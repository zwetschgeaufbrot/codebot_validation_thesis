from __future__ import annotations

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from openai import OpenAI

from .classification import classify_analyses
from .code_parser import load_code_files
from .comparison import LLMComparisonStrategy, run_combined, run_staged, to_comparison_records
from .dimensions import DEFAULT_DIMENSIONS
from .extraction import extract_analysis_summaries
from .models import AnalysisDetail, ComparisonRecord, PaperText, PipelineArtifacts
from .text_parser import parse_pdf
from .writer import write_aggregate, write_intermediates, write_per_paper

LOG = logging.getLogger("codebot_flow")


def _find_dotenv(start: Path) -> Path | None:
    for parent in [start] + list(start.parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def _load_dotenv_fallback(path: Path) -> None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("export "):
                raw = raw[len("export ") :].strip()
            if "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
                value = value[1:-1]
            os.environ[key] = value
    except Exception:
        # Best-effort: ignore .env parsing errors
        return


def _load_dotenv_if_present() -> None:
    env_path = _find_dotenv(Path(__file__).resolve())
    if not env_path:
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        _load_dotenv_fallback(env_path)
        return
    load_dotenv(env_path)


_load_dotenv_if_present()


def _full_report_exists(output_dir: Path, paper_id: str) -> bool:
    json_path = output_dir / f"{paper_id}.json"
    csv_path = output_dir / f"{paper_id}.csv"
    return json_path.exists() and csv_path.exists()


def _load_existing_report(path: Path) -> list[ComparisonRecord]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    records: list[ComparisonRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            records.append(ComparisonRecord(**item))
        except Exception:
            continue
    return records


@dataclass
class RunConfig:
    paper_path: Path
    code_root: Path
    paper_id: str | None = None
    parser: str = "grobid"  # grobid | dpt2 | pypdf | text
    mode: str = "combined"  # combined | staged
    model: str = os.getenv("CODEBOT_MODEL", "gpt-5.1")
    reasoning_effort: str = os.getenv("CODEBOT_REASONING", "medium")
    output_dir: Path = Path("data/raw/codebot/output")
    dimensions: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_DIMENSIONS))
    keep_intermediates: bool = True


def default_client() -> OpenAI:
    api_key = os.getenv("CODEBOT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set CODEBOT_OPENAI_API_KEY or OPENAI_API_KEY before running.")
    return OpenAI(api_key=api_key)


def run_single(config: RunConfig, *, client: OpenAI | None = None) -> PipelineArtifacts:
    client = client or default_client()

    LOG.info("Process: parse paper | %s", config.paper_path)
    paper = parse_pdf(config.paper_path, method=config.parser)
    if config.paper_id:
        paper.paper_id = config.paper_id

    LOG.info("Process: extract analysis summaries")
    summaries = extract_analysis_summaries(
        paper, client=client, model=config.model, reasoning_effort=config.reasoning_effort
    )
    LOG.info("Process: prepare analysis details from summaries")
    details = [
        AnalysisDetail(
            analysis_id=s.analysis_id,
            brief_description=s.brief_description,
            location=s.location,
        )
        for s in summaries
    ]
    LOG.info("Process: classify analyses")
    relevant = classify_analyses(details, client=client, model=config.model, reasoning_effort=config.reasoning_effort)

    LOG.info("Process: load code files | %s", config.code_root)
    code_files = load_code_files(config.code_root)

    strategy = LLMComparisonStrategy(
        client=client, model=config.model, dimensions=config.dimensions, reasoning_effort=config.reasoning_effort
    )

    if config.mode == "staged":
        LOG.info("Process: staged comparison")
        paper_evidence, code_evidence, matches = run_staged(
            relevant,
            paper.text,
            code_files,
            strategy=strategy,
            paper_id=paper.paper_id,
            dimensions=config.dimensions,
        )
    else:
        LOG.info("Process: combined comparison")
        matches = run_combined(
            relevant,
            paper.text,
            code_files,
            strategy=strategy,
            paper_id=paper.paper_id,
            dimensions=config.dimensions,
        )
        paper_evidence = None
        code_evidence = None

    comparisons = to_comparison_records(relevant, matches)

    LOG.info("Process: write outputs | %s", config.output_dir)
    write_per_paper(comparisons, paper.paper_id, config.output_dir)
    if config.keep_intermediates:
        write_intermediates(
            paper.paper_id,
            paper_evidence=paper_evidence,
            code_evidence=code_evidence,
            matches=matches,
            output_dir=config.output_dir,
        )

    return PipelineArtifacts(
        paper=paper,
        analyses=relevant,
        code_files=code_files,
        paper_evidence=paper_evidence,
        code_evidence=code_evidence,
        matches=matches,
        comparisons=comparisons,
    )


def run_multi(
    rows: Iterable[dict],
    *,
    client: OpenAI | None = None,
    parser: str = "grobid",
    mode: str = "combined",
    model: str = os.getenv("CODEBOT_MODEL", "gpt-5.1"),
    reasoning_effort: str = os.getenv("CODEBOT_REASONING", "medium"),
    output_dir: Path = Path("data/raw/codebot/output"),
    dimensions: Mapping[str, str] = DEFAULT_DIMENSIONS,
    keep_intermediates: bool = True,
    parallelism: int = 1,
) -> list[PipelineArtifacts]:
    """rows must contain paper_id, paper_path, code_root."""
    output_dir = Path(output_dir)
    artifacts: list[PipelineArtifacts] = []
    all_records: list[ComparisonRecord] = []
    pending_rows: list[dict] = []
    for row in rows:
        paper_id = row.get("paper_id") or Path(row["paper_path"]).stem
        if _full_report_exists(output_dir, paper_id):
            LOG.info("Run: multi | skip %s (report exists)", paper_id)
            all_records.extend(_load_existing_report(output_dir / f"{paper_id}.json"))
        else:
            pending_rows.append(row)

    if parallelism <= 1 and client is None:
        client = default_client()

    def _run_row(row: dict) -> PipelineArtifacts:
        local_client = client or default_client()
        paper_id = row.get("paper_id") or Path(row["paper_path"]).stem
        cfg = RunConfig(
            paper_path=Path(row["paper_path"]),
            code_root=Path(row["code_root"]),
            paper_id=paper_id,
            parser=parser,
            mode=mode,
            model=model,
            reasoning_effort=reasoning_effort,
            output_dir=output_dir,
            dimensions=dimensions,
            keep_intermediates=keep_intermediates,
        )
        return run_single(cfg, client=local_client)

    if parallelism <= 1 or len(pending_rows) <= 1:
        for row in pending_rows:
            art = _run_row(row)
            artifacts.append(art)
            all_records.extend(art.comparisons or [])
    else:
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            future_to_id = {
                executor.submit(_run_row, row): (row.get("paper_id") or Path(row["paper_path"]).stem)
                for row in pending_rows
            }
            for future in as_completed(future_to_id):
                paper_id = future_to_id[future]
                try:
                    art = future.result()
                except Exception:
                    LOG.exception("Run: multi | failed %s", paper_id)
                    errors.append(paper_id)
                    continue
                artifacts.append(art)
                all_records.extend(art.comparisons or [])
        if errors:
            LOG.error("Run: multi | %d failures: %s", len(errors), ", ".join(errors))
            raise RuntimeError(f"run-multi had {len(errors)} failures: {', '.join(errors)}")

    if all_records:
        write_aggregate(all_records, output_dir / "all_papers")

    return artifacts


__all__ = ["RunConfig", "run_single", "run_multi", "default_client"]
