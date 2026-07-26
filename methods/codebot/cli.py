from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

from .dimensions import DEFAULT_DIMENSIONS
from .runner import RunConfig, default_client, run_multi, run_single
from .comparison import LLMComparisonStrategy, to_comparison_records
from .code_parser import load_code_files
from .extraction import extract_analysis_summaries
from .classification import classify_analyses
from .text_parser import parse_pdf
from .writer import ensure_dir, write_intermediates, write_per_paper
from .models import AnalysisDetail

LOG = logging.getLogger("codebot_flow")


# ----------------- CLI helpers -----------------

def _add_common_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper-id", help="Identifier for the paper (defaults to stem of paper path)")
    parser.add_argument("--parser", default="grobid", choices=["grobid", "dpt2", "pypdf", "text"], help="Paper parsing backend")
    parser.add_argument("--model", default="gpt-5.1", help="OpenAI model to use")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="OpenAI reasoning_effort setting",
    )
    parser.add_argument("--mode", default="combined", choices=["combined", "staged"], help="Comparison mode")
    parser.add_argument("--output-dir", default="data/raw/codebot/output", help="Directory for outputs")
    parser.add_argument("--no-intermediates", action="store_true", help="Skip saving stage artifacts")


# ----------------- Stage commands -----------------

def stage_paper(args) -> None:
    LOG.info("Stage: paper | parse + extract + classify")
    client = default_client()
    paper = parse_pdf(args.paper, method=args.parser)
    if args.paper_id:
        paper.paper_id = args.paper_id

    LOG.info("Stage: paper | extract analysis summaries")
    summaries = extract_analysis_summaries(paper, client=client, model=args.model, reasoning_effort=args.reasoning_effort)
    LOG.info("Stage: paper | prepare analysis details from summaries")
    details = [
        AnalysisDetail(
            analysis_id=s.analysis_id,
            brief_description=s.brief_description,
            location=s.location,
        )
        for s in summaries
    ]
    LOG.info("Stage: paper | classify analyses")
    relevant = classify_analyses(details, client=client, model=args.model, reasoning_effort=args.reasoning_effort)

    state_dir = Path(args.state_dir)
    ensure_dir(state_dir)
    (state_dir / "analyses.json").write_text(json.dumps([a.__dict__ for a in relevant], ensure_ascii=False, indent=2), encoding="utf-8")
    (state_dir / "paper_meta.json").write_text(json.dumps({"paper_id": paper.paper_id}, indent=2), encoding="utf-8")

    # paper evidence per dimension
    LOG.info("Stage: paper | extract dimension evidence from paper")
    strategy = LLMComparisonStrategy(
        client=client, model=args.model, dimensions=DEFAULT_DIMENSIONS, reasoning_effort=args.reasoning_effort
    )
    paper_evidence = []
    for analysis in relevant:
        for dim, definition in DEFAULT_DIMENSIONS.items():
            ev = strategy.extract_paper_dimension(analysis, dim, definition, paper.text)
            ev.paper_id = paper.paper_id
            paper_evidence.append(ev)

    (state_dir / "paper_evidence.json").write_text(
        json.dumps([e.__dict__ for e in paper_evidence], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved analyses + paper evidence to {state_dir}")


def stage_code(args) -> None:
    LOG.info("Stage: code | load analyses + paper evidence")
    client = default_client()
    state_dir = Path(args.state_dir)
    analyses_path = state_dir / "analyses.json"
    paper_ev_path = state_dir / "paper_evidence.json"
    if not analyses_path.exists() or not paper_ev_path.exists():
        raise SystemExit("Run stage paper first to produce analyses.json and paper_evidence.json")

    analyses = [AnalysisDetail(**item) for item in json.loads(analyses_path.read_text(encoding="utf-8"))]
    paper_evidence = { (item["analysis_id"], item["dimension"]): item["value"] for item in json.loads(paper_ev_path.read_text(encoding="utf-8")) }

    LOG.info("Stage: code | load code files")
    code_files = load_code_files(args.code_root)
    meta = json.loads((state_dir / "paper_meta.json").read_text(encoding="utf-8")) if (state_dir / "paper_meta.json").exists() else {}
    paper_id = meta.get("paper_id") or args.paper_id or Path(args.code_root).stem

    LOG.info("Stage: code | extract dimension evidence from code")
    strategy = LLMComparisonStrategy(
        client=client, model=args.model, dimensions=DEFAULT_DIMENSIONS, reasoning_effort=args.reasoning_effort
    )
    code_evidence = []
    for analysis in analyses:
        for dim, definition in DEFAULT_DIMENSIONS.items():
            paper_val = paper_evidence.get((analysis.analysis_id, dim))
            ev = strategy.extract_code_dimension(analysis, dim, definition, code_files, paper_value=paper_val)
            ev.paper_id = paper_id
            code_evidence.append(ev)

    (state_dir / "code_evidence.json").write_text(
        json.dumps([e.__dict__ for e in code_evidence], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved code evidence to {state_dir}")


def stage_judge(args) -> None:
    LOG.info("Stage: judge | load analyses + evidence")
    client = default_client()
    state_dir = Path(args.state_dir)
    analyses_path = state_dir / "analyses.json"
    paper_ev_path = state_dir / "paper_evidence.json"
    code_ev_path = state_dir / "code_evidence.json"
    if not (analyses_path.exists() and paper_ev_path.exists() and code_ev_path.exists()):
        raise SystemExit("Run stage paper and code first to produce analyses.json/paper_evidence.json/code_evidence.json")

    analyses = [AnalysisDetail(**item) for item in json.loads(analyses_path.read_text(encoding="utf-8"))]
    paper_evidence = { (item["analysis_id"], item["dimension"]): item for item in json.loads(paper_ev_path.read_text(encoding="utf-8")) }
    code_evidence = { (item["analysis_id"], item["dimension"]): item for item in json.loads(code_ev_path.read_text(encoding="utf-8")) }

    meta = json.loads((state_dir / "paper_meta.json").read_text(encoding="utf-8")) if (state_dir / "paper_meta.json").exists() else {}
    paper_id = meta.get("paper_id") or args.paper_id or "paper"

    LOG.info("Stage: judge | adjudicate matches")
    strategy = LLMComparisonStrategy(
        client=client, model=args.model, dimensions=DEFAULT_DIMENSIONS, reasoning_effort=args.reasoning_effort
    )
    matches = []
    for analysis in analyses:
        for dim, definition in DEFAULT_DIMENSIONS.items():
            paper_val = (paper_evidence.get((analysis.analysis_id, dim)) or {}).get("value", "")
            code_val = (code_evidence.get((analysis.analysis_id, dim)) or {}).get("value", "")
            m = strategy.judge_dimension(analysis, dim, definition, paper_val, code_val)
            m.paper_id = paper_id
            matches.append(m)

    LOG.info("Stage: judge | write reports")
    comparisons = to_comparison_records(analyses, matches)
    out_dir = Path(args.output_dir)
    write_per_paper(comparisons, paper_id, out_dir)
    write_intermediates(paper_id, paper_evidence=list(paper_evidence.values()), code_evidence=list(code_evidence.values()), matches=matches, output_dir=out_dir)
    print(f"Wrote adjudications + reports to {out_dir}")


# ----------------- Main CLI -----------------

def run_single_cmd(args) -> None:
    LOG.info("Run: single | pipeline start")
    cfg = RunConfig(
        paper_path=Path(args.paper),
        code_root=Path(args.code_root),
        paper_id=args.paper_id,
        parser=args.parser,
        mode=args.mode,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        output_dir=Path(args.output_dir),
        keep_intermediates=not args.no_intermediates,
    )
    run_single(cfg)
    LOG.info("Run: single | pipeline complete")


def run_multi_cmd(args) -> None:
    LOG.info("Run: multi | pipeline start")
    rows: List[dict] = []
    with open(args.pairs_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not {"paper_path", "code_root"}.issubset(row):
                raise SystemExit("pairs CSV must include columns: paper_path, code_root[, paper_id]")
            rows.append({
                "paper_path": row["paper_path"],
                "code_root": row["code_root"],
                "paper_id": row.get("paper_id"),
            })
    run_multi(
        rows,
        parser=args.parser,
        mode=args.mode,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        output_dir=Path(args.output_dir),
        keep_intermediates=not args.no_intermediates,
        parallelism=args.parallelism,
    )
    LOG.info("Run: multi | pipeline complete")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CodeBot modular CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # run-single
    single = sub.add_parser("run-single", help="Run pipeline for one paper + codebase")
    single.add_argument("--paper", required=True, help="Path to PDF or text file")
    single.add_argument("--code-root", required=True, help="Path to code directory or file")
    _add_common_parser_args(single)
    single.set_defaults(func=run_single_cmd)

    # run-multi
    multi = sub.add_parser("run-multi", help="Run pipeline for multiple pairs from CSV")
    multi.add_argument("--pairs-csv", required=True, help="CSV with columns paper_path,code_root[,paper_id]")
    multi.add_argument("--parallelism", type=int, default=1, help="Number of paper-code pairs to run in parallel")
    _add_common_parser_args(multi)
    multi.set_defaults(func=run_multi_cmd)

    # staged commands
    stage = sub.add_parser("stage", help="Run an individual stage (paper/code/judge)")
    stage.add_argument("stage", choices=["paper", "code", "judge"], help="Stage to execute")
    stage.add_argument("--paper", help="Path to paper (required for stage paper)")
    stage.add_argument("--code-root", help="Code root (required for stage code)")
    stage.add_argument("--state-dir", default=".codebot_state", help="Where to persist intermediate JSON between stages")
    stage.add_argument("--paper-id", help="Paper identifier")
    stage.add_argument("--parser", default="grobid", choices=["grobid", "dpt2", "pypdf", "text"], help="Paper parsing backend")
    stage.add_argument("--model", default="gpt-5.1", help="OpenAI model to use")
    stage.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="OpenAI reasoning_effort setting",
    )
    stage.add_argument("--output-dir", default="data/raw/codebot/output", help="Final report destination (judge stage)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="CLI log level")

    def stage_dispatch(args):
        if args.stage == "paper":
            if not args.paper:
                raise SystemExit("--paper is required for stage paper")
            stage_paper(args)
        elif args.stage == "code":
            if not args.code_root:
                raise SystemExit("--code-root is required for stage code")
            stage_code(args)
        else:
            stage_judge(args)

    stage.set_defaults(func=stage_dispatch)

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
