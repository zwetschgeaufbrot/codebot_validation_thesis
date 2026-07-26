from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from .models import ComparisonRecord, DimensionEvidence, MatchDecision


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(records: Sequence[ComparisonRecord], path: Path) -> None:
    ensure_dir(path.parent)
    def _extract_evidence(evidence, key: str) -> str:
        if isinstance(evidence, dict):
            return str(evidence.get(key, "")) if evidence.get(key) is not None else ""
        if isinstance(evidence, list):
            vals = []
            for item in evidence:
                if isinstance(item, dict) and item.get(key):
                    vals.append(str(item.get(key)))
            return "\n\n".join(vals)
        return ""

    fieldnames = [
        "paper_id",
        "analysis_id",
        "brief_description",
        "dimension",
        "paper_value",
        "code_value",
        "match_status",
        "explanation",
        "paper_evidence",
        "code_evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "paper_id": r.paper_id,
                    "analysis_id": r.analysis_id,
                    "brief_description": r.brief_description,
                    "dimension": r.dimension,
                    "paper_value": r.paper_value,
                    "code_value": r.code_value,
                    "match_status": r.match_status,
                    "explanation": r.explanation,
                    "paper_evidence": _extract_evidence(r.evidence, "paper_span"),
                    "code_evidence": _extract_evidence(r.evidence, "code_path"),
                }
            )


def write_per_paper(records: Sequence[ComparisonRecord], paper_id: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    write_json([r.to_dict() for r in records], output_dir / f"{paper_id}.json")
    write_csv(records, output_dir / f"{paper_id}.csv")


def write_intermediates(
    paper_id: str,
    *,
    paper_evidence: Sequence[DimensionEvidence] | None,
    code_evidence: Sequence[DimensionEvidence] | None,
    matches: Sequence[MatchDecision] | None,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    to_dict = lambda obj: obj.__dict__ if hasattr(obj, "__dict__") else obj
    if paper_evidence is not None:
        write_json([to_dict(e) for e in paper_evidence], output_dir / f"{paper_id}_paper_evidence.json")
    if code_evidence is not None:
        write_json([to_dict(e) for e in code_evidence], output_dir / f"{paper_id}_code_evidence.json")
    if matches is not None:
        write_json([to_dict(m) for m in matches], output_dir / f"{paper_id}_matches.json")


def write_aggregate(records: Iterable[ComparisonRecord], path: Path) -> None:
    records_list = list(records)
    write_json([r.to_dict() for r in records_list], path.with_suffix(".json"))
    write_csv(records_list, path.with_suffix(".csv"))


__all__ = [
    "write_per_paper",
    "write_aggregate",
    "write_intermediates",
    "write_json",
    "write_csv",
]
