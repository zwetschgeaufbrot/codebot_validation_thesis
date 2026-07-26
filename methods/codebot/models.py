from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CodeFile:
    path: str
    content: str

    @classmethod
    def from_path(cls, path: Path) -> "CodeFile":
        return cls(path=str(path), content=path.read_text(encoding="utf-8", errors="replace"))


@dataclass
class PaperText:
    paper_id: str
    text: str
    source: str  # e.g., "dpt2", "grobid", "pypdf", "provided"
    meta: Dict[str, Any] | None = None


@dataclass
class AnalysisSummary:
    analysis_id: str
    brief_description: str
    location: str | None = None


@dataclass
class AnalysisDetail(AnalysisSummary):
    model_specification: str | None = None
    variable_specification: str | None = None
    transformation_specification: str | None = None
    inference_specification: str | None = None


@dataclass
class DimensionEvidence:
    paper_id: str
    analysis_id: str
    dimension: str
    value: str
    source: str  # "paper" or "code"
    evidence: Dict[str, Any] | None = None
    llm_raw: str | None = None


@dataclass
class MatchDecision:
    paper_id: str
    analysis_id: str
    dimension: str
    status: str  # match | mismatch | unknown
    explanation: str
    paper_value: str
    code_value: str
    evidence: Dict[str, Any] | None = None
    llm_raw: str | None = None


@dataclass
class ComparisonRecord:
    paper_id: str
    analysis_id: str
    brief_description: str
    dimension: str
    paper_value: str
    code_value: str
    match_status: str
    explanation: str
    evidence: Dict[str, Any] | None = None
    llm_raw: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineArtifacts:
    paper: PaperText
    analyses: List[AnalysisDetail]
    code_files: List[CodeFile]
    paper_evidence: List[DimensionEvidence] | None = None
    code_evidence: List[DimensionEvidence] | None = None
    matches: List[MatchDecision] | None = None
    comparisons: List[ComparisonRecord] | None = None
