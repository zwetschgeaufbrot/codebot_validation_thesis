from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Mapping, Sequence

from openai import OpenAI

from .dimensions import DEFAULT_DIMENSIONS
from .extraction import extract_code_dimension as extract_code_dimension_call
from .extraction import extract_paper_dimension as extract_paper_dimension_call
from .models import (
    AnalysisDetail,
    CodeFile,
    ComparisonRecord,
    DimensionEvidence,
    MatchDecision,
)

LOG = logging.getLogger("codebot_flow")


def _compact_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)


def _parse_json_or_fallback(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


class LLMComparisonStrategy:
    """Default LLM-based strategy with optional staged flow."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5.1",
        dimensions: Mapping[str, str] | None = None,
        reasoning_effort: str = "medium",
    ):
        self.client = client
        self.model = model
        self.dimensions = dict(dimensions or DEFAULT_DIMENSIONS)
        self.reasoning_effort = reasoning_effort

    # --- staged pieces ---
    def extract_paper_dimension(self, analysis: AnalysisDetail, dimension: str, definition: str, paper_text: str) -> DimensionEvidence:
        return extract_paper_dimension_call(
            analysis,
            dimension,
            definition,
            paper_text,
            client=self.client,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    def extract_code_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        code_files: Sequence[CodeFile],
        paper_value: str | None = None,
    ) -> DimensionEvidence:
        return extract_code_dimension_call(
            analysis,
            dimension,
            definition,
            code_files,
            paper_value=paper_value,
            client=self.client,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    def judge_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        paper_value: str,
        code_value: str,
    ) -> MatchDecision:
        LOG.info("API: judge dimension | analysis_id=%s | dimension=%s", analysis.analysis_id, dimension)
        prompt = (
            "Below, you will see a description of a specific analysis conducted within an academic paper, a specific dimension of analysis to be considered, specific quotes from the academic paper that relate to this analyis-dimension pairing, and specific code snippets from the associated analysis code related to this dimension-analysis pairing.\n"
            "Your task is to determine whether the code implementation matches the paper description for this analysis-dimension pairing. You may make one of three judgements: 'match', 'mismatch', or 'unknown'.\n"
            "'match' indicates that the code implementation appears to align with the paper description for the analysis-dimension pairing. 'mismatch' indicates that the code implementation appears to differ from the paper description for the analysis-dimension pairing. 'unknown' indicates that it is unclear whether the code implementation matches the paper description for the analysis-dimension pairing (e.g., due to insufficient information in either source).\n"
            "Return STRICT JSON with keys: status, explanation (<=2 sentences), evidence.\n"
            "status refers to the match/mismatch/unknown judgement. explanation refers to a brief rationale for the judgement. evidence refers to specific supporting information from the paper and code that informed the judgement (e.g., direct quotes, file names, line numbers, etc.).\n\n"
            f"Here is the brief description of the analysis of interest: {_compact_json(analysis.__dict__)}\n\n"
            f"Here is the dimension of interest: {dimension}, which is defined as {definition}\n\n"
            f"Here are the contents extracted from the paper in relation to this: {paper_value}\n\n"
            f"Here are the contents extracted from the code in relation to this: {code_value}\n\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": "You are a strict adjudicator returning JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        payload = _parse_json_or_fallback(raw)
        status = payload.get("status") or payload.get("match_status") or "unknown"
        explanation = payload.get("explanation") or payload.get("reason", "")
        evidence = payload.get("evidence") or {}
        return MatchDecision(
            paper_id=analysis.analysis_id,
            analysis_id=analysis.analysis_id,
            dimension=dimension,
            status=status,
            explanation=explanation,
            paper_value=paper_value,
            code_value=code_value,
            evidence=evidence,
            llm_raw=raw,
        )

    # --- combined legacy flow ---
    # Note from Jamie: this is not currently used in the main pipeline, but retained for potential future use
    def combined_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        paper_text: str,
        code_files: Sequence[CodeFile],
    ) -> MatchDecision:
        LOG.info("API: combined comparison | analysis_id=%s | dimension=%s", analysis.analysis_id, dimension)
        prompt = (
            "Below, you will see the content of an academic paper, a description of a specific analysis conducted within that paper, "
            "a specific analysis dimension to consider, and the full contents of relevant code files.\n"
            "Your task is to (1) extract direct quotes from the paper relevant to the analysis + dimension, "
            "(2) extract direct code snippets from the codebase relevant to the analysis + dimension, and "
            "(3) judge whether the code matches the paper along this dimension.\n"
            "Return STRICT JSON with keys:\n"
            "{\n"
            "  \"status\": \"match\"|\"mismatch\"|\"unknown\",\n"
            "  \"paper_value\": \"direct quote(s) from the paper\",\n"
            "  \"code_value\": \"direct code snippet(s)\",\n"
            "  \"explanation\": \"<=2 sentences\",\n"
            "  \"evidence\": {\n"
            "     \"paper_span\": \"supporting quote(s)\",\n"
            "     \"location\": \"where in the paper the quotes appear (section/page/line if possible)\",\n"
            "     \"code_path\": \"file path\",\n"
            "     \"code_lines\": \"line range or approximate lines\"\n"
            "  }\n"
            "}\n"
            "paper_value should be the most relevant direct quote(s) for this dimension. paper_span can be the same as paper_value "
            "or a longer surrounding excerpt. code_value must include actual code (not only comments). If you cannot find an "
            "appropriate paper or code excerpt, use an empty string for that field and consider status=\"unknown\".\n\n"
            f"Here is the brief description of the analysis: {_compact_json(analysis.__dict__)}\n\n"
            f"Here is the dimension to compare: {dimension}, which is defined as {definition}\n\n"
            "Here is the paper text:\n" + paper_text + "\n\n"
            "Here are the code files (path + content), as a JSON list:\n" + _compact_json([cf.__dict__ for cf in code_files])
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": "You are a rigorous extraction and comparison assistant. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        payload = _parse_json_or_fallback(raw)
        status = payload.get("status") or "unknown"
        explanation = payload.get("explanation") or payload.get("reason", "")
        evidence = payload.get("evidence") or {}
        paper_value = payload.get("paper_value") or ""
        code_value = payload.get("code_value") or ""
        return MatchDecision(
            paper_id=analysis.analysis_id,
            analysis_id=analysis.analysis_id,
            dimension=dimension,
            status=status,
            explanation=explanation,
            paper_value=paper_value,
            code_value=code_value,
            evidence=evidence,
            llm_raw=raw,
        )


# Placeholder for future RAG-based strategy
class RAGComparisonStrategy(LLMComparisonStrategy):
    """Placeholder for future RAG-based retrieval. Swap in retrieval inside extract_* methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hook for RAG vector store / retrieval models


# ---------- Orchestration helpers ----------

def run_staged(
    analyses: Sequence[AnalysisDetail],
    paper_text: str,
    code_files: Sequence[CodeFile],
    *,
    strategy: LLMComparisonStrategy,
    paper_id: str,
    dimensions: Mapping[str, str] | None = None,
) -> tuple[List[DimensionEvidence], List[DimensionEvidence], List[MatchDecision]]:
    dims = dimensions or strategy.dimensions

    paper_evidence: List[DimensionEvidence] = []
    total_analyses = len(analyses)
    total_dims = len(dims)
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: paper evidence | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            ev = strategy.extract_paper_dimension(analysis, dim, definition, paper_text)
            ev.paper_id = paper_id
            paper_evidence.append(ev)

    code_evidence: List[DimensionEvidence] = []
    paper_lookup = {(e.analysis_id, e.dimension): e for e in paper_evidence}
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: code evidence | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            paper_ev = paper_lookup.get((analysis.analysis_id, dim))
            ev = strategy.extract_code_dimension(
                analysis,
                dim,
                definition,
                code_files,
                paper_value=paper_ev.value if paper_ev else None,
            )
            ev.paper_id = paper_id
            code_evidence.append(ev)

    matches: List[MatchDecision] = []
    code_lookup = {(e.analysis_id, e.dimension): e for e in code_evidence}
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: adjudication | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            paper_ev = paper_lookup.get((analysis.analysis_id, dim))
            code_ev = code_lookup.get((analysis.analysis_id, dim))
            paper_val = paper_ev.value if paper_ev else ""
            code_val = code_ev.value if code_ev else ""
            decision = strategy.judge_dimension(analysis, dim, definition, paper_val, code_val)
            decision.paper_id = paper_id
            matches.append(decision)

    return paper_evidence, code_evidence, matches


def run_combined(
    analyses: Sequence[AnalysisDetail],
    paper_text: str,
    code_files: Sequence[CodeFile],
    *,
    strategy: LLMComparisonStrategy,
    paper_id: str,
    dimensions: Mapping[str, str] | None = None,
) -> List[MatchDecision]:
    dims = dimensions or strategy.dimensions
    decisions: List[MatchDecision] = []
    total_analyses = len(analyses)
    total_dims = len(dims)
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: combined comparison | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            decision = strategy.combined_dimension(analysis, dim, definition, paper_text, code_files)
            decision.paper_id = paper_id
            decisions.append(decision)
    return decisions


def to_comparison_records(
    analyses: Sequence[AnalysisDetail],
    decisions: Iterable[MatchDecision],
) -> List[ComparisonRecord]:
    brief_lookup = {a.analysis_id: a.brief_description for a in analyses}
    records: List[ComparisonRecord] = []
    for dec in decisions:
        records.append(
            ComparisonRecord(
                paper_id=dec.paper_id,
                analysis_id=dec.analysis_id,
                brief_description=brief_lookup.get(dec.analysis_id, ""),
                dimension=dec.dimension,
                paper_value=dec.paper_value,
                code_value=dec.code_value,
                match_status=dec.status,
                explanation=dec.explanation,
                evidence=dec.evidence,
                llm_raw=dec.llm_raw,
            )
        )
    return records


__all__ = [
    "LLMComparisonStrategy",
    "RAGComparisonStrategy",
    "run_staged",
    "run_combined",
    "to_comparison_records",
]
