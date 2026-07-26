from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence
from openai import OpenAI

from .dimensions import DEFAULT_DIMENSIONS
from .models import AnalysisDetail, AnalysisSummary, CodeFile, DimensionEvidence, PaperText

LOG = logging.getLogger("codebot_flow")


def _to_json_or_fallback(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def _normalize_items(payload: Any) -> List[Dict[str, Any]]:
    """Return list of analysis items whether payload is dict with analyses or a raw list."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("analyses") or payload.get("data") or []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


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


def extract_analysis_summaries(
    paper: PaperText,
    *,
    client: OpenAI,
    model: str = "gpt-5.1",
    reasoning_effort: str = "medium",
) -> List[AnalysisSummary]:
    LOG.info("API: extract analysis summaries | paper_id=%s", paper.paper_id)
    prompt = (
        "You are extracting analyses conducted by the authors in an academic paper.\n"
        "Return STRICT JSON with a top-level key 'analyses', each item having: analysis_id (int),"
        " analysis_description (2-4 sentence summary including test type and variables, specific enough in description to facilitate future searching and identification of this specific analysis), and location (in-text or table + approximate page/line).\n"
        "Include all analyses meeting any of: t/z tests, univariate ANOVA, linear regression, correlations, Chi-square (proportions), logistic regression, exploratory factor analysis, equivalence tests, Wilcoxon rank/signed, Kruskal-Wallis.\n"
        "If unsure, include it.\n"
        "Be comprehensive in your inclusions: if there are multiple similar analyses (e.g., multiple t-tests), include each separately with distinct IDs. If there are multiple correlations reported in a correlation matrix, include each separate correlation as a distinct analysis.\n"
        "In short: be over-inclusive rather than under-inclusive.\n"
        "Paper text follows:\n" + paper.text
    )

    resp = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[
            {"role": "system", "content": "You are a labelling assistant that outputs valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content or ""
    payload = _to_json_or_fallback(raw)
    analyses = _normalize_items(payload)

    results: List[AnalysisSummary] = []
    for item in analyses:
        try:
            analysis_id = str(item.get("analysis_id"))
            desc = item.get("analysis_description") or item.get("description") or ""
            loc = item.get("location")
            results.append(AnalysisSummary(analysis_id=analysis_id, brief_description=desc, location=loc))
        except Exception:
            continue
    if not results and isinstance(payload, dict) and "raw" in payload:
        # fallback: single item using raw text
        results.append(AnalysisSummary(analysis_id="1", brief_description=payload["raw"], location=None))
    return results


def extract_paper_dimension(
    analysis: AnalysisDetail,
    dimension: str,
    definition: str,
    paper_text: str,
    *,
    client: OpenAI,
    model: str = "gpt-5.1",
    reasoning_effort: str = "medium",
) -> DimensionEvidence:
    LOG.info("API: extract paper dimension | analysis_id=%s | dimension=%s", analysis.analysis_id, dimension)
    prompt = (
        "Below, you will see the content of an academic paper, a description of a specific analysis conducted within that paper, and a specific dimension of analysis to be considered.\n"
        "Your task is to extract direct quotes from the paper in relation to the described analysis and the specific analysis dimension.\n"
        "For example, if the analysis dimension is about 'Variable Specification', you should extract quotes from the manuscript related to the specified analysis in terms of how variables are specified within it (e.g., IV, DV, moderators, mediators, etc.)\n"
        "Return STRICT JSON: {\"paper_value\": str, \"evidence\": {\"paper_span\": str, \"location\": str}}.\n"
        "paper_value refers to the extracted quote(s) most relevant to the dimension. paper_span refers to additional supporting quotes which may be extracted to provide further context. location refers to the section(s) of the paper where the quote(s) were extracted from (ideally with page and/or line numbers where possible).\n\n"
        f"Here is the brief description of the analysis: {_compact_json(analysis.__dict__)}\n\n"
        f"Here is the dimension to compare: {dimension}, which is defined as {definition}\n\n"
        "Here is the paper text:\n" + paper_text
    )
    resp = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[
            {"role": "system", "content": "You extract verbatim quotes and return JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content or ""
    payload = _parse_json_or_fallback(raw)
    paper_value = payload.get("paper_value") or payload.get("value") or payload.get("raw", "")
    evidence = payload.get("evidence") or {}
    return DimensionEvidence(
        paper_id=analysis.analysis_id,  # paper_id to be overwritten by orchestrator
        analysis_id=analysis.analysis_id,
        dimension=dimension,
        value=paper_value,
        source="paper",
        evidence=evidence,
        llm_raw=raw,
    )


def extract_code_dimension(
    analysis: AnalysisDetail,
    dimension: str,
    definition: str,
    code_files: Sequence[CodeFile],
    paper_value: str | None = None,
    *,
    client: OpenAI,
    model: str = "gpt-5.1",
    reasoning_effort: str = "medium",
) -> DimensionEvidence:
    LOG.info("API: extract code dimension | analysis_id=%s | dimension=%s", analysis.analysis_id, dimension)
    prompt = (
        "Below, you will see a description of a specific analysis conducted within an academic paper, a specific dimension of analysis to be considered, and specific quotes from the academic paper that relate to this analyis-dimension pairing.\n"
        "You will also see the content of statistical analysis code files from the repository associated with the paper.\n"
        "Your task is to locate the implementation of this described analysis in the code, and extract direct quotes and code lines from the analysis code that are relevant to the requested dimension.\n"
        "It may be the case that the analysis is not implemented in the code, that the requested dimension is not represented in the code; in such cases, return an empty string for code_value.\n"
        "It may also be the case that the code implementation in relation to the specific analysis dimension differs from the specification in the paper; this is OK. If such discrepancies are present, simply return accurate quotes from the code in relation to the analysis and dimension - they will be judged for similarity at a later step.\n"
        "For example, if the analysis dimension is about 'Variable Specification', you should extract direct snippets from the code related to the specified analysis in terms of how variables are specified within it (e.g., IV, DV, moderators, mediators, etc.)\n"
        "Return STRICT JSON: {\"code_value\": str, \"evidence\": {\"code_path\": str, \"code_lines\": str}}.\n"
        "code_value refers to the extracted code lines most relevant to the analysis-dimension extraction of interest. code_path refers to the path of the code file where the code_value was extracted from (e.g., src/analysis.Rmd). code_lines refers to the actual lines of code that were extracted.\n\n"
        "Note: extracted content from the code should be actual code snippets, not only comments. Comments may be included, but substantive code lines should be the primary focus. If you cannot find any relevant code snippets, use an empty string for code_value.\n\n"
        f"Here is the brief description of the analysis of interest: {_compact_json(analysis.__dict__)}\n\n"
        f"Here is the dimension to compare: {dimension}, which is defined as {definition}\n\n"
        f"Here are the contents extracted from the paper in relation to this: {paper_value or 'n/a'}\n\n"
        f"Here are the code files (path + content), which follow as a JSON list:\n" + _compact_json([cf.__dict__ for cf in code_files])
    )
    resp = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[
            {"role": "system", "content": "You return JSON with directly-quoted code snippets."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content or ""
    payload = _parse_json_or_fallback(raw)
    code_value = payload.get("code_value") or payload.get("value") or payload.get("raw", "")
    evidence = payload.get("evidence") or {}
    return DimensionEvidence(
        paper_id=analysis.analysis_id,  # placeholder
        analysis_id=analysis.analysis_id,
        dimension=dimension,
        value=code_value,
        source="code",
        evidence=evidence,
        llm_raw=raw,
    )


__all__ = [
    "extract_analysis_summaries",
    "extract_paper_dimension",
    "extract_code_dimension",
]
