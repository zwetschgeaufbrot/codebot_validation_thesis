from __future__ import annotations

import json
import logging
from typing import List, Sequence

from openai import OpenAI

from .models import AnalysisDetail

LOG = logging.getLogger("codebot_flow")

RELEVANT_CRITERIA = "t-tests, z-tests, univariate ANOVA (incl. repeated measures), linear regression, parametric/nonparametric correlation, Chi-square for proportions, univariate logistic regression, exploratory factor analysis, equivalence tests, Wilcoxon rank/signed tests, and Kruskal-Wallis tests"


def classify_analyses(
    analyses: Sequence[AnalysisDetail],
    *,
    client: OpenAI,
    model: str = "gpt-5.1",
    reasoning_effort: str = "medium",
) -> List[AnalysisDetail]:
    relevant: List[AnalysisDetail] = []
    total = len(analyses)
    for idx, analysis in enumerate(analyses, start=1):
        LOG.info("API: classify analysis | %s/%s | analysis_id=%s", idx, total, analysis.analysis_id)
        user_prompt = (
            "Classify whether this analysis is relevant, irrelevant, or not_main.\n"
            f"Analyses should be classified as relevant if they fulfill any one of the following criteria: {RELEVANT_CRITERIA}."
            f"Analyses should be classified as not_main if they are power checks, manipulation/attention checks, sanity checks, or nested coefficient tests.\n"
            f"Analyses should be classified as irrelevant if they do not fulfill any of the relevant criteria, and do not quality for any of the not_main criteria.\n"
            "Return STRICT JSON: {\"classification\": "
            "\"relevant\"|\"irrelevant\"|\"not_main\"}.\n"
            f"The brief description of the analysis to be classified is: {json.dumps(analysis.__dict__, ensure_ascii=False)}"
        )
        resp = client.chat.completions.create(
            model=model,
            reasoning_effort=reasoning_effort,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = resp.choices[0].message.content or ""
        try:
            classification = json.loads(raw).get("classification", "irrelevant")
        except Exception:
            classification = "irrelevant"
        if classification == "relevant":
            relevant.append(analysis)
    return relevant


__all__ = ["classify_analyses", "RELEVANT_CRITERIA"]
