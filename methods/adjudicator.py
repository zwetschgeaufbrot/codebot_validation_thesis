#!/usr/bin/env python3
"""
Strategy D matcher for CodeBot vs Human code-checking reports.

Implements:
  1) Bundle (analysis_id -> analysis bundle)
  2) Candidate retrieval (TF-IDF cosine similarity)
  3) Adjudication (heuristic by default; optional OpenAI LLM)
  4) Global assignment (Hungarian)
  5) Post-processing split/merge tags
  6) Export mapping + merged datasets

Usage:
  python match_strategy_d.py

Requirements:
  pip install pandas numpy scikit-learn scipy

Optional (LLM adjudication):
  pip install openai
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations
import argparse
import os
import re
import json
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.optimize import linear_sum_assignment

# load environment variables from a local .env file if present
from dotenv import load_dotenv  # type: ignore
load_dotenv()

# ---------------------------
# Column helpers 
# ---------------------------

def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column that exists in df.

    Matching rules:
    1) Exact match (case-sensitive)
    2) Case-insensitive match after stripping whitespace in df column names
    """
    cols = list(df.columns)
    if not cols:
        return None

    # Exact match first
    for c in candidates:
        if c in df.columns:
            return c

    # Case-insensitive + strip match
    normalized = {str(col).strip().lower(): col for col in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in normalized:
            return normalized[key]

    return None



def coalesce_text_cols(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Return a text Series from the first existing candidate column, else empty strings."""
    col = first_existing_col(df, candidates)
    if col is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


# --- Additional helper functions ---
def coalesce_any_col(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Return a Series from the first existing candidate column, else pd.NA."""
    col = first_existing_col(df, candidates)
    if col is None:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[col]


def to_match_judgement(series: pd.Series) -> pd.Series:
    """Map a column to {1,0,NA} with 1=match, 0=mismatch, NA=other/unknown."""
    def _parse(v):
        if pd.isna(v):
            return pd.NA
        # numeric 0/1
        if isinstance(v, (int, np.integer)) and v in (0, 1):
            return int(v)
        if isinstance(v, (float, np.floating)) and v in (0.0, 1.0):
            return int(v)
        t = str(v).strip().lower()

        # Try numeric strings (e.g., "1", "0", "1.0", "0.0")
        try:
            f = float(t)
            if f in (0.0, 1.0):
                return int(f)
        except Exception:
            pass

        if t in {"1", "match", "matched", "yes", "y", "true", "t"}:
            return 1
        if t in {"0", "mismatch", "mismatched", "no", "n", "false", "f"}:
            return 0
        return pd.NA

    return series.apply(_parse).astype("Int64")


def human_code_location_series(df: pd.DataFrame) -> pd.Series:
    """Create a human_code_location from available columns (code_location, code_file, code_lines)."""
    # If a direct location column exists, use it
    direct = first_existing_col(df, ["code_location", "code_loc", "code_path", "code_reference"])
    if direct is not None:
        return df[direct].fillna("").astype(str)

    code_file = coalesce_text_cols(df, ["code_file", "file", "filepath", "path"])
    code_lines = coalesce_text_cols(df, ["code_lines", "lines", "line_numbers", "line_range"])

    loc = (code_file.str.strip() + ":" + code_lines.str.strip()).str.strip(":")
    # If both missing, return empty
    loc = loc.replace({"": ""})
    return loc


def codebot_code_location_series(df: pd.DataFrame) -> pd.Series:
    """Create a codebot_code_location from available columns."""
    direct = first_existing_col(df, ["code_location", "code_loc", "code_reference", "code_file", "code_lines"])
    if direct is not None and direct in df.columns and direct not in ["code_file", "code_lines"]:
        return df[direct].fillna("").astype(str)

    # If code_file + code_lines exist on CodeBot side
    code_file = coalesce_text_cols(df, ["code_file", "file", "filepath", "path"])
    code_lines = coalesce_text_cols(df, ["code_lines", "lines", "line_numbers", "line_range"])
    loc = (code_file.str.strip() + ":" + code_lines.str.strip()).str.strip(":")
    return loc


# ---------------------------
# Config
# ---------------------------

@dataclass
class PaperConfig:
    paper_id: str
    codebot_csv: str
    human_csv: str


# edit paths for csvs
# TO-DO: make these command-line args
PAPERS: List[PaperConfig] = [
    PaperConfig(
        paper_id="elson_2020",
        codebot_csv="codebot_workflow/pilot_reports/elson_2020.csv",  # <-- update
        human_csv="data/raw/pilot_papers/elson_2020.csv",      # <-- update
    ),
    PaperConfig(
        paper_id="anvari_2023",
        codebot_csv="codebot_workflow/pilot_reports/anvari_2023.csv", # <-- update
        human_csv="data/raw/pilot_papers/anvari_2023.csv",     # <-- update
    ),
]

TOP_K = 15                    # retrieval candidates per human analysis bundle
MIN_ACCEPT_CONF = 0.5        # accept matches with adjudication confidence >= this
FRAGMENT_CONF = 0.70          # allow fragment_of links in post-process if >= this
MAX_TEXT_CHARS = 5000         # truncation for LLM prompt, if enabled

OPENAI_MODEL = "gpt-5.1"  


# ---------------------------
# Helpers
# ---------------------------

def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def detect_report_type(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    if "brief_description" in cols and "paper_text" in cols and "code_text" in cols:
        return "codebot"
    if "manuscript_content" in cols and "code_content" in cols:
        return "human"
    # fallback heuristic
    if "brief_description" in cols:
        return "codebot"
    return "human"

def normalize_dimension(dim: str) -> str:
    d = _safe_str(dim).strip().lower()
    if not d:
        return "other"
    # robust keyword mapping
    if "model" in d:
        return "model"
    if "variable" in d:
        return "variables"
    if "transform" in d:
        return "transform"
    if "infer" in d or "criteria" in d:
        return "inference"
    return "other"

def concat_nonempty(parts: List[str], sep: str = "\n") -> str:
    cleaned = [p for p in (p.strip() for p in parts) if p]
    return sep.join(cleaned)

def build_combined_text(bundle: Dict[str, str], include_brief: bool = True) -> str:
    chunks = []
    if include_brief and bundle.get("brief_description", "").strip():
        chunks.append(f"BRIEF_DESCRIPTION:\n{bundle['brief_description'].strip()}")
    for key in ["model", "variables", "transform", "inference", "other", "notes"]:
        val = bundle.get(key, "").strip()
        if val:
            chunks.append(f"{key.upper()}:\n{val}")
    return "\n\n---\n\n".join(chunks).strip()


# ---------------------------
# Bundling
# ---------------------------

def bundle_report(df: pd.DataFrame, report_type: str, paper_id: str) -> pd.DataFrame:
    """
    Returns one row per analysis_id with dimension-specific concatenated text fields + combined_text.
    """
    df = df.copy()

    # normalize dimension
    if "dimension" in df.columns:
        df["dimension_norm"] = df["dimension"].apply(normalize_dimension)
    else:
        df["dimension_norm"] = "other"

    # normalize analysis_id as string (coders vary; keep as string keys)
    if "analysis_id" not in df.columns:
        raise ValueError(f"[{paper_id}] Missing analysis_id column.")
    df["analysis_id"] = df["analysis_id"].astype(str).str.strip()

    bundles = []
    for analysis_id, g in df.groupby("analysis_id", dropna=False):
        b: Dict[str, str] = {
            "paper_id": paper_id,
            "analysis_id": analysis_id,
            "uid": f"{paper_id}::{analysis_id}",
            "brief_description": "",
            "model": "",
            "variables": "",
            "transform": "",
            "inference": "",
            "other": "",
            "notes": "",
        }

        if report_type == "codebot":
            # brief_description is (usually) invariant across dimensions
            if "brief_description" in g.columns:
                b["brief_description"] = _safe_str(g["brief_description"].dropna().iloc[0]) if g["brief_description"].notna().any() else ""

            # dimension-specific texts: prefer paper_text + code_text
            for dn, gg in g.groupby("dimension_norm"):
                paper_text = concat_nonempty([_safe_str(x) for x in gg.get("paper_text", pd.Series(dtype=str)).tolist()])
                code_text  = concat_nonempty([_safe_str(x) for x in gg.get("code_text", pd.Series(dtype=str)).tolist()])
                joined = concat_nonempty(
                    [
                        f"PAPER_TEXT:\n{paper_text}" if paper_text else "",
                        f"CODE_TEXT:\n{code_text}" if code_text else "",
                    ],
                    sep="\n\n",
                )
                b[dn] = concat_nonempty([b.get(dn, ""), joined], sep="\n\n")

        else:  # human
            # dimension-specific: manuscript_content, code_content, plus locations/files/lines if present
            for dn, gg in g.groupby("dimension_norm"):
                m = concat_nonempty([_safe_str(x) for x in gg.get("manuscript_content", pd.Series(dtype=str)).tolist()])
                loc = concat_nonempty([_safe_str(x) for x in gg.get("manuscript_location", pd.Series(dtype=str)).tolist()])
                c = concat_nonempty([_safe_str(x) for x in gg.get("code_content", pd.Series(dtype=str)).tolist()])
                code_file = concat_nonempty([_safe_str(x) for x in gg.get("code_file", pd.Series(dtype=str)).tolist()])
                code_lines = concat_nonempty([_safe_str(x) for x in gg.get("code_lines", pd.Series(dtype=str)).tolist()])

                joined = concat_nonempty(
                    [
                        f"MANUSCRIPT_LOCATION:\n{loc}" if loc else "",
                        f"MANUSCRIPT_CONTENT:\n{m}" if m else "",
                        f"CODE_FILE:\n{code_file}" if code_file else "",
                        f"CODE_LINES:\n{code_lines}" if code_lines else "",
                        f"CODE_CONTENT:\n{c}" if c else "",
                    ],
                    sep="\n\n",
                )
                b[dn] = concat_nonempty([b.get(dn, ""), joined], sep="\n\n")

            # notes can be cross-dimension
            if "notes" in g.columns:
                b["notes"] = concat_nonempty([_safe_str(x) for x in g["notes"].tolist()])

        b["combined_text"] = build_combined_text(b, include_brief=True)
        bundles.append(b)

    out = pd.DataFrame(bundles)
    # ensure columns exist
    for col in ["paper_id","analysis_id","uid","brief_description","model","variables","transform","inference","other","notes","combined_text"]:
        if col not in out.columns:
            out[col] = ""
    return out


# ---------------------------
# Retrieval
# ---------------------------

def retrieve_candidates(
    human_bundles: pd.DataFrame,
    codebot_bundles: pd.DataFrame,
    top_k: int = TOP_K,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Returns similarity matrix (n_human x n_codebot) and candidate indices list for each human row.
    Uses TF-IDF cosine similarity on combined_text.
    """
    human_texts = human_bundles["combined_text"].fillna("").tolist()
    codebot_texts = codebot_bundles["combined_text"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(human_texts + codebot_texts)
    X_h = X[: len(human_texts)]
    X_c = X[len(human_texts) :]

    sim = cosine_similarity(X_h, X_c)  # n_human x n_codebot

    candidates = []
    for i in range(sim.shape[0]):
        idx = np.argsort(sim[i])[::-1][:top_k].tolist()
        candidates.append(idx)
    return sim, candidates


# ---------------------------
# Adjudication
# ---------------------------

def heuristic_adjudicate_one(
    human_row: pd.Series,
    codebot_rows: pd.DataFrame,
    candidate_idxs: List[int],
    sim_row: np.ndarray,
) -> Dict:
    """
    Offline adjudicator: chooses best candidate by similarity and returns a confidence.
    Also tries to guess split/merge by inspecting brief_description & text overlap heuristics.
    """
    if len(candidate_idxs) == 0:
        return {"selected_j": None, "confidence": 0.0, "link_type": "no_match", "rationale": "No candidates."}

    best_j = candidate_idxs[0]
    best_sim = float(sim_row[best_j])

    # Simple confidence shaping: push high sims upward, low sims downward
    confidence = max(0.0, min(1.0, (best_sim - 0.10) / 0.50))  # sim ~0.60 -> 1.0, sim ~0.10 -> 0.0

    # Determine link_type crudely:
    # If best_sim is very high, exact; else maybe fragment_of.
    if confidence >= 0.85:
        link_type = "exact"
    elif confidence >= 0.70:
        link_type = "fragment_of"
    else:
        link_type = "no_match"

    rationale = f"Heuristic: best cosine similarity={best_sim:.3f} (confidence={confidence:.2f}); link_type={link_type}."

    return {"selected_j": best_j, "confidence": float(confidence), "link_type": link_type, "rationale": rationale}


def openai_adjudicate_one(
    human_row: pd.Series,
    codebot_rows: pd.DataFrame,
    candidate_idxs: List[int],
    sim_row: np.ndarray,
    model: str = OPENAI_MODEL,
) -> Dict:
    """
    LLM adjudicator: choose best match among candidates (or no_match).
    Returns JSON-like dict with selected_j, confidence [0..1], link_type, rationale.

    Requires:
      pip install openai
      export OPENAI_API_KEY=...
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI adjudicator selected but openai package is not installed. pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in the process environment. "
            "A .env file is not loaded automatically unless python-dotenv is installed and load_dotenv() is called. "
            "Fix: `pip install python-dotenv` (already supported by this script) or export OPENAI_API_KEY in your shell."
        )

    client = OpenAI(api_key=api_key)
    print(f"[LLM] Calling OpenAI model={model} candidates={len(candidate_idxs)}")
    start_time = time.time()

    def trunc(s: str) -> str:
        s = s or ""
        s = s.strip()
        return s[:MAX_TEXT_CHARS]

    # Build compact candidate summaries
    cand_blocks = []
    for rank, j in enumerate(candidate_idxs, start=1):
        cb = codebot_rows.iloc[j]
        cand_blocks.append(
            f"[Candidate {rank}] codebot_index={j} sim={sim_row[j]:.3f}\n"
            f"BRIEF_DESCRIPTION: {trunc(_safe_str(cb.get('brief_description','')))}\n"
            f"COMBINED_TEXT:\n{trunc(_safe_str(cb.get('combined_text','')))}"
        )
    candidates_text = "\n\n".join(cand_blocks)

    user_prompt = f"""
You are matching a HUMAN-coded analysis bundle to one of several CodeBot analysis bundles from the SAME paper.

Task:
- Choose the single best CodeBot candidate (or choose NO_MATCH).
- If the human bundle is only a *part* of a broader CodeBot analysis, choose that candidate and set link_type="fragment_of".
- If the human bundle appears to *cover multiple* distinct CodeBot analyses, choose the best candidate and set link_type="covers_multiple".
- Otherwise link_type="exact" for a clean match.

Return STRICT JSON with keys:
- selected_codebot_index: integer or null
- link_type: "exact" | "fragment_of" | "covers_multiple" | "no_match"
- confidence: number in [0,1]
- rationale: short explanation (1-3 sentences)

HUMAN_BUNDLE:
{trunc(_safe_str(human_row.get("combined_text","")))}

CODEBOT_CANDIDATES:
{candidates_text}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only. No markdown."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as e:
        print("[LLM] OpenAI call failed:", repr(e))
        traceback.print_exc()
        raise

    elapsed = time.time() - start_time
    print(f"[LLM] Done in {elapsed:.2f}s")

    content = resp.choices[0].message.content.strip()
    # Parse JSON robustly
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # attempt to extract JSON object if model wrapped it
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))

    selected = data.get("selected_codebot_index", None)
    if selected is None:
        return {"selected_j": None, "confidence": float(data.get("confidence", 0.0)), "link_type": "no_match", "rationale": data.get("rationale","")}

    return {
        "selected_j": int(selected),
        "confidence": float(data.get("confidence", 0.0)),
        "link_type": str(data.get("link_type", "exact")),
        "rationale": str(data.get("rationale", "")),
    }


# ---------------------------
# Global assignment + post-processing
# ---------------------------

def hungarian_assignment(
    n_h: int,
    n_c: int,
    edge_weights: Dict[Tuple[int, int], float],
) -> List[Tuple[int, Optional[int], float]]:
    """
    Maximum-weight bipartite matching (1-1) using Hungarian algorithm.
    Returns list of (human_i, codebot_j or None, weight).
    """
    # Build dense weight matrix
    W = np.zeros((n_h, n_c), dtype=float)
    for (i, j), w in edge_weights.items():
        if 0 <= i < n_h and 0 <= j < n_c:
            W[i, j] = max(W[i, j], w)

    # Pad to square with dummy columns/rows
    n = max(n_h, n_c)
    W_sq = np.zeros((n, n), dtype=float)
    W_sq[:n_h, :n_c] = W

    # Convert to cost for minimization: cost = 1 - weight
    C = 1.0 - W_sq
    row_ind, col_ind = linear_sum_assignment(C)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r >= n_h:
            continue
        if c >= n_c:
            matches.append((r, None, 0.0))
        else:
            matches.append((r, c, float(W[r, c])))
    return matches

def postprocess_splits(
    assignments: List[Tuple[int, Optional[int], float]],
    adjudications: List[Dict],
    min_accept: float = MIN_ACCEPT_CONF,
    fragment_conf: float = FRAGMENT_CONF,
) -> Dict[int, Dict]:
    """
    Converts Hungarian assignments into final decisions with:
      - accept threshold
      - fragment_of extra links where helpful
    Returns dict keyed by human index -> decision dict.
    """
    final: Dict[int, Dict] = {}

    # First pass: accept/reject assigned match
    for i, j, w in assignments:
        adj = adjudications[i]
        if j is None or w < min_accept:
            final[i] = {
                "matched_codebot_index": None,
                "confidence": float(w),
                "link_type": "no_match",
                "rationale": adj.get("rationale", ""),
            }
        else:
            final[i] = {
                "matched_codebot_index": int(j),
                "confidence": float(w),
                "link_type": adj.get("link_type", "exact"),
                "rationale": adj.get("rationale", ""),
            }

    # Second pass: detect many-human -> one-codebot and label as fragment_of for the extras
    codebot_to_humans: Dict[int, List[int]] = {}
    for hi, dec in final.items():
        j = dec["matched_codebot_index"]
        if j is not None:
            codebot_to_humans.setdefault(j, []).append(hi)

    for j, his in codebot_to_humans.items():
        if len(his) <= 1:
            continue
        # Keep the highest confidence as "exact-ish", downgrade others to fragment_of if strong enough.
        his_sorted = sorted(his, key=lambda x: final[x]["confidence"], reverse=True)
        keeper = his_sorted[0]
        for hi in his_sorted[1:]:
            if final[hi]["confidence"] >= fragment_conf:
                final[hi]["link_type"] = "fragment_of"
            else:
                # too weak; drop
                final[hi]["matched_codebot_index"] = None
                final[hi]["link_type"] = "no_match"

        # If keeper was previously fragment_of but is the strongest, bump to exact
        if final[keeper]["link_type"] == "fragment_of":
            final[keeper]["link_type"] = "exact"

    return final


# ---------------------------
# Main per-paper workflow
# ---------------------------

def run_for_paper(
    cfg: PaperConfig,
    adjudicator: str,
    openai_model: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(cfg.codebot_csv):
        raise FileNotFoundError(f"Missing CodeBot CSV: {cfg.codebot_csv}")
    if not os.path.exists(cfg.human_csv):
        raise FileNotFoundError(f"Missing Human CSV: {cfg.human_csv}")

    cb_df = pd.read_csv(cfg.codebot_csv)
    hu_df = pd.read_csv(cfg.human_csv)

    if adjudicator not in {"heuristic", "llm"}:
        raise ValueError(f"Unknown adjudicator: {adjudicator}. Choose 'heuristic' or 'llm'.")

    cb_type = detect_report_type(cb_df)
    hu_type = detect_report_type(hu_df)
    if cb_type != "codebot":
        raise ValueError(f"[{cfg.paper_id}] Expected CodeBot file but detected '{cb_type}'.")
    if hu_type != "human":
        raise ValueError(f"[{cfg.paper_id}] Expected Human file but detected '{hu_type}'.")

    cb_bundles = bundle_report(cb_df, "codebot", cfg.paper_id)
    hu_bundles = bundle_report(hu_df, "human", cfg.paper_id)

    sim, candidates = retrieve_candidates(hu_bundles, cb_bundles, top_k=TOP_K)

    # Adjudicate for each human bundle
    adjudications: List[Dict] = []
    edge_weights: Dict[Tuple[int, int], float] = {}

    for i in range(len(hu_bundles)):
        cand = candidates[i]
        if adjudicator == "llm":
            human_id = hu_bundles.iloc[i].get("analysis_id", i)
            print(f"[LLM] Paper={cfg.paper_id} human_bundle {i+1}/{len(hu_bundles)} analysis_id={human_id} topk={len(cand)}")
            adj = openai_adjudicate_one(
                hu_bundles.iloc[i],
                cb_bundles,
                cand,
                sim[i],
                model=openai_model,
            )
        else:
            adj = heuristic_adjudicate_one(hu_bundles.iloc[i], cb_bundles, cand, sim[i])

        adjudications.append(adj)
        j = adj.get("selected_j", None)
        if j is not None:
            edge_weights[(i, int(j))] = float(adj.get("confidence", 0.0))

    # Global assignment
    assignments = hungarian_assignment(len(hu_bundles), len(cb_bundles), edge_weights)

    # Postprocess splits/thresholding
    final_decisions = postprocess_splits(assignments, adjudications, min_accept=MIN_ACCEPT_CONF, fragment_conf=FRAGMENT_CONF)

    # Build mapping table
    rows = []
    for i in range(len(hu_bundles)):
        dec = final_decisions[i]
        human = hu_bundles.iloc[i]
        cb_idx = dec["matched_codebot_index"]
        codebot = cb_bundles.iloc[cb_idx] if cb_idx is not None else None

        rows.append({
            "paper_id": cfg.paper_id,
            "human_analysis_id": human["analysis_id"],
            "human_uid": human["uid"],
            "codebot_analysis_id": codebot["analysis_id"] if codebot is not None else None,
            "codebot_uid": codebot["uid"] if codebot is not None else None,
            "link_type": dec["link_type"],
            "confidence": dec["confidence"],
            "rationale": dec["rationale"],
            "retrieval_best_cosine": float(sim[i, cb_idx]) if cb_idx is not None else np.nan,
            "codebot_brief_description": codebot["brief_description"] if codebot is not None else None,
        })

    mapping = pd.DataFrame(rows).sort_values(["paper_id","confidence"], ascending=[True, False])

    # Convenience: merged row-level outputs for downstream dimension comparisons
    hu_df_out = hu_df.copy()
    hu_df_out["analysis_id"] = hu_df_out["analysis_id"].astype(str).str.strip()

    cb_df_out = cb_df.copy()
    cb_df_out["analysis_id"] = cb_df_out["analysis_id"].astype(str).str.strip()

    # Add normalized dimension on raw row-level data (so we can join by dimension)
    if "dimension" in hu_df_out.columns:
        hu_df_out["dimension_norm"] = hu_df_out["dimension"].apply(normalize_dimension)
    else:
        hu_df_out["dimension"] = ""
        hu_df_out["dimension_norm"] = "other"

    if "dimension" in cb_df_out.columns:
        cb_df_out["dimension_norm"] = cb_df_out["dimension"].apply(normalize_dimension)
    else:
        cb_df_out["dimension"] = ""
        cb_df_out["dimension_norm"] = "other"

    # Attach codebot match info to each human row (via human_analysis_id)
    mapping_small = mapping[["paper_id", "human_analysis_id", "codebot_analysis_id", "link_type", "confidence"]].copy()
    hu_merged = hu_df_out.merge(mapping_small, left_on="analysis_id", right_on="human_analysis_id", how="left")

    # Within-source match judgements
    # Human: `result` column where 1=match, 0=no match, NA=unknown/too vague/no info
    human_result_raw = coalesce_any_col(hu_df_out, ["result", "results",  "Result", "match_result", "Match_Result", "match", "Match", "match_status", "Match_Status", "judgement", "Judgement"])
    human_match_judgement = to_match_judgement(human_result_raw)

    # CodeBot: `match_status` column mapped to 1=match, 0=mismatch, NA=other
    codebot_status_raw = coalesce_any_col(cb_df_out, ["match_status", "Match_Status", "result", "results", "Result", "judgement", "Judgement", "match", "Match"])
    codebot_match_judgement = to_match_judgement(codebot_status_raw)

    # ---------------------------
    # Build per-paper left-join output requested by user
    # ---------------------------

    # Human fields (robust to schema)
    human_manuscript_content = coalesce_text_cols(hu_df_out, ["manuscript_content", "manuscript_text", "paper_text", "paper_content"])
    human_manuscript_location = coalesce_text_cols(hu_df_out, ["manuscript_location", "paper_location", "location", "section"])
    human_code_content = coalesce_text_cols(hu_df_out, ["code_content", "code_text", "analysis_code"])
    human_code_location = human_code_location_series(hu_df_out)

    # Prepare CodeBot per-(analysis_id, dimension_norm) aggregated rows
    cb_brief = coalesce_text_cols(cb_df_out, ["brief_description"])
    cb_manuscript_content = coalesce_text_cols(cb_df_out, ["paper_text", "manuscript_content", "manuscript_text", "paper_content"])
    cb_manuscript_location = coalesce_text_cols(cb_df_out, ["paper_location", "manuscript_location", "location", "section"])
    cb_code_content = coalesce_text_cols(cb_df_out, ["code_text", "code_content", "analysis_code"])
    cb_code_location = codebot_code_location_series(cb_df_out)

    cb_rows = pd.DataFrame({
        "codebot_analysis_id": cb_df_out["analysis_id"].astype(str).str.strip(),
        "dimension_norm": cb_df_out["dimension_norm"],
        "codebot_brief_description": cb_brief,
        "codebot_manuscript_content": cb_manuscript_content,
        "codebot_manuscript_location": cb_manuscript_location,
        "codebot_code_content": cb_code_content,
        "codebot_code_location": cb_code_location,
        "codebot_match_judgement": codebot_match_judgement,
    })

    # Aggregate in case there are multiple rows per (analysis, dimension)
    def _agg_text(s: pd.Series) -> str:
        vals = [v.strip() for v in s.astype(str).tolist() if v and str(v).strip()]
        # De-duplicate while preserving order
        seen = set()
        out_vals = []
        for v in vals:
            if v in seen:
                continue
            seen.add(v)
            out_vals.append(v)
        return "\n\n".join(out_vals)

    def _agg_match_judgement(s: pd.Series):
        vals = [v for v in s.tolist() if not pd.isna(v)]
        # If any mismatch observed -> mismatch; else if any match observed -> match; else NA
        if any(int(v) == 0 for v in vals):
            return 0
        if any(int(v) == 1 for v in vals):
            return 1
        return pd.NA

    cb_agg = cb_rows.groupby(["codebot_analysis_id", "dimension_norm"], dropna=False).agg({
        "codebot_brief_description": lambda s: _agg_text(s) if _agg_text(s) else "",
        "codebot_manuscript_content": _agg_text,
        "codebot_manuscript_location": _agg_text,
        "codebot_code_content": _agg_text,
        "codebot_code_location": _agg_text,
        "codebot_match_judgement": _agg_match_judgement,
    }).reset_index()

    cb_agg["codebot_match_judgement"] = cb_agg["codebot_match_judgement"].astype("Int64")

    # Start from human rows (left side), and join in match decision + codebot analysis id
    left = pd.DataFrame({
        "analysis_id": hu_df_out["analysis_id"].astype(str).str.strip(),
        "dimension": hu_df_out["dimension"].fillna("").astype(str),
        "dimension_norm": hu_df_out["dimension_norm"].fillna("other").astype(str),
        "human_manuscript_content": human_manuscript_content,
        "human_manuscript_location": human_manuscript_location,
        "human_code_content": human_code_content,
        "human_code_location": human_code_location,
        "human_match_judgement": human_match_judgement,
    })

    left = left.merge(
        mapping_small[["human_analysis_id", "codebot_analysis_id", "link_type", "confidence"]],
        left_on="analysis_id",
        right_on="human_analysis_id",
        how="left",
    )

    # Join in the matching CodeBot info by (codebot_analysis_id, dimension_norm)
    left = left.merge(
        cb_agg,
        on=["codebot_analysis_id", "dimension_norm"],
        how="left",
    )

    # Final column ordering as requested
    left_join = left[[
        "analysis_id",
        "dimension",
        "human_manuscript_content",
        "human_manuscript_location",
        "human_code_content",
        "human_code_location",
        "human_match_judgement",
        "codebot_brief_description",
        "codebot_manuscript_content",
        "codebot_manuscript_location",
        "codebot_code_content",
        "codebot_code_location",
        "codebot_match_judgement",
    ]].copy()

    # Agreement: 1 if both non-missing and equal; 0 if both non-missing and different; NA otherwise
    agree = (left_join["human_match_judgement"].notna()) & (left_join["codebot_match_judgement"].notna())
    left_join["codebot_human_agreement"] = pd.Series([pd.NA] * len(left_join), index=left_join.index, dtype="Int64")
    left_join.loc[agree, "codebot_human_agreement"] = (left_join.loc[agree, "human_match_judgement"].astype(int) == left_join.loc[agree, "codebot_match_judgement"].astype(int)).astype(int)

    # Drop helper columns
    # (dimension_norm, human_analysis_id, confidence are intentionally omitted from the requested output)

    return mapping, hu_merged, cb_df_out, left_join


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match CodeBot vs Human reports (Strategy D).")
    parser.add_argument(
        "--adjudicator",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Which adjudicator to use for matching: 'heuristic' (default) or 'llm'.",
    )
    parser.add_argument(
        "--openai-model",
        default=OPENAI_MODEL,
        help="OpenAI model name to use when --adjudicator llm is selected.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_maps = []
    all_hu_merged = []
    all_cb = []
    all_left_joins = []

    for cfg in PAPERS:
        m, hu_m, cb, left_join = run_for_paper(cfg, adjudicator=args.adjudicator, openai_model=args.openai_model)
        all_maps.append(m)
        all_hu_merged.append(hu_m)
        all_cb.append(cb)
        all_left_joins.append(left_join)

        os.makedirs("out", exist_ok=True)
        m.to_csv(f"out/{cfg.paper_id}_mapping.csv", index=False)
        hu_m.to_csv(f"out/{cfg.paper_id}_human_with_matches.csv", index=False)
        cb.to_csv(f"out/{cfg.paper_id}_codebot_rows.csv", index=False)
        left_join.to_csv(f"out/{cfg.paper_id}_combined_report.csv", index=False)

    mapping = pd.concat(all_maps, ignore_index=True)
    human_rows_with_matches = pd.concat(all_hu_merged, ignore_index=True)
    codebot_rows = pd.concat(all_cb, ignore_index=True)

    os.makedirs("out", exist_ok=True)
    mapping.to_csv("out/mapping.csv", index=False)
    human_rows_with_matches.to_csv("out/human_with_matches.csv", index=False)
    codebot_rows.to_csv("out/codebot_rows.csv", index=False)

    # Quick diagnostics
    print("\n=== Matching summary ===")
    print(mapping["link_type"].value_counts(dropna=False))
    print("\nAccepted matches (>= threshold):", (mapping["confidence"] >= MIN_ACCEPT_CONF).sum())
    print("Total human analyses:", len(mapping))
    print("\nAdjudicator:", args.adjudicator)
    if args.adjudicator == "llm":
        print("Note: using LLM adjudication; ensure OPENAI_API_KEY is available (e.g., via .env or exported env var).")
        print("OpenAI model:", args.openai_model)
    print("\nWrote:")
    print(" - out/mapping.csv")
    print(" - out/human_with_matches.csv")
    print(" - out/codebot_rows.csv")
    print(" - out/<paper_id>_combined_report.csv")


if __name__ == "__main__":
    main()
