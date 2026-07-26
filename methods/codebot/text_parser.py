from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

import requests

from .models import PaperText


def _strip_references(text: str) -> str:
    """Remove reference sections heuristically."""
    pattern = re.compile(r"(?:^|\n)(References|Bibliography|Cited Works)\s*\n", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return text[: match.start()].strip()
    return text


def parse_with_dpt2(pdf_path: Path, api_key: Optional[str] = None, model: str = "dpt-2-latest") -> PaperText:
    token = api_key or os.getenv("CODEBOT_DPT2_API_KEY") or os.getenv("LANDING_AI_API_KEY")
    if not token:
        raise ValueError("DPT-2 API key missing (set CODEBOT_DPT2_API_KEY or LANDING_AI_API_KEY).")

    url = "https://api.va.eu-west-1.landing.ai/v1/ade/parse"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"model": model}

    with pdf_path.open("rb") as f:
        files = {"document": f}
        resp = requests.post(url, files=files, data=data, headers=headers, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    text = payload.get("text") or payload.get("fulltext") or json.dumps(payload)
    text = _strip_references(text)
    return PaperText(paper_id=pdf_path.stem, text=text, source="dpt2", meta={"model": model, "raw": payload})


def parse_with_grobid(pdf_path: Path, grobid_url: str = "https://kermitt2-grobid.hf.space/api/processFulltextDocument") -> PaperText:
    with pdf_path.open("rb") as f:
        files = {"input": f}
        resp = requests.post(grobid_url, files=files, timeout=120)
    resp.raise_for_status()
    xml_content = resp.text

    namespace = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(xml_content)
    body = root.find(".//tei:body", namespace)
    text = "".join(body.itertext()).strip() if body is not None else xml_content
    text = _strip_references(text)
    return PaperText(paper_id=pdf_path.stem, text=text, source="grobid", meta={"grobid_url": grobid_url})


def parse_with_pypdf(pdf_path: Path) -> PaperText:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError("pypdf is required for local PDF parsing. pip install pypdf") from exc

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = _strip_references("\n".join(pages))
    return PaperText(paper_id=pdf_path.stem, text=text, source="pypdf", meta={"pages": len(pages)})


def parse_text_file(path: Path) -> PaperText:
    text = path.read_text(encoding="utf-8", errors="replace")
    return PaperText(paper_id=path.stem, text=text, source="provided-text")


def parse_pdf(
    paper_path: str | Path,
    method: Literal["dpt2", "grobid", "pypdf", "text"] = "grobid",
    *,
    api_key: Optional[str] = None,
    grobid_url: str = "https://kermitt2-grobid.hf.space/api/processFulltextDocument",
) -> PaperText:
    path = Path(paper_path)
    if method == "text":
        return parse_text_file(path)
    if method == "dpt2":
        return parse_with_dpt2(path, api_key=api_key)
    if method == "grobid":
        return parse_with_grobid(path, grobid_url=grobid_url)
    if method == "pypdf":
        return parse_with_pypdf(path)
    raise ValueError(f"Unknown parsing method: {method}")


__all__ = [
    "parse_pdf",
    "parse_with_dpt2",
    "parse_with_grobid",
    "parse_with_pypdf",
    "parse_text_file",
]
