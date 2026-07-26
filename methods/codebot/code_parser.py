from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Set

from .models import CodeFile

# For CodeBot Psychology, we focus on R-related files only
DEFAULT_EXTENSIONS: Set[str] = {
    ".r",
    ".R",
    ".Rmd",
    ".rmd",
    ".qmd",
    ".Qmd",
    ".Rnw",
    ".rnw",
    # ".py",
    # ".ipynb",
    # ".txt",
}


def _iter_files(root: Path, extensions: Set[str]) -> Iterable[Path]:
    if root.is_file():
        yield root
    else:
        for path in root.rglob("*"):
            if path.is_file() and (not extensions or path.suffix in extensions):
                yield path


def load_code_files(root: str | Path, extensions: Sequence[str] | None = None, max_bytes: int | None = None) -> List[CodeFile]:
    root_path = Path(root)
    ext_set = set(extensions) if extensions is not None else DEFAULT_EXTENSIONS

    files: List[CodeFile] = []
    for file_path in _iter_files(root_path, ext_set):
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # skip unreadable files but continue
            continue
        if max_bytes and len(text.encode("utf-8")) > max_bytes:
            text = text[: max_bytes]
        files.append(CodeFile(path=str(file_path), content=text))
    return files


__all__ = ["load_code_files", "DEFAULT_EXTENSIONS"]
