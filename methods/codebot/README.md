# CodeBot Flow (modular CLI + Python)

This folder contains a modular rewrite of the `CodeBot_flow.ipynb` notebook. It supports:
- Single paper + codebase runs
- Multi-paper runs via CSV
- Staged processing (paper → code → judge)

## Install

Create a venv and install deps:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install openai requests pypdf
```

## Environment variables

If a `.env` file is found by walking up from `methods/codebot/codebot`, it is loaded automatically (supports `python-dotenv` if installed, otherwise a simple fallback loader).

- `CODEBOT_OPENAI_API_KEY` (or `OPENAI_API_KEY`) **required**
- `CODEBOT_DPT2_API_KEY` / `LANDING_AI_API_KEY` if using DPT-2 parsing
- Optional defaults:
  - `CODEBOT_MODEL` (default: `gpt-5.1`)
  - `CODEBOT_REASONING` (default: `medium`)

## CLI usage

### Single run

```
python -m codebot.cli run-single \
  --paper /path/to/paper.pdf \
  --code-root /path/to/code \
  --paper-id my_paper_id \
  --parser pypdf \
  --mode combined \
  --model gpt-5.1 \
  --reasoning-effort medium \
  --output-dir data/raw/codebot/output \
  --log-level INFO
```

### Multi run (CSV)

CSV must include `paper_path,code_root` and optionally `paper_id`.

```
paper_path,code_root,paper_id
/abs/path/paper1.pdf,/abs/path/code1,paper1
/abs/path/paper2.pdf,/abs/path/code2,
```

```
python -m codebot.cli run-multi \
  --pairs-csv pairs.csv \
  --parser pypdf \
  --mode combined \
  --parallelism 5 \
  --output-dir data/raw/codebot/output \
  --log-level INFO
```

If `{paper_id}.json` and `{paper_id}.csv` already exist in the output directory, that paper is skipped on re-run.

### Staged run (paper → code → judge)

```
python -m codebot.cli stage paper \
  --paper /path/to/paper.pdf \
  --parser pypdf \
  --state-dir .codebot_state/my_paper \
  --reasoning-effort medium

python -m codebot.cli stage code \
  --code-root /path/to/code \
  --state-dir .codebot_state/my_paper \
  --reasoning-effort medium

python -m codebot.cli stage judge \
  --state-dir .codebot_state/my_paper \
  --output-dir data/raw/codebot/output \
  --reasoning-effort medium
```

## Python API usage

### Single run

```
from pathlib import Path
from codebot_flow.runner import RunConfig, run_single

cfg = RunConfig(
    paper_path=Path("/path/to/paper.pdf"),
    code_root=Path("/path/to/code"),
    paper_id="paper_id",
    parser="pypdf",      # grobid|dpt2|pypdf|text
    mode="staged",       # staged|combined
    model="gpt-5.1",
    reasoning_effort="medium",
    output_dir=Path("pilot_reports"),
)
run_single(cfg)
```

### Multi run

```
from codebot_flow.runner import run_multi

rows = [
    {"paper_path": "/path/paper1.pdf", "code_root": "/path/code1", "paper_id": "paper1"},
    {"paper_path": "/path/paper2.pdf", "code_root": "/path/code2"},
]
run_multi(rows, parser="pypdf", mode="combined", output_dir=Path("pilot_reports"))
```

## Options

### Parsers
- `grobid` (remote) – default
- `dpt2` (remote) – requires API key
- `pypdf` (local) – install `pypdf`
- `text` – read a plain text file

### Comparison modes
- `combined` – single LLM call per analysis × dimension
- `staged` – separate paper evidence, code evidence, then adjudication

### Logging
- CLI: `--log-level DEBUG|INFO|WARNING|ERROR`
- Python: add `logging.basicConfig(...)` in your script

## Outputs

Default output directory: `data/raw/codebot/output`.

Per-paper outputs (in `--output-dir`):
- `{paper_id}.json` and `{paper_id}.csv`
- Optional intermediates if enabled:
  - `{paper_id}_paper_evidence.json`
  - `{paper_id}_code_evidence.json`
  - `{paper_id}_matches.json`

Multi-run also writes aggregate:
- `all_papers.json` and `all_papers.csv`

## Notes

- Staged runs persist intermediates in `--state-dir`.
- The comparison pipeline is structured to support a future RAG-based strategy; swap in a custom strategy in `codebot_flow/comparison.py`.
