from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TABULARIUM_EVAL_DATA") or REPO_ROOT / "dev-corpus")
OUT_DIR = DATA_ROOT / "evals" / "v1v2"


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def out_path(name: str) -> Path:
    return ensure_out_dir() / name
