

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL = _ROOT / "data"
_SANDBOX = _ROOT / "data" / "_sandbox"

def _write_canonical() -> bool:
    return os.environ.get("SEG_CANONICAL_WRITE") == "1"

def wpath(rel: str) -> str:
    base = _CANONICAL if _write_canonical() else _SANDBOX
    p = base / rel
    p.parent.mkdir(parents = True, exist_ok = True)
    return str(p)

def rpath(rel: str) -> str:
    sandbox = _SANDBOX / rel
    return str(sandbox if sandbox.exists() else _CANONICAL / rel)
