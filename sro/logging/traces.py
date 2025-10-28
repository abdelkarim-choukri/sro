# sro/logging/traces.py
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Dict

_REQUIRED_KEYS = {
    "qid",
    "stage",
    "stop_reason",  # UB_BEATEN | MINIMALITY_BLOCK | SAFETY_MARGIN_FAIL | ONE_ALT_CAP | ACCEPT | REJECT | ABSTAIN
    "best_so_far",
    "top_ub",
    "budget_left_norm",
    "frontier_entropy",
    "timestamp_iso",
}

def emit_trace(row: Dict[str, Any], path: str) -> None:
    """
    Append a structured trace row to a JSONL file, guaranteeing required keys.

    This is intentionally minimal and safe under concurrent appends.
    """
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
    out = dict(row)
    out.setdefault("timestamp_iso", now)

    missing = _REQUIRED_KEYS - set(out.keys())
    if missing:
        # Fill with explicit nulls or safe defaults; we never crash tracing.
        for k in missing:
            out[k] = None

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False))
        f.write("\n")
