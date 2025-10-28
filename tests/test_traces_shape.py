# tests/test_traces_shape.py
from __future__ import annotations

import json
import os
from typing import Any, Dict

from sro.logging.traces import emit_trace

REQUIRED = {
    "qid",
    "stage",
    "stop_reason",
    "best_so_far",
    "top_ub",
    "budget_left_norm",
    "frontier_entropy",
    "timestamp_iso",
}

def test_trace_row_schema(tmp_path):
    log_dir = tmp_path / "artifacts" / "logs"
    log_path = log_dir / "traces.jsonl"

    row = {
        "qid": "demo-1",
        "stage": "S1",
        "stop_reason": "ACCEPT",
        "best_so_far": 0.75,
        "top_ub": 0.82,
        "budget_left_norm": 0.5,
        "frontier_entropy": 0.12,
        # omit timestamp to exercise default
    }
    emit_trace(row, str(log_path))

    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) >= 1
    got = json.loads(lines[-1])

    # All required keys present
    assert REQUIRED.issubset(got.keys())

    # Types sanity
    assert isinstance(got["qid"], str)
    assert isinstance(got["stage"], str)
    assert isinstance(got["stop_reason"], str)
    assert isinstance(got["best_so_far"], (int, float))
    assert isinstance(got["top_ub"], (int, float))
    assert isinstance(got["budget_left_norm"], (int, float))
    assert isinstance(got["frontier_entropy"], (int, float))
    assert isinstance(got["timestamp_iso"], str)
