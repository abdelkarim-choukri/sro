# tests/test_traces_more.py
import json
from pathlib import Path
from sro.logging.traces import emit_trace

def _row(qid: str, stage: str, stop: str) -> dict:
    return {
        "qid": qid,
        "stage": stage,
        "stop_reason": stop,
        "best_so_far": 0.0,
        "top_ub": 0.0,
        "budget_left_norm": 1.0,
        "frontier_entropy": 0.0,
        "timestamp_iso": "1970-01-01T00:00:00Z",
    }

def test_traces_append_and_schema(tmp_path: Path):
    log = tmp_path / "traces.jsonl"
    emit_trace(_row("q1", "S1", "ACCEPT"), str(log))
    emit_trace(_row("q2", "S2", "REJECT"), str(log))

    # File exists and has two valid JSONL lines
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for ln in lines:
        obj = json.loads(ln)
        for k in (
            "qid",
            "stage",
            "stop_reason",
            "best_so_far",
            "top_ub",
            "budget_left_norm",
            "frontier_entropy",
            "timestamp_iso",
        ):
            assert k in obj
