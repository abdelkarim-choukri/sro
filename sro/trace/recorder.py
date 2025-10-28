# sro/trace/recorder.py
"""
Does:
    Lightweight, structured trace recorder for SRO-Proof. Writes JSONL lines to a file.
Inputs:
    - run metadata (run_id, seed, profile)
    - events: dicts with required keys:
        { "ts": iso, "type": str, "stage": str, "data": {...}, "run_id": str }
Outputs:
    - artifacts/logs/traces.jsonl (default) or a path you pass.
Notes:
    - Pure Python, no external deps.
    - Thread-safe file writes via OS-level append semantics (open per write).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


@dataclass(frozen=True)
class TraceEvent:
    ts: str
    type: str
    stage: str
    data: dict[str, Any]
    run_id: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TraceRecorder:
    """
    Minimal trace writer. Open-per-write (atomic-ish on Windows too).
    """
    def __init__(self, out_path: str, *, run_id: str | None = None, meta: dict[str, Any] | None = None) -> None:
        self.out_path = out_path
        self.run_id = run_id or str(uuid.uuid4())
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        # write session header
        self.log("session_start", stage="S0", data={"meta": meta or {}})

    def log(self, ev_type: str, *, stage: str, data: dict[str, Any]) -> None:
        ev = TraceEvent(ts=_now_iso(), type=ev_type, stage=stage, data=data, run_id=self.run_id)
        line = ev.to_json()
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def close(self) -> None:
        self.log("session_end", stage="S9", data={})
