# sro/prover/logio.py
from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime, timezone

DAY_FMT = "%Y-%m-%d"
DEFAULT_MAX_MB = 20  # rotate when >20MB unless overridden

def _utc_today():
    return datetime.now(timezone.utc).date()

def _file_utc_day(p: Path):
    try:
        # Use timezone-aware UTC (fixes deprecation)
        return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).date()
    except Exception:
        return _utc_today()


def _get_max_bytes() -> int:
    env = os.getenv("SRO_PROOFS_MAX_MB")
    if env:
        try:
            mb = max(1, int(env))
            return mb * 1024 * 1024
        except Exception:
            pass
    return DEFAULT_MAX_MB * 1024 * 1024

def _next_rotated_name(base: Path, day: str) -> Path:
    # proofs.jsonl -> proofs.YYYY-MM-DD.jsonl (or with numeric suffix if exists)
    stem = base.stem        # "proofs"
    suffix = base.suffix    # ".jsonl"
    candidate = base.with_name(f"{stem}.{day}{suffix}")
    if not candidate.exists():
        return candidate
    # add numeric suffix: proofs.YYYY-MM-DD.1.jsonl, .2.jsonl, ...
    i = 1
    while True:
        c = base.with_name(f"{stem}.{day}.{i}{suffix}")
        if not c.exists():
            return c
        i += 1

def rotate_if_needed(base: Path) -> None:
    """
    Rotate base file when:
      - file size > threshold (DEFAULT_MAX_MB or SRO_PROOFS_MAX_MB), or
      - file's UTC day != today's UTC day, or
      - forced via env SRO_FORCE_ROTATE=1 (useful in tests)
    """
    if not base.exists():
        return
    try:
        force = os.getenv("SRO_FORCE_ROTATE") == "1"
        size = base.stat().st_size
        max_bytes = _get_max_bytes()
        day_file = _file_utc_day(base)
        day_now = _utc_today()
        if force or size > max_bytes or day_file != day_now:
            rot = _next_rotated_name(base, day_file.strftime(DAY_FMT))
            # ensure parent exists; then rename atomically
            base.parent.mkdir(parents=True, exist_ok=True)
            base.replace(rot)
    except Exception:
        # Never crash caller on rotation problems
        return

def append_jsonl(base: Path, obj: dict) -> None:
    """
    Append one JSON record as a single line to `base`, rotating if needed.
    Ensures '\n' newline on Windows and flushes write.
    """
    base.parent.mkdir(parents=True, exist_ok=True)
    rotate_if_needed(base)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with base.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)

def latest_lines(base: Path, n: int) -> list[str]:
    """
    Return the last n lines from `base` (does not scan rotated files).
    """
    if not base.exists():
        return []
    try:
        data = base.read_text(encoding="utf-8")
        lines = data.splitlines()
        return lines[-n:]
    except Exception:
        return []
