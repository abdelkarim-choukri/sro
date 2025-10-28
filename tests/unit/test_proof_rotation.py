# tests/unit/test_proof_rotation.py
from __future__ import annotations

import json
from pathlib import Path


def test_append_and_force_rotate(tmp_path, monkeypatch):
    base: Path = tmp_path / "proofs.jsonl"
    # lazy import to avoid side-effects
    from sro.prover.logio import append_jsonl

    # append one record
    append_jsonl(base, {"k": 1})
    data = base.read_text(encoding="utf-8").splitlines()
    assert len(data) == 1 and '"k": 1' in data[0]

    # force rotation and append another
    monkeypatch.setenv("SRO_FORCE_ROTATE", "1")
    append_jsonl(base, {"k": 2})
    monkeypatch.delenv("SRO_FORCE_ROTATE", raising=False)

    # now base should only have the second line; a rotated file must exist
    files = list(tmp_path.iterdir())
    rotated = [p for p in files if p.name.startswith("proofs.") and p.suffix == ".jsonl"]
    assert rotated, f"no rotated file, files={files}"

    data2 = base.read_text(encoding="utf-8").splitlines()
    assert len(data2) == 1 and '"k": 2' in data2[0]
