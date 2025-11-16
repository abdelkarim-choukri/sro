import os
import json
import pytest
from types import SimpleNamespace

from sro.claims.splitter import split_into_subclaims
from sro.claims.tokenizer import tokenize


def _find_model_path():
    p_l3 = "artifacts/models/splitter_miniL3.int8.onnx"
    p_l6 = "artifacts/models/splitter_miniL6.int8.onnx"
    return p_l3 if os.path.exists(p_l3) else (p_l6 if os.path.exists(p_l6) else "")


def test_onnx_guard_determinism_and_timing():
    model_path = _find_model_path()
    if not model_path:
        pytest.skip("No ONNX artifact found; skip guard test.")

    # Choose a query that tends to avoid high-conf bypass (avoid comma/clear rule break)
    q = "Apple announced the device and reviewers reported issues or delays."

    # Minimal cfg threading the onnx path
    cfg = SimpleNamespace(
        splitter=SimpleNamespace(
            min_gap_tokens=3,
            model=SimpleNamespace(onnx_path=model_path),
        )
    )

    # deterministic qid -> artifact path
    import hashlib
    qid = hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]
    artifact_path = os.path.join(os.getcwd(), "artifacts", "splitter", f"{qid}.jsonl")
    try:
        if os.path.exists(artifact_path):
            os.remove(artifact_path)
    except Exception:
        pass

    outs, adds, rems, model_ms, highs = [], [], [], [], []
    for _ in range(3):
        out = split_into_subclaims(q, cfg)
        outs.append(out)

        # Read last artifact record
        assert os.path.exists(artifact_path), "Expected artifact file to be written by splitter"
        with open(artifact_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            assert lines, "No artifact lines found"
            rec = json.loads(lines[-1])

        tel = rec.get("telemetry", {})
        timings = rec.get("timings", {})  # seconds
        adds.append(int(tel.get("num_model_add", 0)))
        rems.append(int(tel.get("num_model_remove", 0)))
        highs.append(bool(tel.get("high_conf", False)))
        model_ms.append(float(timings.get("model", 0.0)) * 1000.0)

    # Full determinism: identical returns
    assert outs[0] == outs[1] == outs[2], "Public return payload not deterministic across runs"

    # Determinism on model counters & split points
    assert adds[0] == adds[1] == adds[2], f"num_model_add not stable: {adds}"
    assert rems[0] == rems[1] == rems[2], f"num_model_remove not stable: {rems}"
    assert outs[0]["split_points"] == outs[1]["split_points"] == outs[2]["split_points"], "split_points differ across runs"

    # Timing gate
    budget_ms = float(os.environ.get("SPLITTER_MODEL_BUDGET_MS", "250"))
    if all(m == 0.0 for m in model_ms):
        # Likely a high-conf bypass; require all three telemetry flags to be True
        assert all(highs), f"Model times are zero but not all runs report high_conf: highs={highs}, model_ms={model_ms}"
        pytest.skip("Model bypassed due to all-high-confidence rule candidates; skipping timing budget check.")
    else:
        med = sorted(model_ms)[1]
        assert med > 0.0 and med <= budget_ms, f"Median model time {med:.1f}ms exceeds budget {budget_ms}ms"
