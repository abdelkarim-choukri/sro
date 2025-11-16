import os
import json
import hashlib
import pytest
from types import SimpleNamespace

from sro.claims.splitter import split_into_subclaims


def _find_model_path():
    p_l3 = "artifacts/models/splitter_miniL3.int8.onnx"
    p_l6 = "artifacts/models/splitter_miniL6.int8.onnx"
    return p_l3 if os.path.exists(p_l3) else (p_l6 if os.path.exists(p_l6) else "")


def test_force_flag_runs_model_on_high_conf():
    model_path = _find_model_path()
    if not model_path:
        pytest.skip("No ONNX artifact found; skip force-flag test.")

    # Query likely to be high-confidence for rule-based split (comma + conjunction)
    q = "Apple announced the device, and reviewers reported issues."

    cfg = SimpleNamespace(splitter=SimpleNamespace(min_gap_tokens=3, model=SimpleNamespace(onnx_path=model_path)))

    qid = hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]
    artifact_path = os.path.join(os.getcwd(), "artifacts", "splitter", f"{qid}.jsonl")
    try:
        if os.path.exists(artifact_path):
            os.remove(artifact_path)
    except Exception:
        pass

    # Ensure force flag set -> model_ran True even though high_conf should be True
    os.environ["SPLITTER_FORCE_MODEL"] = "1"
    out_forced = split_into_subclaims(q, cfg)
    assert os.path.exists(artifact_path), "Expected artifact to be created"
    with open(artifact_path, "r", encoding="utf-8") as f:
        rec = json.loads([l for l in f if l.strip()][-1])
    tel = rec.get("telemetry", {})
    assert tel.get("high_conf", False) is True, "Expected high_conf for this query"
    assert tel.get("model_ran", False) is True, "SPLITTER_FORCE_MODEL did not cause model to run"

    # Clear force flag -> model_ran should be False (bypassed)
    os.environ.pop("SPLITTER_FORCE_MODEL", None)
    out_unforced = split_into_subclaims(q, cfg)
    with open(artifact_path, "r", encoding="utf-8") as f:
        rec2 = json.loads([l for l in f if l.strip()][-1])
    tel2 = rec2.get("telemetry", {})
    assert tel2.get("high_conf", False) is True, "Expected high_conf to remain True"
    assert tel2.get("model_ran", False) is False, "Model should not have run when force flag cleared"
