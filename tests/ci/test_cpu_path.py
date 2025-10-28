# tests/ci/test_cpu_path.py
import os

def test_cpu_only_path_fast():
    # Enforce CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SRO_DEVICE"] = "cpu"

    # Import here so the env vars take effect
    from sro.nli.backend import NLIBackend  # type: ignore

    nli = NLIBackend()
    out = nli.score_pairs(["a"], ["a"], batch_size=1)
    assert "probs" in out
    assert hasattr(out["probs"], "shape")
    assert int(out["probs"].shape[0]) == 1
