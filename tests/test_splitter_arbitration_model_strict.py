from types import SimpleNamespace
import os
import numpy as np
import sro.claims.tagger_infer as TI
from sro.claims.splitter import split_into_subclaims


CFG = SimpleNamespace(
    splitter=SimpleNamespace(
        variant="L3",
        max_claims=5,
        min_gap_tokens=3,
        max_len_tokens=25,
        ban_pronouns=True,
        high_conf_min_side_len=4,
        verb_lexicon_path="sro/ling/verbs.txt",
        protected_phrases_path="sro/ling/protected_phrases.txt",
        model=SimpleNamespace(onnx_path="artifacts/models/splitter_miniL3.onnx", add_prob=0.70, remove_margin=0.35, quantize_int8=True),
    )
)


def _exists(_):
    return True


def test_add_and_remove_counts(monkeypatch):
    # Pretend model file exists so the code tries to run the model shim
    monkeypatch.setattr(os.path, "exists", _exists)

    def logits_add(self, tokens, prt):
        T = len(tokens)
        L = np.zeros((T, 5), np.float32)
        # put a strong boundary score at the token 'or' if present
        i = next((i for i, t in enumerate(tokens) if t.text.lower() == "or"), None)
        if i is None:
            i = min(3, T - 1)
        # class 3 = BND in our shim convention
        L[i, 3] = 6.0
        return L

    monkeypatch.setattr(TI.TaggerONNX, "encode_logits", logits_add)
    out = split_into_subclaims("Apple announced the device, or delayed later.", CFG)
    assert out["telemetry"].get("num_model_add", 0) >= 1

    def logits_remove(self, tokens, prt):
        T = len(tokens)
        L = np.zeros((T, 5), np.float32)
        # strongly prefer NO-BND (class 4 in this shim)
        L[:, 4] = 6.0
        return L

    monkeypatch.setattr(TI.TaggerONNX, "encode_logits", logits_remove)
    out2 = split_into_subclaims("Apple announced the device, and reviewers reported issues.", CFG)
    # we expect some model_remove telemetry (zero is acceptable but should be present)
    assert "num_model_remove" in out2["telemetry"]
    # boundary count should be small in the remove scenario (rules baseline <=2)
    assert len(out2["split_points"]) <= 3
