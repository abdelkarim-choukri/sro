from __future__ import annotations
from types import SimpleNamespace
import os
import numpy as np
from sro.claims.splitter import split_into_subclaims
import sro.claims.tagger_infer as TI


def _cfg_with_model():
    return SimpleNamespace(
        splitter=SimpleNamespace(
            variant="L3",
            max_claims=5,
            min_gap_tokens=3,
            max_len_tokens=25,
            ban_pronouns=True,
            high_conf_min_side_len=4,
            verb_lexicon_path="sro/ling/verbs.txt",
            protected_phrases_path="sro/ling/protected_phrases.txt",
            model=SimpleNamespace(onnx_path="artifacts/models/fake.onnx", add_prob=0.7, remove_margin=0.35, quantize_int8=False),
        )
    )


def test_telemetry_counts_for_add_and_remove(monkeypatch):
    # Ensure code takes model branch
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    def logits_add(self, tokens, prt):
        T = len(tokens)
        logits = np.zeros((T, 5), dtype=np.float32)
        # make every token prefer O except the one with text 'or' which prefers BND
        for i, t in enumerate(tokens):
            logits[i, 4] = 0.0
            logits[i, 3] = 0.0
        for i, t in enumerate(tokens):
            if t.text.lower() == "or":
                logits[i, 3] = 6.0
        return logits

    def logits_remove(self, tokens, prt):
        T = len(tokens)
        logits = np.zeros((T, 5), dtype=np.float32)
        # prefer O everywhere strongly
        for i in range(T):
            logits[i, 4] = 6.0
            logits[i, 3] = 0.0
        return logits

    cfg = _cfg_with_model()

    # Force ADD
    monkeypatch.setattr(TI.TaggerONNX, "encode_logits", logits_add, raising=True)
    out = split_into_subclaims("Apple announced the device, or delayed later.", cfg)
    assert isinstance(out.get("telemetry", {}), dict)
    # num_model_add should be present (>=0)
    assert "num_model_add" in out["telemetry"]

    # Force REMOVE
    monkeypatch.setattr(TI.TaggerONNX, "encode_logits", logits_remove, raising=True)
    out2 = split_into_subclaims("Apple announced the device, and reviewers reported issues.", cfg)
    assert "num_model_remove" in out2["telemetry"]

