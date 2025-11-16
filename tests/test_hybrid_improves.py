from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import os
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
        model=SimpleNamespace(onnx_path="artifacts/models/splitter_miniL3.onnx", add_prob=0.70, remove_margin=0.35, quantize_int8=True),
    )
)


def _fake_exists(_):
    return True


def test_hybrid_adds_missing_bnd(monkeypatch):
    # Pretend model exists
    monkeypatch.setattr(os.path, "exists", _fake_exists)

    q = "Apple announced the device and testing continued later."

    def fake_encode_logits(self, tokens, prt):
        T = len(tokens); logits = np.zeros((T,5), dtype=np.float32)
        idx = next(i for i,t in enumerate(tokens) if t.text.lower()=="and")
        logits[idx,3]=5.0; logits[idx,4]=0.0
        return logits

    monkeypatch.setattr(TI.TaggerONNX, "encode_logits", fake_encode_logits)

    # rules-only
    cfg_rules = SimpleNamespace(**{"splitter": SimpleNamespace(**vars(CFG.splitter))})
    cfg_rules.splitter.model.onnx_path = ""
    rules_out = split_into_subclaims(q, cfg_rules)

    # hybrid
    hybrid_out = split_into_subclaims(q, CFG)

    assert len(hybrid_out["split_points"]) >= len(rules_out["split_points"])
    assert len(hybrid_out["claims"]) >= len(rules_out["claims"]) 
