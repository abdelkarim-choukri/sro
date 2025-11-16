from __future__ import annotations
from types import SimpleNamespace
from sro.claims.splitter import split_into_subclaims

CFG = SimpleNamespace(
    splitter=SimpleNamespace(
        variant="L3",
        max_claims=5,
        min_gap_tokens=3,
        max_len_tokens=25,
        ban_pronouns=True,
        verb_lexicon_path="sro/ling/verbs.txt",
        protected_phrases_path="sro/ling/protected_phrases.txt",
        high_conf_min_side_len=4,
        model=SimpleNamespace(onnx_path="artifacts/models/splitter_miniL3.onnx", add_prob=0.70, remove_margin=0.35, quantize_int8=True),
    )
)


def test_high_conf_rules_skip_model(monkeypatch):
    # Force os.path.exists to pretend model exists, but ensure bypass happens
    import os
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    # A sentence that yields a high-conf coord split
    q = "Apple announced the iPhone 15 Pro, and reviewers reported reduced weight."
    out = split_into_subclaims(q, CFG)
    # If model had been called and added spurious bnds we’d see differences; here just sanity:
    assert 1 <= len(out["claims"]) <= 5
    assert out["telemetry"]["num_rule_bnd"] >= 1
