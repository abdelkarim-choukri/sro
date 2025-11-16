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
    )
)

def test_determinism_50_runs():
    q = "Apple announced the iPhone 15 Pro, and reviewers said it sold well."
    outs = [split_into_subclaims(q, CFG) for _ in range(50)]
    # all identical
    first = outs[0]
    for o in outs[1:]:
        assert o == first
