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

def test_rules_only_pipeline_three_hops():
    q = "Did Apple announce the iPhone 15 Pro, and did it ship in September, or did reviews report reduced weight?"
    out = split_into_subclaims(q, CFG)
    # claims exist and are <= 5
    assert 2 <= len(out["claims"]) <= 5
    # split points are char indices inside the string
    for c in out["split_points"]:
        assert 0 < c < len(q)
    # prt non-empty for 'September' (month detection)
    assert out["telemetry"]["num_prt"] >= 1
    # no claim starts with a banned pronoun if enabled
    if CFG.splitter.ban_pronouns:
        assert all(not cl["text"].lower().split()[0] in {"i","we","you","he","she","they","it","this","that","these","those"}
                   for cl in out["claims"])

def test_subject_cloning_minimal():
    q = "Apple announced the iPhone 15 Pro, and is expected to ship widely."
    out = split_into_subclaims(q, CFG)
    # Second claim should start with a cloned subject if AUX-led
    if len(out["claims"]) >= 2:
        c2_claim = out["claims"][1]
        # Prefer display_text (augmented) when available; fall back to canonical text.
        c2 = c2_claim.get("display_text", c2_claim.get("text", "")).lower()
        assert c2.startswith("apple ") or c2.startswith("the ") or c2.startswith("a ") or c2.startswith("an ")

def test_filter_single_predicate_and_len():
    q = "Apple announced and shipped and marketed and expanded and reported the results."
    out = split_into_subclaims(q, CFG)
    # Filters should drop multi-predicate fragments
    assert len(out["claims"]) >= 1
    for cl in out["claims"]:
        assert len(cl["text"].split()) <= 25
