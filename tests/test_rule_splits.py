from __future__ import annotations
from types import SimpleNamespace
from sro.claims.tokenizer import tokenize
from sro.claims.rules import propose_rule_splits, propose_protected_spans

CFG = SimpleNamespace(
    splitter=SimpleNamespace(
        min_gap_tokens=3,
        max_len_tokens=25,
        high_conf_min_side_len=4,
        verb_lexicon_path="sro/ling/verbs.txt",
    )
)


def _prep(q: str):
    toks = tokenize(q)
    prt = propose_protected_spans(toks, CFG)
    return toks, prt


def test_coord_and_or_top_level():
    q = "Did Apple announce the iPhone 15 Pro, and did it ship in September, or was it delayed?"
    toks, prt = _prep(q)
    bnds, meta = propose_rule_splits(toks, prt, CFG)
    # expect boundaries after 'and' and 'or'
    chars = [m["char"] for m in meta["rule_bnds"]]
    cuts = [toks[i["i_tok"]].text for i in meta["rule_bnds"]]  # sanity
    assert any(toks[_["i_tok"]].text.lower() == "and" for _ in meta["rule_bnds"])
    assert any(toks[_["i_tok"]].text.lower() == "or" for _ in meta["rule_bnds"])
    # min-gap enforced
    # convert chars back to token indices and assert spacing >= 3
    idxs = []
    for c in bnds:
        for j, t in enumerate(toks):
            if t.end == c:
                idxs.append(j)
                break
    if len(idxs) >= 2:
        assert all(idxs[k + 1] - idxs[k] >= 3 for k in range(len(idxs) - 1))


def test_bare_and_requires_verbs():
    q = "Apple announced the device and the titanium frame."
    toks, prt = _prep(q)
    bnds, meta = propose_rule_splits(toks, prt, CFG)
    # No split: RHS lacks a clear verb
    assert meta["num_rule_bnd"] == 0

    q2 = "Apple announced the device and reviewers reported reduced weight."
    toks, prt = _prep(q2)
    bnds, meta = propose_rule_splits(toks, prt, CFG)
    assert meta["num_rule_bnd"] >= 1
    assert any(toks[it["i_tok"]].text.lower() == "and" for it in meta["rule_bnds"])


def test_clause_markers_with_verb_lookahead():
    q = "Because reviewers reported issues, Apple added a fix."
    toks, prt = _prep(q)
    bnds, meta = propose_rule_splits(toks, prt, CFG)
    # boundary should be before 'Because' clause (i.e., after token before it — which may be none → then skip)
    # In this form, leading marker has no LHS; should skip
    assert meta["num_rule_bnd"] == 0

    q2 = "Apple added a fix because reviewers reported issues."
    toks, prt = _prep(q2)
    bnds, meta = propose_rule_splits(toks, prt, CFG)
    assert meta["num_rule_bnd"] == 1
    assert any(it["kind"] == "clause" for it in meta["rule_bnds"])


def test_prt_clearance_blocks_boundaries():
    q = 'Apple announced “iPhone 15 Pro Max” and shipped it.'
    toks, prt = _prep(q)
    bnds, meta = propose_rule_splits(toks, prt, CFG)
    # The 'and' sits next to a PRT; boundary should still be allowed only if clearance satisfied
    # Ensure no boundary lands inside quotes span
    for it in meta["rule_bnds"]:
        c = it["char"]
        for a, b in prt:
            assert not (a <= c < b)


def test_high_confidence_flags():
    q = "Apple announced the iPhone 15 Pro, and reviewers said it sold well."
    toks, prt = _prep(q)
    bnds, meta = propose_rule_splits(toks, prt, CFG)
    assert meta["num_rule_bnd"] >= 1
    # at least one should be high_conf True (both sides with verbs and enough length)
    assert any(it["high_conf"] for it in meta["rule_bnds"])
