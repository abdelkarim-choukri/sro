from __future__ import annotations
from sro.claims.tokenizer import tokenize
from sro.claims.rules import propose_protected_spans


class DummyCfg:
    protected_phrases_path = "sro/ling/protected_phrases.txt"


def _cover(text, spans):
    covered = set()
    for s, e in spans:
        covered.update(range(s, e))
    return covered


def test_quotes_and_parens_protected():
    s = 'He said, “Apple Inc. (Cupertino, CA) grew 10%”, right?'
    spans = propose_protected_spans(tokenize(s), DummyCfg())
    # ensure quote and paren regions are inside some span
    assert any(s[s0:e0].startswith("“") and s[s0:e0].endswith("”") for s0, e0 in spans)
    assert any("(" in s[s0:e0] and ")" in s[s0:e0] for s0, e0 in spans)


def test_titlecase_runs_and_pairs():
    s = "Harvard University and Massachusetts Institute of Technology partnered with Johnson & Johnson."
    spans = propose_protected_spans(tokenize(s), DummyCfg())
    assert any("Harvard University" in s[a:b] for a, b in spans)
    assert any("Massachusetts Institute of Technology" in s[a:b] for a, b in spans)
    assert any("Johnson & Johnson" in s[a:b] for a, b in spans)


def test_hyphenated_and_acronyms():
    s = "state-of-the-art meets R&D and U.S. standards; Ph.D. candidates applied."
    spans = propose_protected_spans(tokenize(s), DummyCfg())
    assert any("state-of-the-art" in s[a:b] for a, b in spans)
    assert any("R&D" in s[a:b] for a, b in spans)
    assert any("U.S." in s[a:b] for a, b in spans)
    assert any("Ph.D." in s[a:b] for a, b in spans)


def test_dates_numbers_and_units():
    s = "On March 3, 2023 revenue rose from 10–12 to ≥ 15 and hit 5 % then $ 200."
    spans = propose_protected_spans(tokenize(s), DummyCfg())
    # month/day/year
    assert any("March 3, 2023" in s[a:b] for a, b in spans)
    # numeric ranges
    assert any("10–12" in s[a:b] for a, b in spans)
    assert any("≥ 15" in s[a:b] for a, b in spans)
    # % bound to number
    assert any("5 %" in s[a:b] for a, b in spans)
    # currency + number bound
    assert any("$ 200" in s[a:b] for a, b in spans)


def test_urls_emails_and_codes():
    s = "Contact ops@foo.io or visit https://example.com/docs. Stock AAPL moved."
    spans = propose_protected_spans(tokenize(s), DummyCfg())
    assert any("ops@foo.io" in s[a:b] for a, b in spans)
    assert any("https://example.com/docs" in s[a:b] for a, b in spans)
    assert any("AAPL" in s[a:b] for a, b in spans)


def test_merge_and_sort_no_overlap():
    s = '“Alpha Beta” grew 10 % in 2024 (Q1).'
    ts = tokenize(s)
    spans = propose_protected_spans(ts, DummyCfg())
    # Spans are non-overlapping and sorted
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    # Slice sanity: spans map to original indices
    for a, b in spans:
        assert s[a:b] == s[a:b]

