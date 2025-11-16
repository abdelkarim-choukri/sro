from __future__ import annotations
from sro.claims.tokenizer import tokenize, Token

def _texts(tokens):
    return [t.text for t in tokens]

def _spans(tokens):
    return [(t.start, t.end) for t in tokens]

def test_simple_hello_world():
    s = "Hello, World!"
    toks = tokenize(s)
    assert _texts(toks) == ["Hello", ",", "World", "!"]
    # Check indices match original text slices
    for t in toks:
        assert s[t.start:t.end] == t.text
    # Coverage: all non-space chars covered
    non_space = [i for i,c in enumerate(s) if not c.isspace()]
    covered = []
    for t in toks:
        covered.extend(range(t.start, t.end))
    assert sorted(non_space) == sorted(covered)

def test_apostrophes_inside_words_only():
    s = "Apple’s revenue, O’Neil said, can’t drop."
    toks = tokenize(s)
    assert "Apple’s" in _texts(toks)    # Unicode apostrophe inside letters
    assert "O’Neil" in _texts(toks)
    assert "can’t" in _texts(toks)
    # Apostrophe at boundary becomes separate token
    s2 = "'quote' '  ' 'a"
    toks2 = tokenize(s2)
    assert _texts(toks2)[0] == "'"
    assert _texts(toks2)[1] == "quote"
    assert _texts(toks2)[2] == "'"

def test_hyphens_and_unicode_dashes_inside_words_only():
    s = "state-of-the-art A17—Pro COVID–19 end- -start"
    toks = tokenize(s)
    assert "state-of-the-art" in _texts(toks)   # ASCII hyphens internal
    assert "A17—Pro" in _texts(toks)            # em dash internal
    assert "COVID–19" in _texts(toks)           # en dash internal
    # hyphen as punctuation when not between alnum
    assert "-" in _texts(toks)  # standalone hyphen token
    assert "-start".replace("-", "") in "".join(_texts(toks))  # "start" token exists

def test_numbers_with_commas_and_decimal_dot():
    s = "Price was 1,000 and then 3.5 and 1,000,000 but 3, and 3."
    toks = tokenize(s)
    assert "1,000" in _texts(toks)
    assert "3.5" in _texts(toks)
    assert "1,000,000" in _texts(toks)
    # single '3' near comma is tokenized separately, comma is its own token
    assert ["3", ","] in [ _texts(toks)[i:i+2] for i in range(len(toks)-1) ]

def test_currency_and_percent_are_separate_tokens():
    s = "Cost $ 100 % 5 € 200£300 ¥ 400%."
    toks = tokenize(s)
    # Ensure currency and % are separate tokens (never glued to numbers)
    expect = ["Cost", "$", "100", "%", "5", "€", "200", "£", "300", "¥", "400", "%", "."]
    assert _texts(toks) == expect

def test_quotes_parens_are_single_char_tokens():
    s = 'He asked, “Why (now)?”'
    toks = tokenize(s)
    assert _texts(toks) == ["He", "asked", ",", "“", "Why", "(", "now", ")", "?", "”"]

def test_nbsp_and_thin_space_break_numbers_and_words():
    s = "3\u00A0000 and 4\u2009500"
    toks = tokenize(s)
    # NBSP / thin space split tokens
    assert _texts(toks) == ["3", "000", "and", "4", "500"]

def test_indices_end_exclusive_and_full_coverage():
    s = "Did Apple's stock price drop 5% or rise 10%?"
    toks = tokenize(s)
    # end must be exclusive and slice must round-trip
    for t in toks:
        assert s[t.start:t.end] == t.text
    # no overlaps, full coverage of non-space chars
    covered = []
    for t in toks:
        covered.extend(range(t.start, t.end))
    non_space = [i for i, c in enumerate(s) if not c.isspace()]
    assert sorted(covered) == sorted(non_space)

def test_windows_newlines_do_not_break():
    s = "A\r\nB"
    toks = tokenize(s)
    assert _texts(toks) == ["A", "B"]
    # positions: 'A' at 0, '\r' at 1 (space -> skipped), '\n' at 2 (space -> skipped), 'B' at 3
    assert toks[0].start == 0 and toks[0].end == 1
    assert toks[1].start == 3 and toks[1].end == 4

