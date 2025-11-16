from __future__ import annotations
from typing import List, Tuple, Iterable, Sequence, Optional, Dict
import re

from sro.claims.tokenizer import Token

Span = Tuple[int, int]  # [start, end)

_STOPWORDS = {"and", "&", "of", "for", "the"}

_MONTHS = {
    "jan","january","feb","february","mar","march","apr","april","may","jun","june",
    "jul","july","aug","august","sep","sept","september","oct","october","nov","november","dec","december"
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)")
_ORDINAL_RE = re.compile(r"^[0-9]+(st|nd|rd|th)$")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_ALLCAPS_RE = re.compile(r"^[A-Z]{2,}$")
_CODE_RE = re.compile(r"^[A-Z]{2,6}(-[A-Z0-9]{1,6})?$")  # simple ticker/code-ish


def _merge_spans(spans: Iterable[Span]) -> List[Span]:
    s = sorted(spans)
    out: List[Span] = []
    for st, en in s:
        if st >= en:
            continue
        if not out or st > out[-1][1]:
            out.append((st, en))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], en))
    return out


def _span(tokens: List[Token], i: int, j: int) -> Span:
    """Char span covering tokens[i:j] (i inclusive, j exclusive)."""
    return (tokens[i].start, tokens[j - 1].end)


def _is_titlecase(tok: Token) -> bool:
    t = tok.text
    return len(t) >= 2 and t[0].isupper() and any(ch.isalpha() for ch in t[1:])


def _is_allcaps(tok: Token) -> bool:
    return _ALLCAPS_RE.match(tok.text) is not None


def _token_lc(tok: Token) -> str:
    return tok.text.lower()


def _consume_proper_run(tokens: List[Token], i: int, allow_and: bool = True) -> int:
    """Return j > i covering ProperCase run with optional stopwords; else return i.
    Limit to at most 5 tokens.
    If allow_and is False, do not treat 'and'/'&' as in-run stopwords (used for X and Y pairs).
    """
    n = len(tokens)
    j = i
    seen = 0
    if allow_and:
        allowed_stop = _STOPWORDS
    else:
        allowed_stop = {w for w in _STOPWORDS if w not in {"and", "&"}}
    while j < n and (_is_titlecase(tokens[j]) or _token_lc(tokens[j]) in allowed_stop):
        if _is_titlecase(tokens[j]):
            seen += 1
        j += 1
        if j - i >= 5:
            break
    return j if seen >= 1 else i


def _find_pairs_title_and(tokens: List[Token]) -> List[Span]:
    spans: List[Span] = []
    n = len(tokens)
    i = 0
    while i < n:
        if _is_titlecase(tokens[i]):
            L = _consume_proper_run(tokens, i, allow_and=False)
            k = L
            if k < n and _token_lc(tokens[k]) in {"&", "and"}:
                R = _consume_proper_run(tokens, k + 1, allow_and=False)
                if R > k + 1:
                    spans.append(_span(tokens, i, R))
                    i = R
                    continue
        i += 1
    return spans


def _find_quotes_parens(tokens: List[Token]) -> List[Span]:
    spans: List[Span] = []
    opens = []
    QUOTE_OPEN = {'"', "'", "“", "‘"}
    QUOTE_CLOSE = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    BR_OPEN = {"(", "[", "{"}
    BR_CLOSE = {")": "(", "]": "[", "}": "{"}
    for idx, t in enumerate(tokens):
        ch = t.text
        if ch in QUOTE_OPEN:
            opens.append(("Q", ch, idx))
        elif ch in set(QUOTE_CLOSE.values()):
            for k in range(len(opens) - 1, -1, -1):
                typ, o, i = opens[k]
                if typ == "Q" and QUOTE_CLOSE.get(o) == ch:
                    spans.append(_span(tokens, i, idx + 1))
                    opens = opens[:k]
                    break
        elif ch in BR_OPEN:
            opens.append(("B", ch, idx))
        elif ch in BR_CLOSE:
            want = BR_CLOSE[ch]
            for k in range(len(opens) - 1, -1, -1):
                typ, o, i = opens[k]
                if typ == "B" and o == want:
                    spans.append(_span(tokens, i, idx + 1))
                    opens = opens[:k]
                    break
    return spans


def _find_hyphenated(tokens: List[Token]) -> List[Span]:
    spans: List[Span] = []
    for t in tokens:
        if ("-" in t.text) or ("\u2013" in t.text) or ("\u2014" in t.text):
            spans.append((t.start, t.end))
    return spans


def _is_numberish_text(txt: str) -> bool:
    # Accept integer, grouped with commas, optional decimal point.
    if not txt:
        return False
    # Quick path: all digits
    if txt.isdigit():
        return True
    # Commas and/or a single dot, otherwise only digits
    if not all(c.isdigit() or c in {',', '.'} for c in txt):
        return False
    # No more than one dot
    if txt.count('.') > 1:
        return False
    # Dots/commas must be surrounded by digits
    for i, c in enumerate(txt):
        if c in {',', '.'}:
            if i == 0 or i == len(txt) - 1:
                return False
            if not (txt[i - 1].isdigit() and txt[i + 1].isdigit()):
                return False
    return True


def _is_number_token(tok: Token) -> bool:
    return _is_numberish_text(tok.text)


def _find_acronyms_dotted(tokens: List[Token]) -> List[Span]:
    spans: List[Span] = []
    n = len(tokens)

    def is_seg(tok: Token, allow_title_first: bool) -> bool:
        s = tok.text
        if not s.isalpha():
            return False
        if s.isupper():  # U, US, R, D
            return True
        # allow first segment like 'Ph'
        if allow_title_first and len(s) <= 4 and s[0].isupper() and s[1:].islower():
            return True
        return False

    i = 0
    while i < n:
        j = i
        # Require a first segment
        if not is_seg(tokens[j], allow_title_first=True):
            # Special case: U . S . style single letters
            if len(tokens[j].text) == 1 and tokens[j].text.isupper():
                k = j
                ok = False
                while k + 1 < n and tokens[k + 1].text == "." and k + 2 < n and len(tokens[k + 2].text) == 1 and tokens[k + 2].text.isupper():
                    ok = True
                    k += 2
                if ok:
                    # include trailing dot if present
                    end_k = k + 1
                    if end_k < n and tokens[end_k].text == ".":
                        end_k += 1
                    spans.append(_span(tokens, j, end_k))
                    i = end_k
                    continue
            i += 1
            continue

        segs = 1
        k = j
        while k + 2 < n and tokens[k + 1].text in {".", "&"} and is_seg(tokens[k + 2], allow_title_first=False):
            segs += 1
            k += 2
        if segs >= 2:
            end_k = k + 1
            # include trailing dot if present
            if end_k < n and tokens[end_k].text == ".":
                end_k += 1
            spans.append(_span(tokens, j, end_k))
            i = end_k
        else:
            i += 1
    return spans


def _find_dates_numbers(tokens: List[Token]) -> List[Span]:
    spans: List[Span] = []
    n = len(tokens)
    # Years
    for t in tokens:
        if _YEAR_RE.match(t.text):
            spans.append((t.start, t.end))

    # Months + day ordinal/number + optional comma + optional year
    i = 0
    while i < n:
        if _token_lc(tokens[i]) in _MONTHS:
            j = i + 1
            if j < n and (_ORDINAL_RE.match(tokens[j].text) or tokens[j].text.isdigit()):
                j += 1
            if j < n and tokens[j].text == ",":
                j += 1
            if j < n and _YEAR_RE.match(tokens[j].text):
                j += 1
            spans.append(_span(tokens, i, j))
            i = j
            continue
        i += 1

    # Numeric ranges and comparators
    for i in range(n):
        t = tokens[i]
        # ≥/≤ N
        if t.text in {"≥", "≤"} and i + 1 < n and _is_number_token(tokens[i + 1]):
            spans.append(_span(tokens, i, i + 2))
        # N - N or N – N or N — N
        if _is_number_token(t) and i + 2 < n and tokens[i + 1].text in {"-", "–", "—"} and _is_number_token(tokens[i + 2]):
            spans.append(_span(tokens, i, i + 3))

    # Bind % and currencies with adjacent number
    CURRENCIES = {"$", "€", "£", "¥"}
    for i, t in enumerate(tokens):
        if t.text in CURRENCIES:
            # currency + number right
            if i + 1 < n and _is_number_token(tokens[i + 1]):
                spans.append(_span(tokens, i, i + 2))
            # number left + currency (rare, but bind anyway)
            if i - 1 >= 0 and _is_number_token(tokens[i - 1]):
                spans.append(_span(tokens, i - 1, i + 1))
        if t.text == "%":
            # number left + %
            if i - 1 >= 0 and _is_number_token(tokens[i - 1]):
                spans.append(_span(tokens, i - 1, i + 1))
            # % + number right
            if i + 1 < n and _is_number_token(tokens[i + 1]):
                spans.append(_span(tokens, i, i + 2))

    return spans


def _load_curated_phrases(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip().lower() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    except FileNotFoundError:
        return []


def _find_curated_phrases(tokens: List[Token], phrases: List[str]) -> List[Span]:
    spans: List[Span] = []
    if not phrases:
        return spans
    phrase_set = set(phrases)
    n = len(tokens)
    lc = [t.text.lower() for t in tokens]
    max_len = 8
    for i in range(n):
        for L in range(1, min(max_len, n - i) + 1):
            seg = " ".join(lc[i:i + L])
            if seg in phrase_set:
                spans.append(_span(tokens, i, i + L))
    return spans


def _reconstruct_text(tokens: List[Token]) -> str:
    if not tokens:
        return ""
    total_len = tokens[-1].end
    buf = [" "] * total_len
    for t in tokens:
        # Place token.text exactly at its original indices
        buf[t.start:t.end] = list(t.text)
    return "".join(buf)


def propose_protected_spans(tokens: List[Token], cfg) -> List[Span]:
    spans: List[Span] = []
    # Quotes / brackets
    spans += _find_quotes_parens(tokens)
    # TitleCase runs
    spans += _titlecase_runs(tokens)
    # X & Y / X and Y
    spans += _find_pairs_title_and(tokens)
    # Hyphenated tokens
    spans += _find_hyphenated(tokens)
    # Acronyms / dotted forms
    spans += _find_acronyms_dotted(tokens)
    # ALL-CAPS tokens and code-like tokens
    spans += [(t.start, t.end) for t in tokens if _is_allcaps(t)]
    spans += [(t.start, t.end) for t in tokens if _CODE_RE.match(t.text)]
    # Dates / numbers
    spans += _find_dates_numbers(tokens)
    # Curated phrases
    phrases_path = getattr(cfg, "protected_phrases_path", "sro/ling/protected_phrases.txt")
    phrases = _load_curated_phrases(phrases_path)
    spans += _find_curated_phrases(tokens, phrases)
    # URLs / emails over reconstructed original-length text
    full_text = _reconstruct_text(tokens)
    for m in _EMAIL_RE.finditer(full_text):
        spans.append((m.start(), m.end()))
    for m in _URL_RE.finditer(full_text):
        spans.append((m.start(), m.end()))
    # Merge + sort
    return _merge_spans(spans)


def _titlecase_runs(tokens: List[Token]) -> List[Span]:
    spans: List[Span] = []
    n = len(tokens)
    i = 0
    while i < n:
        if _is_titlecase(tokens[i]):
            j = i + 1
            seen_tc = 1
            while j < n:
                if _is_titlecase(tokens[j]) or _token_lc(tokens[j]) in _STOPWORDS:
                    if _is_titlecase(tokens[j]):
                        seen_tc += 1
                    j += 1
                    if j - i >= 5:
                        break
                else:
                    break
            if seen_tc >= 2:
                spans.append(_span(tokens, i, j))
                # advance by one to allow discovering overlapping sub-runs like
                # "Massachusetts Institute of Technology" inside a longer context
                i += 1
                continue
        i += 1
    return spans


# ----------------- Rule-based split proposals -----------------
CLAUSE_MARKERS = {
    "because", "so", "so that", "after", "before", "when", "where", "which", "who", "that",
    # added high-yield clause markers
    "while", "whereas", "although", "since", "unless", "until", "as", "due", "provided", "even"
}


def _in_span(char_idx: int, spans: list[tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= char_idx < b:
            return True
    return False


def _token_in_prt(tok: Token, spans: list[tuple[int, int]]) -> bool:
    return _in_span(tok.start, spans) or _in_span(tok.end - 1, spans)


def _near_prt(i_tok: int, tokens: list[Token], spans: list[tuple[int, int]], clearance: int = 1) -> bool:
    # true if any token within +/- clearance lies in PRT
    n = len(tokens)
    lo = max(0, i_tok - clearance)
    hi = min(n - 1, i_tok + clearance)
    PUNCT = {",", ".", ";", ":", "?", "!", '"', "'", "“", "”", "‘", "’", "(", ")", "[", "]", "{", "}"}
    for j in range(lo, hi + 1):
        # ignore pure punctuation tokens when checking clearance
        if tokens[j].text in PUNCT:
            continue
        if _token_in_prt(tokens[j], spans):
            return True
    return False


def _enforce_min_gap(bnds_char: list[int], min_gap_tokens: int, tokens: list[Token]) -> list[int]:
    if not bnds_char:
        return []
    # Convert char idx to token index for spacing, then back to char with stable selection (leftmost wins)
    idxs = sorted(bnds_char)
    out = [idxs[0]]
    last_tok = _char_to_tok_index(idxs[0], tokens)
    for c in idxs[1:]:
        t = _char_to_tok_index(c, tokens)
        if t - last_tok >= min_gap_tokens:
            out.append(c)
            last_tok = t
    return out


def _char_to_tok_index(char_idx: int, tokens: list[Token]) -> int:
    # returns the token index i such that char_idx == tokens[i].end
    for i, t in enumerate(tokens):
        if t.end == char_idx:
            return i
    # if not exact (shouldn't happen), fall back to nearest left
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].end < char_idx:
            return i
    return 0


def has_verb(token_span: Sequence[Token], lexicon: set[str] | None = None) -> bool:
    # Fast, conservative heuristic: exact match against lexicon (lowercased),
    # plus suffix heuristics: -ed, -ing, -s, and "to" + base-form check.
    if not token_span:
        return False
    lex = lexicon or set()
    lc = [t.text.lower() for t in token_span]
    for w in lc:
        if w in lex:
            return True
        if w.endswith("ed") or w.endswith("ing"):
            if any(ch.isalpha() for ch in w[:-2]):  # avoid "red"/"string"
                return True
        if w.endswith("s") and len(w) > 3:
            # crude present-3sg heuristic, skip obvious nouns by whitelist later if needed
            return True
    # to + base form: "to go", "to build"
    for i in range(len(lc) - 1):
        if lc[i] == "to" and lc[i + 1].isalpha() and lc[i + 1] not in {"the", "a", "an"}:
            return True
    return False


def is_high_conf(bnd_char_idx: int, tokens: list[Token], prt_spans: list[tuple[int, int]], cfg) -> bool:
    min_side = getattr(cfg.splitter, "high_conf_min_side_len", 4)
    min_gap = getattr(cfg.splitter, "min_gap_tokens", 3)

    i = _char_to_tok_index(bnd_char_idx, tokens)
    left = tokens[: i + 1]
    right = tokens[i + 1 :]

    if len(left) < min_side or len(right) < min_side:
        return False
    if _near_prt(i, tokens, prt_spans, clearance=1):
        return False

    # verbs on both sides
    lex = _load_verb_lexicon(getattr(cfg.splitter, "verb_lexicon_path", "sro/ling/verbs.txt"))
    if not (has_verb(left[-min(12, len(left)) :], lex) and has_verb(right[: min(12, len(right))], lex)):
        return False
    # min-gap will be enforced globally; here just return True if local criteria pass
    return True


def _load_verb_lexicon(path: str) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {ln.strip().lower() for ln in f if ln.strip() and not ln.startswith("#")}
    except FileNotFoundError:
        return set()


def _is_top_level(i: int, tokens: list[Token], prt_spans: list[tuple[int, int]]) -> bool:
    # “top-level” = token not inside PRT
    return not _token_in_prt(tokens[i], prt_spans)


def propose_rule_splits(tokens: list[Token], prt_spans: list[tuple[int, int]], cfg) -> tuple[list[int], dict]:
    min_gap = getattr(cfg.splitter, "min_gap_tokens", 3)
    max_len = getattr(cfg.splitter, "max_len_tokens", 25)

    lex = _load_verb_lexicon(getattr(cfg.splitter, "verb_lexicon_path", "sro/ling/verbs.txt"))
    n = len(tokens)
    cand: list[tuple[int, int, str, bool]] = []  # (char_idx, i_tok, kind, high_conf)

    def add_bnd(i_tok: int, kind: str):
        if i_tok < 0 or i_tok >= n - 1:
            return
        # no boundary if adjacent token belongs to PRT (clearance)
        if _near_prt(i_tok, tokens, prt_spans, clearance=1):
            return
        char_idx = tokens[i_tok].end
        # respect side lengths now to avoid silly splits
        left = i_tok + 1
        right = n - (i_tok + 1)
        if left < 2 or right < 2:
            return
        if left > max_len or right > max_len:
            # leave to later stage? For now, skip proposals that would yield > max_len sides
            pass
        hc = is_high_conf(char_idx, tokens, prt_spans, cfg)
        cand.append((char_idx, i_tok, kind, hc))

    # 1) ", and|or" / "; and|or" — split AFTER coordinator token
    # Note: require the coordinator token (and/or) to be top-level; the comma/semicolon
    # itself may be part of a protected span (e.g. within a date) so we check the
    # token after the punctuation for top-level status.
    for i in range(n - 2):
        if tokens[i].text in {",", ";"} and tokens[i + 1].text.lower() in {"and", "or"}:
            if not _is_top_level(i + 1, tokens, prt_spans):
                continue
            add_bnd(i + 1, "coord")

    # 2) bare "and|or": verbs on both sides
    for i in range(1, n - 1):
        if tokens[i].text.lower() in {"and", "or"} and _is_top_level(i, tokens, prt_spans):
            # trim leading/trailing punctuation from each side and require >=4 tokens per side
            def _trim_punct(tok_list):
                if not tok_list:
                    return tok_list
                PUNCT = {",", ".", ";", ":", "?", "!", '"', "'", "“", "”", "‘", "’", "(", ")", "[", "]", "{", "}"}
                a = 0
                b = len(tok_list)
                while a < b and tok_list[a].text in PUNCT:
                    a += 1
                while b > a and tok_list[b - 1].text in PUNCT:
                    b -= 1
                return tok_list[a:b]

            left_ctx = tokens[max(0, i - 12) : i]
            right_ctx = tokens[i + 1 : i + 1 + 12]
            left_trim = _trim_punct(left_ctx)
            right_trim = _trim_punct(right_ctx)
            # require >=3 tokens per side (was 4) to improve recall on QDMR
            if len(left_trim) >= 3 and len(right_trim) >= 3 and has_verb(left_trim, lex) and has_verb(right_trim, lex):
                add_bnd(i, "bare")

    # 3) clause markers — handle multi-word markers first (longest-match)
    i = 0
    while i < n:
        lw = tokens[i].text.lower()
        # multi-word markers
        # multi-word markers: "as long as"
        if i + 2 < n and lw == "as" and tokens[i + 1].text.lower() == "long" and tokens[i + 2].text.lower() == "as":
            marker_i = i
            look = tokens[i + 3 : i + 3 + 5]  # check up to 5 tokens after marker
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 3
            continue
            continue
        if i + 2 < n and lw == "in" and tokens[i + 1].text.lower() == "order" and tokens[i + 2].text.lower() == "to":
            marker_i = i
            look = tokens[i + 3 : i + 3 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 3
            continue
            continue
        if i + 1 < n and lw == "provided" and tokens[i + 1].text.lower() == "that":
            marker_i = i
            look = tokens[i + 2 : i + 2 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 2
            continue
            continue
        if i + 1 < n and lw == "even" and tokens[i + 1].text.lower() == "though":
            marker_i = i
            look = tokens[i + 2 : i + 2 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 2
            continue
        # even if
        if i + 1 < n and lw == "even" and tokens[i + 1].text.lower() == "if":
            marker_i = i
            look = tokens[i + 2 : i + 2 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 2
            continue
        # as if / as though
        if i + 2 < n and lw == "as" and tokens[i + 1].text.lower() in {"if", "though"}:
            marker_i = i
            look = tokens[i + 2 : i + 2 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 2
            continue
        # rather than
        if i + 1 < n and lw == "rather" and tokens[i + 1].text.lower() == "than":
            marker_i = i
            look = tokens[i + 2 : i + 2 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 2
            continue
            continue
        if i + 1 < n and lw == "due" and tokens[i + 1].text.lower() == "to":
            marker_i = i
            look = tokens[i + 2 : i + 2 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 2
            continue
        # longest-match for "so that"
        if i + 1 < n and f"{tokens[i].text.lower()} {tokens[i+1].text.lower()}" == "so that":
            marker_i = i
            look = tokens[i + 2 : i + 2 + 5]
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
            i += 2
            continue
        # single-word clause markers (including 'as', 'since', 'while', etc.)
        if lw in CLAUSE_MARKERS:
            marker_i = i
            look = tokens[i + 1 : i + 1 + 5]
            # special-case: comma + which/who/that -> prefer split before the comma
            if lw in {"which", "who", "that"} and i > 0 and tokens[i - 1].text == "," and _is_top_level(i - 1, tokens, prt_spans):
                # propose split at the token BEFORE the comma when safe
                if i - 2 >= 0:
                    look2 = tokens[i + 1 : i + 1 + 5]
                    if has_verb(look2, lex):
                        add_bnd(i - 2, "clause")
                        i += 1
                        continue
            if has_verb(look, lex):
                add_bnd(marker_i - 1, "clause")
        i += 1

    # de-duplicate by char idx; prefer high_conf True if any
    by_char = {}
    for c, i_tok, kind, hc in cand:
        prev = by_char.get(c)
        if prev is None or (not prev[2] and hc):
            by_char[c] = (i_tok, kind, hc)
    dedup = [(c,) + by_char[c] for c in sorted(by_char.keys())]

    # global min-gap
    bnds_sorted = _enforce_min_gap([c for c, _, _, _ in dedup], min_gap, tokens)

    # rebuild meta aligned to kept indices
    kept = set(bnds_sorted)
    meta_items = []
    for c, i_tok, kind, hc in dedup:
        if c in kept:
            meta_items.append({"char": c, "i_tok": i_tok, "kind": kind, "high_conf": hc})

    meta = {
        "rule_bnds": meta_items,
        "num_rule_bnd": len(meta_items),
        "num_high_conf": sum(1 for it in meta_items if it["high_conf"]),
    }
    return bnds_sorted, meta
