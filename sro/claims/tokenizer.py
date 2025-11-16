from __future__ import annotations
from dataclasses import dataclass
from typing import List

# Deterministic tokenizer with character offsets.
# Rules (non-negotiable):
# - Character offsets are source of truth. end is exclusive.
# - No normalization that shifts indices. Preserve original casing.
# - Hyphens: ASCII '-' and Unicode dashes '–', '—' are inside a token only when BETWEEN letters/digits.
# - Apostrophes: ASCII `'` and Unicode ’ are inside a token only when BETWEEN letters (not digits).
# - Punctuation are separate tokens: , ; . : ( ) [ ] { } " ' “ ” ‘ ’ & ? ! currency ($€£¥) % and others.
#   Do NOT glue % or currency to numbers.
# - Numbers: group continuous digits; allow internal separators:
#     * '.' at most once (decimal), only if surrounded by digits.
#     * ',' any number of times, only if surrounded by digits (e.g., 1,000 or 1,000,000).
#   NBSP/thin-space break numbers (isspace() is True).
# - Coverage: skip whitespace; every non-space char must belong to exactly one token.
# - Standard library only. Deterministic.

@dataclass(frozen=True)
class Token:
    text: str
    start: int  # inclusive char index in original string
    end: int    # exclusive char index in original string

# Unicode sets
HYPHENS = {"-", "\u2013", "\u2014"}          # -, –, —
APOSTROPHES = {"'", "\u2019"}                # ', ’
QUOTE_CHARS = {'"', "'", "“", "”", "‘", "’"}
BRACKETS = {"(", ")", "[", "]", "{", "}"}
CURRENCY = {"$", "€", "£", "¥"}
PUNCT_CORE = {",", ";", ":", ".", "?", "!", "&", "/"}  # conservative: treat slash as punctuation
PERCENT = {"%"}
# Everything not alnum/space/hyphen/apostrophe that we didn't absorb will fall back to 1-char punctuation tokens.

def _is_letter(ch: str) -> bool:
    # Accept Unicode letters via str.isalpha() to support names; indices remain stable.
    return ch.isalpha()

def _is_digit(ch: str) -> bool:
    # Only standard ASCII digits should count as digits for number grouping (deterministic).
    return "0" <= ch <= "9"

def _is_alnum(ch: str) -> bool:
    return _is_letter(ch) or _is_digit(ch)

def _is_space(ch: str) -> bool:
    # Covers spaces, tabs, newlines, NBSP (\u00A0), thin space (\u2009), etc.
    return ch.isspace()

def _is_hyphen(ch: str) -> bool:
    return ch in HYPHENS

def _is_apostrophe(ch: str) -> bool:
    return ch in APOSTROPHES

def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Skip whitespace entirely
        if _is_space(ch):
            i += 1
            continue

        start = i

        # 1) Number block: digits with internal separators ('.' once; ',' multiple),
        #    separators only if surrounded by digits.
        if _is_digit(ch):
            saw_dot = False
            j = i + 1
            while j < n:
                c = text[j]
                if _is_digit(c):
                    j += 1
                    continue
                if c == "." and not saw_dot:
                    # decimal point must be surrounded by digits
                    if j - 1 >= i and _is_digit(text[j - 1]) and (j + 1 < n and _is_digit(text[j + 1])):
                        saw_dot = True
                        j += 1
                        continue
                    else:
                        break
                if c == ",":
                    # thousands separator comma must be surrounded by digits
                    if j - 1 >= i and _is_digit(text[j - 1]) and (j + 1 < n and _is_digit(text[j + 1])):
                        j += 1
                        continue
                    else:
                        break
                # any space-like char breaks
                if _is_space(c):
                    break
                # any other symbol breaks (percent/currency must be separate tokens)
                if _is_alnum(c):
                    # letters after digits → break number (units handled elsewhere)
                    break
                # punctuation: stop number token
                break
            tokens.append(Token(text=text[start:j], start=start, end=j))
            i = j
            continue

        # 2) Word block: letters/digits, with internal hyphens BETWEEN alnum and apostrophes BETWEEN letters
        if _is_alnum(ch):
            j = i + 1
            while j < n:
                c = text[j]
                if _is_alnum(c):
                    j += 1
                    continue
                # keep hyphen only when BETWEEN alnum on both sides
                if _is_hyphen(c) and (j - 1 >= i and _is_alnum(text[j - 1])) and (j + 1 < n and _is_alnum(text[j + 1])):
                    j += 1
                    continue
                # keep apostrophe only when BETWEEN letters (not digits)
                if _is_apostrophe(c) and (j - 1 >= i and _is_letter(text[j - 1])) and (j + 1 < n and _is_letter(text[j + 1])):
                    j += 1
                    continue
                break
            tokens.append(Token(text=text[start:j], start=start, end=j))
            i = j
            continue

        # 3) Single-char punctuation tokens (everything else non-space)
        #    Ensure % and currency are not glued to numbers; they land here as separate tokens.
        #    Quotes and brackets also land here.
        tokens.append(Token(text=ch, start=start, end=start + 1))
        i += 1

    return tokens

