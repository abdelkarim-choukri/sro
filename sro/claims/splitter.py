# sro/claims/splitter.py
from __future__ import annotations

from math import e
import os
import re
from typing import List, Set

from sro.types import Claim

__all__ = ["split", "split_into_claims", "draft_and_claims"]

_DEFAULT_MAX_TOKENS = int(os.getenv("SRO_SPLIT_MAX_TOKENS", "25"))  # max tokens per claim
_MIN_TOKENS = int(os.getenv("SRO_SPLIT_MIN_TOKENS", "4"))           # drop ultra-short

_WORD = re.compile(r"[A-Za-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")  # naive sentence split

_STOPWORDS = {
    "a","an","the","this","that","these","those","of","to","for","in","on","at","by","from","with",
    "and","or","but","as","is","are","was","were","be","been","being","it","its","their","his","her",
    "they","them","we","you","i","he","she","there","here","over","under","into","out","about","up","down"
}

_VERB_LIKE = {
    "is","are","was","were","be","been","being",
    "has","have","had",
    "include","includes","included","including",
    "feature","features","featured","featuring",
    "support","supports","supported","supporting",
    "introduce","introduces","introduced","introducing",
    "announce","announces","announced","announcing",
    "release","releases","released","releasing",
    "launch","launches","launched","launching",
    "confirm","confirms","confirmed","confirming",
    "say","says","said","stated","state","states","claim","claims","claimed"
}

def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]

def _norm(text: str) -> str:
    return " ".join(_tokens(text))

def _has_verb(tokens: list[str]) -> bool:
    for t in tokens:
        if t in _VERB_LIKE:
            return True
        if len(t) >= 4 and t not in _STOPWORDS and (t.endswith("ed") or t.endswith("ing")):
            return True
    return False

def _has_noun_like(tokens: list[str]) -> bool:
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if len(t) >= 3 and not t.isdigit():
            return True
    return False

def _too_vague(text: str) -> bool:
    toks = _tokens(text)
    if len(toks) < _MIN_TOKENS:
        return True
    if not _has_noun_like(toks):
        return True
    if not _has_verb(toks):
        return True
    return False

def _cap_tokens(text: str, max_tokens: int) -> str:
    toks = _tokens(text)
    if len(toks) <= max_tokens:
        return text.strip()
    return " ".join(toks[:max_tokens]).strip()

def _simple_sent_split(draft: str) -> list[str]:
    draft = (draft or "").replace(";", ". ")
    parts = _SENT_SPLIT.split(draft.strip())
    return [p.strip() for p in parts if p and p.strip()]

def split(draft_text: str, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> list[Claim]:
    """
    Split a draft into crisp, deduped, bounded-length claims:
      - naive sentence split
      - dedup by normalized text
      - drop vague sentences (no noun/verb or too short)
      - cap tokens to <= max_tokens
    """
    if not isinstance(draft_text, str) or not draft_text.strip():
        return []

    sents = _simple_sent_split(draft_text)
    seen_norm: set[str] = set()
    out: list[Claim] = []
    k = 1

    for s in sents:
        nrm = _norm(s)
        if not nrm:
            continue
        if nrm in seen_norm:
            continue
        if _too_vague(s):
            continue
        s_capped = _cap_tokens(s, max_tokens)
        seen_norm.add(nrm)
        out.append(Claim(claim_id=f"c{k}", text=s_capped.strip()))
        k += 1

    return out

def split_into_claims(draft_text: str, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> list[Claim]:
    return split(draft_text, max_tokens=max_tokens)


# --- Back-compat shim for tests expecting draft_and_claims here ---
try:
    from sro.compose.answer import draft_and_claims as _draft_and_claims
    def draft_and_claims(*args, **kwargs):
        return _draft_and_claims(*args, **kwargs)
except Exception:
    # If compose.answer cannot be imported for some reason, surface a clear error
    def draft_and_claims(*args, **kwargs):
        raise ImportError(
            "draft_and_claims moved to sro.compose.answer; "
            "ensure sro/compose/answer.py is present and importable"
        ) from e
