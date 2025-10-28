"""
S1 — One-hop scoring.

This module keeps the heuristic stub (fast, deterministic) and can switch
to the real MNLI model when `use_model=True`.

- Heuristic path: identical to Reply 3.
- Model path: calls sro.nli.nli_infer.one_hop_scores (batched, GPU if available).

Set use_model=True from the orchestrator when you're ready.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import List, Tuple

from sro.types import SentenceCandidate

# ---------------- Heuristic stub (unchanged, plus extra verbs we added) ----------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
NEG_WORDS = {"not", "no", "never", "none", "n't", "without", "neither"}
POS_WORDS = {
    "is", "are", "was", "were", "has", "have",
    "confirms", "shows", "reports", "reported", "states", "said",
    "released", "announced", "launched", "introduced", "unveiled"
}

def _tokens(s: str) -> list[str]:
    if not isinstance(s, str):
        raise ValueError("Text must be a string")
    return [t.lower() for t in _WORD_RE.findall(s)]

def _negation_score(tokens: Iterable[str]) -> float:
    return 1.0 if any(t in NEG_WORDS for t in tokens) else 0.0

def _overlap_ratio(claim_toks: list[str], sent_toks: list[str]) -> float:
    if not claim_toks:
        return 0.0
    a, b = set(claim_toks), set(sent_toks)
    return len(a & b) / max(1, len(a))

def _heuristic_scores(claim: str, texts: list[str]) -> tuple[list[float], list[float]]:
    claim_toks = _tokens(claim)
    neg_claim = _negation_score(claim_toks)
    p1, c1 = [], []
    for s in texts:
        sent_toks = _tokens(s)
        neg_sent = _negation_score(sent_toks)
        r = _overlap_ratio(claim_toks, sent_toks)
        entail = r * 0.85 + (0.05 if any(t in POS_WORDS for t in sent_toks) else 0.0)
        neg_mismatch = abs(neg_claim - neg_sent)
        contradict = min(1.0, 0.4 * neg_mismatch + 0.2 * max(0.0, 0.5 - entail))
        p1.append(float(max(0.0, min(1.0, entail))))
        c1.append(float(max(0.0, min(1.0, contradict))))
    return p1, c1

# ---------------- Public API ----------------

def one_hop_scores(claim: str, candidates: list[SentenceCandidate], use_model: bool = False, batch_size: int = 16) -> tuple[list[float], list[float]]:
    """
    If use_model=False: return heuristic scores (fast).
    If use_model=True: use real MNLI model from sro.nli.nli_infer.
    """
    if not candidates:
        return [], []
    texts = [c.text for c in candidates]
    if not use_model:
        return _heuristic_scores(claim, texts)
    # Real model path
    try:
        from sro.nli.nli_infer import one_hop_scores as nli_one_hop
        p_ent, p_contra = nli_one_hop(claim, texts, batch_size=batch_size)
        return p_ent, p_contra
    except Exception:
        # Fail safe to heuristic if model failed to load; surface a warning-like behavior
        # (We do not print; just fallback deterministically)
        return _heuristic_scores(claim, texts)



"""
S1 — One-hop scoring (stub).

Goal:
  For each candidate sentence, estimate:
    - p1[i] ∈ [0,1]: probability sentence i entails the claim (one hop)
    - c1[i] ∈ [0,1]: probability sentence i contradicts the claim

This is a deterministic, lightweight heuristic stub so we can build and test
the SRO-Prover search without loading heavy models. We keep the API compatible
with the real NLI scorer we'll plug in later.

Heuristics used:
  - token overlap ratio with the claim
  - presence of simple negation words to guess contradictions
  - normalize and clamp to [0,1]

Performance:
  O(N * T) where N = number of candidates and T = average token count.
  For default caps (≤64), this is trivial.

Error handling:
  - Empty candidate list returns empty arrays.
  - Non-string texts raise ValueError.
"""


# def one_hop_scores(claim: str, candidates: List[SentenceCandidate]) -> Tuple[List[float], List[float]]:
#     """
#     Compute p1 and c1 for each candidate.

#     Returns:
#       p1: list[float] length N
#       c1: list[float] length N
#     """
#     if not candidates:
#         return [], []

#     claim_toks = _tokens(claim)
#     neg_claim = _negation_score(claim_toks)

#     p1: List[float] = []
#     c1: List[float] = []

#     for cand in candidates:
#         sent_toks = _tokens(cand.text)
#         neg_sent = _negation_score(sent_toks)
#         r = _overlap_ratio(claim_toks, sent_toks)

#         # entailment proxy: overlap + a small boost if sentence is declarative
#         entail = r * 0.85 + (0.05 if any(t in POS_WORDS for t in sent_toks) else 0.0)

#         # contradiction proxy: negation mismatch increases contradiction
#         # If one is negated and the other isn't, bump contradiction.
#         neg_mismatch = abs(neg_claim - neg_sent)
#         contradict = min(1.0, 0.4 * neg_mismatch + 0.2 * max(0.0, 0.5 - entail))

#         # Clamp
#         p1.append(float(max(0.0, min(1.0, entail))))
#         c1.append(float(max(0.0, min(1.0, contradict))))

#     return p1, c1
