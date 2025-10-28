from __future__ import annotations

import math
import re
from typing import Dict, List, Set, Tuple

from sro.types import SentenceCandidate

# _WORD_RE: regex to extract alphanumeric tokens.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokens(s: str) -> set[str]:
    """Return a set of lowercase tokens (simple alnum tokenizer)."""
    if not isinstance(s, str):
        return set()
    return set(t.lower() for t in _WORD_RE.findall(s))

def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets in [0,1]."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _clean01(x: float) -> float:
    """Clamp to [0,1], map NaN/inf → 0.0."""
    if x is None or not (x == x) or math.isinf(x):  # NaN or inf
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def select_frontier_and_pool(
    candidates: list[SentenceCandidate],
    p1: list[float],
    M: int,
    L: int,
    lambda_diversity: float = 0.7,
) -> tuple[list[int], list[int], dict[str, set[str]]]:
    """
    Select a frontier (size ≤ M) via true MMR, then a second-hop pool (size ≤ L)
    ranked by (relevance × novelty).

    Variables:
      M: frontier size (how many first-hop sentences we keep).
      L: second-hop pool size (how many second-hop candidates we consider).
      lambda_diversity (λ): trade-off in MMR: score = λ·relevance − (1−λ)·redundancy.

    Inputs:
      candidates: list of SentenceCandidate with fields (sent_id, text, source_id, ce_score).
      p1: List[float] of 1-hop entailment scores per candidate in [0,1].

    Outputs:
      frontier_idx: indices of selected frontier sentences (≤ M).
      pool2_idx: indices for second-hop pool (≤ L), disjoint from frontier.
      token_cache: {sent_id -> Set[str]} tokens for downstream reuse.
    """
    n = len(candidates)
    if n == 0:
        return [], [], {}
    if len(p1) != n:
        raise ValueError("p1 length must match number of candidates")
    if M <= 0:
        return [], [], {c.sent_id: _tokens(c.text) for c in candidates}
    if L < 0:
        L = 0

    # Clean p1 (NaN/inf → 0, clamp to [0,1](for exmple 1.2 → 1.0))
    p1c = [_clean01(v) for v in p1]

    # Precompute tokens once
    token_cache: dict[str, set[str]] = {c.sent_id: _tokens(c.text) for c in candidates}

    # ----- sorted order by relevance (p1), then ce_score, then sent_id
    # This ensures deterministic tie-breaking.
    def _base_key(i: int):
        """sorted order by relevance (p1), then ce_score, then sent_id (tie-break). Used for MMR tie-breaking. """
        # Sort descending by p1, ce_score; ascending by sent_id for tie
        return (-p1c[i], -(candidates[i].ce_score or 0.0), str(candidates[i].sent_id))

    order = sorted(range(n), key=_base_key)

    # Early-out if M >= N (frontier is everyone)
    if M >= n:
        frontier = order[:]  # copy
        # Pool is empty (we keep them disjoint); or we could fill the rest (none left).
        return frontier, [], token_cache

    # ----- Frontier selection (true MMR) -----
    # Keep running max similarity to the already-selected frontier:
    # max_sim[i] = max_j_in_frontier Jaccard(i, j)
    max_sim = [0.0] * n
    frontier: list[int] = []
    selected = set()

    # Seed with the best relevant item
    first = order[0]
    frontier.append(first)
    selected.add(first)

    # Iteratively add items that maximize λ·p1 − (1−λ)·max_sim
    # Each iteration only updates max_sim against the newly selected item → O(M·N).
    lam = float(lambda_diversity)
    #lam=1: only care about strength (p1c).
    #lam=0: only care about diversity (penalize overlap).
    
    one_minus_lam = 1.0 - lam
    while len(frontier) < min(M, n):
        j_new = frontier[-1]
        tok_j = token_cache[candidates[j_new].sent_id]
        # Update max_sim vs the newly added item
        for i in order:
            if i in selected:
                continue
            s = _jaccard(token_cache[candidates[i].sent_id], tok_j)
            if s > max_sim[i]:
                max_sim[i] = s

        # Select next by MMR
        best_i, best_score = None, float("-inf")
        for i in order:
            if i in selected:
                continue
            mmr = lam * p1c[i] - one_minus_lam * max_sim[i]
            # Deterministic tie-breaker: prefer higher p1, higher ce_score, then sent_id
            if (mmr > best_score) or (mmr == best_score and _base_key(i) < _base_key(best_i if best_i is not None else i)):
                best_score, best_i = mmr, i

        if best_i is None:
            break
        frontier.append(best_i)
        selected.add(best_i)

    # ----- Second-hop pool (relevance × novelty) -----
    # Novelty = 1 - max_sim_to_frontier (already computed).
    rest = [i for i in order if i not in selected]
    scored_rest = []
    for i in rest:
        novelty = 1.0 - max_sim[i]  # in [0,1]
        score = p1c[i] * novelty    # relevance × novelty
        scored_rest.append((i, score))

    scored_rest.sort(key=lambda t: ( -t[1], ) + _base_key(t[0]) )  # stable tie-break
    pool2 = [i for i, _ in scored_rest[: max(0, L)]]

    return frontier, pool2, token_cache
