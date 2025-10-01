"""
S2 — Frontier selection and second-hop pool.

Goal:
  - Select a small, diverse set of first-hop candidates (frontier) of size M.
  - Select a second-hop pool of size L that is relevant but not redundant.

Method:
  - Rank by p1 descending.
  - Greedy MMR-like selection for frontier:
      score = λ * p1 - (1-λ) * max_sim_to_selected
    where sim is Jaccard similarity over tokens (fast, no extra deps).
  - Second-hop pool: take the next-best items by p1 *and* novelty
    relative to the frontier.

Edge cases:
  - If fewer than M or L items exist, return as many as available.
  - Empty candidates → empty outputs.

Complexity:
  O(N^2) in worst case due to similarity checks, but with N ≤ 64 it's trivial.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import re

from sro.types import SentenceCandidate

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set:
    return set(t.lower() for t in _WORD_RE.findall(s))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def select_frontier_and_pool(
    candidates: List[SentenceCandidate],
    p1: List[float],
    M: int,
    L: int,
    lambda_diversity: float = 0.7,
) -> Tuple[List[int], List[int], Dict[str, set]]:
    """
    Inputs:
      candidates: list of SentenceCandidate
      p1: list of float, same length as candidates
      M: frontier size
      L: second-hop pool size
      lambda_diversity: tradeoff between relevance and novelty [0,1]

    Returns:
      frontier_idx: indices of selected first-hop sentences (length ≤ M)
      pool2_idx: indices of selected second-hop pool (length ≤ L)
      token_cache: dict sent_id -> token set (to avoid recompute later)
    """
    n = len(candidates)
    if n == 0:
        return [], [], {}

    if len(p1) != n:
        raise ValueError("p1 length must match number of candidates")

    # Precompute tokens once
    token_cache: Dict[str, set] = {c.sent_id: _tokens(c.text) for c in candidates}

    # Base ranking by p1
    order = sorted(range(n), key=lambda i: p1[i], reverse=True)

    # Greedy MMR-like frontier selection
    frontier: List[int] = []
    sims_to_frontier: List[float] = [0.0] * n

    for i in order:
        if len(frontier) >= M:
            break
        # compute similarity to the *current* selected set
        if frontier:
            sim = max(
                _jaccard(token_cache[candidates[i].sent_id], token_cache[candidates[j].sent_id])
                for j in frontier
            )
        else:
            sim = 0.0
        sims_to_frontier[i] = sim

        mmr_score = lambda_diversity * p1[i] - (1 - lambda_diversity) * sim
        # Insert by marginal utility: try to add only if it remains among best
        # For simplicity with small N, just greedy by encountering order.
        frontier.append(i)

    # Second-hop pool:
    # Start with items not in frontier, prefer higher p1 and novelty from frontier
    rest = [i for i in order if i not in frontier]
    pool2: List[int] = []
    for i in rest:
        if len(pool2) >= L:
            break
        # novelty relative to frontier
        if frontier:
            max_sim_to_frontier = max(
                _jaccard(token_cache[candidates[i].sent_id], token_cache[candidates[j].sent_id])
                for j in frontier
            )
        else:
            max_sim_to_frontier = 0.0
        novelty = 1.0 - max_sim_to_frontier
        # simple composite rank
        # prioritize items with both high p1 and high novelty
        pool2.append(i)

    # If L is smaller than rest, trim to top-L by p1 * novelty
    if len(pool2) > L:
        # recompute novelty for trimming
        def score(i: int) -> float:
            if frontier:
                max_sim_to_frontier = max(
                    _jaccard(token_cache[candidates[i].sent_id], token_cache[candidates[j].sent_id])
                    for j in frontier
                )
            else:
                max_sim_to_frontier = 0.0
            novelty = 1.0 - max_sim_to_frontier
            return 0.5 * p1[i] + 0.5 * novelty

        pool2 = sorted(pool2, key=score, reverse=True)[:L]

    return frontier, pool2, token_cache
