"""
S5 — Bounded search with optional batched two-hop NLI scorer.

Changes vs Reply 3:
- Accept an optional `two_hop_scorer` object with a method:
      score_pairs(pairs: List[Tuple[int,int]]) -> List[float]
  where returned list length == number of input pairs (p2 entail probs).
- Pop pairs in small batches where UB > best_so_far to exploit batching.
- Fallback to the built-in _p2_stub if no scorer is provided.

We still:
- Use max-heap keyed by UB.
- Early stop when top(UB) <= best_so_far.
- Skip non-minimal pairs without spending budget.
"""

from __future__ import annotations
import heapq
from typing import List, Tuple, Dict, Optional

from sro.types import SentenceCandidate
from sro.prover.s4_ub import upper_bound, UBWeights, clamp01


class TwoHopScorer:
    """
    Protocol-like base class. Provide a .score_pairs(List[(i,j)]) -> List[float].
    """
    def score_pairs(self, pairs: List[Tuple[int, int]]) -> List[float]:
        raise NotImplementedError


class SearchResult(Tuple[Optional[Tuple[int, int]], float, int, str]):
    pass


def _p2_stub(feats: Dict[str, float]) -> float:
    p2 = (
        0.60 * feats["max_p1"]
        + 0.20 * feats["entity_overlap"]
        + 0.10 * feats["time_agreement"]
        + 0.10 * (1.0 - feats["distance"])
    )
    return clamp01(p2)


def bounded_search(
    claim: str,
    candidates: List[SentenceCandidate],
    pairs: List[Tuple[int, int]],
    feats: List[Dict[str, float]],
    p1: List[float],
    tau1: float,
    B: int,
    kappa: float,
    ub_weights: UBWeights = UBWeights(),
    two_hop_scorer: Optional[TwoHopScorer] = None,
    batch_size: int = 16,
) -> SearchResult:
    if not pairs:
        return None, 0.0, 0, "NO_PAIRS"

    # Precompute UB and build heap
    heap = []
    for k, f in enumerate(feats):
        ub = upper_bound(f, kappa, ub_weights)
        if not (0.0 <= ub <= 1.0):
            ub = 0.0
        heapq.heappush(heap, (-ub, k))

    best_so_far = max(p1) if p1 else 0.0
    best_pair: Optional[Tuple[int, int]] = None
    best_p2 = 0.0
    evals = 0
    stop_reason = "UB_BEATEN"

    # Batch pop: collect top items whose UB > best_so_far
    while heap and evals < B:
        top_ub = -heap[0][0]
        if top_ub <= best_so_far + 1e-12:
            stop_reason = "UB_BEATEN"
            break

        batch_keys: List[int] = []
        batch_pairs: List[Tuple[int, int]] = []
        batch_feats: List[Dict[str, float]] = []

        while heap and len(batch_keys) < batch_size:
            ub_neg, k = heap[0]  # peek
            ub = -ub_neg
            if ub <= best_so_far + 1e-12:
                break
            heapq.heappop(heap)
            i, j = pairs[k]
            # Minimality prefilter: skip without spending budget
            if p1[i] >= tau1 or p1[j] >= tau1:
                continue
            batch_keys.append(k)
            batch_pairs.append((i, j))
            batch_feats.append(feats[k])

        if not batch_keys:
            # No eligible pairs above best_so_far
            continue

        # Score the batch
        if two_hop_scorer is not None:
            # External scorer returns p2 for the batch in the same order
            p2_batch = two_hop_scorer.score_pairs(batch_pairs)
            if len(p2_batch) != len(batch_keys):
                raise RuntimeError("two_hop_scorer returned mismatched length")
        else:
            # Local heuristic per pair
            p2_batch = [_p2_stub(f) for f in batch_feats]

        # Spend budget
        take = min(len(p2_batch), B - evals)
        evals += take
        for idx in range(take):
            p2 = p2_batch[idx]
            if p2 > best_so_far:
                best_so_far = p2
                best_p2 = p2
                best_pair = batch_pairs[idx]

        if evals >= B:
            stop_reason = "BUDGET_EXCEEDED"
            break

    if not heap and best_pair is None and stop_reason != "BUDGET_EXCEEDED":
        stop_reason = "NO_PAIRS"

    return best_pair, float(best_p2), int(evals), stop_reason
