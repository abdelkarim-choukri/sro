"""
S5 — Bounded search (adds 'top_ub_remaining' to the return for alternation).
"""

from __future__ import annotations
import heapq
from typing import List, Tuple, Dict, Optional

from sro.types import SentenceCandidate
from sro.prover.s4_ub import upper_bound, UBWeights, clamp01


class TwoHopScorer:
    def score_pairs(self, pairs: List[Tuple[int, int]]) -> List[float]:
        raise NotImplementedError


# Return shape now includes: top_ub_remaining (float)
class SearchResult(Tuple[Optional[Tuple[int, int]], float, int, str, float]):
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
        return None, 0.0, 0, "NO_PAIRS", 0.0

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

    while heap and evals < B:
        top_ub = -heap[0][0]
        if top_ub <= best_so_far + 1e-12:
            stop_reason = "UB_BEATEN"
            break

        batch_keys: List[int] = []
        batch_pairs: List[Tuple[int, int]] = []
        batch_feats: List[Dict[str, float]] = []

        while heap and len(batch_keys) < batch_size:
            ub_neg, k = heap[0]
            ub = -ub_neg
            if ub <= best_so_far + 1e-12:
                break
            heapq.heappop(heap)
            i, j = pairs[k]
            if p1[i] >= tau1 or p1[j] >= tau1:  # minimality prefilter
                continue
            batch_keys.append(k)
            batch_pairs.append((i, j))
            batch_feats.append(feats[k])

        if not batch_keys:
            continue

        if two_hop_scorer is not None:
            p2_batch = two_hop_scorer.score_pairs(batch_pairs)
            if len(p2_batch) != len(batch_keys):
                raise RuntimeError("two_hop_scorer returned mismatched length")
        else:
            p2_batch = [_p2_stub(f) for f in batch_feats]

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

    # Top UB remaining on heap (for alternation logic)
    top_ub_remaining = (-heap[0][0]) if heap else 0.0

    if not heap and best_pair is None and stop_reason != "BUDGET_EXCEEDED":
        stop_reason = "NO_PAIRS"

    return best_pair, float(best_p2), int(evals), stop_reason, float(top_ub_remaining)
