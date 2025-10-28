"""
S5 — Bounded search with UB heap, minimality prefilter, and timing hooks.
Returns (best_pair, best_p2, evals, stop_reason, top_ub_remaining).
"""

from __future__ import annotations

import heapq
from contextlib import nullcontext

from sro.prover.s4_ub import UBWeights, clamp01, upper_bound
from sro.types import SentenceCandidate
from sro.utils.timing import StageTimer


class TwoHopScorer:
    def score_pairs(self, pairs: list[tuple[int, int]]) -> list[float]:
        raise NotImplementedError


# Return shape includes: top_ub_remaining (float)
class SearchResult(tuple[tuple[int, int] | None, float, int, str, float]):
    pass


def _p2_stub(feats: dict[str, float]) -> float:
    # Fallback heuristic when a real 2-hop scorer is unavailable (offline stub).
    p2 = (
        0.60 * feats.get("max_p1", 0.0)
        + 0.20 * feats.get("entity_overlap", 0.0)
        + 0.10 * feats.get("time_agreement", 0.0)
        + 0.10 * (1.0 - feats.get("distance", 1.0))
    )
    return clamp01(p2)


def bounded_search(
    claim: str,
    candidates: list[SentenceCandidate],
    pairs: list[tuple[int, int]],
    feats: list[dict[str, float]],
    p1: list[float],
    tau1: float,
    B: int,
    kappa: float,
    ub_weights: UBWeights = UBWeights(),
    two_hop_scorer: TwoHopScorer | None = None,
    batch_size: int = 16,
    timer: StageTimer | None = None,
) -> SearchResult:
    """
    Args:
      claim: claim text (unused by this function but kept for parity)
      candidates: sentence pool (unused here but kept for parity)
      pairs: list of (i, j) indices into candidates
      feats: features per pair (same order as pairs)
      p1: one-hop entail probabilities per candidate
      tau1: minimality threshold (no 2-hop if either leaf alone ≥ tau1)
      B: pair evaluation budget
      kappa: optimism cushion added in UB (used upstream when computing UBs)
      ub_weights: weights for UB computation
      two_hop_scorer: batch scorer (if None, uses heuristic _p2_stub)
      batch_size: scoring batch size
      timer: StageTimer to record S4_ub and S5_search durations
    Returns:
      (best_pair, best_p2, evals, stop_reason, top_ub_remaining)
    """
    # Quick exit if no pairs are provided
    if not pairs:
        return None, 0.0, 0, "NO_PAIRS", 0.0

    # ---------------- S4: build UB heap ----------------
    # Quick exit, but still record S4 timing
    with (timer.stage("S4_ub") if timer else nullcontext()):
        if not pairs:
            return None, 0.0, 0, "NO_PAIRS", 0.0
        heap: list[tuple[float, int]] = []
        for k, f in enumerate(feats):
            ub = upper_bound(f, kappa, ub_weights)
            if not (0.0 <= ub <= 1.0):
                ub = 0.0
            heapq.heappush(heap, (-ub, k))

        best_so_far = max(p1) if p1 else 0.0 #starts as max(p1) (best single-sentence score so far), or 0.0 if no p1.
        best_pair = None #index of best pair found later in S5 (starts None).
        best_p2 = 0.0 #best true 2-hop score found so far (starts 0.0).
        evals = 0 # how many real pair evaluations you’ve done (starts 0).
        stop_reason = "UB_BEATEN" # default reason to report if pruning stops the search (starts "UB_BEATEN").


    # ---------------- S5: bounded search ----------------
    with (timer.stage("S5_search") if timer else nullcontext()):
        while heap and evals < B:
            top_ub = -heap[0][0]
            # Early stop: UB no longer beats best_so_far (tiny epsilon for float noise)
            if top_ub <= best_so_far + 1e-12:
                stop_reason = "UB_BEATEN"
                break

            batch_keys: list[int] = []
            batch_pairs: list[tuple[int, int]] = []
            batch_feats: list[dict[str, float]] = []

            # Pop into a batch while UB still promising
            while heap and len(batch_keys) < batch_size:
                ub_neg, k = heap[0]
                ub = -ub_neg
                if ub <= best_so_far + 1e-12:
                    break
                heapq.heappop(heap)
                i, j = pairs[k]
                # Minimality prefilter: skip if either leaf alone crosses tau1
                if (0 <= i < len(p1) and 0 <= j < len(p1)) and (p1[i] >= tau1 or p1[j] >= tau1):
                    continue
                batch_keys.append(k)
                batch_pairs.append((i, j))
                batch_feats.append(feats[k])

            if not batch_keys:
                # No viable pairs in this window; loop will re-check the next top_ub
                continue

            # Score the batch
            if two_hop_scorer is not None:
                p2_batch = two_hop_scorer.score_pairs(batch_pairs)
                if len(p2_batch) != len(batch_keys):
                    raise RuntimeError("two_hop_scorer returned mismatched length")
            else:
                p2_batch = [_p2_stub(f) for f in batch_feats]

            # Respect global budget B
            take = min(len(p2_batch), B - evals)
            evals += take

            # Update best-so-far
            for idx in range(take):
                p2 = float(p2_batch[idx])
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
