# """
# S4 — Upper bound (UB) computation.

# Goal:
#   UB(i,j) should be a fast, optimistic upper bound of the true two-hop score p2(i,j).
#   That is, UB ≥ p2 most of the time (we'll measure coverage later and tune κ).

# Formula:
#   UB(i,j) = clamp(
#       max_p1
#     + α * entity_overlap
#     + β * time_agreement
#     + γ * ce_max
#     + ζ * (1 - distance)
#     + κ,
#     0, 1
#   )

# Where:
#   - α, β, γ, ζ ∈ [0,1] are weights (can be tuned; we give reasonable defaults).
#   - κ (kappa) is a small optimism cushion.
# """

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict


# @dataclass(frozen=True)
# class UBWeights:
#     alpha: float = 0.20   # entity overlap
#     beta: float = 0.15    # time agreement
#     gamma: float = 0.20   # ce_max
#     zeta: float = 0.20    # (1 - distance)


# def clamp01(x: float) -> float:
#     return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


# def upper_bound(feats: Dict[str, float], kappa: float, w: UBWeights = UBWeights()) -> float:
#     """
#     Compute UB for a given pair's feature dict.
#     """
#     ub = (
#         feats["max_p1"]
#         + w.alpha * feats["entity_overlap"]
#         + w.beta * feats["time_agreement"]
#         + w.gamma * feats["ce_max"]
#         + w.zeta * (1.0 - feats["distance"])
#         + kappa
#     )
#     return clamp01(ub)


# sro/prover/s4_ub.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

@dataclass
class UBWeights:
    w_max_p1: float = 0.60
    w_entity: float = 0.15
    w_time: float = 0.10
    w_inv_dist: float = 0.10
    w_novelty: float = 0.02
    w_ce_max: float = 0.08
    # feature bump
    w_neg_conflict: float = -0.04
    w_src_div: float = 0.04

def upper_bound(
    feats: Dict[str, float],
    kappa: float,
    ub_weights: Optional[UBWeights] = None,
    **kwargs,  # ← accept legacy aliases
) -> float:
    # Back-compat alias: tests may pass w=...
    w = ub_weights or kwargs.get("w") or UBWeights()

    max_p1 = float(feats.get("max_p1", 0.0))
    entity_overlap = float(feats.get("entity_overlap", 0.0))
    time_agreement = float(feats.get("time_agreement", 0.0))
    inv_dist = 1.0 - float(feats.get("distance", 1.0))
    novelty = float(feats.get("novelty", 0.0))
    ce_max = float(feats.get("ce_max", 0.0))
    neg_conflict = float(feats.get("negation_conflict", 0.0))
    src_div = float(feats.get("source_diversity", 0.0))

    base = (
        w.w_max_p1 * max_p1 +
        w.w_entity * entity_overlap +
        w.w_time * time_agreement +
        w.w_inv_dist * max(0.0, inv_dist) +
        w.w_novelty * novelty +
        w.w_ce_max * ce_max +
        w.w_neg_conflict * neg_conflict +
        w.w_src_div * src_div
    )
    return clamp01(base + float(kappa))
