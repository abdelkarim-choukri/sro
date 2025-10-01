"""
S4 — Upper bound (UB) computation.

Goal:
  UB(i,j) should be a fast, optimistic upper bound of the true two-hop score p2(i,j).
  That is, UB ≥ p2 most of the time (we'll measure coverage later and tune κ).

Formula:
  UB(i,j) = clamp(
      max_p1
    + α * entity_overlap
    + β * time_agreement
    + γ * ce_max
    + ζ * (1 - distance)
    + κ,
    0, 1
  )

Where:
  - α, β, γ, ζ ∈ [0,1] are weights (can be tuned; we give reasonable defaults).
  - κ (kappa) is a small optimism cushion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class UBWeights:
    alpha: float = 0.20   # entity overlap
    beta: float = 0.15    # time agreement
    gamma: float = 0.20   # ce_max
    zeta: float = 0.20    # (1 - distance)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def upper_bound(feats: Dict[str, float], kappa: float, w: UBWeights = UBWeights()) -> float:
    """
    Compute UB for a given pair's feature dict.
    """
    ub = (
        feats["max_p1"]
        + w.alpha * feats["entity_overlap"]
        + w.beta * feats["time_agreement"]
        + w.gamma * feats["ce_max"]
        + w.zeta * (1.0 - feats["distance"])
        + kappa
    )
    return clamp01(ub)
