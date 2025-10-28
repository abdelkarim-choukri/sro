# sro/prover/s8_alternation.py
"""
Does:
    Learned alternation policy for switching from 1-hop to 2-hop (at most once).
    Tiny logistic classifier + guardrails (minimality, UB headroom, one alternation max).

IO:
    - PolicyMeta.save/load: JSON file storing {features, weights, bias, threshold, tau1, delta}.
    - decide_alternation_from_pair_scores: convenience hook to use the policy without a full prover.

Inputs to decide():
    feats: dict[str,float] with keys like:
        best_so_far, top_ub, budget_left_norm, frontier_entropy, ub_bandwidth
    alternations_used: int in {0,1}

Outputs:
    dict { "alternate": bool, "prob": float, "reason": str }

Notes:
    Pure logic. No heavy imports at module import time.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = [
    "PolicyMeta",
    "AlternationPolicy",
    "decide_alternation_from_pair_scores",
]


# ---------------------- Model file ----------------------

@dataclass(frozen=True)
class PolicyMeta:
    features: Tuple[str, ...]
    weights: Tuple[float, ...]
    bias: float
    threshold: float = 0.5
    tau1: float = 0.75
    delta: float = 0.10

    @staticmethod
    def load(path: str) -> "PolicyMeta":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        feats = tuple(d["features"])
        w = tuple(float(x) for x in d["weights"])
        if len(w) != len(feats):
            raise ValueError("weights and features length mismatch.")
        return PolicyMeta(
            features=feats,
            weights=w,
            bias=float(d["bias"]),
            threshold=float(d.get("threshold", 0.5)),
            tau1=float(d.get("tau1", 0.75)),
            delta=float(d.get("delta", 0.10)),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "features": list(self.features),
                    "weights": list(self.weights),
                    "bias": self.bias,
                    "threshold": self.threshold,
                    "tau1": self.tau1,
                    "delta": self.delta,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


# ---------------------- Policy core ----------------------

class AlternationPolicy:
    def __init__(self, meta: PolicyMeta):
        self.meta = meta

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _vectorize(self, feats: Dict[str, float]) -> List[float]:
        out: List[float] = []
        for name in self.meta.features:
            v = float(feats.get(name, 0.0))
            if name in ("best_so_far", "top_ub", "budget_left_norm", "ub_bandwidth"):
                v = max(0.0, min(1.0, v))
            out.append(v)
        return out

    def score(self, feats: Dict[str, float]) -> float:
        x = self._vectorize(feats)
        z = sum(w * xi for w, xi in zip(self.meta.weights, x)) + self.meta.bias
        return self._sigmoid(z)

    def decide(self, feats: Dict[str, float], *, alternations_used: int) -> Dict[str, object]:
        # Guardrail: one alternation max
        if int(alternations_used) >= 1:
            return {"alternate": False, "prob": 0.0, "reason": "ONE_ALT_CAP"}

        best = float(feats.get("best_so_far", 0.0))
        top_ub = float(feats.get("top_ub", best))
        # Guardrail: minimality
        if best >= self.meta.tau1:
            return {"alternate": False, "prob": 0.0, "reason": "MINIMALITY_BLOCK"}
        # Guardrail: UB headroom
        if (top_ub - best) < self.meta.delta:
            return {"alternate": False, "prob": 0.0, "reason": "UB_BEATEN"}

        p = self.score(feats)
        alt = bool(p >= self.meta.threshold)
        return {"alternate": alt, "prob": p, "reason": "OK" if alt else "POLICY_REJECT"}


# ---------------------- Convenience hook ----------------------

def decide_alternation_from_pair_scores(
    p1_i: float,
    p1_j: float,
    *,
    alternations_used: int,
    budget_left_norm: float = 1.0,
    policy_path: str | None = None,
) -> Dict[str, object]:
    """
    Minimal integration helper: build features from two 1-hop scores, get UB via s4_ub.upper_bound,
    and run the learned policy with guardrails.

    Returns policy.decide(..) dict.
    """
    import math as _math
    from sro.prover.s4_ub import upper_bound  # uses learned UB if available

    best = float(max(p1_i, p1_j))

    # tiny 2-class entropy from the two 1-hop scores (proxy frontier entropy)
    s = max(p1_i + p1_j, 1e-12)
    q1, q2 = p1_i / s, p1_j / s
    q1 = min(max(q1, 1e-12), 1.0)
    q2 = min(max(q2, 1e-12), 1.0)
    frontier_entropy = float(-((q1 * _math.log(q1)) + (q2 * _math.log(q2))) / _math.log(2.0))

    # UB feature vector (fill missing with 0.0)
    ub_feats = {
        "p1_i": float(p1_i),
        "p1_j": float(p1_j),
        "best_so_far": best,
        "p2": 0.0,
        "max_p1": best,
        "entity_overlap": 0.0,
        "time_agreement": 0.0,
        "distance": 0.0,
        "novelty": 0.0,
        "ce_max": 0.0,
        "negation_conflict": 0.0,
        "source_diversity": 0.0,
    }
    top_ub = float(upper_bound(ub_feats))

    feats = {
        "best_so_far": best,
        "top_ub": top_ub,
        "budget_left_norm": float(max(0.0, min(1.0, budget_left_norm))),
        "frontier_entropy": frontier_entropy,
        "ub_bandwidth": max(0.0, top_ub - best),
    }

    # load policy
    path = policy_path or os.environ.get("SRO_ALT_POLICY", "artifacts/alternation/policy.json")
    meta = PolicyMeta.load(path)
    policy = AlternationPolicy(meta)
    return policy.decide(feats, alternations_used=alternations_used)
