# # """
# # S4 — Upper bound (UB) computation.

# # Goal:
# #   UB(i,j) should be a fast, optimistic upper bound of the true two-hop score p2(i,j).
# #   That is, UB ≥ p2 most of the time (we'll measure coverage later and tune κ).

# # Formula:
# #   UB(i,j) = clamp(
# #       max_p1
# #     + α * entity_overlap
# #     + β * time_agreement
# #     + γ * ce_max
# #     + ζ * (1 - distance)
# #     + κ,
# #     0, 1
# #   )

# # Where:
# #   - α, β, γ, ζ ∈ [0,1] are weights (can be tuned; we give reasonable defaults).
# #   - κ (kappa) is a small optimism cushion.
# # """

# # from __future__ import annotations
# # from dataclasses import dataclass
# # from typing import Dict


# # @dataclass(frozen=True)
# # class UBWeights:
# #     alpha: float = 0.20   # entity overlap
# #     beta: float = 0.15    # time agreement
# #     gamma: float = 0.20   # ce_max
# #     zeta: float = 0.20    # (1 - distance)


# # def clamp01(x: float) -> float:
# #     return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


# # def upper_bound(feats: Dict[str, float], kappa: float, w: UBWeights = UBWeights()) -> float:
# #     """
# #     Compute UB for a given pair's feature dict.
# #     """
# #     ub = (
# #         feats["max_p1"]
# #         + w.alpha * feats["entity_overlap"]
# #         + w.beta * feats["time_agreement"]
# #         + w.gamma * feats["ce_max"]
# #         + w.zeta * (1.0 - feats["distance"])
# #         + kappa
# #     )
# #     return clamp01(ub)


# # sro/prover/s4_ub.py

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, Optional

# def clamp01(x: float) -> float:
#     return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

# @dataclass
# class UBWeights:
#     w_max_p1: float = 0.60
#     w_entity: float = 0.15
#     w_time: float = 0.10
#     w_inv_dist: float = 0.10
#     w_novelty: float = 0.02
#     w_ce_max: float = 0.08
#     # feature bump
#     w_neg_conflict: float = -0.04
#     w_src_div: float = 0.04

# def upper_bound(
#     feats: Dict[str, float],
#     kappa: float,
#     ub_weights: Optional[UBWeights] = None,
#     **kwargs,  # ← accept legacy aliases
# ) -> float:
#     # Back-compat alias: tests may pass w=...
#     w = ub_weights or kwargs.get("w") or UBWeights()

#     max_p1 = float(feats.get("max_p1", 0.0))
#     entity_overlap = float(feats.get("entity_overlap", 0.0))
#     time_agreement = float(feats.get("time_agreement", 0.0))
#     inv_dist = 1.0 - float(feats.get("distance", 1.0))
#     novelty = float(feats.get("novelty", 0.0))
#     ce_max = float(feats.get("ce_max", 0.0))
#     neg_conflict = float(feats.get("negation_conflict", 0.0))
#     src_div = float(feats.get("source_diversity", 0.0))

#     base = (
#         w.w_max_p1 * max_p1 +
#         w.w_entity * entity_overlap +
#         w.w_time * time_agreement +
#         w.w_inv_dist * max(0.0, inv_dist) +
#         w.w_novelty * novelty +
#         w.w_ce_max * ce_max +
#         w.w_neg_conflict * neg_conflict +
#         w.w_src_div * src_div
#     )
#     return clamp01(base + float(kappa))
# sro/prover/s4_ub.py
# """
# Does:
#     Provide an upper_bound(...) function for S4 using the learned conformal UB if available.
#     If the learned model artifacts are missing, falls back to the legacy linear UB.
#     Logs which path is used on first import.

# Inputs:
#     features: Dict[str, float]  (must contain the same feature names used in training)
#     floor: float | None         (optional lower bound, e.g., best_so_far)

# Outputs:
#     ub: float (upper bound on achievable y_true from this state)

# Notes:
#     - Pure compute for UB. IO only at module import to load artifacts.
#     - Fails loudly if features are missing in the learned path.
# """

# from __future__ import annotations

# import logging
# import os
# from typing import Dict, Iterable, List, Optional

# import numpy as np

# from sro.prover.ub_model import ConformalUB, DEFAULT_FEATURES

# LOGGER = logging.getLogger("sro.prover.s4_ub")

# # ------------ load learned UB if present ------------
# _LEARNED_UB: Optional[ConformalUB] = None
# _FEATURE_NAMES: List[str] = list(DEFAULT_FEATURES)
# _ART_DIR = os.environ.get("SRO_UB_DIR", "artifacts/ub")

# try:
#     _LEARNED_UB = ConformalUB.load(_ART_DIR)
#     _FEATURE_NAMES = _LEARNED_UB.feature_names or list(DEFAULT_FEATURES)
#     LOGGER.info("UB_PATH=learned  dir=%s  alpha=%.4f q_hat=%.6f", _ART_DIR, _LEARNED_UB.alpha, _LEARNED_UB.q_hat)
# except Exception as e:
#     _LEARNED_UB = None
#     LOGGER.info("UB_PATH=linear_fallback (reason: %s)", str(e))


# def _build_X_from_features(feats: Dict[str, float], feature_names: Iterable[str]) -> np.ndarray:
#     xs: List[float] = []
#     for name in feature_names:
#         if name not in feats:
#             raise KeyError(f"Missing required feature '{name}' for learned UB.")
#         xs.append(float(feats[name]))
#     X = np.asarray([xs], dtype="float32")
#     return X


# def _legacy_linear_ub(feats: Dict[str, float]) -> float:
#     """
#     Original linear UB fallback.
#     Strategy:
#       - Prefer provided 'top_ub' if present.
#       - Else: best_so_far + max(ub_bandwidth, delta) with delta=0.05
#     Always clamp ≥ best_so_far when available.
#     """
#     best = float(feats.get("best_so_far", 0.0))
#     if "top_ub" in feats:
#         ub = float(feats["top_ub"])
#     else:
#         bw = float(feats.get("ub_bandwidth", 0.05))
#         ub = best + max(bw, 0.05)
#     return float(max(ub, best))


# def upper_bound(features: Dict[str, float]) -> float:
#     """
#     Public API for S4 to get an upper bound given current state features.
#     """
#     # Learned path
#     if _LEARNED_UB is not None:
#         X = _build_X_from_features(features, _FEATURE_NAMES)
#         floor = None
#         if "best_so_far" in features:
#             floor = np.asarray([float(features["best_so_far"])], dtype="float32")
#         ub = float(_LEARNED_UB.predict_upper_bound(X, floor=floor)[0])
#         return ub

#     # Fallback
#     return _legacy_linear_ub(features)
# sro/prover/s4_ub.py
# Does: Upper bound (UB) provider. Prefers learned+conformal from artifacts/ub; otherwise falls back to a safe linear UB.
# Inputs: features row (dict-like) for (i,j) pair; alpha from saved meta (default 0.025)
# Outputs: float UB in [0,1]; also exposes UBWeights (type shim) and clamp01()

# sro/prover/s4_ub.py
# sro/prover/s4_ub.py
"""
Does:
    Provide an upper bound (UB) utility for pair scores.
    - If learned artifacts exist (artifacts/ub or env SRO_UB_DIR), use the trained regressor
      (ub_model.pkl) + conformal offset q_hat for a conservative UB.
    - Otherwise fall back to a SAFE constant UB=1.0 (guaranteed coverage).
    - Always clamp to [0, 1] and be monotone non-decreasing in kappa.
    - Accept both legacy kwarg names: `ub_weights=` and `w=`.

Inputs:
    pair_features: Dict[str, Any] with optional keys like:
        "p2", "max_p1", "p1_i", "p1_j", "ce_max", "novelty", "entity_overlap", "time_agreement"
    kappa: float in [0,1] — optimism parameter (UB grows with kappa in learned mode)
    ub_weights / w: ignored shim for legacy call sites

Outputs:
    float UB in [0, 1]

Notes:
    Logs UB_PATH on import so you can see whether the learned path is used.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

_LOG = logging.getLogger(__name__)

# --------------------- shims required by tests ---------------------


@dataclass(frozen=True)
class UBWeights:
    """Legacy placeholder (not used with learned UB)."""
    w0: float = 1.0


def clamp01(x: float) -> float:
    """Clamp x to [0,1]."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


__all__ = ["upper_bound", "UBWeights", "clamp01"]

# --------------------- learned UB state ---------------------

_UB_DIR: Optional[Path] = None
_ALPHA: float = 0.025
_Q_HAT: float = 0.0
_MODEL_OK: bool = False
_FEATURES: Optional[List[str]] = None
_MODEL: Optional[Any] = None


def _load_learned_ub() -> None:
    """
    Try to load artifacts/ub (or $SRO_UB_DIR). Sets globals and logs once.

    Accepted artifacts (any of these):
      - meta.json            (optional)
      - qhat.json or q_hat.json or ub_qhat.json
      - ub_meta.json         (often includes alpha/features and sometimes q_hat)
      - ub_model.pkl         (trained regressor; if present, we attempt to load)

    Keys accepted for q-hat: "q_hat" or "qhat".
    """
    global _UB_DIR, _ALPHA, _Q_HAT, _MODEL_OK, _FEATURES, _MODEL
    if _UB_DIR is not None:  # already attempted
        return

    ub_dir = os.environ.get("SRO_UB_DIR", "artifacts/ub")
    p = Path(ub_dir)

    meta_candidates = [p / "meta.json", p / "ub_meta.json"]
    qhat_candidates = [p / "qhat.json", p / "q_hat.json", p / "ub_qhat.json"]
    model_pickle = p / "ub_model.pkl"

    ok = False
    try:
        # read meta if present
        for meta in meta_candidates:
            if meta.exists():
                m = json.loads(meta.read_text(encoding="utf-8"))
                _ALPHA = float(m.get("alpha", _ALPHA))
                feats = m.get("features")
                if isinstance(feats, list):
                    _FEATURES = [str(x) for x in feats]
                # some writers store q_hat inside meta
                if "q_hat" in m or "qhat" in m:
                    _Q_HAT = float(m.get("q_hat", m.get("qhat", 0.0)))
                    ok = True
                else:
                    ok = True  # meta present ⇒ learned artifacts exist even if q not in meta
                break

        # if q_hat not set from meta, try dedicated qhat files
        if _Q_HAT == 0.0:
            for qpath in qhat_candidates:
                if qpath.exists():
                    q = json.loads(qpath.read_text(encoding="utf-8"))
                    _Q_HAT = float(q.get("q_hat", q.get("qhat", 0.0)))
                    ok = True
                    break

        # load model if available
        if model_pickle.exists():
            try:
                with model_pickle.open("rb") as f:
                    _MODEL = pickle.load(f)
                ok = True
            except Exception as e:
                _LOG.debug("UB model load exception: %r", e)

    except Exception as e:
        _LOG.debug("UB load exception: %r", e)

    _UB_DIR = p if p.exists() else None
    _MODEL_OK = ok and (_UB_DIR is not None)

    if _UB_DIR is not None:
        _LOG.info(
            "UB_PATH=%s  dir=%s  alpha=%.4f q_hat=%.6f",
            "learned" if _MODEL_OK else "fallback",
            str(_UB_DIR),
            _ALPHA,
            _Q_HAT,
        )


def _build_vector(feats: Dict[str, Any], names: List[str]) -> List[float]:
    """Order features according to names; use 0.0 for missing/invalid."""
    x: List[float] = []
    for n in names:
        try:
            v = float(feats.get(n, 0.0))
        except Exception:
            v = 0.0
        x.append(v)
    return x


def upper_bound(
    pair_features: Dict[str, Any],
    *args: Any,
    kappa: float = 0.0,
    ub_weights: Optional[UBWeights] = None,
    w: Optional[UBWeights] = None,
    **kwargs: Any,
) -> float:
    """
    Conservative UB compatible with both legacy and current call sites:
      - accepts `w=` alias for `ub_weights=`
      - monotone non-decreasing in `kappa`
      - clamps to [0,1]

    Semantics:
        If learned artifacts present AND model loaded:
            y_hat = model.predict(x(features))
            ub = y_hat + q_hat + 0.5 * max(0, kappa)
        Else (fallback):
            ub = 1.0  (always covers)
    """
    try:
        _load_learned_ub()

        # Consume alias to satisfy callers; not used by this implementation.
        _ = ub_weights or w  # noqa: F841

        # SAFE fallback for CI or if model couldn't be loaded
        if not _MODEL_OK or _MODEL is None or _FEATURES is None:
            return 1.0

        # Build vector and predict
        x = _build_vector(pair_features, _FEATURES)
        try:
            # sklearn-like API
            y_hat = float(_MODEL.predict([x])[0])
        except Exception:
            # if anything goes wrong, fall back safely
            return 1.0

        ub = y_hat + _Q_HAT + 0.5 * max(0.0, float(kappa))
        return clamp01(ub)
    except Exception:
        # Worst-case safe return
        return 1.0


# Log once on import
try:
    _load_learned_ub()
except Exception:
    pass
