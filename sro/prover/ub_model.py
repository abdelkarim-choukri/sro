# sro/prover/ub_model.py
"""
Does:
    Learned upper bound model for early-stop with guarantees:
      - Base quantile regressor (sklearn GradientBoostingRegressor with quantile loss)
      - Inductive conformal wrapper computing a high-quantile residual (q_hat)
      - Predict UB(x) = y_hat_base(x) + q_hat  (clamped >= best_so_far if provided)
    Includes feature builder utilities and serialization.

Inputs:
    - Feature matrix X (np.ndarray[float32], shape [N, D]) with a FIXED feature order
    - Targets y_true (np.ndarray[float32], shape [N]) representing realized
      "best attainable" score from a state (higher is better)
    - alpha ∈ (0, 1): violation rate target (e.g., 0.025 for 97.5% coverage)

Outputs:
    - Trained ConformalUB (predict_upper_bound) with save/load helpers
    - Utility functions: build_feature_matrix, compute_coverage

Notes:
    - All core logic is pure. IO lives in save/load only.
    - Deterministic by seed (passed to base estimator).
    - Fails loudly if required features are missing or shapes mismatch.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np

LOGGER = logging.getLogger("sro.prover.ub_model")

# Try to import sklearn. If missing, we throw a clear error at train time.
try:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore
    Pipeline = None  # type: ignore
    train_test_split = None  # type: ignore


DEFAULT_FEATURES: tuple[str, ...] = (
    "best_so_far",
    "top_ub",
    "budget_left_norm",
    "frontier_entropy",
    "frontier_p1_mean",
    "frontier_p1_max",
    "frontier_p1_std",
    "ub_bandwidth",
)


def ensure_features(df_like, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df_like.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}. Present: {list(df_like.columns)}")


def build_feature_matrix(
    df_like,
    feature_names: Sequence[str],
    dtype: str = "float32",
) -> np.ndarray:
    """
    Pure builder: extracts columns in 'feature_names' order and returns X array.

    Invariants:
      - X.shape == (N, D)
      - dtype float32 by default
    """
    ensure_features(df_like, feature_names)
    X = np.asarray(df_like[list(feature_names)].to_numpy(), dtype=dtype)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("Feature matrix shape mismatch.")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return X


def compute_coverage(y_true: np.ndarray, ub_pred: np.ndarray) -> float:
    """
    Coverage = mean( y_true <= ub_pred )
    """
    if y_true.shape != ub_pred.shape:
        raise ValueError("Shapes mismatch in compute_coverage.")
    return float(np.mean(y_true <= ub_pred))


@dataclass
class UBMeta:
    alpha: float
    q_hat: float
    feature_names: list[str]
    base_model: str
    updated_at: str
    version: str = "1.0"


class QuantileUB:
    """
    Wrapper over sklearn quantile regressor using GradientBoostingRegressor(loss='quantile').
    Pipeline: StandardScaler -> GBDT(quantile).
    """

    def __init__(
        self,
        alpha: float = 0.98,
        random_state: int = 42,
        n_estimators: int = 400,
        max_depth: int = 3,
        subsample: float = 0.9,
    ) -> None:
        if GradientBoostingRegressor is None or Pipeline is None:  # pragma: no cover
            raise ImportError("scikit-learn is required. Please `pip install scikit-learn`.")
        self.alpha = float(alpha)
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=self.alpha,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        subsample=subsample,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantileUB:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype="float32")


class ConformalUB:
    """
    Inductive conformal wrapper over a base quantile regressor.

    Training:
      - fit_base on train split
      - compute residuals on calibration split: r = y_true - y_hat_base
      - q_hat = empirical quantile_{1-alpha}(r)
    Inference:
      - UB(x) = y_hat_base(x) + q_hat
      - Optional floor: UB(x) = max(UB(x), floor)  (e.g., best_so_far)
    """

    def __init__(self, base: QuantileUB, alpha: float = 0.025, feature_names: Sequence[str] | None = None):
        self.base = base
        self.alpha = float(alpha)
        self.q_hat: float = 0.0
        self.feature_names: list[str] = list(feature_names) if feature_names is not None else []

    # ----------------- fit -----------------
    def fit(  # pure compute
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> ConformalUB:
        self.base.fit(X_train, y_train)
        y_hat_calib = self.base.predict(X_calib)
        residuals = (y_calib - y_hat_calib).astype("float32")
        self.q_hat = float(np.quantile(residuals, 1.0 - self.alpha))
        return self

    # ----------------- predict -----------------
    def predict_upper_bound(self, X: np.ndarray, floor: np.ndarray | None = None) -> np.ndarray:
        """
        Predict upper bound. If 'floor' is provided (same shape), clamp UB >= floor.
        """
        y_hat = self.base.predict(X)
        ub = y_hat + self.q_hat
        if floor is not None:
            if floor.shape != ub.shape:
                raise ValueError("floor shape mismatch in predict_upper_bound.")
            ub = np.maximum(ub, floor)
        return ub.astype("float32")

    # ----------------- IO -----------------
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # Save base model pipeline
        joblib.dump(self.base.model, os.path.join(out_dir, "ub_model.pkl"))
        # Save meta JSON
        meta = UBMeta(
            alpha=self.alpha,
            q_hat=self.q_hat,
            feature_names=self.feature_names,
            base_model="GBR-quantile",
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        with open(os.path.join(out_dir, "ub_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta.__dict__, f, indent=2)

    @classmethod
    def load(cls, in_dir: str) -> ConformalUB:
        model_path = os.path.join(in_dir, "ub_model.pkl")
        meta_path = os.path.join(in_dir, "ub_meta.json")
        if not (os.path.isfile(model_path) and os.path.isfile(meta_path)):
            raise FileNotFoundError(f"UB artifacts missing in: {in_dir}")
        pipe = joblib.load(model_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        # Rehydrate
        base = QuantileUB(alpha=0.98)  # alpha of base regressor is not critical once trained
        base.model = pipe
        inst = cls(base=base, alpha=float(meta["alpha"]), feature_names=meta.get("feature_names", []))
        inst.q_hat = float(meta["q_hat"])
        return inst
