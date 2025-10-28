# scripts/train_ub.py
"""
Does:
    Train a learned UB with inductive conformal calibration.
    - Reads CSV with required feature cols + y_true
    - Splits into train/calib (random_state seed)
    - Trains base quantile regressor (97-98th percentile)
    - Fits conformal residual q_hat for alpha (default 0.025)
    - Saves model + meta to artifacts/ub/

CLI:
    python -m scripts.train_ub --input data/processed/dev_pairs.csv --alpha 0.025 --seed 42 --out_dir artifacts/ub
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from sro.prover.ub_model import (
    DEFAULT_FEATURES,
    ConformalUB,
    QuantileUB,
    build_feature_matrix,
    compute_coverage,
)

logging.basicConfig(
    level=os.environ.get("SRO_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("scripts.train_ub")


def set_all_seeds(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def _split_train_calib(y: np.ndarray, test_size: float, seed: int):
    # Stratify by y buckets to stabilize splits
    bins = np.clip((y * 10).astype(int), 0, 10)
    from sklearn.model_selection import train_test_split  # local import

    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=bins,
        shuffle=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/ub")
    ap.add_argument("--alpha", type=float, default=0.025, help="target violation rate (e.g., 0.025 for 97.5% coverage)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--calib_size", type=float, default=0.30, help="fraction for calibration split")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if "y_true" not in df.columns:
        raise KeyError("Input CSV must contain 'y_true' column.")

    feature_names = tuple([s.strip() for s in args.features.split(",") if s.strip()])
    X = build_feature_matrix(df, feature_names)
    y = df["y_true"].to_numpy(dtype="float32")

    # Split into train/calib
    train_idx, calib_idx = _split_train_calib(y, test_size=args.calib_size, seed=args.seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_calib, y_calib = X[calib_idx], y[calib_idx]

    # Train base quantile regressor near the top (0.98 default)
    base = QuantileUB(alpha=0.98, random_state=args.seed)
    conf = ConformalUB(base=base, alpha=args.alpha, feature_names=feature_names)
    conf.fit(X_train, y_train, X_calib, y_calib)

    # Quick calibration coverage check on calib split
    ub_calib = conf.predict_upper_bound(X_calib, floor=X_calib[:, feature_names.index("best_so_far")] if "best_so_far" in feature_names else None)
    coverage = compute_coverage(y_calib, ub_calib)
    LOGGER.info("Calibration coverage (should be >= %.1f%%): %.2f%%", (1 - args.alpha) * 100, coverage * 100.0)

    # Save artifacts
    conf.save(args.out_dir)
    LOGGER.info("Saved UB model to: %s", args.out_dir)


if __name__ == "__main__":
    main()
