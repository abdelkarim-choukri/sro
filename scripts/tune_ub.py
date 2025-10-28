# scripts/tune_ub.py
"""
Does:
    Evaluate a saved learned UB on a dataset (held-out preferred).
    Uses the feature list saved in the UB meta by default.

CLI:
    python -m scripts.tune_ub --input data/processed/dev_pairs.csv --conformal
    # or override features explicitly:
    python -m scripts.tune_ub --input ... --features p1_i,p1_j,...
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

from sro.prover.ub_model import (
    ConformalUB,
    build_feature_matrix,
    compute_coverage,
    DEFAULT_FEATURES,
)

logging.basicConfig(
    level=os.environ.get("SRO_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("scripts.tune_ub")


def _resolve_features_arg(arg_val: str, conf: ConformalUB) -> Tuple[str, tuple]:
    """
    Returns (origin, feature_names) where origin in {"model","arg","default"}.
    - If arg_val is empty or "auto": use conf.feature_names if present; else DEFAULT_FEATURES.
    - Else: parse arg_val as CSV list.
    """
    if not arg_val or arg_val.strip().lower() == "auto":
        if conf.feature_names:
            return "model", tuple(conf.feature_names)
        else:
            return "default", DEFAULT_FEATURES
    feats = tuple([s.strip() for s in arg_val.split(",") if s.strip()])
    return "arg", feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--model_dir", type=str, default="artifacts/ub")
    ap.add_argument("--features", type=str, default="auto", help="CSV list or 'auto' to use model's feature_names")
    ap.add_argument("--conformal", action="store_true", help="prints alpha and q_hat from meta")
    args = ap.parse_args()

    # Load model first so we know its feature_names
    conf = ConformalUB.load(args.model_dir)

    # Load data
    df = pd.read_csv(args.input)
    if "y_true" not in df.columns:
        raise KeyError("Input CSV must contain 'y_true' column.")

    origin, feature_names = _resolve_features_arg(args.features, conf)
    X = build_feature_matrix(df, feature_names)
    y = df["y_true"].to_numpy(dtype="float32")

    # Predict UB with optional floor if present in features (rare in your setup)
    floor = None
    if "best_so_far" in feature_names:
        floor = X[:, feature_names.index("best_so_far")]
    ub = conf.predict_upper_bound(X, floor=floor)

    # Metrics
    cov = compute_coverage(y, ub)
    gap = ub - y
    LOGGER.info(
        "Coverage: %.2f%% (target: >= %.2f%%) | Features: %s (%s)",
        cov * 100.0, (1.0 - conf.alpha) * 100.0, ",".join(feature_names), origin
    )
    LOGGER.info(
        "Tightness: mean(UB - y)=%.4f  median=%.4f  p90=%.4f",
        float(np.mean(gap)), float(np.median(gap)), float(np.quantile(gap, 0.90))
    )
    if args.conformal:
        LOGGER.info("Conformal params: alpha=%.4f  q_hat=%.6f", conf.alpha, conf.q_hat)

    print("COVERAGE=%.4f MEAN_GAP=%.4f MED_GAP=%.4f" % (cov, float(np.mean(gap)), float(np.median(gap))))


if __name__ == "__main__":
    main()
