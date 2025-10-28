# scripts/log_search_traces.py
"""
Does:
    Produce a proxy alternation training table from dev_pairs.csv and the learned UB.
    For each row, compute features:
        best_so_far      = max(p1_i, p1_j)
        X = [best_so_far, top_ub, budget_left_norm, frontier_entropy, ub_bandwidth]
    where:
        top_ub           = learned UB prediction on the row's features
        budget_left_norm = 1.0 (proxy; you can wire true budget later)
        frontier_entropy = entropy of softmax([p1_i, p1_j]) normalized by log(2)
        ub_bandwidth     = max(0, top_ub - best_so_far)

    Label (bootstrap):
        y_alt = 1 if (best_so_far < tau1) and ((top_ub - best_so_far) >= delta) else 0

Outputs:
    CSV at --out with columns: best_so_far, top_ub, budget_left_norm, frontier_entropy, ub_bandwidth, y_alt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import List

import numpy as np
import pandas as pd

from sro.prover.ub_model import ConformalUB, build_feature_matrix

logging.basicConfig(
    level=os.environ.get("SRO_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOG = logging.getLogger("scripts.log_search_traces")


def _entropy2(p1: float, p2: float) -> float:
    s = p1 + p2
    if s <= 0:
        return 0.0
    q1, q2 = p1 / s, p2 / s
    # Clip to avoid log(0)
    q1 = min(max(q1, 1e-12), 1.0)
    q2 = min(max(q2, 1e-12), 1.0)
    h = -(q1 * math.log(q1) + q2 * math.log(q2))
    return float(h / math.log(2.0))  # normalize to [0,1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--ub_dir", type=str, default="artifacts/ub")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--tau1", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    conf = ConformalUB.load(args.ub_dir)

    # Build UB predictions on the same features used for training the UB
    X = build_feature_matrix(df, conf.feature_names)
    floor = None
    if "best_so_far" in conf.feature_names:
        floor = X[:, conf.feature_names.index("best_so_far")]
    ub = conf.predict_upper_bound(X, floor=floor)

    # Proxy features
    p1_i = df.get("p1_i", pd.Series([0.0] * len(df), dtype=float)).to_numpy(dtype=float)
    p1_j = df.get("p1_j", pd.Series([0.0] * len(df), dtype=float)).to_numpy(dtype=float)
    best = np.maximum(p1_i, p1_j).astype("float32")

    frontier_entropy = np.array([_entropy2(a, b) for a, b in zip(p1_i, p1_j)], dtype="float32")
    budget_left_norm = np.ones_like(best, dtype="float32")  # proxy = 1.0
    ub_bandwidth = np.maximum(0.0, ub - best).astype("float32")

    # Bootstrap label: headroom & not yet minimal
    y_alt = ((best < args.tau1) & ((ub - best) >= args.delta)).astype("int8")

    out = pd.DataFrame(
        {
            "best_so_far": best,
            "top_ub": ub.astype("float32"),
            "budget_left_norm": budget_left_norm,
            "frontier_entropy": frontier_entropy,
            "ub_bandwidth": ub_bandwidth,
            "y_alt": y_alt,
        }
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    LOG.info("Wrote %s rows=%d", args.out, len(out))


if __name__ == "__main__":
    main()
