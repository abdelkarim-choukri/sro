# scripts/train_alternation.py
"""
Does:
    Train a tiny logistic alternation policy from a traces CSV (from log_search_traces.py or real traces).
    Saves artifacts/alternation/policy.json with weights, bias, feature order, threshold.

Inputs:
    --traces : CSV with columns best_so_far, top_ub, budget_left_norm, frontier_entropy, ub_bandwidth, y_alt
    --out    : path to policy.json (default artifacts/alternation/policy.json)
    --seed   : RNG seed
    --tau1, --delta : guardrail params to embed in the policy file
    --threshold : decision threshold (default 0.5). You can later tune on a dev split.

Outputs:
    policy.json with:
       { features, weights, bias, threshold, tau1, delta, updated_at }
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = ("best_so_far", "top_ub", "budget_left_norm", "frontier_entropy", "ub_bandwidth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True)
    ap.add_argument("--out", type=str, default="artifacts/alternation/policy.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau1", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--threshold", type=float, default=0.50)
    args = ap.parse_args()

    df = pd.read_csv(args.traces)
    for c in list(FEATURES) + ["y_alt"]:
        if c not in df.columns:
            raise KeyError(f"Traces missing required column: {c}")
    X = df[list(FEATURES)].to_numpy(dtype="float32")
    y = df["y_alt"].to_numpy(dtype="int32")

    clf = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=args.seed, class_weight="balanced"
    )
    clf.fit(X, y)
    w = clf.coef_[0].astype("float64")
    b = float(clf.intercept_[0])

    # quick training AUC for sanity
    try:
        probs = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float("nan")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "features": list(FEATURES),
                "weights": [float(x) for x in w],
                "bias": b,
                "threshold": float(args.threshold),
                "tau1": float(args.tau1),
                "delta": float(args.delta),
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "train_auc": auc,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved policy to: {args.out}  (AUC={auc:.3f})")


if __name__ == "__main__":
    main()
