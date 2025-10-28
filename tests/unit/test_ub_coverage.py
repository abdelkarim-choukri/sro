from __future__ import annotations

import csv
import os
from pathlib import Path

from sro.config import apply_env_overrides, load_config
from sro.prover.s4_ub import UBWeights, clamp01, upper_bound


def _UB(feats, kappa, w):
    try:
        return upper_bound(feats, kappa, w)  # positional
    except TypeError:
        return upper_bound(feats, kappa=kappa, ub_weights=w)

def test_dev_pairs_coverage_ge_975():
    cfg = load_config()
    apply_env_overrides(cfg)
    kappa = float(cfg.sro_prover.kappa)
    p2, ub = [], []
    w = UBWeights()
    path = Path("data/processed/dev_pairs.csv")
    assert path.exists(), "Generate with scripts.make_dev_pairs first"
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p2v = float(r["p2"])
            feats = {
                k: float(r[k]) for k in [
                    "max_p1","entity_overlap","time_agreement","distance",
                    "novelty","ce_max","negation_conflict","source_diversity"
                ] if k in r and r[k] != ""
            }
            if not feats:  # skip if row malformed
                continue
            p2.append(p2v)
            ub.append(clamp01(_UB(feats, kappa, w)))
    assert len(p2) > 0
    covered = sum(1 for a,b in zip(p2,ub) if a <= b + 1e-12) / len(p2)
    assert covered >= 0.975, f"Coverage {covered:.4f} < 0.975 at kappa={kappa}"
