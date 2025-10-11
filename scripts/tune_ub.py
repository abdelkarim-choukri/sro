# scripts/tune_ub.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sro.config import load_config, apply_env_overrides
from sro.prover.s4_ub import upper_bound, UBWeights, clamp01

KNOWN_FEATS = {
    "max_p1","entity_overlap","time_agreement","distance","novelty","ce_max",
    "negation_conflict","source_diversity",
}

def _to_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def _parse_feats_from_row(row: Dict[str, str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    # explicit columns first
    for k in KNOWN_FEATS:
        if k in row and row[k] not in (None, ""):
            feats[k] = _to_float(row[k], 0.0)
    if feats:
        return feats
    # JSON blob fallback
    for key in ("feats","features","feats_json","features_json"):
        raw = row.get(key)
        if not raw: continue
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                for k, v in d.items():
                    if k in KNOWN_FEATS:
                        feats[k] = _to_float(v, 0.0)
                if feats:
                    return feats
        except Exception:
            pass
    return feats

def _pick_p2(row: Dict[str, str]) -> Optional[float]:
    for key in ("p2","p_entail","p_true","p","score"):
        if key in row and row[key] not in (None, ""):
            return _to_float(row[key], None)
    return None

def _pick_ub_column(row: Dict[str, str]) -> Optional[float]:
    for key in ("UB","ub","upper_bound"):
        if key in row and row[key] not in (None, ""):
            return _to_float(row[key], None)
    return None

def _load_pairs(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

# Be compatible with positional/keyword UB signatures
def _UB(feats: Dict[str, float], kappa: float, w: UBWeights) -> float:
    try:
        return upper_bound(feats, kappa, w)  # positional
    except TypeError:
        return upper_bound(feats, kappa=kappa, ub_weights=w)  # keyword

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="CSV of dev pairs (needs p2 and S3 features or a feats JSON).")
    ap.add_argument("--kappa", type=float, default=None,
                    help="κ to evaluate; default = cfg.sro_prover.kappa (after env overrides).")
    ap.add_argument("--target", type=float, default=0.975,
                    help="Target coverage for p2 ≤ UB(κ).")
    ap.add_argument("--print_samples", type=int, default=0,
                    help="Print this many worst offenders (p2-UB).")
    args = ap.parse_args()

    cfg = load_config()
    apply_env_overrides(cfg)
    kappa = float(args.kappa) if args.kappa is not None else float(cfg.sro_prover.kappa)
    target = float(args.target)

    rows = _load_pairs(Path(args.input))
    if not rows:
        print(json.dumps({"error":"no_rows"}, ensure_ascii=False)); return

    w = UBWeights()
    p2_list: List[float] = []
    ub0_list: List[Optional[float]] = []
    ubk_list: List[float] = []
    used_fallback = False

    for r in rows:
        p2 = _pick_p2(r)
        if p2 is None:
            continue
        feats = _parse_feats_from_row(r)
        if feats:
            ub0 = clamp01(_UB(feats, 0.0, w))
            ubk = clamp01(_UB(feats, kappa, w))
        else:
            ub_from_col = _pick_ub_column(r)
            if ub_from_col is None:
                continue
            ub0 = None
            ubk = clamp01(_to_float(ub_from_col, 0.0))
            used_fallback = True

        p2_list.append(float(p2))
        ub0_list.append(ub0)
        ubk_list.append(ubk)

    N = len(p2_list)
    if N == 0:
        print(json.dumps({"error":"no_usable_rows"}, ensure_ascii=False)); return

    p2_arr = np.asarray(p2_list, dtype=np.float64)
    ubk_arr = np.asarray(ubk_list, dtype=np.float64)

    covered = (p2_arr <= ubk_arr + 1e-12)
    coverage = float(np.mean(covered))
    calib_error = float(np.mean(np.maximum(0.0, p2_arr - ubk_arr)))

    # A) Suggestion from base UB(0)
    deficits_base = []
    for i in range(N):
        ub0 = ub0_list[i]
        if ub0 is not None:
            d = max(0.0, p2_list[i] - ub0)
        else:
            approx_base = max(0.0, ubk_list[i] - kappa)
            d = max(0.0, p2_list[i] - approx_base)
        deficits_base.append(d)
    q_needed_base = float(np.quantile(np.asarray(deficits_base, dtype=np.float64), target))
    kappa_from_base = float(min(1.0, q_needed_base))

    # B) Suggestion from current κ (increment by the target quantile of current gaps)
    gaps_current = np.maximum(0.0, p2_arr - ubk_arr)
    inc_needed = float(np.quantile(gaps_current, target))
    kappa_from_current = float(min(1.0, kappa + inc_needed))

    kappa_suggest = None
    if coverage < target:
        kappa_suggest = float(max(kappa, max(kappa_from_base, kappa_from_current)))

    out = {
        "N": N,
        "coverage": coverage,
        "target": target,
        "calibration_error": calib_error,
        "kappa_current": kappa,
        "kappa_suggested": kappa_suggest,
        "kappa_from_base": kappa_from_base,
        "kappa_from_current": kappa_from_current,
        "used_precomputed_ub_fallback": used_fallback,
        "notes": (
            "suggestion computed from both base UB(0) and current κ; choose the max"
            if not used_fallback else
            "approximate base used when features missing; current-κ suggestion valid"
        ),
    }
    print(json.dumps(out, indent=2, sort_keys=False))

    if args.print_samples > 0:
        idx = np.argsort(gaps_current)[::-1]
        print("WORST_UNDER_ESTIMATES (p2 - UB(kappa)):")
        for j in idx[: min(args.print_samples, N)]:
            print(f"  #{j}: p2={p2_arr[j]:.4f} ubk={ubk_arr[j]:.4f} gap={gaps_current[j]:.4f}")

if __name__ == "__main__":
    main()
