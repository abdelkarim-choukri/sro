from __future__ import annotations

import argparse
import json
import os
import yaml
from sro.claims.splitter import split_into_subclaims


def _load_cfg():
    # Return raw dict so the splitter’s internal normalizer can handle nested keys
    with open("configs/splitter.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0.0 else 2 * p * r / (p + r)


def _boundary_pr(gold: list[int], pred: list[int]) -> tuple[float, float, float]:
    # If both sides are empty, count as perfect for this item.
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    g, p = set(gold), set(pred)
    tp = len(g & p); fp = len(p - g); fn = len(g - p)
    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec  = 0.0 if tp + fn == 0 else tp / (tp + fn)
    return prec, rec, _f1(prec, rec)


def _prt_pr(gold_spans: list[list[int]], pred_spans: list[list[int]]) -> tuple[float, float, float]:
    # exact-span matching; splitter’s PRT is strict already
    if not gold_spans and not pred_spans:
        return 1.0, 1.0, 1.0
    g = {tuple(x) for x in gold_spans}
    p = {tuple(x) for x in pred_spans}
    tp = len(g & p); fp = len(p - g); fn = len(g - p)
    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec  = 0.0 if tp + fn == 0 else tp / (tp + fn)
    return prec, rec, _f1(prec, rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--debug", action="store_true", help="print first 3 mismatches")
    args = ap.parse_args()

    cfg = _load_cfg()
    force = os.environ.get("SPLITTER_FORCE_MODEL", "").lower() in ("1", "true", "yes")

    n = 0
    b_f1_sum = 0.0
    p_f1_sum = 0.0
    dbg_left = 3

    with open(args.dev, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if args.limit and i >= args.limit:
                break
            row = json.loads(line)
            q = row["q"]
            gold_b = row.get("gold_split_points", [])
            gold_p = row.get("gold_prt_spans", [])

            out = split_into_subclaims(q, cfg)  # passes raw dict; splitter normalizes
            pred_b = out.get("split_points", [])
            pred_p = out.get("protected_spans", [])

            _, _, bf1 = _boundary_pr(gold_b, pred_b)
            _, _, pf1 = _prt_pr(gold_p, pred_p)

            if args.debug and bf1 < 1.0 and dbg_left > 0:
                print("\n[DEBUG] mismatch sample")
                print("Q:", q)
                print("gold_b:", gold_b)
                print("pred_b:", pred_b)
                print("gold_p:", gold_p)
                print("pred_p:", pred_p)
                print("tele:", out.get("telemetry", {}))
                dbg_left -= 1

            b_f1_sum += bf1
            p_f1_sum += pf1
            n += 1

    res = {
        "n": n,
        "force_model": force,
        "boundary_f1_mean": (b_f1_sum / n) if n else 0.0,
        "prt_f1_mean":      (p_f1_sum / n) if n else 0.0,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
