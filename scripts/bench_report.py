from __future__ import annotations
import json, os
from types import SimpleNamespace
from sro.claims.splitter import split_into_subclaims

def load_fixture(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def run_report(fixture_path: str, max_examples: int = 20):
    rows = load_fixture(fixture_path)
    cfg = SimpleNamespace(splitter=SimpleNamespace(min_gap_tokens=3, variant="L3", model=SimpleNamespace(onnx_path="")))
    failures = []
    for i, r in enumerate(rows):
        q = r["q"]
        out = split_into_subclaims(q, cfg=cfg)
        pred = out.get("split_points", [])
        gold = r.get("gold_split_points", [])
        if pred != gold:
            failures.append({"i": i, "q": q, "pred": pred, "gold": gold, "pred_n": len(pred), "gold_n": len(gold)})
        if len(failures) >= max_examples:
            break

    total = len(rows)
    n_fail = len(failures)
    print(f"Total examples: {total}; First {n_fail} failures (of up to {max_examples} shown):\n")
    for f in failures:
        print("---")
        print(f"idx: {f['i']}")
        print(f"q: {f['q']}")
        print(f"pred splits: {f['pred']} (n={f['pred_n']})")
        print(f"gold splits: {f['gold']} (n={f['gold_n']})")
        print()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', default='bench/break_dev_fixture.jsonl')
    ap.add_argument('--max', type=int, default=20)
    args = ap.parse_args()
    if not os.path.exists(args.dev):
        print('Fixture not found:', args.dev)
        raise SystemExit(1)
    run_report(args.dev, args.max)
