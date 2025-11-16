from __future__ import annotations
import argparse, json, os, subprocess, sys
import yaml

CFG_PATH = "configs/splitter.yaml"

def run_eval(dev_path: str, limit: int | None):
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("SPLITTER_FORCE_MODEL", "1")  # force model for sweep
    cmd = [sys.executable, "-u", "scripts/bench_eval.py", "--dev", dev_path]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    out = subprocess.check_output(cmd, text=True, env=env)
    return json.loads(out)

def update_cfg(add_prob: float|None, remove_margin: float|None = None) -> dict:
    cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))
    if add_prob is not None:
        cfg["splitter"]["model"]["add_prob"] = float(add_prob)
    if remove_margin is not None:
        cfg["splitter"]["model"]["remove_margin"] = float(remove_margin)
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", nargs="+", type=float, required=True, help="candidate add_prob values")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--dev", default="bench/break_dev_200.jsonl")
    ap.add_argument("--full", action="store_true", help="also run the best on full dataset")
    args = ap.parse_args()

    print(f"\nSWEEP (limit={args.limit}):")
    best = (None, -1.0)
    for v in args.vals:
        update_cfg(add_prob=v)
        js = run_eval(args.dev, args.limit if args.limit > 0 else None)
        f1 = js["boundary_f1_mean"]
        print(f"add_prob={v:.2f}  n={js['n']}  F1={f1:.3f}  PRT={js['prt_f1_mean']:.3f}")
        if f1 > best[1]:
            best = (v, f1)

    print(f"\nBEST -> add_prob={best[0]}")

    if args.full:
        js = run_eval(args.dev, None)
        print(json.dumps(js, ensure_ascii=False))

if __name__ == "__main__":
    main()
