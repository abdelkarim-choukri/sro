from __future__ import annotations
import argparse, json, os, subprocess, sys, yaml

CFG_PATH = "configs/splitter.yaml"

def _run_eval(dev_path: str, limit: int | None, rules_only: bool):
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    if not rules_only:
        env.setdefault("SPLITTER_FORCE_MODEL", "1")
    cmd = [sys.executable, "-u", "scripts/bench_eval.py", "--dev", dev_path]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    out = subprocess.check_output(cmd, text=True, env=env)
    return json.loads(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="bench/break_dev_fixture.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="limit examples (0=all)")
    args = ap.parse_args()

    orig_cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))
    best_add = float(orig_cfg["splitter"]["model"]["add_prob"])

    # Rules-only ablation: write a temporary rules-only config (disable onnx path)
    cfg0 = dict(orig_cfg)
    cfg0 = cfg0.copy()
    cfg0.setdefault("splitter", {}).setdefault("model", {})
    cfg0["splitter"]["model"]["add_prob"] = 1.0
    cfg0["splitter"]["model"]["remove_margin"] = 9.99
    cfg0["splitter"]["model"]["onnx_path"] = ""
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg0, f, sort_keys=False, allow_unicode=True)
    rules = _run_eval(args.dev, args.limit or None, rules_only=True)

    # Hybrid-best ablation: restore original config but set best add_prob and ensure onnx_path is present
    cfg1 = dict(orig_cfg)
    cfg1.setdefault("splitter", {}).setdefault("model", {})
    cfg1["splitter"]["model"]["add_prob"] = best_add
    # ensure onnx_path is present (keep original value)
    if not cfg1["splitter"]["model"].get("onnx_path"):
        cfg1["splitter"]["model"]["onnx_path"] = orig_cfg["splitter"]["model"].get("onnx_path", "")
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg1, f, sort_keys=False, allow_unicode=True)
    hybrid = _run_eval(args.dev, args.limit or None, rules_only=False)

    # restore original config file
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(orig_cfg, f, sort_keys=False, allow_unicode=True)

    print(json.dumps({
        "rules_only": rules,
        "hybrid_best": hybrid,
        "best_add_prob": best_add
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
