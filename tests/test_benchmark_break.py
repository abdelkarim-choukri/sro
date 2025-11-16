import json, os, pytest, yaml, subprocess, sys

CFG = "configs/benchmarks.yaml"

def _cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["benchmarks"]["break"]

@pytest.mark.skipif(not os.path.exists(_cfg()["dev_path"]), reason="BREAK dev fixture not found locally")
def test_break_benchmark_gates(tmp_path):
    cfg = _cfg()
    # Always point to the 200-subset to cap runtime/variance
    subset = "bench/break_dev_200.jsonl"
    if not os.path.exists(subset):
        subprocess.check_call([sys.executable, "-m", "scripts.make_break_subset",
                               "--in_path", cfg["dev_path"], "--out_path", subset,
                               "--n", str(cfg.get("limit", 200)), "--seed", "13"])
    dev = subset

    g = cfg["gates"]
    limit = int(cfg.get("limit", 200))
    timeout_sec = int(cfg.get("timeout_sec", 600))

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("SPLITTER_FORCE_MODEL", "1")

    # 1) Safety sweep (fast)
    sweep_cmd = [sys.executable, "-u", "scripts/sweep_add_prob.py",
                 "--vals", "0.50", "0.55", "0.60", "0.65",
                 "--limit", str(limit),
                 "--dev", subset]
    p = subprocess.run(sweep_cmd, capture_output=True, text=True, check=True,
                       env=env, timeout=timeout_sec)
    best_f1 = 0.0
    for L in p.stdout.splitlines():
        if L.strip().startswith("add_prob="):
            for tok in L.split():
                if tok.startswith("F1="):
                    best_f1 = max(best_f1, float(tok.split("=",1)[1]))
    if best_f1 < 0.37:
        pytest.fail(f"safety sweep failed: best {limit}-sample boundary F1={best_f1:.3f} < 0.37")

    # 2) Rules vs Hybrid ablation on the same fixed subset
    abl_cmd = [sys.executable, "-u", "scripts/ablate_rules_vs_hybrid.py",
               "--dev", dev, "--limit", str(limit)]
    p = subprocess.run(abl_cmd, capture_output=True, text=True, check=True,
                       env=env, timeout=timeout_sec)
    table = json.loads(p.stdout)

    if table.get("rules_only", {}).get("n", 0) < 50:
        pytest.skip("Sample too small for reliable gate")

    rules_f1 = float(table["rules_only"]["boundary_f1_mean"])
    hybrid_f1 = float(table["hybrid_best"]["boundary_f1_mean"])
    prt = float(table["hybrid_best"]["prt_f1_mean"])

    # Calibrate to the 200-subset (next section explains why)
    assert rules_f1 >= g["rules_only_boundary_f1_min"], f"rules-only gate failed: {rules_f1} < {g['rules_only_boundary_f1_min']}"
    assert hybrid_f1 >= g["hybrid_boundary_f1_min"], f"hybrid gate failed: {hybrid_f1} < {g['hybrid_boundary_f1_min']}"
    assert hybrid_f1 >= rules_f1 + 0.05, f"ablation failed: hybrid {hybrid_f1} not >= rules + 0.05 ({rules_f1})"
    assert prt >= g["prt_f1_min"]
import json, os, pytest, yaml, subprocess, sys

CFG = "configs/benchmarks.yaml"

def _cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["benchmarks"]["break"]

@pytest.mark.skipif(not os.path.exists(_cfg()["dev_path"]), reason="BREAK dev fixture not found locally")
def test_break_benchmark_gates(tmp_path):
    cfg = _cfg()
    dev = cfg["dev_path"]
    # Create a deterministic subset for CI to avoid long runs / pathological ordering
    subset = "bench/break_dev_200.jsonl"
    # If the subset doesn't exist or is smaller than requested, create it deterministically
    if (not os.path.exists(subset)) or (os.path.getsize(subset) == 0):
        subprocess.check_call([sys.executable, "-m", "scripts.make_break_subset",
                               "--in_path", dev, "--out_path", subset,
                               "--n", str(limit), "--seed", "13"])
    out_fixture = subset
    g = cfg["gates"]
    limit = int(cfg.get("limit", 200))
    timeout_sec = int(cfg.get("timeout_sec", 600))

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("SPLITTER_FORCE_MODEL", "1")

    # 1) Safety sweep on a small sample (fast) — fail early if model doesn't perform
    sweep_cmd = [sys.executable, "-u", "scripts/sweep_add_prob.py",
                 "--vals", "0.50", "0.55", "0.60", "0.65",
                 "--limit", str(limit),
                 "--dev", out_fixture]
    p = subprocess.run(sweep_cmd, capture_output=True, text=True, check=True,
                       env=env, timeout=timeout_sec)
    out = p.stdout.splitlines()
    best_f1 = 0.0
    for L in out:
        if L.strip().startswith("add_prob="):
            parts = L.split()
            f1_tok = next((p for p in parts if p.startswith("F1=")), None)
            if f1_tok:
                f1 = float(f1_tok.split("=",1)[1])
                best_f1 = max(best_f1, f1)
    if best_f1 < 0.37:
        pytest.fail(f"safety sweep failed: best {limit}-sample boundary F1={best_f1:.3f} < 0.37")

    # 2) Ablation on a bounded sample (avoid full 3000 in unit CI)
    abl_cmd = [sys.executable, "-u", "scripts/ablate_rules_vs_hybrid.py",
               "--dev", out_fixture, "--limit", str(limit)]
    p = subprocess.run(abl_cmd, capture_output=True, text=True, check=True,
                       env=env, timeout=timeout_sec)
    table = json.loads(p.stdout)

    # Require minimally-sized sample for stability
    if table.get("rules_only", {}).get("n", 0) < 50:
        pytest.skip(f"Sample too small for reliable gate: n={table.get('rules_only',{}).get('n',0)}")

    rules_f1 = float(table["rules_only"]["boundary_f1_mean"])
    hybrid_f1 = float(table["hybrid_best"]["boundary_f1_mean"])
    prt = float(table["hybrid_best"]["prt_f1_mean"])

    assert rules_f1 >= g["rules_only_boundary_f1_min"], f"rules-only gate failed: {rules_f1} < {g['rules_only_boundary_f1_min']}"
    assert hybrid_f1 >= g["hybrid_boundary_f1_min"], f"hybrid gate failed: {hybrid_f1} < {g['hybrid_boundary_f1_min']}"
    assert hybrid_f1 >= rules_f1 + 0.05, f"ablation failed: hybrid {hybrid_f1} not >= rules + 0.05 ({rules_f1})"
    assert prt >= g["prt_f1_min"]
