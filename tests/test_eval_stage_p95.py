import json
import subprocess
import sys

DEV_FIXTURE = "tests/fixtures/splitter_dev.jsonl"


def _run(args):
    p = subprocess.run([sys.executable, "-m", "scripts.eval_splitter"] + args,
                       capture_output=True, text=True, check=True)
    s = p.stdout
    js = json.loads(s[s.find("{"):])
    return js


def test_rules_only_stage_caps():
    js = _run(["--dev", DEV_FIXTURE, "--rules_only"])
    p95 = js["stage_p95"]
    # thresholds in milliseconds
    assert float(p95.get("tokenize_ms", 0.0)) <= 1.5
    assert float(p95.get("prt_ms", 0.0)) <= 2.0
    assert float(p95.get("rules_ms", 0.0)) <= 2.0
    assert float(p95.get("arbitrate_ms", 0.0)) <= 2.0
