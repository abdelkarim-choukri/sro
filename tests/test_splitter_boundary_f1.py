from __future__ import annotations
import json, subprocess, sys
from sro.claims.tokenizer import tokenize

DEV_FIXTURE = "tests/fixtures/splitter_dev.jsonl"

def _run(args):
    proc = subprocess.run([sys.executable, "-m", "scripts.eval_splitter"] + args,
                          capture_output=True, text=True, check=True)
    out = proc.stdout.strip()
    js = json.loads(out[out.find("{"):])
    return js


def test_rules_only_metrics_ok():
    js = _run(["--dev", DEV_FIXTURE, "--rules_only"])
    assert js["boundary_f1_mean"] >= 0.80
    assert js["prt_f1_mean"] >= 0.97
    assert js["time_ms_median"] <= 5.0


def test_hybrid_metrics_ok_or_better():
    js = _run(["--dev", DEV_FIXTURE])
    assert js["boundary_f1_mean"] >= 0.88
    assert js["prt_f1_mean"] >= 0.97
    assert js["over_rate"] <= 0.20
    assert js["under_rate"] <= 0.20
    assert js["time_ms_median"] <= 12.0


def test_per_example_invariants_rules_only():
    js = _run(["--dev", DEV_FIXTURE, "--rules_only"])
    # ensure summary file exists
    with open("artifacts/eval/splitter_summary.json", "r", encoding="utf-8") as f:
        _ = json.load(f)

    # Re-run splitter to get per-example output
    from types import SimpleNamespace
    from sro.claims.splitter import split_into_subclaims
    import yaml
    with open("configs/splitter.yaml", "r", encoding="utf-8") as f:
        cfgd = yaml.safe_load(f)
    def ns(d):
        from types import SimpleNamespace
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    cfg = ns(cfgd)
    # Force rules-only
    if hasattr(cfg.splitter, "model"):
        cfg.splitter.model.onnx_path = ""

    with open(DEV_FIXTURE, "r", encoding="utf-8") as fh:
        items = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]

    for it in items:
        q = it["q"]
        out = split_into_subclaims(q, cfg)
        toks = tokenize(q)
        token_ends = {t.end for t in toks}

        # A) boundaries align to token ends
        for c in out["split_points"]:
            assert c in token_ends, f"Split {c} not at token end for id={it['id']}"

        # B) no split inside PRT
        for c in out["split_points"]:
            for a, b in out["protected_spans"]:
                assert not (a <= c < b), f"Split inside PRT [{a},{b}) for id={it['id']}"

        # C) min-gap in tokens
        ends_ordered = sorted(out["split_points"])
        def char_to_tok_idx(ch):
            for i, t in enumerate(toks):
                if t.end == ch:
                    return i
            return -1
        idxs = [char_to_tok_idx(c) for c in ends_ordered]
        for a, b in zip(idxs, idxs[1:]):
            assert b - a >= cfg.splitter.min_gap_tokens, f"Min-gap violated for id={it['id']}"

        # D) claim filters: length and offsets valid
        for cl in out["claims"]:
            assert len(tokenize(cl["text"])) <= cfg.splitter.max_len_tokens
            assert 0 <= cl["start"] < cl["end"] <= len(q)
            assert q[cl["start"]:cl["end"]].strip() == cl["text"]
