from __future__ import annotations
import json
import argparse
import time
import os
import hashlib
import statistics as stats
from typing import List, Dict, Set, Tuple
from types import SimpleNamespace
import copy

from sro.claims.splitter import split_into_subclaims


def _load_cfg(path: str):
    # Minimal loader: prefer yaml if available, else try to read as JSON-like YAML
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        def ns(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: ns(v) for k, v in d.items()})
            return d
        return ns(y)
    except Exception:
        # fallback: build a minimal namespace with sensible defaults
        return SimpleNamespace(splitter=SimpleNamespace(
            variant="L3",
            min_gap_tokens=3,
            max_claims=5,
            max_len_tokens=25,
            high_conf_min_side_len=4,
            ban_pronouns=True,
            model=SimpleNamespace(onnx_path="", add_prob=0.7, remove_margin=0.35, quantize_int8=False)
        ))


def _qid(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _read_artifact_timings(qid: str) -> dict:
    # Read the last artifact record for this qid and return its timings (seconds)
    path = os.path.join(os.getcwd(), "artifacts", "splitter", f"{qid}.jsonl")
    try:
        with open(path, "r", encoding="utf-8") as f:
            last = None
            for ln in f:
                if ln.strip():
                    last = ln
        if not last:
            return {}
        rec = json.loads(last)
        return rec.get("timings", {}) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _set_boundary_f1(pred: Set[int], gold: Set[int]) -> Dict[str, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    # Special case: both empty => perfect
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _char_cover(spans: List[Tuple[int, int]], text: str) -> Set[int]:
    covered = set()
    for a, b in spans:
        for i in range(a, b):
            if i < 0 or i >= len(text):
                continue
            if not text[i].isspace():
                covered.add(i)
    return covered


def _prt_f1(pred_spans: List[Tuple[int, int]], gold_spans: List[Tuple[int, int]], text: str) -> Dict[str, float]:
    return _set_boundary_f1(_char_cover(pred_spans, text), _char_cover(gold_spans, text))


def _run_once(item, cfg, rules_only: bool):
    q = item["q"]
    cfg_local = copy.deepcopy(cfg)
    if rules_only:
        if hasattr(cfg_local.splitter, "model"):
            cfg_local.splitter.model.onnx_path = ""
    t0 = time.perf_counter_ns()
    out = split_into_subclaims(q, cfg_local)
    dt_ms = (time.perf_counter_ns() - t0) / 1e6
    return out, dt_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--config", default="configs/splitter.yaml")
    ap.add_argument("--rules_only", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = _load_cfg(args.config)
    results = []

    # warmup
    _ = split_into_subclaims("warm up, and exit.", cfg)

    for item in _read_jsonl(args.dev):
        # determinism: 3 runs, must be identical
        outs, times = [], []
        for _ in range(3):
            out, dt_ms = _run_once(item, cfg, args.rules_only)
            outs.append(out)
            times.append(dt_ms)
        if not (outs[0] == outs[1] == outs[2]):
            raise RuntimeError(f"Non-deterministic output for id={item.get('id')}")

        pred = set(outs[0]["split_points"])
        gold = set(item["gold_split_points"])
        bf1 = _set_boundary_f1(pred, gold)
        pf1 = _prt_f1(outs[0]["protected_spans"], item.get("gold_prt_spans", []), item["q"])
        # collect per-stage timings from artifacts (real measured timings), not
        # the zeroed-out timings returned by the API (used for deterministic
        # deep-equality checks).
        qid = _qid(item.get("q", ""))
        art_timings = _read_artifact_timings(qid)
        stages = {}
        stages["tokenize_ms"] = float(art_timings.get("tokenize", 0.0)) * 1000.0
        stages["prt_ms"] = float(art_timings.get("protected_spans", 0.0)) * 1000.0
        stages["rules_ms"] = float(art_timings.get("rules", 0.0)) * 1000.0
        stages["model_ms"] = float(art_timings.get("model", 0.0)) * 1000.0
        stages["arbitrate_ms"] = float(art_timings.get("arbitration", 0.0)) * 1000.0
        stages["clone_ms"] = float(art_timings.get("subject_clone", 0.0)) * 1000.0
        stages["scoring_ms"] = float(art_timings.get("scoring", 0.0)) * 1000.0

        results.append({
            "id": item.get("id"),
            "boundary": bf1,
            "prt": pf1,
            "pred_n": len(pred),
            "gold_n": len(gold),
            "delta_n": len(pred) - len(gold),
            "time_ms": times[1],
            "stages": stages,
        })

    # aggregate
    b_f1s = [r["boundary"]["f1"] for r in results]
    p_f1s = [r["prt"]["f1"] for r in results]
    over = sum(1 for r in results if r["delta_n"] > 0) / max(1, len(results))
    under = sum(1 for r in results if r["delta_n"] < 0) / max(1, len(results))
    t_ms = [r["time_ms"] for r in results]
    # aggregate per-stage p95
    def p95(seq):
        if not seq:
            return 0.0
        s = sorted(seq)
        return s[int(0.95 * (len(s) - 1))]
    stage_keys = ["tokenize_ms", "prt_ms", "rules_ms", "model_ms", "arbitrate_ms", "clone_ms", "scoring_ms"]
    stage_p95 = {k: p95([r["stages"].get(k, 0.0) for r in results]) for k in stage_keys}
    summary = {
        "n": len(results),
        "boundary_f1_mean": sum(b_f1s) / max(1, len(b_f1s)),
        "boundary_f1_median": stats.median(b_f1s) if b_f1s else 0.0,
        "prt_f1_mean": sum(p_f1s) / max(1, len(p_f1s)),
        "over_rate": over,
        "under_rate": under,
        "time_ms_median": stats.median(t_ms) if t_ms else 0.0,
        "time_ms_p95": (sorted(t_ms)[int(0.95 * (len(t_ms) - 1))] if t_ms else 0.0),
        "stage_p95": stage_p95,
    }

    os.makedirs("artifacts/eval", exist_ok=True)
    with open("artifacts/eval/splitter_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_example": results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
