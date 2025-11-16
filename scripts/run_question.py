# scripts/run_question.py
"""
Runner without SROProver:
- Early logging (respects SRO_LOGLEVEL)
- Seed discipline
- Single NLIBackend (logs calibrated temperature)
- --redundancy (default: cosine), passed along when possible
- Demos: --demo_alt, --demo_global_safety
- Observability (P8): structured traces JSONL + optional HTML report (--debug_html)
- Calls legacy entrypoints if present (in this module or scripts.run_question_legacy)
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import random
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch

from sro.nli.backend import NLIBackend  # logs calibrated T on init
from sro.trace.recorder import TraceRecorder


# ---------------- seed discipline ----------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- logging config ----------------
logging.basicConfig(
    level=os.environ.get("SRO_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------- helpers: conflict probs ----------------
def _extract_conflict_probs(out: dict, backend) -> np.ndarray:
    """
    Robustly extract conflict probabilities:
      - If labels include 'contradiction', use that column.
      - Else if binary with 'not_entailment', use that column.
      - Else, if backend exposes label map, pick the right index.
    """
    labels = None
    if "labels" in out:
        labels = out["labels"]
    elif "label_names" in out:
        labels = out["label_names"]
    elif "labels_names" in out:
        labels = out["labels_names"]

    probs = None
    if "probs" in out:
        probs = out["probs"]
    elif "probabilities" in out:
        probs = out["probabilities"]

    if probs is None:
        # No matrix? try per-label arrays
        if "contradiction" in out:
            return np.asarray(out["contradiction"], dtype="float32")
        if "not_entailment" in out:
            return np.asarray(out["not_entailment"], dtype="float32")
        raise KeyError(f"NLI output missing probs: keys={list(out.keys())}")

    probs = np.asarray(probs, dtype="float32")
    if labels is not None:
        labels = [str(x).lower() for x in list(labels)]
        if "contradiction" in labels:
            return probs[:, labels.index("contradiction")].astype("float32")
        if "not_entailment" in labels:
            return probs[:, labels.index("not_entailment")].astype("float32")
        raise KeyError(f"NLI labels missing 'contradiction'/'not_entailment': {labels}")

    # Fall back to backend mappings
    if hasattr(backend, "label_to_index"):
        m = backend.label_to_index
        for name in ("contradiction", "not_entailment"):
            if isinstance(m, dict) and name in m:
                return probs[:, int(m[name])].astype("float32")
    if hasattr(backend, "id2label"):
        d = backend.id2label
        if isinstance(d, dict):
            inv = {str(v).lower(): int(k) for k, v in d.items()}
            for name in ("contradiction", "not_entailment"):
                if name in inv:
                    return probs[:, inv[name]].astype("float32")
    raise KeyError("Cannot resolve conflict column index.")


def _print_config_banner(profile: str, seed: int, kappa_in: float) -> None:
    if profile == "med":
        M, L, B = 8, 24, 64
        tau1, tau2, delta, kappa, eps = 0.75, 0.80, 0.10, kappa_in, 0.02
    elif profile == "tiny":
        M, L, B = 4, 12, 32
        tau1, tau2, delta, kappa, eps = 0.70, 0.78, 0.10, kappa_in, 0.02
    else:
        M, L, B = 8, 24, 64
        tau1, tau2, delta, kappa, eps = 0.75, 0.80, 0.10, kappa_in, 0.02
    logging.info(
        "CONFIG M=%s L=%s B=%s tau1=%.2f tau2=%.2f delta=%.2f kappa=%.2f eps=%.2f seed=%d",
        M, L, B, tau1, tau2, delta, kappa, eps, seed
    )



# ---------------- legacy runner helpers ----------------
def _call_if_exists(obj: object, name: str, *args, **kwargs):
    if hasattr(obj, name):
        fn = getattr(obj, name)
        if callable(fn):
            return fn(*args, **kwargs)
    return None


def _run_legacy_inplace(args: argparse.Namespace, rec: TraceRecorder | None) -> bool:
    """
    Try to run legacy entrypoints defined in THIS module:
    run_question(question) -> run() -> run_cli(args) -> answer(question) -> main(args)
    Return True if any ran without raising.
    """
    try_order = [
        ("run_question", (args.question,) if args.question else ()),
        ("run", ()),
        ("run_cli", (args,)),
        ("answer", (args.question,)) if args.question else None,
        ("main", (args,)),
    ]
    for item in try_order:
        if not item:
            continue
        name, argv = item
        try:
            out = _call_if_exists(globals(), name, *argv)
            if out is not None or name in ("run", "run_cli", "main"):
                _pretty_print_result(out)
                if rec:
                    rec.log("LEGACY_RAN", stage="S9", data={"entry": name})
                    rec.close()
                return True
        except Exception as e:
            logging.error("Legacy %s() in scripts.run_question failed: %s", name, e)
            logging.debug("%s", traceback.format_exc())
    return False


def _run_legacy_module(args: argparse.Namespace, rec: TraceRecorder | None) -> bool:
    """
    Try scripts.run_question_legacy module: prefers main(args) then run(args) then run().
    """
    try:
        legacy = importlib.import_module("scripts.run_question_legacy")
    except ModuleNotFoundError:
        return False
    except Exception:
        logging.error("Failed to import scripts.run_question_legacy:\n%s", traceback.format_exc())
        return False

    for name in ("main", "run"):
        try:
            if hasattr(legacy, name):
                fn = getattr(legacy, name)
                out = fn(args) if "args" in inspect.signature(fn).parameters else fn()
                _pretty_print_result(out)
                if rec:
                    rec.log("LEGACY_RAN", stage="S9", data={"module": "scripts.run_question_legacy", "entry": name})
                    rec.close()
                return True
        except Exception as e:
            logging.error("scripts.run_question_legacy.%s() failed: %s", name, e)
            logging.debug("%s", traceback.format_exc())
    return False


def _pretty_print_result(out: dict[str, Any] | None) -> None:
    if not isinstance(out, dict):
        return
    final = out.get("final_answer") or out.get("answer") or ""
    if final:
        print(f"FINAL ANSWER: {final}")
    claims = out.get("claims_accepted") or out.get("claims") or []
    if claims:
        print(f"CLAIMS ACCEPTED: {len(claims)}")
        for c in claims:
            score = c.get("score", "?")
            margin = c.get("margin", "?")
            cites = c.get("cites", [])
            cid = c.get("id") or c.get("qid") or "c?"
            print(f"  - {cid}: score={score} margin={margin} cites={cites}")
    refs = out.get("references") or out.get("refs") or {}
    if isinstance(refs, dict) and refs:
        print("REFERENCES:")
        for k, v in refs.items():
            print(f"  [{k}] {v}")
    rt = out.get("runtime") or out.get("runtimes") or {}
    if isinstance(rt, dict) and rt:
        rline = " ".join(f"S{k}={v}" for k, v in rt.items() if isinstance(v, str))
        if rline:
            print(f"RUNTIME {rline}")


# ---------------- helpers: entailment probs & inputs ----------------
def _entail_idx_from_backend(backend) -> int:
    if hasattr(backend, "label_to_index"):
        m = backend.label_to_index
        if isinstance(m, dict) and "entailment" in m:
            return int(m["entailment"])
    if hasattr(backend, "index_to_label"):
        for i, name in enumerate(backend.index_to_label):
            if str(name).lower() == "entailment":
                return int(i)
    if hasattr(backend, "id2label"):
        d = backend.id2label
        if isinstance(d, dict):
            for k, v in d.items():
                if str(v).lower() == "entailment":
                    try:
                        return int(k)
                    except Exception:
                        pass
    raise KeyError("Cannot infer entailment index from backend; need 'entailment' mapping.")


def _extract_entailment_probs(out: dict, backend) -> np.ndarray:
    labels = None
    if "labels" in out:
        labels = out["labels"]
    elif "label_names" in out:
        labels = out["label_names"]
    elif "labels_names" in out:
        labels = out["labels_names"]

    probs = None
    if "probs" in out:
        probs = out["probs"]
    elif "probabilities" in out:
        probs = out["probabilities"]

    if probs is None:
        if "entailment" in out:
            return np.asarray(out["entailment"], dtype="float32")
        raise KeyError(f"NLI output missing probs: keys={list(out.keys())}")

    probs = np.asarray(probs, dtype="float32")
    if labels is not None:
        labels = [str(x).lower() for x in list(labels)]
        if "entailment" not in labels:
            raise KeyError(f"NLI labels missing 'entailment': {labels}")
        eidx = labels.index("entailment")
        return probs[:, eidx].astype("float32")
    # no labels → use backend map
    eidx = _entail_idx_from_backend(backend)
    if probs.ndim == 1:
        return probs.astype("float32")
    if probs.ndim == 2 and 0 <= eidx < probs.shape[1]:
        return probs[:, eidx].astype("float32")
    raise ValueError(f"Unexpected probs shape: {probs.shape}")


def _read_claim(args: argparse.Namespace) -> str:
    if args.question:
        return args.question
    path = os.path.join("data", "processed", "dev_claims.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError("No --question provided and data/processed/dev_claims.csv not found.")
    df = pd.read_csv(path)
    for col in ("claim", "hypothesis", "h"):
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val and val.strip():
                return val.strip()
    raise KeyError("dev_claims.csv present but has no claim/hypothesis column.")


def _read_corpus_sentences(path: str = "data/corpus/sentences.txt") -> list[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Corpus file not found: {path}")
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    if not out:
        raise ValueError(f"Corpus file is empty: {path}")
    return out


# ---------------- demos (wired with TraceRecorder) ----------------
def _demo_alternation(nli, args: argparse.Namespace, rec: TraceRecorder | None = None) -> None:
    """
    Demo: score a single claim vs corpus sentences, pick top-2 p1, call alternation policy.
    Purely for wiring; no side effects beyond logs.
    """
    log = logging.getLogger("demo.alt")
    from sro.prover.s4_ub import upper_bound
    from sro.prover.s8_alternation import decide_alternation_from_pair_scores

    claim = _read_claim(args)
    sents = _read_corpus_sentences()
    # score in batches
    p1: list[float] = []
    bs = max(1, int(args.bs_nli1))
    for i in range(0, len(sents), bs):
        out = nli.score_pairs(sents[i:i + bs], [claim] * len(sents[i:i + bs]), batch_size=bs)
        p1.extend(_extract_entailment_probs(out, nli).tolist())
    p1_arr = np.asarray(p1, dtype="float32")
    if p1_arr.size == 0:
        raise RuntimeError("No scores produced.")

    # top-2
    order = np.argsort(-p1_arr)
    i1 = int(order[0])
    i2 = int(order[1]) if p1_arr.size > 1 else i1
    best_i, best_j = float(p1_arr[i1]), float(p1_arr[i2])

    # compute UB for logging (policy also computes one internally)
    ub_feats = {
        "p1_i": best_i, "p1_j": best_j,
        "best_so_far": max(best_i, best_j), "p2": 0.0, "max_p1": max(best_i, best_j),
        "entity_overlap": 0.0, "time_agreement": 0.0, "distance": 0.0, "novelty": 0.0,
        "ce_max": 0.0, "negation_conflict": 0.0, "source_diversity": 0.0,
    }
    top_ub = float(upper_bound(ub_feats, kappa=getattr(args, "kappa", 0.0)))


    dec = decide_alternation_from_pair_scores(
        best_i, best_j, alternations_used=0, budget_left_norm=1.0
    )
    log.info(
        "ALT demo: claim=%r | top1=%.3f [%s]  top2=%.3f [%s]  UB=%.3f  -> %s",
        claim,
        best_i, sents[i1][:80],
        best_j, sents[i2][:80],
        top_ub,
        dec,
    )
    if rec:
        rec.log("ALT_DECISION", stage="S8", data={
            "claim": claim,
            "top1": float(best_i),
            "top2": float(best_j),
            "ub": float(top_ub),
            "decision": dec,
        })


def _demo_global_safety(nli, args: argparse.Namespace, rec: TraceRecorder | None = None) -> None:
    """
    Demo: take two claims, compute best entailment and Cmax_local (conflict) vs toy corpus,
    enforce global safety across the two in sequence.
    """
    from sro.safety.cross_claim import CrossClaimSafety

    log = logging.getLogger("demo.safety")

    # Claims: use --question for the first; hard-code a conflicting second
    claim1 = args.question or "The iPhone 15 Pro features a titanium frame."
    claim2 = "The iPhone 15 base model has titanium."

    sents = _read_corpus_sentences()
    bs = max(1, int(args.bs_nli1))

    def _score_pairwise(claim: str) -> tuple[float, float, str, str]:
        ent_scores: list[float] = []
        conf_scores: list[float] = []
        for i in range(0, len(sents), bs):
            batch = sents[i:i + bs]
            out = nli.score_pairs(batch, [claim] * len(batch), batch_size=bs)
            ent_scores.extend(_extract_entailment_probs(out, nli).tolist())
            conf_scores.extend(_extract_conflict_probs(out, nli).tolist())
        ent = np.asarray(ent_scores, dtype="float32")
        con = np.asarray(conf_scores, dtype="float32")
        i_best = int(np.argmax(ent))
        i_conf = int(np.argmax(con))
        return float(ent[i_best]), float(con[i_conf]), sents[i_best], sents[i_conf]

    safety = CrossClaimSafety(delta=0.10)

    # Claim 1
    b1, c1, s_best1, s_conf1 = _score_pairwise(claim1)
    ok1, info1 = safety.check_and_maybe_update(b1, c1)
    log.info(
        "C1: best=%.3f (\"%s\") | Cmax_local=%.3f (\"%s\") | Cmax_global=%.3f | margin=%.3f >= δ=%.2f ? %s (%s)",
        b1, s_best1[:80], c1, s_conf1[:80], info1["cmax_global"], info1["margin"], info1["threshold"], ok1, info1["reason"]
    )
    if rec:
        rec.log("SAFETY_CLAIM", stage="S7", data={"idx": 1, "claim": claim1, "decision": info1})

    # Claim 2
    b2, c2, s_best2, s_conf2 = _score_pairwise(claim2)
    ok2, info2 = safety.check_and_maybe_update(b2, c2)
    log.info(
        "C2: best=%.3f (\"%s\") | Cmax_local=%.3f (\"%s\") | Cmax_global=%.3f | margin=%.3f >= δ=%.2f ? %s (%s)",
        b2, s_best2[:80], c2, s_conf2[:80], info2["cmax_global"], info2["margin"], info2["threshold"], ok2, info2["reason"]
    )
    if not ok2 and info2["reason"] == "SAFETY_MARGIN_FAIL":
        log.info("blocked_by_global_conflict=true")
    if rec:
        rec.log("SAFETY_CLAIM", stage="S7", data={"idx": 2, "claim": claim2, "decision": info2})


# ---------------- main ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", type=str, default="med", choices=["tiny", "med", "large"])
    ap.add_argument("--bs_nli1", type=int, default=32)
    ap.add_argument("--bs_nli2", type=int, default=32)
    ap.add_argument("--nli_model_dir", type=str, default=None, help="Explicit local dir for NLI model")
    ap.add_argument("--question", type=str, default=None, help="Optional single question to run")
    ap.add_argument("--redundancy", type=str, default="cosine", help="cosine|jaccard (default: cosine)")
    ap.add_argument("--demo_alt", action="store_true", help="Demo the learned alternation policy using corpus sentences")
    ap.add_argument("--demo_global_safety", action="store_true", help="Demo cross-claim safety (global contradiction margin)")
    # Observability (P8)
    ap.add_argument("--trace_out", type=str, default="artifacts/logs/traces.jsonl")
    ap.add_argument("--debug_html", action="store_true", help="Emit a standalone HTML report next to traces")
    ap.add_argument("--kappa", type=float, default=0.80, help="Optimism for UB in [0,1] (demo only)")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    _print_config_banner(args.profile, args.seed, args.kappa)


    # Trace recorder
    rec = TraceRecorder(args.trace_out, meta={"seed": args.seed, "profile": args.profile, "redundancy": args.redundancy})

    # UB warm-load (logs path)
    try:
        import sro.prover.s4_ub as _S4UB  # noqa: F401
        rec.log("UB_INFO", stage="S4", data={"path": os.environ.get("SRO_UB_DIR", "artifacts/ub")})
    except Exception:
        import traceback as _tb
        logging.getLogger(__name__).error("UB warm-load failed:\n%s", _tb.format_exc())

    # Alternation warm-load (log only)
    try:
        policy_path = os.environ.get("SRO_ALT_POLICY", "artifacts/alternation/policy.json")
        logging.getLogger("sro.prover.s8_alternation").info("ALT_POLICY probe path: %s", policy_path)
        if os.path.isfile(policy_path):
            from sro.prover.s8_alternation import AlternationPolicy, PolicyMeta
            _meta = PolicyMeta.load(policy_path)
            _ = AlternationPolicy(_meta)
            logging.getLogger("sro.prover.s8_alternation").info(
                "ALT_POLICY loaded: %s  features=%s  thr=%.2f",
                policy_path, ",".join(_meta.features), _meta.threshold
            )
            rec.log("ALT_POLICY_INFO", stage="S8", data={"path": policy_path, "features": list(_meta.features), "threshold": float(_meta.threshold)})
        else:
            logging.getLogger("sro.prover.s8_alternation").info("ALT_POLICY not found at path; skipping.")
            rec.log("ALT_POLICY_INFO", stage="S8", data={"path": None})
    except Exception:
        import traceback as _tb
        logging.getLogger(__name__).warning("ALT_POLICY warm-load skipped:\n%s", _tb.format_exc())

    # NLI backend
    model_dir = args.nli_model_dir or os.environ.get("SRO_NLI_MODEL_DIR", None)
    nli = NLIBackend(model_name=model_dir)    
    t_eff = float(nli.get_temperature())
    logging.getLogger("sro.nli.backend").info("NLI temperature = %.3f (effective)", t_eff)
    rec.log("NLI_INIT", stage="S1", data={"temperature": t_eff})

    # ---- DEMO: global safety ----
    if args.demo_global_safety:
        _demo_global_safety(nli, args, rec=rec)
        _maybe_render_html(args)
        rec.close()
        return

    # ---- DEMO: alternation ----
    if args.demo_alt:
        _demo_alternation(nli, args, rec=rec)
        _maybe_render_html(args)
        rec.close()
        return

    # Legacy hooks (unchanged)
    if _run_legacy_inplace(args, rec):
        _maybe_render_html(args)
        return
    if _run_legacy_module(args, rec):
        _maybe_render_html(args)
        return

    # No runner found
    rec.log("ERROR", stage="S9", data={"msg": "No runnable pipeline found"})
    _maybe_render_html(args)
    rec.close()
    raise RuntimeError(
        "No runnable pipeline found. "
        "Define a V1 runner or use --demo_alt / --demo_global_safety to exercise wiring."
    )


def _maybe_render_html(args: argparse.Namespace) -> None:
    """Render one-file HTML if --debug_html and trace exists."""
    if not args.debug_html:
        return
    try:
        if not os.path.isfile(args.trace_out):
            return
        out_html = os.path.join(os.path.dirname(os.path.abspath(args.trace_out)), "report.html")
        # Call scripts.trace_view.main programmatically
        import sys

        from scripts.trace_view import main as _render
        old = sys.argv[:]
        sys.argv = ["trace_view.py", "--input", args.trace_out, "--out", out_html]
        _render()
        sys.argv = old
    except Exception as e:
        logging.getLogger(__name__).warning("debug_html render failed: %s", e)


if __name__ == "__main__":
    main()
