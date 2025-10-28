# scripts/eval_claims.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from sro.config import load_config
from sro.prover import SROProver
from sro.retrieval.hybrid import get_initial_candidates, make_fetch_more
from sro.utils.random import set_all_seeds

# Optional cross-encoder reranker (offline-capable; will silently skip if not available)
try:
    from sro.rerank.cross_encoder import CrossEncoderReranker
except Exception:
    CrossEncoderReranker = None  # type: ignore


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_dir(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with qid,question,claim,label,evidence_ids")
    ap.add_argument("--output", required=True, help="Where to write per-row outputs CSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs_nli1", type=int, default=32)
    ap.add_argument("--bs_nli2", type=int, default=32)
    args = ap.parse_args()

    # 1) Determinism first
    set_all_seeds(args.seed)

    # 2) Config & components
    cfg = load_config()
    corpus_path = Path(str(cfg.retrieval.corpus_jsonl))
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}. Build it or run your demo warmup first.")

    # Optional reranker (safe in offline mode)
    reranker = None
    if getattr(cfg.retrieval, "use_cross_encoder", False) and CrossEncoderReranker is not None:
        try:
            reranker = CrossEncoderReranker()
        except Exception:
            reranker = None

    fetch_more = make_fetch_more(
        str(corpus_path),
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        k_fused=cfg.retrieval.k_fused,
    )

    prover = SROProver(cfg, use_real_nli=True, bs_nli1=args.bs_nli1, bs_nli2=args.bs_nli2)
    delta = float(cfg.sro_prover.delta)

    # 3) IO
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = _read_csv(in_path)

    out_rows: list[dict[str, Any]] = []
    n = len(rows)

    # Metrics buckets
    accept_T = accept_F = reject = abstain = safety_viol = 0
    hop1 = hop2 = 0

    # Confusion counts
    TP = FP = TN = FN = 0

    for r in rows:
        qid = r.get("qid", "")
        claim_text = r.get("claim", "") or ""
        label = (r.get("label", "") or "").strip().lower()

        # Retrieval per-claim (use claim text as query for proofing)
        init_claim = get_initial_candidates(
            str(corpus_path),
            claim_text,
            k_bm25=cfg.retrieval.k_bm25,
            k_dense=cfg.retrieval.k_dense,
            k_fused=cfg.retrieval.k_fused,
            mmr_lambda=cfg.retrieval.mmr_lambda,
            rrf_c=cfg.retrieval.rrf_c,
            use_cross_encoder=bool(reranker is not None),
            cross_encoder=reranker,
            rerank_top=cfg.retrieval.rerank_top,
        )

        # Prove
        from sro.types import Claim
        cl = Claim(claim_id=qid or "eval_claim", text=claim_text)
        res = prover.prove(cl, init_claim, fetch_more=fetch_more)

        # Map outcome to status/metrics
        status = res.status
        proof_len = 0
        score = cmax = margin = 0.0
        alt_used = False
        stop_reason = getattr(res, "reason", "")

        if status == "ACCEPT" and res.proof is not None:
            p = res.proof
            proof_len = len(p.leaves or [])
            score = _safe_float(p.score, 0.0)
            cmax = _safe_float(p.cmax, 0.0)
            margin = _safe_float(p.margin, 0.0)
            alt_used = bool(getattr(p, "alternation_used", False))
            stop_reason = getattr(p, "stop_reason", stop_reason)

            if margin < delta:
                safety_viol += 1

            if proof_len == 1:
                hop1 += 1
            elif proof_len == 2:
                hop2 += 1

            if label == "true":
                accept_T += 1; TP += 1
            else:
                accept_F += 1; FP += 1
        elif status == "REJECT":
            reject += 1
            if label == "false":
                TN += 1
            elif label == "true":
                FN += 1
        else:
            status = "ABSTAIN"
            abstain += 1
            # Treat ABSTAIN on true as FN (miss); ABSTAIN on false is neither TN nor FP
            if label == "true":
                FN += 1

        out_rows.append({
            "qid": qid,
            "status": status,
            "score": f"{score:.6f}",
            "cmax": f"{cmax:.6f}",
            "margin": f"{margin:.6f}",
            "proof_len": proof_len,
            "alternation_used": int(alt_used),
            "stop_reason": stop_reason,
        })

    # 4) Aggregate metrics
    accept = accept_T + accept_F
    prec = (accept_T / accept) if accept else 0.0
    reject_rate = reject / n if n else 0.0
    abstain_rate = abstain / n if n else 0.0

    summary = {
        "N": n,
        "accept": accept,
        "accept_true": accept_T,
        "accept_false": accept_F,
        "precision_at_ACCEPT": round(prec, 3),
        "reject_rate": round(reject_rate, 3),
        "abstain_rate": round(abstain_rate, 3),
        "safety_violations": int(safety_viol),
        "proofs": {"1hop": hop1, "2hop": hop2},
        "delta": delta,
        "confusion": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
    }

    # 5) Write per-row output and print + dump summary
    _write_csv(out_path, out_rows, fieldnames=[
        "qid","status","score","cmax","margin","proof_len","alternation_used","stop_reason"
    ])

    print(json.dumps(summary, indent=2))
    print(
        f"METRICS p@ACCEPT={prec:.3f} reject={reject_rate:.3f} abstain={abstain_rate:.3f} "
        f"viol={safety_viol} proofs1={hop1} proofs2={hop2} "
        f"TP={TP} FP={FP} TN={TN} FN={FN}"
    )

    # Dump JSON summary for CI / dashboards
    summary_path = Path("artifacts/logs/eval.summary.json")
    _ensure_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
