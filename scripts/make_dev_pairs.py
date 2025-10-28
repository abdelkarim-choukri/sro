# scripts/make_dev_pairs.py
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# bootstrap: offline + utf-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HF_CACHE = str(Path("models_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE)
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sro.config import load_config
from sro.nli.nli_infer import two_hop_scores
from sro.prover.s1_onehop import one_hop_scores as s1_one_hop
from sro.prover.s2_frontier import _tokens as s2_tokens
from sro.prover.s2_frontier import select_frontier_and_pool
from sro.prover.s3_features import build_pair_features
from sro.retrieval.hybrid import get_initial_candidates
from sro.types import SentenceCandidate
from sro.utils.random import set_all_seeds


def _ensure_demo_corpus(corpus_path: Path) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    if corpus_path.exists():
        return
    docs = [
        {"source_id": "news:1",
         "text": "Apple announced the iPhone 15 lineup in 2023. The iPhone 15 Pro features a titanium frame. "
                 "Preorders began in September, with shipments later that month."},
        {"source_id": "press:1",
         "text": "In September 2023, Apple released the iPhone 15 series. The Pro models introduced a titanium frame for durability."},
        {"source_id": "blog:1",
         "text": "Rumors suggested the Pro might include titanium. However, confirmation came at the September event."}
    ]
    with corpus_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def _read_dev_claims(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd]

def _bruteforce_pairs(idx: list[int], cap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    n = len(idx)
    if n < 2:
        return out
    for a in range(n):
        for b in range(a + 1, n):
            out.append((idx[a], idx[b]))
            if len(out) >= cap:
                return out
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims_csv", type=str, default="data/processed/dev_claims.csv",
                    help="CSV with columns: qid, question, claim, label, evidence_ids")
    ap.add_argument("--output", type=str, default="data/processed/dev_pairs.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs1", type=int, default=32, help="batch size for S1 NLI")
    ap.add_argument("--bs2", type=int, default=32, help="batch size for 2-hop NLI")
    ap.add_argument("--max_pairs_per_claim", type=int, default=128)
    ap.add_argument("--min_pairs", type=int, default=1, help="skip claims that generate < min_pairs")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    cfg = load_config()

    corpus_path = Path(str(cfg.retrieval.corpus_jsonl))
    _ensure_demo_corpus(corpus_path)

    rows = _read_dev_claims(Path(args.claims_csv))
    if not rows:
        print(f"NO_CLAIMS: {args.claims_csv} is empty or missing headers", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "qid", "claim",
        "i", "j", "sent_id_i", "sent_id_j",
        "p1_i", "p1_j", "p2",
        "max_p1", "entity_overlap", "time_agreement", "distance", "novelty", "ce_max",
        "negation_conflict", "source_diversity",
    ]
    total_rows = 0
    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        wr = csv.DictWriter(f_out, fieldnames=fieldnames)
        wr.writeheader()

        for r in rows:
            qid = r.get("qid") or ""
            claim_text = r.get("claim") or r.get("question") or ""
            if not claim_text:
                continue

            # Retrieval
            cands: list[SentenceCandidate] = get_initial_candidates(
                str(corpus_path), claim_text,
                k_bm25=cfg.retrieval.k_bm25,
                k_dense=cfg.retrieval.k_dense,
                k_fused=cfg.retrieval.k_fused,
                mmr_lambda=cfg.retrieval.mmr_lambda,
                rrf_c=cfg.retrieval.rrf_c,
                use_cross_encoder=False,
                cross_encoder=None,
                rerank_top=0,
            )
            if not cands:
                continue

            # S1
            p1, c1 = s1_one_hop(claim_text, cands, use_model=True, batch_size=args.bs1)

            # S2+S3 (first attempt)
            frontier_idx, pool2_idx, token_cache = select_frontier_and_pool(
                cands, p1, M=int(cfg.sro_prover.M), L=int(cfg.sro_prover.L), lambda_diversity=0.7
            )
            claim_tokens = list(s2_tokens(claim_text))
            pairs, feats = build_pair_features(
                claim_tokens, cands, token_cache, frontier_idx, pool2_idx, p1
            )

            # Minimality filter
            tau1 = float(cfg.sro_prover.tau1)
            keep: list[int] = [
                k for k, (i, j) in enumerate(pairs)
                if (i < len(p1) and j < len(p1)) and (p1[i] < tau1 and p1[j] < tau1)
            ]

            # Fallback: if no pairs survived, brute-force among all below-τ1 indices
            if not keep:
                below = [i for i in range(len(cands)) if p1[i] < tau1]
                bf_pairs = _bruteforce_pairs(below, args.max_pairs_per_claim)
                if bf_pairs:
                    # Recompute features using the same token cache and claim tokens
                    # by treating 'frontier' and 'pool2' as the below-τ1 set
                    frontier_idx2 = below
                    pool2_idx2 = below
                    pairs2, feats2 = build_pair_features(
                        claim_tokens, cands, token_cache, frontier_idx2, pool2_idx2, p1
                    )
                    # Map available features for the brute-forced pairs
                    # (build_pair_features will typically include these combos; if not, fallback to a minimal feat dict)
                    feat_map: dict[tuple[int, int], dict[str, float]] = {}
                    for (a, b), F in zip(pairs2, feats2):
                        feat_map[(a, b)] = F
                        feat_map[(b, a)] = F
                    # Replace pairs/feats with the brute-forced subset that we have features for
                    pairs = []
                    feats = []
                    for (i, j) in bf_pairs:
                        F = feat_map.get((i, j))
                        if F is None:
                            # minimal defensible defaults if S3 didn’t emit this combo
                            F = {
                                "max_p1": float(max(p1[i], p1[j])),
                                "entity_overlap": 0.0,
                                "time_agreement": 0.0,
                                "distance": 1.0,
                                "novelty": 0.0,
                                "ce_max": 0.0,
                                "negation_conflict": 0.0,
                                "source_diversity": 0.0,
                            }
                        pairs.append((i, j))
                        feats.append(F)
                    # now all candidates are minimality-valid by construction
                    keep = list(range(len(pairs)))

            # Cap for speed and apply keep
            if not keep or len(keep) < args.min_pairs:
                continue
            keep = keep[: args.max_pairs_per_claim]
            pairs_kept = [pairs[k] for k in keep]
            feats_kept = [feats[k] for k in keep]

            # p2 via two-hop MNLI
            text_pairs: list[tuple[str, str]] = [(cands[i].text, cands[j].text) for (i, j) in pairs_kept]
            p2 = two_hop_scores(claim_text, text_pairs, batch_size=args.bs2)

            # write rows
            for (i, j), F, p2v in zip(pairs_kept, feats_kept, p2):
                wr.writerow({
                    "qid": qid,
                    "claim": claim_text,
                    "i": i, "j": j,
                    "sent_id_i": cands[i].sent_id,
                    "sent_id_j": cands[j].sent_id,
                    "p1_i": f"{p1[i]:.6f}",
                    "p1_j": f"{p1[j]:.6f}",
                    "p2": f"{float(p2v):.6f}",
                    "max_p1": f"{float(F.get('max_p1', 0.0)):.6f}",
                    "entity_overlap": f"{float(F.get('entity_overlap', 0.0)):.6f}",
                    "time_agreement": f"{float(F.get('time_agreement', 0.0)):.6f}",
                    "distance": f"{float(F.get('distance', 0.0)):.6f}",
                    "novelty": f"{float(F.get('novelty', 0.0)):.6f}",
                    "ce_max": f"{float(F.get('ce_max', 0.0)):.6f}",
                    "negation_conflict": f"{float(F.get('negation_conflict', 0.0)):.6f}",
                    "source_diversity": f"{float(F.get('source_diversity', 0.0)):.6f}",
                })
                total_rows += 1

    print(f"Wrote {out_path} ({total_rows} rows)")

if __name__ == "__main__":
    main()
