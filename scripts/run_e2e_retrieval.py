# scripts/run_e2e_retrieval.py
from __future__ import annotations
from pathlib import Path
import json
from sro.config import load_config
from sro.types import Claim
from sro.prover import SROProver
from sro.rerank.cross_encoder import CrossEncoderReranker
from sro.retrieval.hybrid import get_initial_candidates, make_fetch_more

CORPUS = Path("data/corpus/corpus.jsonl")

def ensure_demo_corpus():
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    if CORPUS.exists():
        return
    docs = [
        {
            "source_id": "news:1",
            "text": "Apple announced the iPhone 15 lineup in 2023. The iPhone 15 Pro features a titanium frame. "
                    "Preorders began in September, with shipments later that month."
        },
        {
            "source_id": "press:1",
            "text": "In September 2023, Apple released the iPhone 15 series. The Pro models introduced a titanium frame for durability."
        },
        {
            "source_id": "blog:1",
            "text": "Rumors suggested the Pro might include titanium. However, confirmation came at the September event."
        }
    ]
    with CORPUS.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def main():
    ensure_demo_corpus()
    cfg = load_config()

    # Build (optional) cross-encoder
    reranker = None
    if cfg.retrieval.use_cross_encoder:
        try:
            reranker = CrossEncoderReranker()
        except Exception:
            reranker = None

    # First-pass retrieval (no hand seeds)
    claim = Claim(claim_id="c_ti", text="The iPhone 15 Pro has a titanium frame.")

    init = get_initial_candidates(
        str(cfg.retrieval.corpus_jsonl),
        claim.text,
        k_bm25=cfg.retrieval.k_bm25,
        k_dense=cfg.retrieval.k_dense,
        k_fused=cfg.retrieval.k_fused,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        rrf_c=cfg.retrieval.rrf_c,
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        rerank_top=cfg.retrieval.rerank_top,
    )

    # Alternation-backed fetch_more
    fetch_more = make_fetch_more(
        str(cfg.retrieval.corpus_jsonl),
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        k_fused=cfg.retrieval.k_fused,
    )

    prover = SROProver(cfg, use_real_nli=True, batch_size=16)
    result = prover.prove(claim, init, fetch_more=fetch_more)

    if result.status == "ACCEPT":
        p = result.proof
        print("ACCEPT (alternation_used =", p.alternation_used, ")")
        print("  leaves:", p.leaves)
        print("  score:", round(p.score, 3), "cmax:", round(p.cmax, 3), "margin:", round(p.margin, 3))
        for e in p.edges:
            print("  edge:", e.src, "→", e.dst, e.label, "pE=", round(e.p_entail, 3), "pC=", round(e.p_contradict, 3))
    else:
        print("REJECT:", result.reason)

if __name__ == "__main__":
    main()
