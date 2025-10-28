# scripts/demo_retrieval_fetch.py
from __future__ import annotations

import json
from pathlib import Path

from sro.config import load_config
from sro.prover import SROProver
from sro.rerank.cross_encoder import CrossEncoderReranker
from sro.retrieval.hybrid import make_fetch_more
from sro.types import Claim, SentenceCandidate

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

    # Build fetch_more backed by hybrid retriever (+ optional cross-encoder)
    try:
        reranker = CrossEncoderReranker()  # uses GPU if available
        fetch_more = make_fetch_more(str(CORPUS), use_cross_encoder=True, cross_encoder=reranker, k_fused=24)
    except Exception:
        # Fallback without cross-encoder
        fetch_more = make_fetch_more(str(CORPUS), use_cross_encoder=False, cross_encoder=None, k_fused=24)

    # Claim that requires evidence; start with an intentionally weak initial pool
    claim = Claim(claim_id="c_ti", text="The iPhone 15 Pro has a titanium frame.")

    initial_cands = [
        SentenceCandidate("sA", "Apple released the iPhone 15 lineup in 2023.", "seed:news", 0.60),
        SentenceCandidate("sB", "Preorders opened in mid-September.", "seed:news", 0.40),
        SentenceCandidate("sC", "Shipments started later in September 2023.", "seed:press", 0.45),
    ]

    prover = SROProver(cfg, use_real_nli=True, batch_size=16)
    result = prover.prove(claim, initial_cands, fetch_more=fetch_more)

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
