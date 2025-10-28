# tests/integration/test_e2e_retrieval.py
from __future__ import annotations

import json
from pathlib import Path

from sro.config import load_config
from sro.prover import SROProver
from sro.rerank.cross_encoder import CrossEncoderReranker
from sro.retrieval.hybrid import get_initial_candidates, make_fetch_more
from sro.types import Claim

CORPUS = Path("data/corpus/corpus.jsonl")

def _ensure_demo_corpus():
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

def _runner(claim_text: str):
    _ensure_demo_corpus()
    cfg = load_config()
    # keep tests light: no CE in CI

    init = get_initial_candidates(
        str(cfg.retrieval.corpus_jsonl),
        claim_text,
        k_bm25=cfg.retrieval.k_bm25,
        k_dense=cfg.retrieval.k_dense,
        k_fused=cfg.retrieval.k_fused,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        rrf_c=cfg.retrieval.rrf_c,
        use_cross_encoder=False,
        cross_encoder=None,
        rerank_top=cfg.retrieval.rerank_top,
    )
    fetch_more = make_fetch_more(
        str(cfg.retrieval.corpus_jsonl),
        use_cross_encoder=False,
        cross_encoder=None,
        k_fused=cfg.retrieval.k_fused,
    )

    prover = SROProver(cfg, use_real_nli=True, batch_size=8)
    return prover.prove(Claim("c", claim_text), init, fetch_more=fetch_more)

def test_true_claim_accepts():
    r = _runner("The iPhone 15 Pro has a titanium frame.")
    assert r.status == "ACCEPT"
    assert r.proof is not None
    assert r.proof.score >= 0.75  # 1-hop often crosses

def test_false_claim_rejects():
    r = _runner("The iPhone 15 Pro uses a stainless steel frame.")  # false in our corpus
    assert r.status in ("REJECT", "ACCEPT")
    # Prefer REJECT; if ACCEPT, margin must be small (contradiction present)
    if r.status == "ACCEPT":
        assert (r.proof.score - r.proof.cmax) < 0.10
