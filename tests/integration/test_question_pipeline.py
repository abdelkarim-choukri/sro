from __future__ import annotations
from pathlib import Path
import json

from sro.config import load_config
from sro.claims.splitter import draft_and_claims
from sro.retrieval.hybrid import get_initial_candidates, make_fetch_more
from sro.prover import SROProver
from sro.rerank.cross_encoder import CrossEncoderReranker
from sro.types import Claim

CORPUS = Path("data/corpus/corpus.jsonl")

def _ensure_demo():
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    if CORPUS.exists():
        return
    docs = [
        {"source_id":"news:1","text":"Apple announced the iPhone 15 lineup in 2023. The iPhone 15 Pro features a titanium frame."},
        {"source_id":"press:1","text":"In September 2023, Apple released the iPhone 15 series. The Pro models introduced a titanium frame for durability."}
    ]
    with CORPUS.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

def test_question_end_to_end():
    _ensure_demo()
    cfg = load_config()
    # keep test light: no CE
    reranker = None

    q = "Does the iPhone 15 Pro have a titanium frame?"
    init = get_initial_candidates(
        str(cfg.retrieval.corpus_jsonl), q,
        k_bm25=cfg.retrieval.k_bm25, k_dense=cfg.retrieval.k_dense, k_fused=cfg.retrieval.k_fused,
        mmr_lambda=cfg.retrieval.mmr_lambda, rrf_c=cfg.retrieval.rrf_c,
        use_cross_encoder=False, cross_encoder=None, rerank_top=cfg.retrieval.rerank_top
    )
    draft, claims = draft_and_claims(q, init, max_claims=2)
    assert len(claims) >= 1

    prover = SROProver(cfg, use_real_nli=True, batch_size=8)
    fetch_more = make_fetch_more(str(cfg.retrieval.corpus_jsonl), use_cross_encoder=False, cross_encoder=None, k_fused=cfg.retrieval.k_fused)

    accepted = 0
    for cl in claims:
        init_claim = get_initial_candidates(
            str(cfg.retrieval.corpus_jsonl), cl.text,
            k_bm25=cfg.retrieval.k_bm25, k_dense=cfg.retrieval.k_dense, k_fused=cfg.retrieval.k_fused,
            mmr_lambda=cfg.retrieval.mmr_lambda, rrf_c=cfg.retrieval.rrf_c,
            use_cross_encoder=False, cross_encoder=None, rerank_top=cfg.retrieval.rerank_top
        )
        r = prover.prove(cl, init_claim, fetch_more=fetch_more)
        if r.status == "ACCEPT":
            accepted += 1

    assert accepted >= 1
