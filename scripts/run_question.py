# scripts/run_question.py
from __future__ import annotations
from pathlib import Path
import json
from sro.config import load_config
from sro.claims.splitter import draft_and_claims, pick_top_sentences
from sro.prover import SROProver
from sro.types import SentenceCandidate
from sro.retrieval.hybrid import get_initial_candidates, make_fetch_more
from sro.rerank.cross_encoder import CrossEncoderReranker
from sro.compose.answer import compose_answer_with_citations


CORPUS = Path("data/corpus/corpus.jsonl")

def _norm_cite(s) -> dict:
    """
    Normalize a citation item to {'sent_id': str, 'source_id': str|None}.
    Accepts either a dict or an object with attributes.
    """
    if isinstance(s, dict):
        return {"sent_id": s.get("sent_id"), "source_id": s.get("source_id")}
    # Fallback for object-like items
    return {"sent_id": getattr(s, "sent_id", None), "source_id": getattr(s, "source_id", None)}


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

def compose_answer(accepted_claims: list[dict]) -> str:
    """
    Very simple composer: join proved claim texts.
    Each dict has: {'claim_id','text','citations':[{'sent_id','source_id'}]}
    """
    if not accepted_claims:
        return ""
    return " ".join(item["text"] for item in accepted_claims)

def main():
    ensure_demo_corpus()
    cfg = load_config()

    # Optional cross-encoder for retrieval; safe fallback to None
    reranker = None
    if cfg.retrieval.use_cross_encoder:
        try:
            reranker = CrossEncoderReranker()
        except Exception:
            reranker = None

    # 1) Question
    question = "Does the iPhone 15 Pro have a titanium frame?"

    # 2) First-pass retrieval for the question
    init = get_initial_candidates(
        str(cfg.retrieval.corpus_jsonl),
        question,
        k_bm25=cfg.retrieval.k_bm25,
        k_dense=cfg.retrieval.k_dense,
        k_fused=cfg.retrieval.k_fused,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        rrf_c=cfg.retrieval.rrf_c,
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        rerank_top=cfg.retrieval.rerank_top,
    )

    # 3) Draft and claims (no LLM yet): turn top sentences into small claims
    draft, claims = draft_and_claims(
        question,
        init,
        K=cfg.claims.K,
        min_question_cosine=cfg.claims.min_question_cosine,
        hedge_terms=cfg.claims.hedge_terms,
        reliability_weights=cfg.claims.reliability_weights,
    )

    # 4) Prover setup (with alternation-backed fetch_more)
    fetch_more = make_fetch_more(
        str(cfg.retrieval.corpus_jsonl),
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        k_fused=cfg.retrieval.k_fused,
    )
    prover = SROProver(cfg, use_real_nli=True, batch_size=16)

    # 5) Prove each claim independently
    accepted = []
    for cl in claims:
        # For each claim, we re-retrieve initial candidates dedicated to the claim text
        init_claim = get_initial_candidates(
            str(cfg.retrieval.corpus_jsonl),
            cl.text,
            k_bm25=cfg.retrieval.k_bm25,
            k_dense=cfg.retrieval.k_dense,
            k_fused=cfg.retrieval.k_fused,
            mmr_lambda=cfg.retrieval.mmr_lambda,
            rrf_c=cfg.retrieval.rrf_c,
            use_cross_encoder=bool(reranker is not None),
            cross_encoder=reranker,
            rerank_top=cfg.retrieval.rerank_top,
        )
        result = prover.prove(cl, init_claim, fetch_more=fetch_more)
        if result.status == "ACCEPT" and result.proof is not None:
            # collect a JSON-friendly claim record
            accepted.append({
                "claim_id": cl.claim_id,
                "text": cl.text,
                "citations": [_norm_cite(s) for s in (result.proof.citations or [])],
                "score": result.proof.score,
                "cmax": result.proof.cmax,
                "margin": result.proof.margin,
            })

    # 6) Compose final answer with inline citations
    if not accepted:
        print("ABSTAIN: no claim proved safely.")
        return

    final, refs = compose_answer_with_citations(
        accepted,
        N=2,               # keep it punchy
        theta_dup=0.90,    # dup threshold
        psi_concat=0.40,   # join threshold
        enforce_source_diversity=True, 
    )
    print("FINAL ANSWER:", final)
    print("CLAIMS ACCEPTED:", len(accepted))
    for a in accepted:
        print(f"  - {a['claim_id']}: score={a['score']:.3f} margin={a['margin']:.3f} cites={a['citations']}")
    if refs:
        print("REFERENCES:")
        for m, s in refs:
            print(f"  {m} {s}")
if __name__ == "__main__":
    main()
