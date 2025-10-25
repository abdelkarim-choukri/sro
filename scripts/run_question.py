# scripts/run_question.py
from __future__ import annotations
# === BOOTSTRAP: env + console + warning filters (must be before any HF/ST imports) ===
import json
import os, sys, warnings
from pathlib import Path

# Silence HF hub's deprecated message about local_dir_use_symlinks (it’s harmless noise)
warnings.filterwarnings(
    "ignore",
    message="`local_dir_use_symlinks` parameter is deprecated",
    category=UserWarning,
    module="huggingface_hub.file_download",
)

# Force offline/cache-only behavior and quiet progress bars
HF_CACHE = str(Path("models_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE)
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")  # belt-and-suspenders for progress bars

# UTF-8-safe console on Windows (avoid UnicodeEncodeError: 'gbk')
# On Windows, printing e.g. “Résumé ✨” 
# can raise UnicodeEncodeError: 'gbk' codec can't encode character .... 
# Forcing UTF-8 prevents that.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Reduce transformers logging noise (after env is set, safe to import)
try:
    from transformers import logging as tflog
    tflog.set_verbosity_error()
except Exception:
    pass
# === END BOOTSTRAP ===


from sro.config import load_config, apply_profile, apply_env_overrides, validate_config
import argparse
from sro.claims.splitter import draft_and_claims
from sro.prover import SROProver
from sro.retrieval.hybrid import get_initial_candidates, make_fetch_more
from sro.rerank.cross_encoder import CrossEncoderReranker
from sro.compose.answer import compose_answer_with_citations
from sro.utils.random import set_all_seeds


def _norm_cite(s) -> dict:
    if isinstance(s, dict):
        return {"sent_id": s.get("sent_id"), "source_id": s.get("source_id")}
    return {"sent_id": getattr(s, "sent_id", None), "source_id": getattr(s, "source_id", None)}

def ensure_demo_corpus(corpus_path: Path) -> None:
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

def main():

    import logging, os
    logging.basicConfig(
        level=os.environ.get("SRO_LOGLEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, default="Does the iPhone 15 Pro have a titanium frame?")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs_nli1", type=int, default=32, help="Batch size for 1-hop NLI (S1).")
    ap.add_argument("--bs_nli2", type=int, default=32, help="Batch size for 2-hop NLI (pair scoring).")
    ap.add_argument("--profile", type=str, choices=["low", "med", "high"], default=None,
                    help="Preset for (M,L,B)")
    args = ap.parse_args()

    # determinism
    set_all_seeds(args.seed)

    # config
    cfg = load_config()
    if args.profile: 
        apply_profile(cfg, args.profile)
    apply_env_overrides(cfg) 
    validate_config(cfg)
    
    # show effective knobs
    sp = cfg.sro_prover
    print(
        "CONFIG "
        f"M={sp.M} L={sp.L} B={sp.B} "
        f"tau1={sp.tau1} tau2={sp.tau2} delta={sp.delta} "
        f"kappa={sp.kappa} eps={sp.epsilon}"
    )

    # corpus
    corpus_path = Path(str(cfg.retrieval.corpus_jsonl))
    ensure_demo_corpus(corpus_path)

    # reranker (allow disable via env)
    reranker = None
    use_ce = bool(getattr(cfg.retrieval, "use_cross_encoder", False))
    if os.getenv("SRO_DISABLE_RERANK") == "1":
        use_ce = False
    if use_ce:
        try:
            reranker = CrossEncoderReranker()
        except Exception:
            reranker = None

    # initial retrieval
    init = get_initial_candidates(
        str(corpus_path),
        args.question,
        k_bm25=cfg.retrieval.k_bm25,
        k_dense=cfg.retrieval.k_dense,
        k_fused=cfg.retrieval.k_fused,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        rrf_c=cfg.retrieval.rrf_c,
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        rerank_top=cfg.retrieval.rerank_top,
    )

    # draft & claims
    draft, claims = draft_and_claims(
        args.question, init,
        K=cfg.claims.K,
        min_question_cosine=cfg.claims.min_question_cosine,#A minimum cosine similarity (using sentence embeddings) between claim and question. 
                                                             # If a generated claim is too far from the question, it’s dropped.

        hedge_terms=cfg.claims.hedge_terms,#Words like "might", "reportedly", "could". If a sentence contains these hedging terms, 
                                            #we tone it down (or drop it), so claims are crisp.
        reliability_weights=cfg.claims.reliability_weights,#Source trust weights. Example: { "press": 1.0, "news": 0.9, "blog": 0.3 }. 
                                                            #This boosts/penalizes claims depending on where the sentence came from.
    )

    # alternation fetcher + prover
    fetch_more = make_fetch_more(
        str(corpus_path),
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        k_fused=cfg.retrieval.k_fused,
    )
    prover = SROProver(cfg, use_real_nli=True, bs_nli1=args.bs_nli1, bs_nli2=args.bs_nli2)
 
    accepted = []
    accepted_proofs = []
    for cl in claims:
        init_claim = get_initial_candidates(
            str(corpus_path), cl.text,
            k_bm25=cfg.retrieval.k_bm25,
            k_dense=cfg.retrieval.k_dense,
            k_fused=cfg.retrieval.k_fused,
            mmr_lambda=cfg.retrieval.mmr_lambda,
            rrf_c=cfg.retrieval.rrf_c,
            use_cross_encoder=bool(reranker is not None),
            cross_encoder=reranker,
            rerank_top=cfg.retrieval.rerank_top,
        )
        res = prover.prove(cl, init_claim, fetch_more=fetch_more)
        if res.status == "ACCEPT" and res.proof is not None:
            accepted.append({
                "claim_id": cl.claim_id,
                "text": cl.text,
                "citations": [_norm_cite(s) for s in (res.proof.citations or [])],
                "score": res.proof.score,
                "cmax": res.proof.cmax,
                "margin": res.proof.margin,
            })
            accepted_proofs.append(res.proof)

    if not accepted:
        print("ABSTAIN: no claim proved safely.")
        print(f"RUNTIME evals=0 top_ub=0.0000 bs1={args.bs_nli1} bs2={args.bs_nli2}")
        return

    final, refs = compose_answer_with_citations(
        accepted, N=2, theta_dup=0.90, psi_concat=0.40, enforce_source_diversity=True
    )
    print("FINAL ANSWER:", final)
    print("CLAIMS ACCEPTED:", len(accepted))
    for a in accepted:
        print(f"  - {a['claim_id']}: score={a['score']:.3f} margin={a['margin']:.3f} cites={a['citations']}")
    if refs:
        print("REFERENCES:")
        for m, s in refs:
            print(f"  {m} {s}")

    for p in accepted_proofs:
        t = getattr(p, "timings", {}) or {}
        print(
            "RUNTIME "
            f"S1={t.get('S1_onehop', 0.0):.3f}s "
            f"S2={t.get('S2_frontier', 0.0):.3f}s "
            f"S3={t.get('S3_features', 0.0):.3f}s "
            f"S4={t.get('S4_ub', 0.0):.3f}s "
            f"S5={t.get('S5_search', 0.0):.3f}s "
            f"evals={getattr(p, 'budget_used', 0)} "
            f"top_ub={getattr(p, 'ub_top_remaining', 0.0):.4f} "
            f"bs1={args.bs_nli1} bs2={args.bs_nli2}"
        )

if __name__ == "__main__":
    main()
