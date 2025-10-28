# scripts/run_e2e_retrieval.py


from __future__ import annotations
# === BOOTSTRAP: env + console + warning filters (must be before any HF/ST imports) ===
import os, sys, warnings
from pathlib import Path
import json

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
from sro.utils.random import set_all_seeds
from sro.retrieval.hybrid import get_initial_candidates
from sro.rerank.cross_encoder import CrossEncoderReranker
import argparse
# UTF-8-safe console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Force offline/cache on every process
HF_CACHE = str(Path("models_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE)
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


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


def _brief_text(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default="Does the iPhone 15 Pro have a titanium frame?")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", type=str, choices=["low", "med", "high"], default=None,
                    help="Preset for (M,L,B) — affects downstream prover, here we just show config.")
    ap.add_argument("--use_ce", type=str, choices=["auto", "on", "off"], default="auto",
                    help="Force cross-encoder reranker: auto=cfg, on=force, off=disable")
    args = ap.parse_args()

    # Determinism first
    set_all_seeds(args.seed)

    # Config load + ergonomics
    cfg = load_config()
    if args.profile:
        apply_profile(cfg, args.profile)
    apply_env_overrides(cfg)
    validate_config(cfg)

    # Show effective knobs (useful for CI and debugging)
    sp = cfg.sro_prover
    print(
        "CONFIG "
        f"M={sp.M} L={sp.L} B={sp.B} "
        f"tau1={sp.tau1} tau2={sp.tau2} delta={sp.delta} "
        f"kappa={sp.kappa} eps={sp.epsilon}"
    )

    # Corpus
    corpus_path = Path(str(cfg.retrieval.corpus_jsonl))
    ensure_demo_corpus(corpus_path)

    # Reranker policy
    use_ce_cfg = bool(getattr(cfg.retrieval, "use_cross_encoder", False))
    if args.use_ce == "on":
        use_ce = True
    elif args.use_ce == "off":
        use_ce = False
    else:
        # auto
        use_ce = use_ce_cfg
    # env kill-switch still respected
    if os.getenv("SRO_DISABLE_RERANK") == "1":
        use_ce = False

    reranker = None
    if use_ce:
        try:
            reranker = CrossEncoderReranker()
        except Exception:
            reranker = None

    # Retrieval
    hits = get_initial_candidates(
        str(corpus_path),
        args.query,
        k_bm25=cfg.retrieval.k_bm25,
        k_dense=cfg.retrieval.k_dense,
        k_fused=cfg.retrieval.k_fused,
        mmr_lambda=cfg.retrieval.mmr_lambda,
        rrf_c=cfg.retrieval.rrf_c,
        use_cross_encoder=bool(reranker is not None),
        cross_encoder=reranker,
        rerank_top=cfg.retrieval.rerank_top,
    )

    # Summary
    print(
        "RETRIEVAL "
        f"k_bm25={cfg.retrieval.k_bm25} "
        f"k_dense={cfg.retrieval.k_dense} "
        f"k_fused={cfg.retrieval.k_fused} "
        f"rrf_c={cfg.retrieval.rrf_c} "
        f"mmr_lambda={cfg.retrieval.mmr_lambda} "
        f"ce={'on' if reranker is not None else 'off'}"
    )

    # Dump top results (id, source_id, ce_score, text snippet)
    if not hits:
        print("(no hits)")
        return
    print("TOP HITS:")
    for h in hits[: min(10, len(hits))]:
        sid = getattr(h, "sent_id", "")
        src = getattr(h, "source_id", "")
        ce = getattr(h, "ce_score", None)
        ce_s = f"{ce:.3f}" if isinstance(ce, (float, int)) else "-"
        print(f"  - {sid} [{src}] ce={ce_s} :: {_brief_text(getattr(h, 'text', ''))}")


if __name__ == "__main__":
    main()
