# scripts/run_e2e_retrieval.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, cast

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Silence HF hub's deprecated message about local_dir_use_symlinks (harmless noise)
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
os.environ.setdefault("TQDM_DISABLE", "1")

# UTF-8-safe console on Windows; keep mypy happy by casting to Any
try:
    _stdout = cast(Any, sys.stdout)
    _stderr = cast(Any, sys.stderr)
    if hasattr(_stdout, "reconfigure"):
        _stdout.reconfigure(encoding="utf-8")
    if hasattr(_stderr, "reconfigure"):
        _stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Reduce transformers logging noise (after env is set, safe to import)
try:
    from transformers import logging as tflog
    tflog.set_verbosity_error()
except Exception:
    pass

from sro.config import apply_env_overrides, apply_profile, load_config, validate_config
from sro.rerank.cross_encoder import CrossEncoderReranker
from sro.retrieval.hybrid import get_initial_candidates
from sro.utils.random import set_all_seeds


def ensure_demo_corpus(corpus_path: Path) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    if corpus_path.exists():
        return
    docs = [
        {
            "source_id": "news:1",
            "text": (
                "Apple announced the iPhone 15 lineup in 2023. The iPhone 15 Pro features a titanium frame. "
                "Preorders began in September, with shipments later that month."
            ),
        },
        {
            "source_id": "press:1",
            "text": "In September 2023, Apple released the iPhone 15 series. The Pro models introduced a titanium frame for durability.",
        },
        {
            "source_id": "blog:1",
            "text": "Rumors suggested the Pro might include titanium. However, confirmation came at the September event.",
        },
    ]
    with corpus_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _brief_text(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def main() -> None:
    ap = argparse.ArgumentParser("E2E retrieval demo")
    ap.add_argument("--query", type=str, default="Does the iPhone 15 Pro have a titanium frame?")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--profile",
        type=str,
        choices=["low", "med", "high"],
        default=None,
        help="Preset for (M,L,B) — affects downstream prover; here we just show config.",
    )
    ap.add_argument(
        "--use_ce",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="Force cross-encoder reranker: auto=cfg, on=force, off=disable",
    )
    # NEW (P5 flags)
    ap.add_argument("--use_faiss", action="store_true", help="Enable FAISS dense path if artifacts exist.")
    ap.add_argument("--faiss_dir", type=str, default="artifacts/faiss", help="Directory with FAISS artifacts.")
    ap.add_argument(
        "--retrieval_mix",
        type=str,
        default="default",
        choices=["default", "splade"],
        help="Reserved for SPLADE toggle (disabled; raises NotImplementedError).",
    )
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
        use_ce = use_ce_cfg
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
        # NEW passthroughs
        use_faiss=bool(args.use_faiss),
        faiss_dir=args.faiss_dir,
        retrieval_mix=args.retrieval_mix,
    )

    # Summary
    print(
        "RETRIEVAL "
        f"k_bm25={cfg.retrieval.k_bm25} "
        f"k_dense={cfg.retrieval.k_dense} "
        f"k_fused={cfg.retrieval.k_fused} "
        f"rrf_c={cfg.retrieval.rrf_c} "
        f"mmr_lambda={cfg.retrieval.mmr_lambda} "
        f"ce={'on' if reranker is not None else 'off'} "
        f"faiss={'on' if args.use_faiss else 'off'}"
    )

    # Dump top results
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
