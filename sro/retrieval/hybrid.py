# # sro/retrieval/hybrid.py
# from __future__ import annotations
# from typing import List, Tuple, Optional, Sequence, Union, Dict, Any
# import os
# import logging
# import numpy as np

# from sro.embeddings.backend import EmbeddingBackend
# from sro.retrieval.redundancy import mmr_select_cosine

# LOGGER = logging.getLogger("sro.retrieval.hybrid")


# class SentenceCandidate:
#     """
#     Minimal carrier for retrieval + S2 that is also COMPATIBLE with older
#     sro.prover.* code that expects extra fields.

#     Fields used across code:
#       - sent_id (str): stable id
#       - text (str): sentence string
#       - score (float): generic relevance (we store TF-IDF normalized here)
#       - ce_score (float): cross-encoder score (legacy S2 sorts on this; default 0.0)
#       - bm25 (float|None): optional bm25
#       - dense (float|None): optional dense
#       - source_id (str): provenance/document/source id (legacy S3 uses this); default "corpus"
#       - rank (int): deterministic rank within the retrieval list (optional; default 0)
#     """
#     __slots__ = (
#         "sent_id",
#         "text",
#         "score",
#         "ce_score",
#         "bm25",
#         "dense",
#         "source_id",
#         "rank",
#     )

#     def __init__(
#         self,
#         sent_id: str,
#         text: str,
#         score: float,
#         *,
#         ce_score: Optional[float] = None,
#         bm25: Optional[float] = None,
#         dense: Optional[float] = None,
#         source_id: Optional[str] = None,
#         rank: int = 0,
#     ):
#         self.sent_id = sent_id
#         self.text = text
#         self.score = float(score)
#         self.ce_score = 0.0 if ce_score is None else float(ce_score)
#         self.bm25 = None if bm25 is None else float(bm25)
#         self.dense = None if dense is None else float(dense)
#         self.source_id = source_id if source_id is not None else "corpus"
#         self.rank = int(rank)

#     def __repr__(self) -> str:
#         return f"SentenceCandidate(id={self.sent_id!r}, score={self.score:.3f}, ce={self.ce_score:.3f}, src={self.source_id})"


# # ---------------------------
# # 1) Initial retrieval (offline TF-IDF)
# # ---------------------------

# def _load_default_corpus() -> List[Tuple[str, str]]:
#     """
#     Load fallback corpus from data/corpus/sentences.txt (one sentence per line).
#     Returns: list[(sent_id, text)]
#     """
#     default_path = os.path.join("data", "corpus", "sentences.txt")
#     if not os.path.isfile(default_path):
#         raise FileNotFoundError(
#             "No corpus provided to get_initial_candidates() and default corpus not found.\n"
#             f"Expected: {default_path}\n"
#             "Pass an explicit 'corpus' argument: list[str] or list[(id,str)]"
#         )
#     out: List[Tuple[str, str]] = []
#     with open(default_path, "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             s = line.strip()
#             if not s:
#                 continue
#             out.append((f"s{i}", s))
#     if not out:
#         raise ValueError(f"Default corpus file is empty: {default_path}")
#     return out


# def _normalize_corpus(
#     corpus: Optional[Sequence[Union[str, Tuple[str, str], SentenceCandidate]]]
# ) -> List[Tuple[str, str]]:
#     """
#     Accept:
#       - None -> load default
#       - list[str]
#       - list[(id, text)]
#       - list[SentenceCandidate]
#     Return list[(id, text)]
#     """
#     if corpus is None:
#         return _load_default_corpus()
#     if len(corpus) == 0:
#         return []
#     first = corpus[0]
#     if isinstance(first, SentenceCandidate):
#         return [(c.sent_id, c.text) for c in corpus]  # type: ignore
#     if isinstance(first, tuple) and len(first) == 2 and all(isinstance(x, str) for x in first):
#         return list(corpus)  # type: ignore
#     if isinstance(first, str):
#         return [(f"s{i}", t) for i, t in enumerate(corpus)]  # type: ignore
#     raise TypeError("Unsupported corpus element type. Use list[str], list[(id,str)], or list[SentenceCandidate].")


# def _infer_top_k(default_k: int, kwargs: Dict[str, Any]) -> int:
#     """
#     Compatible with callers that pass k_bm25/k_dense/k_rrf/k/top_k.
#     We choose the max of any provided k-like args, else default_k.
#     """
#     candidates = []
#     for key in ("k_rrf", "k_bm25", "k_dense", "top_k", "k"):
#         if key in kwargs and kwargs[key] is not None:
#             try:
#                 candidates.append(int(kwargs[key]))
#             except Exception:
#                 pass
#     k = max([default_k] + candidates) if candidates else default_k
#     return max(1, k)


# def get_initial_candidates(
#     query: str,
#     corpus: Optional[Sequence[Union[str, Tuple[str, str], SentenceCandidate]]] = None,  # allow 2nd positional
#     *,
#     top_k: Optional[int] = None,
#     seed: int = 42,
#     return_scores: bool = False,
#     **kwargs: Any,
# ) -> Union[List[SentenceCandidate], Tuple[List[SentenceCandidate], np.ndarray]]:
#     """
#     Offline initial retrieval using TF-IDF cosine similarity.

#     Returns:
#       candidates OR (candidates, p1_scores)
#          candidates: List[SentenceCandidate] (len <= top_k)
#          p1_scores : np.ndarray [len(candidates)] in [0,1]  (only if return_scores=True)
#     """
#     import random
#     random.seed(seed)
#     np.random.seed(seed)

#     k = _infer_top_k(default_k=24 if top_k is None else int(top_k), kwargs=kwargs)

#     items = _normalize_corpus(corpus)  # list[(id,text)]
#     if len(items) == 0:
#         return ([] if not return_scores else ([], np.zeros((0,), dtype=np.float32)))
#     ids, texts = zip(*items)

#     try:
#         from sklearn.feature_extraction.text import TfidfVectorizer
#         from sklearn.metrics.pairwise import linear_kernel
#     except Exception as e:
#         raise ImportError("scikit-learn is required for TF-IDF retrieval. `pip install scikit-learn`.") from e

#     vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
#     X = vec.fit_transform(list(texts) + [query])  # last row is query
#     X_docs = X[:-1]
#     X_q = X[-1]

#     sims = linear_kernel(X_docs, X_q)[:, 0]  # cosine
#     if sims.size == 0:
#         return ([] if not return_scores else ([], np.zeros((0,), dtype=np.float32)))

#     smin, smax = float(np.min(sims)), float(np.max(sims))
#     p1 = (sims - smin) / (smax - smin) if smax > smin else np.zeros_like(sims)

#     order = np.lexsort((np.arange(len(p1)), -p1))
#     top = order[: max(0, min(k, len(order)))]

#     cands: List[SentenceCandidate] = [
#     SentenceCandidate(
#         ids[i],
#         texts[i],
#         float(p1[i]),
#         ce_score=0.0,
#         source_id="corpus",
#         rank=ri,
#     )
#     for ri, i in enumerate(top)
# ]

#     p1_scores = np.asarray([p1[i] for i in top], dtype=np.float32)
#     return (cands, p1_scores) if return_scores else cands


# # ---------------------------
# # 2) S2 frontier (cosine default, jaccard fallback)
# # ---------------------------

# _EMBED_BACKEND_SINGLETON: Optional[EmbeddingBackend] = None
# def _get_embed_backend() -> EmbeddingBackend:
#     global _EMBED_BACKEND_SINGLETON
#     if _EMBED_BACKEND_SINGLETON is None:
#         _EMBED_BACKEND_SINGLETON = EmbeddingBackend()
#     return _EMBED_BACKEND_SINGLETON


# def select_frontier_and_pool(
#     candidates: List[SentenceCandidate],
#     p1_scores: np.ndarray,
#     M: int,
#     L: int,
#     *,
#     mmr_lambda: float = 0.5,
#     redundancy: str = "cosine",  # default cosine
#     embed_backend: Optional[EmbeddingBackend] = None,
# ) -> Tuple[List[int], List[int], np.ndarray]:
#     """
#     Select size-M frontier by MMR and size-L 2-hop pool.
#     """
#     N = len(candidates)
#     assert p1_scores.shape == (N,), f"p1_scores must be shape ({N},)"

#     if redundancy == "cosine":
#         backend = embed_backend or _get_embed_backend()
#         texts = [c.text for c in candidates]
#         ids = [getattr(c, "sent_id", None) for c in candidates]
#         out = mmr_select_cosine(
#             texts, ids, p1_scores, M,
#             mmr_lambda=mmr_lambda, sim_threshold=0.9, embed_backend=backend
#         )
#         frontier_idx: List[int] = out["selected_idx"]  # type: ignore
#         max_sim: np.ndarray = out["max_sim"]  # type: ignore
#         novelty: np.ndarray = out["novelty"]  # type: ignore

#         remaining = [i for i in range(N) if i not in frontier_idx]
#         pool_order = sorted(
#             remaining,
#             key=lambda i: (float(novelty[i]) * float(p1_scores[i]), float(p1_scores[i]), -i),
#             reverse=True,
#         )
#         pool_idx = pool_order[: max(0, min(L, len(pool_order)))]
#         return frontier_idx, pool_idx, max_sim

#     # ---------- Legacy Jaccard fallback ----------
#     def _token_set(s: str) -> set:
#         return set(t.lower() for t in s.split())

#     texts = [c.text for c in candidates]
#     sets = [_token_set(t) for t in texts]
#     max_j = np.zeros((N,), dtype=np.float32)
#     frontier_idx: List[int] = []
#     selected = np.zeros((N,), dtype=bool)

#     def _jacc(a: set, b: set) -> float:
#         if not a and not b:
#             return 0.0
#         inter = len(a & b)
#         union = len(a | b)
#         return float(inter) / float(union) if union > 0 else 0.0

#     for _ in range(min(M, N)):
#         best_i, best_mmr = -1, -1e9
#         for i in range(N):
#             if selected[i]:
#                 continue
#             mmr = mmr_lambda * float(p1_scores[i]) - (1.0 - mmr_lambda) * float(max_j[i])
#             if (mmr > best_mmr) or \
#                (mmr == best_mmr and p1_scores[i] > (p1_scores[best_i] if best_i >= 0 else -1e9)) or \
#                (mmr == best_mmr and p1_scores[i] == (p1_scores[best_i] if best_i >= 0 else -1e9) and i < best_i):
#                 best_i, best_mmr = i, mmr
#         if best_i < 0:
#             break
#         frontier_idx.append(best_i)
#         selected[best_i] = True
#         for i in range(N):
#             if selected[i]:
#                 continue
#             j = _jacc(sets[i], sets[best_i])
#             if j > max_j[i]:
#                 max_j[i] = j

#     novelty = 1.0 - max_j
#     remaining = [i for i in range(N) if i not in frontier_idx]
#     pool_order = sorted(
#         remaining,
#         key=lambda i: (float(novelty[i]) * float(p1_scores[i]), float(p1_scores[i]), -i),
#         reverse=True,
#     )
#     pool_idx = pool_order[: max(0, min(L, len(pool_order)))]
#     return frontier_idx, pool_idx, max_j
# sro/retrieval/hybrid.py
# sro/retrieval/hybrid.py
"""
Does:
    Retrieval helpers + frontier selection (S2) with cosine redundancy (default) and
    a legacy Jaccard fallback. Also provides minimal candidate fetchers used by scripts/tests.

Inputs:
    - Corpus path: either a newline TXT ("one sentence per line") or JSONL where each line is
      {"source_id": str, "text": str}. JSONL "text" is split to sentences heuristically.
    - p1_scores: numpy array of 1-hop scores aligned with `candidates`.

Outputs:
    - select_frontier_and_pool(...) -> (frontier_idx, pool_idx, max_sim_array)
    - get_initial_candidates(corpus_path, query, ...) -> List[SentenceCandidate]
    - make_fetch_more(corpus_path) -> closure(k, exclude_ids) -> List[SentenceCandidate]

Notes:
    - Pure logic, deterministic tie-breaks.
    - Embeddings are cached by EmbeddingBackend (LRU + sqlite).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from sro.embeddings.backend import EmbeddingBackend
from sro.retrieval.redundancy import mmr_select_cosine
from .faiss_index import search_faiss

# ---------- Public candidate type ----------

@dataclass
class SentenceCandidate:
    sent_id: str
    text: str
    score: float = 0.0
    source_id: str = "corpus:0"
    ce_score: float = 0.0  # cross-encoder score if available


# ---------- Embedding backend singleton ----------

_EMBED_BACKEND_SINGLETON: Optional[EmbeddingBackend] = None


def _get_embed_backend() -> EmbeddingBackend:
    global _EMBED_BACKEND_SINGLETON
    if _EMBED_BACKEND_SINGLETON is None:
        _EMBED_BACKEND_SINGLETON = EmbeddingBackend()
    return _EMBED_BACKEND_SINGLETON


# ---------- Corpus readers ----------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _read_corpus_lines(corpus_path: Path) -> list[tuple[str, str, str]]:
    """
    Returns list of tuples: (sent_id, source_id, text)
    Supports:
      - TXT: one sentence per line
      - JSONL: {"source_id": "...", "text": "..."} (split to sentences)
    """
    out: list[tuple[str, str, str]] = []
    if not corpus_path.exists():
        return out

    if corpus_path.suffix.lower() == ".jsonl":
        with corpus_path.open("r", encoding="utf-8") as f:
            for doc_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                src = str(obj.get("source_id") or f"doc:{doc_idx}")
                sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
                for si, s in enumerate(sents):
                    sid = f"{src}#s{si}"
                    out.append((sid, src, s))
    else:
        # default: TXT
        with corpus_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                sid = f"s{i}"
                out.append((sid, "corpus:0", s))
    return out


# ---------- Frontier selection (S2) ----------

def select_frontier_and_pool(
    candidates: List[SentenceCandidate],
    p1_scores: np.ndarray,
    M: int,
    L: int,
    *,
    mmr_lambda: float = 0.5,
    redundancy: str = "cosine",  # default cosine
    embed_backend: Optional[EmbeddingBackend] = None,
) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Select size-M frontier by MMR and size-L 2-hop pool.
    Default redundancy = cosine; 'jaccard' kept as fallback.

    Returns:
        frontier_idx: indices of selected frontier (len ≤ M)
        pool_idx: indices for 2-hop pool (len ≤ L, disjoint from frontier)
        max_sim: array[N] of max similarity to any selected frontier (cos or jacc)

    Invariants:
        - p1_scores.shape == (N,)
        - Deterministic tie-breaking: higher p1, then lower index.
    """
    N = len(candidates)
    assert p1_scores.shape == (N,), f"p1_scores must be shape ({N},)"

    if redundancy == "cosine":
        backend = embed_backend or _get_embed_backend()
        texts = [c.text for c in candidates]
        ids = [getattr(c, "sent_id", None) for c in candidates]
        out = mmr_select_cosine(
            texts, ids, p1_scores, M,
            mmr_lambda=mmr_lambda, sim_threshold=0.9, embed_backend=backend
        )
        frontier_idx: List[int] = out["selected_idx"]  # type: ignore
        max_sim: np.ndarray = out["max_sim"]  # type: ignore
        novelty: np.ndarray = out["novelty"]  # type: ignore

        remaining = [i for i in range(N) if i not in frontier_idx]
        pool_order = sorted(
            remaining,
            key=lambda i: (float(novelty[i]) * float(p1_scores[i]), float(p1_scores[i]), -i),
            reverse=True,
        )
        pool_idx = pool_order[: max(0, min(L, len(pool_order)))]
        return frontier_idx, pool_idx, max_sim

    # ---------- Legacy Jaccard fallback ----------
    def _token_set(s: str) -> set[str]:
        return set(t.lower() for t in s.split())

    texts = [c.text for c in candidates]
    sets = [_token_set(t) for t in texts]
    max_j = np.zeros((N,), dtype=np.float32)
    frontier_idx: List[int] = []
    selected = np.zeros((N,), dtype=bool)

    def _jacc(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter) / float(union) if union > 0 else 0.0

    for _ in range(min(M, N)):
        best_i, best_mmr = -1, -1e9
        for i in range(N):
            if selected[i]:
                continue
            mmr = mmr_lambda * float(p1_scores[i]) - (1.0 - mmr_lambda) * float(max_j[i])
            if (mmr > best_mmr) or \
               (mmr == best_mmr and p1_scores[i] > (p1_scores[best_i] if best_i >= 0 else -1e9)) or \
               (mmr == best_mmr and p1_scores[i] == (p1_scores[best_i] if best_i >= 0 else -1e9) and i < best_i):
                best_i, best_mmr = i, mmr
        if best_i < 0:
            break
        frontier_idx.append(best_i)
        selected[best_i] = True
        for i in range(N):
            if selected[i]:
                continue
            j = _jacc(sets[i], sets[best_i])
            if j > max_j[i]:
                max_j[i] = j

    novelty = 1.0 - max_j
    remaining = [i for i in range(N) if i not in frontier_idx]
    pool_order = sorted(
        remaining,
        key=lambda i: (float(novelty[i]) * float(p1_scores[i]), float(p1_scores[i]), -i),
        reverse=True,
    )
    pool_idx = pool_order[: max(0, min(L, len(pool_order)))]
    return frontier_idx, pool_idx, max_j


# ---------- Minimal retrieval helpers ----------

def get_initial_candidates(
    corpus_path: str,
    query_text: str,
    *,
    k_bm25: int,
    k_dense: int,
    k_fused: int,
    mmr_lambda: float,
    rrf_c: int,
    use_cross_encoder: bool,
    cross_encoder: Any,
    rerank_top: int,
    # NEW (P5):
    use_faiss: bool = False,
    faiss_dir: Optional[str] = None,
    retrieval_mix: str = "default",  # reserved for SPLADE toggle
) -> List[Any]:
    """
    Return fused initial candidates for retrieval.

    V2 extensions:
      • Optional FAISS dense path (with clean brute-force fallback).
      • SPLADE guard (disabled behind a flag).
      • Per-stage contribution logs:
            RETRIEVAL bm25_hits=XXX dense_hits=YYY fused_kept=ZZZ

    Deterministic as long as upstream RNG and the embedding backend are seeded.
    """
    import json
    import logging
    import os
    from pathlib import Path
    from types import SimpleNamespace

    logger = logging.getLogger(__name__)

    # --- Guard: SPLADE disabled by default ---
    if retrieval_mix.lower() == "splade":
        raise NotImplementedError(
            "SPLADE path is disabled by default. Enable only with local index and legal license."
        )

    # Try to import the project SentenceCandidate. If missing or ctor incompatible,
    # we'll create lightweight objects with the same attributes.
    try:
        from sro.types import SentenceCandidate  # type: ignore
        _HAS_SC = True
    except Exception:
        SentenceCandidate = None  # type: ignore
        _HAS_SC = False

    def _mk_candidate(sid: str, txt: str, src: str, ce_score: float = 0.0) -> Any:
        """
        Construct a candidate robustly:
          1) Try keyword ctor (sent_id/text/source_id).
          2) Try positional ctors.
          3) Fall back to a SimpleNamespace with needed attrs.
        IMPORTANT: we DO NOT pass 'score' because your dataclass does not accept it.
        """
        if _HAS_SC and SentenceCandidate is not None:
            try:
                return SentenceCandidate(sent_id=sid, text=txt, source_id=src, ce_score=ce_score)  # type: ignore
            except TypeError:
                try:
                    return SentenceCandidate(sid, txt, src)  # type: ignore
                except TypeError:
                    try:
                        return SentenceCandidate(sid, src, txt)  # type: ignore
                    except TypeError:
                        pass
        # Lightweight drop-in with the attrs downstream code uses
        return SimpleNamespace(sent_id=sid, text=txt, source_id=src, ce_score=ce_score)

    def _normalize_items(items: List[Any], default_source: str) -> List[Any]:
        """Convert dicts to objects with attrs; leave existing objects intact."""
        norm: List[Any] = []
        for it in items:
            if hasattr(it, "sent_id") and hasattr(it, "text"):
                norm.append(it)
                continue
            sid = str(it.get("sent_id", ""))
            txt = str(it.get("text", ""))
            src = str(it.get("source_id", it.get("source", default_source)))
            ce_sc = float(it.get("ce_score", 0.0)) if isinstance(it, dict) else 0.0
            norm.append(_mk_candidate(sid, txt, src, ce_sc))
        return norm

    # Fallback corpus read (only used if your private search helpers are unavailable)
    def _fallback_read_jsonl(p: Path) -> List[dict]:
        rows: List[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def _fallback_naive_topk(p: str, top_k: int) -> List[Any]:
        rows = _fallback_read_jsonl(Path(p))
        out: List[Any] = []
        for r in rows[: max(0, int(top_k))]:
            sid = str(r.get("sent_id", ""))
            txt = str(r.get("text", ""))
            src = str(r.get("source_id", r.get("source", "corpus")))
            out.append(_mk_candidate(sid, txt, src, ce_score=0.0))
        return out

    # --- Stage 1: BM25 and Dense pools ---
    if "_bm25_search" in globals():  # type: ignore[name-defined]
        bm25_pool_raw: List[Any] = _bm25_search(corpus_path, query_text, top_k=int(k_bm25))  # type: ignore[name-defined]
    else:
        bm25_pool_raw = _fallback_naive_topk(corpus_path, int(k_bm25))

    if "_dense_search" in globals():  # type: ignore[name-defined]
        dense_pool_raw: List[Any] = _dense_search(corpus_path, query_text, top_k=int(k_dense))  # type: ignore[name-defined]
    else:
        dense_pool_raw = _fallback_naive_topk(corpus_path, int(k_dense))

    bm25_pool: List[Any] = _normalize_items(bm25_pool_raw, default_source="bm25")
    dense_pool: List[Any] = _normalize_items(dense_pool_raw, default_source="dense")

    # --- Optional FAISS path (or deterministic brute-force fallback) BEFORE fusion ---
    if use_faiss and faiss_dir:
        try:
            from sro.retrieval.faiss_index import search_faiss  # type: ignore
            # Try to get your project embedding backend; else fall back to a deterministic hash embed
            embed_backend = None
            if "_get_embedding_backend" in globals():  # type: ignore[name-defined]
                try:
                    embed_backend = _get_embedding_backend()  # type: ignore[name-defined]
                except Exception:
                    embed_backend = None

            if embed_backend is None:
                # Fallback: dim taken from meta.json to match the index; default to 64.
                class _TinyHashEmbed:
                    def __init__(self, dim: int = 64) -> None:
                        self.dim = int(dim)

                    def encode(self, texts: Sequence[str], batch_size: int = 32) -> "np.ndarray":
                        import numpy as np  # local import to avoid hard dep at module import time
                        out = []
                        for t in texts:
                            v = np.zeros(self.dim, dtype=np.float32)
                            b = t.encode("utf-8")
                            for i, ch in enumerate(b):
                                v[(i + ch) % self.dim] += float((ch % 13) - 6)
                            out.append(v)
                        return np.vstack(out)

                dim = 64
                meta_path = os.path.join(str(faiss_dir), "meta.json")
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        dim = int(json.load(f).get("dim", 64))
                except Exception:
                    pass
                embed_backend = _TinyHashEmbed(dim)

            faiss_hits = search_faiss(
                query_text=query_text,
                top_k=max(0, int(k_dense)),
                embed_backend=embed_backend,  # consistent dim with index
                faiss_dir=str(faiss_dir),
            )
            for sid, txt, _sc in faiss_hits:
                dense_pool.append(_mk_candidate(str(sid), str(txt), "faiss", ce_score=0.0))
        except Exception as e:
            logger.warning("FAISS path failed; continuing without it: %r", e)

    # --- Fusion (RRF/MMR/CE) using your existing implementation if available ---
    if "_fuse_and_filter" in globals():  # type: ignore[name-defined]
        fused: List[Any] = _fuse_and_filter(  # type: ignore[name-defined]
            bm25_pool=bm25_pool,
            dense_pool=dense_pool,
            k_fused=int(k_fused),
            mmr_lambda=float(mmr_lambda),
            rrf_c=int(rrf_c),
            use_cross_encoder=bool(use_cross_encoder),
            cross_encoder=cross_encoder,
            rerank_top=int(rerank_top),
        )
    else:
        # Minimal deterministic fallback: BM25 then Dense, keep unique sent_id, cut to k_fused
        seen = set()
        fused_tmp: List[Any] = []
        for cand in bm25_pool + dense_pool:
            sid = getattr(cand, "sent_id", None)
            if sid in seen:
                continue
            seen.add(sid)
            fused_tmp.append(cand)
            if len(fused_tmp) >= int(k_fused):
                break
        fused = fused_tmp

    # --- Per-stage contribution log (P5 acceptance) ---
    logger.info(
        "RETRIEVAL bm25_hits=%d dense_hits=%d fused_kept=%d",
        len(bm25_pool),
        len(dense_pool),
        len(fused),
    )
    return fused


# --- add or replace this function in sro/retrieval/hybrid.py ---
from pathlib import Path
from typing import Callable, List, Optional

def make_fetch_more(
    corpus_path: str | Path,
    *,
    k_fused: int = 24,
    use_cross_encoder: bool = False,
    cross_encoder: object | None = None,
    rerank_top: int = 50,
    **_: object,
) -> Callable[..., List[SentenceCandidate]]:
    """
    Return a fetch-more function that supports BOTH signatures:
      (already_selected: List[SentenceCandidate], n_more: int) -> List[SentenceCandidate]
      (k: int, exclude_ids: Optional[Iterable[str]]) -> List[SentenceCandidate]
    We ignore CE-related kwargs in this offline demo.
    """
    triples = _read_corpus_lines(Path(corpus_path))

    def _by_exclude(k: int, exclude_ids: Optional[Iterable[str]] = None) -> List[SentenceCandidate]:
        k = max(0, int(k))
        excl = set(exclude_ids or [])
        out: List[SentenceCandidate] = []
        for sid, src, text in triples:
            if sid in excl:
                continue
            out.append(SentenceCandidate(sent_id=sid, text=text, source_id=src))
            if len(out) >= k:
                break
        return out

    def _by_already(already: List[SentenceCandidate], n_more: int) -> List[SentenceCandidate]:
        have = {c.sent_id for c in already}
        out: List[SentenceCandidate] = []
        for sid, src, text in triples:
            if sid in have:
                continue
            out.append(SentenceCandidate(sent_id=sid, text=text, source_id=src))
            if len(out) >= max(0, int(n_more)):
                break
        return out

    def _fn(*args, **kwargs) -> List[SentenceCandidate]:
        if len(args) >= 2 and isinstance(args[0], list):
            return _by_already(args[0], int(args[1]))
        if len(args) >= 1:
            k = int(args[0])
            excl = args[1] if len(args) >= 2 else None
            return _by_exclude(k, excl)
        # default: nothing requested
        return []

    return _fn
