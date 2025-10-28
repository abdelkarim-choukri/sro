# sro/retrieval/faiss_index.py
from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAVE_FAISS = False


# ----------------------------
# Protocol: minimal embed backend interface
# ----------------------------
class EmbeddingBackend(Protocol):
    """
    Minimal interface expected by this module.

    Must be deterministic if the global seed is set elsewhere.
    """
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        ...


# ----------------------------
# Utils
# ----------------------------
def _ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_id2text(jsonl_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in _iter_jsonl(jsonl_path):
        sid = str(row["sent_id"])
        txt = str(row["text"])
        out[sid] = txt
    return out


@dataclass(frozen=True)
class _IndexPaths:
    dir: str
    embeddings: str
    ids: str
    faiss_index: str
    meta: str
    corpus_copy: str


def _paths(base_dir: str) -> _IndexPaths:
    d = os.path.abspath(base_dir)
    return _IndexPaths(
        dir=d,
        embeddings=os.path.join(d, "embeddings.npy"),
        ids=os.path.join(d, "ids.npy"),
        faiss_index=os.path.join(d, "index.faiss"),
        meta=os.path.join(d, "meta.json"),
        corpus_copy=os.path.join(d, "corpus.jsonl"),
    )


# ----------------------------
# Public API
# ----------------------------
def build_faiss_index(
    corpus_jsonl: str,
    embed_backend: EmbeddingBackend,
    out_dir: str,
    bs: int = 256,
) -> None:
    """
    Build a local dense index for sentences.

    Reads JSONL with {"sent_id": str, "text": str}.
    Saves:
      - embeddings.npy (float32, L2-normalized)
      - ids.npy (np.str_ array)
      - index.faiss (if FAISS available)
      - meta.json
      - corpus.jsonl (copy for id->text lookup)
    """
    _ensure_dir(out_dir)
    P = _paths(out_dir)

    ids: List[str] = []
    texts: List[str] = []
    for row in _iter_jsonl(corpus_jsonl):
        ids.append(str(row["sent_id"]))
        texts.append(str(row["text"]))

    # Encode in batches
    embs: List[np.ndarray] = []
    n = len(texts)
    for i in range(0, n, bs):
        batch = texts[i : i + bs]
        e = embed_backend.encode(batch, batch_size=bs)
        if not isinstance(e, np.ndarray):
            e = np.asarray(e)
        if e.dtype != np.float32:
            e = e.astype(np.float32)
        embs.append(e)

    X = np.vstack(embs)
    if X.shape[0] != n:
        raise RuntimeError(f"Embedding count mismatch: got {X.shape[0]} vs {n}")

    # Normalize for cosine via inner product
    X = _unit_norm(X)

    # Persist
    np.save(P.embeddings, X)
    np.save(P.ids, np.asarray(ids, dtype=np.str_))
    _save_json(
        P.meta,
        {
            "dim": int(X.shape[1]),
            "size": int(X.shape[0]),
            "have_faiss": bool(_HAVE_FAISS),
        },
    )
    # Keep a corpus copy for id->text lookup
    if os.path.abspath(corpus_jsonl) != os.path.abspath(P.corpus_copy):
        # Cheap copy (line-by-line) to avoid shutil import / permissions weirdness on Windows
        with open(corpus_jsonl, "r", encoding="utf-8") as src, open(P.corpus_copy, "w", encoding="utf-8") as dst:
            for ln in src:
                dst.write(ln)

    # Optional FAISS
    if _HAVE_FAISS:
        dim = int(X.shape[1])
        index = faiss.IndexFlatIP(dim)  # cosine via inner product since normalized
        index.add(X)
        faiss.write_index(index, P.faiss_index)


def search_faiss(
    query_text: str,
    top_k: int,
    embed_backend: EmbeddingBackend,
    faiss_dir: str,
) -> List[Tuple[str, str, float]]:
    """
    Search the local dense index by cosine similarity.

    Returns: List of (sent_id, text, score) with score in [−1, 1] (cosine).
    If FAISS is missing or index file absent, falls back to brute-force cosine over embeddings.npy.
    Never crashes: produces an empty list if inputs are inconsistent.
    """
    P = _paths(faiss_dir)
    if not os.path.exists(P.embeddings) or not os.path.exists(P.ids) or not os.path.exists(P.corpus_copy):
        # clean fallback behavior: nothing to search
        return []

    # Load embeddings + ids + id->text
    X = np.load(P.embeddings)  # (N, D) normalized
    ids = np.load(P.ids)
    id2text = _load_id2text(P.corpus_copy)

    # Encode query
    q = embed_backend.encode([query_text], batch_size=1)
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    if q.dtype != np.float32:
        q = q.astype(np.float32)
    q = _unit_norm(q)
    q = q.reshape(1, -1)
    if q.shape[1] != X.shape[1]:
        # Dimension mismatch — deterministic empty result
        return []

    # Try FAISS search; fall back to brute-force cosine
    scores: np.ndarray
    if _HAVE_FAISS and os.path.exists(P.faiss_index):
        index = faiss.read_index(P.faiss_index)  # type: ignore
        scores_np, idxs_np = index.search(q, min(top_k, X.shape[0]))  # type: ignore
        scores = scores_np[0]
        idxs = idxs_np[0]
    else:
        # cosine since X and q are normalized
        scores = (X @ q.T).reshape(-1)  # (N,)
        # top-k by partial argsort
        k = min(top_k, X.shape[0])
        if k <= 0:
            return []
        idxs = np.argpartition(-scores, k - 1)[:k]
        # sort exact top-k
        order = np.argsort(-scores[idxs], kind="mergesort")
        idxs = idxs[order]

    out: List[Tuple[str, str, float]] = []
    for i in idxs:
        sid = str(ids[i])
        txt = id2text.get(sid, "")
        sc = float(scores[i])
        out.append((sid, txt, sc))
    return out
