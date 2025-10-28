# sro/retrieval/faiss_index.py
from __future__ import annotations

import json
import logging
import os
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore[assignment]
    _HAVE_FAISS = False



# ----------------------------
# Protocol: minimal embed backend interface
# ----------------------------
class EmbeddingBackend(Protocol):
    """Embedding backend must be deterministic if the global seed is set."""
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


# ----------------------------
# Utils
# ----------------------------
def _ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _save_json(path: str, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _detect_keys(sample: dict[str, Any]) -> tuple[str | None, str | None]:
    """Choose id/text keys from a sample row, robust to many schemas."""
    id_candidates = ["sent_id", "id", "sid", "source_id", "doc_id", "uid"]
    text_candidates = ["text", "contents", "content", "passage", "sentence", "body"]
    id_key = next((k for k in id_candidates if k in sample), None)
    txt_key = next((k for k in text_candidates if k in sample), None)
    return id_key, txt_key


def _load_id2text(jsonl_path: str) -> dict[str, str]:
    """Load a stable id->text mapping with schema auto-detection."""
    rows = list(_iter_jsonl(jsonl_path))
    if not rows:
        return {}
    id_key, txt_key = _detect_keys(rows[0])
    if txt_key is None:
        txt_key = "text"
    out: dict[str, str] = {}
    for i, row in enumerate(rows):
        sid = (str(row[id_key]) if id_key and (id_key in row) else f"row:{i}")
        txt = str(row.get(txt_key, ""))
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

    Reads JSONL with at least a text-like field.
    Accepted id keys: sent_id/id/sid/source_id/doc_id/uid; otherwise auto-ids row:<idx>.
    Saves:
      - embeddings.npy (float32, L2-normalized)
      - ids.npy (np.str_ array)
      - index.faiss (if FAISS available and writeable)
      - meta.json
      - corpus.jsonl (copy for id->text lookup)
    """
    _ensure_dir(out_dir)
    P = _paths(out_dir)

    rows: list[dict[str, Any]] = list(_iter_jsonl(corpus_jsonl))
    if not rows:
        # Write empty artifacts deterministically
        np.save(P.embeddings, np.zeros((0, 1), dtype=np.float32))
        np.save(P.ids, np.asarray([], dtype=np.str_))
        _save_json(P.meta, {"dim": 0, "size": 0, "have_faiss": bool(_HAVE_FAISS)})
        # keep an empty copy for completeness
        if os.path.abspath(corpus_jsonl) != os.path.abspath(P.corpus_copy):
            open(P.corpus_copy, "w", encoding="utf-8").close()
        return

    id_key, txt_key = _detect_keys(rows[0])
    if txt_key is None:
        txt_key = "text"

    ids: list[str] = []
    texts: list[str] = []
    for i, row in enumerate(rows):
        sid = (str(row[id_key]) if id_key and (id_key in row) else f"row:{i}")
        txt = str(row.get(txt_key, ""))
        ids.append(sid)
        texts.append(txt)

    # Encode in batches
    embs: list[np.ndarray] = []
    n = len(texts)
    for i in range(0, n, bs):
        batch = texts[i : i + bs]
        e = embed_backend.encode(batch, batch_size=bs)
        e = np.asarray(e, dtype=np.float32)
        embs.append(e)

    X = np.vstack(embs) if embs else np.zeros((0, 1), dtype=np.float32)
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
            "id_key": id_key,
            "text_key": txt_key,
        },
    )
    # Keep a corpus copy for id->text lookup
    if os.path.abspath(corpus_jsonl) != os.path.abspath(P.corpus_copy):
        with open(P.corpus_copy, "w", encoding="utf-8") as dst:
            for r in rows:
                dst.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Optional FAISS binary index (Windows/Unicode-safe fallback)
    if _HAVE_FAISS and X.size > 0:
        try:
            dim = int(X.shape[1])
            index = faiss.IndexFlatIP(dim)  # cosine via inner product on unit vectors
            index.add(X)
            faiss.write_index(index, P.faiss_index)
        except Exception as e:
            # Known issue on Windows with non-ASCII temp dirs; fall back to brute-force.
            logging.warning(
                "FAISS index write failed (%r); continuing without index. "
                "Brute-force cosine over embeddings.npy will be used.", e
            )


def search_faiss(
    query_text: str,
    top_k: int,
    embed_backend: EmbeddingBackend,
    faiss_dir: str,
) -> list[tuple[str, str, float]]:
    """
    Search the local dense index by cosine similarity.

    Returns: List of (sent_id, text, score) with score in [-1, 1] (cosine).
    If FAISS is missing or index file absent, falls back to brute-force cosine over embeddings.npy.
    Never crashes: produces an empty list if inputs are inconsistent.
    """
    P = _paths(faiss_dir)
    if not (os.path.exists(P.embeddings) and os.path.exists(P.ids) and os.path.exists(P.corpus_copy)):
        return []

    # Load embeddings + ids + id->text
    X = np.load(P.embeddings)  # (N, D) normalized
    ids = np.load(P.ids)
    id2text = _load_id2text(P.corpus_copy)

    # Encode query
    q = embed_backend.encode([query_text], batch_size=1)
    q = np.asarray(q, dtype=np.float32)
    q = _unit_norm(q).reshape(1, -1)
    if q.shape[1] != X.shape[1]:
        return []

    # Try FAISS, else brute-force
    if _HAVE_FAISS and os.path.exists(P.faiss_index):
        index = faiss.read_index(P.faiss_index)  # pyright: ignore[reportAttributeAccessIssue]
        scores_np, idxs_np = index.search(q, min(top_k, X.shape[0]))

        scores = scores_np[0]
        idxs = idxs_np[0]
    else:
        scores = (X @ q.T).reshape(-1)  # (N,)
        k = min(top_k, X.shape[0])
        if k <= 0:
            return []
        idxs = np.argpartition(-scores, k - 1)[:k]
        order = np.argsort(-scores[idxs], kind="mergesort")
        idxs = idxs[order]

    out: list[tuple[str, str, float]] = []
    for i in idxs:
        sid = str(ids[i])
        txt = id2text.get(sid, "")
        sc = float(scores[i])
        out.append((sid, txt, sc))
    return out
