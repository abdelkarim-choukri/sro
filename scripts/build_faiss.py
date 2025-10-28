# scripts/build_faiss.py
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np


def _load_embedding_backend():
    """
    Try project backends; if none, return a deterministic offline-safe embed.
    """
    tried: list[str] = []
    for modpath, factory in [
        ("sro.retrieval.hybrid", "_get_embedding_backend"),
        ("sro.retrieval.embedding_backend", "EmbeddingBackend"),
        ("sro.retrieval.embedding", "EmbeddingBackend"),
        ("sro.embedding.backend", "EmbeddingBackend"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[factory])
            obj = getattr(mod, factory)
            # factory function
            if callable(obj):
                be = obj()
                if be is not None:
                    return be
            # class with default ctor
            return obj()
        except Exception as e:
            tried.append(f"{modpath}.{factory}: {e!r}")
            continue

    # Fallback: deterministic tiny hash embed (must match hybrid's fallback algo)
    class _TinyHashEmbed:
        def __init__(self, dim: int = 64) -> None:
            self.dim = int(dim)

        def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
            out: list[np.ndarray] = []
            for t in texts:
                v: np.ndarray = np.zeros(self.dim, dtype=np.float32)
                b = t.encode("utf-8")
                for i, ch in enumerate(b):
                    v[(i + ch) % self.dim] += float((ch % 13) - 6)
                out.append(v)
            return np.vstack(out)

    return _TinyHashEmbed(dim=64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    from sro.retrieval.faiss_index import build_faiss_index  # local import keeps startup light

    backend = _load_embedding_backend()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    build_faiss_index(args.corpus, backend, args.out, bs=int(args.bs))

    # Log summary from meta.json
    meta_path = os.path.join(args.out, "meta.json")
    embeddings_path = os.path.join(args.out, "embeddings.npy")
    ids_path = os.path.join(args.out, "ids.npy")
    index_path = os.path.join(args.out, "index.faiss")

    size = dim = 0
    have_faiss = False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        size = int(meta.get("size", 0))
        dim = int(meta.get("dim", 0))
        have_faiss = bool(meta.get("have_faiss", False))
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info(
        "FAISS build done: size=%d dim=%d have_faiss=%s embeddings=%s ids=%s index=%s",
        size,
        dim,
        have_faiss,
        os.path.exists(embeddings_path),
        os.path.exists(ids_path),
        os.path.exists(index_path),
    )


if __name__ == "__main__":
    main()
