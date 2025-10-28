# scripts/build_faiss.py
from __future__ import annotations

import argparse
import logging
import os
import sys

from sro.retrieval.faiss_index import build_faiss_index

# We try a few likely import locations for your embedding backend.
# Adjust if your repo uses a different path/name.
# The backend must expose: encode(texts: list[str], batch_size: int) -> np.ndarray
def _load_embedding_backend():
    """
    Try project backends; if none, return a deterministic offline-safe embed.
    """
    tried = []
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
            return obj()  # type: ignore[call-arg]
        except Exception as e:
            tried.append(f"{modpath}.{factory}: {e!r}")
            continue

    # Fallback: deterministic tiny hash embed (must match hybrid's fallback algo)
    import numpy as np

    class _TinyHashEmbed:
        def __init__(self, dim: int = 64) -> None:
            self.dim = int(dim)

        def encode(self, texts, batch_size: int = 32):
            out = []
            for t in texts:
                v = np.zeros(self.dim, dtype=np.float32)
                b = t.encode("utf-8")
                for i, ch in enumerate(b):
                    v[(i + ch) % self.dim] += float((ch % 13) - 6)
                out.append(v)
            return np.vstack(out)

    return _TinyHashEmbed(dim=64)


def main() -> None:
    ap = argparse.ArgumentParser("Build embeddings + optional FAISS index (offline-safe).")
    ap.add_argument("--corpus", required=True, help="Path to JSONL with {sent_id, text}.")
    ap.add_argument("--out", required=True, help="Output directory for FAISS artifacts.")
    ap.add_argument("--bs", type=int, default=256, help="Batch size for embedding.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)

    backend = _load_embedding_backend()
    build_faiss_index(args.corpus, backend, args.out, bs=args.bs)

    from sro.retrieval.faiss_index import _paths  # type: ignore
    P = _paths(args.out)
    import json, numpy as np  # noqa

    meta = {}
    if os.path.exists(P.meta):
        with open(P.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

    dim = meta.get("dim", "?")
    size = meta.get("size", "?")
    have_faiss = meta.get("have_faiss", False)

    logging.info(
        f"FAISS build done: size={size} dim={dim} have_faiss={have_faiss} "
        f"embeddings={os.path.exists(P.embeddings)} ids={os.path.exists(P.ids)} "
        f"index={os.path.exists(P.faiss_index)}"
    )


if __name__ == "__main__":
    main()
