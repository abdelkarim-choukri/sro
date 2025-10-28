# tests/test_hybrid_faiss_flag.py
from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
import pytest

from sro.retrieval.faiss_index import build_faiss_index, search_faiss

# Import hybrid to validate SPLADE flag behavior.
from sro.retrieval import hybrid as hybrid_mod  # type: ignore


class FakeEmbed:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for i, ch in enumerate(t.encode("utf-8")):
                v[(i + ch) % self.dim] += 1.0
            out.append(v)
        return np.vstack(out)


def _make_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"sent_id": "a", "text": "alpha beta gamma"},
        {"sent_id": "b", "text": "alpha beta"},
        {"sent_id": "c", "text": "delta epsilon zeta"},
        {"sent_id": "d", "text": "eta theta iota"},
    ]
    with open(corpus, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return corpus


def test_splade_flag_disabled():
    with pytest.raises(NotImplementedError):
        # We don't care about the other args; call the function and expect the guard to fire.
        hybrid_mod.get_initial_candidates(
            corpus_path="",
            query_text="anything",
            k_bm25=1,
            k_dense=1,
            k_fused=1,
            mmr_lambda=0.2,
            rrf_c=10,
            use_cross_encoder=False,
            cross_encoder=None,
            rerank_top=1,
            use_faiss=False,
            faiss_dir=None,
            retrieval_mix="splade",
        )


def test_faiss_fallback_without_index(tmp_path):
    # No FAISS index present -> falls back to brute force over embeddings.npy when available,
    # or returns empty list if nothing to search. Here we create a valid index artifacts dir.
    corpus = _make_corpus(tmp_path)
    out_dir = tmp_path / "faiss"
    be = FakeEmbed(dim=16)
    build_faiss_index(str(corpus), be, str(out_dir), bs=4)

    # Delete faiss binary index to force brute force
    idx_path = out_dir / "index.faiss"
    if os.path.exists(idx_path):
        os.remove(idx_path)

    hits = search_faiss("alpha", top_k=2, embed_backend=be, faiss_dir=str(out_dir))
    assert len(hits) >= 1  # non-empty via brute-force path


@pytest.mark.skipif(
    "faiss" not in globals(), reason="FAISS not installed in environment"
)
def test_faiss_vs_bruteforce_top1_match(tmp_path):
    corpus = _make_corpus(tmp_path)
    out_dir = tmp_path / "faiss"
    be = FakeEmbed(dim=16)
    build_faiss_index(str(corpus), be, str(out_dir), bs=4)

    # With FAISS present — get top1
    hits_faiss = search_faiss("alpha", top_k=3, embed_backend=be, faiss_dir=str(out_dir))
    assert len(hits_faiss) >= 1
    top1_faiss = hits_faiss[0][0]

    # Force brute-force by renaming the index file
    idx = out_dir / "index.faiss"
    if os.path.exists(idx):
        os.rename(idx, out_dir / "index.faiss.bak")

    hits_bf = search_faiss("alpha", top_k=3, embed_backend=be, faiss_dir=str(out_dir))
    assert len(hits_bf) >= 1
    top1_bf = hits_bf[0][0]

    assert top1_bf == top1_faiss
