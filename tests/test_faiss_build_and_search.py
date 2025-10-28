# tests/test_faiss_build_and_search.py
from __future__ import annotations

import json
import os
from typing import List, Sequence

import numpy as np
import pytest

from sro.retrieval.faiss_index import build_faiss_index, search_faiss


class FakeEmbed:
    """
    Deterministic, offline-safe embedding backend for tests.

    It tokenizes on spaces and maps characters to counts. Low-dim but fixed.
    """
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out: List[np.ndarray] = []
        for t in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for i, ch in enumerate(t.encode("utf-8")):
                vec[i % self.dim] += float((ch % 13) - 6)
            out.append(vec)
        return np.vstack(out)


@pytest.mark.skipif(len([m for m in ("faiss",) if m in globals()]) < 0, reason="n/a")
def test_build_then_query(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"sent_id": "s1", "text": "The iPhone 15 Pro has a titanium frame."},
        {"sent_id": "s2", "text": "Bananas are yellow."},
        {"sent_id": "s3", "text": "The frame of iPhone 15 Pro is made of titanium alloy."},
        {"sent_id": "s4", "text": "Mars is the red planet."},
    ]
    with open(corpus, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "faiss"
    be = FakeEmbed(dim=32)
    build_faiss_index(str(corpus), be, str(out_dir), bs=2)

    # Deterministic search, top-1 should be semantically similar sentence
    q = "Does the iPhone 15 Pro have a titanium frame?"
    hits = search_faiss(q, top_k=3, embed_backend=be, faiss_dir=str(out_dir))
    assert len(hits) >= 1
    top1 = hits[0][0]  # sent_id
    assert top1 in {"s1", "s3"}  # either of the two iPhone sentences

    # Determinism: repeated search yields same top1
    hits2 = search_faiss(q, top_k=3, embed_backend=be, faiss_dir=str(out_dir))
    assert hits2[0][0] == top1
