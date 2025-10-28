from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pytest

from sro.retrieval.faiss_index import build_faiss_index, search_faiss


class FakeEmbed:
    """
    Deterministic, offline-safe embedding backend for tests.

    It hashes UTF-8 bytes into a fixed-size vector. This is NOT semantic,
    so tests must NOT rely on exact rank-1 — only inclusion + determinism.
    """
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out: list[np.ndarray] = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            b = t.encode("utf-8")
            for i, ch in enumerate(b):
                v[(i + ch) % self.dim] += float((ch % 13) - 6)
            out.append(v)
        return np.vstack(out)


@pytest.mark.skipif(False, reason="always run")
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

    # Deterministic search; the relevant iPhone/titanium sentences should be in top-3
    q = "Does the iPhone 15 Pro have a titanium frame?"
    hits1 = search_faiss(q, top_k=3, embed_backend=be, faiss_dir=str(out_dir))
    assert len(hits1) >= 1

    top_ids1 = [sid for (sid, _txt, _sc) in hits1]
    assert {"s1", "s3"} & set(top_ids1), f"Expected s1 or s3 in top-3, got {top_ids1}"

    # Determinism: same top_ids across repeated calls
    hits2 = search_faiss(q, top_k=3, embed_backend=be, faiss_dir=str(out_dir))
    top_ids2 = [sid for (sid, _txt, _sc) in hits2]
    assert top_ids2 == top_ids1, "search_faiss should be deterministic for same inputs"
