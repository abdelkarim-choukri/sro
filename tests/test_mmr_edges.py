# tests/test_mmr_edges.py
import numpy as np
from sro.embeddings.backend import EmbeddingBackend
from sro.retrieval.redundancy import mmr_select_cosine

def test_mmr_empty_inputs():
    eb = EmbeddingBackend()
    # N == 0; should return empty selections and sane arrays
    out = mmr_select_cosine([], None, np.zeros((0,), dtype=np.float32), M=0, embed_backend=eb)
    assert out["selected_idx"] == []
    assert out["max_sim"].shape == (0,)
    assert out["novelty"].shape == (0,)

def test_mmr_zero_budget():
    eb = EmbeddingBackend()
    texts = ["a", "b", "c"]
    p1 = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    # M == 0; should not select anything but produce per-item arrays
    out = mmr_select_cosine(texts, None, p1, M=0, embed_backend=eb)
    assert out["selected_idx"] == []
    assert out["max_sim"].shape == (3,)
    assert out["novelty"].shape == (3,)
