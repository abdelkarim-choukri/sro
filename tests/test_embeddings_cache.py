# tests/test_embeddings_cache.py
import os

import numpy as np

from sro.embeddings.backend import EmbeddingBackend


def test_embedding_cache_persistence(tmp_path, monkeypatch):
    # Redirect cache to temp dir
    cache_dir = tmp_path / "artifacts" / "cache"
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.chdir(tmp_path)  # working dir becomes tmp

    eb = EmbeddingBackend()  # should create artifacts/cache/embeddings.sqlite
    v1 = eb.encode("Hello world.", "s1")
    v2 = eb.encode("Hello world.", "s1")
    assert v1.shape == v2.shape == (v1.size,), "Vector must be 1-D"
    # same id -> same vector
    assert np.allclose(v1, v2, atol=1e-6)
    # file exists
    assert os.path.exists("artifacts/cache/embeddings.sqlite")

    # Different text/id -> different vector (usually; allow tiny tolerance)
    v3 = eb.encode("Totally different sentence.", "s2")
    assert not np.allclose(v1, v3, atol=1e-3)
