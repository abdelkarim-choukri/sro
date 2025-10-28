# tests/test_frontier_cosine.py
import numpy as np
from sro.retrieval.redundancy import mmr_select_cosine
from sro.embeddings.backend import EmbeddingBackend

def test_mmr_cosine_filters_paraphrase(monkeypatch, tmp_path):
    # Work under temp to isolate cache
    monkeypatch.chdir(tmp_path)
    eb = EmbeddingBackend()

    # Two paraphrases + one distinct sentence
    texts = [
        "The iPhone 15 Pro features a titanium frame.",
        "A titanium frame was introduced on the iPhone 15 Pro model.",  # paraphrase
        "The base iPhone 15 uses aluminum, not titanium.",
    ]
    ids = ["p1", "p2", "p3"]
    # Make paraphrases more relevant to force MMR to choose between them
    p1 = np.asarray([0.95, 0.93, 0.70], dtype=np.float32)

    out = mmr_select_cosine(texts, ids, p1, M=2, mmr_lambda=0.5, sim_threshold=0.9, embed_backend=eb)
    selected = out["selected_idx"]
    assert len(selected) == 2

    # Ensure we did NOT pick both paraphrases
    # Count how many of {0,1} are included
    included = sum(int(i in selected) for i in (0, 1))
    assert included <= 1, f"Cosine-MMR should avoid selecting both paraphrases; got {selected}"
