# sro/retrieval/redundancy.py
"""
Does:
    Cosine-based redundancy control for S2 frontier selection using MMR.
    Uses L2-normalized sentence embeddings via sro.embeddings.backend. Pure logic.

Inputs:
    texts: List[str]                 # candidate sentences
    ids: List[str|None]              # stable per-sentence ids (for cache); can be None
    p1_scores: np.ndarray [N]        # relevance scores per candidate
    M: int                           # frontier size to select
    mmr_lambda: float                # trade-off (higher favors relevance)
    sim_threshold: float             # cosine >= threshold is treated as full duplicate
    embed_backend: EmbeddingBackend  # encoder instance

Outputs:
    dict:
      selected_idx: List[int]        # indices of selected frontier
      max_sim: np.ndarray [N]        # max cosine sim to selected set
      novelty: np.ndarray [N]        # 1 - max_sim

Notes:
    Deterministic tie-breaking: higher p1, then lower index.
"""
from __future__ import annotations

import numpy as np

from sro.embeddings.backend import EmbeddingBackend


def _ensure_unit_norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms

def mmr_select_cosine(
    texts: list[str],
    ids: list[str | None] | None,
    p1_scores: np.ndarray,
    M: int,
    *,
    mmr_lambda: float = 0.5,
    sim_threshold: float = 0.9,
    embed_backend: EmbeddingBackend,
) -> dict[str, object]:
    assert isinstance(p1_scores, np.ndarray) and p1_scores.ndim == 1, "p1_scores must be 1D np.ndarray"
    N = len(texts)
    assert N == p1_scores.shape[0], "texts and p1_scores length mismatch"
    if ids is None:
        ids = [None] * N
    assert len(ids) == N, "ids length mismatch"

    if N == 0:
        z: np.ndarray = np.zeros((0,), dtype=np.float32)
        return {"selected_idx": [], "max_sim": z, "novelty": z}

    M = int(max(0, min(M, N)))
    if M == 0:
        return {
            "selected_idx": [],
            "max_sim": np.zeros((N,), dtype=np.float32),
            "novelty": np.ones((N,), dtype=np.float32),
        }

    vecs = np.vstack([embed_backend.encode(texts[i], ids[i]) for i in range(N)]).astype(np.float32)
    vecs = _ensure_unit_norm(vecs)

    max_sim: np.ndarray = np.zeros((N,), dtype=np.float32)
    selected: list[int] = []
    in_selected: np.ndarray = np.zeros((N,), dtype=bool)

    # order: higher p1, then lower index
    order = np.lexsort((np.arange(N), -p1_scores))

    for _ in range(M):
        best_i, best_mmr = -1, -1e9
        for i in order:
            if in_selected[i]:
                continue
            mmr = mmr_lambda * float(p1_scores[i]) - (1.0 - mmr_lambda) * float(max_sim[i])
            if (mmr > best_mmr) or \
               (mmr == best_mmr and p1_scores[i] > (p1_scores[best_i] if best_i >= 0 else -1e9)) or \
               (mmr == best_mmr and p1_scores[i] == (p1_scores[best_i] if best_i >= 0 else -1e9) and i < best_i):
                best_i, best_mmr = i, mmr
        if best_i < 0:
            break

        selected.append(best_i)
        in_selected[best_i] = True

        vj = vecs[best_i]
        sims = np.clip(vecs @ vj, -1.0, 1.0)  # cosine = dot for unit vectors
        sims = np.where(sims >= sim_threshold, 1.0, sims)  # near-duplicates→full dup
        max_sim = np.maximum(max_sim, sims.astype(np.float32))

    novelty = 1.0 - max_sim
    return {"selected_idx": selected, "max_sim": max_sim, "novelty": novelty}
