# sro/retrieval/hybrid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
import json
import math
import re
import os
import torch
from sro.utils.st import get_st
import numpy as np
from sentence_transformers import SentenceTransformer

from sro.types import SentenceCandidate
from sro.retrieval.bm25 import BM25OkapiLite

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    return [t.lower() for t in _WORD_RE.findall(s)]

def _split_sentences(text: str) -> List[str]:
    # Simple rule-based splitter; good enough for newsy prose
    # Splits on [.?!] followed by space and capital or EOS.
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+(?=[A-Z\"'])", text.strip())
    # Fallback if none found
    if len(parts) == 1:
        return [text.strip()]
    # Clean
    return [p.strip() for p in parts if p.strip()]

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

@dataclass
class SentenceRow:
    sent_id: str
    source_id: str
    text: str
    tokens: List[str]

class CorpusIndex:
    """
    Loads a JSONL corpus with fields:
      - source_id: str
      - text: str  (document text; we will split into sentences)

    Produces a sentence table, BM25 index, and exposes dense encoding.
    """
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.rows: List[SentenceRow] = []
        self._load()
        self._build_bm25()
        self._dense_model: Optional[SentenceTransformer] = None
        self._dense_matrix: Optional[np.ndarray] = None  # [N, d]

    def _load(self) -> None:
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Corpus JSONL not found: {self.jsonl_path}")
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                source_id = obj.get("source_id", "")
                text = obj.get("text", "")
                sents = _split_sentences(text)
                for k, sent in enumerate(sents):
                    sid = f"{source_id}#s{k}"
                    self.rows.append(SentenceRow(sent_id=sid, source_id=source_id, text=sent, tokens=_tokens(sent)))

    def _build_bm25(self) -> None:
        self._bm25 = BM25OkapiLite([r.tokens for r in self.rows])

    # ---- Dense ----
    def ensure_dense(self) -> None:
        """Lazy-init the dense encoder once."""
        if self._dense_model is None:
            model_id = getattr(self, "dense_model_name", None) or "sentence-transformers/all-MiniLM-L6-v2"
            
            # cache_folder ensures local reuse; device is explicit
            self._dense_model = get_st(model_id)

        if self._dense_matrix is None:
            texts = [r.text for r in self.rows]
            self._dense_matrix = self._dense_model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=128,            # lower to 32 if VRAM is tight
                show_progress_bar=False,
                normalize_embeddings=True, # cosine == dot
            )


    def dense_query(self, query: str) -> np.ndarray:
        assert self._dense_model is not None
        vec = self._dense_model.encode([query], convert_to_numpy=True, batch_size=1, show_progress_bar=False, normalize_embeddings=True)
        return vec[0]

    # ---- Retrieval ----
    def search(self,
               query: str,
               k_bm25: int = 200,
               k_dense: int = 200,
               k_fused: int = 128,
               mmr_lambda: float = 0.7,
               rrf_c: int = 60,
               use_cross_encoder: bool = False,
               cross_encoder=None,
               rerank_top: int = 64) -> List[SentenceCandidate]:
        """
        Return top-k_fused SentenceCandidate for a query using BM25 + Dense + RRF + MMR (+ optional cross-encoder rerank).

        Variables:
          k_bm25: number of BM25 hits to keep before fusion.
          k_dense: number of dense hits to keep before fusion.
          k_fused: final pool size after RRF + MMR.
          mmr_lambda (λ): trade-off for MMR selection.
          rrf_c: RRF constant (larger → more uniform).
          use_cross_encoder: if True, apply cross-encoder rerank on top 'rerank_top' sentences.
        """
        # BM25
        q_tokens = _tokens(query)
        bm25_scores = self._bm25.get_scores(q_tokens)
        # get top-k_bm25 indices
        idx_bm = np.argsort(np.array(bm25_scores))[::-1][:min(k_bm25, len(self.rows))]

        # Dense
        self.ensure_dense()
        qvec = self.dense_query(query)
        sims = (self._dense_matrix @ qvec)  # cosine since normalized
        idx_de = np.argsort(sims)[::-1][:min(k_dense, len(self.rows))]

        # Fusion via RRF
        # Build rank maps
        rank_bm = {int(i): r for r, i in enumerate(idx_bm, start=1)}
        rank_de = {int(i): r for r, i in enumerate(idx_de, start=1)}
        cand_ids = set(rank_bm) | set(rank_de)

        def rrf_score(i: int) -> float:
            rb = rank_bm.get(i)
            rd = rank_de.get(i)
            s = 0.0
            if rb is not None:
                s += 1.0 / (rrf_c + rb)
            if rd is not None:
                s += 1.0 / (rrf_c + rd)
            return s

        # sort by RRF desc
        fused = sorted(cand_ids, key=lambda i: rrf_score(i), reverse=True)

        # MMR on fused list to enforce novelty
        selected: List[int] = []
        tok_sets: List[Set[str]] = [set(self.rows[i].tokens) for i in fused]
        while fused and len(selected) < min(k_fused, len(self.rows)):
            # pick the next best by MMR
            best_i, best_sc = None, -1e9
            for pos, idx in enumerate(fused):
                # relevance proxy: combined signal (bm25 norm + dense sim norm)
                rb = rank_bm.get(idx, len(self.rows))
                rd = rank_de.get(idx, len(self.rows))
                rel = (1.0 / (rrf_c + rb)) + (1.0 / (rrf_c + rd))
                if not selected:
                    mmr = rel
                else:
                    # redundancy: max Jaccard vs selected
                    toks_i = tok_sets[pos]
                    max_sim = 0.0
                    for sel in selected:
                        toks_j = set(self.rows[sel].tokens)
                        # Jaccard similarity
                        inter = len(toks_i & toks_j)
                        uni = len(toks_i | toks_j)
                        s = (inter / uni) if uni else 0.0
                        if s > max_sim:
                            max_sim = s
                    mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                if mmr > best_sc:
                    best_sc, best_i = mmr, idx
            selected.append(best_i)
            # remove best_i from fused list
            fused = [i for i in fused if i != best_i]

        # Optionally rerank with cross-encoder
        final_idx = selected[:]
        if use_cross_encoder and cross_encoder is not None:
            texts = [self.rows[i].text for i in final_idx[:rerank_top]]
            scores = cross_encoder.score_pairs([query] * len(texts), texts)  # higher is better
            order = np.argsort(np.array(scores))[::-1]
            # apply rerank within rerank_top window
            reranked = [final_idx[i] for i in order] + final_idx[len(order):]
            final_idx = reranked

        # Build SentenceCandidate list
        out: List[SentenceCandidate] = []
        # Normalize a composite ce_score into [0,1] for downstream (simple min-max)
        # Use rel (same formula as above) as ce_score base
        rel_vals = []
        for idx in final_idx:
            rb = rank_bm.get(idx, len(self.rows))
            rd = rank_de.get(idx, len(self.rows))
            rel_vals.append((1.0 / (rrf_c + rb)) + (1.0 / (rrf_c + rd)))
        if rel_vals:
            rmin, rmax = min(rel_vals), max(rel_vals)
        else:
            rmin, rmax = 0.0, 1.0

        for k, idx in enumerate(final_idx):
            row = self.rows[idx]
            rel = rel_vals[k] if k < len(rel_vals) else 0.0
            ce = 0.0 if rmax == rmin else (rel - rmin) / (rmax - rmin)
            out.append(SentenceCandidate(sent_id=row.sent_id, text=row.text, source_id=row.source_id, ce_score=float(ce)))
        return out

# Convenience factory to build a fetch_more callback usable by SROProver
def make_fetch_more(corpus_jsonl: str,
                    use_cross_encoder: bool = False,
                    cross_encoder=None,
                    k_fused: int = 24) -> callable:
    """
    Return a fetch_more(claim=..., **kwargs) callback that returns
    up to k_fused SentenceCandidate from the corpus for the given claim string.
    """
    index = CorpusIndex(corpus_jsonl)
    def _fetch_more(**kwargs):
        # kwargs may include: claim, candidates, frontier_idx, pool2_idx, p1, top_ub
        claim_text = kwargs.get("claim", "")
        return index.search(claim_text,
                            k_bm25=200,
                            k_dense=200,
                            k_fused=k_fused,
                            mmr_lambda=0.7,
                            rrf_c=60,
                            use_cross_encoder=use_cross_encoder,
                            cross_encoder=cross_encoder,
                            rerank_top=min(64, k_fused))
    return _fetch_more


def get_initial_candidates(
    corpus_jsonl: str,
    query: str,
    *,
    k_bm25: int = 200,
    k_dense: int = 200,
    k_fused: int = 24,
    mmr_lambda: float = 0.7,
    rrf_c: int = 60,
    use_cross_encoder: bool = False,
    cross_encoder=None,
    rerank_top: int = 64,
):
    """
    First-pass retrieval for initial candidate pool.

    Inputs:
      query: claim text (string).
    Returns:
      List[SentenceCandidate] of length ≤ k_fused, with ce_score ∈ [0,1].
    """
    index = CorpusIndex(corpus_jsonl)
    return index.search(
        query,
        k_bm25=k_bm25,
        k_dense=k_dense,
        k_fused=k_fused,
        mmr_lambda=mmr_lambda,
        rrf_c=rrf_c,
        use_cross_encoder=use_cross_encoder,
        cross_encoder=cross_encoder,
        rerank_top=rerank_top,
    )
