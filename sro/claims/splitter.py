from __future__ import annotations
from typing import List, Tuple, Set, Dict, Optional
import re
import math
import numpy as np
from sentence_transformers import SentenceTransformer

from sro.types import Claim, SentenceCandidate

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokens(s: str) -> Set[str]:
    return set(t.lower() for t in _WORD_RE.findall(s or ""))

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _source_prefix(source_id: str) -> str:
    # "news:1" -> "news"; "press:1" -> "press"
    return (source_id or "").split(":", 1)[0] if source_id else ""

class _STEmbedder:
    """Singleton-ish mini wrapper to avoid reloading the encoder."""
    _model = None
    @classmethod
    def encode(cls, texts: List[str]) -> np.ndarray:
        if cls._model is None:
            # small and fast; matches retrieval model family
            self._dense_model = SentenceTransformer(model_name, cache_folder="models_cache")

        return cls._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # vectors are already normalized; robust fallback
    num = float((a * b).sum())
    return max(-1.0, min(1.0, num))

def _is_hedged(text: str, hedge_patterns: List[re.Pattern]) -> bool:
    low = (text or "").lower()
    return any(p.search(low) is not None for p in hedge_patterns)

def pick_top_sentences(
    question: str,
    cands: List[SentenceCandidate],
    *,
    K: int,
    min_question_cosine: float,
    hedge_terms: List[str],
    reliability_weights: Dict[str, float],
    max_sim: float = 0.85,
) -> List[SentenceCandidate]:
    """
    Select up to K sentences using:
      - hedge filter (drop speculative sentences),
      - question–sentence cosine ≥ min_question_cosine,
      - reliability reweighting of ce_score,
      - novelty (Jaccard ≤ max_sim).
    """
    if not cands:
        return []

    # Compile hedge regexes once
    hedge_patterns = [re.compile(h, flags=re.I) for h in hedge_terms or []]

    # Encode question once
    qv = _STEmbedder.encode([question])[0]
    # Precompute sentence vectors for cosine gate
    sv = _STEmbedder.encode([c.text for c in cands])

    # Compute an adjusted relevance:
    # rel = ce_score * w_src * max(cosine, 0)
    adj_scores: List[Tuple[int, float]] = []
    for i, s in enumerate(cands):
        if _is_hedged(s.text, hedge_patterns):
            continue  # drop hedged
        cos = _cosine(qv, sv[i])
        if cos < min_question_cosine:
            continue  # too off-topic for the question
        w_src = reliability_weights.get(_source_prefix(s.source_id), 1.0)
        rel = (s.ce_score or 0.0) * float(w_src) * max(cos, 0.0)
        adj_scores.append((i, rel))

    if not adj_scores:
        return []

    # Sort by adjusted relevance
    adj_scores.sort(key=lambda t: t[1], reverse=True)
    order = [i for (i, _) in adj_scores]

    # Novelty filter via Jaccard
    chosen: List[SentenceCandidate] = []
    chosen_toks: List[Set[str]] = []
    for i in order:
        tok = _tokens(cands[i].text)
        if any(_jaccard(tok, t) > max_sim for t in chosen_toks):
            continue
        chosen.append(cands[i])
        chosen_toks.append(tok)
        if len(chosen) >= K:
            break

    return chosen

def draft_and_claims(
    question: str,
    initial_cands: List[SentenceCandidate],
    *,
    # New API
    K: Optional[int] = None,
    min_question_cosine: Optional[float] = None,
    hedge_terms: Optional[List[str]] = None,
    reliability_weights: Optional[Dict[str, float]] = None,
    # Backward-compat (old API)
    max_claims: Optional[int] = None,
) -> Tuple[str, List[Claim]]:
    """
    Make a short draft by concatenating top sentences (filtered), and turn each into a claim.

    Backward-compat: if `max_claims` is provided and `K` is None, we use `max_claims`.
    If the new filtering knobs are omitted, we fall back to safe defaults:
      - min_question_cosine = 0.30
      - hedge_terms = basic hedging patterns
      - reliability_weights = {'news':1.0,'press':0.95,'blog':0.60,'seed':0.80,'alt':0.75}
    """
    # Resolve K with backward-compat
    if K is None:
        K = max_claims if max_claims is not None else 3

    # Defaults for new knobs if not passed (test code may omit them)
    if min_question_cosine is None:
        min_question_cosine = 0.30
    if hedge_terms is None:
        hedge_terms = [
            r"\brumor(s|ed)?\b",
            r"\breportedly\b",
            r"\bmay\b",
            r"\bmight\b",
            r"\bpossibly\b",
            r"\bsuggest(ed|s|ing)?\b",
            r"\baccording to (sources|rumors)\b",
        ]
    if reliability_weights is None:
        reliability_weights = {
            "news": 1.00,
            "press": 0.95,
            "blog": 0.60,
            "seed": 0.80,
            "alt": 0.75,
        }

    picked = pick_top_sentences(
        question,
        initial_cands,
        K=K,
        min_question_cosine=min_question_cosine,
        hedge_terms=hedge_terms,
        reliability_weights=reliability_weights,
    )
    if not picked:
        return "", []

    draft = " ".join(s.text.strip() for s in picked)
    claims: List[Claim] = []
    for i, s in enumerate(picked, start=1):
        claims.append(Claim(claim_id=f"c{i}", text=s.text.strip(), is_critical=True))
    return draft, claims
