from __future__ import annotations
from typing import Dict, List, Tuple, Set, Sequence, Optional
import math
import re

from sro.types import SentenceCandidate

# ----------------------------
# Lightweight token helpers
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _tokens(s: str) -> Set[str]:
    """Return a set of lowercase alphanumeric tokens."""
    if not isinstance(s, str):
        return set()
    return set(t.lower() for t in _WORD_RE.findall(s))

def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity ∈[0,1]."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _clean01(x: float) -> float:
    """Clamp to [0,1]; NaN/±inf → 0.0."""
    if x is None or not (x == x) or math.isinf(x):
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _numbers(tokens: Sequence[str]) -> Set[str]:
    """Numeric tokens (simple proxy for entities like years/dates)."""
    return {t for t in tokens if t.isdigit()}

def _years(tokens: Sequence[str]) -> Set[str]:
    """4-digit years as strings (19xx or 20xx)."""
    joined = " ".join(tokens)
    return set(m.group(0) for m in _YEAR_RE.finditer(joined))

# ----------------------------
# Main: build pair features
# ----------------------------
def build_pair_features(
    claim_tokens: Sequence[str],
    candidates: List[SentenceCandidate],
    token_cache: Dict[str, Set[str]],
    idx_frontier: List[int],
    idx_pool2: List[int],
    p1: List[float],
    *,
    # New knobs (optional, default keeps old behavior):
    novelty_gate: float = 0.90,        # drop a pair if either leaf is ≥ this sim to frontier
    hi_sim_thresh: float = 0.20,       # if distance < this, raise a similarity_penalty
) -> Tuple[List[Tuple[int, int]], List[Dict[str, float]]]:
    """
    Build one cheap feature vector per candidate pair (i in frontier, j in pool2, i!=j).

    Variables:
      novelty_gate: float in [0,1], min novelty required vs frontier; if either leaf is
                    too similar to the *existing frontier* (max Jaccard ≥ novelty_gate),
                    we skip that pair entirely (prevents redundant 2-hop).
      hi_sim_thresh: distance (=1-Jaccard(i,j)) threshold under which we set
                    'similarity_penalty' ∈ [0,1] (more penalty for higher similarity).

    Inputs:
      claim_tokens: tokens of the claim (list[str]); lowercase recommended.
      candidates:   SentenceCandidate list (has sent_id, text, source_id, ce_score).
      token_cache:  {sent_id -> Set[str]} from S2 (reused to avoid re-tokenizing).
      idx_frontier: indices of frontier sentences (first-hop).
      idx_pool2:    indices for second-hop pool (disjoint from frontier).
      p1:           one-hop entailment probs per candidate (same order as candidates).

    Outputs:
      pairs: list of (i, j) indices.
      feats: list of feature dicts aligned with 'pairs' (UB uses: max_p1, sum_p1,
             ce_max, entity_overlap, time_agreement, distance, section_novelty;
             we also add 'similarity_penalty' which UB may start consuming).
    """
    n = len(candidates)
    if len(p1) != n:
        raise ValueError("p1 length must match number of candidates")

    # Clean p1 once (safety)
    p1c = [_clean01(x) for x in p1]

    # Build quick access arrays
    tok_by_idx: List[Set[str]] = []
    for i, c in enumerate(candidates):
        tok = token_cache.get(c.sent_id)
        if tok is None:
            tok = _tokens(c.text)  # rare path; keeps function robust
            token_cache[c.sent_id] = tok
        tok_by_idx.append(tok)

    # Precompute frontier token sets and per-candidate max sim to frontier
    frontier_toks = [tok_by_idx[i] for i in idx_frontier if 0 <= i < n]
    max_sim_to_frontier = [0.0] * n
    if frontier_toks:
        for i in range(n):
            sims = ( _jaccard(tok_by_idx[i], ft) for ft in frontier_toks )
            max_sim_to_frontier[i] = max(sims, default=0.0)

    # Claim entity proxy (numbers/years)
    claim_tok_list = list(claim_tokens)
    claim_nums = _numbers(claim_tok_list)
    # We keep years for potential future features; time_agreement below uses years(i) ∩ years(j).
    # claim_years = _years(claim_tok_list)

    def _entity_overlap(i: int, j: int) -> float:
        """Overlap of numeric tokens between claim and union of i/j."""
        if not claim_nums:
            return 0.0
        nums_i = _numbers(tok_by_idx[i])
        nums_j = _numbers(tok_by_idx[j])
        inter = len(claim_nums & (nums_i | nums_j))
        return inter / max(1, len(claim_nums))

    def _time_agreement(i: int, j: int) -> float:
        """1.0 if i/j mention at least one common year; 0.5 if neither has years; else 0.0."""
        yi = _years(tok_by_idx[i])
        yj = _years(tok_by_idx[j])
        if not yi and not yj:
            return 0.5
        return 1.0 if (yi & yj) else 0.0

    def _distance(i: int, j: int) -> float:
        """Distance = 1 - Jaccard(i, j) ∈ [0,1]. Larger is better (more novel)."""
        return 1.0 - _jaccard(tok_by_idx[i], tok_by_idx[j])

    def _section_novelty(i: int, j: int) -> float:
        """1.0 if different source_id; else 0.0 (cheap cross-source novelty proxy)."""
        si = candidates[i].source_id or ""
        sj = candidates[j].source_id or ""
        return 1.0 if si != sj else 0.0

    def _similarity_penalty(dist: float) -> float:
        """
        Penalty in [0,1], larger when pair is too similar (low 'dist').
        Linear ramp: dist < hi_sim_thresh → penalty = (hi_sim_thresh - dist)/hi_sim_thresh.
        """
        thr = max(1e-6, float(hi_sim_thresh))
        if dist >= thr:
            return 0.0
        return (thr - dist) / thr

    pairs: List[Tuple[int, int]] = []
    feats: List[Dict[str, float]] = []

    # Build pairs (i in frontier, j in pool2, i != j) with gating and features
    # Novelty gate: if either leaf is too similar to the frontier, skip the pair early.
    gate = float(novelty_gate)
    for i in idx_frontier:
        if not (0 <= i < n):
            continue
        # If the frontier sim is extremely high for i, it still passes (i is the frontier).
        for j in idx_pool2:
            if not (0 <= j < n) or j == i:
                continue
            if max_sim_to_frontier[j] >= gate:
                continue  # j is redundant w.r.t. frontier; skip

            # Core features (kept stable for UB)
            max_p1 = max(p1c[i], p1c[j])
            sum_p1 = min(1.0, p1c[i] + p1c[j])  # soft cap
            ce_i = candidates[i].ce_score or 0.0
            ce_j = candidates[j].ce_score or 0.0
            # If ce_scores are not normalized, clamp to [0,1] defensively.
            ce_max = max(_clean01(ce_i), _clean01(ce_j))

            ent_ov = _entity_overlap(i, j)
            time_ag = _time_agreement(i, j)
            dist = _distance(i, j)
            sect_nov = _section_novelty(i, j)
            sim_pen = _similarity_penalty(dist)

            pairs.append((i, j))
            feats.append({
                "max_p1": float(max_p1),
                "sum_p1": float(sum_p1),
                "ce_max": float(ce_max),
                "entity_overlap": float(ent_ov),
                "time_agreement": float(time_ag),
                "distance": float(dist),
                "section_novelty": float(sect_nov),
                # New: similarity penalty (not used by UB yet, but useful for inspection/tuning)
                "similarity_penalty": float(sim_pen),
            })

    return pairs, feats
