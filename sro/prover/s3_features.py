"""
S3 — Cheap pair features for (i, j).

Features (all in [0,1] where possible):
  - max_p1: max(p1[i], p1[j])
  - sum_p1: p1[i] + p1[j] (clamped to 1 for stability when used directly)
  - ce_max: max(ce_score[i], ce_score[j]) normalized to [0,1] via min-max over the pool
  - entity_overlap: simple proxy using capitalized tokens + numbers overlap ratio
  - time_agreement: 1 if years match, 0 if conflict, 0.5 if no years present
  - distance: Jaccard distance (1 - Jaccard similarity) over tokens
  - section_novelty: 1 if different source_id else 0

Edge cases:
  - If no min-max range for ce_score (all equal), normalize to 0.5 to avoid division by zero.
  - If i == j, skip (caller should avoid generating i==j pairs).

Performance:
  The pair space is at most M*L (≤ 192 by default), so this is trivial.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import re

from sro.types import SentenceCandidate

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")  # naive year extractor


def _capitalized_tokens(tokens: List[str]) -> set:
    return set(t for t in tokens if t[:1].isalpha() and t[:1].upper() == t[:1])


def _numbers(tokens: List[str]) -> set:
    return set(t for t in tokens if t.isdigit())


def _entity_overlap(claim_tokens: List[str], sent_tokens_i: List[str], sent_tokens_j: List[str]) -> float:
    # crude: union of capitalized + numbers for each sentence vs claim entities
    claim_ents = _capitalized_tokens(claim_tokens) | _numbers(claim_tokens)
    i_ents = _capitalized_tokens(sent_tokens_i) | _numbers(sent_tokens_i)
    j_ents = _capitalized_tokens(sent_tokens_j) | _numbers(sent_tokens_j)
    if not claim_ents:
        return 0.0
    inter = len(claim_ents & (i_ents | j_ents))
    return inter / max(1, len(claim_ents))


def _years(tokens: List[str]) -> List[str]:
    return _YEAR_RE.findall(" ".join(tokens))  # [(19|20), yy] pairs; we only care presence


def _time_agreement(tokens_i: List[str], tokens_j: List[str]) -> float:
    years_i = set(re.findall(_YEAR_RE, " ".join(tokens_i)))
    years_j = set(re.findall(_YEAR_RE, " ".join(tokens_j)))
    if not years_i and not years_j:
        return 0.5  # neutral when no explicit time info
    if years_i & years_j:
        return 1.0
    return 0.0


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_pair_features(
    claim_tokens: List[str],
    candidates: List[SentenceCandidate],
    tokens_by_id: Dict[str, set],
    idx_frontier: List[int],
    idx_pool2: List[int],
    p1: List[float],
) -> Tuple[List[Tuple[int, int]], List[Dict[str, float]]]:
    """
    Returns:
      pairs: list of (i, j) with i from frontier, j from pool2 and i != j
      feats: list of dict feature_name -> float aligned with pairs
    """
    n = len(candidates)
    # Normalize ce_score to [0,1] across selected indices only for stability
    sel = list({*idx_frontier, *idx_pool2})
    ce_vals = [candidates[i].ce_score for i in sel]
    ce_min = min(ce_vals) if ce_vals else 0.0
    ce_max = max(ce_vals) if ce_vals else 1.0
    ce_den = (ce_max - ce_min) or 1.0

    pairs: List[Tuple[int, int]] = []
    feats: List[Dict[str, float]] = []

    for i in idx_frontier:
        for j in idx_pool2:
            if i == j:
                continue
            si = candidates[i]
            sj = candidates[j]
            ti = tokens_by_id[si.sent_id]
            tj = tokens_by_id[sj.sent_id]

            # Features
            max_p1 = max(p1[i], p1[j])
            sum_p1 = min(1.0, p1[i] + p1[j])
            ce_norm = (max(si.ce_score, sj.ce_score) - ce_min) / ce_den if ce_den else 0.5
            ent_ov = _entity_overlap(claim_tokens, list(ti), list(tj))
            time_ag = _time_agreement(list(ti), list(tj))
            sim = _jaccard(set(ti), set(tj))
            distance = 1.0 - sim
            section_novelty = 1.0 if si.source_id != sj.source_id else 0.0

            pairs.append((i, j))
            feats.append({
                "max_p1": float(max_p1),
                "sum_p1": float(sum_p1),
                "ce_max": float(ce_norm),
                "entity_overlap": float(ent_ov),
                "time_agreement": float(time_ag),
                "distance": float(distance),
                "section_novelty": float(section_novelty),
            })

    return pairs, feats
