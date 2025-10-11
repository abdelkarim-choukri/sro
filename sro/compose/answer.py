# sro/compose/answer.py
from __future__ import annotations
from typing import List, Dict, Tuple, Set
import re
import numpy as np
from sro.utils.st import get_st

# ----------------------------
# Lightweight token helpers
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokens(s: str) -> Set[str]:
    return set(t.lower() for t in _WORD_RE.findall(s or ""))

# ----------------------------
# ASCII citation markers
# ----------------------------
def _idx_to_mark(i: int) -> str:
    """Return an ASCII marker like '[1]' for 0-based index i."""
    return f"[{i+1}]"

# ----------------------------
# Sentence-Transformer (cached)
# ----------------------------
def _st():
    # Offline/cache-friendly getter; model is already warmed per project setup.
    return get_st("sentence-transformers/all-MiniLM-L6-v2")

def _embed(texts: List[str]) -> np.ndarray:
    return _st().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip((a * b).sum(), -1.0, 1.0))

# ----------------------------
# Citation utilities
# ----------------------------
def _uniq_sources(citations: List[Dict[str, str]]) -> List[str]:
    """Return unique source_ids in first-appearance order."""
    seen: Set[str] = set()
    out: List[str] = []
    for c in citations or []:
        sid = c.get("source_id") or ""
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out

# ----------------------------
# Composer
# ----------------------------
def compose_answer_with_citations(
    accepted: List[Dict],
    *,
    N: int = 2,               # max sentences in final answer
    theta_dup: float = 0.90,  # θ — duplicate threshold on cosine
    psi_concat: float = 0.40, # ψᶜ — min cosine to join claims in one sentence
    enforce_source_diversity: bool = True,  # prefer different sources
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Inputs:
      accepted: list of dicts with keys:
        'text': str, 'score': float, 'citations': list of {'source_id','sent_id'}.
    Outputs:
      final_answer (str),
      refs (list of (marker, source_id)) where marker is ASCII like '[1]'
    Steps:
      1) sort claims by score desc,
      2) drop near-duplicates (cosine >= θ),
      3) select up to N with optional source diversity,
      4) compact to ≤N sentences with inline markers,
      5) build refs (marker -> source_id).
    """
    if not accepted:
        return "", []

    # 1) sort by score desc
    accepted = sorted(accepted, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    texts = [a["text"] for a in accepted]
    embs = _embed(texts)

    # 2) deduplicate by cosine (keep first in each near-duplicate cluster)
    nondup_idx: List[int] = []
    for i in range(len(accepted)):
        if not nondup_idx:
            nondup_idx.append(i)
            continue
        if any(_cos(embs[i], embs[j]) >= theta_dup for j in nondup_idx):
            continue
        nondup_idx.append(i)

    # Helper: primary source_id for a claim (first unique in its citations)
    def _primary_src(a: Dict) -> str:
        cites = a.get("citations") or []
        for c in cites:
            sid = c.get("source_id") or ""
            if sid:
                return sid
        return ""

    # 3) pick up to N with optional source diversity
    picked_idx: List[int] = []
    used_src: set[str] = set()
    if enforce_source_diversity:
        # first pass: only take claims introducing a new source
        for i in nondup_idx:
            if len(picked_idx) >= N:
                break
            src = _primary_src(accepted[i])
            if src and src in used_src:
                continue
            picked_idx.append(i)
            if src:
                used_src.add(src)
        # second pass: fill remaining slots even if sources repeat
        if len(picked_idx) < N:
            for i in nondup_idx:
                if len(picked_idx) >= N:
                    break
                if i in picked_idx:
                    continue
                picked_idx.append(i)
    else:
        picked_idx = nondup_idx[:max(N, 1)]

    kept = [accepted[i] for i in picked_idx]

    # 4) Build global markers over unique source_ids from kept claims (ASCII markers)
    all_cites: List[Dict[str, str]] = []
    for k in kept:
        all_cites.extend(k.get("citations") or [])
    uniq_srcs = _uniq_sources(all_cites)
    src_to_marker: Dict[str, str] = {src: _idx_to_mark(i) for i, src in enumerate(uniq_srcs)}

    # Attach markers to text and compact to ≤N sentences
    parts: List[str] = []
    prev_vec: np.ndarray | None = None
    for k in kept:
        t = (k.get("text") or "").strip()
        if not t:
            continue
        srcs = _uniq_sources(k.get("citations") or [])
        marks = "".join(src_to_marker[s] for s in srcs if s in src_to_marker)
        seg = f"{t} {marks}".strip()
        if not parts:
            parts.append(seg)
            prev_vec = _embed([t])[0]
            continue
        cur_vec = _embed([t])[0]
        if _cos(prev_vec, cur_vec) >= psi_concat:
            parts[-1] = (parts[-1] + " " + seg).strip()
        else:
            parts.append(seg)
        prev_vec = cur_vec

    parts = parts[:N]  # enforce final cap
    final_answer = " ".join(parts).strip()

    refs: List[Tuple[str, str]] = [(src_to_marker[s], s) for s in uniq_srcs]
    return final_answer, refs




from typing import List, Dict, Tuple, Set, Optional
import re
import numpy as np
from sro.types import SentenceCandidate, Claim
from sro.utils.st import get_st

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
def _tokens(s: str) -> Set[str]:
    return set(t.lower() for t in _WORD_RE.findall(s or ""))

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return (inter / union) if union else 0.0

def _source_prefix(source_id: str) -> str:
    return (source_id or "").split(":", 1)[0] if source_id else ""

class _ST:
    _m = None
    @classmethod
    def enc(cls, texts: List[str]) -> np.ndarray:
        if cls._m is None:
            cls._m = get_st("sentence-transformers/all-MiniLM-L6-v2")
        return cls._m.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip((a*b).sum(), -1.0, 1.0))

def _is_hedged(text: str, pats: List[re.Pattern]) -> bool:
    low = (text or "").lower()
    return any(p.search(low) for p in pats)

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
    if not cands:
        return []
    pats = [re.compile(h, re.I) for h in hedge_terms or []]
    qv = _ST.enc([question])[0]
    sv = _ST.enc([c.text for c in cands])

    scored: List[Tuple[int, float]] = []
    for i, s in enumerate(cands):
        if _is_hedged(s.text, pats): continue
        cos = _cos(qv, sv[i])
        if cos < min_question_cosine: continue
        w_src = reliability_weights.get(_source_prefix(s.source_id), 1.0)
        rel = (s.ce_score or 0.0) * float(w_src) * max(cos, 0.0)
        scored.append((i, rel))
    if not scored:
        return []
    scored.sort(key=lambda t: t[1], reverse=True)
    order = [i for (i, _) in scored]

    picked: List[SentenceCandidate] = []
    toks_list: List[Set[str]] = []
    for i in order:
        tok = _tokens(cands[i].text)
        if any(_jaccard(tok, t) > max_sim for t in toks_list):
            continue
        picked.append(cands[i])
        toks_list.append(tok)
        if len(picked) >= K:
            break
    return picked

def draft_and_claims(
    question: str,
    initial_cands: List[SentenceCandidate],
    *,
    K: Optional[int] = None,
    min_question_cosine: Optional[float] = None,
    hedge_terms: Optional[List[str]] = None,
    reliability_weights: Optional[Dict[str, float]] = None,
    max_claims: Optional[int] = None,
) -> Tuple[str, List[Claim]]:
    if K is None:
        K = max_claims if max_claims is not None else 3
    if min_question_cosine is None:
        min_question_cosine = 0.30
    if hedge_terms is None:
        hedge_terms = [
            r"\brumor(s|ed)?\b", r"\breportedly\b", r"\bmay\b", r"\bmight\b",
            r"\bpossibly\b", r"\bsuggest(ed|s|ing)?\b", r"\baccording to (sources|rumors)\b",
        ]
    if reliability_weights is None:
        reliability_weights = {"news":1.00,"press":0.95,"blog":0.60,"seed":0.80,"alt":0.75}

    picked = pick_top_sentences(
        question, initial_cands, K=K,
        min_question_cosine=min_question_cosine,
        hedge_terms=hedge_terms,
        reliability_weights=reliability_weights,
    )
    if not picked:
        return "", []
    draft = " ".join(s.text.strip() for s in picked)
    claims = [Claim(claim_id=f"c{i}", text=s.text.strip(), is_critical=True) for i, s in enumerate(picked, 1)]
    return draft, claims
