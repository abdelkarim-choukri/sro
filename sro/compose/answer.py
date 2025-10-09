# sro/compose/answer.py
from __future__ import annotations
from typing import List, Dict, Tuple, Set
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sro.utils.st import get_st
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokens(s: str) -> Set[str]:
    return set(t.lower() for t in _WORD_RE.findall(s or ""))

def _st():
    
    return get_st("sentence-transformers/all-MiniLM-L6-v2")

def _embed(texts):
    return _st().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip((a * b).sum(), -1.0, 1.0))

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

def _marker(i: int) -> str:
    # 1-based superscript-like: [¹], [²], ...
    SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    digits = list(str(i))
    sup = "".join(SUP[int(d)] for d in digits)
    return f"[{sup}]"

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
      accepted: list of dicts with keys: 'text', 'score', 'citations' (list of {'source_id','sent_id'}).
    Outputs:
      final_answer (str), refs (list of (marker, source_id))
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

    # 2) deduplicate by cosine
    nondup_idx: List[int] = []
    for i in range(len(accepted)):
        if not nondup_idx:
            nondup_idx.append(i); continue
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
            if len(picked_idx) >= N: break
            src = _primary_src(accepted[i])
            if src and src in used_src:
                continue
            picked_idx.append(i)
            if src:
                used_src.add(src)
        # second pass: fill remaining slots even if sources repeat
        if len(picked_idx) < N:
            for i in nondup_idx:
                if len(picked_idx) >= N: break
                if i in picked_idx: continue
                picked_idx.append(i)
    else:
        picked_idx = nondup_idx[:max(N, 1)]

    kept = [accepted[i] for i in picked_idx]

    # 4) Build global markers over unique source_ids from kept claims
    all_cites: List[Dict[str, str]] = []
    for k in kept:
        all_cites.extend(k.get("citations") or [])
    uniq_srcs = _uniq_sources(all_cites)
    src_to_marker: Dict[str, str] = {src: _marker(i+1) for i, src in enumerate(uniq_srcs)}

    # Attach markers to text and compact to ≤N sentences
    parts: List[str] = []
    prev_vec = None
    for k in kept:
        t = k["text"].strip()
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
