from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import math
import re
import numpy as np

from sro.claims.tokenizer import tokenize, Token
from sro.claims.rules import has_verb

PRONOUNS = {"i","we","you","he","she","they","it","this","that","these","those"}
AUX = {"am","is","are","was","were","be","been","being","has","have","had","can","could","will","would","should","may","might","must","do","does","did"}

def _inside_prt(idx: int, prt: List[Tuple[int,int]]) -> bool:
    return any(a <= idx < b for a,b in prt)

def _tokens_in_span(tokens: List[Token], a: int, b: int) -> List[Token]:
    return [t for t in tokens if not (t.end <= a or t.start >= b)]

def _starts_with_pronoun(txt: str) -> bool:
    parts = txt.strip().split()
    return bool(parts) and parts[0].lower() in PRONOUNS

def _finite_verb_heuristic(tokens: List[Token]) -> int:
    # crude finite count: words ending in s/ed and common finite auxiliaries
    lc = [t.text.lower() for t in tokens]
    count = 0
    for w in lc:
        if w in AUX:
            count += 1
        elif len(w) >= 3 and (w.endswith("ed") or w.endswith("es") or (w.endswith("s") and len(w) > 3)):
            count += 1
    # discount obvious infinitive "to X"
    for i in range(len(lc)-1):
        if lc[i] == "to" and lc[i+1].isalpha():
            count = max(0, count-1)
    return max(0, count)

def _subject_clone(lhs: str, rhs: str) -> str:
    # If RHS starts with AUX (optionally after one punctuation), clone up to 6 tokens from LHS up to first verb.
    rhs_stripped = rhs.lstrip()
    leading = rhs_stripped.split()
    if not leading:
        return rhs
    first = leading[0].lower().strip(".,;:!?")
    if first not in AUX:
        return rhs

    # LHS head: take up to 6 tokens before first verb-like token
    lhs_toks = tokenize(lhs)
    cut = len(lhs_toks)
    for i,t in enumerate(lhs_toks):
        w = t.text.lower()
        if w in AUX or w.endswith("ed") or w.endswith("ing") or (w.endswith("s") and len(w) > 3):
            cut = i
            break
    prefix = " ".join(tt.text for tt in lhs_toks[:min(cut,6)]).strip()
    if not prefix:
        return rhs

    # POS proxy: Accept if prefix starts with determiner, TitleCase, or pronoun
    head = lhs_toks[0].text
    if not (head.lower() in {"the","a","an"} or head[0].isupper() or head.lower() in PRONOUNS):
        return rhs

    # Avoid duplicating punctuation spacing
    return (prefix + " " + rhs_stripped).strip()

def _char_index_for_tok(i_tok: int, tokens: List[Token]) -> int:
    return tokens[i_tok].end

def _tok_index_for_char(char_idx: int, tokens: List[Token]) -> int:
    for i,t in enumerate(tokens):
        if t.end == char_idx:
            return i
    # fallback (shouldn't happen with our pipeline)
    for i in range(len(tokens)-1, -1, -1):
        if tokens[i].end < char_idx:
            return i
    return 0

def _in_prt_tok(i_tok: int, tokens: List[Token], prt: List[Tuple[int,int]]) -> bool:
    c = tokens[i_tok].end
    return any(a <= c < b for a,b in prt)

def _enforce_spacing(chars: List[int], tokens: List[Token], min_gap: int) -> List[int]:
    if not chars:
        return []
    def ti(c):
        for i,t in enumerate(tokens):
            if t.end == c:
                return i
        return 0
    out = [chars[0]]
    last = ti(chars[0])
    for c in chars[1:]:
        t = ti(c)
        if t - last >= min_gap:
            out.append(c); last = t
    return out


def arbitrate_splits(tokens: List[Token], prt_spans: List[Tuple[int,int]],
                     rule_bnds: List[int], model_bnds_info: Optional[Dict[str,object]], cfg) -> Tuple[List[int], Dict[str,int]]:
    """
    Merge rule and model boundaries. model_bnds_info may be empty or None for rules-only.

    Returns (kept_chars, stats) where stats contains num_model_add and num_model_remove.
    """
    min_gap = getattr(cfg.splitter, "min_gap_tokens", 3)
    # model thresholds (nested model section may be missing)
    add_thr = 0.0
    rm_margin = 1e9
    try:
        add_thr = float(getattr(cfg.splitter.model, "add_prob", 0.70))
        rm_margin = float(getattr(cfg.splitter.model, "remove_margin", 0.35))
    except Exception:
        pass

    # 1) Start from rule set; drop any inside PRT (defensive)
    base = sorted({c for c in rule_bnds if not any(a <= c < b for a,b in prt_spans)})

    if not model_bnds_info:
        merged = _enforce_spacing(base, tokens, min_gap)
        return merged, {"num_model_add": 0, "num_model_remove": 0}

    probs = model_bnds_info.get("probs", None)
    logits = model_bnds_info.get("logits", None)
    model_tok = model_bnds_info.get("tok_indices", [])

    add_chars: List[int] = []
    # track candidates that met the probability threshold but were rejected for reasons
    rej_reasons: List[Dict[str, object]] = []  # list of {"char": int, "reasons": [str,...]}
    candidate_add_chars: List[int] = []
    for i_tok in model_tok:
        if probs is not None and probs[i_tok] >= add_thr and not _in_prt_tok(i_tok, tokens, prt_spans):
            add_chars.append(_char_index_for_tok(i_tok, tokens))

        # record 'near_prt' for candidates that meet prob but lie inside/adjacent to PRT
        if probs is not None and probs[i_tok] >= add_thr and _in_prt_tok(i_tok, tokens, prt_spans):
            c = _char_index_for_tok(i_tok, tokens)
            rej_reasons.append({"char": c, "reasons": ["near_prt"]})
        # keep track of candidates that met prob threshold (for later min_gap reasoning)
        if probs is not None and probs[i_tok] >= add_thr:
            candidate_add_chars.append(_char_index_for_tok(i_tok, tokens))

    rm_chars = set()
    if logits is not None:
        # softmax on 5 labels
        maxlog = np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(logits - maxlog)
        probs_all = exps / np.sum(exps, axis=-1, keepdims=True)
        # For rule-removal we compute Δ = log P(O) − log P(BND) at the rule token; if Δ >= rm_margin, remove rule.
        for c in base:
            i_tok = _tok_index_for_char(c, tokens)
            delta = float(np.log(probs_all[i_tok, 4] + 1e-9) - np.log(probs_all[i_tok, 3] + 1e-9))
            if delta >= rm_margin:
                rm_chars.add(c)

    pre_spaced = sorted(set([c for c in base if c not in rm_chars] + add_chars))
    merged = _enforce_spacing(pre_spaced, tokens, min_gap)

    # any candidate_add_chars that were removed by spacing get a 'min_gap' reason
    for c in candidate_add_chars:
        if c not in merged and c not in [r["char"] for r in rej_reasons]:
            rej_reasons.append({"char": c, "reasons": ["min_gap"]})

    # compute stats and list of model-originating adds that survived
    model_add_chars = [c for c in merged if c in add_chars]
    stats = {
        "num_model_add": sum(1 for c in model_add_chars),
        "num_model_remove": sum(1 for c in base if c in rm_chars),
        "model_add_chars": model_add_chars,
        "rej_reasons": rej_reasons,
    }
    return merged, stats

def enforce_filters(q: str, claims: List[Dict], cfg) -> List[Dict]:
    max_len = getattr(cfg.splitter, "max_len_tokens", 25)
    ban_pron = getattr(cfg.splitter, "ban_pronouns", True)
    out = []
    for cl in claims:
        text = cl["text"].strip()
        if not text:
            continue
        toks = tokenize(text)
        if len(toks) > max_len:
            continue
        if ban_pron and _starts_with_pronoun(text):
            continue
        # single predicate heuristic
        if _finite_verb_heuristic(toks) > 1:
            continue
        cl["text"] = text
        out.append(cl)
    return out

def _token_set(s: str) -> set:
    return {t.text.lower() for t in tokenize(s) if t.text.isalpha()}

def select_top_claims(q: str, claims: List[Dict], cfg) -> List[Dict]:
    if not claims:
        return []
    # Delegate scoring to sro.claims.scoring to keep logic compact and testable.
    from sro.claims import scoring

    scored = []
    for i, cl in enumerate(claims):
        s = float(scoring.final_score(q, i, claims))
        claims[i]["score"] = s
        scored.append((i, s))

    k = getattr(cfg.splitter, "max_claims", 5)
    # stable selection: sort indices by score desc preserving left-to-right tie-break
    keep_idxs = sorted(range(len(claims)), key=lambda i: claims[i]["score"], reverse=True)[:k]
    return [claims[i] for i in keep_idxs]

def maybe_subject_clone(claims: List[Dict], tokens: List[Token], cfg) -> List[Dict]:
    """
    Optionally create an augmented display text that prepends a short subject
    prefix from the previous claim when the RHS starts with an auxiliary.

    Important: do NOT mutate the canonical `text`, `start`, or `end` fields so
    that downstream invariants (which expect `text` to be equal to
    q[start:end].strip()) hold. Instead, place the augmented form in
    `display_text` on the claim when applicable.
    """
    if not claims:
        return claims
    out = [claims[0]]
    for i in range(1, len(claims)):
        lhs = out[-1]["text"]
        rhs = claims[i]["text"]
        new_rhs = _subject_clone(lhs, rhs)
        new = dict(claims[i])
        # Keep canonical text unchanged; expose augmented text only as display_text
        if new_rhs != rhs:
            new["display_text"] = new_rhs
        out.append(new)
    return out
