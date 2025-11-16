from __future__ import annotations
from typing import List, Dict
from sro.claims.tokenizer import tokenize

_STOP = {"the","a","an","of","and","or","to","in","on","for","with","by","at","as","is","are","was","were","it","this","that"}

def coverage(q: str, text: str) -> float:
    return min(1.0, len(text)/max(1,len(q)))

def specificity(text: str) -> float:
    words = [t.text.lower() for t in tokenize(text) if t.text.isalpha()]
    if not words: return 0.0
    non_stop = [w for w in words if w not in _STOP]
    return 1.0 - (len(words)-len(non_stop))/len(words)

def independence(i: int, claims: List[Dict]) -> float:
    def toks(s): return {t.text.lower() for t in tokenize(s) if t.text.isalpha() and t.text.lower() not in _STOP}
    this = toks(claims[i]["text"]) if claims else set()
    others = set()
    for j,c in enumerate(claims):
        if j==i: continue
        others |= toks(c["text"])
    if not this and not others: return 0.0
    inter = len(this & others); uni = len(this | others) or 1
    return 1.0 - inter/uni

def lexical_novelty(text: str) -> float:
    words = [t.text.lower() for t in tokenize(text) if t.text.isalpha()]
    if not words: return 0.0
    uniq = set(words)
    content = [w for w in words if w not in _STOP]
    return len(set(content))/len(uniq) if uniq else 0.0

def final_score(q: str, i: int, claims: List[Dict]) -> float:
    c = coverage(q, claims[i]["text"])
    s = specificity(claims[i]["text"])
    ind = independence(i, claims)
    nov = lexical_novelty(claims[i]["text"])
    return 0.45*c + 0.25*s + 0.20*ind + 0.10*nov
