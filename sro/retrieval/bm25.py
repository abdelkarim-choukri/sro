# sro/retrieval/bm25.py
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Dict, List


class BM25OkapiLite:
    """
    Minimal BM25-Okapi for sentences.
    Tokens are pre-tokenized lists of strings per sentence.

    Variables:
      k1: positive float (default 1.5) – BM25 term saturation.
      b:  float in [0,1] (default 0.75) – length normalization.
    """
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.doc_len = [0] * self.N
        self.avgdl = 0.0

        for i, toks in enumerate(corpus_tokens):
            self.doc_len[i] = len(toks)
            seen = set()
            for t in toks:
                if t not in seen:
                    self.doc_freq[t] += 1
                    seen.add(t)
        self.avgdl = sum(self.doc_len) / max(1, self.N)

        # Precompute IDF
        self.idf: dict[str, float] = {}
        for term, df in self.doc_freq.items():
            # BM25 idf with +0.5 smoothing
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _score_doc(self, qtf: Counter, idx: int) -> float:
        dl = self.doc_len[idx]
        denom = self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self.avgdl)))
        score = 0.0
        # term-frequency in doc
        tf_doc = Counter(self.corpus_tokens[idx])
        for term, qcnt in qtf.items():
            if term not in self.idf:
                continue
            tf = tf_doc.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf[term]
            # Okapi term
            num = tf * (self.k1 + 1.0)
            sc = idf * (num / (tf + denom))
            # Weight by query term count (rarely >1 for short queries)
            score += sc * qcnt
        return score

    def get_scores(self, query_tokens: Iterable[str]) -> list[float]:
        qtf = Counter(list(query_tokens))
        return [self._score_doc(qtf, i) for i in range(self.N)]
