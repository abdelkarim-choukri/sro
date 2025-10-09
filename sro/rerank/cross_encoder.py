# sro/rerank/cross_encoder.py
from __future__ import annotations
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sro.utils.hf import ensure_repo_local
from transformers import AutoTokenizer, AutoModelForSequenceClassification
class CrossEncoderReranker:
    """
    Simple cross-encoder reranker. Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (default).
    score_pairs(queries, passages) -> List[float] (higher is better).
    """
    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        local = ensure_repo_local(model_id, local_dir="models_cache")
        self.tok = AutoTokenizer.from_pretrained(local, cache_dir="models_cache")
        self.model = AutoModelForSequenceClassification.from_pretrained(local, cache_dir="models_cache")
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, queries: List[str], passages: List[str], batch_size: int = 32) -> List[float]:
        assert len(queries) == len(passages)
        out: List[float] = []
        for i in range(0, len(queries), batch_size):
            q = queries[i:i+batch_size]
            p = passages[i:i+batch_size]
            enc = self.tokenizer(q, p, padding=True, truncation=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            scores = logits.detach().float().cpu().tolist()
            # Normalize to [0,1] via sigmoid to be comparable; higher still better
            if isinstance(scores, float):
                scores = [scores]
            # map to 0..1
            if hasattr(torch, "sigmoid"):
                with torch.no_grad():
                    sig = torch.sigmoid(torch.tensor(scores)).tolist()
                out.extend(sig)
            else:
                out.extend(scores)
        return out
