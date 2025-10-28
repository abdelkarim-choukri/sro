from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sro.utils.hf import ensure_repo_local


class CrossEncoderReranker:
    """
    Cross-encoder reranker (Roberta/BERT-style).
    - Loads from local cache (via ensure_repo_local) to work offline.
    - GPU if available, else CPU.
    - score_pairs returns a float score per (query, passage) pair.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir: str = "models_cache",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name  # str: HF repo id or local dir
        self.cache_dir = cache_dir    # str: local cache root
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure repo is cached locally (works even in HF offline mode)
        local_dir = ensure_repo_local(self.model_name, local_dir=self.cache_dir)

        # Load tokenizer+model and move to device
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> list[float]:
        """
        Inputs:
          queries: list[str] (len N) or a single str (broadcasted)
          passages: list[str] (len N)
        Output:
          scores: list[float] length N (higher = more relevant)
        """
        # Broadcast single query
        if isinstance(queries, str):
            queries = [queries] * len(passages)
        if len(queries) != len(passages):
            raise ValueError(f"queries (len={len(queries)}) and passages (len={len(passages)}) must have same length")

        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            end = min(len(passages), start + batch_size)
            q = list(queries[start:end])
            p = list(passages[start:end])

            enc = self.tokenizer(
                q, p,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

            logits = self.model(**enc).logits  # [B, D]
            # Robust extraction: if D==1 treat as regression head; else take "positive" logit
            if logits.size(-1) == 1:
                s = logits.squeeze(-1)
            else:
                s = logits[:, -1]
            scores.extend(s.detach().float().cpu().tolist())

        return scores
