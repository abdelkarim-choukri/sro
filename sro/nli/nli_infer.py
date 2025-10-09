"""
Real NLI inference (MNLI) for one-hop and two-hop scoring.

- one_hop_scores: p_entail and p_contradict for (sentence -> claim)
- two_hop_scores: p_entail for ((sent_i + sent_j) -> claim)

Design:
- Uses HuggingFace Transformers.
- Loads once per process; reuses the same model/tokenizer (singleton).
- Auto-selects CUDA if available, else CPU.
- Robust to label id ordering (reads model.config.label2id).
- Batching with truncation to max_length=512.
- Errors are raised with clear messages (no silent failures).

Model default:
- "roberta-large-mnli" (strong baseline). Override via env var SRO_NLI_MODEL.

Notes:
- two-hop "premise" is the concatenation of the two sentences with a separator.
- For speed: keep batch_size in [16, 32] depending on GPU memory.
"""

# sro/nli/nli_infer.py
from __future__ import annotations
import os
from typing import Iterable, List, Sequence, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_DEFAULT_MODEL = os.getenv("SRO_NLI_MODEL", "roberta-large-mnli")
_CACHE_DIR = os.getenv("SRO_CACHE_DIR", "models_cache")  # local HuggingFace cache dir
_MAX_LEN = 512

class _NLIBackend:
    """Internal singleton that holds model & tokenizer and exposes batched scoring."""
    _instance: Optional["_NLIBackend"] = None

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=_CACHE_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=_CACHE_DIR)
        except Exception as e:
            raise RuntimeError(f"Failed to load NLI model '{model_name}': {e}") from e

        self.model.to(self.device)
        self.model.eval()

        # Robust label mapping
        label2id = getattr(self.model.config, "label2id", None) or {}
        self._id_ent = None
        self._id_contra = None
        for k, v in label2id.items():
            kl = str(k).lower()
            if "entail" in kl:
                self._id_ent = int(v)
            elif "contrad" in kl:
                self._id_contra = int(v)

        # Fallback to common MNLI order [contradiction, neutral, entailment]
        if self._id_ent is None or self._id_contra is None:
            num_labels = int(getattr(self.model.config, "num_labels", 3))
            if num_labels == 3:
                self._id_contra = 0 if self._id_contra is None else self._id_contra
                self._id_ent = 2 if self._id_ent is None else self._id_ent
            else:
                raise RuntimeError(
                    f"Cannot infer entail/contradiction label ids (num_labels={num_labels}). "
                    f"Set SRO_NLI_MODEL to an MNLI-compatible head."
                )

    @classmethod
    def get(cls) -> "_NLIBackend":
        if cls._instance is None:
            cls._instance = _NLIBackend(_DEFAULT_MODEL)
        return cls._instance

    @torch.inference_mode()
    def score_pairs(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = 16,
    ) -> Tuple[List[float], List[float]]:
        """Return (p_entail, p_contradict) for each (premise, hypothesis)."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have same length")

        p_entail: List[float] = []
        p_contra: List[float] = []

        for start in range(0, len(premises), batch_size):
            end = min(len(premises), start + batch_size)
            enc = self.tokenizer(
                list(premises[start:end]),
                list(hypotheses[start:end]),
                truncation=True,
                padding=True,
                max_length=_MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # [B, num_labels]
            probs = torch.softmax(logits, dim=-1)

            p_entail.append(probs[:, self._id_ent].detach().float().cpu())
            p_contra.append(probs[:, self._id_contra].detach().float().cpu())

        if p_entail:
            p_entail = torch.cat(p_entail).tolist()
            p_contra = torch.cat(p_contra).tolist()
        else:
            p_entail, p_contra = [], []

        # Clamp defensively
        p_entail = [float(min(1.0, max(0.0, x))) for x in p_entail]
        p_contra = [float(min(1.0, max(0.0, x))) for x in p_contra]
        return p_entail, p_contra

# ---------- Public API ----------

def one_hop_scores(claim: str, sentences: Sequence[str], batch_size: int = 16) -> Tuple[List[float], List[float]]:
    """Per sentence: p1 (entail) and c1 (contradict) for (sentence -> claim)."""
    if not sentences:
        return [], []
    be = _NLIBackend.get()
    prem = list(sentences)
    hyp = [claim] * len(prem)
    p_ent, p_contra = be.score_pairs(prem, hyp, batch_size=batch_size)
    return p_ent, p_contra

def two_hop_scores(
    claim: str,
    pairs: Sequence[Tuple[str, str]],
    sep: str = " [SEP] ",
    batch_size: int = 16,
) -> List[float]:
    """
    For each pair (sent_i, sent_j), compute p2 entail for ((sent_i + SEP + sent_j) -> claim).
    """
    if not pairs:
        return []
    be = _NLIBackend.get()
    prem = [f"{a}{sep}{b}" for (a, b) in pairs]
    hyp = [claim] * len(prem)
    p_ent, _ = be.score_pairs(prem, hyp, batch_size=batch_size)
    return p_ent
