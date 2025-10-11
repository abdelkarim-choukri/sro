"""
Real NLI inference (MNLI) for one-hop and two-hop scoring.

- one_hop_scores: p_entail and p_contradict for (sentence -> claim)
- two_hop_scores: p_entail for ((sent_i + sent_j) -> claim)

Design:
- Uses HuggingFace Transformers.
- Loads once per process; reuses the same model/tokenizer (singleton).
- Auto-selects CUDA if available, else CPU (with per-batch OOM fallback to CPU).
- Robust to label id ordering (reads model.config.label2id/id2label).
- Batching with truncation to max_length=512.
- STRICTLY OFFLINE: loads from local cache only (no snapshot_download / no network).

Model default:
- "roberta-large-mnli" (strong baseline). Override via env var SRO_NLI_MODEL.

Notes:
- two-hop "premise" is the concatenation of the two sentences with a separator.
- For speed: keep batch_size in [16, 32] depending on GPU memory.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------
# Offline/cache discipline
# ------------------------------
# Your repo already holds the hub-style cache under models_cache/
# (e.g., models_cache/models--roberta-large-mnli/...)
# We force transformers to use *only* this local cache.
_CACHE_ROOT = str(Path("models_cache"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _CACHE_ROOT)
os.environ.setdefault("HF_HOME", _CACHE_ROOT)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_DEFAULT_MODEL = os.getenv("SRO_NLI_MODEL", "roberta-large-mnli")
_MAX_LEN = 512


class _NLIBackend:
    """Internal singleton that holds model & tokenizer and exposes batched scoring."""
    _instance: Optional["_NLIBackend"] = None

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._dev_pref = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build strictly from local cache; never hit the network; never snapshot_download.
        kw = dict(cache_dir=_CACHE_ROOT, local_files_only=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kw)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load NLI model '{model_name}' from local cache at {_CACHE_ROOT}. "
                f"Pre-warm the cache or set SRO_NLI_MODEL to an available local model. Details: {e}"
            ) from e

        # Move once to preferred device; we'll OOM-fallback per-batch if needed.
        self.model.to(self._dev_pref)
        self.model.eval()

        # Map label ids robustly
        self._id_ent, self._id_contra = self._resolve_label_ids(self.model)

    @staticmethod
    def _resolve_label_ids(model) -> Tuple[int, int]:
        # Try explicit label2id / id2label first
        ent_id = None
        contra_id = None
        lab2id = getattr(model.config, "label2id", None) or {}
        if lab2id:
            for k, v in lab2id.items():
                kl = str(k).lower()
                if "entail" in kl:
                    ent_id = int(v)
                elif "contrad" in kl:
                    contra_id = int(v)

        # Fallback: common MNLI ordering [contradiction, neutral, entailment]
        if ent_id is None or contra_id is None:
            num_labels = int(getattr(model.config, "num_labels", 3))
            if num_labels == 3:
                contra_id = 0 if contra_id is None else contra_id
                ent_id = 2 if ent_id is None else ent_id
            else:
                raise RuntimeError(
                    f"Cannot infer entail/contradiction label ids (num_labels={num_labels}). "
                    f"Use an MNLI-compatible head or set SRO_NLI_MODEL accordingly."
                )
        return ent_id, contra_id

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

            # Try on preferred device; if CUDA OOM, fallback to CPU for this batch.
            try:
                enc_dev = {k: v.to(self._dev_pref) for k, v in enc.items()}
                logits = self.model(**enc_dev).logits  # [B, num_labels]
            except RuntimeError as oom:
                if "CUDA out of memory" not in str(oom):
                    raise
                # Fallback to CPU just for this batch
                enc_cpu = {k: v.to("cpu") for k, v in enc.items()}
                self.model.to("cpu")
                logits = self.model(**enc_cpu).logits
                # Move back for next batch
                self.model.to(self._dev_pref)

            # Compute probabilities
            if logits.shape[-1] == 1:
                # Degenerate one-logit head: interpret as "entail" score
                ent = logits.squeeze(-1)
                ent_prob = torch.sigmoid(ent)
                contra_prob = 1.0 - ent_prob
            else:
                probs = torch.softmax(logits, dim=-1)
                ent_prob = probs[:, self._id_ent]
                contra_prob = probs[:, self._id_contra]

            p_entail.append(ent_prob.detach().float().cpu())
            p_contra.append(contra_prob.detach().float().cpu())

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
