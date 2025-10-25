# sro/nli/backend.py
"""
Does:
    Offline-first NLI backend around a HF sequence classifier with temperature scaling.
    - Supports BOTH 3-class ({entailment, contradiction, neutral}) and binary ({entailment, not_entailment}).
    - Reads calibration from artifacts/calib/nli_temperature.json.
    - Accepts model by HF repo-id (cache_dir) OR explicit local dir (--model_dir / SRO_NLI_MODEL_DIR).
    - Exposes raw logits (predict_logits) and calibrated probabilities (score_pairs).

Inputs:
    Text pairs (premise, hypothesis).

Outputs:
    score_pairs(): dict with "probs" (and optional "logits").
    predict_logits(): raw logits (no temperature).

Notes:
    * strictly local loading (local_files_only=True). If missing locally, raises a detailed error.
    * temperature is applied as logits / T before softmax.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_LOGGER = logging.getLogger("sro.nli.backend")


# ---------------------------- label handling ----------------------------
def _normalize_nli_label_map(id2label: Mapping[int, str]) -> Dict[str, int]:
    """
    Build a name->index map for NLI classes from an HF id2label.

    Returns one of:
      - {'entailment':i, 'contradiction':i, 'neutral':i}  (3-class)
      - {'entailment':i, 'not_entailment':i}              (binary)

    Fails loudly if neither pattern is present.
    """
    names = {int(k): str(v).strip().lower().replace(" ", "_").replace("-", "_") for k, v in id2label.items()}
    inv: Dict[str, int] = {}
    for i, n in names.items():
        if n in ("entailment", "entails"):
            inv["entailment"] = i
        elif n in ("contradiction", "contradict", "contradictory"):
            inv["contradiction"] = i
        elif n == "neutral":
            inv["neutral"] = i
        elif n in ("not_entailment", "non_entailment", "notentailment"):
            inv["not_entailment"] = i

    if all(k in inv for k in ("entailment", "contradiction", "neutral")):
        return {"entailment": inv["entailment"], "contradiction": inv["contradiction"], "neutral": inv["neutral"]}

    if all(k in inv for k in ("entailment", "not_entailment")):
        return {"entailment": inv["entailment"], "not_entailment": inv["not_entailment"]}

    raise ValueError(f"Model labels do not match supported NLI schemes; got id2label={dict(id2label)}")


def _names_equivalent(a: str, b: str) -> bool:
    """Treat repo-id and local-dir basename as equivalent for calibration matching."""
    def last(seg: str) -> str:
        seg = seg.strip().replace("\\", "/")
        return seg.split("/")[-1]
    return (a == b) or (last(a) == last(b))


def _read_temperature_json(path: str, model_name: str) -> float:
    """
    Read calibration JSON and return T if model matches; else 1.0.
    JSON: {"model": "<name or basename>", "T": float, "updated_at": iso8601}
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or "T" not in payload or "model" not in payload:
            _LOGGER.warning("Calibration JSON schema invalid; ignoring.")
            return 1.0
        if not _names_equivalent(str(payload["model"]), str(model_name)):
            _LOGGER.info(
                f"Calibration JSON model mismatch (have '{payload.get('model')}', need '{model_name}'); ignoring."
            )
            return 1.0
        T = float(payload["T"])
        if not (T > 0):
            _LOGGER.warning("Calibration T <= 0; ignoring.")
            return 1.0
        return float(np.clip(T, 1e-2, 10.0))
    except FileNotFoundError:
        return 1.0
    except Exception as e:
        _LOGGER.warning(f"Failed to read calibration JSON '{path}': {e}; ignoring.")
        return 1.0


def _raise_offline_help(errs: list[str], model_name: str, model_dir: str | None, cache_dir: str) -> None:
    msg = [
        "NLI model not found locally (offline mode). Fix it one of these ways:",
        "1) Point to a local model directory:",
        "   - set env SRO_NLI_MODEL_DIR=path\\to\\model_dir",
        "   - or pass --model_dir path\\to\\model_dir to scripts",
        "   The directory MUST contain at least: config.json, tokenizer.json (or vocab files), model.safetensors (or pytorch_model.bin).",
        "2) Pre-download the repo into a local folder (before going offline), e.g.:",
        "   hf snapshot download MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --local-dir models_cache\\MoritzLaurer\\deberta-v3-large-zeroshot-v2.0 --local-dir-use-symlinks False",
        f"(attempted model_name='{model_name}', model_dir='{model_dir}', cache_dir='{cache_dir}')",
        "Errors:",
        *errs,
    ]
    raise FileNotFoundError("\n".join(msg))


# ---------------------------- batching + core ----------------------------
@dataclass
class _Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class NLIBackend:
    """
    Offline-first NLI wrapper with calibrated scoring.

    Attrs:
        model_name: str                         # id used for calibration matching
        model_dir: str | None                   # explicit local dir if provided
        label_to_index: Dict[str,int]           # tri-class or binary mapping
        index_to_label: list[str]
        is_binary: bool
        temperature: float
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str = "models_cache",
        temperature_json_path: str = "artifacts/calib/nli_temperature.json",
        model_dir: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "SRO_NLI_MODEL", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        )
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or os.environ.get("SRO_NLI_MODEL_DIR")

        errs: list[str] = []
        tokenizer = None
        model = None

        # Try 1: explicit local directory (preferred offline)
        if self.model_dir:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True)
                # lock calibration identity to directory basename
                self.model_name = os.path.basename(os.path.normpath(self.model_dir))
            except Exception as e:
                errs.append(f"[local model_dir] {type(e).__name__}: {e}")

        # Try 2: repo id via cache_dir (offline)
        if tokenizer is None or model is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir, local_files_only=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir, local_files_only=True
                )
            except Exception as e:
                errs.append(f"[cache_dir] {type(e).__name__}: {e}")

        # Try 3: repo id via default HF cache (offline)
        if tokenizer is None or model is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
            except Exception as e:
                errs.append(f"[default cache] {type(e).__name__}: {e}")

        if tokenizer is None or model is None:
            _raise_offline_help(errs, self.model_name, self.model_dir, self.cache_dir)

        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()

        # Labels
        config = self.model.config
        id2label = getattr(config, "id2label", {i: str(i) for i in range(config.num_labels)})
        self.label_to_index = _normalize_nli_label_map(id2label)
        self.is_binary = ("not_entailment" in self.label_to_index) and ("contradiction" not in self.label_to_index)
        self.index_to_label = [str(id2label[i]).lower() for i in range(config.num_labels)]

        # Temperature
        self.temperature = _read_temperature_json(temperature_json_path, self.model_name)
        if self.temperature != 1.0:
            _LOGGER.info(f"NLI temperature = {self.temperature:.3f} (calibrated)")
        else:
            _LOGGER.info("NLI temperature = 1.000 (uncalibrated / default)")

        # Label scheme log
        if self.is_binary:
            _LOGGER.info("NLI label scheme: binary {'entailment','not_entailment'}")
        else:
            _LOGGER.info("NLI label scheme: 3-class {'entailment','contradiction','neutral'}")

    def _make_batches(self, premises: Sequence[str], hypotheses: Sequence[str], batch_size: int) -> Iterable[_Batch]:
        assert len(premises) == len(hypotheses), "premises and hypotheses must be same length"
        N = len(premises)
        for i in range(0, N, batch_size):
            toks = self.tokenizer(
                list(premises[i : i + batch_size]),
                list(hypotheses[i : i + batch_size]),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            yield _Batch(
                input_ids=toks["input_ids"].to(self.device),
                attention_mask=toks["attention_mask"].to(self.device),
            )

    # ---------------------------- API ----------------------------
    def predict_logits(self, premises: Sequence[str], hypotheses: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Raw (uncalibrated) logits [N,C]."""
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in self._make_batches(premises, hypotheses, batch_size):
                logits = self.model(
                    input_ids=batch.input_ids, attention_mask=batch.attention_mask
                ).logits  # [B,C]
                outs.append(logits.detach().cpu().numpy())
        if not outs:
            return np.zeros((0, self.model.config.num_labels), dtype=np.float32)
        return np.concatenate(outs, axis=0)

    def score_pairs(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = 32,
        return_logits: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Calibrated probabilities; applies logits / T then softmax."""
        raw = self.predict_logits(premises, hypotheses, batch_size)
        logits = raw / float(self.temperature)
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z, dtype=np.float64)
        probs = ez / np.sum(ez, axis=1, keepdims=True)
        out = {"probs": probs.astype(np.float64)}
        if return_logits:
            out["logits"] = logits.astype(np.float64)
        return out

    def get_temperature(self) -> float:
        """Return current temperature (1.0 if uncalibrated/missing)."""
        return float(self.temperature)
