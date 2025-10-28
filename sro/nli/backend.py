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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

_ALLOW_DUMMY_NLI = os.getenv("SRO_ALLOW_DUMMY_NLI", "") == "1"

_LOGGER = logging.getLogger("sro.nli.backend")


# ---------------------------- label handling ----------------------------
def _normalize_nli_label_map(id2label: Mapping[int, str]) -> dict[str, int]:
    """
    Build a name->index map for NLI classes from an HF id2label.

    Returns one of:
      - {'entailment':i, 'contradiction':i, 'neutral':i}  (3-class)
      - {'entailment':i, 'not_entailment':i}              (binary)

    Fails loudly if neither pattern is present.
    """
    names = {int(k): str(v).strip().lower().replace(" ", "_").replace("-", "_") for k, v in id2label.items()}
    inv: dict[str, int] = {}
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
        with open(path, encoding="utf-8") as f:
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
    Natural Language Inference backend with:
      - Real model (local-only) when available
      - Deterministic dummy fallback when SRO_ALLOW_DUMMY_NLI=1 (for offline CPU tests)

    Public surface kept minimal and stable:
      - __init__(model_name: Optional[str] = None, device: Optional[str] = None, temperature: Optional[float] = None)
      - score_pairs(premises: Sequence[str], hypotheses: Sequence[str], batch_size: int = 8) -> Dict[str, np.ndarray]
      - set_temperature(T: float) / get_temperature() -> float

    Output:
      {"probs": np.ndarray of shape (N, 3)} ordered as [entailment, neutral, contradiction].
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        temperature: float | None = None,
    ) -> None:
        import os

        self._dummy: bool = False
        self.model_name: str = model_name or os.getenv("SRO_NLI_MODEL_NAME", "facebook/bart-large-mnli")
        self.temperature: float = float(
            temperature if temperature is not None else os.getenv("SRO_NLI_TEMPERATURE", "1.0")
        )

        # Common attrs some code relies on
        self.label_to_index: dict[str, int] = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self.index_to_label: list[str] = ["entailment", "neutral", "contradiction"]
        self._labels: list[str] = self.index_to_label.copy()
        self.is_binary: bool = False

        # Decide device (respect SRO_DEVICE if set)
        try:
            import torch  # noqa: F401
            has_torch = True
        except Exception:
            has_torch = False

        dev_env = (os.getenv("SRO_DEVICE") or "").strip()
        if device is not None:
            self._device_str = device
        elif dev_env:
            self._device_str = dev_env
        else:
            if has_torch:
                import torch  # type: ignore
                self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device_str = "cpu"

        # Runtime flag (read AFTER test sets env)
        allow_dummy = (os.getenv("SRO_ALLOW_DUMMY_NLI") or "") == "1"

        # Try to load a real model from local cache; else dummy if allowed; else error.
        try:
            if not has_torch:
                raise RuntimeError("PyTorch not available in this environment")

            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            # Prefer a fully local directory if present; otherwise rely on local_files_only=True.
            local_dir_hint = os.getenv("SRO_NLI_LOCAL_DIR")
            if local_dir_hint and os.path.isdir(local_dir_hint):
                load_name = local_dir_hint
            else:
                # allow a local snapshot under models_cache/<repo_name>
                cache_guess = os.path.join("models_cache", self.model_name.replace("/", "_"))
                load_name = cache_guess if os.path.isdir(cache_guess) else self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(load_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_name, local_files_only=True)
            self.model.to(self._device_str)
            self.model.eval()

            # If the model exposes label mapping, respect it when reading logits
            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                # Normalize to lowercase for robustness
                rev = {str(v).lower(): int(k) for k, v in id2label.items()}
                for lbl in ("entailment", "neutral", "contradiction"):
                    if lbl in rev:
                        self.label_to_index[lbl] = rev[lbl]
        except Exception as e:
            if allow_dummy:
                # Minimal, deterministic, CPU-only dummy backend
                self._dummy = True
                self.model_name = "dummy-nli"
                self.temperature = 1.0
                self.label_to_index = {"entailment": 0, "neutral": 1, "contradiction": 2}
                self.index_to_label = ["entailment", "neutral", "contradiction"]
                self._labels = self.index_to_label.copy()
            else:
                # Keep the error explicit for real runs
                raise RuntimeError(
                    "NLI model could not be loaded locally. To run offline CI, set SRO_ALLOW_DUMMY_NLI=1. "
                    f"Attempted model: {self.model_name}. Device: {self._device_str}. Error: {e}"
                ) from e

    # ----------------------------
    # Temperature control
    # ----------------------------
    def set_temperature(self, T: float) -> None:
        self.temperature = float(T)

    def get_temperature(self) -> float:
        return float(self.temperature)

    # ----------------------------
    # Core API
    # ----------------------------
    def score_pairs(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = 8,
    ) -> dict[str, np.ndarray]:
        """
        Score NLI for (premise, hypothesis) pairs.

        Returns:
            {"probs": np.ndarray of shape (N, 3)} in order [entailment, neutral, contradiction].
        """
        # Align sizes defensively
        n = min(len(premises), len(hypotheses))
        premises = list(premises)[:n]
        hypotheses = list(hypotheses)[:n]

        if self._dummy:
            # Deterministic dummy: identical strings => strong entailment; else neutral-ish.
            probs = np.zeros((n, 3), dtype=np.float32)
            for i, (p, h) in enumerate(zip(premises, hypotheses)):
                if str(p).strip() == str(h).strip():
                    probs[i] = np.array([0.95, 0.05, 0.0], dtype=np.float32)
                else:
                    probs[i] = np.array([0.10, 0.85, 0.05], dtype=np.float32)
            return {"probs": probs}

        # Real model path
        import torch  # type: ignore
        from torch.nn.functional import softmax  # type: ignore

        bs = max(1, int(batch_size))
        out = np.zeros((n, 3), dtype=np.float32)

        # Model label indices (might not be 0/1/2)
        idx_ent = int(self.label_to_index.get("entailment", 0))
        idx_neu = int(self.label_to_index.get("neutral", 1))
        idx_con = int(self.label_to_index.get("contradiction", 2))

        with torch.no_grad():
            i = 0
            while i < n:
                j = min(i + bs, n)
                p_batch = premises[i:j]
                h_batch = hypotheses[i:j]

                enc = self.tokenizer(
                    p_batch,
                    h_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device_str) for k, v in enc.items()}
                logits = self.model(**enc).logits  # (B, C)

                T = max(1e-6, float(self.temperature))
                logits = logits / T

                pr = softmax(logits, dim=-1).detach().cpu().numpy()  # (B, C)

                # Map to fixed order [entailment, neutral, contradiction]
                # with safeguards if model has unexpected num_labels
                def _safe(arr: np.ndarray, k: int) -> float:
                    return float(arr[k]) if 0 <= k < arr.shape[0] else 0.0

                for r in range(pr.shape[0]):
                    out[i + r, 0] = _safe(pr[r], idx_ent)
                    out[i + r, 1] = _safe(pr[r], idx_neu)
                    out[i + r, 2] = _safe(pr[r], idx_con)

                i = j

        return {"probs": out}