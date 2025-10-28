# scripts/calibrate_nli.py
"""
Does:
    Calibrate the NLI backend with a single temperature scalar T using dev data.
    Reads a CSV of (premise, hypothesis, gold_label), but also supports common aliases:
      premise  ~ {premise,p,context,passage,evidence,evidence_text,ctx,sentence,doc}
      hypothesis ~ {hypothesis,h,claim,claim_text,statement}
      label    ~ {gold_label,label,y,target,verdict,nli_label,fever_label,hover_label}
    Also normalizes FEVER/HoVer and boolean labels:
      SUPPORTS/true -> entailment
      REFUTES/false -> contradiction  (or not_entailment if the backend is binary)
      NEI/ambiguous/unknown -> neutral (or not_entailment if binary)

CLI:
    python -m scripts.calibrate_nli --input ... --out ... --seed 42 --bs 64
    # Column overrides if needed:
    --premise_col question --hypothesis_col claim --label_col label
    # Local model dir:
    --model_dir models_cache\MoritzLaurer\deberta-v3-large-zeroshot-v2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from sro.metrics.calibration import (
    compute_ece,
    compute_mce,
    compute_nll_from_logits,
    fit_temperature_scalar,
    probabilize,
)
from sro.nli.backend import NLIBackend

logger = logging.getLogger("sro.calibrate_nli")
logging.basicConfig(
    level=os.environ.get("SRO_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_PREMISE_ALIASES = {
    "premise","p","context","ctx","passage","evidence","evidence_text","sentence","sent","doc","document",
}
_HYPO_ALIASES = {"hypothesis","h","claim","claim_text","statement"}
_LABEL_ALIASES = {"gold_label","label","y","target","verdict","nli_label","fever_label","hover_label"}

_LABEL_NORMALIZER: dict[str, str] = {
    # entailment
    "entailment": "entailment", "e": "entailment",
    "supports": "entailment", "support": "entailment", "supported": "entailment",
    "true": "entailment",
    # contradiction
    "contradiction": "contradiction", "c": "contradiction",
    "refutes": "contradiction", "refute": "contradiction", "refuted": "contradiction",
    "false": "contradiction",
    # neutral
    "neutral": "neutral", "n": "neutral",
    "nei": "neutral", "not_enough_info": "neutral", "not enough info": "neutral",
    "unknown": "neutral", "ambiguous": "neutral",
}

def _resolve_columns(df: pd.DataFrame, premise_col: str | None, hypothesis_col: str | None, label_col: str | None) -> tuple[str, str, str]:
    cols_lc = {c.lower(): c for c in df.columns}
    def _pick(aliases: set[str]) -> str | None:
        for a in aliases:
            if a in cols_lc:
                return cols_lc[a]
        return None
    p = cols_lc.get(premise_col.lower()) if premise_col else _pick(_PREMISE_ALIASES)
    h = cols_lc.get(hypothesis_col.lower()) if hypothesis_col else _pick(_HYPO_ALIASES)
    y = cols_lc.get(label_col.lower()) if label_col else _pick(_LABEL_ALIASES)
    missing = [name for name, real in [("premise", p), ("hypothesis", h), ("label", y)] if real is None]
    if missing:
        preview = ", ".join(df.columns[:10])
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Available columns: [{preview}{'...' if len(df.columns)>10 else ''}]. "
            "Use --premise_col/--hypothesis_col/--label_col to point to your schema."
        )
    return p, h, y

def _device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _normalize_label_string(s: str) -> str:
    return s.strip().lower().replace("-", "_")

def _map_to_backend_index(name: str, label_to_index: dict[str, int]) -> int:
    """
    Map normalized NLI label name to backend index, supporting binary backends.
    name ∈ {'entailment','contradiction','neutral'} after normalization.
    """
    # direct hit
    if name in label_to_index:
        return label_to_index[name]
    # binary backend fallback
    if "not_entailment" in label_to_index and name in ("contradiction", "neutral"):
        return label_to_index["not_entailment"]
    raise ValueError(f"Cannot map label '{name}' to backend classes {list(label_to_index.keys())}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="CSV path")
    ap.add_argument("--out", required=True, type=str, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (default: auto)")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--premise_col", type=str, default=None)
    ap.add_argument("--hypothesis_col", type=str, default=None)
    ap.add_argument("--label_col", type=str, default=None)
    ap.add_argument("--model_dir", type=str, default=None, help="Path to local HF model directory")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = args.device or _device_auto()
    logger.info(f"Using device: {device}")

    df = pd.read_csv(args.input)
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()

    p_col, h_col, y_col = _resolve_columns(df, args.premise_col, args.hypothesis_col, args.label_col)
    premises: list[str] = df[p_col].astype(str).tolist()
    hypotheses: list[str] = df[h_col].astype(str).tolist()
    raw_labels: list[str] = df[y_col].astype(str).tolist()

    backend = NLIBackend(device=device, model_dir=args.model_dir)
    label_to_index = backend.label_to_index  # tri-class or binary

    normalized_indices: list[int] = []
    bads: list[str] = []
    for s in raw_labels:
        base = _normalize_label_string(s)
        base = _LABEL_NORMALIZER.get(base, base)  # map booleans/fever to {entailment,contradiction,neutral}
        try:
            idx = _map_to_backend_index(base, label_to_index)
            normalized_indices.append(idx)
        except Exception:
            bads.append(s)

    if bads:
        uniq = sorted(set(bads))[:10]
        raise ValueError(
            "Unknown/unsupported gold labels after normalization: "
            + ", ".join(repr(x) for x in uniq)
            + f". Backend classes available: {list(label_to_index.keys())}."
        )

    y_idx = np.asarray(normalized_indices, dtype=np.int64)

    logits = backend.predict_logits(premises, hypotheses, batch_size=args.bs)
    if not isinstance(logits, np.ndarray) or logits.ndim != 2:
        raise RuntimeError("Backend.predict_logits must return a 2D np.ndarray [N, C].")
    if logits.shape[0] != len(y_idx):
        raise RuntimeError("Mismatch between logits rows and labels.")

    nll_before = compute_nll_from_logits(logits, y_idx)
    probs_before = probabilize(logits)
    ece_before = compute_ece(probs_before, y_idx)
    mce_before = compute_mce(probs_before, y_idx)

    T = fit_temperature_scalar(logits, y_idx)
    logger.info(f"Fitted temperature T = {T:.6f}")

    logits_cal = logits / T
    nll_after = compute_nll_from_logits(logits_cal, y_idx)
    probs_after = probabilize(logits_cal)
    ece_after = compute_ece(probs_after, y_idx)
    mce_after = compute_mce(probs_after, y_idx)

    unchanged = (probs_before.argmax(axis=1) == probs_after.argmax(axis=1)).mean()
    logger.info(f"Argmax unchanged on {unchanged * 100:.2f}% of samples.")

    logger.info("=== Calibration Report (dev) ===")
    logger.info(f"NLL  : {nll_before:.4f} -> {nll_after:.4f}")
    logger.info(f"ECE  : {ece_before:.4f} -> {ece_after:.4f}")
    logger.info(f"MCE  : {mce_before:.4f} -> {mce_after:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {"model": backend.model_name, "T": float(T), "updated_at": datetime.now(timezone.utc).isoformat()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote calibration to: {args.out}")

if __name__ == "__main__":
    main()
