# scripts/augment_pairs_y_true.py
"""
Does:
    Append 'y_true' to a pairs CSV by running the calibrated NLI backend.
    Uses sent_id_i / sent_id_j -> sentence text (from data/corpus/sentences.txt),
    then computes entailment(premise, claim) for i and j, and sets:
        y_true = max(y_true_i, y_true_j)
    Also writes y_true_i, y_true_j columns.

Robustness:
    - If a sent_id cannot be found in the corpus file, we SKIP scoring that side
      and set its entailment to 0.0. We log how many were missing.
      This avoids hard failures when your pairs reference sources not present in
      the simple toy corpus.

Inputs:
    --pairs  : CSV from scripts.make_dev_pairs (must include 'claim' and sent_id_i / sent_id_j)
    --out    : output CSV (can overwrite input)
    --bs     : batch size for NLI (default 64)
    --seed   : RNG seed
    --nli_model_dir : explicit local dir for NLI (or use $env:SRO_NLI_MODEL_DIR)

Assumptions:
    - Default corpus lives at data/corpus/sentences.txt, one sentence per line, ids are s0, s1, ...
    - If sent_id looks like "...#s123", we map using the trailing s123.

Outputs:
    - Same CSV + columns: y_true, y_true_i, y_true_j (float32 in [0,1]).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sro.nli.backend import NLIBackend

logging.basicConfig(
    level=os.environ.get("SRO_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("scripts.augment_pairs_y_true")


# ---------- seed discipline ----------
def set_all_seeds(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------- sent_id -> text ----------
def _load_corpus_map(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Default corpus not found at {path}. Create it or point your pipeline to a corpus.\n"
            "Expected one sentence per line; sent_ids are s0, s1, ..."
        )
    m: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            m[f"s{i}"] = s
    if not m:
        raise ValueError(f"Corpus file is empty: {path}")
    return m


_ID_TAIL = re.compile(r"(s\d+)$", re.IGNORECASE)

def _norm_sent_id(raw: str) -> str:
    # If id is already sNNN, return as-is; else try to extract trailing sNNN (e.g., "news:1#s3" -> "s3")
    if raw.startswith("s") and raw[1:].isdigit():
        return raw
    m = _ID_TAIL.search(raw)
    if m:
        return m.group(1)
    return raw  # last resort: return raw


def _maybe_lookup(sent_id: str, m: Dict[str, str]) -> Optional[str]:
    sid = _norm_sent_id(str(sent_id))
    txt = m.get(sid)
    if txt is None:
        return None
    return txt


# ---------- NLI scoring ----------
def _entail_idx_from_backend(backend) -> int:
    """
    Try to recover the entailment column index from the backend object.
    Supports common attributes we used in V2.
    """
    # Preferred: explicit mapping
    if hasattr(backend, "label_to_index"):
        m = getattr(backend, "label_to_index")
        if isinstance(m, dict) and "entailment" in m:
            return int(m["entailment"])
    # Common HF-style attributes
    if hasattr(backend, "index_to_label"):
        seq = getattr(backend, "index_to_label")
        try:
            for i, name in enumerate(seq):
                if str(name).lower() == "entailment":
                    return int(i)
        except Exception:
            pass
    if hasattr(backend, "id2label"):
        d = getattr(backend, "id2label")
        if isinstance(d, dict):
            for k, v in d.items():
                if str(v).lower() == "entailment":
                    try:
                        return int(k)
                    except Exception:
                        continue
    raise KeyError("Cannot infer entailment index from backend. Add label_to_index['entailment'].")


def _score_pairs(backend: NLIBackend, premises: List[str], hypotheses: List[str], bs: int) -> np.ndarray:
    outs: List[np.ndarray] = []
    entail_idx: int = -1  # lazy-resolve once per batch if needed

    for i in range(0, len(premises), bs):
        p = premises[i:i+bs]
        h = hypotheses[i:i+bs]
        out = backend.score_pairs(p, h, batch_size=bs)

        # Try to get labels without using boolean 'or' on arrays
        labels = None
        if isinstance(out, dict):
            if "labels" in out:
                labels = out["labels"]
            elif "label_names" in out:
                labels = out["label_names"]
            elif "labels_names" in out:
                labels = out["labels_names"]

        probs = None
        if isinstance(out, dict):
            if "probs" in out:
                probs = out["probs"]
            elif "probabilities" in out:
                probs = out["probabilities"]

        if probs is None:
            # Fallback: maybe dict has per-label arrays
            if isinstance(out, dict) and "entailment" in out:
                e = np.asarray(out["entailment"], dtype="float32")
                outs.append(e)
                continue
            raise KeyError(f"NLI output missing 'probs'/'probabilities'. Keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")

        probs = np.asarray(probs, dtype="float32")

        # If backend gave labels with probs, use them
        if labels is not None:
            labels = [str(x).lower() for x in list(labels)]
            if probs.ndim == 2 and probs.shape[1] == len(labels) and "entailment" in labels:
                e_idx = labels.index("entailment")
                outs.append(probs[:, e_idx])
                continue
            else:
                raise ValueError(f"Shape mismatch: probs {probs.shape} vs labels {len(labels)}")

        # No labels given: infer entailment index from backend mapping
        if entail_idx < 0:
            entail_idx = _entail_idx_from_backend(backend)

        if probs.ndim == 1:
            # Already a single-column entailment probability
            outs.append(probs.astype("float32"))
        elif probs.ndim == 2 and 0 <= entail_idx < probs.shape[1]:
            outs.append(probs[:, entail_idx])
        else:
            raise ValueError(f"Unexpected probs shape: {probs.shape} and no labels supplied.")

    return np.concatenate(outs).astype("float32") if outs else np.zeros((0,), dtype="float32")

# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nli_model_dir", type=str, default=None)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    df = pd.read_csv(args.pairs)
    # Resolve claim text
    if "claim" not in df.columns:
        raise KeyError("pairs CSV must include a 'claim' column for hypothesis text.")
    claims = df["claim"].astype(str).tolist()

    # Resolve premises via sent_id_i / sent_id_j
    has_i = "sent_id_i" in df.columns
    has_j = "sent_id_j" in df.columns
    if not has_i and not has_j:
        raise KeyError("pairs CSV must contain 'sent_id_i' and/or 'sent_id_j' to map to text.")

    sent_map = _load_corpus_map(os.path.join("data", "corpus", "sentences.txt"))

    # Build lists for i and j, skipping missing mappings
    premises_i, idx_i = [], []
    premises_j, idx_j = [], []
    miss_i, miss_j = [], []

    if has_i:
        for k, sid in enumerate(df["sent_id_i"].astype(str).tolist()):
            if sid and sid != "nan":
                txt = _maybe_lookup(sid, sent_map)
                if txt is None:
                    miss_i.append(k)
                    continue
                premises_i.append(txt)
                idx_i.append(k)
    if has_j:
        for k, sid in enumerate(df["sent_id_j"].astype(str).tolist()):
            if sid and sid != "nan":
                txt = _maybe_lookup(sid, sent_map)
                if txt is None:
                    miss_j.append(k)
                    continue
                premises_j.append(txt)
                idx_j.append(k)

    backend = NLIBackend(model_dir=args.nli_model_dir)

    y_i = np.zeros((len(df),), dtype="float32")
    y_j = np.zeros((len(df),), dtype="float32")

    if idx_i:
        hyps_i = [claims[k] for k in idx_i]
        e_i = _score_pairs(backend, premises_i, hyps_i, args.bs)
        if e_i.shape[0] != len(idx_i):
            raise RuntimeError("Batch size mismatch on i-scores.")
        y_i[idx_i] = e_i
    if miss_i:
        LOGGER.warning("Missing text for %d sent_id_i rows; set their entailment to 0.0.", len(miss_i))

    if idx_j:
        hyps_j = [claims[k] for k in idx_j]
        e_j = _score_pairs(backend, premises_j, hyps_j, args.bs)
        if e_j.shape[0] != len(idx_j):
            raise RuntimeError("Batch size mismatch on j-scores.")
        y_j[idx_j] = e_j
    if miss_j:
        LOGGER.warning("Missing text for %d sent_id_j rows; set their entailment to 0.0.", len(miss_j))

    # y_true = max(y_i, y_j)
    y_true = np.maximum(y_i, y_j).astype("float32")

    df = df.copy()
    df["y_true_i"] = y_i
    df["y_true_j"] = y_j
    df["y_true"] = y_true

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    LOGGER.info(
        "Wrote %s  y_true stats: min=%.4f mean=%.4f p90=%.4f max=%.4f  (missing_i=%d, missing_j=%d)",
        args.out, float(y_true.min()), float(y_true.mean()), float(np.quantile(y_true, 0.90)), float(y_true.max()),
        len(miss_i), len(miss_j)
    )


if __name__ == "__main__":
    main()
