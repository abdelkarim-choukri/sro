"""
UB validation and tuning script.

Goal:
  Given a CSV of pairs (claim, s_i, s_j), compute:
    - p1_i, p1_j: one-hop entailment probs (NLI)
    - p2: two-hop entailment probs (NLI on "s_i [SEP] s_j" → claim)
    - UB(i,j): rule-based upper bound using SRO UB formula and current κ (kappa)
  Report:
    - coverage = % of pairs with UB ≥ p2 (target ≥95%)
    - recommended κ' to reach ≥95% based on the 95th percentile of violations (p2 - UB)+

CSV schema (header required, extra columns ignored):
  required:
    claim, s_i, s_j
  optional:
    src_i, src_j     (for section_novelty; default: different→1.0 if text differs)
    ce_i, ce_j       (float reranker scores; normalized inside per-pair; default 0.5)

Usage:
  python -m scripts.tune_ub --csv data/processed/dev_pairs.csv --batch-size 16
  # optional:
  #   --kappa 0.05              (override config κ)
  #   --sep " [SEP] "           (premise separator for two-hop)
  #   --save artifacts/logs/ub_eval.csv

Assumptions:
  - Small dev CSV (e.g., ≤ a few hundred rows).
  - MNLI model downloads are cached (HuggingFace).
"""

from __future__ import annotations
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from sro.config import load_config
from sro.prover.s4_ub import UBWeights, upper_bound, clamp01

# Real NLI (batched, GPU if available)
from sro.nli.nli_infer import one_hop_scores as nli_one_hop
from sro.nli.nli_infer import two_hop_scores as nli_two_hop

# Light tokenization & feature helpers (align with s2/s3)
import re
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]

def _tokset(s: str) -> set:
    return set(_tokens(s))

def _capitalized_tokens(tokens: List[str]) -> set:
    # For simplicity on lowercased tokens, "capitalized" proxy is empty; fall back to numbers only.
    # We keep code structure identical to s3; with real NER later, this improves.
    return set()

def _numbers(tokens: List[str]) -> set:
    return set(t for t in tokens if t.isdigit())

def _entity_overlap(claim_tokens: List[str], toks_i: List[str], toks_j: List[str]) -> float:
    # Proxy: overlap of numeric tokens (years/dates) between claim and union of i/j
    claim_ents = _numbers(claim_tokens)
    i_ents = _numbers(toks_i)
    j_ents = _numbers(toks_j)
    if not claim_ents:
        return 0.0
    inter = len(claim_ents & (i_ents | j_ents))
    return inter / max(1, len(claim_ents))

def _years(tokens: List[str]) -> set:
    return set(_YEAR_RE.findall(" ".join(tokens)))

def _time_agreement(tokens_i: List[str], tokens_j: List[str]) -> float:
    yi, yj = _years(tokens_i), _years(tokens_j)
    if not yi and not yj:
        return 0.5
    return 1.0 if (yi & yj) else 0.0

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

@dataclass
class Row:
    claim: str
    s_i: str
    s_j: str
    src_i: Optional[str]
    src_j: Optional[str]
    ce_i: Optional[float]
    ce_j: Optional[float]

def _read_csv(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"claim", "s_i", "s_j"}
        missing = required - set(k.strip() for k in r.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        for line in r:
            def _getf(k: str) -> Optional[float]:
                v = (line.get(k) or "").strip()
                if v == "":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            rows.append(Row(
                claim=(line.get("claim") or "").strip(),
                s_i=(line.get("s_i") or "").strip(),
                s_j=(line.get("s_j") or "").strip(),
                src_i=(line.get("src_i") or "").strip() or None,
                src_j=(line.get("src_j") or "").strip() or None,
                ce_i=_getf("ce_i"),
                ce_j=_getf("ce_j"),
            ))
    if not rows:
        raise ValueError("CSV has no data rows.")
    return rows

def _normalize_ce(ce_i: Optional[float], ce_j: Optional[float]) -> float:
    # ce_max normalized per pair; if both missing or equal, fallback to 0.5
    if ce_i is None and ce_j is None:
        return 0.5
    a = ce_i if ce_i is not None else 0.0
    b = ce_j if ce_j is not None else 0.0
    lo = min(a, b); hi = max(a, b)
    den = hi - lo
    if den <= 1e-12:
        return 0.5
    return (hi - lo) / den  # equals 1.0 always; but we keep shape for extensibility

def _section_novelty(src_i: Optional[str], src_j: Optional[str], s_i: str, s_j: str) -> float:
    # Prefer different sources; if missing, treat different texts as different sections
    if src_i and src_j:
        return 1.0 if src_i != src_j else 0.0
    return 1.0 if s_i != s_j else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dev pairs CSV")
    ap.add_argument("--kappa", type=float, default=None, help="Override κ (kappa) for UB")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for NLI")
    ap.add_argument("--sep", type=str, default=" [SEP] ", help="Separator for two-hop premise concatenation")
    ap.add_argument("--save", type=str, default=None, help="Optional path to save per-row results CSV")
    args = ap.parse_args()

    cfg = load_config()
    kappa = float(args.kappa) if args.kappa is not None else float(cfg.sro_prover.kappa)
    rows = _read_csv(args.csv)

    # --- One-hop caching: score unique (claim, sentence) pairs to avoid duplicate NLI calls.
    unique_cs: Dict[Tuple[str, str], Tuple[float, float]] = {}
    # Group sentences per claim for fewer calls
    by_claim: Dict[str, List[str]] = {}
    for r in rows:
        by_claim.setdefault(r.claim, [])
        by_claim[r.claim].append(r.s_i)
        by_claim[r.claim].append(r.s_j)
    # Deduplicate sentence lists
    for c in by_claim:
        by_claim[c] = list({s for s in by_claim[c] if s})

    # Compute one-hop in batches per claim
    for claim, sent_list in by_claim.items():
        if not sent_list:
            continue
        p1, c1 = nli_one_hop(claim, sent_list, batch_size=args.batch_size)
        for s, pe, pc in zip(sent_list, p1, c1):
            unique_cs[(claim, s)] = (float(pe), float(pc))

    # --- Two-hop: collect all pairs and score in a single call (or chunked if large)
    all_pairs_text: List[Tuple[str, str]] = [(r.s_i, r.s_j) for r in rows]
    p2_all = nli_two_hop(rows[0].claim, all_pairs_text, batch_size=args.batch_size, sep=args.sep) \
             if len({r.claim for r in rows}) == 1 \
             else _two_hop_multi_claim(rows, batch_size=args.batch_size, sep=args.sep)

    # Prepare outputs
    out_cols = ["claim", "s_i", "s_j", "p1_i", "p1_j", "p2", "UB", "UB_minus_p2", "violation"]
    out_rows: List[List[str]] = []
    diffs: List[float] = []
    cov_numer = 0

    w = UBWeights()

    for idx, r in enumerate(rows):
        claim = r.claim
        si, sj = r.s_i, r.s_j
        p1_i, c1_i = unique_cs.get((claim, si), (0.0, 0.0))
        p1_j, c1_j = unique_cs.get((claim, sj), (0.0, 0.0))
        p2 = float(p2_all[idx])

        # Build features consistent with s3 (cheap)
        ti = _tokens(si); tj = _tokens(sj); tc = _tokens(claim)
        max_p1 = max(p1_i, p1_j)
        sum_p1 = min(1.0, p1_i + p1_j)
        ce_max = _normalize_ce(r.ce_i, r.ce_j)
        ent_ov = _entity_overlap(tc, ti, tj)
        time_ag = _time_agreement(ti, tj)
        sim = _jaccard(_tokset(si), _tokset(sj))
        distance = 1.0 - sim
        sect_nov = _section_novelty(r.src_i, r.src_j, si, sj)

        feats = {
            "max_p1": float(max_p1),
            "sum_p1": float(sum_p1),
            "ce_max": float(ce_max),
            "entity_overlap": float(ent_ov),
            "time_agreement": float(time_ag),
            "distance": float(distance),
            "section_novelty": float(sect_nov),
        }
        ub = float(upper_bound(feats, kappa=kappa, w=w))
        violation = 1 if ub + 1e-9 < p2 else 0
        if violation == 0:
            cov_numer += 1
        diff = ub - p2
        diffs.append(diff)

        out_rows.append([
            claim, si, sj,
            f"{p1_i:.6f}", f"{p1_j:.6f}", f"{p2:.6f}",
            f"{ub:.6f}", f"{(ub - p2):.6f}", str(violation)
        ])

    N = len(rows)
    coverage = cov_numer / N if N else 0.0
    diffs_arr = np.array(diffs, dtype=np.float32)
    gaps = -np.minimum(diffs_arr, 0.0)  # positive only where UB < p2
    need = float(np.quantile(gaps, 0.95)) if N > 0 else 0.0
    kappa_suggest = clamp01(kappa + need)  # keep result in [0,1]

    print(f"\nUB validation on {N} pairs")
    print(f"  κ (kappa) used: {kappa:.3f}")
    print(f"  Coverage (UB ≥ p2): {coverage*100:.2f}% (target ≥ 95.00%)")
    print(f"  Mean(UB - p2): {float(np.mean(diffs_arr)):.4f} | Min: {float(np.min(diffs_arr)):.4f} | Max: {float(np.max(diffs_arr)):.4f}")
    print(f"  95th percentile violation (needed cushion): {need:.4f}")
    if coverage < 0.95 and need > 0:
        print(f"  Suggested κ': {kappa_suggest:.3f}  (set sro_prover.kappa to this or higher)")
    else:
        print("  κ is sufficient for target coverage.")

    # Save per-row results if requested (CSV)
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w", encoding="utf-8", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(out_cols)
            wcsv.writerows(out_rows)
        print(f"Saved per-row details to: {args.save}")

def _two_hop_multi_claim(rows: List[Row], batch_size: int = 16, sep: str = " [SEP] ") -> List[float]:
    """
    Handle the general case with multiple different claims in the CSV:
    score two-hop per claim in batches and reassemble in the original order.
    """
    # Group indices by claim
    by_claim_idx: Dict[str, List[int]] = {}
    for idx, r in enumerate(rows):
        by_claim_idx.setdefault(r.claim, []).append(idx)
    out = [0.0] * len(rows)
    for claim, idxs in by_claim_idx.items():
        pairs = [(rows[i].s_i, rows[i].s_j) for i in idxs]
        p2 = nli_two_hop(claim, pairs, batch_size=batch_size, sep=sep)
        for k, i in enumerate(idxs):
            out[i] = float(p2[k])
    return out

if __name__ == "__main__":
    main()
