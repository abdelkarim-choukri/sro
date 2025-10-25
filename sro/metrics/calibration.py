# sro/metrics/calibration.py
"""
Does:
    Utilities for calibration metrics and temperature scaling of classifier logits.
    Provides: softmax (probabilize), NLL, ECE, MCE, and a robust fit for
    a single temperature scalar by minimizing NLL.

Inputs:
    Public functions accept NumPy arrays:
        logits: [N, C]
        probs : [N, C]
        y_idx : [N] class indices

Outputs:
    Scalars for metrics; float temperature T for scaling; helper to get probs.

Notes:
    * No torch dependency except for the optimizer inside fit_temperature_scalar.
    * ECE/MCE use equally spaced bins over confidence [0,1], default 15 bins.
    * Temperature scaling preserves argmax for T>0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


# ----------------------------
# Numerically stable softmax
# ----------------------------
def probabilize(logits: np.ndarray) -> np.ndarray:
    """Softmax over last dim; returns probabilities in [0,1] summing to 1 per row."""
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z, dtype=np.float64)
    denom = np.sum(ez, axis=1, keepdims=True)
    # Guard against all -inf (degenerate inputs)
    denom = np.where(denom == 0.0, 1.0, denom)
    return (ez / denom).astype(np.float64)


def compute_nll_from_logits(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Average negative log-likelihood from raw logits."""
    probs = probabilize(logits)
    p_true = np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def compute_ece(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error with equal-width bins on max confidence.
    Returns a scalar in [0,1].
    """
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_idx).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_conf = conf[mask].mean()
        bin_acc = acc[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_mce(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error across bins."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_idx).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_conf = conf[mask].mean()
        bin_acc = acc[mask].mean()
        gaps.append(abs(bin_acc - bin_conf))
    return float(max(gaps) if gaps else 0.0)


@dataclass
class TemperatureFitResult:
    T: float
    nll_before: float
    nll_after: float
    iters: int


def fit_temperature_scalar(
    logits: np.ndarray, y_idx: np.ndarray, max_iter: int = 200, lr: float = 0.01
) -> float:
    """
    Fit a single temperature scalar T>0 minimizing NLL on the provided dev logits.

    We optimize over log_T ∈ ℝ to enforce positivity: T = exp(log_T).
    Uses Adam for stability.

    Returns:
        float T
    """
    assert logits.ndim == 2, "logits must be [N, C]"
    N, C = logits.shape
    assert len(y_idx) == N, "labels length mismatch"

    x = torch.from_numpy(logits).to(torch.float64)
    y = torch.from_numpy(y_idx).to(torch.int64)

    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # start at T=1
    opt = torch.optim.Adam([log_T], lr=lr)

    def _nll(cur_log_T: torch.Tensor) -> torch.Tensor:
        # logits / T = logits * exp(-log_T)
        scaled = x * torch.exp(-cur_log_T)
        # Stable log-softmax
        lse = torch.logsumexp(scaled, dim=1)
        ll = scaled[torch.arange(N), y] - lse
        return -ll.mean()

    best = (np.inf, None)
    it = 0
    for it in range(1, max_iter + 1):
        opt.zero_grad()
        loss = _nll(log_T)
        loss.backward()
        opt.step()
        cur = float(loss.detach().cpu())
        if cur < best[0] - 1e-9:
            best = (cur, float(log_T.detach().cpu()))
        # small patience-like early stopping
        if it > 25 and abs(cur - best[0]) < 1e-7:
            break

    T = float(np.exp(best[1] if best[1] is not None else float(log_T.detach().cpu())))
    # clamp to sane range to avoid pathological artifacts
    T = float(np.clip(T, 1e-2, 10.0))
    return T
