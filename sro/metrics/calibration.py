# sro/metrics/calibration.py
"""
Does: Calibration metrics and 1D temperature fitting utilities.
Inputs: logits (np.ndarray [N, C]), labels (np.ndarray [N] in [0..C-1])
Outputs: ECE, MCE, NLL computations; temperature fit via grid + golden search.

Notes:
- Pure NumPy, deterministic, no SciPy needed.
- Temperature scaling divides logits by T (>0).
- Golden search runs on log T space for stability.
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def probabilize(logits: np.ndarray) -> np.ndarray:
    """Convert logits [N, C] to probs [N, C]."""
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [N, C]")
    return _softmax(logits)


def compute_nll_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood."""
    probs = probabilize(logits)
    n = probs.shape[0]
    p_true = probs[np.arange(n), y.astype(int)]
    # numerical floor
    p_true = np.clip(p_true, 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def _bin_confidence(probs: np.ndarray, y: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-bin (acc, conf, count)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs, confs, counts = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (conf > b0) & (conf <= b1)
        cnt = int(m.sum())
        if cnt == 0:
            accs.append(0.0)
            confs.append(0.0)
            counts.append(0)
        else:
            accs.append(float(correct[m].mean()))
            confs.append(float(conf[m].mean()))
            counts.append(cnt)
    return np.array(accs), np.array(confs), np.array(counts)


def compute_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Expected calibration error (ECE)."""
    acc, c, n = _bin_confidence(probs, y, n_bins)
    if n.sum() == 0:
        return 0.0
    w = n / n.sum()
    return float(np.sum(w * np.abs(acc - c)))


def compute_mce(probs: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Maximum calibration error (MCE)."""
    acc, c, n = _bin_confidence(probs, y, n_bins)
    if n.sum() == 0:
        return 0.0
    return float(np.max(np.abs(acc - c)))


def fit_temperature_scalar(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Fit a single temperature T>0 by minimizing NLL with a robust,
    dependency-free search: coarse grid + golden section on log T.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [N, C]")
    y = y.astype(int)

    def nll_at(T: float) -> float:
        return compute_nll_from_logits(logits / max(T, 1e-8), y)

    # 1) coarse grid on log T in [log 0.25, log 8]
    grid = np.exp(np.linspace(math.log(0.25), math.log(8.0), 41))
    vals = np.array([nll_at(t) for t in grid])
    i_best = int(vals.argmin())
    T0 = float(grid[i_best])

    # 2) golden search on log T around T0
    phi = (1 + 5 ** 0.5) / 2
    left = max(math.log(0.25), math.log(T0) - 1.0)
    right = min(math.log(8.0),  math.log(T0) + 1.0)

    def nll_logt(z: float) -> float:
        return nll_at(math.exp(z))

    # 25 iters is cheap and precise here
    a, b = left, right
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = nll_logt(c)
    fd = nll_logt(d)
    for _ in range(25):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = nll_logt(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = nll_logt(d)
    logT = 0.5 * (a + b)
    T = float(max(1e-6, min(100.0, math.exp(logT))))
    return T
