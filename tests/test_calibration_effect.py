# tests/test_calibration_effect.py
"""
Does:
    Demonstrate that temperature scaling improves calibration on a held-out split
    and preserves argmax (>=99% identical). Synthetic logits avoid model downloads.

Approach:
    * Generate synthetic 3-class logits with controlled overconfidence.
    * Split into train/dev.
    * Fit T on train; evaluate ECE/MCE on dev.
"""
import numpy as np

from sro.metrics.calibration import (
    compute_ece,
    compute_mce,
    compute_nll_from_logits,
    fit_temperature_scalar,
    probabilize,
)


def _make_synthetic(N: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # True class indices
    y = rng.integers(0, 3, size=N)
    # Base logits ~ N(0,1)
    z = rng.normal(size=(N, 3))
    # Push the true class up to create overconfidence mismatch
    for i in range(N):
        z[i, y[i]] += rng.normal(loc=2.0, scale=0.5)
    return z.astype(np.float64), y.astype(np.int64)


def test_temperature_improves_calibration_and_preserves_argmax():
    logits, y = _make_synthetic(5000, seed=123)
    train_logits, dev_logits = logits[:3000], logits[3000:]
    train_y, dev_y = y[:3000], y[3000:]

    # Pre
    ece_before = compute_ece(probabilize(dev_logits), dev_y)
    mce_before = compute_mce(probabilize(dev_logits), dev_y)
    nll_before = compute_nll_from_logits(dev_logits, dev_y)

    # Fit T
    T = fit_temperature_scalar(train_logits, train_y)
    assert T > 0.0

    # Post
    dev_logits_cal = dev_logits / T
    ece_after = compute_ece(probabilize(dev_logits_cal), dev_y)
    mce_after = compute_mce(probabilize(dev_logits_cal), dev_y)
    nll_after = compute_nll_from_logits(dev_logits_cal, dev_y)

    # Improvements
    assert ece_after <= ece_before * 0.5  # ≥50% reduction
    assert mce_after <= mce_before
    assert nll_after <= nll_before

    # Argmax unchanged for >= 99% (in practice 100% with T>0 unless ties)
    argmax_same = (probabilize(dev_logits).argmax(1) == probabilize(dev_logits_cal).argmax(1)).mean()
    assert argmax_same >= 0.99
