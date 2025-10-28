# sro/prover/s7_safety.py
from typing import Tuple


def safe_to_ship(score_star: float, cmax: float, delta: float) -> tuple[bool, str]:
    """
    Safety rule: Ship only if (score_star - cmax) >= delta.
    All inputs are probabilities in [0,1].
    Returns (ok, reason_if_not).
    """
    for name, val in (("score_star", score_star), ("cmax", cmax), ("delta", delta)):
        if not (0.0 <= val <= 1.0):
            return False, f"INVALID_INPUT: {name}={val:.3f} not in [0,1]"

    margin = score_star - cmax
    if margin < delta:
        return False, (
            f"CONTRADICTION_BLOCK: margin={margin:.3f} < delta={delta:.3f} "
            f"(score*={score_star:.3f}, cmax={cmax:.3f})"
        )
    return True, ""
