# sro/safety/cross_claim.py
"""
Does:
    Enforce a global contradiction safety margin across all ACCEPTed claims in a draft.

Definitions:
    - best:           the best entailment score for the candidate claim (in [0,1]).
    - Cmax_local:     the maximum contradiction/conflict score found for THIS claim (in [0,1]).
    - Cmax_global:    the maximum contradiction/conflict score among all PREVIOUSLY ACCEPTED claims (in [0,1]).
    - margin:         best - max(Cmax_local, Cmax_global)
    - δ (delta):      required safety margin. ACCEPT only if margin >= δ.

Usage:
    state = CrossClaimSafety(delta=0.10)
    ok, info = state.check_and_maybe_update(best, c_local)
    if not ok: block and log info["reason"] == "SAFETY_MARGIN_FAIL"
    # When ok is True, state.cmax_global may update to max(old, Cmax_local).

Inputs:
    - Pure floats in [0,1]. No IO, no model calls.

Outputs:
    - (ok: bool, info: dict) with keys {margin, cmax_local, cmax_global, threshold, reason}.

Notes:
    - We UPDATE Cmax_global ONLY on ACCEPT (ok=True). Evaluations that are blocked do NOT
      pollute the global ceiling.
    - The decision rule is INCLUSIVE: margin >= δ is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

EPS = 1e-6  # tolerance for inclusive >= δ comparisons
@dataclass
class CrossClaimSafety:
    delta: float = 0.10 
    cmax_global: float = 0.0

    def reset(self) -> None:
        self.cmax_global = 0.0

    def check(self, best: float, cmax_local: float) -> dict[str, float | str]:
        """Compute margin and return a decision blob WITHOUT updating internal state."""
        b = max(0.0, min(1.0, float(best)))
        c_loc = max(0.0, min(1.0, float(cmax_local)))
        c_ceiling = max(self.cmax_global, c_loc)
        margin = b - c_ceiling
        # FIX: inclusive threshold with tolerance
        ok = bool((margin + EPS) >= float(self.delta))   # <-- changed
        return {
            "ok": ok,
            "margin": margin,
            "cmax_local": c_loc,
            "cmax_global": self.cmax_global,  # ceiling before potential update
            "threshold": float(self.delta),
            "reason": "OK" if ok else "SAFETY_MARGIN_FAIL",
        }

    def check_and_maybe_update(self, best: float, cmax_local: float):
        info = self.check(best, cmax_local)
        ok = bool(info["ok"])  # type: ignore
        if ok:
            self.cmax_global = max(self.cmax_global, float(info["cmax_local"]))  # type: ignore
            info["cmax_global"] = self.cmax_global
        return ok, info