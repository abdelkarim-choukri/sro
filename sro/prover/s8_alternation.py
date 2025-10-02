"""
S8 — Alternation policy (single re-retrieval).

Rule (spec-driven):
  Trigger exactly once if ALL hold:
    - best_so_far < tau2        (we don't have a 2-hop proof at threshold)
    - top_ub < tau2 + epsilon   (the remaining UB ceiling can't likely beat tau2)
    - budget_left > 0           (we still have pair eval budget to try again)

Definitions:
  - best_so_far: max(best 1-hop, best 2-hop) after first search pass
  - top_ub:      top UB remaining on the heap when we stopped
  - tau2:        two-hop accept threshold
  - epsilon:     slack margin (ε)
  - budget_left: B - evals_used (non-negative)
"""

from __future__ import annotations

def should_alternate(best_so_far: float, top_ub: float, tau2: float, epsilon: float, budget_left: int) -> bool:
    if budget_left <= 0:
        return False
    if best_so_far >= tau2:
        return False
    if top_ub >= (tau2 + epsilon):
        return False
    return True
