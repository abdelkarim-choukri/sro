from __future__ import annotations

def should_alternate(
    best_so_far: float,
    top_ub: float,
    tau2: float,
    eps: float,
    budget_left: int,
    evals_first_pass: int | None = None,
) -> bool:
    """
    Decide if we should do a single alternation (targeted re-retrieval).

    v1 rule (robust, simple):
      If we did NOT meet tau2 in the first pass, allow ONE alternation,
      regardless of UB or remaining budget. This lets us surface a strong 1-hop
      after re-retrieval (common in our toy + small-corpus cases).

    We still rely on the prover to enforce "single alternation" globally.
    """
    # If already strong enough, no alternation.
    if best_so_far >= tau2:
        return False
    # Otherwise: do exactly one alternation (caller ensures "once").
    return True
