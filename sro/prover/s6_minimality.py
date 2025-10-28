# sro/prover/s6_minimality.py


def is_minimal(
    p1_i: float,
    p1_j: float,
    tau1: float,
    leaf_i: str | None = None,
    leaf_j: str | None = None,
) -> tuple[bool, str]:
    """
    Minimality rule (2-hop): valid iff NEITHER leaf alone crosses tau1.
    Returns (ok, reason_if_not).
    p1_i, p1_j, tau1 are probabilities in [0,1].
    leaf_i/leaf_j are optional IDs used in the reason string.
    """
    # Fast input checks (defensive; cheap)
    for name, val in (("p1_i", p1_i), ("p1_j", p1_j), ("tau1", tau1)):
        if not (0.0 <= val <= 1.0):
            return False, f"INVALID_INPUT: {name}={val:.3f} not in [0,1]"

    viol = []
    if p1_i >= tau1:
        viol.append(f"{leaf_i or 'i'} p1={p1_i:.3f} >= tau1={tau1:.3f}")
    if p1_j >= tau1:
        viol.append(f"{leaf_j or 'j'} p1={p1_j:.3f} >= tau1={tau1:.3f}")

    if viol:
        return False, "NON_MINIMAL: " + "; ".join(viol)
    return True, ""
