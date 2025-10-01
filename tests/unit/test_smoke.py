# tests/unit/test_minimality_and_safety_boundaries.py
from sro.prover.s6_minimality import is_minimal
from sro.prover.s7_safety import safe_to_ship

def test_minimality_boundary_eq_tau1_is_non_minimal():
    ok, why = is_minimal(0.75, 0.10, 0.75)  # p1_i == tau1
    assert not ok and "NON_MINIMAL" in why

def test_safety_boundary_eq_delta_is_ok():
    ok, why = safe_to_ship(score_star=0.80, cmax=0.70, delta=0.10)  # margin == delta
    assert ok and why == ""
