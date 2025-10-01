"""
Minimality test:
  A 2-hop pair {i,j} must be rejected if any leaf alone has p1 >= tau1.
"""
from sro.prover.s6_minimality import is_minimal

def test_minimality_rejects_when_leaf_crosses_tau1():
    tau1 = 0.75
    # Leaf i crosses tau1
    ok, why = is_minimal(0.80, 0.20, tau1)
    assert not ok
    assert "NON_MINIMAL" in why

def test_minimality_accepts_when_both_below_tau1():
    tau1 = 0.75
    ok, _ = is_minimal(0.74, 0.74, tau1)
    assert ok
