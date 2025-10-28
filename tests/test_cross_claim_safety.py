# tests/test_cross_claim_safety.py
from sro.safety.cross_claim import CrossClaimSafety


def test_accept_updates_global():
    s = CrossClaimSafety(delta=0.10)
    ok, info = s.check_and_maybe_update(best=0.85, cmax_local=0.60)  # margin = 0.25
    assert ok is True
    assert abs(s.cmax_global - 0.60) < 1e-6
    assert info["reason"] == "OK"

def test_reject_does_not_update_global():
    s = CrossClaimSafety(delta=0.10)
    s.cmax_global = 0.60
    ok, info = s.check_and_maybe_update(best=0.68, cmax_local=0.65)  # margin = 0.08 < 0.10
    assert ok is False
    assert abs(s.cmax_global - 0.60) < 1e-6
    assert info["reason"] == "SAFETY_MARGIN_FAIL"

def test_inclusive_threshold_allows_equal():
    s = CrossClaimSafety(delta=0.10)
    s.cmax_global = 0.60
    ok, info = s.check_and_maybe_update(best=0.70, cmax_local=0.60)  # margin = 0.10 == delta
    assert ok is True
    assert abs(s.cmax_global - 0.60) < 1e-6  # local=0.60; global stays 0.60
    assert info["reason"] == "OK"

def test_second_accept_keeps_max_global():
    s = CrossClaimSafety(delta=0.10)
    ok1, _ = s.check_and_maybe_update(best=0.80, cmax_local=0.50)  # accept, global=0.50
    assert ok1
    ok2, _ = s.check_and_maybe_update(best=0.78, cmax_local=0.40)  # accept, global should remain 0.50
    assert ok2
    assert abs(s.cmax_global - 0.50) < 1e-6
