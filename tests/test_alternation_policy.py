# tests/test_alternation_policy.py
from sro.prover.s8_alternation import AlternationPolicy, PolicyMeta


def _policy(tau1=0.75, delta=0.10, thr=0.5):
    meta = PolicyMeta(
        features=("best_so_far","top_ub","budget_left_norm","frontier_entropy","ub_bandwidth"),
        weights=(1.0, 0.0, 0.0, 0.0, 0.0),  # only best_so_far matters for sigmoid in this toy
        bias=0.0, threshold=thr, tau1=tau1, delta=delta
    )
    return AlternationPolicy(meta)

def test_minimality_blocks():
    pol = _policy()
    dec = pol.decide({"best_so_far":0.80, "top_ub":0.90}, alternations_used=0)
    assert dec["alternate"] is False and dec["reason"] == "MINIMALITY_BLOCK"

def test_one_alt_cap():
    pol = _policy()
    dec = pol.decide({"best_so_far":0.10, "top_ub":0.95}, alternations_used=1)
    assert dec["alternate"] is False and dec["reason"] == "ONE_ALT_CAP"

def test_ub_beaten_blocks():
    pol = _policy(delta=0.10)
    # UB - best = 0.09 < delta → block
    dec = pol.decide({"best_so_far":0.50, "top_ub":0.59}, alternations_used=0)
    assert dec["alternate"] is False and dec["reason"] == "UB_BEATEN"

def test_policy_fires_when_headroom_and_not_minimal():
    # Force headroom and below minimality; set weights so sigmoid > threshold
    meta = PolicyMeta(
        features=("best_so_far","top_ub","budget_left_norm","frontier_entropy","ub_bandwidth"),
        weights=(0.0, 0.0, 0.0, 0.0, 10.0),  # strong positive on bandwidth
        bias=0.0, threshold=0.5, tau1=0.75, delta=0.10
    )
    pol = AlternationPolicy(meta)
    dec = pol.decide({"best_so_far":0.40, "top_ub":0.70, "ub_bandwidth":0.30}, alternations_used=0)
    assert dec["reason"] in ("OK","POLICY_REJECT")
    assert dec["alternate"] is True  # with 10x weight on bandwidth, this should tip over threshold
