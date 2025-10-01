"""
UB bounds & monotonicity tests:
  - UB must be in [0,1].
  - Increasing κ (kappa) must not decrease UB (monotone non-decreasing).
"""
from sro.prover.s4_ub import upper_bound, UBWeights

def test_ub_in_unit_interval_and_monotone_in_kappa():
    feats = {
        "max_p1": 0.6,
        "sum_p1": 0.8,
        "ce_max": 0.5,
        "entity_overlap": 0.3,
        "time_agreement": 0.7,
        "distance": 0.2,
        "section_novelty": 1.0,
    }
    w = UBWeights()
    ub0 = upper_bound(feats, kappa=0.00, w=w)
    ub1 = upper_bound(feats, kappa=0.05, w=w)
    ub2 = upper_bound(feats, kappa=0.20, w=w)

    assert 0.0 <= ub0 <= 1.0
    assert 0.0 <= ub1 <= 1.0
    assert 0.0 <= ub2 <= 1.0
    assert ub1 >= ub0 - 1e-12
    assert ub2 >= ub1 - 1e-12
