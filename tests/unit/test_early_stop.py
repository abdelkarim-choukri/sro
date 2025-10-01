"""
Early stop test:
  If best_so_far (from 1-hop) is high and all pair UBs are below it,
  the search should stop with 0 evals and UB_BEATEN.
"""
from sro.types import SentenceCandidate
from sro.prover.s5_bounded_search import bounded_search
from sro.prover.s4_ub import UBWeights

def test_early_stop_without_evals():
    # Construct two candidates; p1 high for one of them
    cands = [
        SentenceCandidate("s1", "A strong single-hop support sentence.", "src1", 0.9),
        SentenceCandidate("s2", "Another sentence with weak overlap.", "src2", 0.1),
    ]
    # Pairs (only one possible pair)
    pairs = [(0,1)]
    # Features deliberately weak so UB < 0.9
    feats = [{
        "max_p1": 0.50,
        "sum_p1": 0.60,
        "ce_max": 0.30,
        "entity_overlap": 0.10,
        "time_agreement": 0.50,
        "distance": 0.30,  # 1 - distance = 0.70
        "section_novelty": 1.0,
    }]
    # p1 sets best_so_far baseline = 0.9
    p1 = [0.9, 0.2]

    best_pair, best_p2, evals, stop_reason = bounded_search(
        claim="irrelevant",
        candidates=cands,
        pairs=pairs,
        feats=feats,
        p1=p1,
        tau1=0.75,
        B=64,
        kappa=0.05,
        ub_weights=UBWeights(),
    )

    assert evals == 0, "No pair should be evaluated when top UB ≤ best_so_far"
    assert stop_reason in ("UB_BEATEN", "NO_PAIRS")
    assert best_pair is None
    assert best_p2 == 0.0
