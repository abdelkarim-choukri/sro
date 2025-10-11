from sro.utils.random import set_all_seeds
from sro.config import load_config
from sro.prover import SROProver
from sro.types import Claim, SentenceCandidate

def build_tiny():
    cfg = load_config()
    prover = SROProver(cfg, use_real_nli=False, bs_nli1=4, bs_nli2=4)  # stub NLI path ok
    claim = Claim(claim_id="c1", text="Demo claim.")
    cands = [
        SentenceCandidate(sent_id="s1", text="A.", source_id="d:1", ce_score=0.5),
        SentenceCandidate(sent_id="s2", text="B.", source_id="d:2", ce_score=0.6),
    ]
    return prover, claim, cands

def test_same_outcome_and_leaves():
    set_all_seeds(42)
    prover, claim, cands = build_tiny()
    r1 = prover.prove(claim, cands)
    set_all_seeds(42)
    prover, claim, cands = build_tiny()
    r2 = prover.prove(claim, cands)
    assert r1.status == r2.status
    if r1.status == "ACCEPT" and r2.status == "ACCEPT":
        assert r1.proof.leaves == r2.proof.leaves
