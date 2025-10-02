"""
Demo that forces alternation to trigger once and then ACCEPT via 1-hop.

We start with candidates that *lack* the key support.
fetch_more() injects a targeted sentence, then the prover ships.
"""

from __future__ import annotations
from sro.config import load_config
from sro.types import Claim, SentenceCandidate
from sro.prover import SROProver

def my_fetch_more(**kwargs):
    # kwargs: claim, candidates, frontier_idx, pool2_idx, p1, top_ub
    return [
        SentenceCandidate(
            "alt1",
            "Apple announced that the iPhone 15 Pro features a titanium frame.",
            "alt:retrieval",
            0.85,
        )
    ]

def main():
    cfg = load_config()
    claim = Claim(claim_id="c_ti", text="The iPhone 15 Pro has a titanium frame.")

    # Initial pool intentionally lacks the key 'titanium frame' sentence
    cands = [
        SentenceCandidate("s1", "Apple released the iPhone 15 lineup in 2023.", "news:1", 0.90),
        SentenceCandidate("s2", "Preorders for iPhone 15 began in mid-September.", "news:2", 0.70),
        SentenceCandidate("s3", "Shipments started later in September 2023.", "press:1", 0.80),
    ]

    prover = SROProver(cfg, use_real_nli=True, batch_size=16)
    result = prover.prove(claim, cands, fetch_more=my_fetch_more)

    if result.status == "ACCEPT":
        p = result.proof
        print("ACCEPT (after alternation)" if p.alternation_used else "ACCEPT")
        print("  alternation_used:", p.alternation_used)
        print("  leaves:", p.leaves)
        print("  score:", round(p.score, 3), "cmax:", round(p.cmax, 3), "margin:", round(p.margin, 3))
        for e in p.edges:
            print("  edge:", e.src, "→", e.dst, e.label, "pE=", round(e.p_entail, 3), "pC=", round(e.p_contradict, 3))
    else:
        print("REJECT:", result.reason)

if __name__ == "__main__":
    main()
