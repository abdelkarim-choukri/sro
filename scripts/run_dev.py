from __future__ import annotations

from sro.config import load_config
from sro.prover import SROProver
from sro.types import Claim, SentenceCandidate


def main():
    cfg = load_config()

    # Claim (string): short statement we must verify
    claim = Claim(claim_id="c1", text="The iPhone 15 was released in 2023.")

    # Candidates (list of SentenceCandidate):
    # Keep one strong support (s1), keep neutrals (s2, s6). Remove lines the MNLI head mislabels as contradiction.
    cands = [
        SentenceCandidate("s1", "Apple released the iPhone 15 lineup in September 2023.", "news:1", 0.90),
        SentenceCandidate("s2", "The iPhone 14 came out in 2022.", "news:2", 0.75),
        SentenceCandidate("s6", "The iPhone 15 series ships later in September 2023.", "press:1", 0.80),
    ]

    prover = SROProver(cfg, use_real_nli=True, batch_size=16)
    result = prover.prove(claim, cands)

    if result.status == "ACCEPT":
        p = result.proof
        print("ACCEPT")
        print("  claim_id:", p.claim_id)
        print("  leaves:", p.leaves)
        print("  score:", round(p.score, 3), "cmax:", round(p.cmax, 3), "margin:", round(p.margin, 3))
        print("  stop_reason:", p.stop_reason)
        for e in p.edges:
            print("  edge:", e.src, "→", e.dst, e.label, "pE=", round(e.p_entail, 3), "pC=", round(e.p_contradict, 3))
        print("  citations:", p.citations)
    else:
        print("REJECT:", result.reason)

if __name__ == "__main__":
    main()
    