"""
Quick NLI sanity check on GPU/CPU.

Run:
  python scripts/test_nli.py
"""

from __future__ import annotations

from sro.nli.nli_infer import one_hop_scores, two_hop_scores


def main():
    claim = "The iPhone 15 was released in 2023."
    sents = [
        "Apple released the iPhone 15 lineup in September 2023.",
        "The iPhone 14 came out in 2022.",
        "Apple did not release any iPhone in 2023.",
    ]
    p1, c1 = one_hop_scores(claim, sents, batch_size=16)
    print("one-hop entail:", [round(x, 3) for x in p1])
    print("one-hop contradict:", [round(x, 3) for x in c1])

    pairs = [
        (sents[1], "In 2023, Apple introduced a new iPhone model."),
        (sents[2], "However, later sources confirmed a 2023 release."),
    ]
    p2 = two_hop_scores(claim, pairs, batch_size=16)
    print("two-hop entail:", [round(x, 3) for x in p2])

if __name__ == "__main__":
    main()
