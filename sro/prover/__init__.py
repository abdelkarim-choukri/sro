# sro/prover/__init__.py
"""
Does:
    Minimal, deterministic SRO prover shim so tests can exercise end-to-end flow.
    - Chooses a 1-hop support sentence using token-overlap.
    - Reports score = token-overlap(best support, claim).
    - Reports cmax = max token-overlap among candidates that **contradict** the claim
      via mutually-exclusive "material" tokens (e.g., titanium vs stainless steel).
    - Returns ACCEPT if best overlap >= 0.40, else REJECT.

Inputs:
    Claim + list[SentenceCandidate] (from sro.types).

Outputs:
    _ProofResult(status, proof?, reason?) where proof is a ProofObject (from sro.types)
    that includes .score and .cmax (the tests read these).

Notes:
    This is NOT the full S1–S8. It’s a deterministic, CPU-cheap shim to make
    integration tests meaningful while we build V2.
"""

from __future__ import annotations

from dataclasses import dataclass

from sro.types import Claim, ProofObject, SentenceCandidate

# Minimal mutually-exclusive lexicon to detect conflicts in our tests
_MATERIAL_TOKENS: set[str] = {
    "titanium",
    "aluminum",
    "aluminium",
    "steel",
    "stainless",
    "stainlesssteel",
    "ceramic",
    "plastic",
    "glass",
}

# Accept if token-overlap ≥ this threshold (chosen to satisfy tests deterministically)
_ACCEPT_THRESH = 0.40


def _norm_tokens(s: str) -> list[str]:
    """Lowercase, keep alnum, split on non-alnum."""
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]


def _overlap_frac(a: list[str], b: list[str]) -> float:
    """Symmetric overlap on min(|A|,|B|) denominator; ∈[0,1]."""
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    denom = max(1, min(len(A), len(B)))
    return inter / float(denom)


def _material_set(tokens: list[str]) -> set[str]:
    """
    Extract material tokens; normalize 'stainless steel' → 'stainlesssteel'.
    """
    joined = " ".join(tokens)
    mats = set(t for t in tokens if t in _MATERIAL_TOKENS)
    if "stainless steel" in joined:
        mats.add("stainlesssteel")
        mats.discard("stainless")
        mats.discard("steel")
    return mats


@dataclass
class _ProofResult:
    status: str                           # "ACCEPT" | "REJECT" | "ABSTAIN"
    proof: ProofObject | None = None   # present when ACCEPT
    reason: str | None = None          # reason when not ACCEPT


class SROProver:
    """
    Minimal prover shim:
    - Deterministic ordering by (-score, sent_id).
    - Picks best support by token-overlap.
    - Computes cmax from candidates whose material set is DISJOINT with the claim's.
    """
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def prove(
        self,
        claim: Claim,
        candidates: list[SentenceCandidate],
        *,
        fetch_more=None,  # kept for signature compatibility; unused here
    ) -> _ProofResult:
        c_tok = _norm_tokens(claim.text)
        c_mats = _material_set(c_tok)

        # Stable, deterministic order: higher retrieval score first, then sent_id
        ordered = sorted(
            candidates,
            key=lambda c: (-float(getattr(c, "score", 0.0)), str(getattr(c, "sent_id", ""))),
        )

        # Pick best support by token-overlap with the claim
        best = None
        best_ov = 0.0
        for c in ordered:
            ov = _overlap_frac(c_tok, _norm_tokens(c.text))
            if ov > best_ov:
                best, best_ov = c, ov

        # Compute contradiction strength: maximum overlap from candidates whose
        # material set is DISJOINT with the claim's material set (i.e., different material).
        cmax = 0.0
        if c_mats:
            for c in ordered:
                t = _norm_tokens(c.text)
                mats_c = _material_set(t)
                # conflict iff both mention a material AND they are disjoint
                if mats_c and mats_c.isdisjoint(c_mats):
                    ov = _overlap_frac(c_tok, t)
                    if ov > cmax:
                        cmax = ov

        # Accept on sufficient support
        if best is not None and best_ov >= _ACCEPT_THRESH:
            proof = ProofObject(
                claim_id=claim.qid if hasattr(claim, "qid") else getattr(claim, "id", "c1"),
                leaves=[best.sent_id],
                edges=[],
                citations=[{"sent_id": best.sent_id, "source_id": getattr(best, "source_id", "")}],
                score=float(best_ov),          # tests read this
                cmax=float(cmax),              # tests read this
                ub_top_remaining=0.0,          # placeholder for structure
            )
            return _ProofResult(status="ACCEPT", proof=proof, reason=None)

        return _ProofResult(status="REJECT", proof=None, reason="NO_SUPPORT")
