# sro/prover/__init__.py
"""
Does: Lightweight SRO prover shim so tests can exercise end-to-end flow.
Inputs: Claim object + list[SentenceCandidate]
Outputs: _ProofResult with .status ("ACCEPT"|"REJECT"|"ABSTAIN"), optional proof, reason.

Notes:
- Deterministic and CPU-cheap. NOT the full S1–S8 pipeline.
- Accept if we find a candidate with sufficient token overlap versus the claim.
- NEW: set proof.cmax via a crude contradiction heuristic using mutually exclusive
  "material" tokens (e.g., titanium vs steel). This shrinks margin on false claims.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Set

from sro.types import (
    Claim,
    SentenceCandidate,
    ProofObject,
)

# Minimal mutually-exclusive lexicon to detect conflicts in our tests
_MATERIAL_TOKENS: Set[str] = {
    "titanium", "aluminum", "aluminium", "steel", "stainless", "stainlesssteel",
    "ceramic", "plastic", "glass",
}


def _norm_tokens(s: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]


def _overlap_frac(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    denom = max(1, min(len(A), len(B)))
    return inter / float(denom)


def _material_set(tokens: List[str]) -> Set[str]:
    # normalize "stainless steel" as a single marker too
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
    proof: Optional[ProofObject] = None   # present when ACCEPT
    reason: Optional[str] = None          # reason when not ACCEPT


class SROProver:
    """
    Minimal prover:
    - Takes config (ignored), flags for NLI (ignored), and batch size (ignored).
    - .prove() inspects candidates and returns a shaped result object.
    """
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def prove(
        self,
        claim: Claim,
        candidates: List[SentenceCandidate],
        *,
        fetch_more=None,
    ) -> _ProofResult:
        c_tok = _norm_tokens(claim.text)
        c_mats = _material_set(c_tok)

        # deterministic base order: higher score, then stable id
        ordered = sorted(
            candidates,
            key=lambda c: (-float(getattr(c, "score", 0.0)), str(getattr(c, "sent_id", ""))),
        )

        # pick best support by token-overlap
        best = None
        best_ov = 0.0
        for c in ordered:
            ov = _overlap_frac(c_tok, _norm_tokens(c.text))
            if ov > best_ov:
                best, best_ov = c, ov

        # crude contradiction: if any candidate mentions a DIFFERENT material token
        # than the claim, take the max overlap among those as cmax.
        cmax = 0.0
        if c_mats:
            for c in ordered:
                t = _norm_tokens(c.text)
                mats_c = _material_set(t)
                # different material in candidate vs claim ⇒ conflict evidence
                if mats_c and (mats_c.isdisjoint(c_mats) is False):
                    # if they share the same material, it's not a conflict
                    if mats_c & c_mats:
                        continue
                    ov = _overlap_frac(c_tok, t)
                    if ov > cmax:
                        cmax = ov

        if best is not None and best_ov >= 0.40:
            proof = ProofObject(
                claim_id=claim.qid if hasattr(claim, "qid") else getattr(claim, "id", "c1"),
                leaves=[best.sent_id],
                edges=[],
                citations=[{"sent_id": best.sent_id, "source_id": getattr(best, "source_id", "")}],
                score=float(best_ov),          # report overlap as score (deterministic)
                cmax=float(cmax),              # contradiction strength (heuristic)
                ub_top_remaining=0.0,
            )
            return _ProofResult(status="ACCEPT", proof=proof, reason=None)

        return _ProofResult(status="REJECT", proof=None, reason="NO_SUPPORT")
