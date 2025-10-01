from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Literal, Dict, Any

# Enumerations (Literal keeps it simple and serializable)
Status = Literal["ACCEPT", "REJECT"]
RejectReason = Literal["NO_PROOF", "CONTRADICTION_BLOCK", "BUDGET_EXCEEDED", "NON_MINIMAL"]
StopReason = Literal["UB_BEATEN", "BUDGET_EXCEEDED", "ALT_DONE", "NO_PAIRS", "EARLY_TAU1"]
Label = Literal["entail", "neutral", "contradict"]

# Claim: a short statement we must verify.
@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    is_critical: bool = True  # if False, we can ship without proving it

# SentenceCandidate: a candidate evidence sentence.
@dataclass(frozen=True)
class SentenceCandidate:
    sent_id: str
    text: str
    source_id: str
    ce_score: float  # cross-encoder/reranker score (arbitrary scale)

# Citation: normalized reference we log + attach to leaves
@dataclass(frozen=True)
class Citation:
    sent_id: str
    source_id: str

# ProofEdge: NLI result between leaf(s) and the claim.
@dataclass(frozen=True)
class ProofEdge:
    src: Tuple[str, ...]          # ("s1",) for 1-hop or ("s1","s2") for 2-hop
    dst: str                      # claim_id
    label: Label                  # "entail" | "neutral" | "contradict"
    p_entail: float               # [0,1]
    p_contradict: float           # [0,1]
    model: Optional[str] = None   # e.g., "deberta-v3-large-mnli"

# ProofObject: everything we log when we ACCEPT.
@dataclass
class ProofObject:
    claim_id: str
    leaves: List[str]                   # leaf sentence IDs (1 or 2)
    edges: List[ProofEdge]              # NLI edge(s)
    citations: List[Citation]           # one per leaf in order
    score: float                        # chosen proof score (best p1 or p2)
    cmax: float                         # strongest one-hop contradiction across candidates
    margin: float                       # score - cmax
    stop_reason: StopReason             # why the search stopped
    alternation_used: bool = False      # whether alternation ran

    def as_dict(self) -> Dict[str, Any]:
        # JSONL-safe view for logging
        return {
            "claim_id": self.claim_id,
            "leaves": list(self.leaves),
            "edges": [asdict(e) for e in self.edges],
            "citations": [asdict(c) for c in self.citations],
            "score": float(self.score),
            "cmax": float(self.cmax),
            "margin": float(self.margin),
            "stop_reason": self.stop_reason,
            "alternation_used": bool(self.alternation_used),
        }

# ProverResult: ACCEPT with proof, or REJECT with reason.
@dataclass
class ProverResult:
    status: Status
    proof: Optional[ProofObject] = None
    reason: Optional[RejectReason] = None
