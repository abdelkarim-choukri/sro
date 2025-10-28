from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# -----------------------
# Simple string enums
# -----------------------
Status = Literal["ACCEPT", "REJECT"]
RejectReason = Literal["NO_PROOF", "CONTRADICTION_BLOCK", "BUDGET_EXCEEDED", "NON_MINIMAL"]
StopReason = Literal["UB_BEATEN", "BUDGET_EXCEEDED", "ALT_DONE", "NO_PAIRS", "EARLY_TAU1"]
Label = Literal["entail", "neutral", "contradict"]

# -----------------------
# Core data types
# -----------------------

@dataclass(frozen=True)
class Claim:
    """Claim: a short statement we must verify."""
    claim_id: str
    text: str
    is_critical: bool = True  # If False, we can ship without proving it.


@dataclass(frozen=True)
class SentenceCandidate:
    """
    Candidate evidence sentence.
    Fields:
      sent_id: unique sentence identifier (e.g., "news:1#s3")
      text: the sentence text
      source_id: document/source id (e.g., "news:1")
      ce_score: fused retrieval/reranker score (normalized to [0,1] by caller)
    """
    sent_id: str
    text: str
    source_id: str
    ce_score: float = 0.0


@dataclass(frozen=True)
class Edge:
    """
    NLI edge between leaf(s) and the claim.
    For 1-hop: src = (sent_id,)
    For 2-hop: src = (sent_id_i, sent_id_j)
    """
    src: tuple[str, ...]        # ("s1",) or ("s1","s2")
    dst: str                    # claim_id
    label: Label                # "entail" | "neutral" | "contradict"
    p_entail: float             # entailment prob in [0,1]
    p_contradict: float         # contradiction prob in [0,1]
    model: str | None = None # optional model name (e.g., "roberta-large-mnli")


@dataclass
class ProofObject:
    claim_id: str
    leaves: list[str]
    edges: list[Edge]
    citations: list[dict[str, str]] = field(default_factory=list)
    score: float = 0.0
    cmax: float = 0.0
    margin: float = 0.0
    stop_reason: StopReason = "UB_BEATEN"
    alternation_used: bool = False
    # --- NEW telemetry ---
    accept_reason: str = ""          # "1hop@tau1" | "2hop@tau2"
    budget_used: int = 0             # # pair evals consumed
    ub_top_remaining: float = 0.0    # best UB left when stopping

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "leaves": list(self.leaves),
            "edges": [asdict(e) for e in self.edges],
            "citations": [dict(c) for c in self.citations],
            "score": float(self.score),
            "cmax": float(self.cmax),
            "margin": float(self.margin),
            "stop_reason": self.stop_reason,
            "alternation_used": bool(self.alternation_used),
            # new fields
            "accept_reason": self.accept_reason,
            "budget_used": int(self.budget_used),
            "ub_top_remaining": float(self.ub_top_remaining),
        }


@dataclass
class ProverResult:
    """
    Result of SRO-Prover:
      - ACCEPT with a ProofObject
      - REJECT with a reason
    """
    status: Status
    proof: ProofObject | None = None
    reason: RejectReason | None = None
