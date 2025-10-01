"""
SROProver — Orchestrate S1→S5 and enforce S6 (minimality) and S7 (safety).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from sro.types import SentenceCandidate, Claim, ProofEdge, ProofObject, ProverResult
from sro.config import Config, load_config
from sro.prover.s1_onehop import one_hop_scores as s1_one_hop
from sro.prover.s2_frontier import select_frontier_and_pool, _tokens as s2_tokens
from sro.prover.s3_features import build_pair_features
from sro.prover.s4_ub import UBWeights
from sro.prover.s5_bounded_search import bounded_search, TwoHopScorer
from sro.prover.s6_minimality import is_minimal
from sro.prover.s7_safety import safe_to_ship

# Optional NLI import for batched 2-hop (only used if use_real_nli=True)
try:
    from sro.nli.nli_infer import two_hop_scores as nli_two_hop
except Exception:
    nli_two_hop = None


class _TwoHopNLIScorer(TwoHopScorer):
    """Adapt nli_infer.two_hop_scores to the TwoHopScorer interface."""
    def __init__(self, claim_text: str, cand_texts: List[str], batch_size: int = 16):
        self.claim_text = claim_text
        self.cand_texts = cand_texts
        self.batch_size = int(batch_size)

    def score_pairs(self, pairs: List[Tuple[int, int]]) -> List[float]:
        if nli_two_hop is None:
            raise RuntimeError("two_hop_scores is not available")
        text_pairs = [(self.cand_texts[i], self.cand_texts[j]) for (i, j) in pairs]
        return nli_two_hop(self.claim_text, text_pairs, batch_size=self.batch_size)


class SROProver:
    def __init__(self, cfg: Optional[Config] = None, use_real_nli: bool = True, batch_size: int = 16):
        self.cfg = cfg or load_config()
        self.use_real_nli = bool(use_real_nli)
        self.batch_size = int(batch_size)

    def prove(self, claim: Claim, candidates: List[SentenceCandidate]) -> ProverResult:
        # --- Validate inputs ---
        if not isinstance(claim.text, str) or not claim.text.strip():
            return ProverResult(status="REJECT", reason="NO_PROOF")
        if not candidates:
            return ProverResult(status="REJECT", reason="NO_PROOF")

        knobs = self.cfg.sro_prover
        M, L, B = int(knobs.M), int(knobs.L), int(knobs.B)
        tau1, tau2, delta, kappa = float(knobs.tau1), float(knobs.tau2), float(knobs.delta), float(knobs.kappa)

        # --- S1: one-hop (model by default) ---
        p1, c1 = s1_one_hop(claim.text, candidates, use_model=self.use_real_nli, batch_size=self.batch_size)
        Cmax = max(c1) if c1 else 0.0
        if p1:
            best1_idx = max(range(len(p1)), key=lambda i: p1[i])
            best1_score = float(p1[best1_idx])
        else:
            best1_idx, best1_score = -1, 0.0

        # --- S2: frontier + pool ---
        frontier_idx, pool2_idx, token_cache = select_frontier_and_pool(candidates, p1, M=M, L=L, lambda_diversity=0.7)

        # --- Path: no pairs possible → consider 1-hop then safety ---
        if not frontier_idx or not pool2_idx:
            if best1_score >= tau1:
                ok_safe, _ = safe_to_ship(best1_score, Cmax, delta)
                if ok_safe:
                    edge = ProofEdge(
                        src=(candidates[best1_idx].sent_id,),
                        dst=claim.claim_id,
                        label="entail",
                        p_entail=best1_score,
                        p_contradict=c1[best1_idx],
                        model="mnli-1hop" if self.use_real_nli else "heuristic-1hop-stub",
                    )
                    proof = ProofObject(
                        claim_id=claim.claim_id,
                        leaves=[candidates[best1_idx].sent_id],
                        edges=[edge],
                        citations=[{"sent_id": candidates[best1_idx].sent_id, "source_id": candidates[best1_idx].source_id}],
                        score=best1_score,
                        cmax=Cmax,
                        margin=best1_score - Cmax,
                        stop_reason="NO_PAIRS",
                        alternation_used=False,
                    )
                    return ProverResult(status="ACCEPT", proof=proof)
                return ProverResult(status="REJECT", reason="CONTRADICTION_BLOCK")
            return ProverResult(status="REJECT", reason="NO_PROOF")

        # --- S3: pair features ---
        claim_tokens = list(s2_tokens(claim.text))
        pairs, feats = build_pair_features(claim_tokens, candidates, token_cache, frontier_idx, pool2_idx, p1)

        # --- S4/S5: bounded search (batch-capable two-hop if available) ---
        scorer = _TwoHopNLIScorer(claim.text, [c.text for c in candidates], batch_size=self.batch_size) \
                 if (self.use_real_nli and nli_two_hop is not None) else None

        best_pair, best_p2, evals, stop_reason = bounded_search(
            claim.text, candidates, pairs, feats,
            p1=p1, tau1=tau1, B=B, kappa=kappa,
            ub_weights=UBWeights(),
            two_hop_scorer=scorer,
            batch_size=self.batch_size,
        )

        # --- Decision: always allow 1-hop fallback if it clears τ₁ ---
        choice: Optional[Tuple[str, object]] = None  # ("1hop", idx) | ("2hop", (i,j))
        score_star = 0.0

        if best_pair is not None and best_p2 >= tau2:
            i, j = best_pair
            ok_min, _ = is_minimal(p1[i], p1[j], tau1)
            if ok_min:
                choice = ("2hop", (i, j))
                score_star = float(best_p2)

        # HARDENED FALLBACK: regardless of search outcome
        if (choice is None) and (best1_score >= tau1):
            choice = ("1hop", best1_idx)
            score_star = best1_score

        if choice is None:
            # Nothing passes thresholds
            if stop_reason == "BUDGET_EXCEEDED":
                return ProverResult(status="REJECT", reason="BUDGET_EXCEEDED")
            return ProverResult(status="REJECT", reason="NO_PROOF")

        # --- S7: safety margin ---
        ok_safe, _ = safe_to_ship(score_star, Cmax, delta)
        if not ok_safe:
            return ProverResult(status="REJECT", reason="CONTRADICTION_BLOCK")

        # --- Build proof ---
        if choice[0] == "1hop":
            idx = int(choice[1])
            leaves = [candidates[idx].sent_id]
            edges = [ProofEdge(
                src=(candidates[idx].sent_id,),
                dst=claim.claim_id,
                label="entail",
                p_entail=p1[idx],
                p_contradict=c1[idx],
                model="mnli-1hop" if self.use_real_nli else "heuristic-1hop-stub",
            )]
            citations = [{"sent_id": candidates[idx].sent_id, "source_id": candidates[idx].source_id}]
        else:
            i, j = choice[1]
            leaves = [candidates[i].sent_id, candidates[j].sent_id]
            edges = [ProofEdge(
                src=(candidates[i].sent_id, candidates[j].sent_id),
                dst=claim.claim_id,
                label="entail",
                p_entail=best_p2,
                p_contradict=max(c1[i], c1[j]),
                model="mnli-2hop" if self.use_real_nli else "heuristic-2hop-stub",
            )]
            citations = [
                {"sent_id": candidates[i].sent_id, "source_id": candidates[i].source_id},
                {"sent_id": candidates[j].sent_id, "source_id": candidates[j].source_id},
            ]

        proof = ProofObject(
            claim_id=claim.claim_id,
            leaves=leaves,
            edges=edges,
            citations=citations,
            score=score_star,
            cmax=Cmax,
            margin=score_star - Cmax,
            stop_reason=stop_reason,
            alternation_used=False,
        )
        return ProverResult(status="ACCEPT", proof=proof)
