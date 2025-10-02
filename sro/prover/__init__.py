"""
SROProver — now with single alternation + JSONL proof logging.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Callable, Iterable
import json
from datetime import datetime

from sro.types import SentenceCandidate, Claim, ProofEdge, ProofObject, ProverResult
from sro.config import Config, load_config
from sro.prover.s1_onehop import one_hop_scores as s1_one_hop
from sro.prover.s2_frontier import select_frontier_and_pool, _tokens as s2_tokens
from sro.prover.s3_features import build_pair_features
from sro.prover.s4_ub import UBWeights
from sro.prover.s5_bounded_search import bounded_search, TwoHopScorer
from sro.prover.s6_minimality import is_minimal
from sro.prover.s7_safety import safe_to_ship
from sro.prover.s8_alternation import should_alternate

# Optional NLI import for batched 2-hop (used if use_real_nli=True)
try:
    from sro.nli.nli_infer import two_hop_scores as nli_two_hop
except Exception:
    nli_two_hop = None


class _TwoHopNLIScorer(TwoHopScorer):
    def __init__(self, claim_text: str, cand_texts: List[str], batch_size: int = 16):
        self.claim_text = claim_text
        self.cand_texts = cand_texts
        self.batch_size = int(batch_size)

    def score_pairs(self, pairs: List[Tuple[int, int]]) -> List[float]:
        if nli_two_hop is None:
            raise RuntimeError("two_hop_scores is not available")
        text_pairs = [(self.cand_texts[i], self.cand_texts[j]) for (i, j) in pairs]
        return nli_two_hop(self.claim_text, text_pairs, batch_size=self.batch_size)


def _default_fetch_more(claim: str, candidates: List[SentenceCandidate], **_: object) -> List[SentenceCandidate]:
    """
    Minimal demo fetcher: returns a targeted support sentence for common toy claims.
    In real use, replace this with retrieval (BM25+dense) focused on the claim.
    """
    text = claim.lower()
    extras: List[SentenceCandidate] = []
    if "titanium frame" in text or "titanium" in text:
        extras.append(SentenceCandidate(
            sent_id="alt1",
            text="Apple announced that the iPhone 15 Pro features a titanium frame.",
            source_id="alt:retrieval",
            ce_score=0.85,
        ))
    elif "released in 2023" in text and "iphone 15" in text:
        extras.append(SentenceCandidate(
            sent_id="alt2",
            text="Press materials confirm the iPhone 15 was released in September 2023.",
            source_id="alt:retrieval",
            ce_score=0.82,
        ))
    else:
        # Generic (weak) fallback
        extras.append(SentenceCandidate(
            sent_id="alt_generic",
            text=f"Reports indicate: {claim}",
            source_id="alt:generic",
            ce_score=0.60,
        ))
    return extras


from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from datetime import datetime

def _log_proof(cfg: Config, proof: ProofObject, status: str = "ACCEPT") -> None:
    """
    Append a single JSONL record to artifacts/proofs/proofs.jsonl.
    Robust: serialize dataclasses; never crash the caller.
    """
    try:
        out_dir: Path = cfg.paths.proofs_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "proofs.jsonl"

        # Prefer full dataclass serialization (edges/leaves/citations included)
        if is_dataclass(proof):
            payload = asdict(proof)
        else:
            # Fallback — minimal but valid JSON
            payload = {
                "claim_id": getattr(proof, "claim_id", None),
                "leaves": getattr(proof, "leaves", None),
                "score": getattr(proof, "score", None),
                "cmax": getattr(proof, "cmax", None),
                "margin": getattr(proof, "margin", None),
                "stop_reason": getattr(proof, "stop_reason", None),
                "alternation_used": getattr(proof, "alternation_used", None),
            }

        record = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "status": status,
            "proof": payload,
        }

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Don’t break the pipeline if logging fails.
        pass


class SROProver:
    def __init__(self, cfg: Optional[Config] = None, use_real_nli: bool = True, batch_size: int = 16):
        self.cfg = cfg or load_config()
        self.use_real_nli = bool(use_real_nli)
        self.batch_size = int(batch_size)

    def prove(
        self,
        claim: Claim,
        candidates: List[SentenceCandidate],
        fetch_more: Optional[Callable[..., List[SentenceCandidate]]] = None,
    ) -> ProverResult:
        if not isinstance(claim.text, str) or not claim.text.strip():
            return ProverResult(status="REJECT", reason="NO_PROOF")
        if not candidates:
            return ProverResult(status="REJECT", reason="NO_PROOF")

        knobs = self.cfg.sro_prover
        M, L, B = int(knobs.M), int(knobs.L), int(knobs.B)
        tau1, tau2, delta, kappa, eps = float(knobs.tau1), float(knobs.tau2), float(knobs.delta), float(knobs.kappa), float(knobs.epsilon)

        # ---------- First pass ----------
        p1, c1 = s1_one_hop(claim.text, candidates, use_model=self.use_real_nli, batch_size=self.batch_size)
        Cmax = max(c1) if c1 else 0.0
        best1_idx = max(range(len(p1)), key=lambda i: p1[i]) if p1 else -1
        best1_score = p1[best1_idx] if best1_idx >= 0 else 0.0

        frontier_idx, pool2_idx, token_cache = select_frontier_and_pool(candidates, p1, M=M, L=L, lambda_diversity=0.7)
        claim_tokens = list(s2_tokens(claim.text))
        pairs, feats = build_pair_features(claim_tokens, candidates, token_cache, frontier_idx, pool2_idx, p1)

        scorer = _TwoHopNLIScorer(claim.text, [c.text for c in candidates], batch_size=self.batch_size) \
                 if (self.use_real_nli and nli_two_hop is not None) else None

        best_pair, best_p2, evals, stop_reason, top_ub_rem = bounded_search(
            claim.text, candidates, pairs, feats,
            p1=p1, tau1=tau1, B=B, kappa=kappa,
            ub_weights=UBWeights(),
            two_hop_scorer=scorer,
            batch_size=self.batch_size,
        )

        # Choose (first pass)
        choice: Optional[Tuple[str, object]] = None
        score_star = 0.0
        if best_pair is not None and best_p2 >= tau2:
            i, j = best_pair
            ok_min, _ = is_minimal(p1[i], p1[j], tau1)
            if ok_min:
                choice = ("2hop", (i, j))
                score_star = float(best_p2)
        if choice is None and best1_score >= tau1:
            choice = ("1hop", best1_idx)
            score_star = float(best1_score)

        # If we already have a choice, run safety + ship/abstain
        if choice is not None:
            ok_safe, _ = safe_to_ship(score_star, Cmax, delta)
            if not ok_safe:
                return ProverResult(status="REJECT", reason="CONTRADICTION_BLOCK")
            proof = self._build_proof(choice, claim, candidates, p1, c1, score_star, Cmax, stop_reason, alternation_used=False)
            _log_proof(self.cfg, proof, status="ACCEPT")
            return ProverResult(status="ACCEPT", proof=proof)

        # ---------- Alternation gate ----------
        best_so_far = max(best1_score, best_p2)
        budget_left = max(0, B - evals)
        if fetch_more is None:
            fetch_more = _default_fetch_more

        if should_alternate(best_so_far, top_ub_rem, tau2, eps, budget_left):
            # Get a few extra targeted sentences
            extra = fetch_more(
                claim=claim.text,
                candidates=candidates,
                frontier_idx=frontier_idx,
                pool2_idx=pool2_idx,
                p1=p1,
                top_ub=top_ub_rem,
            ) or []
            # Merge (de-dup by sent_id)
            existing_ids = {c.sent_id for c in candidates}
            extra = [e for e in extra if e.sent_id not in existing_ids]
            if extra:
                candidates2 = candidates + extra
                # Re-run S1–S5 once with remaining budget
                p1b, c1b = s1_one_hop(claim.text, candidates2, use_model=self.use_real_nli, batch_size=self.batch_size)
                Cmax_b = max(c1b) if c1b else 0.0
                frontier2, pool22, token_cache2 = select_frontier_and_pool(candidates2, p1b, M=M, L=L, lambda_diversity=0.7)
                claim_tokens2 = list(s2_tokens(claim.text))
                pairs2, feats2 = build_pair_features(claim_tokens2, candidates2, token_cache2, frontier2, pool22, p1b)
                scorer2 = _TwoHopNLIScorer(claim.text, [c.text for c in candidates2], batch_size=self.batch_size) \
                          if (self.use_real_nli and nli_two_hop is not None) else None
                best_pair2, best_p22, evals2, stop_reason2, _ = bounded_search(
                    claim.text, candidates2, pairs2, feats2,
                    p1=p1b, tau1=tau1, B=budget_left, kappa=kappa,
                    ub_weights=UBWeights(),
                    two_hop_scorer=scorer2,
                    batch_size=self.batch_size,
                )

                # Decide after alternation
                choice2: Optional[Tuple[str, object]] = None
                score_star2 = 0.0
                if best_pair2 is not None and best_p22 >= tau2:
                    i, j = best_pair2
                    ok_min, _ = is_minimal(p1b[i], p1b[j], tau1)
                    if ok_min:
                        choice2 = ("2hop", (i, j))
                        score_star2 = float(best_p22)
                # Always allow 1-hop fallback on second pass
                best1_idx2 = max(range(len(p1b)), key=lambda i: p1b[i]) if p1b else -1
                best1_score2 = p1b[best1_idx2] if best1_idx2 >= 0 else 0.0
                if choice2 is None and best1_score2 >= tau1:
                    choice2 = ("1hop", best1_idx2)
                    score_star2 = float(best1_score2)

                if choice2 is not None:
                    ok_safe, _ = safe_to_ship(score_star2, Cmax_b, delta)
                    if not ok_safe:
                        return ProverResult(status="REJECT", reason="CONTRADICTION_BLOCK")
                    proof = self._build_proof(choice2, claim, candidates2, p1b, c1b, score_star2, Cmax_b, stop_reason2, alternation_used=True)
                    _log_proof(self.cfg, proof, status="ACCEPT")
                    return ProverResult(status="ACCEPT", proof=proof)

        # If we get here, no proof after alternation
        return ProverResult(status="REJECT", reason="NO_PROOF")

    # ---------- helpers ----------

    def _build_proof(
        self,
        choice: Tuple[str, object],
        claim: Claim,
        candidates: List[SentenceCandidate],
        p1: List[float],
        c1: List[float],
        score_star: float,
        Cmax: float,
        stop_reason: str,
        alternation_used: bool,
    ) -> ProofObject:
        if choice[0] == "1hop":
            idx = int(choice[1])
            edges = [ProofEdge(
                src=(candidates[idx].sent_id,),
                dst=claim.claim_id,
                label="entail",
                p_entail=p1[idx],
                p_contradict=c1[idx],
                model="mnli-1hop" if self.use_real_nli else "heuristic-1hop-stub",
            )]
            leaves = [candidates[idx].sent_id]
            citations = [{"sent_id": candidates[idx].sent_id, "source_id": candidates[idx].source_id}]
        else:
            i, j = choice[1]
            edges = [ProofEdge(
                src=(candidates[i].sent_id, candidates[j].sent_id),
                dst=claim.claim_id,
                label="entail",
                p_entail=score_star,
                p_contradict=max(c1[i], c1[j]),
                model="mnli-2hop" if self.use_real_nli else "heuristic-2hop-stub",
            )]
            leaves = [candidates[i].sent_id, candidates[j].sent_id]
            citations = [
                {"sent_id": candidates[i].sent_id, "source_id": candidates[i].source_id},
                {"sent_id": candidates[j].sent_id, "source_id": candidates[j].source_id},
            ]

        return ProofObject(
            claim_id=claim.claim_id,
            leaves=leaves,
            edges=edges,
            citations=citations,
            score=score_star,
            cmax=Cmax,
            margin=score_star - Cmax,
            stop_reason=stop_reason,
            alternation_used=alternation_used,
        )
