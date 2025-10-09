"""
SROProver — single alternation + JSONL proof logging.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from datetime import datetime, timezone

from sro.types import SentenceCandidate, Claim, ProofObject, ProverResult, Edge
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
    """Thin adapter to real 2-hop NLI scorer."""
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
        extras.append(SentenceCandidate(
            sent_id="alt_generic",
            text=f"Reports indicate: {claim}",
            source_id="alt:generic",
            ce_score=0.60,
        ))
    return extras


def _citation_dicts_from_leaves(leaves: List[str],
                                candidates: List[SentenceCandidate]) -> List[Dict[str, str]]:
    """
    Build [{'sent_id': str, 'source_id': str}, ...] for the given leaf IDs.
    If a leaf ID is not found, return the ID with empty source_id to keep schema stable.
    """
    idx_by_id = {c.sent_id: c for c in candidates}
    cites: List[Dict[str, str]] = []
    for sid in leaves:
        c = idx_by_id.get(sid)
        cites.append({"sent_id": sid, "source_id": (c.source_id if c else "")})
    return cites


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
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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

        # ---------- S1: one-hop ----------
        p1, c1 = s1_one_hop(claim.text, candidates, use_model=self.use_real_nli, batch_size=self.batch_size)
        Cmax = max(c1) if c1 else 0.0
        best1_idx = max(range(len(p1)), key=lambda i: p1[i]) if p1 else -1
        best1_score = p1[best1_idx] if best1_idx >= 0 else 0.0

        # ---------- S2–S3: frontier, pool, features ----------
        frontier_idx, pool2_idx, token_cache = select_frontier_and_pool(candidates, p1, M=M, L=L, lambda_diversity=0.7)
        claim_tokens = list(s2_tokens(claim.text))
        pairs, feats = build_pair_features(claim_tokens, candidates, token_cache, frontier_idx, pool2_idx, p1)

        # ---------- S4–S5: UB + bounded search ----------
        scorer = _TwoHopNLIScorer(claim.text, [c.text for c in candidates], batch_size=self.batch_size) \
                 if (self.use_real_nli and nli_two_hop is not None) else None

        best_pair, best_p2, evals, stop_reason, top_ub_rem = bounded_search(
            claim.text, candidates, pairs, feats,
            p1=p1, tau1=tau1, B=B, kappa=kappa,
            ub_weights=UBWeights(),
            two_hop_scorer=scorer,
            batch_size=self.batch_size,
        )

        # ---------- Choose (first pass) ----------
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

        # ---------- S7: Safety + ship ----------
        if choice is not None:
            ok_safe, _ = safe_to_ship(score_star, Cmax, delta)
            if not ok_safe:
                return ProverResult(status="REJECT", reason="CONTRADICTION_BLOCK")
            acc_reason = "2hop@tau2" if choice[0] == "2hop" else "1hop@tau1"
            proof = self._build_proof(
                choice, claim, candidates, p1, c1,
                score_star, Cmax, stop_reason,
                alternation_used=False,
                accept_reason=acc_reason,
                budget_used=evals,
                ub_top_remaining=top_ub_rem,
            )
            _log_proof(self.cfg, proof, status="ACCEPT")
            return ProverResult(status="ACCEPT", proof=proof)
        # ---------- S8: Alternation gate ----------
        best_so_far = max(best1_score, best_p2)
        budget_left = max(0, B - evals)
        if fetch_more is None:
            fetch_more = _default_fetch_more

        if should_alternate(best_so_far, top_ub_rem, tau2, eps, budget_left):
            extra = fetch_more(
                claim=claim.text,
                candidates=candidates,
                frontier_idx=frontier_idx,
                pool2_idx=pool2_idx,
                p1=p1,
                top_ub=top_ub_rem,
            ) or []

            existing_ids = {c.sent_id for c in candidates}
            extra = [e for e in extra if e.sent_id not in existing_ids]

            if extra:
                candidates2 = candidates + extra
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

                choice2: Optional[Tuple[str, object]] = None
                score_star2 = 0.0
                if best_pair2 is not None and best_p22 >= tau2:
                    i, j = best_pair2
                    ok_min, _ = is_minimal(p1b[i], p1b[j], tau1)
                    if ok_min:
                        choice2 = ("2hop", (i, j))
                        score_star2 = float(best_p22)

                best1_idx2 = max(range(len(p1b)), key=lambda i: p1b[i]) if p1b else -1
                best1_score2 = p1b[best1_idx2] if best1_idx2 >= 0 else 0.0
                if choice2 is None and best1_score2 >= tau1:
                    choice2 = ("1hop", best1_idx2)
                    score_star2 = float(best1_score2)

                if choice2 is not None:
                    ok_safe, _ = safe_to_ship(score_star2, Cmax_b, delta)
                    if not ok_safe:
                        return ProverResult(status="REJECT", reason="CONTRADICTION_BLOCK")

                    acc_reason = "2hop@tau2" if choice2[0] == "2hop" else "1hop@tau1"
                    proof = self._build_proof(
                        choice2, claim, candidates2, p1b, c1b,
                        score_star2, Cmax_b, stop_reason2,
                        alternation_used=True,
                        accept_reason=acc_reason,
                        budget_used=evals + evals2,
                        ub_top_remaining=0.0,
                    )
                    _log_proof(self.cfg, proof, status="ACCEPT")
                    return ProverResult(status="ACCEPT", proof=proof)

        # ---- ALWAYS return a ProverResult if we didn't accept above ----
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
        *,
        alternation_used: bool = False,
        accept_reason: str = "",
        budget_used: int = 0,
        ub_top_remaining: float = 0.0,
    ) -> ProofObject:
        """
        Construct a ProofObject with standardized dict citations.
        - For 1-hop: leaves = [sent_id_i]; edge = ((si,), claim_id, 'entail', p1[i], c1[i])
        - For 2-hop: leaves = [si, sj]; edge = ((si, sj), claim_id, 'entail', p2, max(c1[i], c1[j]))
        """
        kind, payload = choice
        if kind == "1hop":
            i = int(payload)
            if not (0 <= i < len(candidates)) or not (0 <= i < len(p1)) or not (0 <= i < len(c1)):
                raise IndexError(f"_build_proof: 1hop index {i} out of range")
            sid = candidates[i].sent_id
            leaves = [sid]
            edge = Edge(
                src=(sid,),
                dst=claim.claim_id,
                label="entail",
                p_entail=float(p1[i]),
                p_contradict=float(c1[i]),
            )
        elif kind == "2hop":
            i, j = payload
            i = int(i); j = int(j)
            if not (0 <= i < len(candidates) and 0 <= j < len(candidates)):
                raise IndexError(f"_build_proof: 2hop indices {(i,j)} out of range")
            if not (0 <= i < len(c1) and 0 <= j < len(c1)):
                raise IndexError(f"_build_proof: c1 indices {(i,j)} out of range")
            sid_i = candidates[i].sent_id
            sid_j = candidates[j].sent_id
            leaves = [sid_i, sid_j]
            edge = Edge(
                src=(sid_i, sid_j),
                dst=claim.claim_id,
                label="entail",
                p_entail=float(score_star),
                p_contradict=float(max(c1[i], c1[j])),
            )
        else:
            raise ValueError(f"_build_proof: unknown choice kind '{kind}'")

        citations = _citation_dicts_from_leaves(leaves, candidates)
        margin = float(score_star) - float(Cmax)

        return ProofObject(
            claim_id=claim.claim_id,
            leaves=leaves,
            edges=[edge],
            citations=citations,
            score=float(score_star),
            cmax=float(Cmax),
            margin=margin,
            stop_reason=str(stop_reason),
            alternation_used=bool(alternation_used),
            accept_reason=str(accept_reason),
            budget_used=int(budget_used),
            ub_top_remaining=float(ub_top_remaining),
        )
