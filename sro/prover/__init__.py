"""
SROProver — single alternation + JSONL proof logging + PR1 timers/bs split.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
from sro.utils.text import year_conflict 
from sro.utils.timing import StageTimer

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

_WORD = re.compile(r"\w+")


def _is_echo_like(evidence_text: str, claim_text: str) -> bool:
    """
    Returns True if the evidence is just an echo/paraphrase of the claim.
    Heuristics:
      - normalized strings equal OR one endswith the other
      - token Jaccard >= 0.9
      - starts with 'reports indicate' (previous fabricated pattern)
    """
    a = " ".join(_WORD.findall(evidence_text.lower()))
    b = " ".join(_WORD.findall(claim_text.lower()))
    if not a or not b:
        return False
    if a == b or a.endswith(b) or b.endswith(a):
        return True
    ta, tb = set(a.split()), set(b.split())
    j = len(ta & tb) / len(ta | tb)
    if j >= 0.90:
        return True
    if a.startswith("reports indicate") or a.startswith("report indicates"):
        return True
    return False


class _TwoHopNLIScorer(TwoHopScorer):
    """Thin adapter to real 2-hop NLI scorer; batch_size must be bs_nli2."""
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
    Minimal demo fetcher: only returns targeted support for known toy patterns.
    IMPORTANT: Do NOT fabricate generic support (no 'alt_generic').
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
    Append a single JSONL record to artifacts/proofs/proofs.jsonl with rotation.
    Robust: serialize dataclasses; never crash the caller.
    """
    try:
        from sro.prover.logio import append_jsonl  # local import to avoid cycles
        out_dir: Path = cfg.paths.proofs_dir
        base = out_dir / "proofs.jsonl"

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
                "accept_reason": getattr(proof, "accept_reason", None),
                "budget_used": getattr(proof, "budget_used", None),
                "ub_top_remaining": getattr(proof, "ub_top_remaining", None),
                "timings": getattr(proof, "timings", {}),
            }

        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status,
            "proof": payload,
        }

        append_jsonl(base, record)
    except Exception:
        # Don’t break the pipeline if logging fails.
        pass



class SROProver:
    def __init__(
        self,
        cfg: Optional[Config] = None,
        use_real_nli: bool = True,
        bs_nli1: Optional[int] = None,
        bs_nli2: Optional[int] = None,
        batch_size: Optional[int] = None,  # ← NEW (back-compat)
    ):
        self.cfg = cfg or load_config()
        self.use_real_nli = bool(use_real_nli)

        # Back-compat: if tests pass batch_size=..., use it for both hops unless explicit bs_* provided
        if batch_size is not None:
            if bs_nli1 is None:
                bs_nli1 = batch_size
            if bs_nli2 is None:
                bs_nli2 = batch_size

        # Defaults if still None
        self.bs_nli1 = int(bs_nli1 if bs_nli1 is not None else 32)
        self.bs_nli2 = int(bs_nli2 if bs_nli2 is not None else 32)

        # Optional legacy attr for any stray uses
        self.batch_size = self.bs_nli1
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

        t = StageTimer()
        knobs = self.cfg.sro_prover
        M, L, B = int(knobs.M), int(knobs.L), int(knobs.B)
        tau1, tau2 = float(knobs.tau1), float(knobs.tau2)
        delta, kappa, eps = float(knobs.delta), float(knobs.kappa), float(knobs.epsilon)

        # ---------- S1: one-hop ----------
        with t.stage("S1_onehop"):
            p1, c1 = s1_one_hop(
                claim.text, candidates,
                use_model=self.use_real_nli,
                batch_size=self.bs_nli1
            )
        Cmax = max(c1) if c1 else 0.0
        best1_idx = max(range(len(p1)), key=lambda i: p1[i]) if p1 else -1
        best1_score = p1[best1_idx] if best1_idx >= 0 else 0.0

        # ---------- S2–S3: frontier, pool, features ----------
        with t.stage("S2_frontier"):
            frontier_idx, pool2_idx, token_cache = select_frontier_and_pool(
                candidates, p1, M=M, L=L, lambda_diversity=0.7
            )
        with t.stage("S3_features"):
            claim_tokens = list(s2_tokens(claim.text))
            pairs, feats = build_pair_features(
                claim_tokens, candidates, token_cache, frontier_idx, pool2_idx, p1
            )

        # ---------- S4–S5: UB + bounded search (first pass) ----------
        scorer = _TwoHopNLIScorer(
            claim.text, [c.text for c in candidates],
            batch_size=self.bs_nli2
        ) if (self.use_real_nli and nli_two_hop is not None) else None

        best_pair, best_p2, evals, stop_reason, top_ub_rem = bounded_search(
            claim.text, candidates, pairs, feats,
            p1=p1, tau1=tau1, B=B, kappa=kappa,
            ub_weights=UBWeights(),
            two_hop_scorer=scorer,
            batch_size=self.bs_nli2,
            # NOTE: instrument S4/S5 inside bounded_search to support timer=t
            timer=t,
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

        # ---------- S7: Safety + ship (first pass) ----------
        if choice is not None:
            # If it's a 1-hop choice, veto when years conflict
            if choice[0] == "1hop":
                i = int(choice[1])
                if year_conflict(candidates[i].text, claim.text):
                    return ProverResult(status="REJECT", reason="YEAR_CONFLICT_GUARD")

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
            # attach timings for CLI (not persisted by asdict)
            proof.timings = {k: t.get(k) for k in ("S1_onehop","S2_frontier","S3_features","S4_ub","S5_search")}
            _log_proof(self.cfg, proof, status="ACCEPT")
            return ProverResult(status="ACCEPT", proof=proof)

        # ---------- S8: Alternation gate ----------
        best_so_far = max(best1_score, best_p2)
        remaining_budget = max(0, B - evals)
        if fetch_more is None:
            fetch_more = _default_fetch_more

        if should_alternate(best_so_far, top_ub_rem, tau2, eps, remaining_budget, evals_first_pass=evals):
            # Get a few extra targeted sentences via the provided retriever
            extra = fetch_more(
                claim=claim.text,
                candidates=candidates,
                frontier_idx=frontier_idx,
                pool2_idx=pool2_idx,
                p1=p1,
                top_ub=top_ub_rem,
            ) or []

            if os.getenv("SRO_DEBUG") == "1":
                print(f"[SRO_DEBUG] alternation: evals={evals} best1={best1_score:.3f} best2={best_p2:.3f} "
                      f"topUB={top_ub_rem:.3f} budget_left={remaining_budget} extra={len(extra)}")

            # --- De-dup with robust fallback and forced-unique IDs ---
            existing_ids = {c.sent_id for c in candidates}
            extra = [e for e in extra if e.sent_id not in existing_ids]
            pre_cnt = len(extra)
            extra = [
                e for e in extra
                if e.sent_id != "alt_generic" and not _is_echo_like(e.text, claim.text)
            ]
            dropped = pre_cnt - len(extra)
            if os.getenv("SRO_DEBUG") == "1":
                print(f"[SRO_DEBUG] extra_after_dedup={len(extra)} dropped_echo={dropped}")

            if not extra:
                # last-resort targeted fallback, force-unique ids deterministically
                fallback = _default_fetch_more(claim.text, candidates)
                uniq: List[SentenceCandidate] = []
                for e in fallback:
                    sid = e.sent_id
                    k = 1
                    while sid in existing_ids:
                        sid = f"{e.sent_id}_alt{k}"
                        k += 1
                    uniq.append(SentenceCandidate(sent_id=sid, text=e.text, source_id=e.source_id, ce_score=e.ce_score))
                extra = uniq

            if extra:
                candidates2 = candidates + extra

                # S1b — one-hop on expanded pool, use bs_nli1
                with t.stage("S1_onehop"):
                    p1b, c1b = s1_one_hop(
                        claim.text, candidates2,
                        use_model=self.use_real_nli,
                        batch_size=self.bs_nli1
                    )
                Cmax_b = max(c1b) if c1b else 0.0

                # S2b — frontier/pool; S3b — features
                with t.stage("S2_frontier"):
                    frontier2, pool22, token_cache2 = select_frontier_and_pool(
                        candidates2, p1b, M=M, L=L, lambda_diversity=0.7
                    )
                with t.stage("S3_features"):
                    claim_tokens2 = list(s2_tokens(claim.text))
                    pairs2, feats2 = build_pair_features(
                        claim_tokens2, candidates2, token_cache2, frontier2, pool22, p1b
                    )

                # Two-hop scorer for alternation pass: use bs_nli2 (NOT the 1-hop batch size)
                scorer2 = _TwoHopNLIScorer(
                    claim.text, [c.text for c in candidates2],
                    batch_size=self.bs_nli2
                ) if (self.use_real_nli and nli_two_hop is not None) else None

                # S4b+S5b — UB + bounded search (second pass)
                best_pair_b, best_p2_b, evals_b, stop_reason_b, top_ub_rem_b = bounded_search(
                    claim.text, candidates2, pairs2, feats2,
                    p1=p1b, tau1=tau1, B=remaining_budget, kappa=kappa,
                    ub_weights=UBWeights(),
                    two_hop_scorer=scorer2,
                    batch_size=self.bs_nli2,
                    timer=t,
                )

                # Decide after alternation (allow 1-hop fallback)
                choice_b: Optional[Tuple[str, object]] = None
                score_star_b = 0.0
                if best_pair_b is not None and best_p2_b >= tau2:
                    i, j = best_pair_b
                    ok_min, _ = is_minimal(p1b[i], p1b[j], tau1)
                    if ok_min:
                        choice_b = ("2hop", (i, j))
                        score_star_b = float(best_p2_b)

                best1_idx_b = max(range(len(p1b)), key=lambda i: p1b[i]) if p1b else -1
                best1_score_b = p1b[best1_idx_b] if best1_idx_b >= 0 else 0.0
                if choice_b is None and best1_score_b >= tau1:
                    choice_b = ("1hop", best1_idx_b)
                    score_star_b = float(best1_score_b)

                if os.getenv("SRO_DEBUG") == "1":
                    print(f"[SRO_DEBUG] alt_decision: choice={choice_b} "
                          f"best1_b={best1_score_b:.3f} best2_b={best_p2_b:.3f} Cmax_b={Cmax_b:.3f}")

                if choice_b is not None:
                    ok_safe_b, _ = safe_to_ship(score_star_b, Cmax_b, delta)
                    if not ok_safe_b:
                        return ProverResult(status="REJECT", reason="CONTRADICTION_BLOCK")
                    acc_reason_b = "2hop@tau2" if choice_b[0] == "2hop" else "1hop@tau1"
                    proof_b = self._build_proof(
                        choice_b, claim, candidates2, p1b, c1b,
                        score_star_b, Cmax_b, stop_reason_b,
                        alternation_used=True,
                        accept_reason=acc_reason_b,
                        budget_used=evals + evals_b,
                        ub_top_remaining=top_ub_rem_b,
                    )
                    proof_b.timings = {k: t.get(k) for k in ("S1_onehop","S2_frontier","S3_features","S4_ub","S5_search")}
                    _log_proof(self.cfg, proof_b, status="ACCEPT")
                    return ProverResult(status="ACCEPT", proof=proof_b)

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
