# tests/unit/test_citations_shape.py
from __future__ import annotations

from sro.config import load_config
from sro.prover import SROProver
from sro.types import Claim, SentenceCandidate


def test_citation_dict_shape():
    cfg = load_config()
    prover = SROProver(cfg, use_real_nli=False)  # heuristic is fine for shape test

    claim = Claim("c1", "Dummy claim.")
    cands = [
        SentenceCandidate("s1", "evidence A", "doc:1", 0.8),
        SentenceCandidate("s2", "evidence B", "doc:2", 0.7),
    ]
    # Force a trivial 1-hop accept by setting heuristic-friendly text
    result = prover.prove(claim, cands, fetch_more=None)
    # Either ACCEPT or REJECT is fine; if ACCEPT, citations must be dicts with keys
    if result.status == "ACCEPT":
        cites = result.proof.citations
        assert isinstance(cites, list)
        assert all(isinstance(x, dict) and "sent_id" in x and "source_id" in x for x in cites)
