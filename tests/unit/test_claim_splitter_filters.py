from sro.claims.splitter import draft_and_claims
from sro.types import SentenceCandidate
from sro.config import load_config

def test_hedge_and_reliability_filter():
    cfg = load_config()
    question = "Does the iPhone 15 Pro have a titanium frame?"
    cands = [
        SentenceCandidate("news:1#s1","The iPhone 15 Pro features a titanium frame.","news:1",0.95),
        SentenceCandidate("press:1#s1","The Pro models introduced a titanium frame for durability.","press:1",0.90),
        SentenceCandidate("blog:1#s0","Rumors suggested the Pro might include titanium.","blog:1",0.99),
    ]
    draft, claims = draft_and_claims(
        question,
        cands,
        K=3,
        min_question_cosine=cfg.claims.min_question_cosine,
        hedge_terms=cfg.claims.hedge_terms,
        reliability_weights=cfg.claims.reliability_weights,
    )
    assert "Rumors" not in draft
    texts = [c.text for c in claims]
    assert all("Rumor" not in t and "Rumors" not in t for t in texts)
