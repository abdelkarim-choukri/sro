from sro.compose.answer import compose_answer_with_citations


def test_source_diversity_enforced():
    accepted = [
        {"text":"Apple released the iPhone 15 series in 2023.", "score":0.99,
         "citations":[{"sent_id":"press:1#s0","source_id":"press:1"}]},
        {"text":"The Pro model features a titanium frame.", "score":0.98,
         "citations":[{"sent_id":"press:1#s1","source_id":"press:1"}]},
        {"text":"The iPhone 15 Pro has a titanium frame.", "score":0.97,
         "citations":[{"sent_id":"news:1#s1","source_id":"news:1"}]},
    ]
    final, refs = compose_answer_with_citations(accepted, N=2, enforce_source_diversity=True)
    # Expect markers for both press:1 and news:1, not just press:1
    ref_srcs = [s for _, s in refs]
    assert "press:1" in ref_srcs and "news:1" in ref_srcs
