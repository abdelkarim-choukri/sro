from sro.compose.answer import compose_answer_with_citations


def test_compose_inline_cites_and_refs():
    accepted = [
        {"text":"The iPhone 15 Pro features a titanium frame.","score":0.99,
         "citations":[{"sent_id":"news:1#s1","source_id":"news:1"}]},
        {"text":"The Pro models introduced a titanium frame for durability.","score":0.98,
         "citations":[{"sent_id":"press:1#s1","source_id":"press:1"}]},
        {"text":"Apple announced the iPhone 15 lineup in 2023.","score":0.80,
         "citations":[{"sent_id":"press:1#s0","source_id":"press:1"}]},
    ]
    final, refs = compose_answer_with_citations(accepted, N=2)
    assert final  # non-empty
    assert "ⁱ" in final or "[" in final  # has some marker
    # references include unique sources (news:1 and press:1)
    srcs = [s for _, s in refs]
    assert "news:1" in srcs and "press:1" in srcs
