import os
import numpy as np
import pytest

from types import SimpleNamespace
from sro.claims.tokenizer import tokenize
from sro.claims.tagger_infer import TaggerONNX, viterbi_with_constraints

# Helper: minimal cfg with min_gap=3
CFG = SimpleNamespace(splitter=SimpleNamespace(min_gap_tokens=3))


def test_model_inference_non_zero_logits_and_mask():
    # Ensure a model exists; skip if not
    p_l3 = "artifacts/models/splitter_miniL3.int8.onnx"
    p_l6 = "artifacts/models/splitter_miniL6.int8.onnx"
    model_path = p_l3 if os.path.exists(p_l3) else (p_l6 if os.path.exists(p_l6) else "")
    if not model_path:
        pytest.skip("No ONNX available; export first.")

    tagger = TaggerONNX(SimpleNamespace(onnx_path=model_path, quantize_int8=True))

    q = 'Visit https://example.com/docs (alpha), and email ops@foo.io when ready.'
    toks = tokenize(q)

    # PRT spans: URL, (alpha), and email — you already detect these in rules
    # We'll hard-code spans here to test masking.
    prt_spans = []
    # find URL and email spans by char match
    url = "https://example.com/docs"
    email = "ops@foo.io"
    a = q.index(url); prt_spans.append([a, a+len(url)])
    b = q.index("(alpha)"); prt_spans.append([b, b+len("(alpha)")])
    c = q.index(email); prt_spans.append([c, c+len(email)])

    logits = tagger.encode_logits(toks, prt_spans)  # [T,5]
    assert logits.shape[0] == len(toks) and logits.shape[1] == 5

    # Non-zero check (weights are deterministic but not all-zero)
    assert np.any(np.abs(logits) > 1e-6), "Logits are all zeros — export likely failed."

    # Decode with constraints: ensure no BND inside PRT
    tok_bnds = viterbi_with_constraints(logits, prt_spans, CFG.splitter.min_gap_tokens, toks)
    # Convert token indices -> char ends
    char_bnds = [toks[i].end for i in tok_bnds]
    def inside_prt(c):
        for s,e in prt_spans:
            if s <= c < e: return True
        return False
    assert all(not inside_prt(c) for c in char_bnds), "Decoder produced a split inside PRT (mask failed)."
