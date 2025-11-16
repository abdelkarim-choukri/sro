from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np

from sro.claims.tokenizer import Token
import json

# Label order: [PRT_B, PRT_I, PRT_E, BND, O]
LBL_PRT_B, LBL_PRT_I, LBL_PRT_E, LBL_BND, LBL_O = range(5)

class TaggerONNX:
    def __init__(self, model_cfg):
        self.path = getattr(model_cfg, "onnx_path", "")
        self.quant = bool(getattr(model_cfg, "quantize_int8", True))
        # Lazy import to keep stdlib in tests when model is missing
        self._session = None
        self._available = os.path.exists(self.path)

    def _ensure_session(self):
        if not self._available or self._session is not None:
            return
        import onnxruntime as ort  # optional, only if model file exists
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.path, sess_options=opts, providers=providers)

    def encode_logits(self, tokens: List[Token], prt_spans: List[Tuple[int,int]]) -> "np.ndarray":
        """
        Return logits [T,5]. In stub mode (no ONNX), return zeros (deterministic).
        Wordpiece mapping is assumed baked in the exported model; here we pass plain tokens
        to a tiny adapter or just stub.
        """
        T = len(tokens)
        if not self._available:
            return np.zeros((T, 5), dtype=np.float32)

        self._ensure_session()
        # Try to find a manifest next to the ONNX file to discover a tokenizer
        onnx_dir = os.path.dirname(self.path)
        manifest = None
        for fn in os.listdir(onnx_dir):
            if fn.endswith("_export.json") and fn.startswith("splitter"):
                try:
                    with open(os.path.join(onnx_dir, fn), "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    break
                except Exception:
                    continue

        tokenizer = None
        if manifest and manifest.get("hf_model_path"):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(manifest.get("hf_model_path"), local_files_only=True, use_fast=True)
            except Exception:
                tokenizer = None

        # Reconstruct original text from tokens using token.start positions
        # tokens may have .start/.end and .text
        qlen = max((t.end for t in tokens), default=0)
        chars = [" "] * max(qlen, 0)
        for t in tokens:
            s = t.start
            for i, ch in enumerate(t.text):
                if s + i < len(chars):
                    chars[s + i] = ch
                else:
                    chars.extend([" "] * (s + i - len(chars) + 1))
                    chars[s + i] = ch
        q = "".join(chars).rstrip()

        # If we couldn't load a tokenizer or session cannot accept ids, fall back to zeros
        if tokenizer is None:
            # best-effort: return zeros to keep deterministic behavior
            return np.zeros((T, 5), dtype=np.float32)

        # Tokenize with offsets so we can map wordpieces back to token spans
        enc = tokenizer(q, return_offsets_mapping=True, return_tensors="np")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc.get("offset_mapping")[0].tolist()

        # Run ONNX session
        feeds = {}
        # find input names
        inp0 = self._session.get_inputs()[0].name
        inp1 = self._session.get_inputs()[1].name if len(self._session.get_inputs()) > 1 else None
        feeds[inp0] = input_ids
        if inp1 is not None:
            feeds[inp1] = attention_mask
        out = self._session.run(None, feeds)
        # expect [B, S, 5]
        logits_pieces = np.asarray(out[0])
        if logits_pieces.ndim == 3:
            logits_pieces = logits_pieces[0]

        # Map piece logits to token-level by averaging piece logits that overlap each token span
        token_logits = np.zeros((T, logits_pieces.shape[1]), dtype=np.float32)
        for ti, t in enumerate(tokens):
            tstart, tend = t.start, t.end
            contribs = []
            for pi, (ps, pe) in enumerate(offsets):
                # skip empty offsets (special tokens)
                if ps == pe == 0:
                    continue
                # check overlap
                if not (pe <= tstart or ps >= tend):
                    contribs.append(logits_pieces[pi])
            if contribs:
                token_logits[ti] = np.mean(contribs, axis=0)
            else:
                # fallback: take nearest piece by center
                centers = [( (ps+pe)/2.0, pi) for pi,(ps,pe) in enumerate(offsets) if not (ps==pe==0)]
                if centers:
                    center_vals = [abs(c - ((tstart+tend)/2.0)) for c,_ in centers]
                    nearest_idx = centers[int(np.argmin(center_vals))][1]
                    token_logits[ti] = logits_pieces[nearest_idx]
                else:
                    token_logits[ti] = 0.0

        return token_logits


def viterbi_with_constraints(logits: "np.ndarray", prt_spans: List[Tuple[int,int]], min_gap: int, tokens: List[Token]) -> List[int]:
    """
    Constrained decode that forbids BND inside PRT and enforces min-gap in token space.
    This function applies a hard mask to the BND logits for any token whose char END
    lies inside a protected span, then performs a simple argmax decode followed by
    a global min-gap pass over token indices.
    Args:
      logits: np.ndarray[T,5]
      prt_spans: list of (start,end) char spans
      min_gap: min token gap between boundaries
      tokens: list of Token objects (must correspond to logits rows)
    Returns:
      list of token indices where BND fires (decoded and gap-enforced)
    """
    T = logits.shape[0]
    # copy to avoid mutating caller data
    masked = logits.copy()
    # mask out BND logits where token end falls inside any PRT span
    for i in range(T):
        c = tokens[i].end
        for a,b in prt_spans:
            if a <= c < b:
                masked[i, LBL_BND] = -1e9
                break

    pred = np.argmax(masked, axis=-1)  # [T]
    cand = [i for i in range(T) if pred[i] == LBL_BND]

    # Enforce global min-gap over token indices
    if not cand:
        return []
    kept: List[int] = []
    last = -10**9
    for i in cand:
        if i - last >= min_gap:
            kept.append(i)
            last = i
    return kept
