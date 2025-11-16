from __future__ import annotations
import json
import yaml
from types import SimpleNamespace
from sro.claims.splitter import split_into_subclaims


def _cfg_rules_only():
    with open("configs/splitter.yaml", "r", encoding="utf-8") as f:
        cfgd = yaml.safe_load(f)

    def ns(d):
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})

    cfg = ns(cfgd)
    if hasattr(cfg.splitter, "model"):
        cfg.splitter.model.onnx_path = ""
    return cfg


def test_claim_schema_and_canonical_slice():
    q = "Apple announced the iPhone 15 Pro, and is expected to ship widely."
    out = split_into_subclaims(q, _cfg_rules_only())
    for cl in out["claims"]:
        for k in ("id", "text", "start", "end", "reason", "score"):
            assert k in cl
        assert q[cl["start"]:cl["end"]].strip() == cl["text"]
        if "display_text" in cl:
            assert isinstance(cl["display_text"], str) and cl["display_text"].strip()
