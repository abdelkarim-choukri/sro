from __future__ import annotations
from types import SimpleNamespace
import yaml
import random

from sro.claims.splitter import split_into_subclaims
from sro.claims.tokenizer import tokenize


def _cfg_rules_only():
    with open("configs/splitter.yaml", "r", encoding="utf-8") as f:
        cfgd = yaml.safe_load(f)
    def ns(d):
        from types import SimpleNamespace
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    cfg = ns(cfgd)
    # force rules only
    if hasattr(cfg.splitter, "model"):
        cfg.splitter.model.onnx_path = ""
    return cfg


def test_prt_integrity_property_injected_quotes():
    base = "Apple announced the device and reviewers reported reduced weight."
    inserts = ['"A B C"', "“Quoted Phrase”", "(alpha beta)", "[x y z]"]
    cfg = _cfg_rules_only()
    for ins in inserts:
        for pos in [6, 12, 24]:
            q = base[:pos] + " " + ins + " " + base[pos:]
            out = split_into_subclaims(q, cfg)
            start = q.find(ins)
            end = start + len(ins)
            for c in out["split_points"]:
                assert not (start <= c < end), f"Split inside injected PRT span: {ins}"


def test_determinism_100x():
    q = "Apple announced the iPhone 15 Pro, and reviewers reported reduced weight."
    cfg = _cfg_rules_only()
    outs = [split_into_subclaims(q, cfg) for _ in range(100)]
    first = outs[0]
    for o in outs[1:]:
        assert o == first
