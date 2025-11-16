from __future__ import annotations
import os
from types import SimpleNamespace
import yaml

from sro.claims.splitter import split_into_subclaims


def _cfg_rules_only():
    with open("configs/splitter.yaml", "r", encoding="utf-8") as f:
        cfgd = yaml.safe_load(f)
    def ns(d):
        from types import SimpleNamespace
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    cfg = ns(cfgd)
    if hasattr(cfg.splitter, "model"):
        cfg.splitter.model.onnx_path = ""
    return cfg


def test_artifacts_write_errors_are_swallowed(monkeypatch):
    # Load config first (it uses open) then monkeypatch artifact writes to
    # simulate permission errors when writing artifacts.
    cfg = _cfg_rules_only()
    def boom(*a, **k):
        raise PermissionError("nope")
    monkeypatch.setattr(os, "makedirs", boom, raising=True)
    # Only make open raise for subsequent artifact writes; monkeypatch builtins.open
    # with a wrapper that delegates to the real open for non-artifact paths.
    import builtins as _builtins
    real_open = _builtins.open
    def open_wrapper(path, *a, **k):
        p = str(path)
        if "artifacts" in p.replace('\\', '/'):
            raise PermissionError("nope")
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", open_wrapper, raising=True)
    q = "Apple announced the iPhone 15 Pro, and reviewers reported reduced weight."
    out = split_into_subclaims(q, cfg)
    assert isinstance(out, dict) and "claims" in out and "split_points" in out and "telemetry" in out
