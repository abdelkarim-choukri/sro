import random
from types import SimpleNamespace
from sro.claims.splitter import split_into_subclaims
from sro.claims.tokenizer import tokenize


def _cfg_rules_only():
    import yaml
    from types import SimpleNamespace as _NS
    with open("configs/splitter.yaml", "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    def ns(d):
        if isinstance(d, dict):
            return _NS(**{k: ns(v) for k, v in d.items()})
        return d

    cfg = ns(y)
    # force rules-only
    if hasattr(cfg.splitter, "model"):
        try:
            cfg.splitter.model.onnx_path = ""
        except Exception:
            cfg.splitter.model = _NS(onnx_path="")
    return cfg


QUOTES = ['"X Y Z"', "“Alpha Beta”", "(foo bar)", "[lorem ipsum]"]
BASES = [
    "Apple announced the device and reviewers reported results.",
    "The team expanded operations and profits increased later.",
]


def test_no_split_inside_injected_prt_and_gap():
    cfg = _cfg_rules_only()
    for _ in range(50):
        base = random.choice(BASES)
        ins = random.choice(QUOTES)
        pos = random.randint(1, max(1, len(base) - 2))
        q = base[:pos] + " " + ins + " " + base[pos:]
        out = split_into_subclaims(q, cfg)
        # no split inside the injected span
        a = q.find(ins)
        b = a + len(ins)
        assert all(not (a <= c < b) for c in out["split_points"])
        # min-gap enforced in tokens
        toks = list(tokenize(q))
        idxs = []
        for c in sorted(out["split_points"]):
            i = next(i for i, t in enumerate(toks) if t.end == c)
            idxs.append(i)
        for i in range(len(idxs) - 1):
            assert idxs[i + 1] - idxs[i] >= cfg.splitter.min_gap_tokens
