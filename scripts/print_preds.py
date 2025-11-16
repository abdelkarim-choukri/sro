from __future__ import annotations
import json
from types import SimpleNamespace
from sro.claims.splitter import split_into_subclaims
from sro.claims.tokenizer import tokenize


def make_cfg():
    # minimal cfg matching eval harness defaults
    return SimpleNamespace(splitter=SimpleNamespace(
        variant="L3",
        min_gap_tokens=3,
        max_claims=5,
        max_len_tokens=25,
        high_conf_min_side_len=4,
        ban_pronouns=True,
        model=SimpleNamespace(onnx_path="", add_prob=0.7, remove_margin=0.35, quantize_int8=False)
    ))


def main(path: str):
    cfg = make_cfg()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            it = json.loads(ln)
            qid = it.get("id")
            q = it["q"]
            out = split_into_subclaims(q, cfg)
            toks = list(tokenize(q))
            print(f"ID: {qid}")
            print("Q:", q)
            print("Tokens:")
            for t in toks:
                print(f"  '{t.text}' [{t.start},{t.end}]")
            print("Gold splits:", it.get("gold_split_points"))
            print("Pred splits:", out.get("split_points"))
            print("Protected spans:", out.get("protected_spans"))
            print("Display texts:", [c.get("display_text") for c in out.get("claims", [])])
            print("---")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/print_preds.py path/to/fixture.jsonl")
        raise SystemExit(2)
    main(sys.argv[1])
