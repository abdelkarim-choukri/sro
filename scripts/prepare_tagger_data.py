from __future__ import annotations
import json, argparse
from dataclasses import dataclass
from sro.claims.tokenizer import tokenize  # deterministic tokenizer

LABELS = ["PRT_B","PRT_I","PRT_E","BND","O"]
L2ID = {l:i for i,l in enumerate(LABELS)}

@dataclass
class Ex:
    qid: str
    text: str
    tokens: list  # [{text,start,end}]
    labels: list  # per-token ids


def _label_tokens(q: str, gold_splits: set[int], gold_prt: list[list[int]]):
    toks = tokenize(q)
    T = len(toks)
    y = ["O"] * T

    # mark PRT spans
    for s,e in gold_prt:
        covered = [i for i,t in enumerate(toks) if t.start >= s and t.end <= e]
        if not covered:
            continue
        if len(covered) == 1:
            y[covered[0]] = "PRT_E"
        else:
            y[covered[0]] = "PRT_B"
            for k in covered[1:-1]:
                y[k] = "PRT_I"
            y[covered[-1]] = "PRT_E"

    # mark BND at tokens whose end==split char index, but never inside PRT
    for i,t in enumerate(toks):
        if t.end in gold_splits and not y[i].startswith("PRT_"):
            y[i] = "BND"

    return toks, [L2ID[z] for z in y]


def main(in_fixture: str, out_path: str):
    outs = []
    with open(in_fixture, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            q = o["q"]
            gold_splits = set(o.get("gold_split_points", []))
            gold_prt = o.get("gold_prt_spans", [])
            toks, y = _label_tokens(q, gold_splits, gold_prt)
            outs.append({
                "text": q,
                "tokens": [{"text":t.text, "start":t.start, "end":t.end} for t in toks],
                "labels": y
            })
    out_dir = out_path.rsplit("\\",1)[0] if "\\" in out_path else "."
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in outs:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(outs)} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_fixture")
    ap.add_argument("out_jsonl")
    args = ap.parse_args()
    main(args.in_fixture, args.out_jsonl)
