from __future__ import annotations
import json, sys, os, re
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

from sro.claims.tokenizer import tokenize
from sro.claims.rules import propose_protected_spans, propose_rule_splits
from types import SimpleNamespace

# Input JSONL rows: {"question": str, "decomposition": ["step1", "step2", ...]}
# Output JSONL rows (our dev-fixture shape): {"q": str, "gold_split_points": [int,...], "gold_prt_spans": [[s,e],...]}

STEP_SEP_HINT = re.compile(r"\b(and|or|then|after|before|because|so that|which|who|that)\b", re.I)

@dataclass
class OutRow:
    q: str
    gold_split_points: List[int]
    gold_prt_spans: List[List[int]]


def _snap_to_token_end(char_idx: int, toks) -> int | None:
    ends = [t.end for t in toks]
    # nearest token end <= char_idx, else next end if within 1 char
    le = [e for e in ends if e <= char_idx]
    if le:
        return le[-1]
    ge = [e for e in ends if e >= char_idx]
    return ge[0] if ge and ge[0] - char_idx <= 1 else None


def _derive_split_points(q: str, steps: List[str]) -> List[int]:
    # Improved alignment: map desired split count to nearest legal rule candidates
    if len(steps) <= 1:
        return []
    toks = tokenize(q)
    # get PRT spans and rule candidates (use a minimal cfg)
    try:
        prt = propose_protected_spans(toks, cfg=None)
    except TypeError:
        prt = propose_protected_spans(toks)
    cfg_local = SimpleNamespace(splitter=SimpleNamespace(
        variant="L3",
        min_gap_tokens=3,
        max_len_tokens=25,
        model=SimpleNamespace(onnx_path="")
    ))
    # propose_rule_splits expects a cfg with splitter settings
    rule_bnds, meta = propose_rule_splits(toks, prt, cfg_local)
    target = max(0, len(steps) - 1)
    if not rule_bnds:
        # fallback to joiner-based heuristic (best-effort)
        splits: List[int] = []
        cursor = 0
        for k in range(len(steps)-1):
            m = STEP_SEP_HINT.search(q[cursor:])
            cand = None
            if m:
                join_end = _snap_to_token_end(cursor + m.end(), toks)
                cand = join_end
                cursor = cursor + m.end()
            if cand is None:
                mid = int(len(q) * (k+1) / len(steps))
                cand = _snap_to_token_end(mid, toks)
            if cand is not None:
                splits.append(cand)
        splits = sorted(set(splits))
        pruned = []
        prev = -10
        for c in splits:
            if c - prev >= 2:
                pruned.append(c); prev = c
        return pruned
    # Prefer aligning to step substrings when possible, else fall back to even spacing
    desired = []
    for k in range(target):
        step_text = steps[k].strip()
        if not step_text:
            desired.append(int((k+1) * len(q) / len(steps)))
            continue
        # case-insensitive search for the step text in the question
        idx = q.lower().find(step_text.lower())
        if idx != -1:
            desired.append(idx + len(step_text))
        else:
            desired.append(int((k+1) * len(q) / len(steps)))

    snapped = []
    for d in desired:
        # choose nearest candidate by char distance
        c = min(rule_bnds, key=lambda x: abs(x - d))
        snapped.append(c)
    # dedupe, sort, prune close-by boundaries
    snapped = sorted(set(snapped))
    pruned, prev = [], -10
    for c in snapped:
        if c - prev >= 2:
            pruned.append(c)
            prev = c
    return pruned


def _gold_prt(q: str) -> List[List[int]]:
    # reuse rule PRT as gold-proxy; you can later enrich with curated phrases.
    toks = tokenize(q)
    # propose_protected_spans expects a cfg in some variants; call defensively
    try:
        prt = propose_protected_spans(toks, cfg=None)
    except TypeError:
        prt = propose_protected_spans(toks)
    return [[s,e] for (s,e) in prt]


def convert(in_path: str, out_path: str, limit: int | None = None):
    n = 0
    with open(out_path, "w", encoding="utf-8") as fout, open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip(): continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("question_text") or ""
            steps = obj.get("decomposition") or obj.get("qdmr") or []
            if not q or not steps: continue
            row = OutRow(q=q, gold_split_points=_derive_split_points(q, steps), gold_prt_spans=_gold_prt(q))
            fout.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit: break
    print(f"[OK] Wrote {n} items -> {out_path}")


if __name__ == "__main__":
    # python -m scripts.bench_break_to_fixture bench/local/break_dev_highlevel.jsonl bench/break_dev_fixture.jsonl --limit 200
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    convert(args.in_path, args.out_path, args.limit)
