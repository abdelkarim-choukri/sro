"""Export a small BREAK (QDMR high-level) dev subset to our local bench format.

This script prefers the high-level QDMR export (list[str] steps). It validates
that `decomposition` is a list of strings with at least 2 steps and writes
`bench/local/break_dev_highlevel.jsonl`.

Usage:
  python -m scripts.export_break_highlevel --dataset break_data --subset QDMR-high-level --N 200
"""
from __future__ import annotations
import argparse, json, random, re
from datasets import load_dataset

RET = re.compile(r"\breturn\b", re.I)

def qdmr_program_to_steps(prog: str) -> list[str]:
    if not prog or not isinstance(prog, str):
        return []
    parts = []
    for chunk in prog.split(';'):
        m = list(RET.finditer(chunk))
        seg = chunk[m[-1].end():] if m else chunk  # content after last 'return'
        seg = seg.strip()
        if not seg:
            continue
        seg = re.sub(r"#\d+", "", seg)          # drop pointer refs like #1
        seg = re.sub(r"\s+", " ", seg).strip()  # normalize spaces
        if seg:
            parts.append(seg)
    return parts

def main(dataset: str, subset: str, N: int):
    print(f"Loading dataset {dataset} config={subset} (may use local cache)")
    ds = load_dataset(dataset, subset, trust_remote_code=False)
    rows = []
    def to_row(ex):
        q = (ex.get("question_text") or ex.get("question") or "").strip()
        prog_or_list = ex.get("decomposition")
        if isinstance(prog_or_list, list):
            steps = [str(s).strip() for s in prog_or_list if str(s).strip()]
        else:
            steps = qdmr_program_to_steps(str(prog_or_list or ""))
        steps = [s for s in steps if len(s.split()) >= 3]
        if q and len(steps) >= 2:
            return {"question": q, "decomposition": steps}
        return None

    for split in ("validation", "dev", "test", "train"):
        if split in ds:
            for ex in ds[split]:
                r = to_row(ex)
                if r:
                    rows.append(r)

    random.seed(13); random.shuffle(rows)
    rows = rows[:N]
    out = "bench/local/break_dev_highlevel.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/break_data")
    ap.add_argument("--subset", default="QDMR-high-level")
    ap.add_argument("--N", type=int, default=200)
    args = ap.parse_args()
    main(args.dataset, args.subset, args.N)
