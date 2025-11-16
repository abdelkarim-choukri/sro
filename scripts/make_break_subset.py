from __future__ import annotations
import argparse, json, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="bench/break_dev_fixture.jsonl")
    ap.add_argument("--out_path", default="bench/break_dev_200.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    with open(args.in_path, encoding="utf-8") as f:
        for L in f:
            rows.append(L.rstrip("\n"))
    idxs = list(range(len(rows)))
    # stratified-ish spread: shuffle, then take every k; fallback to head if short
    rng.shuffle(idxs)
    k = max(1, len(rows)//args.n)
    pick = [rows[i] for i in idxs[::k]][:args.n]
    with open(args.out_path, "w", encoding="utf-8") as out:
        for L in pick:
            out.write(L + "\n")
    print(f"[OK] wrote {len(pick)} -> {args.out_path}")

if __name__ == "__main__":
    main()
