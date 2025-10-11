# scripts/print_latest_proofs.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

from sro.prover.logio import latest_lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="artifacts/proofs/proofs.jsonl",
                    help="Path to the live proofs jsonl (non-rotated).")
    ap.add_argument("--n", type=int, default=5, help="How many latest lines to print.")
    ap.add_argument("--raw", action="store_true", help="Print raw lines instead of pretty JSON.")
    args = ap.parse_args()

    base = Path(args.file)
    lines = latest_lines(base, args.n)
    if not lines:
        print("(no proofs found)")
        return

    for ln in lines:
        if args.raw:
            print(ln)
        else:
            try:
                obj = json.loads(ln)
                print(json.dumps(obj, indent=2, ensure_ascii=False))
            except Exception:
                # Fallback to raw if line is corrupted
                print(ln)

if __name__ == "__main__":
    main()
