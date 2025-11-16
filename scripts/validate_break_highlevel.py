"""Validate a BREAK high-level export JSONL file.

Checks:
- Each line is JSON with keys: 'question' (str) and 'decomposition' (list[str]).
- decomposition has at least min_steps (default 2).
- each step has at least min_tokens (default 3).

Exit code: 0 if all rows pass, 1 if any row fails.

Usage:
  python -m scripts.validate_break_highlevel bench/local/break_dev_highlevel.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import Iterable


def validate_row(obj: dict, min_steps: int = 2, min_tokens: int = 3) -> Iterable[str]:
    """Yield error messages for the row; empty if valid."""
    if not isinstance(obj, dict):
        yield "row is not an object"
        return
    if "question" not in obj or not isinstance(obj["question"], str) or not obj["question"].strip():
        yield "missing or invalid 'question'"
    steps = obj.get("decomposition")
    if not isinstance(steps, list):
        yield "'decomposition' is not a list"
        return
    if len(steps) < min_steps:
        yield f"'decomposition' has fewer than {min_steps} steps ({len(steps)})"
    for i, s in enumerate(steps):
        if not isinstance(s, str):
            yield f"step[{i}] is not a string"
            continue
        toks = s.strip().split()
        if len(toks) < min_tokens:
            yield f"step[{i}] has fewer than {min_tokens} tokens ({len(toks)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to JSONL file to validate")
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--min-tokens", type=int, default=3)
    args = ap.parse_args()

    path = args.path
    total = 0
    bad = 0
    first_bad_examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except Exception as e:
                    bad += 1
                    if len(first_bad_examples) < 5:
                        first_bad_examples.append((lineno, f"invalid json: {e}"))
                    continue
                errs = list(validate_row(obj, args.min_steps, args.min_tokens))
                if errs:
                    bad += 1
                    if len(first_bad_examples) < 5:
                        first_bad_examples.append((lineno, errs))
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(2)

    print(f"Checked {total} rows: {total-bad} OK, {bad} bad")
    if first_bad_examples:
        print("First bad examples (lineno -> errors):")
        for lineno, errs in first_bad_examples:
            print(f"  {lineno}: {errs}")

    if bad:
        print("Validation FAILED")
        sys.exit(1)
    print("Validation OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
