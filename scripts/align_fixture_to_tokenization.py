from __future__ import annotations
import json, sys
from sro.claims.tokenizer import tokenize


def nearest_end(c, ends):
    if c in ends:
        return c
    candidates = sorted(ends, key=lambda e: abs(e - c))
    if candidates and abs(candidates[0] - c) <= 1:
        return candidates[0]
    return None


def process(path):
    out = []
    changed = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            it = json.loads(ln)
            q = it["q"]
            ends = {t.end for t in tokenize(q)}
            fixed = []
            for c in it["gold_split_points"]:
                n = nearest_end(c, ends)
                if n is None:
                    fixed.append(c)
                else:
                    if n != c:
                        changed += 1
                    fixed.append(n)
            it["gold_split_points"] = fixed
            out.append(it)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n")
    print("Aligned indices (±1) for", changed, "splits")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/align_fixture_to_tokenization.py path/to/fixture.jsonl")
        raise SystemExit(2)
    process(sys.argv[1])
