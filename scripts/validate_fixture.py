from __future__ import annotations
import json, sys
from sro.claims.tokenizer import tokenize
from sro.claims.rules import propose_protected_spans


def validate_item(it):
    q = it["q"]
    toks = tokenize(q)
    ends = {t.end for t in toks}
    prt = propose_protected_spans(toks, type("Cfg", (), {"splitter": None})())
    # 1) every gold split is a token end
    bad_ends = [c for c in it["gold_split_points"] if c not in ends]
    # 2) no gold split lies inside PRT
    inside_prt = []
    for c in it["gold_split_points"]:
        for a, b in prt:
            if a <= c < b:
                inside_prt.append((c, (a, b)))
                break
    # 3) gold PRT spans are merged & ordered
    gp = it.get("gold_prt_spans", [])
    non_merged = []
    last = -1
    for a, b in gp:
        if a >= b or a < last:
            non_merged.append((a, b))
        last = b
    return bad_ends, inside_prt, non_merged


def main(path):
    ok = True
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            it = json.loads(ln)
            be, ip, nm = validate_item(it)
            if be or ip or nm:
                ok = False
                print("ID:", it.get("id"))
                if be:
                    print("  gold_split_points not at token ends:", be)
                if ip:
                    print("  gold splits inside PRT:", ip)
                if nm:
                    print("  gold_prt_spans not merged/ordered:", nm)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_fixture.py path/to/fixture.jsonl")
        raise SystemExit(2)
    main(sys.argv[1])
