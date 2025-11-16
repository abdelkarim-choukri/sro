from __future__ import annotations
import sys
from sro.claims.tokenizer import tokenize

def show(q: str):
    toks = tokenize(q)
    for i, t in enumerate(toks):
        print(f"{i:>3}  {t.text!r:>20}  [{t.start},{t.end})")
    print("token-ends:", [t.end for t in toks])


def main():
    if len(sys.argv) == 2:
        show(sys.argv[1])
    else:
        for ln in sys.stdin:
            s = ln.strip()
            if s:
                print("Q:", s)
                show(s)


if __name__ == "__main__":
    main()
