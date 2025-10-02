from __future__ import annotations
from pathlib import Path

def main():
    p = Path("artifacts/proofs/proofs.jsonl")
    if not p.exists():
        print("No proofs file yet at:", p)
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    n = len(lines)
    take = min(5, n)  # K = 5 last lines
    for line in lines[-take:]:
        print(line)

if __name__ == "__main__":
    main()
