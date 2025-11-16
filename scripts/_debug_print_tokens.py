import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sro.claims.tokenizer import tokenize
qs = [
    'Did Apple announce the iPhone 15 Pro, and did reviewers report reduced weight?',
    '“Johnson & Johnson” expanded in the U.S., and shipments rose.'
]
for q in qs:
    print('Q:', q)
    toks = tokenize(q)
    for i,t in enumerate(toks):
        print(i, repr(t.text), t.start, t.end)
    print('---')
