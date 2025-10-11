import csv
from sro.prover.s4_ub import UBWeights, upper_bound, clamp01

KAPPA = 0.80   # adjust if you want
w = UBWeights()
n = sat = 0

with open("data/processed/dev_pairs.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        feats = {k: float(r[k]) for k in [
            "max_p1","entity_overlap","time_agreement","distance",
            "novelty","ce_max","negation_conflict","source_diversity"
        ] if k in r and r[k] != ""}
        if not feats:
            continue
        ub = clamp01(upper_bound(feats, KAPPA, w))
        n += 1
        if ub >= 0.999:
            sat += 1

print("pairs", n, "ub>=0.999", sat)
