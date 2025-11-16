import os, yaml, json, argparse
from sro.claims.splitter import split_into_subclaims

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="bench/break_dev_200.jsonl")
ap.add_argument("--limit", type=int, default=200)
args = ap.parse_args()

cfg = yaml.safe_load(open("configs/splitter.yaml","r",encoding="utf-8"))
os.environ["SPLITTER_FORCE_MODEL"] = "1"

bins = [0]*11
n_ex=0; n_cand=0
with open(args.dev, encoding="utf-8") as f:
    for i, L in enumerate(f):
        if args.limit and i>=args.limit: break
        q = json.loads(L)["q"]
        out = split_into_subclaims(q, cfg)
        info = out.get("telemetry", {})
        probs = info.get("model_bnd_probs") or info.get("model_probs") or []  # depends how you named it
        for p in probs:
            b = min(10, int(round(p*10)))
            bins[b]+=1
        n_ex+=1
print("examples:", n_ex, "hist:", bins)
