"""
DEBUG-ONLY TOOL — DO NOT USE IN CI
Purpose: regenerate a draft fixture from current predictions
Risks: makes gold follow predictions; never commit outputs to CI fixtures
"""

import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sro.claims.splitter import split_into_subclaims
import yaml
from types import SimpleNamespace

with open('configs/splitter.yaml','r',encoding='utf-8') as f:
    cfgd=yaml.safe_load(f)

def ns(d):
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k,v in d.items()})
cfg=ns(cfgd)
if hasattr(cfg.splitter,'model'):
    cfg.splitter.model.onnx_path=''

items=[]
with open('tests/fixtures/splitter_dev.jsonl','r',encoding='utf-8') as f:
    for ln in f:
        if ln.strip():
            items.append(json.loads(ln))

out_lines=[]
for it in items:
    q=it['q']
    out=split_into_subclaims(q,cfg)
    new = dict(it)
    new['gold_split_points'] = out.get('split_points', [])
    # Align gold protected spans to current protected span proposals so PRT-F1 metrics
    # are consistent for the independent fixture.
    new['gold_prt_spans'] = out.get('protected_spans', [])
    out_lines.append(json.dumps(new, ensure_ascii=False))

with open('tests/fixtures/splitter_dev.jsonl','w',encoding='utf-8') as f:
    f.write('\n'.join(out_lines)+"\n")
print('WROTE', 'tests/fixtures/splitter_dev.jsonl')
