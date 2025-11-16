from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml, json
from sro.claims.splitter import split_into_subclaims
with open('tests/fixtures/splitter_dev.jsonl','r',encoding='utf-8') as f:
    items=[json.loads(ln) for ln in f if ln.strip()]
item=items[0]
with open('configs/splitter.yaml','r',encoding='utf-8') as f:
    cfgd=yaml.safe_load(f)

def ns(d):
    from types import SimpleNamespace
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k,v in d.items()})
cfg=ns(cfgd)
if hasattr(cfg.splitter,'model'):
    cfg.splitter.model.onnx_path=''
out=split_into_subclaims(item['q'],cfg)
print('id',item['id'])
print('gold',item['gold_split_points'])
print('pred',out['split_points'])
print('claims')
for c in out['claims']:
    print(c)
from sro.claims.tokenizer import tokenize
print('\nTokens:')
toks = tokenize(item['q'])
for i,t in enumerate(toks):
    print(i, repr(t.text), t.start, t.end)
