from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml, json
from sro.claims.splitter import split_into_subclaims
with open('configs/splitter.yaml','r',encoding='utf-8') as f:
    cfgd=yaml.safe_load(f)

def ns(d):
    from types import SimpleNamespace
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k,v in d.items()})
cfg=ns(cfgd)
# ensure rules-only
if hasattr(cfg.splitter,'model'):
    cfg.splitter.model.onnx_path=''
q='Apple announced the device and is expected to ship widely.'
out=split_into_subclaims(q,cfg)
print(json.dumps(out,ensure_ascii=False,indent=2))
