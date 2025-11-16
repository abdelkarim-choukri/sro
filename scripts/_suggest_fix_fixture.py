import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sro.claims.tokenizer import tokenize

with open('tests/fixtures/splitter_dev.jsonl','r',encoding='utf-8') as f:
    items=[json.loads(ln) for ln in f if ln.strip()]

for it in items:
    q=it['q']
    toks=tokenize(q)
    token_ends={t.end for t in toks}
    gold=it.get('gold_split_points',[])
    good = all(g in token_ends for g in gold)
    pred=None
    # recompute rules-only pred
    from types import SimpleNamespace
    import yaml
    with open('configs/splitter.yaml','r',encoding='utf-8') as fh:
        cfgd=yaml.safe_load(fh)
    def ns(d):
        from types import SimpleNamespace
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k,v in d.items()})
    cfg=ns(cfgd)
    if hasattr(cfg.splitter,'model'):
        cfg.splitter.model.onnx_path=''
    from sro.claims.splitter import split_into_subclaims
    out=split_into_subclaims(q,cfg)
    pred=out['split_points']
    print(it['id'])
    print('  q=',q)
    print('  gold=',gold,'valid_token_ends=',good)
    print('  pred=',pred)
    # suggest fix: map each gold to nearest token_end
    sug=[]
    for g in gold:
        if g in token_ends:
            sug.append(g)
        else:
            nearest=min(list(token_ends), key=lambda x: abs(x-g))
            sug.append(nearest)
    print('  suggested=',sug)
    print()
