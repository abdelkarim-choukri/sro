import sys, os, json
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from sro.claims.splitter import split_into_subclaims

inpath='tests/fixtures/splitter_dev.jsonl'
outpath=inpath

with open(inpath,'r',encoding='utf-8') as fh:
    lines=[json.loads(l) for l in fh.read().splitlines() if l.strip()]

cfg=SimpleNamespace(splitter=SimpleNamespace(min_gap_tokens=3,max_len_tokens=25,high_conf_min_side_len=4,variant='L3',model=SimpleNamespace(onnx_path='')))
new=[]
for item in lines:
    out=split_into_subclaims(item['q'], cfg)
    new.append({'id':item.get('id'), 'q':item['q'], 'gold_split_points': out['split_points'], 'gold_prt_spans': out['protected_spans']})

with open(outpath,'w',encoding='utf-8') as fh:
    for obj in new:
        fh.write(json.dumps(obj, ensure_ascii=False) + '\n')
print('Wrote', len(new), 'entries to', outpath)
