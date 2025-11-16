import sys
import os
sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
from sro.claims.splitter import split_into_subclaims
import json
cfg=SimpleNamespace(splitter=SimpleNamespace(min_gap_tokens=3,max_len_tokens=25,high_conf_min_side_len=4,variant='L3',model=SimpleNamespace(onnx_path='')))
with open('tests/fixtures/splitter_dev.jsonl','r',encoding='utf-8') as fh:
    item = json.loads(fh.read().splitlines()[0])
print('id',item['id'])
out=split_into_subclaims(item['q'], cfg)
print('split_points', out['split_points'])
print('protected_spans', out['protected_spans'])
print('claims')
for c in out['claims']:
    print(c['start'],c['end'],repr(c['text']))
print('\n--- diagnostic slices ---')
q=item['q']
print('len',len(q))
print('gold',item['gold_split_points'])
for idx in sorted(set(item['gold_split_points'] + out['split_points'])):
    print(idx, repr(q[:idx]))
