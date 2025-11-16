#!/usr/bin/env python3
"""Deterministic split of artifacts/splitter/break_train.jsonl -> train.json, val.json
Usage: python d:\project\scripts\split_train_val.py
"""
import json, hashlib, sys, os
P_DIR = os.path.join('artifacts','splitter')
p = os.path.join(P_DIR, 'break_train.jsonl')
if not os.path.exists(p):
    print('[ERR] missing', p); sys.exit(2)
# read input rows (jsonl)
with open(p, encoding='utf-8') as fh:
    rows = [json.loads(l) for l in fh if l.strip()]
# key = sha1 of text to stabilize ordering, then deterministic split
rows = sorted(rows, key=lambda r: hashlib.sha1(r['text'].encode('utf-8')).hexdigest())
N = len(rows)
k = max(20, int(0.1 * N))
val = rows[:k]
train = rows[k:]
os.makedirs(P_DIR, exist_ok=True)

# Write both train/val as a JSON array (train.json) and newline-delimited JSON (train.jsonl)
def _atomic_write(path, data, mode='w', encoding='utf-8'):
    tmp = path + '.tmp'
    with open(tmp, mode, encoding=encoding) as fh:
        fh.write(data)
    os.replace(tmp, path)

train_json_path = os.path.join(P_DIR, 'train.json')
val_json_path = os.path.join(P_DIR, 'val.json')
train_jsonl_path = os.path.join(P_DIR, 'train.jsonl')
val_jsonl_path = os.path.join(P_DIR, 'val.jsonl')

_atomic_write(train_json_path, json.dumps(train, ensure_ascii=False))
_atomic_write(val_json_path, json.dumps(val, ensure_ascii=False))

_atomic_write(train_jsonl_path, '\n'.join(json.dumps(r, ensure_ascii=False) for r in train) + '\n')
_atomic_write(val_jsonl_path, '\n'.join(json.dumps(r, ensure_ascii=False) for r in val) + '\n')
print('[OK] N=', N, 'train=', len(train), 'val=', len(val))
