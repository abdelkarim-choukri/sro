#!/usr/bin/env python3
import json, collections, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_rows(path_jsonl, path_json):
    if os.path.exists(path_jsonl):
        try:
            with open(path_jsonl, encoding='utf-8') as fh:
                return [json.loads(l) for l in fh if l.strip()]
        except Exception as e:
            print(f"ERROR reading {path_jsonl}: {e}", file=sys.stderr)
            return []
    if os.path.exists(path_json):
        try:
            with open(path_json, encoding='utf-8') as fh:
                return json.load(fh)
        except Exception as e:
            print(f"ERROR reading {path_json}: {e}", file=sys.stderr)
            return []
    print(f"Missing both {path_jsonl} and {path_json}", file=sys.stderr)
    return []


def main():
    for rel in ("artifacts/splitter/train.jsonl", "artifacts/splitter/val.jsonl"):
        full_jsonl = os.path.join(ROOT, rel)
        full_json = full_jsonl.replace('.jsonl', '.json')
        rows = read_rows(full_jsonl, full_json)
        n_rows = len(rows)
        n_tok = sum(len(r.get('tokens', [])) for r in rows)
        flat = [y for r in rows for y in r.get('labels', [])]
        cnt = collections.Counter(flat)
        mismatch_count = sum(1 for r in rows if len(r.get('tokens', [])) != len(r.get('labels', [])))
        # sample first row info
        sample = None
        if rows:
            r0 = rows[0]
            sample = {
                'has_text': 'text' in r0,
                'tokens_shape': [(t.get('start'), t.get('end')) for t in r0.get('tokens', [])[:5]],
                'labels_head': r0.get('labels', [])[:10]
            }
        print(f"{os.path.basename(full_jsonl)} rows={n_rows} tokens={n_tok} label_counts={dict(cnt)} mismatches={mismatch_count} sample={sample}")

if __name__ == '__main__':
    main()
