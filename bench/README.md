# Local BREAK benchmark mirror

Place a small local mirror of the BREAK high-level QDMR data under `bench/local/`.

Recommended file:

`bench/local/break_dev_highlevel.jsonl`

Each line should be a JSON object with at least these fields:

{
  "question": "...",
  "decomposition": ["step1", "step2", ...]
}

Use `python -m scripts.bench_break_to_fixture` to convert to the internal fixture format if needed:

```
python -m scripts.bench_break_to_fixture bench/local/break_dev_highlevel.jsonl bench/break_dev_fixture.jsonl --limit 200
```

Run the benchmark evaluator:

```
python -m scripts.bench_eval --dev bench/break_dev_fixture.jsonl
```

In CI, add `bench/local/break_dev_highlevel.jsonl` to the runner (secure artifact or mirror) to enable the gating test `tests/test_benchmark_break.py`.
