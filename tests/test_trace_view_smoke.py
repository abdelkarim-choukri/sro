import json
import pathlib
import subprocess
import sys
import tempfile


def test_trace_view_generates_html():
    with tempfile.TemporaryDirectory() as td:
        trace = pathlib.Path(td, "t.jsonl")
        out = pathlib.Path(td, "r.html")
        evt = {
            "ts": "2025-01-01T00:00:00Z",
            "type": "NLI_INIT",
            "stage": "S1",
            "data": {"temperature": 1.0},
            "run_id": "r",
        }
        trace.write_text(json.dumps(evt) + "\n", encoding="utf-8")
        code = subprocess.call([sys.executable, "-m", "scripts.trace_view",
                                "--input", str(trace), "--out", str(out)])
        assert code == 0
        assert out.exists() and out.stat().st_size > 0
