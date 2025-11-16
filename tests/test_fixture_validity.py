import subprocess
import sys


def test_fixture_validates_and_has_min_items():
    path = "tests/fixtures/splitter_dev.jsonl"
    # Fail loudly if fixture doesn’t validate against tokenizer/PRT
    subprocess.run([sys.executable, "-m", "scripts.validate_fixture", path], check=True)
    # Also ensure there are at least 12 items (guard against accidental truncation)
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f if _.strip())
    assert n >= 12
