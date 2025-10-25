# tests/test_calibration_io.py
"""
Does:
    Validate calibration JSON schema handling and model-name matching, without requiring
    an actual model to load. Tests the pure helper function.
"""
import json
import os
import tempfile

from sro.nli.backend import _read_temperature_json


def test_read_temperature_json_schema_and_match():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "nli_temperature.json")

        # Bad schema -> 1.0
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"foo": "bar"}, f)
        assert _read_temperature_json(p, "m") == 1.0

        # Wrong model -> 1.0
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"model": "wrong", "T": 2.0, "updated_at": "x"}, f)
        assert _read_temperature_json(p, "right") == 1.0

        # Non-positive T -> 1.0
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"model": "right", "T": 0.0, "updated_at": "x"}, f)
        assert _read_temperature_json(p, "right") == 1.0

        # Good case -> clamped T
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"model": "right", "T": 123.0, "updated_at": "x"}, f)
        assert _read_temperature_json(p, "right") == 10.0  # clamp

        with open(p, "w", encoding="utf-8") as f:
            json.dump({"model": "right", "T": 0.000001, "updated_at": "x"}, f)
        assert _read_temperature_json(p, "right") == 0.01  # clamp
