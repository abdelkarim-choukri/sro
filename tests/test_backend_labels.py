# tests/test_backend_labels.py
import pytest
from sro.nli.backend import _normalize_nli_label_map

def test_normalize_nli_label_map_three_class():
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    m = _normalize_nli_label_map(id2label)
    assert set(m.keys()) == {"entailment", "contradiction", "neutral"}
    assert m["entailment"] == 0
    assert m["neutral"] == 1
    assert m["contradiction"] == 2

def test_normalize_nli_label_map_variants():
    id2label = {0: "Contradictory", 1: "Entails", 2: "Neutral"}
    m = _normalize_nli_label_map(id2label)
    assert m["entailment"] == 1
    assert m["contradiction"] == 0
    assert m["neutral"] == 2

def test_normalize_nli_label_map_binary():
    id2label = {0: "entailment", 1: "not_entailment"}
    m = _normalize_nli_label_map(id2label)
    assert set(m.keys()) == {"entailment", "not_entailment"}
    assert m["entailment"] == 0
    assert m["not_entailment"] == 1

def test_normalize_nli_label_map_missing():
    id2label = {0: "entailment", 1: "neutral"}  # missing contradiction, no not_entailment
    with pytest.raises(ValueError):
        _normalize_nli_label_map(id2label)
