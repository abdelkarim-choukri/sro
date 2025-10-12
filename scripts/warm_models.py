# scripts/warm_models.py
from __future__ import annotations
import os
from pathlib import Path

CACHE = os.getenv("SRO_CACHE_DIR") or os.getenv("HF_HOME") or "models_cache"
Path(CACHE).mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE)

def warm_roberta_mnli():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model = "roberta-large-mnli"
    AutoTokenizer.from_pretrained(model, cache_dir=CACHE)
    AutoModelForSequenceClassification.from_pretrained(model, cache_dir=CACHE)
    print(f"[OK] cached {model} -> {CACHE}")

def warm_st():
    from sentence_transformers import SentenceTransformer
    model = "sentence-transformers/all-MiniLM-L6-v2"
    SentenceTransformer(model, cache_folder=CACHE)
    print(f"[OK] cached {model} -> {CACHE}")

def warm_ce():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    try:
        AutoTokenizer.from_pretrained(model, cache_dir=CACHE)
        AutoModelForSequenceClassification.from_pretrained(model, cache_dir=CACHE)
        print(f"[OK] cached {model} -> {CACHE}")
    except Exception as e:
        print(f"[WARN] could not cache {model}: {e}")

def main():
    warm_roberta_mnli()
    warm_st()
    warm_ce()
    print("[DONE] warm_models")

if __name__ == "__main__":
    main()
