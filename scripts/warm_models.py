# scripts/warm_models.py
from __future__ import annotations
import os
from pathlib import Path
from huggingface_hub import snapshot_download

CACHE = os.getenv("SRO_CACHE_DIR") or os.getenv("HF_HOME") or "models_cache"
Path(CACHE).mkdir(parents=True, exist_ok=True)

# Point every library to the same cache
os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(CACHE) / "hub"))

def prefetch(repo_id: str):
    path = snapshot_download(
        repo_id,
        cache_dir=CACHE,
        local_files_only=False,  # we are ONLINE in the seed job
        allow_patterns="*",
    )
    print(f"[OK] snapshot {repo_id} -> {path}")

def main():
    # Exact repos used in your code/tests
    prefetch("sentence-transformers/all-MiniLM-L6-v2")
    prefetch("roberta-large-mnli")
    prefetch("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[DONE] warm_models")

if __name__ == "__main__":
    main()
