# scripts/warm_models.py
from sro.utils.hf import ensure_repo_local

MODELS = [
    "roberta-large-mnli",
    "sentence-transformers/all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

def main():
    for m in MODELS:
        local = ensure_repo_local(m, local_dir="models_cache")
        print(f"✓ cached {m} at {local}")

if __name__ == "__main__":
    main()
