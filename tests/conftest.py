# tests/conftest.py
from pathlib import Path
import os

def pytest_sessionstart(session):
    cache_root = Path("models_cache").resolve()

    # Force all libs to read/write cache here (no "hub" suffix mismatch).
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root))

    # Also keep HF_HOME for completeness (Transformers builds "<HF_HOME>/hub" if others not set)
    os.environ.setdefault("HF_HOME", str(cache_root))

    # Fully offline for CI/tests
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Silence Windows symlink warnings
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
