# sro/utils/st.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# Pin cache + offline; we set these again here in case someone imports this module directly
_HF_CACHE = str(Path("models_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_CACHE)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# in-process cache
_REG: dict[str, object] = {}

def get_st(model_id: str):
    """
    Lazily load and cache a SentenceTransformer strictly from the local cache.
    We import sentence_transformers here (lazy) so envs are already set by the caller script.
    """
    if model_id in _REG:
        return _REG[model_id]
    from sentence_transformers import SentenceTransformer  # lazy import
    m = SentenceTransformer(model_id, cache_folder=_HF_CACHE)
    _REG[model_id] = m
    return m
