# sro/utils/st.py
from __future__ import annotations
import os
from typing import Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer

# simple in-process cache
_ST_POOL: Dict[Tuple[str, str, str], SentenceTransformer] = {}

def get_st(model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
           cache_dir: str | None = None,
           device: str | None = None) -> SentenceTransformer:
    """
    Return a singleton SentenceTransformer with explicit cache + device.
    cache_dir: where HF & ST read/write; default from env or 'models_cache'.
    device: 'cuda' if available else 'cpu' unless overridden.
    """
    cache = cache_dir or os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("TRANSFORMERS_CACHE") or "models_cache"
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_id, cache, dev)
    if key not in _ST_POOL:
        _ST_POOL[key] = SentenceTransformer(model_id, cache_folder=cache, device=dev)
    return _ST_POOL[key]
