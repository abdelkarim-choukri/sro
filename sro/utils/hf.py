# sro/utils/hf.py
from __future__ import annotations
from typing import Optional, Iterable
import os, time
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

def ensure_repo_local(
    repo_id: str,
    local_dir: str = "models_cache",
    allow_patterns: Optional[Iterable[str]] = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    last: Exception | None = None
    for r in range(retries):
        try:
            return snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=list(allow_patterns) if allow_patterns else None,
                ignore_patterns=None,
                max_workers=4,
            )
        except Exception as e:
            last = e
            time.sleep(backoff * (r + 1))
    raise RuntimeError(f"Failed to prefetch {repo_id}: {last}")
