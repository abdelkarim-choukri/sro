"""
Does:
    Sentence-embedding backend with offline-first loading, L2-normalized outputs,
    and persistent+LRU caching.

Notes:
    - Resolves model_dir to ABSOLUTE path (robust to chdir in tests).
    - Accepts either config.json OR modules.json (SentenceTransformers layout).
    - Clear offline error message shows EXACT hf download command.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger("sro.embeddings.backend")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as _e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


def _has_local_model_files(path: str) -> bool:
    """
    Return True if the directory looks like a valid SentenceTransformers/Transformers
    model checkout. We are permissive:
      - Accept 'config.json' (Transformers layout), OR
      - Accept 'modules.json' (SentenceTransformers layout),
    and also require at least one weights file.
    """
    if not os.path.isdir(path):
        return False
    any_layout = any(
        os.path.isfile(os.path.join(path, fname))
        for fname in ("config.json", "modules.json")
    )
    if not any_layout:
        return False
    has_weights = any(
        os.path.isfile(os.path.join(path, fname))
        for fname in ("model.safetensors", "pytorch_model.bin")
    )
    return has_weights


def _raise_offline_help(model_name: str, model_dir: str | None, cache_dir: str) -> None:
    msg = [
        "Embedding model not found locally (offline mode). Fix in one of these ways:",
        "1) Point to a local model directory containing modules.json/config.json and model weights:",
        "   - set env SRO_EMBED_MODEL_DIR=path\\to\\model_dir",
        "   - or pass model_dir='path\\to\\model_dir' to EmbeddingBackend(...)",
        "2) Pre-download the repo into a local folder BEFORE going offline, e.g.:",
        f"   hf download {model_name} --local-dir {os.path.join(cache_dir, model_name.replace('/', os.sep))}",
        f"(attempted model_name='{model_name}', model_dir='{model_dir}', cache_dir='{cache_dir}')",
    ]
    raise FileNotFoundError("\n".join(msg))


class EmbeddingBackend:
    """
    Offline-first sentence embedding with L2-normalized outputs + caching.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        model_dir: str | None = None,
        cache_dir: str = "models_cache",
        cache_size: int = 1000,
        device: str | None = None,
    ) -> None:
        if SentenceTransformer is None:  # pragma: no cover
            raise ImportError(
                "sentence-transformers package not available. Please `pip install sentence-transformers`."
            )

        self.model_name = model_name or os.environ.get(
            "SRO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        raw_dir = model_dir or os.environ.get("SRO_EMBED_MODEL_DIR")
        # Make model_dir absolute so tests that chdir() don't break relative paths
        self.model_dir = os.path.abspath(raw_dir) if raw_dir else None
        self.cache_dir = cache_dir
        self.device = device  # let SBERT pick CUDA if available

        # Persistent cache (relative to CWD by design; tests chdir into tmp -> isolated DB)
        os.makedirs(os.path.join("artifacts", "cache"), exist_ok=True)
        self.db_path = os.path.join("artifacts", "cache", "embeddings.sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS embeddings (id TEXT PRIMARY KEY, vec BLOB)")
        self.conn.commit()

        # In-memory LRU
        self.cache_size = int(cache_size)
        self._mem: dict[str, np.ndarray] = {}
        self._order: list[str] = []

        # Load model offline-first
        self.model = self._load_model_offline()

    # ------------------------ model loading ------------------------
    def _load_model_offline(self):
        """
        Load SBERT model offline-first:
        - If model_dir is set, load from that directory (absolute).
        - Else try cache_folder=cache_dir (must already be populated).
        - On failure, raise clear FileNotFoundError with instructions.
        """
        try:
            if self.model_dir:
                if not _has_local_model_files(self.model_dir):
                    _raise_offline_help(self.model_name, self.model_dir, self.cache_dir)
                _LOGGER.info("Embedding model: loading from local dir: %s", self.model_dir)
                return SentenceTransformer(self.model_dir, device=self.device)
            else:
                local_path = os.path.join(self.cache_dir, self.model_name.replace("/", os.sep))
                local_path = os.path.abspath(local_path)
                if _has_local_model_files(local_path):
                    _LOGGER.info("Embedding model: loading from cache dir: %s", local_path)
                    return SentenceTransformer(local_path, device=self.device)
                _LOGGER.info(
                    "Embedding model: trying repo id from cache_folder='%s' (must be available offline)", self.cache_dir
                )
                # If HF_HUB_OFFLINE=1 and cache is missing, ST will fail -> caught below
                return SentenceTransformer(self.model_name, cache_folder=self.cache_dir, device=self.device)
        except Exception as e:
            _LOGGER.error("Failed to load embedding model: %s", e)
            _raise_offline_help(self.model_name, self.model_dir, self.cache_dir)
            raise  # unreachable

    # ------------------------ LRU helpers ------------------------
    def _touch(self, key: str) -> None:
        if key in self._mem:
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)

    def _add(self, key: str, vec: np.ndarray) -> None:
        if key in self._mem:
            self._mem[key] = vec
            self._touch(key)
            return
        if len(self._order) >= self.cache_size:
            old = self._order.pop(0)
            self._mem.pop(old, None)
        self._mem[key] = vec
        self._order.append(key)

    # ------------------------ public API ------------------------
    def encode(self, text: str, sent_id: str | None = None) -> np.ndarray:
        """
        Return L2-normalized float32 vector for 'text'. If 'sent_id' is given,
        use it as cache key; else use md5(text).
        """
        key = sent_id or ("md5:" + hashlib.md5(text.encode("utf-8")).hexdigest())

        # in-mem LRU
        v = self._mem.get(key)
        if v is not None:
            self._touch(key)
            return v

        # persistent cache
        cur = self.conn.execute("SELECT vec FROM embeddings WHERE id=?", (key,))
        row = cur.fetchone()
        if row:
            arr = np.frombuffer(row[0], dtype=np.float32)
            self._add(key, arr)
            return arr

        # compute via model (normalize_embeddings=True yields unit vectors)
        vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

        # persist
        self.conn.execute("INSERT OR REPLACE INTO embeddings (id, vec) VALUES (?,?)", (key, vec.tobytes()))
        self.conn.commit()
        # LRU
        self._add(key, vec)
        return vec

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
