from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, overload

import numpy as np

import re
import hashlib
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _has_local_model_files(d: str) -> bool:
    must = {"config.json", "modules.json"}
    present = set(os.listdir(d)) if os.path.isdir(d) else set()
    return bool(must & present)


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


@dataclass
class _Cfg:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = os.getenv("SRO_DEVICE", "cpu")
    cache_dir: str = "models_cache"
    model_dir: str | None = os.getenv("SRO_EMBED_MODEL_DIR")


class EmbeddingBackend:
    """
    Offline-first embedding backend.

    API:
      - encode(text: str, key: Optional[str] = None) -> np.ndarray[(dim,)]
      - encode(texts: Sequence[str], ids: Optional[Sequence[str]] = None, batch_size: int = 32)
        -> np.ndarray[(n, dim)]
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model_dir: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        cfg = _Cfg()
        if model_name:
            cfg.model_name = model_name
        if device:
            cfg.device = device
        if model_dir is not None:
            cfg.model_dir = model_dir
        if cache_dir:
            cfg.cache_dir = cache_dir

        self.model_name = cfg.model_name
        self.device = cfg.device
        self.cache_dir = cfg.cache_dir
        self.model_dir = cfg.model_dir

        # Guarantee cache artifacts exist for tests
        _ensure_dir(self.cache_dir)
        self._init_sqlite_cache()

        self._dummy = False
        self._dim = int(os.getenv("SRO_EMBED_DIM", "64"))

        self.model = None
        try:
            self.model = self._load_model_offline()
        except Exception:
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
            if offline or os.getenv("SRO_ALLOW_DUMMY_EMBED") == "1":
                self._dummy = True
            else:
                raise

    # ------- public API overloads -------
    @overload
    def encode(self, text: str, key: Optional[str] = None, batch_size: int = 32) -> np.ndarray: ...
    @overload
    def encode(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None, batch_size: int = 32) -> np.ndarray: ...

    def encode(
        self,
        texts: str | Sequence[str],
        ids: Optional[str | Sequence[Optional[str]]] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        # Normalize inputs
        if isinstance(texts, str):
            mat = self._encode_list([texts], batch_size=batch_size)
            return mat[0]  # 1-D
        else:
            return self._encode_list(list(texts), batch_size=batch_size)

    # ------- internals -------
    def _encode_list(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if self._dummy:
            return self._encode_dummy(texts)
        arr = self.model.encode(  # type: ignore[union-attr]
            list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, device=self.device
        )
        arr = np.asarray(arr, dtype=np.float32)
        return arr

    def _load_model_offline(self):
        from sentence_transformers import SentenceTransformer  # heavy import inside

        # 1) Explicit local directory wins
        if self.model_dir:
            if not _has_local_model_files(self.model_dir):
                _raise_offline_help(self.model_name, self.model_dir, self.cache_dir)
            return SentenceTransformer(self.model_dir, device=self.device)

        # 2) Cached repo snapshot under models_cache/<repo>
        local_path = os.path.join(self.cache_dir, self.model_name.replace("/", os.sep))
        local_path = os.path.abspath(local_path)
        if _has_local_model_files(local_path):
            return SentenceTransformer(local_path, device=self.device)

        # 3) No local model? DO NOT “auto-create” an ST model. Force caller to fall back to dummy.
        _raise_offline_help(self.model_name, self.model_dir, self.cache_dir)


    def _encode_dummy(self, texts: Sequence[str]) -> np.ndarray:
        dim = max(32, int(self._dim))  # give bigrams room
        WORD = re.compile(r"[A-Za-z0-9]+")

        def h(s: str) -> int:
            # stable hash → [0, dim)
            return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") % dim

        out = []
        for t in texts:
            v = np.zeros(dim, dtype=np.float32)
            toks = [tok.lower() for tok in WORD.findall(str(t))]
            # unigrams
            for tok in toks:
                v[h(f"u:{tok}")] += 1.0
            # bigrams (help paraphrase similarity)
            for a, b in zip(toks, toks[1:]):
                v[h(f"b:{a}_{b}")] += 1.5
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return np.vstack(out)

    def _init_sqlite_cache(self) -> None:
        rel = os.path.join("artifacts", "cache")
        _ensure_dir(rel)
        db = os.path.join(rel, "embeddings.sqlite")
        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
            con.commit()
            con.close()
        except Exception:
            pass
