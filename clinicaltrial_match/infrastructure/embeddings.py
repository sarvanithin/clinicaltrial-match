"""
Sentence-transformers embedding model wrapper.

Manages:
- Model loading (all-MiniLM-L6-v2, 384-dim)
- BLOB serialization/deserialization for SQLite storage
- In-memory numpy cosine similarity index
- Optional FAISS index for large collections (>threshold trials)
"""

from __future__ import annotations

import numpy as np

from clinicaltrial_match.config import EmbeddingConfig
from clinicaltrial_match.infrastructure.db import Database


class EmbeddingIndex:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None
        self._nct_ids: list[str] = []
        self._matrix: np.ndarray | None = None  # shape (N, 384)
        self._faiss_index = None

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._config.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    async def encode_one_async(self, text: str) -> np.ndarray:
        """Run encoding in a thread pool so it never blocks the async event loop."""
        import asyncio
        import functools

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(self.encode_one, text))

    def warmup(self) -> None:
        """Pre-load the model and run a dummy encode so first real request is instant."""
        self._load_model()
        self.encode(["warmup"])

    @staticmethod
    def to_bytes(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def from_bytes(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def load_from_db(self, db: Database) -> None:
        """Load all stored trial embeddings into memory."""
        rows = db.get_all_trial_embeddings()
        if not rows:
            self._nct_ids = []
            self._matrix = None
            return
        self._nct_ids = [r[0] for r in rows]
        vecs = [self.from_bytes(r[1]) for r in rows]
        self._matrix = np.vstack(vecs)
        if self._config.use_faiss and len(self._nct_ids) >= self._config.faiss_rebuild_threshold:
            self._build_faiss()

    def add(self, nct_id: str, vec: np.ndarray) -> None:
        """Add a single trial embedding to the in-memory index."""
        if nct_id in self._nct_ids:
            idx = self._nct_ids.index(nct_id)
            if self._matrix is not None:
                self._matrix[idx] = vec
        else:
            self._nct_ids.append(nct_id)
            if self._matrix is None:
                self._matrix = vec.reshape(1, -1)
            else:
                self._matrix = np.vstack([self._matrix, vec])
        n = len(self._nct_ids)
        if self._config.use_faiss and n >= self._config.faiss_rebuild_threshold:
            self._build_faiss()

    def _build_faiss(self) -> None:
        try:
            import faiss  # type: ignore[import]

            dim = self._matrix.shape[1]  # type: ignore[union-attr]
            index = faiss.IndexFlatIP(dim)
            index.add(self._matrix.astype(np.float32))  # type: ignore[union-attr]
            self._faiss_index = index
        except ImportError:
            pass  # fall back to numpy

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return (nct_id, score) pairs sorted by descending similarity."""
        if self._matrix is None or len(self._nct_ids) == 0:
            return []
        top_k = min(top_k, len(self._nct_ids))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vec.reshape(1, -1).astype(np.float32), top_k)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((self._nct_ids[idx], float(score)))
            return results
        # numpy cosine (vectors already normalized)
        scores = self._matrix @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._nct_ids[i], float(scores[i])) for i in top_indices]

    @property
    def has_index(self) -> bool:
        """True when there are trial embeddings to search against."""
        return self._matrix is not None and len(self._nct_ids) > 0

    @property
    def is_model_ready(self) -> bool:
        """True when the sentence-transformers model is loaded and can encode queries."""
        return self._model is not None

    def size(self) -> int:
        return len(self._nct_ids)
