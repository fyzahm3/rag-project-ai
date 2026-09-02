"""Embedding providers: OpenAI text-embedding-3-small or local HuggingFace MiniLM."""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from app.config import Settings
from app.ingestion.chunker import l2_normalize
from app.utils.errors import RetrievalError

logger = logging.getLogger(__name__)

_OPENAI_BATCH_SIZE = 96


class EmbeddingProvider(Protocol):
    name: str
    dim: int | None

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class HuggingFaceEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.name = model_name
        self.dim: int | None = None
        self._batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading HuggingFace embedding model '%s'", self.name)
            self._model = SentenceTransformer(self.name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 384), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    @property
    def loaded(self) -> bool:
        return self._model is not None


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.name = settings.openai_embedding_model
        self.dim: int | None = settings.openai_embedding_dimensions
        self._settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            if not self._settings.openai_api_key:
                raise RetrievalError("OPENAI_API_KEY is not configured")
            self._client = OpenAI(
                api_key=self._settings.openai_api_key,
                timeout=self._settings.openai_timeout_seconds,
                max_retries=self._settings.openai_max_retries,
            )
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        client = self._ensure_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _OPENAI_BATCH_SIZE):
            batch = texts[start : start + _OPENAI_BATCH_SIZE]
            response = client.embeddings.create(
                model=self.name,
                input=batch,
                dimensions=self.dim,
            )
            vectors.extend(item.embedding for item in response.data)
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dim = int(matrix.shape[1])
        return l2_normalize(matrix)

    @property
    def loaded(self) -> bool:
        return self._client is not None


def get_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return HuggingFaceEmbedder(settings.hf_embedding_model)


class HashingEmbedder:
    """Deterministic dependency-free embedder used by tests and offline smoke runs."""

    def __init__(self, dim: int = 128) -> None:
        self.name = "hashing-embedder"
        self.dim: int | None = dim

    def _vectorize(self, text: str) -> np.ndarray:
        import hashlib

        vector = np.zeros(self.dim or 128, dtype=np.float32)
        tokens = [t for t in text.lower().split() if t]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % vector.shape[0]
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
            second = int.from_bytes(digest[5:9], "little") % vector.shape[0]
            vector[second] += sign * 0.5
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 128), dtype=np.float32)
        return l2_normalize(np.stack([self._vectorize(t) for t in texts]).astype(np.float32))
