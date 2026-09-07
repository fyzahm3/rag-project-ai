"""Vector persistence backends (ChromaDB persistent + pure-NumPy fallback) and dense retrieval."""
from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel

from app.config import Settings
from app.ingestion.chunker import l2_normalize
from app.retrieval.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class RetrievedRecord(BaseModel):
    chunk_id: str
    text: str
    document_name: str
    section_heading: str = "(untitled)"
    page: int | None = None
    strategy: str = "structure"
    score: float = 0.0
    raw_score: float | None = None


class VectorStore(Protocol):
    backend: str

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ...

    def query(self, embedding: np.ndarray, top_k: int) -> list[RetrievedRecord]:
        ...

    def count(self) -> int:
        ...

    def delete_by_document(self, document_name: str) -> int:
        ...


def _record_from_meta(chunk_id: str, text: str, similarity: float, metadata: dict[str, Any]) -> RetrievedRecord:
    return RetrievedRecord(
        chunk_id=chunk_id,
        text=text,
        document_name=str(metadata.get("document_name", "unknown")),
        section_heading=str(metadata.get("section_heading", "(untitled)")),
        page=metadata.get("page"),
        strategy=str(metadata.get("strategy", "structure")),
        score=float(similarity),
        raw_score=float(similarity),
    )


class ChromaVectorStore:
    backend = "chroma"

    def __init__(self, settings: Settings) -> None:
        import chromadb

        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready at %s (%d chunks)",
            settings.collection_name,
            settings.chroma_dir,
            self.count(),
        )

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        clean_meta = [{k: v for k, v in m.items() if v is not None} for m in metadatas]
        self._collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=documents,
            metadatas=clean_meta,
        )

    def query(self, embedding: np.ndarray, top_k: int) -> list[RetrievedRecord]:
        if self.count() == 0 or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        records: list[RetrievedRecord] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, chunk_id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            similarity = 1.0 - distance
            metadata = metas[i] if i < len(metas) and metas[i] else {}
            records.append(_record_from_meta(chunk_id, docs[i] if i < len(docs) else "", similarity, metadata))
        return records

    def count(self) -> int:
        return int(self._collection.count())

    def delete_by_document(self, document_name: str) -> int:
        if self.count() == 0:
            return 0
        existing = self._collection.get(where={"document_name": document_name}, include=[])
        ids = existing.get("ids", [])
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)


class NumpyVectorStore:
    """Brute-force cosine store; used for tests, CI and tiny deployments without chromadb."""

    backend = "numpy"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._docs: dict[str, tuple[str, dict[str, Any]]] = {}

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        with self._lock:
            vectors = l2_normalize(np.asarray(embeddings, dtype=np.float32))
            self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
            for chunk_id, doc, meta in zip(ids, documents, metadatas):
                self._ids.append(chunk_id)
                self._docs[chunk_id] = (doc, meta)

    def query(self, embedding: np.ndarray, top_k: int) -> list[RetrievedRecord]:
        with self._lock:
            if self._matrix is None or not self._ids or top_k <= 0:
                return []
            vector = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
            similarities = (self._matrix @ vector.T).ravel()[: len(self._ids)]
        order = np.argsort(-similarities)[:top_k]
        records: list[RetrievedRecord] = []
        for index in order:
            chunk_id = self._ids[int(index)]
            doc, meta = self._docs.get(chunk_id, ("", {}))
            records.append(_record_from_meta(chunk_id, doc, float(similarities[int(index)]), meta))
        return records

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def delete_by_document(self, document_name: str) -> int:
        with self._lock:
            if not self._ids:
                return 0
            keep_indices = [
                i
                for i, chunk_id in enumerate(self._ids)
                if self._docs.get(chunk_id, ("", {}))[1].get("document_name") != document_name
            ]
            removed = len(self._ids) - len(keep_indices)
            if removed == 0:
                return 0
            keep_set = set(keep_indices)
            for i, chunk_id in enumerate(self._ids):
                if i not in keep_set:
                    self._docs.pop(chunk_id, None)
            self._ids = [self._ids[i] for i in keep_indices]
            self._matrix = self._matrix[keep_indices] if self._matrix is not None else None
            return removed


def get_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store_backend == "chroma":
        return ChromaVectorStore(settings)
    return NumpyVectorStore()


class DenseRetriever:
    """Embeds the query and returns top-N chunks by cosine similarity."""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def query_sync(self, query: str, top_k: int) -> list[RetrievedRecord]:
        matrix = self.embedder.embed([query])
        if matrix.shape[0] == 0:
            return []
        return self.store.query(matrix[0], top_k)

    async def query(self, query: str, top_k: int) -> list[RetrievedRecord]:
        import asyncio

        return await asyncio.to_thread(self.query_sync, query, top_k)
