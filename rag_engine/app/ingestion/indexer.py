"""Ingestion orchestrator: parse -> chunk -> embed -> near-duplicate prune -> dual index."""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from app.config import Settings
from app.ingestion.chunker import Chunk, get_chunker
from app.ingestion.parser import parse_document
from app.retrieval.dense import RetrievedRecord, VectorStore
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.sparse import SparseIndexBase
from app.schemas.ingestion import IngestResponse
from app.utils.errors import IngestionError
from app.utils.text import sha1_id

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        sparse_index: SparseIndexBase,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index

    async def ingest(self, filename: str, content: bytes, strategy: str | None = None) -> IngestResponse:
        return await asyncio.to_thread(self._ingest_sync, filename, content, strategy)

    def _ingest_sync(self, filename: str, content: bytes, strategy: str | None) -> IngestResponse:
        started = time.perf_counter()
        resolved_strategy = (strategy or self.settings.default_chunking_strategy).lower()
        try:
            document = parse_document(filename, content)
            chunker = get_chunker(
                resolved_strategy,
                self.settings,
                embed_fn=self.embedder.embed,
            )
            chunks = chunker.chunk(document)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Ingestion failed for {filename}: {exc}") from exc

        if not chunks:
            raise IngestionError(f"No chunks produced for {filename}; the document may be empty")

        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as exc:
            raise IngestionError(f"Embedding failed for {filename}: {exc}") from exc

        accepted_indices = self._filter_duplicates(chunks, embeddings)
        deduplicated = len(chunks) - len(accepted_indices)
        if not accepted_indices:
            logger.info(
                "All %d candidate chunks from %s were near-duplicates; nothing indexed",
                len(chunks),
                filename,
            )
            return IngestResponse(
                document_name=filename,
                chunks_created=0,
                deduplicated_chunks=deduplicated,
                strategy=resolved_strategy,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        kept_chunks = [chunks[i] for i in accepted_indices]
        kept_embeddings = embeddings[accepted_indices]
        records = [self._to_record(chunk) for chunk in kept_chunks]

        try:
            self.vector_store.add(
                ids=[c.chunk_id for c in kept_chunks],
                embeddings=kept_embeddings,
                documents=[c.text for c in kept_chunks],
                metadatas=[
                    {
                        "document_name": c.document_name,
                        "section_heading": c.section_heading,
                        "page": c.page,
                        "strategy": c.strategy,
                    }
                    for c in kept_chunks
                ],
            )
        except Exception as exc:
            raise IngestionError(f"Vector store write failed for {filename}: {exc}") from exc

        self.sparse_index.add_documents(records)

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Ingested %s: %d chunks created, %d duplicates pruned (%.1f ms)",
            filename,
            len(kept_chunks),
            deduplicated,
            latency_ms,
        )
        return IngestResponse(
            document_name=filename,
            chunks_created=len(kept_chunks),
            deduplicated_chunks=deduplicated,
            strategy=resolved_strategy,
            latency_ms=latency_ms,
        )

    def _to_record(self, chunk: Chunk) -> RetrievedRecord:
        return RetrievedRecord(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            document_name=chunk.document_name,
            section_heading=chunk.section_heading,
            page=chunk.page,
            strategy=chunk.strategy,
        )

    def _filter_duplicates(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[int]:
        """Skip candidates whose cosine similarity vs. the existing index or the
        already-accepted portion of this batch exceeds `dedup_similarity_threshold`."""
        threshold = self.settings.dedup_similarity_threshold
        accepted: list[int] = []
        accepted_vectors: list[np.ndarray] = []

        for index, embedding in enumerate(embeddings):
            vector = np.asarray(embedding, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(vector)) or 1e-12
            unit = vector / norm

            existing_hits = self.vector_store.query(vector, 1)
            if existing_hits and float(existing_hits[0].score) >= threshold:
                logger.debug("Chunk %s pruned: matches existing index", chunks[index].chunk_id)
                continue

            if accepted_vectors:
                stacked = np.stack(accepted_vectors)
                sims = (stacked @ unit) / (np.linalg.norm(stacked, axis=1) + 1e-12)
                if float(np.max(sims)) >= threshold:
                    logger.debug("Chunk %s pruned: intra-batch duplicate", chunks[index].chunk_id)
                    continue

            accepted.append(index)
            accepted_vectors.append(unit)
        return accepted

    def remove_document(self, document_name: str) -> int:
        """Drop every chunk indexed under `document_name` from both stores.

        Used by local-mode file watching to keep the index in sync when a watched
        file is edited (re-ingest replaces stale chunks) or deleted.
        """
        vector_removed = self.vector_store.delete_by_document(document_name)
        sparse_removed = self.sparse_index.remove_by_document(document_name)
        if vector_removed or sparse_removed:
            logger.info(
                "Removed document %s: %d vector chunks, %d sparse entries",
                document_name,
                vector_removed,
                sparse_removed,
            )
        return max(vector_removed, sparse_removed)

    def stats(self) -> dict[str, object]:
        return {
            "vector_store": self.vector_store.backend,
            "chunk_count": self.vector_store.count(),
            "bm25_documents": self.sparse_index.count(),
        }


def make_chunk_uid(document_name: str, heading: str, seq: int) -> str:
    return sha1_id(document_name, heading, seq)
