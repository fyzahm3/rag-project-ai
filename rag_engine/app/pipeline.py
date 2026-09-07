"""End-to-end RAG pipeline orchestration (retrieval -> fusion -> rerank -> generate -> verify)."""
from __future__ import annotations

import asyncio
import logging
import time

from app.config import Settings
from app.generation.generator import DraftAnswer, GroundedGenerator
from app.generation.verifier import CitationVerifier, compute_confidence
from app.ingestion.indexer import Indexer
from app.retrieval.dense import DenseRetriever, RetrievedRecord
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import SparseIndexBase
from app.schemas.eval import StrategyName
from app.schemas.query import QueryRequest, QueryResponse, SourceCitation
from app.utils.errors import RetrievalError
from app.utils.text import mean as safe_mean

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        settings: Settings,
        dense: DenseRetriever,
        sparse: SparseIndexBase,
        reranker: CrossEncoderReranker,
        generator: GroundedGenerator,
        verifier: CitationVerifier,
    ) -> None:
        self.settings = settings
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.generator = generator
        self.verifier = verifier

    async def retrieve(
        self,
        strategy: str,
        query: str,
        k: int,
    ) -> list[RetrievedRecord]:
        try:
            if strategy == "dense":
                return await self.dense.query(query, k)
            if strategy == "sparse":
                return await self.sparse.query(query, k)
            if strategy in {"hybrid_rrf", "hybrid_rerank"}:
                dense_task = asyncio.create_task(
                    self.dense.query(query, self.settings.dense_top_k)
                )
                sparse_task = asyncio.create_task(
                    self.sparse.query(query, self.settings.sparse_top_k)
                )
                dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
                fused = reciprocal_rank_fusion(
                    [dense_hits, sparse_hits],
                    k=self.settings.rrf_k,
                    weights=[self.settings.rrf_dense_weight, self.settings.rrf_sparse_weight],
                )
                pool = fused[: self.settings.candidate_pool_size]
                if strategy == "hybrid_rerank":
                    return await self._rerank_or_slice(query, pool, k)
                return pool[:k]
            raise ValueError(f"Unknown retrieval strategy '{strategy}'")
        except ValueError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed ({strategy}): {exc}") from exc

    async def _rerank_or_slice(
        self,
        query: str,
        pool: list[RetrievedRecord],
        top_k: int,
    ) -> list[RetrievedRecord]:
        if not pool:
            return []
        if self.reranker.ensure_model():
            return await self.reranker.rerank(query, pool, top_k)
        return pool[:top_k]

    def build_citations(self, records: list[RetrievedRecord]) -> list[SourceCitation]:
        return [
            SourceCitation(
                citation_id=index + 1,
                chunk_id=record.chunk_id,
                source_doc=record.document_name,
                section_heading=record.section_heading,
                text=record.text,
                relevance_score=float(min(max(record.score, 0.0), 1.0)),
                page=record.page,
            )
            for index, record in enumerate(records)
        ]

    async def generate(
        self,
        query: str,
        citations: list[SourceCitation],
    ) -> DraftAnswer:
        return await self.generator.generate(query, citations)

    async def ask(self, request: QueryRequest) -> QueryResponse:
        overall_started = time.perf_counter()
        logger.info("ask(): query=%r top_k=%d rerank=%s", request.query, request.top_k, request.enable_reranking)

        retrieval_started = time.perf_counter()
        dense_task = asyncio.create_task(
            self.dense.query(request.query, self.settings.dense_top_k)
        )
        sparse_task = asyncio.create_task(
            self.sparse.query(request.query, self.settings.sparse_top_k)
        )
        try:
            dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        except Exception as exc:
            raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=self.settings.rrf_k,
            weights=[self.settings.rrf_dense_weight, self.settings.rrf_sparse_weight],
        )
        candidate_pool = fused[: self.settings.candidate_pool_size]

        if request.enable_reranking and candidate_pool:
            final_records = await self._rerank_or_slice(
                request.query, candidate_pool, request.top_k
            )
        else:
            final_records = candidate_pool[: request.top_k]

        citations = self.build_citations(final_records)

        generation_started = time.perf_counter()
        draft = await self.generate(request.query, citations)
        verification = await self.verifier.verify(draft.answer, citations)
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000

        mean_relevance = safe_mean([c.relevance_score for c in citations])
        confidence = compute_confidence(
            mean_relevance=mean_relevance,
            verification=verification,
            retrieval_weight=self.settings.confidence_retrieval_weight,
            verification_weight=self.settings.confidence_verification_weight,
            contradiction_penalty=self.settings.contradiction_penalty,
        )

        total_ms = (time.perf_counter() - overall_started) * 1000
        logger.info(
            "ask(): done in %.1f ms (retrieval %.1f ms, generation+verify %.1f ms); "
            "%d citations, confidence=%.3f, engine=%s",
            total_ms,
            retrieval_latency_ms,
            generation_latency_ms,
            len(citations),
            confidence,
            verification.get("engine"),
        )
        return QueryResponse(
            answer=draft.answer,
            citations=citations,
            confidence_score=confidence,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            verification_status=verification,
        )
