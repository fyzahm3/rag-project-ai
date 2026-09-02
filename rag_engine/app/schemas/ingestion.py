"""Ingestion API contracts."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_name": "kubernetes_platform.md",
                "chunks_created": 6,
                "deduplicated_chunks": 0,
                "strategy": "structure",
                "latency_ms": 318.4,
            }
        }
    )

    document_name: str
    chunks_created: int = Field(ge=0)
    deduplicated_chunks: int = Field(ge=0)
    strategy: str
    latency_ms: float = 0.0


class IndexStatus(BaseModel):
    vector_store: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int | None = None
    chunk_count: int
    bm25_documents: int
    reranker_loaded: bool = False
    llm_available: bool = False
