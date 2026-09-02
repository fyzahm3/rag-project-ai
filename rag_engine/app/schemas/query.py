from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How fast are volumes provisioned on the storage layer?",
                "top_k": 5,
                "enable_reranking": True,
                "chunking_strategy": "structure",
            }
        }
    )

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    enable_reranking: bool = True
    chunking_strategy: str = "structure"

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SourceCitation(BaseModel):
    citation_id: int = Field(ge=1)
    chunk_id: str
    source_doc: str
    section_heading: str
    text: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    page: int | None = None


class ClaimVerification(BaseModel):
    claim: str
    citation_ids: list[int]
    status: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]
    rationale: str = ""


class QueryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Volumes on the ceph-rbd-ssd storage class provision in under 900 milliseconds [1].",
                "citations": [
                    {
                        "citation_id": 1,
                        "chunk_id": "kubernetes_platfor_3a91f2",
                        "source_doc": "kubernetes_platform.md",
                        "section_heading": "Storage Layer",
                        "text": "...",
                        "relevance_score": 0.93,
                        "page": None,
                    }
                ],
                "confidence_score": 0.87,
                "retrieval_latency_ms": 41.2,
                "generation_latency_ms": 980.5,
                "verification_status": {
                    "engine": "heuristic",
                    "flag": "ok",
                    "counts": {"SUPPORTED": 1, "CONTRADICTED": 0, "UNVERIFIED": 0},
                    "pass_rate": 1.0,
                    "claims": [],
                },
            }
        }
    )

    answer: str
    citations: list[SourceCitation]
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieval_latency_ms: float
    generation_latency_ms: float
    verification_status: dict[str, Any]
