"""Evaluation API contracts and benchmark dataset models."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

StrategyName = Literal["dense", "sparse", "hybrid_rrf", "hybrid_rerank"]

ALL_STRATEGIES: tuple[StrategyName, ...] = ("dense", "sparse", "hybrid_rrf", "hybrid_rerank")


class GoldenCase(BaseModel):
    question: str = Field(min_length=1)
    expected_answer: str | None = None
    relevant_snippets: list[str] = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("relevant_snippets", "relevant_chunk_ids", "tags")
    @classmethod
    def _strip_lists(cls, value: list[str]) -> list[str]:
        return [v for v in (item.strip() for item in value) if v]

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset_path": "data/benchmarks/golden_qa.jsonl",
                "ks": [3, 5],
                "strategies": ["dense", "sparse", "hybrid_rrf", "hybrid_rerank"],
                "include_generation": True,
                "limit": 20,
            }
        }
    )

    dataset_path: str | None = None
    ks: list[int] = Field(default_factory=lambda: [3, 5], min_length=1)
    strategies: list[StrategyName] = Field(default_factory=lambda: list(ALL_STRATEGIES))
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_generation: bool = False
    limit: int | None = Field(default=None, ge=1)

    @field_validator("ks")
    @classmethod
    def _positive_ks(cls, value: list[int]) -> list[int]:
        if any(k < 1 for k in value):
            raise ValueError("all k values must be >= 1")
        return sorted(set(value))


class RetrievalStrategyMetrics(BaseModel):
    strategy: StrategyName
    num_cases: int
    recall_at_k: dict[str, float]
    mrr: float


class FaithfulnessStats(BaseModel):
    total_answers: int = 0
    fully_grounded_answers: int = 0
    total_claims: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    unverified_claims: int = 0
    groundedness_rate: float = 0.0
    answer_grounding_rate: float = 0.0


class StrategyReport(BaseModel):
    retrieval: RetrievalStrategyMetrics
    faithfulness: FaithfulnessStats | None = None


class EvalReport(BaseModel):
    run_id: str
    dataset: str
    generated_at: datetime
    duration_seconds: float
    num_cases: int
    ks: list[int]
    results: dict[str, StrategyReport]
    markdown_table: str
