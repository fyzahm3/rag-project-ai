"""Central application settings (pydantic-settings, 12-factor via env vars / .env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "rag-engine"
    app_version: str = "1.0.0"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_mb: int = 25

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 2

    embedding_provider: Literal["openai", "huggingface"] = "huggingface"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    vector_store_backend: Literal["chroma", "numpy"] = "chroma"
    collection_name: str = "rag_chunks"
    chroma_dir: Path = Path("data/chroma")

    data_dir: Path = Path("data")
    raw_docs_dir: Path = Path("data/raw_docs")
    index_dir: Path = Path("data/index")
    benchmark_path: Path = Path("data/benchmarks/golden_qa.jsonl")
    reports_dir: Path = Path("data/reports")

    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=50, ge=0, le=1000)
    default_chunking_strategy: Literal["fixed", "structure", "semantic"] = "structure"
    semantic_distance_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    semantic_min_sentences: int = Field(default=2, ge=1)
    min_chunk_chars: int = 80
    max_chunk_chars: int = 1200
    dedup_similarity_threshold: float = Field(default=0.95, gt=0.5, lt=1.0)

    dense_top_k: int = Field(default=10, ge=1)
    sparse_top_k: int = Field(default=10, ge=1)
    candidate_pool_size: int = Field(default=20, ge=1)
    rerank_top_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rrf_dense_weight: float = Field(default=0.7, gt=0.0)
    rrf_sparse_weight: float = Field(default=0.3, gt=0.0)

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    warm_models: bool = True

    verifier_mode: Literal["auto", "llm", "nli", "heuristic"] = "auto"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    entailment_supported_threshold: float = 0.55
    entailment_unverified_threshold: float = 0.35
    evidence_max_chars: int = 1500

    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=900, ge=64)

    confidence_retrieval_weight: float = 0.35
    confidence_verification_weight: float = 0.65
    contradiction_penalty: float = 0.25

    eval_default_ks: list[int] = [3, 5]
    eval_include_generation: bool = False
    eval_max_cases: int | None = None

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    def model_post_init(self, __context) -> None:
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_docs_dir,
            self.chroma_dir,
            self.index_dir,
            self.benchmark_path.parent,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
