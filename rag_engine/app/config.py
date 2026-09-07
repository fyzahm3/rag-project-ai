"""Central application settings (pydantic-settings, 12-factor via env vars / .env).

Two profiles share this one Settings model:
  - "server" (default): the FastAPI service / Docker deployment. Behavior here is
    unchanged from before local mode existed.
  - "local": the tray daemon (app/local/daemon.py). Config additionally loads from a
    user-editable YAML file in the per-OS app data dir (see app/local/config_yaml.py
    and get_settings() below) and a handful of fields switch their *default* value
    (not their type) to local-friendly choices — only for fields the user hasn't
    explicitly set via env var, .env, or the YAML file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.paths import app_data_dir


def _parse_path_list(value: object) -> object:
    """Accept a JSON list, or a comma-separated string, for a list[Path]/list[str] field."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            return value
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return value


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

    # "server" runs the FastAPI service (unchanged production path, Docker deployment,
    # eval harness). "local" runs the tray daemon instead — same Settings model, same
    # Indexer/RAGService, different entry point and a different set of defaults below.
    profile: Literal["server", "local"] = "server"

    # ---- Local mode ----
    watched_paths: list[Path] = Field(default_factory=list)
    excluded_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build",
            "Library", "AppData", "*.app", "*.exe", "*.dll", "*.zip", "*.dmg", "*.iso",
        ]
    )
    max_index_file_size_mb: int = Field(default=25, ge=1)
    local_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    local_top_k: int = Field(default=8, ge=1)
    local_debounce_seconds: float = Field(default=2.0, ge=0.0)
    max_concurrent_indexing: int = Field(default=2, ge=1)

    @field_validator("watched_paths", "excluded_patterns", mode="before")
    @classmethod
    def _parse_local_lists(cls, value: object) -> object:
        return _parse_path_list(value)

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_mb: int = 25

    api_key: str | None = None
    rate_limit_per_minute: int = Field(default=30, ge=1)

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 2

    # Generation LLM provider. Server profile defaults to "openai" (unchanged).
    # Local profile defaults to "ollama" unless OPENAI_API_KEY is set, in which case
    # it defaults to "openai" — either way, GroundedGenerator falls back to the other
    # provider at request time if the preferred one errors, then to extractive.
    llm_provider: Literal["openai", "ollama"] = "openai"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model: str = "qwen2.5:7b"

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
        if self.profile == "local":
            self._apply_local_profile_defaults()
        self.ensure_dirs()

    def _apply_local_profile_defaults(self) -> None:
        """Only touches fields the user did not explicitly set (env var, .env, YAML,
        or a direct kwarg) — model_fields_set reflects all of those sources."""
        set_fields = self.model_fields_set
        base = app_data_dir()

        if "data_dir" not in set_fields:
            self.data_dir = base
        if "chroma_dir" not in set_fields:
            self.chroma_dir = base / "chroma"
        if "index_dir" not in set_fields:
            self.index_dir = base / "index"
        if "reports_dir" not in set_fields:
            self.reports_dir = base / "reports"
        if "raw_docs_dir" not in set_fields:
            self.raw_docs_dir = base / "raw_docs"
        if "benchmark_path" not in set_fields:
            self.benchmark_path = base / "benchmarks" / "golden_qa.jsonl"

        if "embedding_provider" not in set_fields:
            self.embedding_provider = "huggingface"
        if "hf_embedding_model" not in set_fields:
            self.hf_embedding_model = "BAAI/bge-small-en-v1.5"

        if "api_host" not in set_fields:
            self.api_host = "127.0.0.1"

        if "llm_provider" not in set_fields:
            self.llm_provider = "openai" if self.openai_api_key else "ollama"

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
    """The single Settings entrypoint for both profiles.

    Profile is decided by the PROFILE env var (read directly, since Settings itself
    doesn't exist yet at this point) before construction. For "local", the per-OS
    YAML config file is loaded (generated with defaults on first run) and merged in
    as explicit field values — which, per pydantic-settings precedence, take priority
    over .env/env vars for the fields it sets, while anything the YAML doesn't
    mention still falls through to env vars / .env / built-in defaults normally.
    """
    import os

    if os.environ.get("PROFILE", "").strip().lower() == "local":
        from app.local.config_yaml import load_local_yaml_overrides

        overrides = load_local_yaml_overrides()
        overrides["profile"] = "local"
        return Settings(**overrides)
    return Settings()
