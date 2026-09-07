"""Pytest bootstrap: expose the rag_engine package root on sys.path, plus the shared
offline fixtures (dependency-free fakes) used across test modules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.config import Settings
from app.generation.generator import GroundedGenerator
from app.generation.verifier import CitationVerifier
from app.ingestion.indexer import Indexer
from app.pipeline import RAGService
from app.retrieval.dense import DenseRetriever, NumpyVectorStore
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import SparseIndex


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        raw_docs_dir=tmp_path / "raw_docs",
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
        benchmark_path=tmp_path / "benchmarks" / "golden_qa.jsonl",
        reports_dir=tmp_path / "reports",
        vector_store_backend="numpy",
        embedding_provider="huggingface",
        warm_models=False,
        verifier_mode="heuristic",
        min_chunk_chars=40,
        log_level="WARNING",
    )


@pytest.fixture()
def components(settings: Settings):
    embedder = HashingEmbedder(dim=128)
    store = NumpyVectorStore()
    sparse = SparseIndex(settings.index_dir / "bm25_corpus.jsonl")
    dense = DenseRetriever(embedder=embedder, store=store)
    indexer = Indexer(settings=settings, embedder=embedder, vector_store=store, sparse_index=sparse)
    reranker = CrossEncoderReranker(settings)
    generator = GroundedGenerator(settings)
    verifier = CitationVerifier(settings)
    rag = RAGService(
        settings=settings,
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        generator=generator,
        verifier=verifier,
    )
    return {"settings": settings, "embedder": embedder, "store": store, "sparse": sparse,
            "dense": dense, "indexer": indexer, "reranker": reranker,
            "generator": generator, "verifier": verifier, "rag": rag}
