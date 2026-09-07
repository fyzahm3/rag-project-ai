"""Tests for app.tray.LocalDaemon: the server-thread + watcher-thread wiring behind
the tray app, independent of pystray/webbrowser (only touched in main(), not in
LocalDaemon itself)."""
from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings
from app.generation.generator import GroundedGenerator
from app.generation.verifier import CitationVerifier
from app.ingestion.indexer import Indexer
from app.main import AppContext
from app.pipeline import RAGService
from app.retrieval.dense import DenseRetriever, NumpyVectorStore
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import FTS5SparseIndex
from app.tray import LocalDaemon


def _local_settings(tmp_path: Path, watch_dir: Path, **overrides) -> Settings:
    kwargs = dict(
        profile="local",
        data_dir=tmp_path,
        raw_docs_dir=tmp_path / "raw_docs",
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
        benchmark_path=tmp_path / "benchmarks" / "golden_qa.jsonl",
        reports_dir=tmp_path / "reports",
        vector_store_backend="numpy",
        warm_models=False,
        log_level="WARNING",
        min_chunk_chars=10,
        watched_paths=[watch_dir],
        api_port=0,  # let the OS pick a free port; nothing in these tests hits it over HTTP
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _fake_local_context(settings: Settings) -> AppContext:
    """A profile=local AppContext built from lightweight, dependency-free fakes
    (HashingEmbedder, NumpyVectorStore) instead of build_context()'s real
    sentence-transformers/torch-backed embedder — LocalDaemon accepts an injected
    ctx precisely so tests don't need those heavy deps installed."""
    embedder = HashingEmbedder(dim=64)
    store = NumpyVectorStore()
    sparse = FTS5SparseIndex(settings.index_dir / "fts5_index.sqlite3")
    dense = DenseRetriever(embedder=embedder, store=store)
    indexer = Indexer(settings=settings, embedder=embedder, vector_store=store, sparse_index=sparse)
    reranker = CrossEncoderReranker(settings)
    generator = GroundedGenerator(settings)
    verifier = CitationVerifier(settings)
    rag = RAGService(settings=settings, dense=dense, sparse=sparse, reranker=reranker,
                      generator=generator, verifier=verifier)
    return AppContext(settings=settings, embedder=embedder, store=store, sparse=sparse,
                       dense=dense, indexer=indexer, reranker=reranker, generator=generator,
                       verifier=verifier, rag=rag)


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_reindex_now_before_start_is_a_safe_noop(tmp_path: Path):
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    settings = _local_settings(tmp_path, watch_dir)
    daemon = LocalDaemon(settings, ctx=_fake_local_context(settings))
    daemon.reindex_now()  # must not raise even though start() was never called


def test_url_reflects_configured_host_and_port(tmp_path: Path):
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    settings = _local_settings(tmp_path, watch_dir, api_host="127.0.0.1", api_port=8787)
    daemon = LocalDaemon(settings, ctx=_fake_local_context(settings))
    assert daemon.url == "http://127.0.0.1:8787"


def test_start_crawls_watch_dirs_and_reindex_now_picks_up_new_files(tmp_path: Path):
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    (watch_dir / "one.md").write_text("# One\n\nFirst document body text.", encoding="utf-8")

    # A long debounce means the live watcher itself won't have fired during this
    # test's lifetime, so any pickup of a post-start file has to come from the
    # explicit reindex_now() call, not the live watcher noticing on its own.
    settings = _local_settings(tmp_path, watch_dir, local_debounce_seconds=120.0)
    daemon = LocalDaemon(settings, ctx=_fake_local_context(settings))
    try:
        daemon.start()
        assert _wait_until(lambda: daemon.ctx.store.count() > 0), "initial crawl never indexed one.md"

        (watch_dir / "two.md").write_text("# Two\n\nSecond document body text.", encoding="utf-8")
        count_before = daemon.ctx.store.count()
        daemon.reindex_now()
        assert _wait_until(lambda: daemon.ctx.store.count() > count_before), \
            "reindex_now() never picked up the new file"
    finally:
        daemon.stop()
