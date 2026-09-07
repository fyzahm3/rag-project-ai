"""Offline tests for local-mode indexing: delete-by-document plumbing and IndexSyncer,
independent of watchdog/pystray/tkinter (none of which are required to run these)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.ingestion.indexer import Indexer
from app.local.filters import exceeds_size_limit, is_excluded
from app.local.sync import IndexSyncer
from app.retrieval.dense import NumpyVectorStore
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.sparse import FTS5SparseIndex

SAMPLE_A = "# Doc A\n\nThe launch window opens at 0900 UTC and closes at 1100 UTC."
SAMPLE_B = "# Doc B\n\nThe launch window opens at 0900 UTC and closes at 1100 UTC, revised."


def test_vector_and_sparse_stores_delete_by_document(components):
    indexer: Indexer = components["indexer"]
    store = components["store"]
    sparse = components["sparse"]

    asyncio.run(indexer.ingest("/tmp/docs/a.md", SAMPLE_A.encode("utf-8")))
    assert store.count() > 0
    assert sparse.count() > 0

    removed = indexer.remove_document("/tmp/docs/a.md")
    assert removed > 0
    assert store.count() == 0
    assert sparse.count() == 0


def test_remove_document_only_affects_matching_document(components):
    indexer: Indexer = components["indexer"]
    store = components["store"]

    asyncio.run(indexer.ingest("/tmp/docs/a.md", SAMPLE_A.encode("utf-8")))
    asyncio.run(indexer.ingest("/tmp/docs/b.md", SAMPLE_B.encode("utf-8")))
    total_before = store.count()
    assert total_before > 0

    indexer.remove_document("/tmp/docs/a.md")
    assert 0 < store.count() < total_before


def test_index_syncer_reindexes_changed_file_without_duplication(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    store = components["store"]
    syncer = IndexSyncer(indexer)

    file_path = tmp_path / "watched.md"
    file_path.write_text(SAMPLE_A, encoding="utf-8")

    assert asyncio.run(syncer.index_path(file_path)) is True
    first_count = store.count()
    assert first_count > 0

    # Simulate an editor save with modified content: re-indexing must replace, not
    # accumulate, chunks for this document.
    file_path.write_text(SAMPLE_B, encoding="utf-8")
    assert asyncio.run(syncer.index_path(file_path)) is True
    second_count = store.count()
    assert second_count > 0
    assert second_count <= first_count + 1  # replaced, not appended alongside stale chunks


def test_index_syncer_remove_path(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    store = components["store"]
    syncer = IndexSyncer(indexer)

    file_path = tmp_path / "watched.md"
    file_path.write_text(SAMPLE_A, encoding="utf-8")
    asyncio.run(syncer.index_path(file_path))
    assert store.count() > 0

    assert syncer.remove_path(file_path) is True
    assert store.count() == 0


def test_index_syncer_skips_unsupported_extensions(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    syncer = IndexSyncer(indexer)

    file_path = tmp_path / "notes.exe"
    file_path.write_bytes(b"not a real document")
    assert asyncio.run(syncer.index_path(file_path)) is False


def test_index_syncer_scan_directory(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    store = components["store"]
    syncer = IndexSyncer(indexer)

    (tmp_path / "one.md").write_text(SAMPLE_A, encoding="utf-8")
    (tmp_path / "two.txt").write_text("Plain text notes about the launch window.", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    indexed = asyncio.run(syncer.scan_directory(tmp_path))
    assert indexed == 2
    assert store.count() > 0


def test_config_parses_comma_separated_watched_paths(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, watched_paths="/tmp/a, /tmp/b")
    assert settings.watched_paths == [Path("/tmp/a"), Path("/tmp/b")]


def _local_settings(tmp_path: Path, **overrides) -> Settings:
    """profile='local' Settings with every path field pinned inside tmp_path, so
    _apply_local_profile_defaults() never touches the real per-OS app data dir."""
    kwargs = dict(
        profile="local",
        data_dir=tmp_path,
        raw_docs_dir=tmp_path / "raw_docs",
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
        benchmark_path=tmp_path / "benchmarks" / "golden_qa.jsonl",
        reports_dir=tmp_path / "reports",
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def test_local_profile_defaults_embedding_and_llm_provider(tmp_path: Path):
    settings = _local_settings(tmp_path)
    assert settings.embedding_provider == "huggingface"
    assert settings.hf_embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.llm_provider == "ollama"
    assert settings.api_host == "127.0.0.1"


def test_local_profile_prefers_openai_when_key_present(tmp_path: Path):
    settings = _local_settings(tmp_path, openai_api_key="sk-test")
    assert settings.llm_provider == "openai"


def test_local_profile_does_not_override_explicit_choices(tmp_path: Path):
    settings = _local_settings(
        tmp_path, embedding_provider="openai", llm_provider="openai", api_host="0.0.0.0"
    )
    assert settings.embedding_provider == "openai"
    assert settings.llm_provider == "openai"
    assert settings.api_host == "0.0.0.0"


def test_server_profile_is_unaffected_by_local_defaults(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    assert settings.profile == "server"
    assert settings.embedding_provider == "huggingface"  # unchanged built-in default
    assert settings.hf_embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.llm_provider == "openai"


def test_is_excluded_matches_directory_name_and_glob_patterns():
    patterns = ["node_modules", ".git", "*.zip"]
    assert is_excluded(Path("/repo/node_modules/pkg/index.js"), patterns) is True
    assert is_excluded(Path("/repo/.git/HEAD"), patterns) is True
    assert is_excluded(Path("/repo/archive.zip"), patterns) is True
    assert is_excluded(Path("/repo/src/notes.md"), patterns) is False


def test_exceeds_size_limit(tmp_path: Path):
    small = tmp_path / "small.txt"
    small.write_bytes(b"x" * 100)
    assert exceeds_size_limit(small, max_mb=1) is False

    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (2 * 1024 * 1024))
    assert exceeds_size_limit(big, max_mb=1) is True


def test_index_syncer_respects_excluded_patterns(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    syncer = IndexSyncer(indexer, excluded_patterns=["node_modules"])

    excluded_dir = tmp_path / "node_modules"
    excluded_dir.mkdir()
    excluded_file = excluded_dir / "readme.md"
    excluded_file.write_text(SAMPLE_A, encoding="utf-8")

    assert asyncio.run(syncer.index_path(excluded_file)) is False


def test_index_syncer_respects_size_limit(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    syncer = IndexSyncer(indexer, max_index_file_size_mb=1)

    big_file = tmp_path / "huge.txt"
    big_file.write_bytes(b"x" * (2 * 1024 * 1024))

    assert asyncio.run(syncer.index_path(big_file)) is False


def test_index_syncer_skips_reindex_when_content_unchanged(tmp_path: Path, settings: Settings):
    """With FTS5SparseIndex's file-tracking, an unchanged file (e.g. a mtime-only
    touch) should not be re-embedded on the next pass."""
    embedder = HashingEmbedder(dim=128)
    store = NumpyVectorStore()
    sparse = FTS5SparseIndex(settings.index_dir / "fts5_index.sqlite3")
    indexer = Indexer(settings=settings, embedder=embedder, vector_store=store, sparse_index=sparse)
    syncer = IndexSyncer(indexer)

    file_path = tmp_path / "watched.md"
    file_path.write_text(SAMPLE_A, encoding="utf-8")

    assert asyncio.run(syncer.index_path(file_path)) is True
    first_count = store.count()
    assert first_count > 0

    # Unchanged content -> skipped (no remove+reingest cycle).
    assert asyncio.run(syncer.index_path(file_path)) is False
    assert store.count() == first_count

    # Changed content -> re-indexed.
    file_path.write_text(SAMPLE_B, encoding="utf-8")
    assert asyncio.run(syncer.index_path(file_path)) is True
