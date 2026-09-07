"""Offline tests for local-mode indexing: delete-by-document plumbing, the crawler
(IndexSyncer) and live watcher (FolderWatcher) debounce logic, independent of
watchdog/pystray/tkinter (none of which are required to run these)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.ingestion.filters import exceeds_size_limit, is_excluded
from app.ingestion.indexer import Indexer
from app.ingestion.watcher import FolderWatcher, IndexSyncer
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


def test_crawl_indexes_supported_files_and_reports_stats(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    store = components["store"]
    syncer = IndexSyncer(indexer)

    (tmp_path / "one.md").write_text(SAMPLE_A, encoding="utf-8")
    (tmp_path / "two.txt").write_text("Plain text notes about the launch window.", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    stats = asyncio.run(syncer.crawl([tmp_path]))
    assert stats.scanned == 3
    assert stats.indexed == 2
    assert stats.skipped == 1
    assert stats.errors == 0
    assert store.count() > 0


def test_crawl_second_pass_skips_unchanged_files(tmp_path: Path, settings: Settings):
    # Needs FTS5SparseIndex (local profile) for skip-if-unchanged to apply at all;
    # BM25SparseIndex has no per-file tracking, so it always re-indexes (covered by
    # test_crawl_indexes_supported_files_and_reports_stats instead). The watched
    # content also has to live outside settings.index_dir, or the crawl would walk
    # into the sparse index's own storage directory.
    embedder = HashingEmbedder(dim=128)
    store = NumpyVectorStore()
    sparse = FTS5SparseIndex(settings.index_dir / "fts5_index.sqlite3")
    indexer = Indexer(settings=settings, embedder=embedder, vector_store=store, sparse_index=sparse)
    syncer = IndexSyncer(indexer)

    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "one.md").write_text(SAMPLE_A, encoding="utf-8")

    first = asyncio.run(syncer.crawl([content_dir]))
    assert first.scanned == 1
    assert first.indexed == 1

    second = asyncio.run(syncer.crawl([content_dir]))
    assert second.scanned == 1
    assert second.indexed == 0
    assert second.skipped == 1


def test_crawl_missing_root_is_reported_not_raised(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    syncer = IndexSyncer(indexer)
    missing = tmp_path / "does-not-exist"

    stats = asyncio.run(syncer.crawl([missing]))
    assert stats.scanned == 0


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


def test_index_syncer_skips_reindex_when_stat_unchanged(tmp_path: Path, settings: Settings):
    """With FTS5SparseIndex's file-tracking, an unchanged file (same mtime/size) is
    skipped on the next pass without even being read."""
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

    # Changed content (different size) -> re-indexed.
    file_path.write_text(SAMPLE_B, encoding="utf-8")
    assert asyncio.run(syncer.index_path(file_path)) is True


def test_local_profile_chunk_ids_are_path_and_index_based(tmp_path: Path, settings: Settings):
    from app.ingestion.chunker import StructureAwareChunker
    from app.ingestion.parser import parse_document
    from app.utils.text import path_doc_id

    local_settings = settings.model_copy(update={"profile": "local"})
    doc = parse_document("notes.md", SAMPLE_A.encode("utf-8"))
    chunks = StructureAwareChunker(local_settings).chunk(doc)
    assert chunks
    expected_prefix = path_doc_id("notes.md")
    for chunk in chunks:
        assert chunk.chunk_id == f"{expected_prefix}:{chunk.seq}"


def test_server_profile_chunk_ids_are_content_based(settings: Settings):
    from app.ingestion.chunker import StructureAwareChunker
    from app.ingestion.parser import parse_document

    doc = parse_document("notes.md", SAMPLE_A.encode("utf-8"))
    chunks = StructureAwareChunker(settings).chunk(doc)
    assert chunks
    for chunk in chunks:
        assert ":" not in chunk.chunk_id  # sha1_id output, not the path:index scheme


def test_folder_watcher_debounces_rapid_events_to_one_pending_timer(tmp_path: Path, components):
    """A burst of on_modified events for the same path should cancel the previous
    pending timer and leave exactly one scheduled, not one per event."""

    async def scenario():
        indexer: Indexer = components["indexer"]
        syncer = IndexSyncer(indexer)
        watcher = FolderWatcher(syncer, [tmp_path], debounce_seconds=10.0)
        watcher._loop = asyncio.get_running_loop()

        path = tmp_path / "watched.md"
        watcher._reschedule(path, removed=False)
        first_handle = watcher._pending[path]
        watcher._reschedule(path, removed=False)
        second_handle = watcher._pending[path]

        assert first_handle.cancelled()
        assert second_handle is not first_handle
        assert not second_handle.cancelled()
        assert len(watcher._pending) == 1

        second_handle.cancel()

    asyncio.run(scenario())
