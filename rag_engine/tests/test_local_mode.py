"""Offline tests for local-mode indexing: delete-by-document plumbing and IndexSyncer,
independent of watchdog/pystray/tkinter (none of which are required to run these)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.ingestion.indexer import Indexer
from app.local.sync import IndexSyncer

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


def test_config_parses_comma_separated_watch_dirs(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path,
        deployment_mode="local",
        local_watch_dirs="/tmp/a, /tmp/b",
    )
    assert settings.local_watch_dirs == [Path("/tmp/a"), Path("/tmp/b")]
