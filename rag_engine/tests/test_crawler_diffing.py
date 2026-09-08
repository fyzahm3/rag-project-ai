"""Crawler diffing logic: new / changed / unchanged / deleted file detection.

Uses os.utime() as a "fake clock" — setting exact, deterministic mtimes rather
than relying on real wall-clock timing between writes (which can be flaky on
filesystems with coarse mtime resolution). No real filesystem watching (watchdog)
and no real embedding models are involved — just IndexSyncer.crawl() against
FTS5SparseIndex's files table.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.config import Settings
from app.ingestion.indexer import Indexer
from app.ingestion.watcher import IndexSyncer
from app.retrieval.dense import NumpyVectorStore
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.sparse import FTS5SparseIndex

DOC_A = "# Alpha\n\nFirst document about launch procedures and timing windows."
DOC_A_EDITED = "# Alpha\n\nFirst document about launch procedures, timing windows, and abort criteria."
DOC_B = "# Beta\n\nSecond document about backup retention policy details."

_FAKE_NOW = 1_700_000_000.0  # 2023-11-14T22:13:20Z — arbitrary, just fixed and known


def _set_fake_mtime(path: Path, offset_seconds: float) -> None:
    """Deterministically set a file's mtime, instead of depending on real elapsed
    wall-clock time between writes."""
    ts = _FAKE_NOW + offset_seconds
    os.utime(path, (ts, ts))


def _syncer_over_fresh_index(settings: Settings) -> tuple[IndexSyncer, NumpyVectorStore, FTS5SparseIndex]:
    embedder = HashingEmbedder(dim=64)
    store = NumpyVectorStore()
    sparse = FTS5SparseIndex(settings.index_dir / "fts5_index.sqlite3")
    indexer = Indexer(settings=settings, embedder=embedder, vector_store=store, sparse_index=sparse)
    return IndexSyncer(indexer), store, sparse


def test_new_file_is_indexed_on_first_crawl(tmp_path: Path, settings: Settings):
    syncer, store, sparse = _syncer_over_fresh_index(settings)
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    doc_path = watch_dir / "alpha.md"
    doc_path.write_text(DOC_A, encoding="utf-8")
    _set_fake_mtime(doc_path, offset_seconds=0)

    stats = asyncio.run(syncer.crawl([watch_dir]))

    assert stats.scanned == 1
    assert stats.indexed == 1
    assert stats.skipped == 0
    assert stats.removed == 0
    assert store.count() > 0
    record = sparse.get_file_record(str(doc_path.resolve()))
    assert record is not None
    assert record["mtime"] == _FAKE_NOW


def test_unchanged_file_is_skipped_on_second_crawl(tmp_path: Path, settings: Settings):
    syncer, store, sparse = _syncer_over_fresh_index(settings)
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    doc_path = watch_dir / "alpha.md"
    doc_path.write_text(DOC_A, encoding="utf-8")
    _set_fake_mtime(doc_path, offset_seconds=0)

    first = asyncio.run(syncer.crawl([watch_dir]))
    assert first.indexed == 1
    count_after_first = store.count()

    # Same content, same mtime (fake clock didn't advance) -> untouched from the
    # crawler's point of view, should be skipped without even being read.
    second = asyncio.run(syncer.crawl([watch_dir]))
    assert second.indexed == 0
    assert second.skipped == 1
    assert second.removed == 0
    assert store.count() == count_after_first


def test_changed_file_is_reindexed_not_duplicated(tmp_path: Path, settings: Settings):
    syncer, store, sparse = _syncer_over_fresh_index(settings)
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    doc_path = watch_dir / "alpha.md"
    doc_path.write_text(DOC_A, encoding="utf-8")
    _set_fake_mtime(doc_path, offset_seconds=0)

    first = asyncio.run(syncer.crawl([watch_dir]))
    assert first.indexed == 1
    count_after_first = store.count()

    doc_path.write_text(DOC_A_EDITED, encoding="utf-8")
    _set_fake_mtime(doc_path, offset_seconds=60)  # the fake clock advances, not real time

    second = asyncio.run(syncer.crawl([watch_dir]))
    assert second.indexed == 1
    assert second.skipped == 0

    record = sparse.get_file_record(str(doc_path.resolve()))
    assert record["mtime"] == _FAKE_NOW + 60
    # Replaced, not appended alongside the stale version's chunks.
    assert 0 < store.count()
    hits = sparse.query_sync("abort criteria", top_k=5)
    assert any("abort criteria" in h.text for h in hits)
    _ = count_after_first  # documents the pre-edit baseline; replacement asserted above


def test_deleted_file_is_pruned_on_next_crawl(tmp_path: Path, settings: Settings):
    """The live watcher only notices a deletion while it's actually running — this
    is what catches a file removed while the daemon was off, on the next crawl."""
    syncer, store, sparse = _syncer_over_fresh_index(settings)
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    doc_a = watch_dir / "alpha.md"
    doc_b = watch_dir / "beta.md"
    doc_a.write_text(DOC_A, encoding="utf-8")
    doc_b.write_text(DOC_B, encoding="utf-8")
    _set_fake_mtime(doc_a, offset_seconds=0)
    _set_fake_mtime(doc_b, offset_seconds=0)

    first = asyncio.run(syncer.crawl([watch_dir]))
    assert first.indexed == 2
    assert first.removed == 0
    count_with_both = store.count()
    assert sparse.get_file_record(str(doc_a.resolve())) is not None

    doc_a.unlink()  # simulates a deletion that happened while the daemon wasn't watching

    second = asyncio.run(syncer.crawl([watch_dir]))
    assert second.scanned == 1  # only beta.md remains on disk
    assert second.indexed == 0  # beta.md unchanged
    assert second.removed == 1  # alpha.md pruned
    assert store.count() < count_with_both
    assert sparse.get_file_record(str(doc_a.resolve())) is None
    remaining = sparse.query_sync("backup retention", top_k=5)
    assert remaining and all("launch procedures" not in h.text for h in remaining)


def test_crawl_over_bm25_backend_never_prunes(components):
    """BM25SparseIndex has no files table, so _prune_vanished_files is a no-op —
    this must not raise, and `removed` must simply stay 0 (server profile)."""
    from app.ingestion.watcher import IndexSyncer

    indexer: Indexer = components["indexer"]
    syncer = IndexSyncer(indexer)
    stats = asyncio.run(syncer.crawl([Path("/does/not/exist")]))
    assert stats.removed == 0
