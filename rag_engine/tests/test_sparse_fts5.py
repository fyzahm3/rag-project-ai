"""Direct tests for FTS5SparseIndex: the local-profile sparse backend."""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.retrieval.dense import RetrievedRecord
from app.retrieval.sparse import FTS5SparseIndex, get_sparse_index


def _record(chunk_id: str, doc_path: str, text: str) -> RetrievedRecord:
    return RetrievedRecord(
        chunk_id=chunk_id,
        text=text,
        document_name=doc_path,
        section_heading="Intro",
        page=None,
        strategy="structure",
    )


def test_add_query_count(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    records = [
        _record("c1", "/docs/a.md", "The launch window opens at 0900 UTC."),
        _record("c2", "/docs/b.md", "Backups run hourly and are retained for thirty days."),
    ]
    added = index.add_documents(records)
    assert added == 2
    assert index.count() == 2

    # Re-adding the same chunk ids is a no-op (dedup by chunk_id).
    assert index.add_documents(records) == 0
    assert index.count() == 2

    results = index.query_sync("launch window", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert 0.0 < results[0].score <= 1.0


def test_query_ranks_by_bm25_and_normalizes_scores(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    index.add_documents(
        [
            _record("c1", "/docs/a.md", "storage storage storage volumes provision quickly"),
            _record("c2", "/docs/b.md", "storage volumes provision"),
            _record("c3", "/docs/c.md", "completely unrelated content about weather"),
        ]
    )
    results = index.query_sync("storage volumes", top_k=10)
    ids = [r.chunk_id for r in results]
    assert "c3" not in ids
    assert set(ids) == {"c1", "c2"}
    assert results[0].score == 1.0  # top hit is min-max normalized to 1.0


def test_query_empty_index_and_empty_query(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    assert index.query_sync("anything", top_k=5) == []
    index.add_documents([_record("c1", "/docs/a.md", "some content")])
    assert index.query_sync("", top_k=5) == []
    assert index.query_sync("   ", top_k=5) == []


def test_async_query_matches_sync(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    index.add_documents([_record("c1", "/docs/a.md", "the launch window opens at 0900 UTC")])
    results = asyncio.run(index.query("launch window", top_k=5))
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_remove_by_document(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    index.add_documents(
        [
            _record("c1", "/docs/a.md", "alpha content"),
            _record("c2", "/docs/a.md", "alpha more content"),
            _record("c3", "/docs/b.md", "beta content"),
        ]
    )
    assert index.count() == 3
    removed = index.remove_by_document("/docs/a.md")
    assert removed == 2
    assert index.count() == 1
    assert index.remove_by_document("/docs/a.md") == 0


def test_remove_by_path_prefix(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    index.add_documents(
        [
            _record("c1", "/docs/project_a/one.md", "alpha content"),
            _record("c2", "/docs/project_a/two.md", "alpha more content"),
            _record("c3", "/docs/project_b/one.md", "beta content"),
        ]
    )
    removed = index.remove_by_path_prefix("/docs/project_a/")
    assert removed == 2
    assert index.count() == 1
    remaining = index.query_sync("beta", top_k=5)
    assert remaining[0].document_name == "/docs/project_b/one.md"


def test_incremental_update_replaces_content_not_appends(tmp_path: Path):
    """The FTS5 index itself has no 'UPDATE a chunk's text' operation — the
    incremental-update pattern used throughout this codebase (IndexSyncer included)
    is remove-then-reinsert for the same document, and search must reflect only the
    new content afterward, never both."""
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    index.add_documents([_record("c1", "/docs/a.md", "the original launch time is 0900")])
    assert index.query_sync("0900", top_k=5)

    index.remove_by_document("/docs/a.md")
    index.add_documents([_record("c1-v2", "/docs/a.md", "the revised launch time is 1400")])

    # "0900" only ever existed in the removed version — an OR-token match on it
    # must find nothing now, not a low-scored leftover.
    stale = index.query_sync("0900", top_k=5)
    assert stale == []
    fresh = index.query_sync("1400", top_k=5)
    assert len(fresh) == 1
    assert fresh[0].chunk_id == "c1-v2"


def test_list_file_paths_under_prefix(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    index.upsert_file_record("/docs/project_a/one.md", mtime=1.0, size=1, content_hash="h1")
    index.upsert_file_record("/docs/project_a/two.md", mtime=1.0, size=1, content_hash="h2")
    index.upsert_file_record("/docs/project_b/one.md", mtime=1.0, size=1, content_hash="h3")

    under_a = index.list_file_paths_under("/docs/project_a")
    assert set(under_a) == {"/docs/project_a/one.md", "/docs/project_a/two.md"}

    under_everything = index.list_file_paths_under("/docs")
    assert len(under_everything) == 3

    assert index.list_file_paths_under("/docs/nonexistent") == []


def test_file_record_upsert_and_get(tmp_path: Path):
    index = FTS5SparseIndex(tmp_path / "fts5.sqlite3")
    assert index.get_file_record("/docs/a.md") is None

    index.upsert_file_record("/docs/a.md", mtime=100.0, size=42, content_hash="abc123")
    record = index.get_file_record("/docs/a.md")
    assert record is not None
    assert record["content_hash"] == "abc123"
    assert record["size"] == 42

    index.upsert_file_record("/docs/a.md", mtime=200.0, size=99, content_hash="def456")
    updated = index.get_file_record("/docs/a.md")
    assert updated["content_hash"] == "def456"
    assert updated["size"] == 99


def test_get_sparse_index_factory_selects_backend_by_profile(tmp_path: Path):
    from app.config import Settings
    from app.retrieval.sparse import BM25SparseIndex

    server_settings = Settings(data_dir=tmp_path / "server")
    assert isinstance(get_sparse_index(server_settings), BM25SparseIndex)

    local_settings = Settings(
        profile="local",
        data_dir=tmp_path / "local",
        raw_docs_dir=tmp_path / "local" / "raw_docs",
        chroma_dir=tmp_path / "local" / "chroma",
        index_dir=tmp_path / "local" / "index",
        benchmark_path=tmp_path / "local" / "benchmarks" / "golden_qa.jsonl",
        reports_dir=tmp_path / "local" / "reports",
    )
    assert isinstance(get_sparse_index(local_settings), FTS5SparseIndex)
