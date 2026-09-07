"""Tests for GET /v1/find (fast, no-rerank/no-LLM search) and the local-only
POST /v1/open, plus the FTS5 filename/path boosting and schema-migration logic
that backs them."""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.indexer import Indexer
from app.retrieval.dense import DenseRetriever, NumpyVectorStore
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.find import find_files
from app.retrieval.sparse import FTS5SparseIndex

SAMPLE_A = "# Launch Guide\n\nThe launch window opens at 0900 UTC and closes at 1100 UTC."
SAMPLE_B = "# Backup Policy\n\nBackups run hourly and are retained for thirty days."


def _local_settings(tmp_path: Path) -> Settings:
    return Settings(
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
    )


def _local_components(tmp_path: Path):
    settings = _local_settings(tmp_path)
    embedder = HashingEmbedder(dim=64)
    store = NumpyVectorStore()
    sparse = FTS5SparseIndex(settings.index_dir / "fts5_index.sqlite3")
    dense = DenseRetriever(embedder=embedder, store=store)
    indexer = Indexer(settings=settings, embedder=embedder, vector_store=store, sparse_index=sparse)
    return settings, embedder, store, sparse, dense, indexer


def test_find_sync_ranks_filename_match_above_content_only_match(tmp_path: Path):
    settings, embedder, store, sparse, dense, indexer = _local_components(tmp_path)
    asyncio.run(indexer.ingest("/docs/launch-guide.md", SAMPLE_A.encode("utf-8")))
    asyncio.run(indexer.ingest(
        "/docs/other.md",
        b"launch launch launch launch launch mentioned repeatedly in body text here",
    ))

    hits = sparse.find_sync("launch", top_k=10)
    assert hits
    # filename "launch-guide.md" should outrank a body that merely repeats "launch".
    assert hits[0].document_name == "/docs/launch-guide.md"


def test_find_sync_prefix_matches_partial_filename(tmp_path: Path):
    settings, embedder, store, sparse, dense, indexer = _local_components(tmp_path)
    asyncio.run(indexer.ingest("/docs/readme.md", SAMPLE_A.encode("utf-8")))

    hits = sparse.find_sync("read", top_k=10)
    assert any(h.document_name == "/docs/readme.md" for h in hits)


def test_find_sync_boosts_path_segment_match(tmp_path: Path):
    settings, embedder, store, sparse, dense, indexer = _local_components(tmp_path)
    asyncio.run(indexer.ingest("/docs/project-x/config.md", SAMPLE_A.encode("utf-8")))
    asyncio.run(indexer.ingest("/docs/other/config.md", SAMPLE_B.encode("utf-8")))

    hits = sparse.find_sync("project", top_k=10)
    assert hits
    assert hits[0].document_name == "/docs/project-x/config.md"


def test_query_sync_is_unaffected_by_filename_column(tmp_path: Path):
    """/v1/ask's sparse retrieval (query_sync) must stay content-only even though
    the same table now also indexes filename/path_tokens for /v1/find."""
    settings, embedder, store, sparse, dense, indexer = _local_components(tmp_path)
    asyncio.run(indexer.ingest("/docs/launch.md", b"nothing relevant to the search term in body"))
    asyncio.run(indexer.ingest("/docs/other.md", SAMPLE_A.encode("utf-8")))  # body mentions "launch"

    hits = sparse.query_sync("launch", top_k=10)
    paths = {h.document_name for h in hits}
    assert paths == {"/docs/other.md"}  # not /docs/launch.md, whose filename matches but body doesn't


def test_bm25_find_falls_back_to_content_query(components):
    """Server profile (BM25SparseIndex) has no filename column — find() should
    still work as a plain content search rather than erroring."""
    indexer: Indexer = components["indexer"]
    sparse = components["sparse"]
    # rank_bm25's classic BM25Okapi idf is log((N - n + 0.5) / (n + 0.5)), which is
    # <= 0 (and gets clamped to "no results" by query_sync) once a term appears in
    # half or more of the corpus — so this needs 3+ docs with the query terms
    # confined to just one of them, not the usual 2-doc test corpus.
    asyncio.run(indexer.ingest("/docs/a.md", SAMPLE_A.encode("utf-8")))
    asyncio.run(indexer.ingest("/docs/b.md", SAMPLE_B.encode("utf-8")))
    asyncio.run(indexer.ingest("/docs/c.md", b"# Unrelated\n\nCompletely different subject matter entirely."))

    hits = asyncio.run(sparse.find("launch window", top_k=5))
    assert hits


def test_find_files_dedupes_by_path_and_builds_snippets(tmp_path: Path):
    settings, embedder, store, sparse, dense, indexer = _local_components(tmp_path)
    asyncio.run(indexer.ingest("/docs/launch.md", SAMPLE_A.encode("utf-8")))
    asyncio.run(indexer.ingest("/docs/backup.md", SAMPLE_B.encode("utf-8")))

    hits = asyncio.run(find_files(dense, sparse, settings, "launch window"))
    assert hits
    paths = [h.path for h in hits]
    assert len(paths) == len(set(paths))  # no duplicate paths even if multiple chunks matched
    assert hits[0].path == "/docs/launch.md"
    assert hits[0].snippet
    # mtime falls back to the files-table record since /docs/launch.md isn't a real path
    assert hits[0].mtime is None or isinstance(hits[0].mtime, float)


def test_fts5_schema_migration_rebuilds_old_database(tmp_path: Path):
    db_path = tmp_path / "old.sqlite3"
    old_schema = """
    CREATE TABLE IF NOT EXISTS files (
        path TEXT PRIMARY KEY, mtime REAL, size INTEGER, content_hash TEXT, indexed_at REAL
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_id UNINDEXED, doc_path UNINDEXED, section_heading UNINDEXED,
        page UNINDEXED, strategy UNINDEXED, content
    );
    """
    conn = sqlite3.connect(str(db_path))
    conn.executescript(old_schema)
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, doc_path, section_heading, page, strategy, content) "
        "VALUES ('c1', '/docs/a.md', 'Intro', NULL, 'structure', 'stale content')"
    )
    conn.commit()
    conn.close()

    index = FTS5SparseIndex(db_path)
    # Migration wipes the old (schema-incompatible) rows; a re-crawl is expected to
    # repopulate — the important thing is it didn't crash and the new columns exist.
    assert index.count() == 0
    index._conn.execute("SELECT filename, path_tokens FROM chunks_fts LIMIT 1")


def _build_app(ctx):
    from app.main import create_app

    return create_app(ctx)


def _local_app_context(tmp_path: Path):
    from app.generation.generator import GroundedGenerator
    from app.generation.verifier import CitationVerifier
    from app.main import AppContext
    from app.pipeline import RAGService
    from app.retrieval.reranker import CrossEncoderReranker

    settings, embedder, store, sparse, dense, indexer = _local_components(tmp_path)
    reranker = CrossEncoderReranker(settings)
    generator = GroundedGenerator(settings)
    verifier = CitationVerifier(settings)
    rag = RAGService(settings=settings, dense=dense, sparse=sparse, reranker=reranker,
                      generator=generator, verifier=verifier)
    return AppContext(settings=settings, embedder=embedder, store=store, sparse=sparse,
                       dense=dense, indexer=indexer, reranker=reranker, generator=generator,
                       verifier=verifier, rag=rag)


def test_v1_find_endpoint_via_server_profile_client(components):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from app.main import AppContext, create_app

    indexer: Indexer = components["indexer"]
    asyncio.run(indexer.ingest("/docs/launch.md", SAMPLE_A.encode("utf-8")))

    ctx = AppContext(
        settings=components["settings"], embedder=components["embedder"], store=components["store"],
        sparse=components["sparse"], dense=components["dense"], indexer=indexer,
        reranker=components["reranker"], generator=components["generator"],
        verifier=components["verifier"], rag=components["rag"],
    )
    app = create_app(ctx)
    with fastapi_testclient.TestClient(app) as client:
        resp = client.get("/v1/find", params={"q": "launch window"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "launch window"
        assert body["results"]
        assert "latency_ms" in body

        # /v1/open doesn't exist under the server profile.
        open_resp = client.post("/v1/open", json={"path": "/docs/launch.md"})
        assert open_resp.status_code == 404


def test_v1_open_requires_loopback_and_indexed_path(tmp_path: Path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    ctx = _local_app_context(tmp_path)
    asyncio.run(ctx.indexer.ingest("/docs/launch.md", SAMPLE_A.encode("utf-8")))
    ctx.sparse.upsert_file_record("/docs/launch.md", mtime=1.0, size=10, content_hash="x")

    app = _build_app(ctx)

    # Non-loopback caller: rejected regardless of path.
    with fastapi_testclient.TestClient(app, client=("203.0.113.5", 12345)) as remote_client:
        resp = remote_client.post("/v1/open", json={"path": "/docs/launch.md"})
        assert resp.status_code == 403

    # Loopback caller, but a path that was never indexed: rejected.
    with fastapi_testclient.TestClient(app, client=("127.0.0.1", 12345)) as local_client:
        resp = local_client.post("/v1/open", json={"path": "/not/indexed.md"})
        assert resp.status_code == 404

    # Loopback caller, indexed path: subprocess call is mocked so this doesn't
    # actually try to open a fake file on the test machine.
    with fastapi_testclient.TestClient(app, client=("127.0.0.1", 12345)) as local_client:
        import unittest.mock

        with unittest.mock.patch("subprocess.run") as mock_run, \
             unittest.mock.patch("platform.system", return_value="Darwin"):
            resp = local_client.post("/v1/open", json={"path": "/docs/launch.md"})
            assert resp.status_code == 200
            assert resp.json()["opened"] == "/docs/launch.md"
            mock_run.assert_called_once_with(["open", "/docs/launch.md"], check=False)


def test_root_serves_local_static_page_only_under_local_profile(tmp_path: Path, components):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from app.main import AppContext, create_app

    server_ctx = AppContext(
        settings=components["settings"], embedder=components["embedder"], store=components["store"],
        sparse=components["sparse"], dense=components["dense"], indexer=components["indexer"],
        reranker=components["reranker"], generator=components["generator"],
        verifier=components["verifier"], rag=components["rag"],
    )
    with fastapi_testclient.TestClient(create_app(server_ctx)) as client:
        resp = client.get("/")
        assert resp.headers["content-type"].startswith("application/json")

    local_ctx = _local_app_context(tmp_path)
    with fastapi_testclient.TestClient(_build_app(local_ctx)) as client:
        resp = client.get("/")
        assert resp.headers["content-type"].startswith("text/html")
        assert "Search your files" in resp.text
