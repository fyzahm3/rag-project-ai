"""Sparse retrieval, two interchangeable backends behind one interface:

  - BM25SparseIndex: rank-bm25 over an in-memory index persisted as JSONL. Used by
    the "server" profile and the evaluation harness (kept exactly as before).
  - FTS5SparseIndex: SQLite FTS5, used by the "local" profile. Local-mode files get
    edited/deleted on disk far more often than server-mode uploads do, and rebuilding
    a full BM25Okapi index on every keystroke-triggered save doesn't scale the same
    way FTS5's incremental insert/delete does.

Both return `list[RetrievedRecord]` from `query`/`query_sync`, so
app/retrieval/fusion.py and app/pipeline.py need no knowledge of which one is active
— get_sparse_index() below is the only place that decides.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.config import Settings
from app.retrieval.dense import RetrievedRecord
from app.utils.text import tokenize

logger = logging.getLogger(__name__)


class SparseIndexBase(Protocol):
    def add_documents(self, records: list[RetrievedRecord]) -> int:
        ...

    def remove_by_document(self, document_name: str) -> int:
        ...

    def query_sync(self, query: str, top_k: int) -> list[RetrievedRecord]:
        ...

    async def query(self, query: str, top_k: int) -> list[RetrievedRecord]:
        ...

    def count(self) -> int:
        ...


class BM25SparseIndex:
    """BM25Okapi index persisted as a JSONL sidecar next to the vector store."""

    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._entries: list[dict[str, Any]] = []
        self._bm25 = None
        self._load()

    def _load(self) -> None:
        if not self.persist_path.exists():
            logger.info("Sparse corpus %s not found; starting empty", self.persist_path)
            return
        entries: list[dict[str, Any]] = []
        with self.persist_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        self._entries = entries
        self._rebuild()
        logger.info("Loaded BM25 corpus with %d documents from %s", len(entries), self.persist_path)

    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi

        if not self._entries:
            self._bm25 = None
            return
        tokenized = [tokenize(entry["text"]) for entry in self._entries]
        self._bm25 = BM25Okapi(tokenized)

    def _persist(self) -> None:
        tmp_path = self.persist_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp_path.replace(self.persist_path)

    def add_documents(self, records: list[RetrievedRecord]) -> int:
        if not records:
            return 0
        with self._lock:
            existing = {entry["chunk_id"] for entry in self._entries}
            added = 0
            for record in records:
                if record.chunk_id in existing:
                    continue
                self._entries.append(
                    {
                        "chunk_id": record.chunk_id,
                        "text": record.text,
                        "metadata": {
                            "document_name": record.document_name,
                            "section_heading": record.section_heading,
                            "page": record.page,
                            "strategy": record.strategy,
                        },
                    }
                )
                existing.add(record.chunk_id)
                added += 1
            self._rebuild()
            self._persist()
        return added

    def query_sync(self, query: str, top_k: int) -> list[RetrievedRecord]:
        with self._lock:
            if self._bm25 is None or not self._entries or top_k <= 0:
                return []
            scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
            if scores.size == 0 or float(scores.max()) <= 0.0:
                return []
            order = np.argsort(-scores)[:top_k]
            max_score = float(scores.max())
            results: list[RetrievedRecord] = []
            for index in order:
                score = float(scores[int(index)])
                if score <= 0.0:
                    continue
                entry = self._entries[int(index)]
                metadata = entry.get("metadata", {})
                results.append(
                    RetrievedRecord(
                        chunk_id=entry["chunk_id"],
                        text=entry["text"],
                        document_name=str(metadata.get("document_name", "unknown")),
                        section_heading=str(metadata.get("section_heading", "(untitled)")),
                        page=metadata.get("page"),
                        strategy=str(metadata.get("strategy", "structure")),
                        score=score / max_score,
                        raw_score=score,
                    )
                )
            return results

    async def query(self, query: str, top_k: int) -> list[RetrievedRecord]:
        import asyncio

        return await asyncio.to_thread(self.query_sync, query, top_k)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def remove_by_document(self, document_name: str) -> int:
        with self._lock:
            before = len(self._entries)
            self._entries = [
                e for e in self._entries
                if e.get("metadata", {}).get("document_name") != document_name
            ]
            removed = before - len(self._entries)
            if removed:
                self._rebuild()
                self._persist()
            return removed


_FTS5_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime REAL,
    size INTEGER,
    content_hash TEXT,
    indexed_at REAL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_path UNINDEXED,
    section_heading UNINDEXED,
    page UNINDEXED,
    strategy UNINDEXED,
    content
);
"""


class FTS5SparseIndex:
    """SQLite FTS5 sparse index (local profile). Same query/add/remove/count surface
    as BM25SparseIndex, plus a `files` table (path/mtime/size/content_hash/indexed_at)
    that local-mode file watching can use to skip re-indexing unchanged files, and a
    doc-path-prefix delete for dropping every chunk under a removed watched folder.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_FTS5_SCHEMA)
        self._conn.commit()

    # -- chunk index ---------------------------------------------------------

    def add_documents(self, records: list[RetrievedRecord]) -> int:
        if not records:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            added = 0
            for record in records:
                exists = cur.execute(
                    "SELECT 1 FROM chunks_fts WHERE chunk_id = ? LIMIT 1", (record.chunk_id,)
                ).fetchone()
                if exists:
                    continue
                cur.execute(
                    "INSERT INTO chunks_fts "
                    "(chunk_id, doc_path, section_heading, page, strategy, content) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.chunk_id,
                        record.document_name,
                        record.section_heading,
                        record.page,
                        record.strategy,
                        record.text,
                    ),
                )
                added += 1
            self._conn.commit()
            return added

    def query_sync(self, query: str, top_k: int) -> list[RetrievedRecord]:
        if top_k <= 0:
            return []
        fts_query = self._to_match_query(query)
        if not fts_query:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT chunk_id, doc_path, section_heading, page, strategy, content, "
                    "bm25(chunks_fts) AS rank "
                    "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, top_k),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 query failed for %r: %s", query, exc)
                return []
        if not rows:
            return []
        # SQLite's bm25() is *more negative = more relevant*; invert and min-max
        # normalize into (0, 1] to match RetrievedRecord.score's convention.
        inverted = [-row[6] for row in rows]
        max_score = max(inverted) if inverted else 0.0
        results: list[RetrievedRecord] = []
        for row, inv in zip(rows, inverted):
            chunk_id, doc_path, heading, page, strategy, content, _rank = row
            results.append(
                RetrievedRecord(
                    chunk_id=chunk_id,
                    text=content,
                    document_name=doc_path,
                    section_heading=heading or "(untitled)",
                    page=page,
                    strategy=strategy or "structure",
                    score=(inv / max_score) if max_score > 0 else 0.0,
                    raw_score=inv,
                )
            )
        return results

    async def query(self, query: str, top_k: int) -> list[RetrievedRecord]:
        import asyncio

        return await asyncio.to_thread(self.query_sync, query, top_k)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0])

    def remove_by_document(self, document_name: str) -> int:
        with self._lock:
            (removed,) = self._conn.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE doc_path = ?", (document_name,)
            ).fetchone()
            if removed == 0:
                return 0
            self._conn.execute("DELETE FROM chunks_fts WHERE doc_path = ?", (document_name,))
            self._conn.execute("DELETE FROM files WHERE path = ?", (document_name,))
            self._conn.commit()
            return int(removed)

    def remove_by_path_prefix(self, prefix: str) -> int:
        """Delete every chunk (and file record) whose doc_path starts with `prefix` —
        for dropping an entire folder that was removed from watched_paths."""
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._lock:
            (removed,) = self._conn.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE doc_path LIKE ? ESCAPE '\\'", (pattern,)
            ).fetchone()
            if removed == 0:
                return 0
            self._conn.execute("DELETE FROM chunks_fts WHERE doc_path LIKE ? ESCAPE '\\'", (pattern,))
            self._conn.execute("DELETE FROM files WHERE path LIKE ? ESCAPE '\\'", (pattern,))
            self._conn.commit()
            return int(removed)

    @staticmethod
    def _to_match_query(query: str) -> str:
        tokens = tokenize(query)
        if not tokens:
            return ""
        # OR (not FTS5's default AND) so partial term overlap still ranks via bm25(),
        # matching BM25Okapi's behavior rather than requiring every term present.
        return " OR ".join(f'"{token}"' for token in tokens)

    # -- file tracking (for skip-if-unchanged re-indexing) -------------------

    def upsert_file_record(self, path: str, mtime: float, size: int, content_hash: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO files (path, mtime, size, content_hash, indexed_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, "
                "content_hash=excluded.content_hash, indexed_at=excluded.indexed_at",
                (path, mtime, size, content_hash, time.time()),
            )
            self._conn.commit()

    def get_file_record(self, path: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT path, mtime, size, content_hash, indexed_at FROM files WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            return None
        return {"path": row[0], "mtime": row[1], "size": row[2], "content_hash": row[3], "indexed_at": row[4]}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def get_sparse_index(settings: Settings) -> SparseIndexBase:
    """Factory mirroring app.retrieval.dense.get_vector_store(): the one place that
    decides BM25 (server profile, eval harness) vs FTS5 (local profile)."""
    if settings.profile == "local":
        return FTS5SparseIndex(settings.index_dir / "fts5_index.sqlite3")
    return BM25SparseIndex(settings.index_dir / "bm25_corpus.jsonl")
