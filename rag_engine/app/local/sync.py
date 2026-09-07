"""Bridges filesystem paths to Indexer.ingest/remove_document for local-mode indexing.

Kept independent of any watching mechanism (watchdog, polling, ...) so it can be
exercised directly in tests without a real filesystem watcher running.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app.ingestion.indexer import Indexer
from app.ingestion.parser import SUPPORTED_EXTENSIONS
from app.local.filters import exceeds_size_limit, is_excluded

logger = logging.getLogger(__name__)


class IndexSyncer:
    def __init__(
        self,
        indexer: Indexer,
        strategy: str | None = None,
        excluded_patterns: list[str] | None = None,
        max_index_file_size_mb: int = 25,
    ) -> None:
        self.indexer = indexer
        self.strategy = strategy
        self.excluded_patterns = excluded_patterns or []
        self.max_index_file_size_mb = max_index_file_size_mb

    def supports(self, path: Path) -> bool:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        if is_excluded(path, self.excluded_patterns):
            return False
        if path.is_file() and exceeds_size_limit(path, self.max_index_file_size_mb):
            logger.info("Skipping %s: exceeds %d MB limit", path, self.max_index_file_size_mb)
            return False
        return True

    async def index_path(self, path: Path) -> bool:
        """(Re-)index a single file. Safe to call repeatedly (e.g. on every save).

        Skips the work entirely if the file's content hash matches what's already
        indexed (tracked via the sparse index's `files` table, when available —
        FTS5SparseIndex in local mode; a no-op for BM25SparseIndex, which doesn't
        track per-file metadata).
        """
        if not self.supports(path) or not path.is_file():
            return False
        try:
            content = path.read_bytes()
            stat = path.stat()
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return False
        if not content:
            return False

        document_name = str(path.resolve())
        content_hash = hashlib.sha1(content).hexdigest()
        sparse = self.indexer.sparse_index
        if hasattr(sparse, "get_file_record"):
            existing = sparse.get_file_record(document_name)
            if existing is not None and existing.get("content_hash") == content_hash:
                return False

        self.indexer.remove_document(document_name)
        try:
            response = await self.indexer.ingest(document_name, content, self.strategy)
        except Exception as exc:
            logger.warning("Failed to index %s: %s", path, exc)
            return False

        if hasattr(sparse, "upsert_file_record"):
            sparse.upsert_file_record(
                document_name, mtime=stat.st_mtime, size=stat.st_size, content_hash=content_hash
            )
        logger.info("Indexed %s -> %d chunks", path.name, response.chunks_created)
        return True

    def remove_path(self, path: Path) -> bool:
        document_name = str(path.resolve())
        removed = self.indexer.remove_document(document_name)
        if removed:
            logger.info("Removed %s from index", path.name)
        return bool(removed)

    async def scan_directory(self, directory: Path) -> int:
        if not directory.exists():
            logger.warning("Watch directory does not exist, skipping: %s", directory)
            return 0
        count = 0
        for path in directory.rglob("*"):
            if is_excluded(path, self.excluded_patterns):
                continue
            if path.is_file() and self.supports(path):
                if await self.index_path(path):
                    count += 1
        return count
