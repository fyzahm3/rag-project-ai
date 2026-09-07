"""Bridges filesystem paths to Indexer.ingest/remove_document for local-mode indexing.

Kept independent of any watching mechanism (watchdog, polling, ...) so it can be
exercised directly in tests without a real filesystem watcher running.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.indexer import Indexer
from app.ingestion.parser import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class IndexSyncer:
    def __init__(self, indexer: Indexer, strategy: str | None = None) -> None:
        self.indexer = indexer
        self.strategy = strategy

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    async def index_path(self, path: Path) -> bool:
        """(Re-)index a single file. Safe to call repeatedly (e.g. on every save)."""
        if not self.supports(path) or not path.is_file():
            return False
        try:
            content = path.read_bytes()
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return False
        if not content:
            return False

        document_name = str(path.resolve())
        self.indexer.remove_document(document_name)
        try:
            response = await self.indexer.ingest(document_name, content, self.strategy)
        except Exception as exc:
            logger.warning("Failed to index %s: %s", path, exc)
            return False
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
            if path.is_file() and self.supports(path):
                if await self.index_path(path):
                    count += 1
        return count
