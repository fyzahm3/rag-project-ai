"""Crawler + live watcher for local-mode indexing.

Two collaborating pieces, kept in one module since they share the same
skip/index/remove logic and are always used together (see app/local/daemon.py):

  - IndexSyncer: the initial crawl (stat-based skip-if-unchanged, so an unchanged
    file is never even read) and the per-file index/remove primitives, both bounded
    to a configurable number of concurrent embed calls so a large first-run crawl
    doesn't peg the CPU.
  - FolderWatcher: watchdog-based live watching (FSEvents on macOS,
    ReadDirectoryChangesW on Windows) with *true* per-path debounce — a rapid burst
    of saves to the same file cancels and reschedules a single pending reindex
    rather than firing one reindex per event.

Deletes (crawl-detected or live) remove a file's chunks from both the vector store
and the sparse index via Indexer.remove_document, which fans out to whichever
backend is active (see app/retrieval/dense.py, app/retrieval/sparse.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.filters import exceeds_size_limit, is_excluded
from app.ingestion.indexer import Indexer
from app.ingestion.parser import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

_PROGRESS_LOG_EVERY = 100


@dataclass
class CrawlStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    errors: int = 0
    removed: int = 0


class IndexSyncer:
    def __init__(
        self,
        indexer: Indexer,
        strategy: str | None = None,
        excluded_patterns: list[str] | None = None,
        max_index_file_size_mb: int = 25,
        max_concurrent_indexing: int = 2,
    ) -> None:
        self.indexer = indexer
        self.strategy = strategy
        self.excluded_patterns = excluded_patterns or []
        self.max_index_file_size_mb = max_index_file_size_mb
        self.max_concurrent_indexing = max(1, max_concurrent_indexing)
        self._semaphore = asyncio.Semaphore(self.max_concurrent_indexing)

    def supports(self, path: Path) -> bool:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        if is_excluded(path, self.excluded_patterns):
            return False
        if path.is_file() and exceeds_size_limit(path, self.max_index_file_size_mb):
            logger.info("Skipping %s: exceeds %d MB limit", path, self.max_index_file_size_mb)
            return False
        return True

    def _unchanged(self, document_name: str, stat) -> bool:
        """Cheap pre-check against the sparse index's files table — (mtime, size)
        only, no file read — so an untouched file costs one stat() call, not a
        read+hash+embed. A no-op (always False) when the active sparse backend
        doesn't track file metadata (BM25SparseIndex, i.e. server profile)."""
        sparse = self.indexer.sparse_index
        get_file_record = getattr(sparse, "get_file_record", None)
        if get_file_record is None:
            return False
        record = get_file_record(document_name)
        if record is None:
            return False
        return record.get("mtime") == stat.st_mtime and record.get("size") == stat.st_size

    async def index_path(self, path: Path) -> bool:
        """(Re-)index a single file. Safe to call repeatedly (e.g. on every save).
        Returns True only if it actually (re-)indexed the file."""
        if not self.supports(path) or not path.is_file():
            return False
        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning("Could not stat %s: %s", path, exc)
            return False

        document_name = str(path.resolve())
        if self._unchanged(document_name, stat):
            return False

        async with self._semaphore:
            try:
                content = path.read_bytes()
            except OSError as exc:
                logger.warning("Could not read %s: %s", path, exc)
                return False
            if not content:
                return False

            self.indexer.remove_document(document_name)
            try:
                response = await self.indexer.ingest(document_name, content, self.strategy)
            except Exception as exc:
                logger.warning("Failed to index %s: %s", path, exc)
                return False

            upsert_file_record = getattr(self.indexer.sparse_index, "upsert_file_record", None)
            if upsert_file_record is not None:
                content_hash = hashlib.sha1(content).hexdigest()
                upsert_file_record(
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

    async def crawl(self, roots: list[Path]) -> CrawlStats:
        """Walk every root, indexing new/changed files, skipping unchanged ones, and
        pruning files that vanished since the last crawl (the live watcher only
        catches a deletion that happens while it's actually running — a fresh crawl
        is what notices one that happened while the daemon was off). Launches
        indexing concurrently (bounded by max_concurrent_indexing) as files are
        discovered, rather than walking-then-indexing in two separate phases, so
        progress is visible throughout a long first-run crawl rather than all at
        once at the end."""
        stats = CrawlStats()
        pending: set[asyncio.Task] = set()
        backpressure_limit = self.max_concurrent_indexing * 4
        seen_paths: set[str] = set()

        for root in roots:
            if not root.exists():
                logger.warning("Watch path does not exist, skipping: %s", root)
                continue
            for path in root.rglob("*"):
                if not path.is_file() or is_excluded(path, self.excluded_patterns):
                    continue
                stats.scanned += 1
                seen_paths.add(str(path.resolve()))
                pending.add(asyncio.create_task(self._crawl_one(path, stats)))
                if len(pending) >= backpressure_limit:
                    _done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if stats.scanned % _PROGRESS_LOG_EVERY == 0:
                    logger.info(
                        "Crawl progress: %d scanned, %d indexed, %d skipped, %d errors",
                        stats.scanned, stats.indexed, stats.skipped, stats.errors,
                    )
        if pending:
            await asyncio.wait(pending)

        stats.removed = self._prune_vanished_files(roots, seen_paths)

        logger.info(
            "Crawl complete: %d scanned, %d indexed, %d skipped, %d removed, %d errors",
            stats.scanned, stats.indexed, stats.skipped, stats.removed, stats.errors,
        )
        return stats

    def _prune_vanished_files(self, roots: list[Path], seen_paths: set[str]) -> int:
        """Compare what the sparse index already knows about under each root
        against what this crawl actually found, and drop entries for anything
        that's gone missing. A no-op for backends without file tracking
        (BM25SparseIndex, i.e. server profile)."""
        list_known = getattr(self.indexer.sparse_index, "list_file_paths_under", None)
        if list_known is None:
            return 0
        removed = 0
        for root in roots:
            for known_path in list_known(str(root.resolve())):
                if known_path not in seen_paths and not Path(known_path).exists():
                    if self.indexer.remove_document(known_path):
                        removed += 1
        return removed

    async def _crawl_one(self, path: Path, stats: CrawlStats) -> None:
        try:
            did_index = await self.index_path(path)
        except Exception as exc:
            stats.errors += 1
            logger.warning("Error indexing %s during crawl: %s", path, exc)
            return
        if did_index:
            stats.indexed += 1
        else:
            stats.skipped += 1


class FolderWatcher:
    def __init__(
        self,
        syncer: IndexSyncer,
        watch_dirs: list[Path],
        debounce_seconds: float = 2.0,
    ) -> None:
        self.syncer = syncer
        self.watch_dirs = watch_dirs
        self.debounce_seconds = debounce_seconds
        self._observer = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: dict[Path, asyncio.TimerHandle] = {}

    async def start(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as exc:
            raise RuntimeError(
                "Local mode requires 'watchdog'. Install it with: "
                "pip install -r requirements-local.txt"
            ) from exc

        self._loop = asyncio.get_running_loop()
        handler = self._build_handler(FileSystemEventHandler)
        self._observer = Observer()

        watched_any = False
        for directory in self.watch_dirs:
            if not directory.exists():
                logger.warning("watched_paths entry does not exist, skipping: %s", directory)
                continue
            self._observer.schedule(handler, str(directory), recursive=True)
            watched_any = True
        if watched_any:
            self._observer.start()
            logger.info("Watching %d folder(s) for changes", len(self.watch_dirs))

        await self.syncer.crawl(self.watch_dirs)

    def _build_handler(self, base_cls):
        watcher = self

        class Handler(base_cls):
            def on_created(self, event) -> None:
                if not event.is_directory:
                    watcher._debounced_schedule(Path(event.src_path), removed=False)

            def on_modified(self, event) -> None:
                if not event.is_directory:
                    watcher._debounced_schedule(Path(event.src_path), removed=False)

            def on_deleted(self, event) -> None:
                if not event.is_directory:
                    watcher._debounced_schedule(Path(event.src_path), removed=True)

            def on_moved(self, event) -> None:
                if not event.is_directory:
                    watcher._debounced_schedule(Path(event.src_path), removed=True)
                    watcher._debounced_schedule(Path(event.dest_path), removed=False)

        return Handler()

    def _debounced_schedule(self, path: Path, removed: bool) -> None:
        # Called from watchdog's own OS thread; hop onto the daemon's asyncio loop
        # before touching self._pending (not thread-safe otherwise).
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._reschedule, path, removed)

    def _reschedule(self, path: Path, removed: bool) -> None:
        # Runs on the event loop thread: cancel any still-pending timer for this
        # exact path and start a fresh one, so a burst of saves to one file
        # collapses into a single reindex after things settle down.
        existing = self._pending.get(path)
        if existing is not None:
            existing.cancel()
        self._pending[path] = self._loop.call_later(self.debounce_seconds, self._fire, path, removed)

    def _fire(self, path: Path, removed: bool) -> None:
        self._pending.pop(path, None)
        asyncio.ensure_future(self._handle(path, removed))

    async def _handle(self, path: Path, removed: bool) -> None:
        if removed:
            if not path.exists():
                self.syncer.remove_path(path)
        elif path.exists():
            await self.syncer.index_path(path)

    def stop(self) -> None:
        for handle in self._pending.values():
            handle.cancel()
        self._pending.clear()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
