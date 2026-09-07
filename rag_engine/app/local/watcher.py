"""Cross-platform folder watcher (watchdog) that debounces filesystem events into
IndexSyncer calls. Requires the `watchdog` package (see requirements-local.txt)."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.local.sync import IndexSyncer

logger = logging.getLogger(__name__)


class FolderWatcher:
    def __init__(
        self,
        syncer: IndexSyncer,
        watch_dirs: list[Path],
        debounce_seconds: float = 1.5,
    ) -> None:
        self.syncer = syncer
        self.watch_dirs = watch_dirs
        self.debounce_seconds = debounce_seconds
        self._observer = None
        self._loop: asyncio.AbstractEventLoop | None = None

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
                logger.warning("LOCAL_WATCH_DIRS entry does not exist, skipping: %s", directory)
                continue
            self._observer.schedule(handler, str(directory), recursive=True)
            watched_any = True
        if watched_any:
            self._observer.start()
            logger.info("Watching %d folder(s) for changes", len(self.watch_dirs))

        for directory in self.watch_dirs:
            await self.syncer.scan_directory(directory)

    def _build_handler(self, base_cls):
        watcher = self

        class Handler(base_cls):
            def on_created(self, event) -> None:
                if not event.is_directory:
                    watcher._schedule(Path(event.src_path), removed=False)

            def on_modified(self, event) -> None:
                if not event.is_directory:
                    watcher._schedule(Path(event.src_path), removed=False)

            def on_deleted(self, event) -> None:
                if not event.is_directory:
                    watcher._schedule(Path(event.src_path), removed=True)

            def on_moved(self, event) -> None:
                if not event.is_directory:
                    watcher._schedule(Path(event.src_path), removed=True)
                    watcher._schedule(Path(event.dest_path), removed=False)

        return Handler()

    def _schedule(self, path: Path, removed: bool) -> None:
        # watchdog calls handlers from its own OS-thread; hop back onto the daemon's
        # asyncio loop so IndexSyncer's async ingest path can run there.
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._handle(path, removed), self._loop)

    async def _handle(self, path: Path, removed: bool) -> None:
        # Debounce: editors/OSes often fire several events per save.
        await asyncio.sleep(self.debounce_seconds)
        if removed:
            if not path.exists():
                self.syncer.remove_path(path)
        elif path.exists():
            await self.syncer.index_path(path)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
