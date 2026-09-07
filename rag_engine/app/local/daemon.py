"""Local profile entry point: watches configured folders, keeps them indexed, and
serves search from a tray icon — using the same Settings/build_context/RAGService as
the FastAPI server (app.main), selected via PROFILE=local.

Run with:  python -m app.local.daemon   (or scripts/run_local_daemon.py)
"""
from __future__ import annotations

import asyncio
import logging
import threading

from app.config import Settings, get_settings
from app.local.sync import IndexSyncer
from app.local.tray import run_tray
from app.local.watcher import FolderWatcher
from app.main import build_context, configure_logging

logger = logging.getLogger("rag_engine.local")


async def _run_watcher(settings: Settings, syncer: IndexSyncer, stop_event: threading.Event) -> None:
    watch_dirs = [d.expanduser() for d in settings.watched_paths]
    if not watch_dirs:
        logger.warning(
            "No watched_paths configured; nothing will be indexed. Edit config.yaml "
            "(see the path logged above) and restart."
        )
        return

    watcher = FolderWatcher(syncer, watch_dirs, debounce_seconds=settings.local_debounce_seconds)
    await watcher.start()
    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.5)
    finally:
        watcher.stop()


def main() -> int:
    settings = get_settings()
    if settings.profile != "local":
        raise SystemExit(
            "PROFILE is not 'local'. Set PROFILE=local (env var) before running the "
            "tray daemon; use `uvicorn app.main:app` for the server profile."
        )
    configure_logging(settings)

    from app.utils.paths import local_config_path

    logger.info("Starting local daemon (config: %s)", local_config_path())
    logger.info("Watched paths: %s", settings.watched_paths)

    ctx = build_context(settings)
    syncer = IndexSyncer(
        ctx.indexer,
        strategy=settings.default_chunking_strategy,
        excluded_patterns=settings.excluded_patterns,
        max_index_file_size_mb=settings.max_index_file_size_mb,
    )
    stop_event = threading.Event()

    watcher_thread = threading.Thread(
        target=lambda: asyncio.run(_run_watcher(settings, syncer, stop_event)),
        daemon=True,
        name="local-folder-watcher",
    )
    watcher_thread.start()

    try:
        run_tray(settings, ctx.rag, stop_event)
    finally:
        stop_event.set()
        watcher_thread.join(timeout=5)
        logger.info("Local daemon stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
