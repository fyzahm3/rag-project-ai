"""Local deployment mode entry point: watches configured folders, keeps them indexed,
and serves search from a tray icon — using the same Settings/build_context/RAGService
as the FastAPI server (app.main), selected via DEPLOYMENT_MODE=local.

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
    watch_dirs = [d.expanduser() for d in settings.local_watch_dirs]
    if not watch_dirs:
        logger.warning(
            "No LOCAL_WATCH_DIRS configured; nothing will be indexed. "
            "Set LOCAL_WATCH_DIRS=/path/one,/path/two in .env."
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
    if settings.deployment_mode != "local":
        raise SystemExit(
            "DEPLOYMENT_MODE is not 'local'. Set DEPLOYMENT_MODE=local (env var or .env) "
            "before running the tray daemon; use `uvicorn app.main:app` for the server mode."
        )
    configure_logging(settings)
    logger.info("Starting local daemon (watch dirs: %s)", settings.local_watch_dirs)

    ctx = build_context(settings)
    syncer = IndexSyncer(ctx.indexer, strategy=settings.default_chunking_strategy)
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
