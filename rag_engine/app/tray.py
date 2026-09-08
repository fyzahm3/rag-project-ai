"""The day-to-day entry point for local mode: one script that runs the FastAPI
server (bound to 127.0.0.1, local profile defaults the port to 8787), the folder
crawler/watcher, and a tray icon on top of both.

Tray menu:
  - Open Search   -> opens the web UI (GET /, app/static/index.html) in the
                      default browser
  - Reindex now   -> triggers a fresh crawl of watched_paths on demand, without
                      waiting for the live watcher to notice anything
  - Open config file -> opens config.yaml in the default text editor
  - Quit          -> stops the watcher and the HTTP server cleanly

Run with:  python -m app.tray   (or scripts/run_local_daemon.py, a thin wrapper)
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.watcher import FolderWatcher, IndexSyncer
from app.main import AppContext, build_context, configure_logging, create_app

logger = logging.getLogger("rag_engine.tray")


def _configure_frozen_environment() -> None:
    """When running as a packaged app (see packaging/tray.spec), point Hugging
    Face's cache at the model bundled alongside the executable, if present, so a
    bundled embedding model is actually found instead of re-downloaded."""
    if getattr(sys, "frozen", False):
        bundle_dir = Path(getattr(sys, "_MEIPASS", "."))
        hf_cache = bundle_dir / "hf_cache"
        if hf_cache.exists():
            os.environ.setdefault("HF_HOME", str(hf_cache))
            logger.info("Using bundled model cache: %s", hf_cache)


def _open_in_default_app(path: Path) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif system == "Windows":
            import os

            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        logger.warning("Could not open %s: %s", path, exc)


class LocalDaemon:
    """Owns the HTTP server thread and the watcher's asyncio loop, and exposes the
    thread-safe controls the tray menu calls into."""

    def __init__(self, settings: Settings, ctx: AppContext | None = None) -> None:
        import uvicorn

        self.settings = settings
        self.ctx = ctx if ctx is not None else build_context(settings)
        self.syncer = IndexSyncer(
            self.ctx.indexer,
            strategy=settings.default_chunking_strategy,
            excluded_patterns=settings.excluded_patterns,
            max_index_file_size_mb=settings.max_index_file_size_mb,
            max_concurrent_indexing=settings.max_concurrent_indexing,
        )
        self.watch_dirs = [d.expanduser() for d in settings.watched_paths]

        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(self.ctx), host=settings.api_host, port=settings.api_port, log_level="warning"
            )
        )
        self._server_thread = threading.Thread(target=self._run_server, daemon=True, name="local-http-server")
        self._watcher_thread = threading.Thread(target=self._run_watcher, daemon=True, name="local-folder-watcher")
        self._watcher_loop: asyncio.AbstractEventLoop | None = None
        self._folder_watcher: FolderWatcher | None = None
        self._loop_ready = threading.Event()
        self._watcher_stop = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.settings.api_host}:{self.settings.api_port}"

    def start(self) -> None:
        self._server_thread.start()
        self._watcher_thread.start()
        self._loop_ready.wait(timeout=10)

    def _run_server(self) -> None:
        asyncio.run(self._server.serve())

    def _run_watcher(self) -> None:
        asyncio.run(self._watch_main())

    async def _watch_main(self) -> None:
        self._watcher_loop = asyncio.get_running_loop()
        self._loop_ready.set()
        if self.watch_dirs:
            self._folder_watcher = FolderWatcher(
                self.syncer, self.watch_dirs, debounce_seconds=self.settings.local_debounce_seconds
            )
            await self._folder_watcher.start()
        else:
            logger.warning(
                "No watched_paths configured; nothing will be indexed. Edit "
                "config.yaml and choose 'Reindex now' (or restart)."
            )
        while not self._watcher_stop.is_set():
            await asyncio.sleep(0.5)
        if self._folder_watcher is not None:
            self._folder_watcher.stop()

    def reindex_now(self) -> None:
        """Thread-safe: schedules a fresh crawl on the watcher's own asyncio loop
        (called from the tray's own thread, so this can't run the crawl directly)."""
        if not self.watch_dirs:
            logger.warning("Cannot reindex: no watched_paths configured.")
            return
        if self._watcher_loop is None:
            logger.warning("Watcher isn't running yet; try again shortly.")
            return
        asyncio.run_coroutine_threadsafe(self.syncer.crawl(self.watch_dirs), self._watcher_loop)
        logger.info("Reindex triggered for %s", self.watch_dirs)

    def stop(self) -> None:
        self._server.should_exit = True
        self._watcher_stop.set()
        self._server_thread.join(timeout=5)
        self._watcher_thread.join(timeout=5)


def _build_icon_image():
    from PIL import Image, ImageDraw

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, size - 8, size - 8), fill=(66, 133, 244, 255))
    draw.ellipse((22, 22, 42, 42), fill=(255, 255, 255, 255))
    return image


def main() -> int:
    _configure_frozen_environment()
    settings = get_settings()
    if settings.profile != "local":
        raise SystemExit(
            "PROFILE is not 'local'. Set PROFILE=local (env var) before running the "
            "tray app; use `uvicorn app.main:app` for the server profile."
        )
    configure_logging(settings)

    try:
        import pystray
    except ImportError as exc:
        raise RuntimeError(
            "The tray app requires 'pystray' and 'Pillow'. Install with: "
            "pip install -r requirements-local.txt"
        ) from exc

    from app.utils.paths import local_config_path

    daemon = LocalDaemon(settings)
    logger.info("Starting local daemon (config: %s, serving %s)", local_config_path(), daemon.url)
    daemon.start()

    def on_open_search(icon, item) -> None:
        webbrowser.open(daemon.url)

    def on_reindex(icon, item) -> None:
        daemon.reindex_now()

    def on_open_config(icon, item) -> None:
        _open_in_default_app(local_config_path())

    def on_quit(icon, item) -> None:
        daemon.stop()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open Search", on_open_search, default=True),
        pystray.MenuItem("Reindex now", on_reindex),
        pystray.MenuItem("Open config file", on_open_config),
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("rag-local", _build_icon_image(), "Local RAG Search", menu)
    try:
        icon.run()
    finally:
        daemon.stop()
        logger.info("Local daemon stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
