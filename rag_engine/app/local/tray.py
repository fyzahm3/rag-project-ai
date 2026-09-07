"""System tray icon (pystray) wiring: opens the search window, shows a quit item.

Requires `pystray` and `Pillow` (see requirements-local.txt). Runs on macOS and
Windows via pystray's native backends.
"""
from __future__ import annotations

import logging
import threading

from app.config import Settings
from app.local.search_ui import SearchWindow
from app.pipeline import RAGService

logger = logging.getLogger(__name__)


def _build_icon_image():
    from PIL import Image, ImageDraw

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, size - 8, size - 8), fill=(66, 133, 244, 255))
    draw.ellipse((22, 22, 42, 42), fill=(255, 255, 255, 255))
    return image


def run_tray(settings: Settings, rag: RAGService, stop_event: threading.Event) -> None:
    """Blocks on the tray icon's own event loop until Quit is chosen."""
    try:
        import pystray
    except ImportError as exc:
        raise RuntimeError(
            "Local mode requires 'pystray' and 'Pillow'. Install with: "
            "pip install -r requirements-local.txt"
        ) from exc

    search_window = SearchWindow(rag=rag, top_k=settings.local_top_k)

    def on_search(icon, item) -> None:
        search_window.show()

    def on_quit(icon, item) -> None:
        stop_event.set()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Search…", on_search, default=True),
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("rag-local", _build_icon_image(), "Local RAG Search", menu)
    icon.run()
