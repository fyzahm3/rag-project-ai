"""Minimal always-on-top Tkinter search window for local mode (stdlib-only UI deps)."""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

from app.pipeline import RAGService
from app.schemas.query import QueryRequest

logger = logging.getLogger(__name__)


def open_in_default_app(path: str) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", path], check=False)
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:
        logger.warning("Could not open %s: %s", path, exc)


class SearchWindow:
    """Runs its own Tk mainloop on a dedicated thread so it doesn't block the tray
    icon's loop or the watcher's asyncio loop; each query runs the async RAG pipeline
    via its own `asyncio.run` call on a worker thread."""

    def __init__(self, rag: RAGService, top_k: int) -> None:
        self.rag = rag
        self.top_k = top_k
        self._root: tk.Tk | None = None
        self._thread: threading.Thread | None = None

    def show(self) -> None:
        if self._thread and self._thread.is_alive():
            if self._root is not None:
                self._root.after(0, self._root.deiconify)
                self._root.after(0, self._root.lift)
                self._root.after(0, self._root.focus_force)
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("Local Search")
        root.geometry("640x420")
        root.attributes("-topmost", True)

        query_var = tk.StringVar()
        entry = ttk.Entry(root, textvariable=query_var, font=("Helvetica", 14))
        entry.pack(fill="x", padx=10, pady=10)
        entry.focus_set()

        results = tk.Listbox(root, font=("Helvetica", 11))
        results.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        status = ttk.Label(root, text="Type a query and press Enter · double-click a result to open it")
        status.pack(fill="x", padx=10, pady=(0, 10))

        state: dict[str, list] = {"citations": []}

        def populate(response) -> None:
            state["citations"] = response.citations
            results.delete(0, tk.END)
            for citation in response.citations:
                results.insert(
                    tk.END, f"[{citation.citation_id}] {citation.source_doc} — {citation.section_heading}"
                )
            status.config(
                text=f"{len(response.citations)} result(s) · confidence {response.confidence_score:.2f}"
            )

        def run_search(_event=None) -> None:
            query = query_var.get().strip()
            if not query:
                return
            status.config(text="Searching…")

            def worker() -> None:
                try:
                    response = asyncio.run(self.rag.ask(QueryRequest(query=query, top_k=self.top_k)))
                except Exception as exc:
                    logger.warning("Local search failed: %s", exc)
                    root.after(0, lambda: status.config(text=f"Error: {exc}"))
                    return
                root.after(0, lambda: populate(response))

            threading.Thread(target=worker, daemon=True).start()

        def open_selected(_event=None) -> None:
            selection = results.curselection()
            if not selection:
                return
            citation = state["citations"][selection[0]]
            open_in_default_app(citation.source_doc)

        entry.bind("<Return>", run_search)
        results.bind("<Double-Button-1>", open_selected)
        root.bind("<Escape>", lambda _e: root.withdraw())

        root.mainloop()
