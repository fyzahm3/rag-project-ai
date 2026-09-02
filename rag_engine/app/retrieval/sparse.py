"""BM25 sparse retrieval over a persisted, tokenized corpus."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.retrieval.dense import RetrievedRecord
from app.utils.text import tokenize

logger = logging.getLogger(__name__)


class SparseIndex:
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
