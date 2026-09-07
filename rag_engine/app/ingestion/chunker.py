"""Three switchable chunking strategies: fixed, structure-aware, and semantic-boundary."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel

from app.config import Settings
from app.ingestion.parser import ParsedDocument
from app.utils.text import path_doc_id, sha1_id, split_sentences, truncate

EmbedFn = Callable[[list[str]], np.ndarray]

STRATEGIES: tuple[str, ...] = ("fixed", "structure", "semantic")


class Chunk(BaseModel):
    chunk_id: str
    document_name: str
    section_heading: str
    text: str
    page: int | None = None
    strategy: str
    seq: int


class BaseChunker(ABC):
    strategy_name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        raw = self._raw_chunks(document)
        chunks: list[Chunk] = []
        min_chars = self.settings.min_chunk_chars
        for seq, (heading, text, page) in enumerate(raw):
            text = text.strip()
            if not text:
                continue
            if len(text) > self.settings.max_chunk_chars:
                text = truncate(text, self.settings.max_chunk_chars)
            if len(text) < min_chars and len(raw) > 1:
                continue
            if self.settings.profile == "local":
                # Path+index based, not content based: stable across re-indexing a
                # changed file (unchanged chunk positions keep their id), and lets
                # the local watcher delete a whole file's chunks by id prefix alone.
                chunk_id = f"{path_doc_id(document.document_name)}:{seq}"
            else:
                chunk_id = sha1_id(document.document_name, heading, seq, text[:120])
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_name=document.document_name,
                    section_heading=heading,
                    text=text,
                    page=page,
                    strategy=self.strategy_name,
                    seq=seq,
                )
            )
        return chunks

    @abstractmethod
    def _raw_chunks(self, document: ParsedDocument) -> list[tuple[str, str, int | None]]:
        """Return (section_heading, text, page) triples before finalization."""


def _window_slices(text: str, size: int, step: int) -> list[tuple[int, str]]:
    slices: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        slices.append((start, text[start:end].strip()))
        if end >= n:
            break
        start += step
    return [s for s in slices if s[1]]


class FixedChunker(BaseChunker):
    """Sliding character window with configurable overlap across the whole document."""

    strategy_name = "fixed"

    def _raw_chunks(self, document: ParsedDocument) -> list[tuple[str, str, int | None]]:
        full_text = document.full_text
        spans: list[tuple[int, str]] = []
        cursor = 0
        for section in document.sections:
            block_parts = [p for p in (section.heading, section.text) if p]
            block_len = sum(len(p) + 2 for p in block_parts)
            spans.append((cursor, section.heading))
            cursor += max(block_len, 1)

        size = self.settings.chunk_size
        step = max(1, size - self.settings.chunk_overlap)
        results: list[tuple[str, str, int | None]] = []
        starts = [offset for offset, _ in spans]
        for offset, piece in _window_slices(full_text, size, step):
            index = bisect_right(starts, offset) - 1
            heading = spans[max(index, 0)][1] if spans else "(untitled)"
            page = next(
                (s.page for s in document.sections if s.heading == heading and s.page is not None),
                None,
            )
            results.append((heading, piece, page))
        return results


class StructureAwareChunker(BaseChunker):
    """Keeps markdown/HTML/text sections intact; falls back to windows inside oversized sections."""

    strategy_name = "structure"

    def _raw_chunks(self, document: ParsedDocument) -> list[tuple[str, str, int | None]]:
        size = self.settings.chunk_size
        step = max(1, size - self.settings.chunk_overlap)
        results: list[tuple[str, str, int | None]] = []
        for section in document.sections:
            heading = section.heading or "(untitled)"
            body = normalize_paragraphs(section.text)
            if not body:
                continue
            if len(body) <= size:
                results.append((heading, body, section.page))
                continue
            for _, piece in _window_slices(body, size, step):
                results.append((heading, piece, section.page))
        return results


class SemanticBoundaryChunker(BaseChunker):
    """Splits where cosine distance between consecutive sentence embeddings exceeds a threshold."""

    strategy_name = "semantic"

    def __init__(self, settings: Settings, embed_fn: EmbedFn) -> None:
        super().__init__(settings)
        if embed_fn is None:
            raise ValueError("SemanticBoundaryChunker requires an embed_fn")
        self._embed_fn = embed_fn

    def _raw_chunks(self, document: ParsedDocument) -> list[tuple[str, str, int | None]]:
        size = self.settings.chunk_size
        threshold = self.settings.semantic_distance_threshold
        min_sentences = self.settings.semantic_min_sentences
        results: list[tuple[str, str, int | None]] = []

        for section in document.sections:
            heading = section.heading or "(untitled)"
            sentences = [s for s in split_sentences(section.text) if s.strip()]
            if not sentences:
                continue
            total_chars = sum(len(s) for s in sentences)
            if len(sentences) <= 1 or total_chars <= size:
                results.append((heading, " ".join(sentences), section.page))
                continue

            vectors = l2_normalize(np.asarray(self._embed_fn(sentences), dtype=np.float32))
            distances = 1.0 - np.clip(vectors[:-1] @ vectors[1:].T, -1.0, 1.0).diagonal()

            groups: list[list[str]] = [[sentences[0]]]
            for i, distance in enumerate(distances):
                if float(distance) > threshold:
                    groups.append([sentences[i + 1]])
                else:
                    groups[-1].append(sentences[i + 1])

            merged = self._merge_small_groups(groups, min_sentences)
            for segment_sentences in merged:
                segment_text = " ".join(segment_sentences)
                if len(segment_text) <= size:
                    results.append((heading, segment_text, section.page))
                    continue
                step = max(1, size - self.settings.chunk_overlap)
                for _, piece in _window_slices(segment_text, size, step):
                    results.append((heading, piece, section.page))
        return results

    @staticmethod
    def _merge_small_groups(groups: list[list[str]], min_sentences: int) -> list[list[str]]:
        merged: list[list[str]] = []
        for group in groups:
            if merged and (len(group) < min_sentences or len(merged[-1]) < min_sentences):
                merged[-1].extend(group)
            else:
                merged.append(group)
        return merged


def normalize_paragraphs(text: str) -> str:
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe


def get_chunker(
    strategy: str,
    settings: Settings,
    embed_fn: EmbedFn | None = None,
) -> BaseChunker:
    normalized = (strategy or settings.default_chunking_strategy).lower()
    if normalized == "fixed":
        return FixedChunker(settings)
    if normalized == "structure":
        return StructureAwareChunker(settings)
    if normalized == "semantic":
        if embed_fn is None:
            raise ValueError("strategy='semantic' requires an embedding function")
        return SemanticBoundaryChunker(settings, embed_fn)
    raise ValueError(f"Unknown chunking strategy '{strategy}'. Expected one of {STRATEGIES}")


def batch_chunks_to_texts(chunks: Sequence[Chunk]) -> list[str]:
    return [f"{c.section_heading}\n\n{c.text}" if c.section_heading else c.text for c in chunks]


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))
