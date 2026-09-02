"""Document parsing, multi-strategy chunking and dual-index (dense+sparse) ingestion."""

from app.ingestion.chunker import (
    BaseChunker,
    Chunk,
    FixedChunker,
    SemanticBoundaryChunker,
    StructureAwareChunker,
    get_chunker,
)
from app.ingestion.indexer import Indexer
from app.ingestion.parser import ParsedDocument, ParsedSection, parse_document

__all__ = [
    "BaseChunker",
    "FixedChunker",
    "StructureAwareChunker",
    "SemanticBoundaryChunker",
    "get_chunker",
    "Chunk",
    "ParsedDocument",
    "ParsedSection",
    "parse_document",
    "Indexer",
]
