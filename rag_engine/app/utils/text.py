"""Deterministic text utilities shared across ingestion, retrieval and generation."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been being but by for from has have if in into is it its of on or
    our per so than that the their them then these they this to was were what when where which
    who will with would you your does do did can could should may might must about after all
    also any because both each how more most no not now only other over same some such under
    up very we us""".split()
)

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LONG_SENTENCE_CAP = 600


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = [_WHITESPACE_RE.sub(" ", p.replace("\n", " ")).strip() for p in paragraphs]
    return "\n\n".join(p for p in cleaned if p)


def split_sentences(text: str) -> list[str]:
    """Regex sentence segmentation with a hard cap for pathological sentences."""
    sentences: list[str] = []
    for paragraph in normalize_text(text).split("\n\n"):
        pieces = _SENTENCE_SPLIT_RE.split(paragraph)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            while len(piece) > _LONG_SENTENCE_CAP:
                cut = piece.rfind("; ", 0, _LONG_SENTENCE_CAP)
                if cut == -1:
                    cut = piece.rfind(", ", 0, _LONG_SENTENCE_CAP)
                if cut == -1:
                    cut = _LONG_SENTENCE_CAP
                head, piece = piece[: cut + 1].strip(), piece[cut + 1 :].strip()
                if head:
                    sentences.append(head)
            if piece:
                sentences.append(piece)
    return sentences


@lru_cache(maxsize=100_000)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenization used by BM25 and lexical scoring."""
    tokens = [t for t in _tokenize_cached(text) if len(t) > 1]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def token_coverage(claim_tokens: list[str], evidence_tokens: list[str]) -> float:
    """Fraction of claim content tokens present in the evidence."""
    if not claim_tokens:
        return 0.0
    evidence = set(evidence_tokens)
    hits = sum(1 for tok in claim_tokens if tok in evidence)
    return hits / len(claim_tokens)


def sha1_id(*parts: object) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return digest.hexdigest()[:16]


def path_doc_id(path: str) -> str:
    """Stable, content-independent id for a file path (local-mode chunk_id prefix)."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[: cut if cut > 0 else max_chars].rstrip()


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
