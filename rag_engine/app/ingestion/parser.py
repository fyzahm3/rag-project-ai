"""Multi-format document loaders (PDF via PyMuPDF, Markdown, HTML, plain text)."""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Callable

from app.utils.errors import ParsingError
from app.utils.text import normalize_text
from pydantic import BaseModel, Field


class ParsedSection(BaseModel):
    heading: str = "(untitled)"
    text: str = ""
    page: int | None = None


class ParsedDocument(BaseModel):
    document_name: str
    doc_type: str
    sections: list[ParsedSection] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        parts: list[str] = []
        for section in self.sections:
            if section.heading and section.heading != "(untitled)":
                parts.append(section.heading)
            parts.append(section.text)
        return normalize_text("\n\n".join(parts))


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".txt", ".html", ".htm"})

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TXT_HEADING_RES = [
    re.compile(r"^#{1,6}\s+\S"),
    re.compile(r"^[A-Z][A-Z0-9 \-_/&+,\.]{2,60}$"),
    re.compile(r"^\d+(\.\d+){0,3}\s+\S.{0,70}$"),
    re.compile(r"^[A-Za-z][\w \-/]{2,58}:$"),
]
_HTML_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_HTML_SKIP = {"script", "style", "nav", "footer", "noscript", "head"}
_PDF_BODY_SIZE_RATIO = 1.15
_PDF_MAX_HEADING_CHARS = 90


def parse_document(filename: str, content: bytes) -> ParsedDocument:
    """Dispatch to the correct parser based on the file extension."""
    suffix = Path(filename).suffix.lower()
    name = Path(filename).name
    if suffix == ".pdf":
        return _parse_pdf(content, name)
    try:
        raw = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParsingError(f"Unable to decode {name} as UTF-8 text") from exc
    if suffix == ".md":
        return _parse_markdown(raw, name)
    if suffix in {".html", ".htm"}:
        return _parse_html(raw, name)
    if suffix == ".txt":
        return _parse_txt(raw, name)
    raise ParsingError(f"Unsupported file type '{suffix}' for {name}")


def _flush(sections: list[ParsedSection], heading: str, buffer: list[str], page: int | None) -> None:
    text = normalize_text("\n".join(buffer))
    if text:
        sections.append(ParsedSection(heading=heading or "(untitled)", text=text, page=page))


def _parse_markdown(raw: str, name: str) -> ParsedDocument:
    sections: list[ParsedSection] = []
    heading = "(untitled)"
    buffer: list[str] = []
    for line in raw.splitlines():
        match = _MD_HEADING_RE.match(line.strip())
        if match:
            _flush(sections, heading, buffer, None)
            buffer = []
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    _flush(sections, heading, buffer, None)
    if not sections:
        raise ParsingError(f"No extractable text found in {name}")
    return ParsedDocument(document_name=name, doc_type="markdown", sections=sections)


def _parse_txt(raw: str, name: str) -> ParsedDocument:
    sections: list[ParsedSection] = []
    heading = "(untitled)"
    buffer: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        is_heading = bool(stripped) and len(stripped) <= 72 and any(
            pattern.match(stripped) for pattern in _TXT_HEADING_RES
        )
        if is_heading:
            _flush(sections, heading, buffer, None)
            buffer = []
            heading = stripped.rstrip(":").strip()
        else:
            buffer.append(line)
    _flush(sections, heading, buffer, None)
    if not sections:
        raise ParsingError(f"No extractable text found in {name}")
    return ParsedDocument(document_name=name, doc_type="text", sections=sections)


def _parse_html(raw: str, name: str) -> ParsedDocument:
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all(list(_HTML_SKIP)):
        tag.decompose()

    sections: list[ParsedSection] = []
    state = {"heading": "(untitled)", "buffer": []}

    def flush() -> None:
        _flush(sections, state["heading"], state["buffer"], None)
        state["buffer"] = []

    def walk(node) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    state["buffer"].append(text)
            elif isinstance(child, Tag):
                if child.name in _HTML_HEADINGS:
                    flush()
                    state["heading"] = child.get_text(" ", strip=True) or state["heading"]
                elif child.name in {"ul", "ol", "table", "pre"}:
                    text = child.get_text("\n", strip=True)
                    if text:
                        state["buffer"].append(text)
                else:
                    walk(child)

    root = soup.body if soup.body else soup
    walk(root)
    flush()
    if not sections:
        raise ParsingError(f"No extractable text found in {name}")
    return ParsedDocument(document_name=name, doc_type="html", sections=sections)


def _pdf_page_lines(content: bytes) -> list[tuple[int, list[tuple[str, float]]]]:
    import fitz

    pages: list[tuple[int, list[tuple[str, float]]]] = []
    with fitz.open(stream=content, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            lines: list[tuple[str, float]] = []
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue
                    max_size = max((span.get("size", 10.0) for span in spans), default=10.0)
                    lines.append((text, float(max_size)))
            pages.append((page_number, lines))
    return pages


def _parse_pdf(content: bytes, name: str) -> ParsedDocument:
    try:
        pages = _pdf_page_lines(content)
    except Exception as exc:
        raise ParsingError(f"PyMuPDF failed to parse {name}: {exc}") from exc

    all_sizes = [size for _, lines in pages for _, size in lines]
    body_size = statistics.median(all_sizes) if all_sizes else 10.0
    heading_threshold = body_size * _PDF_BODY_SIZE_RATIO

    sections: list[ParsedSection] = []
    heading = ""
    buffer: list[str] = []
    current_page: int | None = None
    previous_was_heading = False

    for page_number, lines in pages:
        for text, size in lines:
            is_heading_line = size >= heading_threshold and len(text) <= _PDF_MAX_HEADING_CHARS
            if is_heading_line:
                if previous_was_heading and buffer == [""]:
                    heading = f"{heading} {text}".strip()
                    continue
                _flush(sections, heading or "(untitled)", buffer, current_page)
                buffer = []
                heading = text
                current_page = page_number
                previous_was_heading = True
                continue
            buffer.append(text)
            previous_was_heading = False
        if not lines:
            continue

    _flush(sections, heading or "(untitled)", buffer, current_page)
    if not sections:
        raise ParsingError(
            f"No extractable text found in {name}; the PDF may be scanned images without an OCR layer"
        )
    return ParsedDocument(document_name=name, doc_type="pdf", sections=sections)


def markdown_to_html(raw_markdown: str) -> str:
    """Utility conversion used by tooling that prefers rendered HTML."""
    import markdown as markdown_lib

    return markdown_lib.markdown(raw_markdown, extensions=["extra", "tables"])
