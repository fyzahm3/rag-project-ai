"""Exclusion matching for crawling/watching local filesystem paths."""
from __future__ import annotations

import fnmatch
from pathlib import Path


def is_excluded(path: Path, patterns: list[str]) -> bool:
    """True if any path component matches a bare directory/file-name pattern (e.g.
    "node_modules", ".git"), or the filename matches a glob pattern (e.g. "*.zip")."""
    name = path.name
    parts = path.parts
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            if fnmatch.fnmatch(name, pattern):
                return True
        elif pattern in parts:
            return True
    return False


def exceeds_size_limit(path: Path, max_mb: int) -> bool:
    try:
        return path.stat().st_size > max_mb * 1024 * 1024
    except OSError:
        return False
