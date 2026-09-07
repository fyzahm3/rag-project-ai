"""Per-OS application data directory for local (profile=local) mode."""
from __future__ import annotations

import os
import platform
from pathlib import Path

_APP_NAME = "RagSearch"


def app_data_dir() -> Path:
    """macOS: ~/Library/Application Support/RagSearch/
    Windows: %LOCALAPPDATA%\\RagSearch\\
    Anything else (Linux/CI): ~/.local/share/RagSearch/ as a reasonable fallback."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / _APP_NAME
    return Path.home() / ".local" / "share" / _APP_NAME


def local_config_path() -> Path:
    return app_data_dir() / "config.yaml"
