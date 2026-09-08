"""User-editable local-mode config file (per-OS app data dir), generated with sensible
defaults on first run. Only imported/used when PROFILE=local, so PyYAML
(requirements-local.txt) is never required by the server profile."""
from __future__ import annotations

import logging
from pathlib import Path

from app.utils.paths import local_config_path

logger = logging.getLogger(__name__)

DEFAULT_WATCHED_PATHS = ["~/Documents", "~/Desktop"]

_CONFIG_TEMPLATE = """\
# RagSearch local mode configuration.
# Edit this file, then restart the daemon (or choose "Reindex now" from the tray
# menu for watched_paths/excluded_patterns changes) to apply changes. Anything left
# commented out uses the built-in default. Secrets (like an OpenAI API key) are
# better set as an environment variable than pasted here, since this file is not
# encrypted.

# Folders to index and watch for changes. Add as many as you like.
{watched_paths_block}

# Directory- and file-name patterns to skip while scanning/watching (glob-style).
excluded_patterns:
  - node_modules
  - .git
  - venv
  - .venv
  - __pycache__
  - dist
  - build
  - Library
  - AppData
  - "*.app"
  - "*.exe"
  - "*.dll"
  - "*.zip"
  - "*.dmg"
  - "*.iso"

# Skip any file larger than this, regardless of extension (protects against a huge
# PDF or an accidentally-matched binary eating time/memory).
# max_index_file_size_mb: 25

# Embedding model — runs locally on-device, no API key needed.
# embedding_provider: huggingface
# hf_embedding_model: BAAI/bge-small-en-v1.5

# Device for local models: auto | cpu | cuda | mps
# local_device: auto

# Generation LLM. Uses a local Ollama model by default (requires Ollama running:
# https://ollama.com). Set openai_api_key here or export OPENAI_API_KEY to use
# OpenAI instead — if a key is present, OpenAI is used; otherwise Ollama.
# llm_provider: ollama
# ollama_base_url: http://localhost:11434/v1
# ollama_chat_model: qwen2.5:7b
# openai_api_key:

# Results shown per query in the tray search window.
# local_top_k: 8
"""


def _render_watched_paths_block(watched_paths: list[str] | None) -> str:
    paths = watched_paths if watched_paths else DEFAULT_WATCHED_PATHS
    lines = ["watched_paths:"]
    lines.extend(f"  - {p}" for p in paths)
    return "\n".join(lines)


def render_config_yaml(watched_paths: list[str] | None = None) -> str:
    """Render the config file text. `watched_paths` (if given) is written in
    *active* (uncommented) form — used by the first-run wizard when the user
    picked specific folders; omitted, it falls back to the same
    Documents/Desktop default the silent auto-generation path has always used."""
    return _CONFIG_TEMPLATE.format(watched_paths_block=_render_watched_paths_block(watched_paths))


def write_default_local_config(path: Path, watched_paths: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config_yaml(watched_paths), encoding="utf-8")
    logger.info("Wrote local config to %s", path)


def load_local_yaml_overrides(path: Path | None = None) -> dict:
    """Load config.yaml as a dict of Settings field overrides, generating the file
    with defaults first if it doesn't exist yet."""
    config_path = path or local_config_path()
    if not config_path.exists():
        write_default_local_config(config_path)
        return {}

    import yaml

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning("Could not parse %s (%s); using built-in defaults", config_path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("%s did not contain a YAML mapping; ignoring", config_path)
        return {}
    return {k: v for k, v in data.items() if v is not None}
