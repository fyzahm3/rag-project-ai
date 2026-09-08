"""Tests for the local-mode YAML config file: default generation + override loading."""
from __future__ import annotations

from pathlib import Path

from app.local.config_yaml import (
    load_local_yaml_overrides,
    render_config_yaml,
    write_default_local_config,
)


def test_generates_default_config_on_first_run(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    assert not config_path.exists()

    overrides = load_local_yaml_overrides(config_path)

    assert config_path.exists()
    assert overrides == {}  # nothing to override yet — the file is all comments/defaults
    content = config_path.read_text(encoding="utf-8")
    assert "watched_paths" in content
    assert "~/Documents" in content


def test_write_default_config_is_idempotent_content(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_default_local_config(config_path)
    first = config_path.read_text(encoding="utf-8")
    write_default_local_config(config_path)
    second = config_path.read_text(encoding="utf-8")
    assert first == second


def test_loads_explicit_overrides_from_existing_file(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "watched_paths:\n"
        "  - /Users/me/Notes\n"
        "excluded_patterns:\n"
        "  - node_modules\n"
        "llm_provider: ollama\n"
        "ollama_chat_model: llama3.1:8b\n",
        encoding="utf-8",
    )

    overrides = load_local_yaml_overrides(config_path)

    assert overrides["watched_paths"] == ["/Users/me/Notes"]
    assert overrides["excluded_patterns"] == ["node_modules"]
    assert overrides["llm_provider"] == "ollama"
    assert overrides["ollama_chat_model"] == "llama3.1:8b"


def test_ignores_null_values_and_malformed_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("watched_paths:\nllm_provider: ollama\n", encoding="utf-8")
    overrides = load_local_yaml_overrides(config_path)
    assert "watched_paths" not in overrides  # null -> dropped, falls through to built-in default
    assert overrides["llm_provider"] == "ollama"

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("watched_paths: [unclosed\n", encoding="utf-8")
    assert load_local_yaml_overrides(bad_path) == {}


def test_render_config_yaml_writes_user_chosen_paths_active_not_commented():
    content = render_config_yaml(watched_paths=["/Users/me/Projects", "/Users/me/Notes"])
    assert "watched_paths:\n  - /Users/me/Projects\n  - /Users/me/Notes" in content
    assert "~/Documents" not in content  # user's picks replace the default, not append


def test_render_config_yaml_falls_back_to_default_paths_when_none_given():
    content = render_config_yaml()
    assert "watched_paths:\n  - ~/Documents\n  - ~/Desktop" in content


def test_wizard_writes_config_with_user_chosen_paths_that_parse_as_overrides(tmp_path: Path):
    """The first-run wizard's write path (write_default_local_config with explicit
    watched_paths) must round-trip through the normal YAML loader like any other
    config.yaml, since that's how first_run.py and get_settings() consume it."""
    config_path = tmp_path / "config.yaml"
    write_default_local_config(config_path, watched_paths=["/data/one", "/data/two"])

    overrides = load_local_yaml_overrides(config_path)
    assert overrides["watched_paths"] == ["/data/one", "/data/two"]


def test_overrides_feed_directly_into_settings(tmp_path: Path):
    from app.config import Settings

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"watched_paths:\n  - {tmp_path / 'Notes'}\n"
        "llm_provider: ollama\n"
        "ollama_chat_model: llama3.1:8b\n",
        encoding="utf-8",
    )
    overrides = load_local_yaml_overrides(config_path)
    settings = Settings(
        profile="local",
        data_dir=tmp_path / "appdata",
        raw_docs_dir=tmp_path / "appdata" / "raw_docs",
        chroma_dir=tmp_path / "appdata" / "chroma",
        index_dir=tmp_path / "appdata" / "index",
        benchmark_path=tmp_path / "appdata" / "benchmarks" / "golden_qa.jsonl",
        reports_dir=tmp_path / "appdata" / "reports",
        **overrides,
    )
    assert settings.watched_paths == [tmp_path / "Notes"]
    assert settings.ollama_chat_model == "llama3.1:8b"
