"""First-run wizard for local mode, called by scripts/setup.sh / setup.ps1 after
dependencies are installed:
  1. Prompt for which folders to index (default: Documents/Desktop), write config.yaml
     (skipped if a config already exists — never clobbers a returning user's edits)
  2. Run the initial crawl over those folders, with progress logged as it goes
  3. Print the exact command to start the daemon, and offer to install the
     auto-start-on-login helper for this OS
"""
from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Local profile must be selected before Settings/get_settings() are ever touched,
# since that's what decides whether config.yaml gets loaded at all.
os.environ["PROFILE"] = "local"

from app.config import get_settings  # noqa: E402
from app.ingestion.watcher import IndexSyncer  # noqa: E402
from app.local.config_yaml import DEFAULT_WATCHED_PATHS, write_default_local_config  # noqa: E402
from app.main import build_context, configure_logging  # noqa: E402
from app.utils.paths import local_config_path  # noqa: E402


def _prompt_watched_paths() -> list[str]:
    default_display = ", ".join(DEFAULT_WATCHED_PATHS)
    print(f"Which folders should be indexed? [default: {default_display}]")
    response = input("Enter comma-separated paths, or press Enter to accept the default: ").strip()
    if not response:
        return DEFAULT_WATCHED_PATHS
    return [part.strip() for part in response.split(",") if part.strip()]


def _run_wizard_if_needed() -> None:
    config_path = local_config_path()
    if config_path.exists():
        print(f"Using existing config: {config_path}")
        return
    watched_paths = _prompt_watched_paths()
    write_default_local_config(config_path, watched_paths=watched_paths)
    print(f"Wrote {config_path} watching: {', '.join(watched_paths)}")
    print("(Edit this file any time to add/remove folders or tweak other settings.)")


def _run_initial_crawl() -> None:
    settings = get_settings()
    configure_logging(settings)
    ctx = build_context(settings)
    syncer = IndexSyncer(
        ctx.indexer,
        strategy=settings.default_chunking_strategy,
        excluded_patterns=settings.excluded_patterns,
        max_index_file_size_mb=settings.max_index_file_size_mb,
        max_concurrent_indexing=settings.max_concurrent_indexing,
    )
    watch_dirs = [d.expanduser() for d in settings.watched_paths]
    if not watch_dirs:
        print("No watched_paths configured; skipping the initial crawl.")
        return
    print(f"Indexing {', '.join(str(d) for d in watch_dirs)} — this may take a while the first time...")
    stats = asyncio.run(syncer.crawl(watch_dirs))
    print(f"Done: {stats.scanned} scanned, {stats.indexed} indexed, {stats.skipped} skipped, {stats.errors} errors.")


def _offer_autostart() -> None:
    system = platform.system()
    installer = ROOT / "scripts" / ("install_macos.sh" if system == "Darwin" else "install_windows.ps1")
    if system not in ("Darwin", "Windows"):
        return
    response = input("Start this automatically when you log in? [y/N]: ").strip().lower()
    if response not in ("y", "yes"):
        print("Skipping auto-start. You can run this later:")
        print(f"  {installer}")
        return
    if system == "Darwin":
        subprocess.run(["bash", str(installer)], check=False)
    else:
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(installer)], check=False
        )


def main() -> int:
    _run_wizard_if_needed()
    _run_initial_crawl()
    print()
    print("Setup complete. Start the tray app with:")
    print(f"  python {ROOT / 'app' / 'tray.py'}")
    _offer_autostart()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
