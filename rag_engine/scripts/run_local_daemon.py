"""Convenience launcher for local mode: `python scripts/run_local_daemon.py`.

Requires PROFILE=local (env var) plus the extra dependencies in
requirements-local.txt (watchdog, pystray, Pillow, PyYAML). watched_paths and
everything else is normally configured via the auto-generated config.yaml — see
README's "Local mode" section for its path per OS.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tray import main

if __name__ == "__main__":
    raise SystemExit(main())
