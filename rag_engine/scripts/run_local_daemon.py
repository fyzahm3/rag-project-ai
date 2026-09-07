"""Convenience launcher for local mode: `python scripts/run_local_daemon.py`.

Requires DEPLOYMENT_MODE=local and LOCAL_WATCH_DIRS set (env var or .env), plus the
extra dependencies in requirements-local.txt (watchdog, pystray, Pillow).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.local.daemon import main

if __name__ == "__main__":
    raise SystemExit(main())
