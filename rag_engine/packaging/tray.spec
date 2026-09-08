"""PyInstaller spec for the local-mode tray app (app/tray.py).

Produces a onedir bundle per OS — a double-clickable RagSearch.app on macOS
(LSUIElement, so it's menu-bar-only with no Dock icon), a RagSearch/ folder with
RagSearch.exe at its root on Windows. This is the packaged alternative to running
from source (see README's "Local mode" section for both paths); from-source is the
one documented as primary, since building per-OS binaries is optional extra work.

Bundles the on-device embedding model (BAAI/bge-small-en-v1.5) if it's already in
the local Hugging Face cache, so the packaged app doesn't need network access on
first run. It won't be there on a clean checkout — run the app from source once
first (with network access, so sentence-transformers downloads and caches it), then
build. If it's missing at build time, this spec just skips bundling it and the
packaged app falls back to downloading it on first run instead (no build error).

Build (from the repo root, with `pip install -r requirements-local.txt pyinstaller`):
    pyinstaller packaging/tray.spec

Output: dist/RagSearch/ (Windows/Linux) or dist/RagSearch.app (macOS).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None
REPO_ROOT = Path(SPECPATH).resolve().parent
APP_NAME = "RagSearch"
EMBEDDING_MODEL_REPO_ID = "BAAI/bge-small-en-v1.5"


def _embedding_model_cache_snapshot() -> Path | None:
    """Locate the already-downloaded HF snapshot dir for EMBEDDING_MODEL_REPO_ID,
    if any is cached locally."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(EMBEDDING_MODEL_REPO_ID, local_files_only=True))
    except Exception:
        return None


extra_datas: list[tuple[str, str]] = [
    (str(REPO_ROOT / "app" / "static"), "app/static"),
]

model_snapshot = _embedding_model_cache_snapshot()
if model_snapshot is not None:
    # Mirror the cache layout HF expects (hub/models--org--name/snapshots/<rev>)
    # under a bundle-local hf_cache/ dir; app/tray.py points HF_HOME there when
    # running as a frozen bundle (see _configure_frozen_environment).
    snapshots_dir = model_snapshot.parent  # .../snapshots
    model_root = snapshots_dir.parent  # .../models--org--name
    rel_target = Path("hf_cache") / "hub" / model_root.name / "snapshots" / model_snapshot.name
    extra_datas.append((str(model_snapshot), str(rel_target)))
    print(f"[tray.spec] Bundling cached embedding model from {model_snapshot}")
else:
    print(
        f"[tray.spec] {EMBEDDING_MODEL_REPO_ID} not found in the local Hugging Face "
        "cache; building without it. Run the app from source once first (with "
        "network access) so it's cached, then rebuild, to bundle it."
    )

collected_datas: list[tuple[str, str]] = []
collected_hidden: list[str] = []
for package in ("sentence_transformers", "transformers", "torch", "chromadb", "tiktoken"):
    try:
        pkg_datas, _pkg_binaries, pkg_hidden = collect_all(package)
        collected_datas.extend(pkg_datas)
        collected_hidden.extend(pkg_hidden)
    except Exception:
        pass

a = Analysis(
    [str(REPO_ROOT / "app" / "tray.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=[*extra_datas, *collected_datas],
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
        "pystray._darwin",
        "pystray._win32",
        "pystray._xorg",
        *collected_hidden,
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "pytest_asyncio"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier=f"com.ragsearch.{APP_NAME.lower()}",
        info_plist={
            "LSUIElement": True,  # menu-bar-only: no Dock icon, no app-switcher entry
            "NSHighResolutionCapable": True,
        },
    )
