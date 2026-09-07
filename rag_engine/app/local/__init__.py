"""Local profile support code that isn't part of the core ingestion pipeline.

The crawler/watcher lives in app.ingestion.watcher (an ingestion concern); the tray
app, HTTP server thread and watcher wiring live in app.tray (the actual entry
point — see app/tray.py). This package currently just holds the per-OS YAML config
loader (app.local.config_yaml).
"""
