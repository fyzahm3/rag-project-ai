#!/usr/bin/env bash
# One-shot setup for local mode on macOS/Linux: checks Python, creates a venv,
# installs requirements-local.txt, checks for Ollama, then hands off to
# first_run.py for the config wizard + initial crawl.
#
# Usage:
#   ./scripts/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== 1/6: Checking for Python 3.11+ =="
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    major="${version%%.*}"
    minor="${version##*.}"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.11+ not found."
  echo "Install it from https://www.python.org/downloads/ (or via Homebrew: brew install python@3.12)"
  echo "then re-run this script."
  exit 1
fi
echo "Using $("$PYTHON_BIN" --version) at $(command -v "$PYTHON_BIN")"

echo ""
echo "== 2/6: Creating virtual environment (.venv) =="
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "== 3/6: Installing dependencies (requirements-local.txt) =="
pip install --upgrade pip --quiet
pip install -r requirements-local.txt

echo ""
echo "== 4/6: Checking for Ollama =="
if command -v ollama >/dev/null 2>&1; then
  echo "Found: $(ollama --version 2>/dev/null || echo ollama)"
else
  echo "Ollama not found. It's used for local generation and is optional:"
  echo "  - Install it from https://ollama.com to answer questions with a local model, or"
  echo "  - Skip it and set OPENAI_API_KEY instead (config.yaml or your shell environment) to"
  echo "    use OpenAI, or skip both and search will still work (results just won't get a"
  echo "    synthesized answer from /v1/ask — /v1/find's file search works either way)."
fi

echo ""
echo "== 5/6: First-run setup (config + initial index) =="
PROFILE=local python scripts/first_run.py

echo ""
echo "== 6/6: Done =="
