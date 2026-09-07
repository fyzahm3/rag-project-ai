#!/usr/bin/env bash
# Installs a launchd LaunchAgent that runs the local search daemon (app/tray.py,
# PROFILE=local) at every login. OFF by default — this script does nothing until
# you run it yourself. See the bottom of its output for how to undo it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.$(whoami).ragsearch"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
  PYTHON_BIN="$VENV_PYTHON"
else
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "$HOME/Library/LaunchAgents"

# Re-running this script should replace, not duplicate, a previous install.
if launchctl list "$LABEL" >/dev/null 2>&1; then
  echo "Unloading existing agent before reinstalling..."
  launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${REPO_ROOT}/scripts/run_local_daemon.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PROFILE</key>
        <string>local</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>${REPO_ROOT}/ragsearch.log</string>
    <key>StandardErrorPath</key>
    <string>${REPO_ROOT}/ragsearch.err.log</string>
</dict>
</plist>
PLIST

launchctl load -w "$PLIST_PATH"

echo "Installed and loaded launchd agent:"
echo "  $PLIST_PATH"
echo ""
echo "It will run this at every login:"
echo "  PROFILE=local $PYTHON_BIN $REPO_ROOT/scripts/run_local_daemon.py"
echo ""
echo "Logs:"
echo "  $REPO_ROOT/ragsearch.log"
echo "  $REPO_ROOT/ragsearch.err.log"
echo ""
echo "To undo:"
echo "  launchctl unload '$PLIST_PATH'"
echo "  rm '$PLIST_PATH'"
