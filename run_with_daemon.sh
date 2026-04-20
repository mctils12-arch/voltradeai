#!/bin/bash
# VolTradeAI — Daemon Supervisor
# ================================
# Starts voltrade_daemon.py alongside Node.js with auto-restart on crash.
#
# MATCHES RAILWAY CONFIG: uses the exact start command from railway.json.
# To activate, change Railway startCommand to "./run_with_daemon.sh".
#
# Disable via env var VOLTRADE_DAEMON_ENABLED=false (no code revert needed).

set -e

NODE_CMD="node --max-old-space-size=512 dist/index.cjs"

if [ "$VOLTRADE_DAEMON_ENABLED" = "false" ]; then
  echo "[supervisor] VOLTRADE_DAEMON_ENABLED=false — running Node only"
  exec $NODE_CMD
fi

if [ ! -f voltrade_daemon.py ]; then
  echo "[supervisor] voltrade_daemon.py not found — running Node only"
  exec $NODE_CMD
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[supervisor] python3 not found — running Node only"
  exec $NODE_CMD
fi

# Background auto-restart loop for the daemon
(
  while true; do
    echo "[supervisor] Starting voltrade_daemon.py"
    python3 voltrade_daemon.py
    echo "[supervisor] daemon exited ($?) — restart in 2s"
    sleep 2
  done
) &
DAEMON_PID=$!

trap 'kill $DAEMON_PID 2>/dev/null; exit 0' SIGTERM SIGINT

echo "[supervisor] Starting: $NODE_CMD"
exec $NODE_CMD
