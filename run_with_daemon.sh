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

# Heap sizing (2026-07-05, HUMAN-AUTHORIZED edit of this frozen file —
# Priority-1 postmortem): the hardcoded 512MB cap OOM-crash-looped prod
# every ~61s once the vessel archive outgrew what the boot folds could
# hold ("JavaScript heap out of memory" at ~509MB in Railway logs). Read
# the container's cgroup memory limit and give Node 60% of it, reserving
# headroom for the Python daemon (self-capped 1GB) + OS. Override with
# VOLTRADE_NODE_HEAP_MB; floor 512, cap 6144.
LIMIT_BYTES=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "")
if [[ "$LIMIT_BYTES" =~ ^[0-9]+$ ]] && [ "$LIMIT_BYTES" -lt 274877906944 ]; then
  LIMIT_MB=$((LIMIT_BYTES / 1048576))
else
  LIMIT_MB=2048   # unknown or unlimited cgroup: conservative default
fi
HEAP_MB=${VOLTRADE_NODE_HEAP_MB:-$((LIMIT_MB * 6 / 10))}
[ "$HEAP_MB" -lt 512 ] && HEAP_MB=512
[ "$HEAP_MB" -gt 6144 ] && HEAP_MB=6144
echo "[supervisor] cgroup limit ${LIMIT_MB}MB -> node --max-old-space-size=${HEAP_MB}"

NODE_CMD="node --max-old-space-size=$HEAP_MB dist/index.cjs"

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
