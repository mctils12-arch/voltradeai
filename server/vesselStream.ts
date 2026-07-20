// Single source of truth for whether the AIS vessel layer is enabled, and
// for the boot-time connect decision. Extracted out of routes.ts so both are
// unit-testable without dragging in registerRoutes' heavy deps (auth.ts opens
// a real sqlite handle at import time).
//
// KNOWN BROKEN #9: the aisstream websocket used to connect lazily, only when
// the first /api/data/vessels request called ensureVesselStream(). Every
// deploy therefore left the vessels layer (and its archive recording) cold
// until a visitor opened the map. bootVesselStream() closes that gap by
// invoking the same connect function once at server startup.

export function vesselStreamEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.AISSTREAM_KEY);
}

export function bootVesselStream(env: NodeJS.ProcessEnv, connect: () => void): void {
  if (vesselStreamEnabled(env)) connect();
}

// GAP THIS CLOSES (found 2026-07-20, live): bootVesselStream fixed the
// COLD-START case (KNOWN BROKEN #9) — connect once at boot instead of
// waiting for the first request. But after boot, ensureVesselStream() was
// ONLY ever called again by an incoming /api/data/vessels request. If the
// upstream websocket drops (network blip, aisstream-side disconnect) and
// the map goes untouched for a while (or a caller only ever hits the route
// while the socket looks CONNECTING/OPEN, so no reconnect fires), the feed
// — and everything downstream of it, including the 60s archive tick, which
// itself never calls ensureVesselStream() — can stay silently dead
// indefinitely. Live-verified 2026-07-20: archive's last vessels file was
// 2026-07-19-23:00Z (~14h stale), and repeated /api/data/vessels requests
// over several minutes did not recover it. ensureVesselStream() is already
// idempotent (no-ops when OPEN/CONNECTING) — this just gives it a periodic
// trigger independent of site traffic, same fix shape as bootVesselStream,
// injectable timer so it's unit-testable without a real setInterval/socket.
export function scheduleVesselHeartbeat(
  env: NodeJS.ProcessEnv,
  connect: () => void,
  intervalMs: number,
  setIntervalFn: (fn: () => void, ms: number) => { unref?: () => void },
): void {
  if (!vesselStreamEnabled(env)) return;
  const handle = setIntervalFn(connect, intervalMs);
  handle?.unref?.();
}
