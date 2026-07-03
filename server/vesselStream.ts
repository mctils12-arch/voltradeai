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
