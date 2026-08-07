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

/** Health of the AIS websocket feed (repair 2026-08-06, full-code-review
 *  findings, adversarially verified): (1) the socket never redialed after a
 *  close without visitor traffic — permanent, unrefillable archive gaps;
 *  (2) a half-open socket keeps readyState 1 forever with no keepalive, so
 *  connection state alone cannot detect a dead feed. aisstream's global
 *  subscription is a firehose — minutes of total silence IS an outage.
 *  Pure function: the watchdog, the layers status override, and the route's
 *  honesty fields all consume this one verdict. */
export interface VesselFeedHealth {
  connected: boolean;
  /** ms since the last frame, null before the first frame ever */
  silentMs: number | null;
  /** connected but silent past the threshold — terminate + redial */
  zombie: boolean;
  /** the feed cannot be called live: disconnected, or a zombie */
  down: boolean;
}

export const VESSEL_SILENT_THRESHOLD_MS = 3 * 60_000;

export function vesselFeedHealth(
  readyState: number | null,
  lastMsgAtMs: number,
  nowMs: number,
  silentThresholdMs: number = VESSEL_SILENT_THRESHOLD_MS,
): VesselFeedHealth {
  const connected = readyState === 1;
  const silentMs = lastMsgAtMs > 0 ? Math.max(0, nowMs - lastMsgAtMs) : null;
  const zombie = connected && silentMs !== null && silentMs > silentThresholdMs;
  // never-received-a-frame while "connected" counts as down once the
  // threshold has passed since we cannot distinguish it from a dead pipe —
  // but with no frame ever and no timestamp, only disconnection is provable
  const down = !connected || zombie;
  return { connected, silentMs, zombie, down };
}
