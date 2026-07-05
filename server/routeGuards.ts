/**
 * routeGuards.ts — hard-deadline + self-expiring in-flight slots for
 * routes that fetch upstreams in-request.
 *
 * [REPAIR 2026-07-05] Production outage anatomy: /api/data/trains shares
 * one in-flight promise across concurrent requests (rate-limit-friendly
 * dedup). One fetch got stuck past its per-source AbortSignal timeouts
 * (upstream/undici pathology — the archive shows the feed healthy at
 * 05:00 UTC, dead after), and because `.finally()` only clears the slot
 * when the promise SETTLES, every request forever after awaited the same
 * dead promise: a permanent outage from one bad fetch, surviving upstream
 * recovery, fixed only by a redeploy. /api/data/aircraft carried the
 * identical latent pattern per bbox key.
 *
 * Two defenses, both required:
 *  - raceDeadline: no request ever awaits an in-flight fetch longer than
 *    the deadline — it falls to the route's stale-beats-spinner path.
 *  - expired slots: a slot older than maxAge is abandoned (a fresh fetch
 *    starts); if the orphan ever settles, its identity-guarded cleanup
 *    no-ops instead of clobbering the replacement.
 *
 * Pure module (no imports) — unit-testable without dragging in auth/sqlite
 * (the node test-runner hang lesson).
 */

/** Route-level hard deadline: no caller waits longer than the route
 *  budget. Never on the fetch promise itself (upstream timeouts own
 *  that) — this bounds the WAIT, letting a slow-but-alive fetch finish
 *  in the background and land in the cache for the next request. */
export const ROUTE_DEADLINE_MS = 15_000;

/** An in-flight slot older than this is presumed stuck and abandoned —
 *  generous vs the worst legitimate chain (aircraft: 3 providers x 12s
 *  serial = 36s) so live-but-slow work still completes into the cache. */
export const INFLIGHT_MAX_AGE_MS = 45_000;

export interface InflightSlot<T> {
  p: Promise<T>;
  at: number;
}

export function raceDeadline<T>(p: Promise<T>, ms: number, label = "route"): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    // no unref(): the timer must survive an idle event loop (test runner;
    // graceful shutdown mid-request) — it's cleared the moment p settles,
    // so at worst one stuck fetch holds shutdown for the deadline window.
    const t = setTimeout(() => reject(new Error(`${label}: upstream still pending after ${ms}ms`)), ms);
    p.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); },
    );
  });
}

export function slotExpired<T>(slot: InflightSlot<T> | null | undefined, nowMs: number, maxAgeMs = INFLIGHT_MAX_AGE_MS): boolean {
  return !slot || nowMs - slot.at > maxAgeMs;
}

/** Makes a slot whose cleanup is IDENTITY-GUARDED: when the promise
 *  settles it clears itself via `clear(slot)` only if the holder still
 *  points at THIS slot — an abandoned orphan settling late can never
 *  clobber its replacement. Rejections are absorbed here (the route's
 *  await sees them through raceDeadline; the slot chain itself must
 *  never produce an unhandled rejection). */
export function makeSlot<T>(
  make: () => Promise<T>,
  nowMs: number,
  clear: (self: InflightSlot<T>) => void,
): InflightSlot<T> {
  const slot: InflightSlot<T> = { p: make(), at: nowMs };
  slot.p.then(() => clear(slot), () => clear(slot));
  slot.p.catch(() => {});
  return slot;
}
