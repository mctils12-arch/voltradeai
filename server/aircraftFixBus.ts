// aircraftFixBus.ts — in-process fan-out of every live aircraft batch any
// upstream path fetched (FLIGHT PROGRAM B1, 2026-09-28).
//
// The viewport tiling chain already fetches, every ~30s, exactly the discs
// viewers are looking at. The global snapshot (globalSnapshot.ts) REUSES
// those results instead of refetching them: fetchDiscs publishes each
// successful disc here, the snapshot subscribes. Same for the global-scope
// archiver (mil/ladd/pia). Publishers never depend on a subscriber existing,
// and a throwing subscriber can never break a publisher's fetch path.

import type { AircraftPoint } from "./datacoreArchive";

export interface FixBatch {
  /** upstream provider key (adsblol | airplaneslive | adsbfi | opensky) —
   *  provenance travels with the batch, never inferred later */
  provider: string;
  /** which path fetched it (viewport | sweep-disc | sweep-type | scopes | opensky) */
  origin: string;
  /** mapped rows (mapPointAircraft / OpenSky normalizer output); seen_at_ms
   *  = the fix's own time when the upstream sends one per row (OpenSky) */
  aircraft: Array<AircraftPoint & { seen_at_ms?: number | null }>;
  /** when WE received it (ms) */
  fetchedAt: number;
  /** the upstream's own generation clock when it sent one, in MILLISECONDS.
   *  Publishers pass the provider's `now` through providerClockMs(): adsb.lol
   *  and airplanes.live send ms, adsb.fi sends seconds. */
  upstreamNowMs?: number | null;
  /** set when the batch is one point-query disc (viewport credit for the sweep) */
  disc?: { lat: number; lon: number; radiusNm: number } | null;
}

/** A provider `now` below this is SECONDS, at or above it MILLISECONDS:
 *  1e11 ms is March 1973 and 1e11 s is the year 5138, so no real clock sits
 *  on the wrong side of it. */
export const PROVIDER_CLOCK_SECONDS_BELOW = 1e11;

/**
 * A provider's own response clock (`now`) in milliseconds.
 *
 * The three community providers share the readsb row shape but NOT the unit
 * of `now`: adsb.lol and airplanes.live send milliseconds, adsb.fi sends
 * seconds with a fraction (1794880068.001, read live 2026-11-16). A fix's
 * time is this clock minus the row's own position age, so the seconds form
 * read as milliseconds dates every adsb.fi fix in January 1970 — and the
 * worldwide snapshot then refuses each one as far too old. That happened to
 * every adsb.fi answer of the viewport chain and of the coverage gap-fill
 * until 2026-11-16.
 *
 * Returns null when `v` is not a positive finite number (the caller then
 * falls back to its own receive time). `ourNowMs`, when given, is an upper
 * bound: a provider clock ahead of ours never dates a fix in the future.
 */
export function providerClockMs(v: unknown, ourNowMs?: number): number | null {
  if (typeof v !== "number" || !Number.isFinite(v) || v <= 0) return null;
  const ms = v < PROVIDER_CLOCK_SECONDS_BELOW ? v * 1000 : v;
  return ourNowMs != null && Number.isFinite(ourNowMs) ? Math.min(ourNowMs, ms) : ms;
}

export type FixSubscriber = (b: FixBatch) => void;

const subscribers = new Set<FixSubscriber>();

export function subscribeFixes(fn: FixSubscriber): () => void {
  subscribers.add(fn);
  return () => { subscribers.delete(fn); };
}

export function publishFixes(b: FixBatch): void {
  if (!subscribers.size) return;
  for (const fn of Array.from(subscribers)) {
    try { fn(b); } catch (e) {
      console.error("[aircraft-fix-bus] subscriber:", e instanceof Error ? e.message : String(e));
    }
  }
}

/** test hook */
export function subscriberCount(): number {
  return subscribers.size;
}
