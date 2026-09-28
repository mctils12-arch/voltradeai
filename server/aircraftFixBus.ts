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
  /** the upstream's own generation clock when it sent one (readsb `now`, ms) */
  upstreamNowMs?: number | null;
  /** set when the batch is one point-query disc (viewport credit for the sweep) */
  disc?: { lat: number; lon: number; radiusNm: number } | null;
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
