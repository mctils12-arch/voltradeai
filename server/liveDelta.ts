// Live-endpoint delta helpers — SCALE S1(b) (research/scale_program.md:
// "VESSELS DELTA: handler emits no `time` and ignores `since` → full
// re-ship every poll; give it the aircraft treatment").
//
// Pure functions so the snapshot/delta logic is testable without express
// or the AIS websocket. The vessels route keeps a short-TTL snapshot cache
// keyed by the expanded bbox; a client re-polling with ?since= equal to the
// snapshot's time gets a tiny {unchanged} answer instead of the full list
// (the datamap client already speaks this protocol for aircraft — same
// sinceRef machinery, zero client changes needed).

export interface LiveVesselPos {
  lat: number;
  lon: number;
  sog: number | null;
  cog: number | null;
  name: string;
  at: number;
}

export interface LiveVesselStatic {
  shiptype: number | null;
  destination: string | null;
  name: string | null;
}

export interface VesselBbox {
  lamin: number;
  lamax: number;
  lomin: number;
  lomax: number;
}

/** Positions older than this are dropped from snapshots (matches the
 *  handler's long-standing 20-minute freshness rule). */
export const VESSEL_FRESH_MS = 20 * 60_000;
/** Snapshot cache TTL. MUST exceed the client's vessels poll interval
 *  (20s, datamap.tsx wireLivePoints) — the `unchanged` short-circuit can
 *  only fire when a poll lands on a still-live cache entry with the same
 *  `time`. SCALE S-A2 (research/scale_program.md): this was 15_000, i.e.
 *  SHORTER than the 20s poll, so every single poll missed the cache and
 *  rebuilt a fresh snapshot with a new `time` — `unchanged` was dead code,
 *  and the full ~2MB raw payload re-shipped on every poll. Raised to 30s
 *  (2x poll interval, matching AIRCRAFT_TTL_MS's ratio to its 15s poll in
 *  server/routes.ts) for headroom against timer jitter/backgrounded tabs. */
export const VESSEL_SNAPSHOT_TTL_MS = 30_000;

/** Whether a cached snapshot entry is stale and must be rebuilt. Pure/
 *  exported so the TTL-vs-poll-interval invariant is unit-testable
 *  without express (S-A2: the bug was invisible without this). */
export function shouldRebuildSnapshot(
  hit: { at: number } | undefined,
  now: number,
  ttlMs: number = VESSEL_SNAPSHOT_TTL_MS,
): boolean {
  return !hit || now - hit.at >= ttlMs;
}
/** Max vessels per payload (raised 5000→15000 in the WebGL-symbol era —
 *  the cap discloses itself via coverage_note when it bites). */
export const VESSEL_CAP = 15000;

/** Expand a requested bbox OUTWARD to 0.1° so nearby viewports share one
 *  snapshot-cache entry. Outward (floor/ceil), never nearest-rounding:
 *  a cache key must never crop vessels the viewport actually asked for. */
export function expandBbox1dp(lamin: number, lamax: number, lomin: number, lomax: number): VesselBbox {
  const f = (v: number) => Math.floor(v * 10) / 10;
  const c = (v: number) => Math.ceil(v * 10) / 10;
  return {
    lamin: Math.max(-85, f(lamin)),
    lamax: Math.min(85, c(lamax)),
    lomin: Math.max(-180, f(lomin)),
    lomax: Math.min(180, c(lomax)),
  };
}

export interface VesselSnapshot {
  time: string;
  count: number;
  total_in_view: number;
  coverage?: "partial";
  coverage_note?: string;
  vessels: Array<{
    mmsi: string;
    name: string | null;
    lat: number;
    lon: number;
    sog: number | null;
    cog: number | null;
    shiptype: number | null;
    destination: string | null;
  }>;
}

/** The vessels payload body — extracted verbatim from the route handler
 *  (bbox + freshness filter, statics join, cap with honest disclosure),
 *  plus the `time` stamp that makes the delta protocol possible. */
export function buildVesselSnapshot(
  positions: Map<string, LiveVesselPos>,
  statics: Map<string, LiveVesselStatic>,
  bbox: VesselBbox,
  now: number,
  cap: number = VESSEL_CAP,
): VesselSnapshot {
  const cutoff = now - VESSEL_FRESH_MS;
  const vessels: VesselSnapshot["vessels"] = [];
  let totalInView = 0;
  positions.forEach((v, mmsi) => {
    if (v.at < cutoff) return;
    if (v.lat < bbox.lamin || v.lat > bbox.lamax || v.lon < bbox.lomin || v.lon > bbox.lomax) return;
    totalInView += 1;
    if (vessels.length >= cap) return;
    const st = statics.get(mmsi);
    vessels.push({
      mmsi, name: st?.name || v.name, lat: v.lat, lon: v.lon,
      sog: v.sog, cog: v.cog,
      shiptype: st?.shiptype ?? null,
      destination: st?.destination ?? null,
    });
  });
  const capped = totalInView > vessels.length;
  return {
    time: String(now),
    count: vessels.length,
    total_in_view: totalInView,
    ...(capped ? {
      coverage: "partial" as const,
      coverage_note: `showing ${vessels.length.toLocaleString()} of ${totalInView.toLocaleString()} vessels in view — zoom in to see the rest`,
    } : {}),
    vessels,
  };
}

/** True when the client's ?since= matches the payload's time — answer
 *  {unchanged} instead of re-shipping. A payload without a time can never
 *  be "unchanged" (fail open to the full payload, never to silence). */
export function sinceUnchanged(data: { time?: unknown }, since: unknown): boolean {
  if (data.time == null) return false;
  const s = String(since ?? "");
  return s !== "" && s === String(data.time);
}
