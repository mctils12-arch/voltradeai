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
/** Snapshot cache TTL — polls land every ~20s, so one snapshot serves a
 *  poll cycle across all tabs without going stale past one interval. */
export const VESSEL_SNAPSHOT_TTL_MS = 15_000;
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
