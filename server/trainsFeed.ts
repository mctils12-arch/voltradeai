// trainsFeed.ts — pure mapping + source metadata for the live trains layer
// (RAW). No fetch, no db, no env: routes.ts owns transport (shared-upstream
// cache/backoff), this module owns the per-source JSON -> unified-train
// mapping so it is unit-testable with real sample payloads (same pattern as
// vesselStream.ts / providerCompliance.ts).
//
// LAUNCH SOURCES (licensing checked first, per the standing rule):
//  - Finland  · Digitraffic (Fintraffic) — CC BY 4.0, no key, plain JSON.
//    https://rata.digitraffic.fi/api/v1/train-locations/latest
//  - Norway   · Entur vehicles API — open (NLOD), free, requires only an
//    ET-Client-Name header. GraphQL, mode:RAIL.
// US NOTE (logged so no session chases it): freight rail positions are
// PROPRIETARY (Class I railroads sell them; no free source exists). Amtrak
// passenger positions have no clean official free JSON — candidate for a
// later source evaluation, not launch.

export interface UnifiedTrain {
  id: string;            // country-prefixed stable id
  country: string;       // "FI" | "NO" (coverage tag)
  lat: number;
  lon: number;
  speed_kmh: number | null;
  bearing: number | null;
  label: string | null;  // train number / line ref
  ts: number | null;     // unix seconds of the position fix
}

export const TRAIN_SOURCES = [
  { key: "digitraffic", country: "FI", label: "Finland — Digitraffic (CC BY 4.0)" },
  { key: "entur", country: "NO", label: "Norway — Entur (NLOD)" },
] as const;

/** Digitraffic /train-locations/latest: [{trainNumber, departureDate,
 *  timestamp, location:{coordinates:[lon,lat]}, speed(km/h)}] */
export function mapDigitraffic(raw: any): UnifiedTrain[] {
  if (!Array.isArray(raw)) return [];
  const out: UnifiedTrain[] = [];
  for (const r of raw) {
    const co = r?.location?.coordinates;
    if (!Array.isArray(co) || co.length < 2 || r?.trainNumber == null) continue;
    out.push({
      id: `FI-${r.trainNumber}-${r.departureDate || ""}`,
      country: "FI",
      lat: co[1], lon: co[0],
      speed_kmh: typeof r.speed === "number" ? r.speed : null,
      bearing: null, // Digitraffic publishes no heading
      label: `Train ${r.trainNumber}`,
      ts: r.timestamp ? Math.floor(Date.parse(r.timestamp) / 1000) : null,
    });
  }
  return out;
}

/** Entur vehicles GraphQL: {data:{vehicles:[{vehicleId, lastUpdated,
 *  line:{lineRef}, location:{latitude,longitude}, speed, bearing}]}} */
export function mapEntur(raw: any): UnifiedTrain[] {
  const vehicles = raw?.data?.vehicles;
  if (!Array.isArray(vehicles)) return [];
  const out: UnifiedTrain[] = [];
  for (const v of vehicles) {
    const loc = v?.location;
    if (!v?.vehicleId || loc?.latitude == null || loc?.longitude == null) continue;
    out.push({
      id: `NO-${v.vehicleId}`,
      country: "NO",
      lat: loc.latitude, lon: loc.longitude,
      speed_kmh: typeof v.speed === "number" ? Math.round(v.speed * 3.6) : null, // m/s -> km/h
      bearing: typeof v.bearing === "number" ? v.bearing : null,
      label: v.line?.lineRef ? String(v.line.lineRef).split(":").pop() || null : null,
      ts: v.lastUpdated ? Math.floor(Date.parse(v.lastUpdated) / 1000) : null,
    });
  }
  return out;
}

export const ENTUR_VEHICLES_QUERY =
  "{vehicles(mode:RAIL){vehicleId lastUpdated line{lineRef} location{latitude longitude} speed bearing}}";

/** VIEWER-INDEPENDENT CAPTURE (2026-09-30, human: "when on the bot page
 *  don't run the data page, but still capture the data"). The trains
 *  archive was fed ONLY by /api/data/trains requests — with nobody on the
 *  map, rail capture (and the feed's dead-air liveness) stopped. A boot-
 *  started background tick now refreshes the shared cache when no viewer
 *  has in the last TRAINS_BG_TICK_MS, so upstream load never exceeds what
 *  one viewer already caused (the route's 30 s cache). */
export const TRAINS_BG_TICK_MS = 60_000;
export function trainsBackgroundDue(cacheAt: number | null, inflight: boolean, now: number): boolean {
  if (inflight) return false;
  return cacheAt == null || now - cacheAt >= TRAINS_BG_TICK_MS;
}
