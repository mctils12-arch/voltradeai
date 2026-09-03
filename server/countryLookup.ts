/**
 * countryLookup.ts — point -> country reverse lookup over the Natural Earth
 * 1:110m admin0 boundaries already vendored at
 * datacore/boundaries/ne_110m_admin0.json (public domain; already served
 * raw as the "boundaries" map layer, server/routes.ts:1759).
 *
 * Built as the regionOf(lat, lon) join server/gasFlareCandidates.ts's own
 * header names as a GATE 1 prerequisite ("needs a country-boundary join —
 * candidatesByRegion() below takes that join as an injected function so a
 * future session can supply it without touching this module"). Generic
 * beyond that one caller — any future country-aggregate gate-1/gate-2 check
 * can reuse this without its own boundary-parsing code.
 *
 * Point-in-polygon reuses plantsUnderAlerts.ts's even-odd ray-cast
 * (pointInAnyRing/ringsBbox) — the same algorithm already established and
 * tested in this codebase, not a second implementation of the same math.
 *
 * SIMPLIFICATION (documented, not hidden): OUTER rings only — a hole in an
 * admin0 polygon (e.g. an enclosed enclave country) is not subtracted, so a
 * point that falls inside an enclave's outer ring is attributed to BOTH the
 * enclave and its surrounding country by a naive "first match" scan. This
 * module checks enclave-shaped small countries first by AREA (smallest
 * bbox area wins on a tie) specifically to resolve that case correctly at
 * 1:110m resolution — see `countryOf`'s doc comment. Acceptable for a
 * country-level rank-correlation gate-1 test; NOT precise enough for a
 * parcel/facility-level claim.
 */
import datacoreBoundaries from "../datacore/boundaries/ne_110m_admin0.json";
import { pointInAnyRing, ringsBbox, type Ring } from "./plantsUnderAlerts";

interface CountryEntry {
  name: string;
  iso3: string;
  rings: Ring[];
  bbox: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  bboxArea: number;
}

type GeoJSONPolygon = { type: "Polygon"; coordinates: number[][][] };
type GeoJSONMultiPolygon = { type: "MultiPolygon"; coordinates: number[][][][] };

function outerRingsOf(geometry: GeoJSONPolygon | GeoJSONMultiPolygon): Ring[] {
  if (geometry.type === "Polygon") {
    const outer = geometry.coordinates[0];
    return outer ? [outer as Ring] : [];
  }
  // MultiPolygon: outer ring (index 0) of each constituent polygon.
  return geometry.coordinates
    .map((poly) => poly[0])
    .filter((r): r is number[][] => !!r) as Ring[];
}

let COUNTRIES: CountryEntry[] | null = null;

function loadCountries(): CountryEntry[] {
  if (COUNTRIES) return COUNTRIES;
  const fc = datacoreBoundaries as unknown as {
    features: Array<{ properties: { name: string; iso3: string }; geometry: GeoJSONPolygon | GeoJSONMultiPolygon }>;
  };
  const out: CountryEntry[] = [];
  for (const f of fc.features) {
    const rings = outerRingsOf(f.geometry);
    if (!rings.length) continue;
    const bbox = ringsBbox(rings);
    if (!bbox) continue;
    const [minLon, minLat, maxLon, maxLat] = bbox;
    out.push({
      name: f.properties.name,
      iso3: f.properties.iso3,
      rings,
      bbox,
      bboxArea: Math.max(0, maxLon - minLon) * Math.max(0, maxLat - minLat),
    });
  }
  // Smallest-bbox-area first so an enclave (e.g. Lesotho inside South
  // Africa) is tested — and matched — before its larger surrounding
  // country, resolving the hole-less-outer-ring ambiguity noted above.
  out.sort((a, b) => a.bboxArea - b.bboxArea);
  COUNTRIES = out;
  return out;
}

/** Reverse-geocodes (lat, lon) to an ISO3 country code, or null if the point
 *  falls in no admin0 polygon at this resolution (open ocean, Antarctica —
 *  not covered by this 177-feature dataset). Pure aside from the one-time
 *  lazy load of the vendored boundary file. */
export function countryOf(lat: number, lon: number): string | null {
  const countries = loadCountries();
  for (const c of countries) {
    const [minLon, minLat, maxLon, maxLat] = c.bbox;
    if (lon < minLon || lon > maxLon || lat < minLat || lat > maxLat) continue;
    if (pointInAnyRing(lon, lat, c.rings)) return c.iso3;
  }
  return null;
}

/** The country's own bounding box as "lamin,lamax,lomin,lomax" — the exact
 *  format server/bot.ts's `/api/diag/archive?bbox=...` probe expects, so a
 *  gate-1 script can scope an archive query to one country without
 *  re-deriving the box by hand. Null if the iso3 code isn't in the dataset. */
export function countryBboxParam(iso3: string): string | null {
  const c = loadCountries().find((x) => x.iso3 === iso3.toUpperCase());
  if (!c) return null;
  const [minLon, minLat, maxLon, maxLat] = c.bbox;
  return `${minLat},${maxLat},${minLon},${maxLon}`;
}

/** Country display name for an iso3 code, or null if unknown. */
export function countryName(iso3: string): string | null {
  return loadCountries().find((x) => x.iso3 === iso3.toUpperCase())?.name ?? null;
}

/** Antimeridian-aware version of countryBboxParam(): a country whose admin0
 *  geometry crosses the dateline (e.g. Russia's Chukotka exclave, part of the
 *  110m dataset's own separate MultiPolygon rings at lon -180..-169.9) has a
 *  naive single min/max-lon bbox spanning almost the ENTIRE GLOBE's longitude
 *  range, not just that country — confirmed live 2026-09-01
 *  (research/experiments.md, GAS FLARE CANDIDATES gate-1 run) when that bbox
 *  truncated at an archive probe's per-day row cap on ordinary global
 *  wildfire noise before enough real rows for the country came through.
 *  Splits the country's rings into a "far west" exclave cluster (any ring
 *  whose own lon span sits entirely under -90 — the dateline-wrapped part)
 *  and everything else, and returns ONE bbox per cluster instead of one bbox
 *  spanning both. Every non-crossing country returns the same single bbox
 *  countryBboxParam() would. Null if the iso3 code isn't in the dataset. */
export function countryBboxParams(iso3: string): string[] | null {
  const c = loadCountries().find((x) => x.iso3 === iso3.toUpperCase());
  if (!c) return null;
  const [minLon, , maxLon] = c.bbox;
  if (maxLon - minLon <= 180) return [countryBboxParam(iso3)!];
  const exclave: Ring[] = [];
  const main: Ring[] = [];
  for (const ring of c.rings) {
    const lons = ring.map((pt) => pt[0]);
    (Math.max(...lons) < -90 ? exclave : main).push(ring);
  }
  const params: string[] = [];
  for (const rings of [main, exclave]) {
    if (!rings.length) continue;
    const bbox = ringsBbox(rings);
    if (!bbox) continue;
    const [lo, la, hi, ha] = bbox;
    params.push(`${la},${ha},${lo},${hi}`);
  }
  return params.length ? params : [countryBboxParam(iso3)!];
}
