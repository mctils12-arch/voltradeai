/**
 * gemMethaneProximity.ts — joins GEM methane-plume detections
 * (server/gemMethane.ts) to the nearest GEM oil/gas-extraction or coal-mine
 * asset (server/gemMethaneAssets.ts) by great-circle distance. This is
 * GATE-2(a) of the "GEM METHANE-PLUME × EXTRACTION-REGISTRY PROXIMITY"
 * hypothesis (research/open_questions.md) — the join itself, so the map
 * can visually sanity-check clustering per that entry's stated next step.
 *
 * STILL RAW, NOT A SIGNAL: this reports which catalogued GEM asset is
 * geometrically nearest a plume detection, with the actual distance —
 * a factual proximity measurement, not a claim that the asset caused the
 * plume (flaring, pipeline leaks, and unrelated nearby infrastructure all
 * produce false-positive-looking proximity). GATE-2(b) — a repeat-
 * detection count/rate per asset — is `computeAssetPlumeStats()` below
 * (2026-07-20): still a descriptive stat, not a signal. Gate 2(c)-(d) — a
 * same-universe base rate and matching against operators' own disclosed
 * emissions — are NOT built here; see the open_questions.md entry for
 * what's still required before this clears gate 2 as a tradeable signal.
 *
 * MATCH_RADIUS_KM is a stated, tunable constant, not a tuned parameter —
 * chosen from GEM's own location-accuracy vocabulary (wellpad/mine
 * footprints run hundreds of meters to a couple km; "exact" vs "approximate"
 * location accuracy on both sides adds slop): a plume within 2km of an
 * asset is called a candidate match; farther than that, honestly unmatched
 * rather than guessed. AMBIGUOUS_MARGIN_KM flags plumes where a second
 * asset sits nearly as close as the nearest — those get ambiguousMatch:true
 * instead of a silently arbitrary pick between two similarly-plausible sites.
 */
import type { MethanePlume, GemProvenance } from "./gemMethane";
import { cachedGemMethane } from "./gemMethane";
import type { GemAsset, GemAssetsProvenance } from "./gemMethaneAssets";
import { cachedGemAssets } from "./gemMethaneAssets";

export const MATCH_RADIUS_KM = 2;
export const AMBIGUOUS_MARGIN_KM = 0.5;

const EARTH_RADIUS_KM = 6371;

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

/** Great-circle distance in km (haversine). */
export function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(a)));
}

const CELL_DEG = 0.5; // ~55km at the equator — comfortably wider than any realistic search radius

function cellKey(lat: number, lon: number): string {
  return `${Math.floor(lat / CELL_DEG)},${Math.floor(lon / CELL_DEG)}`;
}

/** Buckets assets into ~0.5°×0.5° cells so each plume only scans its own
 *  cell + the 8 neighbors instead of the full O(plumes × assets) product
 *  (3,474 × ~12,400 ≈ 43M comparisons brute-force — the grid cuts that to
 *  a small constant per plume). */
export function buildAssetGrid(assets: GemAsset[]): Map<string, GemAsset[]> {
  const grid = new Map<string, GemAsset[]>();
  for (const a of assets) {
    const key = cellKey(a.lat, a.lon);
    const bucket = grid.get(key);
    if (bucket) bucket.push(a);
    else grid.set(key, [a]);
  }
  return grid;
}

export interface NearestAsset {
  kind: GemAsset["kind"];
  id: string;
  name: string | null;
  operator: string | null;
  owner: string | null;
  parent: string | null;
  distanceKm: number;
}

export interface PlumeWithAsset extends MethanePlume {
  nearestAsset: NearestAsset | null;
  ambiguousMatch: boolean;
}

/** Finds the nearest and second-nearest asset to (lat, lon) within the grid's
 *  3×3 neighborhood. Returns nulls when the neighborhood has no assets at
 *  all (never expands further — a plume with nothing within ~55km has no
 *  honest match to report). */
function nearestTwo(
  lat: number, lon: number, grid: Map<string, GemAsset[]>,
): { nearest: { asset: GemAsset; distanceKm: number } | null; secondKm: number | null } {
  const [cLat, cLon] = [Math.floor(lat / CELL_DEG), Math.floor(lon / CELL_DEG)];
  let best: { asset: GemAsset; distanceKm: number } | null = null;
  let second: number | null = null;
  for (let dLat = -1; dLat <= 1; dLat++) {
    for (let dLon = -1; dLon <= 1; dLon++) {
      const bucket = grid.get(`${cLat + dLat},${cLon + dLon}`);
      if (!bucket) continue;
      for (const asset of bucket) {
        const d = haversineKm(lat, lon, asset.lat, asset.lon);
        if (!best || d < best.distanceKm) {
          second = best ? best.distanceKm : second;
          best = { asset, distanceKm: d };
        } else if (second === null || d < second) {
          second = d;
        }
      }
    }
  }
  return { nearest: best, secondKm: second };
}

/** Joins each plume to its nearest asset. Pure function — no I/O, fully
 *  unit-testable against small fixtures. */
export function joinPlumesToAssets(plumes: MethanePlume[], assets: GemAsset[]): PlumeWithAsset[] {
  const grid = buildAssetGrid(assets);
  return plumes.map((plume) => {
    if (plume.lat === null || plume.lon === null) {
      return { ...plume, nearestAsset: null, ambiguousMatch: false };
    }
    const { nearest, secondKm } = nearestTwo(plume.lat, plume.lon, grid);
    if (!nearest || nearest.distanceKm > MATCH_RADIUS_KM) {
      return { ...plume, nearestAsset: null, ambiguousMatch: false };
    }
    const ambiguousMatch = secondKm !== null && secondKm - nearest.distanceKm < AMBIGUOUS_MARGIN_KM;
    return {
      ...plume,
      nearestAsset: {
        kind: nearest.asset.kind,
        id: nearest.asset.id,
        name: nearest.asset.name,
        operator: nearest.asset.operator,
        owner: nearest.asset.owner,
        parent: nearest.asset.parent,
        distanceKm: nearest.distanceKm,
      },
      ambiguousMatch,
    };
  });
}

export interface AssetPlumeStat {
  assetId: string;
  kind: GemAsset["kind"];
  name: string | null;
  operator: string | null;
  owner: string | null;
  parent: string | null;
  /** Total unambiguous plume detections matched to this asset. */
  detectionCount: number;
  firstObservedAt: string | null;
  lastObservedAt: string | null;
  /** Days between the first and last DATED detection; null when fewer than
   *  two dated detections exist (a single sighting has no span to measure). */
  spanDays: number | null;
  /** Dated-detection count / spanDays × 365.25 — null whenever spanDays is
   *  null, or the two dated detections land on the same day (no honest
   *  rate from a zero-length span). One detection is noise, not a rate —
   *  gate-2(b) of the GEM METHANE-PLUME × EXTRACTION-REGISTRY PROXIMITY
   *  hypothesis (research/open_questions.md); this stat alone is NOT a
   *  signal (no base rate or disclosed-intensity comparison yet — see (c)/(d)). */
  ratePerYear: number | null;
  meanEmissionsKgHr: number | null;
}

const MS_PER_DAY = 86_400_000;

/** GEM's "Observation Date" field (server/gemMethane.ts's `observedAt`) is
 *  documented as ISO 8601 but ships a bare 2-digit UTC offset ("+00"
 *  instead of "+00:00"/"Z") — live-verified against the real archive,
 *  every one of 3,474 rows uses this exact shape. `new Date()` silently
 *  returns Invalid Date for it (no throw), which would silently zero out
 *  every spanDays/ratePerYear below without this normalization. Patches
 *  only that one known shape; anything still unparseable returns NaN so
 *  its caller treats it as no-date rather than miscounting. */
export function parseObservedAtMs(iso: string): number {
  const direct = Date.parse(iso);
  if (!Number.isNaN(direct)) return direct;
  return Date.parse(iso.replace(/([+-]\d{2})$/, "$1:00"));
}

/** Groups matched, unambiguous plumes by nearest asset and computes a
 *  repeat-detection count/rate per asset. Pure function — no I/O.
 *  Ambiguous matches are excluded entirely: a plume that could as
 *  plausibly belong to a different asset shouldn't inflate either
 *  candidate's count. Assets with zero unambiguous matches don't appear
 *  (nothing to report). Sorted by detectionCount, most-detected first —
 *  callers wanting a different order (rate, mean emissions) re-sort
 *  client-side; this function states one honest default, not a ranking
 *  claim. */
export function computeAssetPlumeStats(plumes: PlumeWithAsset[]): AssetPlumeStat[] {
  const groups = new Map<string, PlumeWithAsset[]>();
  for (const p of plumes) {
    if (!p.nearestAsset || p.ambiguousMatch) continue;
    const key = p.nearestAsset.id;
    const bucket = groups.get(key);
    if (bucket) bucket.push(p);
    else groups.set(key, [p]);
  }
  const out: AssetPlumeStat[] = [];
  groups.forEach((group, assetId) => {
    const asset = group[0].nearestAsset!;
    const dates = group.map((p) => p.observedAt).filter((d): d is string => d != null).sort();
    const firstObservedAt = dates[0] ?? null;
    const lastObservedAt = dates.length > 0 ? dates[dates.length - 1] : null;
    let spanDays: number | null = null;
    let ratePerYear: number | null = null;
    if (dates.length >= 2 && firstObservedAt && lastObservedAt) {
      const firstMs = parseObservedAtMs(firstObservedAt);
      const lastMs = parseObservedAtMs(lastObservedAt);
      if (!Number.isNaN(firstMs) && !Number.isNaN(lastMs)) {
        const span = (lastMs - firstMs) / MS_PER_DAY;
        if (span > 0) { spanDays = span; ratePerYear = (dates.length / span) * 365.25; }
      }
    }
    const emissions = group.map((p) => p.emissionsKgHr).filter((e): e is number => e != null);
    const meanEmissionsKgHr = emissions.length > 0 ? emissions.reduce((a, b) => a + b, 0) / emissions.length : null;
    out.push({
      assetId, kind: asset.kind, name: asset.name, operator: asset.operator,
      owner: asset.owner, parent: asset.parent, detectionCount: group.length,
      firstObservedAt, lastObservedAt, spanDays, ratePerYear, meanEmissionsKgHr,
    });
  });
  return out.sort((a, b) => b.detectionCount - a.detectionCount);
}

export interface MethaneProximityResult {
  plumes: PlumeWithAsset[];
  matchedCount: number;
  ambiguousCount: number;
  assetStats: AssetPlumeStat[];
  plumeProvenance: GemProvenance;
  assetProvenance: GemAssetsProvenance;
}

let cached: MethaneProximityResult | null | undefined;

/** Combines the cached plume + asset loaders and joins them, cached once
 *  per process (both inputs are static reference data — see gemMethane.ts
 *  and gemMethaneAssets.ts for the re-ingest cadence). Returns null when
 *  the plume file itself is unavailable (matches cachedGemMethane's
 *  degrade-to-null contract); a missing/partial asset registry still
 *  yields plumes, just with fewer matches. */
export function cachedGemMethaneProximity(): MethaneProximityResult | null {
  if (cached !== undefined) return cached;
  const plumeHit = cachedGemMethane();
  if (!plumeHit) {
    cached = null;
    return cached;
  }
  const assetHit = cachedGemAssets();
  const joined = joinPlumesToAssets(plumeHit.plumes, assetHit.assets);
  cached = {
    plumes: joined,
    matchedCount: joined.filter((p) => p.nearestAsset !== null).length,
    ambiguousCount: joined.filter((p) => p.ambiguousMatch).length,
    assetStats: computeAssetPlumeStats(joined),
    plumeProvenance: plumeHit.provenance,
    assetProvenance: assetHit.provenance,
  };
  return cached;
}

/** Test-only: clears the module cache. */
export function _resetGemMethaneProximityCacheForTests(): void {
  cached = undefined;
}
