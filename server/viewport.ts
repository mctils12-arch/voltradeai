/**
 * viewport.ts — SCALE PROGRAM S1(a): VIEWPORT-BOUNDED SERVING.
 *
 * A reusable, pure, injectable serve-time filter for the shared datacore
 * serving path. STORAGE fidelity and RENDER fidelity are separate problems
 * (research/scale_program.md): the archive keeps everything; a data request
 * may carry the map's current bounding box and the server returns only the
 * features inside that view. Adding more data OUTSIDE the viewport then costs
 * the rendered payload nothing ("off-screen = free").
 *
 * This module is a pure geometry filter — no network, no clock, no archive
 * access. It never thins storage; callers apply it at SERVE time only. When
 * no bbox is supplied the caller returns its full set unchanged (backward
 * compatible), so other layers and the no-bbox default are untouched.
 *
 * A `bbox` string is "minLon,minLat,maxLon,maxLat" (GeoJSON/Leaflet bbox
 * order — lon first). parseBbox validates ranges and rejects garbage without
 * throwing; inBbox is boundary-inclusive and honours antimeridian wrap.
 */

export interface Bbox {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
}

/**
 * Parse a "minLon,minLat,maxLon,maxLat" bbox string. Returns null (never
 * throws) on anything malformed: wrong arity, non-finite numbers, out-of-range
 * coordinates, or a degenerate (zero-height / zero-width) box.
 *
 * Latitude MUST ascend (minLat < maxLat) — there is no antimeridian analog for
 * latitude. Longitude MAY descend (minLon > maxLon): that encodes a box that
 * wraps across the antimeridian (e.g. 170 .. -170), which inBbox handles.
 * A zero-width longitude span (minLon === maxLon) is rejected as degenerate.
 */
export function parseBbox(str: unknown): Bbox | null {
  if (typeof str !== "string") return null;
  const parts = str.split(",");
  if (parts.length !== 4) return null;
  const nums = parts.map((p) => Number(p.trim()));
  if (nums.some((n) => !Number.isFinite(n))) return null;
  const [minLon, minLat, maxLon, maxLat] = nums;
  if (minLat < -90 || minLat > 90 || maxLat < -90 || maxLat > 90) return null;
  if (minLon < -180 || minLon > 180 || maxLon < -180 || maxLon > 180) return null;
  if (minLat >= maxLat) return null; // latitude must ascend; no wrap for lat
  if (minLon === maxLon) return null; // zero-width box is garbage
  return { minLon, minLat, maxLon, maxLat };
}

/**
 * Is the point (lon, lat) inside the bbox? Boundary-INCLUSIVE on all four
 * edges. Non-finite coordinates are never inside. When minLon > maxLon the box
 * wraps across the antimeridian and a point matches if it is on either side of
 * the +/-180 seam.
 */
export function inBbox(lon: number, lat: number, b: Bbox): boolean {
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return false;
  if (lat < b.minLat || lat > b.maxLat) return false;
  if (b.minLon <= b.maxLon) return lon >= b.minLon && lon <= b.maxLon;
  // antimeridian wrap: e.g. minLon=170, maxLon=-170 covers 170..180 and -180..-170
  return lon >= b.minLon || lon <= b.maxLon;
}

/**
 * Return only the features whose (lon, lat) — extracted by `getLatLon`, which
 * returns [lon, lat] or null to skip — fall inside the bbox. Single-pass O(n):
 * `getLatLon` is invoked exactly once per feature and no feature is compared
 * against any other, so cost is linear in the input regardless of how many
 * survive. This is what keeps the rendered payload flat as total data grows.
 */
export function filterByViewport<T>(
  features: readonly T[],
  b: Bbox,
  getLatLon: (f: T) => [number, number] | null | undefined
): T[] {
  const out: T[] = [];
  for (const f of features) {
    const ll = getLatLon(f);
    if (!ll) continue;
    if (inBbox(ll[0], ll[1], b)) out.push(f);
  }
  return out;
}

/**
 * Serve-time convenience for a layer route: parse `bboxStr`, and if it is a
 * valid bbox AND `payload[arrayKey]` is an array, return a copy with that array
 * viewport-filtered and the count fields rewritten (no silent caps — the
 * envelope states pre-filter count and the drop). Otherwise return `payload`
 * UNCHANGED (same reference) — absent-or-invalid bbox is byte-for-byte the
 * current behavior. Kept OUT of the route handler body so the handler does not
 * grow (the S1 sweep reuses this one helper across every point layer).
 */
export function applyViewport<T>(
  payload: any,
  bboxStr: unknown,
  arrayKey: string,
  getLatLon: (f: T) => [number, number] | null | undefined
): any {
  const b = parseBbox(bboxStr);
  if (!b || !payload || !Array.isArray(payload[arrayKey])) return payload;
  const arr = payload[arrayKey] as T[];
  const before = arr.length;
  const inView = filterByViewport(arr, b, getLatLon);
  return {
    ...payload,
    [arrayKey]: inView,
    count: inView.length,
    viewport_filtered: true,
    count_before_viewport: before,
    count_dropped_offscreen: before - inView.length,
  };
}
