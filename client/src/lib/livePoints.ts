// Live-points tick helpers — PERF session #2 (user-reported lag/freezes).
// Pure + tested; datamap's wireLivePoints consumes them. Each function
// encodes one profiled waste source in the aircraft/vessels poll pipeline:
//
//  (a) velocity vectors were BUILT and setData'd (~a second full pass over
//      up to 10k records + a second structured-clone to the map worker)
//      on every tick even though their layer is invisible below minzoom 6;
//  (b) any fresh snapshot's count (jiggling ±20 per tick on thousands)
//      defeated setStatus's no-op render bail, re-rendering the whole
//      5,800-line page component every 15-30s;
//  (c) every pan settle refetched the SAME server-side 250nm circle under
//      a new 0.1°-rounded cache key — full download + parse + rebuild for
//      a viewport that barely moved.

/** The velocity-vector layer's minzoom (see wireLivePoints addLayer). */
export const VEC_MIN_ZOOM = 6;
/** Build slightly BEFORE visibility so crossing the boundary shows vectors
 *  immediately (the zoomend rebuild covers the between-ticks gap). */
export const VEC_BUILD_ZOOM = VEC_MIN_ZOOM - 0.5;

/** Vectors are invisible below their layer minzoom — building/shipping them
 *  there is pure waste (profiled cause (a)). */
export function shouldBuildVectors(zoom: number): boolean {
  return Number.isFinite(zoom) && zoom >= VEC_BUILD_ZOOM;
}

/**
 * Quantize a live count so identical-payload render bails actually engage
 * (profiled cause (b)): exact below 500 (small counts are information),
 * nearest-25 above (display staleness ≤ 24 on counts in the thousands —
 * the count is a rolling live estimate either way). NEVER used for data
 * decisions — display/status only.
 */
export function quantizeLiveCount(count: number): number {
  if (!Number.isFinite(count) || count < 0) return 0;
  return count <= 500 ? Math.round(count) : Math.round(count / 25) * 25;
}

/** What one successful fetch covered (camera at fetch time). */
export interface FetchFootprint {
  lat: number;
  lng: number;
  zoom: number;
  /** half of the viewport's larger span, in degrees latitude-equivalent —
   *  a proxy for the server's clamped 50-250nm serve radius. */
  halfSpanDeg: number;
}

const NM_PER_DEG_LAT = 60;
const SERVE_RADIUS_MIN_DEG = 50 / NM_PER_DEG_LAT; // server clamp floor (50nm)
const SERVE_RADIUS_MAX_DEG = 250 / NM_PER_DEG_LAT; // server clamp ceiling (250nm)

/** Build a footprint from map state (pure; caller passes bounds numbers). */
export function fetchFootprint(
  lat: number, lng: number, zoom: number,
  north: number, south: number, east: number, west: number,
): FetchFootprint {
  const latSpan = Math.abs(north - south);
  const lngSpan = Math.abs(east - west) * Math.cos((lat * Math.PI) / 180);
  return { lat, lng, zoom, halfSpanDeg: Math.max(latSpan, lngSpan) / 2 };
}

/**
 * Should a moveend trigger a refetch? (profiled cause (c)). Yes when the
 * camera left ~20% of the last fetch's served radius or changed zoom band;
 * no for jitter pans inside coverage. CONSERVATIVE by design: the 15-30s
 * interval still polls regardless (cheap `unchanged` answers), so skipping
 * a redundant moveend fetch never lets data age beyond one interval.
 */
export function needsRefetch(prev: FetchFootprint | null, next: FetchFootprint): boolean {
  if (!prev) return true;
  if (Math.abs(next.zoom - prev.zoom) >= 0.5) return true;
  const servedRadiusDeg = Math.min(
    SERVE_RADIUS_MAX_DEG,
    Math.max(SERVE_RADIUS_MIN_DEG, prev.halfSpanDeg),
  );
  const dLat = Math.abs(next.lat - prev.lat);
  const dLng = Math.abs(next.lng - prev.lng) * Math.cos(((prev.lat + next.lat) / 2) * Math.PI / 180);
  const movedDeg = Math.sqrt(dLat * dLat + dLng * dLng);
  return movedDeg > 0.2 * servedRadiusDeg;
}
