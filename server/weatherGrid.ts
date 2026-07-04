/**
 * weatherGrid.ts — sampled point grid for wind vectors + temperature labels
 * (weather-upgrade directive 2026-07-04). HONEST SOURCING: OWM's free tile
 * API is a raster color field with NO vector data; direction/speed/temp
 * numbers come from the free current-weather POINT API, sampled on a coarse
 * grid. The grid density is bounded by the 60-calls/min shared budget — the
 * UI states the real sampling spacing and never renders arrows denser than
 * the data (no faked barb density).
 *
 * Budget math: <=40 points per snapped-viewport bucket, shared cache 10-min
 * TTL, minute-level upstream guard — worst case bursts stay under the free
 * tier with headroom for the tile proxy.
 *
 * Pure module: URL/grid/parse/snap logic testable without network.
 */

export const GRID_MAX_POINTS = 40;
export const GRID_TTL_MS = 10 * 60_000;
export const UPSTREAM_PER_MIN_GUARD = 45; // leave headroom under OWM's 60/min

export interface GridPoint { la: number; lo: number }
export interface WeatherSample { la: number; lo: number; tc: number | null; wd: number | null; ws: number | null }

/** Snap a bbox to a coarse bucket so nearby viewports share one cache entry
 *  (and one upstream burst). The quantum itself is quantized to a power-of-two
 *  ladder — deriving it from the raw span would give slightly-different
 *  viewports slightly-different quanta and defeat sharing. Edges snap OUTWARD
 *  (floor/ceil) so the bucket always covers the viewport. */
export function snapBbox(s: number, w: number, n: number, e: number): { s: number; w: number; n: number; e: number } {
  const spanLa = Math.max(1, n - s), spanLo = Math.max(1, e - w);
  const ladder = (span: number) => Math.pow(2, Math.round(Math.log2(span / 4)));
  const qLa = ladder(spanLa), qLo = ladder(spanLo);
  const down = (v: number, q: number) => Math.floor(v / q) * q;
  const up = (v: number, q: number) => Math.ceil(v / q) * q;
  return {
    s: Math.max(-85, down(s, qLa)), w: Math.max(-180, down(w, qLo)),
    n: Math.min(85, up(n, qLa)), e: Math.min(180, up(e, qLo)),
  };
}

export function bboxKey(b: { s: number; w: number; n: number; e: number }): string {
  return [b.s, b.w, b.n, b.e].map((v) => v.toFixed(2)).join(",");
}

/** Evenly spaced grid, capped at maxPoints — density follows the cap, never
 *  the other way around. */
export function gridPoints(b: { s: number; w: number; n: number; e: number },
                           maxPoints = GRID_MAX_POINTS): GridPoint[] {
  const aspect = Math.max(0.3, Math.min(3, (b.e - b.w) / Math.max(1e-6, b.n - b.s)));
  let cols = Math.max(3, Math.round(Math.sqrt(maxPoints * aspect)));
  let rows = Math.max(3, Math.floor(maxPoints / cols));
  while (cols * rows > maxPoints) cols--;
  const out: GridPoint[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out.push({
        la: b.s + ((r + 0.5) / rows) * (b.n - b.s),
        lo: b.w + ((c + 0.5) / cols) * (b.e - b.w),
      });
    }
  }
  return out;
}

/** Real sampling spacing in km (for the honesty label: "sampled ~every N km"). */
export function spacingKm(b: { s: number; w: number; n: number; e: number }, nPoints: number): number {
  const midLa = (b.s + b.n) / 2;
  const wKm = Math.abs(b.e - b.w) * 111.32 * Math.cos((midLa * Math.PI) / 180);
  const hKm = Math.abs(b.n - b.s) * 110.57;
  return Math.round(Math.sqrt((wKm * hKm) / Math.max(1, nPoints)));
}

export function owmPointUrl(la: number, lo: number, key: string): string {
  return `https://api.openweathermap.org/data/2.5/weather?lat=${la.toFixed(3)}&lon=${lo.toFixed(3)}&units=metric&appid=${encodeURIComponent(key)}`;
}

/** Parses one current-weather response into the compact sample. Missing
 *  fields stay null — the UI skips them, never invents. */
export function parseOwmPoint(la: number, lo: number, j: any): WeatherSample {
  return {
    la: Math.round(la * 100) / 100,
    lo: Math.round(lo * 100) / 100,
    tc: typeof j?.main?.temp === "number" ? Math.round(j.main.temp * 10) / 10 : null,
    wd: typeof j?.wind?.deg === "number" ? Math.round(j.wind.deg) : null,
    ws: typeof j?.wind?.speed === "number" ? Math.round(j.wind.speed * 10) / 10 : null,
  };
}
