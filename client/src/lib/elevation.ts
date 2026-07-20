// GROUND ELEVATION WITHOUT THE 3D MESH (human 2026-07-20: "can we take
// the msl and if we know the ground point and have the whole world mapped
// can we calculate the agl") — yes: the same global terrarium DEM the
// map's mesh uses is a plain public tile set, so AGL never needs the
// Terrain toggle. This module fetches and decodes terrarium tiles
// directly (AWS Open Data Terrain Tiles — the exact bucket the seafloor
// mesh already consumes, so the licensing/attribution posture is already
// established) and answers point queries from a small decoded-tile cache.
//
// HONESTY: this is the same measured DEM the 3D terrain renders — SRTM/
// ETOPO-class ground truth, not an estimate. Queries return null while a
// tile is in flight (callers show "—" and re-read on their next tick,
// never 0-as-ground).
//
// Pure math (tile addressing, terrarium decode, bilinear sampling) is
// exported for unit tests; only fetch/decode touches the DOM.

/** sampling zoom: ~150 m/px at mid-latitudes — right for an AGL readout
 *  and a track ground profile; one tile spans ~80 km so a long track
 *  needs only a handful. */
export const ELEV_TILE_Z = 9;
/** decoded-tile LRU cap (each ≈ 128 KB as Int16 meters). */
export const ELEV_CACHE_CAP = 40;

const TILE_URL = (z: number, x: number, y: number) =>
  `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;

/** lon/lat → tile x/y (float — integer part is the tile, fraction the
 *  in-tile position). Standard web-mercator tiling. Pure. */
export function tileCoord(lon: number, lat: number, z: number): { x: number; y: number } {
  const n = Math.pow(2, z);
  let x = ((lon + 180) / 360) * n;
  x = ((x % n) + n) % n; // antimeridian wrap
  const latR = (Math.max(-85.051129, Math.min(85.051129, lat)) * Math.PI) / 180;
  let y = ((1 - Math.log(Math.tan(latR) + 1 / Math.cos(latR)) / Math.PI) / 2) * n;
  y = Math.max(0, Math.min(n - 1e-9, y)); // float epsilon at the poles
  return { x, y };
}

/** terrarium RGB → meters. Pure. */
export function terrariumDecode(r: number, g: number, b: number): number {
  return r * 256 + g + b / 256 - 32768;
}

/** bilinear sample of a decoded Int16 meter grid (size×size). fx/fy in
 *  pixel coordinates (0..size-1 range, clamped). Pure. */
export function bilinearSample(grid: Int16Array, size: number, fx: number, fy: number): number {
  const cx = Math.max(0, Math.min(size - 1, fx));
  const cy = Math.max(0, Math.min(size - 1, fy));
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  // clamp the far neighbors INSIDE the grid: at the exact edge (and on a
  // 1×1 grid) the +1 read would be out of bounds — undefined × weight 0
  // is NaN, not 0
  const x1 = Math.min(x0 + 1, size - 1), y1 = Math.min(y0 + 1, size - 1);
  const dx = cx - x0, dy = cy - y0;
  const g = (x: number, y: number) => grid[y * size + x];
  return (
    g(x0, y0) * (1 - dx) * (1 - dy) +
    g(x1, y0) * dx * (1 - dy) +
    g(x0, y1) * (1 - dx) * dy +
    g(x1, y1) * dx * dy
  );
}

interface TileEntry {
  state: "pending" | "ready" | "error";
  grid?: Int16Array;
  size?: number;
  at: number;
}

const cache = new Map<string, TileEntry>();

const evictIfNeeded = () => {
  if (cache.size <= ELEV_CACHE_CAP) return;
  // drop the oldest ready/error entries first (never in-flight ones)
  const entries = Array.from(cache.entries())
    .filter(([, e]) => e.state !== "pending")
    .sort((a, b) => a[1].at - b[1].at);
  for (const [k] of entries) {
    if (cache.size <= ELEV_CACHE_CAP) break;
    cache.delete(k);
  }
};

const fetchTile = (z: number, tx: number, ty: number, key: string) => {
  cache.set(key, { state: "pending", at: Date.now() });
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    try {
      const size = img.naturalWidth || 256;
      const cv = document.createElement("canvas");
      cv.width = size;
      cv.height = size;
      const ctx = cv.getContext("2d", { willReadFrequently: true });
      if (!ctx) throw new Error("no 2d ctx");
      ctx.drawImage(img, 0, 0);
      const px = ctx.getImageData(0, 0, size, size).data;
      const grid = new Int16Array(size * size);
      for (let i = 0, p = 0; i < grid.length; i++, p += 4) {
        grid[i] = Math.round(terrariumDecode(px[p], px[p + 1], px[p + 2]));
      }
      cache.set(key, { state: "ready", grid, size, at: Date.now() });
      evictIfNeeded();
    } catch {
      cache.set(key, { state: "error", at: Date.now() });
    }
  };
  img.onerror = () => { cache.set(key, { state: "error", at: Date.now() }); };
  img.src = TILE_URL(z, tx, ty);
};

/**
 * Ground elevation (meters MSL) at a point, or null while the covering
 * tile is loading / failed (a miss triggers the fetch — callers simply
 * re-read on their next tick). Synchronous by design: the flight-card
 * tick and the track painter both run at a cadence where "answer next
 * tick" is invisible.
 */
export function groundElevationSync(lon: number, lat: number): number | null {
  if (typeof document === "undefined") return null;
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  const { x, y } = tileCoord(lon, lat, ELEV_TILE_Z);
  const tx = Math.floor(x), ty = Math.floor(y);
  const key = `${ELEV_TILE_Z}/${tx}/${ty}`;
  const e = cache.get(key);
  if (!e) { fetchTile(ELEV_TILE_Z, tx, ty, key); return null; }
  if (e.state !== "ready" || !e.grid || !e.size) return null;
  e.at = Date.now();
  return bilinearSample(e.grid, e.size, (x - tx) * e.size, (y - ty) * e.size);
}

/** Kick fetches for every tile a track touches (fire-and-forget — the
 *  next repaint after tiles land picks the values up). */
export function prefetchElevation(points: ReadonlyArray<{ lon: number; lat: number }>): void {
  if (typeof document === "undefined") return;
  points.forEach((p) => {
    if (!Number.isFinite(p.lon) || !Number.isFinite(p.lat)) return;
    const { x, y } = tileCoord(p.lon, p.lat, ELEV_TILE_Z);
    const key = `${ELEV_TILE_Z}/${Math.floor(x)}/${Math.floor(y)}`;
    if (!cache.has(key)) fetchTile(ELEV_TILE_Z, Math.floor(x), Math.floor(y), key);
  });
}
