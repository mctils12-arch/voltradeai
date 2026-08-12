// moonTiles — LROC WAC tile STREAMING + MOSAIC for the focused-close Moon
// (CELESTIAL v2 B4). The browser fetches Trek tiles DIRECTLY (CORS `*`,
// crossOrigin="anonymous"), decodes them OFF-thread (createImageBitmap), and
// stitches the visible near-side region into ONE high-res equirect mosaic
// that moonSurface samples as its detail overlay. Discipline mirrors
// spaceAssets: nothing loads until the Moon is focused-close (zero startup
// cost), decode is off-thread, main-thread readback is banded (no task
// >16ms), the mosaic is EVICTED on zoom-out and fully freed on dispose.
//
// Split: the PLANNING + STITCH math is pure/testable (moonTiles.test.ts); the
// manager owns fetch/canvas and is constructed only from the mounted frame,
// with injectable fetch/decode seams so the pipeline is exercisable headless.

import {
  MOON_TREK,
  MOON_TREK_CREDIT,
  type TrekScheme,
  type TileId,
  type BboxDeg,
  type MosaicWindow,
  matrixWidth,
  degPerTile,
  clampLatDeg,
  zoomForResolution,
  tilesForBbox,
  mosaicWindow,
  tileUrl,
} from "./lroc.ts";

export { MOON_TREK_CREDIT };

export interface TexImageLike {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

// ── planning (pure) ─────────────────────────────────────────────────────────

/** Mosaic edge cap, px (GPU budget): the stitched RGBA never exceeds this on
 *  either side — 2048² RGBA ≈ 16 MB, under the ~4096 mobile texture cap. This
 *  is the real bound (tiles below is a safety net). */
export const MOON_MOSAIC_MAX_PX = 2048;

/** Hard safety ceiling on tiles per mosaic (fetch bound). At 256px/tile,
 *  MOON_MOSAIC_MAX_PX already limits an axis to 8 tiles ⇒ ≤ 64. */
export const MOON_MAX_TILES = 72;

/** Concurrent tile fetches while a mosaic builds (a mosaic can be dozens of
 *  tiles; serial would take seconds). Bounded so outstanding requests stay
 *  polite to the service. */
export const MOON_FETCH_CONCURRENCY = 6;

export interface MoonTarget {
  z: number;
  bbox: BboxDeg;
  tiles: TileId[];
  mosaic: MosaicWindow;
}

/**
 * Plan the mosaic for a sub-camera point at the given on-screen resolution.
 * Picks the finest native level meeting `pxPerDeg`, then backs OFF one level
 * at a time until the region fits within MOON_MAX_TILES (bounded fetch +
 * memory). `halfSpanDeg` is the mosaic half-width around the sub-point; lat
 * is clamped to the poles. Returns null if nothing resolves (degenerate).
 */
export function planMoonTarget(
  subLonDeg: number,
  subLatDeg: number,
  pxPerDeg: number,
  halfSpanDeg: number,
  opts: { scheme?: TrekScheme; maxTiles?: number; minZ?: number; maxPx?: number } = {},
): MoonTarget | null {
  const scheme = opts.scheme ?? MOON_TREK;
  const maxTiles = opts.maxTiles ?? MOON_MAX_TILES;
  const maxPx = opts.maxPx ?? MOON_MOSAIC_MAX_PX;
  const minZ = opts.minZ ?? 2;
  const latMax = clampLatDeg(subLatDeg + halfSpanDeg);
  const latMin = clampLatDeg(subLatDeg - halfSpanDeg);
  const bbox: BboxDeg = {
    lonMin: subLonDeg - halfSpanDeg,
    lonMax: subLonDeg + halfSpanDeg,
    latMin,
    latMax,
  };
  // finest native level whose mosaic fits the pixel/tile budget: start at the
  // demanded resolution and back off until the stitched mosaic is within
  // MOON_MOSAIC_MAX_PX (the real GPU bound — coarser levels are never worth
  // fetching since a finer one that fits always looks better).
  let z = zoomForResolution(pxPerDeg, scheme, minZ);
  for (; z >= minZ; z--) {
    const tiles = tilesForBbox(bbox, z);
    if (!tiles.length) continue;
    const mosaic = mosaicWindow(tiles, scheme);
    if (mosaic && mosaic.pxW <= maxPx && mosaic.pxH <= maxPx && tiles.length <= maxTiles) {
      return { z, bbox, tiles, mosaic };
    }
  }
  return null;
}

/** A stable key for a target — the manager rebuilds only when it changes. */
export function targetKey(t: MoonTarget): string {
  return `${t.z}|${t.tiles[0]?.x},${t.tiles[0]?.y}|${t.mosaic.cols}x${t.mosaic.rows}`;
}

// ── stitch (pure) ───────────────────────────────────────────────────────────

export interface TilePlacement {
  tile: TileId;
  /** column index within the mosaic (0 = west), row index (0 = north). */
  col: number;
  row: number;
}

/** Where each tile sits in the mosaic grid (handles the wrapped west anchor).
 *  Order follows tilesForBbox (row-major from the NW corner). */
export function tilePlacements(tiles: TileId[]): TilePlacement[] {
  if (!tiles.length) return [];
  const z = tiles[0].z;
  const w = matrixWidth(z);
  const firstX = tiles[0].x;
  const firstY = tiles[0].y;
  return tiles.map((tile) => ({
    tile,
    col: ((tile.x - firstX) % w + w) % w,
    row: tile.y - firstY,
  }));
}

/**
 * Blit an already-decoded tile RGBA (`tileRgba`, tilePx²) into the mosaic
 * buffer `out` (mosaic.pxW × mosaic.pxH RGBA) at grid (col,row), for the
 * mosaic rows that fall in [rowStart,rowEnd) — chunkable so a full mosaic
 * readback never busts a frame. No-op for a placement outside the band.
 */
export function blitTile(
  tileRgba: TexImageLike,
  place: TilePlacement,
  mosaic: MosaicWindow,
  tilePx: number,
  out: Uint8ClampedArray,
  rowStart: number,
  rowEnd: number,
): void {
  const oy0 = place.row * tilePx;
  const ox0 = place.col * tilePx;
  const y0 = Math.max(oy0, rowStart);
  const y1 = Math.min(oy0 + tilePx, rowEnd, mosaic.pxH);
  const x1 = Math.min(ox0 + tilePx, mosaic.pxW);
  for (let y = y0; y < y1; y++) {
    const sy = y - oy0;
    let di = (y * mosaic.pxW + ox0) * 4;
    let si = sy * tilePx * 4;
    for (let x = ox0; x < x1; x++) {
      out[di] = tileRgba.data[si];
      out[di + 1] = tileRgba.data[si + 1];
      out[di + 2] = tileRgba.data[si + 2];
      // REAL alpha, not a hardcoded 255 (NAC tier, 2026-08-12): a NAC strip
      // tile's transparent margin must stay transparent so the alpha-gated
      // sampler falls through to the WAC tier under it. Opaque JPEG sources
      // (WAC and every planet mosaic) decode to alpha 255 everywhere, so
      // this is byte-identical for them.
      out[di + 3] = tileRgba.data[si + 3];
      di += 4;
      si += 4;
    }
  }
}

// ── the manager (DOM side — constructed only by the mounted frame) ──────────

export interface MoonMosaic {
  tex: TexImageLike;
  /** DetailOverlay window fields (moonSurface consumes these directly). */
  lonMin: number;
  lonSpan: number;
  latMax: number;
  latSpan: number;
  z: number;
  tiles: number;
}

export interface MoonTileStats {
  /** mosaic bytes resident (0 when evicted). */
  bytes: number;
  /** tiles composited into the current mosaic. */
  tiles: number;
  z: number | null;
  /** worst single main-thread readback band, ms (the ≤16ms proof). */
  maxChunkMs: number;
  building: boolean;
  mainThreadDecode: boolean;
}

export interface MoonTileManager {
  /** per settled close frame: ensure a mosaic for this sub-point/resolution. */
  request(subLonDeg: number, subLatDeg: number, pxPerDeg: number, halfSpanDeg: number): void;
  /** best resident mosaic (null when none / evicted). */
  current(): MoonMosaic | null;
  /** drop the mosaic + tile cache (zoom-out eviction). */
  clear(): void;
  stats(): MoonTileStats;
  onUpdate(cb: () => void): void;
  dispose(): void;
}

export interface MoonTileDeps {
  scheme?: TrekScheme;
  /** fetch a tile as a decoded RGBA image (off-thread). Browser default uses
   *  fetch + createImageBitmap + a banded canvas readback. */
  fetchTile?: (url: string, signal: AbortSignal) => Promise<TexImageLike>;
  /** rows per readback band (bounded main-thread task). */
  bandRows?: number;
  /** coarsest level worth planning (NAC site managers pass ~10 — a z2 tile
   *  of a landing-site strip is one 404, not a fallback). Default 2. */
  minZ?: number;
}

async function defaultFetchTile(url: string, signal: AbortSignal): Promise<TexImageLike> {
  const res = await fetch(url, { signal, mode: "cors" });
  if (!res.ok) throw new Error(`${res.status}`);
  const blob = await res.blob();
  const bmp = await createImageBitmap(blob);
  const w = bmp.width;
  const h = bmp.height;
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  try { bmp.close(); } catch { /* closed */ }
  const img = ctx.getImageData(0, 0, w, h);
  return { data: img.data, width: w, height: h };
}

export function createMoonTileManager(deps: MoonTileDeps = {}): MoonTileManager {
  const scheme = deps.scheme ?? MOON_TREK;
  const fetchTile = deps.fetchTile ?? defaultFetchTile;
  // 32 (down from 64) halves the worst tile-readback band so it stays under the
  // 16ms frame budget even on a loaded SwiftShader machine (a 2048px-wide mosaic
  // band measured ~19ms at 64 rows during the 2026-07-19 crisp-motion drive).
  const bandRows = deps.bandRows ?? 32;
  const tilePx = scheme.tilePx;

  const tileCache = new Map<string, TexImageLike>(); // key: z/x/y
  let mosaic: MoonMosaic | null = null;
  let builtKey = "";
  let building = false;
  let pending: MoonTarget | null = null;
  let disposed = false;
  // clear()/dispose() bump this; an in-flight build() captures it at the start
  // and bails if it changed, so a build that was streaming when the mosaic was
  // EVICTED never re-populates it afterwards (the __vtMoonTileBytes → 0 contract
  // must hold even mid-build — B5 continuous refresh made this reachable).
  let gen = 0;
  let maxChunkMs = 0;
  const mainThreadDecode = false;
  let update: () => void = () => {};
  const abort = new AbortController();

  const mirror = (): void => {
    try { (globalThis as any).__vtMoonTileBytes = mosaic ? mosaic.tex.data.byteLength : 0; } catch { /* sandboxed */ }
  };
  const tkey = (t: TileId): string => `${t.z}/${t.x}/${t.y}`;

  async function build(target: MoonTarget): Promise<void> {
    if (building || disposed) return;
    building = true;
    const myGen = gen; // bail if clear()/dispose() bumps this mid-build
    try {
      const placements = tilePlacements(target.tiles);
      // fetch every missing tile with bounded concurrency (a mosaic is dozens
      // of tiles; decode is off-thread inside fetchTile). A 404/network miss
      // leaves that tile base-only — silent degrade, never a broken mosaic.
      const missing = target.tiles.filter((t) => !tileCache.has(tkey(t)));
      let next = 0;
      const worker = async (): Promise<void> => {
        while (next < missing.length && !disposed && myGen === gen) {
          const t = missing[next++];
          const k = tkey(t);
          if (tileCache.has(k)) continue;
          try {
            const img = await fetchTile(tileUrl(t, scheme), abort.signal);
            if (disposed) return;
            if (tileCache.size > 160) tileCache.clear(); // bounded cache
            tileCache.set(k, img);
          } catch { /* missing tile → base fallback */ }
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(MOON_FETCH_CONCURRENCY, Math.max(1, missing.length)) }, () => worker()),
      );
      if (disposed || myGen !== gen) return; // evicted mid-fetch → drop the work
      // supersede: if a newer target arrived while we fetched (the camera kept
      // moving), don't waste work stitching this stale mosaic — bail and let
      // finally{} rebuild for the freshest pose, so the FIRST mosaic shown is
      // the one that matches where the camera actually settled.
      if (pending && targetKey(pending) !== targetKey(target)) return;
      const m = target.mosaic;
      const out = new Uint8ClampedArray(m.pxW * m.pxH * 4);
      // banded readback: blit tiles into the mosaic buffer in row bands, each
      // a bounded main-thread task (yield between bands)
      let composited = 0;
      for (let y = 0; y < m.pxH; y += bandRows) {
        const yEnd = Math.min(y + bandRows, m.pxH);
        const t0 = nowMs();
        for (const p of placements) {
          const img = tileCache.get(tkey(p.tile));
          if (!img) continue;
          blitTile(img, p, m, tilePx, out, y, yEnd);
        }
        const ms = nowMs() - t0;
        if (ms > maxChunkMs) maxChunkMs = ms;
        if (yEnd < m.pxH) await new Promise((r) => setTimeout(r, 0));
      }
      for (const p of placements) if (tileCache.has(tkey(p.tile))) composited++;
      if (disposed || myGen !== gen) return; // evicted mid-stitch → don't repopulate
      mosaic = {
        tex: { data: out, width: m.pxW, height: m.pxH },
        lonMin: m.lonMin,
        lonSpan: m.lonSpan,
        latMax: m.latMax,
        latSpan: m.latSpan,
        z: m.z,
        tiles: composited,
      };
      builtKey = targetKey(target);
      mirror();
      update();
    } finally {
      building = false;
      // a request that arrived mid-build supersedes: rebuild for it
      if (pending && !disposed) {
        const next = pending;
        pending = null;
        if (targetKey(next) !== builtKey) void build(next);
      }
    }
  }

  return {
    request(subLonDeg, subLatDeg, pxPerDeg, halfSpanDeg): void {
      if (disposed) return;
      const target = planMoonTarget(subLonDeg, subLatDeg, pxPerDeg, halfSpanDeg, { scheme, minZ: deps.minZ });
      if (!target) return;
      const key = targetKey(target);
      if (key === builtKey && mosaic) return; // already resident
      if (building) { pending = target; return; }
      void build(target);
    },
    current(): MoonMosaic | null {
      return mosaic;
    },
    clear(): void {
      // bump the generation FIRST (before any early-out) so a build in flight —
      // even one that hasn't cached a tile yet — bails instead of repopulating
      gen++;
      if (!mosaic && !tileCache.size) return;
      mosaic = null;
      builtKey = "";
      pending = null;
      tileCache.clear();
      mirror();
      update();
    },
    stats(): MoonTileStats {
      return {
        bytes: mosaic ? mosaic.tex.data.byteLength : 0,
        tiles: mosaic ? mosaic.tiles : 0,
        z: mosaic ? mosaic.z : null,
        maxChunkMs,
        building,
        mainThreadDecode,
      };
    },
    onUpdate(cb): void {
      update = cb;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      gen++;
      try { abort.abort(); } catch { /* already */ }
      tileCache.clear();
      mosaic = null;
      pending = null;
      mirror();
      update = () => {};
    },
  };
}

function nowMs(): number {
  try {
    return performance.now();
  } catch {
    return Date.now();
  }
}

// re-exports the frame needs alongside the manager
export { degPerTile, MOON_TREK };
