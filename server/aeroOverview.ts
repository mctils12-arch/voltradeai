// aeroOverview.ts — raster helpers that let an FAA chart be seen at zoom
// levels the FAA never cut tiles for (server/aeroCharts.ts).
//
// ZOOM OUT (below the FAA's lowest level): a tile at z is the 2x2 block of
// its four children at z+1, each shrunk to half size. Repeating that builds
// an overview pyramid down to a whole-country view — what ForeFlight shows
// when you zoom out on a sectional. Honest limit: it is a picture of the
// chart at a scale the FAA never drew, so text is not legible at country
// scale; airspace, airways and colors are.
//
// FILL (inside the FAA's band, where a deeper level has no tile but the
// level above does — IFR Low z12 exists only inside Area-chart footprints):
// the parent's quadrant is enlarged 2x, so the map stays one continuous
// chart instead of a patchwork of chart and satellite.
//
// CPU: libvips via sharp, which runs on libuv's thread pool — the event loop
// is never blocked (a pure-JS JPEG path measured ~200 ms per overview tile
// ON the main thread, the exact stall class the 2026-10-02 audit flagged).

import sharp from "sharp";

export const TILE_PX = 256;
const HALF = TILE_PX / 2;

export interface EncodedTile {
  body: Buffer;
  contentType: "image/jpeg" | "image/png";
}

/** JPEG quality for fully-opaque overview tiles: chart linework stays crisp
 *  at the reduced scale, ~3-5x smaller than PNG. */
export const OVERVIEW_JPEG_QUALITY = 85;

/** Child order matches the quadtree: [NW, NE, SW, SE] = (2x,2y), (2x+1,2y),
 *  (2x,2y+1), (2x+1,2y+1). */
export function childTiles(z: number, x: number, y: number): Array<{ z: number; x: number; y: number }> {
  return [
    { z: z + 1, x: 2 * x, y: 2 * y },
    { z: z + 1, x: 2 * x + 1, y: 2 * y },
    { z: z + 1, x: 2 * x, y: 2 * y + 1 },
    { z: z + 1, x: 2 * x + 1, y: 2 * y + 1 },
  ];
}

/** Encode raw RGBA: JPEG when every pixel is opaque (smaller), PNG when any
 *  transparency must survive (coverage edges, where the satellite shows
 *  through). Null when nothing is visible at all — callers cache that as a
 *  cheap "no chart here" marker. */
async function encodeRaw(raw: Buffer, w: number, h: number): Promise<EncodedTile | null> {
  const img = () => sharp(raw, { raw: { width: w, height: h, channels: 4 } });
  const st = await img().stats();
  const alpha = st.channels[3];
  if (!alpha || alpha.max === 0) return null;
  if (st.isOpaque) {
    return { body: await img().removeAlpha().jpeg({ quality: OVERVIEW_JPEG_QUALITY, mozjpeg: true }).toBuffer(), contentType: "image/jpeg" };
  }
  return { body: await img().png({ compressionLevel: 6 }).toBuffer(), contentType: "image/png" };
}

/** Shrink each present child (JPEG or PNG bytes) into its quadrant and
 *  encode. Resizing premultiplies alpha, so transparent coverage edges never
 *  darken the chart. A child that fails to decode is treated as absent. */
export async function composeOverview(children: ReadonlyArray<Buffer | null>): Promise<EncodedTile | null> {
  if (children.length !== 4) throw new Error("composeOverview needs exactly 4 children");
  const layers: sharp.OverlayOptions[] = [];
  for (let q = 0; q < 4; q++) {
    const c = children[q];
    if (!c || c.length === 0) continue;
    let small: Buffer;
    try {
      small = await sharp(c).ensureAlpha().resize(HALF, HALF, { fit: "fill", kernel: "lanczos3" }).raw().toBuffer();
    } catch {
      continue;
    }
    layers.push({ input: small, raw: { width: HALF, height: HALF, channels: 4 }, left: (q % 2) * HALF, top: Math.floor(q / 2) * HALF });
  }
  if (!layers.length) return null;
  const raw = await sharp({ create: { width: TILE_PX, height: TILE_PX, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite(layers).raw().toBuffer();
  return encodeRaw(raw, TILE_PX, TILE_PX);
}

/** The quarter of `parent` that covers child (cx, cy) — qx = cx & 1,
 *  qy = cy & 1 — enlarged 2x. Null when the parent does not decode or that
 *  quarter is empty. */
export function upscaleQuadrant(parent: Buffer, qx: 0 | 1, qy: 0 | 1): Promise<EncodedTile | null> {
  return fillUnder(null, parent, qx, qy);
}

/** Fill a tile's transparent pixels from its parent's quarter, enlarged 2x —
 *  for charts whose deepest level only exists in places (IFR Low z12 is
 *  drawn only inside Area-chart footprints; elsewhere the FAA serves a fully
 *  transparent PNG, not a 404). `tile` null = upstream had nothing at all.
 *  Null when there is nothing to fill: the tile is already fully opaque, or
 *  the parent quarter is empty too. */
export async function fillUnder(tile: Buffer | null, parent: Buffer, qx: 0 | 1, qy: 0 | 1): Promise<EncodedTile | null> {
  if (tile && tile.length) {
    try {
      if ((await sharp(tile).stats()).isOpaque) return null;
    } catch {
      tile = null; // undecodable upstream bytes: treat as a hole
    }
  }
  let base: Buffer;
  try {
    base = await sharp(parent).ensureAlpha()
      .extract({ left: qx * HALF, top: qy * HALF, width: HALF, height: HALF })
      .resize(TILE_PX, TILE_PX, { fit: "fill", kernel: "cubic" })
      .raw().toBuffer();
  } catch {
    return null;
  }
  let raw = base;
  if (tile && tile.length) {
    try {
      raw = await sharp(base, { raw: { width: TILE_PX, height: TILE_PX, channels: 4 } })
        .composite([{ input: await sharp(tile).ensureAlpha().resize(TILE_PX, TILE_PX).png().toBuffer() }])
        .raw().toBuffer();
    } catch {
      raw = base;
    }
  }
  return encodeRaw(raw, TILE_PX, TILE_PX);
}
