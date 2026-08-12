// lroc — PURE WMTS TILE MATH for the NASA Moon Trek LROC WAC global mosaic
// (CELESTIAL v2 B4 "MOON FOR REAL", directive §4, 2026-07-18). NO network
// here — every function is deterministic lon/lat ↔ z/y/x arithmetic plus the
// tile→equirect-UV mapping the surface renderer samples. `npx tsx --test`-able.
//
// SOURCE + SCHEME (probed from the service's own WMTSCapabilities.xml on
// 2026-07-19, curl evidence in the B4 report — nothing here is assumed):
//   endpoint:  https://trek.nasa.gov/tiles/Moon/EQ/
//              LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm
//   URL order: /{z}/{y}/{x}.jpg   (row before column — WMTS TileRow/TileCol)
//   licence:   NASA/LRO LROC WAC global mosaic, PUBLIC DOMAIN.
//   CORS:      Access-Control-Allow-Origin: * — the browser fetches tiles
//              directly for WebGL/canvas textures (crossOrigin="anonymous"),
//              no server relay (contrast CelesTrak, which firewalls Railway).
//   projection: EQ = equirectangular / plate carrée, CRS EPSG:104903
//               (Moon 2000, SELENOGRAPHIC lon/lat, positive East).
//   bbox:      lon [-180, 180], lat [-90, 90]; TopLeftCorner = (-180, +90).
//   tiles:     256 × 256 px, grayscale JPEG.
//   matrix:    level z has MatrixWidth = 2^(z+1) columns, MatrixHeight = 2^z
//              rows (verbatim from the capabilities TileMatrix list) — so a
//              tile spans (180 / 2^z)° in BOTH axes (square in degrees). The
//              service leniently answers x = MatrixWidth with 200 (an extra
//              wrap column); we build strictly against MatrixWidth and never
//              request out of range.
//   depth:     max native level z = 8 (z = 9 → HTTP 404). At z = 8 that is
//              512 × 256 tiles = 131072 × 65536 px ≈ 364 px/°, i.e. ~76 m/px
//              at the equator — ~13× finer than the 8k base texture.
//
// ALIGNMENT: the base moon_8k texture (textureSphere.equirectUV) already
// places lon 0 (the sub-Earth prime meridian) at u = 0.5, lon increasing
// EAST with u, and the Moon's IAU orientation comes from rotation.ts. The
// WMTS bbox uses the SAME selenographic frame (−180 at the left edge, East
// positive), so a tile's lon/lat maps through equirectUV with no re-frame —
// tiles land on the same features the 8k already renders. The surface
// renderer samples both at the identical body lon/lat, so alignment is
// inherited by construction (pinned in lroc.test.ts against Tycho/Imbrium).
//
// BODY-GENERIC: every function takes the scheme constants as an argument-
// free default but the shape (2^(z+1)×2^z plate carrée) is the NASA Trek EQ
// convention shared by Mars/Mercury/etc. — a sibling body reuses this module
// by pointing MOON_TREK.baseUrl elsewhere. B4 builds + tests the Moon only.

export interface TrekScheme {
  /** URL prefix up to and including .../default028mm (no trailing slash). */
  baseUrl: string;
  /** native tile edge, px. */
  tilePx: number;
  /** deepest level that returns imagery (z beyond this 404s). */
  maxZ: number;
  /** tile file extension (Trek serves JPEG for most bodies, PNG for a few —
   *  Venus's Magellan mosaic is PNG). Defaults to "jpg" so MOON_TREK and every
   *  existing caller stay byte-identical. */
  ext?: "jpg" | "png";
}

export const MOON_TREK: TrekScheme = {
  baseUrl:
    "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm",
  tilePx: 256,
  maxZ: 8,
};

/** visible on-screen credit (public-domain courtesy line, shown whenever a
 *  Trek tile is resident — parallels spaceAssets.MILKY_WAY_CREDIT). */
export const MOON_TREK_CREDIT = "Moon: NASA LRO · LROC WAC 303 ppd · trek.nasa.gov";

// ── B5 "PLANET SURFACES" — sibling Trek EQ mosaics for the rocky bodies ──────
// (CELESTIAL v2 B5, directive §5, 2026-07-19). Every one of these was probed
// from its own WMTSCapabilities.xml + a live tile fetch on 2026-07-19 (curl
// evidence in the B5 report — NOTHING assumed from the Moon). All four bodies
// share the EXACT NASA Trek EQ plate-carrée convention this module already
// encodes — TileMatrixSet default028mm, TopLeftCorner (−180,+90), 256 px tiles,
// MatrixWidth = 2^(z+1), MatrixHeight = 2^z — so matrixWidth/degPerTile/
// tileColForLon/… are shared verbatim; ONLY the endpoint, the deepest served
// level (each 404s one past its maxZ), and the tile extension differ per body.
// The gas giants (Jupiter/Saturn/Uranus/Neptune) have NO solid surface and NO
// Trek mosaic — they keep their cloud-band sprites and are absent here.
//
//   MARS    z0..7 (z8→404), .jpg — Viking MDIM21 colour, 232 m/px source;
//           z7 = 512×256 tiles ≈ 325 m/px equator (~8× the 8k base sprite).
//   MERCURY z0..7 (z8→404), .jpg — MESSENGER MDIS BDR, 166 m/px source;
//           z7 ≈ 234 m/px equator (~8× the base sprite).
//   VENUS   z0..5 (z6→404), .png — Magellan C3-MDIR SAR radar mosaic, 2025 m/px
//           source; z5 = 64×32 tiles ≈ 2.32 km/px equator (~2× the base sprite).

export const MARS_TREK: TrekScheme = {
  baseUrl:
    "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm",
  tilePx: 256,
  maxZ: 7,
  ext: "jpg",
};
export const MARS_TREK_CREDIT = "Mars: NASA/USGS Viking MDIM21 · trek.nasa.gov";

export const MERCURY_TREK: TrekScheme = {
  baseUrl:
    "https://trek.nasa.gov/tiles/Mercury/EQ/Mercury_MESSENGER_MDIS_Basemap_BDR_Mosaic_Global_166m/1.0.0/default/default028mm",
  tilePx: 256,
  maxZ: 7,
  ext: "jpg",
};
export const MERCURY_TREK_CREDIT = "Mercury: NASA/JHUAPL MESSENGER MDIS · trek.nasa.gov";

export const VENUS_TREK: TrekScheme = {
  baseUrl:
    "https://trek.nasa.gov/tiles/Venus/EQ/Venus_Magellan_C3-MDIR_Global_Mosaic_2025m/1.0.0/default/default028mm",
  tilePx: 256,
  maxZ: 5,
  ext: "png",
};
export const VENUS_TREK_CREDIT = "Venus: NASA/JPL Magellan C3-MDIR · trek.nasa.gov";

/** A body that has a real Trek surface mosaic: its scheme + on-screen credit. */
export interface TrekBody {
  scheme: TrekScheme;
  credit: string;
}

/** Registry of every body B5 streams a real surface for (the Moon from B4 +
 *  the rocky planets). Keyed by the same body id spaceFrame/rotation use. A
 *  body absent here (gas giants, the Sun) simply never gets a surface patch. */
export const TREK_BODIES: Record<string, TrekBody> = {
  moon: { scheme: MOON_TREK, credit: MOON_TREK_CREDIT },
  mars: { scheme: MARS_TREK, credit: MARS_TREK_CREDIT },
  mercury: { scheme: MERCURY_TREK, credit: MERCURY_TREK_CREDIT },
  venus: { scheme: VENUS_TREK, credit: VENUS_TREK_CREDIT },
};

/** The Trek surface config for a body id, or null if it has none. */
export function trekBody(id: string): TrekBody | null {
  return TREK_BODIES[id] ?? null;
}

// ── APOLLO NAC SITE MOSAICS (human 2026-08-12: "can you see the Rover? is
// there more imagery? the tracks the flag?") ────────────────────────────────
// Trek hosts a dedicated LROC NAC mosaic strip per landing site — REAL
// ~0.5 m/px imagery in which the descent stages, experiment packages, rover
// tracks and foot trails are actually resolved (confirmed by live tile
// fetch 2026-08-12: the Eagle descent stage + the trail to Little West
// Crater are visible in the z15 tile at Tranquility Base). Every entry
// below was probed live on 2026-08-12: WMTSCapabilities.xml + a tile at
// the landing coordinates. A12/A15 declare a z16 matrix but 404 it — the
// deepest SERVED level is what maxZ records. Out-of-strip and beyond-maxZ
// requests 404 cleanly (the manager treats those as holes; the alpha-gated
// sampler falls back to the WAC tier under them). PNG gray+alpha.
//
// HONESTY: these are NASA/ASU-processed public-domain mosaics streamed
// as-is — we do not enhance, colorize, or annotate the pixels. What you
// see at max zoom is what LRO photographed.

export interface NacSite {
  /** apolloSites.ts site id — the join key for markers/cards. */
  id: string;
  scheme: TrekScheme;
  /** product coverage strip (selenographic degrees, from the Trek catalog). */
  bbox: BboxDeg;
  credit: string;
}

const nacScheme = (label: string, maxZ: number): TrekScheme => ({
  baseUrl: `https://trek.nasa.gov/tiles/Moon/EQ/${label}/1.0.0/default/default028mm`,
  tilePx: 256,
  maxZ,
  ext: "png",
});

export const APOLLO_NAC_SITES: NacSite[] = [
  { id: "apollo11", scheme: nacScheme("LRO_NAC_Apollo11_Mosaic_p", 15),
    bbox: { lonMin: 23.4484977, lonMax: 23.5396839, latMin: 0.1464484, latMax: 1.1149051 },
    credit: "Apollo 11 site: NASA LRO · LROC NAC ~0.5 m/px · trek.nasa.gov" },
  { id: "apollo12", scheme: nacScheme("LRO_NAC_Apollo12_Mosaic_p", 15),
    bbox: { lonMin: -23.4442323, lonMax: -23.3571898, latMin: -3.4713148, latMax: -2.5018766 },
    credit: "Apollo 12 site: NASA LRO · LROC NAC ~0.5 m/px · trek.nasa.gov" },
  { id: "apollo14", scheme: nacScheme("LRO_NAC_Apollo14_Mosaic_p", 15),
    bbox: { lonMin: -17.4900752, lonMax: -17.3907622, latMin: -4.1920876, latMax: -3.2253601 },
    credit: "Apollo 14 site: NASA LRO · LROC NAC ~0.5 m/px · trek.nasa.gov" },
  { id: "apollo15", scheme: nacScheme("LRO_NAC_Apollo15_Mosaic_p", 15),
    bbox: { lonMin: 3.581094, lonMax: 3.689933, latMin: 25.7965264, latMax: 26.7635835 },
    credit: "Apollo 15 site: NASA LRO · LROC NAC ~0.5 m/px · trek.nasa.gov" },
  { id: "apollo16", scheme: nacScheme("LRO_NAC_Apollo16_Mosaic_p", 14),
    bbox: { lonMin: 15.3788327, lonMax: 15.5545346, latMin: -9.6060973, latMax: -8.6616783 },
    credit: "Apollo 16 site: NASA LRO · LROC NAC ~1.3 m/px · trek.nasa.gov" },
  { id: "apollo17", scheme: nacScheme("NAC_DTM_APOLLO17_MOSAIC_120CM", 14),
    bbox: { lonMin: 29.9059014, lonMax: 31.6579562, latMin: 19.3905439, latMax: 21.3035034 },
    credit: "Apollo 17 site: NASA LRO · LROC NAC ~1.2 m/px · trek.nasa.gov" },
];

/** px/deg the WAC global mosaic tops out at (its deepest level) — beyond
 *  this the screen demands more than WAC has, and a NAC site strip (when
 *  the view is inside one) is the only honest source of more detail. */
export const NAC_ACTIVATE_PX_PER_DEG = 256 / (180 / 2 ** MOON_TREK.maxZ);

/** The NAC site whose coverage strip contains this surface point, or null.
 *  All six strips sit well inside ±90° lon (near side), so a plain range
 *  test needs no wrap handling. */
export function nacSiteFor(lonDeg: number, latDeg: number): NacSite | null {
  const lon = normLonDeg(lonDeg);
  for (const s of APOLLO_NAC_SITES) {
    if (lon >= s.bbox.lonMin && lon <= s.bbox.lonMax &&
        latDeg >= s.bbox.latMin && latDeg <= s.bbox.latMax) return s;
  }
  return null;
}

/** columns at level z (MatrixWidth). */
export function matrixWidth(z: number): number {
  return 2 ** (z + 1);
}

/** rows at level z (MatrixHeight). */
export function matrixHeight(z: number): number {
  return 2 ** z;
}

/** degrees spanned by one tile edge at level z (equal in lon and lat). */
export function degPerTile(z: number): number {
  return 180 / 2 ** z;
}

/** native tile resolution in pixels per degree at level z. */
export function tilePxPerDeg(z: number, s: TrekScheme = MOON_TREK): number {
  return s.tilePx / degPerTile(z);
}

/** wrap a longitude into [-180, 180). */
export function normLonDeg(lon: number): number {
  return (((lon + 180) % 360) + 360) % 360 - 180;
}

/** clamp a latitude into [-90, 90]. */
export function clampLatDeg(lat: number): number {
  return lat < -90 ? -90 : lat > 90 ? 90 : lat;
}

/** tile COLUMN (x) containing a longitude at level z, in [0, width). Wraps. */
export function tileColForLon(lonDeg: number, z: number): number {
  const w = matrixWidth(z);
  const x = Math.floor(((normLonDeg(lonDeg) + 180) / 360) * w);
  return ((x % w) + w) % w;
}

/** tile ROW (y) containing a latitude at level z, in [0, height). */
export function tileRowForLat(latDeg: number, z: number): number {
  const h = matrixHeight(z);
  const y = Math.floor(((90 - clampLatDeg(latDeg)) / 180) * h);
  return y < 0 ? 0 : y >= h ? h - 1 : y;
}

export interface TileId {
  z: number;
  x: number;
  y: number;
}

/** WMTS tile URL — note the /{z}/{y}/{x}.ext (row before column) order. The
 *  extension is the scheme's `ext` (default "jpg" — Venus is "png"). */
export function tileUrl(t: TileId, s: TrekScheme = MOON_TREK): string {
  return `${s.baseUrl}/${t.z}/${t.y}/${t.x}.${s.ext ?? "jpg"}`;
}

export interface BboxDeg {
  lonMin: number;
  lonMax: number;
  latMin: number;
  latMax: number;
}

/** selenographic lon/lat extent of a tile (lonMin at its west edge, latMax
 *  at its north edge — TopLeftCorner convention). */
export function tileBboxDeg(t: TileId): BboxDeg {
  const dpt = degPerTile(t.z);
  const lonMin = -180 + t.x * dpt;
  const latMax = 90 - t.y * dpt;
  return { lonMin, lonMax: lonMin + dpt, latMin: latMax - dpt, latMax };
}

/**
 * The smallest native level whose tiles resolve at least `wantPxPerDeg`
 * pixels per degree (clamped to the scheme's maxZ). Screen-resolution
 * matching: the caller passes the on-screen pixels-per-degree the Moon
 * currently draws at, so tiles are never fetched finer than the display can
 * show. `minZ` keeps a sane floor (a whole-hemisphere mosaic).
 */
export function zoomForResolution(
  wantPxPerDeg: number,
  s: TrekScheme = MOON_TREK,
  minZ = 2,
): number {
  let z = minZ;
  while (z < s.maxZ && tilePxPerDeg(z, s) < wantPxPerDeg) z++;
  return z;
}

/**
 * Every tile covering a lon/lat bbox at level z. Latitude rows are clamped
 * to the matrix; longitude columns WRAP (a bbox straddling the ±180 seam
 * yields the correct wrapped column list, never a giant 0..width span). The
 * returned tiles are row-major (north→south, then west→east) and de-duped.
 */
export function tilesForBbox(bbox: BboxDeg, z: number): TileId[] {
  const w = matrixWidth(z);
  const y0 = tileRowForLat(bbox.latMax, z);
  const y1 = tileRowForLat(bbox.latMin, z);
  const cx0 = tileColForLon(bbox.lonMin, z);
  // number of columns spanned, following the (possibly wrapping) lon range
  const span = Math.max(0, bbox.lonMax - bbox.lonMin);
  const nCols = span >= 360 ? w : Math.min(w, Math.floor(span / degPerTile(z)) + 2);
  const out: TileId[] = [];
  const seen = new Set<number>();
  for (let y = y0; y <= y1; y++) {
    for (let i = 0; i < nCols; i++) {
      const x = (cx0 + i) % w;
      const key = y * w + x;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ z, x, y });
    }
  }
  return out;
}

/**
 * Where a lon/lat sample falls inside a MOSAIC built from `tiles` (row-major
 * as returned by tilesForBbox): the mosaic's own equirect window. Callers
 * stitch tiles left→right (increasing wrapped lon) and top→bottom
 * (decreasing lat), so the mosaic spans:
 *   lonMin = west edge of the first column, lonSpan = nCols · degPerTile
 *   latMax = north edge of the first row,  latSpan = nRows · degPerTile
 * with lon measured CONTINUOUSLY from lonMin (unwrapped) so a seam-crossing
 * mosaic stays monotonic. Returns null for an empty tile list.
 */
export interface MosaicWindow {
  z: number;
  cols: number;
  rows: number;
  /** west edge of the mosaic, degrees (may be < −180: unwrapped anchor). */
  lonMin: number;
  lonSpan: number;
  /** north edge of the mosaic, degrees. */
  latMax: number;
  latSpan: number;
  /** pixel dimensions if each tile contributes tilePx (pre-cap). */
  pxW: number;
  pxH: number;
}

export function mosaicWindow(tiles: TileId[], s: TrekScheme = MOON_TREK): MosaicWindow | null {
  if (!tiles.length) return null;
  const z = tiles[0].z;
  const dpt = degPerTile(z);
  const cols = new Set<number>();
  const rows = new Set<number>();
  for (const t of tiles) {
    cols.add(t.x);
    rows.add(t.y);
  }
  const nCols = cols.size;
  const nRows = rows.size;
  // tilesForBbox emits row-major from the NW corner, so tiles[0] anchors the
  // west edge (its x may be a wrapped column near +180 — lonMin then exceeds
  // +180 unwrapped, and the sampler measures sample lon continuously from it).
  const first = tiles[0];
  return {
    z,
    cols: nCols,
    rows: nRows,
    lonMin: -180 + first.x * dpt,
    lonSpan: nCols * dpt,
    latMax: 90 - first.y * dpt,
    latSpan: nRows * dpt,
    pxW: nCols * s.tilePx,
    pxH: nRows * s.tilePx,
  };
}
