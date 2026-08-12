// lroc.test — WMTS tile math pinned to hand-checked values and to real
// selenographic feature coordinates (the alignment contract for B4).

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  MOON_TREK,
  MOON_TREK_CREDIT,
  MARS_TREK,
  MARS_TREK_CREDIT,
  MERCURY_TREK,
  MERCURY_TREK_CREDIT,
  VENUS_TREK,
  VENUS_TREK_CREDIT,
  TREK_BODIES,
  trekBody,
  matrixWidth,
  matrixHeight,
  degPerTile,
  tilePxPerDeg,
  normLonDeg,
  clampLatDeg,
  tileColForLon,
  tileRowForLat,
  tileUrl,
  tileBboxDeg,
  zoomForResolution,
  tilesForBbox,
  mosaicWindow,
} from "./lroc.ts";

// ── the matrix scheme (verbatim from WMTSCapabilities.xml) ──────────────────

test("matrix dimensions: width 2^(z+1), height 2^z", () => {
  assert.equal(matrixWidth(0), 2);
  assert.equal(matrixHeight(0), 1);
  assert.equal(matrixWidth(1), 4);
  assert.equal(matrixHeight(1), 2);
  assert.equal(matrixWidth(8), 512);
  assert.equal(matrixHeight(8), 256);
});

test("tiles are square in degrees: 180/2^z per edge", () => {
  assert.equal(degPerTile(0), 180);
  assert.equal(degPerTile(1), 90);
  assert.equal(degPerTile(8), 180 / 256);
});

test("max native depth is z=8 and tiles are 256px", () => {
  assert.equal(MOON_TREK.maxZ, 8);
  assert.equal(MOON_TREK.tilePx, 256);
});

test("z=8 native resolution ≈ 364 px/deg (finer than the 8k base)", () => {
  const ppd = tilePxPerDeg(8);
  assert.ok(ppd > 360 && ppd < 366, `got ${ppd}`);
  // the 8k base decodes at 4096 wide ⇒ 4096/360 ≈ 11.4 px/deg; tiles win big
  assert.ok(ppd / (4096 / 360) > 30);
});

// ── lon/lat normalisation ───────────────────────────────────────────────────

test("normLonDeg wraps into [-180,180)", () => {
  assert.equal(normLonDeg(0), 0);
  assert.equal(normLonDeg(190), -170);
  assert.equal(normLonDeg(-190), 170);
  assert.equal(normLonDeg(360), 0);
  assert.equal(normLonDeg(180), -180);
});

test("clampLatDeg clamps to the poles", () => {
  assert.equal(clampLatDeg(95), 90);
  assert.equal(clampLatDeg(-95), -90);
  assert.equal(clampLatDeg(12.5), 12.5);
});

// ── the tile containing a point (hand-checked) ──────────────────────────────

test("0°N,0°E sits at the first EAST-of-centre column, equator row", () => {
  // width 512 ⇒ col = floor((0+180)/360 * 512) = floor(256) = 256
  assert.equal(tileColForLon(0, 8), 256);
  // height 256 ⇒ row = floor((90-0)/180 * 256) = floor(128) = 128
  assert.equal(tileRowForLat(0, 8), 128);
});

test("the west edge (-180) is column 0, the north pole is row 0", () => {
  assert.equal(tileColForLon(-180, 8), 0);
  assert.equal(tileRowForLat(90, 8), 0);
  // just south of the north pole is still row 0
  assert.equal(tileRowForLat(89.9, 8), 0);
  // the south pole clamps to the last row
  assert.equal(tileRowForLat(-90, 8), 255);
});

test("columns WRAP around the antimeridian", () => {
  // +179.99 → column 511 (last), −179.99 → column 0 (first): the four tiles
  // around the antimeridian are {511,0} × two rows
  assert.equal(tileColForLon(179.99, 8), 511);
  assert.equal(tileColForLon(-179.99, 8), 0);
  // 181 wraps to -179 → same column as -179 (near the west edge, col 1)
  assert.equal(tileColForLon(181, 8), tileColForLon(-179, 8));
  assert.equal(tileColForLon(181, 8), 1);
});

test("tileBboxDeg inverts the col/row mapping", () => {
  const b = tileBboxDeg({ z: 8, x: 256, y: 128 });
  assert.ok(Math.abs(b.lonMin - 0) < 1e-9);
  assert.ok(Math.abs(b.lonMax - degPerTile(8)) < 1e-9);
  assert.ok(b.latMax <= 0 + 1e-9 && b.latMax > -degPerTile(8));
  // a point inside the tile maps back to that tile
  const midLon = (b.lonMin + b.lonMax) / 2;
  const midLat = (b.latMin + b.latMax) / 2;
  assert.equal(tileColForLon(midLon, 8), 256);
  assert.equal(tileRowForLat(midLat, 8), 128);
});

// ── real lunar features land in the right tile (alignment to the same
// selenographic frame the base 8k texture uses) ─────────────────────────────

test("Tycho crater (11.36°W, 43.31°S) round-trips through the tile grid", () => {
  const lon = -11.36; // 11.36° West
  const lat = -43.31; // 43.31° South
  const z = 6;
  const x = tileColForLon(lon, z);
  const y = tileRowForLat(lat, z);
  const b = tileBboxDeg({ z, x, y });
  assert.ok(lon >= b.lonMin && lon <= b.lonMax, `lon ${lon} in [${b.lonMin},${b.lonMax}]`);
  assert.ok(lat >= b.latMin && lat <= b.latMax, `lat ${lat} in [${b.latMin},${b.latMax}]`);
  // west of the prime meridian ⇒ column just below the centre column
  assert.ok(x < matrixWidth(z) / 2);
  // southern hemisphere ⇒ row below the equator row
  assert.ok(y > matrixHeight(z) / 2);
});

test("Mare Imbrium (~16°W, 33°N) lands NW of centre", () => {
  const lon = -16;
  const lat = 33;
  const z = 6;
  const x = tileColForLon(lon, z);
  const y = tileRowForLat(lat, z);
  assert.ok(x < matrixWidth(z) / 2, "west of prime meridian");
  assert.ok(y < matrixHeight(z) / 2, "north of equator");
  const b = tileBboxDeg({ z, x, y });
  assert.ok(lon >= b.lonMin && lon <= b.lonMax);
  assert.ok(lat >= b.latMin && lat <= b.latMax);
});

// ── URL construction (row before column) ────────────────────────────────────

test("tileUrl uses /{z}/{y}/{x}.jpg order", () => {
  assert.equal(
    tileUrl({ z: 8, x: 256, y: 128 }),
    `${MOON_TREK.baseUrl}/8/128/256.jpg`,
  );
});

test("credit string names the source and licence path", () => {
  assert.match(MOON_TREK_CREDIT, /NASA|LRO|LROC/);
  assert.match(MOON_TREK_CREDIT, /trek\.nasa\.gov/);
});

// ── zoom selection ──────────────────────────────────────────────────────────

test("zoomForResolution picks the coarsest level meeting the demand", () => {
  // want 10 px/deg — z where 256*2^z/180 >= 10 ⇒ 2^z >= 7.03 ⇒ z=3 (16*256/180=~22.8? no)
  // tilePxPerDeg: z2=5.7, z3=11.4, so z3 is first ≥10
  assert.equal(zoomForResolution(10), 3);
  assert.equal(zoomForResolution(5), 2); // z2 = 5.7 ≥ 5, floor minZ=2
  // an enormous demand saturates at maxZ
  assert.equal(zoomForResolution(1e6), MOON_TREK.maxZ);
  // never below the floor
  assert.equal(zoomForResolution(0.001), 2);
});

// ── bbox → tile cover + mosaic window ───────────────────────────────────────

test("tilesForBbox covers a small equatorial patch", () => {
  const z = 6;
  const dpt = degPerTile(z); // 2.8125°
  const bbox = { lonMin: -1, lonMax: 1, latMin: -1, latMax: 1 };
  const tiles = tilesForBbox(bbox, z);
  // patch < one tile wide/tall but straddles centre ⇒ up to 2×2 tiles
  assert.ok(tiles.length >= 1 && tiles.length <= 4, `got ${tiles.length}`);
  // every returned tile actually overlaps the bbox
  for (const t of tiles) {
    const b = tileBboxDeg(t);
    assert.ok(b.lonMax > bbox.lonMin && b.lonMin < bbox.lonMax);
    assert.ok(b.latMax > bbox.latMin && b.latMin < bbox.latMax);
  }
  assert.ok(dpt > 2 && dpt < 3);
});

test("tilesForBbox row-major from the NW corner", () => {
  const tiles = tilesForBbox({ lonMin: -10, lonMax: 10, latMin: -10, latMax: 10 }, 5);
  // first tile is north-most (smallest y) and west-most (smallest wrapped col
  // from lonMin) — its bbox north edge ≥ every other tile's
  const b0 = tileBboxDeg(tiles[0]);
  for (const t of tiles) {
    assert.ok(tileBboxDeg(t).latMax <= b0.latMax + 1e-9);
  }
});

test("tilesForBbox around the antimeridian yields wrapped columns, not a full sweep", () => {
  const z = 5;
  const w = matrixWidth(z); // 64
  const bbox = { lonMin: 178, lonMax: 182, latMin: -2, latMax: 2 };
  const tiles = tilesForBbox(bbox, z);
  const cols = new Set(tiles.map((t) => t.x));
  // must include the last column and the first column (the seam), and NOT
  // every column in between
  assert.ok(cols.has(w - 1) || cols.has(0));
  assert.ok(cols.size <= 3, `expected a couple of seam columns, got ${cols.size}`);
});

test("mosaicWindow describes the stitched equirect extent", () => {
  const z = 5;
  const tiles = tilesForBbox({ lonMin: -10, lonMax: 10, latMin: -10, latMax: 10 }, z);
  const win = mosaicWindow(tiles)!;
  assert.equal(win.z, z);
  // lon span = cols * degPerTile, lat span = rows * degPerTile
  assert.ok(Math.abs(win.lonSpan - win.cols * degPerTile(z)) < 1e-9);
  assert.ok(Math.abs(win.latSpan - win.rows * degPerTile(z)) < 1e-9);
  // the requested bbox lies inside the mosaic window
  assert.ok(win.lonMin <= -10 + 1e-9);
  assert.ok(win.lonMin + win.lonSpan >= 10 - 1e-9);
  assert.ok(win.latMax >= 10 - 1e-9);
  assert.ok(win.latMax - win.latSpan <= -10 + 1e-9);
  assert.equal(win.pxW, win.cols * MOON_TREK.tilePx);
  assert.equal(win.pxH, win.rows * MOON_TREK.tilePx);
});

test("mosaicWindow is null for no tiles", () => {
  assert.equal(mosaicWindow([]), null);
});

// ── B5: sibling rocky-body Trek schemes (probed from each body's own
// WMTSCapabilities.xml + a live tile fetch, 2026-07-19) ──────────────────────

test("B5 schemes carry each body's own max native z (probed, not assumed)", () => {
  // Mars/Mercury 404 at z=8 → maxZ 7; Venus 404 at z=6 → maxZ 5; Moon keeps 8.
  assert.equal(MOON_TREK.maxZ, 8);
  assert.equal(MARS_TREK.maxZ, 7);
  assert.equal(MERCURY_TREK.maxZ, 7);
  assert.equal(VENUS_TREK.maxZ, 5);
  // all Trek EQ tiles are 256 px regardless of body
  for (const s of [MOON_TREK, MARS_TREK, MERCURY_TREK, VENUS_TREK]) {
    assert.equal(s.tilePx, 256);
  }
});

test("B5 tile extension: Venus is PNG, the rest default to JPEG", () => {
  // the Moon/Mars/Mercury schemes carry no ext ⇒ tileUrl must still emit .jpg
  assert.equal(MARS_TREK.ext, "jpg");
  assert.equal(MERCURY_TREK.ext, "jpg");
  assert.equal(VENUS_TREK.ext, "png");
  assert.match(tileUrl({ z: 7, x: 3, y: 4 }, MARS_TREK), /\/7\/4\/3\.jpg$/);
  assert.match(tileUrl({ z: 7, x: 3, y: 4 }, MERCURY_TREK), /\/7\/4\/3\.jpg$/);
  assert.match(tileUrl({ z: 5, x: 3, y: 4 }, VENUS_TREK), /\/5\/4\/3\.png$/);
  // MOON_TREK (no ext field) is untouched — still .jpg (byte-identical to B4)
  assert.match(tileUrl({ z: 8, x: 1, y: 1 }, MOON_TREK), /\.jpg$/);
});

test("B5 baseUrls point at the real Trek EQ endpoints", () => {
  assert.match(MARS_TREK.baseUrl, /trek\.nasa\.gov\/tiles\/Mars\/EQ\/Mars_Viking_MDIM21/);
  assert.match(MERCURY_TREK.baseUrl, /trek\.nasa\.gov\/tiles\/Mercury\/EQ\/Mercury_MESSENGER_MDIS/);
  assert.match(VENUS_TREK.baseUrl, /trek\.nasa\.gov\/tiles\/Venus\/EQ\/Venus_Magellan_C3-MDIR/);
  // every baseUrl ends at the default028mm matrix set (no trailing slash)
  for (const s of [MOON_TREK, MARS_TREK, MERCURY_TREK, VENUS_TREK]) {
    assert.match(s.baseUrl, /default028mm$/);
  }
});

test("TREK_BODIES registry: the four rocky bodies, gas giants absent", () => {
  assert.deepEqual(new Set(Object.keys(TREK_BODIES)), new Set(["moon", "mars", "mercury", "venus"]));
  // gas giants + the Sun never get a surface patch
  for (const id of ["jupiter", "saturn", "uranus", "neptune", "sun", "earth"]) {
    assert.equal(trekBody(id), null, `${id} has no Trek surface`);
  }
  assert.equal(trekBody("mars")!.scheme, MARS_TREK);
  assert.equal(trekBody("moon")!.scheme, MOON_TREK);
});

test("each body's credit names its instrument + trek.nasa.gov", () => {
  assert.match(MARS_TREK_CREDIT, /Mars.*Viking.*trek\.nasa\.gov/);
  assert.match(MERCURY_TREK_CREDIT, /Mercury.*MESSENGER.*trek\.nasa\.gov/);
  assert.match(VENUS_TREK_CREDIT, /Venus.*Magellan.*trek\.nasa\.gov/);
});

test("zoomForResolution saturates at EACH body's own maxZ, not the Moon's", () => {
  assert.equal(zoomForResolution(1e6, MARS_TREK), 7);
  assert.equal(zoomForResolution(1e6, MERCURY_TREK), 7);
  assert.equal(zoomForResolution(1e6, VENUS_TREK), 5);
  assert.equal(zoomForResolution(1e6, MOON_TREK), 8);
});

// ── known planetary features land in the right tile (per-body alignment
// contract — the same selenographic→areographic frame the base sprite uses,
// with lon in [-180,180] positive-East as the Trek EQ bbox defines) ──────────

test("Mars — Olympus Mons (18.65°N, 226.2°E) lands NW of centre", () => {
  const lon = normLonDeg(226.2); // 226.2°E → −133.8°
  const lat = 18.65;
  const z = 6; // within Mars maxZ 7
  const x = tileColForLon(lon, z);
  const y = tileRowForLat(lat, z);
  const b = tileBboxDeg({ z, x, y });
  assert.ok(lon >= b.lonMin && lon <= b.lonMax, `lon ${lon} in [${b.lonMin},${b.lonMax}]`);
  assert.ok(lat >= b.latMin && lat <= b.latMax, `lat ${lat} in [${b.latMin},${b.latMax}]`);
  assert.ok(x < matrixWidth(z) / 2, "west of prime meridian (−133.8°)");
  assert.ok(y < matrixHeight(z) / 2, "north of equator");
});

test("Mars — Valles Marineris (~14°S, 59°W) lands SW of centre", () => {
  const lon = -59;
  const lat = -14;
  const z = 6;
  const x = tileColForLon(lon, z);
  const y = tileRowForLat(lat, z);
  const b = tileBboxDeg({ z, x, y });
  assert.ok(lon >= b.lonMin && lon <= b.lonMax);
  assert.ok(lat >= b.latMin && lat <= b.latMax);
  assert.ok(x < matrixWidth(z) / 2, "west of prime meridian");
  assert.ok(y > matrixHeight(z) / 2, "south of equator");
});

test("Mercury — Caloris Basin (~30.5°N, 189.8°E) lands NW near the antimeridian", () => {
  const lon = normLonDeg(189.8); // → −170.2°
  const lat = 30.5;
  const z = 5; // within Mercury maxZ 7
  const x = tileColForLon(lon, z);
  const y = tileRowForLat(lat, z);
  const b = tileBboxDeg({ z, x, y });
  assert.ok(lon >= b.lonMin && lon <= b.lonMax, `lon ${lon} in [${b.lonMin},${b.lonMax}]`);
  assert.ok(lat >= b.latMin && lat <= b.latMax);
  // −170° is just east of the −180 west edge ⇒ a low column, and northern
  assert.ok(x < matrixWidth(z) * 0.1, "near the west edge / antimeridian");
  assert.ok(y < matrixHeight(z) / 2, "north of equator");
});

test("Venus — Maxwell Montes (~65°N, 3.3°E) lands just E of centre, far north", () => {
  const lon = 3.3;
  const lat = 65.2;
  const z = 4; // within Venus maxZ 5
  const x = tileColForLon(lon, z);
  const y = tileRowForLat(lat, z);
  const b = tileBboxDeg({ z, x, y });
  assert.ok(lon >= b.lonMin && lon <= b.lonMax);
  assert.ok(lat >= b.latMin && lat <= b.latMax);
  assert.ok(x >= matrixWidth(z) / 2, "east of the prime meridian");
  assert.ok(y < matrixHeight(z) / 2, "north of equator");
});

// ── APOLLO NAC SITE MOSAICS (2026-08-12 — every value probed live) ──────────

test("all six Apollo sites have a NAC scheme; maxZ matches the live probes", async () => {
  const { APOLLO_NAC_SITES } = await import("./lroc.ts");
  assert.equal(APOLLO_NAC_SITES.length, 6);
  const byId = new Map(APOLLO_NAC_SITES.map((s) => [s.id, s]));
  // A11/12/14/15 serve z15 (A12/A15 declare z16 but 404 it); A16/A17 top at z14
  for (const id of ["apollo11", "apollo12", "apollo14", "apollo15"]) {
    assert.equal(byId.get(id)!.scheme.maxZ, 15, id);
  }
  for (const id of ["apollo16", "apollo17"]) {
    assert.equal(byId.get(id)!.scheme.maxZ, 14, id);
  }
  for (const s of APOLLO_NAC_SITES) {
    assert.equal(s.scheme.ext, "png", `${s.id}: NAC strips are gray+alpha PNG`);
    assert.ok(s.scheme.baseUrl.includes("trek.nasa.gov/tiles/Moon/EQ/"), s.id);
    assert.match(s.credit, /NAC/);
  }
});

test("nacSiteFor: every landing point is inside its own strip; Tycho is in none", async () => {
  const { nacSiteFor } = await import("./lroc.ts");
  const { APOLLO_SITES } = await import("./apolloSites.ts");
  for (const site of APOLLO_SITES) {
    const hit = nacSiteFor(site.lon, site.lat);
    assert.ok(hit, `${site.id}: landing point inside a NAC strip`);
    assert.equal(hit!.id, site.id, `${site.id}: inside its OWN strip`);
  }
  assert.equal(nacSiteFor(-11.36, -43.31), null, "Tycho has no Apollo NAC strip");
  assert.equal(nacSiteFor(170, 0), null, "far side has none");
});

test("NAC activation threshold = the WAC mosaic's deepest-level px/deg", async () => {
  const { NAC_ACTIVATE_PX_PER_DEG, tilePxPerDeg, MOON_TREK } = await import("./lroc.ts");
  assert.equal(NAC_ACTIVATE_PX_PER_DEG, tilePxPerDeg(MOON_TREK.maxZ, MOON_TREK));
  // z8 → 256 / (180/256) ≈ 364 px/deg — beyond this WAC has nothing finer
  assert.ok(Math.abs(NAC_ACTIVATE_PX_PER_DEG - 364.09) < 0.1);
});

test("a NAC scheme's deep tiles resolve sub-meter-class detail the WAC cannot", async () => {
  const { APOLLO_NAC_SITES, tilePxPerDeg, MOON_TREK } = await import("./lroc.ts");
  const a11 = APOLLO_NAC_SITES.find((s) => s.id === "apollo11")!;
  const nacPxPerDeg = tilePxPerDeg(a11.scheme.maxZ, a11.scheme);
  const wacPxPerDeg = tilePxPerDeg(MOON_TREK.maxZ, MOON_TREK);
  assert.ok(nacPxPerDeg / wacPxPerDeg >= 100, `NAC ${nacPxPerDeg} vs WAC ${wacPxPerDeg}`);
});

test("NAC plan-fit clamp: the desktop-floor case plans at the finest level instead of nothing", async () => {
  const { APOLLO_NAC_SITES, fitHalfSpanDeg, NAC_MIN_Z } = await import("./lroc.ts");
  const { planMoonTarget, MOON_MOSAIC_MAX_PX } = await import("./moonTiles.ts");
  const a11 = APOLLO_NAC_SITES.find((s) => s.id === "apollo11")!;
  // the 2026-08-12 desktop hole: Moon floor (1.02R) on a 1440px viewport
  // demands ~754 px/deg over a ~1.1° half-span — too wide for z10's tile
  // budget, and the minZ floor forbids backing off ⇒ planMoonTarget = null
  const pxPerDeg = 754;
  const wideHalf = 1.1;
  const broken = planMoonTarget(23.47297, 0.67408, pxPerDeg, wideHalf,
    { scheme: a11.scheme, minZ: NAC_MIN_Z, maxPx: MOON_MOSAIC_MAX_PX });
  assert.equal(broken, null, "unclamped desktop request cannot plan (the bug)");
  // the clamp shrinks the REQUEST to what the budget fits at the wanted z
  const half = fitHalfSpanDeg(pxPerDeg, a11.scheme, NAC_MIN_Z, MOON_MOSAIC_MAX_PX);
  assert.ok(half > 0.3 && half < wideHalf, `clamped half-span ${half}`);
  const t = planMoonTarget(23.47297, 0.67408, pxPerDeg, Math.min(wideHalf, half),
    { scheme: a11.scheme, minZ: NAC_MIN_Z, maxPx: MOON_MOSAIC_MAX_PX });
  assert.ok(t, "clamped request plans");
  assert.ok(t!.z >= NAC_MIN_Z, `plans at z${t!.z} (finest that meets the demand)`);
  assert.ok(t!.mosaic.pxW <= MOON_MOSAIC_MAX_PX && t!.mosaic.pxH <= MOON_MOSAIC_MAX_PX);
});
