// GEBCO v2 seafloor asset ratchet (EARTH TWIN E2 v2) — the committed
// PMTiles archives written by scripts/gebco_seafloor_tiles.py are decoded
// here with the REAL `pmtiles` reader (the same library the map client uses),
// so a malformed archive can never ship looking green (the gridTiles.test.ts
// lesson: magic alone missed a broken power_us once — this goes further and
// reads actual tiles back out).
//
// What it pins, against the committed assets + their provenance sidecar:
//  - archive integrity: header fields, bounds == provenance extent, PNG type
//  - depth truth: the tile over the Challenger Deep decodes to hadal depths
//    (< -10,000 m) via the terrarium encoding the client consumes
//  - TID honesty: every decoded cell is a documented GEBCO TID code or the
//    -1 no-data marker — nothing invented, nothing averaged into nonsense
//  - provenance honesty: verbatim GEBCO attribution (single source of truth:
//    datacore/gebco/tid_decode.json), not-for-navigation flag, measured group
//    shares that sum to 1, regional-coverage note
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { PMTiles, TileType, zxyToTileId, type Source } from "pmtiles";

const here = path.dirname(new URL(import.meta.url).pathname);
const tilesDir = path.join(here, "..", "client", "public", "tiles");
const demPath = path.join(tilesDir, "seafloor_gebco_mariana.pmtiles");
const tidPath = path.join(tilesDir, "seafloor_tid_mariana.pmtiles");
const provPath = path.join(tilesDir, "seafloor_gebco_mariana.provenance.json");
const decodeTable = JSON.parse(
  fs.readFileSync(path.join(here, "..", "datacore", "gebco", "tid_decode.json"), "utf8"),
);
const prov = JSON.parse(fs.readFileSync(provPath, "utf8"));

/** fs-backed pmtiles Source (the lib's FileSource is browser-File-only). */
class FsSource implements Source {
  constructor(private p: string) {}
  getKey() {
    return this.p;
  }
  async getBytes(offset: number, length: number) {
    const fd = fs.openSync(this.p, "r");
    try {
      const buf = Buffer.alloc(length);
      const n = fs.readSync(fd, buf, 0, length, offset);
      return { data: buf.buffer.slice(buf.byteOffset, buf.byteOffset + n) };
    } finally {
      fs.closeSync(fd);
    }
  }
}

/** Decode the pipeline's PNGs (8-bit RGB, filter 0 — asserted). */
function pngPixels(png: Buffer): { w: number; h: number; rows: Buffer[] } {
  assert.equal(png.subarray(0, 8).toString("latin1"), "\x89PNG\r\n\x1a\n", "PNG signature");
  const w = png.readUInt32BE(16);
  const h = png.readUInt32BE(20);
  let idat = Buffer.alloc(0);
  let off = 8;
  while (off < png.length) {
    const len = png.readUInt32BE(off);
    const tag = png.subarray(off + 4, off + 8).toString("latin1");
    if (tag === "IDAT") idat = Buffer.concat([idat, png.subarray(off + 8, off + 8 + len)]);
    off += 12 + len;
  }
  const raw = zlib.inflateSync(idat);
  const stride = w * 3 + 1;
  const rows: Buffer[] = [];
  for (let j = 0; j < h; j++) {
    const line = raw.subarray(j * stride, (j + 1) * stride);
    assert.equal(line[0], 0, "pipeline scanlines use filter 0");
    rows.push(Buffer.from(line.subarray(1)));
  }
  return { w, h, rows };
}

const terrarium = (r: number, g: number, b: number) => r * 256 + g + b / 256 - 32768;

// Challenger Deep (~142.59E, 11.37N) tile at z9, standard slippy math.
const DEEP_LON = 142.59;
const DEEP_LAT = 11.37;
const Z = 9;
const N = 2 ** Z;
const DEEP_X = Math.floor(((DEEP_LON + 180) / 360) * N);
const DEEP_Y = Math.floor(
  ((1 - Math.asinh(Math.tan((DEEP_LAT * Math.PI) / 180)) / Math.PI) / 2) * N,
);

test("depth archive: valid PMTiles header, png tiles, bounds = provenance extent", async () => {
  const pm = new PMTiles(new FsSource(demPath));
  const h = await pm.getHeader();
  assert.equal(h.tileType, TileType.Png);
  assert.equal(h.minZoom, 0);
  assert.equal(h.maxZoom, 9);
  assert.ok(h.clustered, "writer emits clustered archives");
  const bb = prov.coverage.bbox;
  assert.ok(Math.abs(h.minLon - bb.west) < 1e-6 && Math.abs(h.maxLon - bb.east) < 1e-6);
  assert.ok(Math.abs(h.minLat - bb.south) < 1e-6 && Math.abs(h.maxLat - bb.north) < 1e-6);
  const meta = (await pm.getMetadata()) as Record<string, unknown>;
  assert.ok(String(meta.attribution).includes("GEBCO Bathymetric Compilation Group"));
  assert.equal(meta.encoding, "terrarium");
});

test("depth truth: the Challenger Deep tile decodes to hadal depths through the client's terrarium math", async () => {
  const pm = new PMTiles(new FsSource(demPath));
  const t = await pm.getZxy(Z, DEEP_X, DEEP_Y);
  assert.ok(t, `tile ${Z}/${DEEP_X}/${DEEP_Y} must exist inside the covered region`);
  const { w, h, rows } = pngPixels(Buffer.from(t.data));
  assert.equal(w, 256);
  assert.equal(h, 256);
  let min = Infinity;
  let max = -Infinity;
  for (const row of rows) {
    for (let i = 0; i < w; i++) {
      const v = terrarium(row[i * 3], row[i * 3 + 1], row[i * 3 + 2]);
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  assert.ok(min < -10000, `Challenger Deep tile min ${min} must be hadal (< -10,000 m)`);
  assert.ok(min > -11100, `deeper than the ocean gets (${min}) — encoding broken`);
  assert.ok(max <= 0, "an all-ocean tile never decodes above sea level");
});

test("TID archive: every decoded cell is a documented code or the -1 no-data marker", async () => {
  const pm = new PMTiles(new FsSource(tidPath));
  const h = await pm.getHeader();
  assert.equal(h.tileType, TileType.Png);
  const documented = new Set(Object.keys(decodeTable.codes).map(Number));
  const t = await pm.getZxy(Z, DEEP_X, DEEP_Y);
  assert.ok(t, "TID tile must pair with the depth tile");
  const { w, rows } = pngPixels(Buffer.from(t.data));
  const seen = new Set<number>();
  for (const row of rows) {
    for (let i = 0; i < w; i++) {
      seen.add(Math.round(terrarium(row[i * 3], row[i * 3 + 1], row[i * 3 + 2])));
    }
  }
  for (const v of seen) {
    assert.ok(v === -1 || documented.has(v), `undocumented TID value ${v} in shipped tile`);
  }
  // this well-surveyed trench tile must carry direct measurements
  assert.ok(
    [...seen].some((v) => v >= 10 && v <= 17),
    "expected direct-measurement codes over the Challenger Deep",
  );
});

test("provenance sidecar: verbatim attribution, not-for-navigation, measured shares, honest regional coverage", () => {
  assert.equal(prov.attribution, decodeTable.source.attribution,
    "attribution must be the ONE string from datacore/gebco/tid_decode.json (single source of truth)");
  assert.equal(prov.notForNavigation, true);
  assert.equal(prov.coverage.native_resolution_arcsec, 15);
  assert.match(prov.coverage.note, /REGIONAL subset/);
  assert.equal(prov.files.bathymetry, path.basename(demPath));
  assert.equal(prov.files.tid, path.basename(tidPath));
  const shares = Object.values(prov.tid.group_share_of_covered_cells) as number[];
  const sum = shares.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 0.01, `group shares must sum to ~1 (got ${sum})`);
  for (const code of Object.keys(prov.tid.histogram_by_code)) {
    assert.ok(code in decodeTable.codes, `provenance histogram carries undocumented code ${code}`);
  }
});

test("tile-id cross-check: the Python writer's Hilbert ordering matches the real pmtiles library", () => {
  // the same pins live in test_gebco_seafloor_tiles.py — if either side drifts,
  // archives stop being readable and this catches which side moved
  assert.equal(zxyToTileId(0, 0, 0), 0);
  assert.equal(zxyToTileId(1, 0, 0), 1);
  assert.equal(zxyToTileId(1, 0, 1), 2);
  assert.equal(zxyToTileId(1, 1, 1), 3);
  assert.equal(zxyToTileId(1, 1, 0), 4);
  assert.equal(zxyToTileId(2, 0, 0), 5);
});
