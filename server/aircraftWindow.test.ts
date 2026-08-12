import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { archiveAircraftAt, type AircraftPoint } from "./datacoreArchive";
import {
  readWindow, lodStepSec, lonInBBox, hourName, WINDOW_DEFAULT_CAPS,
} from "./aircraftWindow";

const T0 = Math.floor(Date.parse("2026-08-10T12:00:00Z") / 1000);

function tmpBase(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "vt-window-"));
}

function fix(hex: string, tSec: number, lat: number, lon: number, altM: number | null,
             extra: Partial<AircraftPoint> = {}): AircraftPoint & { tSec: number } {
  return {
    tSec, icao24: hex, registration: (extra as any).registration ?? null,
    type: (extra as any).type ?? null, callsign: (extra as any).callsign,
    lat, lon, altitude_m: altM, on_ground: false,
    velocity_ms: null, heading: null, category: null,
    ...extra,
  } as AircraftPoint & { tSec: number };
}

test("lodStepSec: close zoom keeps everything, world zoom decimates hardest", () => {
  assert.equal(lodStepSec(12), 0);
  assert.equal(lodStepSec(9), 0);
  assert.equal(lodStepSec(7), 60);
  assert.equal(lodStepSec(5), 300);
  assert.equal(lodStepSec(3), 600);
  assert.equal(lodStepSec(1), 900);
});

test("lonInBBox handles ordinary boxes and the antimeridian seam", () => {
  assert.ok(lonInBBox(-100, -110, -90));
  assert.ok(!lonInBBox(-80, -110, -90));
  // seam box: 170..-170 wraps the dateline
  assert.ok(lonInBBox(175, 170, -170));
  assert.ok(lonInBBox(-175, 170, -170));
  assert.ok(!lonInBBox(0, 170, -170));
});

test("hourName matches the archive's hour-file basename convention", () => {
  assert.equal(hourName(T0), "2026-08-10-12");
});

test("window ∩ bbox: hexes outside the box or the time range never appear", async () => {
  const base = tmpBase();
  archiveAircraftAt([
    fix("aaaaaa", T0 + 60, 40.0, -100.0, 10000),
    fix("aaaaaa", T0 + 120, 40.1, -100.1, 10050),
    fix("bbbbbb", T0 + 60, 55.0, -100.0, 9000),   // north of the box
    fix("cccccc", T0 - 7200, 40.0, -100.0, 8000), // before the window
  ], base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
  });
  assert.equal(r.hexes_seen, 1);
  assert.equal(r.hexes[0].i, "aaaaaa");
  assert.equal(r.hexes[0].points.length, 2);
  assert.ok(r.coverage.complete);
  fs.rmSync(base, { recursive: true, force: true });
});

test("one stream pass multiplexes every hex — and carries rg/c/ty metadata", async () => {
  const base = tmpBase();
  archiveAircraftAt([
    fix("aaaaaa", T0 + 10, 40.0, -100.0, 10000, { registration: "N123AB", callsign: "TEST1", type: "C172" } as any),
    fix("dddddd", T0 + 20, 41.0, -101.0, 11000),
    fix("eeeeee", T0 + 30, 42.0, -102.0, null),
  ], base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
  });
  assert.equal(r.hexes_seen, 3);
  const a = r.hexes.find((h) => h.i === "aaaaaa")!;
  assert.equal(a.rg, "N123AB");
  assert.equal(a.ty, "C172");
  const e = r.hexes.find((h) => h.i === "eeeeee")!;
  assert.equal(e.points[0][3], null, "missing altitude stays null, never invented");
  fs.rmSync(base, { recursive: true, force: true });
});

test("same-second dedupe: the altitude-bearing fix wins (fullTrackAsync rule)", async () => {
  const base = tmpBase();
  archiveAircraftAt([
    fix("aaaaaa", T0 + 60, 40.0, -100.0, null),
    fix("aaaaaa", T0 + 60, 40.0001, -100.0001, 10000), // same second, has altitude
  ], base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
  });
  assert.equal(r.hexes[0].points.length, 1);
  assert.equal(r.hexes[0].points[0][3], 10000);
  fs.rmSync(base, { recursive: true, force: true });
});

test("LOD decimation: low zoom thins to the step, the LAST point always survives", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  for (let k = 0; k < 60; k++) fixes.push(fix("aaaaaa", T0 + k * 30, 40 + k * 0.01, -100, 10000));
  archiveAircraftAt(fixes, base);
  const close = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
  });
  assert.equal(close.hexes[0].points.length, 60, "zoom 10: every fix");
  const wide = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 5, baseDir: base,
  });
  const pts = wide.hexes[0].points;
  assert.ok(pts.length < 10, `zoom 5 decimates 30s fixes to >=300s spacing (${pts.length})`);
  assert.equal(pts[pts.length - 1][0], T0 + 59 * 30, "track end survives decimation");
  for (let k = 1; k < pts.length - 1; k++) {
    assert.ok(pts[k][0] - pts[k - 1][0] >= 300, "spacing respects the step");
  }
  fs.rmSync(base, { recursive: true, force: true });
});

test("hex cap is honest: hexes_seen counts everything, the note says zoom in", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  for (let k = 0; k < 8; k++) {
    const hex = `a${k}a${k}a${k}`;
    // more fixes for lower k → deterministic most-active-first ordering
    for (let j = 0; j < 10 - k; j++) fixes.push(fix(hex, T0 + j * 60 + k, 40 + k * 0.1, -100, 9000));
  }
  archiveAircraftAt(fixes, base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
    caps: { maxHexes: 3 },
  });
  assert.equal(r.hexes_seen, 8);
  assert.equal(r.hexes.length, 3);
  assert.match(r.note || "", /returned 3 of 8/);
  assert.equal(r.hexes[0].i, "a0a0a0", "most-active hex first");
  fs.rmSync(base, { recursive: true, force: true });
});

test("per-hex cap keeps the NEWEST points and flags truncation", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  for (let k = 0; k < 30; k++) fixes.push(fix("aaaaaa", T0 + k * 30, 40, -100, 9000));
  archiveAircraftAt(fixes, base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
    caps: { maxPointsPerHex: 5 },
  });
  const h = r.hexes[0];
  assert.equal(h.points.length, 5);
  assert.ok(h.truncated);
  assert.equal(h.raw_count, 30);
  assert.equal(h.points[4][0], T0 + 29 * 30, "newest end kept");
  fs.rmSync(base, { recursive: true, force: true });
});

test("file budget: newest-first scan reports honest partial coverage", async () => {
  const base = tmpBase();
  // three separate hours
  archiveAircraftAt([fix("aaaaaa", T0 + 10, 40, -100, 9000)], base);
  archiveAircraftAt([fix("bbbbbb", T0 + 3610, 40, -100, 9000)], base);
  archiveAircraftAt([fix("cccccc", T0 + 7210, 40, -100, 9000)], base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3 * 3600, zoom: 10, baseDir: base,
    caps: { maxFiles: 2 },
  });
  // newest two hours scanned; the oldest hour (aaaaaa) honestly missing
  assert.equal(r.hexes_seen, 2);
  assert.ok(!r.hexes.some((h) => h.i === "aaaaaa"));
  assert.equal(r.coverage.complete, false);
  assert.equal(r.coverage.files_scanned, 2);
  assert.ok(r.coverage.scanned_from > T0, "scanned_from narrows to what was streamed");
  assert.match(r.note || "", /scan budget hit/);
  fs.rmSync(base, { recursive: true, force: true });
});

test("gzipped hour files stream identically to plain ones", async () => {
  const base = tmpBase();
  archiveAircraftAt([
    fix("aaaaaa", T0 + 10, 40, -100, 9000),
    fix("aaaaaa", T0 + 40, 40.1, -100.1, 9100),
  ], base);
  // gzip the hour file in place (compressOldHours' output shape)
  const dir = path.join(base, "aircraft");
  const f = fs.readdirSync(dir).find((x) => x.endsWith(".jsonl"))!;
  const fp = path.join(dir, f);
  fs.writeFileSync(fp + ".gz", zlib.gzipSync(fs.readFileSync(fp)));
  fs.unlinkSync(fp);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
  });
  assert.equal(r.hexes_seen, 1);
  assert.equal(r.hexes[0].points.length, 2);
  fs.rmSync(base, { recursive: true, force: true });
});

test("empty archive and inverted windows answer honestly, never throw", async () => {
  const base = tmpBase();
  const empty = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
  });
  assert.equal(empty.hexes_seen, 0);
  assert.match(empty.note || "", /no archive yet/);
  const inverted = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0 + 3600, toSec: T0, zoom: 10, baseDir: base,
  });
  assert.equal(inverted.hexes_seen, 0);
  assert.match(inverted.note || "", /empty window/);
  fs.rmSync(base, { recursive: true, force: true });
});

test("default caps are the charter's stated bounds", () => {
  assert.equal(WINDOW_DEFAULT_CAPS.maxHexes, 300);
  assert.equal(WINDOW_DEFAULT_CAPS.maxPointsPerHex, 600);
  assert.equal(WINDOW_DEFAULT_CAPS.maxTotalPoints, 60_000);
  assert.equal(WINDOW_DEFAULT_CAPS.maxFiles, 192);
});
