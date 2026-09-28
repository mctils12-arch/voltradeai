import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { archiveAircraftAt, type AircraftPoint } from "./datacoreArchive";
import {
  readWindow, lodStepSec, lonInBBox, hourName, WINDOW_DEFAULT_CAPS, WINDOW_GAP_SEC,
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

/** T-4 vessels-parity fixture: writes vessel hour-file lines directly in the
 *  same shape archiveVessels() produces (server/datacoreArchive.ts), since
 *  there is no vessel equivalent of archiveAircraftAt's shouldWrite-bypass
 *  backfill writer — this is the read-side contract to pin, not a new
 *  production writer this test doesn't need. */
function writeVesselHour(base: string, hourStartSec: number,
                          rows: Array<{ mmsi: string; tSec: number; lat: number; lon: number; name?: string }>) {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const lines = rows.map((r) => JSON.stringify({
    t: r.tSec, i: r.mmsi, c: r.name, la: r.lat, lo: r.lon,
  }));
  fs.appendFileSync(path.join(dir, `${hourName(hourStartSec)}.jsonl`), lines.join("\n") + "\n");
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

test("stepSecOverride: T-2 explicit step overrides the zoom-derived LOD default", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  for (let k = 0; k < 60; k++) fixes.push(fix("aaaaaa", T0 + k * 30, 40 + k * 0.01, -100, 10000));
  archiveAircraftAt(fixes, base);
  // zoom 12 alone would keep every 30s fix (lodStepSec(12) === 0); an
  // explicit 900s override must thin it anyway.
  const overridden = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 12, stepSecOverride: 900, baseDir: base,
  });
  assert.equal(overridden.step_sec, 900);
  const pts = overridden.hexes[0].points;
  assert.ok(pts.length < 10, `900s override thins 30s fixes (${pts.length})`);
  for (let k = 1; k < pts.length - 1; k++) {
    assert.ok(pts[k][0] - pts[k - 1][0] >= 900, "spacing respects the override, not the zoom");
  }
  assert.equal(pts[pts.length - 1][0], T0 + 59 * 30, "track end still survives");
  fs.rmSync(base, { recursive: true, force: true });
});

test("stepSecOverride: 0 forces full fidelity even at a world zoom that would otherwise decimate hardest", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  for (let k = 0; k < 5; k++) fixes.push(fix("aaaaaa", T0 + k * 30, 40 + k * 0.01, -100, 10000));
  archiveAircraftAt(fixes, base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 1, stepSecOverride: 0, baseDir: base,
  });
  assert.equal(r.step_sec, 0);
  assert.equal(r.hexes[0].points.length, 5, "override=0 keeps every fix despite zoom 1 (lodStepSec would be 900)");
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

test("T-4 vessels parity: kind=\"vessels\" reads the vessels/ dir, mmsi as id, no altitude/rg/ty invented", async () => {
  const base = tmpBase();
  writeVesselHour(base, T0, [
    { mmsi: "244010352", tSec: T0 + 60, lat: 51.9, lon: 4.1, name: "MSC OSCAR" },
    { mmsi: "244010352", tSec: T0 + 120, lat: 51.91, lon: 4.11, name: "MSC OSCAR" },
    { mmsi: "366123456", tSec: T0 + 90, lat: 52.0, lon: 4.2 }, // no name broadcast
  ]);
  const r = await readWindow({
    kind: "vessels",
    bbox: { w: 3, s: 51, e: 5, n: 53 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
  });
  assert.equal(r.kind, "vessels");
  assert.equal(r.hexes_seen, 2);
  const ship = r.hexes.find((h) => h.i === "244010352")!;
  assert.ok(ship, "mmsi used as the track id, same field the aircraft path calls icao24");
  assert.equal(ship.c, "MSC OSCAR");
  assert.equal(ship.points.length, 2);
  const unnamed = r.hexes.find((h) => h.i === "366123456")!;
  assert.equal(unnamed.c, undefined, "missing name stays undefined, never invented");
  assert.equal(unnamed.points[0][3], null, "vessels carry no altitude — stays null like a missing aircraft reading");
  fs.rmSync(base, { recursive: true, force: true });
});

test("T-4 vessels parity: the honest hex-cap note says \"vessels\", not a hardcoded \"aircraft\"", async () => {
  const base = tmpBase();
  writeVesselHour(base, T0, [
    { mmsi: "100000001", tSec: T0 + 10, lat: 51.9, lon: 4.1 },
    { mmsi: "100000002", tSec: T0 + 10, lat: 51.9, lon: 4.1 },
    { mmsi: "100000003", tSec: T0 + 10, lat: 51.9, lon: 4.1 },
  ]);
  const r = await readWindow({
    kind: "vessels",
    bbox: { w: 3, s: 51, e: 5, n: 53 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base,
    caps: { maxHexes: 1 },
  });
  assert.match(r.note || "", /returned 1 of 3 vessels/);
  fs.rmSync(base, { recursive: true, force: true });
});

test("default caps are the charter's stated bounds", () => {
  assert.equal(WINDOW_DEFAULT_CAPS.maxHexes, 300);
  assert.equal(WINDOW_DEFAULT_CAPS.maxPointsPerHex, 600);
  assert.equal(WINDOW_DEFAULT_CAPS.maxTotalPoints, 60_000);
  assert.equal(WINDOW_DEFAULT_CAPS.maxFiles, 192);
});

// ── FLIGHT PROGRAM replay (2026-09-28): close approaches + honest gaps ───────

test("closeApproaches: computed on UN-decimated fixes, present even at a coarse step", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  // two aircraft 2 nm apart laterally, 500 ft vertically, 30 s fixes for 20 min
  for (let k = 0; k < 40; k++) {
    fixes.push(fix("aaaaaa", T0 + k * 30, 40.0, -100 + k * 0.05, 10668, { callsign: "AAL1" } as any));
    fixes.push(fix("bbbbbb", T0 + k * 30, 40.0 + 2 / 60, -100 + k * 0.05, 10820, { callsign: "UAL2" } as any));
    fixes.push(fix("cccccc", T0 + k * 30, 44.0, -95 + k * 0.05, 10668)); // far away
  }
  archiveAircraftAt(fixes, base);
  // step 3600: the returned points are ≥ 1 h apart — a scan on those would
  // find nothing under the ±90 s rule; the scan must use the raw fixes
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 3, stepSecOverride: 3600, baseDir: base,
  });
  assert.ok(r.hexes.every((h) => h.points.length <= 2));
  assert.ok(Array.isArray(r.closeApproaches));
  assert.equal(r.closeApproaches!.length, 1);
  const ca = r.closeApproaches![0];
  assert.equal(ca.a, "aaaaaa");
  assert.equal(ca.b, "bbbbbb");
  assert.equal(ca.ca, "AAL1");
  assert.ok(Math.abs(ca.horizNm - 2) < 0.05, `horiz ${ca.horizNm}`);
  assert.ok(Math.abs(ca.vertFt - 499) < 5, `vert ${ca.vertFt}`);
  assert.match(ca.basis, /per our recorded ADS-B data — not an official loss-of-separation report/);
  assert.ok(["high", "medium", "low"].includes(ca.confidence));
  assert.equal(r.closeApproachesMeta!.evaluated_hexes, 3);
  assert.equal(r.closeApproachesMeta!.found, 1);
  assert.equal(r.closeApproachesMeta!.capped, false);
  fs.rmSync(base, { recursive: true, force: true });
});

test("closeApproaches: checked over ALL hexes seen, not only the returned (capped) ones", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  // a busy hex that wins the hex cap, plus a quiet close pair that does not
  for (let k = 0; k < 100; k++) fixes.push(fix("aaaaaa", T0 + k * 30, 36.0, -105 + k * 0.01, 9000));
  for (let k = 0; k < 10; k++) {
    fixes.push(fix("dddddd", T0 + k * 30, 41.0, -100 + k * 0.05, 11000));
    fixes.push(fix("eeeeee", T0 + k * 30, 41.0 + 1 / 60, -100 + k * 0.05, 11000));
  }
  archiveAircraftAt(fixes, base);
  const r = await readWindow({
    bbox: { w: -110, s: 35, e: -90, n: 45 },
    fromSec: T0, toSec: T0 + 3600, zoom: 10, baseDir: base, caps: { maxHexes: 1 },
  });
  assert.equal(r.hexes.length, 1);
  assert.equal(r.hexes[0].i, "aaaaaa");
  assert.equal(r.closeApproaches!.length, 1, "the capped-out pair is still reported");
  assert.equal(r.closeApproaches![0].a, "dddddd");
  assert.equal(r.closeApproachesMeta!.evaluated_hexes, 3);
  fs.rmSync(base, { recursive: true, force: true });
});

test("closeApproaches: vessels never carry them; closeApproaches:false disables", async () => {
  const base = tmpBase();
  writeVesselHour(base, T0, [
    { mmsi: "111111111", tSec: T0 + 10, lat: 40, lon: -70 },
    { mmsi: "222222222", tSec: T0 + 10, lat: 40.001, lon: -70 },
  ]);
  const v = await readWindow({ kind: "vessels", bbox: { w: -80, s: 30, e: -60, n: 50 }, fromSec: T0, toSec: T0 + 3600, zoom: 8, baseDir: base });
  assert.equal(v.closeApproaches, undefined);
  archiveAircraftAt([fix("aaaaaa", T0 + 10, 40, -100, 10000)], base);
  const off = await readWindow({ bbox: { w: -110, s: 35, e: -90, n: 45 }, fromSec: T0, toSec: T0 + 3600, zoom: 8, baseDir: base, closeApproaches: false });
  assert.equal(off.closeApproaches, undefined);
  fs.rmSync(base, { recursive: true, force: true });
});

test("gaps: a REAL raw hole > 10 min is marked; decimation spacing is not", async () => {
  const base = tmpBase();
  const fixes: Array<AircraftPoint & { tSec: number }> = [];
  // 30 s fixes for 20 min, a 25-min signal loss, then 30 s fixes for 10 min
  for (let k = 0; k < 40; k++) fixes.push(fix("aaaaaa", T0 + k * 30, 40 + k * 0.01, -100, 10000));
  const resume = T0 + 39 * 30 + 25 * 60;
  for (let k = 0; k < 20; k++) fixes.push(fix("aaaaaa", resume + k * 30, 41 + k * 0.01, -100, 10000));
  archiveAircraftAt(fixes, base);
  for (const stepSecOverride of [0, 60, 900]) {
    const r = await readWindow({
      bbox: { w: -110, s: 35, e: -90, n: 45 }, fromSec: T0, toSec: T0 + 7200, zoom: 10,
      stepSecOverride, baseDir: base,
    });
    const h = r.hexes[0];
    assert.ok(Array.isArray(h.gaps) && h.gaps.length === 1, `step ${stepSecOverride}: one gap (${JSON.stringify(h.gaps)})`);
    const idx = h.points.findIndex((p) => p[0] === h.gaps![0]);
    assert.ok(idx > 0, "gap marks a returned point");
    assert.ok(h.points[idx][0] >= resume, "the marked point is on the far side of the hole");
    assert.ok(h.points[idx - 1][0] <= T0 + 39 * 30, "…and its predecessor on the near side");
  }
  // continuous 30 s track decimated to 15-min steps: no gap claimed
  const base2 = tmpBase();
  const cont: Array<AircraftPoint & { tSec: number }> = [];
  for (let k = 0; k < 120; k++) cont.push(fix("bbbbbb", T0 + k * 30, 40 + k * 0.01, -100, 10000));
  archiveAircraftAt(cont, base2);
  const r2 = await readWindow({ bbox: { w: -110, s: 35, e: -90, n: 45 }, fromSec: T0, toSec: T0 + 7200, zoom: 3, stepSecOverride: 900, baseDir: base2 });
  assert.equal(r2.hexes[0].gaps, undefined);
  assert.equal(WINDOW_GAP_SEC, 600);
  fs.rmSync(base, { recursive: true, force: true });
  fs.rmSync(base2, { recursive: true, force: true });
});
