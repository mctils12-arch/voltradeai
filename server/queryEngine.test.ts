// ANALYST CONSOLE W4 battery: cross-layer geo-temporal query over synthetic
// archive dirs shaped exactly like the real writers emit (datacoreArchive
// hourly position lines {t,i,la,lo}; nasaFirms FireDetection; nwsAlerts
// AlertRec; usgsWater GaugeObs day files). Same fixture pattern as
// siteTimeline.test.ts.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  queryWindow, queryWindowCached, _resetQueryCache,
  SUPPORTED_LAYERS, RADIUS_KM_CAP, DAYS_CAP, TOP_ENTITIES_CAP, EVENTS_CAP,
  querySnapshot, snapshotWindow, SNAPSHOT_POINTS_CAP,
} from "./queryEngine";
import { RAW_RETENTION_DAYS } from "./datacoreArchive";

beforeEach(() => _resetQueryCache());

const LAT = 35.94, LON = -96.74;           // Cushing OK
const NOW = Date.parse("2026-07-05T12:00:00Z");
const DAY = "2026-07-05";
const T0 = Math.floor(NOW / 1000) - 3600;  // 11:00Z, inside today

function tmpBase(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "qe-"));
}

/** Hourly position file, real writer naming: YYYY-MM-DD-HH.jsonl(.gz). */
function writeHour(base: string, kind: string, stamp: string, recs: any[], gz = false) {
  const dir = path.join(base, kind);
  fs.mkdirSync(dir, { recursive: true });
  const body = recs.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${stamp}.jsonl.gz`), zlib.gzipSync(body));
  else fs.writeFileSync(path.join(dir, `${stamp}.jsonl`), body);
}

/** Daily event file, real writer naming: YYYY-MM-DD.jsonl(.gz). */
function writeDay(base: string, kind: string, day: string, recs: any[], gz = false) {
  const dir = path.join(base, kind);
  fs.mkdirSync(dir, { recursive: true });
  const body = recs.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(body));
  else fs.writeFileSync(path.join(dir, `${day}.jsonl`), body);
}

test("counts points inside radius+window only; outside radius or window excluded", async () => {
  const base = tmpBase();
  writeHour(base, "aircraft", "2026-07-05-11", [
    { t: T0, i: "aaa", la: 35.95, lo: -96.75 },       // ~2 km: in
    { t: T0 + 60, i: "bbb", la: 35.96, lo: -96.73 },  // ~2 km: in
    { t: T0 + 120, i: "ccc", la: 51.0, lo: 0.0 },     // far: out
  ]);
  // file named INSIDE the window but record timestamp outside it — record-level guard
  const oldT = Math.floor((NOW - 10 * 86400_000) / 1000);
  writeHour(base, "aircraft", "2026-07-04-09", [
    { t: oldT, i: "ddd", la: 35.95, lo: -96.75 },
  ]);
  // file named outside the window — must not even be opened for counting
  writeHour(base, "aircraft", "2026-06-20-11", [
    { t: oldT, i: "eee", la: 35.95, lo: -96.75 },
  ]);
  const r = await queryWindow({ lat: LAT, lon: LON, layers: ["aircraft"] }, base, NOW);
  assert.equal(r.layers.aircraft.points, 2);
  assert.equal(r.layers.aircraft.freshness, new Date((T0 + 60) * 1000).toISOString());
  assert.equal(r.layers.aircraft.provenance.includes("archive"), true);
});

test("byDay has absent days absent, never zero-filled", async () => {
  const base = tmpBase();
  writeHour(base, "aircraft", "2026-07-05-11", [
    { t: T0, i: "aaa", la: 35.95, lo: -96.75 },
  ]);
  const r = await queryWindow({ lat: LAT, lon: LON, layers: ["aircraft"] }, base, NOW);
  assert.deepEqual(Object.keys(r.layers.aircraft.byDay), [DAY]);
  assert.equal(r.layers.aircraft.byDay[DAY], 1);
});

test("topEntities ranked by points with first/last seen, capped at 10", async () => {
  const base = tmpBase();
  // 12 entities, entity ek gets k+1 points — e11 (12 pts) must rank first
  const recs: any[] = [];
  for (let k = 0; k < 12; k++) {
    for (let j = 0; j <= k; j++) {
      recs.push({ t: T0 + k * 100 + j, i: `e${k}`, la: 35.95, lo: -96.75 });
    }
  }
  writeHour(base, "vessels", "2026-07-05-11", recs);
  const r = await queryWindow({ lat: LAT, lon: LON, layers: ["vessels"] }, base, NOW);
  const top = r.layers.vessels.topEntities!;
  assert.equal(top.length, TOP_ENTITIES_CAP);
  assert.equal(r.caps.topEntities, TOP_ENTITIES_CAP, "cap stated in envelope");
  assert.equal(top[0].id, "e11");
  assert.equal(top[0].points, 12);
  assert.equal(top[0].firstSeen, new Date((T0 + 1100) * 1000).toISOString());
  assert.equal(top[0].lastSeen, new Date((T0 + 1111) * 1000).toISOString());
  assert.ok(!top.some((e) => e.id === "e0" || e.id === "e1"), "two smallest fall off the cap");
});

test("alert events surface newest-first with the 50 cap stated", async () => {
  const base = tmpBase();
  const many = Array.from({ length: 60 }, (_, i) => ({
    id: `a${i}`, event: `Alert ${i}`, severity: "Minor",
    sent: `2026-07-05T${String(Math.floor(i / 3)).padStart(2, "0")}:${String((i % 3) * 20).padStart(2, "0")}:00Z`,
    lat: 35.94, lon: -96.74,
  }));
  writeDay(base, "nwsalerts", DAY, many);
  const r = await queryWindow({ lat: LAT, lon: LON, layers: ["alerts"] }, base, NOW);
  const a = r.layers.alerts;
  assert.equal(a.points, 60, "points counts everything matched, before the events cap");
  assert.equal(a.events!.length, EVENTS_CAP);
  assert.equal(r.caps.events, EVENTS_CAP, "cap stated in envelope");
  assert.equal(a.events![0].label, "Alert 59", "newest first");
  assert.equal(a.events![0].severity, "Minor");
  assert.equal(a.freshness, "2026-07-05T19:40:00Z");
});

test("unknown layer names are filtered with the rejection stated in the envelope", async () => {
  const base = tmpBase();
  const r = await queryWindow({ lat: LAT, lon: LON, layers: ["aircraft", "submarines"] }, base, NOW);
  assert.deepEqual(r.query.layers, ["aircraft"]);
  assert.deepEqual(r.rejected_layers, ["submarines"]);
  assert.deepEqual(Object.keys(r.layers), ["aircraft"]);
});

test("caps enforced and visible: radiusKm clamps to 250, days clamps to raw retention", async () => {
  const base = tmpBase();
  const r = await queryWindow({ lat: LAT, lon: LON, radiusKm: 9999, days: 365, layers: ["aircraft"] }, base, NOW);
  assert.equal(r.query.radiusKm, RADIUS_KM_CAP);
  assert.equal(r.query.radiusKm, 250);
  assert.equal(r.query.days, DAYS_CAP);
  assert.equal(r.query.days, RAW_RETENTION_DAYS);
  assert.deepEqual(r.caps, { radiusKm: 250, days: RAW_RETENTION_DAYS, topEntities: 10, events: 50 });
});

test("gzipped day files are read for both position and event layers", async () => {
  const base = tmpBase();
  writeHour(base, "trains", "2026-07-05-10", [
    { t: T0 - 3600, i: "FI-62", co: "FI", la: 35.95, lo: -96.75 },
  ], true);
  writeDay(base, "fires", DAY, [
    { id: "f1", lat: 35.9, lon: -96.7, confidence: "high", frp: 12.5, acq_date: DAY, acq_time: "0930" },
  ], true);
  const r = await queryWindow({ lat: LAT, lon: LON, layers: ["trains", "fires"] }, base, NOW);
  assert.equal(r.layers.trains.points, 1);
  assert.equal(r.layers.trains.topEntities![0].id, "FI-62");
  assert.equal(r.layers.fires.points, 1);
  assert.equal(r.layers.fires.events![0].label, "Fire detection (high confidence)");
  assert.equal(r.layers.fires.events![0].value, 12.5);
  assert.equal(r.layers.fires.freshness, "2026-07-05T09:30:00Z");
});

test("cache: hit within TTL returns the same object without re-scanning; reset and TTL expiry re-scan", async () => {
  const base = tmpBase();
  writeHour(base, "aircraft", "2026-07-05-11", [
    { t: T0, i: "aaa", la: 35.95, lo: -96.75 },
  ]);
  const req = { lat: LAT, lon: LON, layers: ["aircraft"] };
  const r1 = await queryWindowCached(req, base, NOW);
  assert.equal(r1.layers.aircraft.points, 1);
  // mutate the fixture: a re-scan would now see 2 points
  fs.appendFileSync(path.join(base, "aircraft", "2026-07-05-11.jsonl"),
    JSON.stringify({ t: T0 + 60, i: "bbb", la: 35.95, lo: -96.75 }) + "\n");
  const r2 = await queryWindowCached(req, base, NOW + 60_000);
  assert.equal(r2, r1, "same object back — no re-scan inside TTL");
  assert.equal(r2.layers.aircraft.points, 1);
  // TTL expiry re-scans
  const r3 = await queryWindowCached(req, base, NOW + 6 * 60_000);
  assert.equal(r3.layers.aircraft.points, 2);
  // reset hook re-scans immediately
  fs.appendFileSync(path.join(base, "aircraft", "2026-07-05-11.jsonl"),
    JSON.stringify({ t: T0 + 120, i: "ccc", la: 35.95, lo: -96.75 }) + "\n");
  _resetQueryCache();
  const r4 = await queryWindowCached(req, base, NOW + 6 * 60_000);
  assert.equal(r4.layers.aircraft.points, 3);
});

test("empty/missing archive dirs degrade to zero results with freshness null, never throw", async () => {
  const r = await queryWindow({ lat: LAT, lon: LON }, "/nonexistent/base", NOW);
  assert.deepEqual(r.query.layers, [...SUPPORTED_LAYERS], "default = all supported layers");
  for (const name of SUPPORTED_LAYERS) {
    const l = r.layers[name];
    assert.equal(l.points, 0, name);
    assert.deepEqual(l.byDay, {}, name);
    assert.equal(l.freshness, null, name);
    assert.ok(l.provenance.length > 0, name);
  }
  assert.ok(r.note.includes("thinning"), "honesty note present");
});

test("invalid lat/lon rejected before any scan or cache touch", async () => {
  await assert.rejects(() => queryWindow({ lat: 999, lon: 0 }, "/nonexistent", NOW), /invalid lat\/lon/);
  await assert.rejects(() => queryWindowCached({ lat: 0, lon: NaN }, "/nonexistent", NOW), /invalid lat\/lon/);
});

test("gauge events carry name and value; zone-only (geo-less) records are excluded", async () => {
  const base = tmpBase();
  writeDay(base, "usgswater", DAY, [
    { site: "07153000", name: "Cimarron River nr Cushing", param: "00065", d: "2026-07-05T06:00:00-05:00", v: 4.2, q: "P", lat: 36.0, lon: -96.77 },
  ]);
  writeDay(base, "nwsalerts", DAY, [
    { id: "z1", event: "Zone Alert", severity: "Minor", sent: "2026-07-05T08:00:00Z", lat: null, lon: null },
  ]);
  const r = await queryWindow({ lat: LAT, lon: LON, layers: ["gauges", "alerts"] }, base, NOW);
  assert.equal(r.layers.gauges.points, 1);
  assert.equal(r.layers.gauges.events![0].label, "Cimarron River nr Cushing");
  assert.equal(r.layers.gauges.events![0].value, 4.2);
  assert.equal(r.layers.alerts.points, 0, "zone-only alerts have no geometry — excluded, stated at the stream");
  assert.equal(r.layers.alerts.freshness, null);
});

// ── W3 TIME SCRUBBER: querySnapshot ──────────────────────────────────────────

test("snapshot: position layer reads exactly the requested hour bucket", () => {
  const base = tmpBase();
  writeHour(base, "aircraft", "2026-07-05-11", [
    { t: T0, i: "aaa", la: 35.95, lo: -96.75 },
    { t: T0 + 60, i: "bbb", la: 36.0, lo: -97.0 },
  ]);
  // a different hour must not leak in
  writeHour(base, "aircraft", "2026-07-05-10", [
    { t: T0 - 3600, i: "ccc", la: 40.0, lo: -100.0 },
  ]);
  const r = querySnapshot("aircraft", "2026-07-05T11:30:00Z", undefined, base, NOW);
  assert.equal(r.mode, "position");
  assert.equal(r.bucket_at, "2026-07-05T11:00:00.000Z");
  assert.equal(r.count, 2);
  assert.deepEqual(r.data.map((p) => p.id).sort(), ["aaa", "bbb"]);
  assert.equal(r.freshness, new Date((T0 + 60) * 1000).toISOString());
  assert.equal(r.viewport_filtered, false);
  assert.equal(r.capped, false);
});

test("snapshot: event layer reads exactly the requested day bucket and keeps lat/lon", () => {
  const base = tmpBase();
  writeDay(base, "fires", DAY, [
    { id: "f1", lat: 35.9, lon: -96.7, confidence: "high", frp: 12.5, acq_date: DAY, acq_time: "0930" },
  ]);
  const r = querySnapshot("fires", "2026-07-05T11:00:00Z", undefined, base, NOW);
  assert.equal(r.mode, "event");
  assert.equal(r.bucket_at, "2026-07-05T00:00:00.000Z");
  assert.equal(r.count, 1);
  assert.equal(r.data[0].lat, 35.9);
  assert.equal(r.data[0].lon, -96.7);
  assert.equal(r.data[0].label, "Fire detection (high confidence)");
  assert.equal(r.data[0].value, 12.5);
});

test("snapshot: bbox filters and states before/after counts; absent bbox leaves data unfiltered", () => {
  const base = tmpBase();
  writeHour(base, "vessels", "2026-07-05-11", [
    { t: T0, i: "in-box", la: 35.95, lo: -96.75 },
    { t: T0, i: "out-box", la: 10.0, lo: 10.0 },
  ]);
  const r = querySnapshot("vessels", "2026-07-05T11:15:00Z", "-100,30,-90,40", base, NOW);
  assert.equal(r.viewport_filtered, true);
  assert.equal(r.count_before_viewport, 2);
  assert.equal(r.count, 1);
  assert.equal(r.count_dropped_offscreen, 1);
  assert.equal(r.data[0].id, "in-box");

  const unfiltered = querySnapshot("vessels", "2026-07-05T11:15:00Z", undefined, base, NOW);
  assert.equal(unfiltered.viewport_filtered, false);
  assert.equal(unfiltered.count, 2);
});

test("snapshot: point cap is enforced and stated, never silent", () => {
  const base = tmpBase();
  const many = Array.from({ length: SNAPSHOT_POINTS_CAP + 50 }, (_, i) => ({ t: T0 + i, i: `p${i}`, la: 35.9, lo: -96.7 }));
  writeHour(base, "aircraft", "2026-07-05-11", many);
  const r = querySnapshot("aircraft", "2026-07-05T11:00:00Z", undefined, base, NOW);
  assert.equal(r.capped, true);
  assert.equal(r.count, SNAPSHOT_POINTS_CAP);
});

test("snapshot: unknown layer, invalid timestamp, and out-of-window all throw stated errors", () => {
  assert.throws(() => querySnapshot("submarines", "2026-07-05T11:00:00Z", undefined, "/nonexistent", NOW), /unknown layer/);
  assert.throws(() => querySnapshot("aircraft", "not-a-date", undefined, "/nonexistent", NOW), /invalid "at"/);
  const { minMs } = snapshotWindow(NOW);
  const tooOld = new Date(minMs - 3600_000).toISOString();
  assert.throws(() => querySnapshot("aircraft", tooOld, undefined, "/nonexistent", NOW), /outside the retained raw window/);
});

test("snapshot: missing archive dir degrades to zero points, never throws", () => {
  const r = querySnapshot("aircraft", "2026-07-05T11:00:00Z", undefined, "/nonexistent/base", NOW);
  assert.equal(r.count, 0);
  assert.equal(r.freshness, null);
  assert.deepEqual(r.data, []);
});

test("snapshotWindow bounds span exactly RAW_RETENTION_DAYS ending at now", () => {
  const { minMs, maxMs } = snapshotWindow(NOW);
  assert.equal(maxMs, NOW);
  assert.equal(maxMs - minMs, RAW_RETENTION_DAYS * 86400_000);
});
