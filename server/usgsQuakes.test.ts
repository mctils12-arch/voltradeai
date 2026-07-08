import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  QUAKES_FEED_URL,
  parseQuakesGeoJson,
  fetchQuakes,
  archiveQuakes,
  gzipOldQuakeDays,
  bootQuakesPoll,
} from "./usgsQuakes";

// ROOT VALIDATION LADDER gate 1 (DATA) fixture — captured verbatim from a
// live GET of QUAKES_FEED_URL during this build (2026-07-08), not hand-built.
const LIVE_SAMPLE = {
  type: "FeatureCollection",
  metadata: {
    generated: 1783478127000,
    url: QUAKES_FEED_URL,
    title: "USGS Magnitude 2.5+ Earthquakes, Past Day",
    status: 200,
    api: "2.7.0",
    count: 2,
  },
  features: [
    {
      type: "Feature",
      properties: {
        mag: 5,
        place: "15 km NNE of Xunchang, China",
        time: 1783476528776,
        updated: 1783477428040,
        tz: null,
        url: "https://earthquake.usgs.gov/earthquakes/eventpage/us6000taui",
        detail: "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/us6000taui.geojson",
        felt: null, cdi: null, mmi: null, alert: null,
        status: "reviewed", tsunami: 0, sig: 385, net: "us", code: "6000taui",
        ids: ",us6000taui,", sources: ",us,", types: ",origin,phase-data,",
        nst: 80, dmin: 7.402, rms: 0.82, gap: 35, magType: "mww", type: "earthquake",
        title: "M 5.0 - 15 km NNE of Xunchang, China",
      },
      geometry: { type: "Point", coordinates: [104.7549, 28.5871, 10] },
      id: "us6000taui",
    },
    {
      type: "Feature",
      properties: {
        mag: 4.4,
        place: "64 km WSW of Jiquilillo, Nicaragua",
        time: 1783474635530,
        updated: 1783475492040,
        tz: null,
        url: "https://earthquake.usgs.gov/earthquakes/eventpage/us6000tauc",
        status: "reviewed", tsunami: 0, sig: 298, net: "us", code: "6000tauc",
        magType: "mb", type: "earthquake",
      },
      geometry: { type: "Point", coordinates: [-87.9449, 12.4263, 67.471] },
      id: "us6000tauc",
    },
  ],
};

test("parseQuakesGeoJson: maps real USGS feed shape (lon/lat/depth from geometry, id preserved)", () => {
  const rows = parseQuakesGeoJson(LIVE_SAMPLE, "2026-07-08");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].id, "us6000taui");
  assert.equal(rows[0].mag, 5);
  assert.equal(rows[0].place, "15 km NNE of Xunchang, China");
  assert.equal(rows[0].lon, 104.7549);
  assert.equal(rows[0].lat, 28.5871);
  assert.equal(rows[0].depth, 10);
  assert.equal(rows[0].time, 1783476528776);
  assert.equal(rows[0].tsunami, false);
  assert.equal(rows[0].magType, "mww");
  assert.equal(rows[0].rt, "2026-07-08");
});

test("parseQuakesGeoJson: sparse properties (no url/status extras) don't throw, missing fields null", () => {
  const rows = parseQuakesGeoJson(LIVE_SAMPLE, "2026-07-08");
  assert.equal(rows[1].id, "us6000tauc");
  assert.equal(rows[1].magType, "mb");
});

test("parseQuakesGeoJson: events with no id are dropped (nothing to dedup/archive against)", () => {
  const rows = parseQuakesGeoJson({ features: [{ properties: { mag: 3 }, geometry: { coordinates: [1, 2, 3] } }] }, "2026-07-08");
  assert.equal(rows.length, 0);
});

test("parseQuakesGeoJson: empty/missing features returns no rows", () => {
  assert.deepEqual(parseQuakesGeoJson({}, "2026-07-08"), []);
  assert.deepEqual(parseQuakesGeoJson({ features: [] }, "2026-07-08"), []);
});

test("parseQuakesGeoJson: tsunami flag accepts both numeric 1 and boolean true", () => {
  const j = { features: [
    { id: "a", properties: { tsunami: 1 }, geometry: { coordinates: [0, 0, 0] } },
    { id: "b", properties: { tsunami: true }, geometry: { coordinates: [0, 0, 0] } },
    { id: "c", properties: {}, geometry: { coordinates: [0, 0, 0] } },
  ] };
  const rows = parseQuakesGeoJson(j, "2026-07-08");
  assert.equal(rows[0].tsunami, true);
  assert.equal(rows[1].tsunami, true);
  assert.equal(rows[2].tsunami, false);
});

test("fetchQuakes: hits QUAKES_FEED_URL with a UA header and parses the response", async () => {
  let calledUrl = "";
  let calledInit: any = null;
  const fetchImpl = async (url: string, init: any) => {
    calledUrl = url; calledInit = init;
    return { ok: true, status: 200, text: async () => JSON.stringify(LIVE_SAMPLE) };
  };
  const rows = await fetchQuakes(fetchImpl as any, Date.parse("2026-07-08T00:00:00Z"));
  assert.equal(calledUrl, QUAKES_FEED_URL);
  assert.ok(calledInit.headers["User-Agent"]);
  assert.equal(rows.length, 2);
});

test("fetchQuakes: throws with the HTTP status on a non-ok response", async () => {
  const fetchImpl = async () => ({ ok: false, status: 503, text: async () => "" });
  await assert.rejects(() => fetchQuakes(fetchImpl as any), /503/);
});

test("archiveQuakes: fresh events write a day-file, dedup suppresses re-archiving unchanged rows", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quakes-"));
  const now = Date.parse("2026-07-08T12:00:00Z");
  const events = parseQuakesGeoJson(LIVE_SAMPLE, "2026-07-08");
  const n1 = archiveQuakes(events, dir, now);
  assert.equal(n1, 2);
  const fp = path.join(dir, "earthquakes", "2026-07-08.jsonl");
  assert.ok(fs.existsSync(fp));
  assert.equal(fs.readFileSync(fp, "utf8").trim().split("\n").length, 2);
  // re-archiving the exact same events (same `updated`) writes nothing new
  const n2 = archiveQuakes(events, dir, now + 60_000);
  assert.equal(n2, 0);
  assert.equal(fs.readFileSync(fp, "utf8").trim().split("\n").length, 2);
});

test("archiveQuakes: a revised `updated` timestamp for a known id re-archives it (magnitude/location review)", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quakes-"));
  const now = Date.parse("2026-07-08T12:00:00Z");
  // unique ids (not reused from LIVE_SAMPLE) — archiveQuakes's dedup state is
  // module-level (matches nasaFirms.ts's precedent), so a colliding id from
  // an earlier test in this same process would already be "seen".
  const base = { id: "us_test_revise_1", mag: 4.1, place: "test", lat: 1, lon: 2, depth: 3,
    time: 1, updated: 1000, tsunami: false, sig: null, net: "us", magType: "mb",
    type: "earthquake", status: "automatic", url: null, rt: "2026-07-08" };
  archiveQuakes([base], dir, now);
  const revised = { ...base, mag: 5.2, updated: base.updated + 1000, status: "reviewed" };
  const n2 = archiveQuakes([revised], dir, now + 60_000);
  assert.equal(n2, 1);
  const fp = path.join(dir, "earthquakes", "2026-07-08.jsonl");
  const lines = fs.readFileSync(fp, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, 2); // both the original and the revision, append-only
  assert.equal(lines[1].mag, 5.2);
  assert.equal(lines[1].status, "reviewed");
});

test("gzipOldQuakeDays: gzips day-files older than 2 days, leaves recent ones alone", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quakes-"));
  const now = Date.parse("2026-07-08T12:00:00Z");
  const qDir = path.join(dir, "earthquakes");
  fs.mkdirSync(qDir, { recursive: true });
  fs.writeFileSync(path.join(qDir, "2026-07-01.jsonl"), '{"id":"old"}\n');
  fs.writeFileSync(path.join(qDir, "2026-07-08.jsonl"), '{"id":"new"}\n');
  const n = gzipOldQuakeDays(dir, now);
  assert.equal(n, 1);
  assert.ok(fs.existsSync(path.join(qDir, "2026-07-01.jsonl.gz")));
  assert.ok(!fs.existsSync(path.join(qDir, "2026-07-01.jsonl")));
  assert.ok(fs.existsSync(path.join(qDir, "2026-07-08.jsonl"))); // untouched, not gzipped
});

test("bootQuakesPoll: keyless — starts polling unconditionally, idempotent across repeat calls", () => {
  assert.doesNotThrow(() => {
    bootQuakesPoll(3600_000);
    bootQuakesPoll(3600_000); // second call is a no-op (module-level `polling` guard)
  });
});
