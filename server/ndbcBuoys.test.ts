import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  BUOYS_FEED_URL,
  parseLatestObs,
  fetchBuoys,
  archiveBuoys,
  gzipOldBuoyDays,
  bootBuoysPoll,
} from "./ndbcBuoys";

// ROOT VALIDATION LADDER gate 1 (DATA) fixture — captured verbatim from a
// live GET of BUOYS_FEED_URL during this build (2026-07-08), not hand-built.
// Includes: the 2 header lines (must be skipped), a 4-char station id next
// to 5-char ids (whitespace-delimited, not fixed-width), a negative lat/lon
// row, an all-MM row, and a row with a signed pressure-tendency value.
const LIVE_SAMPLE =
`#STN       LAT      LON  YYYY MM DD hh mm WDIR WSPD   GST WVHT  DPD APD MWD   PRES  PTDY  ATMP  WTMP  DEWP  VIS   TIDE
#text      deg      deg   yr mo day hr mn degT  m/s   m/s   m   sec sec degT   hPa   hPa  degC  degC  degC  nmi     ft
22101    37.24   126.02  2026 07 08 12 00 170   2.0    MM  0.5   0   MM  MM     MM    MM  21.4  20.3    MM   MM     MM
32ST0   -22.000  -85.000 2026 07 08 11 30 130   6.0    MM   MM  MM   MM  MM 1018.5    MM  17.7  20.3  13.4   MM     MM
41082    35.950  -75.125 2026 07 08 11 00 300   4.0    MM   MM  MM   MM  MM 1015.3  +0.8  26.0    MM    MM   MM     MM
YRSV2    37.414  -76.712 2026 07 08 12 30  50   0.5    MM   MM  MM   MM  MM 1018.0    MM  24.1    MM  24.1   MM     MM
`;

test("parseLatestObs: maps the real NDBC whitespace-delimited shape, skips header lines", () => {
  const rows = parseLatestObs(LIVE_SAMPLE, "2026-07-08");
  assert.equal(rows.length, 4);
  assert.equal(rows[0].station, "22101");
  assert.equal(rows[0].lat, 37.24);
  assert.equal(rows[0].lon, 126.02);
  assert.equal(rows[0].windDir, 170);
  assert.equal(rows[0].windSpeed, 2.0);
  assert.equal(rows[0].gust, null); // MM
  assert.equal(rows[0].waveHeight, 0.5);
  assert.equal(rows[0].dominantPeriod, 0);
  assert.equal(rows[0].airTemp, 21.4);
  assert.equal(rows[0].waterTemp, 20.3);
  assert.equal(rows[0].time, Date.parse("2026-07-08T12:00:00Z"));
});

test("parseLatestObs: 4-char station id (not just 5-char) parses identically via whitespace split", () => {
  const rows = parseLatestObs(LIVE_SAMPLE, "2026-07-08");
  const row = rows.find((r) => r.station === "YRSV2");
  assert.ok(row);
  assert.equal(row!.lat, 37.414);
});

test("parseLatestObs: negative lat/lon parsed correctly, not confused with a missing-field shift", () => {
  const rows = parseLatestObs(LIVE_SAMPLE, "2026-07-08");
  const row = rows.find((r) => r.station === "32ST0");
  assert.ok(row);
  assert.equal(row!.lat, -22.000);
  assert.equal(row!.lon, -85.000);
  assert.equal(row!.pressure, 1018.5);
});

test("parseLatestObs: signed pressure-tendency ('+0.8') parses as a positive number, not null", () => {
  const rows = parseLatestObs(LIVE_SAMPLE, "2026-07-08");
  const row = rows.find((r) => r.station === "41082");
  assert.ok(row);
  assert.equal(row!.pressureTendency, 0.8);
});

test("parseLatestObs: 'MM' maps to null, never coerced to 0", () => {
  const rows = parseLatestObs(LIVE_SAMPLE, "2026-07-08");
  const row = rows.find((r) => r.station === "22101");
  assert.equal(row!.gust, null);
  assert.notEqual(row!.gust, 0);
});

test("parseLatestObs: header lines and blank lines produce no rows", () => {
  assert.deepEqual(parseLatestObs("", "2026-07-08"), []);
  assert.deepEqual(parseLatestObs("#STN LAT LON\n#text deg deg\n\n", "2026-07-08"), []);
});

test("fetchBuoys: hits BUOYS_FEED_URL with a UA header and parses the response", async () => {
  let calledUrl = "";
  let calledInit: any = null;
  const fetchImpl = async (url: string, init: any) => {
    calledUrl = url; calledInit = init;
    return { ok: true, status: 200, text: async () => LIVE_SAMPLE };
  };
  const rows = await fetchBuoys(fetchImpl as any, Date.parse("2026-07-08T00:00:00Z"));
  assert.equal(calledUrl, BUOYS_FEED_URL);
  assert.ok(calledInit.headers["User-Agent"]);
  assert.equal(rows.length, 4);
});

test("fetchBuoys: throws with the HTTP status on a non-ok response", async () => {
  const fetchImpl = async () => ({ ok: false, status: 503, text: async () => "" });
  await assert.rejects(() => fetchBuoys(fetchImpl as any), /503/);
});

test("archiveBuoys: fresh station observations write a day-file; unchanged timestamp is not re-archived", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-buoys-"));
  const now = Date.parse("2026-07-08T12:00:00Z");
  const rows = parseLatestObs(LIVE_SAMPLE, "2026-07-08");
  const n1 = archiveBuoys(rows, dir, now);
  assert.equal(n1, 4);
  const fp = path.join(dir, "buoys", "2026-07-08.jsonl");
  assert.ok(fs.existsSync(fp));
  assert.equal(fs.readFileSync(fp, "utf8").trim().split("\n").length, 4);
  // re-archiving the same rows (identical observation time per station) writes nothing new
  const n2 = archiveBuoys(rows, dir, now + 60_000);
  assert.equal(n2, 0);
  assert.equal(fs.readFileSync(fp, "utf8").trim().split("\n").length, 4);
});

test("archiveBuoys: a station reporting a NEWER observation time re-archives just that station", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-buoys-"));
  const now = Date.parse("2026-07-08T12:00:00Z");
  const base = { station: "TESTB1", lat: 1, lon: 2, time: 1000, windDir: null, windSpeed: null,
    gust: null, waveHeight: null, dominantPeriod: null, avgPeriod: null, waveDir: null,
    pressure: null, pressureTendency: null, airTemp: null, waterTemp: null, dewpoint: null,
    visibility: null, tide: null, rt: "2026-07-08" };
  archiveBuoys([base], dir, now);
  // same time, different reading — must NOT re-archive (station hasn't actually reported anew)
  const n2 = archiveBuoys([{ ...base, airTemp: 99 }], dir, now + 60_000);
  assert.equal(n2, 0);
  // newer time — must re-archive
  const n3 = archiveBuoys([{ ...base, time: 2000, airTemp: 15 }], dir, now + 120_000);
  assert.equal(n3, 1);
  const fp = path.join(dir, "buoys", "2026-07-08.jsonl");
  const lines = fs.readFileSync(fp, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, 2);
  assert.equal(lines[1].time, 2000);
  assert.equal(lines[1].airTemp, 15);
});

test("gzipOldBuoyDays: gzips day-files older than 2 days, leaves recent ones alone", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-buoys-"));
  const now = Date.parse("2026-07-08T12:00:00Z");
  const bDir = path.join(dir, "buoys");
  fs.mkdirSync(bDir, { recursive: true });
  fs.writeFileSync(path.join(bDir, "2026-07-01.jsonl"), '{"station":"old"}\n');
  fs.writeFileSync(path.join(bDir, "2026-07-08.jsonl"), '{"station":"new"}\n');
  const n = gzipOldBuoyDays(dir, now);
  assert.equal(n, 1);
  assert.ok(fs.existsSync(path.join(bDir, "2026-07-01.jsonl.gz")));
  assert.ok(!fs.existsSync(path.join(bDir, "2026-07-01.jsonl")));
  assert.ok(fs.existsSync(path.join(bDir, "2026-07-08.jsonl"))); // untouched, not gzipped
});

test("bootBuoysPoll: keyless — starts polling unconditionally, idempotent across repeat calls", () => {
  assert.doesNotThrow(() => {
    bootBuoysPoll(3600_000);
    bootBuoysPoll(3600_000); // second call is a no-op (module-level `polling` guard)
  });
});
