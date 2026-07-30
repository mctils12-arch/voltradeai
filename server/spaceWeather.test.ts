// SWPC space-weather battery (open_questions.md SPACE WEATHER hypothesis,
// gate-1 archiver). Fixtures mirror the live services.swpc.noaa.gov shapes
// probed 2026-07-29: Kp rows, scales keyed "-1".."3", alert messages with
// embedded message codes, one-row solar-wind summaries, ovation 1° grid
// with 0..359 longitudes. Offline: fetch is injected, archive uses tmpdirs.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  parseKp, parseScales, parseSwpcAlerts, parseWindSummary, parseOvation,
  parseXray, classifyFlare,
  fetchSpaceWeather, archiveSpaceWeather, gzipOldSpaceWeatherDays, conditionsRow,
} from "./spaceWeather";

const KP_FIX = [
  { time_tag: "2026-07-29T06:00:00", Kp: 1.67, a_running: 6, station_count: 8 },
  { time_tag: "2026-07-29T09:00:00", Kp: 3.33, a_running: 18, station_count: 8 },
  { time_tag: "bad", Kp: "not-a-number" },
];

const SCALES_FIX = {
  "0": { DateStamp: "2026-07-29", TimeStamp: "12:37:00", R: { Scale: "1", Text: "minor", MinorProb: null, MajorProb: null }, S: { Scale: "0", Text: "none", Prob: null }, G: { Scale: "2", Text: "moderate" } },
  "1": { DateStamp: "2026-07-29", TimeStamp: "12:37:00", R: { Scale: null, Text: null, MinorProb: "30", MajorProb: "1" }, S: { Scale: null, Text: null, Prob: "1" }, G: { Scale: "0", Text: "none" } },
  "2": { DateStamp: "2026-07-30", TimeStamp: "00:00:00", R: { Scale: null, Text: null, MinorProb: "25", MajorProb: "1" }, S: { Scale: null, Text: null, Prob: "1" }, G: { Scale: "1", Text: "minor" } },
  "3": { DateStamp: "2026-07-31", TimeStamp: "00:00:00", R: { Scale: null, Text: null, MinorProb: "25", MajorProb: "1" }, S: { Scale: null, Text: null, Prob: "1" }, G: { Scale: "0", Text: "none" } },
  "-1": { DateStamp: "2026-07-28", TimeStamp: "12:37:00", R: { Scale: "0", Text: "none", MinorProb: null, MajorProb: null }, S: { Scale: "0", Text: "none", Prob: null }, G: { Scale: "0", Text: "none" } },
};

const ALERTS_FIX = [
  {
    product_id: "EF3A",
    issue_datetime: "2026-07-27 14:53:19.230",
    message: "Space Weather Message Code: ALTEF3\r\nSerial Number: 3722\r\nIssue Time: 2026 Jul 27 1453 UTC\r\n\r\nCONTINUED ALERT: Electron 2MeV Integral Flux exceeded 1,000pfu\nBegin Time: 2026 Jul 26 1534 UTC",
  },
  { product_id: "K05W", issue_datetime: "2026-07-27 02:11:00.000", message: "Space Weather Message Code: WARK05\r\nSerial Number: 2101\r\n\r\nWARNING: Geomagnetic K-index of 5 expected" },
  { product_id: "noMsg", issue_datetime: "2026-07-27 00:00:00.000" }, // dropped: no message
];

const XRAY_FIX = [
  { time_tag: "2026-07-29T12:00:00Z", satellite: 18, energy: "0.05-0.4nm", flux: 1.2e-7, electron_contaminaton: false },
  { time_tag: "2026-07-29T12:00:00Z", satellite: 18, energy: "0.1-0.8nm", flux: 3.4e-6, electron_contaminaton: false },
  { time_tag: "2026-07-29T12:01:00Z", energy: "0.1-0.8nm", flux: 5.1e-6 },
  { time_tag: "bad", flux: "not-a-number", energy: "0.1-0.8nm" },
  { time_tag: "2026-07-29T12:02:00Z", energy: "0.1-0.8nm" }, // no flux -> dropped
];

// 1° grid rows: [lon 0..359, lat, prob%]
const OVATION_FIX = {
  "Observation Time": "2026-07-29T12:30:00Z",
  "Forecast Time": "2026-07-29T13:51:00Z",
  coordinates: [
    [0, 64, 12], [1, 64, 8],     // same 2° cell → max 12
    [0, 65, 3],                   // adjacent lat, same 2°-cell as above (64..66? no: floor(65/2)*2=64) → folds in
    [350, 66, 5],                 // lon 350 → -10
    [200, -70, 40],               // lon 200 → -160, southern oval
    [10, 40, 1],                  // below minVal 2 → dropped from cells but counts for max? (1 < 40 so no)
    [11, 41, 0],
  ],
};

test("parseKp: maps rows, skips malformed, carries vintage", () => {
  const recs = parseKp(KP_FIX, "2026-07-29");
  assert.equal(recs.length, 2);
  assert.deepEqual(recs[1], { t: "2026-07-29T09:00:00", kp: 3.33, a: 18, n: 8, rt: "2026-07-29" });
});

test("parseScales: observed current vs forecast rows never blended", () => {
  const { current, forecast } = parseScales(SCALES_FIX);
  assert.ok(current);
  assert.equal(current!.kind, "current");
  assert.equal(current!.r, "1");
  assert.equal(current!.g, "2");
  assert.equal(current!.rMinorProb, null, "observed row carries no probabilities");
  assert.equal(forecast.length, 3);
  assert.equal(forecast[0].kind, "forecast");
  assert.equal(forecast[0].r, null, "forecast R is a probability, never a scale reading");
  assert.equal(forecast[0].rMinorProb, 30);
  assert.equal(forecast[1].g, "1", "G forecast scale carried as forecast");
});

test("parseSwpcAlerts: id = product+issue, code/serial/title extracted, no-message rows dropped", () => {
  const recs = parseSwpcAlerts(ALERTS_FIX, "2026-07-29");
  assert.equal(recs.length, 2);
  assert.equal(recs[0].id, "EF3A:2026-07-27 14:53:19.230");
  assert.equal(recs[0].code, "ALTEF3");
  assert.equal(recs[0].serial, 3722);
  assert.match(recs[0].title!, /^CONTINUED ALERT: Electron/);
  assert.equal(recs[1].code, "WARK05");
  assert.match(recs[1].title!, /^WARNING: Geomagnetic/);
});

test("parseWindSummary: one-row summaries; missing feed -> nulls, never fabricated", () => {
  const w = parseWindSummary([{ proton_speed: 329, time_tag: "2026-07-29T12:32:00Z" }], [{ bt: 4, bz_gsm: -6, time_tag: "2026-07-29T12:34:00Z" }]);
  assert.equal(w.speedKms, 329);
  assert.equal(w.bzNt, -6);
  assert.equal(w.magAt, "2026-07-29T12:34:00Z");
  const empty = parseWindSummary(null, undefined);
  assert.equal(empty.speedKms, null);
  assert.equal(empty.btNt, null);
});

test("parseOvation: lon normalized, 2-degree MAX aggregation, threshold applied, times carried", () => {
  const g = parseOvation(OVATION_FIX, 2, 2);
  assert.equal(g.obs, "2026-07-29T12:30:00Z");
  assert.equal(g.forecast, "2026-07-29T13:51:00Z");
  assert.equal(g.max, 40, "max reads the FULL grid before thresholding");
  const at = (lonW: number, latS: number) => g.cells.find((c) => c[0] === lonW && c[1] === latS);
  assert.equal(at(0, 64)![2], 12, "aggregation takes max, never mean — thinning must not dim the oval");
  assert.equal(at(-10, 66)![2], 5, "lon 350 -> -10");
  assert.equal(at(-160, -70)![2], 40, "lon 200 -> -160, southern hemisphere kept");
  assert.equal(at(10, 40), undefined, "below-threshold cells dropped");
  for (const [lonW] of g.cells) assert.ok(lonW >= -180 && lonW < 180);
});

test("parseXray: maps rows, skips malformed/no-flux, normalizes NOAA's misspelled contamination field", () => {
  const recs = parseXray(XRAY_FIX);
  assert.equal(recs.length, 3);
  assert.deepEqual(recs[0], { time_tag: "2026-07-29T12:00:00Z", satellite: 18, energy: "0.05-0.4nm", flux: 1.2e-7, electronContamination: false });
  assert.equal(recs[2].satellite, null, "missing satellite -> null, not fabricated");
});

test("classifyFlare: official GOES A/B/C/M/X boundary table", () => {
  assert.equal(classifyFlare(3.4e-6).label, "C3.4");
  assert.equal(classifyFlare(1e-4).cls, "X");
  assert.equal(classifyFlare(9.9e-5).cls, "M");
  assert.equal(classifyFlare(1e-5).cls, "M");
  assert.equal(classifyFlare(1e-6).cls, "C");
  assert.equal(classifyFlare(1e-7).cls, "B");
  assert.equal(classifyFlare(5e-9).cls, "A", "below A floor still classifies as A, fractional magnitude");
});

test("fetchSpaceWeather: one dead feed never blanks the rest", async () => {
  const fetchImpl = async (url: string) => {
    if (url.includes("ovation")) return { ok: false, status: 503, text: async () => "" };
    if (url.includes("k-index")) return { ok: true, status: 200, text: async () => JSON.stringify(KP_FIX) };
    if (url.includes("noaa-scales")) return { ok: true, status: 200, text: async () => JSON.stringify(SCALES_FIX) };
    if (url.includes("alerts")) return { ok: true, status: 200, text: async () => JSON.stringify(ALERTS_FIX) };
    if (url.includes("solar-wind-speed")) return { ok: true, status: 200, text: async () => JSON.stringify([{ proton_speed: 400, time_tag: "2026-07-29T12:00:00Z" }]) };
    if (url.includes("xrays")) return { ok: true, status: 200, text: async () => JSON.stringify(XRAY_FIX) };
    return { ok: true, status: 200, text: async () => JSON.stringify([{ bt: 5, bz_gsm: 2, time_tag: "2026-07-29T12:00:00Z" }]) };
  };
  const pull = await fetchSpaceWeather(fetchImpl as any, Date.parse("2026-07-29T13:00:00Z"));
  assert.equal(pull.aurora, null);
  assert.equal(pull.kp.length, 2);
  assert.ok(pull.scales.current);
  assert.equal(pull.alerts.length, 2);
  assert.equal(pull.wind.speedKms, 400);
  assert.equal(pull.xray.length, 3);
  assert.equal(pull.errors.length, 1);
  assert.match(pull.errors[0], /503/);
});

test("archive: kp dedup by time_tag, alerts by id, conditions by upstream stamp, xray by time_tag+energy; gz lifecycle", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "swpc-"));
  const now = Date.parse("2026-07-29T13:00:00Z");
  const pull = {
    kp: parseKp(KP_FIX, "2026-07-29"),
    scales: parseScales(SCALES_FIX),
    alerts: parseSwpcAlerts(ALERTS_FIX, "2026-07-29"),
    wind: parseWindSummary([{ proton_speed: 329, time_tag: "t1" }], [{ bt: 4, bz_gsm: 1, time_tag: "t2" }]),
    aurora: parseOvation(OVATION_FIX),
    xray: parseXray(XRAY_FIX),
    errors: [],
  };
  const first = archiveSpaceWeather(pull as any, base, now);
  assert.deepEqual(first, { kp: 2, alerts: 2, conditions: 1, xray: 3 });
  // identical re-poll: nothing new anywhere (conditions stamp unchanged)
  const again = archiveSpaceWeather(pull as any, base, now + 60_000);
  assert.deepEqual(again, { kp: 0, alerts: 0, conditions: 0, xray: 0 });
  // a new Kp bin arrives -> exactly that row + a new conditions stamp
  const pull2 = { ...pull, kp: [...pull.kp, { t: "2026-07-29T12:00:00", kp: 2.0, a: 9, n: 8, rt: "2026-07-29" }] };
  const third = archiveSpaceWeather(pull2 as any, base, now + 120_000);
  assert.deepEqual(third, { kp: 1, alerts: 0, conditions: 1, xray: 0 });
  const dir = path.join(base, "spaceweather");
  const kpLines = fs.readFileSync(path.join(dir, "kp-2026-07-29.jsonl"), "utf8").trim().split("\n");
  assert.equal(kpLines.length, 3);
  const xrayLines = fs.readFileSync(path.join(dir, "xray-2026-07-29.jsonl"), "utf8").trim().split("\n");
  assert.equal(xrayLines.length, 3, "same time_tag, different energy -> both kept (composite key)");
  // gz: files older than 2 days compress and remain readable
  fs.writeFileSync(path.join(dir, "kp-2026-07-25.jsonl"), JSON.stringify({ t: "old" }) + "\n");
  const n = gzipOldSpaceWeatherDays(base, now);
  assert.equal(n, 1);
  assert.ok(!fs.existsSync(path.join(dir, "kp-2026-07-25.jsonl")));
  const gz = zlib.gunzipSync(fs.readFileSync(path.join(dir, "kp-2026-07-25.jsonl.gz"))).toString("utf8");
  assert.match(gz, /"old"/);
  fs.rmSync(base, { recursive: true, force: true });
});

test("conditionsRow: composite stamp keys the dedup; latest Kp row wins", () => {
  const pull = {
    kp: parseKp(KP_FIX, "2026-07-29"),
    scales: parseScales(SCALES_FIX),
    alerts: [],
    wind: parseWindSummary([{ proton_speed: 329, time_tag: "tS" }], [{ bt: 4, bz_gsm: 1, time_tag: "tM" }]),
    aurora: parseOvation(OVATION_FIX),
    xray: [],
    errors: [],
  };
  const row = conditionsRow(pull as any, Date.parse("2026-07-29T13:00:00Z"));
  assert.equal(row.kp, 3.33, "latest 3-hour bin");
  assert.equal(row.g, "2");
  assert.equal(row.auroraMax, 40);
  assert.ok(row.id.includes("2026-07-29T09:00:00") && row.id.includes("tS") && row.id.includes("tM"));
});
