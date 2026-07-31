// NOAA SWPC rtsw normalizer + feed-freshness battery (space-weather
// hardening Phases 0.3-0.5). Golden fixture at __fixtures__/rtsw_sample.json
// mirrors the live json/rtsw/rtsw_{wind,mag}_1m.json shape probed 2026-07-31:
// newest-first, THREE-plus spacecraft interleaved at overlapping time_tags
// (ACE/IMAP/SOLAR1 seen live, active on SOLAR1, DSCOVR absent — source is an
// OPEN string enum), overall_quality 2 rows carrying null proton_*, plus a
// stale tail of old timestamps. Offline: no network anywhere.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeRtswRows, latestActive, assessStaleness, noaaUtcMs } from "./spaceweatherNormalizer";
import {
  FEED_MAX_AGE_MS, computeFreshness, newestFeedAt,
  parseKp, parseScales, parseSwpcAlerts, parseWindSummary, parseOvation, parseXray,
} from "./spaceWeather";
import type { SwFeedKey } from "./spaceWeather";

const here = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE: unknown = JSON.parse(fs.readFileSync(path.join(here, "__fixtures__", "rtsw_sample.json"), "utf8"));
const NOW = Date.parse("2026-07-31T12:05:00Z"); // 3 min after the fixture's newest row

test("normalizeRtswRows: golden mapping of the live rtsw shape — NOAA names in, our names out", () => {
  const rows = normalizeRtswRows(FIXTURE);
  assert.equal(rows.length, 10, "every well-formed fixture row survives");
  // the active source's newest QUALITY row, mapped field by field
  assert.deepEqual(rows[2], {
    time_tag: "2026-07-31T12:01:00Z",
    source: "SOLAR1",
    active: true,
    quality: 0,
    speedKms: 412.3,
    densityPcc: 5.2,
    tempK: 98321,
    btNt: 6.0,
    bzNt: -2.1,
    phiGsm: 140.0,
    thetaGsm: -11.5,
  });
  // interleaving preserved: three spacecraft share time_tag 12:00 with different values
  const at1200 = rows.filter((r) => r.time_tag === "2026-07-31T12:00:00Z");
  assert.deepEqual(at1200.map((r) => r.source).sort(), ["ACE", "IMAP", "SOLAR1"]);
  assert.equal(new Set(at1200.map((r) => r.speedKms)).size, 3, "per-spacecraft values stay distinct");
});

test("nulls never propagate as NaN: quality-2 proton_* stay null; junk values land as null", () => {
  const rows = normalizeRtswRows(FIXTURE);
  const q2 = rows.find((r) => r.quality === 2 && r.source === "SOLAR1")!;
  assert.equal(q2.speedKms, null);
  assert.equal(q2.densityPcc, null);
  assert.equal(q2.tempK, null);
  assert.equal(q2.btNt, 6.1, "mag fields on the same row keep their readings");
  for (const row of rows) {
    for (const [k, v] of Object.entries(row)) {
      assert.ok(typeof v !== "number" || Number.isFinite(v), `${row.source}@${row.time_tag} ${k} is non-finite`);
    }
  }
  // defensive: junk numerics -> null, and an unknown spacecraft passes through
  const junk = normalizeRtswRows([
    { time_tag: "2026-07-31T12:00:00Z", source: "SOME-NEW-SAT", active: false, overall_quality: "x", proton_speed: "not-a-number", proton_density: "", bt: undefined },
  ]);
  assert.equal(junk.length, 1);
  assert.equal(junk[0].source, "SOME-NEW-SAT", "source is an OPEN enum — never a hardcoded spacecraft list");
  assert.equal(junk[0].quality, null);
  assert.equal(junk[0].speedKms, null);
  assert.equal(junk[0].densityPcc, null);
  assert.equal(junk[0].btNt, null);
});

test("normalizeRtswRows: non-arrays and rows without time_tag+source are rejected, not guessed at", () => {
  assert.deepEqual(normalizeRtswRows(null), []);
  assert.deepEqual(normalizeRtswRows(undefined), []);
  assert.deepEqual(normalizeRtswRows({}), []);
  assert.deepEqual(normalizeRtswRows("nope"), []);
  const rows = normalizeRtswRows([{ source: "ACE" }, { time_tag: "2026-07-31T12:00:00Z" }, 42, null]);
  assert.equal(rows.length, 0);
});

test("latestActive: the active source's newest QUALITY row wins over its newer quality-2/null-speed row", () => {
  const rows = normalizeRtswRows(FIXTURE);
  const hit = latestActive(rows)!;
  assert.ok(hit);
  assert.equal(hit.source, "SOLAR1", "only the active source can win");
  assert.equal(hit.time_tag, "2026-07-31T12:01:00Z", "the newer 12:02 row is quality 2 with null speed — skipped");
  assert.equal(hit.speedKms, 412.3);
  // requireQuality:false -> the newest active row wins even degraded, nulls intact
  const loose = latestActive(rows, { requireQuality: false })!;
  assert.equal(loose.time_tag, "2026-07-31T12:02:00Z");
  assert.equal(loose.quality, 2);
  assert.equal(loose.speedKms, null, "null speed served as null — never NaN, never fabricated");
  // no active rows at all -> null
  assert.equal(latestActive(rows.filter((r) => !r.active)), null);
  assert.equal(latestActive([]), null);
});

test("latestActive: ordering-independent — shuffled and reversed fixtures give the same answer", () => {
  const rows = normalizeRtswRows(FIXTURE);
  const expected = latestActive(rows)!;
  const expectedLoose = latestActive(rows, { requireQuality: false })!;
  // deterministic LCG shuffle — reproducible failures, no seed dependency
  let s = 42;
  const rand = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let trial = 0; trial < 5; trial++) {
    const shuffled = [...rows];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    assert.deepEqual(latestActive(shuffled), expected, `trial ${trial}: shuffle changed the strict answer`);
    assert.deepEqual(latestActive(shuffled, { requireQuality: false }), expectedLoose, `trial ${trial}: shuffle changed the loose answer`);
  }
  assert.deepEqual(latestActive([...rows].reverse()), expected, "oldest-first input gives the same answer");
});

test("assessStaleness: fresh passes, the fixture's stale tail flags, missing/garbage is stale with null age", () => {
  const rows = normalizeRtswRows(FIXTURE);
  const fresh = latestActive(rows)!;
  const f = assessStaleness(fresh.time_tag, 15 * 60_000, NOW);
  assert.equal(f.stale, false);
  assert.equal(f.ageMs, 4 * 60_000, "12:01 vs 12:05");
  const tailNewest = rows
    .filter((r) => r.time_tag.startsWith("2026-07-29"))
    .map((r) => r.time_tag)
    .sort()
    .pop()!;
  const st = assessStaleness(tailNewest, 15 * 60_000, NOW);
  assert.equal(st.stale, true, "the stale tail flags");
  assert.ok(st.ageMs! > 24 * 3_600_000);
  assert.deepEqual(assessStaleness(null, 15 * 60_000, NOW), { stale: true, ageMs: null });
  assert.deepEqual(assessStaleness("garbage", 15 * 60_000, NOW), { stale: true, ageMs: null });
});

test("noaaUtcMs: NOAA's Z-less time tags parse as UTC, never host-local", () => {
  assert.equal(noaaUtcMs("2026-07-31T12:00:00"), Date.parse("2026-07-31T12:00:00Z"));
  assert.equal(noaaUtcMs("2026-07-27 14:53:19.230"), Date.parse("2026-07-27T14:53:19.230Z"), "alerts-style space separator");
  assert.equal(noaaUtcMs("2026-07-31T12:00:00Z"), Date.parse("2026-07-31T12:00:00Z"), "explicit Z untouched");
  assert.equal(noaaUtcMs(null), null);
  assert.equal(noaaUtcMs(""), null);
  assert.equal(noaaUtcMs("not-a-time"), null);
});

test("freshness thresholds map per feed: 15 min wind summaries + X-ray, 6 h Kp/scales/alerts, 30 min OVATION", () => {
  assert.equal(FEED_MAX_AGE_MS.windSpeed, 15 * 60_000);
  assert.equal(FEED_MAX_AGE_MS.windMag, 15 * 60_000);
  assert.equal(FEED_MAX_AGE_MS.xray, 15 * 60_000);
  assert.equal(FEED_MAX_AGE_MS.kp, 6 * 3_600_000);
  assert.equal(FEED_MAX_AGE_MS.scales, 6 * 3_600_000);
  assert.equal(FEED_MAX_AGE_MS.alerts, 6 * 3_600_000);
  assert.equal(FEED_MAX_AGE_MS.ovation, 30 * 60_000);
  assert.deepEqual(
    Object.keys(FEED_MAX_AGE_MS).sort(),
    ["alerts", "kp", "ovation", "scales", "windMag", "windSpeed", "xray"],
    "exactly the seven archived feeds — no more, no fewer",
  );
});

test("computeFreshness: each feed ages against ITS OWN threshold; one stale feed trips anyStale", () => {
  const now = Date.parse("2026-07-29T14:00:00Z");
  const pull = {
    kp: parseKp(
      [
        { time_tag: "2026-07-29T12:00:00", Kp: 2.0, a_running: 9, station_count: 8 },
        { time_tag: "2026-07-29T09:00:00", Kp: 3.33, a_running: 18, station_count: 8 },
      ],
      "2026-07-29",
    ), // deliberately newest-FIRST: newest-picking must not trust array order
    scales: parseScales({
      "0": { DateStamp: "2026-07-29", TimeStamp: "12:37:00", R: { Scale: "0" }, S: { Scale: "0" }, G: { Scale: "1" } },
    }),
    alerts: parseSwpcAlerts(
      [{ product_id: "K05W", issue_datetime: "2026-07-29 10:11:00.000", message: "Space Weather Message Code: WARK05\r\nSerial Number: 2101\r\n\r\nWARNING: Geomagnetic K-index of 5 expected" }],
      "2026-07-29",
    ),
    wind: parseWindSummary(
      [{ proton_speed: 400, time_tag: "2026-07-29T13:40:00Z" }], // 20 min > 15 min -> stale
      [{ bt: 5, bz_gsm: 2, time_tag: "2026-07-29T13:55:00Z" }], // 5 min -> fresh
    ),
    aurora: parseOvation({
      "Observation Time": "2026-07-29T13:20:00Z", // 40 min > 30 min -> stale
      "Forecast Time": "2026-07-29T14:30:00Z",
      coordinates: [[0, 64, 12]],
    }),
    xray: parseXray([{ time_tag: "2026-07-29T13:52:00Z", satellite: 18, energy: "0.1-0.8nm", flux: 3.4e-6 }]),
    errors: [],
  };
  const { feeds, anyStale } = computeFreshness(pull as any, now);
  // 6-hour feeds, all well inside
  assert.deepEqual(feeds.kp, { at: "2026-07-29T12:00:00", ageS: 2 * 3600, stale: false });
  assert.equal(feeds.scales.stale, false);
  assert.equal(feeds.scales.ageS, 83 * 60, "12:37 UTC scales stamp vs 14:00");
  assert.equal(feeds.alerts.stale, false);
  // 15-min feeds: speed stale at 20 min, mag fresh at 5 min, xray fresh at 8 min
  assert.deepEqual(feeds.windSpeed, { at: "2026-07-29T13:40:00Z", ageS: 20 * 60, stale: true });
  assert.deepEqual(feeds.windMag, { at: "2026-07-29T13:55:00Z", ageS: 5 * 60, stale: false });
  assert.equal(feeds.xray.stale, false);
  // 30-min ovation stale at 40 min — and freshness reads the OBSERVATION time,
  // never the (future) forecast target time
  assert.deepEqual(feeds.ovation, { at: "2026-07-29T13:20:00Z", ageS: 40 * 60, stale: true });
  assert.equal(anyStale, true);
  // Kp newest-picking ignored the newest-first array order
  assert.equal(newestFeedAt(pull as any).kp, "2026-07-29T12:00:00");
});

test("computeFreshness: empty pull is honestly all-stale with null ages; fully fresh pull is anyStale=false", () => {
  const empty = {
    kp: [], scales: { current: null, forecast: [] }, alerts: [],
    wind: { speedKms: null, speedAt: null, btNt: null, bzNt: null, magAt: null },
    aurora: null, xray: [], errors: ["all seven down"],
  };
  const e = computeFreshness(empty as any, NOW);
  for (const k of Object.keys(FEED_MAX_AGE_MS) as SwFeedKey[]) {
    assert.deepEqual(e.feeds[k], { at: null, ageS: null, stale: true }, `${k}: no record -> stale, null age`);
  }
  assert.equal(e.anyStale, true);

  const now = Date.parse("2026-07-29T13:00:00Z");
  const fresh = {
    kp: [{ t: "2026-07-29T12:00:00", kp: 2, a: 9, n: 8, rt: "2026-07-29" }],
    scales: { current: { kind: "current", date: "2026-07-29", time: "12:37:00", r: "0", s: "0", g: "0", rMinorProb: null, rMajorProb: null, sProb: null }, forecast: [] },
    alerts: [{ id: "x", productId: "x", issued: "2026-07-29 12:40:00.000", code: null, serial: null, title: null, message: "m", rt: "2026-07-29" }],
    wind: { speedKms: 400, speedAt: "2026-07-29T12:55:00Z", btNt: 5, bzNt: 1, magAt: "2026-07-29T12:56:00Z" },
    aurora: { obs: "2026-07-29T12:45:00Z", forecast: null, max: 10, aggDeg: 2, minVal: 2, cells: [] },
    xray: [{ time_tag: "2026-07-29T12:58:00Z", satellite: 18, energy: "0.1-0.8nm", flux: 1e-6, electronContamination: null }],
    errors: [],
  };
  const f = computeFreshness(fresh as any, now);
  assert.equal(f.anyStale, false);
  for (const k of Object.keys(FEED_MAX_AGE_MS) as SwFeedKey[]) {
    assert.equal(f.feeds[k].stale, false, `${k} fresh`);
    assert.ok(f.feeds[k].ageS! >= 0);
  }
});
