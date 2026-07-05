// fleet-utilization battery (BUILD ORDER 3 #1, 2026-07-05): sessionization
// ground truth, week bucketing, spine join with ground-point and
// non-corporate exclusions. Synthetic archive + spine fixtures throughout.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { foldSessions, weekStart, buildFleetSeries, _resetFleetCache, SESSION_GAP_MIN } from "./fleetUtilization";
import { _resetAircraftEntityCache } from "./aircraftEntities";

beforeEach(() => { _resetFleetCache(); _resetAircraftEntityCache(); });

const MON = Date.parse("2026-06-29T00:00:00Z") / 1000; // a Monday

test("weekStart maps any weekday to its Monday", () => {
  assert.equal(weekStart(MON), "2026-06-29");
  assert.equal(weekStart(MON + 6 * 86400 + 3600), "2026-06-29", "Sunday still same week");
  assert.equal(weekStart(MON + 7 * 86400), "2026-07-06");
});

test("foldSessions: gap splits flights; hours are lower-bound span sums", () => {
  const t0 = MON + 12 * 3600;
  const f2 = t0 + 1800 + (SESSION_GAP_MIN + 5) * 60;   // gap > threshold -> new flight
  const times = [
    t0, t0 + 600, t0 + 1200, t0 + 1800,                // 30-min flight
    f2, f2 + 1200, f2 + 1800,                          // 30-min second flight (intra-gaps < threshold)
  ];
  const weekly = foldSessions(times);
  assert.deepEqual(Object.keys(weekly), ["2026-06-29"]);
  assert.equal(weekly["2026-06-29"].f, 2);
  assert.equal(weekly["2026-06-29"].h, 1);
  // single point = flight of 0 hours (seen once, still a movement)
  const single = foldSessions([t0]);
  assert.equal(single["2026-06-29"].f, 1);
  assert.equal(single["2026-06-29"].h, 0);
  assert.deepEqual(foldSessions([]), {});
});

test("buildFleetSeries: joins corporate spine owners, excludes ground points and non-corporates", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fleet-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  const spineFp = path.join(base, "spine.json");
  fs.writeFileSync(spineFp, JSON.stringify({
    entities: {
      abc123: { n_number: "N1CORP", owner: "ACME JETS INC", registrant_type: "corporation" },
      def456: { n_number: "N2IND", owner: "SOME PERSON", registrant_type: "individual" },
    },
  }));
  const t0 = MON + 10 * 3600;
  const lines = [
    { t: t0, i: "abc123" }, { t: t0 + 900, i: "abc123" },
    { t: t0 + 950, i: "abc123", g: true },                 // ground point excluded
    { t: t0 + 8 * 86400, i: "abc123" },                    // next week, second flight
    { t: t0, i: "def456" },                                // individual -> excluded
    { t: t0, i: "zzz999" },                                // not in spine -> excluded
  ];
  fs.writeFileSync(path.join(base, "aircraft", "2026-06-29-10.jsonl"),
    lines.slice(0, 3).map((l) => JSON.stringify(l)).join("\n") + "\n");
  fs.writeFileSync(path.join(base, "aircraft", "2026-07-07-10.jsonl.gz"),
    zlib.gzipSync(lines.slice(3).map((l) => JSON.stringify(l)).join("\n") + "\n"));

  const series = await buildFleetSeries(base, spineFp);
  assert.equal(series.length, 1, "only the corporate registrant survives");
  const s = series[0];
  assert.equal(s.owner, "ACME JETS INC");
  assert.equal(s.n_airframes, 1);
  assert.equal(s.weekly["2026-06-29"].f, 1);
  assert.equal(s.weekly["2026-06-29"].h, 0.25);
  assert.equal(s.weekly["2026-07-06"].f, 1);
  assert.equal(Object.keys(s.weekly).length, 2, "weeks without coverage are absent, not zero");
});

test("buildFleetSeries: missing archive or spine degrades to empty", async () => {
  assert.deepEqual(await buildFleetSeries("/nonexistent", "/nonexistent/spine.json"), []);
});
