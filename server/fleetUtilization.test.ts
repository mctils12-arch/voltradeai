// fleet-utilization battery (BUILD ORDER 3 #1, 2026-07-05): sessionization
// ground truth, week bucketing, spine join with ground-point and
// non-corporate exclusions. Synthetic archive + spine fixtures throughout.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { foldSessions, weekStart, buildFleetSeries, preserveWeeklyBeforeRollup, _resetFleetCache, SESSION_GAP_MIN } from "./fleetUtilization";
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

// REPAIR (found 2026-07-22, same root cause + fix as datacoreArchive.ts's
// streamJsonlLines): readline.Interface re-emits a piped-in stream's error
// on ITSELF too, independent of the stream.on("error", ...) guard here —
// unlistened, a truncated/corrupt .gz crashed the WHOLE PROCESS. See
// datacoreArchive.test.ts for the full writeup + minimal repro.
// preserveWeeklyBeforeRollup (found this session while trying to actually
// run the BUILD ORDER 3d mining pass): datacoreArchive deletes raw aircraft
// hour files past RAW_RETENTION_DAYS, so buildFleetSeries above can only
// ever see 7 days of history no matter how long the archive has run. These
// three tests pin the permanent-archive fix: fold-then-survive-deletion,
// idempotent-before-deletion, and additive-across-ticks.
const hourFile = (ms: number) => new Date(ms).toISOString().slice(0, 13).replace("T", "-") + ".jsonl";

test("preserveWeeklyBeforeRollup folds an aged-out file into the permanent archive, survives its deletion, and does not double-count if called again before the file is deleted", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fleet-preserve-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  const spineFp = path.join(base, "spine.json");
  fs.writeFileSync(spineFp, JSON.stringify({
    entities: { abc123: { n_number: "N1CORP", owner: "ACME JETS INC", registrant_type: "corporation" } },
  }));
  const oldMs = (MON + 10 * 3600) * 1000;
  const t0 = Math.floor(oldMs / 1000);
  const fname = hourFile(oldMs);
  fs.writeFileSync(path.join(base, "aircraft", fname),
    [{ t: t0, i: "abc123" }, { t: t0 + 900, i: "abc123" }].map((l) => JSON.stringify(l)).join("\n") + "\n");
  const nowMs = oldMs + 8 * 86400_000; // 8 days later: past the 7-day cutoff

  const r1 = await preserveWeeklyBeforeRollup(base, nowMs, 7, spineFp);
  assert.equal(r1.filesFolded, 1);
  assert.equal(r1.ownersTouched, 1);

  // idempotent: raw file still present (rollup hasn't deleted it yet) -> no-op
  const r2 = await preserveWeeklyBeforeRollup(base, nowMs, 7, spineFp);
  assert.equal(r2.filesFolded, 0, "already-folded file is not re-processed");

  // simulate the generic rollup deleting the raw file afterward
  fs.unlinkSync(path.join(base, "aircraft", fname));

  const series = await buildFleetSeries(base, spineFp);
  assert.equal(series.length, 1);
  assert.equal(series[0].owner, "ACME JETS INC");
  assert.equal(series[0].weekly[weekStart(t0)].f, 1, "one session preserved, not doubled by the idempotent second call");
});

test("preserveWeeklyBeforeRollup: a week spanning two rollup ticks accumulates additively, not overwritten", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fleet-preserve2-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  const spineFp = path.join(base, "spine.json");
  fs.writeFileSync(spineFp, JSON.stringify({
    entities: { abc123: { n_number: "N1CORP", owner: "ACME JETS INC", registrant_type: "corporation" } },
  }));
  const day1Ms = (MON + 10 * 3600) * 1000;
  const day2Ms = day1Ms + 3600_000; // one hour later, same day/week, separate hour file

  const f1 = hourFile(day1Ms);
  fs.writeFileSync(path.join(base, "aircraft", f1), JSON.stringify({ t: Math.floor(day1Ms / 1000), i: "abc123" }) + "\n");
  const r1 = await preserveWeeklyBeforeRollup(base, day1Ms + 8 * 86400_000, 7, spineFp);
  assert.equal(r1.filesFolded, 1);
  fs.unlinkSync(path.join(base, "aircraft", f1));

  const f2 = hourFile(day2Ms);
  fs.writeFileSync(path.join(base, "aircraft", f2), JSON.stringify({ t: Math.floor(day2Ms / 1000), i: "abc123" }) + "\n");
  const r2 = await preserveWeeklyBeforeRollup(base, day2Ms + 8 * 86400_000, 7, spineFp);
  assert.equal(r2.filesFolded, 1);
  fs.unlinkSync(path.join(base, "aircraft", f2));

  const series = await buildFleetSeries(base, spineFp);
  assert.equal(series[0].weekly[weekStart(Math.floor(day1Ms / 1000))].f, 2,
    "both hour files' sessions accumulate in the same week across two preserve calls, not overwritten");
});

test("buildFleetSeries surfaces historical-archive-only owners (no current live-window airframes) from the permanent weekly archive", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fleet-histonly-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  const spineFp = path.join(base, "spine.json");
  fs.writeFileSync(spineFp, JSON.stringify({ entities: {} }));
  const oldMs = (MON + 10 * 3600) * 1000;
  const t0 = Math.floor(oldMs / 1000);
  const fname = hourFile(oldMs);
  // callsign-resolved operator (DAL prefix) — resolves without a spine entry
  fs.writeFileSync(path.join(base, "aircraft", fname),
    [{ t: t0, i: "dal001", c: "DAL123" }, { t: t0 + 900, i: "dal001", c: "DAL123" }].map((l) => JSON.stringify(l)).join("\n") + "\n");
  await preserveWeeklyBeforeRollup(base, oldMs + 8 * 86400_000, 7, spineFp);
  fs.unlinkSync(path.join(base, "aircraft", fname));

  const series = await buildFleetSeries(base, spineFp);
  assert.equal(series.length, 1);
  assert.equal(series[0].owner, "DELTA AIR LINES");
  assert.equal(series[0].n_airframes, 0, "no current-window airframes; still surfaced from the permanent archive");
  assert.equal(series[0].registrant_type, "historical-archive-only");
  assert.equal(series[0].weekly[weekStart(t0)].f, 1);
});

test("buildFleetSeries resolves (never crashes the process) on a truncated/corrupt gzip file", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fleet-trunc-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  const spineFp = path.join(base, "spine.json");
  fs.writeFileSync(spineFp, JSON.stringify({
    entities: { abc123: { n_number: "N1CORP", owner: "ACME JETS INC", registrant_type: "corporation" } },
  }));
  const good = zlib.gzipSync(JSON.stringify({ t: MON, i: "abc123" }) + "\n");
  fs.writeFileSync(path.join(base, "aircraft", "2026-06-29-10.jsonl.gz"), good.subarray(0, good.length - 4));
  await buildFleetSeries(base, spineFp);
});
