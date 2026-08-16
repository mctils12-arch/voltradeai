/**
 * Hermetic tests for the GNSS integrity gate-2 signal surface — band
 * verdict statistics (moved here from scripts/gnss_integrity_gate2.ts,
 * same assertions, now testing the canonical location), the rolling-
 * window day picker, and the live end-to-end summary computation against
 * a temp archive dir. No network. Runs via `npm run test:node`.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  evaluateBands, gate2Verdict, recentDays, computeGnssIntegritySignal,
  WRITER_LIVE_SINCE, CANDIDATE_BBOX, CONTROL_BBOX,
  type DiagCell,
} from "./gnssIntegritySignal";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-gnss-signal-"));

function cell(band: string, origin: string, n_total: number, n_zero: number): DiagCell {
  return { band, origin, n_total, n_zero, distinct_airframes: 1 };
}

test("evaluateBands: excludes non-broadcast origin cells (mlat-derived rows are not the aircraft's own GPS reading)", () => {
  const candidate = [cell("cruise", "broadcast", 84, 3), cell("cruise", "ground", 35, 35)];
  const control = [cell("cruise", "broadcast", 9641, 19)];
  const verdicts = evaluateBands(candidate, control);
  assert.equal(verdicts.length, 1);
  assert.equal(verdicts[0].band, "cruise");
  assert.equal(verdicts[0].candidate_k, 3);
  assert.equal(verdicts[0].candidate_n, 84);
});

test("evaluateBands: reproduces the live 2026-08-13 Baltic-vs-control finding (cruise + mid elevated, low is not)", () => {
  const candidate = [
    cell("cruise", "broadcast", 84, 3),
    cell("mid", "broadcast", 146, 8),
    cell("low", "broadcast", 295, 7),
  ];
  const control = [
    cell("cruise", "broadcast", 9641, 19),
    cell("mid", "broadcast", 6194, 16),
    cell("low", "broadcast", 16229, 351),
  ];
  const verdicts = evaluateBands(candidate, control);
  const byBand = Object.fromEntries(verdicts.map((v) => [v.band, v]));
  assert.equal(byBand.cruise.elevated, true);
  assert.ok(byBand.cruise.p_value < 0.001);
  assert.equal(byBand.mid.elevated, true);
  assert.ok(byBand.mid.p_value < 0.001);
  assert.equal(byBand.low.elevated, false); // 7/295=2.4% vs control 351/16229=2.16% — consistent with base rate
});

test("evaluateBands: a band missing from either side (or zero-n) is skipped, not divide-by-zero", () => {
  const candidate = [cell("cruise", "broadcast", 84, 3), cell("ground", "broadcast", 0, 0)];
  const control = [cell("cruise", "broadcast", 9641, 19)]; // no "ground" band
  const verdicts = evaluateBands(candidate, control);
  assert.equal(verdicts.length, 1);
  assert.equal(verdicts[0].band, "cruise");
});

test("gate2Verdict: PASS requires elevation in an expected-elevated band", () => {
  const verdicts = [
    { band: "cruise", candidate_k: 3, candidate_n: 84, control_rate: 0.002, expected_under_null: 0.17, p_value: 0.0006, elevated: true, expected_to_elevate: true },
    { band: "low", candidate_k: 7, candidate_n: 295, control_rate: 0.0216, expected_under_null: 6.37, p_value: 0.4, elevated: false, expected_to_elevate: false },
  ];
  assert.equal(gate2Verdict(verdicts), "PASS");
});

test("gate2Verdict: FAIL when an expected-null band ALSO shows elevation (artifact pattern, not a targeted signature)", () => {
  const verdicts = [
    { band: "cruise", candidate_k: 3, candidate_n: 84, control_rate: 0.002, expected_under_null: 0.17, p_value: 0.0006, elevated: true, expected_to_elevate: true },
    { band: "low", candidate_k: 50, candidate_n: 295, control_rate: 0.0216, expected_under_null: 6.37, p_value: 0.0001, elevated: true, expected_to_elevate: false },
  ];
  assert.equal(gate2Verdict(verdicts), "FAIL");
});

test("gate2Verdict: INCONCLUSIVE when nothing clears the bar anywhere", () => {
  const verdicts = [
    { band: "cruise", candidate_k: 1, candidate_n: 84, control_rate: 0.002, expected_under_null: 0.17, p_value: 0.15, elevated: false, expected_to_elevate: true },
  ];
  assert.equal(gate2Verdict(verdicts), "INCONCLUSIVE");
});

test("recentDays: floors at WRITER_LIVE_SINCE regardless of how large maxDays is", () => {
  const now = Date.parse("2026-08-16T12:00:00Z");
  const days = recentDays(30, WRITER_LIVE_SINCE, now);
  assert.equal(days[days.length - 1], WRITER_LIVE_SINCE, "never reaches before the writer went live");
  assert.ok(days.every((d) => d >= WRITER_LIVE_SINCE));
  assert.equal(days[0], "2026-08-16", "most recent day first");
});

test("recentDays: returns nothing before the writer's own start date", () => {
  const now = Date.parse("2026-08-10T12:00:00Z"); // before WRITER_LIVE_SINCE
  const days = recentDays(21, WRITER_LIVE_SINCE, now);
  assert.deepEqual(days, []);
});

test("recentDays: caps at maxDays even when the floor is far in the past", () => {
  const now = Date.parse("2026-09-01T00:00:00Z");
  const days = recentDays(5, WRITER_LIVE_SINCE, now);
  assert.equal(days.length, 5);
});

test("computeGnssIntegritySignal: reads both regions live, reports honest freshness, never fabricates a day", async () => {
  const base = tmp();
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  // Candidate (Baltic) — elevated nic==0 rate at cruise.
  const baltic = Array.from({ length: 20 }, (_, i) =>
    `{"t":${i},"i":"bal${i}","la":55,"lo":20,"al":9000,"ni":${i < 6 ? 0 : 8},"pt":"adsb_icao"}`).join("\n") + "\n";
  // Control (NY) — ordinary low background rate at cruise.
  const control = Array.from({ length: 200 }, (_, i) =>
    `{"t":${i},"i":"ny${i}","la":40.7,"lo":-74,"al":9000,"ni":${i < 2 ? 0 : 8},"pt":"adsb_icao"}`).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-08-11-00.jsonl"), baltic + control);
  const now = Date.parse("2026-08-12T12:00:00Z");
  const summary = await computeGnssIntegritySignal(21, base, now);
  assert.equal(summary.kind, "signal");
  assert.equal(summary.root_id, "gnss_integrity_adsb");
  assert.equal(summary.gate.status, "gate2_pass");
  assert.deepEqual(summary.freshness.candidate.days_read, ["2026-08-11"]);
  assert.deepEqual(summary.freshness.control.days_read, ["2026-08-11"]);
  assert.ok(summary.freshness.candidate.days_missing.includes("2026-08-12"), "the second requested day has no archive files and is reported missing, not silently dropped");
  assert.ok(summary.bands.length > 0);
  assert.ok(summary.caveats.length >= 3, "gate-1-partial / small-sample / not-tradeable caveats are always present");
  assert.equal(summary.region.candidate_bbox[0], CANDIDATE_BBOX.lamin);
  assert.equal(summary.region.control_bbox[0], CONTROL_BBOX.lamin);
  fs.rmSync(base, { recursive: true, force: true });
});

test("computeGnssIntegritySignal: no archive data at all still returns a well-formed, honestly-empty summary", async () => {
  const base = tmp();
  const now = Date.parse("2026-08-12T00:00:00Z");
  const summary = await computeGnssIntegritySignal(21, base, now);
  assert.equal(summary.bands.length, 0);
  assert.equal(summary.verdict, "INCONCLUSIVE");
  assert.deepEqual(summary.freshness.candidate.days_read, []);
  fs.rmSync(base, { recursive: true, force: true });
});
