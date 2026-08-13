import { test } from "node:test";
import assert from "node:assert/strict";
import { evaluateBands, gate2Verdict, type DiagCell } from "./gnss_integrity_gate2";

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
