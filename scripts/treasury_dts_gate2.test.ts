// treasury_dts_gate2 battery: date-window math, YoY growth, correlation,
// and the p-value/incomplete-beta implementation (A/B-verified this
// session against Python's scipy.stats.t.cdf after finding and fixing a
// real bug — see treasury_dts_gate2.ts's own betacf comment).
import { test } from "node:test";
import assert from "node:assert/strict";
import { monthRange, yoyGrowth, pearson, pValueForR } from "./treasury_dts_gate2";

test("monthRange: inclusive, wraps year boundaries", () => {
  assert.deepEqual(monthRange("2023-11", "2024-02"), ["2023-11", "2023-12", "2024-01", "2024-02"]);
  assert.deepEqual(monthRange("2024-05", "2024-05"), ["2024-05"]);
});

test("yoyGrowth: null when either endpoint is missing or the prior value is zero, never NaN/Infinity", () => {
  const m = new Map([["2024-01", 110], ["2023-01", 100], ["2024-02", 5], ["2023-02", 0]]);
  assert.equal(yoyGrowth(m, "2024-01"), (110 - 100) / 100);
  assert.equal(yoyGrowth(m, "2024-03"), null, "current month missing");
  assert.equal(yoyGrowth(m, "2025-01"), null, "prior-year month missing");
  assert.equal(yoyGrowth(m, "2024-02"), null, "prior value is zero — division would be Infinity/NaN");
});

test("pearson: +1 for a perfect line, -1 for perfect inverse", () => {
  assert.ok(Math.abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1) < 1e-9);
  assert.ok(Math.abs(pearson([1, 2, 3, 4], [8, 6, 4, 2]) - -1) < 1e-9);
});

// pValueForR — the exact bug this session found and fixed: the original
// ibeta() flipped its own return value (`x < bar ? cf : 1 - cf`) instead of
// recomputing the continued fraction with swapped (a,b,1-x) parameters in
// the x >= bar branch, silently wrong whenever a test's t-statistic landed
// in that branch (which most large-df, small-|r| cases do). Every value
// below is a live A/B against scipy.stats.t.cdf run this session.
test("pValueForR: matches scipy.stats.t.cdf-derived two-tailed p-values", () => {
  const cases: [number, number, number][] = [
    [0.1627287566700037, 31, 0.38176253489935286],
    [0.3, 31, 0.10106616727684603],
    [0.5, 20, 0.024769558804109693],
    [0.9, 10, 0.0003871562499999648],
    [0.22288644413114633, 30, 0.23645648258140373], // Test B's own lead-1-month reading
  ];
  for (const [r, n, expected] of cases) {
    const got = pValueForR(r, n);
    assert.ok(Math.abs(got - expected) < 1e-6, `r=${r} n=${n}: got ${got}, expected ${expected}`);
  }
});

test("pValueForR: r=0 gives p=1 (no evidence against the null); df<=0 gives p=1 (undefined test, never a false PASS)", () => {
  assert.ok(Math.abs(pValueForR(0, 31) - 1) < 1e-9);
  assert.equal(pValueForR(0.5, 2), 1);
  assert.equal(pValueForR(0.5, 1), 1);
});

test("pValueForR: symmetric in the sign of r (a negative correlation is exactly as significant as the same-magnitude positive one)", () => {
  assert.equal(pValueForR(0.4, 25), pValueForR(-0.4, 25));
});
