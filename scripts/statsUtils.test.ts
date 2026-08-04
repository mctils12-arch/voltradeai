import { test } from "node:test";
import assert from "node:assert/strict";
import { clusterMeanTTest, tCrit005, survivesAtCrit005 } from "./statsUtils";

test("clusterMeanTTest: zero-variance cluster means give NaN t (guarded, not Infinity)", () => {
  const r = clusterMeanTTest([0.01, 0.01, 0.01]);
  assert.equal(r.n, 3);
  assert.equal(r.mean, 0.01);
  assert.equal(r.std, 0);
  assert.equal(r.se, 0);
  assert.ok(Number.isNaN(r.t));
  assert.equal(r.df, 2);
});

test("clusterMeanTTest: matches hand-computed t for a known small sample", () => {
  // values: 1, 2, 3, 4, 5 -> mean=3, sample std=sqrt(2.5)=1.5811, se=std/sqrt(5)=0.7071
  // t = mean/se = 3 / 0.7071 = 4.2426
  const r = clusterMeanTTest([1, 2, 3, 4, 5]);
  assert.equal(r.n, 5);
  assert.equal(r.mean, 3);
  assert.ok(Math.abs(r.std - 1.5811) < 1e-3);
  assert.ok(Math.abs(r.se - 0.7071) < 1e-3);
  assert.ok(Math.abs(r.t - 4.2426) < 1e-3);
  assert.equal(r.df, 4);
});

test("clusterMeanTTest: n=0 and n=1 are guarded, not NaN-propagating crashes", () => {
  const zero = clusterMeanTTest([]);
  assert.equal(zero.n, 0);
  assert.ok(Number.isNaN(zero.mean));
  assert.equal(zero.df, 0);

  const one = clusterMeanTTest([0.05]);
  assert.equal(one.n, 1);
  assert.equal(one.mean, 0.05);
  assert.equal(one.std, 0); // n>1 required for sample variance
  assert.equal(one.se, 0);
  assert.ok(Number.isNaN(one.t));
});

test("tCrit005: known table values at small df, normal approximation beyond df=30", () => {
  assert.equal(tCrit005(1), 12.706);
  assert.equal(tCrit005(15), 2.131);
  assert.equal(tCrit005(30), 2.042);
  assert.equal(tCrit005(31), 1.96);
  assert.equal(tCrit005(100), 1.96);
  assert.ok(Number.isNaN(tCrit005(0)));
});

test("survivesAtCrit005: a flat |t|>2 heuristic would wrongly pass a df=15 result that doesn't clear 5%", () => {
  // t=2.05 clears the naive |t|>2 heuristic but NOT the true df=15 critical
  // value of 2.131 — this is exactly the failure mode this helper exists
  // to catch (few-cluster samples need the higher, df-correct bar).
  const nearMiss = { n: 16, mean: 0.01, std: 0.02, se: 0.005, t: 2.05, df: 15 };
  assert.equal(survivesAtCrit005(nearMiss), false);

  const clears = { n: 16, mean: 0.02, std: 0.02, se: 0.005, t: 3.5, df: 15 };
  assert.equal(survivesAtCrit005(clears), true);
});
