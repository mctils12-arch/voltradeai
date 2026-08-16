import { test } from "node:test";
import assert from "node:assert/strict";
import { clusterMeanTTest, tCrit005, survivesAtCrit005, binomialUpperTailP } from "./statsUtils";

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

test("binomialUpperTailP: matches hand-computed P(X>=8) for Binomial(10, 0.5)", () => {
  // P(8)+P(9)+P(10) = (C(10,8)+C(10,9)+C(10,10)) / 2^10 = (45+10+1)/1024
  const expected = 56 / 1024;
  assert.ok(Math.abs(binomialUpperTailP(8, 10, 0.5) - expected) < 1e-9);
});

test("binomialUpperTailP: edge cases (k<=0 always 1, p0<=0 always 0 for k>0, p0>=1 always 1)", () => {
  assert.equal(binomialUpperTailP(0, 100, 0.01), 1);
  assert.equal(binomialUpperTailP(-1, 100, 0.01), 1);
  assert.equal(binomialUpperTailP(5, 100, 0), 0);
  assert.equal(binomialUpperTailP(5, 100, 1), 1);
});

test("binomialUpperTailP: rare elevated count against a large-n low base rate gives a small p-value (the gate-2 use case)", () => {
  // n=84, p0=0.0020 (control cruise/broadcast zero-rate), k=3 observed —
  // matches the live 2026-08-13 gnss_integrity Baltic-vs-control run.
  const p = binomialUpperTailP(3, 84, 0.0020);
  assert.ok(p < 0.001, `expected p<0.001, got ${p}`);
  // and a count consistent with the null (k=0) should NOT look significant
  const pNull = binomialUpperTailP(0, 84, 0.0020);
  assert.equal(pNull, 1);
});

test("binomialUpperTailP: monotone in k (more extreme counts never get a larger p-value)", () => {
  const p1 = binomialUpperTailP(2, 200, 0.01);
  const p2 = binomialUpperTailP(5, 200, 0.01);
  const p3 = binomialUpperTailP(10, 200, 0.01);
  assert.ok(p1 > p2);
  assert.ok(p2 > p3);
});
