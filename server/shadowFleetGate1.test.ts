// GATE 1 (DATA) case-control enrichment test — research/open_questions.md
// "SHADOW-FLEET SIGNAL": "Gate passes if our gap/loiter detections are
// significantly ENRICHED for reference-list vessels vs a size-matched
// random tanker sample ... (odds ratio with CI, not eyeballing)." Every
// case here uses synthetic MMSI sets — no archive/network access, matching
// this module's own pure-statistics contract.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  contingencyCounts, oddsRatio, oddsRatioCI, seededSample,
  buildCaseControlUniverse, evaluateEnrichment,
} from "./shadowFleetGate1";

test("contingencyCounts: hand-verified 2x2 table over a small synthetic universe", () => {
  const candidates = new Set(["c1", "c2", "c3"]); // gap/loiter-flagged
  const reference = new Set(["c1", "x1", "x2"]); // OFAC reference list
  const universe = ["c1", "c2", "c3", "n1", "n2", "x1", "x2"];
  // c1: candidate & reference -> a. c2,c3: candidate & !reference -> b (x2).
  // x1,x2: !candidate & reference -> c (x2). n1,n2: !candidate & !reference -> d (x2).
  const t = contingencyCounts(candidates, reference, universe);
  assert.deepEqual(t, { a: 1, b: 2, c: 2, d: 2 });
});

test("contingencyCounts: a duplicate MMSI in universe is counted once", () => {
  const candidates = new Set(["c1"]);
  const reference = new Set(["c1"]);
  const t = contingencyCounts(candidates, reference, ["c1", "c1", "c1"]);
  assert.deepEqual(t, { a: 1, b: 0, c: 0, d: 0 });
});

test("oddsRatio: no zero cells, all-equal table is exactly 1", () => {
  assert.equal(oddsRatio({ a: 3, b: 3, c: 3, d: 3 }), 1);
});

test("oddsRatio: hand-verified value, no correction needed", () => {
  // OR = (a*d)/(b*c) = (20*95)/(80*5) = 1900/400 = 4.75
  const or_ = oddsRatio({ a: 20, b: 80, c: 5, d: 95 });
  assert.ok(Math.abs(or_ - 4.75) < 1e-9, `expected 4.75, got ${or_}`);
});

test("oddsRatio: Haldane-Anscombe +0.5 correction applied to every cell when any cell is zero", () => {
  // a=0 triggers correction on all four cells: (0.5*10.5)/(10.5*0.5) = 1 exactly
  const or_ = oddsRatio({ a: 0, b: 10, c: 0, d: 10 });
  assert.equal(or_, 1);
});

test("oddsRatio: without correction a zero cell would be 0 or Infinity — correction prevents both", () => {
  const or_ = oddsRatio({ a: 5, b: 0, c: 5, d: 5 });
  assert.ok(Number.isFinite(or_) && or_ > 0);
});

test("oddsRatioCI: a clearly enriched synthetic case (n=200) has a CI entirely above 1", () => {
  const t = { a: 20, b: 80, c: 5, d: 95 };
  const ci = oddsRatioCI(t);
  assert.ok(ci.lower > 1, `expected lower bound > 1, got ${ci.lower}`);
  assert.ok(ci.lower < oddsRatio(t), "lower bound sits below the point estimate");
  assert.ok(ci.upper > oddsRatio(t), "upper bound sits above the point estimate");
});

test("oddsRatioCI: a null/random case (no real association) straddles 1", () => {
  // candidates and controls have near-identical reference-hit rates
  const t = { a: 10, b: 90, c: 10, d: 90 };
  const ci = oddsRatioCI(t);
  assert.ok(ci.lower < 1 && ci.upper > 1, `expected a straddling CI, got [${ci.lower}, ${ci.upper}]`);
});

test("oddsRatioCI: log-symmetric around the point estimate in log-space", () => {
  const t = { a: 20, b: 80, c: 5, d: 95 };
  const ci = oddsRatioCI(t);
  const or_ = oddsRatio(t);
  const lowGap = Math.log(or_) - Math.log(ci.lower);
  const highGap = Math.log(ci.upper) - Math.log(or_);
  assert.ok(Math.abs(lowGap - highGap) < 1e-9, "Woolf's method is symmetric in log-odds space, not raw-ratio space");
});

test("seededSample: reproducible given the same seed", () => {
  const pool = Array.from({ length: 50 }, (_, i) => `m${i}`);
  const s1 = seededSample(pool, 10, 42);
  const s2 = seededSample(pool, 10, 42);
  assert.deepEqual(s1, s2);
});

test("seededSample: different seeds produce a different draw (not a fixed constant order)", () => {
  const pool = Array.from({ length: 50 }, (_, i) => `m${i}`);
  const s1 = seededSample(pool, 10, 1);
  const s2 = seededSample(pool, 10, 2);
  assert.notDeepEqual(s1, s2);
});

test("seededSample: requesting more than the pool size returns the whole pool, no duplicates", () => {
  const pool = ["a", "b", "c"];
  const s = seededSample(pool, 10, 7);
  assert.equal(s.length, 3);
  assert.deepEqual(new Set(s), new Set(pool));
});

test("buildCaseControlUniverse: controls never overlap cases, and are size-matched to them", () => {
  const candidates = new Set(["c1", "c2"]);
  const pool = ["c1", "c2", "p1", "p2", "p3", "p4"];
  const universe = buildCaseControlUniverse(candidates, pool, 5);
  assert.equal(universe.length, 4, "2 cases + 2 size-matched controls");
  const controls = universe.filter((m) => !candidates.has(m));
  assert.equal(controls.length, 2);
  for (const c of controls) assert.ok(!candidates.has(c), "no candidate ever drawn as its own control");
});

test("buildCaseControlUniverse: fewer eligible controls than cases uses every eligible control, no duplicate draw", () => {
  const candidates = new Set(["c1", "c2", "c3"]);
  const pool = ["c1", "c2", "c3", "p1"]; // only 1 non-candidate in the pool
  const universe = buildCaseControlUniverse(candidates, pool, 5);
  assert.equal(universe.length, 4, "3 cases + only 1 available control, not padded or duplicated");
});

test("evaluateEnrichment: insufficient_n flags a universe with fewer than 5 total reference hits", () => {
  const candidates = new Set(["c1"]);
  const reference = new Set(["c1"]);
  const pool = ["c1", "p1", "p2", "p3", "p4", "p5"];
  const v = evaluateEnrichment(candidates, reference, pool, 1);
  assert.equal(v.n_reference_hits, 1);
  assert.equal(v.insufficient_n, true);
});

test("evaluateEnrichment: enriched=true only when the CI lies entirely above 1, matching oddsRatioCI directly", () => {
  // Construct a large, clearly-enriched synthetic archive: 100 candidates
  // (20 on the reference list), 100-control pool candidates are excluded
  // from (5 of the eligible controls on the reference list).
  const candidates = new Set(Array.from({ length: 100 }, (_, i) => `cand${i}`));
  const referenceInCandidates = Array.from({ length: 20 }, (_, i) => `cand${i}`);
  const referenceInPool = Array.from({ length: 5 }, (_, i) => `pool${i}`);
  const reference = new Set([...referenceInCandidates, ...referenceInPool]);
  const pool = [
    ...Array.from(candidates), // detectors' own candidates also live in the tanker pool
    ...Array.from({ length: 100 }, (_, i) => `pool${i}`),
  ];
  const v = evaluateEnrichment(candidates, reference, pool, 99);
  assert.equal(v.n_candidates, 100);
  assert.equal(v.n_controls, 100, "size-matched to the 100 candidates");
  assert.equal(v.enriched, v.ci_95.lower > 1, "enriched is derived from the CI, never a separate judgment call");
  assert.equal(v.seed, 99, "seed is echoed back for independent re-verification");
});

test("evaluateEnrichment: deterministic given the same seed (independently re-runnable)", () => {
  const candidates = new Set(["c1", "c2", "c3"]);
  const reference = new Set(["c1", "p9"]);
  const pool = [...candidates, ...Array.from({ length: 20 }, (_, i) => `p${i}`)];
  const v1 = evaluateEnrichment(candidates, reference, pool, 123);
  const v2 = evaluateEnrichment(candidates, reference, pool, 123);
  assert.deepEqual(v1, v2);
});
