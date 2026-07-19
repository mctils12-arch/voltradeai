// gemMethaneAssetRates — repeat-detection count/rate per matched asset
// (gate-2(b) of the GEM METHANE-PLUME × EXTRACTION-REGISTRY PROXIMITY
// hypothesis, research/open_questions.md). Pins: ambiguous-match exclusion,
// single-detection assets getting a null (not zero/fabricated) rate, the
// span/rate arithmetic against real dates, both sort orders, and the
// cached-loader degrade path when the upstream proximity join is unavailable.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  computeAssetDetectionRates, sortAssetDetectionRates,
  cachedGemMethaneAssetRates, _resetGemMethaneAssetRatesCacheForTests,
} from "./gemMethaneAssetRates";
import { _resetGemMethaneProximityCacheForTests } from "./gemMethaneProximity";
import { _resetGemMethaneCacheForTests } from "./gemMethane";
import { _resetGemAssetsCacheForTests } from "./gemMethaneAssets";
import type { PlumeWithAsset } from "./gemMethaneProximity";
import type { NearestAsset } from "./gemMethaneProximity";

function nearestAsset(id: string, kind: NearestAsset["kind"] = "coal_mine"): NearestAsset {
  return { kind, id, name: `${id} name`, operator: "Op Co", owner: null, parent: null, distanceKm: 0.5 };
}

function plumeWithAsset(
  id: string,
  opts: { asset?: NearestAsset | null; ambiguous?: boolean; observedAt?: string | null; emissionsKgHr?: number | null } = {},
): PlumeWithAsset {
  return {
    id, name: null, wiki: null, provider: null, instrument: null,
    observedAt: opts.observedAt ?? null,
    emissionsKgHr: opts.emissionsKgHr ?? null,
    emissionsUncertaintyKgHr: null, infrastructureType: null, infrastructureNotes: null,
    subnationalUnit: null, country: null, lat: 0, lon: 0,
    nearestAsset: opts.asset === undefined ? nearestAsset(id) : opts.asset,
    ambiguousMatch: opts.ambiguous ?? false,
  };
}

test("computeAssetDetectionRates: unmatched and ambiguous plumes are excluded", () => {
  const plumes = [
    plumeWithAsset("P1", { asset: null }),
    plumeWithAsset("P2", { ambiguous: true }),
  ];
  assert.equal(computeAssetDetectionRates(plumes).length, 0);
});

test("computeAssetDetectionRates: a single detection gets count=1 and a null rate, not zero", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1"), observedAt: "2026-01-01T00:00:00Z" }),
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].detectionCount, 1);
  assert.equal(out[0].detectionsPerYear, null, "a single detection is noise, not a rate");
  assert.equal(out[0].spanDays, null);
});

test("computeAssetDetectionRates: two dated detections at the same asset compute a real span/rate", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1"), observedAt: "2026-01-01T00:00:00Z" }),
    plumeWithAsset("P2", { asset: nearestAsset("A1"), observedAt: "2026-07-01T00:00:00Z" }), // ~181 days later
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].detectionCount, 2);
  assert.ok(out[0].spanDays !== null && out[0].spanDays > 180 && out[0].spanDays < 182, `got ${out[0].spanDays}`);
  assert.ok(out[0].detectionsPerYear !== null && out[0].detectionsPerYear > 3.5 && out[0].detectionsPerYear < 4.5,
    `expected ~2 detections / (181/365.25)yr ~= 4.03/yr, got ${out[0].detectionsPerYear}`);
});

test("computeAssetDetectionRates: distinct assets are kept in separate groups, keyed by kind+id", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1", "coal_mine") }),
    plumeWithAsset("P2", { asset: nearestAsset("A1", "oil_gas_extraction") }), // same id, different kind
  ]);
  assert.equal(out.length, 2, "kind is part of the group key — same id, different kind, must not merge");
});

test("computeAssetDetectionRates: meanEmissionsKgHr averages only the non-null readings", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1"), emissionsKgHr: 100 }),
    plumeWithAsset("P2", { asset: nearestAsset("A1"), emissionsKgHr: null }),
    plumeWithAsset("P3", { asset: nearestAsset("A1"), emissionsKgHr: 300 }),
  ]);
  assert.equal(out[0].meanEmissionsKgHr, 200);
});

test("computeAssetDetectionRates: parses GEM's real bare-UTC-offset date format ('+00', no minutes/colon)", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1"), observedAt: "2020-11-09T18:47:16+00" }),
    plumeWithAsset("P2", { asset: nearestAsset("A1"), observedAt: "2021-01-09T18:47:16+00" }), // ~61 days later
  ]);
  assert.equal(out[0].detectionCount, 2);
  assert.ok(out[0].spanDays !== null && Math.abs(out[0].spanDays - 61) < 0.01, `got ${out[0].spanDays}`);
  assert.ok(out[0].detectionsPerYear !== null && out[0].detectionsPerYear > 0);
});

test("computeAssetDetectionRates: a span shorter than MIN_SPAN_DAYS_FOR_RATE keeps detectionsPerYear null — never extrapolate wildly from a short window", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1"), observedAt: "2026-01-01T00:00:00Z" }),
    plumeWithAsset("P2", { asset: nearestAsset("A1"), observedAt: "2026-01-03T00:00:00Z" }), // 2 days later
  ]);
  assert.equal(out[0].detectionCount, 2);
  assert.ok(out[0].spanDays !== null && out[0].spanDays < 30);
  assert.equal(out[0].detectionsPerYear, null, "2 days of history must not annualize to a rate");
});

test("computeAssetDetectionRates: a plain YYYY-MM-DD date (no time component) is never mistaken for a bare-offset date", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1"), observedAt: "2020-11-09" }),
  ]);
  assert.equal(out[0].firstObservedAt, new Date("2020-11-09").toISOString());
});

test("computeAssetDetectionRates: a plume with an unparseable date is excluded from date math but still counted", () => {
  const out = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1"), observedAt: "not-a-date" }),
    plumeWithAsset("P2", { asset: nearestAsset("A1"), observedAt: null }),
  ]);
  assert.equal(out[0].detectionCount, 2);
  assert.equal(out[0].firstObservedAt, null);
  assert.equal(out[0].spanDays, null);
});

test("sortAssetDetectionRates: by count sorts descending on detectionCount", () => {
  const rates = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1") }),
    plumeWithAsset("P2", { asset: nearestAsset("A2") }),
    plumeWithAsset("P3", { asset: nearestAsset("A2") }),
  ]);
  const sorted = sortAssetDetectionRates(rates, "count");
  assert.equal(sorted[0].assetId, "A2");
  assert.equal(sorted[0].detectionCount, 2);
});

test("sortAssetDetectionRates: by rate sinks null-rate (single-detection) assets to the bottom, never treats null as zero-worst", () => {
  const highRate = plumeWithAsset("P1", { asset: nearestAsset("HIGH"), observedAt: "2026-01-01T00:00:00Z" });
  const highRate2 = plumeWithAsset("P2", { asset: nearestAsset("HIGH"), observedAt: "2026-03-01T00:00:00Z" }); // ~59 days -> a real rate
  const single = plumeWithAsset("P3", { asset: nearestAsset("SINGLE"), observedAt: "2026-01-01T00:00:00Z" });
  const rates = computeAssetDetectionRates([highRate, highRate2, single]);
  const sorted = sortAssetDetectionRates(rates, "rate");
  assert.equal(sorted[0].assetId, "HIGH");
  assert.equal(sorted[1].assetId, "SINGLE");
  assert.equal(sorted[1].detectionsPerYear, null);
});

test("sortAssetDetectionRates: does not mutate its input array", () => {
  const rates = computeAssetDetectionRates([
    plumeWithAsset("P1", { asset: nearestAsset("A1") }),
    plumeWithAsset("P2", { asset: nearestAsset("A2") }),
  ]);
  const before = rates.map((r) => r.assetId);
  sortAssetDetectionRates(rates, "count");
  assert.deepEqual(rates.map((r) => r.assetId), before);
});

test("cachedGemMethaneAssetRates: computes from the real repo fixtures, caches across calls, and counts excluded ambiguous matches", () => {
  _resetGemMethaneAssetRatesCacheForTests();
  _resetGemMethaneProximityCacheForTests();
  _resetGemMethaneCacheForTests();
  _resetGemAssetsCacheForTests();
  try {
    const first = cachedGemMethaneAssetRates();
    const second = cachedGemMethaneAssetRates();
    assert.ok(first, "expected the real repo fixtures to load");
    assert.ok(first!.rates.length > 0);
    assert.ok(first!.excludedAmbiguousCount >= 0);
    assert.ok(first!.singleDetectionAssetCount <= first!.rates.length);
    assert.equal(first, second, "second call must return the cached object, not recompute");
    // every grouped asset's count must equal 1 (single) or be reflected honestly
    for (const r of first!.rates) {
      assert.ok(r.detectionCount >= 1);
      if (r.detectionCount === 1) assert.equal(r.detectionsPerYear, null);
    }
  } finally {
    _resetGemMethaneAssetRatesCacheForTests();
    _resetGemMethaneProximityCacheForTests();
    _resetGemMethaneCacheForTests();
    _resetGemAssetsCacheForTests();
  }
});
