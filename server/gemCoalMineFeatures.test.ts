// gemCoalMineFeatures — GEM "Coal Mine Boundaries and Methane Sources"
// RAW overlay (CLAUDE.md RAW-vs-SIGNAL surface rule). Pins the drop-not-
// infer rule (no id / no geometry), the per-feature build_version lookup
// (this file has no top-level provenance object, unlike methane_emitters),
// the missing/corrupt-file degrade path, and the per-process cache.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  normalizeCoalMineFeatures,
  findBuildVersion,
  loadGemCoalMineFeatures,
  cachedGemCoalMineFeatures,
  _resetGemCoalMineFeaturesCacheForTests,
} from "./gemCoalMineFeatures";

function mkGzFixture(body: any): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemcoalfeat-"));
  const fp = path.join(dir, "coal_mine_features.geojson.gz");
  fs.writeFileSync(fp, zlib.gzipSync(JSON.stringify(body)));
  return fp;
}

const FULL_FEATURE = {
  type: "Feature",
  properties: {
    id: "M0086.B1",
    "mine feature category": "mine boundary",
    "mine feature subcategory": "surface",
    "data source date": "2025-02-07 00:00:00",
    notes: "a very long citation blob that should never reach the API payload",
    description: "Olive Downs mine boundary",
    "GEM Mine ID": "M0086",
    "Owners": "Pembroke Resources [100%]",
    "Parent Company": "Pembroke Resources",
    "GEM Wiki Page (ENG)": "https://www.gem.wiki/Olive_Downs_coal_mine",
    "Coal Grade": "Thermal & Met",
    "Mine Name": "Olive Downs Coal Mine",
    "Country / Area": "Australia",
    build_version: "Coal Mine Boundaries and Methane Sources - version 1.0.2 (built on May 12 2026 09.50 EST)",
  },
  geometry: { type: "Polygon", coordinates: [[[148.1, -22.1], [148.2, -22.1], [148.2, -22.2], [148.1, -22.1]]] },
};

test("normalizeCoalMineFeatures: maps GEM's raw properties to the clean schema, drops notes", () => {
  const out = normalizeCoalMineFeatures([FULL_FEATURE as any]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "M0086.B1",
    mineId: "M0086",
    mineName: "Olive Downs Coal Mine",
    category: "mine boundary",
    subcategory: "surface",
    description: "Olive Downs mine boundary",
    owners: "Pembroke Resources [100%]",
    parent: "Pembroke Resources",
    country: "Australia",
    coalGrade: "Thermal & Met",
    wiki: "https://www.gem.wiki/Olive_Downs_coal_mine",
    dataSourceDate: "2025-02-07 00:00:00",
    geometry: FULL_FEATURE.geometry,
  });
  assert.equal((out[0] as any).notes, undefined, "the bulky citation prose must never reach the payload");
});

test("normalizeCoalMineFeatures: GEM's own blank cells stay null, never fabricated", () => {
  const sparse = {
    type: "Feature",
    properties: { id: "M0999.V1", "mine feature category": "ventilation system" },
    geometry: { type: "Point", coordinates: [10, 20] },
  };
  const out = normalizeCoalMineFeatures([sparse as any]);
  assert.equal(out.length, 1);
  assert.equal(out[0].mineId, null);
  assert.equal(out[0].mineName, null);
  assert.equal(out[0].owners, null);
});

test("normalizeCoalMineFeatures: drops features with no id (nothing to key on)", () => {
  const noId = { ...FULL_FEATURE, properties: { ...FULL_FEATURE.properties, id: undefined } };
  assert.equal(normalizeCoalMineFeatures([noId as any]).length, 0);
});

test("normalizeCoalMineFeatures: drops features with no geometry (nothing to place on a map)", () => {
  const noGeom = { ...FULL_FEATURE, geometry: undefined };
  assert.equal(normalizeCoalMineFeatures([noGeom as any]).length, 0);
});

test("normalizeCoalMineFeatures: never throws on empty or garbage input", () => {
  assert.deepEqual(normalizeCoalMineFeatures([]), []);
  assert.deepEqual(normalizeCoalMineFeatures(undefined as any), []);
});

test("findBuildVersion: reads the release string off the first feature that carries one", () => {
  const noVersion = { type: "Feature", properties: { id: "X" }, geometry: { type: "Point", coordinates: [0, 0] } };
  assert.equal(findBuildVersion([noVersion as any, FULL_FEATURE as any]), FULL_FEATURE.properties.build_version);
});

test("findBuildVersion: null when nothing carries a build_version", () => {
  const noVersion = { type: "Feature", properties: { id: "X" }, geometry: { type: "Point", coordinates: [0, 0] } };
  assert.equal(findBuildVersion([noVersion as any]), null);
  assert.equal(findBuildVersion([]), null);
});

test("loadGemCoalMineFeatures: reads + gunzips the fixture, extracts build_version + features", () => {
  const fp = mkGzFixture({ type: "FeatureCollection", name: "test", features: [FULL_FEATURE] });
  const hit = loadGemCoalMineFeatures(fp);
  assert.ok(hit);
  assert.equal(hit!.buildVersion, FULL_FEATURE.properties.build_version);
  assert.equal(hit!.features.length, 1);
  assert.equal(hit!.features[0].id, "M0086.B1");
});

test("loadGemCoalMineFeatures: a missing file degrades to null, never throws", () => {
  assert.equal(loadGemCoalMineFeatures("/nonexistent/path/coal_mine_features.geojson.gz"), null);
});

test("loadGemCoalMineFeatures: a corrupt (non-gzip) file degrades to null, never throws", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemcoalfeat-corrupt-"));
  const fp = path.join(dir, "coal_mine_features.geojson.gz");
  fs.writeFileSync(fp, "not actually gzip");
  assert.equal(loadGemCoalMineFeatures(fp), null);
});

test("cachedGemCoalMineFeatures: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemCoalMineFeaturesCacheForTests();
  try {
    // Exercised against the actual checked-in
    // datacore/gem/coal_mine_features.geojson.gz — a realistic
    // integration check as well as a cache-identity check.
    const first = cachedGemCoalMineFeatures();
    const second = cachedGemCoalMineFeatures();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemCoalMineFeaturesCacheForTests();
  }
});
