// gemMethane — GMET satellite methane-plume RAW overlay (static reference
// dataset, CLAUDE.md RAW-vs-SIGNAL surface rule). Pins the raw-GEM-column
// normalization, the drop-not-infer rule (no id / no coordinates), null
// honesty for GEM's own blank cells, the missing/corrupt-file degrade
// path, and the per-process cache.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { normalizePlumes, loadGemMethane, cachedGemMethane, _resetGemMethaneCacheForTests } from "./gemMethane";

function mkGzFixture(body: any): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemmethane-"));
  const fp = path.join(dir, "methane_emitters.json.gz");
  fs.writeFileSync(fp, zlib.gzipSync(JSON.stringify(body)));
  return fp;
}

const FULL_ROW = {
  "GEM Methane Plume ID": "CH4T1",
  "Name": "California State Methane Observation 2020-11-09, 1",
  "GEM Wiki": "https://www.gem.wiki/CH4T1",
  "Satellite Data Provider": "CarbonMapper",
  "Instrument": "Global Airborne Observatory",
  "Observation Date": "2020-11-09T18:47:16+00",
  "Emissions (kg/hr)": 37.54922973,
  "Emissions Uncertainty (kg/hr)": 10.09112315,
  "Type of Infrastructure": "wellpad",
  "Infrastructure Notes": "Appears to be on a wellpad, but no associated well data",
  "Subnational Unit": "California State",
  "Country/Area": "United States",
  "Plume Origin Latitude": 35.08225086,
  "Plume Origin Longitude": -119.3044323,
};

test("normalizePlumes: maps GEM's raw spreadsheet columns to the clean schema", () => {
  const out = normalizePlumes([FULL_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "CH4T1",
    name: "California State Methane Observation 2020-11-09, 1",
    wiki: "https://www.gem.wiki/CH4T1",
    provider: "CarbonMapper",
    instrument: "Global Airborne Observatory",
    observedAt: "2020-11-09T18:47:16+00",
    emissionsKgHr: 37.54922973,
    emissionsUncertaintyKgHr: 10.09112315,
    infrastructureType: "wellpad",
    infrastructureNotes: "Appears to be on a wellpad, but no associated well data",
    subnationalUnit: "California State",
    country: "United States",
    lat: 35.08225086,
    lon: -119.3044323,
  });
});

test("normalizePlumes: GEM's own blank cells stay null, never fabricated", () => {
  const sparse = {
    "GEM Methane Plume ID": "CH4T2",
    "Name": "Sparse row",
    "Observation Date": "2020-11-09T18:47:16+00",
    "Plume Origin Latitude": 35.3679308,
    "Plume Origin Longitude": -119.6849953,
    // no Emissions, no Type of Infrastructure, no Instrument
  };
  const out = normalizePlumes([sparse]);
  assert.equal(out.length, 1);
  assert.equal(out[0].emissionsKgHr, null);
  assert.equal(out[0].infrastructureType, null);
  assert.equal(out[0].instrument, null);
});

test("normalizePlumes: drops rows with no id (nothing to key on)", () => {
  const noId = { ...FULL_ROW, "GEM Methane Plume ID": undefined };
  assert.equal(normalizePlumes([noId]).length, 0);
});

test("normalizePlumes: drops rows with no origin coordinates (nothing to map)", () => {
  const noCoords = { ...FULL_ROW, "Plume Origin Latitude": undefined, "Plume Origin Longitude": undefined };
  assert.equal(normalizePlumes([noCoords]).length, 0);
});

test("normalizePlumes: never throws on an empty or garbage rows array", () => {
  assert.deepEqual(normalizePlumes([]), []);
  assert.deepEqual(normalizePlumes(undefined as any), []);
});

test("loadGemMethane: reads + gunzips the fixture and carries provenance through", () => {
  const fp = mkGzFixture({
    provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0", release: "GMET_V3_TEST" },
    plumes: [FULL_ROW],
    pipelines: [{ unrelated: "sheet" }], // present in the real file — must be ignored, not read
  });
  const hit = loadGemMethane(fp);
  assert.ok(hit);
  assert.equal(hit!.provenance.release, "GMET_V3_TEST");
  assert.equal(hit!.plumes.length, 1);
  assert.equal(hit!.plumes[0].id, "CH4T1");
});

test("loadGemMethane: a missing file degrades to null, never throws", () => {
  assert.equal(loadGemMethane("/nonexistent/path/methane_emitters.json.gz"), null);
});

test("loadGemMethane: a corrupt (non-gzip) file degrades to null, never throws", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemmethane-corrupt-"));
  const fp = path.join(dir, "methane_emitters.json.gz");
  fs.writeFileSync(fp, "not actually gzip");
  assert.equal(loadGemMethane(fp), null);
});

test("cachedGemMethane: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemMethaneCacheForTests();
  try {
    // No fp param on cachedGemMethane() (it resolves the real repo file via
    // repoDataPath internally) — exercised against the actual checked-in
    // datacore/gem/methane_emitters.json.gz, which is a realistic
    // integration check as well as a cache-identity check.
    const first = cachedGemMethane();
    const second = cachedGemMethane();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemMethaneCacheForTests();
  }
});
