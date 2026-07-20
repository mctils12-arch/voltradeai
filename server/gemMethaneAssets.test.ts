// gemMethaneAssets — GEM oil/gas-extraction + coal-mine registries,
// normalized to the common point-asset shape the proximity join consumes.
// Pins the field-name mapping, the drop-not-infer rule (no id / no
// coordinates), the coal-tracker's null-operator honesty (no separate
// Operator column exists there), and the missing-file degrade path.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  normalizeOilGasFields, normalizeCoalMines, loadGemAssets,
  cachedGemAssets, _resetGemAssetsCacheForTests,
} from "./gemMethaneAssets";
import { repoDataPath } from "./repoFiles";

function mkGzFixture(name: string, body: any): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemassets-"));
  const fp = path.join(dir, name);
  fs.writeFileSync(fp, zlib.gzipSync(JSON.stringify(body)));
  return fp;
}

const OIL_GAS_ROW = {
  "Unit ID": "L100000321014",
  "Unit Name": "10103FF/Cold Lake Oil Asset (Alberta, Canada)",
  "Fuel type": "oil",
  "Operator": "Strathcona Resources",
  "Owner(s)": "",
  "Parent(s)": "",
  "Latitude": "54.5340000",
  "Longitude": "-110.3941000",
};

const COAL_MINE_ROW = {
  "GEM Mine ID": "M3354",
  "Mine Name": "#1 Coal Mine",
  "Owners": "Eagle One Mining Inc",
  "Parent Company": "Eagle One Mining Inc [100.0%]",
  "Latitude": 37.406111,
  "Longitude": -82.882778,
};

test("normalizeOilGasFields: maps GEM's raw columns, string lat/lon parsed", () => {
  const out = normalizeOilGasFields([OIL_GAS_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "L100000321014", kind: "oil_gas_extraction",
    name: "10103FF/Cold Lake Oil Asset (Alberta, Canada)",
    operator: "Strathcona Resources", owner: null, parent: null,
    lat: 54.534, lon: -110.3941,
  });
});

test("normalizeOilGasFields: drops rows with no Unit ID or no coordinates", () => {
  assert.equal(normalizeOilGasFields([{ ...OIL_GAS_ROW, "Unit ID": undefined }]).length, 0);
  assert.equal(normalizeOilGasFields([{ ...OIL_GAS_ROW, "Latitude": undefined }]).length, 0);
});

test("normalizeCoalMines: maps GEM's raw columns, operator stays null (no source column)", () => {
  const out = normalizeCoalMines([COAL_MINE_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "M3354", kind: "coal_mine",
    name: "#1 Coal Mine",
    operator: null, owner: "Eagle One Mining Inc", parent: "Eagle One Mining Inc [100.0%]",
    lat: 37.406111, lon: -82.882778,
  });
});

test("normalizeCoalMines: drops rows with no GEM Mine ID or no coordinates", () => {
  assert.equal(normalizeCoalMines([{ ...COAL_MINE_ROW, "GEM Mine ID": undefined }]).length, 0);
  assert.equal(normalizeCoalMines([{ ...COAL_MINE_ROW, "Longitude": undefined }]).length, 0);
});

test("normalizeOilGasFields/normalizeCoalMines: never throw on empty or garbage rows", () => {
  assert.deepEqual(normalizeOilGasFields([]), []);
  assert.deepEqual(normalizeCoalMines(undefined as any), []);
});

test("loadGemAssets: reads + gunzips both fixtures and combines them", () => {
  const oilGasFp = mkGzFixture("oil_gas_extraction.json.gz", {
    provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0", release: "OGE_TEST" },
    fields: [OIL_GAS_ROW],
  });
  const coalMineFp = mkGzFixture("coal_mine_tracker.json.gz", {
    provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0", release: "CMT_TEST" },
    non_closed: [COAL_MINE_ROW],
  });
  const hit = loadGemAssets(oilGasFp, coalMineFp);
  assert.equal(hit.assets.length, 2);
  assert.equal(hit.provenance.oilGas?.release, "OGE_TEST");
  assert.equal(hit.provenance.coalMine?.release, "CMT_TEST");
});

test("loadGemAssets: a missing registry drops just that registry's assets, never throws", () => {
  const coalMineFp = mkGzFixture("coal_mine_tracker.json.gz", {
    provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0", release: "CMT_TEST" },
    non_closed: [COAL_MINE_ROW],
  });
  const hit = loadGemAssets("/nonexistent/oil_gas_extraction.json.gz", coalMineFp);
  assert.equal(hit.assets.length, 1);
  assert.equal(hit.assets[0].kind, "coal_mine");
  assert.equal(hit.provenance.oilGas, null);
});

test("RATCHET [REPAIR 2026-07-20]: loadGemAssets resolves BOTH registries under the real " +
     "production directory shape (dist/ only, no repo-tree datacore/) — the exact layout " +
     "script/build.ts must stage into and the exact gap that let matchedCount silently go to 0 " +
     "on prod after gate-2(a) shipped (repoFiles.test.ts's build.ts string-pin didn't catch it " +
     "because this file was never listed there; this test proves the READ side under that same " +
     "shape instead of trusting a repo-tree checkout to mask the miss, same class of gap this " +
     "suite's own cachedGemAssets test above has always had).", () => {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), "gemassets-prodlayout-"));
  const distGemDir = path.join(cwd, "dist", "datacore", "gem");
  fs.mkdirSync(distGemDir, { recursive: true });
  fs.writeFileSync(path.join(distGemDir, "oil_gas_extraction.json.gz"), zlib.gzipSync(JSON.stringify({
    provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0", release: "OGE_PROD" },
    fields: [OIL_GAS_ROW],
  })));
  fs.writeFileSync(path.join(distGemDir, "coal_mine_tracker.json.gz"), zlib.gzipSync(JSON.stringify({
    provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0", release: "CMT_PROD" },
    non_closed: [COAL_MINE_ROW],
  })));
  const hit = loadGemAssets(
    repoDataPath("datacore/gem/oil_gas_extraction.json.gz", cwd),
    repoDataPath("datacore/gem/coal_mine_tracker.json.gz", cwd),
  );
  assert.equal(hit.assets.length, 2, "both registries must resolve via the dist/ fallback, not degrade to empty");
  assert.equal(hit.provenance.oilGas?.release, "OGE_PROD");
  assert.equal(hit.provenance.coalMine?.release, "CMT_PROD");
});

test("cachedGemAssets: caches across calls (parsed once per process)", () => {
  _resetGemAssetsCacheForTests();
  try {
    const first = cachedGemAssets();
    const second = cachedGemAssets();
    assert.ok(first.assets.length > 0, "expected the real repo fixtures to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemAssetsCacheForTests();
  }
});
