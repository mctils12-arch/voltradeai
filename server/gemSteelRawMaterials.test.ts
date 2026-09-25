// gemSteelRawMaterials — GEM steel-industry met-coal/iron-ore national
// balance sheet RAW overlay (CLAUDE.md RAW-vs-SIGNAL surface rule). Pins
// the "Global" world-aggregate exclusion, the numeric/"unknown"-sentinel
// normalization, the missing/corrupt-file degrade path, and the
// country-name-alias choropleth join (the genuinely new part of this
// module vs. the rest of the point-layer GEM family) — including the
// ecological-fallacy-style guard that every admin0 polygon survives the
// join whether or not GEM reports a value for it.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  normalizeCountryBalance,
  loadGemSteelRawMaterials,
  cachedGemSteelRawMaterials,
  _resetGemSteelRawMaterialsCacheForTests,
  joinCountryChoropleth,
  cachedSteelRawMaterialsGeoJSON,
  _resetSteelRawMaterialsGeoCacheForTests,
  COUNTRY_NAME_ALIASES,
} from "./gemSteelRawMaterials";

function mkFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemsteelraw-"));
  const fp = path.join(dir, "steel_raw_materials.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

const FULL_ROW = {
  Country: "Algeria",
  "Met coal mined (ttpa)": 0,
  "Iron ore mined (ttpa)": 2288.25,
  "Met coal consumed by pig iron production (ttpa)": 1170,
  "Iron ore consumed by pig iron production (ttpa)": 2400,
  "Iron ore consumed by DRI production (ttpa)": 5880,
  "Total iron ore consumed by pig iron and DRI production (ttpa)": 8280,
  "Pig iron produced (ttpa)": 1500,
  "DRI produced (ttpa)": 4200,
};

test("normalizeCountryBalance: maps GEM's raw columns to the clean schema", () => {
  const out = normalizeCountryBalance([FULL_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    country: "Algeria",
    metCoalMinedTtpa: 0,
    ironOreMinedTtpa: 2288.25,
    metCoalConsumedPigIronTtpa: 1170,
    ironOreConsumedPigIronTtpa: 2400,
    ironOreConsumedDriTtpa: 5880,
    ironOreConsumedTotalTtpa: 8280,
    pigIronProducedTtpa: 1500,
    driProducedTtpa: 4200,
  });
});

test("normalizeCountryBalance: a legitimately reported 0 stays 0, never coerced to null", () => {
  const out = normalizeCountryBalance([{ ...FULL_ROW, "Iron ore mined (ttpa)": 0 }]);
  assert.equal(out[0].ironOreMinedTtpa, 0);
});

test("normalizeCountryBalance: GEM's own 'unknown' sentinel degrades to null, never guessed", () => {
  const out = normalizeCountryBalance([{ ...FULL_ROW, "Pig iron produced (ttpa)": "unknown" }]);
  assert.equal(out[0].pigIronProducedTtpa, null);
});

test("normalizeCountryBalance: drops the 'Global' world-aggregate row — not a real country", () => {
  const out = normalizeCountryBalance([FULL_ROW, { ...FULL_ROW, Country: "Global" }]);
  assert.equal(out.length, 1);
  assert.equal(out[0].country, "Algeria");
});

test("normalizeCountryBalance: drops rows with no country name", () => {
  assert.equal(normalizeCountryBalance([{ ...FULL_ROW, Country: undefined }]).length, 0);
  assert.equal(normalizeCountryBalance([{ ...FULL_ROW, Country: "" }]).length, 0);
});

test("loadGemSteelRawMaterials: reads a fixture file end-to-end, provenance intact", () => {
  const fp = mkFixture({
    provenance: {
      attribution: "Global Energy Monitor",
      license: "CC BY 4.0 (per-release Copyright sheets)",
      release: "Production-Consumption-of-Met-Coal-Iron-Ore-by-Steel-Industry-December-2025-Standard-Copy-V1.xlsx",
    },
    country_balance: [FULL_ROW],
    counts: { country_balance: 1 },
  });
  const hit = loadGemSteelRawMaterials(fp);
  assert.ok(hit);
  assert.equal(hit!.attribution, "Global Energy Monitor");
  assert.equal(hit!.balances.length, 1);
});

test("loadGemSteelRawMaterials: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemSteelRawMaterials("/nonexistent/path/steel_raw_materials.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemsteelraw-bad-"));
  const fp = path.join(dir, "steel_raw_materials.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemSteelRawMaterials(fp), null);
});

test("cachedGemSteelRawMaterials: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemSteelRawMaterialsCacheForTests();
  try {
    // Exercised against the actual checked-in datacore/gem/steel_raw_materials.json.
    const first = cachedGemSteelRawMaterials();
    const second = cachedGemSteelRawMaterials();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemSteelRawMaterialsCacheForTests();
  }
});

// ── country-choropleth geometry join ─────────────────────────────────────

const FAKE_ADMIN0 = {
  type: "FeatureCollection" as const,
  features: [
    { type: "Feature" as const, properties: { name: "Algeria", iso3: "DZA" }, geometry: { type: "Polygon", coordinates: [] } },
    { type: "Feature" as const, properties: { name: "United States of America", iso3: "USA" }, geometry: { type: "Polygon", coordinates: [] } },
    { type: "Feature" as const, properties: { name: "Monaco", iso3: "MCO" }, geometry: { type: "Polygon", coordinates: [] } },
  ],
};

test("joinCountryChoropleth: matches a direct name and carries every field into snake_case properties", () => {
  const geo = joinCountryChoropleth(FAKE_ADMIN0, normalizeCountryBalance([FULL_ROW]));
  const algeria = geo.features.find((f) => (f.properties as any).name === "Algeria")!;
  assert.equal((algeria.properties as any).has_data, true);
  assert.equal((algeria.properties as any).iron_ore_mined_ttpa, 2288.25);
  assert.equal((algeria.properties as any).pig_iron_produced_ttpa, 1500);
});

test("joinCountryChoropleth: resolves a GEM name through COUNTRY_NAME_ALIASES to the NE spelling", () => {
  assert.equal(COUNTRY_NAME_ALIASES["United States"], "United States of America");
  const geo = joinCountryChoropleth(
    FAKE_ADMIN0,
    normalizeCountryBalance([{ ...FULL_ROW, Country: "United States", "Iron ore mined (ttpa)": 33240 }]),
  );
  const usa = geo.features.find((f) => (f.properties as any).name === "United States of America")!;
  assert.equal((usa.properties as any).has_data, true);
  assert.equal((usa.properties as any).iron_ore_mined_ttpa, 33240);
});

test("joinCountryChoropleth: a polygon with no matching GEM record keeps its shape, has_data:false, all fields null (never dropped, never a false zero)", () => {
  const geo = joinCountryChoropleth(FAKE_ADMIN0, normalizeCountryBalance([FULL_ROW]));
  assert.equal(geo.features.length, 3, "every admin0 feature survives the join");
  const monaco = geo.features.find((f) => (f.properties as any).name === "Monaco")!;
  assert.equal((monaco.properties as any).has_data, false);
  assert.equal((monaco.properties as any).iron_ore_mined_ttpa, null);
});

test("joinCountryChoropleth: 'Global' never reaches the join (excluded upstream by normalizeCountryBalance)", () => {
  const geo = joinCountryChoropleth(FAKE_ADMIN0, normalizeCountryBalance([FULL_ROW, { ...FULL_ROW, Country: "Global" }]));
  assert.equal(geo.features.every((f) => (f.properties as any).name !== "Global"), true);
});

test("cachedSteelRawMaterialsGeoJSON: joins the real repo balance sheet onto the real repo admin0 boundaries", () => {
  _resetGemSteelRawMaterialsCacheForTests();
  _resetSteelRawMaterialsGeoCacheForTests();
  try {
    const hit = cachedSteelRawMaterialsGeoJSON();
    assert.ok(hit, "expected the real repo fixtures to load and join");
    assert.ok(hit!.geo.features.length > 100, "expected the full ~177-feature admin0 set");
    assert.ok(hit!.matched > 0, "expected at least some countries to match by name");
    assert.equal(hit!.matched + hit!.unmatched, hit!.geo.features.length);
  } finally {
    _resetGemSteelRawMaterialsCacheForTests();
    _resetSteelRawMaterialsGeoCacheForTests();
  }
});
