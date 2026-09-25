// gemIronSteelPlants — GEM "Global Iron and Steel Tracker" RAW overlay
// (CLAUDE.md RAW-vs-SIGNAL surface rule). Pins the "Main production
// equipment" -> primary-technology classifier (live-verified priority
// order against the real release's 87 distinct semicolon-list
// combinations), the packed "lat, lon" Coordinates parser (same shape as
// iron_ore_mines.json), the mixed year-number/ISO-string/"unknown" date
// columns, GEM's own sentinel strings for unreported values, the
// drop-not-infer rule (no coordinates, no name, no unique id), and the
// missing/corrupt-file degrade path.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  classifyProductionTechnology,
  parseGemCoordinates,
  toDateOrNull,
  normalizeIronSteelPlants,
  loadGemIronSteelPlants,
  cachedGemIronSteelPlants,
  _resetGemIronSteelPlantsCacheForTests,
  classifyUnitStatus,
  normalizeSteelUnits,
  loadGemSteelUnits,
  groupSteelUnitsByPlant,
  type SteelFurnaceUnit,
} from "./gemIronSteelPlants";

function mkFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemironsteel-"));
  const fp = path.join(dir, "iron_steel_plants.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

const FULL_ROW = {
  "GEM plant ID": "P100000120882",
  "Plant name (English)": "Aba Iron and Steel Payas plant",
  Coordinates: "36.7474130, 36.2173300",
  "Coordinate accuracy": "exact",
  Municipality: "Payas",
  "Subnational unit": "Hatay",
  "Country/area": "Türkiye",
  Region: "Europe",
  "Category steel product": "crude, semi-finished, finished rolled",
  "Main production equipment": "BF; BOF; EAF",
  "Workforce size": 900,
  "Start date": 1983,
  "Retired date": "unknown",
  "Idled date": "N/A",
  Owner: "ABA Çelik Demir LŞ",
  "Parent (English)": "ABA Çelik Demir LŞ [100.0%]",
  "SOE status": "N/A",
  "GEM wiki page": "https://www.gem.wiki/Aba_Iron_and_Steel_Payas_plant",
};

test("classifyProductionTechnology: BF beats every other listed technology (integrated route)", () => {
  assert.equal(classifyProductionTechnology("BF; BOF; EAF"), "bf_bof");
  assert.equal(classifyProductionTechnology("BF"), "bf_bof");
  assert.equal(classifyProductionTechnology("BF; DRI; EAF"), "bf_bof");
});

test("classifyProductionTechnology: DRI beats EAF/IF when no BF is listed (green-steel-adjacent route)", () => {
  assert.equal(classifyProductionTechnology("DRI; EAF"), "dri");
  assert.equal(classifyProductionTechnology("DRI"), "dri");
  assert.equal(classifyProductionTechnology("DRI; IF"), "dri");
});

test("classifyProductionTechnology: EAF-only and IF-only buckets", () => {
  assert.equal(classifyProductionTechnology("EAF"), "eaf");
  assert.equal(classifyProductionTechnology("IF"), "if");
  assert.equal(classifyProductionTechnology("EAF; IF"), "eaf");
});

test("classifyProductionTechnology: a lone BOF (no stated BF) still buckets as the integrated route", () => {
  assert.equal(classifyProductionTechnology("BOF"), "bf_bof");
});

test("classifyProductionTechnology: unrecognized/blank/unspecified values fall to 'other', never guessed", () => {
  assert.equal(classifyProductionTechnology("Iron other/unspecified; Steel other/unspecified"), "other");
  assert.equal(classifyProductionTechnology(""), "other");
  assert.equal(classifyProductionTechnology(null), "other");
  assert.equal(classifyProductionTechnology(undefined), "other");
  assert.equal(classifyProductionTechnology("something-new"), "other");
});

test("classifyProductionTechnology: case-insensitive", () => {
  assert.equal(classifyProductionTechnology("bf; bof"), "bf_bof");
  assert.equal(classifyProductionTechnology("eaf"), "eaf");
});

test("parseGemCoordinates: parses GEM's packed 'lat, lon' string", () => {
  assert.deepEqual(parseGemCoordinates("36.7474130, 36.2173300"), { lat: 36.747413, lon: 36.21733 });
  assert.deepEqual(parseGemCoordinates("-23.5, 150.9"), { lat: -23.5, lon: 150.9 });
});

test("parseGemCoordinates: malformed/missing input returns null, never a guessed coordinate", () => {
  assert.equal(parseGemCoordinates(undefined), null);
  assert.equal(parseGemCoordinates(null), null);
  assert.equal(parseGemCoordinates(""), null);
  assert.equal(parseGemCoordinates("not-coordinates"), null);
  assert.equal(parseGemCoordinates("34.01,61.5,1.0"), null);
  assert.equal(parseGemCoordinates(1234), null);
});

test("toDateOrNull: a bare year-number formats to its integer string", () => {
  assert.equal(toDateOrNull(1983), "1983");
  assert.equal(toDateOrNull(1983.0), "1983");
});

test("toDateOrNull: a real ISO date string passes through unchanged", () => {
  assert.equal(toDateOrNull("2024-09-07"), "2024-09-07");
});

test("toDateOrNull: sentinel/missing values degrade to null", () => {
  assert.equal(toDateOrNull("unknown"), null);
  assert.equal(toDateOrNull("N/A"), null);
  assert.equal(toDateOrNull(null), null);
  assert.equal(toDateOrNull(undefined), null);
});

test("normalizeIronSteelPlants: maps GEM's raw columns to the clean schema", () => {
  const out = normalizeIronSteelPlants([FULL_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "P100000120882",
    name: "Aba Iron and Steel Payas plant",
    technology: "bf_bof",
    technologyRaw: "BF; BOF; EAF",
    categorySteelProduct: "crude, semi-finished, finished rolled",
    country: "Türkiye",
    region: "Europe",
    municipality: "Payas",
    subnationalUnit: "Hatay",
    coordinateAccuracy: "exact",
    workforceSize: 900,
    startDate: "1983",
    retiredDate: null,
    idledDate: null,
    owner: "ABA Çelik Demir LŞ",
    parent: "ABA Çelik Demir LŞ [100.0%]",
    soeStatus: null,
    wiki: "https://www.gem.wiki/Aba_Iron_and_Steel_Payas_plant",
    lat: 36.747413,
    lon: 36.21733,
    units: [],
  });
});

test("normalizeIronSteelPlants: attaches furnace units by 'GEM plant ID' when a units map is supplied", () => {
  const unit: SteelFurnaceUnit = {
    plantId: "P100000120882",
    unitId: "U100000144311",
    name: "unknown EAF (1)",
    furnaceType: "eaf",
    status: "operating",
    capacityTtpa: 1100,
    startDate: null,
    retiredDate: null,
    manufacturer: "AyTekno",
  };
  const map = new Map([["P100000120882", [unit]]]);
  const out = normalizeIronSteelPlants([FULL_ROW], map);
  assert.deepEqual(out[0].units, [unit]);
});

test("normalizeIronSteelPlants: a plant with no matching entry in the units map gets units: []", () => {
  const map = new Map([["some-other-plant-id", [{} as SteelFurnaceUnit]]]);
  const out = normalizeIronSteelPlants([FULL_ROW], map);
  assert.deepEqual(out[0].units, []);
});

test("normalizeIronSteelPlants: GEM's own 'unknown'/'N/A' sentinel cells stay null, never fabricated", () => {
  const out = normalizeIronSteelPlants([FULL_ROW]);
  assert.equal(out[0].retiredDate, null); // "unknown"
  assert.equal(out[0].idledDate, null); // "N/A"
  assert.equal(out[0].soeStatus, null); // "N/A"
});

test("normalizeIronSteelPlants: drops rows with no usable coordinates (nothing to place on a map)", () => {
  assert.equal(normalizeIronSteelPlants([{ ...FULL_ROW, Coordinates: "not-coordinates" }]).length, 0);
  assert.equal(normalizeIronSteelPlants([{ ...FULL_ROW, Coordinates: undefined }]).length, 0);
});

test("normalizeIronSteelPlants: drops rows with no name or no id", () => {
  assert.equal(normalizeIronSteelPlants([{ ...FULL_ROW, "Plant name (English)": undefined }]).length, 0);
  assert.equal(normalizeIronSteelPlants([{ ...FULL_ROW, "GEM plant ID": undefined }]).length, 0);
});

test("loadGemIronSteelPlants: reads a fixture file end-to-end, provenance intact", () => {
  const fp = mkFixture({
    provenance: {
      attribution: "Global Energy Monitor",
      license: "CC BY 4.0 (per-release Copyright sheets)",
      release: "Plant-level_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx",
    },
    plants: [FULL_ROW],
    counts: { plants: 1 },
  });
  const hit = loadGemIronSteelPlants(fp);
  assert.ok(hit);
  assert.equal(hit!.release, "Plant-level_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx");
  assert.equal(hit!.attribution, "Global Energy Monitor");
  assert.equal(hit!.plants.length, 1);
});

test("loadGemIronSteelPlants: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemIronSteelPlants("/nonexistent/path/iron_steel_plants.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemironsteel-bad-"));
  const fp = path.join(dir, "iron_steel_plants.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemIronSteelPlants(fp), null);
});

const FULL_UNIT_ROW = {
  "GEM plant ID": "P100000120882",
  "GEM unit ID": "U100000144311",
  "Unit name": "unknown EAF (1)",
  "Unit status": "operating",
  "Current capacity (ttpa)": 1100,
  "Start date": "unknown",
  "Retired date": "unknown",
  "Furnace manufacturer": "AyTekno",
};

test("classifyUnitStatus: recognizes GEM's own 8 catalogued values, case-insensitive", () => {
  assert.equal(classifyUnitStatus("operating"), "operating");
  assert.equal(classifyUnitStatus("Operating Pre-Retirement"), "operating pre-retirement");
  assert.equal(classifyUnitStatus("MOTHBALLED"), "mothballed");
  assert.equal(classifyUnitStatus("cancelled"), "cancelled");
});

test("classifyUnitStatus: unrecognized/blank/missing values fall to 'unknown', never guessed", () => {
  assert.equal(classifyUnitStatus(""), "unknown");
  assert.equal(classifyUnitStatus(null), "unknown");
  assert.equal(classifyUnitStatus(undefined), "unknown");
  assert.equal(classifyUnitStatus("idle"), "unknown");
});

test("normalizeSteelUnits: maps GEM's raw unit columns to the clean schema", () => {
  const out = normalizeSteelUnits("eaf", [FULL_UNIT_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    plantId: "P100000120882",
    unitId: "U100000144311",
    name: "unknown EAF (1)",
    furnaceType: "eaf",
    status: "operating",
    capacityTtpa: 1100,
    startDate: null, // "unknown" sentinel
    retiredDate: null, // "unknown" sentinel
    manufacturer: "AyTekno",
  });
});

test("normalizeSteelUnits: drops rows with no plant id or no unit id", () => {
  assert.equal(normalizeSteelUnits("eaf", [{ ...FULL_UNIT_ROW, "GEM plant ID": undefined }]).length, 0);
  assert.equal(normalizeSteelUnits("eaf", [{ ...FULL_UNIT_ROW, "GEM unit ID": undefined }]).length, 0);
});

test("normalizeSteelUnits: a missing/'unknown' capacity degrades to null, never fabricated", () => {
  const out = normalizeSteelUnits("bof", [{ ...FULL_UNIT_ROW, "Current capacity (ttpa)": "unknown" }]);
  assert.equal(out[0].capacityTtpa, null);
});

function mkUnitsFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemsteelunits-"));
  const fp = path.join(dir, "steel_units.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

test("loadGemSteelUnits: reads a fixture file end-to-end, concatenating all 4 sheets", () => {
  const fp = mkUnitsFixture({
    provenance: {
      attribution: "Global Energy Monitor",
      license: "CC BY 4.0 (per-release Copyright sheets)",
      release: "Steel_unit_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx",
    },
    eaf: [FULL_UNIT_ROW],
    bof: [{ ...FULL_UNIT_ROW, "GEM unit ID": "U2", "Unit status": "retired" }],
    induction: [],
    open_hearth: [],
    counts: { eaf: 1, bof: 1, induction: 0, open_hearth: 0 },
  });
  const hit = loadGemSteelUnits(fp);
  assert.ok(hit);
  assert.equal(hit!.release, "Steel_unit_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx");
  assert.equal(hit!.units.length, 2);
  assert.equal(hit!.units[0].furnaceType, "eaf");
  assert.equal(hit!.units[1].furnaceType, "bof");
});

test("loadGemSteelUnits: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemSteelUnits("/nonexistent/path/steel_units.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemsteelunits-bad-"));
  const fp = path.join(dir, "steel_units.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemSteelUnits(fp), null);
});

test("groupSteelUnitsByPlant: groups multiple units under the same plant id, preserves order", () => {
  const a: SteelFurnaceUnit = { ...normalizeSteelUnits("eaf", [FULL_UNIT_ROW])[0] };
  const b: SteelFurnaceUnit = { ...normalizeSteelUnits("bof", [{ ...FULL_UNIT_ROW, "GEM unit ID": "U2" }])[0] };
  const c: SteelFurnaceUnit = { ...normalizeSteelUnits("eaf", [{ ...FULL_UNIT_ROW, "GEM plant ID": "OTHER", "GEM unit ID": "U3" }])[0] };
  const grouped = groupSteelUnitsByPlant([a, b, c]);
  assert.deepEqual(grouped.get("P100000120882"), [a, b]);
  assert.deepEqual(grouped.get("OTHER"), [c]);
  assert.equal(grouped.size, 2);
});

test("cachedGemIronSteelPlants: joins real furnace-unit data from datacore/gem/steel_units.json onto real plants", () => {
  _resetGemIronSteelPlantsCacheForTests();
  try {
    const hit = cachedGemIronSteelPlants();
    assert.ok(hit, "expected the real repo fixture to load");
    const withUnits = hit!.plants.filter((p) => p.units.length > 0);
    assert.ok(withUnits.length > 0, "expected at least one real plant to have joined furnace units");
    for (const p of withUnits.slice(0, 5)) {
      for (const u of p.units) assert.equal(u.plantId, p.id);
    }
  } finally {
    _resetGemIronSteelPlantsCacheForTests();
  }
});

test("cachedGemIronSteelPlants: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemIronSteelPlantsCacheForTests();
  try {
    // Exercised against the actual checked-in
    // datacore/gem/iron_steel_plants.json — a realistic integration check
    // as well as a cache-identity check.
    const first = cachedGemIronSteelPlants();
    const second = cachedGemIronSteelPlants();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemIronSteelPlantsCacheForTests();
  }
});
