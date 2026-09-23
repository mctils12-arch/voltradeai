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
  });
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
