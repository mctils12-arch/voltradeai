// gemChemicals — GEM "Global Chemicals Inventory" RAW overlay (CLAUDE.md
// RAW-vs-SIGNAL surface rule). Pins the "Feedstock" -> feedstock-family
// classifier (priority order over the release's real semicolon-list
// combinations), the packed "lat, lon" Coordinates parser (same shape as
// iron_steel_plants.json/iron_ore_mines.json), GEM's own sentinel strings
// for unreported values, the drop-not-infer rule (no coordinates, no name,
// no unique id), and the missing/corrupt-file degrade path.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  classifyFeedstockFamily,
  parseGemCoordinates,
  normalizeChemicalPlants,
  loadGemChemicals,
  cachedGemChemicals,
  _resetGemChemicalsCacheForTests,
} from "./gemChemicals";

function mkFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemchemicals-"));
  const fp = path.join(dir, "chemicals.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

const FULL_ROW = {
  "GEM plant ID": "P100000140001",
  "Plant name (English)": "Fertial Annaba Ammonia Plant",
  "Owner (English)": "Fertiberia [66%]; Algerian Government [33%]",
  Municipality: "Annaba",
  "Subnational unit": "Annaba Province",
  "Country/area": "Algeria",
  Region: "Africa",
  Coordinates: "36.866303, 7.768517",
  "Coordinate accuracy": "approximate",
  "GEM wiki page": "https://gem.wiki/Fertial_Annaba_Ammonia_Plant",
  "Primary products": "ammonia",
  "Secondary products": "urea",
  Feedstock: "natural gas",
};

test("classifyFeedstockFamily: coal beats every other listed feedstock", () => {
  assert.equal(classifyFeedstockFamily("coal"), "coal");
  assert.equal(classifyFeedstockFamily("coal; natural gas"), "coal");
  assert.equal(classifyFeedstockFamily("coal; natural gas"), "coal");
});

test("classifyFeedstockFamily: natural gas beats petroleum/NGL when no coal is listed", () => {
  assert.equal(classifyFeedstockFamily("natural gas"), "natural_gas");
  assert.equal(classifyFeedstockFamily("naphtha; natural gas"), "natural_gas");
  assert.equal(classifyFeedstockFamily("crude oil; natural gas"), "natural_gas");
  assert.equal(classifyFeedstockFamily("methane"), "natural_gas");
  assert.equal(classifyFeedstockFamily("liquefied natural gas (LNG)"), "natural_gas");
  assert.equal(classifyFeedstockFamily("coke oven gas"), "natural_gas");
});

test("classifyFeedstockFamily: petroleum-liquid tokens bucket when no coal/gas is listed", () => {
  assert.equal(classifyFeedstockFamily("naphtha"), "petroleum");
  assert.equal(classifyFeedstockFamily("crude oil"), "petroleum");
  assert.equal(classifyFeedstockFamily("ethane; naphtha"), "petroleum");
  assert.equal(classifyFeedstockFamily("condensate"), "petroleum");
});

test("classifyFeedstockFamily: NGL tokens bucket when no coal/gas/petroleum-liquid is listed", () => {
  assert.equal(classifyFeedstockFamily("ethane"), "ngl");
  assert.equal(classifyFeedstockFamily("ethane; propane"), "ngl");
  assert.equal(classifyFeedstockFamily("liquid petroleum gas (LPG)"), "ngl");
  assert.equal(classifyFeedstockFamily("mixed C4"), "ngl");
});

test("classifyFeedstockFamily: low-carbon tokens bucket when nothing higher-priority is listed", () => {
  assert.equal(classifyFeedstockFamily("green hydrogen"), "low_carbon");
  assert.equal(classifyFeedstockFamily("carbon dioxide; green hydrogen"), "low_carbon");
  assert.equal(classifyFeedstockFamily("biomass"), "low_carbon");
  assert.equal(classifyFeedstockFamily("bioethanol"), "low_carbon");
});

test("classifyFeedstockFamily: unrecognized/blank/unknown values fall to 'other', never guessed", () => {
  assert.equal(classifyFeedstockFamily("methanol"), "other");
  assert.equal(classifyFeedstockFamily("ethylene"), "other");
  assert.equal(classifyFeedstockFamily("benzene"), "other");
  assert.equal(classifyFeedstockFamily("unknown"), "other");
  assert.equal(classifyFeedstockFamily(""), "other");
  assert.equal(classifyFeedstockFamily(null), "other");
  assert.equal(classifyFeedstockFamily(undefined), "other");
});

test("classifyFeedstockFamily: case-insensitive", () => {
  assert.equal(classifyFeedstockFamily("COAL"), "coal");
  assert.equal(classifyFeedstockFamily("Natural Gas"), "natural_gas");
});

test("parseGemCoordinates: parses GEM's packed 'lat, lon' string", () => {
  assert.deepEqual(parseGemCoordinates("36.866303, 7.768517"), { lat: 36.866303, lon: 7.768517 });
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

test("normalizeChemicalPlants: maps GEM's raw columns to the clean schema", () => {
  const out = normalizeChemicalPlants([FULL_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "P100000140001",
    name: "Fertial Annaba Ammonia Plant",
    feedstockFamily: "natural_gas",
    feedstockRaw: "natural gas",
    primaryProducts: "ammonia",
    secondaryProducts: "urea",
    country: "Algeria",
    region: "Africa",
    municipality: "Annaba",
    subnationalUnit: "Annaba Province",
    coordinateAccuracy: "approximate",
    owner: "Fertiberia [66%]; Algerian Government [33%]",
    wiki: "https://gem.wiki/Fertial_Annaba_Ammonia_Plant",
    lat: 36.866303,
    lon: 7.768517,
  });
});

test("normalizeChemicalPlants: GEM's own 'unknown' sentinel feedstock stays null-raw but classifies as 'other'", () => {
  const out = normalizeChemicalPlants([{ ...FULL_ROW, Feedstock: "unknown" }]);
  assert.equal(out[0].feedstockRaw, null);
  assert.equal(out[0].feedstockFamily, "other");
});

test("normalizeChemicalPlants: drops rows with no usable coordinates (nothing to place on a map)", () => {
  assert.equal(normalizeChemicalPlants([{ ...FULL_ROW, Coordinates: "not-coordinates" }]).length, 0);
  assert.equal(normalizeChemicalPlants([{ ...FULL_ROW, Coordinates: undefined }]).length, 0);
});

test("normalizeChemicalPlants: drops rows with no name or no id", () => {
  assert.equal(normalizeChemicalPlants([{ ...FULL_ROW, "Plant name (English)": undefined }]).length, 0);
  assert.equal(normalizeChemicalPlants([{ ...FULL_ROW, "GEM plant ID": undefined }]).length, 0);
});

test("loadGemChemicals: reads a fixture file end-to-end, provenance intact", () => {
  const fp = mkFixture({
    provenance: {
      attribution: "Global Energy Monitor",
      license: "CC BY 4.0 (per-release Copyright sheets)",
      release: "Plant-level-data-Global-Chemicals-Inventory-November-2025-V1.xlsx",
    },
    plants: [FULL_ROW],
    counts: { plants: 1 },
  });
  const hit = loadGemChemicals(fp);
  assert.ok(hit);
  assert.equal(hit!.release, "Plant-level-data-Global-Chemicals-Inventory-November-2025-V1.xlsx");
  assert.equal(hit!.attribution, "Global Energy Monitor");
  assert.equal(hit!.plants.length, 1);
});

test("loadGemChemicals: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemChemicals("/nonexistent/path/chemicals.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemchemicals-bad-"));
  const fp = path.join(dir, "chemicals.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemChemicals(fp), null);
});

test("cachedGemChemicals: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemChemicalsCacheForTests();
  try {
    // Exercised against the actual checked-in datacore/gem/chemicals.json —
    // a realistic integration check as well as a cache-identity check.
    const first = cachedGemChemicals();
    const second = cachedGemChemicals();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemChemicalsCacheForTests();
  }
});
