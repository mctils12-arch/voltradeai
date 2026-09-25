// gemSteelUnits — GEM "Global Iron and Steel Tracker" furnace-UNIT detail
// (datacore/gem/steel_units.json), joined onto iron_steel_plants.json by
// "GEM plant ID" (server/routes.ts's /api/data/iron-steel-plants route).
// Pins: the per-furnace-type normalizer, the drop-not-infer join-key rule,
// the counts-across-every-status / capacity-sums-operating-only split
// aggregateUnitsByPlant implements, and the missing/corrupt-file degrade
// path (matching gemIronSteelPlants.test.ts's own pattern for this file
// family).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  normalizeSteelUnits,
  aggregateUnitsByPlant,
  loadGemSteelUnits,
  cachedGemSteelUnits,
  cachedPlantFurnaceSummaries,
  _resetGemSteelUnitsCacheForTests,
} from "./gemSteelUnits";

function mkFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemsteelunits-"));
  const fp = path.join(dir, "steel_units.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

const EAF_ROW = {
  "GEM plant ID": "P1", "GEM unit ID": "U1", "Unit name": "EAF 1",
  "Unit status": "operating", "Current capacity (ttpa)": 1100,
};
const BOF_ROW = {
  "GEM plant ID": "P1", "GEM unit ID": "U2", "Unit name": "BOF 1",
  "Unit status": "retired", "Current capacity (ttpa)": 1950,
};

test("normalizeSteelUnits: maps GEM's raw columns to the clean schema, tags the given furnace type", () => {
  const out = normalizeSteelUnits([EAF_ROW], "eaf");
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    plantId: "P1", unitId: "U1", name: "EAF 1", furnaceType: "eaf",
    status: "operating", capacityTtpa: 1100,
  });
});

test("normalizeSteelUnits: GEM's own 'unknown' sentinel and missing capacity stay null, never fabricated", () => {
  const out = normalizeSteelUnits([{ "GEM plant ID": "P1", "Unit status": "unknown", "Current capacity (ttpa)": "unknown" }], "induction");
  assert.equal(out[0].status, null);
  assert.equal(out[0].capacityTtpa, null);
});

test("normalizeSteelUnits: drops rows with no plant ID (nothing to join to)", () => {
  assert.equal(normalizeSteelUnits([{ "Unit name": "orphan" }], "bof").length, 0);
  assert.equal(normalizeSteelUnits([{ "GEM plant ID": "unknown" }], "bof").length, 0);
});

test("aggregateUnitsByPlant: counts every unit across ALL statuses, never drops a retired unit from the count", () => {
  const summaries = aggregateUnitsByPlant([
    ...normalizeSteelUnits([EAF_ROW], "eaf"),
    ...normalizeSteelUnits([BOF_ROW], "bof"),
  ]);
  const s = summaries.get("P1")!;
  assert.equal(s.eaf, 1);
  assert.equal(s.bof, 1);
  assert.equal(s.induction, 0);
  assert.equal(s.openHearth, 0);
});

test("aggregateUnitsByPlant: sums capacity across operating/operating-pre-retirement units ONLY, excludes retired", () => {
  const summaries = aggregateUnitsByPlant([
    ...normalizeSteelUnits([EAF_ROW], "eaf"), // operating, 1100
    ...normalizeSteelUnits([BOF_ROW], "bof"), // retired, 1950 — must NOT count
  ]);
  const s = summaries.get("P1")!;
  assert.equal(s.operatingCount, 1);
  assert.equal(s.operatingCapacityTtpa, 1100, "retired unit's capacity must not inflate the operating total");
});

test("aggregateUnitsByPlant: 'operating pre-retirement' counts as operating capacity too", () => {
  const summaries = aggregateUnitsByPlant(
    normalizeSteelUnits([{ "GEM plant ID": "P2", "Unit status": "operating pre-retirement", "Current capacity (ttpa)": 500 }], "induction"),
  );
  assert.equal(summaries.get("P2")!.operatingCapacityTtpa, 500);
});

test("aggregateUnitsByPlant: a plant with only non-operating units reports operatingCapacityTtpa null, never 0", () => {
  const summaries = aggregateUnitsByPlant(normalizeSteelUnits([BOF_ROW], "bof")); // retired only
  const s = summaries.get("P1")!;
  assert.equal(s.operatingCount, 0);
  assert.equal(s.operatingCapacityTtpa, null);
});

test("aggregateUnitsByPlant: a plant absent from the unit list simply has no map entry (not a zero-filled row)", () => {
  const summaries = aggregateUnitsByPlant(normalizeSteelUnits([EAF_ROW], "eaf"));
  assert.equal(summaries.has("P404"), false);
});

test("loadGemSteelUnits: reads a fixture file end-to-end across all four furnace-type arrays, provenance intact", () => {
  const fp = mkFixture({
    provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0 (per-release Copyright sheets)", release: "Steel_unit_data_June_2026_V1.xlsx" },
    eaf: [EAF_ROW], bof: [BOF_ROW], induction: [], open_hearth: [],
    counts: { eaf: 1, bof: 1, induction: 0, open_hearth: 0 },
  });
  const hit = loadGemSteelUnits(fp);
  assert.ok(hit);
  assert.equal(hit!.release, "Steel_unit_data_June_2026_V1.xlsx");
  assert.equal(hit!.attribution, "Global Energy Monitor");
  assert.equal(hit!.units.length, 2);
  assert.deepEqual(hit!.units.map((u) => u.furnaceType).sort(), ["bof", "eaf"]);
});

test("loadGemSteelUnits: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemSteelUnits("/nonexistent/path/steel_units.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemsteelunits-bad-"));
  const fp = path.join(dir, "steel_units.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemSteelUnits(fp), null);
});

test("cachedGemSteelUnits / cachedPlantFurnaceSummaries: cache across calls against the real repo fixture", () => {
  _resetGemSteelUnitsCacheForTests();
  try {
    const first = cachedGemSteelUnits();
    const second = cachedGemSteelUnits();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
    const s1 = cachedPlantFurnaceSummaries();
    const s2 = cachedPlantFurnaceSummaries();
    assert.ok(s1);
    assert.equal(s1, s2, "second call must return the cached map, not re-aggregate");
    // Live-verified plant from the real release (see gemSteelUnits.ts's
    // header comment): P100000120882 has exactly one operating EAF unit
    // at 1100 ttpa.
    const known = s1!.get("P100000120882");
    assert.ok(known, "expected the real repo release to catalogue at least one unit for this plant");
    assert.equal(known!.eaf, 1);
    assert.equal(known!.operatingCapacityTtpa, 1100);
  } finally {
    _resetGemSteelUnitsCacheForTests();
  }
});
