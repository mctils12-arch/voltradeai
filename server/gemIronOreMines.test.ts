// gemIronOreMines — GEM "Global Iron Ore Mines Tracker" RAW overlay
// (CLAUDE.md RAW-vs-SIGNAL surface rule). Pins the Operating status ->
// 7-bucket classifier (live-verified against the real release's exact
// value set, including its "unknown" bucket used as a real GEM value,
// not just a fallback), the packed "lat, lon" Coordinates parser (this
// release has no separate Latitude/Longitude columns, unlike
// coal_terminals.json), GEM's own sentinel strings for unreported
// numbers (including "unkonwn" — a typo present in the checked-in
// release itself), the drop-not-infer rule (no coordinates, no name, no
// unique id), and the missing/corrupt-file degrade path.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  classifyMineStatus,
  parseGemCoordinates,
  normalizeIronOreMines,
  loadGemIronOreMines,
  cachedGemIronOreMines,
  _resetGemIronOreMinesCacheForTests,
} from "./gemIronOreMines";

function mkFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemironore-"));
  const fp = path.join(dir, "iron_ore_mines.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

const FULL_ROW = {
  "GEM Asset ID": "P100000129290",
  "Asset name (English)": "Ghoryan Mine",
  Coordinates: "34.012090, 61.564895",
  "Coordinate accuracy": "approximate",
  Municipality: "unknown",
  "Subnational unit": "Herat",
  "Country/Area": "Afghanistan",
  Region: "Asia Pacific",
  "Production 2024 (ttpa)": 1408,
  "Production 2023 (ttpa)": "N/A",
  "Production 2022 (ttpa)": "unkonwn",
  "Design capacity (ttpa)": 2500,
  "Total reserves (proven and probable, thousand metric tonnes)": "unknown",
  "Total resource (inferred, indicated and measured, thousand metric tonnes)": 90000,
  "Operating status": "proposed",
  "Start date": "unknown",
  "Stop date": "N/A",
  Owner: "unknown",
  "Owner GEM Entity ID": "E100000132388",
  Parent: "--",
  "Parent GEM Entity ID": "--",
  "GEM wiki page URL": "https://www.gem.wiki/Ghoryan_Mine",
};

test("classifyMineStatus: the 7 catalogued buckets map directly, including 'unknown' as a real value", () => {
  assert.equal(classifyMineStatus("operating"), "operating");
  assert.equal(classifyMineStatus("proposed"), "proposed");
  assert.equal(classifyMineStatus("mothballed"), "mothballed");
  assert.equal(classifyMineStatus("retired"), "retired");
  assert.equal(classifyMineStatus("shelved"), "shelved");
  assert.equal(classifyMineStatus("cancelled"), "cancelled");
  assert.equal(classifyMineStatus("unknown"), "unknown");
});

test("classifyMineStatus: case-insensitive, and an unrecognized/blank value falls back to 'unknown', never guessed", () => {
  assert.equal(classifyMineStatus("Operating"), "operating");
  assert.equal(classifyMineStatus("OPERATING"), "operating");
  assert.equal(classifyMineStatus("something-new"), "unknown");
  assert.equal(classifyMineStatus(""), "unknown");
  assert.equal(classifyMineStatus(null), "unknown");
  assert.equal(classifyMineStatus(undefined), "unknown");
});

test("parseGemCoordinates: parses GEM's packed 'lat, lon' string", () => {
  assert.deepEqual(parseGemCoordinates("34.012090, 61.564895"), { lat: 34.01209, lon: 61.564895 });
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

test("normalizeIronOreMines: maps GEM's raw columns to the clean schema", () => {
  const out = normalizeIronOreMines([FULL_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "P100000129290",
    name: "Ghoryan Mine",
    status: "proposed",
    statusRaw: "proposed",
    country: "Afghanistan",
    region: "Asia Pacific",
    municipality: null,
    subnationalUnit: "Herat",
    coordinateAccuracy: "approximate",
    production2024Kt: 1408,
    production2023Kt: null,
    production2022Kt: null,
    designCapacityKt: 2500,
    totalReservesKt: null,
    totalResourceKt: 90000,
    startDate: null,
    stopDate: null,
    owner: null,
    parent: null,
    wiki: "https://www.gem.wiki/Ghoryan_Mine",
    lat: 34.01209,
    lon: 61.564895,
  });
});

test("normalizeIronOreMines: GEM's own 'unknown'/'N/A'/'--' sentinel cells stay null, never fabricated — including the 'unkonwn' typo GEM's real release ships", () => {
  const out = normalizeIronOreMines([FULL_ROW]);
  assert.equal(out[0].municipality, null); // "unknown"
  assert.equal(out[0].production2023Kt, null); // "N/A"
  assert.equal(out[0].production2022Kt, null); // "unkonwn" (source typo)
  assert.equal(out[0].totalReservesKt, null); // "unknown"
  assert.equal(out[0].owner, null); // "unknown"
  assert.equal(out[0].parent, null); // "--"
});

test("normalizeIronOreMines: a real numeric production/reserve value survives as a number, distinct from a sentinel", () => {
  const out = normalizeIronOreMines([FULL_ROW]);
  assert.equal(out[0].production2024Kt, 1408);
  assert.equal(out[0].designCapacityKt, 2500);
  assert.equal(out[0].totalResourceKt, 90000);
});

test("normalizeIronOreMines: drops rows with no usable coordinates (nothing to place on a map)", () => {
  assert.equal(normalizeIronOreMines([{ ...FULL_ROW, Coordinates: "not-coordinates" }]).length, 0);
  assert.equal(normalizeIronOreMines([{ ...FULL_ROW, Coordinates: undefined }]).length, 0);
});

test("normalizeIronOreMines: drops rows with no name or no id", () => {
  assert.equal(normalizeIronOreMines([{ ...FULL_ROW, "Asset name (English)": undefined }]).length, 0);
  assert.equal(normalizeIronOreMines([{ ...FULL_ROW, "GEM Asset ID": undefined }]).length, 0);
});

test("normalizeIronOreMines: an unrecognized Operating status string degrades to 'unknown', keeping statusRaw for the honest original", () => {
  const out = normalizeIronOreMines([{ ...FULL_ROW, "Operating status": "care-and-maintenance" }]);
  assert.equal(out[0].status, "unknown");
  assert.equal(out[0].statusRaw, "care-and-maintenance");
});

test("loadGemIronOreMines: reads a fixture file end-to-end, provenance intact", () => {
  const fp = mkFixture({
    provenance: {
      attribution: "Global Energy Monitor",
      license: "CC BY 4.0 (per-release Copyright sheets)",
      release: "Global-Iron-Ore-Mines-Tracker-August-2025-V1.xlsx",
    },
    mines: [FULL_ROW],
    counts: { mines: 1 },
  });
  const hit = loadGemIronOreMines(fp);
  assert.ok(hit);
  assert.equal(hit!.release, "Global-Iron-Ore-Mines-Tracker-August-2025-V1.xlsx");
  assert.equal(hit!.attribution, "Global Energy Monitor");
  assert.equal(hit!.mines.length, 1);
});

test("loadGemIronOreMines: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemIronOreMines("/nonexistent/path/iron_ore_mines.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemironore-bad-"));
  const fp = path.join(dir, "iron_ore_mines.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemIronOreMines(fp), null);
});

test("cachedGemIronOreMines: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemIronOreMinesCacheForTests();
  try {
    // Exercised against the actual checked-in
    // datacore/gem/iron_ore_mines.json — a realistic integration check
    // as well as a cache-identity check.
    const first = cachedGemIronOreMines();
    const second = cachedGemIronOreMines();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemIronOreMinesCacheForTests();
  }
});
