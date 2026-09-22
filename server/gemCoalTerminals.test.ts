// gemCoalTerminals — GEM "Global Coal Terminals Tracker" RAW overlay
// (CLAUDE.md RAW-vs-SIGNAL surface rule). Pins the Terminal Type ->
// 5-bucket classifier (live-verified against the real release's 13
// distinct free-text values), the drop-not-infer rule (no lat/lon, no
// name, no unique id), the "GEM Terminal ID" non-uniqueness finding (53
// live collisions — "GEM Unit/Phase ID" is this module's id instead),
// and the missing/corrupt-file degrade path.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  classifyTerminalType,
  normalizeCoalTerminals,
  loadGemCoalTerminals,
  cachedGemCoalTerminals,
  _resetGemCoalTerminalsCacheForTests,
} from "./gemCoalTerminals";

function mkFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemcoalterm-"));
  const fp = path.join(dir, "coal_terminals.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

const FULL_ROW = {
  "Coal Terminal Name": "Balaclava Island Coal Terminal",
  "Parent Port Name": "-",
  "Wiki URL": "https://www.gem.wiki/Balaclava_Island_Coal_Terminal",
  Status: "Cancelled",
  Owner: "Glencore",
  "Capacity (Mt)": 35,
  "Product Type": "Coal",
  "Terminal Type": "Exports",
  "Start Year": "-",
  "Retired Year": "-",
  Location: "Gladstone",
  "State/Province": "Queensland",
  "Country/Area": "Australia",
  Subregion: "Australia and New Zealand",
  Region: "Oceania",
  Latitude: -23.560116,
  Longitude: 150.909863,
  "Location Accuracy": "Approximate",
  "GEM Terminal ID": "T1006",
  "GEM Unit/Phase ID": "T01006",
};

test("classifyTerminalType: single roles map directly", () => {
  assert.equal(classifyTerminalType("Exports"), "exports");
  assert.equal(classifyTerminalType("Imports"), "imports");
  assert.equal(classifyTerminalType("Domestic"), "domestic");
});

test("classifyTerminalType: comma-joined multi-role strings are 'mixed', never a guessed single role", () => {
  assert.equal(classifyTerminalType("Exports, Imports"), "mixed");
  assert.equal(classifyTerminalType("Domestic, Imports"), "mixed");
  assert.equal(classifyTerminalType("Domestic, Exports, Imports"), "mixed");
  assert.equal(classifyTerminalType("Imports, Domestic"), "mixed");
});

test("classifyTerminalType: blank/dash/unrecognized text is 'unstated', never guessed", () => {
  assert.equal(classifyTerminalType("-"), "unstated");
  assert.equal(classifyTerminalType(""), "unstated");
  assert.equal(classifyTerminalType(null), "unstated");
  assert.equal(classifyTerminalType(undefined), "unstated");
});

test("normalizeCoalTerminals: maps GEM's raw columns to the clean schema", () => {
  const out = normalizeCoalTerminals([FULL_ROW]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "T01006",
    terminalId: "T1006",
    name: "Balaclava Island Coal Terminal",
    parentPort: null,
    status: "Cancelled",
    typeClass: "exports",
    typeRaw: "Exports",
    productType: "Coal",
    capacityMt: 35,
    owner: "Glencore",
    country: "Australia",
    region: "Oceania",
    startYear: null,
    retiredYear: null,
    locationAccuracy: "Approximate",
    wiki: "https://www.gem.wiki/Balaclava_Island_Coal_Terminal",
    lat: -23.560116,
    lon: 150.909863,
  });
});

test("normalizeCoalTerminals: GEM's own dash cells stay null, never fabricated", () => {
  const out = normalizeCoalTerminals([{ ...FULL_ROW, Owner: "-", "Start Year": "-" }]);
  assert.equal(out[0].owner, null);
  assert.equal(out[0].startYear, null);
});

test("normalizeCoalTerminals: a real numeric Start Year survives as a string", () => {
  const out = normalizeCoalTerminals([{ ...FULL_ROW, "Start Year": 1983 }]);
  assert.equal(out[0].startYear, "1983");
});

test("normalizeCoalTerminals: drops rows with no lat/lon (nothing to place on a map)", () => {
  assert.equal(normalizeCoalTerminals([{ ...FULL_ROW, Latitude: "-", Longitude: "-" }]).length, 0);
  assert.equal(normalizeCoalTerminals([{ ...FULL_ROW, Latitude: undefined }]).length, 0);
});

test("normalizeCoalTerminals: drops rows with no name or no unique id", () => {
  assert.equal(normalizeCoalTerminals([{ ...FULL_ROW, "Coal Terminal Name": undefined }]).length, 0);
  assert.equal(normalizeCoalTerminals([{ ...FULL_ROW, "GEM Unit/Phase ID": undefined }]).length, 0);
});

test("normalizeCoalTerminals: 'GEM Terminal ID' repeats across multiple berths at one terminal — this module's id is the Unit/Phase ID instead", () => {
  const rowA = { ...FULL_ROW, "GEM Terminal ID": "T1073", "GEM Unit/Phase ID": "T01073", "Coal Terminal Name": "Dalrymple Bay Coal Terminal" };
  const rowB = { ...FULL_ROW, "GEM Terminal ID": "T1073", "GEM Unit/Phase ID": "T01315", "Coal Terminal Name": "Dalrymple Bay Coal Terminal" };
  const out = normalizeCoalTerminals([rowA, rowB]);
  assert.equal(out.length, 2, "both berths survive — GEM Terminal ID colliding must not merge/drop either row");
  assert.equal(out[0].terminalId, "T1073");
  assert.equal(out[1].terminalId, "T1073");
  assert.notEqual(out[0].id, out[1].id);
});

test("loadGemCoalTerminals: reads a fixture file end-to-end, provenance intact", () => {
  const fp = mkFixture({
    provenance: {
      attribution: "Global Energy Monitor",
      license: "CC BY 4.0 (per-release Copyright sheets)",
      release: "Global-Coal-Terminals-Tracker-December-2024.xlsx",
    },
    terminals: [FULL_ROW],
    counts: { terminals: 1 },
  });
  const hit = loadGemCoalTerminals(fp);
  assert.ok(hit);
  assert.equal(hit!.release, "Global-Coal-Terminals-Tracker-December-2024.xlsx");
  assert.equal(hit!.attribution, "Global Energy Monitor");
  assert.equal(hit!.terminals.length, 1);
});

test("loadGemCoalTerminals: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemCoalTerminals("/nonexistent/path/coal_terminals.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemcoalterm-bad-"));
  const fp = path.join(dir, "coal_terminals.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemCoalTerminals(fp), null);
});

test("cachedGemCoalTerminals: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemCoalTerminalsCacheForTests();
  try {
    // Exercised against the actual checked-in
    // datacore/gem/coal_terminals.json — a realistic integration check
    // as well as a cache-identity check.
    const first = cachedGemCoalTerminals();
    const second = cachedGemCoalTerminals();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
  } finally {
    _resetGemCoalTerminalsCacheForTests();
  }
});
