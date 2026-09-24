// gemLngCarriers — GEM "Global LNG Carrier Tracker" RAW overlay (CLAUDE.md
// RAW-vs-SIGNAL surface rule). Pins the shipyard-aggregation honesty
// framing (research/open_questions.md's 2026-09-22 GEM-suite backlog
// entry: "needs the shipyard honesty framing before shipping" — the only
// coordinate this release carries is a BUILD location, and live-verified
// only 32 distinct ones exist across 1,143 carriers, one shipbuilder each),
// the "Status" -> lifecycle classifier, the drop-not-infer rule (no yard
// coordinates, no shipbuilder name), capacity summation, and the
// missing/corrupt-file degrade path.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  classifyCarrierStatus,
  slugifyShipyardId,
  normalizeLngShipyards,
  loadGemLngShipyards,
  cachedGemLngShipyards,
  _resetGemLngShipyardsCacheForTests,
} from "./gemLngCarriers";

function mkFixture(body: unknown): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemlngcarriers-"));
  const fp = path.join(dir, "lng_carriers.json");
  fs.writeFileSync(fp, JSON.stringify(body));
  return fp;
}

function carrier(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    "IMO number": 9443401,
    Name: "Aamira",
    Status: "active",
    Shipowner: "Nakilat",
    Shipbuilder: "Samsung Heavy Industries",
    "Shipbuilder yard country/area": "South Korea",
    "Yard location latitude": 34.89804,
    "Yard location longitude": 128.6045,
    "Yard location accuracy": "exact",
    Capacity: 266000,
    "Capacity units": "cbm",
    ...over,
  };
}

test("classifyCarrierStatus: the 3 catalogued statuses map exactly", () => {
  assert.equal(classifyCarrierStatus("active"), "active");
  assert.equal(classifyCarrierStatus("on order"), "on_order");
  assert.equal(classifyCarrierStatus("proposed"), "proposed");
});

test("classifyCarrierStatus: case-insensitive, and unrecognized/blank falls to 'other', never guessed", () => {
  assert.equal(classifyCarrierStatus("ACTIVE"), "active");
  assert.equal(classifyCarrierStatus("On Order"), "on_order");
  assert.equal(classifyCarrierStatus("retired"), "other");
  assert.equal(classifyCarrierStatus(""), "other");
  assert.equal(classifyCarrierStatus(null), "other");
  assert.equal(classifyCarrierStatus(undefined), "other");
});

test("slugifyShipyardId: lowercases, collapses non-alnum runs to one dash, trims edges", () => {
  assert.equal(slugifyShipyardId("Samsung Heavy Industries"), "samsung-heavy-industries");
  assert.equal(slugifyShipyardId("HD Hyundai Heavy Industries"), "hd-hyundai-heavy-industries");
  assert.equal(slugifyShipyardId("  Kawasaki  Heavy!!  Industries  "), "kawasaki-heavy-industries");
});

test("normalizeLngShipyards: aggregates multiple carriers at the same yard into one point", () => {
  const out = normalizeLngShipyards([
    carrier({ Name: "Aamira" }),
    carrier({ Name: "Adam LNG", Status: "on order", Capacity: 162000 }),
    carrier({ Name: "Unnamed", Status: "proposed", Capacity: undefined }),
  ]);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], {
    id: "samsung-heavy-industries",
    shipbuilder: "Samsung Heavy Industries",
    country: "South Korea",
    lat: 34.89804,
    lon: 128.6045,
    coordinateAccuracy: "exact",
    carrierCount: 3,
    activeCount: 1,
    onOrderCount: 1,
    proposedCount: 1,
    otherCount: 0,
    totalCapacityCbm: 266000 + 162000,
    knownCapacityCount: 2,
  });
});

test("normalizeLngShipyards: distinct yard coordinates produce distinct shipyards, sorted by carrier count descending", () => {
  const out = normalizeLngShipyards([
    carrier({ Shipbuilder: "Kawasaki Heavy Industries", "Shipbuilder yard country/area": "Japan",
              "Yard location latitude": 34.327891, "Yard location longitude": 133.832534 }),
    carrier({ Name: "B" }),
    carrier({ Name: "C" }),
  ]);
  assert.equal(out.length, 2);
  assert.equal(out[0].shipbuilder, "Samsung Heavy Industries");
  assert.equal(out[0].carrierCount, 2);
  assert.equal(out[1].shipbuilder, "Kawasaki Heavy Industries");
  assert.equal(out[1].carrierCount, 1);
});

test("normalizeLngShipyards: drops rows with no yard coordinates (proposed carriers with no yard assigned yet)", () => {
  const out = normalizeLngShipyards([
    carrier({ "Yard location latitude": undefined, "Yard location longitude": undefined, Status: "proposed" }),
  ]);
  assert.equal(out.length, 0);
});

test("normalizeLngShipyards: drops rows with no shipbuilder name", () => {
  const out = normalizeLngShipyards([carrier({ Shipbuilder: undefined })]);
  assert.equal(out.length, 0);
});

test("normalizeLngShipyards: a yard with no carriers carrying a known capacity reports null, not zero", () => {
  const out = normalizeLngShipyards([carrier({ Capacity: undefined })]);
  assert.equal(out[0].totalCapacityCbm, null);
  assert.equal(out[0].knownCapacityCount, 0);
});

test("loadGemLngShipyards: reads a fixture file end-to-end, provenance and totalCarriers intact", () => {
  const fp = mkFixture({
    provenance: {
      attribution: "Global Energy Monitor",
      license: "CC BY 4.0 (per-release Copyright sheets)",
      release: "LNG-Carrier-Tracker-December-2025-release.xlsx",
    },
    carriers: [carrier({ Name: "Aamira" }), carrier({ Name: "Adam LNG" })],
    counts: { carriers: 2 },
  });
  const hit = loadGemLngShipyards(fp);
  assert.ok(hit);
  assert.equal(hit!.release, "LNG-Carrier-Tracker-December-2025-release.xlsx");
  assert.equal(hit!.attribution, "Global Energy Monitor");
  assert.equal(hit!.shipyards.length, 1);
  assert.equal(hit!.totalCarriers, 2);
});

test("loadGemLngShipyards: a missing/corrupt file degrades to null, never throws", () => {
  assert.equal(loadGemLngShipyards("/nonexistent/path/lng_carriers.json"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gemlngcarriers-bad-"));
  const fp = path.join(dir, "lng_carriers.json");
  fs.writeFileSync(fp, "{not json");
  assert.equal(loadGemLngShipyards(fp), null);
});

test("cachedGemLngShipyards: caches across calls (same object reference, parsed once per process)", () => {
  _resetGemLngShipyardsCacheForTests();
  try {
    // Exercised against the actual checked-in datacore/gem/lng_carriers.json —
    // a realistic integration check as well as a cache-identity check.
    const first = cachedGemLngShipyards();
    const second = cachedGemLngShipyards();
    assert.ok(first, "expected the real repo fixture to load");
    assert.equal(first, second, "second call must return the cached object, not re-parse");
    // Live-verified count from this session's own data read: 32 distinct
    // shipyards across 1,125 LOCATED carriers (1,143 total in the release;
    // 18 have no yard assigned yet — all "proposed" — and are dropped by
    // normalizeLngShipyards's drop-not-infer rule, so totalCarriers counts
    // only the ones actually placed on the map). Pinned so a future GEM
    // release swap is a visible, deliberate re-pin, not a silent drift.
    assert.equal(first!.shipyards.length, 32);
    assert.equal(first!.totalCarriers, 1125);
  } finally {
    _resetGemLngShipyardsCacheForTests();
  }
});
