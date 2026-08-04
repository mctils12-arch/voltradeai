// NRC daily Power Reactor Status Reports battery: pipe-delimited parse,
// unit->registry-plant name reconciliation (gate 1), event-identity
// archive dedup, latest-day-only cache.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseReactorStatus, reactorStatusUrl, latestDay, normalizePlantName,
  matchToRegistry, EXPECTED_REGISTRY_ONLY, fetchReactorStatus,
  archiveReactorStatus, refreshReactorStatus, latestReactorStatus,
  loadRegistryNuclearPlants, joinToPlants,
} from "./nrcReactorStatus";

const SAMPLE = [
  "ReportDt|Unit|Power",
  "8/3/2026 12:00:00 AM|Arkansas Nuclear 1|100",
  "8/3/2026 12:00:00 AM|Beaver Valley 1|97",
  "8/3/2026 12:00:00 AM|Palo Verde 2|0",
  "8/2/2026 12:00:00 AM|Arkansas Nuclear 1|100",
].join("\n");

test("url: one file per calendar year", () => {
  assert.equal(reactorStatusUrl(2026), "https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/2026/2026PowerStatus.txt");
});

test("parse: pipe-delimited, header dropped, dates normalized to YYYY-MM-DD, malformed lines skipped", () => {
  const rows = parseReactorStatus(SAMPLE);
  assert.equal(rows.length, 4);
  assert.deepEqual(rows[0], { date: "2026-08-03", unit: "Arkansas Nuclear 1", power: 100 });
  assert.equal(rows[2].power, 0, "0% (shutdown) is a real value, not treated as missing");
  assert.deepEqual(parseReactorStatus("garbage\n\nA|B"), []);
  assert.deepEqual(parseReactorStatus("1/1/2026|Unit X|"), [{ date: "2026-01-01", unit: "Unit X", power: null }]);
});

test("latestDay: max date string across rows", () => {
  assert.equal(latestDay(parseReactorStatus(SAMPLE)), "2026-08-03");
  assert.equal(latestDay([]), "");
});

test("normalizePlantName: strips punctuation, hyphens, and one suffix phrase", () => {
  assert.equal(normalizePlantName("Davis-Besse"), "davis besse");
  assert.equal(normalizePlantName("D.C. Cook"), "dc cook");
  assert.equal(normalizePlantName("Clinton Power Station"), "clinton");
  assert.equal(normalizePlantName("Wolf Creek Generating Station"), "wolf creek");
  assert.equal(normalizePlantName("Waterford 3"), "waterford 3", "no suffix word to strip; unit-strip is a separate step");
});

test("matchToRegistry: live-shaped panel matches 100%, with documented (not silent) retired-plant gaps", () => {
  const nrcUnits = [
    "Beaver Valley 1", "Beaver Valley 2", "Arkansas Nuclear 1", "D.C. Cook 1",
    "Davis-Besse", "Clinton", "River Bend Station 1", "Hope Creek 1", "Salem 1",
    "Susquehanna 1", "Watts Bar 1",
  ];
  const registry = [
    "Beaver Valley", "Arkansas Nuclear One", "Donald C Cook", "Davis Besse",
    "Clinton Power Station", "River Bend", "PSEG Hope Creek Generating Station",
    "PSEG Salem Generating Station", "TalenEnergy Susquehanna",
    "Watts Bar Nuclear Plant", "Indian Point 2",
  ];
  const r = matchToRegistry(nrcUnits, registry);
  assert.equal(r.matched, nrcUnits.length);
  assert.deepEqual(r.unmatched, []);
  assert.deepEqual(r.unexpectedRegistryGaps, [], "no silently-dropped registry entry");
  assert.ok(r.expectedGaps.includes("indian point 2"), "documented retired-plant gap surfaces, not silently dropped");
  assert.equal(r.matchRate, 1);
});

test("matchToRegistry: a genuinely unexplained registry gap is reported, never silently swallowed", () => {
  const r = matchToRegistry(["Beaver Valley 1"], ["Beaver Valley", "Some New Plant"]);
  assert.deepEqual(r.unexpectedRegistryGaps, ["Some New Plant"]);
  assert.deepEqual(r.expectedGaps, []);
});

test("matchToRegistry: an NRC unit with no registry counterpart is reported unmatched, not silently dropped", () => {
  const r = matchToRegistry(["Totally Unknown Unit 1"], ["Beaver Valley"]);
  assert.deepEqual(r.unmatched, ["Totally Unknown Unit 1"]);
  assert.equal(r.matched, 0);
});

test("EXPECTED_REGISTRY_ONLY carries only documented-retired plants", () => {
  assert.ok(EXPECTED_REGISTRY_ONLY.has("indian point 2"));
  assert.ok(EXPECTED_REGISTRY_ONLY.has("duane arnold"));
});

test("loadRegistryNuclearPlants: filters to fuel==='nuclear', keeps the row's original index", () => {
  const rows: [string, number, string, string, number, number, number][] = [
    ["Grand Coulee", 6809, "hydro", "USBR", 47.9575, -118.9773, 1],
    ["Beaver Valley", 1836.6, "nuclear", "Energy Harbor", 40.6234, -80.434, 1],
    ["Palo Verde", 4209.6, "nuclear", "APS", 33.3881, -112.8617, 1],
  ];
  const plants = loadRegistryNuclearPlants(rows);
  assert.equal(plants.length, 2);
  assert.deepEqual(plants[0], { idx: 1, name: "Beaver Valley", mw: 1836.6, owner: "Energy Harbor", lat: 40.6234, lon: -80.434 });
  assert.equal(plants[1].idx, 2, "index is the position in the UNFILTERED array (matches entityGraph.ts's plantFacilityId)");
});

test("joinToPlants: groups units under their plant, buckets status off the mean reported power", () => {
  const registry = loadRegistryNuclearPlants([
    ["Beaver Valley", 1836.6, "nuclear", "Energy Harbor", 40.6234, -80.434, 1],
    ["Palo Verde", 4209.6, "nuclear", "APS", 33.3881, -112.8617, 1],
  ]);
  const rows = [
    { date: "2026-08-04", unit: "Beaver Valley 1", power: 100 },
    { date: "2026-08-04", unit: "Beaver Valley 2", power: 90 }, // avg 95 -> full (>= threshold)
    { date: "2026-08-04", unit: "Palo Verde 1", power: 0 },
    { date: "2026-08-04", unit: "Palo Verde 2", power: 3 },      // avg 1.5 -> outage
  ];
  const plants = joinToPlants(rows, registry);
  assert.equal(plants.length, 2);
  const bv = plants.find((p) => p.name === "Beaver Valley")!;
  assert.equal(bv.units.length, 2);
  assert.equal(bv.avgPower, 95);
  assert.equal(bv.status, "full");
  assert.equal(bv.idx, 0);
  const pv = plants.find((p) => p.name === "Palo Verde")!;
  assert.equal(pv.avgPower, 1.5);
  assert.equal(pv.status, "outage");
});

test("joinToPlants: a single-unit outage at an otherwise-full multi-unit plant reads as 'reduced', not 'outage' — the down unit stays visible in units[]", () => {
  const registry = loadRegistryNuclearPlants([["Beaver Valley", 1836.6, "nuclear", "Energy Harbor", 40.6234, -80.434, 1]]);
  const rows = [
    { date: "2026-08-04", unit: "Beaver Valley 1", power: 100 },
    { date: "2026-08-04", unit: "Beaver Valley 2", power: 0 },
  ];
  const plants = joinToPlants(rows, registry);
  assert.equal(plants[0].status, "reduced");
  assert.deepEqual(plants[0].units, [{ unit: "Beaver Valley 1", power: 100 }, { unit: "Beaver Valley 2", power: 0 }]);
});

test("joinToPlants: null-power units are excluded from the average, not treated as 0; all-null resolves to 'unknown'", () => {
  const registry = loadRegistryNuclearPlants([["Beaver Valley", 1836.6, "nuclear", "Energy Harbor", 40.6234, -80.434, 1]]);
  assert.equal(joinToPlants([{ date: "2026-08-04", unit: "Beaver Valley 1", power: 100 }, { date: "2026-08-04", unit: "Beaver Valley 2", power: null }], registry)[0].avgPower, 100);
  assert.equal(joinToPlants([{ date: "2026-08-04", unit: "Beaver Valley 1", power: null }], registry)[0].status, "unknown");
});

test("joinToPlants: an unresolved unit is dropped, not guessed at a nearby plant", () => {
  const registry = loadRegistryNuclearPlants([["Beaver Valley", 1836.6, "nuclear", "Energy Harbor", 40.6234, -80.434, 1]]);
  const plants = joinToPlants([{ date: "2026-08-04", unit: "Totally Unknown Unit 1", power: 100 }], registry);
  assert.deepEqual(plants, []);
});

test("fetch: keeps only the newest day's rows, on non-ok logs and returns empty", async () => {
  const ok = async () => ({ ok: true, status: 200, text: async () => SAMPLE });
  const rows = await fetchReactorStatus(ok as any, Date.parse("2026-08-03T12:00:00Z"));
  assert.equal(rows.length, 3, "only 2026-08-03 rows, the 2026-08-02 Arkansas row excluded");
  assert.ok(rows.every((r) => r.date === "2026-08-03"));

  const bad = async () => ({ ok: false, status: 500, text: async () => "err" });
  assert.deepEqual(await fetchReactorStatus(bad as any), []);

  const throws = async () => { throw new Error("network down"); };
  assert.deepEqual(await fetchReactorStatus(throws as any), []);
});

test("archive: date|unit dedup across fetches and restarts", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nrc-"));
  const rows = parseReactorStatus(SAMPLE).filter((r) => r.date === "2026-08-03");
  const now = Date.parse("2026-08-03T12:00:00Z");
  assert.equal(archiveReactorStatus(rows, base, now), 3);
  assert.equal(archiveReactorStatus(rows, base, now), 0, "same date/unit never re-archives");
  const plus = [...rows, { date: "2026-08-04", unit: "Arkansas Nuclear 1", power: 100 }];
  assert.equal(archiveReactorStatus(plus, base, now), 1, "new day's row lands");
});

test("refresh: cache holds only the newest day's rows, plus the registry-joined plants view", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nrc-"));
  const ok = async () => ({ ok: true, status: 200, text: async () => SAMPLE });
  await refreshReactorStatus(ok as any, Date.parse("2026-08-03T12:00:00Z"), base);
  const hit = latestReactorStatus();
  assert.ok(hit);
  assert.equal(hit!.date, "2026-08-03");
  assert.ok(hit!.rows.every((r) => r.date === "2026-08-03"));
  // real registry join against the actual us_power_plants.json, not a
  // fixture — SAMPLE's units (Arkansas Nuclear 1, Beaver Valley 1, Palo
  // Verde 2) all resolve to real registry plants, so this proves the
  // wiring end-to-end, not just that joinToPlants works in isolation.
  assert.ok(Array.isArray(hit!.plants));
  assert.ok(hit!.plants.length >= 2, "Arkansas Nuclear + Beaver Valley + Palo Verde all resolve against the real registry");
  const arkansas = hit!.plants.find((p) => p.name.includes("Arkansas"));
  assert.ok(arkansas, "Arkansas Nuclear 1 (100%) resolves to a registry plant");
  assert.equal(arkansas!.status, "full");
});
