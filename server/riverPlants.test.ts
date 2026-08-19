// riverPlants cross-tie — pure proximity join, same test discipline as
// firesFacilities.ts: known distances, radius cutoff, dedupe, capacity
// aggregation, and never-fabricated values.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gaugePoints, plantsNearGauges, powerplantTable, type PlantTuple } from "./riverPlants.ts";

const here = path.dirname(fileURLToPath(import.meta.url));

// St. Louis gauge (real barge-corridor site).
const STL = { site: "07010000", name: "Mississippi R at St. Louis, MO", lat: 38.629, lon: -90.180 };

test("gaugePoints: dedupes a site's stage+discharge series to one point, drops bad coords", () => {
  const pts = gaugePoints([
    { site: "07010000", name: "STL", lat: 38.629, lon: -90.180 },   // stage
    { site: "07010000", name: "STL", lat: 38.629, lon: -90.180 },   // discharge, same site
    { site: "05586100", name: "Illinois R", lat: null, lon: -90.6 }, // missing lat -> dropped
    { site: "06934500", name: "Missouri R", lat: 38.71, lon: -91.44 },
  ]);
  assert.equal(pts.length, 2);
  assert.deepEqual(pts.map((p) => p.site).sort(), ["06934500", "07010000"]);
});

test("plantsNearGauges: keeps plants inside the radius, drops those outside", () => {
  const plants: PlantTuple[] = [
    ["Near Plant", 500, "coal", "Op A", 38.65, -90.20, 1],   // ~2-3 km from STL
    ["Far Plant", 900, "gas", "Op B", 41.0, -95.0, 1],       // hundreds of km away
  ];
  const hits = plantsNearGauges([STL], plants, 25);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].plant_count, 1);
  assert.equal(hits[0].plants[0].name, "Near Plant");
  assert.equal(hits[0].total_capacity_mw, 500);
  assert.deepEqual(hits[0].capacity_by_fuel, { coal: 500 });
});

test("plantsNearGauges: nearest-first plants, most-exposed-capacity-first gauges, fuel breakdown", () => {
  const gauges = [STL, { site: "06934500", name: "Missouri R at Hermann", lat: 38.71, lon: -91.44 }];
  const plants: PlantTuple[] = [
    ["Coal A", 800, "coal", "Op", 38.66, -90.19, 1],   // near STL, ~4 km
    ["Gas B", 300, "gas", "Op", 38.63, -90.17, 1],     // nearer STL, ~1 km
    ["Nuke C", 1200, "nuclear", "Op", 38.72, -91.45, 1], // near Hermann
  ];
  const hits = plantsNearGauges(gauges, plants, 30);
  // STL has 800+300=1100 MW, Hermann has 1200 MW -> Hermann first (more exposed)
  assert.equal(hits[0].site, "06934500");
  assert.equal(hits[1].site, "07010000");
  // STL plants nearest-first: Gas B (~1km) before Coal A (~4km)
  assert.equal(hits[1].plants[0].name, "Gas B");
  assert.equal(hits[1].plants[1].name, "Coal A");
  assert.deepEqual(hits[1].capacity_by_fuel, { gas: 300, coal: 800 });
  assert.equal(hits[1].total_capacity_mw, 1100);
});

test("plantsNearGauges: a non-numeric capacity counts as 0, never a fabricated guess", () => {
  const plants: PlantTuple[] = [
    ["No-MW Plant", NaN as any, "hydro", "Op", 38.64, -90.18, 1],
  ];
  const hits = plantsNearGauges([STL], plants, 25);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].plants[0].capacity_mw, 0);
  assert.equal(hits[0].total_capacity_mw, 0);
});

test("plantsNearGauges: a gauge with no nearby plant is omitted entirely", () => {
  const plants: PlantTuple[] = [["Far", 100, "gas", "Op", 10.0, 10.0, 1]];
  assert.deepEqual(plantsNearGauges([STL], plants, 25), []);
});

// REGRESSION (2026-08-19): /api/data/plants-near-rivergauges and
// /api/data/plants-under-alerts both 500'd in production ("t is not iterable")
// for their whole lifetime — the routes cast the JSON MODULE straight to
// PlantTuple[], but it is a {_doc, source, fuels, count, plants} wrapper. The
// pure joins above were fully tested with synthetic arrays, so nothing caught
// the extraction step. These tests pin the real shipped shape.
test("powerplantTable: extracts the tuple rows from the SHIPPED dataset shape", () => {
  const raw = JSON.parse(fs.readFileSync(
    path.join(here, "..", "datacore", "powerplants", "us_power_plants.json"), "utf8"));
  const plants = powerplantTable(raw);
  assert.ok(Array.isArray(plants), "the routes iterate this — it must be an array");
  assert.equal(plants.length, raw.count, "row count must match the dataset's own count");
  assert.ok(plants.length > 9000, `expected ~9.8k US plants, got ${plants.length}`);
  const [name, mw, fuel, owner, lat, lon] = plants[0];
  assert.equal(typeof name, "string");
  assert.equal(typeof mw, "number");
  assert.equal(typeof fuel, "string");
  assert.equal(typeof owner, "string");
  assert.equal(typeof lat, "number");
  assert.equal(typeof lon, "number");
  // the cross-tie joins must actually consume it without throwing
  assert.doesNotThrow(() => plantsNearGauges([STL], plants, 25));
  assert.ok(plantsNearGauges([STL], plants, 25).length > 0, "St. Louis has plants within 25km");
});

test("powerplantTable: a bare array passes through; an unusable shape yields []", () => {
  const rows: PlantTuple[] = [["P", 100, "gas", "Op", 38.64, -90.18, 1]];
  assert.deepEqual(powerplantTable(rows), rows);
  assert.deepEqual(powerplantTable({ plants: rows }), rows);
  assert.deepEqual(powerplantTable({ count: 3 }), []);
  assert.deepEqual(powerplantTable(null), []);
});

test("neither cross-tie route casts the powerplant module straight to an array", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(!/datacorePowerplants as unknown as PlantTuple\[\]/.test(routes),
    "the JSON module is a {plants:[...]} wrapper — cast it and the route 500s at runtime; use powerplantTable()");
  assert.equal((routes.match(/powerplantTable\(datacorePowerplants\)/g) || []).length, 2,
    "both plants-* cross-tie routes must read the table through powerplantTable()");
});
