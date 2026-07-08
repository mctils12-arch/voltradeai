// riverPlants cross-tie — pure proximity join, same test discipline as
// firesFacilities.ts: known distances, radius cutoff, dedupe, capacity
// aggregation, and never-fabricated values.
import { test } from "node:test";
import assert from "node:assert/strict";
import { gaugePoints, plantsNearGauges, type PlantTuple } from "./riverPlants.ts";

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
