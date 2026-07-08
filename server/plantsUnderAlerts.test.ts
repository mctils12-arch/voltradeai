// plantsUnderAlerts cross-tie — pure point-in-polygon join. Tests: ray-cast
// containment, bbox pre-filter correctness, per-fuel aggregation, most-exposed
// ordering, zone-only (no-polygon) skip, and never-fabricated capacity.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  pointInRing, pointInAnyRing, ringsBbox, plantsUnderAlerts,
  type AlertLike, type Ring,
} from "./plantsUnderAlerts.ts";
import type { PlantTuple } from "./riverPlants.ts";

// A unit square [0,0]-[10,10] in [lon,lat].
const SQUARE: Ring = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];

test("pointInRing: inside vs outside a simple square", () => {
  assert.equal(pointInRing(5, 5, SQUARE), true);
  assert.equal(pointInRing(15, 5, SQUARE), false);
  assert.equal(pointInRing(-1, -1, SQUARE), false);
});

test("pointInAnyRing: true if inside any one of several rings", () => {
  const far: Ring = [[100, 100], [110, 100], [110, 110], [100, 110], [100, 100]];
  assert.equal(pointInAnyRing(5, 5, [far, SQUARE]), true);
  assert.equal(pointInAnyRing(50, 50, [far, SQUARE]), false);
});

test("ringsBbox: tight axis-aligned bounds", () => {
  assert.deepEqual(ringsBbox([SQUARE]), [0, 0, 10, 10]);
  assert.equal(ringsBbox([]), null);
});

test("plantsUnderAlerts: keeps plants inside the polygon, drops those outside", () => {
  const alerts: AlertLike[] = [
    { id: "a1", event: "Severe Thunderstorm Warning", severity: "Severe", area: "X", ends: null, rings: [SQUARE] },
  ];
  const plants: PlantTuple[] = [
    ["In Plant", 500, "gas", "Op", 5, 5, 1],       // (lat 5, lon 5) inside
    ["Out Plant", 900, "coal", "Op", 5, 50, 1],    // lon 50 outside
  ];
  const hits = plantsUnderAlerts(alerts, plants);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].plant_count, 1);
  assert.equal(hits[0].plants[0].name, "In Plant");
  assert.equal(hits[0].total_capacity_mw, 500);
  assert.deepEqual(hits[0].capacity_by_fuel, { gas: 500 });
});

test("plantsUnderAlerts: most-exposed-capacity alert first, biggest plant first, fuel breakdown", () => {
  const other: Ring = [[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]];
  const alerts: AlertLike[] = [
    { id: "small", event: "Flood Warning", severity: "Moderate", area: "A", ends: null, rings: [SQUARE] },
    { id: "big", event: "Hurricane Warning", severity: "Extreme", area: "B", ends: null, rings: [other] },
  ];
  const plants: PlantTuple[] = [
    ["P1", 300, "gas", "Op", 5, 5, 1],       // in SQUARE
    ["P2", 2000, "nuclear", "Op", 25, 25, 1], // in other
    ["P3", 800, "coal", "Op", 25, 24, 1],     // in other
  ];
  const hits = plantsUnderAlerts(alerts, plants);
  // "big" alert exposes 2800 MW vs "small" 300 -> big first
  assert.equal(hits[0].id, "big");
  assert.equal(hits[0].total_capacity_mw, 2800);
  // biggest plant within the alert first
  assert.equal(hits[0].plants[0].name, "P2");
  assert.deepEqual(hits[0].capacity_by_fuel, { nuclear: 2000, coal: 800 });
  assert.equal(hits[1].id, "small");
});

test("plantsUnderAlerts: a zone-only alert (no polygon rings) is skipped", () => {
  const alerts: AlertLike[] = [
    { id: "zone", event: "Heat Advisory", severity: "Minor", area: "Z", ends: null, rings: [] },
  ];
  const plants: PlantTuple[] = [["P", 100, "gas", "Op", 5, 5, 1]];
  assert.deepEqual(plantsUnderAlerts(alerts, plants), []);
});

test("plantsUnderAlerts: non-numeric capacity counts as 0, never fabricated", () => {
  const alerts: AlertLike[] = [
    { id: "a", event: "Tornado Warning", severity: "Extreme", area: "T", ends: null, rings: [SQUARE] },
  ];
  const plants: PlantTuple[] = [["NoMW", NaN as any, "hydro", "Op", 5, 5, 1]];
  const hits = plantsUnderAlerts(alerts, plants);
  assert.equal(hits[0].plants[0].capacity_mw, 0);
  assert.equal(hits[0].total_capacity_mw, 0);
});
