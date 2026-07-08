// fires × facilities cross-tie (Pillar 6 inference) — the join math.
import { test } from "node:test";
import assert from "node:assert/strict";
import { haversineKm, firesNearFacilities, type Facility, type FireLike } from "./firesFacilities.ts";

test("haversineKm: known distances", () => {
  assert.equal(Math.round(haversineKm(0, 0, 0, 0)), 0);
  // ~1 degree of latitude ≈ 111 km
  assert.ok(Math.abs(haversineKm(0, 0, 1, 0) - 111.2) < 1);
  // NYC (40.71,-74.01) → LA (34.05,-118.24) ≈ 3936 km
  assert.ok(Math.abs(haversineKm(40.71, -74.01, 34.05, -118.24) - 3936) < 30);
});

const SITE: Facility = { id: "cushing", name: "Cushing", lat: 35.95, lon: -96.76, category: "tank_farm" };

test("firesNearFacilities: counts within radius, nearest, max frp; excludes far facilities", () => {
  const fires: FireLike[] = [
    { lat: 35.96, lon: -96.75, frp: 12 },   // ~1.4 km — inside
    { lat: 35.99, lon: -96.80, frp: 40 },   // ~5 km — inside, strongest
    { lat: 36.30, lon: -96.76, frp: 5 },    // ~39 km — outside 25km
    { lat: 10.0, lon: 10.0, frp: 99 },      // far — outside
  ];
  const hits = firesNearFacilities([SITE], fires, 25);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].fires_within, 2);
  assert.equal(hits[0].max_frp, 40);
  assert.ok(hits[0].nearest_km > 0 && hits[0].nearest_km < 3);
});

test("firesNearFacilities: facility with no nearby fire is omitted", () => {
  const far: Facility = { id: "x", name: "X", lat: -40, lon: 170 };
  const fires: FireLike[] = [{ lat: 35.96, lon: -96.75, frp: 12 }];
  assert.deepEqual(firesNearFacilities([far], fires, 25), []);
});

test("firesNearFacilities: sorted nearest-first; bad coords skipped, never crash", () => {
  const s2: Facility = { id: "b", name: "B", lat: 36.5, lon: -96.76 };
  const fires: FireLike[] = [
    { lat: 35.951, lon: -96.761, frp: null }, // ~0.15km from cushing
    { lat: 36.51, lon: -96.76, frp: 3 },      // ~1.1km from B
    { lat: NaN, lon: NaN, frp: 1 },           // junk — skipped
  ];
  const hits = firesNearFacilities([s2, SITE], fires, 25);
  assert.equal(hits[0].id, "cushing", "nearest-first ordering");
  assert.equal(hits[0].max_frp, null, "null frp stays null (never fabricated)");
});
