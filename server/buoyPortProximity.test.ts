// buoyPortProximity — synthetic buoy/port join tests, no fs/network.
import { test } from "node:test";
import assert from "node:assert/strict";
import type { PortDef } from "./portDwell";
import type { BuoyObs } from "./ndbcBuoys";
import {
  MAX_JOIN_KM, buoysNearPort, nearestBuoyPerPort, buoyPortProximityReport, hasWaveHeight,
} from "./buoyPortProximity";

const LA: PortDef = { id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, radius_km: 5 };
const HOUSTON: PortDef = { id: "port_houston", name: "Port of Houston (Barbours Cut)", lat: 29.677, lon: -95.006, radius_km: 5 };
const SEATTLE: PortDef = { id: "port_seattle", name: "Port of Seattle", lat: 47.582, lon: -122.3474, radius_km: 5 };

function buoy(station: string, lat: number | null, lon: number | null, waveHeight: number | null = 1.2): BuoyObs {
  return {
    station, lat, lon, time: Date.UTC(2026, 7, 1), windDir: null, windSpeed: null, gust: null,
    waveHeight, dominantPeriod: null, avgPeriod: null, waveDir: null, pressure: null,
    pressureTendency: null, airTemp: null, waterTemp: null, dewpoint: null, visibility: null,
    tide: null, rt: "2026-08-01",
  };
}

test("buoysNearPort: exact-coordinate buoy is 0km and included", () => {
  const near = buoysNearPort(LA, [buoy("46222", LA.lat, LA.lon)]);
  assert.equal(near.length, 1);
  assert.equal(near[0].station, "46222");
  assert.equal(near[0].distance_km, 0);
});

test("buoysNearPort: a buoy on the other coast is excluded", () => {
  // Real NDBC station 46026 sits ~20km off San Francisco — ~600km from LA, well outside MAX_JOIN_KM.
  const near = buoysNearPort(LA, [buoy("46026", 37.75, -122.82)]);
  assert.equal(near.length, 0);
});

test("buoysNearPort: sorted nearest-first when multiple stations qualify", () => {
  const far = buoy("F1", LA.lat + 1.5, LA.lon); // ~166km north
  const near = buoy("N1", LA.lat + 0.3, LA.lon); // ~33km north
  const result = buoysNearPort(LA, [far, near]);
  assert.deepEqual(result.map((m) => m.station), ["N1", "F1"]);
  assert.ok(result[0].distance_km < result[1].distance_km);
});

test("buoysNearPort: null lat/lon rows are skipped, never treated as distance 0", () => {
  const near = buoysNearPort(LA, [buoy("MALFORMED", null, null), buoy("MALFORMED2", LA.lat, null)]);
  assert.equal(near.length, 0);
});

test("nearestBuoyPerPort: honest null when no station is within range", () => {
  const map = nearestBuoyPerPort([SEATTLE], [buoy("46222", LA.lat, LA.lon)]);
  assert.equal(map.get("port_seattle"), null);
});

test("nearestBuoyPerPort: picks the closest of several candidates per port", () => {
  const closeToHouston = buoy("42035", 29.3, -94.4); // real NDBC Galveston-area buoy, ~85km from the port
  const farFromHouston = buoy("42019", 27.9, -95.35); // real NDBC buoy further offshore, ~200km
  const map = nearestBuoyPerPort([HOUSTON], [farFromHouston, closeToHouston]);
  assert.equal(map.get("port_houston")?.station, "42035");
});

test("buoyPortProximityReport: matched + unmatched partition covers every port exactly once", () => {
  const report = buoyPortProximityReport(
    [LA, HOUSTON, SEATTLE],
    [buoy("46222", LA.lat + 0.1, LA.lon), buoy("42035", 29.3, -94.4)],
    MAX_JOIN_KM,
    Date.UTC(2026, 7, 9),
  );
  assert.equal(report.ports_total, 3);
  assert.equal(report.ports_matched, 2);
  assert.equal(report.matches.length, 2);
  assert.deepEqual(report.unmatched_ports.map((p) => p.portId), ["port_seattle"]);
  assert.equal(report.generated_at, new Date(Date.UTC(2026, 7, 9)).toISOString());
  assert.equal(report.max_join_km, MAX_JOIN_KM);
});

test("buoyPortProximityReport: empty buoy list leaves every port honestly unmatched", () => {
  const report = buoyPortProximityReport([LA, HOUSTON], []);
  assert.equal(report.ports_matched, 0);
  assert.equal(report.matches.length, 0);
  assert.equal(report.unmatched_ports.length, 2);
});

test("MAX_JOIN_KM default is respected by a custom, tighter radius", () => {
  const near = buoy("N1", LA.lat + 0.3, LA.lon); // ~33km
  assert.equal(buoysNearPort(LA, [near], 10).length, 0);
  assert.equal(buoysNearPort(LA, [near], 50).length, 1);
});

// LIVE-PROBED HONESTY FINDING (2026-08-09): the nearest station to every one
// of the 9 real ports turned out to be a harbor/C-MAN gauge with
// waveHeight=null, not an open-ocean wave buoy — geographic nearness alone
// answers the wrong question for the sea-state hypothesis. These pin the
// `hasWaveHeight` filter that fixes that.
test("hasWaveHeight: true only for a non-null waveHeight reading", () => {
  assert.equal(hasWaveHeight(buoy("A", 0, 0, 1.2)), true);
  assert.equal(hasWaveHeight(buoy("A", 0, 0, 0)), true); // a real 0m reading is not "missing"
  assert.equal(hasWaveHeight(buoy("A", 0, 0, null)), false);
});

test("buoysNearPort + hasWaveHeight filter: a closer non-wave station is skipped in favor of a farther wave-reporting one", () => {
  const harborGauge = buoy("HARBOR", LA.lat + 0.01, LA.lon, null); // ~1km, no wave sensor
  const waveBuoy = buoy("46256", LA.lat + 0.07, LA.lon, 1.0); // ~8km, real sensor
  const near = buoysNearPort(LA, [harborGauge, waveBuoy], MAX_JOIN_KM, hasWaveHeight);
  assert.deepEqual(near.map((m) => m.station), ["46256"]);
});

test("nearestBuoyPerPort + hasWaveHeight filter: honest null when only non-wave stations are in range", () => {
  const harborGauge = buoy("HARBOR", LA.lat + 0.01, LA.lon, null);
  const map = nearestBuoyPerPort([LA], [harborGauge], MAX_JOIN_KM, hasWaveHeight);
  assert.equal(map.get("port_la"), null);
});

test("buoyPortProximityReport + hasWaveHeight filter: default (unfiltered) call is unaffected", () => {
  // Regression guard: adding the filter param must not change the existing default-call behavior.
  const report = buoyPortProximityReport([LA], [buoy("HARBOR", LA.lat, LA.lon, null)]);
  assert.equal(report.ports_matched, 1);
  assert.equal(report.matches[0].station, "HARBOR");
});
