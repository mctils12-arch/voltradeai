// FLIGHT PROGRAM B1 — the static world disc plan: exact cover of every
// mask box (dense haversine sampling), every major airport region covered,
// remote ocean/Antarctica deliberately NOT, deterministic.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildWorldPlan, worldPlan, coveredByPlan, haversineNm, inBox, wrapLon,
  TRAFFIC_MASK, PLAN_RADIUS_NM, PLAN_LAT_MIN, PLAN_LAT_MAX,
} from "./globalDiscPlan";

const plan = worldPlan();

// ~60 major airports on every inhabited continent + the island airports
// long-haul traffic depends on (IATA: lat, lon).
const AIRPORTS: Record<string, [number, number]> = {
  ATL: [33.64, -84.43], DFW: [32.90, -97.04], DEN: [39.86, -104.67], ORD: [41.98, -87.90],
  LAX: [33.94, -118.41], JFK: [40.64, -73.78], SEA: [47.45, -122.31], MIA: [25.79, -80.29],
  YYZ: [43.68, -79.63], YVR: [49.19, -123.18], ANC: [61.17, -150.00], HNL: [21.32, -157.92],
  MEX: [19.44, -99.07], PTY: [9.07, -79.38], SJU: [18.44, -66.00], BDA: [32.36, -64.68],
  BOG: [4.70, -74.15], LIM: [-12.02, -77.11], GRU: [-23.43, -46.47], EZE: [-34.82, -58.54],
  SCL: [-33.39, -70.79], LHR: [51.47, -0.45], CDG: [49.01, 2.55], AMS: [52.31, 4.76],
  FRA: [50.04, 8.56], MAD: [40.47, -3.56], FCO: [41.80, 12.25], IST: [41.26, 28.74],
  SVO: [55.97, 37.41], OSL: [60.19, 11.10], HEL: [60.32, 24.96], KEF: [63.99, -22.62],
  LPA: [27.93, -15.39], PDL: [37.74, -25.70], CAI: [30.12, 31.41], ADD: [8.98, 38.80],
  NBO: [-1.32, 36.93], LOS: [6.58, 3.32], JNB: [-26.14, 28.25], CPT: [-33.97, 18.60],
  MRU: [-20.43, 57.68], SEZ: [-4.67, 55.52], DXB: [25.25, 55.36], DOH: [25.27, 51.61],
  RUH: [24.96, 46.70], TLV: [32.01, 34.89], DEL: [28.56, 77.10], BOM: [19.09, 72.87],
  MLE: [4.19, 73.53], CMB: [7.18, 79.88], PEK: [40.08, 116.58], PVG: [31.14, 121.81],
  CAN: [23.39, 113.30], HKG: [22.31, 113.91], TPE: [25.08, 121.23], ICN: [37.46, 126.44],
  HND: [35.55, 139.78], NRT: [35.77, 140.39], BKK: [13.69, 100.75], SIN: [1.36, 103.99],
  KUL: [2.75, 101.71], CGK: [-6.13, 106.66], MNL: [14.51, 121.02], GUM: [13.48, 144.80],
  SYD: [-33.95, 151.18], MEL: [-37.67, 144.84], PER: [-31.94, 115.97], AKL: [-37.01, 174.79],
  NAN: [-17.76, 177.44], PPT: [-17.55, -149.61], ALA: [43.35, 77.04], TAS: [41.26, 69.28],
  NOU: [-22.01, 166.21], APW: [-13.83, -172.01],
};

test("every major airport region is inside some plan disc", () => {
  const missing = Object.entries(AIRPORTS).filter(([, [la, lo]]) => !coveredByPlan(la, lo, plan)).map(([k]) => k);
  assert.deepEqual(missing, [], `uncovered airports: ${missing.join(", ")}`);
  assert.ok(Object.keys(AIRPORTS).length >= 30);
});

test("exact cover: every point of every mask box (0.5° grid) is within the disc radius (haversine)", () => {
  let checked = 0;
  const misses: string[] = [];
  for (const b of TRAFFIC_MASK) {
    for (let la = b.lat0; la <= b.lat1 + 1e-9; la += 0.5) {
      for (let lo = b.lon0; lo <= b.lon1 + 1e-9; lo += 0.5) {
        checked++;
        if (!coveredByPlan(la, lo, plan) && misses.length < 5) misses.push(`${b.name} ${la},${lo}`);
      }
    }
  }
  assert.deepEqual(misses, []);
  assert.ok(checked > 100_000, `dense sample (${checked} points)`);
});

test("remote open ocean and Antarctica are deliberately NOT swept (no receivers — pure cost)", () => {
  const remote: [string, number, number][] = [
    ["central Pacific", 0, -140], ["south Pacific", -40, -120], ["south Atlantic", -40, -15],
    ["central Indian Ocean", -30, 80], ["Southern Ocean", -60, 0], ["Antarctica", -75, 0],
    ["high Arctic", 85, 0],
  ];
  for (const [name, la, lo] of remote) assert.equal(coveredByPlan(la, lo, plan), false, name);
  for (const d of plan) {
    assert.ok(d.lat >= PLAN_LAT_MIN - 0.01 && d.lat <= PLAN_LAT_MAX + 0.01, `disc ${d.id} poleward of plan`);
    assert.equal(d.radiusNm, PLAN_RADIUS_NM);
    assert.ok(d.lon >= -180 && d.lon < 180);
  }
});

test("plan is deterministic, sized sanely, and ids are stable indexes", () => {
  const again = buildWorldPlan();
  assert.deepEqual(again, plan);
  plan.forEach((d, i) => assert.equal(d.id, i));
  assert.ok(plan.length > 300 && plan.length < 1200, `plan size ${plan.length}`);
  // dense regions carry the higher first-pass prior
  const nearJfk = plan.filter((d) => haversineNm(d.lat, d.lon, 40.64, -73.78) <= d.radiusNm);
  assert.ok(nearJfk.some((d) => d.prior >= 300));
});

test("helpers: wrapLon / inBox across the antimeridian", () => {
  assert.equal(wrapLon(190), -170);
  assert.equal(wrapLon(-190), 170);
  assert.equal(wrapLon(180), -180);
  const fijiW = TRAFFIC_MASK.find((b) => b.name === "south-pacific-islands-w")!;
  assert.equal(inBox(-17.76, 177.44, fijiW), true);
  assert.equal(inBox(-17.76, -180, fijiW), true, "180 and -180 are the same meridian");
  assert.equal(inBox(0, 177, fijiW), false);
});
