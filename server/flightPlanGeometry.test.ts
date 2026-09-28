// Tests for shared/flightPlanGeometry.ts. Lives under server/ because the
// test gate (scripts/gated_tests.sh) collects server/*.test.ts and
// client/**/*.test.ts only — a shared/*.test.ts would never run in CI.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  haversineNm, greatCircle, crossTrackNm, splitPlanAt, estimateVerticalProfile,
  replanFromPosition, unwrapLons, densifyPlan, normLon, angleDiffDeg, initialBearingDeg,
  cumulativeNm, typicalCruiseFt, CLIMB_FT_PER_NM, DESCENT_FT_PER_NM, FT_PER_M,
  REPLAN_MIN_AHEAD_NM, type PlanPoint, type LatLon,
} from "../shared/flightPlanGeometry";

const KSFO = { lat: 37.619, lon: -122.375 };
const WSSS = { lat: 1.3502, lon: 103.994 };
const pp = (lat: number, lon: number, altM: number | null = null, altEstimated = altM == null, name?: string): PlanPoint =>
  ({ lat, lon, altM, altEstimated, ...(name ? { name } : {}) });
const near = (a: number, b: number, tol: number, msg?: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg ?? ""} expected ${b} ± ${tol}, got ${a}`);

// ── basics ──────────────────────────────────────────────────────────────────

test("haversineNm: 1° of latitude ≈ 60nm; symmetric; zero for identical points", () => {
  near(haversineNm({ lat: 0, lon: 0 }, { lat: 1, lon: 0 }), 60.04, 0.1);
  assert.equal(haversineNm(KSFO, KSFO), 0);
  near(haversineNm(KSFO, WSSS), haversineNm(WSSS, KSFO), 1e-9);
  near(haversineNm(KSFO, WSSS), 7340, 30, "SFO-SIN great circle");
});

test("antimeridian: 179E -> 179W is 2° of longitude, never the long way", () => {
  const a = { lat: 0, lon: 179 }, b = { lat: 0, lon: -179 };
  near(haversineNm(a, b), 120.08, 0.2);
  const gc = greatCircle(a, b, 10);
  assert.ok(gc.length >= 13);
  for (const p of gc) assert.ok(Math.abs(p.lon) >= 178.99, `point ${p.lon} left the short arc`);
  const un = unwrapLons(gc);
  for (let i = 1; i < un.length; i++) assert.ok(Math.abs(un[i].lon - un[i - 1].lon) < 180);
  assert.ok(un[un.length - 1].lon > 180, "unwrapped end continues past 180");
});

test("normLon / angleDiffDeg / initialBearingDeg", () => {
  assert.equal(normLon(190), -170);
  assert.equal(normLon(-190), 170);
  assert.equal(normLon(45), 45);
  assert.equal(angleDiffDeg(350, 10), 20);
  assert.equal(angleDiffDeg(10, 190), 180);
  near(initialBearingDeg({ lat: 0, lon: 0 }, { lat: 0, lon: 10 }), 90, 1e-6);
  near(initialBearingDeg({ lat: 0, lon: 0 }, { lat: 10, lon: 0 }), 0, 1e-6);
});

test("greatCircle: endpoints exact, spacing <= step, zero-length is a single point", () => {
  const gc = greatCircle(KSFO, WSSS, 100);
  assert.deepEqual(gc[0], { lat: KSFO.lat, lon: KSFO.lon });
  assert.deepEqual(gc[gc.length - 1], { lat: WSSS.lat, lon: WSSS.lon });
  for (let i = 1; i < gc.length; i++) assert.ok(haversineNm(gc[i - 1], gc[i]) <= 100.01);
  // the SFO-SIN great circle crosses the antimeridian northbound of Hawaii's latitude
  assert.ok(gc.some((p) => p.lon > 150) && gc.some((p) => p.lon < -150));
  near(cumulativeNm(gc)[gc.length - 1], haversineNm(KSFO, WSSS), 0.5, "densified length ≈ GC length");
  assert.deepEqual(greatCircle(KSFO, KSFO), [{ lat: KSFO.lat, lon: KSFO.lon }]);
});

// ── cross-track ─────────────────────────────────────────────────────────────

test("crossTrackNm: equator segment — distance, side sign, along, projection", () => {
  const line: LatLon[] = [{ lat: 0, lon: 0 }, { lat: 0, lon: 10 }];
  const n = crossTrackNm({ lat: 0.1, lon: 5 }, line)!; // north of an eastbound track = LEFT
  near(n.nm, 6.0, 0.05);
  assert.ok(n.signedNm < 0, "left of track is negative");
  near(n.alongNm, 300.2, 0.5);
  assert.equal(n.segIndex, 0);
  near(n.proj.lat, 0, 1e-9);
  near(n.proj.lon, 5, 1e-6);
  const s = crossTrackNm({ lat: -0.1, lon: 5 }, line)!;
  assert.ok(s.signedNm > 0, "right of track is positive");
});

test("crossTrackNm: beyond a segment end measures to the endpoint; picks the nearest segment", () => {
  const line: LatLon[] = [{ lat: 0, lon: 0 }, { lat: 0, lon: 10 }, { lat: 10, lon: 10 }];
  const beyond = crossTrackNm({ lat: 0, lon: -1 }, line)!;
  near(beyond.nm, 60, 0.2);
  assert.equal(beyond.alongNm, 0);
  const onSecond = crossTrackNm({ lat: 5, lon: 10.1 }, line)!;
  assert.equal(onSecond.segIndex, 1);
  near(onSecond.nm, 6.0, 0.1);
  near(onSecond.alongNm, 600.4 + 300.2, 1.5);
});

test("crossTrackNm: single-point, zero-length segment and empty polylines", () => {
  assert.equal(crossTrackNm({ lat: 0, lon: 0 }, []), null);
  const one = crossTrackNm({ lat: 1, lon: 0 }, [{ lat: 0, lon: 0 }])!;
  near(one.nm, 60.04, 0.1);
  assert.equal(one.segIndex, 0);
  const dup = crossTrackNm({ lat: 1, lon: 0 }, [{ lat: 0, lon: 0 }, { lat: 0, lon: 0 }])!;
  near(dup.nm, 60.04, 0.1);
});

test("crossTrackNm: antimeridian segment", () => {
  const r = crossTrackNm({ lat: 0.5, lon: 180 }, [{ lat: 0, lon: 179 }, { lat: 0, lon: -179 }])!;
  near(r.nm, 30, 0.2);
  near(r.alongNm, 60, 0.3);
});

// ── split ───────────────────────────────────────────────────────────────────

test("splitPlanAt: ahead starts exactly at the projected position, behind ends there", () => {
  const plan = [pp(0, 0, 0, false, "A"), pp(0, 10, 10000, false, "B"), pp(0, 20, 0, false, "C")];
  const { behind, ahead, crossTrackNm: xt } = splitPlanAt(plan, { lat: 0.2, lon: 5 });
  near(xt!, 12, 0.1);
  assert.deepEqual(behind[behind.length - 1], ahead[0]);
  near(ahead[0].lat, 0, 1e-9);
  near(ahead[0].lon, 5, 1e-6);
  near(ahead[0].altM!, 5000, 1);
  assert.equal(ahead[0].altEstimated, true, "interpolated altitude is an estimate");
  assert.deepEqual(ahead.slice(1).map((p) => p.name), ["B", "C"]);
  assert.deepEqual(behind.slice(0, -1).map((p) => p.name), ["A"]);
});

test("splitPlanAt: at a vertex no duplicate point; single-point and empty plans", () => {
  const plan = [pp(0, 0, null, true, "A"), pp(0, 10, null, true, "B"), pp(0, 20, null, true, "C")];
  const { behind, ahead } = splitPlanAt(plan, { lat: 0, lon: 10 });
  assert.equal(ahead[0].name, "B");
  assert.deepEqual(ahead.map((p) => p.name), ["B", "C"]);
  assert.deepEqual(behind.map((p) => p.name), ["A", "B"]);
  const single = splitPlanAt([pp(1, 1)], { lat: 0, lon: 0 });
  assert.equal(single.ahead.length, 1);
  const empty = splitPlanAt([], { lat: 0, lon: 0 });
  assert.deepEqual(empty, { behind: [], ahead: [], crossTrackNm: null, alongNm: null });
});

// ── vertical profile ────────────────────────────────────────────────────────

test("estimateVerticalProfile: climbs at 250ft/nm, cruises, descends on 3°; all flagged estimated", () => {
  const route = densifyPlan([pp(0, 0), pp(0, 10)], 10); // ~600nm
  const prof = estimateVerticalProfile(route, 0, 0, 30000);
  assert.ok(prof.every((p) => p.altM != null && p.altEstimated));
  const toc = prof.find((p) => p.name === "TOC (est.)")!;
  const tod = prof.find((p) => p.name === "TOD (est.)")!;
  assert.ok(toc && tod, "top of climb/descent inserted");
  const cum = cumulativeNm(prof);
  near(cum[prof.indexOf(toc)], 30000 / CLIMB_FT_PER_NM, 1);
  near(cum[prof.length - 1] - cum[prof.indexOf(tod)], 30000 / DESCENT_FT_PER_NM, 1);
  near(DESCENT_FT_PER_NM, 318.4, 0.2);
  near(toc.altM! * FT_PER_M, 30000, 5);
  assert.equal(prof[0].altM, 0);
  assert.equal(prof[prof.length - 1].altM, 0);
  const max = Math.max(...prof.map((p) => p.altM!));
  near(max * FT_PER_M, 30000, 5);
});

test("estimateVerticalProfile: short hop never reaches cruise — peaks where climb meets descent", () => {
  const route = densifyPlan([pp(0, 0), pp(0, 1)], 5); // ~60nm
  const prof = estimateVerticalProfile(route, 100, 200, 35000);
  const peak = prof.find((p) => p.name === "peak (est.)");
  assert.ok(peak, "peak vertex inserted");
  assert.ok(peak!.altM! * FT_PER_M < 35000);
  assert.ok(prof.every((p) => p.altM! * FT_PER_M <= 35000 + 1));
  assert.equal(prof[0].altM, 100);
  assert.equal(prof[prof.length - 1].altM, 200);
});

test("estimateVerticalProfile: filed/observed altitudes are kept; null cruise uses the stage-length prior", () => {
  const prof = estimateVerticalProfile([pp(0, 0), pp(0, 5, 11000, false, "FIX"), pp(0, 10)], 0, 0, null);
  const fix = prof.find((p) => p.name === "FIX")!;
  assert.equal(fix.altM, 11000);
  assert.equal(fix.altEstimated, false);
  assert.equal(typicalCruiseFt(600), 36000);
  assert.equal(typicalCruiseFt(50), 12000);
  assert.deepEqual(estimateVerticalProfile([], 0, 0, 30000), []);
  const one = estimateVerticalProfile([pp(0, 0)], 50, 0, 30000);
  assert.equal(one[0].altM, 50);
  assert.equal(one[0].altEstimated, true);
});

// ── re-plan ─────────────────────────────────────────────────────────────────

const eastbound = densifyPlan([pp(0, 0, null, true, "ORIG"), pp(0, 10, null, true, "DEST")], 25);

test("replanFromPosition: rejoins at the first vertex >= 20nm ahead within ±70° of track", () => {
  // 30nm north of the plan, heading east-south-east back toward it
  const r = replanFromPosition(eastbound, { lat: 0.5, lon: 3, altM: 9000 }, 110);
  assert.equal(r.mode, "REJOIN");
  assert.equal(r.points[0].name, "present position");
  assert.equal(r.points[0].altM, 9000);
  assert.equal(r.points[0].altEstimated, false);
  const cum = cumulativeNm(eastbound);
  const projAlong = crossTrackNm({ lat: 0.5, lon: 3 }, eastbound)!.alongNm;
  assert.ok(cum[r.rejoinIndex!] - projAlong >= REPLAN_MIN_AHEAD_NM);
  assert.ok(angleDiffDeg(initialBearingDeg({ lat: 0.5, lon: 3 }, eastbound[r.rejoinIndex!]), 110) <= 70);
  assert.equal(r.points[r.points.length - 1].name, "DEST");
  // rejoined tail is the original plan verbatim
  assert.deepEqual(r.points.slice(-3), eastbound.slice(-3));
});

test("replanFromPosition: heading away from every vertex -> direct great circle to destination", () => {
  // flying due north, plan is to the east: nothing within ±70° of 000 except far-ahead vertices?
  const r = replanFromPosition(eastbound, { lat: 2, lon: 1 }, 270);
  assert.equal(r.mode, "DIRECT");
  assert.equal(r.rejoinIndex, null);
  assert.equal(r.points[r.points.length - 1].name, "DEST");
  assert.ok(r.points.slice(1, -1).every((p) => p.altM == null && p.altEstimated));
});

test("replanFromPosition: without a track the heading test is skipped", () => {
  const r = replanFromPosition(eastbound, { lat: 0.5, lon: 3 }, null);
  assert.equal(r.mode, "REJOIN");
});

test("replanFromPosition: near the destination (< 20nm left) goes direct; empty and single-point plans", () => {
  const r = replanFromPosition(eastbound, { lat: 0.1, lon: 9.9 }, 90);
  assert.equal(r.mode, "DIRECT");
  assert.equal(replanFromPosition([], { lat: 0, lon: 0 }, 0).mode, "EMPTY");
  const one = replanFromPosition([pp(0, 5, null, true, "X")], { lat: 0, lon: 0 }, 90);
  assert.equal(one.mode, "DIRECT");
  assert.equal(one.points[one.points.length - 1].name, "X");
});

test("replanFromPosition: antimeridian-crossing plan rejoins across the dateline", () => {
  const plan = densifyPlan([pp(10, 170, null, true, "W"), pp(10, -170, null, true, "E")], 25);
  const r = replanFromPosition(plan, { lat: 10.6, lon: 175 }, 95);
  assert.equal(r.mode, "REJOIN");
  for (let i = 1; i < r.points.length; i++) {
    assert.ok(haversineNm(r.points[i - 1], r.points[i]) < 60, "no leg jumps the long way round");
  }
});

test("densifyPlan: keeps originals, bounded spacing, inserted points left for the profile estimate", () => {
  const d = densifyPlan([pp(0, 0, 1000, false, "A"), pp(0, 2, 3000, false, "B")], 30);
  assert.equal(d[0].name, "A");
  assert.equal(d[0].altM, 1000);
  assert.equal(d[d.length - 1].name, "B");
  for (let i = 1; i < d.length; i++) assert.ok(haversineNm(d[i - 1], d[i]) <= 30.01);
  assert.ok(d.slice(1, -1).every((p) => p.altEstimated && p.altM == null));
  assert.equal(densifyPlan([pp(0, 0), pp(0, 0)]).length, 1, "zero-length plan collapses to one point");
});

test("REGRESSION (live smoke 2026-09-28): a densified airport-to-airport plan cruises, not at field elevation", () => {
  // SFO (4 m) -> SIN (7 m): densify used to interpolate the two FIELD
  // elevations across 7,000 nm, so the profile estimate never ran mid-route
  const plan = densifyPlan([pp(KSFO.lat, KSFO.lon, 4, false, "SFO"), pp(WSSS.lat, WSSS.lon, 7, false, "SIN")], 50);
  const prof = estimateVerticalProfile(plan, 4, 7, 36000);
  const mid = prof[prof.length >> 1];
  near(mid.altM! * FT_PER_M, 36000, 5, "mid-route altitude is the cruise estimate");
  assert.equal(mid.altEstimated, true);
  assert.equal(prof[0].altM, 4);
  assert.equal(prof[prof.length - 1].altM, 7);
});
