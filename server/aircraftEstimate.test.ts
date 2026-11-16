// aircraftEstimate — the estimated position and path served beside an aircraft's last real fix
// (2026-11-16, owner: "keep the model moving by dead reckoning between updates ... where there is no data, show an
// estimated path along the filed route, drawn dashed so it's clearly estimated"). Every expected number below is
// worked out in the test from speed x time and spherical geometry, not read back from the code.
// Run: npx tsx --test server/aircraftEstimate.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import {
  estimateAircraft, motionBetween, pointAlong, DEAD_RECKONING_MAX_SEC, DEAD_RECKONING_SHORT_SEC, ESTIMATE_MAX_AGE_SEC,
  ESTIMATE_MIN_GS_KT, PATH_AHEAD_SEC, ROUTE_JOIN_MAX_NM, SPEED_MARGIN, type EstimateFix, type EstimateRoute,
} from "../shared/aircraftEstimate";
import { crossTrackNm, cumulativeNm, haversineNm, initialBearingDeg } from "../shared/flightPlanGeometry";
import { createEstimator, registerAircraftEstimateRoute, ESTIMATE_PLAN_TTL_MS, type EstimateDeps, type EstimateLiveRow, type EstimatePlan } from "./aircraftEstimate";

const T0 = 1_800_000_000_000;
const here = path.dirname(fileURLToPath(import.meta.url));
const fixAt = (o: Partial<EstimateFix> = {}): EstimateFix => ({ lat: 40, lon: -100, altFt: 36000, gsKt: 480, trkDeg: 90, onGround: false, fixMs: T0, ...o });
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

test("dead reckoning: 480 kt for 60 s is 8.0 nm straight along the last track; the answer says it is an estimate", () => {
  const e = estimateAircraft(fixAt(), T0 + 60_000);
  assert.equal(e.estimated, true);
  assert.equal(e.method, "dead_reckoning");
  assert.equal(e.ageSec, 60);
  assert.equal(e.distanceNm, 8); // 480 nm/h x 60 s / 3600
  near(haversineNm({ lat: 40, lon: -100 }, { lat: e.lat!, lon: e.lon! }), 8, 0.01, "distance from the fix");
  near(initialBearingDeg({ lat: 40, lon: -100 }, { lat: e.lat!, lon: e.lon! }), 90, 0.2, "bearing from the fix");
  assert.equal(e.withinNm, 9.2); // 8.0 x 1.15
  assert.equal(SPEED_MARGIN, 1.15);
  assert.equal(e.altFt, 36000);
  assert.equal(e.altBasis, "held");
  assert.equal(e.route, null);
  assert.deepEqual(e.path[0], [40, -100, T0], "the path starts at the real fix");
  assert.equal(e.path[e.path.length - 1][2], T0 + DEAD_RECKONING_MAX_SEC * 1000, "and runs no further than dead reckoning is allowed");
  assert.match(e.reason, /straight ahead \(no route available\)/);
  assert.ok(e.assumptions.includes("no turn since the last fix"));
});

test("without a route nothing is estimated past 120 s: no position, only the circle the aircraft must be inside", () => {
  assert.equal(DEAD_RECKONING_MAX_SEC, 120);
  assert.equal(estimateAircraft(fixAt(), T0 + 120_000).method, "dead_reckoning");
  const e = estimateAircraft(fixAt(), T0 + 121_000);
  assert.equal(e.method, "none");
  assert.equal(e.lat, null);
  assert.equal(e.lon, null);
  assert.deepEqual(e.path, []);
  near(e.withinNm!, (480 * 121 / 3600) * 1.15, 0.06, "withinNm");
  assert.match(e.reason, /121 s old, past the 120 s a straight line is used for, and no route available/);
});

test("nothing is extrapolated on the ground, under 40 kt, without speed or track, from a future fix, or after 6 hours", () => {
  for (const [f, re] of [
    [fixAt({ onGround: true }), /on the ground/],
    [fixAt({ gsKt: 39 }), /under 40 kt/],
    [fixAt({ gsKt: null }), /no ground speed or track/],
    [fixAt({ trkDeg: null }), /no ground speed or track/],
  ] as const) {
    const e = estimateAircraft(f, T0 + 10_000);
    assert.equal(e.method, "none");
    assert.equal(e.lat, null);
    assert.match(e.reason, re);
  }
  assert.equal(ESTIMATE_MIN_GS_KT, 40);
  assert.equal(estimateAircraft(fixAt(), T0 - 1).method, "none");
  const old = estimateAircraft(fixAt(), T0 + (ESTIMATE_MAX_AGE_SEC + 1) * 1000, { points: [{ lat: 40, lon: -100 }, { lat: 40, lon: -60 }], source: "FILED_FAA" });
  assert.equal(old.method, "none");
  assert.match(old.reason, /more than 6 hours old/);
});

// a route due north then due east, so each leg's length is plain arithmetic: 60 nm per degree of latitude
const NORTH_THEN_EAST: EstimateRoute = { source: "FILED_FAA", points: [{ lat: 40, lon: -100 }, { lat: 41, lon: -100 }, { lat: 41, lon: -98 }] };

test("along the route: 360 kt for 900 s is 90 nm, 60 up the first leg and 30 along the second, turn included", () => {
  const fix = fixAt({ gsKt: 360, trkDeg: 0 });
  const e = estimateAircraft(fix, T0 + 900_000, NORTH_THEN_EAST);
  assert.equal(e.method, "along_route");
  assert.equal(e.distanceNm, 90);
  const cum = cumulativeNm(NORTH_THEN_EAST.points);
  near(cum[1], 60.04, 0.05, "first leg is one degree of latitude");
  // 90 nm along: 30 nm (less the 0.04) past the corner, on the second leg
  const corner = NORTH_THEN_EAST.points[1];
  near(haversineNm(corner, { lat: e.lat!, lon: e.lon! }), 90 - cum[1], 0.02, "distance past the corner");
  near(crossTrackNm({ lat: e.lat!, lon: e.lon! }, NORTH_THEN_EAST.points)!.nm, 0, 0.01, "the estimate is on the route");
  assert.ok(e.lon! > -100 && e.lon! < -98);
  near(e.trkDeg!, 90, 1, "pointing along the second leg");
  assert.deepEqual(e.route, { source: "FILED_FAA", crossTrackAtFixNm: 0, remainingNm: Math.round((cum[2] - 90) * 10) / 10, reachedEnd: false });
  assert.equal(e.altFt, 36000);
  // the dashed path: starts at the real fix, passes the corner, ends 10 minutes past the estimate, all of it on the route
  assert.deepEqual(e.path[0], [40, -100, T0]);
  assert.ok(e.path.some((p) => p[0] === 41 && p[1] === -100), "the route's corner is a path point");
  assert.equal(e.path[e.path.length - 1][2], T0 + (900 + PATH_AHEAD_SEC) * 1000);
  for (const p of e.path) assert.ok(crossTrackNm({ lat: p[0], lon: p[1] }, NORTH_THEN_EAST.points)!.nm < 0.05, "every path point is on the route");
  for (let i = 1; i < e.path.length; i++) assert.ok(e.path[i][2] >= e.path[i - 1][2], "path times never go back");
  assert.match(e.reason, /advanced along the FILED_FAA route at the last ground speed/);
  assert.ok(e.assumptions.some((a) => /stayed on the FILED_FAA route/.test(a)));
});

test("the source of the route is carried through: a predicted route is never called filed", () => {
  const e = estimateAircraft(fixAt({ gsKt: 360, trkDeg: 0 }), T0 + 300_000, { ...NORTH_THEN_EAST, source: "ROUTE_DB_PREDICTED" });
  assert.equal(e.route?.source, "ROUTE_DB_PREDICTED");
  assert.match(e.reason, /ROUTE_DB_PREDICTED/);
  assert.doesNotMatch(JSON.stringify(e), /FILED/);
});

test("the estimate stops at the route's last point and says so", () => {
  const total = cumulativeNm(NORTH_THEN_EAST.points)[2];
  const e = estimateAircraft(fixAt({ gsKt: 360, trkDeg: 0 }), T0 + 3 * 3600_000, NORTH_THEN_EAST); // 1,080 nm of time on a ~150 nm route
  assert.equal(e.method, "along_route");
  assert.equal(e.lat, 41);
  assert.equal(e.lon, -98);
  assert.equal(e.route?.reachedEnd, true);
  assert.equal(e.route?.remainingNm, 0);
  near(e.distanceNm!, total, 0.06, "distance is the route's length, not speed x time");
  assert.match(e.reason, /to its last point and stopped there/);
  const last = e.path[e.path.length - 1];
  assert.deepEqual([last[0], last[1]], [41, -98], "the path ends at the route's end");
});

test("with a route: straight ahead for the first 30 s only, then the route takes over", () => {
  assert.equal(DEAD_RECKONING_SHORT_SEC, 30);
  const fix = fixAt({ gsKt: 360, trkDeg: 0 });
  const young = estimateAircraft(fix, T0 + 30_000, NORTH_THEN_EAST);
  assert.equal(young.method, "dead_reckoning");
  assert.match(young.reason, /straight ahead for the first 30 s, then along the route/);
  assert.equal(young.path[young.path.length - 1][2], T0 + 30_000);
  assert.equal(estimateAircraft(fix, T0 + 31_000, NORTH_THEN_EAST).method, "along_route");
});

test("a route the aircraft was not on is not used: too far from it, or flying another way", () => {
  // 0.2 degrees of longitude at 40N is 9.2 nm: more than the 8 nm join limit
  assert.equal(ROUTE_JOIN_MAX_NM, 8);
  const far = fixAt({ lon: -100.2, gsKt: 360, trkDeg: 0 });
  const e1 = estimateAircraft(far, T0 + 60_000, NORTH_THEN_EAST);
  assert.equal(e1.method, "dead_reckoning");
  assert.match(e1.reason, /9\.2 nm from the route \(more than 8 nm\)/);
  const e2 = estimateAircraft(far, T0 + 300_000, NORTH_THEN_EAST);
  assert.equal(e2.method, "none");
  assert.equal(e2.lat, null);
  // on the route but heading south, away from it
  const away = estimateAircraft(fixAt({ gsKt: 360, trkDeg: 180 }), T0 + 300_000, NORTH_THEN_EAST);
  assert.equal(away.method, "none");
  assert.match(away.reason, /track \(180°\) was not along the route \(0°\)/);
  assert.equal(estimateAircraft(fixAt({ gsKt: 360, trkDeg: 0 }), T0 + 300_000, { source: "FILED_FAA", points: [{ lat: 40, lon: -100 }] }).method, "none");
});

test("pointAlong and motionBetween: the two pieces of arithmetic the estimate rests on", () => {
  const cum = cumulativeNm(NORTH_THEN_EAST.points);
  const mid = pointAlong(NORTH_THEN_EAST.points, cum, cum[1] / 2);
  near(mid.pos.lat, 40.5, 1e-6, "halfway up the first leg");
  near(mid.brgDeg!, 0, 0.01, "heading north");
  assert.equal(pointAlong(NORTH_THEN_EAST.points, cum, 1e9).atEnd, true);
  // one minute of latitude is one nautical mile; covered in 10 s it is 360 kt, due north
  const m = motionBetween({ lat: 40, lon: -100, tMs: T0 }, { lat: 40 + 1 / 60, lon: -100, tMs: T0 + 10_000 });
  near(m!.gsKt, 360, 0.5, "ground speed");
  near(m!.trkDeg, 0, 0.1, "track");
  assert.equal(motionBetween({ lat: 40, lon: -100, tMs: T0 }, { lat: 41, lon: -100, tMs: T0 + 301_000 }), null, "fixes more than 5 minutes apart give no speed");
  assert.equal(motionBetween({ lat: 40, lon: -100, tMs: T0 }, { lat: 40, lon: -100, tMs: T0 + 10_000 }), null, "no movement, no track");
  assert.equal(motionBetween({ lat: 40, lon: -100, tMs: T0 }, { lat: 41, lon: -100, tMs: T0 }), null);
});

// ── the endpoint ─────────────────────────────────────────────────────────────
function rig(o: { row?: EstimateLiveRow | null; tail?: Array<{ t: number; la: number; lo: number; al?: number | null }>; plan?: EstimatePlan | Error } = {}) {
  let clock = T0;
  const planCalls: unknown[] = [];
  const state = { row: o.row ?? null as EstimateLiveRow | null };
  const deps: EstimateDeps = {
    liveRow: () => state.row,
    archiveTail: async () => o.tail ?? [],
    plan: async (q) => {
      planCalls.push(q);
      if (o.plan instanceof Error) throw o.plan;
      return o.plan ?? { source: "NONE", points: [] };
    },
    now: () => clock,
  };
  return { est: createEstimator(deps), deps, planCalls, state, tick: (ms: number) => { clock += ms; } };
}
const ROW = (ageMs: number): EstimateLiveRow => ({ lat: 40, lon: -100, altFt: 36000, gsKt: 360, trk: 0, gnd: false, callsign: "TST1", seenAt: T0 - ageMs, src: "adsblol" });
const FILED: EstimatePlan = { source: "FILED_FAA", label: "Filed (FAA)", points: NORTH_THEN_EAST.points };

test("endpoint: a young fix is dead-reckoned with NO plan lookup; the real fix and the estimate are separate objects", async () => {
  const r = rig({ row: ROW(10_000), plan: FILED });
  const b = await r.est.estimate("AB8C8E");
  assert.equal(b.kind, "estimate");
  assert.equal(b.hex, "ab8c8e");
  assert.deepEqual({ lat: b.fix!.lat, lon: b.fix!.lon, fixMs: b.fix!.fixMs, ageSec: b.fix!.ageSec, source: b.fix!.source, provider: b.fix!.provider },
    { lat: 40, lon: -100, fixMs: T0 - 10_000, ageSec: 10, source: "snapshot", provider: "adsblol" });
  assert.equal(b.estimate!.method, "dead_reckoning");
  assert.equal(b.estimate!.estimated, true);
  assert.equal(b.estimate!.distanceNm, 1); // 360 kt x 10 s
  assert.equal(r.planCalls.length, 0, "this route is polled: no plan is resolved for a fix under 30 s old");
  assert.equal(b.plan, null);
  assert.match(b.estimate!.reason, /the route is looked up once the fix is older than 30 s/);
  assert.match(b.honesty, /NOT an observed position/);
});

test("endpoint: an older fix is advanced along the plan /plan/:hex would serve, looked up once per fix", async () => {
  const r = rig({ row: ROW(300_000), plan: FILED });
  const b = await r.est.estimate("ab8c8e");
  assert.equal(b.estimate!.method, "along_route");
  assert.equal(b.estimate!.distanceNm, 30); // 360 kt x 300 s
  assert.deepEqual(b.plan, { source: "FILED_FAA", label: "Filed (FAA)", privacy: null, used: true, note: null });
  assert.deepEqual(r.planCalls, [{ hex: "ab8c8e", callsign: "TST1", lat: 40, lon: -100, altFt: 36000, trkDeg: 0, fixT: T0 - 300_000 }]);
  r.tick(5_000);
  const again = await r.est.estimate("ab8c8e");
  assert.equal(again.estimate!.distanceNm, 30.5, "the estimate moves on with the clock");
  assert.equal(r.planCalls.length, 1, "same fix inside the cache time: the plan is not resolved again");
  r.state.row = { ...ROW(0), seenAt: T0 + 5_000 - 40_000 }; // a NEW fix, 40 s old
  await r.est.estimate("ab8c8e");
  assert.equal(r.planCalls.length, 2, "a new fix resolves the plan again");
  r.tick(ESTIMATE_PLAN_TTL_MS + 1);
  await r.est.estimate("ab8c8e");
  assert.equal(r.planCalls.length, 3, "and so does an expired cache entry");
});

test("endpoint: no live row — the archive's newest position is the basis, its speed and track worked out and flagged", async () => {
  // two archived positions 60 s apart, one minute of latitude each 10 s: 6 nm in 60 s = 360 kt, due north
  const tail = [{ t: (T0 - 660_000) / 1000, la: 39.9, lo: -100, al: 10973 }, { t: (T0 - 600_000) / 1000, la: 40, lo: -100, al: 10973 }];
  const r = rig({ tail, plan: FILED });
  const b = await r.est.estimate("ab8c8e");
  assert.equal(b.fix!.source, "archive");
  assert.equal(b.fix!.motionDerived, true);
  near(b.fix!.gsKt!, 360, 1, "derived ground speed");
  near(b.fix!.trkDeg!, 0, 0.1, "derived track");
  assert.equal(b.fix!.altFt, Math.round(10973 * 3.28084));
  assert.equal(b.fix!.ageSec, 600);
  assert.equal(b.estimate!.method, "along_route");
  near(b.estimate!.distanceNm!, 60, 0.3, "360 kt for 600 s");
  assert.ok(b.estimate!.assumptions.some((a) => /worked out from the last two archived positions/.test(a)));
  // one archived position alone gives no speed: no estimate, and the reason says so
  const one = await rig({ tail: [tail[1]], plan: FILED }).est.estimate("ab8c8e");
  assert.equal(one.estimate!.method, "none");
  assert.match(one.estimate!.reason, /no ground speed or track/);
});

test("endpoint: nothing held anywhere is said plainly; a bad hex is refused", async () => {
  const b = await rig().est.estimate("ab8c8e");
  assert.equal(b.fix, null);
  assert.equal(b.estimate, null);
  assert.match(b.reason!, /no position held for this aircraft/);
  await assert.rejects(rig().est.estimate("zz"), /hex required/);
});

test("endpoint: a withheld (LADD) or missing plan never becomes a route; a failed lookup is reported, not hidden", async () => {
  const ladd = await rig({ row: ROW(300_000), plan: { source: "NONE", label: "No plan", privacy: "LADD", points: [] } }).est.estimate("ab8c8e");
  assert.equal(ladd.estimate!.method, "none");
  assert.equal(ladd.estimate!.lat, null);
  assert.deepEqual(ladd.plan, { source: "NONE", label: "No plan", privacy: "LADD", used: false, note: null });
  // inside two minutes the straight line still works without any plan
  const young = await rig({ row: ROW(90_000), plan: { source: "NONE", privacy: "LADD", points: [] } }).est.estimate("ab8c8e");
  assert.equal(young.estimate!.method, "dead_reckoning");
  assert.equal(young.plan!.used, false);
  const failed = await rig({ row: ROW(300_000), plan: new Error("route DB timeout") }).est.estimate("ab8c8e");
  assert.equal(failed.estimate!.method, "none");
  assert.deepEqual(failed.plan, { source: "NONE", label: null, privacy: null, used: false, note: "the route lookup failed: route DB timeout" });
});

test("the route: GET /api/data/aircraft/estimate/:hex on a real Express app, never cached, and wired in routes.ts", async () => {
  const r = rig({ row: ROW(300_000), plan: FILED });
  const app = express();
  registerAircraftEstimateRoute(app, r.deps);
  const server = app.listen(0);
  try {
    const port = (server.address() as { port: number }).port;
    const res = await fetch(`http://127.0.0.1:${port}/api/data/aircraft/estimate/AB8C8E`);
    assert.equal(res.status, 200);
    assert.equal(res.headers.get("cache-control"), "no-store");
    const body = await res.json() as { kind: string; estimate: { method: string; estimated: boolean } };
    assert.equal(body.kind, "estimate");
    assert.equal(body.estimate.method, "along_route");
    assert.equal(body.estimate.estimated, true);
    assert.equal((await fetch(`http://127.0.0.1:${port}/api/data/aircraft/estimate/nothex`)).status, 400);
  } finally {
    server.close();
  }
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.match(routes, /registerAircraftEstimateRoute\(app, \{\s*liveRow: \(hex\) => globalAircraft\.snapshot\.get\(hex\),\s*archiveTail: \(hex\) => recentTrackCached\("aircraft", hex\),\s*plan: \(q\) => resolveFlightPlan\(q, getFlightPlanContext\(\)\),\s*\}\);/);
  // the estimate module itself reads no SWIM/FAA source: the plan comes only through resolveFlightPlan's own withholding
  const src = fs.readFileSync(path.join(here, "aircraftEstimate.ts"), "utf8") + fs.readFileSync(path.join(here, "..", "shared", "aircraftEstimate.ts"), "utf8");
  assert.doesNotMatch(src, /from "\.\/swim|from "\.\/laddRegistry|from "\.\/flightPlans"/);
});
