// FORWARD-ONLY SEAM + TERMINAL VECTORING (2026-09-30) — regression for the
// live AAL892R -> KAUS screenshot: an aircraft being vectored ~5 nm beside its
// FILED arrival, 25 nm out, drew a connector SIDEWAYS to the perpendicular
// foot on the route and then a 90° corner onto it. Pins: the server's
// "present position" vertex is the aircraft's ACTUAL position followed by a
// join vertex AHEAD in its direction of travel; the connector is flagged as
// ATC vectors (terminalVectoring + point.vectors) inside the no-judge radius;
// an on-route en-route aircraft is unchanged; plus the shared pure rule.
// Run: npx tsx --test server/flightPlanSeam.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  RouteDbClient, HistoryIndex, DeviationTracker, resolveFlightPlan, parsePlanQuery, markVectorsUntil, vectoringNote,
  DEVIATION_TERMINAL_NM, ROUTESET_URL, type PlanContext,
} from "./flightPlans";
import { SwimPlanStore, parseSfdpsMessages } from "./swimSfdps";
import {
  haversineNm, initialBearingDeg, angleDiffDeg, crossTrackNm, forwardJoinOnPlan, chooseForwardJoin,
  isSynthesizedFixName, isTerminalVectoring, seamLeadNm, cumulativeNm, interpolateGC,
  PRESENT_POSITION_ON_PLAN_NAME, PRESENT_POSITION_NAME, TERMINAL_AREA_NM, VECTORING_MIN_XT_NM, SEAM_MIN_LEAD_NM,
  type PlanPoint, type LatLon,
} from "../shared/flightPlanGeometry";

const NOW = Date.parse("2026-09-28T18:00:00Z");

function ctx(opts: { swim?: SwimPlanStore; routes?: Record<string, unknown> } = {}): PlanContext {
  const now = () => NOW;
  return {
    routeDb: new RouteDbClient({
      now, batchWindowMs: 1,
      fetchImpl: (async (url: string) => {
        if (url === ROUTESET_URL) return new Response("", { status: 201 });
        const cs = String(url).split("/").pop()!.replace(".json", "");
        const r = opts.routes?.[cs];
        return r ? new Response(JSON.stringify(r), { status: 200, headers: { "content-type": "application/json" } })
          : new Response("nf", { status: 404 });
      }) as unknown as typeof fetch,
    }),
    history: new HistoryIndex({ now, scan: async () => new Map() }),
    swim: opts.swim ?? new SwimPlanStore(),
    tracker: new DeviationTracker(),
    now,
    recentFixes: async () => [],
    historyBudgetMs: 200,
  };
}

const q = (extra: Record<string, unknown>) =>
  parsePlanQuery("a1b2c3", extra) as Exclude<ReturnType<typeof parsePlanQuery>, { error: string }>;

// A FILED arrival straight down the 97.662°W meridian into KAUS (fixture fix
// names; positions carried in the message so no gazetteer lookup is needed).
const AUS_FIXES: Array<[string, number, number]> = [
  ["FIXAA", 32.0, -97.4], ["FIXBB", 31.2, -97.662], ["FIXCC", 30.8, -97.662], ["FIXDD", 30.55, -97.662],
];
const AUS_ARRIVAL = `<m:MessageCollection xmlns:m="urn:x"><message><flight source="FH" timestamp="2026-09-28T17:00:00Z">
    <flightIdentification aircraftIdentification="AAL892R"/><gufi>g-aus</gufi>
    <departure departurePoint="KDFW"/><arrival arrivalPoint="KAUS"/>
    <requestedAltitude><simple uom="FEET">34000</simple></requestedAltitude>
    <agreed><route nasRouteText="KDFW..FIXAA..FIXBB..FIXCC..FIXDD..KAUS"><expandedRoute>
      ${AUS_FIXES.map(([n, la, lo]) => `<routePoint><point fix="${n}"><location><pos>${la} ${lo}</pos></location></point></routePoint>`).join("")}
    </expandedRoute></route></agreed></flight></message></m:MessageCollection>`;
const KAUS_LL = { lat: 30.1975, lon: -97.662 };
/** 25 nm north of KAUS, 5 nm east of the arrival */
const VECTORED = { lat: KAUS_LL.lat + 25 / 60, lon: KAUS_LL.lon + 5 / (60 * Math.cos(((KAUS_LL.lat + 25 / 60) * Math.PI) / 180)) };
const TRK_TO_KAUS = Math.round(initialBearingDeg(VECTORED, KAUS_LL));

function ausSwim(): SwimPlanStore {
  const swim = new SwimPlanStore();
  for (const f of parseSfdpsMessages(AUS_ARRIVAL)) swim.upsert(f, NOW - 60_000);
  return swim;
}

const KSFO_KLAX = {
  callsign: "SKW5000", airport_codes: "KSFO-KLAX",
  _airports: [
    { name: "San Francisco International Airport", icao: "KSFO", iata: "SFO", lat: 37.618999, lon: -122.375, alt_meters: 3.96 },
    { name: "Los Angeles International Airport", icao: "KLAX", iata: "LAX", lat: 33.942501, lon: -118.407997, alt_meters: 38.1 },
  ],
};

// ── the regression ──────────────────────────────────────────────────────────

test("regression AAL892R: 5 nm beside the filed arrival 25 nm out -> connector joins AHEAD, never sideways/backwards; terminalVectoring", async () => {
  assert.equal(DEVIATION_TERMINAL_NM, TERMINAL_AREA_NM, "one terminal radius, shared with the client");
  const r = await resolveFlightPlan(q({
    callsign: "AAL892R", lat: String(VECTORED.lat), lon: String(VECTORED.lon), alt: "9000", trk: String(TRK_TO_KAUS),
  }), ctx({ swim: ausSwim() }));
  assert.equal(r.source, "FILED_FAA");
  assert.equal(r.pathEstimated, false);
  assert.equal(r.terminalVectoring, true, "inside 40 nm of KAUS and ~5 nm off the filed route");
  assert.ok(r.honesty.includes(vectoringNote("FILED_FAA")));
  assert.match(r.honesty, /ATC vectors — not part of the filed route/);

  const hi = r.points.findIndex((p) => p.name === PRESENT_POSITION_ON_PLAN_NAME);
  assert.ok(hi >= 0, "the present-position vertex is in the plan");
  const here = r.points[hi], join = r.points[hi + 1], next = r.points[hi + 2];
  assert.ok(haversineNm(here, VECTORED) < 0.01, "present position = the aircraft's ACTUAL position, not the perpendicular foot");
  assert.equal(here.altEstimated, false);
  assert.equal(here.vectors, true, "the connector is flagged as ATC vectors");
  assert.ok(r.points.slice(hi + 1).every((p) => !p.vectors), "only the connector is flagged — the filed route ahead is not");

  const conn = initialBearingDeg(here, join);
  assert.ok(angleDiffDeg(conn, TRK_TO_KAUS) <= 90, `never backwards (dot with track >= 0): ${conn} vs trk ${TRK_TO_KAUS}`);
  const interior = 180 - angleDiffDeg(conn, initialBearingDeg(join, next));
  assert.ok(interior >= 110, `connector meets the next plan leg at ${interior.toFixed(1)}° (>= 110°, no 90° corner)`);
  const filed: LatLon[] = [...AUS_FIXES.map(([, la, lo]) => ({ lat: la, lon: lo })), KAUS_LL];
  const xt = crossTrackNm(VECTORED, filed)!;
  assert.ok(xt.nm > 4.5 && xt.nm < 5.5, `fixture is ~5 nm off (${xt.nm})`);
  assert.ok(haversineNm(here, join) >= seamLeadNm(xt.nm) - 0.01, "join lead >= max(3 nm, 2 x cross-track)");
  // contrast: the OLD perpendicular connector formed a ~90° corner here
  const footInterior = 180 - angleDiffDeg(initialBearingDeg(VECTORED, xt.proj), 180);
  assert.ok(footInterior < 110, `the old connector (${footInterior.toFixed(0)}°) would have failed`);
  // named filed fixes survive SFDPS -> densify -> profile -> cap (labels need them)
  for (const nm of ["FIXAA", "FIXBB", "FIXCC"]) assert.ok(r.points.some((p) => p.name === nm), `${nm} kept`);
});

test("forward seam: an on-route en-route plane is unchanged — no vectoring, no flags, connector continues along the route", async () => {
  // exactly ON the KSFO-KLAX great circle (the predicted path), mid-route
  const on = interpolateGC({ lat: 37.618999, lon: -122.375 }, { lat: 33.942501, lon: -118.407997 }, 0.5);
  const r = await resolveFlightPlan(q({ callsign: "SKW5000", lat: String(on.lat), lon: String(on.lon), alt: "30000", trk: "137" }),
    ctx({ routes: { SKW5000: KSFO_KLAX } }));
  assert.equal(r.source, "ROUTE_DB_PREDICTED");
  assert.equal(r.terminalVectoring, false);
  assert.ok(r.points.every((p) => !p.vectors), "nothing flagged as vectors en route");
  assert.doesNotMatch(r.honesty, /ATC vectors/);
  const hi = r.points.findIndex((p) => p.name === PRESENT_POSITION_ON_PLAN_NAME);
  const here = r.points[hi], join = r.points[hi + 1], next = r.points[hi + 2];
  assert.ok(haversineNm(here, on) < 0.01);
  assert.ok(haversineNm(here, join) <= 10.01, "joins the very next vertex ahead (server spacing <= 10 nm), no lead skip");
  assert.ok(angleDiffDeg(initialBearingDeg(here, join), initialBearingDeg(join, next)) < 1, "connector continues along the route");
});

test("forward seam: the vectored aircraft with NO track still never joins behind its projection", async () => {
  const r = await resolveFlightPlan(q({ callsign: "AAL892R", lat: String(VECTORED.lat), lon: String(VECTORED.lon), alt: "9000" }),
    ctx({ swim: ausSwim() }));
  const hi = r.points.findIndex((p) => p.name === PRESENT_POSITION_ON_PLAN_NAME);
  assert.ok(r.points[hi + 1].lat < VECTORED.lat - 0.05, "the join lies south of the aircraft (ahead, toward KAUS)");
  assert.equal(r.terminalVectoring, true);
});

test("markVectorsUntil flags the whole connector (incl. an inserted profile vertex) and nothing from the join on", () => {
  const pts: PlanPoint[] = [
    { lat: 0, lon: 0, altM: 3000, altEstimated: false, name: PRESENT_POSITION_ON_PLAN_NAME },
    { lat: 0.05, lon: 0, altM: 2500, altEstimated: true, name: "TOD (est.)" },
    { lat: 0.1, lon: 0, altM: null, altEstimated: true },
    { lat: 0.2, lon: 0, altM: null, altEstimated: true, name: "FIX" },
  ];
  assert.deepEqual(markVectorsUntil(pts, 0, { lat: 0.1, lon: 0 }).map((p) => !!p.vectors), [true, true, false, false]);
  assert.equal(pts[0].vectors, undefined, "input not mutated");
  assert.deepEqual(markVectorsUntil(pts, 0, { lat: 9, lon: 9 }).map((p) => !!p.vectors), [true, false, false, false]);
});

// ── the shared pure rule ────────────────────────────────────────────────────

/** a route due SOUTH along lon 0, a vertex every 2 nm for 60 nm */
const SOUTH: LatLon[] = Array.from({ length: 31 }, (_, i) => ({ lat: 1 - (i * 2) / 60, lon: 0 }));

test("chooseForwardJoin: on the route -> the next vertex ahead (unchanged rule)", () => {
  const pos = { lat: 1 - 5 / 60, lon: 0 }; // exactly on the route, 5 nm down it
  const fj = forwardJoinOnPlan(SOUTH, pos, 180)!;
  assert.equal(fj.join.rule, "ON_ROUTE");
  assert.equal(fj.join.index, fj.firstAhead);
  assert.equal(fj.firstAhead, 3, "the vertex at 6 nm");
});

test("chooseForwardJoin: off the route -> ahead, >= lead away, within ±70° of track, never behind (track known or not)", () => {
  const pos = { lat: 1 - 10 / 60, lon: 4 / 60 }; // 4 nm east, 10 nm down
  for (const trk of [200, null]) {
    const fj = forwardJoinOnPlan(SOUTH, pos, trk)!;
    assert.equal(fj.join.rule, "FORWARD");
    const j = SOUTH[fj.join.index];
    assert.ok(j.lat < pos.lat, "ahead (south) of the aircraft");
    assert.ok(haversineNm(pos, j) >= Math.max(SEAM_MIN_LEAD_NM, 8) - 1e-6);
    if (trk != null) assert.ok(angleDiffDeg(initialBearingDeg(pos, j), trk) <= 70);
  }
});

test("chooseForwardJoin fallbacks: destination when forward, else the next vertex ahead — never a vertex behind", () => {
  // flying NORTH (away from the route's direction) 3 nm east of it: nothing
  // ahead is within ±70° of track and the destination is behind the aircraft
  const pos = { lat: 1 - 10 / 60, lon: 3 / 60 };
  const fj = forwardJoinOnPlan(SOUTH, pos, 0)!;
  assert.equal(fj.join.rule, "FALLBACK_NEXT");
  assert.equal(fj.join.index, fj.firstAhead, "the next vertex AHEAD of the projection, not behind it");
  // short plan whose only candidate is too close -> destination
  const short = SOUTH.slice(0, 7); // 12 nm long
  const pos2 = { lat: 1 - 8 / 60, lon: 3 / 60 };
  const fj2 = forwardJoinOnPlan(short, pos2, 190)!;
  assert.equal(fj2.join.rule, "FALLBACK_DEST");
  assert.equal(fj2.join.index, short.length - 1);
  // past the end: nothing ahead
  assert.equal(forwardJoinOnPlan(short, { lat: 0, lon: 0 }, 180), null);
  // generic view form
  const cum = cumulativeNm(SOUTH);
  const view = { n: SOUTH.length, lat: (i: number) => SOUTH[i].lat, lon: (i: number) => SOUTH[i].lon, alongNm: (i: number) => cum[i] };
  assert.equal(chooseForwardJoin(view, SOUTH.length, 0, 5, pos, 180), null);
});

test("synthesized names and the vectoring predicate", () => {
  for (const n of [PRESENT_POSITION_NAME, PRESENT_POSITION_ON_PLAN_NAME, "TOC (est.)", "TOD (est.)", "peak (est.)"]) {
    assert.equal(isSynthesizedFixName(n), true, n);
  }
  for (const n of ["FIXCC", "SNS", "LAX", "", null]) assert.equal(isSynthesizedFixName(n), false, String(n));
  assert.equal(isTerminalVectoring(25, 5), true);
  assert.equal(isTerminalVectoring(25, VECTORING_MIN_XT_NM), false, "exactly at the threshold is still on the route");
  assert.equal(isTerminalVectoring(TERMINAL_AREA_NM + 1, 5), false, "outside the terminal area");
  assert.equal(isTerminalVectoring(null, 5), false);
});
