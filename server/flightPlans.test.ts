import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import express from "express";
import type { AddressInfo } from "node:net";
import {
  parseRouteDbEntry, selectLeg, legPlausibleNm, RouteDbClient, ROUTESET_URL, routeFileUrl,
  scanCallsignFixes, completedTrips, pickHistoryTrip, HistoryIndex, thinPath,
  DeviationTracker, currentFlightFixes, appendFlightEvent, flightEventsDir,
  resolveFlightPlan, parsePlanQuery, registerFlightPlanRoutes, sanitizeCallsign,
  DEVIATION_OFF_NM, type PlanContext, type RouteDbRoute, type FlightEventRecord, type HistoryTrip,
  type FlightPlanResponse,
} from "./flightPlans";
import { SwimPlanStore, parseSfdpsMessages } from "./swimSfdps";
import { densifyPlan, haversineNm, type PlanPoint } from "../shared/flightPlanGeometry";

// ── fixtures: live adsb.lol route files (captured 2026-09-28) ───────────────
const UAL1 = {
  callsign: "UAL1", number: "1", airline_code: "UAL", airport_codes: "KSFO-WSSS", _airport_codes_iata: "SFO-SIN",
  _airports: [
    { name: "San Francisco International Airport", icao: "KSFO", iata: "SFO", location: "San Francisco", countryiso2: "US", lat: 37.618999, lon: -122.375, alt_feet: 13.0, alt_meters: 3.96 },
    { name: "Singapore Changi International Airport", icao: "WSSS", iata: "SIN", location: "Singapore", countryiso2: "SG", lat: 1.35019, lon: 103.994003, alt_feet: 22.0, alt_meters: 6.71 },
  ],
};
const SWA1234 = {
  callsign: "SWA1234", number: "1234", airline_code: "SWA", airport_codes: "KDAL-KPHX-KLAS", _airport_codes_iata: "DAL-PHX-LAS",
  _airports: [
    { name: "Dallas Love Field", icao: "KDAL", iata: "DAL", location: "Dallas", countryiso2: "US", lat: 32.847099, lon: -96.851799, alt_feet: 487.0, alt_meters: 148.44 },
    { name: "Phoenix Sky Harbor International Airport", icao: "KPHX", iata: "PHX", location: "Phoenix", countryiso2: "US", lat: 33.434299, lon: -112.012001, alt_feet: 1135.0, alt_meters: 345.95 },
    { name: "McCarran International Airport", icao: "KLAS", iata: "LAS", location: "Las Vegas", countryiso2: "US", lat: 36.080101, lon: -115.152, alt_feet: 2181.0, alt_meters: 664.77 },
  ],
};
const KSFO_KLAX = {
  callsign: "SKW5000", airport_codes: "KSFO-KLAX",
  _airports: [
    { name: "San Francisco International Airport", icao: "KSFO", iata: "SFO", lat: 37.618999, lon: -122.375, alt_meters: 3.96 },
    { name: "Los Angeles International Airport", icao: "KLAX", iata: "LAX", lat: 33.942501, lon: -118.407997, alt_meters: 38.1 },
  ],
};

const json = (x: unknown, status = 200) => new Response(JSON.stringify(x), { status, headers: { "content-type": "application/json" } });

// ── route DB parsing + leg selection ────────────────────────────────────────

test("parseRouteDbEntry: airports in code order with elevations; unknown/garbage -> null", () => {
  const r = parseRouteDbEntry(UAL1)!;
  assert.equal(r.callsign, "UAL1");
  assert.deepEqual(r.airports.map((a) => a.icao), ["KSFO", "WSSS"]);
  assert.equal(r.airports[0].elevM, 4);
  assert.equal(r.airports[0].iata, "SFO");
  assert.equal(r.plausible, null, "static route files carry no plausibility flag");
  assert.deepEqual(parseRouteDbEntry(SWA1234)!.airports.map((a) => a.icao), ["KDAL", "KPHX", "KLAS"]);
  assert.equal(parseRouteDbEntry({ callsign: "X1", airport_codes: "unknown", _airports: [] }), null);
  assert.equal(parseRouteDbEntry(null), null);
  assert.equal(parseRouteDbEntry({ callsign: "UAL1", airport_codes: "KSFO-WSSS", _airports: [UAL1._airports[0]] }), null);
  assert.equal(parseRouteDbEntry({ ...UAL1, plausible: 0 }, { lat: 1, lon: 2 })!.plausible, false);
});

test("selectLeg: multi-leg route picks the leg consistent with the aircraft's position", () => {
  const r = parseRouteDbEntry(SWA1234)!;
  const overTexas = selectLeg(r, { lat: 32.5, lon: -103 }, 270)!; // west Texas, westbound
  assert.equal(overTexas.legIndex, 0);
  assert.equal(overTexas.origin.icao, "KDAL");
  assert.equal(overTexas.destination.icao, "KPHX");
  assert.equal(overTexas.plausible, true);
  assert.equal(overTexas.positionChecked, true);
  const nwOfPhx = selectLeg(r, { lat: 34.8, lon: -113.6 }, 315)!;
  assert.equal(nwOfPhx.legIndex, 1);
  assert.equal(nwOfPhx.destination.icao, "KLAS");
});

test("selectLeg: a route that does not fit the position is rejected, never drawn wrong", () => {
  const r = parseRouteDbEntry(UAL1)!;
  const overLondon = selectLeg(r, { lat: 51.47, lon: -0.45 }, 90)!;
  assert.equal(overLondon.plausible, false);
  assert.match(overLondon.reason, /beyond the .* plausibility bound/);
  // bound scales with leg length (long-haul tracks sit far off the great circle)
  assert.equal(legPlausibleNm(100), 150);
  assert.equal(legPlausibleNm(7000), 1050);
});

test("selectLeg: no position — single leg accepted UNCHECKED, multi-leg unresolvable", () => {
  const one = selectLeg(parseRouteDbEntry(UAL1)!, null)!;
  assert.equal(one.positionChecked, false);
  assert.equal(one.plausible, true);
  assert.equal(selectLeg(parseRouteDbEntry(SWA1234)!, null), null);
});

test("selectLeg: upstream implausible flag vetoes only near where it was evaluated", () => {
  const vetoed: RouteDbRoute = { ...parseRouteDbEntry(UAL1)!, plausible: false, plausibleAt: { lat: 37.7, lon: -122.5 } };
  assert.equal(selectLeg(vetoed, { lat: 37.7, lon: -122.6 })!.plausible, false);
  assert.equal(selectLeg(vetoed, { lat: 30, lon: -150 })!.plausible, true, "far from the veto point our own check rules");
});

// ── route DB client (no network: injected fetch) ────────────────────────────

test("RouteDbClient: routeset empty 201 parks it; static route file answers; cache serves repeats", async () => {
  const calls: string[] = [];
  let now = 1_000_000;
  const client = new RouteDbClient({
    now: () => now, batchWindowMs: 1,
    fetchImpl: (async (url: string) => {
      calls.push(String(url));
      if (url === ROUTESET_URL) return new Response("", { status: 201 });
      if (url === routeFileUrl("UAL1")) return json(UAL1);
      return new Response("<html>404</html>", { status: 404 });
    }) as unknown as typeof fetch,
  });
  const a = await client.get("ual1", { lat: 37.6, lon: -122.3 });
  assert.deepEqual(a.route!.airports.map((x) => x.icao), ["KSFO", "WSSS"]);
  assert.equal(a.fetchedAt, now);
  assert.deepEqual(calls, [ROUTESET_URL, "https://vrs-standing-data.adsb.lol/routes/UA/UAL1.json"]);
  assert.ok(client.routesetDownUntil > now, "routeset parked after an empty answer");
  now += 3600_000; // 1h later: still within the 12h TTL
  const b = await client.get("UAL1", { lat: 40, lon: -140 });
  assert.equal(calls.length, 2, "served from cache");
  assert.equal(b.fetchedAt, 1_000_000);
  const miss = await client.get("ZZZ9999", null);
  assert.equal(miss.route, null);
  assert.equal(miss.error, null, "404 is an honest negative, not an error");
  const missAgain = await client.get("ZZZ9999", null);
  assert.equal(calls.length, 3, "negative answer cached too");
  assert.equal(missAgain.route, null);
  assert.equal(client.size, 2);
});

test("RouteDbClient: concurrent lookups batch into ONE routeset POST", async () => {
  const bodies: Array<{ planes: Array<Record<string, unknown>> }> = [];
  const client = new RouteDbClient({
    batchWindowMs: 5,
    fetchImpl: (async (url: string, init?: RequestInit) => {
      if (url === ROUTESET_URL) {
        bodies.push(JSON.parse(String(init?.body)));
        return json([{ ...UAL1, plausible: true }, { ...SWA1234, plausible: false }]);
      }
      throw new Error("static file must not be fetched when routeset answers");
    }) as unknown as typeof fetch,
  });
  const [a, b] = await Promise.all([
    client.get("UAL1", { lat: 37.6, lon: -122.3 }),
    client.get("SWA1234", { lat: 33.4, lon: -112 }),
  ]);
  assert.equal(bodies.length, 1);
  assert.deepEqual(bodies[0].planes.map((p) => String(p.callsign)).sort(), ["SWA1234", "UAL1"]);
  assert.deepEqual(Object.keys(bodies[0].planes[0]).sort(), ["callsign", "lat", "lng"]);
  assert.equal(a.route!.plausible, true);
  assert.equal(b.route!.plausible, false);
  assert.deepEqual(b.route!.plausibleAt, { lat: 33.4, lon: -112 });
});

test("RouteDbClient: provider failure -> backoff, last good answer served STALE, never a throw", async () => {
  let now = 5_000_000;
  let fail = false;
  let fetches = 0;
  const client = new RouteDbClient({
    now: () => now,
    fetchImpl: (async (url: string) => {
      fetches++;
      if (fail) throw new Error("ECONNRESET");
      if (url === ROUTESET_URL) return new Response("", { status: 201 });
      return json(UAL1);
    }) as unknown as typeof fetch,
  });
  await client.get("UAL1", null);
  now += 13 * 3600_000; // expired
  fail = true;
  const r = await client.get("UAL1", null);
  assert.equal(r.stale, true);
  assert.deepEqual(r.route!.airports.map((a) => a.icao), ["KSFO", "WSSS"]);
  assert.match(r.error!, /ECONNRESET/);
  assert.match(client.lastError!, /ECONNRESET/);
  const before = fetches;
  const again = await client.get("UAL1", null);
  assert.equal(fetches, before, "in backoff: no upstream call");
  assert.equal(again.stale, true);
  const none = await client.get("DAL1", null);
  assert.equal(none.route, null);
  assert.match(none.error!, /backoff/);
});

// ── history ─────────────────────────────────────────────────────────────────

const H0 = Date.parse("2026-09-27T15:00:00Z") / 1000;
/** a realistic complete KSFO->KLAX trip: ground, climb, cruise, descent, ground */
function sfoLaxTrip(t0: number, callsign = "SKW5000"): Array<Record<string, unknown>> {
  const pts: Array<[number, number, number | null, boolean]> = [
    [37.6190, -122.3750, null, true], [37.5000, -122.2000, 1500, false], [37.0000, -121.6000, 7000, false],
    [36.3000, -120.9000, 10500, false], [35.5000, -120.0000, 10500, false], [34.8000, -119.3000, 7000, false],
    [34.2000, -118.7000, 2000, false], [33.9425, -118.4080, null, true],
  ];
  return pts.map(([la, lo, al, g], i) => ({ t: t0 + i * 300, i: "a1b2c3", c: callsign, la, lo, ...(al != null ? { al } : {}), ...(g ? { g: true } : {}) }));
}

function writeArchive(dir: string, rows: Array<Record<string, unknown>>) {
  const byHour = new Map<string, string[]>();
  for (const r of rows) {
    const h = new Date((r.t as number) * 1000).toISOString().slice(0, 13).replace("T", "-");
    byHour.set(h, [...(byHour.get(h) || []), JSON.stringify(r)]);
  }
  fs.mkdirSync(path.join(dir, "aircraft"), { recursive: true });
  for (const [h, lines] of Array.from(byHour.entries())) fs.appendFileSync(path.join(dir, "aircraft", `${h}.jsonl`), lines.join("\n") + "\n");
}

test("scanCallsignFixes: exact callsign match (no prefix bleed), grouped by hex, bounded by lookback", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "fp-hist-"));
  try {
    writeArchive(dir, [
      ...sfoLaxTrip(H0),
      { t: H0 + 60, i: "ffffff", c: "SKW50001", la: 1, lo: 1 },            // prefix of another callsign
      { t: H0 + 90, i: "eeeeee", c: "SKW5000", la: 2, lo: 2, al: 9000 },   // same callsign, other tail
      { t: H0 - 5 * 86400, i: "a1b2c3", c: "SKW5000", la: 3, lo: 3 },      // outside lookback
    ]);
    const m = await scanCallsignFixes("SKW5000", dir, { nowSec: H0 + 7200 });
    assert.deepEqual(Array.from(m.keys()).sort(), ["a1b2c3", "eeeeee"]);
    assert.equal(m.get("a1b2c3")!.length, 8);
    assert.ok(m.get("a1b2c3")!.every((f, i, a) => i === 0 || f.t >= a[i - 1].t), "sorted by time");
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("completedTrips / pickHistoryTrip: only complete airport-to-airport flights; endpoints must match", () => {
  const fixes = sfoLaxTrip(H0).map((r) => ({ t: r.t as number, la: r.la as number, lo: r.lo as number, al: (r.al as number) ?? null, c: "SKW5000", g: !!r.g }));
  const trips = completedTrips(new Map([["a1b2c3", fixes]]), H0 + 7200);
  assert.equal(trips.length, 1);
  assert.equal(trips[0].trip.from_airport!.id, "KSFO");
  assert.equal(trips[0].trip.to_airport!.id, "KLAX");
  assert.ok(trips[0].points.every((p) => p.altEstimated), "last flight's altitudes are an estimate for this one");
  const ksfo = parseRouteDbEntry(KSFO_KLAX)!.airports;
  assert.ok(pickHistoryTrip(trips, { origin: ksfo[0], destination: ksfo[1] }));
  assert.equal(pickHistoryTrip(trips, { origin: ksfo[1], destination: ksfo[0] }), null, "reverse direction is not a match");
  assert.ok(pickHistoryTrip(trips, { origin: null, destination: null }));
  // an in-progress flight (ended < 10 min ago) is never "last time"
  assert.equal(completedTrips(new Map([["a1b2c3", fixes]]), fixes[fixes.length - 1].t + 60).length, 0);
  // a trip that never landed (signal lost airborne) is not complete
  assert.equal(completedTrips(new Map([["a1b2c3", fixes.slice(0, 5)]]), H0 + 7200).length, 0);
});

test("HistoryIndex: cached per callsign, budgeted (null when the scan is slow, cache fills after)", async () => {
  let scans = 0;
  let release: () => void = () => {};
  const gate = new Promise<void>((r) => { release = r; });
  const fixes = sfoLaxTrip(H0).map((r) => ({ t: r.t as number, la: r.la as number, lo: r.lo as number, al: (r.al as number) ?? null, c: "SKW5000", g: !!r.g }));
  const idx = new HistoryIndex({
    now: () => (H0 + 7200) * 1000, baseDir: () => "/nonexistent",
    scan: async () => { scans++; await gate; return new Map([["a1b2c3", fixes]]); },
  });
  assert.equal(await idx.trips("SKW5000", 20), null, "over budget -> null, request not blocked");
  release();
  await new Promise((r) => setTimeout(r, 20));
  const t = await idx.trips("SKW5000", 20);
  assert.equal(t!.length, 1);
  await idx.trips("SKW5000", 20);
  assert.equal(scans, 1, "one scan; the rest are cache hits");
  assert.equal(idx.size, 1);
});

test("thinPath keeps endpoints and bounds the count", () => {
  const pts = Array.from({ length: 1000 }, (_, i) => ({ lat: 0, lon: i / 100 }));
  const t = thinPath(pts, 50);
  assert.ok(t.length <= 51);
  assert.deepEqual(t[0], pts[0]);
  assert.deepEqual(t[t.length - 1], pts[999]);
});

// ── deviation state machine ─────────────────────────────────────────────────

const pp = (lat: number, lon: number, name?: string): PlanPoint => ({ lat, lon, altM: 10000, altEstimated: true, ...(name ? { name } : {}) });
const EAST = densifyPlan([pp(0, 0, "ORIG"), pp(0, 10, "DEST")], 25);

test("DeviationTracker: OFF_PLAN only after 3 consecutive en-route fixes > 8nm; hysteresis back < 4nm", () => {
  const recs: FlightEventRecord[] = [];
  const tr = new DeviationTracker({ append: (r) => recs.push(r) });
  const t = tr.touch("abc123", "TST1", "ROUTE_DB_PREDICTED", "k1", EAST, 0);
  let clock = 1_000_000;
  const fix = (lat: number, lon: number, dt = 60_000) => { clock += dt; tr.observe(t, { t: clock, lat, lon, altM: 10000, trk: 90 }); };
  fix(0.2, 3); fix(0.2, 3.1);                 // 12nm off twice
  assert.equal(t.state, "UNKNOWN");
  fix(0.05, 3.2);                             // 3nm: streak broken, judged on plan
  assert.equal(t.state, "ON_PLAN");
  fix(0.2, 3.3); fix(0.2, 3.4);
  assert.equal(t.state, "ON_PLAN");
  fix(0.2, 3.5);                              // third consecutive
  assert.equal(t.state, "OFF_PLAN");
  assert.equal(t.since, clock);
  assert.ok(t.crossTrackNm! > DEVIATION_OFF_NM);
  assert.deepEqual(t.events.map((e) => e.type), ["DEVIATION_START", "REPLANNED"]);
  assert.equal(t.replan!.points[0].name, "present position");
  const offSince = t.since;
  fix(0.0, 3.6, 10_000);                      // 10s later: too close to count
  assert.equal(t.state, "OFF_PLAN");
  fix(0.1, 3.7);                              // 6nm: inside the hysteresis band
  assert.equal(t.state, "OFF_PLAN");
  assert.equal(t.since, offSince);
  fix(0.03, 3.8);                             // 1.8nm: back on plan
  assert.equal(t.state, "ON_PLAN");
  assert.equal(t.replan, null);
  assert.deepEqual(t.events.map((e) => e.type), ["DEVIATION_START", "REPLANNED", "DEVIATION_END"]);
  assert.deepEqual(recs.map((r) => r.type), ["DEVIATION_START", "REPLANNED", "DEVIATION_END"]);
  assert.equal(recs[0].hex, "abc123");
  assert.equal(recs[0].cs, "TST1");
  assert.equal(recs[0].src, "ROUTE_DB_PREDICTED");
  assert.equal(recs[0].la, 0.2);
});

test("DeviationTracker: terminal-area fixes never start a deviation; older fixes never rewind state", () => {
  const tr = new DeviationTracker();
  const t = tr.touch("abc124", "TST2", "ROUTE_DB_PREDICTED", "k", EAST, 0);
  for (let i = 1; i <= 5; i++) tr.observe(t, { t: i * 60_000, lat: 0.4, lon: 0.2 }); // 24nm off, 26nm from ORIG
  assert.equal(t.state, "UNKNOWN");
  assert.equal(t.offCount, 0);
  tr.observe(t, { t: 10 * 60_000, lat: 0, lon: 5 });
  tr.observe(t, { t: 60_000, lat: 5, lon: 5 }); // stale fix
  assert.equal(t.state, "ON_PLAN");
  assert.equal(t.crossTrackNm, 0);
});

test("DeviationTracker: plan change resets the machine (events kept); idle eviction and cap", () => {
  const tr = new DeviationTracker({ max: 2, idleMs: 1000 });
  const t = tr.touch("aaaaaa", "A", "ROUTE_DB_PREDICTED", "k1", EAST, 0);
  t.state = "OFF_PLAN"; t.events.push({ t: 1, type: "DEVIATION_START", detail: "x" });
  const t2 = tr.touch("aaaaaa", "A", "FILED_FAA", "k2", EAST, 10);
  assert.equal(t2.state, "UNKNOWN");
  assert.equal(t2.events.length, 1);
  tr.touch("bbbbbb", "B", "ROUTE_DB_PREDICTED", "k", EAST, 20);
  tr.touch("cccccc", "C", "ROUTE_DB_PREDICTED", "k", EAST, 30);
  assert.equal(tr.size, 2, "capped");
  assert.equal(tr.get("aaaaaa"), undefined, "least recently requested evicted");
  tr.touch("dddddd", "D", "ROUTE_DB_PREDICTED", "k", EAST, 5000);
  assert.equal(tr.size, 1, "idle > 2h (here 1s) evicted");
});

test("currentFlightFixes keeps only fixes after the last > 45 min gap", () => {
  const f = [{ t: 0 }, { t: 60 }, { t: 60 + 46 * 60 }, { t: 60 + 47 * 60 }];
  assert.deepEqual(currentFlightFixes(f).map((x) => x.t), [2820, 2880]);
  assert.equal(currentFlightFixes([]).length, 0);
});

test("appendFlightEvent: append-only JSONL under <archive>/flight_events/<UTC day>.jsonl", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "fp-ev-"));
  try {
    const t = Date.parse("2026-09-28T23:59:00Z");
    const rec: FlightEventRecord = { t, hex: "abc123", cs: "TST1", type: "DEVIATION_START", detail: "d", src: "ROUTE_DB_PREDICTED", xt: 12.3, la: 1, lo: 2 };
    await appendFlightEvent(rec, dir);
    await appendFlightEvent({ ...rec, type: "DEVIATION_END" }, dir);
    assert.equal(flightEventsDir(dir), path.join(dir, "flight_events"));
    const lines = fs.readFileSync(path.join(dir, "flight_events", "2026-09-28.jsonl"), "utf8").trim().split("\n");
    assert.deepEqual(lines.map((l) => JSON.parse(l).type), ["DEVIATION_START", "DEVIATION_END"]);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

// ── the endpoint contract ───────────────────────────────────────────────────

const NOW = Date.parse("2026-09-28T18:00:00Z");

function ctxWith(opts: {
  routes?: Record<string, unknown>; fail?: boolean; historyFixes?: Map<string, any[]>; swim?: SwimPlanStore;
  recent?: Array<{ t: number; la: number; lo: number; al?: number | null }>; now?: () => number;
} = {}): PlanContext {
  const now = opts.now ?? (() => NOW);
  return {
    routeDb: new RouteDbClient({
      now, batchWindowMs: 1,
      fetchImpl: (async (url: string) => {
        if (opts.fail) throw new Error("upstream down");
        if (url === ROUTESET_URL) return new Response("", { status: 201 });
        const cs = String(url).split("/").pop()!.replace(".json", "");
        const r = opts.routes?.[cs];
        return r ? json(r) : new Response("nf", { status: 404 });
      }) as unknown as typeof fetch,
    }),
    history: new HistoryIndex({ now, scan: async () => opts.historyFixes ?? new Map() }),
    swim: opts.swim ?? new SwimPlanStore(),
    tracker: new DeviationTracker(),
    now,
    recentFixes: async () => opts.recent ?? [],
    historyBudgetMs: 500,
  };
}

const CONTRACT_KEYS = [
  "ageSec", "callsign", "cruiseAltEstimated", "cruiseAltFt", "destination", "deviation", "events", "fetchedAt",
  "hex", "honesty", "label", "origin", "originalPoints", "pathEstimated", "points", "source",
].sort();

function assertContract(r: FlightPlanResponse) {
  assert.deepEqual(Object.keys(r).sort(), CONTRACT_KEYS);
  assert.ok(["FILED_FAA", "ROUTE_DB_PREDICTED", "HISTORY_PREDICTED", "NONE"].includes(r.source));
  assert.equal(typeof r.label, "string");
  assert.equal(typeof r.honesty, "string");
  assert.ok(r.honesty.length > 20);
  assert.deepEqual(Object.keys(r.deviation).sort(), ["crossTrackNm", "since", "state"]);
  assert.ok(["ON_PLAN", "OFF_PLAN", "UNKNOWN"].includes(r.deviation.state));
  assert.ok(Array.isArray(r.points) && Array.isArray(r.events));
  for (const p of r.points) {
    assert.equal(typeof p.lat, "number");
    assert.equal(typeof p.lon, "number");
    assert.ok(p.altM === null || typeof p.altM === "number");
    assert.equal(typeof p.altEstimated, "boolean");
  }
  for (const a of [r.origin, r.destination]) {
    if (a) assert.deepEqual(Object.keys(a).sort(), ["elevM", "iata", "icao", "lat", "lon", "name"]);
  }
  assert.equal(typeof r.fetchedAt, "number");
  assert.equal(typeof r.ageSec, "number");
}

const q = (extra: Record<string, unknown>) => parsePlanQuery("a1b2c3", extra) as Exclude<ReturnType<typeof parsePlanQuery>, { error: string }>;

test("contract: ROUTE_DB_PREDICTED — labeled PREDICTED, great-circle, estimated profile anchored on the observed altitude", async () => {
  // (35.5, -120.1) sits on the KSFO-KLAX great circle
  const r = await resolveFlightPlan(q({ callsign: "SKW5000", lat: "35.5", lon: "-120.1", alt: "30000", trk: "135" }), ctxWith({ routes: { SKW5000: KSFO_KLAX } }));
  assertContract(r);
  assert.equal(r.source, "ROUTE_DB_PREDICTED");
  assert.match(r.label, /^PREDICTED — usual route for SKW5000 \(adsb\.lol route DB\), great-circle path$/);
  assert.match(r.honesty, /PREDICTED, not filed/);
  assert.equal(r.pathEstimated, true);
  assert.equal(r.origin!.icao, "KSFO");
  assert.equal(r.destination!.icao, "KLAX");
  assert.equal(r.cruiseAltEstimated, true);
  assert.equal(r.points[0].name, "SFO");
  assert.equal(r.points[r.points.length - 1].name, "LAX");
  const here = r.points.find((p) => p.name === "present position (on plan)")!;
  assert.ok(here, "plan carries the aircraft's projection");
  assert.equal(here.altEstimated, false, "observed altitude is not an estimate");
  assert.equal(here.altM, Math.round(30000 / 3.28084));
  assert.ok(r.points.filter((p) => p !== here && p.altM !== r.origin!.elevM && p.altM !== r.destination!.elevM).every((p) => p.altEstimated));
  assert.equal(r.originalPoints, null);
  assert.equal(r.deviation.state, "ON_PLAN");
  assert.ok(r.deviation.crossTrackNm! < 8);
});

test("contract: an estimated cruise is raised to the observed altitude (curtain never drops ahead of the aircraft)", async () => {
  const r = await resolveFlightPlan(q({ callsign: "SKW5000", lat: "35.5", lon: "-120.1", alt: "33000", trk: "135" }), ctxWith({ routes: { SKW5000: KSFO_KLAX } }));
  assert.equal(r.cruiseAltFt, 33000, "stage-length prior for ~300nm is 32000; observed 33000 wins");
  assert.equal(r.cruiseAltEstimated, true);
  const here = r.points.findIndex((p) => p.name === "present position (on plan)");
  assert.ok(r.points[here + 1].altM! >= r.points[here].altM! - 1, "no step down right after the aircraft");
});

test("contract: HISTORY_PREDICTED outranks the route DB when last time's trip matches its endpoints", async () => {
  const fixes = sfoLaxTrip(H0).map((x) => ({ t: x.t as number, la: x.la as number, lo: x.lo as number, al: (x.al as number) ?? null, c: "SKW5000", g: !!x.g }));
  const r = await resolveFlightPlan(q({ callsign: "SKW5000", lat: "36.2", lon: "-120.1" }),
    ctxWith({ routes: { SKW5000: KSFO_KLAX }, historyFixes: new Map([["a1b2c3", fixes]]) }));
  assertContract(r);
  assert.equal(r.source, "HISTORY_PREDICTED");
  assert.match(r.label, /^PREDICTED — the path SKW5000 actually flew on its last recorded KSFO→KLAX trip \(2026-09-27, our ADS-B archive\)$/);
  assert.equal(r.origin!.icao, "KSFO");
  assert.ok(r.points.length >= 8);
  assert.ok(r.points.some((p) => p.altM === 10500), "recorded altitudes carried");
});

test("contract: FILED_FAA outranks every prediction", async () => {
  const swim = new SwimPlanStore();
  const FH = `<m:MessageCollection xmlns:m="urn:x"><message><flight source="FH" timestamp="2026-09-28T17:00:00Z">
    <flightIdentification aircraftIdentification="SKW5000"/><gufi>g-1</gufi>
    <departure departurePoint="KSFO"/><arrival arrivalPoint="KLAX"/>
    <requestedAltitude><simple uom="FEET">24000</simple></requestedAltitude>
    <agreed><route nasRouteText="KSFO..SNS..KLAX"><expandedRoute>
      <routePoint><point fix="SNS"><location><pos>36.6636 -121.6031</pos></location></point></routePoint>
    </expandedRoute></route></agreed></flight></message></m:MessageCollection>`;
  for (const f of parseSfdpsMessages(FH)) swim.upsert(f, NOW - 60_000);
  const r = await resolveFlightPlan(q({ callsign: "SKW5000", lat: "36.2", lon: "-120.1" }), ctxWith({ routes: { SKW5000: KSFO_KLAX }, swim }));
  assertContract(r);
  assert.equal(r.source, "FILED_FAA");
  assert.equal(r.label, "FILED — FAA SWIM flight plan KSFO→KLAX");
  assert.equal(r.pathEstimated, false);
  assert.equal(r.cruiseAltFt, 24000);
  assert.equal(r.cruiseAltEstimated, false);
  assert.equal(r.origin!.icao, "KSFO", "filed aerodrome resolved through the OurAirports ident index");
  assert.ok(r.points.some((p) => p.name === "SNS"), "filed route point kept");
  assert.equal(r.ageSec, 60);
});

test("contract: implausible route -> NONE with the honest reason, nothing drawn", async () => {
  const r = await resolveFlightPlan(q({ callsign: "UAL1", lat: "51.47", lon: "-0.45" }), ctxWith({ routes: { UAL1 } }));
  assertContract(r);
  assert.equal(r.source, "NONE");
  assert.deepEqual(r.points, []);
  assert.match(r.honesty, /does not fit the aircraft's position/);
});

test("contract: provider down with nothing cached -> NONE (200), never a 500; no callsign -> NONE", async () => {
  const r = await resolveFlightPlan(q({ callsign: "UAL1", lat: "37.6", lon: "-122.3" }), ctxWith({ fail: true }));
  assertContract(r);
  assert.equal(r.source, "NONE");
  assert.match(r.honesty, /route lookup failed/);
  const nocs = await resolveFlightPlan(q({ lat: "37.6", lon: "-122.3" }), ctxWith());
  assert.equal(nocs.source, "NONE");
  assert.match(nocs.honesty, /no callsign/);
});

test("contract: OFF_PLAN — re-planned from the real position, original plan kept, events recorded", async () => {
  let now = NOW;
  const ctx = ctxWith({ routes: { SKW5000: KSFO_KLAX }, now: () => now });
  let r: FlightPlanResponse | null = null;
  // 3 fixes ~30nm west of the SFO-LAX great circle, a minute apart
  for (const [lat, lon] of [[36.4, -121.4], [36.2, -121.2], [36.0, -121.0]]) {
    now += 60_000;
    r = await resolveFlightPlan(q({ callsign: "SKW5000", lat: String(lat), lon: String(lon), alt: "20000", trk: "140" }), ctx);
  }
  if (!r) throw new Error("no response");
  assertContract(r);
  assert.equal(r.deviation.state, "OFF_PLAN");
  assert.ok((r.deviation.crossTrackNm ?? 0) > 8);
  assert.equal(r.points[0].name, "present position");
  assert.equal(r.points[0].altEstimated, false);
  assert.ok(Array.isArray(r.originalPoints) && r.originalPoints.length > 2);
  assert.equal(r.originalPoints![0].name, "SFO");
  assert.equal(r.points[r.points.length - 1].name, "LAX");
  assert.deepEqual(r.events.map((e) => e.type), ["DEVIATION_START", "REPLANNED"]);
  assert.match(r.honesty, /OFF its plan/);
  assert.ok(haversineNm(r.points[0], { lat: 36.0, lon: -121.0 }) < 0.01, "re-plan starts exactly at the aircraft");
});

test("contract: archive fixes seed the state machine on the first request", async () => {
  const t0 = Math.floor(NOW / 1000) - 600;
  const recent = [[36.4, -121.4], [36.2, -121.2], [36.0, -121.0]].map(([la, lo], i) => ({ t: t0 + i * 120, la, lo, al: 6000 }));
  const r = await resolveFlightPlan(q({ callsign: "SKW5000" }), ctxWith({ routes: { SKW5000: KSFO_KLAX }, recent }));
  assert.equal(r.deviation.state, "OFF_PLAN", "already off plan before anyone clicked");
});

test("contract: FILED text-only plan (great-circle stand-in) never reports OFF_PLAN — deviation stays UNKNOWN", async () => {
  const swim = new SwimPlanStore();
  const FH = `<m:MessageCollection xmlns:m="urn:x"><message><flight source="FH" timestamp="2026-09-28T17:00:00Z">
    <flightIdentification aircraftIdentification="SKW5000"/><gufi>g-2</gufi>
    <departure departurePoint="KSFO"/><arrival arrivalPoint="KLAX"/>
    <requestedAltitude><simple uom="FEET">24000</simple></requestedAltitude>
    <agreed><route nasRouteText="KSFO..SNS..KLAX"/></agreed></flight></message></m:MessageCollection>`;
  for (const f of parseSfdpsMessages(FH)) swim.upsert(f, NOW - 60_000);
  let now = NOW;
  const ctx = ctxWith({ swim, now: () => now });
  let r: FlightPlanResponse | null = null;
  for (const [lat, lon] of [[36.4, -121.4], [36.2, -121.2], [36.0, -121.0], [35.9, -120.9]]) {
    now += 60_000;
    r = await resolveFlightPlan(q({ callsign: "SKW5000", lat: String(lat), lon: String(lon), alt: "20000", trk: "140" }), ctx);
  }
  if (!r) throw new Error("no response");
  assert.equal(r.source, "FILED_FAA");
  assert.equal(r.pathEstimated, true);
  assert.equal(r.deviation.state, "UNKNOWN");
  assert.equal(r.deviation.crossTrackNm, null);
  assert.equal(r.originalPoints, null);
  assert.deepEqual(r.events, []);
  assert.match(r.honesty, /Deviation from the filed route is not assessed/);
});

test("parsePlanQuery: validation", () => {
  assert.deepEqual(parsePlanQuery("XYZ", {}), { error: "icao24 hex required (6 hex characters)" });
  assert.ok("error" in parsePlanQuery("a1b2c3", { lat: "91", lon: "0" }));
  assert.ok("error" in parsePlanQuery("a1b2c3", { lat: "10" }));
  assert.ok("error" in parsePlanQuery("a1b2c3", { lat: "1", lon: "2", trk: "400" }));
  assert.ok("error" in parsePlanQuery("a1b2c3", { alt: "abc" }));
  const ok = parsePlanQuery("A1B2C3", { callsign: " ual1 ", lat: "37.6", lon: "-122.3", alt: "35000", trk: "270", t: "1759000000" }) as any;
  assert.equal(ok.hex, "a1b2c3");
  assert.equal(ok.callsign, "UAL1");
  assert.equal(ok.altFt, 35000);
  assert.equal(ok.fixT, 1759000000000, "seconds normalized to ms");
  assert.equal((parsePlanQuery("a1b2c3", { altM: "1000" }) as any).altFt, 3280.84);
  assert.equal((parsePlanQuery("a1b2c3", { callsign: "<script>" }) as any).callsign, null);
  assert.equal(sanitizeCallsign("N123AB"), "N123AB");
  assert.equal(sanitizeCallsign("A"), null);
});

test("routes: /plan/:hex and /plan-status wired; bad hex is a 400; status exposes no secrets", async () => {
  const app = express();
  registerFlightPlanRoutes(app, ctxWith({ routes: { SKW5000: KSFO_KLAX } }));
  const server = app.listen(0);
  try {
    const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    const bad = await fetch(`${base}/api/data/aircraft/plan/zzz`);
    assert.equal(bad.status, 400);
    const ok = await fetch(`${base}/api/data/aircraft/plan/a1b2c3?callsign=SKW5000&lat=36.2&lon=-120.1&alt=30000&trk=135`);
    assert.equal(ok.status, 200);
    const body = (await ok.json()) as FlightPlanResponse;
    assertContract(body);
    assert.equal(body.source, "ROUTE_DB_PREDICTED");
    const st = await (await fetch(`${base}/api/data/aircraft/plan-status`)).json();
    for (const k of ["swimConfigured", "swimConnected", "solclientAvailable", "routeDbCacheSize", "lastRouteDbError", "trackedDeviations", "swimProducts", "swim"]) {
      assert.ok(k in st, `plan-status.${k}`);
    }
    assert.equal(st.routeDbCacheSize, 1);
    assert.equal(st.trackedDeviations, 1);
    assert.deepEqual(Object.keys(st.swimProducts), ["SFDPS", "TFMS", "STDDS", "NOTAM", "ITWS"]);
    assert.ok(!/PASSWORD=|"password"/i.test(JSON.stringify(st)));
  } finally { server.close(); }
});

test("history trips type sanity (compile-time contract for agent D consumers)", () => {
  const h: HistoryTrip | null = null;
  assert.equal(h, null);
});

test("contract: a filed plan whose only placed points are the airport endpoints stays estimated (no false filed-route claim)", async () => {
  const swim = new SwimPlanStore();
  const FH = `<m:MessageCollection xmlns:m="urn:x"><message><flight source="FH" timestamp="2026-09-28T17:00:00Z">
    <flightIdentification aircraftIdentification="SKW5001"/><gufi>g-2</gufi>
    <departure departurePoint="KSFO"/><arrival arrivalPoint="KLAX"/>
    <agreed><route nasRouteText="KSFO..SNS..KLAX"><expandedRoute>
      <routePoint><point><location><pos>37.619 -122.375</pos></location></point></routePoint>
      <routePoint><point><location><pos>33.9425 -118.4081</pos></location></point></routePoint>
    </expandedRoute></route></agreed></flight></message></m:MessageCollection>`;
  for (const f of parseSfdpsMessages(FH)) swim.upsert(f, NOW - 60_000);
  const r = await resolveFlightPlan(q({ callsign: "SKW5001", lat: "36.2", lon: "-120.1" }), ctxWith({ routes: {}, swim }));
  assert.equal(r.source, "FILED_FAA");
  assert.equal(r.pathEstimated, true);
});
