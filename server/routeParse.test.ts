// FULL ROUTE-STRING PARSER (deviation program P1, 2026-10-14):
// navAirways.parseFiledRoute + the CIFP SID/STAR hook + the plan's
// unresolved-segment flag. Fixtures are real NASR / CIFP data (the committed
// nasr_fixes / nasr_airways tables and the KAUS CIFP sample).
// Run: npx tsx --test server/routeParse.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseFiledRoute, parseLatLonToken, tokenizeRoute, type ParsedRoute } from "./navAirways";
import { lookupFix } from "./navFixes";
import { CifpIndex, CifpStore, cycleAt, procedureRouteFixes, matchProcedure } from "./cifp";
import { cifpProcedureHook, filedCandidate, DeviationTracker, _resetRouteParseCache, type AirportRef } from "./flightPlans";
import { parseSfdpsMessages, SwimPlanStore, parserShadowCounters, _resetSfdpsCountersForTests } from "./swimSfdps";
import { densifyPlan, estimateVerticalProfile, splitPlanAt, type PlanPoint } from "../shared/flightPlanGeometry";

const here = path.dirname(fileURLToPath(import.meta.url));
const idx = CifpIndex.build(fs.readFileSync(path.join(here, "__fixtures__", "cifp_kaus_sample.txt")));
idx.external = (id) => lookupFix(id);
const AP = (icao: string, lat: number, lon: number): AirportRef => ({ icao, iata: null, name: icao, lat, lon, elevM: 0 });
const KSFO = AP("KSFO", 37.619, -122.375), KDEN = AP("KDEN", 39.8617, -104.6732);
const KAUS = AP("KAUS", 30.1945, -97.6699), KDFW = AP("KDFW", 32.8968, -97.038);
const ctx = (o: AirportRef, d: AirportRef, extra = {}) =>
  ({ departure: o.icao, arrival: d.icao, origin: o, destination: d, ...extra });
const names = (r: ParsedRoute) => r.points.map((p) => p.name);

test("tokenize: speed/level groups, DCT/IFR markers, ETA suffixes and FIX/speed stripped; './.' kept", () => {
  const t = tokenizeRoute("N0450F350 DCT OAL/N0460F370 J80 MLF DCT KDEN/2041 IFR");
  assert.deepEqual(t.tokens, ["OAL", "J80", "MLF", "KDEN"]);
  assert.equal(t.stripped, 5);
  assert.deepEqual(tokenizeRoute("KORD./.FWA326067..BRYEN").tokens, ["KORD", "/", "FWA326067", "BRYEN"]);
  assert.deepEqual(tokenizeRoute(null), { tokens: [], stripped: 0 });
});

test("DCT-only: every named fix placed in order, kind direct, nothing unresolved", () => {
  const r = parseFiledRoute("KSFO DCT OAL DCT ILC DCT MLF DCT SAKES DCT KDEN", ctx(KSFO, KDEN));
  assert.deepEqual(names(r), ["OAL", "ILC", "MLF", "SAKES"]);
  assert.equal(r.kind, "direct");
  assert.deepEqual(r.unresolved, []);
  assert.ok(r.points.every((p) => p.source === "fix" && !p.gapBefore));
});

test("airway chain: FIX AWY FIX expands along the airway; AWY AWY joins at the common fix", () => {
  const r = parseFiledRoute("KSFO..OAL.J80.JNC..KDEN", ctx(KSFO, KDEN));
  assert.deepEqual(names(r), ["OAL", "ILC", "MLF", "SAKES", "JNC"], "J80's interior fixes, in order");
  assert.ok(r.points.slice(1).every((p) => p.source === "airway"));
  assert.equal(r.kind, "airway");
  assert.equal(r.airways, 1);
  // J80 then Q480 (no fix between them in the text): routed along J80 to the
  // first fix both airways share, then Q480 to the exit
  const chain = parseFiledRoute("SAKES J80 Q480 KYLOH", {});
  assert.ok(chain.airways >= 2, `both airways expanded (${chain.airways})`);
  assert.equal(names(chain)[0], "SAKES");
  assert.equal(names(chain)[names(chain).length - 1], "KYLOH");
  const z = names(chain).indexOf("ZANDR");
  assert.ok(z > 0 && names(chain)[z + 1] === "AIR", "J80 to ZANDR (the first fix on both), then Q480 onward");
  assert.deepEqual(chain.unresolved, []);
  // an airway NASR no longer lists stays an UNRESOLVED gap between real fixes
  const gone = parseFiledRoute("KSFO..OAL.J999.JNC..KDEN", ctx(KSFO, KDEN));
  assert.deepEqual(gone.unresolved, ["J999"]);
  assert.deepEqual(gone.points.find((p) => p.name === "JNC")?.gapBefore, ["J999"]);
});

test("SID + STAR with transitions expand from FAA CIFP (KAUS sample); without CIFP they are unresolved gaps", () => {
  const hook = cifpProcedureHook(idx);
  const text = "KAUS.AUS7.JCT..ACT.BLEWE5.KAUS";
  const r = parseFiledRoute(text, ctx(KAUS, KAUS, { procedure: hook }));
  assert.equal(r.sid?.id, "AUS7");
  assert.equal(r.sid?.transition, "JCT");
  assert.equal(r.sid?.expanded, true);
  assert.equal(r.star?.id, "BLEWE5");
  assert.equal(r.star?.transition, "ACT");
  // SID: common route (CWK) then the JCT enroute transition; STAR: the ACT
  // transition then the common route
  assert.deepEqual(names(r), ["CWK", "AMUSE", "JCT", "ACT", "BLEWE", "SEWZY", "CWK"]);
  assert.deepEqual(r.unresolved, []);
  const sidPts = r.points.filter((p) => p.source === "sid").map((p) => p.name);
  assert.deepEqual(sidPts, ["CWK", "AMUSE", "JCT"]);
  // the same text with no CIFP loaded: the procedures are gaps, the fixes stay
  const bare = parseFiledRoute(text, ctx(KAUS, KAUS));
  assert.deepEqual(bare.unresolved.slice(0, 2), ["AUS7", "BLEWE5"]);
  assert.equal(bare.sid?.expanded, false);
  // ...and on this KAUS→KAUS loop the two bare fixes have no corridor to
  // rule out same-named points elsewhere: they are not placed either
  assert.deepEqual(bare.points, []);
  assert.deepEqual(bare.unresolved.slice(2).sort(), ["ACT", "JCT"]);
  // the realistic case: DFW→AUS with the STAR expanded, nothing unresolved
  const dfw = parseFiledRoute("KDFW..ACT.BLEWE5.KAUS", ctx(KDFW, KAUS, { procedure: hook }));
  assert.deepEqual(names(dfw), ["ACT", "BLEWE", "SEWZY", "CWK"]);
  assert.equal(dfw.kind, "direct");
  // version mismatch is expanded but SAID (BLEWE4 filed, the cycle carries BLEWE5)
  const old = parseFiledRoute("KDFW..ACT.BLEWE4.KAUS", ctx(KDFW, KAUS, { procedure: hook }));
  assert.equal(old.star?.id, "BLEWE5");
  assert.equal(old.star?.versionMismatch, true);
  // procedureRouteFixes: runway transitions never appear (the runway is not filed)
  const aus7 = matchProcedure(idx.airport("KAUS"), "SID", "AUS7")!;
  assert.deepEqual(procedureRouteFixes(idx, idx.airport("KAUS")!, aus7.proc, "SAT").map((f) => f.name), ["CWK", "GARDS", "BETTI", "SAT"]);
});

test("oceanic lat/lon: every published form, ARINC 5-character included; NAT track unresolved without a definition", () => {
  assert.deepEqual(parseLatLonToken("3559N/09818W"), { lat: 35 + 59 / 60, lon: -(98 + 18 / 60) });
  assert.deepEqual(parseLatLonToken("4620N07805W"), { lat: 46 + 20 / 60, lon: -(78 + 5 / 60) });
  assert.deepEqual(parseLatLonToken("46N078W"), { lat: 46, lon: -78 });
  assert.deepEqual(parseLatLonToken("5030N"), { lat: 50, lon: -30 }, "ARINC: N = north/west");
  assert.deepEqual(parseLatLonToken("50N30"), { lat: 50, lon: -130 }, "letter in the middle = longitude + 100");
  assert.deepEqual(parseLatLonToken("4530E"), { lat: 45, lon: 30 });
  assert.deepEqual(parseLatLonToken("4530S"), { lat: -45, lon: 30 });
  const sec = parseLatLonToken("462030N0780515W")!;
  assert.ok(Math.abs(sec.lat - (46 + 20 / 60 + 30 / 3600)) < 1e-9 && Math.abs(sec.lon + (78 + 5 / 60 + 15 / 3600)) < 1e-9);
  for (const bad of ["9930N", "4699N07805W", "NOTLL", "4620X", "462N0780W"]) assert.equal(parseLatLonToken(bad), null, bad);
  const r = parseFiledRoute("KJFK..MERIT.HFD.PUT.BOS..NATW..5530N..5540N/05000W", { departure: "KJFK" });
  assert.deepEqual(names(r).slice(-2), ["5530N", "5540N/05000W"]);
  assert.deepEqual(r.points.find((p) => p.name === "5530N")?.gapBefore, ["NATW"], "the NAT track is an honest gap");
  const withTrack = parseFiledRoute("BOS..NATW..5540N/05000W", { oceanicTrack: (id) => (id === "NATW" ? [{ name: "5542N", lat: 55, lon: -42 }] : null) });
  assert.ok(names(withTrack).includes("5542N"));
});

test("malformed / hostile strings never throw: unknown tokens are kept as unresolved, nothing invented", () => {
  for (const t of ["@@@ ...// ZZZZ9 J999 1234 ???", "", "....", "J80", "./.", "N0450F350", "\u0000￿", "A".repeat(5000), "KSFO..J80..KDEN"]) {
    const r = parseFiledRoute(t, ctx(KSFO, KDEN));
    assert.ok(Array.isArray(r.points) && Array.isArray(r.unresolved), JSON.stringify(t.slice(0, 20)));
  }
  const r = parseFiledRoute("@@@ ...// ZZZZ9 J999 1234 ???", ctx(KSFO, KDEN));
  assert.deepEqual(r.points, []);
  assert.equal(r.kind, null);
  assert.ok(r.unresolved.includes("J999") && r.unresolved.includes("ZZZZ9"));
});

test("namesake guard + './.': a same-named fix far away is unresolved; an amended route marks the leg into its first point", () => {
  // ILC is in Nevada: on a BOS→ATL route it is a same-named point elsewhere
  const KBOS = AP("KBOS", 42.3656, -71.0096), KATL = AP("KATL", 33.6367, -84.4281);
  const far = parseFiledRoute("KBOS..ILC..KATL", ctx(KBOS, KATL));
  assert.deepEqual(far.points, []);
  assert.ok(far.unresolved.includes("ILC"));
  const am = parseFiledRoute("KSFO./.OAL..ILC..MLF..KDEN", ctx(KSFO, KDEN));
  assert.equal(am.amendedFromPresentPosition, true);
  assert.deepEqual(am.points[0].gapBefore, ["./."]);
});

test("regression (gate-1, 2026-10-14): an international plan amended from present position keeps its US route; a procedure-only route is placed", () => {
  // EGLL→KDFW: the first US fix is 4,000 nm from the departure. The departure
  // is no namesake anchor (and after "./." the leg's start is unknown).
  const EGLL = AP("EGLL", 51.4706, -0.4619);
  const r = parseFiledRoute("EGLL./.WWSHR.Q29.KLYNE..PXV..LIT..FEWWW.SEEVR4.KDFW", ctx(EGLL, KDFW));
  assert.ok(r.points.length >= 4, `placed ${names(r).join(" ")}`);
  assert.ok(names(r).includes("PXV") && names(r).includes("LIT"));
  assert.equal(r.amendedFromPresentPosition, true);
  assert.notEqual(r.kind, null);
  // KAUS→KDFW filing nothing but a SID and its transition (N253T's shape)
  const sidOnly = parseFiledRoute("KAUS.AUS7.JCT..KDFW", ctx(KAUS, KDFW, { procedure: cifpProcedureHook(idx) }));
  assert.deepEqual(names(sidOnly), ["CWK", "AMUSE", "JCT"]);
  assert.equal(sidOnly.kind, "direct");
});

test("plan: unresolved legs carry unresolvedNext through densify, the profile and the split; deviation is not judged along them", () => {
  _resetRouteParseCache();
  const store = new SwimPlanStore();
  const xml = `<MessageCollection><message><flight source="FH" timestamp="2026-10-14T12:00:00Z"><flightIdentification aircraftIdentification="UAL9"/><gufi>g9</gufi><departure departurePoint="KSFO"/><arrival arrivalPoint="KDEN"/><route nasRouteText="KSFO..OAL.J999.JNC..KDEN"/></flight></message></MessageCollection>`;
  for (const f of parseSfdpsMessages(xml)) store.upsert(f, 1_000);
  const plan = store.lookup("UAL9", 2_000)!;
  const cand = filedCandidate(plan, (id) => (id === "KSFO" ? KSFO : id === "KDEN" ? KDEN : null))!;
  assert.equal(cand.pathEstimated, false);
  const pts = cand.points;
  const oal = pts.findIndex((p) => p.name === "OAL"), jnc = pts.findIndex((p) => p.name === "JNC");
  assert.ok(oal >= 0 && jnc > oal);
  assert.ok(pts.slice(oal, jnc).every((p) => p.unresolvedNext), "every vertex of OAL→JNC (J999) is flagged");
  assert.ok(!pts[jnc].unresolvedNext && !pts[0].unresolvedNext, "resolved legs are not");
  assert.match(cand.honesty, /could not be placed \(J999\)/);
  assert.deepEqual(cand.filedRoute?.unresolved, ["J999"]);
  const prof = estimateVerticalProfile(pts, 0, 1600, 35000);
  const mid = prof.find((p) => p.name === "TOC (est.)" || p.name === "TOD (est.)");
  assert.ok(mid, "profile marks exist");
  const gc: PlanPoint[] = densifyPlan([{ lat: 0, lon: 0, altM: null, altEstimated: true, unresolvedNext: true }, { lat: 0, lon: 10, altM: null, altEstimated: true }], 50);
  assert.ok(gc.slice(0, -1).every((p) => p.unresolvedNext) && !gc[gc.length - 1].unresolvedNext);
  assert.equal(splitPlanAt(gc, { lat: 0.01, lon: 5.05 }).ahead[0].unresolvedNext, true);
  // 30 nm off the unresolved OAL→JNC leg for 5 en-route fixes: never OFF_PLAN
  const tracker = new DeviationTracker();
  const t = tracker.touch("abc123", "UAL9", "FILED_FAA", "k", prof, 0);
  for (let i = 0; i < 5; i++) tracker.observe(t, { t: (i + 1) * 120_000, lat: 38.9, lon: -112 + i * 0.3 });
  assert.notEqual(t.state, "OFF_PLAN");
});

test("ingest shadow: the parser runs beside the gate-1 chain and counts gains/losses without changing stored points", () => {
  _resetSfdpsCountersForTests();
  const mk = (cs: string, text: string) => `<MessageCollection><message><flight source="FH" timestamp="2026-10-14T12:00:00Z"><flightIdentification aircraftIdentification="${cs}"/><departure departurePoint="KSFO"/><arrival arrivalPoint="KDEN"/><route nasRouteText="${text}"/></flight></message></MessageCollection>`;
  const [a] = parseSfdpsMessages(mk("A1", "KSFO..OAL..ILC..MLF..KDEN"));
  const [b] = parseSfdpsMessages(mk("B1", "KSFO..OAL.J80.JNC.TBARR4.KDEN"));
  assert.equal(a.routeKind, "direct", "the stored (ingest) placement is the chain's, unchanged");
  assert.ok(b.routePoints.length > 0);
  assert.equal(parserShadowCounters.routes, 2);
  assert.equal(parserShadowCounters.lost, 0);
  assert.equal(parserShadowCounters.procedureTokens, 1);
});

test("CifpStore.peek never blocks: null before a load, starts one in the background, then serves it", async () => {
  let fetches = 0;
  const store = new CifpStore({
    dir: fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "cifp-peek-")),
    fetchImpl: (async () => { fetches++; return new Response("x", { status: 503 }); }) as unknown as typeof fetch,
    now: () => cycleAt(Date.UTC(2026, 9, 14)).effectiveMs + 1000,
  });
  assert.equal(store.peek(), null);
  await new Promise((r) => setTimeout(r, 20));
  assert.ok(fetches >= 1, "the background load ran");
  const before = fetches;
  assert.equal(store.peek(), null, "a failed load is not retried inside the backoff window");
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(fetches, before);
  store._setIndex(idx, cycleAt(Date.UTC(2026, 9, 14)));
  assert.equal(store.peek()?.idx, idx);
});
