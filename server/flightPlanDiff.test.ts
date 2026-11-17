// shared/flightPlanDiff.ts: the classified amendment diff, the position at a
// time, and the grouping of the deviation state machine's verdicts into
// off-route episodes. Pure functions, no fixtures on disk.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  classifyPlanChange, positionAtTime, buildOffRouteEpisodes, isRouteRestatement, remainingRouteTokens,
  REROUTE_MIN_LATERAL_NM, POSITION_NEAREST_MAX_MS, POSITION_INTERP_MAX_GAP_MS, AMENDMENT_LINK_MS,
  type JudgedFix, type PlanShape, type TrackFix,
} from "../shared/flightPlanDiff";
import { densifyPlan, haversineNm, interpolateGC, TERMINAL_AREA_NM, type LatLon } from "../shared/flightPlanGeometry";

// FAA NASR positions (server/navFixes, read 2026-11-16)
const FIX: Record<string, LatLon> = {
  KSFO: { lat: 37.6198, lon: -122.3748 }, KLAX: { lat: 33.9425, lon: -118.408 }, KLAS: { lat: 36.0834, lon: -115.1518 },
  SNS: { lat: 36.66384, lon: -121.60318 }, AVE: { lat: 35.64698, lon: -119.97861 }, FIM: { lat: 34.35669, lon: -118.88129 },
  PXN: { lat: 36.71546, lon: -120.77868 }, EHF: { lat: 35.48455, lon: -119.09729 }, GMN: { lat: 34.80403, lon: -118.86136 },
  BSR: { lat: 36.1813, lon: -121.64211 }, HEC: { lat: 34.79702, lon: -116.46292 },
};
function shape(names: string[], extra: Partial<PlanShape> = {}): PlanShape {
  const fixes = names.map((n) => ({ ...FIX[n], name: n }));
  return {
    departure: names[0], arrival: names[names.length - 1], cruiseAltFt: 24000,
    routeText: names.join(".."), fixes,
    path: densifyPlan(fixes.map((f) => ({ ...f, altM: null, altEstimated: true })), 10),
    pathEstimated: false, amendedFromPresentPosition: false, ...extra,
  };
}
const V1 = shape(["KSFO", "SNS", "AVE", "FIM", "KLAX"]);

test("REROUTE: new fixes, dropped fixes, the first leg that differs and the lateral size of the change", () => {
  const c = classifyPlanChange(V1, shape(["KSFO", "SNS", "PXN", "EHF", "GMN", "KLAX"]));
  assert.deepEqual(c.kinds, ["REROUTE", "FIXES_ADDED", "FIXES_REMOVED"]);
  assert.deepEqual(c.fixesAdded, ["PXN", "EHF", "GMN"]);
  assert.deepEqual(c.fixesRemoved, ["AVE", "FIM"]);
  assert.deepEqual(c.fixesRemovedFromStart, [], "SNS is shared and comes first: nothing was dropped from the start");
  assert.equal(c.geometry.changed, true);
  const leg = c.geometry.firstDifferingLeg!;
  assert.equal(leg.index, 2);
  assert.equal(leg.from!.name, "SNS"); // the last point both versions share
  assert.equal(leg.to!.name, "PXN");   // where the new route goes
  assert.equal(leg.was!.name, "AVE");  // where the old one went
  // the two routes are furthest apart near PXN / AVE: 29.7 nm
  assert.ok(c.geometry.maxLateralNm! > 25 && c.geometry.maxLateralNm! < 35, `lateral ${c.geometry.maxLateralNm}`);
  assert.equal(c.geometry.lateralEstimated, false);
  assert.equal(c.destination, null);
  assert.equal(c.cruiseAltFt, null);
  assert.equal(c.routeTextChanged, true);
});

test("the two airports are never listed as fixes; a destination change is its own kind (and a reroute)", () => {
  const c = classifyPlanChange(V1, shape(["KSFO", "SNS", "AVE", "HEC", "KLAS"]));
  assert.ok(c.kinds.includes("DESTINATION_CHANGE") && c.kinds.includes("REROUTE"));
  assert.deepEqual(c.destination, { from: "KLAX", to: "KLAS" });
  assert.deepEqual(c.fixesAdded, ["HEC"]);
  assert.deepEqual(c.fixesRemoved, ["FIM"]);
  assert.ok(!c.kinds.includes("SHORTCUT"));
});

test("ALTITUDE_CHANGE alone: old and new cruise, no geometry change", () => {
  const c = classifyPlanChange(V1, { ...V1, cruiseAltFt: 28000 });
  assert.deepEqual(c.kinds, ["ALTITUDE_CHANGE"]);
  assert.deepEqual(c.cruiseAltFt, { from: 24000, to: 28000, deltaFt: 4000 });
  assert.equal(c.geometry.changed, false);
  assert.equal(c.geometry.firstDifferingLeg, null);
  assert.equal(c.geometry.maxLateralNm, null);
  // a cruise that was unknown before is not a change (the store's own rule)
  assert.deepEqual(classifyPlanChange({ ...V1, cruiseAltFt: null }, V1).kinds, []);
});

test("SHORTCUT: interior fixes dropped and none added", () => {
  const c = classifyPlanChange(V1, shape(["KSFO", "SNS", "FIM", "KLAX"]));
  assert.ok(c.kinds.includes("SHORTCUT") && c.kinds.includes("FIXES_REMOVED"));
  assert.deepEqual(c.fixesRemoved, ["AVE"]);
  assert.deepEqual(c.fixesAdded, []);
  // AVE sits ~9 nm off the direct SNS-FIM line: a real, small sideways move
  assert.ok(c.geometry.maxLateralNm! >= REROUTE_MIN_LATERAL_NM);
  assert.ok(c.kinds.includes("REROUTE"));
});

test("a fix added ON an unchanged leg is FIXES_ADDED, not a reroute", () => {
  const mid = interpolateGC(FIX.SNS, FIX.AVE, 0.5);
  const next = shape(["KSFO", "SNS", "AVE", "FIM", "KLAX"]);
  next.fixes.splice(2, 0, { ...mid, name: "MIDPT" });
  next.routeText = "KSFO..SNS..MIDPT..AVE..FIM..KLAX";
  next.path = densifyPlan(next.fixes.map((f) => ({ ...f, altM: null, altEstimated: true })), 10);
  const c = classifyPlanChange(V1, next);
  assert.deepEqual(c.kinds, ["FIXES_ADDED"]);
  assert.deepEqual(c.fixesAdded, ["MIDPT"]);
  assert.ok(c.geometry.maxLateralNm! < REROUTE_MIN_LATERAL_NM);
  assert.equal(c.geometry.changed, true, "the point list changed even though the line did not move");
});

// ── present-position routes ("./.") ─────────────────────────────────────────
// Read live 2026-11-17 on 111 flights: of 307 recorded changes, 22% restated
// the same remaining route and 33% dropped fixes from the start. Before that
// reading the test here pinned every "./." route as a DIRECT_TO with the
// dropped fixes "removed"; a fix dropped from the start is removed only when
// the aircraft had not reached it.

/** a present-position version: no leg out of the airport is filed, so the
 *  drawn path starts at the airport and jumps to the first element */
function fromPresent(kept: string[], routeText: string, positionToken?: { name: string } & LatLon): PlanShape {
  const s = shape(["KSFO", ...kept, "KLAX"], { amendedFromPresentPosition: true, routeText });
  if (positionToken) {
    s.fixes.splice(1, 0, positionToken);
    s.path = densifyPlan(s.fixes.map((f) => ({ ...f, altM: null, altEstimated: true })), 10);
  }
  return s;
}
const between = (a: string, b: string, f: number): LatLon => interpolateGC(FIX[a], FIX[b], f);

test("present position: the same route restated is not an amendment", () => {
  const c = classifyPlanChange(V1, fromPresent(["SNS", "AVE", "FIM"], "KSFO./.SNS..AVE..FIM..KLAX"));
  assert.deepEqual(c.kinds, [], "before: ROUTE_TEXT_ONLY or DIRECT_TO on every restatement");
  assert.equal(c.restatement, true);
  assert.equal(c.geometry.changed, false);
  assert.equal(c.directTo, null);
  assert.deepEqual(c.fromPresentPosition, { judged: true, positionBasis: null, fixesFlown: [], fixesSkipped: [], offRouteNm: null });
});

test("present position: a fix dropped from the start that lay BEHIND the aircraft was flown, not removed", () => {
  const next = fromPresent(["AVE", "FIM"], "KSFO./.AVE..FIM..KLAX");
  const c = classifyPlanChange(V1, next, { at: between("SNS", "AVE", 0.5) });
  assert.deepEqual(c.kinds, []);
  assert.equal(c.restatement, true);
  assert.deepEqual(c.fixesRemoved, []);
  assert.deepEqual(c.fixesRemovedFromStart, ["SNS"]);
  assert.deepEqual(c.fromPresentPosition, { judged: true, positionBasis: "ADSB_FIX", fixesFlown: ["SNS"], fixesSkipped: [], offRouteNm: 0 });
  assert.match(c.geometry.note!, /1 lay behind it \(flown\) and 0 ahead of it \(skipped\)/);
});

test("present position: a fix dropped from the start that lay AHEAD of the aircraft was skipped: a direct-to", () => {
  // the aircraft is a third of the way from KSFO to SNS, off to the side; cleared direct AVE
  const here = { lat: between("KSFO", "SNS", 0.33).lat, lon: between("KSFO", "SNS", 0.33).lon - 0.3 };
  const c = classifyPlanChange(V1, fromPresent(["AVE", "FIM"], "KSFO./.AVE..FIM..KLAX"), { at: here });
  assert.deepEqual(c.fixesRemoved, ["SNS"]);
  assert.deepEqual(c.fromPresentPosition?.fixesSkipped, ["SNS"]);
  assert.deepEqual(c.fromPresentPosition?.fixesFlown, []);
  assert.equal(c.fromPresentPosition?.positionBasis, "ADSB_FIX");
  assert.ok(c.kinds.includes("DIRECT_TO"));
  assert.ok(c.kinds.includes("FIXES_REMOVED"));
  assert.deepEqual(c.directTo, { fix: "AVE" });
  assert.ok(c.kinds.includes("REROUTE"), "SNS is miles from the line now flown");
  assert.ok(c.geometry.maxLateralNm! >= REROUTE_MIN_LATERAL_NM);
  assert.ok(!c.kinds.includes("SHORTCUT"), "a fix skipped at the start is not an interior shortcut");
  assert.equal(c.restatement, false);
});

test("present position: the FAA's own position token decides, and is never listed as a fix", () => {
  // "SNS300020": the aircraft is 20 nm short of SNS; the route now goes direct AVE
  const token = { name: "SNS300020", ...between("KSFO", "SNS", 0.7) };
  const next = fromPresent(["AVE", "FIM"], "KSFO./.SNS300020..AVE..FIM..KLAX", token);
  const c = classifyPlanChange(V1, next);
  assert.equal(c.fromPresentPosition?.positionBasis, "FAA_ROUTE_TEXT");
  assert.deepEqual(c.fromPresentPosition?.fixesSkipped, ["SNS"]);
  assert.deepEqual(c.fixesAdded, [], "the position token is where the aircraft was, not a fix ATC added");
  assert.ok(c.kinds.includes("DIRECT_TO"));
  // the token moving along the same remaining route is nothing
  const later = fromPresent(["AVE", "FIM"], "KSFO./.AVE310040..AVE..FIM..KLAX", { name: "AVE310040", ...between("SNS", "AVE", 0.5) });
  const again = classifyPlanChange(next, later);
  assert.deepEqual(again.kinds, []);
  assert.equal(again.restatement, true);
  assert.equal(again.geometry.maxLateralNm, 0, "before: the aircraft's own offset from the route read as a reroute");
});

test("present position: with no position known the dropped fixes are neither flagged nor called flown", () => {
  // CHANGED ASSERTIONS (2026-11-17). This case used to assert kinds includes DIRECT_TO,
  // directTo { fix: "AVE" } and fixesRemoved ["SNS"]. With no position there is no way to
  // tell a flown fix from a skipped one, and live data showed most are flown.
  const c = classifyPlanChange(V1, fromPresent(["AVE", "FIM"], "KSFO./.AVE..FIM..KLAX"));
  assert.ok(!c.kinds.includes("DIRECT_TO"));
  assert.equal(c.directTo, null);
  assert.deepEqual(c.fixesRemoved, []);
  assert.deepEqual(c.fixesRemovedFromStart, ["SNS"]);
  assert.deepEqual(c.fromPresentPosition, { judged: false, positionBasis: null, fixesFlown: [], fixesSkipped: [], offRouteNm: null });
  assert.ok(!c.kinds.includes("SHORTCUT"), "a fix dropped from the start is not an interior shortcut");
  assert.ok(!c.kinds.includes("REROUTE"), "the route still ahead lies on the old one");
  assert.equal(c.geometry.maxLateralNm, 0);
  assert.match(c.geometry.note!, /position is not known.*not determined/);
});

test("present position: a change further along the route is still an amendment", () => {
  // from the aircraft's position, and FIM replaced by GMN
  const c = classifyPlanChange(V1, fromPresent(["AVE", "GMN"], "KSFO./.AVE..GMN..KLAX"), { at: between("SNS", "AVE", 0.5) });
  assert.deepEqual(c.fixesAdded, ["GMN"]);
  assert.deepEqual(c.fixesRemoved, ["FIM"]);
  assert.deepEqual(c.fromPresentPosition?.fixesFlown, ["SNS"]);
  assert.ok(c.kinds.includes("REROUTE") && c.kinds.includes("DIRECT_TO"));
});

test("present position: the full text sent again after it names the flown part once more, and adds nothing", () => {
  const truncated = fromPresent(["AVE", "FIM"], "KSFO./.AVE..FIM..KLAX");
  const c = classifyPlanChange(truncated, V1);
  assert.deepEqual(c.kinds, [], "before: REROUTE + FIXES_ADDED (SNS) on every return of the full text");
  assert.deepEqual(c.fixesAdded, []);
  assert.equal(c.restatement, true);
  // an altitude change riding on a restatement is still an altitude change, and only that
  const alt = classifyPlanChange(truncated, { ...V1, cruiseAltFt: 28000 });
  assert.deepEqual(alt.kinds, ["ALTITUDE_CHANGE"]);
});

test("route text: what is still ahead, and when one text only restates another (live strings, 2026-11-17)", () => {
  assert.deepEqual(remainingRouteTokens("KCVG./.APE227059..AHTIY..PSB..PONCT.JFUND2.KBOS/0314"),
    { tokens: ["AHTIY", "PSB", "PONCT", "JFUND2", "KBOS"], fromPresentPosition: true, positionToken: "APE227059" });
  assert.deepEqual(remainingRouteTokens("KORD..MOBLE..ADIME..GERBS.Q186.WEVEL..ETG.MIP4.KLGA/0258").tokens,
    ["MOBLE", "ADIME", "GERBS", "Q186", "WEVEL", "ETG", "MIP4", "KLGA"]);
  const R = isRouteRestatement;
  // the position token moved
  assert.equal(R("KCVG./.AIR254047..PONCT.JFUND2.KBOS/0314", "KCVG./.AIR051011..PONCT.JFUND2.KBOS/0314"), true);
  // "./." put in front of a route held in full
  assert.equal(R("CYUL..FAWNS..BUGSY..SYR..MGGT/0557", "CYUL./.FAWNS..BUGSY..SYR..MGGT/0557"), true);
  // the older, fuller text after a present-position route
  assert.equal(R("KORD./.ADIME..GERBS.Q186.WEVEL..ETG.MIP4.KLGA/0258", "KORD..MOBLE..ADIME..GERBS.Q186.WEVEL..ETG.MIP4.KLGA/0258"), true);
  // fixes dropped: NOT a restatement (flown or skipped is for the position to decide)
  assert.equal(R("KORD..MOBLE..ADIME..GERBS.Q186.WEVEL..ETG.MIP4.KLGA/0258", "KORD./.ADIME..GERBS.Q186.WEVEL..ETG.MIP4.KLGA/0258"), false);
  assert.equal(R("KCVG./.APE227059..AHTIY..PSB..PONCT.JFUND2.KBOS/0314", "KCVG./.AIR254047..PONCT.JFUND2.KBOS/0314"), false);
  // a different route ahead, two full texts, identical texts, missing text
  assert.equal(R("KJFK./.EMI064061..SCHUL.OZZZI2.KATL/0333", "KJFK..RBV.Q430.BYRDD.J48.MOL..FLASK.OZZZI2.KATL/0333"), false);
  assert.equal(R("KSFO..SNS..AVE..KLAX", "KSFO..PXN..AVE..KLAX"), false);
  assert.equal(R("KSFO./.AVE..KLAX", "KSFO./.AVE..KLAX"), false);
  assert.equal(R(null, "KSFO./.AVE..KLAX"), false);
});

test("ROUTE_TEXT_ONLY: the text changed, the geometry we placed did not", () => {
  const c = classifyPlanChange(V1, { ...V1, routeText: "KSFO.SSTIK5.SNS..AVE..FIM..KLAX" });
  assert.deepEqual(c.kinds, ["ROUTE_TEXT_ONLY"]);
  assert.equal(c.geometry.changed, false);
  assert.equal(c.geometry.maxLateralNm, 0);
});

test("a great-circle stand-in on either side flags the distance as an estimate; no path = not measured, said so", () => {
  const est = shape(["KSFO", "KLAX"], { pathEstimated: true });
  const c = classifyPlanChange(est, V1);
  assert.equal(c.geometry.lateralEstimated, true);
  assert.match(c.geometry.note!, /great-circle stand-in/);
  const none = classifyPlanChange(V1, { ...shape(["KSFO", "SNS", "PXN", "KLAX"]), path: [] });
  assert.equal(none.geometry.maxLateralNm, null);
  assert.ok(!none.kinds.includes("REROUTE"), "an unmeasured change is not called a reroute");
  assert.match(none.geometry.note!, /not measured/);
  assert.ok(none.kinds.includes("FIXES_ADDED"));
});

test("unnamed route points: a moved unnamed point is still a geometry change", () => {
  const a = shape(["KSFO", "SNS", "AVE", "KLAX"]);
  const b = shape(["KSFO", "SNS", "AVE", "KLAX"]);
  delete a.fixes[2].name; delete b.fixes[2].name;
  b.fixes[2] = { ...FIX.EHF };
  b.path = densifyPlan(b.fixes.map((f) => ({ ...f, altM: null, altEstimated: true })), 10);
  b.routeText = "KSFO..SNS..3529N/11906W..KLAX";
  const c = classifyPlanChange(a, b);
  assert.equal(c.geometry.changed, true);
  assert.ok(c.kinds.includes("REROUTE"));
  assert.deepEqual(c.fixesAdded, []);
});

// ── position at a time ──────────────────────────────────────────────────────
const T0 = Date.parse("2026-11-16T17:00:00Z");
const TRACK: TrackFix[] = [0, 60, 120, 180].map((s, i) => ({
  tMs: T0 + s * 1000, ...interpolateGC(FIX.SNS, FIX.AVE, i / 3), altFt: 20000 + i * 1000,
}));

test("positionAtTime: exact fix", () => {
  const r = positionAtTime(TRACK, T0 + 60_000);
  assert.equal(r.reason, null);
  assert.equal(r.position!.method, "EXACT");
  assert.equal(r.position!.interpolated, false);
  assert.equal(r.position!.fixTimeMs, T0 + 60_000);
  assert.equal(r.position!.gapSec, 0);
  assert.equal(r.position!.lat, TRACK[1].lat);
  assert.equal(r.position!.altFt, 21000);
});

test("positionAtTime: the nearest fix within the tolerance is used as it is, with its own time and the gap", () => {
  const r = positionAtTime(TRACK, T0 + 60_000 + 20_000);
  assert.equal(r.position!.method, "NEAREST");
  assert.equal(r.position!.interpolated, false);
  assert.equal(r.position!.fixTimeMs, T0 + 60_000);
  assert.equal(r.position!.gapSec, 20);
  assert.deepEqual([r.position!.lat, r.position!.lon], [TRACK[1].lat, TRACK[1].lon], "a real fix, not a blend");
  // the later fix wins when it is the closer one
  assert.equal(positionAtTime(TRACK, T0 + 60_000 + 45_000).position!.fixTimeMs, T0 + 120_000);
  assert.equal(POSITION_NEAREST_MAX_MS, 30_000);
});

test("positionAtTime: between two fixes further than the tolerance the point is interpolated and FLAGGED", () => {
  const sparse: TrackFix[] = [TRACK[0], { ...TRACK[3], tMs: T0 + 200_000 }];
  const r = positionAtTime(sparse, T0 + 100_000);
  assert.equal(r.reason, null);
  assert.equal(r.position!.method, "INTERPOLATED");
  assert.equal(r.position!.interpolated, true);
  assert.deepEqual(r.position!.between, { beforeMs: T0, afterMs: T0 + 200_000, spanSec: 200 });
  assert.equal(r.position!.gapSec, 100);
  const mid = interpolateGC(sparse[0], sparse[1], 0.5);
  assert.ok(haversineNm(r.position!, mid) < 0.01);
  assert.equal(r.position!.altFt, 21500);
});

test("positionAtTime: no fix close enough = null, with the reason in words", () => {
  const far: TrackFix[] = [TRACK[0], { ...TRACK[3], tMs: T0 + POSITION_INTERP_MAX_GAP_MS + 60_000 }];
  const gap = positionAtTime(far, T0 + 120_000);
  assert.equal(gap.position, null);
  assert.match(gap.reason!, /nearest ADS-B fix is 120 s from that time \(tolerance 30 s\).*360 s apart \(interpolation limit 300 s\)/);
  assert.match(positionAtTime(TRACK, T0 - 600_000).reason!, /track starts after that time/);
  assert.match(positionAtTime(TRACK, T0 + 900_000).reason!, /track ends before that time/);
  assert.deepEqual(positionAtTime([], T0), { position: null, reason: "no ADS-B fixes are held for this aircraft" });
});

// ── episodes ────────────────────────────────────────────────────────────────
const jf = (i: number, state: JudgedFix["state"], offCount: number, xt: number, extra: Partial<JudgedFix> = {}): JudgedFix => ({
  tMs: T0 + i * 60_000, lat: 36 - i * 0.05, lon: -121 + i * 0.05, version: 1, state, offCount,
  crossTrackNm: xt, distToDestNm: 200, ...extra,
});

test("episode with no amendment: start is the FIRST off-route fix, detection the third; cause not determined", () => {
  const fixes = [
    jf(0, "ON_PLAN", 0, 0.4), jf(1, "ON_PLAN", 0, 0.6),
    jf(2, "ON_PLAN", 1, 9.5), jf(3, "ON_PLAN", 2, 16.4), jf(4, "OFF_PLAN", 3, 15.2), jf(5, "OFF_PLAN", 4, 14.1),
    jf(6, "OFF_PLAN", 0, 6), // inside the hysteresis band: still off
    jf(7, "ON_PLAN", 0, 2.1), jf(8, "ON_PLAN", 0, 0.3),
  ];
  const eps = buildOffRouteEpisodes(fixes, [], T0 + 9 * 60_000);
  assert.equal(eps.length, 1);
  const e = eps[0];
  assert.equal(e.startMs, T0 + 2 * 60_000);
  assert.deepEqual(e.start, { lat: fixes[2].lat, lon: fixes[2].lon });
  assert.equal(e.detectedAtMs, T0 + 4 * 60_000);
  assert.equal(e.endMs, T0 + 7 * 60_000);
  assert.deepEqual(e.end, { lat: fixes[7].lat, lon: fixes[7].lon });
  assert.equal(e.endReason, "REJOINED");
  assert.equal(e.maxCrossTrackNm, 16.4, "the largest distance, including the fixes before the state machine called it");
  assert.equal(e.fixes, 5, "fixes 2..6, counted from the first one off the route");
  assert.equal(e.classification, "NO_AMENDMENT_RECEIVED");
  assert.match(e.cause, /^cause not determined/);
  assert.equal(e.linkedAmendmentVersion, null);
  assert.equal(e.againstVersion, 1);
  assert.ok(!/weather/.test(e.classification), "no cause is invented");
});

test("episode ended by an amendment: linked to it, with the lag; the next version starts clean", () => {
  const fixes = [
    jf(0, "ON_PLAN", 0, 0.5), jf(1, "ON_PLAN", 1, 9), jf(2, "ON_PLAN", 2, 14), jf(3, "OFF_PLAN", 3, 19), jf(4, "OFF_PLAN", 4, 23),
    jf(5, "UNKNOWN", 0, 0.2, { version: 2 }), jf(6, "ON_PLAN", 0, 0.3, { version: 2 }),
  ];
  const eps = buildOffRouteEpisodes(fixes, [{ version: 2, atMs: T0 + 4.5 * 60_000 }], T0 + 7 * 60_000);
  assert.equal(eps.length, 1);
  const e = eps[0];
  assert.equal(e.endReason, "AMENDED");
  assert.equal(e.endMs, T0 + 4 * 60_000, "ends at the last off-route fix before the amendment");
  assert.equal(e.classification, "AMENDMENT_FOLLOWED");
  assert.equal(e.linkedAmendmentVersion, 2);
  assert.equal(e.amendmentLagSec, 210); // 4.5 min after fix 1 (the first off-route fix)
  assert.match(e.cause, /^reroute: an amendment to the filed route took effect 210 s after/);
  assert.equal(e.maxCrossTrackNm, 23);
});

test("an amendment shortly after a rejoin is linked; one far later is reported but does not explain the start", () => {
  const fixes = [
    jf(0, "ON_PLAN", 1, 9), jf(1, "ON_PLAN", 2, 12), jf(2, "OFF_PLAN", 3, 12), jf(3, "ON_PLAN", 0, 1),
    jf(4, "ON_PLAN", 0, 1), jf(40, "ON_PLAN", 0, 1),
  ];
  const soon = buildOffRouteEpisodes(fixes, [{ version: 2, atMs: T0 + 3 * 60_000 + AMENDMENT_LINK_MS - 1 }], T0 + 41 * 60_000)[0];
  assert.equal(soon.classification, "AMENDMENT_FOLLOWED");
  const late = buildOffRouteEpisodes(fixes, [{ version: 2, atMs: T0 + 3 * 60_000 + AMENDMENT_LINK_MS + 1 }], T0 + 41 * 60_000)[0];
  assert.equal(late.classification, "NO_AMENDMENT_RECEIVED");
  assert.equal(late.linkedAmendmentVersion, null);
  // an episode that is still open when a much later amendment replaces the route
  const long = [jf(0, "ON_PLAN", 1, 9), jf(1, "ON_PLAN", 2, 12), jf(2, "OFF_PLAN", 3, 12)];
  for (let i = 3; i < 30; i++) long.push(jf(i, "OFF_PLAN", i + 1, 20));
  long.push(jf(31, "ON_PLAN", 0, 0.5, { version: 2 }));
  const e = buildOffRouteEpisodes(long, [{ version: 2, atMs: T0 + 30.5 * 60_000 }], T0 + 32 * 60_000)[0];
  assert.equal(e.endReason, "AMENDED");
  assert.equal(e.linkedAmendmentVersion, 2);
  assert.equal(e.amendmentLagSec, 1830);
  assert.equal(e.classification, "AMENDMENT_FOLLOWED", "the amendment came within the link window of the last off-route fix");
});

test("open episodes: ONGOING while the track is fresh, TRACK_ENDED when it is not; no end time is made up", () => {
  const fixes = [jf(0, "ON_PLAN", 1, 9), jf(1, "ON_PLAN", 2, 12), jf(2, "OFF_PLAN", 3, 16), jf(3, "OFF_PLAN", 4, 18)];
  const live = buildOffRouteEpisodes(fixes, [], T0 + 4 * 60_000)[0];
  assert.equal(live.endReason, "ONGOING");
  assert.equal(live.endMs, null);
  assert.equal(live.end, null);
  assert.equal(live.lastOffMs, T0 + 3 * 60_000);
  assert.equal(buildOffRouteEpisodes(fixes, [], T0 + 60 * 60_000)[0].endReason, "TRACK_ENDED");
});

test("terminal area: an en-route episode stops being judged there, and vectoring near the destination is its own class", () => {
  const fixes = [
    jf(0, "ON_PLAN", 1, 9, { distToDestNm: 80 }), jf(1, "ON_PLAN", 2, 12, { distToDestNm: 70 }),
    jf(2, "OFF_PLAN", 3, 14, { distToDestNm: 60 }), jf(3, "OFF_PLAN", 0, 11, { distToDestNm: TERMINAL_AREA_NM - 5 }),
    jf(4, "OFF_PLAN", 0, 6, { distToDestNm: 20 }), jf(5, "OFF_PLAN", 0, 1.2, { distToDestNm: 8 }),
  ];
  const eps = buildOffRouteEpisodes(fixes, [], T0 + 6 * 60_000);
  const enroute = eps.find((e) => e.classification === "NO_AMENDMENT_RECEIVED")!;
  assert.equal(enroute.endReason, "ENTERED_TERMINAL_AREA");
  assert.equal(enroute.endMs, T0 + 3 * 60_000);
  const vect = eps.find((e) => e.classification === "TERMINAL_VECTORING")!;
  assert.equal(vect.startMs, T0 + 3 * 60_000);
  assert.equal(vect.endMs, T0 + 5 * 60_000, "back within the vectoring threshold of the route");
  assert.equal(vect.maxCrossTrackNm, 11);
  assert.match(vect.cause, /terminal area/);
  assert.equal(vect.linkedAmendmentVersion, null);
  assert.equal(eps.length, 2);
});

test("no off-route fixes = no episodes; an empty track = no episodes", () => {
  assert.deepEqual(buildOffRouteEpisodes([jf(0, "ON_PLAN", 0, 1), jf(1, "ON_PLAN", 0, 2)], [], T0), []);
  assert.deepEqual(buildOffRouteEpisodes([], [{ version: 2, atMs: T0 }], T0), []);
});
