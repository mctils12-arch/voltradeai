import { test } from "node:test";
import assert from "node:assert/strict";
import { airwaySegment, expandRouteText, placeDirectFixes, classifyUnplacedRoute, unresolvedRouteTokens, fewFixesProfile, resolveRadialToken, placeDirectFixesWithRadials } from "./navAirways";
import { lookupFix } from "./navFixes";
import { haversineNm, initialBearingDeg } from "../shared/flightPlanGeometry";
import { parseSfdpsMessages, routeShapeCounters, unplacedReasonCounters, routeShapeSamples, _resetSfdpsCountersForTests } from "./swimSfdps";

// J80 in NASR cycle 2026-09-03 runs OAL ILC MLF SAKES JNC ... (real data)
test("airwaySegment returns the fixes between two idents in travel order, either direction", () => {
  assert.deepEqual(airwaySegment("J80", "ILC", "JNC"), ["ILC", "MLF", "SAKES", "JNC"]);
  assert.deepEqual(airwaySegment("J80", "JNC", "ILC"), ["JNC", "SAKES", "MLF", "ILC"]);
});

test("airwaySegment refuses unknown airways, off-airway fixes and identical ends", () => {
  assert.equal(airwaySegment("J99999", "ILC", "JNC"), null);
  assert.equal(airwaySegment("J80", "ILC", "QZQZQ"), null);
  assert.equal(airwaySegment("J80", "ILC", "ILC"), null);
});

test("expandRouteText places FIX AIRWAY FIX interiors and drops what it cannot bound", () => {
  const pts = expandRouteText("KBOS..ILC.J80.JNC..KATL");
  assert.deepEqual(pts.map((p) => p.name), ["ILC", "MLF", "SAKES", "JNC"]);
  assert.ok(pts.every((p) => Number.isFinite(p.lat) && Number.isFinite(p.lon)));
  // unknown airway (J99999) or unbounded fix: nothing is expanded, so nothing is invented
  assert.deepEqual(expandRouteText("ILC.J99999.JNC"), []);
  assert.deepEqual(expandRouteText("ILC.J80.QZQZQ"), []);
  // but a bad airway elsewhere in the route does not poison a good expansion
  assert.deepEqual(expandRouteText("ILC.J80.JNC..QZQZQ.J99999.WHITE").map((p) => p.name), ["ILC", "MLF", "SAKES", "JNC", "WHITE"]);
  assert.deepEqual(expandRouteText(null), []);
  assert.deepEqual(expandRouteText("KBOS..KATL"), []);
});

test("a message with route text only (no expandedRoute) gets airway-expanded route points", () => {
  _resetSfdpsCountersForTests();
  const xml = `<MessageCollection><message><flight source="FH" timestamp="2026-09-30T12:00:00Z">
    <flightIdentification aircraftIdentification="DAL1"/>
    <route nasRouteText="KBOS..ILC.J80.JNC..KATL"/></flight></message></MessageCollection>`;
  const [f] = parseSfdpsMessages(xml);
  assert.deepEqual(f.routePoints.map((p) => p.name), ["ILC", "MLF", "SAKES", "JNC"]);
  assert.equal(routeShapeCounters.airwayExpanded, 1);
});

test("placeDirectFixes places pure direct-fix routes only, never across an airway or an implausible leg", () => {
  // real NASR fixes ILC, MLF, SAKES (on J80) used as direct-to fixes
  assert.deepEqual(placeDirectFixes("KBOS..ILC..MLF..SAKES..KATL").map((p) => p.name), ["ILC", "MLF", "SAKES"]);
  assert.deepEqual(placeDirectFixes("ILC..MLF"), []); // <3 fixes
  assert.deepEqual(placeDirectFixes("ILC..MLF..SAKES.J80.JNC"), []); // airway present: refuse, no chord across it
  assert.deepEqual(placeDirectFixes("ILC..MLF..BOS..SAKES"), []); // MLF..BOS >1200nm: ident collision guard
  assert.deepEqual(placeDirectFixes(null), []);
});

test("route-text-only message with direct fixes counts directFixPlaced", () => {
  _resetSfdpsCountersForTests();
  const xml = `<MessageCollection><message><flight source="FH" timestamp="2026-09-30T12:00:00Z">
    <flightIdentification aircraftIdentification="DAL2"/>
    <route nasRouteText="KBOS..ILC..MLF..SAKES..KATL"/></flight></message></MessageCollection>`;
  const [f] = parseSfdpsMessages(xml);
  assert.deepEqual(f.routePoints.map((p) => p.name), ["ILC", "MLF", "SAKES"]);
  assert.equal(routeShapeCounters.directFixPlaced, 1);
  assert.equal(routeShapeCounters.airwayExpanded, 0);
});

test("classifyUnplacedRoute names why route text stays unplaced (report-only diagnostic)", () => {
  assert.equal(classifyUnplacedRoute(null), "noText");
  assert.equal(classifyUnplacedRoute("KBOS..QZQZQ..KATL"), "noFixResolved");
  assert.equal(classifyUnplacedRoute("KBOS..ILC..MLF..KATL"), "fewFixes");
  assert.equal(classifyUnplacedRoute("ILC.J80.QZQZQ"), "airwayUnbounded");
  assert.equal(classifyUnplacedRoute("ILC..MLF..BOS..SAKES"), "legTooLong");
  assert.equal(classifyUnplacedRoute("ILC..MLF..SAKES"), "other"); // placeable => classifier says other; callers only ask when unplaced
});

test("unplaced routes increment unplacedReasonCounters, placed ones do not", () => {
  _resetSfdpsCountersForTests();
  const mk = (t: string) => `<MessageCollection><message><flight source="FH" timestamp="2026-09-30T12:00:00Z"><flightIdentification aircraftIdentification="X1"/><route nasRouteText="${t}"/></flight></message></MessageCollection>`;
  parseSfdpsMessages(mk("KBOS..ILC..MLF..KATL"));
  parseSfdpsMessages(mk("KBOS..ILC..MLF..SAKES..KATL"));
  assert.equal(unplacedReasonCounters.fewFixes, 1);
  assert.equal(Object.values(unplacedReasonCounters).reduce((a, b) => a + b, 0), 1);
});

test("unresolved-token census covers fewFixes plans only and is report-only", () => {
  const u = unresolvedRouteTokens("ILC..QZQZQ.NOPE1");
  assert.ok(u.includes("QZQZQ") && u.includes("NOPE1") && !u.includes("ILC"));
  _resetSfdpsCountersForTests();
  const mk = (t: string) => `<MessageCollection><message><flight source="FH" timestamp="2026-09-30T12:00:00Z"><flightIdentification aircraftIdentification="X1"/><route nasRouteText="${t}"/></flight></message></MessageCollection>`;
  parseSfdpsMessages(mk("KBOS..ILC..MLF..KATL"));            // fewFixes -> counted
  parseSfdpsMessages(mk("ILC..MLF..SAKES"));                  // placed -> not counted
  const r = routeShapeSamples();
  assert.equal(r.unresolvedTexts.length, 1);
  assert.equal(r.unresolvedTexts[0].reason, "fewFixes");
  assert.ok(r.unresolvedTokens.every((x) => x.token !== "ILC" && x.token !== "SAKES"));
});

test("fewFixes profile counts resolved fixes, lat/lon and radial tokens and is report-only", () => {
  const f = fewFixesProfile("KBOS..ILC..4203N/08500W..SNS285053..MLF..KATL");
  assert.equal(f.resolved, 2);
  assert.equal(f.latlon, 1);
  assert.equal(f.radial, 1);
  _resetSfdpsCountersForTests();
  const mk = (t: string) => `<MessageCollection><message><flight source="FH" timestamp="2026-09-30T12:00:00Z"><flightIdentification aircraftIdentification="X1"/><route nasRouteText="${t}"/></flight></message></MessageCollection>`;
  parseSfdpsMessages(mk("KBOS..ILC..4203N/08500W..MLF..KATL"));
  const p = routeShapeSamples().fewFixesProfile;
  assert.equal(p.total, 1);
  assert.equal(p.resolved2, 1);
  assert.equal(p.withLatLon, 1);
  assert.equal(p.latlonReaches3, 1);
});

test("resolveRadialToken: magnetic radial + filed east variation = true bearing, distance exact, no guessing", () => {
  const base = lookupFix("SNS")!; // SNS filed variation 17E (1965)
  const p = resolveRadialToken("SNS285053")!;
  assert.ok(Math.abs(haversineNm(base, p) - 53) < 0.05);
  assert.ok(Math.abs(initialBearingDeg(base, p) - 302) < 0.2);
  assert.equal(p.magYear, 1965);
  const w = resolveRadialToken("PVD090012")!; // 14W: true bearing 76
  assert.ok(Math.abs(initialBearingDeg(lookupFix("PVD")!, w) - 76) < 0.2);
  assert.equal(resolveRadialToken("SAKES285053"), null); // fix without a navaid variation
  assert.equal(resolveRadialToken("SNS400053"), null); // radial > 360
  assert.equal(resolveRadialToken("SNS285000"), null); // zero distance
  assert.equal(resolveRadialToken("QZQ285053"), null); // unknown ident
  assert.equal(resolveRadialToken("SNS28505"), null);
});

test("radial shadow resolver clears the >=3 bar only with real placed points and never alters placement", () => {
  const t = "KBOS..ILC..SNS285053..MLF..KATL";
  const r = placeDirectFixesWithRadials(t);
  assert.deepEqual(r.points.map((x) => x.name), ["ILC", "SNS285053", "MLF"]);
  assert.equal(r.radial, 1);
  assert.deepEqual(placeDirectFixesWithRadials("KBOS..ILC..SNS285053..KATL").points, []); // only 2 points
  assert.deepEqual(placeDirectFixesWithRadials("ILC.J80.JNC..SNS285053").points, []); // airway refuses
  _resetSfdpsCountersForTests();
  const [f] = parseSfdpsMessages(`<MessageCollection><message><flight source="FH" timestamp="2026-09-30T12:00:00Z"><flightIdentification aircraftIdentification="X1"/><route nasRouteText="${t}"/></flight></message></MessageCollection>`);
  assert.deepEqual(f.routePoints, []); // shadow only: placement unchanged
  const c = routeShapeSamples().fewFixesProfile;
  assert.equal(c.radialShadowPlaced, 1);
  assert.equal(c.radialShadowTokens, 1);
});
