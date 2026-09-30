import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseXmlLite, attr, findAll, parseSfdpsMessages, altitudeFeet, SwimPlanStore, planDiff,
  rootElementName, classifySfdpsPayload, flightChunks, needsFullParse, lightFlight,
  handleSfdpsPayload, sfdpsCounters, sfdpsStatus, startSfdps, routePointsOf, packRoute,
  _resetSfdpsCountersForTests, SWIM_ROUTE_MAX_POINTS, routeShapeSamples, xmlShape, type StoredSwimPlan,
} from "./swimSfdps";
import { _resetSwimConnectorForTests } from "./swimConnector";

// ── hand-written SFDPS fixtures ─────────────────────────────────────────────
// Modeled on the FAA SFDPS FIXM 3.0 + NAS-extension message shape
// (MessageCollection > message > flight with source=FH/AH/HZ/HX). The exact
// production schema could not be verified offline; the parser is tolerant by
// design and these fixtures deliberately mix prefix and point encodings.
const NS = `xmlns:ns2="http://www.fixm.aero/base/3.0" xmlns:ns3="http://www.fixm.aero/flight/3.0" ` +
  `xmlns:ns5="http://www.faa.aero/nas/3.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"`;

const FH = `<?xml version="1.0" encoding="UTF-8"?>
<ns5:MessageCollection ${NS}>
  <!-- flight plan information -->
  <message xsi:type="ns5:FlightMessageType">
    <flight centre="ZBW" source="FH" system="SLC" timestamp="2026-09-28T14:02:11.123Z" xsi:type="ns5:NasFlightType">
      <arrival arrivalPoint="KATL">
        <runwayPositionAndTime><runwayTime><estimated time="2026-09-28T16:40:00Z"/></runwayTime></runwayPositionAndTime>
      </arrival>
      <departure departurePoint="KBOS">
        <runwayPositionAndTime><runwayTime><estimated time="2026-09-28T14:30:00Z"/></runwayTime></runwayPositionAndTime>
      </departure>
      <flightIdentification aircraftIdentification="DAL1234" computerId="45A" siteSpecificPlanId="3312"/>
      <flightStatus fdpsFlightStatus="PROPOSED"/>
      <gufi codeSpace="urn:uuid">5a8c6a52-2f2b-4d1e-9a4f-0f0b8b2f1c11</gufi>
      <requestedAltitude><ns5:simple uom="FEET">35000.0</ns5:simple></requestedAltitude>
      <agreed>
        <route nasRouteText="KBOS.SSOXS5.SSOXS..QZQZQ..SEY..HTO.J150.OOD..BROSS..KATL &amp; notes" initialFlightRules="IFR">
          <ns5:expandedRoute>
            <ns5:routePoint>
              <ns2:point xsi:type="ns2:FixPointType" fix="KBOS">
                <ns2:location srsName="urn:ogc:def:crs:EPSG::4326"><ns2:pos>42.3630 -71.0064</ns2:pos></ns2:location>
              </ns2:point>
            </ns5:routePoint>
            <ns5:routePoint>
              <ns2:point fix="SSOXS"><ns2:location><ns2:pos>42.0203 -70.4480</ns2:pos></ns2:location></ns2:point>
            </ns5:routePoint>
            <ns5:routePoint>
              <!-- a fix with no position: must be skipped, never guessed -->
              <ns2:point fix="QZQZQ"/>
            </ns5:routePoint>
            <ns5:routePoint>
              <ns2:point designator="HTO"><ns2:position latitude="40.9195" longitude="-72.3166"/></ns2:point>
              <ns5:altitude uom="FL">350</ns5:altitude>
            </ns5:routePoint>
            <ns5:routePoint>
              <ns2:point fix="OOD"><ns2:location><ns2:pos>39.6358 -75.3031</ns2:pos></ns2:location></ns2:point>
            </ns5:routePoint>
            <ns5:routePoint>
              <ns2:point fix="KATL"><ns2:location><ns2:pos>33.6367 -84.4281</ns2:pos></ns2:location></ns2:point>
            </ns5:routePoint>
          </ns5:expandedRoute>
        </route>
      </agreed>
    </flight>
  </message>
</ns5:MessageCollection>`;

const FH_TEXT_ONLY = `<ns5:MessageCollection ${NS}>
  <message><flight source="FH" timestamp="2026-09-28T15:00:00Z">
    <flightIdentification aircraftIdentification="N123AB"/>
    <gufi>gufi-text-only</gufi>
    <departure departurePoint="KTEB"/><arrival arrivalPoint="KPBI"/>
    <requestedAltitude><simple uom="FL">410</simple></requestedAltitude>
    <agreed><route nasRouteText="KTEB..WHITE..J209..KPBI"/></agreed>
  </flight></message>
</ns5:MessageCollection>`;

const AH = FH.replace('source="FH"', 'source="AH"')
  .replace("timestamp=\"2026-09-28T14:02:11.123Z\"", "timestamp=\"2026-09-28T14:20:00Z\"")
  .replace("<ns2:pos>39.6358 -75.3031</ns2:pos>", "<ns2:pos>39.9000 -76.0000</ns2:pos>")
  .replace("J150.OOD", "J150.XYZ");

const HZ = `<ns5:MessageCollection ${NS}>
  <message><flight source="HZ" timestamp="2026-09-28T14:45:00Z">
    <flightIdentification aircraftIdentification="DAL1234"/>
    <gufi>5a8c6a52-2f2b-4d1e-9a4f-0f0b8b2f1c11</gufi>
    <enRoute><position positionTime="2026-09-28T14:45:00Z">
      <position srsName="urn:ogc:def:crs:EPSG::4326"><pos>41.5 -72.1</pos></position>
      <altitude uom="FEET">28000.0</altitude>
    </position></enRoute>
  </flight></message>
</ns5:MessageCollection>`;

const HX = `<ns5:MessageCollection ${NS}><message><flight source="HX" timestamp="2026-09-28T17:00:00Z">
  <flightIdentification aircraftIdentification="DAL1234"/><gufi>5a8c6a52-2f2b-4d1e-9a4f-0f0b8b2f1c11</gufi>
</flight></message></ns5:MessageCollection>`;

// FIXM 4.x element forms (aerodromes as elements, flight level as element)
const FIXM4 = `<fx:Flight xmlns:fx="http://www.fixm.aero/flight/4.2">
  <fx:flight>
    <fx:aircraftIdentification>UAL1</fx:aircraftIdentification>
    <fx:gufi>gufi-4x</fx:gufi>
    <fx:departure><fx:departureAerodrome><fx:locationIndicator>KSFO</fx:locationIndicator></fx:departureAerodrome></fx:departure>
    <fx:arrival><fx:destinationAerodrome><fx:locationIndicator>WSSS</fx:locationIndicator></fx:destinationAerodrome></fx:arrival>
    <fx:routeTrajectoryGroup><fx:agreed><fx:routeInformation>
      <fx:cruisingLevel><fx:flightLevel uom="FL">370</fx:flightLevel></fx:cruisingLevel>
      <fx:routeText>KSFO DCT PYE ... WSSS</fx:routeText>
    </fx:routeInformation></fx:agreed></fx:routeTrajectoryGroup>
  </fx:flight>
</fx:Flight>`;

// ── XML-lite ────────────────────────────────────────────────────────────────

test("parseXmlLite: prefixes stripped, attributes kept, entities + CDATA decoded, comments skipped", () => {
  const root = parseXmlLite(`<?xml version="1.0"?><!-- c --><a:root x:k="1 &amp; 2" plain='q'><b>t&lt;1</b><c/><d><![CDATA[<raw>]]></d></a:root>`)!;
  const r = root.children[0];
  assert.equal(r.name, "root");
  assert.equal(attr(r, "k"), "1 & 2");
  assert.equal(attr(r, "plain"), "q");
  assert.deepEqual(r.children.map((c) => c.name), ["b", "c", "d"]);
  assert.equal(r.children[0].text, "t<1");
  assert.equal(r.children[2].text, "<raw>");
  assert.equal(findAll(root, "c").length, 1);
});

test("parseXmlLite: malformed input degrades instead of throwing; oversized is refused", () => {
  assert.ok(parseXmlLite("<a><b>unclosed"));
  assert.equal(parseXmlLite("x".repeat(5 * 1024 * 1024)), null);
  assert.deepEqual(parseSfdpsMessages("not xml at all"), []);
  assert.deepEqual(parseSfdpsMessages(""), []);
});

// ── FIXM extraction ─────────────────────────────────────────────────────────

test("parseSfdpsMessages: FH flight plan — identity, aerodromes, filed cruise, route text, expanded points", () => {
  const [f] = parseSfdpsMessages(FH);
  assert.equal(f.gufi, "5a8c6a52-2f2b-4d1e-9a4f-0f0b8b2f1c11");
  assert.equal(f.callsign, "DAL1234");
  assert.equal(f.departure, "KBOS");
  assert.equal(f.arrival, "KATL");
  assert.equal(f.cruiseAltFt, 35000);
  assert.equal(f.routeText, "KBOS.SSOXS5.SSOXS..QZQZQ..SEY..HTO.J150.OOD..BROSS..KATL & notes");
  assert.equal(f.messageType, "FH");
  assert.equal(f.isAmendment, false);
  assert.equal(f.isCancellation, false);
  assert.equal(f.timestamp, Date.parse("2026-09-28T14:02:11.123Z"));
  assert.equal(f.departureTime, Date.parse("2026-09-28T14:30:00Z"));
  // 6 route points, QZQZQ (a fictitious ident: not in the NASR gazetteer) has no position -> 5 placed, in order
  assert.deepEqual(f.routePoints.map((p) => p.name), ["KBOS", "SSOXS", "HTO", "OOD", "KATL"]);
  assert.deepEqual({ lat: f.routePoints[0].lat, lon: f.routePoints[0].lon }, { lat: 42.363, lon: -71.0064 });
  assert.equal(f.routePoints[2].lat, 40.9195, "latitude/longitude attribute form accepted");
  assert.equal(f.routePoints[2].altFt, 35000, "per-point FL350 -> feet");
  assert.equal(f.routePoints[0].altFt, undefined);
});

test("parseSfdpsMessages: route-text-only plan keeps text + aerodromes, no invented points; FL cruise", () => {
  const [f] = parseSfdpsMessages(FH_TEXT_ONLY);
  assert.equal(f.callsign, "N123AB");
  assert.equal(f.departure, "KTEB");
  assert.equal(f.arrival, "KPBI");
  assert.equal(f.cruiseAltFt, 41000);
  assert.equal(f.routeText, "KTEB..WHITE..J209..KPBI");
  assert.deepEqual(f.routePoints, []);
});

test("parseSfdpsMessages: amendment, track-only and cancellation message types", () => {
  const [a] = parseSfdpsMessages(AH);
  assert.equal(a.messageType, "AH");
  assert.equal(a.isAmendment, true);
  const [h] = parseSfdpsMessages(HZ);
  assert.equal(h.messageType, "HZ");
  assert.equal(h.departure, null);
  assert.deepEqual(h.position, { lat: 41.5, lon: -72.1, altFt: 28000 });
  const [x] = parseSfdpsMessages(HX);
  assert.equal(x.isCancellation, true);
});

test("parseSfdpsMessages: FIXM 4.x element forms are accepted", () => {
  const [f] = parseSfdpsMessages(FIXM4);
  assert.equal(f.callsign, "UAL1");
  assert.equal(f.departure, "KSFO");
  assert.equal(f.arrival, "WSSS");
  assert.equal(f.cruiseAltFt, 37000);
  assert.equal(f.routeText, "KSFO DCT PYE ... WSSS");
});

test("altitudeFeet: FEET / FL / METERS / VFR / garbage", () => {
  const n = (x: string) => parseXmlLite(x)!.children[0];
  assert.equal(altitudeFeet(n(`<a><simple uom="FEET">12000</simple></a>`)), 12000);
  assert.equal(altitudeFeet(n(`<a uom="FL">250</a>`)), 25000);
  assert.equal(altitudeFeet(n(`<a><altitude uom="METERS">3000</altitude></a>`)), 9843);
  assert.equal(altitudeFeet(n(`<a><vfr/></a>`)), null);
  assert.equal(altitudeFeet(n(`<a><simple uom="FEET">abc</simple></a>`)), null);
  assert.equal(altitudeFeet(null), null);
});

// ── store ───────────────────────────────────────────────────────────────────


// ── classification (no DOM for ignored services) ────────────────────────────

const AIXM = `<?xml version="1.0" encoding="UTF-8"?>
<message:AIXMBasicMessage xmlns:message="http://www.aixm.aero/schema/5.1/message" xmlns:aixm="http://www.aixm.aero/schema/5.1" gml:id="m1">
  <message:hasMember><aixm:Airspace gml:id="SAA-1"/></message:hasMember>
</message:AIXMBasicMessage>`;
const STATUS = `<?xml version="1.0"?><ns2:StatusMessage xmlns:ns2="urn:us:gov:dot:faa:atm:status"><service>SFDPS</service><state>UP</state></ns2:StatusMessage>`;
const GENERAL = `<ns5:MessageCollection ${NS}><message xsi:type="ns5:NasGeneralMessageType"><text>WX ADVISORY</text></message></ns5:MessageCollection>`;

test("rootElementName skips BOM, declaration, comments and DOCTYPE", () => {
  assert.equal(rootElementName(`﻿<?xml version="1.0"?>\n<!-- x --><!DOCTYPE a><ns5:MessageCollection a="1">`), "MessageCollection");
  assert.equal(rootElementName("   <flight/>"), "flight");
  assert.equal(rootElementName("plain text"), null);
  assert.equal(rootElementName(""), null);
});

test("classifySfdpsPayload: the four services on the one SFDPS queue", () => {
  assert.equal(classifySfdpsPayload(FH), "FLIGHT");
  assert.equal(classifySfdpsPayload(HZ), "FLIGHT");
  assert.equal(classifySfdpsPayload(AIXM), "AIRSPACE_AIXM");
  assert.equal(classifySfdpsPayload(STATUS), "STATUS");
  assert.equal(classifySfdpsPayload(GENERAL), "GENERAL_MESSAGE");
  assert.equal(classifySfdpsPayload("<foo><bar/></foo>"), "UNKNOWN");
  assert.equal(classifySfdpsPayload("garbage"), "UNKNOWN");
});

test("flightChunks: one chunk per flight, message tag captured, flightIdentification not mistaken for flight", () => {
  const two = `<ns5:MessageCollection ${NS}>` +
    `<message source="HZ" timestamp="2026-09-28T14:45:00Z"><flight><flightIdentification aircraftIdentification="AAL1"/><gufi>g-a</gufi></flight></message>` +
    `<message><flight source="OH"><flightIdentification aircraftIdentification="AAL2"/><gufi>g-b</gufi></flight></message>` +
    `</ns5:MessageCollection>`;
  const c = flightChunks(two);
  assert.equal(c.length, 2);
  assert.ok(c[0].chunk.startsWith("<flight>") && c[0].chunk.endsWith("</flight>"));
  assert.match(c[0].messageTag || "", /source="HZ"/);
  assert.match(c[1].chunk, /AAL2/);
  assert.equal(flightChunks(FH).length, 1);
});

test("needsFullParse: plan-bearing types/contents only", () => {
  const [fh] = flightChunks(FH);
  assert.equal(needsFullParse(fh.chunk, "FH"), true);
  const [hz] = flightChunks(HZ);
  assert.equal(needsFullParse(hz.chunk, "HZ"), false);
  assert.equal(needsFullParse("<flight><agreed><route nasRouteText='X'/></agreed></flight>", "OH"), true);
});

test("lightFlight: identity + filed aerodromes + status by regex", () => {
  const chunk = `<flight source="HZ" timestamp="2026-09-28T14:45:00Z"><arrival arrivalPoint="KATL"/><departure departurePoint="KBOS"/>` +
    `<flightIdentification aircraftIdentification="DAL1234"/><flightStatus fdpsFlightStatus="ACTIVE"/><gufi codeSpace="urn:uuid">g-1</gufi></flight>`;
  const f = lightFlight(chunk, null)!;
  assert.equal(f.callsign, "DAL1234");
  assert.equal(f.gufi, "g-1");
  assert.equal(f.departure, "KBOS");
  assert.equal(f.arrival, "KATL");
  assert.equal(f.messageType, "HZ");
  assert.equal(f.flightStatus, "ACTIVE");
  assert.equal(f.timestamp, Date.parse("2026-09-28T14:45:00Z"));
  assert.deepEqual(f.routePoints, []);
  assert.equal(lightFlight(`<flight source="HZ"/>`, null), null, "no identity -> nothing");
  assert.equal(lightFlight(chunk.replace("ACTIVE", "COMPLETED"), null)!.isCompleted, true);
});

// ── store ───────────────────────────────────────────────────────────────────

test("SwimPlanStore: track messages never erase the filed route; amendments are detected and reported", () => {
  const amended: string[] = [];
  const store = new SwimPlanStore({ onAmended: (_p, d) => amended.push(d) });
  const t0 = Date.parse("2026-09-28T14:03:00Z");
  for (const f of parseSfdpsMessages(FH)) store.upsert(f, t0);
  for (const f of parseSfdpsMessages(HZ)) store.upsert(f, t0 + 60_000);
  let p = store.lookup("DAL1234", t0 + 61_000)!;
  assert.equal(routePointsOf(p).length, 5, "HZ kept the route");
  assert.equal(amended.length, 0);
  for (const f of parseSfdpsMessages(AH)) store.upsert(f, t0 + 120_000);
  assert.equal(amended.length, 1);
  assert.match(amended[0], /route text changed/);
  assert.match(amended[0], /expanded route changed/);
  p = store.lookup("DAL1234", t0 + 121_000)!;
  assert.equal(p.amendments, 1);
  assert.equal(routePointsOf(p)[3].lat, 39.9);
  assert.equal(store.size, 1, "same GUFI = same entry");
  for (const f of parseSfdpsMessages(HX)) store.upsert(f, t0 + 180_000);
  assert.equal(store.lookup("DAL1234", t0 + 181_000), null, "cancellation drops the plan");
});

test("SwimPlanStore: COMPLETED status removes the plan (store holds active flights only)", () => {
  const store = new SwimPlanStore();
  for (const f of parseSfdpsMessages(FH)) store.upsert(f, 1000);
  assert.ok(store.lookup("DAL1234", 1001));
  const done = parseSfdpsMessages(FH.replace('fdpsFlightStatus="PROPOSED"', 'fdpsFlightStatus="COMPLETED"').replace('source="FH"', 'source="HZ"'));
  for (const f of done) store.upsert(f, 2000);
  assert.equal(store.lookup("DAL1234", 2001), null);
  assert.equal(store.size, 0);
});

test("SwimPlanStore: AH with no drawable change is still recorded honestly", () => {
  const amended: string[] = [];
  const store = new SwimPlanStore({ onAmended: (_p, d) => amended.push(d) });
  for (const f of parseSfdpsMessages(FH)) store.upsert(f, 1000);
  for (const f of parseSfdpsMessages(FH.replace('source="FH"', 'source="AH"'))) store.upsert(f, 2000);
  assert.deepEqual(amended, ["amendment message (no route, destination or cruise change detected)"]);
});

test("SwimPlanStore: TTL expiry, LRU cap, bare entries are not plans, no-GUFI keying", () => {
  const store = new SwimPlanStore({ ttlMs: 10_000, max: 2 });
  const base = parseSfdpsMessages(FH_TEXT_ONLY)[0];
  store.upsert({ ...base, gufi: null, callsign: "AAA1" }, 0);
  store.upsert({ ...base, gufi: null, callsign: "AAA1", cruiseAltFt: 39000 }, 1000);
  assert.equal(store.size, 1, "no-GUFI follow-up merges into the live callsign entry");
  assert.equal(store.lookup("AAA1", 1500)!.cruiseAltFt, 39000);
  store.upsert({ ...base, gufi: "g2", callsign: "BBB2" }, 2000);
  store.upsert({ ...base, gufi: "g3", callsign: "CCC3" }, 3000);
  assert.equal(store.size, 2, "capped");
  assert.equal(store.lookup("AAA1", 3000), null, "oldest evicted");
  assert.equal(store.lookup("CCC3", 20_000), null, "expired");
  const trackOnly = parseSfdpsMessages(HZ)[0];
  store.upsert({ ...trackOnly, gufi: "g4", callsign: "DDD4" }, 20_000);
  assert.equal(store.lookup("DDD4", 20_001), null, "an entry with no aerodromes is not a plan");
});

test("packRoute / routePointsOf: compact round trip, capped point count", () => {
  const pts = [{ lat: 42.363, lon: -71.0064, name: "KBOS" }, { lat: 40.9195, lon: -72.3166, altFt: 35000 }];
  const packed = packRoute(pts);
  assert.equal(packed.route.length, 6);
  const plan = { route: packed.route, routeNames: packed.routeNames } as StoredSwimPlan;
  assert.deepEqual(routePointsOf(plan), [{ lat: 42.363, lon: -71.0064, name: "KBOS" }, { lat: 40.9195, lon: -72.3166, altFt: 35000 }]);
  const many = Array.from({ length: SWIM_ROUTE_MAX_POINTS + 50 }, (_, i) => ({ lat: i / 100, lon: 0 }));
  assert.equal(packRoute(many).route.length, SWIM_ROUTE_MAX_POINTS * 3);
});

test("planDiff: destination / cruise changes are named", () => {
  const store = new SwimPlanStore();
  store.upsert(parseSfdpsMessages(FH)[0], 1);
  const a = store.lookup("DAL1234", 2)!;
  assert.equal(planDiff(a, { ...a }), null);
  assert.equal(planDiff(a, { ...a, arrival: "KCLT" }), "destination KATL→KCLT");
  assert.equal(planDiff(a, { ...a, cruiseAltFt: 37000 }), "cruise 35000→37000 ft");
});

// ── payload routing + counters ──────────────────────────────────────────────

test("handleSfdpsPayload: flights into the store, other services counted and ignored, per-type counters", () => {
  _resetSfdpsCountersForTests();
  const store = new SwimPlanStore();
  assert.equal(handleSfdpsPayload(FH, store, 1000), 1);
  assert.equal(handleSfdpsPayload(HZ, store, 2000), 1);
  assert.equal(handleSfdpsPayload(AIXM, store, 3000), 0);
  assert.equal(handleSfdpsPayload(STATUS, store, 3000), 0);
  assert.equal(handleSfdpsPayload(GENERAL, store, 3000), 0);
  assert.equal(handleSfdpsPayload("<<not xml", store, 3000), 0);
  const c = sfdpsCounters();
  assert.deepEqual(c.byService, { FLIGHT: 2, AIRSPACE_AIXM: 1, GENERAL_MESSAGE: 1, STATUS: 1, UNKNOWN: 1 });
  assert.deepEqual(c.flightByType, { FH: 1, HZ: 1 });
  assert.equal(c.fullParses, 1, "FH parsed fully");
  assert.equal(c.lightParses, 1, "HZ took the light path");
  const p = store.lookup("DAL1234", 3000)!;
  assert.equal(routePointsOf(p).length, 5);
  assert.equal(p.updatedAt, 2000, "HZ refreshed the TTL clock");
  assert.equal(sfdpsStatus(store).storeSize, 1);
});

test("handleSfdpsPayload: a flight first seen mid-air via a track message records its FILED aerodromes", () => {
  _resetSfdpsCountersForTests();
  const store = new SwimPlanStore();
  const hz = `<ns5:MessageCollection ${NS}><message><flight source="HZ" timestamp="2026-09-28T14:45:00Z">` +
    `<arrival arrivalPoint="KSEA"/><departure departurePoint="KJFK"/>` +
    `<flightIdentification aircraftIdentification="JBU263"/><gufi>g-jbu</gufi></flight></message></ns5:MessageCollection>`;
  handleSfdpsPayload(hz, store, 5000);
  const p = store.lookup("JBU263", 5001)!;
  assert.equal(p.departure, "KJFK");
  assert.equal(p.arrival, "KSEA");
  assert.equal(p.route.length, 0, "no route invented");
});

test("startSfdps: zero cost without env; routes consumer payloads through the SFDPS handler", async () => {
  _resetSwimConnectorForTests();
  _resetSfdpsCountersForTests();
  let imported = 0;
  const h = await startSfdps(new SwimPlanStore(), { env: {} as NodeJS.ProcessEnv, importer: async () => { imported++; return {}; } });
  assert.equal(imported, 0);
  assert.equal(h.status().configured, false);
  assert.equal(sfdpsStatus().configured, false);
});

// ── route-shape sampler: gate-1 instrument for the empty-routePoints bug ────
const UNPLACED = (id: string) => `<m:MessageCollection xmlns:m="urn:x"><message><flight source="FH" timestamp="2026-09-29T12:00:00Z">
  <flightIdentification aircraftIdentification="${id}"/>
  <agreed><route nasRouteText="KBOS..HTO..KATL"><expandedRoute>
    <routePoint><nasFix fixName="QZQZR" lat="40N" lon="073W"/></routePoint>
    <routePoint><nasFix fixName="QZQZS"/></routePoint>
  </expandedRoute></route></agreed></flight></message></m:MessageCollection>`;

test("route-shape sampler: unplaced route is recorded as SHAPE only (no real values), deduped with a count", () => {
  _resetSfdpsCountersForTests();
  parseSfdpsMessages(UNPLACED("DAL123"));
  parseSfdpsMessages(UNPLACED("UAL456"));
  const r = routeShapeSamples();
  assert.equal(r.counters.expandedNoPoints, 2);
  assert.equal(r.counters.placed, 0);
  assert.equal(r.samples.length, 1, "identical shapes dedupe");
  assert.equal(r.samples[0].count, 2);
  assert.equal(r.samples[0].where, "expandedRoute");
  assert.match(r.samples[0].shape, /routePoint\(nasFix\[fixName=AA\+,lat=99A,lon=99\+A\]/);
  assert.doesNotMatch(r.samples[0].shape, /QZQZR|QZQZS|KBOS|40N|073W|DAL123/, "no real values leak");
});

test("route-shape sampler: placed routes only bump the counter; missing expandedRoute is labelled", () => {
  _resetSfdpsCountersForTests();
  const placed = `<message><flight source="FH"><flightIdentification aircraftIdentification="DAL1"/>
    <agreed><route><expandedRoute><routePoint><pos>40.1 -70.2</pos></routePoint></expandedRoute></route></agreed></flight></message>`;
  parseSfdpsMessages(placed);
  parseSfdpsMessages(`<message><flight source="FH"><flightIdentification aircraftIdentification="DAL2"/><agreed><route nasRouteText="A B"/></agreed></flight></message>`);
  const r = routeShapeSamples();
  assert.equal(r.counters.placed, 1);
  assert.equal(r.counters.noExpanded, 1);
  assert.equal(r.samples.length, 1);
  assert.equal(r.samples[0].where, "route(no expandedRoute)");
});

test("route-shape sampler: bounded samples and depth", () => {
  _resetSfdpsCountersForTests();
  for (let i = 0; i < 20; i++) parseSfdpsMessages(`<message><flight source="FH"><flightIdentification aircraftIdentification="AAL${i}"/><agreed><route><expandedRoute><p${i}/></expandedRoute></route></agreed></flight></message>`);
  assert.ok(routeShapeSamples().samples.length <= 6);
  const deep = parseXmlLite("<a><b><c><d><e><f><g><h><i><j><k>x</k></j></i></h></g></f></e></d></c></b></a>")!;
  assert.ok(!xmlShape(deep).includes("k{"), "depth-capped");
});

// ── NASR gazetteer: fix NAMES resolve to positions (live SFDPS carries no coords) ──
const NAME_ONLY = (pts: string) => `<message><flight source="FH"><flightIdentification aircraftIdentification="DAL9"/>
  <agreed><route nasRouteText="X"><expandedRoute>${pts}</expandedRoute></route></agreed></flight></message>`;

test("fix-name-only routePoints are placed from the FAA NASR gazetteer; unknown names and offset points are not", () => {
  _resetSfdpsCountersForTests();
  const [f] = parseSfdpsMessages(NAME_ONLY(`
    <routePoint><point fix="HTO"/></routePoint>
    <routePoint><point fix="QZQZQ"/></routePoint>
    <routePoint><point fix="OOD"><distance uom="NM">12.5</distance><radial uom="DEG">270.0</radial></point></routePoint>
    <routePoint><point fix="bos"/></routePoint>`));
  assert.deepEqual(f.routePoints.map((p) => p.name), ["HTO", "bos"], "unknown + place-bearing-distance offset are not placed");
  assert.deepEqual({ lat: f.routePoints[0].lat, lon: f.routePoints[0].lon }, { lat: 40.919, lon: -72.3167 });
  assert.ok(Math.abs(f.routePoints[1].lat - 42.357) < 0.01, "navaid ident resolves, case-insensitive");
  assert.equal(routeShapeSamples().counters.placed, 1);
});

test("explicit coordinates win over the gazetteer", () => {
  const [f] = parseSfdpsMessages(NAME_ONLY(`<routePoint><point fix="HTO" lat="1.5" lon="2.5"/></routePoint>`));
  assert.deepEqual({ lat: f.routePoints[0].lat, lon: f.routePoints[0].lon }, { lat: 1.5, lon: 2.5 });
});
