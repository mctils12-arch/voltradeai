// SWIM NOTAM (FNS digital NOTAM) parser — the non-TFR fixture is a trimmed
// copy of a REAL message from our queue (2026-10-14, SMF ILS); the TFR
// fixtures keep that exact element structure with FDC/TFR content (no TFR
// had reached the sample buffer yet — capture now buckets them separately).
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseNotamPayload, NotamStore, qlineCenter, notamTime, isTfr, NOTAM_TFR_DEFAULT_TTL_MS } from "./swimNotam";

const msg = (o: { cls: string; q?: string; text: string; type?: string; id?: string; start?: string; end?: string; airspace?: string; coords?: string; radius?: string }) =>
  `<?xml version="1.0"?><message:AIXMBasicMessage xmlns:message="m" xmlns:gml="g" xmlns:event="e" xmlns:aixm="a" xmlns:fnse="f"><message:hasMember><event:Event gml:id="E1"><event:timeSlice><event:EventTimeSlice gml:id="T1"><event:scenario>${o.cls === "FDC" ? "15" : "980"}</event:scenario><event:textNOTAM><event:NOTAM gml:id="N1"><event:series>A</event:series><event:number>1381</event:number><event:year>2026</event:year><event:type>${o.type ?? "N"}</event:type><event:issued>2026-10-14T23:20:00.000Z</event:issued><event:selectionCode>${o.q ?? ""}</event:selectionCode><event:minimumFL>000</event:minimumFL><event:maximumFL>180</event:maximumFL><event:coordinates>${o.coords ?? "3841N12135W"}</event:coordinates><event:radius>${o.radius ?? "5"}</event:radius><event:location>SMF</event:location><event:effectiveStart>${o.start ?? "202610151530"}</event:effectiveStart><event:effectiveEnd>${o.end ?? "202610151630"}</event:effectiveEnd><event:text>${o.text}</event:text></event:NOTAM></event:textNOTAM><event:extension><fnse:EventExtension gml:id="X1"><fnse:classification>${o.cls}</fnse:classification><fnse:xovernotamID>${o.id ?? "10/024"}</fnse:xovernotamID><fnse:icaoLocation>KSMF</fnse:icaoLocation></fnse:EventExtension></event:extension></event:EventTimeSlice></event:timeSlice></event:Event></message:hasMember>${o.airspace ?? ""}<message:hasMember><aixm:AirportHeliport gml:id="AH1"><aixm:ARP><aixm:ElevatedPoint><gml:pos>38.695444 -121.590772</gml:pos></aixm:ElevatedPoint></aixm:ARP></aixm:AirportHeliport></message:hasMember></message:AIXMBasicMessage>`;

const ILS = msg({ cls: "INTL", q: "QITAU", text: "NAV ILS RWY 17R CAT II/III NA" });
const AIRSPACE = `<message:hasMember><aixm:Airspace gml:id="AS1"><aixm:geometryComponent><aixm:Surface><gml:patches><gml:PolygonPatch><gml:exterior><gml:LinearRing><gml:posList>38.70 -121.70 38.80 -121.50 38.60 -121.40 38.70 -121.70</gml:posList></gml:LinearRing></gml:exterior></gml:PolygonPatch></gml:patches></aixm:Surface></aixm:geometryComponent></aixm:Airspace></message:hasMember>`;
const TFR_POLY = msg({ cls: "FDC", id: "6/1234", text: "CA..AIRSPACE SACRAMENTO, CA..TEMPORARY FLIGHT RESTRICTIONS WI AN AREA DEFINED AS ... PURSUANT TO 14 CFR SECTION 91.137(A)(2)", airspace: AIRSPACE, start: "202610150000", end: "202610170000" });
const TFR_CIRCLE = msg({ cls: "FDC", id: "6/2222", text: "VIP MOVEMENT ... 99.7 SPECIAL SECURITY INSTRUCTIONS", coords: "3854N07702W", radius: "30", start: "202610150000", end: "202610160000" });

test("Q-line center and NOTAM times decode", () => {
  assert.deepEqual(qlineCenter("3841N12135W"), [38.6833, -121.5833]);
  assert.deepEqual(qlineCenter("3341S15112E"), [-33.6833, 151.2]);
  assert.equal(qlineCenter("garbage"), null);
  assert.equal(notamTime("202610151530"), Date.UTC(2026, 9, 15, 15, 30));
  assert.equal(notamTime("PERM"), null);
});

test("a real non-TFR NOTAM: every field parsed, no geometry (runway/airport positions are never a restriction area)", () => {
  const [n] = parseNotamPayload(ILS);
  assert.equal(n.id, "INTL:10/024");
  assert.equal(n.qcode, "QITAU");
  assert.equal(n.location, "KSMF");
  assert.equal(n.text, "NAV ILS RWY 17R CAT II/III NA");
  assert.equal(n.start, Date.UTC(2026, 9, 15, 15, 30));
  assert.equal(n.radiusNm, 5);
  assert.equal(n.geometry, null);
  assert.equal(isTfr(n), false);
});

test("TFR with airspace geometry → exact polygon; TFR without → Q-line circle labeled approximate", () => {
  const [p] = parseNotamPayload(TFR_POLY);
  assert.equal(p.geometry!.kind, "polygon");
  assert.deepEqual((p.geometry as { rings: Array<Array<[number, number]>> }).rings[0][1], [38.8, -121.5]);
  assert.equal(p.maxFl, 180);
  const [c] = parseNotamPayload(TFR_CIRCLE);
  assert.deepEqual(c.geometry, { kind: "circle", center: [38.9, -77.0333], radiusNm: 30, approximate: true });
  assert.ok(isTfr({ classification: "INTL", qcode: "QRTCA", text: "AREA ACTIVE" }), "ICAO Q-code for a temporary restricted area");
  assert.ok(!isTfr({ classification: "DOM", qcode: null, text: "TFR" }), "only FDC text counts");
});

test("store: active TFRs only (window, cancellation); non-TFRs are counted by classification", () => {
  const s = new NotamStore();
  const now = Date.UTC(2026, 9, 15, 12, 0);
  for (const x of [ILS, TFR_POLY, TFR_CIRCLE]) s.ingest(x, now);
  assert.deepEqual(s.active(now).map((t) => t.id).sort(), ["FDC:6/1234", "FDC:6/2222"]);
  assert.deepEqual(s.stats().byClassification, { INTL: 1, FDC: 2 });
  assert.deepEqual(s.active(Date.UTC(2026, 9, 16, 12, 0)).map((t) => t.id), ["FDC:6/1234"], "the 24 h one ended");
  s.ingest(msg({ cls: "FDC", id: "6/1234", type: "C", text: "TEMPORARY FLIGHT RESTRICTIONS CANCELED", airspace: AIRSPACE }), now);
  assert.ok(!s.active(now).some((t) => t.id === "FDC:6/1234"), "a cancellation removes it");
  const noEnd = new NotamStore();
  noEnd.ingest(msg({ cls: "FDC", id: "6/3333", text: "FLIGHT RESTRICTION", end: "PERM" }), now);
  const t = noEnd.active(now)[0];
  assert.equal(t.end, null);
  assert.equal(noEnd.active(Date.parse("2026-10-14T23:20:00Z") + NOTAM_TFR_DEFAULT_TTL_MS + 1).length, 0, "kept a bounded time past issue");
  assert.equal(s.stats().parseErrors, 0);
  assert.deepEqual(parseNotamPayload("<x/>"), []);
});

test("ids without a cross-over id stay unique per location", () => {
  const strip = (x: string, loc: string) => x.replace(/<fnse:xovernotamID>[^<]*<\/fnse:xovernotamID>/, "").replace("KSMF", loc);
  const a = parseNotamPayload(strip(ILS, "KCNY"))[0];
  const b = parseNotamPayload(strip(ILS, "KDAL"))[0];
  assert.equal(a.id, "KCNY:A1381/2026");
  assert.notEqual(a.id, b.id);
});
