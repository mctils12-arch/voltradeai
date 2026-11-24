// SWIM NOTAM (FNS digital NOTAM) parser — the non-TFR fixture is a trimmed
// copy of a REAL message from our queue (2026-10-14, SMF ILS); the TFR
// fixtures keep that exact element structure with FDC/TFR content (no TFR
// had reached the sample buffer yet — capture now buckets them separately).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
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
  // a restriction notice with no airspace geometry and no index circle cannot
  // be drawn: it is counted, so the map can say its list is short by them
  assert.equal(s.stats().tfrs_without_shape, 0, "every restriction above carried a shape; the plain ILS notice is not counted");
  const bare = (x: string) => x.replace(/<event:coordinates>[^<]*<\/event:coordinates>/, "").replace(/<event:radius>[^<]*<\/event:radius>/, "");
  const shapeless = new NotamStore();
  // (2026-11-22: the text of this fixture was "WI AN AREA DEFINED AS 3NM RADIUS OF 370905N0834549W", which is
  // now READ, see the tests below; a wording the reader does not follow keeps this count's meaning)
  shapeless.ingest(bare(msg({ cls: "FDC", id: "6/4444", text: "TEMPORARY FLIGHT RESTRICTIONS WITHIN A 3 NMR OF 370905N/0834549W PURSUANT TO 14 CFR SECTION 91.137(A)(1)" })), now);
  assert.deepEqual({ held: shapeless.stats().tfrs, without_shape: shapeless.stats().tfrs_without_shape, listed: shapeless.active(now).length }, { held: 0, without_shape: 1, listed: 0 });
  shapeless.ingest(bare(msg({ cls: "FDC", id: "6/4444", type: "C", text: "TEMPORARY FLIGHT RESTRICTIONS CANCELED" })), now);
  assert.equal(shapeless.stats().tfrs_without_shape, 1, "a cancellation is not a restriction that could not be drawn");
  shapeless.ingest(bare(ILS), now);
  assert.equal(shapeless.stats().tfrs_without_shape, 1, "a notice that is not a flight restriction is not counted");
  assert.deepEqual(parseNotamPayload("<x/>"), []);
});

test("ids without a cross-over id stay unique per location", () => {
  const strip = (x: string, loc: string) => x.replace(/<fnse:xovernotamID>[^<]*<\/fnse:xovernotamID>/, "").replace("KSMF", loc);
  const a = parseNotamPayload(strip(ILS, "KCNY"))[0];
  const b = parseNotamPayload(strip(ILS, "KDAL"))[0];
  assert.equal(a.id, "KCNY:A1381/2026");
  assert.notEqual(a.id, b.id);
});

// ── the outline read from the notice's words (2026-11-22) ──────────────────
// A REAL message: FDC 6/9306 as the FAA notice feed delivered it to
// production on 2026-11-21 23:27Z (the capture's TFR sample, unedited). It
// has no airspace geometry, no index point and no radius; before this change
// the store dropped it while tfr.faa.gov listed it first.
const FDC_9306 = fs.readFileSync(new URL("./__fixtures__/notam_fdc_6_9306.xml", import.meta.url), "utf8");
const bareMsg = (x: string) => x.replace(/<event:coordinates>[^<]*<\/event:coordinates>/, "").replace(/<event:radius>[^<]*<\/event:radius>/, "");

test("the real US notice production dropped: its outline is read from its text, labeled as such, and it is listed", () => {
  const [r] = parseNotamPayload(FDC_9306);
  assert.equal(r.classification, "FDC");
  assert.equal(r.citation, "FDC 6/9306", "copied from the notice's own plain-text form");
  assert.equal(r.center, null);
  assert.equal(r.radiusNm, null);
  assert.equal(r.start, Date.UTC(2026, 10, 22, 12, 0));
  assert.equal(r.end, Date.UTC(2026, 10, 29, 23, 0), "an end marked EST (estimated) is still its end");
  assert.deepEqual(r.geometry, {
    kind: "polygon", from: "text", unread: 0,
    rings: [[[37.866667, -79.875], [37.85, -79.833333], [37.808333, -79.866667], [37.816667, -79.9], [37.866667, -79.875]]],
  });
  const s = new NotamStore();
  const now = Date.UTC(2026, 10, 22, 13, 0);
  s.ingest(FDC_9306, now);
  assert.deepEqual({ held: s.stats().tfrs, without_shape: s.stats().tfrs_without_shape, listed: s.active(now).length }, { held: 1, without_shape: 0, listed: 1 });
  // it survives the keep-file round trip with its label
  const back = new NotamStore();
  assert.deepEqual(back.restore(JSON.parse(JSON.stringify(s.snapshot(now))), now), { tfrs: 1, dropped: 0 });
  assert.equal((back.active(now)[0].geometry as { from?: string }).from, "text");
  // and a cancellation, which carries no area in its words, removes it
  s.ingest(FDC_9306.replace("<event:type>N</event:type>", "<event:type>C</event:type>").replace(/<event:text>[^<]*<\/event:text>/, "<event:text>TEMPORARY FLIGHT RESTRICTIONS CANCELED</event:text>"), now);
  assert.deepEqual({ held: s.stats().tfrs, without_shape: s.stats().tfrs_without_shape }, { held: 0, without_shape: 0 });
});

test("which shape wins: geometry as sent, then the text's outline, then the index circle; a notice with no citation line has none", () => {
  const AREA = "TEMPORARY FLIGHT RESTRICTIONS WI AN AREA DEFINED AS 3NM RADIUS OF 370905N0834549W (LOZ090020) SFC-3000FT PURSUANT TO 14 CFR SECTION 91.137(A)(1)";
  const sent = parseNotamPayload(msg({ cls: "FDC", text: AREA, airspace: AIRSPACE }))[0];
  assert.deepEqual(Object.keys(sent.geometry!).sort(), ["kind", "rings"], "geometry as sent is never labeled as read from text");
  const words = parseNotamPayload(msg({ cls: "FDC", text: AREA }))[0].geometry as { kind: string; from?: string; rings: unknown[][] };
  assert.deepEqual({ kind: words.kind, from: words.from, rings: words.rings.length, corners: words.rings[0].length }, { kind: "polygon", from: "text", rings: 1, corners: 73 },
    "the text's own area is drawn in place of the index circle, which only encloses it");
  // an area the reader cannot follow: the index circle, as before
  const cut = parseNotamPayload(msg({ cls: "FDC", text: AREA.replace("0834549W", "0834549") }))[0];
  assert.equal(cut.geometry!.kind, "circle");
  // one area read, one not: what was read is drawn and the other is counted
  const half = parseNotamPayload(bareMsg(msg({ cls: "FDC", text: `${AREA}. WI AN AREA DEFINED AS 380000N0840000W TO 380100N0840000 TO POINT OF ORIGIN` })))[0];
  assert.deepEqual({ from: (half.geometry as { from?: string }).from, unread: (half.geometry as { unread?: number }).unread }, { from: "text", unread: 1 });
  // the text is read only for a flight restriction: the same words on another notice make no shape
  assert.equal(parseNotamPayload(msg({ cls: "DOM", text: AREA }))[0].geometry, null);
  assert.equal(sent.citation, undefined, "the trimmed fixture has no plain-text form");
});

test("an area the text excludes: cut out of the outline when it crosses its edge; `uncut` is on the row only when one could not be cut (2026-11-24)", () => {
  const AREA = "TEMPORARY FLIGHT RESTRICTIONS WI AN AREA DEFINED AS 30NM RADIUS OF 264047N0800327W (PBI094001.6) SFC-17999FT MSL PURSUANT TO 14 CFR SECTION 91.141";
  const plain = parseNotamPayload(msg({ cls: "FDC", text: AREA }))[0].geometry as { rings: unknown[][]; uncut?: number };
  assert.deepEqual({ corners: plain.rings[0].length, has: "uncut" in plain }, { corners: 73, has: false });
  const notched = parseNotamPayload(msg({ cls: "FDC", text: `${AREA}; EXCLUDING AN AREA DEFINED AS 2NM RADIUS OF 261150N0801015W (FLL004007.4) SFC-17999FT MSL` }))[0].geometry as { from?: string; rings: unknown[][]; uncut?: number };
  assert.deepEqual({ from: notched.from, has: "uncut" in notched }, { from: "text", has: false });
  assert.ok(notched.rings[0].length > 73, "the outline carries the notch");
  const inner = parseNotamPayload(msg({ cls: "FDC", text: `${AREA}; EXCLUDING AN AREA DEFINED AS 2NM RADIUS OF 264047N0801327W SFC-3000FT MSL` }))[0].geometry as { rings: unknown[][]; uncut?: number };
  assert.deepEqual({ corners: inner.rings[0].length, uncut: inner.uncut }, { corners: 73, uncut: 1 });
});
