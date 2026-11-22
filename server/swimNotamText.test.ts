// swimNotamText.test — the outline read from a flight-restriction notice's
// words. Every notice text here is REAL (the FAA's own, read 2026-11-22),
// and the expected shapes are the FAA's own published polygons for the same
// notices (server/__fixtures__/tfr_text_gate1_sample.json), compared with
// the gate-1 check's own measure (scripts/tfr_text_gate1.ts).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { textAreas, TEXT_ARC_STEP_DEG, TEXT_MAX_AREAS } from "./swimNotamText";
import { judgeNotice, outlineDistanceNm, toleranceNm, detailText, shapesByNotice, type Ring } from "../scripts/tfr_text_gate1";
import { haversineNm } from "../shared/flightPlanGeometry";

interface Sample { notice: string; type: string; text: string; faa: Ring[] }
const SAMPLE = (JSON.parse(fs.readFileSync(new URL("./__fixtures__/tfr_text_gate1_sample.json", import.meta.url), "utf8")) as { notices: Sample[] }).notices;
const by = (n: string) => SAMPLE.find((s) => s.notice === n)!;
const ll = (p: [number, number]) => ({ lat: p[0], lon: p[1] });

test("FDC 6/9306, the notice production dropped on 2026-11-22: four corners TO POINT OF ORIGIN, exactly the FAA's polygon", () => {
  const s = by("6/9306");
  const t = textAreas(s.text);
  assert.deepEqual({ areas: t.areasInText, unread: t.unread, rings: t.rings.length }, { areas: 1, unread: 0, rings: 1 });
  assert.deepEqual(t.rings[0], [[37.866667, -79.875], [37.85, -79.833333], [37.808333, -79.866667], [37.816667, -79.9], [37.866667, -79.875]]);
  assert.ok(outlineDistanceNm(s.faa[0], t.rings[0]) < 0.001, "the FAA's own polygon has the same four corners");
  assert.equal(judgeNotice(s.notice, s.type, s.faa, s.text).verdict, "agree");
});

test("a radius: every corner is that far from the point, the outline is closed, and it lies on the FAA's 36-sided circle", () => {
  const s = by("6/2478"); // "5NM RADIUS OF 311647N1002234W"
  const t = textAreas(s.text);
  assert.equal(t.rings.length, 1);
  const ring = t.rings[0];
  assert.equal(ring.length, 360 / TEXT_ARC_STEP_DEG + 1);
  assert.deepEqual(ring[0], ring[ring.length - 1]);
  const c = { lat: 31 + 16 / 60 + 47 / 3600, lon: -(100 + 22 / 60 + 34 / 3600) };
  for (const p of ring) assert.ok(Math.abs(haversineNm(c, ll(p)) - 5) < 0.002);
  const d = outlineDistanceNm(s.faa[0], ring);
  assert.ok(d <= toleranceNm(s.faa[0]), `distance ${d}`);
  assert.equal(judgeNotice(s.notice, s.type, s.faa, s.text).verdict, "agree");
});

test("arcs, clockwise and counterclockwise, and several areas in one notice: each FAA polygon has an outline within tolerance", () => {
  for (const n of ["6/2520", "6/4116", "6/2481"]) {
    const s = by(n);
    const r = judgeNotice(s.notice, s.type, s.faa, s.text);
    assert.equal(r.verdict, "agree", `${n}: ${JSON.stringify(r)}`);
    assert.equal(r.read, s.faa.length, `${n}: one outline per FAA polygon`);
    assert.ok(r.worst_matched_nm! < 0.1, `${n}: ${r.worst_matched_nm} NM`);
  }
  // FDC 6/2481 writes five areas; two are the same footprint at different heights: one outline, as the FAA publishes it
  assert.equal(textAreas(by("6/2481").text).areasInText, 5);
  assert.equal(textAreas(by("6/2481").text).rings.length, 4);
  // the direction matters: the same words with the turn reversed are a different outline, and the measure says so
  const s = by("6/2520");
  const flipped = s.text.replace("THEN CLOCKWISE", "THEN COUNTERCLOCKWISE");
  assert.notEqual(flipped, s.text);
  assert.notEqual(judgeNotice(s.notice, s.type, s.faa, flipped).verdict, "agree");
});

test("what cannot be followed is left unread and counted, never guessed", () => {
  // real: seconds of 68 in the longitude ("295612N0900368W"); the FAA drew something for it, we do not
  const bad = textAreas(by("6/9071").text);
  assert.deepEqual({ areas: bad.areasInText, unread: bad.unread, rings: bad.rings.length }, { areas: 1, unread: 1, rings: 0 });
  // real: "WI AREA DEFINED AS" without "AN" is not the wording this reader reads (filed, not tuned to fit)
  assert.deepEqual(textAreas(by("6/9287").text), { rings: [], areasInText: 0, unread: 0 });
  const A = "TEMPORARY FLIGHT RESTRICTIONS WI AN AREA DEFINED AS ";
  const unread = (words: string) => assert.deepEqual(textAreas(A + words), { rings: [], areasInText: 1, unread: 1 }, words);
  unread("375200N0795230W TO 375100N0795000W TO 374830N0795200W SFC-4000FT"); // never returns to the point of origin
  unread("375200N0795230W TO 375100N0795000W TO 2611221200-2611292300 END PART 1 OF 2 !FDC 6/9306 ZDC PART 2 OF 2 374830N0795200W TO POINT OF ORIGIN"); // cut by a part break
  unread("375200N0795230W TO 375100N0795000 TO 374830N0795200W TO POINT OF ORIGIN"); // a point without its E or W
  unread("375200N0795230W TO POINT OF ORIGIN"); // one point is not an area
  unread("375200N0795230W TO 375100N0795000W TO POINT OF ORIGIN"); // nor are two
  unread("THE DCA VOR/DME 299 DEGREE RADIAL AT 30 NM TO POINT OF ORIGIN"); // a navaid radial
  unread("0NM RADIUS OF 375200N0795230W"); // no radius
  unread("900NM RADIUS OF 375200N0795230W"); // not a flight restriction's radius
  unread("375200N0795230W THEN CLOCKWISE ON A 5NM ARC CENTERED ON 375100N0795000W SFC-4000FT"); // an arc to nowhere
  // other wordings and non-text are simply not areas
  assert.deepEqual(textAreas("WITHIN A 1 NMR OF 325321N/0964835W OR THE CVE085004.8"), { rings: [], areasInText: 0, unread: 0 });
  assert.deepEqual(textAreas("ACTIVATED WI FLW COORD: 454910N 0155901E - 454804N 0155912E - 454759N 0155736E"), { rings: [], areasInText: 0, unread: 0 });
  for (const v of [null, undefined, "", 7 as unknown as string]) assert.deepEqual(textAreas(v), { rings: [], areasInText: 0, unread: 0 });
});

test("hemispheres, lower case, line breaks and the cap on areas", () => {
  const t = textAreas("temporary flight restrictions wi an area\r\ndefined as 1NM RADIUS OF 334700S1511500E (SY000001) sfc-400ft");
  assert.equal(t.rings.length, 1);
  assert.ok(t.rings[0].every(([la, lo]) => la < -33.7 && la > -33.9 && lo > 151.2 && lo < 151.3));
  const many = Array.from({ length: TEXT_MAX_AREAS + 3 }, (_, i) => `WI AN AREA DEFINED AS 1NM RADIUS OF 33${String(10 + i)}00N1191500W SFC-400FT.`).join(" ");
  const m = textAreas("TEMPORARY FLIGHT RESTRICTIONS " + many);
  assert.deepEqual({ areas: m.areasInText, rings: m.rings.length, unread: m.unread }, { areas: TEXT_MAX_AREAS + 3, rings: TEXT_MAX_AREAS, unread: 3 });
});

test("the gate-1 check's own parts: the notice text out of a detail file, shapes by notice number, the four verdicts", () => {
  assert.equal(detailText("<Not><txtDescrUSNS>!FDC 6/1 A &amp; B &lt;1NM&gt;</txtDescrUSNS></Not>"), "!FDC 6/1 A & B <1NM>");
  assert.equal(detailText("<Not></Not>"), null);
  const shapes = shapesByNotice({ features: [
    { properties: { NOTAM_KEY: "6/9306-1-FDC-F" }, geometry: { type: "Polygon", coordinates: [[[-79.9, 37.8], [-79.8, 37.8], [-79.8, 37.9], [-79.9, 37.8]]] } },
    { properties: { NOTAM_KEY: "6/9306-2-FDC-F" }, geometry: { type: "MultiPolygon", coordinates: [[[[-70, 30], [-69, 30], [-69, 31], [-70, 30]]]] } },
    { properties: {}, geometry: { type: "Polygon", coordinates: [] } },
  ] });
  assert.equal(shapes.get("6/9306")!.length, 2);
  assert.deepEqual(shapes.get("6/9306")![0][0], [37.8, -79.9], "corners become [lat, lon]");
  const s = by("6/9306");
  const far: Ring = s.faa[0].map(([la, lo]) => [la + 1, lo]);
  assert.equal(judgeNotice("x", null, [far], s.text).verdict, "disagree");
  assert.equal(judgeNotice("x", null, [s.faa[0], far], s.text).verdict, "partial");
  assert.equal(judgeNotice("x", null, s.faa, "NO AREA WORDS HERE AT ALL, ONLY A NOTICE OF SOMETHING").verdict, "not_read");
  assert.ok(Math.abs(toleranceNm(s.faa[0]) - 0.1) < 0.02, "a shape a few miles across is held to about 0.1 NM");
});
