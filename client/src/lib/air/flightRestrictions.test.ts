// flightRestrictions.test — the /data map's Flight restrictions layer, parsed
// from the shape GET /api/data/swim/notam/tfrs answers with. The first row is
// the production answer of 2026-11-21 18:07Z (text trimmed); the others are
// the shapes server/swimNotam.ts can send and the shapes the layer must
// survive.
import test from "node:test";
import assert from "node:assert/strict";
import {
  parseFlightRestrictions, restrictionFeatures, restrictionKind, rulesCited, circleRing, statusWords, feedWords, limitWords,
  periodWords, restrictionCard, agoWords, chipTime, limitChip, drawRestrictions, removeRestrictions,
  RESTRICTION_LAYERS, RESTRICTION_SOURCES, RESTRICTION_CLICK_LAYERS, RESTRICTION_KIND_COLOR, RESTRICTION_KIND_LABEL, RESTRICTION_KIND_ORDER, RULE_WORDS, CIRCLE_POINTS,
  FLIGHT_RESTRICTION_MAX_RING_POINTS, FLIGHT_RESTRICTIONS_MAX_FEATURES, FLIGHT_RESTRICTIONS_HEAP_BUDGET_BYTES,
} from "./flightRestrictions.ts";
import { haversineNm } from "../../../../shared/flightPlanGeometry.ts";

const NOW = Date.UTC(2026, 10, 21, 18, 7, 30);
const H = 3600_000;
const U = { km: (km: number, d = 0) => `${km.toFixed(d)} km`, meters: (m: number) => `${Math.round(m)} m` };

const ZAGREB = {
  id: "LDZL:A4134/2026", series: "A", number: "4134", year: "2026", type: "N", classification: "INTL", location: "LDZL", qcode: "QRTCA",
  issued: NOW - 20 * 60_000, start: NOW - 20 * 60_000, end: NOW + 9 * H, minFl: 0, maxFl: 999, center: [45.8167, 15.95], radiusNm: 2,
  text: "TEMPORARY RESERVED AREA 'ZAGREB CENTAR' ACTIVATED WI FLW COORD: \r\n454910N 0155901E - 454804N 0155912E DUE TO UAS ACTIVITIES.", scenario: "982",
  geometry: { kind: "circle", center: [45.8167, 15.95], radiusNm: 2, approximate: true },
};
const row = (over: Record<string, unknown>) => ({ ...ZAGREB, ...over });
const SQUARE = [[38.0, -77.5], [38.0, -77.0], [38.5, -77.0], [38.5, -77.5], [38.0, -77.5]];
const VIP = row({
  id: "FDC:6/1234", classification: "FDC", location: "KZDC", qcode: null, start: NOW + 2 * H, end: NOW + 5 * H, minFl: 0, maxFl: 180, center: null, radiusNm: null,
  text: "TEMPORARY FLIGHT RESTRICTIONS. PURSUANT TO 14 CFR SECTION 91.141 AND 99.7, AIRCRAFT FLIGHT OPERATIONS ARE PROHIBITED",
  geometry: { kind: "polygon", rings: [SQUARE, SQUARE.map(([la, lo]) => [la + 1, lo])] },
});
const answer = (tfrs: unknown[], over: Record<string, unknown> = {}) => ({
  at: NOW, tfrs, stats: { notams: 423, tfrs: 18, tfrs_without_shape: 0, byClassification: {}, parseErrors: 0 },
  kept: { saved_every_s: 60, last_saved_at: NOW - 30_000, last_save: "ok", restored: null },
  feed: { connected: true, last_message_at: NOW - 12_000 },
  source: "FAA SWIM FNS NOTAM (SCDS, NDRB pre-approved for public release; not for operational use)", ...over,
});

test("the production row: an ICAO temporary restricted area, drawn as the notice's index circle and marked as enclosing", () => {
  const v = parseFlightRestrictions(answer([ZAGREB]))!;
  assert.equal(v.rows.length, 1);
  const r = v.rows[0];
  assert.equal(r.kind, "icao_temporary");
  assert.deepEqual(r.rules, []);
  assert.equal(r.shape, "enclosing");
  assert.equal(r.notice, "LDZL A4134/2026");
  assert.deepEqual(r.center, [15.95, 45.8167], "GeoJSON order: longitude first");
  assert.equal(r.inForce, true);
  assert.equal(r.rings.length, 1);
  assert.equal(r.rings[0].length, CIRCLE_POINTS + 1);
  assert.deepEqual(r.rings[0][0], r.rings[0][CIRCLE_POINTS], "the ring is closed");
  // every corner is 2 nautical miles from the index point
  for (const [lon, lat] of r.rings[0]) assert.ok(Math.abs(haversineNm({ lat: 45.8167, lon: 15.95 }, { lat, lon }) - 2) < 0.01);
  assert.equal(r.text!.includes("\r"), false, "carriage returns from the feed are not shown");
  assert.equal(v.held, 18);
  assert.equal(v.kept, true);
  assert.deepEqual(v.feed, { connected: true, lastMessageAt: NOW - 12_000 });
});

test("kind is read from the section the notice cites, then the Q-code subject, and is never guessed", () => {
  assert.deepEqual(rulesCited("PURSUANT TO 14 CFR SECTION 91.141 AND 99.7 ... 91.141"), ["91.141", "99.7"]);
  assert.equal(restrictionKind("PURSUANT TO 14 CFR 91.137(A)(1)", "QRTCA"), "hazard", "a cited section wins over the Q-code");
  assert.equal(restrictionKind("SECTION 91.138", null), "hazard");
  assert.equal(restrictionKind("SECTION 91.139", null), "hazard");
  assert.equal(restrictionKind("SECTION 91.141", null), "security");
  assert.equal(restrictionKind("SECTION 99.7 SPECIAL SECURITY INSTRUCTIONS", null), "security");
  assert.equal(restrictionKind("SECTION 91.143", null), "space");
  assert.equal(restrictionKind("SECTION 91.145", null), "event");
  assert.equal(restrictionKind("AREA ACTIVATED", "qrtca"), "icao_temporary");
  assert.equal(restrictionKind("TEMPORARY FLIGHT RESTRICTIONS IN EFFECT", null), "not_cited");
  assert.equal(restrictionKind("FREQ 191.137 CHANGED", null), "not_cited", "a number that only contains 91.137 is not a citation");
  assert.equal(restrictionKind(null, null), "not_cited");
  // every kind has a label and its own colour; every rule has words
  assert.equal(new Set(RESTRICTION_KIND_ORDER.map((k) => RESTRICTION_KIND_COLOR[k])).size, RESTRICTION_KIND_ORDER.length);
  for (const k of RESTRICTION_KIND_ORDER) assert.ok(RESTRICTION_KIND_LABEL[k]);
  assert.deepEqual(Object.keys(RULE_WORDS).sort(), ["91.137", "91.138", "91.139", "91.141", "91.143", "91.145", "99.7"]);
});

test("a notice with airspace geometry: its own outline, several outlines are separate areas, the symbol sits at the mean corner", () => {
  const v = parseFlightRestrictions(answer([VIP]))!;
  const r = v.rows[0];
  assert.equal(r.shape, "own");
  assert.equal(r.kind, "security");
  assert.deepEqual(r.rules, ["91.141", "99.7"]);
  assert.equal(r.inForce, false, "starts in 2 hours");
  assert.equal(r.rings.length, 2);
  assert.deepEqual(r.rings[0], [[-77.5, 38], [-77, 38], [-77, 38.5], [-77.5, 38.5], [-77.5, 38]]);
  assert.deepEqual(r.center, [-77.25, 38.25]);
  const f = restrictionFeatures(v.rows);
  assert.equal(f.areas.features.length, 1);
  assert.equal(f.areas.features[0].geometry.type, "MultiPolygon");
  assert.equal((f.areas.features[0].geometry.coordinates as unknown[]).length, 2, "two polygons, not one polygon with a hole");
  assert.deepEqual(f.areas.features[0].properties, { id: "FDC:6/1234", kind: "security", color: RESTRICTION_KIND_COLOR.security, shape: "own", in_force: false });
  assert.deepEqual(f.points.features[0].geometry, { type: "Point", coordinates: [-77.25, 38.25] });
  assert.deepEqual(f.points.features[0].properties, f.areas.features[0].properties);
});

test("order and the feature cap: in force first, then the soonest to start; the cap cuts from the end and is counted", () => {
  const later = row({ id: "B:1", start: NOW + 5 * H });
  const sooner = row({ id: "B:2", start: NOW + 1 * H });
  const v = parseFlightRestrictions(answer([later, sooner, ZAGREB]))!;
  assert.deepEqual(v.rows.map((r) => r.id), ["LDZL:A4134/2026", "B:2", "B:1"]);
  const capped = parseFlightRestrictions(answer([later, sooner, ZAGREB]), 2)!;
  assert.deepEqual(capped.rows.map((r) => r.id), ["LDZL:A4134/2026", "B:2"]);
  assert.equal(capped.cut, 1);
  assert.match(statusWords(capped, NOW), /1 in force, 1 starting within 6 hours of 18 held · 1 more not drawn \(limit 2,000\) · not the FAA's complete list/);
  assert.equal(FLIGHT_RESTRICTIONS_MAX_FEATURES, 2000, "the server holds at most 2,000 (swimNotam NOTAM_MAX_TFRS)");
  assert.ok(FLIGHT_RESTRICTIONS_HEAP_BUDGET_BYTES >= FLIGHT_RESTRICTIONS_MAX_FEATURES * FLIGHT_RESTRICTION_MAX_RING_POINTS * 2 * 8, "the declared budget covers the caps");
});

test("a very long outline is thinned to the corner cap and stays closed", () => {
  const n = 5000;
  const ring = Array.from({ length: n }, (_, i) => [40 + Math.sin((2 * Math.PI * i) / n), -100 + Math.cos((2 * Math.PI * i) / n)]);
  const v = parseFlightRestrictions(answer([row({ id: "BIG:1", geometry: { kind: "polygon", rings: [ring] } })]))!;
  const out = v.rows[0].rings[0];
  assert.equal(out.length, FLIGHT_RESTRICTION_MAX_RING_POINTS + 1);
  assert.deepEqual(out[0], out[out.length - 1]);
});

test("rows that cannot be drawn are counted, never invented", () => {
  const v = parseFlightRestrictions(answer([
    ZAGREB,
    row({ id: "X:1", geometry: null }),
    row({ id: "X:2", geometry: { kind: "blob" } }),
    row({ id: "X:3", geometry: { kind: "circle", center: [45, 15], radiusNm: 0 } }),
    row({ id: "X:4", geometry: { kind: "circle", center: [95, 15], radiusNm: 2 } }),
    row({ id: "X:5", geometry: { kind: "polygon", rings: [[[1, 1], [2, 2]]] } }),
    row({ id: "", geometry: ZAGREB.geometry }),
    ZAGREB, // the same notice twice: drawn once
    null,
  ]))!;
  assert.equal(v.rows.length, 1);
  assert.equal(v.unreadable, 8);
  assert.match(statusWords(v, NOW), /8 could not be read/);
  assert.equal(parseFlightRestrictions(null), null);
  assert.equal(parseFlightRestrictions({ tfrs: "no" }), null);
  assert.equal(parseFlightRestrictions([]), null);
});

test("a circle across the 180th meridian is not torn", () => {
  const ring = circleRing([179.99, 10], 30);
  const lons = ring.map((p) => p[0]);
  assert.ok(Math.max(...lons) - Math.min(...lons) < 2, "longitudes stay continuous with the centre's");
  assert.ok(Math.max(...lons) > 180, "the far side is past 180, not wrapped to -180");
});

test("the status line: counts, what is held, and how old the data is", () => {
  assert.equal(statusWords(parseFlightRestrictions(answer([ZAGREB, VIP]))!, NOW), "1 in force, 1 starting within 6 hours of 18 held · not the FAA's complete list · last notice received 12 s ago");
  // restriction notices that arrived without a shape are counted on the line
  const short = parseFlightRestrictions(answer([ZAGREB], { stats: { notams: 423, tfrs: 18, tfrs_without_shape: 7 } }))!;
  assert.equal(short.withoutShape, 7);
  assert.equal(statusWords(short, NOW), "1 in force, 0 starting within 6 hours of 18 held · 7 restriction notices arrived without a shape and are not drawn · not the FAA's complete list · last notice received 12 s ago");
  assert.match(statusWords(parseFlightRestrictions(answer([], { stats: { tfrs: 3, tfrs_without_shape: 1 } }))!, NOW), /1 restriction notice arrived without a shape and is not drawn/);
  assert.equal(statusWords(parseFlightRestrictions(answer([]))!, NOW), "None in force or starting within 6 hours among the 18 held from the FAA notice feed · not the FAA's complete list · last notice received 12 s ago");
  const quiet = parseFlightRestrictions(answer([], { feed: { connected: true, last_message_at: NOW - 11 * 60_000 } }))!;
  assert.equal(feedWords(quiet, NOW), "notice feed quiet, last notice received 11 min ago");
  const down = parseFlightRestrictions(answer([], { feed: { connected: false, last_message_at: NOW - 3 * H } }))!;
  assert.equal(feedWords(down, NOW), "notice feed disconnected, last notice received 3 h ago");
  const never = parseFlightRestrictions(answer([], { feed: { connected: true, last_message_at: null } }))!;
  assert.equal(feedWords(never, NOW), "no notice received since the server started");
  // an older server that does not send the feed block: the layer says it cannot tell, it does not claim to be live
  const old = parseFlightRestrictions(answer([], { feed: undefined, stats: undefined }))!;
  assert.equal(statusWords(old, NOW), "None in force or starting within 6 hours · not the FAA's complete list · the server does not say when the last notice arrived");
  assert.equal(old.withoutShape, null);
  assert.equal(agoWords(NOW - 89_000, NOW), "89 s ago");
  assert.equal(agoWords(NOW - 90_000, NOW), "2 min ago");
  assert.equal(agoWords(NOW + 5_000, NOW), "0 s ago", "a clock a little ahead never reads negative");
});

test("limits and period wording", () => {
  assert.equal(limitWords(0, U), "surface");
  assert.equal(limitWords(999, U), "unlimited");
  assert.equal(limitWords(85, U), "FL085 (2591 m)");
  assert.equal(limitWords(null, U), "not given");
  const [z, vip] = parseFlightRestrictions(answer([ZAGREB, VIP]))!.rows;
  assert.equal(periodWords(z, NOW), "in force until Nov 22 03:07Z");
  assert.equal(periodWords(vip, NOW), "starts 20:07Z, until 23:07Z");
  assert.equal(periodWords({ ...z, endMs: null }, NOW), "in force, no end published");
});

test("the card: the enclosing circle is said to be larger than the restriction; distances go through the unit formatter", () => {
  const [z, vip] = parseFlightRestrictions(answer([ZAGREB, VIP]))!.rows;
  const c = restrictionCard(z, NOW, U, true);
  assert.equal(c.title, "Flight restriction LDZL A4134/2026");
  assert.equal(c.subtitle, "Temporary restricted area (ICAO notice) · in force until Nov 22 03:07Z");
  assert.deepEqual(c.stats, [
    { label: "Status", value: "active" }, { label: "Ends", value: "Nov 22" },
    { label: "Lower", value: "surface" }, { label: "Upper", value: "unlim." },
  ]);
  // four chips share the card's width: about seven characters each (measured in the harness, which fails on a cut chip)
  for (const x of c.stats) assert.ok(x.value.length <= 7 && x.label.length <= 7, `${x.label}: ${x.value}`);
  assert.equal(chipTime(NOW + H, NOW), "19:07Z");
  assert.equal(chipTime(NOW + 9 * H, NOW), "Nov 22");
  assert.equal(chipTime(null, NOW), "n/a");
  assert.equal(limitChip(null), "n/a");
  const fact = (card: typeof c, label: string) => card.facts.find((f) => f.label === label)?.value;
  assert.equal(fact(c, "Drawn shape"), "a circle of 3.7 km around the notice's index point. It encloses the restriction and is larger than it; the outline itself was not sent as geometry");
  assert.equal(fact(c, "Rule cited"), "none cited; ICAO Q-code subject RT, temporary restricted area");
  assert.equal(fact(c, "Issued by"), "international");
  assert.equal(fact(c, "Q-code, as sent"), "QRTCA");
  assert.match(c.body, /^Notice text, as sent:\nTEMPORARY RESERVED AREA 'ZAGREB CENTAR'/);
  assert.match(c.body, /stays listed until its published end/);
  assert.match(c.body, /This is not the FAA's complete list: .* The FAA's own list is at tfr\.faa\.gov\./);
  assert.match(c.body, /not for flight planning or any operational use\.$/);
  const d = restrictionCard(vip, NOW, U, false);
  assert.equal(fact(d, "Drawn shape"), "the restriction's own outline, as sent (2 areas)");
  assert.equal(fact(d, "Rule cited"), "14 CFR 91.141 (presidential and other parties), 14 CFR 99.7 (special security instructions)");
  assert.equal(fact(d, "Starts"), "20:07Z");
  assert.equal(fact(d, "Q-code, as sent"), undefined);
  assert.deepEqual(d.stats[0], { label: "Starts", value: "20:07Z" });
  assert.equal(d.stats[3].value, "FL180", "the chip is short so it is never cut");
  assert.equal(fact(d, "Upper limit"), "FL180 (5486 m)");
  assert.equal(fact(d, "Lower limit"), "surface");
  assert.match(d.body, /The list starts again when the server restarts/);
  assert.doesNotMatch(c.body + d.body + JSON.stringify(c.facts) + JSON.stringify(d.facts), /undefined|NaN|\bnull\b/);
});

// a map that records what is on it, the way MapLibre holds sources and layers
function fakeMap() {
  const sources = new Map<string, { data: unknown; setData(d: unknown): void }>();
  const layers = new Map<string, Record<string, unknown>>();
  let broken = false;
  return {
    sources, layers, breakIt: () => { broken = true; },
    getLayer: (id: string) => layers.get(id),
    removeLayer: (id: string) => { if (broken) throw new Error("map removed"); layers.delete(id); },
    getSource: (id: string) => sources.get(id),
    removeSource: (id: string) => {
      for (const l of layers.values()) if (l.source === id) throw new Error(`source ${id} still used by ${String(l.id)}`);
      sources.delete(id);
    },
    addSource: (id: string, spec: unknown) => { const src = { data: (spec as { data: unknown }).data, setData(d: unknown) { src.data = d; } }; sources.set(id, src); },
    addLayer: (spec: unknown) => { const l = spec as Record<string, unknown>; if (!sources.has(String(l.source))) throw new Error("no source"); layers.set(String(l.id), l); },
  };
}

test("drawing: two sources and five layers once, data replaced after that, everything removed on teardown", () => {
  const m = fakeMap();
  const v = parseFlightRestrictions(answer([ZAGREB, VIP]))!;
  assert.equal(drawRestrictions(m, v.rows), true, "the first draw adds the layers");
  assert.deepEqual([...m.sources.keys()], [...RESTRICTION_SOURCES]);
  assert.deepEqual([...m.layers.keys()], [...RESTRICTION_LAYERS]);
  // SYMBOLS NOT DOTS: the point layer is a symbol drawing the registered glyph, never a circle
  const sym = m.layers.get("flight-restrictions-sym")!;
  assert.equal(sym.type, "symbol");
  assert.equal((sym.layout as Record<string, unknown>)["icon-image"], "vt-airspace");
  for (const l of m.layers.values()) assert.notEqual(l.type, "circle");
  // the index circle is dashed, the restriction's own outline is not
  assert.deepEqual((m.layers.get("flight-restrictions-line-enclosing")!.paint as Record<string, unknown>)["line-dasharray"], [3, 2]);
  assert.equal((m.layers.get("flight-restrictions-line")!.paint as Record<string, unknown>)["line-dasharray"], undefined);
  assert.deepEqual(m.layers.get("flight-restrictions-line-enclosing")!.filter, ["==", ["get", "shape"], "enclosing"]);
  assert.deepEqual(m.layers.get("flight-restrictions-line")!.filter, ["==", ["get", "shape"], "own"]);
  // an outline read from the notice's text is dotted: a third style, never the solid "as sent" line
  assert.deepEqual((m.layers.get("flight-restrictions-line-text")!.paint as Record<string, unknown>)["line-dasharray"], [0.6, 1.6]);
  assert.deepEqual(m.layers.get("flight-restrictions-line-text")!.filter, ["==", ["get", "shape"], "text"]);
  assert.ok(RESTRICTION_CLICK_LAYERS.includes("flight-restrictions-line-text"));
  // the tinted inside does not take clicks
  assert.ok(!RESTRICTION_CLICK_LAYERS.includes("flight-restrictions-fill"));
  // a refresh replaces the data and adds nothing
  assert.equal(drawRestrictions(m, [v.rows[0]]), false);
  assert.equal(m.layers.size, 5);
  assert.equal((m.sources.get("flight-restrictions-pts")!.data as { features: unknown[] }).features.length, 1);
  // teardown: layers first (a source in use cannot be removed), then the sources
  assert.equal(removeRestrictions(m), true);
  assert.equal(m.layers.size + m.sources.size, 0);
  assert.equal(removeRestrictions(m), true, "removing twice is harmless");
  assert.equal(removeRestrictions(null), true);
  // a map already torn down refuses: reported, not thrown
  drawRestrictions(m, v.rows);
  m.breakIt();
  assert.equal(removeRestrictions(m), false);
});

// The production notice of 2026-11-21 23:27Z, FDC 6/9306, as server/swimNotam.ts
// now serves it: no index point, no index limits, no Q-code, the outline read
// from its words (server/swimNotam.test.ts parses the real message to this).
const LEXINGTON = {
  id: "KZDC:9306/", citation: "FDC 6/9306", series: null, number: "9306", year: null, type: "N", classification: "FDC", location: "KZDC", qcode: null,
  issued: NOW - H, start: NOW - H, end: NOW + 8 * 24 * H, minFl: null, maxFl: null, center: null, radiusNm: null, scenario: "986",
  text: "VA..AIRSPACE 20NM W LEXINGTON, VA..TEMPORARY FLIGHT RESTRICTIONS. PURSUANT TO 14 CFR SECTION 91.137(A)(2) TEMPORARY FLIGHT RESTRICTIONS ARE IN EFFECT TO PROVIDE A SAFE ENVIRONMENT FOR FIRE FIGHTING ACFT OPS. WI AN AREA DEFINED AS 375200N0795230W (MOL272036.5) TO 375100N0795000W (MOL270034.6) TO 374830N0795200W (MOL266036.4) TO 374900N0795400W (MOL268037.9) TO POINT OF ORIGIN SFC-4000FT",
  geometry: { kind: "polygon", from: "text", unread: 0, rings: [[[37.866667, -79.875], [37.85, -79.833333], [37.808333, -79.866667], [37.816667, -79.9], [37.866667, -79.875]]] },
};

test("a US notice whose outline was read from its text: its own third shape, cited as people cite it, and the card says how it was drawn", () => {
  const v = parseFlightRestrictions(answer([LEXINGTON, VIP, ZAGREB]))!;
  assert.equal(v.unreadable, 0);
  const r = v.rows.find((x) => x.id === "KZDC:9306/")!;
  assert.equal(r.shape, "text", "never 'own': the notice sent no geometry");
  assert.equal(r.notice, "FDC 6/9306");
  assert.equal(r.kind, "hazard");
  assert.equal(r.unreadAreas, 0);
  assert.deepEqual(r.rings[0][0], [-79.875, 37.866667], "GeoJSON order");
  assert.ok(Math.abs(r.center[0] - -79.86875) < 1e-4 && Math.abs(r.center[1] - 37.835417) < 1e-4, "no index point: the symbol sits at the mean corner");
  // the other two shapes keep their names, and no other row gains unread areas
  assert.deepEqual(v.rows.map((x) => [x.notice, x.shape, x.unreadAreas]).sort(), [["FDC 6/1234", "own", 0], ["FDC 6/9306", "text", 0], ["LDZL A4134/2026", "enclosing", 0]]);
  const f = restrictionFeatures(v.rows).areas.features.find((x) => x.properties.id === "KZDC:9306/")!;
  assert.equal(f.properties.shape, "text");
  const c = restrictionCard(r, NOW, U, true);
  const fact = (label: string) => c.facts.find((x) => x.label === label)?.value;
  assert.equal(c.title, "Flight restriction FDC 6/9306");
  assert.equal(fact("Drawn shape"), "an outline read from the notice's text; the notice sent no geometry. Circles and arcs are drawn as short straight sides");
  assert.equal(fact("Rule cited"), "14 CFR 91.137 (disaster or hazard area)");
  assert.equal(fact("Lower limit"), "not given");
  assert.equal(fact("Issued by"), "FAA Flight Data Center");
  assert.match(c.body, /This notice has no index limits; its lower and upper limits are written in its text\. /);
  assert.doesNotMatch(c.body, /index values/);
  assert.match(c.body, /SFC-4000FT/, "the limits are in the notice text the card prints");
  for (const x of c.stats) assert.ok(x.value.length <= 7 && x.label.length <= 7, `${x.label}: ${x.value}`);
  assert.doesNotMatch(c.body + JSON.stringify(c.facts), /undefined|NaN|\bnull\b/);
  // areas the text defines that could not be followed are said, with the count
  const part = parseFlightRestrictions(answer([{ ...LEXINGTON, geometry: { ...LEXINGTON.geometry, unread: 2, rings: [LEXINGTON.geometry.rings[0], SQUARE] } }]))!.rows[0];
  assert.equal(part.unreadAreas, 2);
  assert.equal(restrictionCard(part, NOW, U, true).facts.find((x) => x.label === "Drawn shape")!.value,
    "an outline read from the notice's text (2 areas); the notice sent no geometry. Circles and arcs are drawn as short straight sides. 2 more areas in the text could not be read and are not drawn");
  // a server that does not send the citation: the id, as before
  const { citation, ...old } = LEXINGTON;
  assert.equal(citation, "FDC 6/9306");
  assert.equal(parseFlightRestrictions(answer([old]))!.rows[0].notice, "KZDC 9306/");
});
