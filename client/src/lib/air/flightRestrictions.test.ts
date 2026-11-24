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
  listWords, listOk, textCheckWords, faaLimitWords, faaLimitChip, FEED_ROW_WITH_LIST_SENTENCE, LIST_RECORD_SENTENCE, listedSinceWords,
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
  // areas the text excludes that the server could not cut out of the outline are said, with the count; none sent = nothing said
  assert.equal(part.uncutAreas, 0);
  const whole = parseFlightRestrictions(answer([{ ...LEXINGTON, geometry: { ...LEXINGTON.geometry, uncut: 1 } }]))!.rows[0];
  assert.equal(whole.uncutAreas, 1);
  assert.equal(restrictionCard(whole, NOW, U, true).facts.find((x) => x.label === "Drawn shape")!.value,
    "an outline read from the notice's text; the notice sent no geometry. Circles and arcs are drawn as short straight sides. The text excludes 1 area that could not be cut out of this outline: the restriction is smaller than drawn there");
  for (const bad of [-1, "2", null]) assert.equal(parseFlightRestrictions(answer([{ ...LEXINGTON, geometry: { ...LEXINGTON.geometry, uncut: bad } }]))!.rows[0].uncutAreas, 0);
  // a server that does not send the citation: the id, as before
  const { citation, ...old } = LEXINGTON;
  assert.equal(citation, "FDC 6/9306");
  assert.equal(parseFlightRestrictions(answer([old]))!.rows[0].notice, "KZDC 9306/");
});

// ── rows read from the FAA's public list (server/faaTfrList.ts, 2026-11-22) ──
// The row is what the server builds from the FAA's real detail file for
// FDC 6/9306 (server/__fixtures__/faa_tfr_list/detail_6_9306.xml); the second
// is 6/9276's four areas with their limits as that file writes them.
const LIST_ROW = {
  id: "FAA:6/9306", source: "faa_list", citation: "FDC 6/9306", classification: "FDC", series: null, number: "6/9306", year: null, type: null, qcode: null, scenario: null,
  location: "ZDC", issued: NOW - 19 * H, start: NOW - 6 * H, end: NOW + 7 * 24 * H, minFl: null, maxFl: null, center: null, radiusNm: null,
  text: "!FDC 6/9306 ZDC VA..AIRSPACE 20NM W LEXINGTON, VA..TEMPORARY FLIGHT RESTRICTIONS WI AN AREA DEFINED AS 375200N0795230W ... PURSUANT TO 14 CFR SECTION 91.137(A)(2)",
  geometry: { kind: "polygon", from: "faa_list", rings: [[[37.866667, -79.875], [37.85, -79.833333], [37.808333, -79.866667], [37.816667, -79.9], [37.866667, -79.875]]] },
  faa: { type: "HAZARDS", rule: "91.137(a)(2)", place: "20NM W LEXINGTON, VIRGINIA", areas_without_outline: 0,
    areas: [{ name: "Hazard Area1", lower: { value: 0, uom: "FT", ref: "ALT" }, upper: { value: 4000, uom: "FT", ref: "ALT" }, scheduled: true }] },
};
const MAYPORT = {
  ...LIST_ROW, id: "FAA:6/9276", citation: "FDC 6/9276", number: "6/9276", location: "ZJX", start: NOW + 2 * H, end: NOW + 3 * 24 * H,
  text: "!FDC 6/9276 ZJX FL..AIRSPACE MAYPORT, FL..TEMPORARY FLIGHT RESTRICTIONS. PURSUANT TO 49 USC 40103(B)(3) ... 14 CFR 99.7",
  geometry: { kind: "polygon", from: "faa_list", rings: [SQUARE, SQUARE.map(([la, lo]) => [la + 1, lo])] },
  faa: { type: "SECURITY", rule: "99.7", place: "MAYPORT, FLORIDA", areas_without_outline: 1, areas: [
    { name: "Area A", lower: { value: 0, uom: "FT", ref: "HEI" }, upper: { value: 2500, uom: "FT", ref: "ALT" }, scheduled: false },
    { name: null, lower: { value: 320, uom: "FL", ref: "ALT" }, upper: { value: 430, uom: "FL", ref: "ALT" }, scheduled: false },
    { name: "Area E", lower: { value: 1200, uom: "FT", ref: "ALT" }, upper: null, scheduled: false },
  ] },
};
const LIST_OK = { status: "ok", read_at: NOW - 4 * 60_000, last_attempt_at: NOW - 4 * 60_000, last_cause: "ok", listed: 48, with_outline: 36, without_outline: 12, detail_pending: 0, also_in_feed: 1, not_drawn: [], restored: false, served: 2 };
const listAnswer = (tfrs: unknown[], faa_list: unknown) => ({ at: NOW, tfrs, stats: { tfrs: 230, tfrs_without_shape: 0 }, kept: {}, feed: { connected: true, last_message_at: NOW - 20_000 }, faa_list });

test("a row from the FAA's public list: its own fourth shape, the FAA's outline, and what the FAA's file says about it", () => {
  const v = parseFlightRestrictions(listAnswer([LIST_ROW, MAYPORT, ZAGREB], LIST_OK))!;
  assert.equal(v.unreadable, 0);
  const r = v.rows.find((x) => x.id === "FAA:6/9306")!;
  assert.equal(r.shape, "faa");
  assert.equal(r.notice, "FDC 6/9306");
  assert.equal(r.kind, "hazard");
  assert.equal(r.inForce, true);
  assert.deepEqual(r.rings[0][0], [-79.875, 37.866667]);
  assert.deepEqual(r.faa, { type: "HAZARDS", rule: "91.137(a)(2)", place: "20NM W LEXINGTON, VIRGINIA", areasWithoutOutline: 0,
    areas: [{ name: "Hazard Area1", lower: { value: 0, uom: "FT", ref: "ALT" }, upper: { value: 4000, uom: "FT", ref: "ALT" }, scheduled: true }] });
  // a feed row never carries the list's facts
  assert.equal(v.rows.find((x) => x.id === "LDZL:A4134/2026")!.faa, null);
  assert.deepEqual(v.faaList, { status: "ok", readAt: NOW - 4 * 60_000, listed: 48, withOutline: 36, withoutOutline: 12, pending: 0, alsoInFeed: 1, served: 2, textCheck: null });
  // drawn on the solid line with the outlines the feed sent, never on the dotted or dashed one; its source rides apart
  const f = restrictionFeatures(v.rows).areas.features.find((x) => x.properties.id === "FAA:6/9306")!;
  assert.equal(f.properties.shape, "own");
  assert.equal(f.properties.from_list, true);
  assert.equal(restrictionFeatures(v.rows).areas.features.find((x) => x.properties.id === "LDZL:A4134/2026")!.properties.from_list, undefined);
});

test("the status line with the FAA's list read: when, how many listed, how many have no outline, and what is still feed-only", () => {
  const v = parseFlightRestrictions(listAnswer([LIST_ROW, MAYPORT, ZAGREB], LIST_OK))!;
  const s = statusWords(v, NOW);
  assert.equal(s, "2 in force, 1 starting within 6 hours (2 from the FAA's public list, 1 from the notice feed, which holds 230)"
    + " · FAA public list read 4 min ago: 48 US restrictions listed, 36 with a published outline, 12 with none (not drawn) · other countries: only what the notice feed has sent"
    + " · last notice received 20 s ago");
  assert.ok(!s.includes("not the FAA's complete list"));
  assert.match(listWords({ ...v.faaList!, pending: 3, withoutOutline: 0 }, NOW)!, /36 with a published outline, 3 not yet read ·/);
});

test("the status line without a usable read of the FAA's list says the layer is not complete, and why", () => {
  const base = parseFlightRestrictions(listAnswer([ZAGREB], LIST_OK))!.faaList!;
  assert.equal(listWords({ ...base, status: "never", readAt: null }, NOW), "not the FAA's complete list: the FAA's public list has not been read yet");
  assert.equal(listWords({ ...base, status: "unreachable", readAt: null }, NOW), "not the FAA's complete list: the FAA's public list could not be reached");
  assert.equal(listWords({ ...base, status: "stale", readAt: NOW - 30 * H }, NOW), "not the FAA's complete list: the FAA's public list was last read 30 h ago, too long ago to show");
  assert.equal(listWords({ ...base, status: "off" }, NOW), "not the FAA's complete list: the FAA's public list is not being read");
  assert.equal(listOk({ ...base, status: "stale" }), false);
  assert.equal(listOk(null), false);
  // an older server (no faa_list in the answer): the words the layer has always said
  const old = parseFlightRestrictions({ ...listAnswer([ZAGREB], undefined) })!;
  assert.equal(old.faaList, null);
  assert.equal(listWords(null, NOW), null);
  assert.ok(statusWords(old, NOW).includes(" · not the FAA's complete list · "));
  // a status the client does not know is not read as ok
  assert.equal(parseFlightRestrictions(listAnswer([ZAGREB], { ...LIST_OK, status: "fine" }))!.faaList, null);
});

test("the list row's card: named as the FAA's list, the FAA's outline, limits as the FAA's file writes them through the unit formatter", () => {
  const v = parseFlightRestrictions(listAnswer([LIST_ROW, MAYPORT], LIST_OK))!;
  const one = restrictionCard(v.rows.find((x) => x.id === "FAA:6/9306")!, NOW, U, true, v.faaList);
  assert.equal(one.title, "Flight restriction FDC 6/9306");
  assert.equal(one.sourceTag, "FAA public list");
  const fact = (c: typeof one, label: string) => c.facts.find((f) => f.label === label)?.value;
  assert.equal(fact(one, "Drawn shape"), "the FAA's published outline for this notice, copied from tfr.faa.gov; it was not sent by the notice feed");
  assert.equal(fact(one, "FAA list type"), "HAZARDS");
  assert.equal(fact(one, "Place"), "20NM W LEXINGTON, VIRGINIA");
  assert.equal(fact(one, "Lower limit"), "surface");
  assert.equal(fact(one, "Upper limit"), "1219 m above sea level");
  assert.equal(fact(one, "Hours"), "certain hours only, written in the notice text; listed here for its whole period");
  assert.equal(fact(one, "Rule cited"), "14 CFR 91.137 (disaster or hazard area)");
  assert.deepEqual(one.stats.map((s) => s.value).slice(2), ["surface", "4000ft"]);
  assert.match(one.body, /^Notice text, as the FAA publishes it:\n!FDC 6\/9306/);
  assert.match(one.body, /FAA public list of flight restrictions \(tfr\.faa\.gov\), read 4 min ago\. The outline and the limits are the FAA's own, copied as published\./);
  assert.ok(!one.body.includes("notice feed (SWIM"), "a list row is never called a feed row");
  assert.match(one.body, /not for flight planning or any operational use\.$/);

  const many = restrictionCard(v.rows.find((x) => x.id === "FAA:6/9276")!, NOW, U, true, v.faaList);
  assert.equal(fact(many, "Area A"), "surface to 762 m above sea level");
  assert.equal(fact(many, "Area 2"), "FL320 (9754 m) to FL430 (13106 m)");
  assert.equal(fact(many, "Area E"), "366 m above sea level to not given");
  assert.equal(fact(many, "Hours"), undefined);
  assert.equal(fact(many, "Drawn shape"), "the FAA's published outline for this notice (2 areas), copied from tfr.faa.gov; it was not sent by the notice feed. 1 more area of this notice has no published outline and is not drawn");
  assert.deepEqual(many.stats.map((s) => s.value).slice(2), ["varies", "varies"]);
  assert.equal(many.stats[0].label, "Starts");

  assert.equal(faaLimitWords({ value: 3000, uom: "FT", ref: "HEI" }, U), "914 m above the ground");
  assert.equal(faaLimitWords({ value: 3000, uom: "FT", ref: null }, U), "914 m (reference not given)");
  assert.equal(faaLimitChip(null), "n/a");
  assert.equal(faaLimitChip({ value: 430, uom: "FL", ref: "ALT" }), "FL430");
});

test("a feed row's card says where it stands against the FAA's list only when that list was read", () => {
  const v = parseFlightRestrictions(listAnswer([ZAGREB], LIST_OK))!;
  const withList = restrictionCard(v.rows[0], NOW, U, true, v.faaList);
  assert.ok(withList.body.includes(FEED_ROW_WITH_LIST_SENTENCE));
  assert.ok(!withList.body.includes("This is not the FAA's complete list"));
  for (const list of [null, { ...v.faaList!, status: "unreachable" as const, readAt: null }]) {
    const without = restrictionCard(v.rows[0], NOW, U, true, list);
    assert.ok(without.body.includes("This is not the FAA's complete list"));
    assert.ok(!without.body.includes(FEED_ROW_WITH_LIST_SENTENCE));
  }
  // the fourth argument alone (the call every older caller makes) is unchanged
  assert.equal(restrictionCard(v.rows[0], NOW, U, true).body, restrictionCard(v.rows[0], NOW, U, true, null).body);
});

// ── since when the server's record of the FAA's list has held a notice (server/faaTfrArchive.ts, 2026-11-22) ──
test("a list row says since when it has been on the FAA's list, as this server's record has it; a row the server says nothing for is unchanged", () => {
  const first = NOW - 26 * H; // 2026-11-20 16:07Z
  const v = parseFlightRestrictions(listAnswer([
    { ...LIST_ROW, listed_since: { first_seen: first, backfill: false } },
    { ...MAYPORT, listed_since: { first_seen: NOW - 3 * H, backfill: true } },
  ], LIST_OK))!;
  const seen = v.rows.find((x) => x.id === "FAA:6/9306")!, began = v.rows.find((x) => x.id === "FAA:6/9276")!;
  assert.deepEqual(seen.listedSince, { firstSeenMs: first, backfill: false });
  assert.deepEqual(began.listedSince, { firstSeenMs: NOW - 3 * H, backfill: true });
  const fact = (r: typeof seen) => restrictionCard(r, NOW, U, true, v.faaList).facts.find((f) => f.label === "On the FAA's list")?.value;
  assert.equal(fact(seen), "first seen there Nov 20 16:07Z, by this server's read of the list (every 30 minutes)");
  // already listed when the record began: the time is the record's start, and the card does not call it the notice's
  assert.equal(fact(began), "already listed when this server's record of the list began, 15:07Z");
  assert.ok(restrictionCard(seen, NOW, U, true, v.faaList).body.includes(LIST_RECORD_SENTENCE));
  // another year is written out (a standing restriction stays listed for years)
  assert.equal(listedSinceWords({ firstSeenMs: Date.UTC(2025, 2, 4, 9, 30), backfill: false }, NOW), "first seen there Mar 4 09:30Z, 2025, by this server's read of the list (every 30 minutes)");

  // an older server, or a notice the record does not hold: no fact, no sentence, the card as it was
  const plain = parseFlightRestrictions(listAnswer([LIST_ROW], LIST_OK))!.rows[0];
  assert.equal(plain.listedSince, null);
  const card = restrictionCard(plain, NOW, U, true, v.faaList);
  assert.equal(card.facts.some((f) => f.label === "On the FAA's list"), false);
  assert.ok(!card.body.includes(LIST_RECORD_SENTENCE));
  // a malformed value is not a time; a feed row never carries one
  for (const bad of [null, "then", {}, { first_seen: "x" }, { first_seen: 0 }, { first_seen: -5, backfill: true }]) {
    assert.equal(parseFlightRestrictions(listAnswer([{ ...LIST_ROW, listed_since: bad }], LIST_OK))!.rows[0].listedSince, null, JSON.stringify(bad));
  }
  assert.equal(parseFlightRestrictions(listAnswer([{ ...ZAGREB, listed_since: { first_seen: first, backfill: false } }], LIST_OK))!.rows[0].listedSince, null);
});

test("the text reader's running check: said only when a notice has been in both sources, in counts, on the status line and on a text-drawn row's card", () => {
  const SINCE = NOW - 26 * H;
  const check = (o: Record<string, unknown>) => ({ since: SINCE, last_judged_at: NOW - H, compared: 0, agree: 0, partial: 0, disagree: 0, not_read: 0, feed_sent_geometry: 0, worst_agree_nm: null, tolerance: "t", not_agreeing: [], dropped: 0, ...o });
  const view = (tc: unknown, rows: unknown[] = [LIST_ROW]) => parseFlightRestrictions(listAnswer(rows, { ...LIST_OK, text_check: tc }))!;
  // an older server, a malformed block, and a check that has judged nothing yet: nothing is said
  for (const tc of [undefined, null, "x", { agree: "8" }, { agree: 1, partial: 0, disagree: 0 }, { agree: -1, partial: 0, disagree: 0, not_read: 0 }]) assert.equal(view(tc).faaList!.textCheck, null, JSON.stringify(tc));
  const none = view(check({}));
  assert.deepEqual(none.faaList!.textCheck, { sinceMs: SINCE, compared: 0, agree: 0, partial: 0, disagree: 0, notRead: 0, worstAgreeNm: null });
  assert.equal(textCheckWords(none.faaList, NOW), null);
  assert.equal(textCheckWords(null, NOW), null);
  assert.ok(!statusWords(none, NOW).includes("checked against"));

  const all = view(check({ compared: 8, agree: 8, worst_agree_nm: 0.031 }));
  assert.deepEqual(all.faaList!.textCheck, { sinceMs: SINCE, compared: 8, agree: 8, partial: 0, disagree: 0, notRead: 0, worstAgreeNm: 0.031 });
  const w = textCheckWords(all.faaList, NOW)!;
  assert.match(w, /^our reading of notice text checked against the FAA's outline on 8 notices since \S.*Z: 8 of 8 read match$/);
  // on the status line between the list's words and the feed's
  assert.match(statusWords(all, NOW), / · other countries: only what the notice feed has sent · our reading of notice text checked .*: 8 of 8 read match · last notice received 20 s ago$/);

  // the server's `compared` is not trusted over its own parts
  const mixed = view(check({ compared: 99, agree: 5, partial: 1, disagree: 1, not_read: 2 }));
  assert.equal(mixed.faaList!.textCheck!.compared, 7);
  assert.match(textCheckWords(mixed.faaList, NOW)!, /on 9 notices since .*: 5 of 7 read match, 1 in part, 1 differs, 2 not read$/);
  assert.match(textCheckWords(view(check({ agree: 1, disagree: 2 })).faaList, NOW)!, /: 1 of 3 read match, 2 differ$/);
  assert.match(textCheckWords(view(check({ not_read: 1 })).faaList, NOW)!, /on 1 notice since .*: 1 not read$/);
  assert.match(textCheckWords(view(check({ agree: 1, since: null })).faaList, NOW)!, /on 1 notice: 1 of 1 read matches$/);

  // the card of a row drawn from its text carries the check; a row with any other shape does not
  const withText = view(check({ agree: 8 }), [LEXINGTON, MAYPORT, ZAGREB]);
  const textRow = withText.rows.find((x) => x.shape === "text")!;
  const fact = restrictionCard(textRow, NOW, U, true, withText.faaList).facts.find((x) => x.label === "Reader checked")!;
  assert.match(fact.value, /^Our reading of notice text checked against the FAA's outline on 8 notices since .*: 8 of 8 read match\. These are the notices both the feed and the FAA's list held; it is a check of the reader, not of this notice$/);
  for (const r of withText.rows.filter((x) => x.shape !== "text")) assert.equal(restrictionCard(r, NOW, U, true, withText.faaList).facts.find((x) => x.label === "Reader checked"), undefined, r.id);
  assert.equal(restrictionCard(textRow, NOW, U, true, none.faaList).facts.find((x) => x.label === "Reader checked"), undefined, "nothing judged, nothing said");
  assert.equal(restrictionCard(textRow, NOW, U, true).facts.find((x) => x.label === "Reader checked"), undefined);
});
