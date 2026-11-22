// FLIGHT RESTRICTIONS ([PRODUCT] 2026-11-21, UI coverage batch B, part 4): the
// /data map's "Flight restrictions" layer. Data: GET /api/data/swim/notam/tfrs
// (server/swimNotam.ts): the restrictions the FAA's notice feed (SWIM FNS
// NOTAM) lists that are in force now or start within 6 hours.
//
// RAW, as received: nothing here is inferred.
//   KIND is read from two published tables and nothing else: the 14 CFR
//   section the notice's own text cites (91.137 .. 91.145, 99.7), or, with no
//   section cited, the ICAO Q-code subject "RT" (temporary restricted area).
//   A notice that cites neither is "rule not cited"; it is never guessed.
//   SHAPE is one of three things and the map draws them differently: the
//   restriction's own outline when the notice carried airspace geometry
//   (solid); the outline read from the notice's own words when it carried
//   none, which is how US Flight Data Center notices arrive (dotted;
//   server/swimNotamText.ts reads one published wording and leaves any area
//   it cannot follow unread and counted); or the notice's index circle
//   (centre and radius), which ENCLOSES the restriction and is larger than
//   it (dashed). The text reading is checked against the FAA's own
//   published shapes (scripts/tfr_text_gate1.ts; 2026-11-22: of 49 listed
//   notices with an FAA shape, 42 read, 41 within tolerance of every FAA
//   polygon, none read wrongly).
//   LIMITS are the notice's index flight levels (000 = surface, 999 =
//   unlimited); they can be wider than the limits written in the text, and
//   the card says so.
// NOT A COMPLETE LIST. Checked 2026-11-21 18:40Z against the FAA's own list
// (tfr.faa.gov): 66 restrictions there, 18 held here, 1 of them in force.
// The server lists a restriction only when the feed has sent its notice
// since the server began keeping the list AND a shape could be had from it
// (airspace geometry, an area its text defines, or an index circle). A
// restriction issued before the list was kept (2026-11-21) is absent until
// the FAA sends it again. The status line, the legend and every card say
// the list is not complete, and the status line counts the restriction
// notices that arrived without a shape (route field
// stats.tfrs_without_shape). (research/open_questions.md, 2026-11-21 and
// 2026-11-22 flight restrictions.)
// The server's list is saved each minute and read back at a start; a
// cancellation sent while the server was down is not known, so such a
// restriction stays listed until its published end (card source line).
// Pure; pinned by flightRestrictions.test.ts.

import { destinationPoint } from "../../../../shared/flightPlanGeometry.js";
import { zuluAt } from "./airportTraffic.js";

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);

export const FLIGHT_RESTRICTIONS_REFRESH_MS = 60_000;
/** Law IV: the most restrictions the layer draws. The server holds at most
 *  2,000 (swimNotam NOTAM_MAX_TFRS); above the cap the ones in force are
 *  kept first, then the soonest to start, and the status line says how many
 *  were left out. */
export const FLIGHT_RESTRICTIONS_MAX_FEATURES = 2_000;
/** Law IV: one outline may carry this many corners; a longer one is thinned evenly. */
export const FLIGHT_RESTRICTION_MAX_RING_POINTS = 512;
/** Law IV: GeoJSON held by the two sources at the caps above (2,000 x 512 corners x 2 numbers), a ceiling, not a typical figure. */
export const FLIGHT_RESTRICTIONS_HEAP_BUDGET_BYTES = 48 * 1024 * 1024;
/** corners of a drawn index circle */
export const CIRCLE_POINTS = 64;
/** a connected feed with no notice for this long is "quiet" (same figure as the Streams block) */
export const NOTICE_FEED_QUIET_MS = 5 * 60_000;
const NM_TO_KM = 1.852;
const FL_TO_M = 30.48;

export type RestrictionKind = "security" | "hazard" | "space" | "event" | "icao_temporary" | "not_cited";

/** 14 CFR sections a flight-restriction notice cites, with the section's own subject */
export const RULE_WORDS: Readonly<Record<string, string>> = {
  "91.137": "disaster or hazard area",
  "91.138": "national disaster area in Hawaii",
  "91.139": "emergency air traffic rules",
  "91.141": "presidential and other parties",
  "91.143": "space flight operations",
  "91.145": "aerial demonstration or major sporting event",
  "99.7": "special security instructions",
};
const RULE_KIND: Readonly<Record<string, RestrictionKind>> = {
  "91.137": "hazard", "91.138": "hazard", "91.139": "hazard",
  "91.141": "security", "99.7": "security", "91.143": "space", "91.145": "event",
};
export const RESTRICTION_KIND_LABEL: Readonly<Record<RestrictionKind, string>> = {
  security: "Security (91.141, 99.7)",
  hazard: "Disaster, hazard or emergency (91.137 to 91.139)",
  space: "Space flight operations (91.143)",
  event: "Air show or major sporting event (91.145)",
  icao_temporary: "Temporary restricted area (ICAO notice)",
  not_cited: "Flight restriction, rule not cited",
};
export const RESTRICTION_KIND_COLOR: Readonly<Record<RestrictionKind, string>> = {
  security: "#f87171", hazard: "#fb923c", space: "#a78bfa", event: "#60a5fa", icao_temporary: "#2dd4bf", not_cited: "#94a3b8",
};
export const RESTRICTION_KIND_ORDER: readonly RestrictionKind[] = ["security", "hazard", "space", "event", "icao_temporary", "not_cited"];
/** who issued the notice, as the feed classifies it */
export const CLASSIFICATION_WORDS: Readonly<Record<string, string>> = {
  FDC: "FAA Flight Data Center", DOM: "US domestic", INTL: "international", MIL: "military",
};

const RULE_RE = /\b(91\.1(?:37|38|39|41|43|45)|99\.7)\b/g;
/** the 14 CFR sections a notice's text cites, in the order it cites them, each once */
export function rulesCited(text: string | null): string[] {
  const out: string[] = [];
  for (const m of (text ?? "").matchAll(RULE_RE)) if (!out.includes(m[1])) out.push(m[1]);
  return out;
}
/** Kind from the first section cited; else the ICAO Q-code subject RT; else not cited. */
export function restrictionKind(text: string | null, qcode: string | null): RestrictionKind {
  const rules = rulesCited(text);
  if (rules.length) return RULE_KIND[rules[0]];
  return /^QRT/i.test(qcode ?? "") ? "icao_temporary" : "not_cited";
}

export type LonLat = [number, number];
export interface FlightRestriction {
  id: string;
  /** the notice as people cite it: "LDZL A4134/2026", or "FDC 6/9306" when the notice carries that form */
  notice: string;
  location: string | null; classification: string | null; qcode: string | null;
  kind: RestrictionKind; rules: string[];
  issuedMs: number | null; startMs: number | null; endMs: number | null;
  /** index flight levels as sent (hundreds of feet; 0 = surface, 999 = unlimited) */
  minFl: number | null; maxFl: number | null;
  /** "own" = the restriction's outline as sent; "text" = the outline read from the notice's words;
   *  "enclosing" = the notice's index circle */
  shape: "own" | "text" | "enclosing";
  /** shape "text" only: areas the text defines that could not be followed and are not drawn */
  unreadAreas: number;
  /** outlines in GeoJSON order [lon, lat], each closed; several = several separate areas */
  rings: LonLat[][];
  /** where the symbol sits: the notice's index point, or the mean corner of the first outline */
  center: LonLat;
  radiusNm: number | null;
  text: string | null;
  inForce: boolean;
}
export interface NoticeFeed { connected: boolean; lastMessageAt: number | null }
export interface FlightRestrictionsView {
  atMs: number | null;
  rows: FlightRestriction[];
  /** rows the answer carried that could not be read (no usable shape or id) */
  unreadable: number;
  /** rows left out by the feature cap */
  cut: number;
  /** restrictions the server holds, including those starting later than 6 hours from now; null = not said */
  held: number | null;
  /** restriction notices received since the server started that carried no shape; null = the server does not say */
  withoutShape: number | null;
  feed: NoticeFeed | null;
  /** true = the server saves its list across a restart */
  kept: boolean;
  source: string | null;
}

const latOk = (v: unknown): v is number => fin(v) && Math.abs(v) <= 90;
const lonOk = (v: unknown): v is number => fin(v) && Math.abs(v) <= 180;

/** The index circle as a closed outline of CIRCLE_POINTS corners (great-circle
 *  distance). Longitudes stay continuous with the centre's, so a circle that
 *  crosses the 180th meridian is not torn. */
export function circleRing(center: LonLat, radiusNm: number, points = CIRCLE_POINTS): LonLat[] {
  const ring: LonLat[] = [];
  for (let i = 0; i < points; i++) {
    const p = destinationPoint({ lat: center[1], lon: center[0] }, (360 * i) / points, radiusNm);
    let lon = p.lon;
    if (lon - center[0] > 180) lon -= 360; else if (lon - center[0] < -180) lon += 360;
    ring.push([lon, p.lat]);
  }
  ring.push(ring[0]);
  return ring;
}

/** server ring [[lat, lon], ...] -> closed GeoJSON ring, thinned evenly above the corner cap; null when unusable */
function ownRing(raw: unknown): LonLat[] | null {
  if (!Array.isArray(raw)) return null;
  let pts: LonLat[] = [];
  for (const p of raw) {
    if (!Array.isArray(p) || !latOk(p[0]) || !lonOk(p[1])) return null;
    pts.push([p[1], p[0]]);
  }
  if (pts.length > 1 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]) pts.pop();
  if (pts.length < 3) return null;
  if (pts.length > FLIGHT_RESTRICTION_MAX_RING_POINTS) {
    const step = pts.length / FLIGHT_RESTRICTION_MAX_RING_POINTS;
    pts = Array.from({ length: FLIGHT_RESTRICTION_MAX_RING_POINTS }, (_, i) => pts[Math.floor(i * step)]);
  }
  pts.push(pts[0]);
  return pts;
}

/** GET /api/data/swim/notam/tfrs -> the layer's rows; null on another shape. */
export function parseFlightRestrictions(raw: unknown, max = FLIGHT_RESTRICTIONS_MAX_FEATURES): FlightRestrictionsView | null {
  const o = obj(raw);
  if (!o || !Array.isArray(o.tfrs)) return null;
  const atMs = fin(o.at) ? o.at : null;
  const now = atMs ?? Date.now();
  let unreadable = 0;
  const rows: FlightRestriction[] = [];
  const seen = new Set<string>();
  for (const tv of o.tfrs) {
    const t = obj(tv);
    const id = t ? str(t.id) : null;
    const g = t ? obj(t.geometry) : null;
    if (!t || !id || !g || seen.has(id)) { unreadable++; continue; }
    const qc = Array.isArray(t.center) && latOk(t.center[0]) && lonOk(t.center[1]) ? ([t.center[1], t.center[0]] as LonLat) : null;
    let shape: FlightRestriction["shape"], rings: LonLat[][], center: LonLat | null, radiusNm: number | null = null;
    if (g.kind === "polygon") {
      rings = (Array.isArray(g.rings) ? g.rings : []).map(ownRing).filter((r): r is LonLat[] => !!r);
      if (!rings.length) { unreadable++; continue; }
      shape = g.from === "text" ? "text" : "own";
      const first = rings[0].slice(0, -1);
      center = qc ?? [first.reduce((s, p) => s + p[0], 0) / first.length, first.reduce((s, p) => s + p[1], 0) / first.length];
    } else if (g.kind === "circle") {
      const c = Array.isArray(g.center) && latOk(g.center[0]) && lonOk(g.center[1]) ? ([g.center[1], g.center[0]] as LonLat) : null;
      if (!c || !fin(g.radiusNm) || g.radiusNm <= 0) { unreadable++; continue; }
      shape = "enclosing"; center = c; radiusNm = g.radiusNm; rings = [circleRing(c, g.radiusNm)];
    } else { unreadable++; continue; }
    seen.add(id);
    const text = str(t.text)?.replace(/\r\n?/g, "\n") ?? null;
    const qcode = str(t.qcode);
    const startMs = fin(t.start) ? t.start : null;
    rows.push({
      id, notice: str(t.citation) ?? id.replace(":", " "), location: str(t.location), classification: str(t.classification), qcode,
      kind: restrictionKind(text, qcode), rules: rulesCited(text),
      issuedMs: fin(t.issued) ? t.issued : null, startMs, endMs: fin(t.end) ? t.end : null,
      minFl: fin(t.minFl) ? t.minFl : null, maxFl: fin(t.maxFl) ? t.maxFl : null,
      shape, unreadAreas: shape === "text" && fin(g.unread) && g.unread > 0 ? Math.floor(g.unread) : 0, rings, center, radiusNm, text, inForce: startMs === null || startMs <= now,
    });
  }
  // in force first, then the soonest to start; the cap cuts from the end
  rows.sort((a, b) => Number(b.inForce) - Number(a.inForce) || (a.startMs ?? 0) - (b.startMs ?? 0) || a.id.localeCompare(b.id));
  const cut = Math.max(0, rows.length - max);
  if (cut) rows.length = max;
  const stats = obj(o.stats), feed = obj(o.feed), kept = obj(o.kept);
  return {
    atMs, rows, unreadable, cut,
    held: stats && fin(stats.tfrs) ? stats.tfrs : null,
    withoutShape: stats && fin(stats.tfrs_without_shape) ? stats.tfrs_without_shape : null,
    feed: feed ? { connected: feed.connected === true, lastMessageAt: fin(feed.last_message_at) ? feed.last_message_at : null } : null,
    kept: !!kept, source: str(o.source),
  };
}

interface Feature { type: "Feature"; geometry: Record<string, unknown>; properties: Record<string, unknown> }
export interface FeatureCollection { type: "FeatureCollection"; features: Feature[] }

/** The two sources the layer draws: the areas (one MultiPolygon per restriction) and one symbol point each. */
export function restrictionFeatures(rows: FlightRestriction[]): { areas: FeatureCollection; points: FeatureCollection } {
  const props = (r: FlightRestriction) => ({ id: r.id, kind: r.kind, color: RESTRICTION_KIND_COLOR[r.kind], shape: r.shape, in_force: r.inForce });
  return {
    areas: { type: "FeatureCollection", features: rows.map((r) => ({ type: "Feature", geometry: { type: "MultiPolygon", coordinates: r.rings.map((ring) => [ring]) }, properties: props(r) })) },
    points: { type: "FeatureCollection", features: rows.map((r) => ({ type: "Feature", geometry: { type: "Point", coordinates: r.center }, properties: props(r) })) },
  };
}

/** said on the status line and in the legend */
export const NOT_COMPLETE = "not the FAA's complete list";
/** said on every card */
export const NOT_COMPLETE_SENTENCE = "This is not the FAA's complete list: a restriction is here only when the feed has sent its notice since the server began keeping the list (2026-11-21) and a shape could be had from it. The FAA's own list is at tfr.faa.gov.";

/** "3 min ago" / "45 s ago" / "2 h ago" */
export function agoWords(ms: number, nowMs: number): string {
  const s = Math.max(0, Math.round((nowMs - ms) / 1000));
  return s < 90 ? `${s} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
}

/** How old the layer's data is, in the feed's own terms (Law V). */
export function feedWords(v: FlightRestrictionsView, nowMs: number): string {
  const f = v.feed;
  if (!f) return "the server does not say when the last notice arrived";
  const last = f.lastMessageAt !== null ? `last notice received ${agoWords(f.lastMessageAt, nowMs)}` : "no notice received since the server started";
  if (!f.connected) return `notice feed disconnected, ${last}`;
  return f.lastMessageAt !== null && nowMs - f.lastMessageAt > NOTICE_FEED_QUIET_MS ? `notice feed quiet, ${last}` : last;
}

/** The layer's status line in the layers panel. */
export function statusWords(v: FlightRestrictionsView, nowMs: number): string {
  const n = v.rows.length;
  const starting = v.rows.filter((r) => !r.inForce).length;
  const held = v.held !== null ? ` of ${v.held} held` : "";
  const head = n === 0
    ? `None in force or starting within 6 hours${v.held !== null ? ` among the ${v.held} held from the FAA notice feed` : ""}`
    : `${n - starting} in force, ${starting} starting within 6 hours${held}`;
  const tail = [
    v.cut ? `${v.cut} more not drawn (limit ${FLIGHT_RESTRICTIONS_MAX_FEATURES.toLocaleString("en-US")})` : null,
    v.unreadable ? `${v.unreadable} could not be read` : null,
    v.withoutShape ? `${v.withoutShape} restriction notice${v.withoutShape === 1 ? "" : "s"} arrived without a shape and ${v.withoutShape === 1 ? "is" : "are"} not drawn` : null,
    NOT_COMPLETE,
    feedWords(v, nowMs),
  ].filter(Boolean);
  return [head, ...tail].join(" · ");
}

export interface UnitWords { km: (km: number, digits?: number) => string; meters: (m: number) => string }

/** "surface" / "unlimited" / "FL085 (8,500 ft)" / "not given" */
export function limitWords(fl: number | null, u: UnitWords): string {
  if (fl === null) return "not given";
  if (fl <= 0) return "surface";
  if (fl >= 999) return "unlimited";
  return `FL${String(Math.round(fl)).padStart(3, "0")} (${u.meters(fl * FL_TO_M)})`;
}

/** A chip holds about seven characters (the card draws four across): "surface" / "unlim." / "FL085" / "n/a".
 *  The Details rows carry the full words and the converted height. */
export function limitChip(fl: number | null): string {
  if (fl === null) return "n/a";
  return fl <= 0 ? "surface" : fl >= 999 ? "unlim." : `FL${String(Math.round(fl)).padStart(3, "0")}`;
}
/** A time for a chip: "22:45Z" on the UTC day of `nowMs`, else the day, "Nov 22". The card's second line and the
 *  Details rows carry day and time together. */
export function chipTime(ms: number | null, nowMs: number): string {
  if (ms === null) return "n/a";
  const full = zuluAt(ms, nowMs);
  return full.length <= 6 ? full : full.slice(0, full.lastIndexOf(" "));
}

/** "in force until 22:45Z" / "starts 23:10Z, until Nov 22 04:59Z" / "in force, no end published" */
export function periodWords(r: FlightRestriction, nowMs: number): string {
  const end = r.endMs !== null ? `until ${zuluAt(r.endMs, nowMs)}` : "no end published";
  if (!r.inForce && r.startMs !== null) return `starts ${zuluAt(r.startMs, nowMs)}, ${end}`;
  return r.endMs !== null ? `in force ${end}` : `in force, ${end}`;
}

export interface RestrictionCard {
  title: string; subtitle: string; body: string; sourceTag: string;
  stats: Array<{ label: string; value: string }>;
  facts: Array<{ label: string; value: string }>;
}

/** Everything the detail card shows for one restriction. */
export function restrictionCard(r: FlightRestriction, nowMs: number, u: UnitWords, kept: boolean): RestrictionCard {
  const rule = r.rules.length
    ? r.rules.map((x) => `14 CFR ${x} (${RULE_WORDS[x]})`).join(", ")
    : r.kind === "icao_temporary" ? "none cited; ICAO Q-code subject RT, temporary restricted area" : "none cited in the notice";
  const areas = r.rings.length > 1 ? ` (${r.rings.length} areas)` : "";
  const shape = r.shape === "own"
    ? `the restriction's own outline, as sent${areas}`
    : r.shape === "text"
    ? `an outline read from the notice's text${areas}; the notice sent no geometry. Circles and arcs are drawn as short straight sides`
      + (r.unreadAreas ? `. ${r.unreadAreas} more area${r.unreadAreas === 1 ? "" : "s"} in the text could not be read and ${r.unreadAreas === 1 ? "is" : "are"} not drawn` : "")
    : `a circle of ${u.km((r.radiusNm ?? 0) * NM_TO_KM, 1)} around the notice's index point. It encloses the restriction and is larger than it; the outline itself was not sent as geometry`;
  const cls = r.classification ? CLASSIFICATION_WORDS[r.classification.toUpperCase()] ?? r.classification : null;
  const facts = [
    { label: "Notice", value: r.notice },
    { label: "Kind", value: RESTRICTION_KIND_LABEL[r.kind] },
    { label: "Rule cited", value: rule },
    { label: "Drawn shape", value: shape },
    r.startMs !== null ? { label: r.inForce ? "In force since" : "Starts", value: zuluAt(r.startMs, nowMs) } : null,
    { label: "Ends", value: r.endMs !== null ? zuluAt(r.endMs, nowMs) : "no end published" },
    { label: "Lower limit", value: limitWords(r.minFl, u) },
    { label: "Upper limit", value: limitWords(r.maxFl, u) },
    r.issuedMs !== null ? { label: "Issued", value: zuluAt(r.issuedMs, nowMs) } : null,
    r.location ? { label: "Location code", value: r.location } : null,
    cls ? { label: "Issued by", value: cls } : null,
    r.qcode ? { label: "Q-code, as sent", value: r.qcode } : null,
  ].filter((f): f is { label: string; value: string } => !!f);
  const restart = kept
    ? "The server keeps its list across a restart; a cancellation sent while it was restarting is not known, so such a restriction stays listed until its published end."
    : "The list starts again when the server restarts: a restriction sent before then is missing until the FAA sends it again.";
  return {
    title: `Flight restriction ${r.notice}`,
    subtitle: `${RESTRICTION_KIND_LABEL[r.kind]} · ${periodWords(r, nowMs)}`,
    stats: [
      { label: r.inForce ? "Status" : "Starts", value: r.inForce ? "active" : chipTime(r.startMs, nowMs) },
      { label: "Ends", value: chipTime(r.endMs, nowMs) },
      { label: "Lower", value: limitChip(r.minFl) },
      { label: "Upper", value: limitChip(r.maxFl) },
    ],
    facts,
    sourceTag: "FAA notice feed",
    body: `${r.text ? `Notice text, as sent:\n${r.text}\n\n` : ""}`
      + (r.minFl === null && r.maxFl === null
      ? "This notice has no index limits; its lower and upper limits are written in its text. "
      : "The lower and upper limits are the notice's index values; they can be wider than the limits written in its text. ")
      + `FAA notice feed (SWIM FNS NOTAM), as received. ${NOT_COMPLETE_SENTENCE} ${restart} Displayed as-is; not for flight planning or any operational use.`,
  };
}

// ── drawing ──────────────────────────────────────────────────────────────
// The five MapLibre layers and two GeoJSON sources, added, updated and
// removed here so the teardown is one tested function (RENDERING & MOTION
// LAW IV). Nothing here listens to a map event (Law I).

/** the part of a MapLibre map this layer uses */
export interface RestrictionMap {
  getLayer(id: string): unknown;
  removeLayer(id: string): unknown;
  getSource(id: string): unknown;
  removeSource(id: string): unknown;
  addSource(id: string, spec: unknown): unknown;
  addLayer(spec: unknown): unknown;
}
export const RESTRICTION_SOURCES = ["flight-restrictions-areas", "flight-restrictions-pts"] as const;
/** bottom to top */
export const RESTRICTION_LAYERS = ["flight-restrictions-fill", "flight-restrictions-line", "flight-restrictions-line-text", "flight-restrictions-line-enclosing", "flight-restrictions-sym"] as const;
/** the layers that open the card: the symbol and the outlines. The tinted
 *  inside does not, so a plane, airport or site inside a restriction keeps its click. */
export const RESTRICTION_CLICK_LAYERS = ["flight-restrictions-sym", "flight-restrictions-line", "flight-restrictions-line-text", "flight-restrictions-line-enclosing"];
export const RESTRICTION_ICON = "vt-airspace";

/** Put the rows on the map: the first call adds sources and layers, later calls only replace the data. Returns true when it added them. */
export function drawRestrictions(map: RestrictionMap, rows: FlightRestriction[]): boolean {
  const fc = restrictionFeatures(rows);
  const areas = map.getSource(RESTRICTION_SOURCES[0]) as { setData(d: unknown): void } | undefined;
  const pts = map.getSource(RESTRICTION_SOURCES[1]) as { setData(d: unknown): void } | undefined;
  if (areas && pts) { areas.setData(fc.areas); pts.setData(fc.points); return false; }
  removeRestrictions(map); // a half-added set (one source without the other) starts clean
  map.addSource(RESTRICTION_SOURCES[0], { type: "geojson", data: fc.areas });
  map.addSource(RESTRICTION_SOURCES[1], { type: "geojson", data: fc.points });
  const faint = (on: number, off: number) => ["case", ["get", "in_force"], on, off];
  map.addLayer({
    id: "flight-restrictions-fill", type: "fill", source: RESTRICTION_SOURCES[0],
    paint: { "fill-color": ["get", "color"], "fill-opacity": faint(0.16, 0.05) },
  });
  map.addLayer({
    id: "flight-restrictions-line", type: "line", source: RESTRICTION_SOURCES[0], filter: ["==", ["get", "shape"], "own"],
    paint: { "line-color": ["get", "color"], "line-width": 1.6, "line-opacity": faint(0.95, 0.55) },
  });
  map.addLayer({
    id: "flight-restrictions-line-text", type: "line", source: RESTRICTION_SOURCES[0], filter: ["==", ["get", "shape"], "text"],
    paint: { "line-color": ["get", "color"], "line-width": 1.8, "line-dasharray": [0.6, 1.6], "line-opacity": faint(0.95, 0.55) },
  });
  map.addLayer({
    id: "flight-restrictions-line-enclosing", type: "line", source: RESTRICTION_SOURCES[0], filter: ["==", ["get", "shape"], "enclosing"],
    paint: { "line-color": ["get", "color"], "line-width": 1.4, "line-dasharray": [3, 2], "line-opacity": faint(0.95, 0.55) },
  });
  map.addLayer({
    id: "flight-restrictions-sym", type: "symbol", source: RESTRICTION_SOURCES[1],
    layout: { "icon-image": RESTRICTION_ICON, "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.4, 8, 0.7], "icon-allow-overlap": true, "icon-ignore-placement": true },
    paint: { "icon-color": ["get", "color"], "icon-halo-color": "rgba(5,10,19,0.95)", "icon-halo-width": 1.3, "icon-opacity": faint(1, 0.6) },
  });
  return true;
}

/** Take everything this layer added off the map. false = the map refused (it is already torn down). */
export function removeRestrictions(map: RestrictionMap | null | undefined): boolean {
  if (!map) return true;
  try {
    for (const id of [...RESTRICTION_LAYERS].reverse()) if (map.getLayer(id)) map.removeLayer(id);
    for (const id of RESTRICTION_SOURCES) if (map.getSource(id)) map.removeSource(id);
    return true;
  } catch (e) {
    return false;
  }
}
