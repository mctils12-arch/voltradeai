// FLIGHT PROGRAM (2026-09-28) — FAA SWIM SFDPS adapter (FILED_FAA flight plans).
//
// SFDPS (SWIM Flight Data Publication Service) publishes, on ONE durable
// Solace queue "<user>.FDPS.<uuid>.OUT" (VPN "FDPS", tcps SMF), FOUR services
// mixed together: Flight FIXM, En Route General Message Publication, Airspace
// AIXM (special-activity airspace), and Status. Only Flight FIXM becomes flight
// plans here; the other three are classified by a cheap root-element sniff,
// COUNTED (per-type counters on /api/data/aircraft/plan-status) and acked —
// never parsed, never logged per message. Airspace AIXM is a future layer.
//
// Transport lives in server/swimConnector.ts (shared with the future TFMS /
// STDDS / NOTAM adapters); this module is only SFDPS semantics. Env (all five
// required; inert — no import, no socket — until they exist):
//   SWIM_SFDPS_URL  SWIM_SFDPS_VPN  SWIM_SFDPS_USER  SWIM_SFDPS_PASSWORD
//   SWIM_SFDPS_QUEUE
//
// VOLUME: nationwide Flight FIXM runs hundreds of messages/sec at peak, most
// of them track/handoff updates. Each flight element is sniffed by string ops
// first; only messages that can change a PLAN (FH plan, AH amendment, HX
// cancel, or anything carrying route/altitude elements) get the full
// XML-lite parse. Everything else takes a regex light path (identity +
// aerodromes + status) that refreshes TTL and, for a flight first seen
// mid-air, records its FILED departure/arrival. The plan store is compact
// (Float32-packed route) and bounded: 20k entries, 6h since last message,
// removed on cancel/complete.
//
// HONESTY: a FILED plan is exactly what the message carried. When only route
// TEXT is present (no expanded point positions), the consumer draws the great
// circle between the filed departure and arrival and labels it estimated —
// named fixes are never geocoded by guess.
//
// SCHEMA NOTE: the SFDPS FIXM 3.0 + NAS-extension element names below could
// not be verified against live traffic offline; the extractor is tolerant
// (namespace-prefix agnostic, attribute or element forms, FIXM 4.x shapes)
// and every field degrades to null rather than guessing.

import { lookupFix } from "./navFixes";
import { startSwimProduct, swimProductStatus, type SwimConnectorHandle, type SwimProductOptions } from "./swimConnector";

// ── 1a. XML-lite ────────────────────────────────────────────────────────────
export interface XNode {
  /** local name (namespace prefix stripped) */
  name: string;
  /** attributes by their FULL name (prefix kept); use attr() for local lookup */
  attrs: Record<string, string>;
  children: XNode[];
  text: string;
}

const XML_MAX_BYTES = 4 * 1024 * 1024;

function decodeEntities(s: string): string {
  if (s.indexOf("&") < 0) return s;
  return s.replace(/&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos);/gi, (m, e: string) => {
    const k = e.toLowerCase();
    if (k === "amp") return "&";
    if (k === "lt") return "<";
    if (k === "gt") return ">";
    if (k === "quot") return '"';
    if (k === "apos") return "'";
    const code = k.startsWith("#x") ? parseInt(k.slice(2), 16) : parseInt(k.slice(1), 10);
    return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : m;
  });
}
const localName = (n: string) => { const i = n.lastIndexOf(":"); return i >= 0 ? n.slice(i + 1) : n; };

/** Minimal well-formed-XML reader: elements, attributes, text, CDATA;
 *  comments / PIs / DOCTYPE skipped. Returns a synthetic "#document" root, or
 *  null for oversized input. Malformed input degrades (unclosed tags are
 *  closed at EOF) rather than throwing. */
export function parseXmlLite(xml: string): XNode | null {
  if (typeof xml !== "string" || xml.length > XML_MAX_BYTES) return null;
  const root: XNode = { name: "#document", attrs: {}, children: [], text: "" };
  const stack: XNode[] = [root];
  const re = /<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!\[CDATA\[([\s\S]*?)\]\]>|<!DOCTYPE[^>]*>|<\/\s*([^\s>]+)\s*>|<([^\s/>!?]+)((?:\s+[^\s=/>]+\s*=\s*(?:"[^"]*"|'[^']*'))*)\s*(\/?)>/g;
  const attrRe = /([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  let last = 0;
  let m: RegExpExecArray | null;
  const top = () => stack[stack.length - 1];
  while ((m = re.exec(xml))) {
    if (m.index > last) {
      const t = xml.slice(last, m.index);
      if (t.trim()) top().text += decodeEntities(t);
    }
    last = re.lastIndex;
    if (m[1] !== undefined) { top().text += m[1]; continue; } // CDATA
    if (m[2] !== undefined) { // close tag: pop to the matching open element
      const nm = localName(m[2]);
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].name === nm) { stack.length = i; break; }
      }
      continue;
    }
    if (m[3] !== undefined) {
      const node: XNode = { name: localName(m[3]), attrs: {}, children: [], text: "" };
      let a: RegExpExecArray | null;
      attrRe.lastIndex = 0;
      while ((a = attrRe.exec(m[4] || ""))) node.attrs[a[1]] = decodeEntities(a[2] ?? a[3] ?? "");
      top().children.push(node);
      if (m[5] !== "/") stack.push(node);
    }
  }
  return root;
}

/** attribute by LOCAL name (prefix-agnostic) */
export function attr(n: XNode | null | undefined, local: string): string | null {
  if (!n) return null;
  if (local in n.attrs) return n.attrs[local];
  for (const k of Object.keys(n.attrs)) if (localName(k) === local) return n.attrs[k];
  return null;
}
/** depth-first descendants with a given local name (excluding n itself) */
export function findAll(n: XNode | null | undefined, name: string, out: XNode[] = []): XNode[] {
  if (!n) return out;
  for (const c of n.children) {
    if (c.name === name) out.push(c);
    findAll(c, name, out);
  }
  return out;
}
export function findFirst(n: XNode | null | undefined, name: string): XNode | null {
  if (!n) return null;
  for (const c of n.children) {
    if (c.name === name) return c;
    const d = findFirst(c, name);
    if (d) return d;
  }
  return null;
}
const textOf = (n: XNode | null | undefined): string | null => {
  const t = n?.text?.trim();
  return t ? t : null;
};
/** first non-empty attribute (by local name) among `names` on n or any descendant */
function deepAttr(n: XNode | null | undefined, names: string[]): string | null {
  if (!n) return null;
  for (const k of names) { const v = attr(n, k); if (v && v.trim()) return v.trim(); }
  for (const c of n.children) { const v = deepAttr(c, names); if (v) return v; }
  return null;
}

// ── 1b. FIXM / SFDPS extraction ─────────────────────────────────────────────
export interface SwimRoutePoint {
  lat: number; lon: number;
  name?: string;
  /** a per-point altitude when the message carried one (feet) */
  altFt?: number | null;
}

export interface SwimFlightMessage {
  gufi: string | null;
  callsign: string | null;
  /** departure / arrival aerodrome identifiers as carried (ICAO normally) */
  departure: string | null;
  arrival: string | null;
  /** filed (requested) cruise altitude in feet; falls back to ATC-assigned */
  cruiseAltFt: number | null;
  routeText: string | null;
  /** expanded route point positions, in route order (may be empty) */
  routePoints: SwimRoutePoint[];
  /** SFDPS message source code as carried (e.g. FH flight plan, AH amendment,
   *  HZ track, HX cancellation — SFDPS 'source' attribute) */
  messageType: string | null;
  isAmendment: boolean;
  /** cancelled / dropped (HX message or fdpsFlightStatus CANCELLED/DROPPED) */
  isCancellation: boolean;
  /** fdpsFlightStatus COMPLETED — the flight is over; its plan leaves the store */
  isCompleted: boolean;
  /** message timestamp (ms epoch) */
  timestamp: number | null;
  /** best departure time known (actual > estimated), ms epoch */
  departureTime: number | null;
  /** en-route position report when carried */
  position: { lat: number; lon: number; altFt: number | null } | null;
  flightStatus: string | null;
}

const ALT_M_TO_FT = 3.28084;
/** altitude in FEET from a FIXM altitude-ish element (<simple uom="FEET">,
 *  <altitude uom="FL">, bare text). null when absent / VFR / unparseable. */
export function altitudeFeet(n: XNode | null | undefined): number | null {
  if (!n) return null;
  // NAS: <requestedAltitude><simple uom="FEET">; 4.x: <cruisingLevel>
  // <flightLevel uom="FL">350</...>; bare: <altitude uom="FEET">35000</...>
  const withText = (x: XNode): XNode | null => {
    if (textOf(x) != null) return x;
    for (const c of x.children) { const d = withText(c); if (d) return d; }
    return null;
  };
  const leaf = findFirst(n, "simple") || findFirst(n, "altitude") || withText(n) || n;
  const raw = textOf(leaf) ?? attr(leaf, "value");
  if (raw == null) return null;
  const v = parseFloat(raw);
  if (!Number.isFinite(v)) return null;
  const uom = (attr(leaf, "uom") || attr(n, "uom") || "FEET").toUpperCase();
  let ft = v;
  if (uom === "FL") ft = v * 100;
  else if (uom === "M" || uom === "METERS" || uom === "METRES") ft = v * ALT_M_TO_FT;
  else if (uom === "SM" || uom === "S") ft = v * 10 * ALT_M_TO_FT; // ICAO metric level (tens of meters)
  if (ft < -2000 || ft > 80000) return null;
  return Math.round(ft);
}

/** parse "lat lon" (GML pos, EPSG:4326 axis order) or "lat,lon" */
function parsePos(s: string | null): { lat: number; lon: number } | null {
  if (!s) return null;
  const parts = s.trim().split(/[\s,]+/).map(Number);
  if (parts.length < 2 || !parts.every(Number.isFinite)) return null;
  const [lat, lon] = parts;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}
/** a position under node n: <pos>, or lat/lon (latitude/longitude) attributes */
function positionIn(n: XNode): { lat: number; lon: number } | null {
  const pos = parsePos(textOf(findFirst(n, "pos")));
  if (pos) return pos;
  const stack: XNode[] = [n];
  while (stack.length) {
    const c = stack.pop()!;
    const la = attr(c, "latitude") ?? attr(c, "lat");
    const lo = attr(c, "longitude") ?? attr(c, "lon") ?? attr(c, "lng");
    if (la != null && lo != null) {
      const p = parsePos(`${la} ${lo}`);
      if (p) return p;
    }
    stack.push(...c.children);
  }
  return null;
}
const isoMs = (s: string | null | undefined): number | null => {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
};
const cleanId = (s: string | null): string | null => {
  const v = (s || "").trim().toUpperCase();
  return /^[A-Z0-9]{2,8}$/.test(v) ? v : null;
};

function aerodrome(n: XNode | null, pointAttr: string): string | null {
  if (!n) return null;
  // FIXM 3.0 NAS extension: <departure departurePoint="KBOS"> / <arrival arrivalPoint="KATL">
  const direct = cleanId(attr(n, pointAttr));
  if (direct) return direct;
  // FIXM 3.0 core / 4.x: <departureAerodrome><locationIndicator>KBOS</...> or code="KBOS"
  for (const nm of ["locationIndicator", "code", "icaoCode", "name"]) {
    const t = cleanId(textOf(findFirst(n, nm)));
    if (t) return t;
  }
  return cleanId(deepAttr(n, ["locationIndicator", "code", "aerodromeIdentifier"]));
}

// ── 1c. route-shape sampler (gate-1 instrument, content-shape only) ─────────
// Live 2026-09-29: 59/59 FILED plans arrived with route TEXT but zero placed
// route points, so the parser's `expandedRoute > routePoint > pos|lat/lon`
// walk does not match the real FIXM shape. This records the STRUCTURE of the
// route subtree (element names, attribute names, value SHAPES with digits->9
// and letters->A) for the first few distinct shapes seen — no real values, no
// identifiers — so the parser can be fixed against evidence, not a guess.
export const ROUTE_SHAPE_MAX_SAMPLES = 6;
const ROUTE_SHAPE_MAX_CHARS = 2400;
const ROUTE_SHAPE_MAX_DEPTH = 8;
const shapeOfValue = (v: string): string =>
  v.trim().slice(0, 24).replace(/[0-9]/g, "9").replace(/[A-Za-z]/g, "A").replace(/(.)\1{2,}/g, "$1$1+");

export function xmlShape(n: XNode, depth = 0): string {
  const attrs = Object.entries(n.attrs).slice(0, 12).map(([k, v]) => `${k}=${shapeOfValue(v)}`);
  const txt = n.text.trim();
  const out = n.name + (attrs.length ? `[${attrs.join(",")}]` : "") + (txt ? `{${shapeOfValue(txt)}}` : "");
  if (depth >= ROUTE_SHAPE_MAX_DEPTH || !n.children.length) return out;
  const kids: string[] = [];
  for (let i = 0; i < n.children.length;) {
    let j = i + 1;
    while (j < n.children.length && n.children[j].name === n.children[i].name) j++;
    kids.push(xmlShape(n.children[i], depth + 1) + (j - i > 1 ? `×${j - i}` : ""));
    i = j;
  }
  return `${out}(${kids.join(" ")})`;
}

export interface RouteShapeSample { where: string; shape: string; count: number; firstSeenAt: number }
const routeShapes = new Map<string, RouteShapeSample>();
export const routeShapeCounters = { placed: 0, expandedNoPoints: 0, noExpanded: 0 };

function recordRouteShape(flight: XNode, agreed: XNode | null, expanded: XNode | null, placed: number): void {
  if (placed > 0) { routeShapeCounters.placed++; return; }
  if (expanded) routeShapeCounters.expandedNoPoints++; else routeShapeCounters.noExpanded++;
  const target = expanded ?? findFirst(agreed, "route") ?? findFirst(flight, "route");
  const where = expanded ? "expandedRoute" : target ? "route(no expandedRoute)" : "flight(no route)";
  let shape = target ? xmlShape(target) : `flight(${flight.children.map((c) => c.name).join(" ")})`;
  if (shape.length > ROUTE_SHAPE_MAX_CHARS) shape = shape.slice(0, ROUTE_SHAPE_MAX_CHARS) + "…";
  const key = where + "|" + shape;
  const hit = routeShapes.get(key);
  if (hit) { hit.count++; return; }
  if (routeShapes.size < ROUTE_SHAPE_MAX_SAMPLES) routeShapes.set(key, { where, shape, count: 1, firstSeenAt: Date.now() });
}
export function routeShapeSamples(): { counters: typeof routeShapeCounters; samples: RouteShapeSample[] } {
  return { counters: { ...routeShapeCounters }, samples: [...routeShapes.values()] };
}

function extractFlight(flight: XNode, message: XNode | null): SwimFlightMessage {
  const fid = findFirst(flight, "flightIdentification");
  const callsign = cleanId(attr(fid, "aircraftIdentification"))
    ?? cleanId(textOf(findFirst(flight, "aircraftIdentification")));
  const gufi = textOf(findFirst(flight, "gufi")) ?? attr(flight, "gufi");

  const dep = findFirst(flight, "departure");
  const arr = findFirst(flight, "arrival");
  const departure = aerodrome(dep, "departurePoint") ?? aerodrome(findFirst(flight, "departureAerodrome"), "departurePoint");
  const arrival = aerodrome(arr, "arrivalPoint") ?? aerodrome(findFirst(flight, "destinationAerodrome"), "arrivalPoint");

  // FILED cruise: requestedAltitude (NAS) / cruisingLevel (4.x); ATC-assigned
  // altitude is only the fallback (it is a clearance, not the plan)
  const cruiseAltFt = altitudeFeet(findFirst(flight, "requestedAltitude"))
    ?? altitudeFeet(findFirst(flight, "cruisingLevel"))
    ?? altitudeFeet(findFirst(flight, "assignedAltitude"));

  // route text: NAS extension carries nasRouteText as an attribute of <route>;
  // 4.x carries <routeText>; accept either
  let routeText: string | null = null;
  for (const r of findAll(flight, "route")) {
    routeText = attr(r, "nasRouteText") ?? attr(r, "routeText") ?? textOf(findFirst(r, "routeText"));
    if (routeText) break;
  }
  routeText = routeText ?? textOf(findFirst(flight, "routeText")) ?? textOf(findFirst(flight, "nasRouteText"));

  // expanded route: prefer the one under <agreed> (current cleared route),
  // else the first found anywhere
  const agreed = findFirst(flight, "agreed");
  const expanded = findFirst(agreed, "expandedRoute") ?? findFirst(flight, "expandedRoute");
  const routePoints: SwimRoutePoint[] = [];
  if (expanded) {
    for (const rp of findAll(expanded, "routePoint")) {
      const name = deepAttr(rp, ["fix", "nasFixName", "designator", "fixName", "name", "point"]) ?? undefined;
      // explicit coordinates win; else resolve the fix NAME against the FAA NASR
      // gazetteer (live SFDPS carries names only). A place-bearing-distance point
      // (distance/radial children) is an OFFSET from the fix, not the fix — not
      // placed. An unknown name stays unplaced, never guessed.
      const offset = findFirst(rp, "distance") || findFirst(rp, "radial");
      const p = positionIn(rp) ?? (offset ? null : lookupFix(name));
      if (!p) continue;
      const altFt = altitudeFeet(findFirst(rp, "altitude") ?? findFirst(rp, "level"));
      routePoints.push({ ...p, ...(name ? { name } : {}), ...(altFt != null ? { altFt } : {}) });
    }
  }

  recordRouteShape(flight, agreed, expanded, routePoints.length);

  const source = attr(flight, "source") ?? attr(message, "source");
  const messageType = source ? source.trim().toUpperCase() : null;
  const statusNode = findFirst(flight, "flightStatus");
  const flightStatus = attr(statusNode, "fdpsFlightStatus") ?? attr(statusNode, "flightStatus") ?? textOf(statusNode);
  const isCancellation = messageType === "HX" || /CANCEL|DROPPED/i.test(flightStatus || "");
  const isCompleted = /COMPLETED/i.test(flightStatus || "");
  const isAmendment = messageType === "AH" || !!findFirst(flight, "amendment");

  const timestamp = isoMs(attr(flight, "timestamp")) ?? isoMs(attr(message, "timestamp"));
  let departureTime: number | null = null;
  if (dep) {
    for (const k of ["actual", "estimated", "controlled", "target"]) {
      const node = findFirst(dep, k);
      const t = isoMs(attr(node, "time") ?? textOf(node));
      if (t != null) { departureTime = t; break; }
    }
    departureTime = departureTime ?? isoMs(deepAttr(dep, ["time", "timeValue"]));
  }

  let position: SwimFlightMessage["position"] = null;
  const enRoute = findFirst(flight, "enRoute");
  const posNode = enRoute ? findFirst(enRoute, "position") : null;
  if (posNode) {
    const p = positionIn(posNode);
    if (p) position = { ...p, altFt: altitudeFeet(findFirst(posNode, "altitude")) };
  }

  return {
    gufi: gufi ? gufi.trim() : null, callsign, departure, arrival, cruiseAltFt,
    routeText: routeText ? routeText.trim() : null, routePoints, messageType,
    isAmendment, isCancellation, isCompleted, timestamp, departureTime, position,
    flightStatus: flightStatus || null,
  };
}

/** Parse one SFDPS payload (a MessageCollection, a single message, or a bare
 *  flight) into flight messages. Never throws; unparseable input -> []. */
export function parseSfdpsMessages(xml: string): SwimFlightMessage[] {
  let root: XNode | null;
  try { root = parseXmlLite(xml); } catch { return []; }
  if (!root) return [];
  const out: SwimFlightMessage[] = [];
  const walk = (n: XNode, message: XNode | null) => {
    for (const c of n.children) {
      if (c.name === "flight") { out.push(extractFlight(c, message)); continue; }
      walk(c, c.name === "message" ? c : message);
    }
  };
  walk(root, null);
  return out.filter((f) => f.callsign || f.gufi);
}


// ── 2. cheap classification (no DOM for ignored services) ───────────────────
export type SfdpsService = "FLIGHT" | "AIRSPACE_AIXM" | "GENERAL_MESSAGE" | "STATUS" | "UNKNOWN";

/** local name of the document's root element, skipping BOM / XML
 *  declaration / comments / DOCTYPE — string ops only */
export function rootElementName(payload: string): string | null {
  let i = 0;
  const n = payload.length;
  while (i < n) {
    const c = payload.charCodeAt(i);
    if (c === 0xfeff || c === 32 || c === 9 || c === 10 || c === 13) { i++; continue; }
    if (payload.startsWith("<?", i)) { const e = payload.indexOf("?>", i); if (e < 0) return null; i = e + 2; continue; }
    if (payload.startsWith("<!--", i)) { const e = payload.indexOf("-->", i); if (e < 0) return null; i = e + 3; continue; }
    if (payload.startsWith("<!", i)) { const e = payload.indexOf(">", i); if (e < 0) return null; i = e + 1; continue; }
    if (payload[i] !== "<") return null;
    const m = /^<([A-Za-z_][\w.\-]*:)?([A-Za-z_][\w.\-]*)/.exec(payload.slice(i, i + 256));
    return m ? m[2] : null;
  }
  return null;
}

const FLIGHT_OPEN_RE = /<((?:[\w.\-]+:)?)flight(?=[\s>\/])/g;

/** Which of the four services a payload belongs to: root name first, then a
 *  bounded look at the head for the message type. */
export function classifySfdpsPayload(payload: string): SfdpsService {
  const root = rootElementName(payload);
  if (!root) return "UNKNOWN";
  if (/aixm|^saa|airspace/i.test(root)) return "AIRSPACE_AIXM";
  if (/status/i.test(root)) return "STATUS";
  if (/general/i.test(root)) return "GENERAL_MESSAGE";
  const head = payload.slice(0, 4096);
  if (/GeneralMessage/i.test(head)) return "GENERAL_MESSAGE";
  FLIGHT_OPEN_RE.lastIndex = 0;
  if (root === "flight" || root === "Flight" || FLIGHT_OPEN_RE.test(payload)) return "FLIGHT";
  if (/aixm/i.test(head)) return "AIRSPACE_AIXM";
  if (/status/i.test(head)) return "STATUS";
  return "UNKNOWN";
}

/** raw <flight> elements (open tag through matching close) plus the open tag
 *  of the enclosing <message>, without building a DOM */
export function flightChunks(payload: string): Array<{ chunk: string; messageTag: string | null }> {
  const out: Array<{ chunk: string; messageTag: string | null }> = [];
  const re = new RegExp(FLIGHT_OPEN_RE.source, "g");
  let m: RegExpExecArray | null;
  while ((m = re.exec(payload))) {
    const start = m.index;
    const openEnd = payload.indexOf(">", start);
    if (openEnd < 0) break;
    let end: number;
    if (payload[openEnd - 1] === "/") end = openEnd + 1; // self-closing
    else {
      const close = `</${m[1]}flight>`;
      const c = payload.indexOf(close, openEnd);
      end = c < 0 ? payload.length : c + close.length;
    }
    // nearest preceding <message ...> open tag (message-level attributes)
    let messageTag: string | null = null;
    const mm = /<(?:[\w.\-]+:)?message(?=[\s>])[^>]*>/g;
    mm.lastIndex = Math.max(0, start - 4096);
    let x: RegExpExecArray | null;
    while ((x = mm.exec(payload)) && x.index < start) messageTag = x[0];
    out.push({ chunk: payload.slice(start, end), messageTag });
    re.lastIndex = end;
  }
  return out;
}

const attrRx = (name: string) => new RegExp(`\\s(?:[\\w.\\-]+:)?${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`);
const RX = {
  source: attrRx("source"),
  timestamp: attrRx("timestamp"),
  aircraftIdentification: attrRx("aircraftIdentification"),
  departurePoint: attrRx("departurePoint"),
  arrivalPoint: attrRx("arrivalPoint"),
  fdpsFlightStatus: attrRx("fdpsFlightStatus"),
  gufi: /<(?:[\w.\-]+:)?gufi(?:\s[^>]*)?>([^<]+)</,
  planBearing: /expandedRoute|[Rr]outeText|requestedAltitude|cruisingLevel|<(?:[\w.\-]+:)?amendment[\s>]/,
};
const rx1 = (re: RegExp, s: string | null | undefined): string | null => {
  if (!s) return null;
  const m = re.exec(s);
  return m ? ((m[1] ?? m[2] ?? "").trim() || null) : null;
};
const openTagOf = (chunk: string) => chunk.slice(0, Math.max(0, chunk.indexOf(">") + 1));
const sourceOf = (chunk: string, messageTag: string | null) =>
  (rx1(RX.source, openTagOf(chunk)) ?? rx1(RX.source, messageTag))?.toUpperCase() ?? null;

/** Plan-bearing flight messages get the full parse; the rest the light path. */
export function needsFullParse(chunk: string, source: string | null): boolean {
  if (source === "FH" || source === "AH" || source === "HX") return true;
  return RX.planBearing.test(chunk);
}

/** Light path: identity + aerodromes + status by regex (no DOM). */
export function lightFlight(chunk: string, messageTag: string | null): SwimFlightMessage | null {
  const open = openTagOf(chunk);
  const source = sourceOf(chunk, messageTag);
  const callsign = cleanId(rx1(RX.aircraftIdentification, chunk));
  const gufi = rx1(RX.gufi, chunk);
  if (!callsign && !gufi) return null;
  const status = rx1(RX.fdpsFlightStatus, chunk);
  return {
    gufi, callsign,
    departure: cleanId(rx1(RX.departurePoint, chunk)),
    arrival: cleanId(rx1(RX.arrivalPoint, chunk)),
    cruiseAltFt: null, routeText: null, routePoints: [],
    messageType: source,
    isAmendment: false,
    isCancellation: source === "HX" || /CANCEL|DROPPED/i.test(status || ""),
    isCompleted: /COMPLETED/i.test(status || ""),
    timestamp: isoMs(rx1(RX.timestamp, open) ?? rx1(RX.timestamp, messageTag)),
    departureTime: null, position: null, flightStatus: status,
  };
}

// ── 3. compact, bounded store ───────────────────────────────────────────────
export const SWIM_PLAN_TTL_MS = 6 * 3600_000;
export const SWIM_STORE_MAX = 20_000;
export const SWIM_ROUTE_MAX_POINTS = 250;
export const SWIM_ROUTE_TEXT_MAX = 1000;

export interface StoredSwimPlan {
  key: string;
  gufi: string | null;
  callsign: string | null;
  departure: string | null;
  arrival: string | null;
  cruiseAltFt: number | null;
  routeText: string | null;
  /** packed [lat, lon, altFt|NaN] triplets (Float32: ~1m precision) */
  route: Float32Array;
  /** route point names joined by "|" ("" = unnamed) */
  routeNames: string;
  messageType: string | null;
  flightStatus: string | null;
  timestamp: number | null;
  departureTime: number | null;
  firstSeen: number;
  updatedAt: number;
  amendments: number;
}

export function packRoute(points: SwimRoutePoint[]): { route: Float32Array; routeNames: string } {
  const pts = points.slice(0, SWIM_ROUTE_MAX_POINTS);
  const route = new Float32Array(pts.length * 3);
  pts.forEach((p, i) => { route[i * 3] = p.lat; route[i * 3 + 1] = p.lon; route[i * 3 + 2] = p.altFt ?? NaN; });
  return { route, routeNames: pts.map((p) => (p.name || "").replace(/\|/g, "/")).join("|") };
}

/** unpack a stored route (lat/lon rounded to 5 dp to shed Float32 noise) */
export function routePointsOf(p: StoredSwimPlan): SwimRoutePoint[] {
  const names = p.routeNames ? p.routeNames.split("|") : [];
  const out: SwimRoutePoint[] = [];
  for (let i = 0; i < p.route.length / 3; i++) {
    const alt = p.route[i * 3 + 2];
    const name = names[i];
    out.push({
      lat: Math.round(p.route[i * 3] * 1e5) / 1e5,
      lon: Math.round(p.route[i * 3 + 1] * 1e5) / 1e5,
      ...(name ? { name } : {}),
      ...(Number.isFinite(alt) ? { altFt: Math.round(alt) } : {}),
    });
  }
  return out;
}

const routeSig = (r: Float32Array) => {
  let s = "";
  for (let i = 0; i < r.length; i += 3) s += `${r[i].toFixed(3)},${r[i + 1].toFixed(3)};`;
  return s;
};

/** What changed between two versions of a plan, or null when nothing the
 *  plan-drawer cares about did. */
export function planDiff(prev: StoredSwimPlan, next: StoredSwimPlan): string | null {
  const parts: string[] = [];
  if (next.arrival && prev.arrival && next.arrival !== prev.arrival) parts.push(`destination ${prev.arrival}→${next.arrival}`);
  if (next.routeText && prev.routeText && next.routeText !== prev.routeText) parts.push("route text changed");
  if (next.route.length && prev.route.length && routeSig(next.route) !== routeSig(prev.route)) {
    parts.push(`expanded route changed (${prev.route.length / 3}→${next.route.length / 3} points)`);
  }
  if (next.cruiseAltFt != null && prev.cruiseAltFt != null && next.cruiseAltFt !== prev.cruiseAltFt) {
    parts.push(`cruise ${prev.cruiseAltFt}→${next.cruiseAltFt} ft`);
  }
  return parts.length ? parts.join("; ") : null;
}

const utcDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);
const EMPTY_ROUTE = new Float32Array(0);

export class SwimPlanStore {
  private byKey = new Map<string, StoredSwimPlan>(); // insertion order = last-message (LRU) order
  private keysByCallsign = new Map<string, Set<string>>();

  constructor(
    private opts: {
      ttlMs?: number; max?: number;
      onAmended?: (plan: StoredSwimPlan, detail: string) => void;
    } = {},
  ) {}

  get size(): number { return this.byKey.size; }

  private keyFor(m: SwimFlightMessage, now: number): string | null {
    if (m.gufi) return `gufi:${m.gufi}`;
    if (!m.callsign) return null;
    // no GUFI: an existing live entry for the callsign is the same flight
    const existing = this.lookup(m.callsign, now, true);
    if (existing) return existing.key;
    return `cs:${m.callsign}|${utcDay(m.departureTime ?? m.timestamp ?? now)}`;
  }

  /** merge one message; returns the amendment detail when the plan changed */
  upsert(m: SwimFlightMessage, now = Date.now()): { key: string | null; amended: string | null } {
    const key = this.keyFor(m, now);
    if (!key) return { key: null, amended: null };
    const prev = this.byKey.get(key);
    if (m.isCancellation || m.isCompleted) {
      if (prev) this.remove(key);
      return { key, amended: null };
    }
    // fields absent from this message keep their previous values (track /
    // handoff messages carry no route and must not erase the filed one)
    const packed = m.routePoints.length ? packRoute(m.routePoints) : null;
    const merged: StoredSwimPlan = {
      key,
      gufi: m.gufi ?? prev?.gufi ?? null,
      callsign: m.callsign ?? prev?.callsign ?? null,
      departure: m.departure ?? prev?.departure ?? null,
      arrival: m.arrival ?? prev?.arrival ?? null,
      cruiseAltFt: m.cruiseAltFt ?? prev?.cruiseAltFt ?? null,
      routeText: m.routeText ? m.routeText.slice(0, SWIM_ROUTE_TEXT_MAX) : (prev?.routeText ?? null),
      route: packed ? packed.route : (prev?.route ?? EMPTY_ROUTE),
      routeNames: packed ? packed.routeNames : (prev?.routeNames ?? ""),
      messageType: m.messageType ?? prev?.messageType ?? null,
      flightStatus: m.flightStatus ?? prev?.flightStatus ?? null,
      timestamp: m.timestamp ?? prev?.timestamp ?? null,
      departureTime: m.departureTime ?? prev?.departureTime ?? null,
      firstSeen: prev?.firstSeen ?? now,
      updatedAt: now,
      amendments: prev?.amendments ?? 0,
    };
    let amended: string | null = null;
    if (prev) {
      // a geometry/destination/cruise change is an amendment whatever the
      // message type; an AH message that changed none of those is still
      // recorded (honestly described) so the event log mirrors the feed
      amended = planDiff(prev, merged)
        ?? (m.isAmendment ? "amendment message (no route, destination or cruise change detected)" : null);
      if (amended) merged.amendments++;
    }
    this.byKey.delete(key);
    this.byKey.set(key, merged);
    if (merged.callsign) {
      const set = this.keysByCallsign.get(merged.callsign) || new Set<string>();
      set.add(key);
      this.keysByCallsign.set(merged.callsign, set);
    }
    this.evict(now);
    if (amended) {
      try { this.opts.onAmended?.(merged, amended); } catch (e) {
        console.error("[swim] amendment hook:", e instanceof Error ? e.message : String(e));
      }
    }
    return { key, amended };
  }

  private remove(key: string) {
    const p = this.byKey.get(key);
    this.byKey.delete(key);
    if (p?.callsign) {
      const set = this.keysByCallsign.get(p.callsign);
      set?.delete(key);
      if (set && !set.size) this.keysByCallsign.delete(p.callsign);
    }
  }

  /** most recently updated live plan for a callsign. Entries with neither a
   *  departure nor an arrival are not plans (includeBare is for keying). */
  lookup(callsign: string, now = Date.now(), includeBare = false): StoredSwimPlan | null {
    const set = this.keysByCallsign.get(callsign);
    if (!set) return null;
    let best: StoredSwimPlan | null = null;
    for (const k of Array.from(set)) {
      const p = this.byKey.get(k);
      if (!p) { set.delete(k); continue; }
      if (now - p.updatedAt > (this.opts.ttlMs ?? SWIM_PLAN_TTL_MS)) { this.remove(k); continue; }
      if (!includeBare && !p.departure && !p.arrival) continue;
      if (!best || p.updatedAt > best.updatedAt) best = p;
    }
    return best;
  }

  evict(now = Date.now()): void {
    const ttl = this.opts.ttlMs ?? SWIM_PLAN_TTL_MS;
    const max = this.opts.max ?? SWIM_STORE_MAX;
    for (const [k, p] of Array.from(this.byKey.entries())) {
      if (now - p.updatedAt > ttl) this.remove(k);
      else break; // LRU order: the rest are newer
    }
    while (this.byKey.size > max) {
      const oldest = this.byKey.keys().next().value;
      if (oldest === undefined) break;
      this.remove(oldest);
    }
  }
}

// ── 4. SFDPS payload handling + counters ────────────────────────────────────
export const SFDPS_ENV_PREFIX = "SWIM_SFDPS";

export interface SfdpsCounters {
  byService: Record<SfdpsService, number>;
  /** flight messages by SFDPS source code (FH, AH, HZ, OH, ...; bounded) */
  flightByType: Record<string, number>;
  fullParses: number;
  lightParses: number;
  parseErrors: number;
}

const counters: SfdpsCounters = {
  byService: { FLIGHT: 0, AIRSPACE_AIXM: 0, GENERAL_MESSAGE: 0, STATUS: 0, UNKNOWN: 0 },
  flightByType: {}, fullParses: 0, lightParses: 0, parseErrors: 0,
};

function countFlightType(code: string | null) {
  let k = code && /^[A-Z]{2,3}$/.test(code) ? code : "OTHER";
  if (!(k in counters.flightByType) && Object.keys(counters.flightByType).length >= 32) k = "OTHER";
  counters.flightByType[k] = (counters.flightByType[k] || 0) + 1;
}

/** Route one SFDPS payload: flights into the store, every other service
 *  counted and ignored. Returns flight messages applied. Never throws. */
export function handleSfdpsPayload(payload: string, store: SwimPlanStore, now = Date.now()): number {
  let service: SfdpsService;
  try { service = classifySfdpsPayload(payload); } catch { service = "UNKNOWN"; }
  counters.byService[service]++;
  if (service !== "FLIGHT") return 0;
  let applied = 0;
  let chunks: Array<{ chunk: string; messageTag: string | null }>;
  try { chunks = flightChunks(payload); } catch { counters.parseErrors++; return 0; }
  for (const { chunk, messageTag } of chunks) {
    try {
      const source = sourceOf(chunk, messageTag);
      countFlightType(source);
      let msgs: SwimFlightMessage[];
      if (needsFullParse(chunk, source)) {
        counters.fullParses++;
        // re-wrap in the message tag so message-level attributes still apply
        msgs = parseSfdpsMessages(messageTag ? `${messageTag}${chunk}</message>` : chunk);
      } else {
        counters.lightParses++;
        const f = lightFlight(chunk, messageTag);
        msgs = f ? [f] : [];
      }
      for (const f of msgs) { store.upsert(f, now); applied++; }
    } catch {
      counters.parseErrors++;
    }
  }
  return applied;
}

export function sfdpsCounters(): SfdpsCounters {
  return { ...counters, byService: { ...counters.byService }, flightByType: { ...counters.flightByType } };
}

/** Start the SFDPS consumer (no-op unless all SWIM_SFDPS_* env vars exist). */
export function startSfdps(
  store: SwimPlanStore,
  opts: Omit<Partial<SwimProductOptions>, "onPayload" | "product" | "envPrefix"> = {},
): Promise<SwimConnectorHandle> {
  return startSwimProduct({
    ...opts,
    product: "SFDPS",
    envPrefix: SFDPS_ENV_PREFIX,
    onPayload: (payload, now) => { handleSfdpsPayload(payload, store, now); },
  });
}

export function sfdpsStatus(store?: SwimPlanStore) {
  return { ...swimProductStatus("SFDPS", SFDPS_ENV_PREFIX), counters: sfdpsCounters(), storeSize: store?.size ?? 0, routeShape: routeShapeSamples() };
}

/** test seam */
export function _resetSfdpsCountersForTests(): void {
  counters.byService = { FLIGHT: 0, AIRSPACE_AIXM: 0, GENERAL_MESSAGE: 0, STATUS: 0, UNKNOWN: 0 };
  counters.flightByType = {}; counters.fullParses = 0; counters.lightParses = 0; counters.parseErrors = 0;
  routeShapes.clear(); routeShapeCounters.placed = 0; routeShapeCounters.expandedNoPoints = 0; routeShapeCounters.noExpanded = 0;
}
