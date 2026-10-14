// Airway expansion for SFDPS filed routes. ~52% of live plans carry only
// nasRouteText ("KBOS..HTO.J150.OOD..KATL"); the airway between two named fixes
// is published in FAA NASR AWY_BASE (public domain, scripts/build_nasr_airways.py).
// Only the unambiguous pattern FIX AIRWAY FIX is expanded, and only when both
// fixes lie on the airway; anything else (SID/STAR names, unknown idents,
// airway-to-airway joins, a fix not on the airway) is left unplaced — never guessed.
import fs from "fs";
import path from "path";
import { repoDataPath } from "./repoFiles";
import { lookupFix, lookupMagVar } from "./navFixes";
import { airportByIdent } from "./airportsIndex";
import { EARTH_RADIUS_NM, destinationPoint } from "../shared/flightPlanGeometry";

let table: Map<string, string[][]> | null = null;
let cycle: string | null = null;

export function loadNavAirways(jsonPath?: string): number {
  if (table) return table.size;
  const p = jsonPath || repoDataPath(path.join("datacore", "aircraft", "nasr_airways.json"));
  try {
    const d = JSON.parse(fs.readFileSync(p, "utf-8"));
    table = new Map(Object.entries(d.airways || {}) as [string, string[][]][]);
    cycle = d.cycle ?? null;
  } catch (e: unknown) {
    console.error("[navAirways] load:", e instanceof Error ? e.message : e);
    table = new Map();
  }
  return table.size;
}

/** idents strictly along `airway` from `from` to `to` inclusive, in travel order;
 *  null when no variant holds both or the match is ambiguous (different variants
 *  give different segments). */
export function airwaySegment(airway: string, from: string, to: string): string[] | null {
  if (!table) loadNavAirways();
  const variants = table!.get(airway.toUpperCase());
  if (!variants) return null;
  const hits: string[][] = [];
  for (const v of variants) {
    const i = v.indexOf(from), j = v.indexOf(to);
    if (i < 0 || j < 0 || i === j) continue;
    hits.push(i < j ? v.slice(i, j + 1) : v.slice(j, i + 1).reverse());
  }
  if (hits.length === 0) return null;
  const key = hits[0].join(" ");
  return hits.every((h) => h.join(" ") === key) ? hits[0] : null;
}

const AIRWAY_RE = /^(?:[JVQTABGRLMNH]|UL|UA|UB|UG|UM|UN|UR|UT|UV|UW|UY|UZ)\d{1,3}$/;

/** Place the airway-expanded interior of a filed route: tokens `FIX AWY FIX`
 *  become the fixes along the airway. Returns ordered points (deduped at joins). */
export function expandRouteText(routeText: string | null | undefined): { lat: number; lon: number; name: string }[] {
  if (!routeText) return [];
  if (!table) loadNavAirways();
  // "&"-notes and speed/level groups are not route elements; dots separate tokens
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  const names: string[] = [];
  const push = (n: string) => { if (names[names.length - 1] !== n) names.push(n); };
  let expandedAny = false;
  for (let k = 0; k < toks.length; k++) {
    const t = toks[k];
    if (table!.has(t) && AIRWAY_RE.test(t) && names.length && k + 1 < toks.length) {
      const seg = airwaySegment(t, names[names.length - 1], toks[k + 1]);
      if (seg) { for (const n of seg) push(n); expandedAny = true; k++; continue; }
      continue; // airway we cannot bound: drop it, keep the surrounding fixes
    }
    if (lookupFix(t)) push(t);
  }
  // bare named fixes alone are a separate, untested step: emit only when an airway
  // was actually expanded (the ~52% text-only case this module exists for)
  if (!expandedAny) return [];
  const out: { lat: number; lon: number; name: string }[] = [];
  for (const n of names) { const p = lookupFix(n); if (p) out.push({ ...p, name: n }); }
  return out;
}

const DIRECT_MAX_LEG_NM = 1200; // a longer leg between two "fixes" means an ident collision, not a route

function legNm(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const r = Math.PI / 180;
  const h = Math.sin(((b.lat - a.lat) * r) / 2) ** 2 +
    Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(((b.lon - a.lon) * r) / 2) ** 2;
  return 2 * EARTH_RADIUS_NM * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** Place a PURE direct-fix filed route (`SID..FIX..FIX..FIX..STAR`, no airway
 *  anywhere). >=3 resolved fixes required, consecutive legs must be plausible,
 *  and any airway token refuses the whole route (skipping it would draw a
 *  straight line across a segment we could not expand). */
export function placeDirectFixes(routeText: string | null | undefined): { lat: number; lon: number; name: string }[] {
  if (!routeText) return [];
  if (!table) loadNavAirways();
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  if (toks.some((t) => AIRWAY_RE.test(t) && table!.has(t))) return [];
  const out: { lat: number; lon: number; name: string }[] = [];
  for (const t of toks) {
    const p = lookupFix(t);
    if (p && out[out.length - 1]?.name !== t) out.push({ ...p, name: t });
  }
  if (out.length < 3) return [];
  for (let i = 1; i < out.length; i++) if (legNm(out[i - 1], out[i]) > DIRECT_MAX_LEG_NM) return [];
  return out;
}

/** Corridor bound for {@link placeFixesInCorridor}: the path through every
 *  fix may be at most this multiple of the airport-to-airport distance. */
export const CORRIDOR_MAX_DETOUR = 1.3;
/** Below this airport-to-airport distance a 1–2 fix route is left unplaced
 *  (terminal-area hops: the fixes ARE the departure/arrival procedures). */
export const CORRIDOR_MIN_DIRECT_NM = 40;
/** A fix this close to either airport adds nothing (often the airport's own VOR). */
const CORRIDOR_AIRPORT_NM = 2;

/** Place the "fewFixes" class (1–2 resolved fixes, no airway — 5,033 of the
 *  live unplaced plans on 2026-10-14) that placeDirectFixes refuses because,
 *  WITHOUT context, one or two names cannot be told from a same-named point
 *  elsewhere. The filed departure and arrival airports are that context:
 *  every fix must lie in the corridor between them (via-fix distance at most
 *  CORRIDOR_MAX_DETOUR x direct), and so must the whole path. A fix that
 *  resolves to a namesake far away (BEARZ resolves to Indiana) fails the
 *  bound and the route stays unplaced — never guessed. Any airway-shaped
 *  token refuses the route (an unexpanded airway would be drawn straight). */
export function placeFixesInCorridor(
  routeText: string | null | undefined,
  origin: { lat: number; lon: number },
  destination: { lat: number; lon: number },
): { lat: number; lon: number; name: string }[] {
  if (!routeText) return [];
  if (!table) loadNavAirways();
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  // stricter than placeDirectFixes: ANY airway-shaped token refuses, even one
  // NASR does not list (an unexpandable airway would be drawn as a straight leg)
  if (toks.some((t) => AIRWAY_RE.test(t))) return [];
  const direct = legNm(origin, destination);
  if (!(direct >= CORRIDOR_MIN_DIRECT_NM)) return [];
  const out: { lat: number; lon: number; name: string }[] = [];
  for (const t of toks) {
    const p = lookupFix(t);
    if (!p || out[out.length - 1]?.name === t) continue;
    if (legNm(origin, p) <= CORRIDOR_AIRPORT_NM || legNm(p, destination) <= CORRIDOR_AIRPORT_NM) continue;
    if (legNm(origin, p) + legNm(p, destination) > CORRIDOR_MAX_DETOUR * direct) return [];
    out.push({ ...p, name: t });
  }
  if (out.length === 0) return [];
  let via = legNm(origin, out[0]) + legNm(out[out.length - 1], destination);
  for (let i = 1; i < out.length; i++) via += legNm(out[i - 1], out[i]);
  return via <= CORRIDOR_MAX_DETOUR * direct ? out : [];
}

export type UnplacedReason = "noText" | "noFixResolved" | "fewFixes" | "airwayUnbounded" | "legTooLong" | "other";

/** Report-only: WHY a route text yields no placed points via expandRouteText /
 *  placeDirectFixes (diagnostic counter for the gate-1 funnel; never alters placement). */
export function classifyUnplacedRoute(routeText: string | null | undefined): UnplacedReason {
  if (!routeText || !routeText.trim()) return "noText";
  if (!table) loadNavAirways();
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  const hasAirway = toks.some((t) => AIRWAY_RE.test(t) && table!.has(t));
  const fixes: { lat: number; lon: number; name: string }[] = [];
  for (const t of toks) {
    const p = lookupFix(t);
    if (p && fixes[fixes.length - 1]?.name !== t) fixes.push({ ...p, name: t });
  }
  if (hasAirway) return "airwayUnbounded";
  if (fixes.length === 0) return "noFixResolved";
  if (fixes.length < 3) return "fewFixes";
  for (let i = 1; i < fixes.length; i++) if (legNm(fixes[i - 1], fixes[i]) > DIRECT_MAX_LEG_NM) return "legTooLong";
  return "other";
}

/** Report-only: route-text tokens that do NOT resolve in the NASR fix table (SID/STAR names,
 *  "DCT", airport idents, junk). Feeds the gate-1 unplaced-token diagnostic; never alters placement. */
export function unresolvedRouteTokens(routeText: string | null | undefined): string[] {
  if (!routeText) return [];
  if (!table) loadNavAirways();
  return routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter((t) => t && !lookupFix(t));
}

const LATLON_TOKEN_RE = /^\d{2,4}[NS]\/?\d{3,5}[EW]$/;
const RADIAL_TOKEN_RE = /^[A-Z]{2,4}\d{6}$/;

/** Report-only: shape of a fewFixes route text — how many distinct NASR fixes resolve, how many
 *  lat/lon waypoint tokens (pure geometry, no source needed) and fix-radial-distance tokens (need
 *  VOR magnetic variation) it carries. Sizes what a legitimate resolver could recover toward the
 *  >=3-fix bar; never alters placement. */
export function fewFixesProfile(routeText: string | null | undefined): { resolved: number; latlon: number; radial: number } {
  if (!routeText) return { resolved: 0, latlon: 0, radial: 0 };
  if (!table) loadNavAirways();
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  let resolved = 0, latlon = 0, radial = 0, prev = "";
  for (const t of toks) {
    if (lookupFix(t)) { if (prev !== t) resolved++; }
    else if (LATLON_TOKEN_RE.test(t)) latlon++;
    else if (RADIAL_TOKEN_RE.test(t)) radial++;
    prev = t;
  }
  return { resolved, latlon, radial };
}

const RADIAL_PARTS_RE = /^([A-Z]{2,4})(\d{3})(\d{3})$/;
const RADIAL_MAX_NM = 400; // a fix-radial-distance offset beyond this is a parse collision, not a waypoint

/** Resolve a fix-radial-distance token (`SNS285053` = SNS radial 285 at 53 NM). The radial is MAGNETIC
 *  from the station, so true bearing = radial + the station's FILED variation (east positive). Only
 *  navaids carrying a variation resolve; anything else returns null — never guessed. */
export function resolveRadialToken(token: string): { lat: number; lon: number; name: string; magYear: number } | null {
  const m = RADIAL_PARTS_RE.exec(token.toUpperCase());
  if (!m) return null;
  const radial = Number(m[2]), dist = Number(m[3]);
  if (radial > 360 || dist < 1 || dist > RADIAL_MAX_NM) return null;
  const base = lookupFix(m[1]), mv = lookupMagVar(m[1]);
  if (!base || !mv) return null;
  const p = destinationPoint(base, (((radial + mv.deg) % 360) + 360) % 360, dist);
  return { ...p, name: token.toUpperCase(), magYear: mv.year };
}

/** SHADOW ONLY (gate-1 resolver, not wired into placement): the points placeDirectFixes would give
 *  if fix-radial-distance tokens were also placed. Same bar as placeDirectFixes (>=3 points, no airway,
 *  plausible legs). `radial` counts how many of the points came from radial tokens. */
export function placeDirectFixesWithRadials(routeText: string | null | undefined): { points: { lat: number; lon: number; name: string }[]; radial: number } {
  const none = { points: [], radial: 0 };
  if (!routeText) return none;
  if (!table) loadNavAirways();
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  if (toks.some((t) => AIRWAY_RE.test(t) && table!.has(t))) return none;
  const points: { lat: number; lon: number; name: string }[] = [];
  let radial = 0;
  for (const t of toks) {
    const fix = lookupFix(t);
    const p = fix ? { ...fix, name: t } : resolveRadialToken(t);
    if (!p || points[points.length - 1]?.name === p.name) continue;
    if (!fix) radial++;
    points.push({ lat: p.lat, lon: p.lon, name: p.name });
  }
  if (points.length < 3) return none;
  for (let i = 1; i < points.length; i++) if (legNm(points[i - 1], points[i]) > DIRECT_MAX_LEG_NM) return none;
  return { points, radial };
}

// ── FULL ROUTE-STRING PARSER (2026-10-14, deviation program P1) ─────────────
// One pass over an FAA NAS route text ("KBOS..HTO.J150.OOD..KATL") or an
// ICAO Item 15 route ("N0450F350 DCT HTO J150 OOD DCT"), built from the
// primitives above (airwaySegment, resolveRadialToken, the NASR gazetteer)
// instead of refusing a whole route over one token it cannot place. Every
// element is either PLACED (a point with its source) or UNRESOLVED (kept,
// in order, as the gap before the next placed point) — an unresolved stretch
// is drawn lighter and deviation is never judged along it. Never throws.
//
// Elements: DCT / ".." directs; named fixes + navaids (NASR); airports at the
// route's ends (the plan adds them) and mid-route (airportsIndex); lat/lon in
// every published form (DDMMN/DDDMMW with or without "/", DDN DDDW, DDMMSS,
// ARINC 5-character oceanic "5030N" / "50N30"); fix-radial-distance
// (FWA326067); airway chains with entry/exit fixes, including airway-to-
// airway joins at their common fix; SIDs / STARs with transitions (expanded by
// the caller's CIFP-backed `procedure` hook; unresolved without it); NAT and
// PACOTS track designators (the caller's `oceanicTrack` hook; unresolved
// without it); speed/level groups (N0450F350, M082F350, K0830S1010, also
// "FIX/N0450F350") and IFR/VFR markers are stripped; "./." (FAA: route
// amended from present position) marks the leg before the next point as
// unknown; "KCMH/2041" (ETA suffix) is the airport.
//
// Namesake guard (the same rules the gate-1-validated chain used): a named
// point more than DIRECT_MAX_LEG_NM from the previous placed point is a
// same-named fix elsewhere and stays unresolved; a route that places only
// 1-2 named points is kept only inside the filed airports' corridor
// (CORRIDOR_MAX_DETOUR), and without the airports it places nothing.

export type RoutePointSource = "fix" | "airport" | "airway" | "radial" | "latlon" | "sid" | "star" | "oceanic";

export interface ParsedRoutePoint {
  lat: number;
  lon: number;
  name: string;
  source: RoutePointSource;
  /** route elements between the previous placed point (or the departure) and
   *  this one that could not be placed — the leg arriving here is unresolved */
  gapBefore?: string[];
}

export interface ParsedProcedure {
  /** the token as filed ("BLEWE4") */
  filed: string;
  /** the CIFP procedure used ("BLEWE5"), null when not expanded */
  id: string | null;
  transition: string | null;
  expanded: boolean;
  versionMismatch?: boolean;
}

export interface ParsedRoute {
  points: ParsedRoutePoint[];
  /** unresolved elements after the last placed point (the leg into the arrival) */
  gapAfter: string[];
  /** every unresolved element, in route order */
  unresolved: string[];
  /** speed/level groups and IFR/VFR markers removed (never route points) */
  stripped: number;
  /** FAA "./." — the route was amended from the aircraft's present position */
  amendedFromPresentPosition: boolean;
  sid: ParsedProcedure | null;
  star: ParsedProcedure | null;
  /** airway segments expanded between their entry and exit fixes */
  airways: number;
  /** the placement class (back-compatible with SwimRouteKind; null = nothing placed) */
  kind: "airway" | "direct" | "radial" | "corridor" | null;
}

export interface RouteParseContext {
  /** filed departure / arrival idents as carried ("KBOS") */
  departure?: string | null;
  arrival?: string | null;
  /** their positions (namesake / corridor context) */
  origin?: { lat: number; lon: number } | null;
  destination?: { lat: number; lon: number } | null;
  /** SID/STAR expansion (CIFP). `transition` is the adjacent filed token. */
  procedure?: ((kind: "SID" | "STAR", airport: string, token: string, transition: string | null) =>
    { id: string; transition: string | null; versionMismatch?: boolean; points: { name: string; lat: number; lon: number }[] } | null) | null;
  /** NAT / PACOTS track definitions (none are ingested yet) */
  oceanicTrack?: ((id: string) => { name: string; lat: number; lon: number }[] | null) | null;
}

/** ICAO speed/level group: N0450F350, M082F350, K0830S1010, N0450A045, N0450VFR */
const SPEED_LEVEL_RE = /^[NKM]\d{3,4}(?:[FAMS]\d{3,4}|VFR)$/;
const MARKER_TOKENS = new Set(["DCT", "IFR", "VFR"]);
const ETA_SUFFIX_RE = /^([A-Z0-9]{3,4})\/\d{4}$/;
/** a SID/STAR designator as filed: name + version digit (+ optional letter) */
const PROC_TOKEN_RE = /^[A-Z]{3,5}\d[A-Z]?$/;
const OCEANIC_TRACK_RE = /^(?:NAT[A-Z]|PACOTS?\d{1,2}|PACOTS?[A-Z])$/;
const ICAO_AIRPORT_RE = /^[A-Z]{4}$/;
/** ARINC 424 5-character oceanic: N = N/W, E = N/E, S = S/E, W = S/W */
const ARINC_HEMI: Record<string, [number, number]> = { N: [1, -1], E: [1, 1], S: [-1, 1], W: [-1, -1] };

const dms = (d: string, m?: string, s?: string): number | null => {
  const mm = m ? Number(m) : 0, ss = s ? Number(s) : 0;
  if (mm >= 60 || ss >= 60) return null;
  return Number(d) + mm / 60 + ss / 3600;
};

/** Pure: a lat/lon route element in any published form, or null. */
export function parseLatLonToken(token: string): { lat: number; lon: number } | null {
  const t = token.toUpperCase();
  let lat: number | null = null, lon: number | null = null;
  let m = /^(\d{2})(\d{2})?(\d{2})?([NS])\/?(\d{3})(\d{2})?(\d{2})?([EW])$/.exec(t);
  if (m && !!m[3] === !!m[7] && (!!m[2] || !m[3]) && (!!m[6] || !m[7])) {
    const la = dms(m[1], m[2], m[3]), lo = dms(m[5], m[6], m[7]);
    if (la != null && lo != null) { lat = la * (m[4] === "S" ? -1 : 1); lon = lo * (m[8] === "W" ? -1 : 1); }
  } else if ((m = /^(\d{2})(\d{2})([NESW])$/.exec(t))) {
    const [sLa, sLo] = ARINC_HEMI[m[3]];
    lat = Number(m[1]) * sLa; lon = Number(m[2]) * sLo;
  } else if ((m = /^(\d{2})([NESW])(\d{2})$/.exec(t))) {
    const [sLa, sLo] = ARINC_HEMI[m[2]];
    lat = Number(m[1]) * sLa; lon = (100 + Number(m[3])) * sLo;
  }
  if (lat == null || lon == null || Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

/** Pure: the route's elements, normalized (notes, speed/level groups and
 *  markers removed; "./." kept as the "/" element; ETA suffixes cut). */
export function tokenizeRoute(routeText: string | null | undefined): { tokens: string[]; stripped: number } {
  if (!routeText) return { tokens: [], stripped: 0 };
  const out: string[] = [];
  let stripped = 0;
  for (let t of routeText.toUpperCase().split("&")[0].split(/[\s.]+/)) {
    if (!t) continue;
    if (SPEED_LEVEL_RE.test(t) || MARKER_TOKENS.has(t)) { stripped++; continue; }
    const sl = /^(.+)\/([NKM]\d{3,4}(?:[FAMS]\d{3,4}|VFR))$/.exec(t);
    if (sl) { t = sl[1]; stripped++; }
    const eta = ETA_SUFFIX_RE.exec(t);
    if (eta) t = eta[1];
    out.push(t);
  }
  return { tokens: out, stripped };
}

const isAirwayToken = (t: string) => AIRWAY_RE.test(t);

/** where airway `a` (entered at `from`, in the direction of `toward` when
 *  given) meets airway `b`: the first fix after `from` that lies on both */
function airwayJoin(a: string, from: string, b: string): string | null {
  if (!table) loadNavAirways();
  const va = table!.get(a), vb = table!.get(b);
  if (!va || !vb) return null;
  const onB = new Set(vb.flat());
  const found = new Set<string>();
  for (const v of va) {
    const i = v.indexOf(from);
    if (i < 0) continue;
    // either direction along the variant: the join is the nearest common fix
    for (const dir of [1, -1]) {
      for (let k = i + dir; k >= 0 && k < v.length; k += dir) {
        if (onB.has(v[k])) { found.add(v[k]); break; }
      }
    }
  }
  return found.size === 1 ? Array.from(found)[0] : null;
}

/**
 * Pure (given its hooks): parse a filed route into placed points + unresolved
 * gaps. See the section header for the grammar and the guards.
 */
export function parseFiledRoute(routeText: string | null | undefined, ctx: RouteParseContext = {}): ParsedRoute {
  const res: ParsedRoute = {
    points: [], gapAfter: [], unresolved: [], stripped: 0, amendedFromPresentPosition: false,
    sid: null, star: null, airways: 0, kind: null,
  };
  try {
    if (!table) loadNavAirways();
    const { tokens, stripped } = tokenizeRoute(routeText);
    res.stripped = stripped;
    const dep = ctx.departure?.toUpperCase() ?? null, arr = ctx.arrival?.toUpperCase() ?? null;
    // the route's own end airports are the plan's endpoints, not route points
    let lo = 0, hi = tokens.length - 1;
    while (lo <= hi && (tokens[lo] === dep || (lo === 0 && !dep && ICAO_AIRPORT_RE.test(tokens[lo]) && !lookupFix(tokens[lo]) && airportByIdent(tokens[lo])))) lo++;
    while (hi >= lo && (tokens[hi] === arr || (hi === tokens.length - 1 && !arr && ICAO_AIRPORT_RE.test(tokens[hi]) && !lookupFix(tokens[hi]) && airportByIdent(tokens[hi])))) hi--;
    const depId = dep ?? (lo > 0 ? tokens[0] : null);
    const arrId = arr ?? (hi < tokens.length - 1 ? tokens[tokens.length - 1] : null);

    let gap: string[] = [];
    const unresolved = (t: string) => { gap.push(t); res.unresolved.push(t); };
    const lastPt = () => res.points[res.points.length - 1] ?? null;
    // the namesake guard compares CONSECUTIVE placed points only (the chain's
    // rule): the departure is no anchor — an international plan's text starts
    // at its first US fix, and after "./." the leg's start is unknown
    let prevLL: { lat: number; lon: number } | null = null;
    const place = (p: { lat: number; lon: number }, name: string, source: RoutePointSource, guard: boolean): boolean => {
      const last = lastPt();
      if (last && last.name === name) return true; // the same point twice (join / transition fix)
      if (guard && prevLL && legNm(prevLL, p) > DIRECT_MAX_LEG_NM) return false; // namesake elsewhere
      res.points.push({ lat: p.lat, lon: p.lon, name, source, ...(gap.length ? { gapBefore: gap } : {}) });
      gap = [];
      prevLL = p;
      return true;
    };
    const placeNamed = (name: string, source: RoutePointSource): boolean => {
      const p = lookupFix(name);
      return !!p && place(p, name, source, true);
    };

    for (let k = lo; k <= hi; k++) {
      const t = tokens[k];
      if (t === "/") { res.amendedFromPresentPosition = true; unresolved("./."); prevLL = null; continue; }

      // SID: the first element after the departure; STAR: the last before the arrival
      const isProc = PROC_TOKEN_RE.test(t) && !lookupFix(t);
      if (isProc && (k === lo || k === hi)) {
        const kind = k === lo ? "SID" : "STAR";
        const apt = kind === "SID" ? depId : arrId;
        const adj = kind === "SID" ? tokens[k + 1] ?? null : tokens[k - 1] ?? null;
        const exp = apt && ctx.procedure ? ctx.procedure(kind, apt, t, adj) : null;
        const info: ParsedProcedure = {
          filed: t, id: exp?.id ?? null, transition: exp?.transition ?? null, expanded: !!exp?.points.length,
          ...(exp?.versionMismatch ? { versionMismatch: true } : {}),
        };
        if (kind === "SID") res.sid = info; else res.star = info;
        if (exp && exp.points.length) {
          // procedure fixes come from CIFP for THIS airport: no namesake risk
          for (const p of exp.points) place(p, p.name, kind === "SID" ? "sid" : "star", false);
        } else {
          unresolved(t);
        }
        continue;
      }

      if (isAirwayToken(t)) {
        const entry = lastPt();
        let next = tokens[k + 1];
        // an airway needs a placed entry fix and an exit inside the route
        if (!table!.has(t) || !entry || !next || k + 1 > hi) { unresolved(t); continue; }
        // airway-to-airway: route along `t` to its join with the next airway
        let via = t;
        let ok = true;
        while (next && isAirwayToken(next) && table!.has(next)) {
          const join = airwayJoin(via, lastPt()!.name, next);
          const seg = join ? airwaySegment(via, lastPt()!.name, join) : null;
          if (!seg) { ok = false; break; }
          for (const n of seg.slice(1)) placeNamed(n, "airway");
          res.airways++;
          via = next; k++; next = tokens[k + 1];
        }
        if (!ok || !next) { unresolved(via); continue; }
        const seg = airwaySegment(via, lastPt()!.name, next);
        if (!seg) { unresolved(via); continue; }
        for (const n of seg.slice(1)) {
          if (!placeNamed(n, "airway")) unresolved(n);
        }
        res.airways++;
        k++; // the exit fix was placed as the segment's last point
        continue;
      }

      if (OCEANIC_TRACK_RE.test(t)) {
        const trk = ctx.oceanicTrack ? ctx.oceanicTrack(t) : null;
        if (trk && trk.length) { for (const p of trk) place(p, p.name, "oceanic", false); } else unresolved(t);
        continue;
      }

      const ll = parseLatLonToken(t);
      if (ll) { place(ll, t, "latlon", false); continue; }
      const rad = resolveRadialToken(t);
      if (rad) { if (!place(rad, t, "radial", true)) unresolved(t); continue; }
      if (placeNamed(t, "fix")) continue;
      if (ICAO_AIRPORT_RE.test(t)) {
        const a = airportByIdent(t);
        if (a && place({ lat: a.la, lon: a.lo }, t, "airport", true)) continue;
      }
      unresolved(t);
    }
    res.gapAfter = gap;

    // ── placement class + the few-point corridor guard ──
    // airports resolve by unique ICAO ident and lat/lon is exact: only NASR
    // names (fixes, navaids, radial bases) can be a same-named point elsewhere
    const named = res.points.filter((p) => p.source === "fix" || p.source === "radial");
    const core = res.points.filter((p) => p.source !== "sid" && p.source !== "star");
    // CIFP-expanded SID/STAR points anchor the route at its airports, so they
    // count toward the >= 3 points the direct rule needs (placeDirectFixes)
    if (res.airways > 0) res.kind = "airway";
    else if (res.points.length >= 3 || (core.length === 0 && res.points.length > 0)) {
      // >= 3 points, or a route that is all CIFP procedure (no namesake risk)
      res.kind = named.some((p) => p.source === "radial") ? "radial" : "direct";
    }
    else if (core.length > 0) {
      // 1-2 points: a same-named fix elsewhere cannot be ruled out without
      // the filed airports (placeFixesInCorridor's rule, gate-1 2026-10-14)
      const o = ctx.origin, d = ctx.destination;
      const fits = named.length === 0 || (!!o && !!d && legNm(o, d) >= CORRIDOR_MIN_DIRECT_NM && named.every((p) => inCorridor(p, o!, d!)));
      if (fits) res.kind = "corridor";
      else {
        // demote the unguardable named points to unresolved, in route order:
        // each one (and its own gap) joins the gap before the next kept point
        const kept: ParsedRoutePoint[] = [];
        let carry: string[] = [];
        for (const p of res.points) {
          const keepIt = p.source !== "fix" && p.source !== "radial";
          if (!keepIt) { carry.push(...(p.gapBefore ?? []), p.name); continue; }
          const g = [...carry, ...(p.gapBefore ?? [])];
          kept.push({ ...p, ...(g.length ? { gapBefore: g } : {}) });
          carry = [];
        }
        const demoted = res.points.filter((p) => !kept.some((q) => q.name === p.name && q.source === p.source)).map((p) => p.name);
        res.unresolved = [...res.unresolved, ...demoted];
        res.gapAfter = [...carry, ...res.gapAfter];
        res.points = kept;
        res.kind = kept.some((p) => p.source !== "sid" && p.source !== "star") ? "corridor" : null;
      }
    }
  } catch (e: unknown) {
    // never crash a feed over one route string: keep what was placed, say so
    console.warn("[navAirways] parseFiledRoute:", e instanceof Error ? e.message : e);
  }
  return res;
}

/** a point lies in the filed airports' corridor (placeFixesInCorridor's per-fix rule) */
function inCorridor(p: { lat: number; lon: number }, o: { lat: number; lon: number }, d: { lat: number; lon: number }): boolean {
  return legNm(o, p) + legNm(p, d) <= CORRIDOR_MAX_DETOUR * legNm(o, d);
}

export const navAirwaysCycle = (): string | null => cycle;
export function resetNavAirways(): void { table = null; cycle = null; }
