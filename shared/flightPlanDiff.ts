// Flight plan versions (2026-11-16): the pure half.
//
// Three questions, each a pure function so the server module that stores the
// versions (server/flightPlanVersionStore.ts) and the one that serves them
// (server/flightPlanHistory.ts) stay free of geometry:
//
//   classifyPlanChange  what an amendment changed between two versions of one
//                       flight plan (reroute, fixes added/removed, shortcut,
//                       direct-to, destination, cruise altitude), where on the
//                       route the change starts and how far sideways it goes
//   positionAtTime      where the aircraft was when the amendment took effect,
//                       from its own ADS-B fixes; never a silent interpolation
//   buildOffRouteEpisodes
//                       the stretches where the flown track left the route in
//                       force, from the deviation state machine's own verdict
//                       per fix (server/flightPlans.ts DeviationTracker); this
//                       file only groups those verdicts, it does not judge
//
// HONESTY: nothing here knows WHY an aircraft left its route. An episode is
// either followed by an amendment (then it was a reroute) or it is not, and
// the cause is reported as not determined. No weather, no traffic, no guess.

import {
  crossTrackNm, interpolateGC, isTerminalVectoring, TERMINAL_AREA_NM, type LatLon,
} from "./flightPlanGeometry";

// ── 1. classified diff ──────────────────────────────────────────────────────

export type PlanChangeKind =
  | "REROUTE"            // the placed route geometry moved sideways by >= REROUTE_MIN_LATERAL_NM
  | "FIXES_ADDED"        // named points on the new route that the old one did not have
  | "FIXES_REMOVED"      // named points on the old route that the new one dropped
  | "SHORTCUT"           // interior fixes dropped and none added: direct between the ones kept
  | "DIRECT_TO"          // amended from the aircraft's position ("./.") AND the route ahead changed
  | "DESTINATION_CHANGE"
  | "ALTITUDE_CHANGE"    // filed cruise altitude
  | "ROUTE_TEXT_ONLY";   // the text changed, the geometry we could place did not

/** A geometry change smaller than this is the same line with different points
 *  on it (a fix added along an unchanged leg), not a reroute. */
export const REROUTE_MIN_LATERAL_NM = 1;

/** One version of a plan, reduced to what the comparison needs. */
export interface PlanShape {
  departure: string | null;
  arrival: string | null;
  cruiseAltFt: number | null;
  routeText: string | null;
  /** placed route vertices in route order; `name` where the plan named them */
  fixes: Array<LatLon & { name?: string }>;
  /** the drawable path (may be densified) the lateral comparison measures */
  path: LatLon[];
  /** true when `path` is a great-circle stand-in, not placed route geometry */
  pathEstimated: boolean;
  /** the route text starts from the aircraft's position at the time ("./.") */
  amendedFromPresentPosition: boolean;
}

export interface RoutePointRef { name: string | null; lat: number; lon: number }

export interface PlanChange {
  kinds: PlanChangeKind[];
  /** named fixes only, in route order; the two airports and a
   *  present-position token are never listed */
  fixesAdded: string[];
  /** fixes the new route no longer has. For a present-position route a fix
   *  dropped from the START is listed here only when it was SKIPPED (it lay
   *  ahead of the aircraft); see fromPresentPosition. */
  fixesRemoved: string[];
  /** every fix that sat BEFORE the first fix both versions share and is gone,
   *  flown or skipped */
  fixesRemovedFromStart: string[];
  destination: { from: string; to: string } | null;
  cruiseAltFt: { from: number; to: number; deltaFt: number } | null;
  /** DIRECT_TO only: the first named fix of the amended route */
  directTo: { fix: string | null } | null;
  routeTextChanged: boolean;
  /** set when the new version is a present-position route ("./."): what
   *  happened to the fixes dropped from its start. `judged` is false when
   *  fixes were dropped and no position for that moment is known; they are
   *  then listed in fixesRemovedFromStart only and nothing is flagged for
   *  them. offRouteNm is how far the aircraft was from the old route. */
  fromPresentPosition: {
    judged: boolean;
    positionBasis: "FAA_ROUTE_TEXT" | "ADSB_FIX" | null;
    fixesFlown: string[];
    fixesSkipped: string[];
    offRouteNm: number | null;
  } | null;
  /** the route text changed only in where it starts (a present-position
   *  restatement, flown fixes dropped, or the full text again): nothing about
   *  the route ahead changed. Such a version is not an amendment. */
  restatement: boolean;
  geometry: {
    changed: boolean;
    /** the first leg that differs: `from` is the last point both versions
     *  share (null = the routes differ from their first point), `to` is where
     *  the new route goes next, `was` where the old one went */
    firstDifferingLeg: { index: number; from: RoutePointRef | null; to: RoutePointRef | null; was: RoutePointRef | null } | null;
    /** largest distance between the two paths, nm (each path's vertices
     *  against the other path; for a present-position amendment only the new
     *  path against the old one, because the old path's flown part is not a
     *  change). null when either version has no path. */
    maxLateralNm: number | null;
    /** true when either path is a great-circle stand-in: the number then
     *  compares an estimate, not two filed routes */
    lateralEstimated: boolean;
    note: string | null;
  };
}

const ref = (p: (LatLon & { name?: string }) | undefined): RoutePointRef | null =>
  p ? { name: p.name ?? null, lat: p.lat, lon: p.lon } : null;
const posKey = (p: LatLon) => `${p.lat.toFixed(3)},${p.lon.toFixed(3)}`;
/** two route points are the same point: same name, or (either unnamed) same place */
const samePoint = (a: LatLon & { name?: string }, b: LatLon & { name?: string }) =>
  a.name && b.name ? a.name === b.name : posKey(a) === posKey(b);
const round1 = (x: number) => Math.round(x * 10) / 10;

function maxDistanceNm(from: LatLon[], to: LatLon[]): number | null {
  if (!from.length || !to.length) return null;
  let max = 0;
  for (const p of from) {
    const xt = crossTrackNm(p, to);
    if (xt && xt.nm > max) max = xt.nm;
  }
  return max;
}

// ── present-position routes ────────────────────────────────────────────────
//
// The FAA re-issues an airborne flight's route from where the aircraft is:
// "KORD./.ADIME..GERBS" (the part before "./." is gone), often with the
// aircraft's own position as the first element, a fix-radial-distance token
// ("KCVG./.APE227059..AHTIY": 59 nm out on the APE 227 radial). The same
// stream also keeps sending the full text. Read live 2026-11-17 on 111
// flights, of 307 recorded changes: 16% restated the same remaining route
// (the position token moved, or "./." appeared), 6% were the older, fuller
// text again, and 33% dropped fixes from the start. A fix dropped from the
// start was either flown already (nothing was amended) or skipped by a
// clearance direct to a later fix (a real amendment). Only the aircraft's
// position at that moment tells the two apart, so that is what decides.

/** a fix-radial-distance token: navaid ident, 3-digit radial, 3-digit nm */
export const PRESENT_POSITION_TOKEN = /^[A-Z0-9]{2,5}\d{6}$/;
/** a dropped fix further than this AHEAD of the aircraft along the old route
 *  was skipped, not flown. Wide enough for a position a few miles stale. */
export const HEAD_AHEAD_TOLERANCE_NM = 5;

/** The route elements still ahead, from the route text alone: for a
 *  present-position route what follows "./." (without the position token),
 *  otherwise everything after the departure airport. The trailing "/hhmm"
 *  (time en route) is not part of the route. */
export function remainingRouteTokens(routeText: string | null | undefined): { tokens: string[]; fromPresentPosition: boolean; positionToken: string | null } {
  const text = String(routeText ?? "").trim().replace(/\/\d{4}$/, "");
  const cut = text.lastIndexOf("./.");
  if (cut >= 0) {
    const tokens = text.slice(cut + 3).split(/\.+/).filter(Boolean);
    const positionToken = tokens.length && PRESENT_POSITION_TOKEN.test(tokens[0]) ? tokens.shift()! : null;
    return { tokens, fromPresentPosition: true, positionToken };
  }
  return { tokens: text.split(/\.+/).filter(Boolean).slice(1), fromPresentPosition: false, positionToken: null };
}

const endsWith = (long: string[], short: string[]) =>
  short.length <= long.length && short.every((t, i) => long[long.length - short.length + i] === t);

/** Does `next` only RESTATE the route `prev` already gave? True when the
 *  remaining route is the same element for element (the position token moved,
 *  or "./." was put in front of a route held in full), and when `prev` is a
 *  present-position route and `next` is the full text whose end is exactly
 *  that remaining route (the older text sent again). A present-position route
 *  with FEWER elements left is not a restatement: fixes were dropped. */
export function isRouteRestatement(prevText: string | null | undefined, nextText: string | null | undefined): boolean {
  if (!prevText || !nextText || prevText === nextText) return false;
  const a = remainingRouteTokens(prevText), b = remainingRouteTokens(nextText);
  if (!a.fromPresentPosition && !b.fromPresentPosition) return false;
  if (!a.tokens.length || !b.tokens.length) return false;
  if (a.tokens.length === b.tokens.length) return endsWith(a.tokens, b.tokens);
  return a.fromPresentPosition && !b.fromPresentPosition && endsWith(b.tokens, a.tokens);
}

type Fix = LatLon & { name?: string };
interface KnownPart { names: string[]; fixes: Fix[]; path: LatLon[]; position: LatLon | null }

/** The part of a version its route text actually describes. For a
 *  present-position route that excludes the departure airport (the leg out of
 *  it is not in the text) and the position token, which is where the aircraft
 *  was and not a point on the route. */
function knownPart(s: PlanShape, airports: Set<string>): KnownPart {
  let fixes: Fix[] = s.fixes.filter((f) => !!f.name && !airports.has(f.name));
  let path = s.path;
  let position: LatLon | null = null;
  if (s.amendedFromPresentPosition) {
    const first = fixes[0];
    if (first?.name && PRESENT_POSITION_TOKEN.test(first.name)) {
      position = { lat: first.lat, lon: first.lon };
      fixes = fixes.slice(1);
    }
    // the drawn path starts at the departure airport with a leg nobody filed
    // (and may be densified): the described route starts at the first fix kept.
    // With no kept fix on the path, only the destination is still described.
    const from = fixes[0] ?? s.fixes[s.fixes.length - 1];
    const at = from ? path.findIndex((p) => posKey(p) === posKey(from)) : -1;
    path = at >= 0 ? path.slice(at) : [];
  }
  return { names: fixes.map((f) => f.name!), fixes, path, position };
}

/** What changed from `prev` to `next`. Pure; never throws on empty shapes.
 *  `at` is where the aircraft was when `next` took effect, from its own
 *  ADS-B fixes; it is used only when `next` is a present-position route whose
 *  text carries no position of its own. */
export function classifyPlanChange(prev: PlanShape, next: PlanShape, opts: { at?: LatLon | null } = {}): PlanChange {
  const kinds: PlanChangeKind[] = [];
  const airports = new Set([prev.departure, prev.arrival, next.departure, next.arrival].filter((x): x is string => !!x));
  const pk = knownPart(prev, airports), nk = knownPart(next, airports);
  const prevNames = pk.names, nextNames = nk.names;
  const prevSet = new Set(prevNames), nextSet = new Set(nextNames);
  const uniq = (xs: string[]) => Array.from(new Set(xs));
  const presentPosition = next.amendedFromPresentPosition;
  const eitherPresent = presentPosition || prev.amendedFromPresentPosition;
  const firstShared = prevNames.findIndex((n) => nextSet.has(n));
  const fixesRemovedFromStart = uniq(firstShared < 0 ? [] : prevNames.slice(0, firstShared).filter((n) => !nextSet.has(n)));
  // the older, fuller text after a present-position route names the flown
  // part again: those are not fixes ATC added
  const nextFirstShared = nextNames.findIndex((n) => prevSet.has(n));
  const renamedHead = new Set(prev.amendedFromPresentPosition && nextFirstShared > 0 ? nextNames.slice(0, nextFirstShared) : []);
  const fixesAdded = uniq(nextNames.filter((n) => !prevSet.has(n) && !renamedHead.has(n)));

  // fixes dropped from the start of a present-position route: flown, or skipped?
  let head: PlanChange["fromPresentPosition"] = null;
  const position = presentPosition ? (nk.position ?? opts.at ?? null) : null;
  if (presentPosition) {
    const basis: NonNullable<PlanChange["fromPresentPosition"]>["positionBasis"] = nk.position ? "FAA_ROUTE_TEXT" : opts.at ? "ADSB_FIX" : null;
    const flown: string[] = [], skipped: string[] = [];
    const here = position && prev.path.length >= 2 ? crossTrackNm(position, prev.path) : null;
    if (here) {
      for (const n of fixesRemovedFromStart) {
        const f = pk.fixes.find((x) => x.name === n);
        const xt = f ? crossTrackNm(f, prev.path) : null;
        if (xt && xt.alongNm > here.alongNm + HEAD_AHEAD_TOLERANCE_NM) skipped.push(n);
        else flown.push(n);
      }
    }
    head = {
      judged: !!here || !fixesRemovedFromStart.length,
      positionBasis: here ? basis : null,
      fixesFlown: flown, fixesSkipped: skipped,
      offRouteNm: here ? round1(here.nm) : null,
    };
  }
  const interiorRemoved = prevNames.filter((n) => !nextSet.has(n) && !fixesRemovedFromStart.includes(n));
  // a fix dropped from the start counts as removed only when it was skipped
  // (or when the route is not a present-position one, as before)
  const fixesRemoved = uniq(presentPosition ? [...(head?.fixesSkipped ?? []), ...interiorRemoved] : prevNames.filter((n) => !nextSet.has(n)));

  // geometry: the placed vertices, in order
  const a = prev.fixes, b = next.fixes;
  let i = 0;
  while (i < a.length && i < b.length && samePoint(a[i], b[i])) i++;
  const vertexListsDiffer = a.length > 0 && b.length > 0 && !(i === a.length && i === b.length);
  const routeTextChanged = !!prev.routeText && !!next.routeText && prev.routeText !== next.routeText;
  const destination = prev.arrival && next.arrival && prev.arrival !== next.arrival
    ? { from: prev.arrival, to: next.arrival } : null;
  // the paths are compared whenever anything about the route changed: a change
  // in UNNAMED points shows only there. A present-position route describes
  // only what is still ahead, so only that part is measured against the other
  // version's whole path.
  let maxLateralNm: number | null = null;
  if ((vertexListsDiffer || routeTextChanged || destination) && prev.path.length >= 2 && next.path.length >= 2) {
    const forward = prev.amendedFromPresentPosition && !presentPosition ? null : maxDistanceNm(nk.path, prev.path);
    const back = presentPosition ? null
      : prev.amendedFromPresentPosition ? maxDistanceNm(pk.path, next.path)
      : maxDistanceNm(prev.path, next.path);
    // a present-position route with a known position: how far the route moved
    // at each fix it no longer has (skipped or dropped further on), measured
    // from the line the aircraft now flies — its position, then the route kept
    let removedLateralNm = 0;
    if (position && fixesRemoved.length) {
      const flownNow = [position, ...(nk.path.length ? nk.path : next.fixes.slice(-1))];
      for (const n of fixesRemoved) {
        const f = pk.fixes.find((x) => x.name === n);
        const off = f && flownNow.length >= 2 ? crossTrackNm(f, flownNow) : null;
        if (off && off.nm > removedLateralNm) removedLateralNm = off.nm;
      }
    }
    maxLateralNm = round1(Math.max(forward ?? 0, back ?? 0, removedLateralNm));
  }
  // between two full routes a changed point list is a change, as before. With
  // a present-position route on either side the lists always differ (one has
  // no start), so only named fixes added, removed or skipped count.
  const fixesDiffer = eitherPresent ? (fixesAdded.length > 0 || fixesRemoved.length > 0) : vertexListsDiffer;
  const changed = fixesDiffer || (maxLateralNm != null && maxLateralNm >= REROUTE_MIN_LATERAL_NM);
  const lateralEstimated = prev.pathEstimated || next.pathEstimated;
  const notes: string[] = [];
  if (changed && maxLateralNm == null) notes.push("one of the two versions has no placed path, so the size of the change is not measured");
  if (changed && lateralEstimated) notes.push("one of the two paths is a great-circle stand-in for route text we could not place, so the distance compares an estimate");
  if (presentPosition && fixesRemovedFromStart.length) {
    notes.push(head && head.positionBasis
      ? `the route was amended from the aircraft's position at the time (${head.positionBasis === "FAA_ROUTE_TEXT" ? "given in the route text" : "its ADS-B fix"}): of the fixes dropped from the start, ${head.fixesFlown.length} lay behind it (flown) and ${head.fixesSkipped.length} ahead of it (skipped)`
      : "the route was amended from the aircraft's position at the time, and that position is not known: whether the fixes dropped from the start were flown or skipped is not determined, so they are not flagged");
  } else if (changed && presentPosition) {
    notes.push("the route was amended from the aircraft's position at the time: only the part still ahead is compared");
  }

  const cruiseAltFt = prev.cruiseAltFt != null && next.cruiseAltFt != null && prev.cruiseAltFt !== next.cruiseAltFt
    ? { from: prev.cruiseAltFt, to: next.cruiseAltFt, deltaFt: next.cruiseAltFt - prev.cruiseAltFt } : null;
  // direct-to: a present-position route that CHANGED what lies ahead. One that
  // only dropped flown fixes, or restated the route, cleared nothing.
  const directTo = presentPosition && changed ? { fix: nextNames[0] ?? null } : null;

  if (changed && maxLateralNm != null && maxLateralNm >= REROUTE_MIN_LATERAL_NM) kinds.push("REROUTE");
  if (fixesAdded.length) kinds.push("FIXES_ADDED");
  if (fixesRemoved.length) kinds.push("FIXES_REMOVED");
  if (changed && !fixesAdded.length && interiorRemoved.length && !destination) kinds.push("SHORTCUT");
  if (directTo) kinds.push("DIRECT_TO");
  if (destination) kinds.push("DESTINATION_CHANGE");
  if (cruiseAltFt) kinds.push("ALTITUDE_CHANGE");
  if (routeTextChanged && !changed && !eitherPresent) kinds.push("ROUTE_TEXT_ONLY");

  return {
    kinds, fixesAdded, fixesRemoved, fixesRemovedFromStart, destination, cruiseAltFt, directTo, routeTextChanged,
    fromPresentPosition: head,
    restatement: routeTextChanged && eitherPresent && !changed && !destination,
    geometry: {
      changed,
      firstDifferingLeg: fixesDiffer && vertexListsDiffer ? { index: i, from: ref(i > 0 ? b[i - 1] : undefined), to: ref(b[i]), was: ref(a[i]) } : null,
      maxLateralNm,
      lateralEstimated: changed && lateralEstimated,
      note: notes.length ? notes.join("; ") : null,
    },
  };
}

// ── 2. where the aircraft was at a given time ───────────────────────────────

export interface TrackFix { tMs: number; lat: number; lon: number; altFt?: number | null }

/** a fix this close to the asked time IS the position at that time */
export const POSITION_EXACT_MS = 1_000;
/** the nearest fix is used as-is up to this far from the asked time. At a
 *  450 kt cruise 30 s is 3.75 nm, which is the position error this allows. */
export const POSITION_NEAREST_MAX_MS = 30_000;
/** between two fixes no further apart than this, the position is interpolated
 *  along the great circle between them and FLAGGED; a turn inside the gap is
 *  not seen. Beyond it nothing is returned. */
export const POSITION_INTERP_MAX_GAP_MS = 5 * 60_000;

export type PositionMethod = "EXACT" | "NEAREST" | "INTERPOLATED";

export interface PositionAtTime {
  lat: number; lon: number; altFt: number | null;
  method: PositionMethod;
  /** true only for INTERPOLATED: the point lies between two real fixes and is
   *  not itself a fix */
  interpolated: boolean;
  /** the real fix used (EXACT / NEAREST), or the nearer of the two bracketing
   *  fixes (INTERPOLATED) */
  fixTimeMs: number;
  /** |asked time - fixTimeMs| in seconds */
  gapSec: number;
  /** INTERPOLATED only: the two real fixes the point lies between */
  between: { beforeMs: number; afterMs: number; spanSec: number } | null;
}

/** Position at `atMs` from fixes sorted by time. Returns the position, or null
 *  with the reason in plain words. */
export function positionAtTime(fixes: TrackFix[], atMs: number): { position: PositionAtTime | null; reason: string | null } {
  if (!fixes.length) return { position: null, reason: "no ADS-B fixes are held for this aircraft" };
  // last fix at or before atMs
  let lo = 0, hi = fixes.length - 1, before = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (fixes[mid].tMs <= atMs) { before = mid; lo = mid + 1; } else hi = mid - 1;
  }
  const b = before >= 0 ? fixes[before] : null;
  const a = before + 1 < fixes.length ? fixes[before + 1] : null;
  const nearest = !b ? a! : !a ? b : (atMs - b.tMs <= a.tMs - atMs ? b : a);
  const gapMs = Math.abs(atMs - nearest.tMs);
  const gapSec = Math.round(gapMs / 1000);
  if (gapMs <= POSITION_NEAREST_MAX_MS) {
    return {
      position: {
        lat: nearest.lat, lon: nearest.lon, altFt: nearest.altFt ?? null,
        method: gapMs <= POSITION_EXACT_MS ? "EXACT" : "NEAREST", interpolated: false,
        fixTimeMs: nearest.tMs, gapSec, between: null,
      },
      reason: null,
    };
  }
  if (b && a && a.tMs - b.tMs <= POSITION_INTERP_MAX_GAP_MS) {
    const f = (atMs - b.tMs) / (a.tMs - b.tMs);
    const g = interpolateGC(b, a, f);
    const p = { lat: Math.round(g.lat * 1e5) / 1e5, lon: Math.round(g.lon * 1e5) / 1e5 };
    const alt = b.altFt != null && a.altFt != null ? Math.round(b.altFt + (a.altFt - b.altFt) * f) : null;
    return {
      position: {
        lat: p.lat, lon: p.lon, altFt: alt, method: "INTERPOLATED", interpolated: true,
        fixTimeMs: nearest.tMs, gapSec,
        between: { beforeMs: b.tMs, afterMs: a.tMs, spanSec: Math.round((a.tMs - b.tMs) / 1000) },
      },
      reason: null,
    };
  }
  const tol = POSITION_NEAREST_MAX_MS / 1000, lim = POSITION_INTERP_MAX_GAP_MS / 1000;
  const why = !b ? "the track starts after that time"
    : !a ? "the track ends before that time"
    : `the fixes either side are ${Math.round((a.tMs - b.tMs) / 1000)} s apart (interpolation limit ${lim} s)`;
  return { position: null, reason: `the nearest ADS-B fix is ${gapSec} s from that time (tolerance ${tol} s) and ${why}` };
}

// ── 3. off-route episodes ───────────────────────────────────────────────────

/** One fix with the deviation state machine's verdict after it was observed. */
export interface JudgedFix {
  tMs: number; lat: number; lon: number;
  /** the plan version in force at this fix (the one it was judged against) */
  version: number;
  state: "ON_PLAN" | "OFF_PLAN" | "UNKNOWN";
  /** consecutive en-route fixes beyond the off threshold so far */
  offCount: number;
  /** cross-track distance to that version's route, nm */
  crossTrackNm: number | null;
  /** distance to that version's destination end, nm */
  distToDestNm: number | null;
}

export type OffRouteEnd = "REJOINED" | "AMENDED" | "ENTERED_TERMINAL_AREA" | "TRACK_ENDED" | "ONGOING";
export type OffRouteClass = "AMENDMENT_FOLLOWED" | "NO_AMENDMENT_RECEIVED" | "TERMINAL_VECTORING";

/** an amendment this long after an episode's start, or after its end, is
 *  linked to it (the clearance comes first, the data entry after) */
export const AMENDMENT_LINK_MS = 10 * 60_000;
/** an open episode whose last fix is older than this is TRACK_ENDED, not ONGOING */
export const EPISODE_ONGOING_MAX_AGE_MS = 5 * 60_000;

export interface OffRouteEpisode {
  /** the first of the consecutive off-route fixes */
  startMs: number;
  start: LatLon;
  /** when the state machine called it (the fix that met the threshold) */
  detectedAtMs: number;
  /** null while ONGOING / TRACK_ENDED */
  endMs: number | null;
  end: LatLon | null;
  endReason: OffRouteEnd;
  /** the last fix seen off the route */
  lastOffMs: number;
  lastOff: LatLon;
  maxCrossTrackNm: number | null;
  /** off-route fixes in the episode, from its start */
  fixes: number;
  againstVersion: number;
  classification: OffRouteClass;
  /** always stated; never a guess */
  cause: string;
  linkedAmendmentVersion: number | null;
  /** amendment time minus startMs, seconds */
  amendmentLagSec: number | null;
}

const CAUSE_UNKNOWN = "cause not determined (no amendment was received, and nothing here knows about weather, traffic or controller instructions)";
const CAUSE_VECTORS = "inside the destination's terminal area and off the route: radar vectors to the approach are normal there (inferred from position only)";

/**
 * Group per-fix verdicts into episodes. `fixes` in time order, each already
 * judged against the version in force at its time. `amendments` are the
 * versions that replaced an earlier one, with the time each took effect.
 */
export function buildOffRouteEpisodes(
  fixes: JudgedFix[],
  amendments: Array<{ version: number; atMs: number }>,
  nowMs: number,
): OffRouteEpisode[] {
  const out: OffRouteEpisode[] = [];
  let open: OffRouteEpisode | null = null;
  let vect: OffRouteEpisode | null = null;
  let pending: JudgedFix | null = null;
  let pendingMax: number | null = null; // largest distance in the run of off-route fixes so far
  let prev: JudgedFix | null = null;
  const pt = (f: JudgedFix): LatLon => ({ lat: f.lat, lon: f.lon });
  const close = (e: OffRouteEpisode, reason: OffRouteEnd, at: JudgedFix | null) => {
    e.endReason = reason;
    if (at) { e.endMs = at.tMs; e.end = pt(at); }
    out.push(e);
  };
  const blank = (f: JudgedFix, startAt: JudgedFix, cls: OffRouteClass): OffRouteEpisode => ({
    startMs: startAt.tMs, start: pt(startAt), detectedAtMs: f.tMs, endMs: null, end: null, endReason: "ONGOING",
    lastOffMs: f.tMs, lastOff: pt(f), maxCrossTrackNm: f.crossTrackNm,
    fixes: cls === "TERMINAL_VECTORING" ? 1 : Math.max(1, f.offCount), againstVersion: f.version,
    classification: cls, cause: cls === "TERMINAL_VECTORING" ? CAUSE_VECTORS : CAUSE_UNKNOWN,
    linkedAmendmentVersion: null, amendmentLagSec: null,
  });
  const grow = (e: OffRouteEpisode, f: JudgedFix) => {
    e.lastOffMs = f.tMs; e.lastOff = pt(f); e.fixes++;
    if (f.crossTrackNm != null && (e.maxCrossTrackNm == null || f.crossTrackNm > e.maxCrossTrackNm)) e.maxCrossTrackNm = f.crossTrackNm;
  };

  for (const f of fixes) {
    const versionChanged = !!prev && prev.version !== f.version;
    if (versionChanged) {
      // a new version took effect: the old route is no longer the one in force
      if (open) {
        // ends at the last off-route fix before the amendment (the amendment's own time is on the amendment)
        open.endMs = open.lastOffMs; open.end = open.lastOff;
        open.linkedAmendmentVersion = f.version;
        close(open, "AMENDED", null); open = null;
      }
      if (vect) { vect.endMs = vect.lastOffMs; vect.end = vect.lastOff; close(vect, "AMENDED", null); vect = null; }
      pending = null; pendingMax = null;
    }
    const inTerminal = f.distToDestNm != null && f.distToDestNm < TERMINAL_AREA_NM;
    // the first of a run of off-route fixes (the state machine needs several before it calls it)
    if (f.offCount === 0) { pending = null; pendingMax = null; }
    else if (f.offCount === 1 && (versionChanged || !prev || prev.offCount === 0)) { pending = f; pendingMax = f.crossTrackNm; }
    else if (pending && f.crossTrackNm != null && (pendingMax == null || f.crossTrackNm > pendingMax)) pendingMax = f.crossTrackNm;

    if (open) {
      if (f.state !== "OFF_PLAN") { close(open, "REJOINED", f); open = null; }
      else if (inTerminal) { close(open, "ENTERED_TERMINAL_AREA", f); open = null; } // not judged in there
      else grow(open, f);
    } else if (f.state === "OFF_PLAN" && !inTerminal && (versionChanged || !prev || prev.state !== "OFF_PLAN")) {
      open = blank(f, pending ?? f, "NO_AMENDMENT_RECEIVED");
      if (pendingMax != null && (open.maxCrossTrackNm == null || pendingMax > open.maxCrossTrackNm)) open.maxCrossTrackNm = pendingMax;
      pending = null; pendingMax = null;
    }

    // terminal vectoring: each fix inside the destination's terminal area and off the route
    if (isTerminalVectoring(f.distToDestNm, f.crossTrackNm)) {
      if (vect) grow(vect, f); else vect = blank(f, f, "TERMINAL_VECTORING");
    } else if (vect) { close(vect, "REJOINED", f); vect = null; }
    prev = f;
  }
  const last = fixes[fixes.length - 1];
  const tail: OffRouteEnd = last && nowMs - last.tMs <= EPISODE_ONGOING_MAX_AGE_MS ? "ONGOING" : "TRACK_ENDED";
  if (open) close(open, tail, null);
  if (vect) close(vect, tail, null);

  // link amendments: one that took effect from the episode's start up to
  // AMENDMENT_LINK_MS after its end made the departure a reroute
  const sorted = amendments.slice().sort((x, y) => x.atMs - y.atMs);
  for (const e of out) {
    if (e.classification === "TERMINAL_VECTORING") continue;
    const until = (e.endMs ?? e.lastOffMs) + AMENDMENT_LINK_MS;
    const hit = e.linkedAmendmentVersion != null
      ? sorted.find((x) => x.version === e.linkedAmendmentVersion)
      : sorted.find((x) => x.atMs >= e.startMs && x.atMs <= until);
    if (!hit) continue;
    e.linkedAmendmentVersion = hit.version;
    e.amendmentLagSec = Math.round((hit.atMs - e.startMs) / 1000);
    if (hit.atMs - e.startMs <= AMENDMENT_LINK_MS || (e.endMs != null && hit.atMs - e.endMs <= AMENDMENT_LINK_MS)) {
      e.classification = "AMENDMENT_FOLLOWED";
      e.cause = `reroute: an amendment to the filed route took effect ${e.amendmentLagSec} s after the aircraft left it`;
    } else {
      e.cause = `${CAUSE_UNKNOWN}; an amendment did arrive, ${e.amendmentLagSec} s after the aircraft left the route, too late to explain the start`;
    }
  }
  return out.sort((x, y) => x.startMs - y.startMs);
}
