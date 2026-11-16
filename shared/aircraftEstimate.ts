// shared/aircraftEstimate.ts — where an aircraft PROBABLY is when its last
// real position is getting old, and the path it is estimated to be on.
// (2026-11-16, owner: "keep the model moving by dead reckoning between
// updates ... where there is no data, show an estimated path along the filed
// route, drawn dashed so it's clearly estimated".)
//
// EVERYTHING this module returns is an ESTIMATE and says so (`estimated:
// true`, the method, the fix it started from and how old that fix is, the
// assumptions). It never returns a position as if it were observed, and it
// returns NO position when the inputs do not support one.
//
// Two methods, in this order:
//   dead_reckoning — the last broadcast track and ground speed, flown
//     straight from the last real fix. Aircraft turn, so it is only used for
//     DEAD_RECKONING_MAX_SEC, or for DEAD_RECKONING_SHORT_SEC when a route is
//     available to take over.
//   along_route — the fix is projected onto the route the aircraft was on
//     (within ROUTE_JOIN_MAX_NM and pointing along it) and advanced along the
//     route at the last ground speed. It stops at the route's end. The route
//     is whatever the caller supplies: a filed plan, or a predicted one; its
//     source is carried through so a predicted route is never shown as filed.
//
// Altitude is HELD at the last reported value. The feed's vertical rate is
// not an input here, so a climbing or descending aircraft's estimated
// altitude is its last real one (`altBasis: "held"`).
//
// `withinNm` is not a statistical error bar. It is the distance the aircraft
// can have covered since the fix at up to SPEED_MARGIN times its last ground
// speed: the aircraft is inside that circle around its LAST FIX unless it
// sped up by more than the margin.
//
// Pure: no I/O, no clock. Shared so the server endpoint and any client that
// wants per-frame motion between answers run the same arithmetic.

import {
  angleDiffDeg, crossTrackNm, cumulativeNm, destinationPoint, haversineNm, initialBearingDeg, interpolateGC,
  type LatLon,
} from "./flightPlanGeometry";

/** straight-line dead reckoning is used this long when no route can take over */
export const DEAD_RECKONING_MAX_SEC = 120;
/** ... and only this long when a route can */
export const DEAD_RECKONING_SHORT_SEC = 30;
/** a fix farther than this from the route is not "on" it (flightPlans DEVIATION_OFF_NM) */
export const ROUTE_JOIN_MAX_NM = 8;
/** ... nor is one whose track differs from the route's direction by more than this */
export const ROUTE_JOIN_MAX_TURN_DEG = 60;
/** no estimate of any kind from a fix older than this */
export const ESTIMATE_MAX_AGE_SEC = 6 * 3600;
/** below this ground speed (taxi, hover, junk) nothing is extrapolated */
export const ESTIMATE_MIN_GS_KT = 40;
/** `withinNm` allows the ground speed to have risen by this factor */
export const SPEED_MARGIN = 1.15;
/** the estimated path runs this far past the estimated position */
export const PATH_AHEAD_SEC = 600;
export const PATH_MAX_POINTS = 240;
const PATH_STEP_SEC = 30;

export type EstimateMethod = "dead_reckoning" | "along_route" | "none";

export interface EstimateFix {
  lat: number; lon: number;
  altFt: number | null;
  gsKt: number | null;
  trkDeg: number | null;
  onGround: boolean;
  /** the fix's OWN time, epoch ms */
  fixMs: number;
  /** true when gsKt/trkDeg were worked out from two archived positions, not broadcast */
  motionDerived?: boolean;
}

export interface EstimateRoute {
  /** the route in flight order */
  points: LatLon[];
  /** e.g. FILED_FAA | HISTORY_PREDICTED | ROUTE_DB_PREDICTED (carried through unchanged) */
  source: string;
}

export interface AircraftEstimate {
  /** always true: nothing here is an observed position */
  estimated: true;
  method: EstimateMethod;
  /** why this method was used, or why there is no estimate */
  reason: string;
  /** the time the estimate is for, epoch ms */
  at: number;
  basisFixMs: number;
  ageSec: number;
  lat: number | null;
  lon: number | null;
  /** the last reported altitude, held (null when the fix had none) */
  altFt: number | null;
  altBasis: "held" | "unknown";
  /** track over ground at the estimated position, degrees (the route's direction when along_route) */
  trkDeg: number | null;
  /** ground speed assumed, knots */
  gsKt: number | null;
  /** distance covered since the fix under the method's assumption, nm */
  distanceNm: number | null;
  /** see the header: a circle around the LAST FIX, not an error bar around the estimate */
  withinNm: number | null;
  /** along_route only */
  route: null | {
    source: string;
    /** how far the fix was from the route, nm */
    crossTrackAtFixNm: number;
    /** route distance left after the estimated position, nm */
    remainingNm: number;
    /** the estimate ran to the route's last point and stopped there */
    reachedEnd: boolean;
  };
  /** the estimated path for a dashed line: [lat, lon, epoch ms], starting AT the real fix */
  path: Array<[number, number, number]>;
  assumptions: string[];
}

const r5 = (x: number): number => Math.round(x * 1e5) / 1e5;
const r1 = (x: number): number => Math.round(x * 10) / 10;
const finite = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);

function none(fix: EstimateFix, nowMs: number, reason: string, withinNm: number | null = null): AircraftEstimate {
  return {
    estimated: true, method: "none", reason, at: nowMs, basisFixMs: fix.fixMs,
    ageSec: r1(Math.max(0, (nowMs - fix.fixMs) / 1000)),
    lat: null, lon: null, altFt: finite(fix.altFt) ? fix.altFt : null, altBasis: finite(fix.altFt) ? "held" : "unknown",
    trkDeg: null, gsKt: finite(fix.gsKt) ? fix.gsKt : null, distanceNm: null, withinNm,
    route: null, path: [], assumptions: [],
  };
}

/** the point `nm` along a polyline from its first vertex, and the direction of travel there */
export function pointAlong(points: LatLon[], cum: number[], nm: number): { pos: LatLon; brgDeg: number | null; atEnd: boolean } {
  const total = cum[cum.length - 1];
  if (nm >= total) {
    const last = points[points.length - 1];
    const prev = points.length > 1 ? points[points.length - 2] : null;
    return { pos: last, brgDeg: prev && haversineNm(prev, last) > 1e-6 ? initialBearingDeg(prev, last) : null, atEnd: true };
  }
  let i = 0;
  while (i < cum.length - 2 && cum[i + 1] <= nm) i++;
  const seg = cum[i + 1] - cum[i];
  const f = seg > 1e-9 ? (nm - cum[i]) / seg : 0;
  const pos = interpolateGC(points[i], points[i + 1], Math.max(0, Math.min(1, f)));
  return { pos, brgDeg: seg > 1e-9 ? initialBearingDeg(pos, points[i + 1]) : null, atEnd: false };
}

/**
 * The estimate for `nowMs` from the last real fix. `route` is optional; it is
 * used only when the fix was on it and heading along it.
 */
export function estimateAircraft(fix: EstimateFix, nowMs: number, route: EstimateRoute | null = null): AircraftEstimate {
  const ageSec = (nowMs - fix.fixMs) / 1000;
  if (!finite(fix.lat) || !finite(fix.lon) || !finite(fix.fixMs)) return none(fix, nowMs, "the last fix has no usable position or time");
  if (!(ageSec >= 0)) return none(fix, nowMs, "the last fix is newer than the requested time");
  if (fix.onGround) return none(fix, nowMs, "on the ground at the last fix: nothing is extrapolated");
  if (!finite(fix.gsKt) || !finite(fix.trkDeg)) return none(fix, nowMs, "the last fix carried no ground speed or track");
  if (fix.gsKt < ESTIMATE_MIN_GS_KT) return none(fix, nowMs, `ground speed under ${ESTIMATE_MIN_GS_KT} kt at the last fix: nothing is extrapolated`);
  const gs = fix.gsKt;
  const here: LatLon = { lat: fix.lat, lon: fix.lon };
  const distNm = (gs * ageSec) / 3600;
  const withinNm = r1(distNm * SPEED_MARGIN);
  if (ageSec > ESTIMATE_MAX_AGE_SEC) {
    return none(fix, nowMs, `the last fix is more than ${ESTIMATE_MAX_AGE_SEC / 3600} hours old: no estimate`, withinNm);
  }
  const base = {
    estimated: true as const, at: nowMs, basisFixMs: fix.fixMs, ageSec: r1(ageSec),
    altFt: finite(fix.altFt) ? fix.altFt : null, altBasis: (finite(fix.altFt) ? "held" : "unknown") as "held" | "unknown",
    gsKt: gs, distanceNm: r1(distNm), withinNm,
  };
  const motion = fix.motionDerived
    ? "ground speed and track worked out from the last two archived positions (not broadcast values)"
    : "ground speed and track as last broadcast";
  const altNote = base.altFt == null ? "no altitude was reported at the last fix" : "altitude held at the last reported value";

  // is the aircraft on the supplied route, pointing along it?
  let onRoute: null | { cum: number[]; alongNm: number; xtNm: number } = null;
  let offRouteWhy = route ? "" : "no route available";
  if (route && route.points.length >= 2) {
    const xt = crossTrackNm(here, route.points);
    const cum = cumulativeNm(route.points);
    if (!xt) offRouteWhy = "no route available";
    else if (xt.nm > ROUTE_JOIN_MAX_NM) offRouteWhy = `the last fix was ${r1(xt.nm)} nm from the route (more than ${ROUTE_JOIN_MAX_NM} nm)`;
    else {
      const dir = pointAlong(route.points, cum, xt.alongNm).brgDeg;
      if (dir != null && angleDiffDeg(dir, fix.trkDeg) > ROUTE_JOIN_MAX_TURN_DEG) {
        offRouteWhy = `the last track (${Math.round(fix.trkDeg)}°) was not along the route (${Math.round(dir)}°)`;
      } else onRoute = { cum, alongNm: xt.alongNm, xtNm: xt.nm };
    }
  } else if (route) offRouteWhy = "the route has fewer than two points";

  const drLimit = onRoute ? DEAD_RECKONING_SHORT_SEC : DEAD_RECKONING_MAX_SEC;
  if (ageSec <= drLimit) {
    const pos = destinationPoint(here, fix.trkDeg, distNm);
    const path: Array<[number, number, number]> = [[r5(here.lat), r5(here.lon), fix.fixMs]];
    // straight ahead only as far as dead reckoning itself is allowed
    for (let s = PATH_STEP_SEC; s <= drLimit; s += PATH_STEP_SEC) {
      const p = destinationPoint(here, fix.trkDeg, (gs * s) / 3600);
      path.push([r5(p.lat), r5(p.lon), fix.fixMs + s * 1000]);
    }
    return {
      ...base, method: "dead_reckoning",
      reason: onRoute
        ? `fix ${Math.round(ageSec)} s old: straight ahead for the first ${DEAD_RECKONING_SHORT_SEC} s, then along the route`
        : `fix ${Math.round(ageSec)} s old: straight ahead (${offRouteWhy})`,
      lat: r5(pos.lat), lon: r5(pos.lon), trkDeg: fix.trkDeg, route: null, path,
      assumptions: [motion, "no turn since the last fix", altNote],
    };
  }
  if (!onRoute || !route) {
    return none(fix, nowMs,
      `the last fix is ${Math.round(ageSec)} s old, past the ${DEAD_RECKONING_MAX_SEC} s a straight line is used for, and ${offRouteWhy}`,
      withinNm);
  }
  const total = onRoute.cum[onRoute.cum.length - 1];
  const target = onRoute.alongNm + distNm;
  const at = pointAlong(route.points, onRoute.cum, target);
  const path: Array<[number, number, number]> = [[r5(here.lat), r5(here.lon), fix.fixMs]];
  const endSec = Math.min(ageSec + PATH_AHEAD_SEC, ((total - onRoute.alongNm) / gs) * 3600);
  const startProj = pointAlong(route.points, onRoute.cum, onRoute.alongNm).pos;
  path.push([r5(startProj.lat), r5(startProj.lon), fix.fixMs]);
  // every route vertex between the fix and the path's end, plus even time steps between them
  const stops: number[] = [];
  for (let i = 0; i < onRoute.cum.length; i++) {
    const sec = ((onRoute.cum[i] - onRoute.alongNm) / gs) * 3600;
    if (sec > 0 && sec < endSec) stops.push(sec);
  }
  const step = Math.max(PATH_STEP_SEC, endSec / (PATH_MAX_POINTS / 2));
  for (let s = step; s < endSec; s += step) stops.push(s);
  stops.push(endSec);
  stops.sort((a, b) => a - b);
  let lastSec = -1;
  for (const s of stops) {
    if (s - lastSec < 1e-6 || path.length >= PATH_MAX_POINTS) continue;
    const p = pointAlong(route.points, onRoute.cum, onRoute.alongNm + (gs * s) / 3600).pos;
    path.push([r5(p.lat), r5(p.lon), Math.round(fix.fixMs + s * 1000)]);
    lastSec = s;
  }
  return {
    ...base, method: "along_route",
    reason: at.atEnd
      ? `fix ${Math.round(ageSec)} s old: advanced along the ${route.source} route to its last point and stopped there`
      : `fix ${Math.round(ageSec)} s old: advanced along the ${route.source} route at the last ground speed`,
    lat: r5(at.pos.lat), lon: r5(at.pos.lon), trkDeg: at.brgDeg == null ? null : r1(at.brgDeg),
    distanceNm: r1(Math.min(distNm, total - onRoute.alongNm)),
    route: {
      source: route.source, crossTrackAtFixNm: r1(onRoute.xtNm),
      remainingNm: r1(Math.max(0, total - target)), reachedEnd: at.atEnd,
    },
    path,
    assumptions: [
      motion, "ground speed unchanged since the last fix",
      `the aircraft stayed on the ${route.source} route (it was ${r1(onRoute.xtNm)} nm from it at the last fix)`,
      altNote,
    ],
  };
}

/** Ground speed (kt) and track (deg) between two positions with times, or null when they cannot give one
 *  (same time, same place, or more than `maxGapSec` apart). For a basis fix read from the archive. */
export function motionBetween(
  a: { lat: number; lon: number; tMs: number }, b: { lat: number; lon: number; tMs: number }, maxGapSec = 300,
): { gsKt: number; trkDeg: number } | null {
  const dt = (b.tMs - a.tMs) / 1000;
  if (!(dt > 0) || dt > maxGapSec) return null;
  const nm = haversineNm(a, b);
  if (!(nm > 0.01)) return null;
  return { gsKt: r1((nm / dt) * 3600), trkDeg: r1(initialBearingDeg(a, b)) };
}
