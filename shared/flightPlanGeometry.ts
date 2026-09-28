// FLIGHT PROGRAM (2026-09-28) — shared, PURE flight-plan geometry.
//
// Owned by the server flight-plan builder (server/flightPlans.ts); the client
// planned-route curtain imports the same functions so both sides split, measure
// and re-plan with one definition. No dependencies, no I/O, no Date.now().
//
// All spherical math runs on unit 3-vectors (not lat/lon deltas), so it is
// antimeridian-safe by construction: a segment from 179°E to 179°W is 2° long,
// never 358°. Output longitudes are normalized to [-180, 180); renderers that
// need a continuous line across the antimeridian call unwrapLons() on the way
// to the GPU (MapLibre's globe wants lon deltas < 180 between neighbors).
//
// HONESTY: nothing here invents data. Estimated altitudes are always returned
// with altEstimated=true; a re-planned path is a prediction and its inserted
// points carry altM=null (estimated later) until a real altitude exists.

export const EARTH_RADIUS_NM = 3440.065;
export const FT_PER_M = 3.28084;
const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

export interface LatLon { lat: number; lon: number }

export interface PlanPoint extends LatLon {
  /** meters MSL; null = unknown (estimateVerticalProfile fills it) */
  altM: number | null;
  /** true for any altitude that is not FILED (plan) or OBSERVED (live fix) */
  altEstimated: boolean;
  name?: string;
}

// ── vector helpers ──────────────────────────────────────────────────────────
type V3 = [number, number, number];

function toV(p: LatLon): V3 {
  const la = p.lat * D2R, lo = p.lon * D2R;
  const c = Math.cos(la);
  return [c * Math.cos(lo), c * Math.sin(lo), Math.sin(la)];
}
function toLL(v: V3): LatLon {
  return { lat: Math.atan2(v[2], Math.hypot(v[0], v[1])) * R2D, lon: normLon(Math.atan2(v[1], v[0]) * R2D) };
}
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
function unit(a: V3): V3 | null {
  const l = len(a);
  return l < 1e-15 ? null : scale(a, 1 / l);
}
/** central angle (radians) between two unit vectors — atan2 form stays
 *  accurate for both tiny and near-antipodal separations */
const angle = (a: V3, b: V3) => Math.atan2(len(cross(a, b)), dot(a, b));

// ── scalar helpers ──────────────────────────────────────────────────────────
/** normalize a longitude to [-180, 180) */
export function normLon(lon: number): number {
  if (!Number.isFinite(lon)) return lon;
  if (lon >= -180 && lon < 180) return lon; // exact passthrough: no float drift on in-range input
  return (((lon + 180) % 360) + 360) % 360 - 180;
}

/** absolute difference between two bearings, in [0, 180] */
export function angleDiffDeg(a: number, b: number): number {
  const d = Math.abs((((a - b) % 360) + 360) % 360);
  return d > 180 ? 360 - d : d;
}

export function haversineNm(a: LatLon, b: LatLon): number {
  const dLa = (b.lat - a.lat) * D2R;
  const dLo = (b.lon - a.lon) * D2R; // sin²(dLo/2) is 360°-periodic: antimeridian-safe
  const s = Math.sin(dLa / 2) ** 2 + Math.cos(a.lat * D2R) * Math.cos(b.lat * D2R) * Math.sin(dLo / 2) ** 2;
  return 2 * EARTH_RADIUS_NM * Math.asin(Math.min(1, Math.sqrt(s)));
}

/** initial true bearing a -> b, degrees [0, 360) */
export function initialBearingDeg(a: LatLon, b: LatLon): number {
  const la1 = a.lat * D2R, la2 = b.lat * D2R, dLo = (b.lon - a.lon) * D2R;
  const y = Math.sin(dLo) * Math.cos(la2);
  const x = Math.cos(la1) * Math.sin(la2) - Math.sin(la1) * Math.cos(la2) * Math.cos(dLo);
  return ((Math.atan2(y, x) * R2D) + 360) % 360;
}

/** point at fraction f (0..1) along the great circle a -> b */
export function interpolateGC(a: LatLon, b: LatLon, f: number): LatLon {
  const va = toV(a), vb = toV(b);
  const d = angle(va, vb);
  if (d < 1e-12) return { lat: a.lat, lon: normLon(a.lon) };
  const s = Math.sin(d);
  if (s < 1e-12) {
    // antipodal: the great circle is undefined — linear fallback keeps the
    // function total (no flight plan is antipodal in one segment)
    return { lat: a.lat + (b.lat - a.lat) * f, lon: normLon(a.lon + (b.lon - a.lon) * f) };
  }
  const k1 = Math.sin((1 - f) * d) / s, k2 = Math.sin(f * d) / s;
  return toLL(add(scale(va, k1), scale(vb, k2)));
}

/** Great-circle polyline a -> b, both endpoints included, spaced <= stepNm.
 *  A zero-length pair returns the single point [a]. */
export function greatCircle(a: LatLon, b: LatLon, stepNm = 50): LatLon[] {
  const dist = haversineNm(a, b);
  const first = { lat: a.lat, lon: normLon(a.lon) };
  if (!(dist > 1e-6)) return [first];
  const n = Math.max(1, Math.ceil(dist / Math.max(0.1, stepNm)));
  const out: LatLon[] = [first];
  for (let i = 1; i < n; i++) out.push(interpolateGC(a, b, i / n));
  out.push({ lat: b.lat, lon: normLon(b.lon) });
  return out;
}

/** cumulative along-path distance (nm) at every vertex; [0] = 0 */
export function cumulativeNm(points: LatLon[]): number[] {
  const out: number[] = new Array(points.length);
  let acc = 0;
  for (let i = 0; i < points.length; i++) {
    if (i > 0) acc += haversineNm(points[i - 1], points[i]);
    out[i] = acc;
  }
  return out;
}

export function polylineLengthNm(points: LatLon[]): number {
  return points.length < 2 ? 0 : cumulativeNm(points)[points.length - 1];
}

/** Densify a plan so no segment exceeds stepNm. Inserted points are
 *  great-circle interpolations with altM=null (estimateVerticalProfile fills
 *  them, flagged estimated). Altitude is deliberately NOT interpolated from
 *  the neighbors: between an origin and a destination those are FIELD
 *  elevations, and interpolating them would put the whole route at ground
 *  level (caught by the live smoke: SFO-SIN at 5 m mid-Pacific). */
export function densifyPlan(points: PlanPoint[], stepNm = 50): PlanPoint[] {
  if (points.length < 2) return points.map((p) => ({ ...p, lon: normLon(p.lon) }));
  const out: PlanPoint[] = [{ ...points[0], lon: normLon(points[0].lon) }];
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    const gc = greatCircle(a, b, stepNm);
    for (let k = 1; k < gc.length - 1; k++) {
      out.push({ lat: gc[k].lat, lon: gc[k].lon, altM: null, altEstimated: true });
    }
    if (gc.length > 1 || haversineNm(a, b) > 1e-6) out.push({ ...b, lon: normLon(b.lon) });
  }
  return out;
}

/** Make longitudes continuous (neighbor deltas < 180°) for rendering a line
 *  across the antimeridian. Returns copies; lats untouched. */
export function unwrapLons<T extends LatLon>(points: T[]): T[] {
  const out: T[] = [];
  let prev: number | null = null;
  for (const p of points) {
    let lon = p.lon;
    if (prev != null) {
      while (lon - prev > 180) lon -= 360;
      while (lon - prev < -180) lon += 360;
    }
    out.push({ ...p, lon });
    prev = lon;
  }
  return out;
}

// ── cross-track ─────────────────────────────────────────────────────────────
export interface CrossTrackResult {
  /** unsigned distance (nm) from the point to the nearest spot on the polyline */
  nm: number;
  /** signed cross-track (nm): + = right of the direction of travel, - = left
   *  (0-sign when the nearest spot is a vertex beyond a segment's ends) */
  signedNm: number;
  /** index of the segment [segIndex, segIndex+1] holding the nearest spot
   *  (0 for a single-point polyline) */
  segIndex: number;
  /** fraction 0..1 along that segment */
  segFraction: number;
  /** distance along the whole polyline from its first vertex to the nearest spot */
  alongNm: number;
  /** the nearest spot itself (the point projected onto the plan) */
  proj: LatLon;
}

/** Distance from `point` to a polyline of great-circle segments, plus where
 *  along the polyline the nearest spot lies. null only for an empty polyline. */
export function crossTrackNm(point: LatLon, polyline: LatLon[]): CrossTrackResult | null {
  if (!polyline.length) return null;
  const p = toV(point);
  if (polyline.length === 1) {
    return {
      nm: haversineNm(point, polyline[0]), signedNm: 0, segIndex: 0, segFraction: 0, alongNm: 0,
      proj: { lat: polyline[0].lat, lon: normLon(polyline[0].lon) },
    };
  }
  let best: CrossTrackResult | null = null;
  let cum = 0;
  for (let i = 0; i < polyline.length - 1; i++) {
    const A = toV(polyline[i]), B = toV(polyline[i + 1]);
    const segAng = angle(A, B);
    let distAng: number, alongAng: number, signed = 0, proj: V3;
    const n = segAng > 1e-12 ? unit(cross(A, B)) : null;
    if (!n) {
      // zero-length (or antipodal) segment: measure to its start vertex
      distAng = angle(p, A); alongAng = 0; proj = A;
    } else {
      const s = Math.max(-1, Math.min(1, dot(p, n)));
      const C = unit(sub(p, scale(n, s))) || A; // p's foot on the great circle
      const along = Math.atan2(dot(cross(A, C), n), dot(A, C)); // signed angle A -> C
      if (along < 0) { distAng = angle(p, A); alongAng = 0; proj = A; }
      else if (along > segAng) { distAng = angle(p, B); alongAng = segAng; proj = B; }
      else { distAng = Math.abs(Math.asin(s)); alongAng = along; proj = C; signed = -Math.asin(s); }
    }
    const nm = distAng * EARTH_RADIUS_NM;
    if (!best || nm < best.nm - 1e-9) {
      best = {
        nm, signedNm: signed * EARTH_RADIUS_NM, segIndex: i,
        segFraction: segAng > 1e-12 ? alongAng / segAng : 0,
        alongNm: cum + alongAng * EARTH_RADIUS_NM,
        proj: toLL(proj),
      };
    }
    cum += segAng * EARTH_RADIUS_NM;
  }
  return best;
}

// ── split at the aircraft ───────────────────────────────────────────────────
const SAME_POINT_NM = 0.01;

/** Split a plan at the aircraft's position projected onto it. `behind` ends
 *  and `ahead` starts EXACTLY at that projected point (altitude interpolated
 *  from its neighbors, flagged estimated when interpolated). */
export function splitPlanAt(points: PlanPoint[], pos: LatLon): {
  behind: PlanPoint[]; ahead: PlanPoint[]; crossTrackNm: number | null; alongNm: number | null;
} {
  if (!points.length) return { behind: [], ahead: [], crossTrackNm: null, alongNm: null };
  const xt = crossTrackNm(pos, points) as CrossTrackResult;
  if (points.length === 1) {
    return { behind: [{ ...points[0] }], ahead: [{ ...points[0] }], crossTrackNm: xt.nm, alongNm: 0 };
  }
  const i = xt.segIndex;
  const a = points[i], b = points[i + 1];
  const atA = haversineNm(xt.proj, a) <= SAME_POINT_NM;
  const atB = !atA && haversineNm(xt.proj, b) <= SAME_POINT_NM;
  let pp: PlanPoint;
  if (atA) pp = { ...a };
  else if (atB) pp = { ...b };
  else {
    const f = xt.segFraction;
    const both = a.altM != null && b.altM != null;
    pp = {
      lat: xt.proj.lat, lon: xt.proj.lon,
      altM: both ? Math.round((a.altM as number) + ((b.altM as number) - (a.altM as number)) * f) : null,
      altEstimated: a.altEstimated || b.altEstimated || !both || a.altM !== b.altM,
    };
  }
  const behind = [...points.slice(0, atA ? i : i + 1), pp];
  const ahead = [pp, ...points.slice(atB ? i + 2 : i + 1)];
  return { behind, ahead, crossTrackNm: xt.nm, alongNm: xt.alongNm };
}

// ── vertical profile (ESTIMATED) ────────────────────────────────────────────
/** typical-jet climb gradient estimate (ft per nm over ground) */
export const CLIMB_FT_PER_NM = 250;
/** 3° descent path: tan(3°) × 6076.12 ft/nm ≈ 318 ft/nm */
export const DESCENT_FT_PER_NM = Math.tan(3 * D2R) * 6076.12;

/** Stage-length cruise estimate when no cruise altitude was filed. A rough
 *  prior (short hops cruise low); always reported cruiseAltEstimated=true. */
export function typicalCruiseFt(distanceNm: number): number {
  if (!(distanceNm > 0)) return 10000;
  if (distanceNm < 100) return 12000;
  if (distanceNm < 250) return 24000;
  if (distanceNm < 500) return 32000;
  return 36000;
}

function pointAtAlong(points: PlanPoint[], cum: number[], x: number): { seg: number; ll: LatLon } {
  for (let i = 0; i < points.length - 1; i++) {
    if (x <= cum[i + 1] || i === points.length - 2) {
      const segLen = cum[i + 1] - cum[i];
      const f = segLen > 0 ? Math.min(1, Math.max(0, (x - cum[i]) / segLen)) : 0;
      return { seg: i, ll: interpolateGC(points[i], points[i + 1], f) };
    }
  }
  return { seg: 0, ll: { lat: points[0].lat, lon: points[0].lon } };
}

/** Fill every point whose altM is null with an ESTIMATED altitude from a
 *  climb / cruise / descent profile:
 *    alt(x) = min(cruise, origin + 250 ft/nm · x, dest + 318 ft/nm · (D − x))
 *  which also handles short flights that never reach cruise (the climb and
 *  descent lines meet below it). Top-of-climb / top-of-descent (or the peak)
 *  vertices are inserted so the curtain's shape is right. Points that already
 *  carry an altitude (FILED per-point, OBSERVED, or recorded) are kept as-is. */
export function estimateVerticalProfile(
  points: PlanPoint[],
  originElevM: number | null,
  destElevM: number | null,
  cruiseFt: number | null,
): PlanPoint[] {
  if (!points.length) return [];
  const o = (originElevM ?? 0) * FT_PER_M;
  const d = (destElevM ?? 0) * FT_PER_M;
  if (points.length === 1) {
    const p = points[0];
    return [p.altM != null ? { ...p } : { ...p, altM: Math.round(o / FT_PER_M), altEstimated: true }];
  }
  const cum = cumulativeNm(points);
  const D = cum[cum.length - 1];
  const cruise = cruiseFt ?? typicalCruiseFt(D);
  const altFtAt = (x: number) => Math.min(cruise, o + CLIMB_FT_PER_NM * x, d + DESCENT_FT_PER_NM * (D - x));

  // profile breakpoints to insert (only where the plan will be estimated)
  const marks: Array<{ x: number; name: string }> = [];
  const xToc = (cruise - o) / CLIMB_FT_PER_NM;
  const xTod = D - (cruise - d) / DESCENT_FT_PER_NM;
  if (xToc > 0 && xTod < D && xToc < xTod) {
    marks.push({ x: xToc, name: "TOC (est.)" }, { x: xTod, name: "TOD (est.)" });
  } else {
    const xPeak = (d - o + DESCENT_FT_PER_NM * D) / (CLIMB_FT_PER_NM + DESCENT_FT_PER_NM);
    if (xPeak > 0 && xPeak < D) marks.push({ x: xPeak, name: "peak (est.)" });
  }

  const withMarks: Array<{ p: PlanPoint; x: number }> = points.map((p, i) => ({ p, x: cum[i] }));
  for (const m of marks) {
    const { seg, ll } = pointAtAlong(points, cum, m.x);
    const a = points[seg], b = points[seg + 1];
    // never override a segment whose ends are both known (filed/observed)
    if (a.altM != null && b.altM != null) continue;
    if (withMarks.some((w) => Math.abs(w.x - m.x) < 0.5)) continue;
    withMarks.push({ p: { lat: ll.lat, lon: ll.lon, altM: null, altEstimated: true, name: m.name }, x: m.x });
  }
  withMarks.sort((u, v) => u.x - v.x);
  return withMarks.map(({ p, x }) =>
    p.altM != null ? { ...p } : { ...p, altM: Math.round(altFtAt(x) / FT_PER_M), altEstimated: true });
}

// ── re-plan from the aircraft's real position ───────────────────────────────
/** rejoin candidates must be at least this far ahead along the plan */
export const REPLAN_MIN_AHEAD_NM = 20;
/** ...and within this many degrees of the aircraft's current track */
export const REPLAN_MAX_TURN_DEG = 70;

export interface ReplanResult {
  points: PlanPoint[];
  /** REJOIN = back onto the plan at rejoinIndex; DIRECT = great circle to the
   *  destination; EMPTY = no plan to rejoin (only the present position) */
  mode: "REJOIN" | "DIRECT" | "EMPTY";
  /** index in the ORIGINAL plan where the path rejoins it */
  rejoinIndex: number | null;
}

/** Re-plan from where the aircraft really is: rejoin the plan at the first
 *  downstream vertex that is >= 20nm ahead along-track AND within ±70° of
 *  the current track (no trk -> the heading test is skipped), else go direct
 *  great-circle to the destination. The first point is the present position
 *  (altitude = the observed one when given). Inserted connector points carry
 *  altM=null so the caller's profile estimate fills them, flagged estimated. */
export function replanFromPosition(
  plan: PlanPoint[],
  pos: LatLon & { altM?: number | null },
  trkDeg: number | null | undefined,
  stepNm = 50,
): ReplanResult {
  const here: PlanPoint = {
    lat: pos.lat, lon: normLon(pos.lon),
    altM: pos.altM ?? null, altEstimated: pos.altM == null, name: "present position",
  };
  if (!plan.length) return { points: [here], mode: "EMPTY", rejoinIndex: null };
  const connect = (to: PlanPoint): PlanPoint[] =>
    greatCircle(here, to, stepNm).slice(1, -1).map((ll) => ({ lat: ll.lat, lon: ll.lon, altM: null, altEstimated: true }));
  const xt = crossTrackNm(here, plan) as CrossTrackResult;
  const cum = cumulativeNm(plan);
  const useTrk = trkDeg != null && Number.isFinite(trkDeg);
  for (let i = xt.segIndex + 1; i < plan.length; i++) {
    if (cum[i] - xt.alongNm < REPLAN_MIN_AHEAD_NM) continue;
    if (useTrk && angleDiffDeg(initialBearingDeg(here, plan[i]), trkDeg as number) > REPLAN_MAX_TURN_DEG) continue;
    return { points: [here, ...connect(plan[i]), ...plan.slice(i).map((p) => ({ ...p }))], mode: "REJOIN", rejoinIndex: i };
  }
  const dest = plan[plan.length - 1];
  if (haversineNm(here, dest) <= SAME_POINT_NM) return { points: [here], mode: "DIRECT", rejoinIndex: null };
  return { points: [here, ...connect(dest), { ...dest }], mode: "DIRECT", rejoinIndex: null };
}
