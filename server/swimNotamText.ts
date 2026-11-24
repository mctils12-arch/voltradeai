// swimNotamText.ts — a flight restriction's outline, read from the words of
// its notice ([PRODUCT] 2026-11-22).
//
// WHY: US Flight Data Center restriction notices reach us on the FAA notice
// feed with no airspace geometry and no index circle; the area is written in
// the text only. Read on production 2026-11-22 00:10Z: notice FDC 6/9306
// (a firefighting restriction 20 NM west of Lexington, Virginia) arrived
// that way and was dropped, while the FAA's own public list showed it first.
//
// WHAT IS READ: one published wording, the one the FAA's notice system
// writes, and nothing else:
//   WI AN AREA DEFINED AS <n>NM RADIUS OF <point>
//   WI AN AREA DEFINED AS <point> TO <point> ... TO [THE] POINT OF ORIGIN
//   ... <point> THEN [COUNTER]CLOCKWISE ON A <n>NM ARC CENTERED ON <point> TO <point> ...
// A point is DDMMSS[.s]N DDDMMSS[.s]W. Anything else in that position (a
// navaid radial, "1 NMR OF", a point missing its hemisphere letter, a list
// cut by "END PART 1 OF 4") leaves that area UNREAD: it is counted and no
// outline is made up for it. Other countries' wordings are not read.
//
// EXCLUDED AREAS (2026-11-24): the same system writes
//   ...; EXCLUDING AN AREA DEFINED AS <n>NM RADIUS OF <point> ...
// after an area, most often on a VIP restriction (a 30 NM ring with an
// airport's surroundings left out). Until this date those words were not
// read at all and the area was drawn whole: production's running check
// judged FDC 6/9335 partial on its first pass for exactly that (the FAA's
// outline is notched twice, ours was a full circle, 2.4 NM apart). An
// exclusion that crosses the edge of the area before it is now cut out of
// that outline, as the FAA draws it. One that cannot be cut (it lies wholly
// inside, where an outline has no hole to give it; its own shape could not
// be read; the two edges cross more than once) is COUNTED as `uncut` and
// the area stays whole, which encloses the true one.
//
// GROUND TRUTH (ladder gate 1, scripts/tfr_text_gate1.ts): the FAA
// publishes its own shape for each listed restriction (tfr.faa.gov). The
// check reads every listed notice's text with this parser and measures the
// distance between the two outlines; the result is in research/.
//
// A circle or an arc is drawn as short straight sides (5 degrees each), as
// the FAA's own published shapes are (10 degrees each). Pure.

import { destinationPoint, haversineNm, initialBearingDeg, type LatLon } from "../shared/flightPlanGeometry";

/** degrees of arc per straight side of a circle or an arc */
export const TEXT_ARC_STEP_DEG = 5;
/** areas read from one notice; more are counted as unread */
export const TEXT_MAX_AREAS = 16;
/** corners of one outline; a longer one is unread */
export const TEXT_MAX_RING_POINTS = 512;
/** a radius above this is not a flight restriction's (the largest standing one, Washington, is 60 NM) */
export const TEXT_MAX_RADIUS_NM = 250;

export interface TextAreas {
  /** closed outlines, corners as [lat, lon]; an outline the text writes twice is here once */
  rings: Array<Array<[number, number]>>;
  /** times the text says "an area defined as" */
  areasInText: number;
  /** of those, the ones this reader could not follow to their end */
  unread: number;
  /** times the text says "excluding an area defined as" after an area that was read and overlapping it;
   *  present only when the text has one */
  excluded?: number;
  /** of those, the ones NOT cut out of the outline (the outline there is larger than the restriction);
   *  present with `excluded` */
  uncut?: number;
}

/** group 1 is set for an exclusion */
const AREA_RE = /\b(?:WI(?:THIN)?|(EXCLUDING)) AN AREA DEFINED AS\b/g;
const PT = String.raw`(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([NS])\s?(\d{3})(\d{2})(\d{2}(?:\.\d+)?)([EW])`;
/** a point, then an optional bracket naming it from a navaid: "(MOL272036.5)" */
const POINT_RE = new RegExp(String.raw`^\s*${PT}(?:\s*\([^)]{0,40}\))?`);
const RADIUS_RE = new RegExp(String.raw`^\s*(\d+(?:\.\d+)?)\s?NM RADIUS OF\s+${PT}(?:\s*\([^)]{0,40}\))?`);
const ARC_RE = new RegExp(String.raw`^\s*,?\s*THEN (COUNTER)?CLOCKWISE ON AN? (\d+(?:\.\d+)?)\s?NM ARC CENTERED ON\s+${PT}(?:\s*\([^)]{0,40}\))?`);
const TO_RE = /^\s*,?\s*TO\b/;
const ORIGIN_RE = /^\s*(?:THE\s+)?POINT OF ORIGIN\b/;

function point(m: RegExpExecArray, at: number): LatLon | null {
  const la = (+m[at] + +m[at + 1] / 60 + +m[at + 2] / 3600) * (m[at + 3] === "S" ? -1 : 1);
  const lo = (+m[at + 4] + +m[at + 5] / 60 + +m[at + 6] / 3600) * (m[at + 7] === "W" ? -1 : 1);
  if (+m[at + 1] >= 60 || +m[at + 2] >= 60 || +m[at + 5] >= 60 || +m[at + 6] >= 60) return null;
  return Math.abs(la) <= 90 && Math.abs(lo) <= 180 ? { lat: la, lon: lo } : null;
}
const r6 = (v: number) => Math.round(v * 1e6) / 1e6;
const corner = (p: LatLon): [number, number] => [r6(p.lat), r6(p.lon)];
const radiusOk = (r: number) => Number.isFinite(r) && r > 0 && r <= TEXT_MAX_RADIUS_NM;

function circle(c: LatLon, rNm: number): Array<[number, number]> {
  const ring: Array<[number, number]> = [];
  for (let b = 0; b < 360; b += TEXT_ARC_STEP_DEG) ring.push(corner(destinationPoint(c, b, rNm)));
  ring.push(ring[0]);
  return ring;
}

/** corners strictly between `from` and `to` on an arc around `c` */
function arc(c: LatLon, rNm: number, from: LatLon, to: LatLon, clockwise: boolean): LatLon[] {
  const b1 = initialBearingDeg(c, from), b2 = initialBearingDeg(c, to);
  let sweep = clockwise ? (b2 - b1 + 360) % 360 : (b1 - b2 + 360) % 360;
  // the same point at both ends is the whole circle
  if (sweep < 1e-6 && haversineNm(from, to) < 0.01) sweep = 360;
  const n = Math.floor(sweep / TEXT_ARC_STEP_DEG);
  const out: LatLon[] = [];
  for (let i = 1; i <= n; i++) {
    const d = i * TEXT_ARC_STEP_DEG;
    if (sweep - d < 1e-6) break;
    out.push(destinationPoint(c, clockwise ? b1 + d : b1 - d, rNm));
  }
  return out;
}

/** one area, starting just after "AN AREA DEFINED AS"; null = not followed to its end */
function readArea(s: string): Array<[number, number]> | null {
  const rad = RADIUS_RE.exec(s);
  if (rad) {
    const c = point(rad, 2), r = Number(rad[1]);
    return c && radiusOk(r) ? circle(c, r) : null;
  }
  let m = POINT_RE.exec(s);
  const first = m ? point(m, 1) : null;
  if (!m || !first) return null;
  const pts: LatLon[] = [first];
  let rest = s.slice(m[0].length);
  for (;;) {
    if (pts.length > TEXT_MAX_RING_POINTS) return null;
    const prev = pts[pts.length - 1];
    const a = ARC_RE.exec(rest);
    if (a) {
      const c = point(a, 3), r = Number(a[2]);
      rest = rest.slice(a[0].length);
      const to = TO_RE.exec(rest);
      if (!c || !radiusOk(r) || !to) return null;
      rest = rest.slice(to[0].length);
      const closes = ORIGIN_RE.test(rest);
      m = closes ? null : POINT_RE.exec(rest);
      const end = closes ? first : m ? point(m, 1) : null;
      if (!end) return null;
      pts.push(...arc(c, r, prev, end, !a[1]));
      if (closes) break;
      pts.push(end);
      rest = rest.slice(m![0].length);
      continue;
    }
    const to = TO_RE.exec(rest);
    if (!to) return null;
    rest = rest.slice(to[0].length);
    if (ORIGIN_RE.test(rest)) break;
    m = POINT_RE.exec(rest);
    const p = m ? point(m, 1) : null;
    if (!m || !p) return null;
    pts.push(p);
    rest = rest.slice(m[0].length);
  }
  if (pts.length < 3 || pts.length > TEXT_MAX_RING_POINTS) return null;
  const ring = pts.map(corner);
  ring.push(ring[0]);
  return ring;
}

type XY = [number, number];
function inside(p: XY, poly: XY[]): boolean {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0]) c = !c;
  }
  return c;
}
/** where segment a-b meets segment c-d, or null */
function meet(a: XY, b: XY, c: XY, d: XY): XY | null {
  const rx = b[0] - a[0], ry = b[1] - a[1], sx = d[0] - c[0], sy = d[1] - c[1];
  const den = rx * sy - ry * sx;
  if (Math.abs(den) < 1e-12) return null;
  const t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / den, u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / den;
  return t >= -1e-9 && t <= 1 + 1e-9 && u >= -1e-9 && u <= 1 + 1e-9 ? [a[0] + t * rx, a[1] + t * ry] : null;
}
/** the one run of true values in a cyclic list: [first, last] indexes, or null when there is none or more than one */
function oneRun(v: boolean[]): [number, number] | null {
  const n = v.length;
  let starts = 0, first = -1;
  for (let i = 0; i < n; i++) if (v[i] && !v[(i + n - 1) % n]) { starts++; first = i; }
  if (starts !== 1) return null;
  let last = first;
  while (v[(last + 1) % n]) last = (last + 1) % n;
  return [first, last];
}

/**
 * `area` with the excluded outline `cut` taken out of its edge. Both are closed rings of [lat, lon].
 * "apart" = the two do not overlap, nothing to cut. null = they overlap and this could not cut it
 * (the exclusion lies wholly inside, or the edges cross more than once): the caller counts it.
 * Worked on a flat map around the area's first corner, in NM: at a restriction's size (tens of NM)
 * the flattening moves a crossing point by far less than the outline's own 5-degree sides do.
 */
export function cutOut(area: Array<[number, number]>, cut: Array<[number, number]>): Array<[number, number]> | "apart" | null {
  const lat0 = area[0][0], lon0 = area[0][1], k = Math.cos((lat0 * Math.PI) / 180) * 60;
  const xy = (p: [number, number]): XY => [(p[1] - lon0) * k, (p[0] - lat0) * 60];
  const back = (p: XY): [number, number] => [r6(lat0 + p[1] / 60), r6(lon0 + p[0] / k)];
  const A = area.slice(0, -1), C = cut.slice(0, -1);
  if (A.length < 3 || C.length < 3) return null;
  const a = A.map(xy), c = C.map(xy);
  const aIn = a.map((p) => inside(p, c)), cIn = c.map((p) => inside(p, a));
  const nA = aIn.filter(Boolean).length, nC = cIn.filter(Boolean).length;
  if (!nA && !nC) {
    // no corner of either inside the other: apart, unless their edges still cross (a band across the area)
    for (let i = 0; i < a.length; i++) for (let j = 0; j < c.length; j++) {
      if (meet(a[i], a[(i + 1) % a.length], c[j], c[(j + 1) % c.length])) return null;
    }
    return "apart";
  }
  if (!nC || nC === C.length || nA === A.length) return null;
  const n = A.length, m = C.length;
  // the area's corners that stay: one run (when no corner of the area is inside the cut, all of them,
  // and the cut bites one side between two corners: that side is found below)
  const keep = nA ? oneRun(aIn.map((v) => !v)) : null;
  const bite = oneRun(cIn);
  if (!bite || (nA && !keep)) return null;
  const [s, e] = bite;
  // the two sides of the cut that cross the area's edge: into the run at s, out of it at e
  const cutIn: [XY, XY] = [c[(s + m - 1) % m], c[s]], cutOutSide: [XY, XY] = [c[e], c[(e + 1) % m]];
  let from: number, to: number; // the area's kept corners, in order, from..to
  if (keep) { from = keep[0]; to = keep[1]; } else {
    // no corner inside: both crossings are on one side of the area; keep every corner, starting after that side
    let side = -1;
    for (let i = 0; i < n; i++) if (meet(a[i], a[(i + 1) % n], cutIn[0], cutIn[1]) && meet(a[i], a[(i + 1) % n], cutOutSide[0], cutOutSide[1])) { side = i; break; }
    if (side < 0) return null;
    from = (side + 1) % n; to = side;
  }
  const leave: [XY, XY] = [a[to], a[(to + 1) % n]], rejoin: [XY, XY] = [a[(from + n - 1) % n], a[from]];
  // leaving the area's edge we enter the cut at one end of its inside run and walk to the other
  let viaEnd = meet(leave[0], leave[1], cutOutSide[0], cutOutSide[1]);
  const viaStart = meet(leave[0], leave[1], cutIn[0], cutIn[1]);
  // one side of the area crossed twice: the crossing met first along that side is where we leave it
  const d2 = (p: XY) => (p[0] - leave[0][0]) ** 2 + (p[1] - leave[0][1]) ** 2;
  if (viaEnd && viaStart && d2(viaStart) < d2(viaEnd)) viaEnd = null;
  const x1 = viaEnd ?? viaStart;
  const x2 = viaEnd ? meet(rejoin[0], rejoin[1], cutIn[0], cutIn[1]) : meet(rejoin[0], rejoin[1], cutOutSide[0], cutOutSide[1]);
  if (!x1 || !x2) return null;
  const ring: Array<[number, number]> = [];
  for (let i = from; ; i = (i + 1) % n) { ring.push(A[i]); if (i === to) break; }
  ring.push(back(x1));
  if (viaEnd) for (let j = e; ; j = (j + m - 1) % m) { ring.push(C[j]); if (j === s) break; }
  else for (let j = s; ; j = (j + 1) % m) { ring.push(C[j]); if (j === e) break; }
  ring.push(back(x2));
  ring.push(ring[0]);
  return ring.length <= TEXT_MAX_RING_POINTS + 1 ? ring : null;
}

/** Every area a notice's text defines in the FAA's wording. */
export function textAreas(text: string | null | undefined): TextAreas {
  const out: TextAreas = { rings: [], areasInText: 0, unread: 0 };
  let excluded = 0, uncut = 0;
  if (typeof text !== "string" || text.length < 30) return out;
  const t = text.toUpperCase().replace(/\s+/g, " ");
  AREA_RE.lastIndex = 0;
  const seen = new Set<string>();
  // the area the text last defined: an exclusion is taken out of that one (it follows it in the same sentence)
  let last: { ring: Array<[number, number]> } | null = null;
  const areas: Array<{ ring: Array<[number, number]> }> = [];
  for (let m = AREA_RE.exec(t); m; m = AREA_RE.exec(t)) {
    const words = t.slice(m.index + m[0].length, m.index + m[0].length + 20_000);
    if (m[1]) {
      // an exclusion after an area that was not read has nothing to be cut from: the area is already counted unread
      if (!last) continue;
      const hole = readArea(words);
      const cut = hole ? cutOut(last.ring, hole) : null;
      // an excluded area that does not reach this one takes nothing from it
      if (cut === "apart") continue;
      excluded++;
      if (cut) last.ring = cut; else uncut++;
      continue;
    }
    out.areasInText++;
    const ring = areas.length < TEXT_MAX_AREAS ? readArea(words) : null;
    if (!ring) { out.unread++; last = null; continue; }
    last = { ring };
    areas.push(last);
  }
  for (const a of areas) {
    // the same outline written twice (two height bands over one footprint,
    // FDC 6/2481) is one outline to draw
    const key = JSON.stringify(a.ring);
    if (!seen.has(key)) { seen.add(key); out.rings.push(a.ring); }
  }
  if (excluded) { out.excluded = excluded; out.uncut = uncut; }
  return out;
}
