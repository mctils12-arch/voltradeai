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
}

const AREA_RE = /\bWI(?:THIN)? AN AREA DEFINED AS\b/g;
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

/** Every area a notice's text defines in the FAA's wording. */
export function textAreas(text: string | null | undefined): TextAreas {
  const out: TextAreas = { rings: [], areasInText: 0, unread: 0 };
  if (typeof text !== "string" || text.length < 30) return out;
  const t = text.toUpperCase().replace(/\s+/g, " ");
  AREA_RE.lastIndex = 0;
  const seen = new Set<string>();
  for (let m = AREA_RE.exec(t); m; m = AREA_RE.exec(t)) {
    out.areasInText++;
    const ring = out.rings.length < TEXT_MAX_AREAS ? readArea(t.slice(m.index + m[0].length, m.index + m[0].length + 20_000)) : null;
    if (!ring) { out.unread++; continue; }
    // the same outline written twice (two height bands over one footprint,
    // FDC 6/2481) is one outline to draw
    const key = JSON.stringify(ring);
    if (!seen.has(key)) { seen.add(key); out.rings.push(ring); }
  }
  return out;
}
