// RUNWAY DETECTION (2026-10-14, human request: "know what runway they are
// landing on based on their heading and position ... activate the plane for a
// takeoff when it's on the runway"). Pure geometry over FAA CIFP runway ends
// (cifp.ts PG records: threshold position, magnetic bearing, length) and an
// aircraft's broadcast ADS-B fix. Every answer is an INFERENCE from position +
// track against published runway geometry, labeled as such — never an ATC
// runway assignment (we have no ATC data).
//
// States (one fix at a time; `recentlyAirborne` is the only history used):
//   LINED_UP      on a runway, aligned with it, < ROLL_KT
//   TAKEOFF_ROLL  on a runway, aligned, >= ROLL_KT, not airborne recently
//   LANDING_ROLL  on a runway, aligned, >= ROLL_KT, airborne within the window
//   ON_FINAL      airborne, descending/level, on the extended centreline
//                 within FINAL_MAX_NM before the threshold
//   DEPARTED      airborne, climbing out past the runway's far end, aligned
// Tolerances are deliberately loose enough for ADS-B position error
// (NACp 8 = < 93 m) and tight enough that parallel runways (CLT 18L/18C/18R,
// ~1,000–5,000 ft apart) do not collide: the closest centreline wins.

import type { AirportData } from "./cifp";

export type RunwayState = "LINED_UP" | "TAKEOFF_ROLL" | "LANDING_ROLL" | "ON_FINAL" | "DEPARTED";

export interface RunwayEnd {
  /** runway designator as painted at THIS end ("36C") */
  id: string;
  lat: number;
  lon: number;
  /** true heading of travel when departing from / landing on this end */
  headingTrue: number;
  /** usable length (m): threshold-to-opposite-threshold when both are coded */
  lengthM: number;
}

export interface RunwayFix {
  lat: number;
  lon: number;
  /** barometric altitude, ft MSL (null = not broadcast) */
  altFt: number | null;
  onGround: boolean;
  gsKt: number | null;
  /** broadcast ground track, deg true */
  trk: number | null;
  /** vertical rate, ft/min (null = unknown) */
  vsFpm?: number | null;
}

export interface RunwayMatch {
  state: RunwayState;
  runway: string;
  /** metres along the centreline from the matched end's threshold (+ = down the runway) */
  alongM: number;
  /** signed metres off the centreline */
  crossM: number;
}

export const ROLL_KT = 40;
export const ON_RWY_CROSS_M = 75;        // runway half-width ~23–30 m + ADS-B error
export const ON_RWY_END_SLACK_M = 60;
export const ON_RWY_MAX_AGL_FT = 150;
export const ALIGN_RUNWAY_DEG = 20;
export const ALIGN_FINAL_DEG = 15;
export const FINAL_MAX_NM = 12;
export const FINAL_MAX_AGL_FT = 4500;
export const DEPART_MAX_NM = 8;
export const DEPART_MAX_AGL_FT = 6000;
const M_PER_NM = 1852;
const FT_PER_M = 3.28084;

const angDiff = (a: number, b: number) => { const d = Math.abs((((a - b) % 360) + 540) % 360 - 180); return d; };

function bearingDeg(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const la1 = (a.lat * Math.PI) / 180, la2 = (b.lat * Math.PI) / 180, dlo = ((b.lon - a.lon) * Math.PI) / 180;
  const y = Math.sin(dlo) * Math.cos(la2);
  const x = Math.cos(la1) * Math.sin(la2) - Math.sin(la1) * Math.cos(la2) * Math.cos(dlo);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

/** local east/north metres of `p` relative to `o` (flat earth: fine within ~20 nm) */
function enu(o: { lat: number; lon: number }, p: { lat: number; lon: number }): [number, number] {
  const k = Math.cos((o.lat * Math.PI) / 180);
  return [(p.lon - o.lon) * 111_320 * k, (p.lat - o.lat) * 110_540];
}

const RWY_RE = /^RW(\d{2})([LRC]?)$/;
function oppositeId(id: string): string | null {
  const m = RWY_RE.exec(id);
  if (!m) return null;
  const n = ((Number(m[1]) + 17) % 36) + 1;
  const side = m[2] === "L" ? "R" : m[2] === "R" ? "L" : m[2];
  return `RW${String(n).padStart(2, "0")}${side}`;
}

/** Pure: every usable runway end of a CIFP airport. Heading is the geometric
 *  bearing to the opposite threshold when both ends are coded (exact), else
 *  magnetic bearing + the airport's variation. Length is threshold-to-
 *  threshold when both are coded, else the coded length. Ends we cannot
 *  orient are skipped — never guessed. */
export function runwayEnds(ap: Pick<AirportData, "runways" | "magVar">): RunwayEnd[] {
  const out: RunwayEnd[] = [];
  for (const r of Array.from(ap.runways.values())) {
    if (!RWY_RE.test(r.id)) continue;
    const oppId = oppositeId(r.id);
    const o = oppId ? ap.runways.get(oppId) : undefined;
    let heading: number | null = null;
    let lengthM: number | null = null;
    if (o) {
      heading = bearingDeg(r.ll, o.ll);
      const [x, y] = enu(r.ll, o.ll);
      lengthM = Math.hypot(x, y);
    } else if (r.bearingMag != null) {
      heading = (r.bearingMag + ap.magVar + 360) % 360;
    }
    if (lengthM == null && r.lengthFt != null) lengthM = r.lengthFt / FT_PER_M;
    if (heading == null || lengthM == null || !(lengthM > 0)) continue;
    out.push({ id: r.id.replace(/^RW/, ""), lat: r.ll.lat, lon: r.ll.lon, headingTrue: Math.round(heading * 10) / 10, lengthM });
  }
  return out;
}

/** Pure: the runway state of ONE fix against an airport's runway ends, or null.
 *  `elevFt` = airport elevation (for AGL); `recentlyAirborne` separates a
 *  landing roll from a takeoff roll. The closest centreline wins. */
export function classifyRunwayFix(
  fix: RunwayFix,
  ends: RunwayEnd[],
  opts: { elevFt: number | null; recentlyAirborne?: boolean } = { elevFt: null },
): RunwayMatch | null {
  if (fix.trk == null || !Number.isFinite(fix.trk)) return null; // no direction, no runway claim
  const agl = fix.onGround ? 0 : fix.altFt != null && opts.elevFt != null ? fix.altFt - opts.elevFt : null;
  const groundish = fix.onGround || (agl != null && agl <= ON_RWY_MAX_AGL_FT);
  let best: (RunwayMatch & { score: number }) | null = null;
  const consider = (m: RunwayMatch, score: number) => { if (!best || score < best.score) best = { ...m, score }; };
  for (const e of ends) {
    const [x, y] = enu(e, fix);
    const h = (e.headingTrue * Math.PI) / 180;
    const along = x * Math.sin(h) + y * Math.cos(h);
    const cross = x * Math.cos(h) - y * Math.sin(h);
    const dHdg = angDiff(fix.trk, e.headingTrue);
    if (groundish) {
      if (dHdg <= ALIGN_RUNWAY_DEG && along >= -ON_RWY_END_SLACK_M && along <= e.lengthM + ON_RWY_END_SLACK_M
          && Math.abs(cross) <= ON_RWY_CROSS_M) {
        const gs = fix.gsKt ?? 0;
        const state: RunwayState = gs < ROLL_KT ? "LINED_UP" : opts.recentlyAirborne ? "LANDING_ROLL" : "TAKEOFF_ROLL";
        consider({ state, runway: e.id, alongM: along, crossM: cross }, Math.abs(cross) / ON_RWY_CROSS_M);
      }
      continue;
    }
    if (agl == null) continue; // airborne states need a height above the field
    // final approach: before the threshold, inside a widening lateral cone
    const before = -along;
    if (dHdg <= ALIGN_FINAL_DEG && before > 0 && before <= FINAL_MAX_NM * M_PER_NM && agl <= FINAL_MAX_AGL_FT
        && (fix.vsFpm == null || fix.vsFpm <= 300)) {
      const tol = 150 + 0.04 * before;
      if (Math.abs(cross) <= tol) consider({ state: "ON_FINAL", runway: e.id, alongM: along, crossM: cross }, Math.abs(cross) / tol);
    }
    // departure: past the far end, climbing out on the runway heading
    const past = along - e.lengthM;
    if (dHdg <= ALIGN_RUNWAY_DEG && past > -e.lengthM * 0.5 && past <= DEPART_MAX_NM * M_PER_NM && agl <= DEPART_MAX_AGL_FT
        && (fix.vsFpm == null || fix.vsFpm >= -300)) {
      const tol = 300 + 0.06 * Math.max(0, past);
      if (Math.abs(cross) <= tol) consider({ state: "DEPARTED", runway: e.id, alongM: along, crossM: cross }, Math.abs(cross) / tol);
    }
  }
  if (!best) return null;
  const { score: _score, ...m } = best as RunwayMatch & { score: number };
  void _score;
  return m;
}

/** Pure: the runway a trip DEPARTED from, from its earliest recorded fixes
 *  (oldest first). The first fix that classifies as on-runway or departed
 *  wins; later fixes are too far out to attribute. */
export function departureRunway(
  fixes: RunwayFix[],
  ends: RunwayEnd[],
  elevFt: number | null,
  maxFixes = 15,
): RunwayMatch | null {
  for (const f of fixes.slice(0, maxFixes)) {
    const m = classifyRunwayFix(f, ends, { elevFt, recentlyAirborne: false });
    if (m && (m.state === "TAKEOFF_ROLL" || m.state === "DEPARTED" || m.state === "LINED_UP")) return m;
  }
  return null;
}

/** Card wording for a state (honest: inferred, not assigned). */
export function runwayStateLabel(m: RunwayMatch): string {
  switch (m.state) {
    case "LINED_UP": return `Lined up on runway ${m.runway}`;
    case "TAKEOFF_ROLL": return `Taking off — runway ${m.runway}`;
    case "LANDING_ROLL": return `Landing roll — runway ${m.runway}`;
    case "ON_FINAL": return `On final — runway ${m.runway}`;
    case "DEPARTED": return `Departed runway ${m.runway}`;
  }
}

export const RUNWAY_HONESTY =
  "Inferred from the aircraft's broadcast ADS-B position and track against FAA CIFP runway geometry — not an ATC runway assignment.";
