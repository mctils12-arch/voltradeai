// ATC FREQUENCIES (2026-10-14, human request: "is there a way to get the ATC
// frequencies, the ARTCC, or anything"). The FAA publishes every tower,
// ground, clearance, ATIS and approach/departure frequency, the center (RCAG)
// transmitter sites, and the ARTCC boundaries in NASR (public domain;
// scripts/build_nasr_atc.py -> datacore/aircraft/nasr_atc.json). This module
// picks, for one aircraft fix, the published frequencies that apply to where
// it is and what it is doing:
//   RUNWAY    lined up / rolling / on final / just departed -> that runway's
//             tower (+ the departure sector once airborne)
//   SURFACE   on the ground at an airport -> ground, clearance, ATIS (towered)
//             or CTAF / UNICOM (non-towered)
//   TERMINAL  airborne below FL180 within TERMINAL_MAX_NM of an airport with
//             published approach/departure sectors -> the sector containing
//             the aircraft's magnetic bearing + altitude from that airport
//   ENROUTE   otherwise -> the ARTCC whose LOW/HIGH boundary contains the
//             aircraft, and its nearest center transmitters of that stratum
// HONESTY: these are PUBLISHED frequencies chosen by geometry. The frequency
// the crew is actually tuned to is not public: controllers split and combine
// sectors through the day, and center sector boundaries are not published —
// so center frequencies are "nearest transmitters", never "the sector".

import fs from "fs";
import path from "path";
import { repoDataPath } from "./repoFiles";

export type AtcKind = "T" | "G" | "C" | "A" | "AD" | "AP" | "DP" | "CT" | "UN";
export type AtcStrata = "L" | "H" | "LH" | "U";
type LatLon = [number, number];

export interface AtcAirport {
  id: string; lid: string; la: number; lo: number;
  /** magnetic variation, degrees (east +, west -) */
  mv?: number;
  twr?: string; art?: string;
  /** tower / approach radio call ("CHARLOTTE") */
  tc?: string; ac?: string;
  /** tower hours ("24", "0600-2200" local) */
  h?: string;
  /** [MHz, kind, sectorization text, call when it differs from tc/ac] */
  f: Array<[number, AtcKind, string, string?]>;
}

export interface AtcData {
  cycle: string | null;
  airports: AtcAirport[];
  /** [lat, lon, ARTCC, MHz, strata, site] */
  rcag: Array<[number, number, string, number, AtcStrata, string]>;
  artcc: Record<string, { n: string; L?: LatLon[]; H?: LatLon[]; U?: LatLon[][] }>;
}

export type AtcRole = "Tower" | "Ground" | "Clearance" | "ATIS" | "Approach" | "Departure" | "Center" | "CTAF" | "UNICOM";
export type AtcPhase = "RUNWAY" | "SURFACE" | "TERMINAL" | "ENROUTE";

export interface AtcEntry {
  role: AtcRole;
  /** spoken call sign of the position ("Charlotte Tower") */
  call: string;
  mhz: number;
  /** the published sectorization / runway text, or the center site name */
  note: string | null;
  /** true when geometry singles this entry out; false = one of several candidates */
  likely: boolean;
}

export interface AtcFix {
  lat: number; lon: number;
  altFt: number | null;
  onGround: boolean;
  vsFpm?: number | null;
}

export interface AtcContext {
  phase: AtcPhase | null;
  airport: string | null;
  artcc: { id: string; name: string; stratum: "LOW" | "HIGH" | "OCEANIC" } | null;
  entries: AtcEntry[];
}

/** Within this of an airport reference point on the ground = at that airport. */
export const SURFACE_MAX_NM = 4;
/** Approach/departure sectors are looked up within this of an airport. */
export const TERMINAL_MAX_NM = 40;
/** nearest airports (with approach/departure frequencies) examined for a sector match */
export const TERMINAL_CANDIDATES = 8;
/** Class A floor: at or above, the aircraft is with a center. */
export const CENTER_FLOOR_FT = 18_000;
export const CENTER_MAX_NM = 150;
export const MAX_CENTER_FREQS = 3;
export const ATC_HONESTY =
  "Published FAA NASR frequencies, chosen by the aircraft's position, altitude and inferred phase of flight. " +
  "The frequency the crew is actually on is not public: controllers split and combine sectors, and center sector boundaries are not published.";

const M_PER_NM = 1852;

function distNm(la1: number, lo1: number, la2: number, lo2: number): number {
  const r1 = (la1 * Math.PI) / 180, r2 = (la2 * Math.PI) / 180;
  const a = Math.sin((r2 - r1) / 2) ** 2 + Math.cos(r1) * Math.cos(r2) * Math.sin((((lo2 - lo1) * Math.PI) / 180) / 2) ** 2;
  return (2 * 6_371_000 * Math.asin(Math.sqrt(a))) / M_PER_NM;
}

function bearingTrue(la1: number, lo1: number, la2: number, lo2: number): number {
  const r1 = (la1 * Math.PI) / 180, r2 = (la2 * Math.PI) / 180, dlo = ((lo2 - lo1) * Math.PI) / 180;
  const y = Math.sin(dlo) * Math.cos(r2);
  const x = Math.cos(r1) * Math.sin(r2) - Math.sin(r1) * Math.cos(r2) * Math.cos(dlo);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

function rayCast(lat: number, lon: number, ring: LatLon[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [yi, xi] = ring[i], [yj, xj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

const unwrapped = new WeakMap<LatLon[], LatLon[] | null>();
/** a ring that crosses the antimeridian (oceanic Oakland/Anchorage), with its
 *  west longitudes moved to 180..360 so it is one simple polygon; null if it
 *  does not cross */
function unwrapRing(ring: LatLon[]): LatLon[] | null {
  if (!unwrapped.has(ring)) {
    const crosses = ring.some(([, lo], i) => Math.abs(lo - ring[(i + 1) % ring.length][1]) > 180);
    unwrapped.set(ring, crosses ? ring.map(([la, lo]) => [la, lo < 0 ? lo + 360 : lo] as LatLon) : null);
  }
  return unwrapped.get(ring) ?? null;
}

export function pointInRing(lat: number, lon: number, ring: LatLon[]): boolean {
  const u = unwrapRing(ring);
  return u ? rayCast(lat, lon < 0 ? lon + 360 : lon, u) : rayCast(lat, lon, ring);
}

/** Pure: the ARTCC whose boundary for the aircraft's stratum contains it
 *  (LOW below FL180, HIGH at/above; oceanic UNLIMITED rings as a fallback). */
export function artccAt(data: AtcData, lat: number, lon: number, altFt: number | null): AtcContext["artcc"] {
  const high = altFt != null && altFt >= CENTER_FLOOR_FT;
  for (const [id, a] of Object.entries(data.artcc)) {
    const ring = high ? a.H : a.L;
    if (ring && pointInRing(lat, lon, ring)) return { id, name: a.n, stratum: high ? "HIGH" : "LOW" };
  }
  for (const [id, a] of Object.entries(data.artcc)) {
    if (a.U?.some((r) => pointInRing(lat, lon, r))) return { id, name: a.n, stratum: "OCEANIC" };
  }
  return null;
}

export interface Sector {
  /** magnetic bearing range from the airport, clockwise inclusive */
  fromMag: number | null; toMag: number | null;
  minFt: number | null; maxFt: number | null;
  runways: string[];
  side: "N" | "S" | "E" | "W" | null;
  /** false = a condition we cannot evaluate (time windows, navaid radials, "when X closed") */
  understood: boolean;
}

const SECTOR_WORDS = new Set(["FT", "ABV", "BLW", "AND", "&", "SFC", "TO", "HIGH", "LOW", "ABOVE", "BELOW", "INCL"]);
/** a descriptive label beside a bearing range ("150-219 SOUTH", "320-149 EAST/NORTH") */
const LABEL_WORD = /^(NORTH|SOUTH|EAST|WEST)(\/(NORTH|SOUTH|EAST|WEST))*$/;
const SIDES: Record<string, Sector["side"]> = { NORTH: "N", SOUTH: "S", EAST: "E", WEST: "W" };
const COMPASS: Record<string, number> = { N: 0, NE: 45, E: 90, SE: 135, S: 180, SW: 225, W: 270, NW: 315 };

/** Pure: read a NASR SECTORIZATION string. Anything outside the forms below
 *  is `understood: false` and never used to single an entry out. */
export function parseSector(raw: string): Sector {
  const s = raw.toUpperCase().replace(/\s+/g, " ").trim();
  const out: Sector = { fromMag: null, toMag: null, minFt: null, maxFt: null, runways: [], side: null, understood: true };
  if (!s) return out;
  if (s in SIDES) { out.side = SIDES[s]; return out; }
  const cp = /^(N|NE|E|SE|S|SW|W|NW)-(N|NE|E|SE|S|SW|W|NW)$/.exec(s); // SFO "NW-E"
  if (cp) { out.fromMag = COMPASS[cp[1]]; out.toMag = COMPASS[cp[2]]; return out; }
  if (/^RWYS? /.test(s)) {
    out.runways = Array.from(s.slice(4).matchAll(/\b(\d{2}[LRC]?)\b/g), (m) => m[1]);
    out.understood = out.runways.length > 0;
    return out;
  }
  let rest = s;
  const br = /^(\d{3})-(\d{3})\b/.exec(rest);
  if (br) {
    out.fromMag = Number(br[1]) % 360; out.toMag = Number(br[2]) % 360;
    rest = rest.slice(br[0].length);
  }
  const max = /(\d{3,5}) FT (?:&|AND) BLW|BLW (\d{3,5}) FT|SFC TO (\d{3,5})(?: FT)?/.exec(rest);
  if (max) { out.maxFt = Number(max[1] ?? max[2] ?? max[3]); rest = rest.replace(max[0], " "); }
  const min = /ABV (\d{3,5}) FT|(\d{3,5}) FT (?:&|AND) ABV/.exec(rest);
  if (min) { out.minFt = Number(min[1] ?? min[2]); rest = rest.replace(min[0], " "); }
  const left = rest.split(/[\s,]+/).filter(Boolean);
  out.understood = (br != null || max != null || min != null) && left.every((w) => SECTOR_WORDS.has(w) || (br != null && LABEL_WORD.test(w)));
  return out;
}

function inBearing(b: number, from: number, to: number): boolean {
  return from <= to ? b >= from && b <= to : b >= from || b <= to;
}

/** Pure: does a parsed sector contain this aircraft (magnetic bearing from the
 *  airport, altitude)? null = the sector says nothing we can test. */
export function sectorContains(sec: Sector, bearingMag: number, altFt: number | null): boolean | null {
  if (!sec.understood) return null;
  let tested = false;
  if (sec.fromMag != null && sec.toMag != null) {
    tested = true;
    if (!inBearing(bearingMag, sec.fromMag, sec.toMag)) return false;
  }
  if (sec.side) {
    tested = true;
    const want = { N: [270, 90], E: [0, 180], S: [90, 270], W: [180, 360] }[sec.side] as [number, number];
    if (!inBearing(bearingMag, want[0], want[1] % 360)) return false;
  }
  if (sec.minFt != null || sec.maxFt != null) {
    if (altFt == null) return null;
    tested = true;
    if (sec.minFt != null && altFt <= sec.minFt) return false;
    if (sec.maxFt != null && altFt > sec.maxFt) return false;
  }
  return tested ? true : null;
}

const title = (s: string) => s.toLowerCase().replace(/\b([a-z])/g, (m) => m.toUpperCase());
const ROLE: Record<AtcKind, AtcRole> = {
  T: "Tower", G: "Ground", C: "Clearance", A: "ATIS", AD: "Approach", AP: "Approach", DP: "Departure", CT: "CTAF", UN: "UNICOM",
};

const ATIS_NOTE: Record<string, string> = { ARR: "arrivals", DEP: "departures" };

function entryOf(ap: AtcAirport, row: AtcAirport["f"][number], likely: boolean, role?: AtcRole): AtcEntry {
  const [mhz, kind, sector, callOverride] = row;
  const r = role ?? ROLE[kind];
  const base = callOverride ?? (kind === "AD" || kind === "AP" || kind === "DP" ? ap.ac : ap.tc);
  const call = r === "CTAF" || r === "UNICOM" || !base ? `${ap.id} ${r}` : `${title(base)} ${r}`;
  const note = kind === "A" ? (ATIS_NOTE[sector] ?? (sector || null)) : sector || null;
  return { role: r, call, mhz, note, likely };
}

/** runway designators match ignoring the leading zero ("1L" vs "01L") */
const sameRunway = (a: string, b: string) => a.replace(/^0/, "") === b.replace(/^0/, "");

function towerEntries(ap: AtcAirport, runway: string | null): AtcEntry[] {
  const towers = ap.f.filter((r) => r[1] === "T");
  const hit = runway ? towers.filter((r) => parseSector(r[2]).runways.some((x) => sameRunway(x, runway))) : [];
  const general = towers.filter((r) => !r[2]);
  const pick = hit.length ? hit : general.length ? general : towers;
  const likely = hit.length === 1 || (!hit.length && pick.length === 1);
  const out = pick.map((r) => entryOf(ap, r, likely));
  if (ap.h && ap.h !== "24") for (const e of out) e.note = [e.note, `tower hours ${ap.h} local`].filter(Boolean).join(" · ");
  return out;
}

/** approach/departure entries of `ap` that contain the aircraft, role by
 *  vertical rate; `matched` = singled out by a published sector, not just the
 *  airport's general approach frequency */
function sectorEntries(ap: AtcAirport, fix: AtcFix): { entries: AtcEntry[]; matched: boolean } {
  const bMag = (bearingTrue(ap.la, ap.lo, fix.lat, fix.lon) - (ap.mv ?? 0) + 360) % 360;
  const rows = ap.f.filter((r) => r[1] === "AD" || r[1] === "AP" || r[1] === "DP");
  const vs = fix.vsFpm ?? 0;
  const roleFor = (k: AtcKind): AtcRole => (k === "DP" ? "Departure" : k === "AP" ? "Approach" : vs > 300 ? "Departure" : "Approach");
  // a departure-only / approach-only row is skipped when the climb/descent contradicts it
  const fits = (k: AtcKind) => !((k === "DP" && vs < -300) || (k === "AP" && vs > 300));
  const matched = rows.filter((r) => fits(r[1]) && sectorContains(parseSector(r[2]), bMag, fix.altFt) === true);
  if (matched.length) return { entries: matched.map((r) => entryOf(ap, r, matched.length === 1, roleFor(r[1]))), matched: true };
  const general = rows.filter((r) => fits(r[1]) && !r[2]);
  return { entries: general.map((r) => entryOf(ap, r, general.length === 1, roleFor(r[1]))), matched: false };
}

const STRATA_FOR = (altFt: number | null): AtcStrata[] =>
  altFt != null && altFt >= 35_000 ? ["U", "H", "LH"] : altFt != null && altFt >= CENTER_FLOOR_FT ? ["H", "LH"] : ["L", "LH"];

function centerEntries(data: AtcData, fix: AtcFix, artcc: NonNullable<AtcContext["artcc"]>): AtcEntry[] {
  const want = STRATA_FOR(fix.altFt);
  const near = data.rcag
    .filter((r) => r[2] === artcc.id && want.includes(r[4]))
    .map((r) => ({ r, d: distNm(fix.lat, fix.lon, r[0], r[1]) }))
    .filter((x) => x.d <= CENTER_MAX_NM)
    .sort((a, b) => a.d - b.d);
  const out: AtcEntry[] = [];
  const seen = new Set<number>();
  for (const { r } of near) {
    if (seen.has(r[3])) continue;
    seen.add(r[3]);
    out.push({ role: "Center", call: `${title(artcc.name)} Center`, mhz: r[3], note: `${title(r[5].replace(/\/A\/?$/, ""))} transmitter`, likely: false });
    if (out.length >= MAX_CENTER_FREQS) break;
  }
  return out;
}

/** Pure: airports with published frequencies within `maxNm`, nearest first. */
export function airportsNear(data: AtcData, lat: number, lon: number, maxNm: number, pred?: (a: AtcAirport) => boolean): Array<{ ap: AtcAirport; nm: number }> {
  const dLat = maxNm / 60;
  const out: Array<{ ap: AtcAirport; nm: number }> = [];
  for (const ap of data.airports) {
    if (Math.abs(ap.la - lat) > dLat) continue;
    if (pred && !pred(ap)) continue;
    const nm = distNm(lat, lon, ap.la, ap.lo);
    if (nm <= maxNm) out.push({ ap, nm });
  }
  return out.sort((a, b) => a.nm - b.nm);
}

const RUNWAY_PHASE_STATES = new Set(["LINED_UP", "TAKEOFF_ROLL", "LANDING_ROLL", "ON_FINAL", "DEPARTED"]);

/** Pure: the published frequencies that apply to one fix.
 *  `runway` = the inferred runway state (server/runwayDetect.ts) and the
 *  ICAO id of the airport it was matched at, when there is one. */
export function atcContext(
  data: AtcData,
  fix: AtcFix,
  runway: { state: string; runway: string; airport: string } | null = null,
): AtcContext {
  const artcc = fix.onGround ? null : artccAt(data, fix.lat, fix.lon, fix.altFt);
  const byId = (id: string) => data.airports.find((a) => a.id === id || a.lid === id) ?? null;

  if (runway && RUNWAY_PHASE_STATES.has(runway.state)) {
    const ap = byId(runway.airport);
    if (ap) {
      const entries = towerEntries(ap, runway.runway);
      if (runway.state === "DEPARTED") entries.push(...sectorEntries(ap, { ...fix, vsFpm: Math.max(fix.vsFpm ?? 0, 301) }).entries);
      if (entries.length) return { phase: "RUNWAY", airport: ap.id, artcc, entries };
    }
  }

  if (fix.onGround) {
    const at = airportsNear(data, fix.lat, fix.lon, SURFACE_MAX_NM)[0]?.ap;
    if (!at) return { phase: null, airport: null, artcc, entries: [] };
    const towered = at.f.some((r) => r[1] === "T");
    const entries: AtcEntry[] = [];
    if (towered) {
      // a split ground ("WEST"/"EAST", "RWY 10/28") is listed whole: the
      // published split follows ramps and taxiways, not the reference point
      const grounds = at.f.filter((r) => r[1] === "G");
      for (const r of grounds) entries.push(entryOf(at, r, grounds.length === 1));
      for (const r of at.f) if (r[1] === "C") entries.push(entryOf(at, r, false));
      for (const r of at.f) if (r[1] === "A") entries.push(entryOf(at, r, false));
      if (at.h && at.h !== "24") for (const r of at.f) if (r[1] === "CT") entries.push({ ...entryOf(at, r, false), note: `when the tower is closed (open ${at.h} local)` });
    } else {
      const ct = at.f.filter((r) => r[1] === "CT");
      for (const r of ct) entries.push(entryOf(at, r, ct.length === 1));
      for (const r of at.f) if (r[1] === "UN" && !ct.some((c) => c[0] === r[0])) entries.push(entryOf(at, r, false));
      for (const r of at.f) if (r[1] === "C") entries.push(entryOf(at, r, false));
    }
    return { phase: "SURFACE", airport: at.id, artcc, entries };
  }

  if (fix.altFt == null || fix.altFt < CENTER_FLOOR_FT) {
    // a published sector that contains the aircraft (usually the primary
    // airport's, e.g. CLT "001-119 8000 FT & BLW") beats the nearest
    // satellite field's general approach frequency
    const cands = airportsNear(data, fix.lat, fix.lon, TERMINAL_MAX_NM, (a) => a.f.some((r) => r[1] === "AD" || r[1] === "AP" || r[1] === "DP"))
      .slice(0, TERMINAL_CANDIDATES)
      .map(({ ap }) => ({ ap, ...sectorEntries(ap, fix) }))
      .filter((c) => c.entries.length);
    const pick = cands.find((c) => c.matched) ?? cands[0];
    if (pick) return { phase: "TERMINAL", airport: pick.ap.id, artcc, entries: pick.entries };
  }

  if (artcc) return { phase: "ENROUTE", airport: null, artcc, entries: centerEntries(data, fix, artcc) };
  return { phase: null, airport: null, artcc: null, entries: [] };
}

let cache: AtcData | null = null;

/** Lazy singleton (1.1 MB JSON, parsed on first use, never at boot). A missing
 *  file degrades to an empty table and is logged — never thrown at callers. */
export function loadAtcData(jsonPath?: string): AtcData {
  if (cache) return cache;
  const p = jsonPath || repoDataPath(path.join("datacore", "aircraft", "nasr_atc.json"));
  try {
    const d = JSON.parse(fs.readFileSync(p, "utf-8"));
    cache = { cycle: d.cycle ?? null, airports: d.airports || [], rcag: d.rcag || [], artcc: d.artcc || {} };
  } catch (e: unknown) {
    console.error("[atc] load:", e instanceof Error ? e.message : e);
    cache = { cycle: null, airports: [], rcag: [], artcc: {} };
  }
  return cache;
}

/** test hook */
export function _resetAtcData(): void { cache = null; }
