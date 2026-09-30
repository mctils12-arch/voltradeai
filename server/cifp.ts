// FAA CIFP (Coded Instrument Flight Procedures, ARINC 424-18) — the
// authoritative VECTOR half of "plate on the map": every SID / STAR /
// instrument approach at US airports as coded legs, turned into GeoJSON.
//
// Source: https://aeronav.faa.gov/Upload_313-d/cifp/CIFP_<YYMMDD>.zip (FAA,
// public domain, 28-day AIRAC cycle). ~9 MB zipped / ~53 MB text. The file is
// extracted ONCE per cycle to the container's /tmp (never the nearly-full
// /data volume; the previous cycle's copy is deleted) and indexed by byte
// offset: airport blocks are read on demand (~100 kB each) and parsed into a
// small LRU; only the global fix tables (enroute waypoints, navaids — ~40k
// entries) stay resident.
//
// GEOMETRY HONESTY. ARINC 424 legs with a fixed terminator (IF, TF, CF, DF,
// RF, AF, FC) are drawn exactly between published coordinates. Legs whose end
// depends on aircraft performance or ATC (CA/VA/FA "to an altitude", VI/CI
// "to intercept", VM/FM "manual termination", hold sizes, procedure turns)
// are drawn as an honest approximation and flagged `approx: true` with the
// assumption in `approxNote`. Turn anticipation / fly-by arcs are never
// drawn — the path is the leg sequence, not a flight simulation.
// NOT FOR NAVIGATION.

import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { EARTH_RADIUS_NM } from "../shared/flightPlanGeometry";

// ── AIRAC cycle math ────────────────────────────────────────────────────────

const DAY_MS = 86_400_000;
export const CYCLE_MS = 28 * DAY_MS;
/** AIRAC 2601 became effective 2026-01-22 09:01Z; every cycle is 28 days. */
export const CYCLE_EPOCH_MS = Date.UTC(2026, 0, 22, 9, 1);

export interface AiracCycle {
  /** "2609" */
  ident: string;
  effective: string;
  expires: string;
  effectiveMs: number;
  /** "260903" (CIFP zip naming) */
  yymmdd: string;
}

export function cycleFromEffective(effMs: number): AiracCycle {
  const d = new Date(effMs);
  const y = d.getUTCFullYear();
  const idx = Math.floor((effMs - Date.UTC(y, 0, 1)) / CYCLE_MS) + 1;
  const yy = String(y % 100).padStart(2, "0");
  const p2 = (n: number) => String(n).padStart(2, "0");
  return {
    ident: `${yy}${p2(idx)}`,
    effective: new Date(effMs).toISOString(),
    expires: new Date(effMs + CYCLE_MS).toISOString(),
    effectiveMs: effMs,
    yymmdd: `${yy}${p2(d.getUTCMonth() + 1)}${p2(d.getUTCDate())}`,
  };
}

/** The AIRAC cycle in force at `ms` (offset -1 = previous, +1 = next). */
export function cycleAt(ms: number, offset = 0): AiracCycle {
  const k = Math.floor((ms - CYCLE_EPOCH_MS) / CYCLE_MS) + offset;
  return cycleFromEffective(CYCLE_EPOCH_MS + k * CYCLE_MS);
}

export const cifpZipUrl = (c: AiracCycle) => `https://aeronav.faa.gov/Upload_313-d/cifp/CIFP_${c.yymmdd}.zip`;

// ── minimal ZIP reader (central directory + inflateRaw; no dependency) ───────

export function extractZipEntry(zip: Buffer, want: (name: string) => boolean, maxBytes = 200 * 1024 * 1024): { name: string; data: Buffer } | null {
  let eocd = -1;
  for (let i = zip.length - 22; i >= Math.max(0, zip.length - 65_557); i--) {
    if (zip.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("not a zip (no end-of-central-directory)");
  const count = zip.readUInt16LE(eocd + 10);
  let p = zip.readUInt32LE(eocd + 16);
  for (let k = 0; k < count && p + 46 <= zip.length; k++) {
    if (zip.readUInt32LE(p) !== 0x02014b50) throw new Error("corrupt zip central directory");
    const method = zip.readUInt16LE(p + 10);
    const csize = zip.readUInt32LE(p + 20);
    const usize = zip.readUInt32LE(p + 24);
    const nlen = zip.readUInt16LE(p + 28), xlen = zip.readUInt16LE(p + 30), clen = zip.readUInt16LE(p + 32);
    const local = zip.readUInt32LE(p + 42);
    const name = zip.toString("utf8", p + 46, p + 46 + nlen);
    p += 46 + nlen + xlen + clen;
    if (!want(name)) continue;
    if (usize > maxBytes) throw new Error(`zip entry ${name} too large (${usize})`);
    const lnl = zip.readUInt16LE(local + 26), lxl = zip.readUInt16LE(local + 28);
    const start = local + 30 + lnl + lxl;
    const raw = zip.subarray(start, start + csize);
    if (method === 0) return { name, data: Buffer.from(raw) };
    if (method === 8) return { name, data: zlib.inflateRawSync(raw, { maxOutputLength: maxBytes }) };
    throw new Error(`unsupported zip method ${method}`);
  }
  return null;
}

// ── ARINC 424 field parsing ─────────────────────────────────────────────────

export interface LL { lat: number; lon: number }

/** "N30174456" + "W097354238" -> decimal degrees (seconds in hundredths). */
export function parseArincLatLon(s: string): LL | null {
  const m = /^([NS])(\d{2})(\d{2})(\d{4})([EW])(\d{3})(\d{2})(\d{4})$/.exec(s);
  if (!m) return null;
  const lat = (Number(m[2]) + Number(m[3]) / 60 + Number(m[4]) / 360_000) * (m[1] === "S" ? -1 : 1);
  const lon = (Number(m[6]) + Number(m[7]) / 60 + Number(m[8]) / 360_000) * (m[5] === "W" ? -1 : 1);
  return { lat, lon };
}
const llAt = (line: string, i: number) => parseArincLatLon(line.slice(i, i + 19));

/** "E0040" -> +4.0 (east positive); "W0110" -> -11.0; "T0000" true-referenced -> 0 */
export function parseMagVar(s: string): number | null {
  const m = /^([EWT])(\d{4})$/.exec(s);
  if (!m) return null;
  const v = Number(m[2]) / 10;
  return m[1] === "W" ? -v : v;
}

export type AltDesc = "AT" | "AT_OR_ABOVE" | "AT_OR_BELOW" | "BETWEEN" | "GS_INTERCEPT" | "OTHER";
export interface AltConstraint { desc: AltDesc; code: string; alt1Ft: number | null; alt2Ft: number | null; text: string }

function parseAltField(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  if (/^FL\d{3}$/.test(t)) return Number(t.slice(2)) * 100;
  if (/^-?\d{1,5}$/.test(t)) return Number(t);
  return null;
}
const fmtAlt = (ft: number) => (ft >= 18_000 ? `FL${Math.round(ft / 100)}` : String(ft));

export function parseAltitude(code: string, a1: string, a2: string): AltConstraint | null {
  const alt1 = parseAltField(a1), alt2 = parseAltField(a2);
  if (alt1 == null && alt2 == null) return null;
  const c = code.trim();
  let desc: AltDesc = "OTHER";
  let text = "";
  if (c === "" && alt1 != null) { desc = "AT"; text = fmtAlt(alt1); } else if (c === "+" && alt1 != null) { desc = "AT_OR_ABOVE"; text = `${fmtAlt(alt1)}A`; } else if (c === "-" && alt1 != null) { desc = "AT_OR_BELOW"; text = `${fmtAlt(alt1)}B`; } else if (c === "B" && alt1 != null && alt2 != null) { desc = "BETWEEN"; text = `${fmtAlt(alt2)}–${fmtAlt(alt1)}`; } else if ((c === "G" || c === "H" || c === "I" || c === "J") && alt1 != null) {
    desc = "GS_INTERCEPT"; text = `${fmtAlt(alt1)} (GS)`;
  } else if (alt1 != null) { text = fmtAlt(alt1); }
  return { desc, code: c, alt1Ft: alt1, alt2Ft: alt2, text };
}

export interface FixRef { ident: string; region: string; section: string; sub: string }

export interface CifpLeg {
  seq: number;
  routeType: string;
  transition: string;
  fix: FixRef | null;
  /** waypoint description code (col 40-43) */
  desc: string;
  turn: "L" | "R" | null;
  pathTerm: string;
  recNav: FixRef | null;
  /** arc radius (nm) for RF */
  arcRadiusNm: number | null;
  thetaDeg: number | null;
  rhoNm: number | null;
  /** course, degrees; `courseTrue` says whether it is already true */
  courseDeg: number | null;
  courseTrue: boolean;
  distNm: number | null;
  timeMin: number | null;
  alt: AltConstraint | null;
  speedKt: number | null;
  vertAngleDeg: number | null;
  center: FixRef | null;
  /** first leg of the missed approach (col 42 'M') */
  missedStart: boolean;
}

const num10 = (s: string) => (/^\d+$/.test(s.trim()) ? Number(s.trim()) / 10 : null);

/** Parse one SID/STAR/approach primary record (PD/PE/PF). Continuation
 *  records (col 39 not '0'/'1') return null. */
export function parseProcedureLeg(line: string): CifpLeg | null {
  if (line.length < 120) return null;
  const cont = line[38];
  if (cont !== "0" && cont !== "1") return null;
  const fixIdent = line.slice(29, 34).trim();
  const ref = (ident: string, region: string, section: string, sub: string): FixRef | null =>
    ident ? { ident, region, section, sub } : null;
  const course = line.slice(70, 74);
  const dist = line.slice(74, 78);
  const turn = line[43];
  const spd = line.slice(99, 102).trim();
  const va = line.slice(102, 106).trim();
  const arc = line.slice(56, 62).trim();
  return {
    seq: Number(line.slice(26, 29)),
    routeType: line[19],
    transition: line.slice(20, 25).trim(),
    fix: ref(fixIdent, line.slice(34, 36), line[36], line[37]),
    desc: line.slice(39, 43),
    turn: turn === "L" || turn === "R" ? turn : null,
    pathTerm: line.slice(47, 49),
    recNav: ref(line.slice(50, 54).trim(), line.slice(54, 56), line[78], line[79]),
    arcRadiusNm: /^\d+$/.test(arc) ? Number(arc) / 1000 : null,
    thetaDeg: num10(line.slice(62, 66)),
    rhoNm: num10(line.slice(66, 70)),
    courseDeg: /^\d{3}T$/.test(course) ? Number(course.slice(0, 3)) : num10(course),
    courseTrue: /^\d{3}T$/.test(course),
    distNm: dist.startsWith("T") ? null : num10(dist),
    timeMin: dist.startsWith("T") ? num10(dist.slice(1)) : null,
    alt: parseAltitude(line[82], line.slice(84, 89), line.slice(89, 94)),
    speedKt: /^\d+$/.test(spd) ? Number(spd) : null,
    vertAngleDeg: /^-?\d+$/.test(va) ? Number(va) / 100 : null,
    center: ref(line.slice(106, 111).trim(), line.slice(112, 114), line[114], line[115]),
    missedStart: line[41] === "M",
  };
}

// ── airport model ───────────────────────────────────────────────────────────

export type ProcKind = "SID" | "STAR" | "IAP";
export interface ProcTransition { routeType: string; id: string; legs: CifpLeg[] }
export interface CifpProcedure { kind: ProcKind; id: string; transitions: ProcTransition[] }

export interface Runway { id: string; ll: LL; bearingMag: number | null; lengthFt: number | null }
export interface AirportData {
  icao: string;
  name: string;
  ll: LL;
  magVar: number;
  elevFt: number | null;
  terminalFixes: Map<string, LL>;
  runways: Map<string, Runway>;
  localizers: Map<string, LL>;
  procedures: CifpProcedure[];
}

export function parseAirportBlock(icao: string, lines: string[]): AirportData | null {
  let head: AirportData | null = null;
  const terminalFixes = new Map<string, LL>();
  const runways = new Map<string, Runway>();
  const localizers = new Map<string, LL>();
  const groups = new Map<string, { kind: ProcKind; id: string; tr: Map<string, ProcTransition> }>();
  for (const line of lines) {
    const sub = line[12];
    if (sub === "A") {
      const ll = llAt(line, 32);
      if (!ll) continue;
      const elev = line.slice(56, 61).trim();
      head = {
        icao, name: line.slice(93, 123).trim(), ll,
        magVar: parseMagVar(line.slice(51, 56)) ?? 0,
        elevFt: /^-?\d+$/.test(elev) ? Number(elev) : null,
        terminalFixes, runways, localizers, procedures: [],
      };
    } else if (sub === "C") {
      const ll = llAt(line, 32);
      if (ll) terminalFixes.set(line.slice(13, 18).trim(), ll);
    } else if (sub === "G") {
      const ll = llAt(line, 32);
      if (!ll) continue;
      const len = line.slice(22, 27).trim(), brg = line.slice(27, 31).trim();
      runways.set(line.slice(13, 18).trim(), {
        id: line.slice(13, 18).trim(), ll,
        bearingMag: /^\d{4}$/.test(brg) ? Number(brg) / 10 : null,
        lengthFt: /^\d+$/.test(len) ? Number(len) : null,
      });
    } else if (sub === "I") {
      const ll = llAt(line, 32);
      if (ll) localizers.set(line.slice(13, 17).trim(), ll);
    } else if (sub === "D" || sub === "E" || sub === "F") {
      const leg = parseProcedureLeg(line);
      if (!leg) continue;
      const kind: ProcKind = sub === "D" ? "SID" : sub === "E" ? "STAR" : "IAP";
      const id = line.slice(13, 19).trim();
      const gk = `${sub}|${id}`;
      let g = groups.get(gk);
      if (!g) { g = { kind, id, tr: new Map() }; groups.set(gk, g); }
      const tk = `${leg.routeType}|${leg.transition}`;
      let t = g.tr.get(tk);
      if (!t) { t = { routeType: leg.routeType, id: leg.transition, legs: [] }; g.tr.set(tk, t); }
      t.legs.push(leg);
    }
  }
  if (!head) return null;
  head.procedures = Array.from(groups.values()).map((g) => ({
    kind: g.kind, id: g.id,
    transitions: Array.from(g.tr.values()).map((t) => ({ ...t, legs: t.legs.sort((a, b) => a.seq - b.seq) })),
  }));
  return head;
}

// ── the index (global fixes + airport byte ranges) ─────────────────────────

export interface NavaidInfo { ll: LL; name: string }

export class CifpIndex {
  readonly waypoints = new Map<string, LL>();        // EA  ident|region
  readonly navaids = new Map<string, NavaidInfo>();  // D / DB / PN  ident|region
  /** independent gazetteer (FAA NASR FIX/NAV, server/navFixes.ts): fallback
   *  when a CIFP reference does not resolve, and a cross-check when it does */
  external: ((ident: string) => LL | null) | null = null;
  private blocks = new Map<string, [number, number]>();
  private cache = new Map<string, AirportData | null>();
  static readonly AIRPORT_LRU = 48;

  constructor(private readBlock: (start: number, end: number) => string, public readonly cycle: string | null = null) {}

  /** Single pass over the whole file (as a latin1 string or buffer). */
  static build(text: Buffer | string, readBlock?: (s: number, e: number) => string, cycle: string | null = null): CifpIndex {
    const buf = typeof text === "string" ? Buffer.from(text, "latin1") : text;
    const idx = new CifpIndex(readBlock ?? ((s, e) => buf.toString("latin1", s, e)), cycle);
    let pos = 0;
    let curAp: string | null = null, curStart = 0;
    const closeAp = (end: number) => {
      if (curAp && !idx.blocks.has(curAp)) idx.blocks.set(curAp, [curStart, end]);
      curAp = null;
    };
    while (pos < buf.length) {
      let nl = buf.indexOf(10, pos);
      if (nl < 0) nl = buf.length;
      const line = buf.toString("latin1", pos, Math.min(nl, pos + 132));
      const lineStart = pos;
      pos = nl + 1;
      if (line.length < 60 || line[0] !== "S") { closeAp(lineStart); continue; }
      const sec = line[4], sub5 = line[5];
      if (sec === "P" && sub5 === " ") {
        const ap = line.slice(6, 10).trim();
        if (ap !== curAp) { closeAp(lineStart); curAp = ap; curStart = lineStart; }
        continue;
      }
      closeAp(lineStart);
      if (line[21] !== "0" && line[21] !== "1") continue; // continuation record (fix/navaid layout: col 22)
      const ident = line.slice(13, 18).trim();
      const region = line.slice(19, 21);
      if (sec === "E" && sub5 === "A") {
        const ll = llAt(line, 32);
        if (ll) idx.waypoints.set(`${ident}|${region}`, ll);
      } else if (sec === "D" || (sec === "P" && sub5 === "N")) {
        const nid = line.slice(13, 17).trim();
        const ll = llAt(line, 32) ?? llAt(line, 55);
        if (ll) idx.navaids.set(`${nid}|${region}`, { ll, name: line.slice(93, 123).trim() });
      }
    }
    closeAp(buf.length);
    return idx;
  }

  airportIds(): string[] { return Array.from(this.blocks.keys()); }
  hasAirport(icao: string): boolean { return this.blocks.has(icao.toUpperCase()); }

  airport(icaoRaw: string): AirportData | null {
    const icao = icaoRaw.toUpperCase();
    if (this.cache.has(icao)) {
      const v = this.cache.get(icao) ?? null;
      this.cache.delete(icao); this.cache.set(icao, v); // LRU touch
      return v;
    }
    const r = this.blocks.get(icao);
    const data = r ? parseAirportBlock(icao, this.readBlock(r[0], r[1]).split(/\r?\n/)) : null;
    this.cache.set(icao, data);
    while (this.cache.size > CifpIndex.AIRPORT_LRU) this.cache.delete(this.cache.keys().next().value as string);
    return data;
  }

  navaidName(ident: string): string | null {
    for (const [k, v] of Array.from(this.navaids.entries())) if (k.startsWith(`${ident}|`)) return v.name;
    return null;
  }

  /** Position of a coded reference from CIFP itself (section/subsection
   *  decide the table; ident+region is the key). */
  resolvePrimary(f: FixRef | null, ap: AirportData): LL | null {
    if (!f) return null;
    const key = `${f.ident}|${f.region}`;
    if (f.section === "P" && f.sub === "C") return ap.terminalFixes.get(f.ident) ?? this.waypoints.get(key) ?? null;
    if (f.section === "P" && f.sub === "G") return ap.runways.get(f.ident)?.ll ?? null;
    if (f.section === "P" && f.sub === "I") return ap.localizers.get(f.ident) ?? null;
    if (f.section === "E" && f.sub === "A") return this.waypoints.get(key) ?? null;
    if (f.section === "D" || (f.section === "P" && f.sub === "N")) return this.navaids.get(key)?.ll ?? null;
    return ap.terminalFixes.get(f.ident) ?? this.waypoints.get(key) ?? this.navaids.get(key)?.ll ?? null;
  }

  /** CIFP first; the NASR gazetteer only for fixes/navaids CIFP did not
   *  resolve (never for runways or localizers, which NASR FIX/NAV lacks). */
  resolve(f: FixRef | null, ap: AirportData): LL | null {
    const p = this.resolvePrimary(f, ap);
    if (p || !f || !this.external) return p;
    if (f.section === "P" && (f.sub === "G" || f.sub === "I")) return null;
    return this.external(f.ident);
  }
}

// ── procedure naming ────────────────────────────────────────────────────────

export const APPROACH_TYPES: Record<string, string> = {
  I: "ILS", L: "LOC", B: "LOC BC", R: "RNAV (GPS)", H: "RNAV (RNP)", V: "VOR", D: "VOR/DME", S: "VOR",
  N: "NDB", Q: "NDB/DME", P: "GPS", X: "LDA", U: "SDF", G: "IGS", J: "GLS", T: "TACAN", W: "MLS", F: "FMS",
};

export interface ApproachIdent { typeCode: string; typeName: string; runway: string | null; variant: string | null }
export function parseApproachIdent(id: string): ApproachIdent {
  const m = /^([A-Z])(\d{2}[LRCB]?)(?:-?([A-Z]))?$/.exec(id);
  if (m) return { typeCode: m[1], typeName: APPROACH_TYPES[m[1]] ?? m[1], runway: m[2], variant: m[3] ?? null };
  const c = /^([A-Z])[A-Z]*-?([A-Z])$/.exec(id); // circling: e.g. VDM-A
  return { typeCode: id[0], typeName: APPROACH_TYPES[id[0]] ?? id[0], runway: null, variant: c ? c[2] : null };
}
export function approachName(id: string): string {
  const a = parseApproachIdent(id);
  if (!a.runway) return `${a.typeName}${a.variant ? `-${a.variant}` : ""}`;
  return `${a.typeName}${a.variant ? ` ${a.variant}` : ""} RWY ${a.runway}`;
}

/** Transition role by kind + ARINC route type. */
export function segmentRole(kind: ProcKind, routeType: string): string {
  if (kind === "IAP") return routeType === "A" ? "approach transition" : "final";
  const n = Number(routeType);
  if (kind === "SID") return n === 1 || n === 4 || n === 7 || routeType === "F" || routeType === "T" ? "runway transition" : n === 2 || n === 5 || n === 8 || routeType === "M" ? "common" : "enroute transition";
  return n === 1 || n === 4 || n === 7 || routeType === "F" ? "enroute transition" : n === 2 || n === 5 || n === 8 || routeType === "M" ? "common" : "runway transition";
}

// ── leg geometry ────────────────────────────────────────────────────────────

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;
export function destinationPoint(p: LL, brgDeg: number, distNm: number): LL {
  const d = distNm / EARTH_RADIUS_NM, b = brgDeg * D2R, la = p.lat * D2R, lo = p.lon * D2R;
  const la2 = Math.asin(Math.sin(la) * Math.cos(d) + Math.cos(la) * Math.sin(d) * Math.cos(b));
  const lo2 = lo + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(la), Math.cos(d) - Math.sin(la) * Math.sin(la2));
  return { lat: la2 * R2D, lon: ((lo2 * R2D + 540) % 360) - 180 };
}
export function distNm(a: LL, b: LL): number {
  const la1 = a.lat * D2R, la2 = b.lat * D2R, dla = la2 - la1, dlo = (b.lon - a.lon) * D2R;
  const h = Math.sin(dla / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(dlo / 2) ** 2;
  return 2 * EARTH_RADIUS_NM * Math.asin(Math.min(1, Math.sqrt(h)));
}
export function bearingDeg(a: LL, b: LL): number {
  const la1 = a.lat * D2R, la2 = b.lat * D2R, dlo = (b.lon - a.lon) * D2R;
  const y = Math.sin(dlo) * Math.cos(la2);
  const x = Math.cos(la1) * Math.sin(la2) - Math.sin(la1) * Math.cos(la2) * Math.cos(dlo);
  return (Math.atan2(y, x) * R2D + 360) % 360;
}

/** Local flat frame (nm) about a centre for intersections. */
const toXY = (p: LL, c: LL): [number, number] => [(p.lon - c.lon) * 60 * Math.cos(c.lat * D2R), (p.lat - c.lat) * 60];
const fromXY = (x: number, y: number, c: LL): LL => ({ lat: c.lat + y / 60, lon: c.lon + x / (60 * Math.cos(c.lat * D2R)) });

/** Ray from p along brg intersected with the line through q at brg2. Returns
 *  distance along the ray (nm), or null when parallel / behind. */
export function rayLineIntersect(p: LL, brg: number, q: LL, brg2: number): number | null {
  const [qx, qy] = toXY(q, p);
  const dx = Math.sin(brg * D2R), dy = Math.cos(brg * D2R);
  const ex = Math.sin(brg2 * D2R), ey = Math.cos(brg2 * D2R);
  const den = dx * ey - dy * ex;
  if (Math.abs(den) < 1e-6) return null;
  const t = (qx * ey - qy * ex) / den;
  return t > 0 ? t : null;
}
/** Ray from p along brg until it is `r` nm from centre c (first crossing ahead). */
export function rayCircleIntersect(p: LL, brg: number, c: LL, r: number): number | null {
  const [cx, cy] = toXY(c, p);
  const dx = Math.sin(brg * D2R), dy = Math.cos(brg * D2R);
  const b = -2 * (dx * cx + dy * cy), cc = cx * cx + cy * cy - r * r;
  const disc = b * b - 4 * cc;
  if (disc < 0) return null;
  const s = Math.sqrt(disc);
  const ts = [(-b - s) / 2, (-b + s) / 2].filter((t) => t > 0.05);
  return ts.length ? Math.min(...ts) : null;
}

export function arcPoints(center: LL, from: LL, to: LL, turn: "L" | "R" | null, radiusNm?: number): LL[] {
  const r = radiusNm ?? (distNm(center, from) + distNm(center, to)) / 2;
  const a0 = bearingDeg(center, from), a1 = bearingDeg(center, to);
  let sweep = ((a1 - a0) + 360) % 360; // clockwise (right turn)
  if (turn === "L") sweep -= 360;
  else if (turn == null && sweep > 180) sweep -= 360; // shortest when unspecified
  const steps = Math.max(4, Math.ceil(Math.abs(sweep) / 5));
  const out: LL[] = [];
  for (let i = 0; i <= steps; i++) out.push(destinationPoint(center, a0 + (sweep * i) / steps, r));
  out[out.length - 1] = to;
  return out;
}

/** approximation constants — each one is stated in the leg's approxNote */
export const CLIMB_GRADIENT_FT_PER_NM = 200;   // TERPS standard minimum climb gradient
export const HOLD_SPEED_KT = 210;              // hold leg length from time at a typical terminal speed
export const MANUAL_TERM_STUB_NM = 3;          // VM/FM: "expect vectors" drawn as a short stub
export const APPROX_FALLBACK_NM = 5;

export interface LegGeom {
  leg: CifpLeg;
  coords: LL[];
  approx: boolean;
  approxNote: string | null;
  end: LL | null;
}

/**
 * Build geometry for a leg sequence. `start` = where the previous segment
 * ended (null for the first leg). Unknown leg types / unresolvable fixes
 * produce no geometry for that leg and are counted by the caller — never
 * guessed into place.
 */
export function buildLegs(legs: CifpLeg[], ap: AirportData, idx: CifpIndex, start: LL | null, startAltFt: number | null): LegGeom[] {
  const out: LegGeom[] = [];
  let cur = start;
  let curAlt = startAltFt ?? ap.elevFt ?? 0;
  const trueCourse = (l: CifpLeg) => (l.courseDeg == null ? null : l.courseTrue ? l.courseDeg : (l.courseDeg + ap.magVar + 360) % 360);
  for (let i = 0; i < legs.length; i++) {
    const l = legs[i];
    const pt = l.pathTerm;
    const fixLL = idx.resolve(l.fix, ap);
    const crs = trueCourse(l);
    let coords: LL[] = [];
    let approx = false;
    let note: string | null = null;
    let end: LL | null = null;
    const toAlt = () => {
      const target = l.alt?.alt1Ft ?? curAlt + 1000;
      const d = Math.min(12, Math.max(1, (target - curAlt) / CLIMB_GRADIENT_FT_PER_NM));
      return { d, note: `ends at ${l.alt?.text ?? "an altitude"}: length assumes a ${CLIMB_GRADIENT_FT_PER_NM} ft/nm climb` };
    };
    switch (pt) {
      case "IF":
        end = fixLL; coords = fixLL ? [fixLL] : [];
        break;
      case "TF": case "CF": case "DF":
        if (fixLL) { end = fixLL; coords = cur ? [cur, fixLL] : [fixLL]; }
        break;
      case "RF": {
        const c = idx.resolve(l.center, ap);
        if (fixLL && cur && c) { coords = arcPoints(c, cur, fixLL, l.turn, l.arcRadiusNm ?? undefined); end = fixLL; } else if (fixLL) { coords = cur ? [cur, fixLL] : [fixLL]; end = fixLL; approx = true; note = "arc centre not resolvable — drawn straight"; }
        break;
      }
      case "AF": {
        const nav = idx.resolve(l.recNav, ap);
        if (fixLL && cur && nav) { coords = arcPoints(nav, cur, fixLL, l.turn, l.rhoNm ?? undefined); end = fixLL; } else if (fixLL) { coords = cur ? [cur, fixLL] : [fixLL]; end = fixLL; approx = true; note = "DME arc navaid not resolvable — drawn straight"; }
        break;
      }
      case "CA": case "VA": case "FA": {
        const from = pt === "FA" ? fixLL : cur;
        if (from && crs != null) {
          const t = toAlt();
          end = destinationPoint(from, crs, t.d); coords = [from, end]; approx = true;
          note = pt === "VA" ? `heading (wind not modelled); ${t.note}` : t.note;
        }
        break;
      }
      case "CD": case "VD": case "FD": {
        const from = pt === "FD" ? fixLL : cur;
        const nav = idx.resolve(l.recNav, ap);
        if (from && crs != null) {
          const d = nav && l.distNm != null ? rayCircleIntersect(from, crs, nav, l.distNm) : null;
          end = destinationPoint(from, crs, d ?? APPROX_FALLBACK_NM); coords = [from, end];
          approx = d == null || pt === "VD";
          note = d == null ? `DME terminator not resolvable — ${APPROX_FALLBACK_NM} nm stub` : pt === "VD" ? "heading leg (wind not modelled)" : null;
        }
        break;
      }
      case "CR": case "VR": {
        const nav = idx.resolve(l.recNav, ap);
        if (cur && crs != null) {
          const radialTrue = l.thetaDeg != null ? (l.thetaDeg + ap.magVar + 360) % 360 : null;
          const d = nav && radialTrue != null ? rayLineIntersect(cur, crs, nav, radialTrue) : null;
          end = destinationPoint(cur, crs, d ?? APPROX_FALLBACK_NM); coords = [cur, end];
          approx = d == null || pt === "VR";
          note = d == null ? `radial terminator not resolvable — ${APPROX_FALLBACK_NM} nm stub` : pt === "VR" ? "heading leg (wind not modelled)" : null;
        }
        break;
      }
      case "CI": case "VI": {
        const nx = legs[i + 1];
        const nfix = nx ? idx.resolve(nx.fix, ap) : null;
        const ncrs = nx ? trueCourse(nx) : null;
        if (cur && crs != null) {
          const d = nfix && ncrs != null ? rayLineIntersect(cur, crs, nfix, ncrs) : null;
          const dd = d != null && d < 40 ? d : APPROX_FALLBACK_NM;
          end = destinationPoint(cur, crs, dd); coords = [cur, end];
          approx = true; note = "intercept point depends on where the aircraft joins the next course";
        }
        break;
      }
      case "FC": {
        if (fixLL && crs != null && l.distNm != null) { end = destinationPoint(fixLL, crs, l.distNm); coords = [cur ?? fixLL, fixLL, end].filter((p, k, a) => k === 0 || p !== a[k - 1]); }
        break;
      }
      case "FM": case "VM": {
        const from = pt === "FM" ? fixLL : cur;
        if (from && crs != null) { end = destinationPoint(from, crs, MANUAL_TERM_STUB_NM); coords = [cur ?? from, from, end]; approx = true; note = "manual termination — expect radar vectors (stub only)"; }
        break;
      }
      case "HA": case "HF": case "HM": {
        if (fixLL && crs != null) {
          const legNm = l.distNm ?? (l.timeMin ?? 1) * HOLD_SPEED_KT / 60;
          coords = [...(cur && cur !== fixLL ? [cur] : []), ...holdRacetrack(fixLL, crs, legNm, l.turn ?? "R")];
          end = fixLL; approx = true;
          note = l.distNm != null ? "holding pattern (leg length published; turn radius assumed)" : `holding pattern (${l.timeMin ?? 1} min legs at an assumed ${HOLD_SPEED_KT} kt)`;
        }
        break;
      }
      case "PI": {
        if (fixLL && crs != null) {
          const out1 = destinationPoint(fixLL, crs, l.distNm ?? 10);
          coords = [cur ?? fixLL, fixLL, out1]; end = fixLL; approx = true; note = "procedure turn — outbound leg only (turn geometry not drawn)";
        }
        break;
      }
      default:
        if (fixLL) { end = fixLL; coords = cur ? [cur, fixLL] : [fixLL]; approx = true; note = `leg type ${pt} drawn straight to its fix`; }
    }
    if (l.alt?.alt1Ft != null) curAlt = l.alt.alt1Ft;
    out.push({ leg: l, coords, approx, approxNote: note, end });
    if (end) cur = end;
  }
  return out;
}

export function holdRacetrack(fix: LL, inboundTrue: number, legNm: number, turn: "L" | "R"): LL[] {
  const outbound = (inboundTrue + 180) % 360;
  const side = turn === "R" ? 90 : -90;
  const w = Math.max(1.5, Math.min(4, legNm * 0.6)); // turn diameter, ~standard-rate at terminal speeds
  const a = fix;
  const abeam = destinationPoint(a, (inboundTrue + side + 360) % 360, w);
  const outEnd = destinationPoint(abeam, outbound, legNm);
  const inStart = destinationPoint(a, outbound, legNm);
  const c1 = destinationPoint(a, (inboundTrue + side + 360) % 360, w / 2);
  const c2 = destinationPoint(inStart, (inboundTrue + side + 360) % 360, w / 2);
  return [a, ...arcPoints(c1, a, abeam, turn, w / 2).slice(1), outEnd, ...arcPoints(c2, outEnd, inStart, turn, w / 2).slice(1), a];
}

// ── GeoJSON ─────────────────────────────────────────────────────────────────

export interface PathFeature {
  type: "Feature";
  geometry: { type: "LineString"; coordinates: Array<[number, number]> } | { type: "Point"; coordinates: [number, number] };
  properties: Record<string, string | number | boolean | null>;
}
export const PATH_MAX_FEATURES = 1500;

const r6 = (x: number) => Math.round(x * 1e6) / 1e6;
const lonlat = (p: LL): [number, number] => [r6(p.lon), r6(p.lat)];

export interface ProcedurePath {
  type: "FeatureCollection";
  features: PathFeature[];
  meta: {
    airport: string; procedure: string; kind: ProcKind; name: string;
    transitions: Array<{ id: string; role: string; routeType: string }>;
    selectedTransition: string | null;
    legs: number; approxLegs: number; unresolvedLegs: number; truncated: boolean;
    bbox: [number, number, number, number] | null;
    nasrCrossCheck: { checked: number; maxDiffNm: number; disagreeing: string[]; placedFromNasr: string[] } | null;
  };
}

/**
 * All transitions of one procedure as leg LineStrings + fix Points.
 * Segment chaining: SID runway→common→enroute, STAR enroute→common→runway,
 * IAP approach transition→final (the final's legs after the first
 * missed-approach leg are tagged `missed: true`). Every transition is drawn
 * from its own first fix; `transition` narrows the drawn set to that
 * transition plus the common/final route.
 */
export function procedurePath(idx: CifpIndex, ap: AirportData, proc: CifpProcedure, transition: string | null = null): ProcedurePath {
  const features: PathFeature[] = [];
  const fixSeen = new Map<string, PathFeature>();
  let legsN = 0, approxN = 0, unresolved = 0;
  let xChecked = 0, xMax = 0;
  const xDisagree: string[] = [];
  const fromNasr: string[] = [];
  const trs = proc.transitions.map((t) => ({ t, role: segmentRole(proc.kind, t.routeType) }));
  const isCore = (role: string) => role === "common" || role === "final";
  const drawn = trs.filter(({ t, role }) => !transition || isCore(role) || t.id === transition);
  let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
  const grow = (p: LL) => { minLon = Math.min(minLon, p.lon); maxLon = Math.max(maxLon, p.lon); minLat = Math.min(minLat, p.lat); maxLat = Math.max(maxLat, p.lat); };
  for (const { t, role } of drawn) {
    let start: LL | null = null;
    // a SID's common/enroute part starts where the runway transition ended —
    // take the first fix of this transition; runway transitions start at the runway
    if (proc.kind === "SID" && role === "runway transition") {
      const rw = ap.runways.get(t.id.startsWith("RW") ? t.id.replace(/B$/, "L") : "") ?? null;
      start = rw?.ll ?? null;
    }
    const geoms = buildLegs(t.legs, ap, idx, start, null);
    let missed = false;
    for (const g of geoms) {
      legsN++;
      if (g.leg.missedStart) missed = true;
      if (g.coords.length < 2) {
        if (g.leg.pathTerm !== "IF" && !g.coords.length) unresolved++;
      } else {
        if (g.approx) approxN++;
        if (features.length < PATH_MAX_FEATURES) {
          g.coords.forEach(grow);
          features.push({
            type: "Feature",
            geometry: { type: "LineString", coordinates: g.coords.map(lonlat) },
            properties: {
              kind: "leg", transition: t.id || null, role, routeType: t.routeType, seq: g.leg.seq,
              pathTerm: g.leg.pathTerm, fix: g.leg.fix?.ident ?? null, approx: g.approx, approxNote: g.approxNote,
              missed, selected: !transition || isCore(role) || t.id === transition,
            },
          });
        }
      }
      const primary = idx.resolvePrimary(g.leg.fix, ap);
      const fll = primary ?? idx.resolve(g.leg.fix, ap);
      if (g.leg.fix && fll) {
        const key = g.leg.fix.ident;
        const altText = g.leg.alt?.text ?? null;
        const d = g.leg.desc;
        const prev = fixSeen.get(key);
        const role2 = d[3] === "A" ? "IAF" : d[3] === "B" ? "IF" : d[3] === "F" ? "FAF" : d[3] === "M" ? "MAP" : d[3] === "I" ? "FACF" : null;
        if (!prev) {
          grow(fll);
          // independent cross-check against the NASR gazetteer (same FAA
          // source family, different product): a disagreement is surfaced
          const ext = primary && idx.external && !(g.leg.fix.section === "P" && (g.leg.fix.sub === "G" || g.leg.fix.sub === "I")) ? idx.external(key) : null;
          const diff = ext && primary ? distNm(ext, primary) : null;
          if (diff != null) { xChecked++; xMax = Math.max(xMax, diff); if (diff > NASR_AGREE_NM) xDisagree.push(key); }
          if (!primary) fromNasr.push(key);
          const f: PathFeature = {
            type: "Feature", geometry: { type: "Point", coordinates: lonlat(fll) },
            properties: {
              kind: "fix", ident: key, alt: altText, speedKt: g.leg.speedKt, role: role2, missed,
              flyover: d[1] === "Y", label: [key, altText, g.leg.speedKt ? `${g.leg.speedKt}K` : null].filter(Boolean).join(" "),
              source: primary ? "CIFP" : "NASR", nasrDiffNm: diff != null ? Math.round(diff * 1000) / 1000 : null,
              fixKind: fixKindOf(g.leg.fix),
            },
          };
          fixSeen.set(key, f);
          if (features.length < PATH_MAX_FEATURES) features.push(f);
        } else if (altText && !prev.properties.alt) {
          prev.properties.alt = altText;
          prev.properties.label = [key, altText].join(" ");
        }
      }
    }
  }
  return {
    type: "FeatureCollection",
    features,
    meta: {
      airport: ap.icao, procedure: proc.id, kind: proc.kind,
      name: proc.kind === "IAP" ? approachName(proc.id) : proc.id,
      transitions: trs.map(({ t, role }) => ({ id: t.id, role, routeType: t.routeType })),
      selectedTransition: transition,
      legs: legsN, approxLegs: approxN, unresolvedLegs: unresolved,
      truncated: features.length >= PATH_MAX_FEATURES,
      bbox: Number.isFinite(minLon) ? [r6(minLon), r6(minLat), r6(maxLon), r6(maxLat)] : null,
      nasrCrossCheck: idx.external
        ? { checked: xChecked, maxDiffNm: Math.round(xMax * 1000) / 1000, disagreeing: xDisagree, placedFromNasr: fromNasr }
        : null,
    },
  };
}

/** What a coded reference IS (drives the map symbol): VHF/NDB navaids,
 *  runway thresholds, localizers, everything else a waypoint/intersection. */
export function fixKindOf(f: FixRef): "navaid" | "runway" | "localizer" | "waypoint" {
  if (f.section === "D" || (f.section === "P" && f.sub === "N")) return "navaid";
  if (f.section === "P" && f.sub === "G") return "runway";
  if (f.section === "P" && f.sub === "I") return "localizer";
  return "waypoint";
}

/** CIFP vs NASR positions for the same fix further apart than this are
 *  reported (both are FAA products; they should agree to survey precision) */
export const NASR_AGREE_NM = 0.1;

/** Fixes a procedure references (idents -> position), for plate georeference. */
export function procedureFixes(idx: CifpIndex, ap: AirportData, proc: CifpProcedure): Map<string, LL> {
  const out = new Map<string, LL>();
  for (const t of proc.transitions) {
    for (const l of t.legs) {
      for (const f of [l.fix, l.recNav, l.center]) {
        const ll = idx.resolve(f, ap);
        if (f && ll && !out.has(f.ident)) {
          out.set(f.ident, ll);
          // localizer idents print on plates with the I- prefix ("I-VNK")
          if (f.section === "P" && f.sub === "I" && f.ident.startsWith("I")) out.set(`I-${f.ident.slice(1)}`, ll);
        }
      }
    }
  }
  return out;
}

// ── the store: download once per cycle to /tmp, index, serve ───────────────

export interface CifpStoreDeps {
  fetchImpl?: typeof fetch;
  now?: () => number;
  dir?: string;
  timeoutMs?: number;
  /** independent fix gazetteer (NASR) attached to every built index */
  external?: ((ident: string) => LL | null) | null;
}
export const CIFP_FAILURE_BACKOFF_MS = 10 * 60_000;

export class CifpStore {
  private idx: CifpIndex | null = null;
  private idxCycle: AiracCycle | null = null;
  private loading: Promise<CifpIndex> | null = null;
  private lastError: { at: number; message: string } | null = null;
  private readonly fetchImpl: typeof fetch;
  private readonly now: () => number;
  readonly dir: string;
  private readonly timeoutMs: number;
  private readonly external: ((ident: string) => LL | null) | null;

  constructor(deps: CifpStoreDeps = {}) {
    this.fetchImpl = deps.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
    this.now = deps.now ?? (() => Date.now());
    this.dir = deps.dir ?? path.join(os.tmpdir(), "voltrade_cifp");
    this.timeoutMs = deps.timeoutMs ?? 120_000;
    this.external = deps.external ?? null;
  }

  status() {
    return {
      loaded: !!this.idx, cycle: this.idxCycle,
      airports: this.idx ? this.idx.airportIds().length : 0,
      lastError: this.lastError,
    };
  }

  /** The index for the cycle in force (downloads on first use / new cycle). */
  async get(): Promise<{ idx: CifpIndex; cycle: AiracCycle }> {
    const want = cycleAt(this.now());
    if (this.idx && this.idxCycle && this.idxCycle.ident === want.ident) return { idx: this.idx, cycle: this.idxCycle };
    if (this.lastError && this.now() - this.lastError.at < CIFP_FAILURE_BACKOFF_MS && this.idx && this.idxCycle) {
      return { idx: this.idx, cycle: this.idxCycle }; // serve the previous cycle while backing off
    }
    if (this.lastError && this.now() - this.lastError.at < CIFP_FAILURE_BACKOFF_MS && !this.idx) {
      throw new Error(`CIFP unavailable: ${this.lastError.message}`);
    }
    if (!this.loading) {
      this.loading = this.load(want).finally(() => { this.loading = null; });
    }
    const idx = await this.loading;
    return { idx, cycle: this.idxCycle! };
  }

  private async load(want: AiracCycle): Promise<CifpIndex> {
    const errors: string[] = [];
    for (const c of [want, cycleAt(want.effectiveMs, -1)]) {
      const file = path.join(this.dir, c.ident, "FAACIFP18");
      try {
        if (!fs.existsSync(file)) await this.download(c, file);
        const text = await fs.promises.readFile(file);
        const idx = CifpIndex.build(text, (s, e) => readRange(file, s, e), c.ident);
        idx.external = this.external;
        this.idx = idx; this.idxCycle = c;
        // serving the PREVIOUS cycle: keep the current cycle's failure so the
        // backoff stops every request from re-hammering the FAA
        this.lastError = errors.length ? { at: this.now(), message: `serving ${c.ident}; ${errors.join("; ")}` } : null;
        this.pruneOld(c.ident);
        return idx;
      } catch (e: unknown) {
        errors.push(`${c.ident}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    this.lastError = { at: this.now(), message: errors.join("; ") };
    console.error(`[cifp] load failed — ${this.lastError.message}`);
    if (this.idx) return this.idx;
    throw new Error(`CIFP unavailable: ${this.lastError.message}`);
  }

  private async download(c: AiracCycle, file: string): Promise<void> {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), this.timeoutMs);
    try {
      const r = await this.fetchImpl(cifpZipUrl(c), { signal: ac.signal, headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" } });
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${cifpZipUrl(c)}`);
      const zip = Buffer.from(await r.arrayBuffer());
      const ent = extractZipEntry(zip, (n) => /(^|\/)FAACIFP18$/.test(n));
      if (!ent) throw new Error("FAACIFP18 not found in the CIFP zip");
      await fs.promises.mkdir(path.dirname(file), { recursive: true });
      const tmp = `${file}.part`;
      await fs.promises.writeFile(tmp, ent.data);
      await fs.promises.rename(tmp, file);
    } finally {
      clearTimeout(timer);
    }
  }

  private pruneOld(keep: string): void {
    let names: string[] = [];
    try { names = fs.readdirSync(this.dir); } catch (e: unknown) { console.warn("[cifp] prune list:", e instanceof Error ? e.message : e); return; }
    for (const n of names) {
      if (n === keep || !/^\d{4}$/.test(n)) continue;
      try { fs.rmSync(path.join(this.dir, n), { recursive: true, force: true }); } catch (e: unknown) { console.warn("[cifp] prune:", e instanceof Error ? e.message : e); }
    }
  }

  /** test hook */
  _setIndex(idx: CifpIndex, cycle: AiracCycle): void { this.idx = idx; this.idxCycle = cycle; }
}

function readRange(file: string, start: number, end: number): string {
  const fd = fs.openSync(file, "r");
  try {
    const b = Buffer.alloc(end - start);
    fs.readSync(fd, b, 0, b.length, start);
    return b.toString("latin1");
  } finally {
    fs.closeSync(fd);
  }
}
