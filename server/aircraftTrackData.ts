// aircraftTrackData.ts — the aircraft TRACK and FLEET data the mobile app
// reads (owner request 2026-11-16: "flown track at true altitude, with
// altitude, speed and time at each point; vertical rate, ground speed and
// heading for each position; fleet history: all past flights for saved tail
// numbers from the archive, with live ADS-B data on top"). Data layer only;
// registered from routes.ts with ONE line. Contract for the app session:
// docs/aircraft_app_data_contract.md.
//
//   GET /api/data/aircraft/track/:hex?from=&to=&max=
//   GET /api/data/aircraft/fleet?tails=N843S,N123AB&hexes=ab8c8e
//
// WHERE THE VALUES COME FROM (nothing here is fetched from a provider):
//   - the position archive's hour files on the LOCAL disk (datacoreArchive):
//     t/la/lo/al/v/h on every line; since 2026-10-14 the `x` block (geometric
//     and barometric altitude in ft, both broadcast vertical rates in ft/min,
//     roll when the aircraft sent it); since 2026-11-16 the flat ag/vr/vs/ak/ft
//     keys on the lines that carry no x (schema_version 5);
//   - the worldwide live snapshot + intent store (globalAircraft) for the
//     live fix on top;
//   - since 2026-11-17, for a TRACKED tail, its permanent per-plane log
//     (trackedPlaneLog.ts) for the part of a window that is older than the
//     oldest hour file on the local disk. Those points say origin tracked_log.
//
// HONESTY RULES this module holds (tests: aircraftTrackData.test.ts):
//   - `t` is the FIX'S time. Most archive lines are stamped with OUR receive
//     time; the feed's position age (`sp`) is subtracted and the point says
//     so (t_src). A line with no age recorded keeps its stamp, labelled.
//   - alt_m is the GEOMETRIC (GNSS) altitude where the aircraft broadcast
//     one — metres above the WGS84 ellipsoid, NOT above mean sea level — and
//     otherwise the BAROMETRIC pressure altitude (1013.25 hPa, uncorrected).
//     alt_kind says which on every point; "unrecorded" where the archive
//     line cannot tell. Nothing is ever called "true altitude".
//   - a vertical rate is "broadcast" (the aircraft's own value) or "derived"
//     (computed here from two consecutive fixes, with their spacing) — never
//     unlabelled, never derived across a gap or between two kinds of altitude.
//   - track_deg is the track over the ground (archive `h`), not the nose
//     heading. Pitch is never broadcast. Bank is in roll_deg only where the
//     aircraft sent it; the client derives the rest and must label it.
//   - a gap in coverage is LISTED and never bridged: no point is invented.
//   - history reaches only as far back as the local raw hour files do; the
//     response states the oldest hour on disk and the retention limit.
//   - FAA/SWIM data is not read here (LADD): every field is community ADS-B.
//     The one FAA input is the public registry's N-number/Mode-S pairing,
//     used only to resolve a tail number to a hex, and named when used.

import fs from "fs";
import path from "path";
import type { Express } from "express";
import { archiveBaseDir, oldestRawHour, RAW_RETENTION_DAYS, streamJsonlLines } from "./datacoreArchive";
import { readTrackedLogRows, trackedLogSummary, EMPTY_LOG_SUMMARY, type PlaneLogSummary, type TrackedLogReadStats } from "./trackedPlaneLog";
import { scanArchiveRows, splitTrips, type ArchivedFix, type ArchiveScanStats, type NearestAirportFn, type Trip } from "./aircraftTrips";
import { IntentArchiveDecoder, type IntentValues } from "./aircraftIntent";
import { aircraftLiveLookup, type AircraftLiveLookup, type AircraftLiveState } from "./globalAircraft";
import { nearestAirport } from "./airportsIndex";
import { loadEntitySpine } from "./aircraftEntities";
import { normalizeReg, TRACKED_CAP, type TrackedPlane } from "./trackedPlanes";
import { HOT_TIER_FLOOR_DAYS, effectiveRetentionDays, retentionConfig } from "./archiveOffload";
import { r2ConfigFromEnv } from "./r2Client";
import { registerHeapGauge, trackMap } from "./heapDiag";

// ── limits (every one is stated in the responses) ───────────────────────────
/** default window when no from/to is given */
export const TRACK_DEFAULT_WINDOW_S = 24 * 3600;
/** widest window one track request may ask for (at most 49 hour files read) */
export const TRACK_MAX_WINDOW_S = 48 * 3600;
export const TRACK_DEFAULT_MAX_POINTS = 2_000;
export const TRACK_MAX_POINTS = 5_000;
export const TRACK_MIN_POINTS = 2;
/** consecutive fixes further apart than this are a GAP (the archive's longest
 *  thinning cadence is 5 min, on the ground; airborne it is 75 s) */
export const TRACK_GAP_S = 300;
/** a vertical rate is derived only from two fixes at most this far apart */
export const DERIVED_VR_MAX_DT_S = 120;
/** a receive-stamped line may be stamped this much later than its fix */
export const RECEIVE_SKEW_MAX_S = 600;
/** fixes of one aircraft held in memory per request (newest kept) */
export const TRACK_SCAN_CAP = 50_000;
/** fleet: fixes held per aircraft / per request while flights are split
 *  (31 bytes each in column buffers; the newest are kept, the rest counted) */
export const FLEET_SCAN_CAP_PER_HEX = 250_000;
export const FLEET_SCAN_CAP_TOTAL = 1_000_000;
/** aircraft per fleet request — the tracked list's own cap */
export const FLEET_MAX_AIRCRAFT = TRACKED_CAP;
/** flights listed per aircraft (newest first); the total is always stated */
export const FLEET_MAX_FLIGHTS = 200;
export const FLEET_CACHE_MS = 5 * 60_000;
export const TRACK_CACHE_MS = 30_000;
/** fixes held across ALL cached track reads (about 200 bytes each); entries
 *  past their 30 s are dropped on the next request, the oldest first above this */
export const TRACK_CACHE_MAX_FIXES = 100_000;
/** a live fix older than this is a last known position, not "live" */
export const LIVE_FRESH_S = 120;
/** ... and one older than this is not served as live at all (snapshot rule) */
export const LIVE_MAX_AGE_S = 600;
/** tail -> hex by the archive's `rg` field: newest hour files only */
export const REG_SCAN_MAX_FILES = 72;
export const REG_SCAN_MAX_BYTES = 64 * 1024 * 1024;
const REG_FOUND_TTL_MS = 6 * 3600_000;
const REG_MISS_TTL_MS = 15 * 60_000;

const HEX_RE = /^[0-9a-f]{6}$/;
const FT_TO_M = 0.3048;
const KT_TO_MS = 0.5144; // the pipeline's own factor (aircraftTiling.mapPointAircraft)

// ── one archive line -> one fix ─────────────────────────────────────────────
export type AltKind = "geometric" | "barometric" | "unrecorded";
export type TimeSrc = "fix" | "receive_minus_age" | "unadjusted";
export type RateKind = "barometric" | "geometric";

export interface TrackFix {
  /** fix time, epoch s (see t_src) */
  t: number;
  t_src: TimeSrc;
  la: number;
  lo: number;
  /** the archive's `al`, m: barometric when the aircraft sent one, else geometric */
  al: number | null;
  alt_geom_m: number | null;
  alt_baro_m: number | null;
  g: boolean;
  /** ground speed m/s (archive `v`, whole m/s) */
  gs: number | null;
  /** track over ground, deg (archive `h`, whole deg) */
  trk: number | null;
  /** BROADCAST vertical rate m/s (0.1); null when the line has none */
  vr: number | null;
  vr_kind: RateKind | null;
  /** broadcast roll, deg (+ right wing down), when the aircraft sent it */
  roll: number | null;
  c: string | null;
  rg: string | null;
  pv: string | null;
  pt: string | null;
  /** set on a fix read from the tracked tails' permanent log (trackedPlaneLog.ts) */
  store?: "tracked_log";
}

const num = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
const str = (x: unknown): string | null => (typeof x === "string" && x ? x : null);
const fpmToMs = (fpm: number): number => Math.round(fpm * 0.0508) / 10; // 1 ft/min = 0.00508 m/s

/** Pure: one parsed archive line (+ its decoded `x` state, or null) -> a fix.
 *  null when the line has no usable time or position. */
export function fixFromLine(row: Record<string, unknown>, x: IntentValues | null): TrackFix | null {
  const t0 = num(row.t), la = num(row.la), lo = num(row.lo);
  if (t0 == null || la == null || lo == null) return null;
  const sp = num(row.sp);
  let t = Math.floor(t0);
  let tSrc: TimeSrc = "unadjusted";
  if (row.ft === 1) tSrc = "fix";
  else if (sp != null && sp >= 0) { t = Math.round(t0 - sp); tSrc = "receive_minus_age"; }
  const al = num(row.al);
  let geom: number | null = null;
  let baro: number | null = null;
  let vr: number | null = null;
  let vrKind: RateKind | null = null;
  if (x) {
    if (x.alt_geom != null) geom = Math.round(x.alt_geom * FT_TO_M);
    // `al` IS the barometric altitude whenever the aircraft sent one
    if (x.alt_baro != null) baro = al ?? Math.round(x.alt_baro * FT_TO_M);
    if (x.baro_rate != null) { vr = fpmToMs(x.baro_rate); vrKind = "barometric"; }
    else if (x.geom_rate != null) { vr = fpmToMs(x.geom_rate); vrKind = "geometric"; }
  }
  // flat keys (schema_version 5): the record on lines that carry no x
  const ag = num(row.ag);
  if (geom == null && ag != null) geom = ag;
  if (baro == null && row.ak === "b" && al != null) baro = al;
  if (geom == null && row.ak === "g" && al != null) geom = al;
  if (vr == null) {
    const flat = num(row.vr);
    if (flat != null) { vr = flat; vrKind = row.vs === "b" ? "barometric" : row.vs === "g" ? "geometric" : null; }
  }
  return {
    t, t_src: tSrc, la, lo, al, alt_geom_m: geom, alt_baro_m: baro, g: !!row.g,
    gs: num(row.v), trk: num(row.h), vr, vr_kind: vrKind, roll: x ? x.roll : null,
    c: str(row.c), rg: str(row.rg), pv: str(row.pv), pt: str(row.pt),
  };
}

/** alt_m and which altitude it is: geometric where broadcast, else barometric,
 *  else the archive's unlabelled `al`. */
export function bestAltitude(f: TrackFix): { alt_m: number | null; alt_kind: AltKind | null } {
  if (f.alt_geom_m != null) return { alt_m: f.alt_geom_m, alt_kind: "geometric" };
  if (f.alt_baro_m != null) return { alt_m: f.alt_baro_m, alt_kind: "barometric" };
  if (f.al != null) return { alt_m: f.al, alt_kind: "unrecorded" };
  return { alt_m: null, alt_kind: null };
}

const fixScore = (f: TrackFix): number =>
  (f.al != null || f.alt_geom_m != null ? 4 : 0) + (f.t_src === "fix" ? 2 : f.t_src === "receive_minus_age" ? 1 : 0) + (f.vr != null ? 1 : 0);
const lightScore = (f: ArchivedFix): number => (f.al != null ? 1 : 0);

/** Pure: time-sort, then drop repeats of the same physical fix. Two lines are
 *  the same fix when they share a second (the better-recorded one is kept), or
 *  when position and altitude are identical within 2 s (one upstream fix
 *  written twice — e.g. stamped by two paths — rounds to neighbouring seconds). */
export function cleanFixes<T extends ArchivedFix>(
  fixes: T[], score: (f: T) => number,
): { fixes: T[]; duplicates_dropped: number } {
  const sorted = fixes.slice().sort((a, b) => a.t - b.t);
  const out: T[] = [];
  let dropped = 0;
  for (const f of sorted) {
    const last = out[out.length - 1];
    if (last && last.t === f.t) {
      if (score(f) > score(last)) out[out.length - 1] = f;
      dropped++;
      continue;
    }
    if (last && f.t - last.t <= 2 && f.la === last.la && f.lo === last.lo && (f.al ?? null) === (last.al ?? null) && !!f.g === !!last.g) {
      dropped++;
      continue;
    }
    out.push(f);
  }
  return { fixes: out, duplicates_dropped: dropped };
}

export interface HexRead {
  fixes: TrackFix[];
  duplicates_dropped: number;
  /** oldest fixes dropped because the per-request cap was hit */
  dropped_over_cap: number;
}

/** What the tracked tails' permanent log added to one track read. */
export interface TrackedLogPart {
  /** was the log read for this window? Only when the window starts before the
   *  oldest hour file on the local disk (or there is none) and a log exists */
  read: boolean;
  why: string;
  /** fixes in the response that came from the log (after de-duplication) */
  fixes_from_log: number;
  months: TrackedLogReadStats | null;
}

/** Pure: the hour files' fixes plus the permanent log's, as one cleaned series.
 *  The hour-file fixes go first, so where both hold the same second the
 *  better-recorded one is kept and a tie stays with the hour file. */
export function mergeTrackedLogFixes(hourFixes: TrackFix[], logFixes: TrackFix[], cap: number): HexRead {
  const c = cleanFixes([...hourFixes, ...logFixes], fixScore);
  const over = Math.max(0, c.fixes.length - cap);
  return { fixes: over ? c.fixes.slice(-cap) : c.fixes, duplicates_dropped: c.duplicates_dropped, dropped_over_cap: over };
}

/** One aircraft's permanent-log fixes in a window (newest `cap` kept). */
export async function readTrackedLogFixes(
  baseDir: string, hex: string, o: { fromSec: number; toSec: number; cap: number },
): Promise<{ fixes: TrackFix[]; dropped_over_cap: number; months: TrackedLogReadStats }> {
  const fixes: TrackFix[] = [];
  let over = 0;
  const months = await readTrackedLogRows(baseDir, hex, { fromSec: o.fromSec, toSec: o.toSec, slackSec: RECEIVE_SKEW_MAX_S }, (row) => {
    const f = fixFromLine(row, null);
    if (!f || f.t < o.fromSec || f.t > o.toSec) return;
    f.store = "tracked_log";
    fixes.push(f);
    if (fixes.length >= 2 * o.cap) { over += fixes.length - o.cap; fixes.splice(0, fixes.length - o.cap); } // months stream oldest first
  });
  return { fixes, dropped_over_cap: over, months };
}

/** ONE streamed pass over the local hour files for these hexes. Times are the
 *  fixes' own (see fixFromLine); the window is applied to those. */
export async function readHexFixes(
  hexes: readonly string[], baseDir: string,
  opts: { fromSec?: number; toSec?: number; decodeIntent: boolean; capPerHex: number },
): Promise<{ byHex: Map<string, HexRead>; scan: ArchiveScanStats }> {
  const acc = new Map<string, { fixes: TrackFix[]; over: number }>();
  for (const h of hexes) acc.set(h, { fixes: [], over: 0 });
  let curFile = "";
  let dec = new IntentArchiveDecoder();
  const scan = await scanArchiveRows("aircraft", hexes, baseDir, {
    fromSec: opts.fromSec,
    // a receive-stamped line can be stamped later than its fix
    toSec: opts.toSec == null ? undefined : opts.toSec + RECEIVE_SKEW_MAX_S,
  }, (row, file) => {
    if (file !== curFile) { curFile = file; dec = new IntentArchiveDecoder(); } // x chains are per hour file
    // every line of the aircraft feeds the decoder, in or out of the window
    const x = opts.decodeIntent ? dec.feed(row) : null;
    const f = fixFromLine(row, x);
    if (!f) return;
    if (opts.fromSec != null && f.t < opts.fromSec) return;
    if (opts.toSec != null && f.t > opts.toSec) return;
    const a = acc.get(String(row.i));
    if (!a) return;
    a.fixes.push(f);
    if (a.fixes.length >= 2 * opts.capPerHex) { // files stream oldest first: drop the oldest half
      a.over += a.fixes.length - opts.capPerHex;
      a.fixes.splice(0, a.fixes.length - opts.capPerHex);
    }
  });
  const byHex = new Map<string, HexRead>();
  for (const [h, a] of acc) {
    const c = cleanFixes(a.fixes, fixScore);
    let over = a.over;
    let fixes = c.fixes;
    if (fixes.length > opts.capPerHex) { over += fixes.length - opts.capPerHex; fixes = fixes.slice(-opts.capPerHex); }
    byHex.set(h, { fixes, duplicates_dropped: c.duplicates_dropped, dropped_over_cap: over });
  }
  return { byHex, scan };
}

// ── fleet read: every local hour file, many aircraft, bounded memory ────────
/** Growable column buffers for one aircraft's fixes (31 bytes a fix, where an
 *  object per fix costs about 200). At the cap the OLDEST quarter is dropped
 *  and counted — hour files stream oldest first. */
class FixColumns {
  n = 0;
  dropped = 0;
  private t = new Float64Array(256);
  private la = new Float64Array(256);
  private lo = new Float64Array(256);
  private al = new Int32Array(256);
  private g = new Uint8Array(256);
  private c = new Uint16Array(256);
  constructor(private readonly cap: number) {}

  push(t: number, la: number, lo: number, al: number | null, g: boolean, c: number): void {
    if (this.n >= this.cap) {
      const drop = Math.max(1, this.cap >> 2);
      for (const a of [this.t, this.la, this.lo, this.al, this.g, this.c]) a.copyWithin(0, drop, this.n);
      this.n -= drop;
      this.dropped += drop;
    }
    if (this.n >= this.t.length) {
      const size = Math.min(this.cap, this.t.length * 2);
      const grow = <A extends Float64Array | Int32Array | Uint8Array | Uint16Array>(old: A, fresh: A): A => { fresh.set(old); return fresh; };
      this.t = grow(this.t, new Float64Array(size));
      this.la = grow(this.la, new Float64Array(size));
      this.lo = grow(this.lo, new Float64Array(size));
      this.al = grow(this.al, new Int32Array(size));
      this.g = grow(this.g, new Uint8Array(size));
      this.c = grow(this.c, new Uint16Array(size));
    }
    const i = this.n++;
    this.t[i] = t; this.la[i] = la; this.lo[i] = lo;
    this.al[i] = al == null ? NO_ALT : Math.round(al);
    this.g[i] = g ? 1 : 0;
    this.c[i] = c;
  }

  fixes(callsigns: string[]): ArchivedFix[] {
    const out: ArchivedFix[] = new Array(this.n);
    for (let i = 0; i < this.n; i++) {
      out[i] = {
        t: this.t[i], la: this.la[i], lo: this.lo[i],
        al: this.al[i] === NO_ALT ? null : this.al[i],
        c: this.c[i] ? callsigns[this.c[i]] : null, g: this.g[i] === 1,
      };
    }
    return out;
  }
}
const NO_ALT = -(2 ** 31); // Int32 sentinel: no altitude on this fix

/** ONE streamed pass over ALL local hour files for these hexes, reduced to
 *  each aircraft's flights. Fix times are the fixes' own (fixFromLine). */
export async function readFleetHistories(
  hexes: readonly string[], baseDir: string, nearest?: NearestAirportFn,
  caps: { perHex?: number; total?: number } = {},
): Promise<{ byHex: Map<string, AircraftHistory>; scan: ArchiveScanStats }> {
  const perHex = Math.max(1, Math.min(caps.perHex ?? FLEET_SCAN_CAP_PER_HEX,
    Math.floor((caps.total ?? FLEET_SCAN_CAP_TOTAL) / Math.max(1, hexes.length))));
  const cols = new Map<string, { col: FixColumns; regs: Set<string> }>();
  for (const h of hexes) cols.set(h, { col: new FixColumns(perHex), regs: new Set() });
  const callsigns: string[] = [""]; // index 0 = none
  const callIdx = new Map<string, number>();
  const push = (row: Record<string, unknown>): boolean => {
    const f = fixFromLine(row, null);
    const a = cols.get(String(row.i));
    if (!f || !a) return false;
    let ci = 0;
    if (f.c) {
      ci = callIdx.get(f.c) ?? 0;
      if (!ci && callsigns.length < 65_535) { ci = callsigns.length; callsigns.push(f.c); callIdx.set(f.c, ci); }
    }
    if (f.rg && a.regs.size < 5) a.regs.add(f.rg);
    a.col.push(f.t, f.la, f.lo, f.al, f.g, ci);
    return true;
  };
  // FIRST the tracked tails' permanent log, and only its part OLDER than the
  // oldest local hour file (all of it when there is none): the fixes reach
  // the column buffers oldest first, which is what their cap assumes
  const oldestMs = oldestRawHour("aircraft", baseDir);
  const beforeSec = oldestMs == null ? undefined : oldestMs / 1000 - 1;
  const fromLog = new Map<string, number>();
  for (const h of hexes) {
    let n = 0;
    await readTrackedLogRows(baseDir, h, { toSec: beforeSec, slackSec: 0 }, (row) => { if (push(row)) n++; });
    fromLog.set(h, n);
  }
  const scan = await scanArchiveRows("aircraft", hexes, baseDir, {}, (row) => { push(row); });
  const byHex = new Map<string, AircraftHistory>();
  for (const [h, a] of cols) {
    // one aircraft's fixes as objects at a time: sort, de-duplicate, split
    const c = cleanFixes(a.col.fixes(callsigns), lightScore);
    byHex.set(h, historyFromFixes(c.fixes, {
      dropped_over_cap: a.col.dropped, duplicates_dropped: c.duplicates_dropped, regs: Array.from(a.regs),
      from_tracked_log: fromLog.get(h) ?? 0,
    }, nearest));
  }
  return { byHex, scan };
}

// ── track assembly (pure) ───────────────────────────────────────────────────
export const TRACK_FIELDS = [
  "t", "lat", "lon", "alt_m", "alt_kind", "alt_geom_m", "alt_baro_m", "gs_ms", "track_deg",
  "vr_ms", "vr_source", "vr_kind", "vr_dt_s", "on_ground", "roll_deg", "t_src", "src", "pos_type", "origin",
] as const;

export const TRACK_UNITS: Record<(typeof TRACK_FIELDS)[number], string> = {
  t: "epoch seconds (UTC): the time of the fix, see t_src",
  lat: "degrees, 4 decimals (about 11 m)",
  lon: "degrees, 4 decimals",
  alt_m: "metres, whole: alt_geom_m where the aircraft broadcast a geometric altitude, else alt_baro_m, else the archive's unlabelled altitude; null on the ground or unknown. alt_kind says which",
  alt_kind: "geometric | barometric | unrecorded | null",
  alt_geom_m: "metres above the WGS84 ellipsoid (GNSS height; NOT above mean sea level: the two differ by the local geoid height, about -105 to +85 m), as broadcast; null when not broadcast or not recorded",
  alt_baro_m: "metres, pressure altitude on the 1013.25 hPa standard setting, as broadcast, not corrected for local pressure or temperature; null when not broadcast or not recorded",
  gs_ms: "ground speed, metres per second, whole (broadcast knots, converted; about +-0.5 m/s)",
  track_deg: "track over the ground, degrees true, whole. NOT the nose heading",
  vr_ms: "vertical rate, metres per second, 0.1, + climbing; null when neither broadcast nor derivable",
  vr_source: "broadcast | derived | null",
  vr_kind: "barometric | geometric | null: which rate was broadcast, or which altitude a derived rate was computed from (null = not recorded)",
  vr_dt_s: "derived rates only: seconds between the two fixes the rate was computed from",
  on_ground: "boolean, as broadcast",
  roll_deg: "bank angle, degrees, + right wing down, ONLY where the aircraft broadcast it (a minority do); null otherwise",
  t_src: "fix | receive_minus_age | unadjusted",
  src: "provider that served the fix: adsblol | adsbfi | airplaneslive | opensky | null (not recorded)",
  pos_type: "the feed's position source: adsb_icao, adsr_icao, mlat (ground-computed), tisb_* (ground-derived), mode_s ...; null = not recorded",
  origin: "archive | tracked_log | live",
};

export const TRACK_ENUMS = {
  alt_kind: {
    geometric: "GNSS height above the WGS84 ellipsoid, broadcast by the aircraft",
    barometric: "pressure altitude, broadcast by the aircraft",
    unrecorded: "the archive line does not say which: it is the barometric altitude when the aircraft sent one, else its geometric altitude",
  },
  vr_source: {
    broadcast: "the aircraft's own vertical rate (ft/min, converted)",
    derived: "computed here: (altitude of this fix - altitude of the previous fix) / vr_dt_s, both the same kind of altitude; an average over vr_dt_s, from altitudes rounded to 1 m and times to 1 s",
  },
  t_src: {
    fix: "the archive line was written at the fix's own time (provider clock minus position age)",
    receive_minus_age: "the line was stamped when we received it; the feed's position age was subtracted (good to about 1-2 s)",
    unadjusted: "the line's stamp as written: no position age was recorded, so it may be later than the fix",
  },
  origin: {
    archive: "a line of our position archive (the shared hour files)",
    tracked_log: "a line of the permanent per-plane log kept for tails on the server's tracked list; read only for the part of a window that is older than the oldest hour file on the local disk",
    live: "the newest fix in the worldwide live snapshot, newer than the last archived line",
  },
} as const;

export const TRACK_ATTITUDE_NOTE = {
  pitch: "NOT in the data: no aircraft broadcasts pitch. Derive it on the client from vr_ms and gs_ms (flight path angle = atan(vr_ms / gs_ms)); that is the flight path angle, not the nose pitch, and must be shown as an estimate.",
  bank: "In the data ONLY where roll_deg is not null (the aircraft answered a Mode S enhanced-surveillance request; most rows are null). Otherwise derive it on the client from the change of track_deg over time and gs_ms (coordinated turn: bank = atan(gs_ms * turn_rate_rad_s / 9.81)) and show it as an estimate.",
  heading: "track_deg is the direction of travel over the ground. The nose heading differs from it by the wind correction angle and is not in this response.",
} as const;

export type TrackRow = [
  number, number, number, number | null, AltKind | null, number | null, number | null, number | null, number | null,
  number | null, "broadcast" | "derived" | null, RateKind | null, number | null, boolean, number | null, TimeSrc,
  string | null, string | null, "archive" | "tracked_log" | "live",
];

export interface TrackGap { from_t: number; to_t: number; seconds: number }

export interface BuiltTrack {
  rows: TrackRow[];
  gaps: TrackGap[];
  points_total: number;
  decimation: {
    applied: boolean; points_total: number; points_returned: number; max: number;
    stride: number | null; gap_endpoints_kept: boolean; method: string;
  };
  counts: {
    vr_broadcast: number; vr_derived: number; vr_none: number;
    alt_geometric: number; alt_barometric: number; alt_unrecorded: number; alt_none: number;
    roll_broadcast: number;
  };
}

/** The altitude pair a rate may be derived from: both geometric, or both
 *  barometric, or both unlabelled — never one of each. */
function sameKindAltitudes(a: TrackFix, b: TrackFix): { a: number; b: number; kind: RateKind | null } | null {
  if (a.alt_geom_m != null && b.alt_geom_m != null) return { a: a.alt_geom_m, b: b.alt_geom_m, kind: "geometric" };
  if (a.alt_baro_m != null && b.alt_baro_m != null) return { a: a.alt_baro_m, b: b.alt_baro_m, kind: "barometric" };
  const unlabelled = (f: TrackFix) => f.alt_geom_m == null && f.alt_baro_m == null && f.al != null;
  if (unlabelled(a) && unlabelled(b)) return { a: a.al as number, b: b.al as number, kind: null };
  return null;
}

/** Pure: cleaned, time-ordered fixes (+ which of them is the live one) -> the
 *  wire rows, the gaps, and a stated decimation. Rates and gaps are computed on
 *  the FULL series, before any thinning. */
export function buildTrack(
  fixes: TrackFix[],
  o: { max: number; gapSec?: number; liveIndex?: number | null },
): BuiltTrack {
  const gapSec = o.gapSec ?? TRACK_GAP_S;
  const n = fixes.length;
  const counts = { vr_broadcast: 0, vr_derived: 0, vr_none: 0, alt_geometric: 0, alt_barometric: 0, alt_unrecorded: 0, alt_none: 0, roll_broadcast: 0 };
  const rows: TrackRow[] = [];
  const gaps: TrackGap[] = [];
  const mustKeep = new Set<number>();
  if (n) { mustKeep.add(0); mustKeep.add(n - 1); }
  for (let i = 0; i < n; i++) {
    const f = fixes[i];
    const prev = i > 0 ? fixes[i - 1] : null;
    const dt = prev ? f.t - prev.t : 0;
    if (prev && dt > gapSec) {
      gaps.push({ from_t: prev.t, to_t: f.t, seconds: dt });
      mustKeep.add(i - 1);
      mustKeep.add(i);
    }
    let vr = f.vr;
    let vrSource: "broadcast" | "derived" | null = vr != null ? "broadcast" : null;
    let vrKind = f.vr_kind;
    let vrDt: number | null = null;
    if (vr == null && prev && dt > 0 && dt <= DERIVED_VR_MAX_DT_S && !f.g && !prev.g) {
      const pair = sameKindAltitudes(prev, f);
      if (pair) {
        vr = Math.round(((pair.b - pair.a) / dt) * 10) / 10;
        vrSource = "derived";
        vrKind = pair.kind;
        vrDt = dt;
      }
    }
    if (vrSource === "broadcast") counts.vr_broadcast++;
    else if (vrSource === "derived") counts.vr_derived++;
    else counts.vr_none++;
    const alt = bestAltitude(f);
    if (alt.alt_kind === "geometric") counts.alt_geometric++;
    else if (alt.alt_kind === "barometric") counts.alt_barometric++;
    else if (alt.alt_kind === "unrecorded") counts.alt_unrecorded++;
    else counts.alt_none++;
    if (f.roll != null) counts.roll_broadcast++;
    rows.push([
      f.t, f.la, f.lo, alt.alt_m, alt.alt_kind, f.alt_geom_m, f.alt_baro_m, f.gs, f.trk,
      vr, vrSource, vrKind, vrDt, f.g, f.roll, f.t_src, f.pv, f.pt,
      o.liveIndex === i ? "live" : f.store === "tracked_log" ? "tracked_log" : "archive",
    ]);
  }
  const max = o.max;
  if (n <= max) {
    return {
      rows, gaps, points_total: n, counts,
      decimation: { applied: false, points_total: n, points_returned: n, max, stride: null, gap_endpoints_kept: true, method: "none: every fix in the window is returned" },
    };
  }
  // thinning: every stride-th fix, plus (when they fit) the first and last fix
  // and both sides of every gap, so a listed gap always has its two end points
  const endpointsFit = mustKeep.size < max;
  const budget = endpointsFit ? max - mustKeep.size : max;
  const stride = Math.ceil(n / budget);
  const kept: TrackRow[] = [];
  for (let i = 0; i < n; i++) {
    if (i % stride === 0 || (endpointsFit && mustKeep.has(i))) kept.push(rows[i]);
  }
  return {
    rows: kept, gaps, points_total: n, counts,
    decimation: {
      applied: true, points_total: n, points_returned: kept.length, max, stride, gap_endpoints_kept: endpointsFit,
      method: endpointsFit
        ? `every ${stride}th fix in time order is returned, plus the first fix, the last fix and the fix on each side of every listed gap; the dropped fixes are real fixes that are simply not sent. vr_ms values were computed before thinning and are unchanged by it`
        : `every ${stride}th fix in time order is returned; there are more gap end points than max, so they are NOT all kept — read the gaps list, not the spacing of the points`,
    },
  };
}

// ── live fix on top ─────────────────────────────────────────────────────────
/** The worldwide snapshot's row for this aircraft as a fix. The row carries
 *  one altitude (barometric first); the geometric altitude, the vertical rate
 *  and roll come from the intent store and are used only while BOTH the fix
 *  and those values are fresh (the store serves values under 30 s old). */
export function liveFix(s: AircraftLiveState, nowMs: number): TrackFix {
  const r = s.row;
  const fresh = nowMs - r.seenAt <= 30_000;
  const v: Partial<IntentValues> = fresh && s.intent ? s.intent.values : {};
  const al = r.altFt == null ? null : Math.round(r.altFt * FT_TO_M);
  const geom = v.alt_geom != null ? Math.round(v.alt_geom * FT_TO_M) : null;
  const baro = v.alt_baro != null && al != null ? al : null;
  let vr: number | null = null;
  let vrKind: RateKind | null = null;
  if (v.baro_rate != null) { vr = fpmToMs(v.baro_rate); vrKind = "barometric"; }
  else if (v.geom_rate != null) { vr = fpmToMs(v.geom_rate); vrKind = "geometric"; }
  return {
    t: Math.round(r.seenAt / 1000), t_src: "fix", la: r.lat, lo: r.lon, al,
    alt_geom_m: geom, alt_baro_m: baro, g: r.gnd,
    gs: r.gsKt == null ? null : Math.round(r.gsKt * KT_TO_MS),
    trk: r.trk == null ? null : Math.round(r.trk),
    vr, vr_kind: vrKind, roll: v.roll ?? null,
    c: r.callsign, rg: r.reg, pv: r.src, pt: null,
  };
}

// ── how far back the archive really goes ────────────────────────────────────
export interface ArchiveReach {
  tier_read: "local";
  /** start of the oldest aircraft hour file on the local disk (epoch s); null = none */
  local_oldest_hour_t: number | null;
  /** raw fixes are deleted everywhere after this many days */
  raw_retention_days: number;
  /** with the cold tier on, local days older than this may already be evicted */
  local_floor_days: number;
  cold_tier_configured: boolean;
  note: string;
}

const HOUR_FILE_RE = /^(\d{4}-\d{2}-\d{2})-(\d{2})\.jsonl(\.gz)?$/;

export function archiveReach(baseDir: string, env: NodeJS.ProcessEnv = process.env): ArchiveReach {
  let oldest: number | null = null;
  try {
    for (const f of fs.readdirSync(path.join(baseDir, "aircraft")).sort()) {
      const m = HOUR_FILE_RE.exec(f);
      if (!m) continue;
      const ms = Date.parse(`${m[1]}T${m[2]}:00:00Z`);
      if (Number.isFinite(ms)) { oldest = ms / 1000; break; }
    }
  } catch (e) {
    oldest = null; // no aircraft directory yet: nothing is held, and the response says so
  }
  const cold = r2ConfigFromEnv(env) !== null;
  const days = effectiveRetentionDays(cold, retentionConfig(env));
  const floor = cold ? HOT_TIER_FLOOR_DAYS : RAW_RETENTION_DAYS;
  return {
    tier_read: "local",
    local_oldest_hour_t: oldest,
    raw_retention_days: days,
    local_floor_days: floor,
    cold_tier_configured: cold,
    note: `This endpoint reads the raw hour files on the server's local disk only. Raw fixes are kept ${days} days in total and then deleted everywhere`
      + (cold ? `; days older than about ${floor} days may already have been moved to the cold tier (R2), which this endpoint does NOT read (GET /api/data/aircraft/window reads it, by area and time)` : "")
      + ". Nothing older is served from them: past the raw window only one coarse line per aircraft per day survives (no altitude, no flights). "
      + "A flight we never received (no receiver in range, or nobody was archiving that area) is not in the archive at all. "
      + "The one exception to the age limit: a tail on the server's tracked list has a permanent log of its own, read for what is older than the local hour files (see tracked_log).",
  };
}

// ── tail number -> hex ──────────────────────────────────────────────────────
export type ResolvedBy = "given" | "tracked_registry" | "live_feed" | "faa_registry" | "archive_registration";

export interface TailResolution {
  hex: string | null;
  resolved_by: ResolvedBy | null;
  basis: string;
  /** other sources that name a DIFFERENT hex for this tail */
  conflicts?: Array<{ hex: string; by: ResolvedBy }>;
}

const RESOLVE_BASIS: Record<ResolvedBy, string> = {
  given: "the request named the hex",
  tracked_registry: "the hex the tracked-plane poller learned from adsb.lol's answer to a query for this registration",
  live_feed: "a live row in the worldwide snapshot carries this registration (the feed's aircraft database, not a broadcast field)",
  faa_registry: "the FAA Releasable Aircraft Registry's Mode S code for this N-number (public registry, build of 2026-07-05, only aircraft our archive held then; an aircraft can transmit a different code)",
  archive_registration: "an archived fix carries this registration (the `rg` field) in the newest hour files",
};

export interface ResolveSources {
  tracked: TrackedPlane[];
  liveByReg: (reg: string) => string | null;
  faaByReg: (reg: string) => string | null;
}

/** Pure: a tail's hex from the sources that cost nothing, in a fixed order.
 *  Never guessed: no source -> hex null, and the caller says why. */
export function resolveTail(reg: string, src: ResolveSources): TailResolution {
  const cands: Array<{ hex: string; by: ResolvedBy }> = [];
  const add = (hex: string | null | undefined, by: ResolvedBy) => {
    const h = String(hex || "").toLowerCase();
    if (HEX_RE.test(h)) cands.push({ hex: h, by });
  };
  add(src.tracked.find((p) => p.reg === reg)?.hex, "tracked_registry");
  add(src.liveByReg(reg), "live_feed");
  add(src.faaByReg(reg), "faa_registry");
  if (!cands.length) return { hex: null, resolved_by: null, basis: "" };
  const first = cands[0];
  const conflicts = cands.filter((c) => c.hex !== first.hex);
  return { hex: first.hex, resolved_by: first.by, basis: RESOLVE_BASIS[first.by], ...(conflicts.length ? { conflicts } : {}) };
}

let faaIndex: { of: object; byReg: Map<string, string> } | null = null;
/** N-number -> hex from the committed FAA registry spine (reverse index, built once). */
export function faaHexForReg(reg: string, spine: Record<string, { n_number: string }> | null = loadEntitySpine()): string | null {
  if (!spine) return null;
  if (!faaIndex || faaIndex.of !== spine) {
    const byReg = new Map<string, string>();
    for (const [hex, rec] of Object.entries(spine)) {
      const n = normalizeReg(rec?.n_number);
      if (n) byReg.set(n, hex.toLowerCase());
    }
    faaIndex = { of: spine, byReg };
  }
  return faaIndex.byReg.get(reg) ?? null;
}

export interface RegScanResult {
  found: Map<string, { hex: string; t: number }>;
  files_scanned: number;
  bytes_scanned: number;
  /** start (epoch s) of the oldest hour file scanned; null = none */
  oldest_hour_t: number | null;
}

const RG_RE = /"rg":"([^"]+)"/;

/** Bounded scan of the NEWEST local hour files for lines whose `rg` is one of
 *  `regs` (newest file first; stops when all are found or a bound is hit). */
export async function scanArchiveForRegs(
  regs: ReadonlySet<string>, baseDir: string,
  o: { maxFiles?: number; maxBytes?: number } = {},
): Promise<RegScanResult> {
  const res: RegScanResult = { found: new Map(), files_scanned: 0, bytes_scanned: 0, oldest_hour_t: null };
  if (!regs.size) return res;
  const dir = path.join(baseDir, "aircraft");
  let files: string[];
  try { files = fs.readdirSync(dir).filter((f) => HOUR_FILE_RE.test(f)).sort().reverse(); } catch (e) { return res; }
  const maxFiles = o.maxFiles ?? REG_SCAN_MAX_FILES;
  const maxBytes = o.maxBytes ?? REG_SCAN_MAX_BYTES;
  for (const f of files) {
    if (res.found.size >= regs.size || res.files_scanned >= maxFiles || res.bytes_scanned >= maxBytes) break;
    const fp = path.join(dir, f);
    let size: number;
    try { size = fs.statSync(fp).size; } catch (e) { continue; }
    res.files_scanned++;
    res.bytes_scanned += size;
    const m = HOUR_FILE_RE.exec(f);
    if (m) res.oldest_hour_t = Date.parse(`${m[1]}T${m[2]}:00:00Z`) / 1000;
    await streamJsonlLines(fp, f.endsWith(".gz"), (line) => {
      const g = RG_RE.exec(line);
      if (!g) return;
      const reg = normalizeReg(g[1]);
      if (!reg || !regs.has(reg)) return;
      let row: { i?: unknown; t?: unknown };
      try { row = JSON.parse(line) as { i?: unknown; t?: unknown }; } catch (e) { return; }
      const hex = String(row.i || "").toLowerCase();
      const t = num(row.t);
      if (!HEX_RE.test(hex) || t == null) return;
      const prev = res.found.get(reg);
      if (!prev || t > prev.t) res.found.set(reg, { hex, t });
    });
  }
  return res;
}

// ── request parsing (strict) ────────────────────────────────────────────────
export type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

const intParam = (v: unknown): number | null | "bad" => {
  if (v === undefined || v === null || v === "") return null;
  const s = String(v);
  return /^\d{1,12}$/.test(s) ? parseInt(s, 10) : "bad";
};

export function parseTrackQuery(q: Record<string, unknown>, nowSec: number): Parsed<{ from: number; to: number; max: number; defaulted: boolean }> {
  const from = intParam(q.from), to = intParam(q.to), max = intParam(q.max);
  if (from === "bad" || to === "bad") return { ok: false, error: "from and to must be whole epoch seconds" };
  if (max === "bad") return { ok: false, error: `max must be a whole number from ${TRACK_MIN_POINTS} to ${TRACK_MAX_POINTS}` };
  if (max != null && (max < TRACK_MIN_POINTS || max > TRACK_MAX_POINTS)) {
    return { ok: false, error: `max must be from ${TRACK_MIN_POINTS} to ${TRACK_MAX_POINTS}` };
  }
  if ((from == null) !== (to == null) && from == null) return { ok: false, error: "to needs from: give both, or from alone (to = now), or neither (the last 24 hours)" };
  const toSec = to ?? nowSec;
  const fromSec = from ?? toSec - TRACK_DEFAULT_WINDOW_S;
  if (!(toSec > fromSec)) return { ok: false, error: "to must be after from" };
  if (toSec - fromSec > TRACK_MAX_WINDOW_S) {
    return { ok: false, error: `the window is limited to ${TRACK_MAX_WINDOW_S / 3600} hours per request (asked for ${Math.ceil((toSec - fromSec) / 3600)} h); ask for one flight at a time (GET /api/data/aircraft/fleet lists each flight's start and end)` };
  }
  return { ok: true, value: { from: fromSec, to: toSec, max: max ?? TRACK_DEFAULT_MAX_POINTS, defaulted: from == null && to == null } };
}

export interface FleetAsk { tail: string | null; hex: string | null }

export function parseFleetQuery(q: Record<string, unknown>): Parsed<{ asks: FleetAsk[]; explicit: boolean }> {
  const list = (v: unknown): string[] | "bad" => {
    if (v === undefined || v === null || v === "") return [];
    if (typeof v !== "string") return "bad"; // a repeated parameter arrives as an array
    return v.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
  };
  const tails = list(q.tails), hexes = list(q.hexes);
  if (tails === "bad" || hexes === "bad") return { ok: false, error: "tails and hexes are each ONE comma-separated value" };
  const asks: FleetAsk[] = [];
  const seen = new Set<string>();
  const bad: string[] = [];
  for (const raw of tails) {
    const reg = normalizeReg(raw);
    if (!reg) { bad.push(raw.slice(0, 16)); continue; }
    if (!seen.has(`t:${reg}`)) { seen.add(`t:${reg}`); asks.push({ tail: reg, hex: null }); }
  }
  for (const raw of hexes) {
    const hex = raw.toLowerCase();
    if (!HEX_RE.test(hex)) { bad.push(raw.slice(0, 16)); continue; }
    if (!seen.has(`h:${hex}`)) { seen.add(`h:${hex}`); asks.push({ tail: null, hex }); }
  }
  if (bad.length) return { ok: false, error: `not a tail number or a 6-digit hex: ${bad.slice(0, 5).join(", ")}` };
  if (asks.length > FLEET_MAX_AIRCRAFT) return { ok: false, error: `at most ${FLEET_MAX_AIRCRAFT} aircraft per request (asked for ${asks.length})` };
  return { ok: true, value: { asks, explicit: asks.length > 0 } };
}

// ── fleet: per-aircraft history (pure over fixes) ───────────────────────────
export interface AircraftHistory {
  flights: Trip[];
  flight_count: number;
  flights_listed: number;
  archive: {
    fixes: number; first_fix_t: number | null; last_fix_t: number | null;
    /** oldest fixes not kept because the per-aircraft cap was hit (flights
     *  before first_fix_t may exist in the archive and are not listed) */
    fixes_dropped_over_cap: number;
    duplicates_dropped: number;
    /** registrations the archived fixes carried (at most 5) */
    registrations_seen: string[];
  };
  /** lines read from the tracked tails' permanent log (its part older than
   *  the local hour files; before de-duplication and the cap). 0 = none read */
  tracked_log_lines: number;
}

/** Pure: one aircraft's cleaned, time-ordered fixes -> its flights (the trips
 *  route's splitter and airport check, unchanged) and what the archive holds. */
export function historyFromFixes(
  fixes: ArchivedFix[],
  meta: { dropped_over_cap: number; duplicates_dropped: number; regs: string[]; from_tracked_log?: number },
  nearest?: NearestAirportFn,
): AircraftHistory {
  const trips = splitTrips(fixes, undefined, undefined, undefined, nearest);
  return {
    flights: trips.slice(0, FLEET_MAX_FLIGHTS),
    flight_count: trips.length,
    flights_listed: Math.min(trips.length, FLEET_MAX_FLIGHTS),
    archive: {
      fixes: fixes.length,
      first_fix_t: fixes.length ? fixes[0].t : null,
      last_fix_t: fixes.length ? fixes[fixes.length - 1].t : null,
      fixes_dropped_over_cap: meta.dropped_over_cap,
      duplicates_dropped: meta.duplicates_dropped,
      registrations_seen: meta.regs,
    },
    tracked_log_lines: meta.from_tracked_log ?? 0,
  };
}

export interface FleetLive {
  seen_t: number; age_s: number; fresh: boolean;
  lat: number; lon: number;
  alt_m: number | null; alt_kind: "barometric_first"; on_ground: boolean;
  gs_ms: number | null; track_deg: number | null; callsign: string | null; src: string;
}

export function fleetLive(s: AircraftLiveState | null, nowMs: number): { live: FleetLive | null; live_note: string } {
  if (!s) return { live: null, live_note: "no signal now: the worldwide snapshot holds no fix for this aircraft (it keeps fixes for 10 minutes)" };
  const age = Math.max(0, Math.round((nowMs - s.row.seenAt) / 1000));
  if (age > LIVE_MAX_AGE_S) return { live: null, live_note: `no signal now: the newest fix is ${age} s old (older than ${LIVE_MAX_AGE_S} s is not served as live)` };
  const r = s.row;
  return {
    live: {
      seen_t: Math.round(r.seenAt / 1000), age_s: age, fresh: age <= LIVE_FRESH_S,
      lat: r.lat, lon: r.lon,
      alt_m: r.altFt == null ? null : Math.round(r.altFt * FT_TO_M), alt_kind: "barometric_first", on_ground: r.gnd,
      gs_ms: r.gsKt == null ? null : Math.round(r.gsKt * KT_TO_MS), track_deg: r.trk == null ? null : Math.round(r.trk),
      callsign: r.callsign, src: r.src,
    },
    live_note: age <= LIVE_FRESH_S
      ? "a fix from the last 2 minutes"
      : `last known position, ${age} s old: older than ${LIVE_FRESH_S} s is not a live position`,
  };
}

// ── routes ──────────────────────────────────────────────────────────────────
export const TRACK_SOURCE = "our own archive of community ADS-B ground-receiver data (adsb.lol, ODbL 1.0; adsb.fi and airplanes.live rows are non-commercial and carry src adsbfi / airplaneslive), as broadcast";
export const TRACK_HONESTY =
  "Every point is a fix our servers received and archived, or (origin live) the newest fix in the live snapshot. Nothing is interpolated: "
  + "where coverage stopped there is a listed gap, and the points on either side of it must not be joined. The archive keeps one fix per "
  + "30 s to 5 min for most aircraft (more through turns and climbs, every fix for tracked tails and while an aircraft is selected), so the "
  + "track is a sample of the flight, not every broadcast. alt_m is geometric (GNSS, above the WGS84 ellipsoid) where the aircraft sent it and "
  + "barometric pressure altitude otherwise; alt_kind says which on every point and neither is height above the ground. A vertical rate marked "
  + "derived was computed here from two fixes. track_deg is the track over the ground, not the heading. Pitch is not in the data; bank only "
  + "where roll_deg is given.";
export const FLEET_HONESTY =
  "Flights are split from the fixes our archive holds for each aircraft: a silence over 45 minutes, or 15 minutes on the ground between two "
  + "airborne stretches, starts a new one. The archive is thinned, so start and end times are the first and last fix we kept (durations are "
  + "lower bounds), and a flight outside receiver coverage or before the first fix we hold is missing, not listed as empty. quality says "
  + "whether a flight was seen from the ground up and back down; signal_lost_airborne means coverage ended in the air, nothing more. Community "
  + "ADS-B only: no FAA flight-plan or SWIM field is in this response.";

export const TRACKED_LOG_TRACK_NOTE =
  "A tail on the server's tracked list has a permanent log of its own: every fix received for it, not thinned, with no age limit, capped in "
  + "bytes (oldest month deleted first). It is read here only for the part of a window that is older than the oldest hour file on the local disk; its "
  + "points say origin tracked_log. first_fix_t / last_fix_t are the oldest and newest fix that log holds. An aircraft that was never "
  + "tracked has none, and a tracked tail's log starts the day it was tracked (2026-11-17 at the earliest).";

export const TRACKED_LOG_FLEET_NOTE =
  "For a tail on the server's tracked list, flights older than the local hour files come from its permanent log (every fix received, "
  + "no age limit, capped in bytes with the oldest month deleted first; tracked_log_lines on each aircraft says how many lines it gave). "
  + "It starts the day the tail was tracked, 2026-11-17 at the earliest; removing a tail from the list keeps its log.";

export interface TrackDataDeps {
  baseDir?: string;
  now?: () => number;
  env?: NodeJS.ProcessEnv;
  /** worldwide snapshot lookup (default: the one globalAircraft registered) */
  live?: () => AircraftLiveLookup | null;
  /** the server's tracked tail list (routes.ts passes the poller's registry) */
  tracked?: () => TrackedPlane[];
  nearest?: NearestAirportFn;
  faaByReg?: (reg: string) => string | null;
}

interface Cached<T> { at: number; data: T }
function cacheSet<T>(m: Map<string, Cached<T>>, key: string, v: Cached<T>, cap: number): void {
  m.delete(key);
  m.set(key, v);
  while (m.size > cap) {
    const oldest = m.keys().next().value;
    if (oldest === undefined) break;
    m.delete(oldest);
  }
}

export function registerAircraftTrackDataRoutes(app: Express, deps: TrackDataDeps = {}): void {
  const now = deps.now ?? Date.now;
  const env = deps.env ?? process.env;
  const base = () => deps.baseDir || archiveBaseDir();
  const live = deps.live ?? aircraftLiveLookup;
  const tracked = deps.tracked ?? (() => []);
  // QC-2 rule of the trips route: fixed-wing fields only
  const nearest: NearestAirportFn = deps.nearest ?? ((la, lo) => nearestAirport(la, lo, 6, ["L", "M", "S", "W"]));
  const faaByReg = deps.faaByReg ?? ((reg: string) => faaHexForReg(reg));
  const fail = (res: { status: (n: number) => { json: (b: unknown) => unknown } }, where: string, e: unknown) => {
    console.error(`[aircraft-track-data] ${where}:`, e instanceof Error ? e.message : String(e));
    res.status(500).json({ error: `${where} failed`, detail: e instanceof Error ? e.message : String(e) });
  };

  // ── track ────────────────────────────────────────────────────────────────
  interface ArchivePart { read: HexRead; scan: ArchiveScanStats; tracked_log: TrackedLogPart }
  const trackCache = new Map<string, Cached<ArchivePart>>();
  const trackInflight = new Map<string, Promise<ArchivePart>>();
  const cachedTrackFixes = (): number => { let n = 0; for (const c of trackCache.values()) n += c.data.read.fixes.length; return n; };
  registerHeapGauge("aircraftTrackData.trackCache", () => ({ entries: trackCache.size, cap: 16, fixes: cachedTrackFixes(), fixes_cap: TRACK_CACHE_MAX_FIXES }));
  // the archive part is read on whole-minute bounds (so a client polling "the
  // last 24 h" shares one read for TRACK_CACHE_MS) and cut to the exact
  // window afterwards; the live fix is added fresh on every response
  const readTrack = (hex: string, fromExact: number, toExact: number): Promise<ArchivePart> => {
    const from = Math.floor(fromExact / 60) * 60;
    const to = Math.ceil(toExact / 60) * 60;
    const key = `${hex}|${from}|${to}`;
    for (const [k, c] of trackCache) if (now() - c.at > TRACK_CACHE_MS) trackCache.delete(k); // nothing stale is held
    const hit = trackCache.get(key);
    if (hit) return Promise.resolve(hit.data);
    const pending = trackInflight.get(key);
    if (pending) return pending;
    const p = readHexFixes([hex], base(), { fromSec: from, toSec: to, decodeIntent: true, capPerHex: TRACK_SCAN_CAP })
      .then(async (r): Promise<ArchivePart> => {
        let read = r.byHex.get(hex) ?? { fixes: [], duplicates_dropped: 0, dropped_over_cap: 0 };
        // the tracked tails' permanent log, for the part of the window the
        // local hour files no longer (or never) hold
        const oldestMs = oldestRawHour("aircraft", base());
        let tracked_log: TrackedLogPart;
        if (oldestMs != null && from >= oldestMs / 1000) {
          tracked_log = { read: false, why: "not read: the window starts inside what the local hour files hold", fixes_from_log: 0, months: null };
        } else {
          // only the log's fixes from BEFORE the oldest hour file: inside the
          // hour files' reach the answer is exactly what it was without the log
          const logTo = oldestMs == null ? to : Math.min(to, oldestMs / 1000 - 1);
          const lg = await readTrackedLogFixes(base(), hex, { fromSec: from, toSec: logTo, cap: TRACK_SCAN_CAP });
          if (!lg.months.months_held) {
            tracked_log = { read: false, why: "no permanent log is held for this aircraft (it is kept only for tails on the tracked list, from the day they are tracked)", fixes_from_log: 0, months: null };
          } else {
            const merged = mergeTrackedLogFixes(read.fixes, lg.fixes, TRACK_SCAN_CAP);
            read = {
              fixes: merged.fixes,
              duplicates_dropped: read.duplicates_dropped + merged.duplicates_dropped,
              dropped_over_cap: read.dropped_over_cap + lg.dropped_over_cap + merged.dropped_over_cap,
            };
            tracked_log = {
              read: true, why: "the window starts before the oldest hour file on the local disk, so the permanent log was read for the part before it",
              fixes_from_log: merged.fixes.reduce((n, f) => n + (f.store === "tracked_log" ? 1 : 0), 0), months: lg.months,
            };
          }
        }
        const part = { read, scan: r.scan, tracked_log };
        cacheSet(trackCache, key, { at: now(), data: part }, 16);
        while (trackCache.size > 1 && cachedTrackFixes() > TRACK_CACHE_MAX_FIXES) {
          const oldest = trackCache.keys().next().value;
          if (oldest === undefined) break;
          trackCache.delete(oldest);
        }
        return part;
      })
      .finally(() => { trackInflight.delete(key); });
    trackInflight.set(key, p);
    return p;
  };

  app.get("/api/data/aircraft/track/:hex", async (req, res) => {
    const hex = String(req.params.hex || "").toLowerCase();
    if (!HEX_RE.test(hex)) return res.status(400).json({ error: "icao24 hex required (6 hex digits)" });
    const t = now();
    const nowSec = Math.floor(t / 1000);
    const q = parseTrackQuery(req.query as Record<string, unknown>, nowSec);
    if (!q.ok) return res.status(400).json({ error: q.error });
    const { from, to, max } = q.value;
    try {
      const part = await readTrack(hex, from, to);
      const fixes = part.read.fixes.filter((f) => f.t >= from && f.t <= to);
      // the live fix on top: only when it is newer than the last archived fix
      const st = live()?.byHex(hex, t) ?? null;
      const lastT = fixes.length ? fixes[fixes.length - 1].t : null;
      let liveIndex: number | null = null;
      let liveInfo: { appended: boolean; seen_t: number | null; age_s: number | null; reason: string };
      if (!st) liveInfo = { appended: false, seen_t: null, age_s: null, reason: "no signal now: the worldwide snapshot holds no fix for this aircraft" };
      else {
        const lf = liveFix(st, t);
        const age = Math.max(0, nowSec - lf.t);
        if (age > LIVE_MAX_AGE_S) liveInfo = { appended: false, seen_t: lf.t, age_s: age, reason: `the newest fix is ${age} s old: not served as live` };
        else if (lf.t < from || lf.t > to) liveInfo = { appended: false, seen_t: lf.t, age_s: age, reason: "the newest live fix is outside the requested window" };
        else if (lastT != null && lf.t <= lastT + 1) liveInfo = { appended: false, seen_t: lf.t, age_s: age, reason: "the archive already holds this fix (or a newer one)" };
        else {
          fixes.push(lf);
          liveIndex = fixes.length - 1;
          liveInfo = { appended: true, seen_t: lf.t, age_s: age, reason: age <= LIVE_FRESH_S ? "the newest live fix, newer than the last archived line" : `last known position, ${age} s old: not a live position` };
        }
      }
      const built = buildTrack(fixes, { max, liveIndex });
      const listed = built.gaps.slice(0, 200);
      const logHeld: PlaneLogSummary = await trackedLogSummary(base(), hex).catch((e: unknown) => {
        console.error("[aircraft-track-data] tracked log summary:", e instanceof Error ? e.message : String(e));
        return { ...EMPTY_LOG_SUMMARY };
      });
      res.set("Cache-Control", "public, max-age=10");
      res.json({
        at: t, hex, kind: "raw",
        range: { from, to, defaulted_to_last_24h: q.value.defaulted, max_window_s: TRACK_MAX_WINDOW_S },
        fields: TRACK_FIELDS, units: TRACK_UNITS, enums: TRACK_ENUMS,
        count: built.rows.length,
        rows: built.rows,
        live: liveInfo,
        gaps: {
          threshold_s: TRACK_GAP_S,
          count: built.gaps.length,
          listed,
          rule: `two consecutive fixes more than ${TRACK_GAP_S} s apart are a gap in what we received or kept; nothing is interpolated across it and a client must not draw a line across it`,
        },
        decimation: built.decimation,
        coverage: {
          first_fix_t: fixes.length ? fixes[0].t : null,
          last_fix_t: fixes.length ? fixes[fixes.length - 1].t : null,
          points_in_window: built.points_total,
          ...built.counts,
          duplicates_dropped: part.read.duplicates_dropped,
          fixes_dropped_over_cap: part.read.dropped_over_cap,
          scan_cap: TRACK_SCAN_CAP,
          hour_files: part.scan,
          archive: archiveReach(base(), env),
          // the permanent per-plane log of tracked tails: what is held, and whether this window read it
          tracked_log: {
            held: logHeld.months > 0, ...logHeld,
            read_for_this_window: part.tracked_log.read, why: part.tracked_log.why,
            fixes_in_window_from_log: fixes.reduce((n, f) => n + (f.store === "tracked_log" ? 1 : 0), 0),
            months_read: part.tracked_log.months,
            note: TRACKED_LOG_TRACK_NOTE,
          },
          note: fixes.length === 0
            ? "no fix for this aircraft in this window on the local archive: it was not received, not archived, or the window is older than the archive holds"
            : undefined,
        },
        attitude: TRACK_ATTITUDE_NOTE,
        source: TRACK_SOURCE,
        honesty: TRACK_HONESTY,
      });
    } catch (e) {
      fail(res, "track read", e);
    }
  });

  // ── fleet ────────────────────────────────────────────────────────────────
  const histCache = new Map<string, Cached<AircraftHistory>>();
  const regCache = new Map<string, Cached<{ hex: string; t: number } | null>>();
  trackMap("aircraftTrackData.histCache", histCache, 64);
  trackMap("aircraftTrackData.regCache", regCache, 256);
  let lastRegScan: { files_scanned: number; bytes_scanned: number; oldest_hour_t: number | null } | null = null;
  // archive scans run one at a time: concurrent fleet requests queue, then
  // find the per-aircraft cache filled
  let scanChain: Promise<unknown> = Promise.resolve();
  const serial = <T>(fn: () => Promise<T>): Promise<T> => {
    const p = scanChain.then(fn, fn);
    scanChain = p.catch(() => undefined);
    return p;
  };

  app.get("/api/data/aircraft/fleet", async (req, res) => {
    const parsed = parseFleetQuery(req.query as Record<string, unknown>);
    if (!parsed.ok) return res.status(400).json({ error: parsed.error });
    const t = now();
    const trackedPlanes = tracked();
    // no tails given: the server's tracked list IS the saved fleet
    const asks: FleetAsk[] = parsed.value.explicit
      ? parsed.value.asks
      : trackedPlanes.slice(0, FLEET_MAX_AIRCRAFT).map((p) => ({ tail: p.reg, hex: null }));
    try {
      const lk = live();
      const src: ResolveSources = {
        tracked: trackedPlanes,
        liveByReg: (reg) => lk?.byReg(reg)?.hex ?? null,
        faaByReg,
      };
      const resolved = asks.map((a): { ask: FleetAsk; r: TailResolution } => (a.hex
        ? { ask: a, r: { hex: a.hex, resolved_by: "given", basis: RESOLVE_BASIS.given } }
        : { ask: a, r: resolveTail(a.tail as string, src) }));
      // tails nothing cheap resolved: ONE bounded scan of the newest hour files
      const need = new Set<string>();
      for (const x of resolved) {
        if (x.r.hex || !x.ask.tail) continue;
        const c = regCache.get(x.ask.tail);
        if (c && t - c.at <= (c.data ? REG_FOUND_TTL_MS : REG_MISS_TTL_MS)) continue;
        need.add(x.ask.tail);
      }
      if (need.size) {
        const scan = await serial(() => scanArchiveForRegs(need, base()));
        lastRegScan = { files_scanned: scan.files_scanned, bytes_scanned: scan.bytes_scanned, oldest_hour_t: scan.oldest_hour_t };
        for (const reg of need) cacheSet(regCache, reg, { at: now(), data: scan.found.get(reg) ?? null }, 256);
      }
      for (const x of resolved) {
        if (x.r.hex || !x.ask.tail) continue;
        const c = regCache.get(x.ask.tail)?.data ?? null;
        x.r = c
          ? { hex: c.hex, resolved_by: "archive_registration", basis: `${RESOLVE_BASIS.archive_registration} (newest such fix at ${c.t})` }
          : {
            hex: null, resolved_by: null,
            basis: `unresolved: not in the tracked list with a learned hex, no live row carries this registration, not in the FAA registry spine, and no archived fix in the newest ${lastRegScan ? lastRegScan.files_scanned : REG_SCAN_MAX_FILES} hour files carries it (older hours are not searched). Pass the hex in hexes= if you know it, or add the tail to the tracked list so its hex is learned when it next broadcasts`,
          };
      }
      // history: ONE pass over the local archive for every hex not cached
      const hexes = Array.from(new Set(resolved.map((x) => x.r.hex).filter((h): h is string => !!h)));
      const stale = hexes.filter((h) => { const c = histCache.get(h); return !c || t - c.at > FLEET_CACHE_MS; });
      let scanStats: ArchiveScanStats | null = null;
      if (stale.length) {
        scanStats = await serial(async () => {
          const still = stale.filter((h) => { const c = histCache.get(h); return !c || now() - c.at > FLEET_CACHE_MS; });
          if (!still.length) return null;
          const r = await readFleetHistories(still, base(), nearest);
          for (const [h, hist] of r.byHex) cacheSet(histCache, h, { at: now(), data: hist }, 64);
          return r.scan;
        });
      }
      const aircraft = resolved.map((x) => {
        const hex = x.r.hex;
        const tp = trackedPlanes.find((p) => (x.ask.tail && p.reg === x.ask.tail) || (hex && p.hex && p.hex.toLowerCase() === hex)) ?? null;
        const hist = hex ? histCache.get(hex) : undefined;
        const lv = hex ? fleetLive(lk?.byHex(hex, t) ?? null, t) : { live: null, live_note: "no hex resolved, so no live lookup was made" };
        return {
          tail: x.ask.tail ?? tp?.reg ?? null,
          hex,
          resolved_by: x.r.resolved_by,
          resolution: { basis: x.r.basis, ...(x.r.conflicts ? { conflicts: x.r.conflicts } : {}) },
          tracked: tp ? { reg: tp.reg, added_t: Math.round(tp.added_at / 1000), last_seen_t: tp.last_seen ? Math.round(tp.last_seen / 1000) : null } : null,
          flights: hist ? hist.data.flights : [],
          flight_count: hist ? hist.data.flight_count : 0,
          flights_listed: hist ? hist.data.flights_listed : 0,
          history_as_of: hist ? Math.round(hist.at / 1000) : null,
          archive: hist ? hist.data.archive : null,
          tracked_log_lines: hist ? hist.data.tracked_log_lines : 0,
          live: lv.live,
          live_note: lv.live_note,
        };
      });
      res.set("Cache-Control", "public, max-age=30");
      res.json({
        at: t, kind: "raw",
        asked: parsed.value.explicit ? "the tails/hexes in the request" : "no tails given: the server's tracked list",
        count: aircraft.length,
        aircraft,
        limits: {
          max_aircraft: FLEET_MAX_AIRCRAFT, max_flights_listed: FLEET_MAX_FLIGHTS,
          history_cache_s: FLEET_CACHE_MS / 1000,
          fixes_cap_per_aircraft: Math.min(FLEET_SCAN_CAP_PER_HEX, Math.floor(FLEET_SCAN_CAP_TOTAL / Math.max(1, aircraft.length))),
        },
        flight_fields: {
          start_t: "epoch s, first fix of the flight", end_t: "epoch s, last fix", duration_s: "end_t - start_t (a lower bound: the archive is thinned)",
          fixes: "archived fixes in the flight", from: "first fix {la, lo} degrees", to: "last fix {la, lo} degrees",
          from_airport: "nearest fixed-wing airport within 6 km of the first fix (OurAirports) or null", to_airport: "same for the last fix",
          max_alt_m: "highest archived altitude, metres: barometric where the aircraft sent one, else geometric (the archive's al)",
          callsigns: "callsigns seen", quality: "complete | taxi_only | partial_start | signal_lost_airborne | partial_both", quality_basis: "why",
          is_flight: "false = ground movement only", verified: "both ends agree with an airport's position and elevation",
          bbox: "[minLon, minLat, maxLon, maxLat]",
        },
        coverage: {
          archive: archiveReach(base(), env),
          hour_files: scanStats,
          registration_scan: lastRegScan,
          tracked_log: TRACKED_LOG_FLEET_NOTE,
          note: "times are the fixes' own (the same clock as GET /api/data/aircraft/track/:hex, so a flight's start_t/end_t can be passed to it as from/to); GET /api/data/aircraft/trips/:hex uses the archive's stamps and can differ by the feed's position age (under a minute).",
        },
        source: TRACK_SOURCE,
        honesty: FLEET_HONESTY,
      });
    } catch (e) {
      fail(res, "fleet read", e);
    }
  });
}
