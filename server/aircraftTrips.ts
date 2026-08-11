// Plane-tracking T2 (human directive 2026-08-08): per-plane TRIP history
// from our own archive. Splits a hex's archived fixes into flights and
// serves them honestly bounded: raw fixes survive RAW_RETENTION_DAYS (30),
// so trips reach back at most that far — the coverage note says so rather
// than implying a full history. Sessionization reconciles the two existing
// splitters (fleetUtilization.foldSessions' 45-min airborne gap;
// trackModel.trimToCurrentFlight's ground-dwell rule): a gap > 45 min
// between consecutive fixes OR a sustained on-ground dwell >= 15 min with
// airborne fixes on both sides starts a new trip.

import fs from "fs";
import path from "path";
import {
  streamJsonlLines, RAW_RETENTION_DAYS,
} from "./datacoreArchive";

export const TRIP_GAP_SEC = 45 * 60;
export const TRIP_GROUND_DWELL_SEC = 15 * 60;
/** fixes below this count are position noise, not a trip worth listing */
export const TRIP_MIN_FIXES = 3;

export interface ArchivedFix {
  t: number; la: number; lo: number;
  al?: number | null; c?: string | null; g?: boolean;
}

export interface Trip {
  start_t: number;
  end_t: number;
  duration_s: number;
  fixes: number;
  from: { la: number; lo: number };
  to: { la: number; lo: number };
  max_alt_m: number | null;
  callsigns: string[];
  bbox: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  quality: TripQuality;      // QC-1 shape classification (2026-08-11)
  quality_basis: string;     // the honest one-line reason
  /** QC-2 (2026-08-11): airport verification — a COMPLETE flight takes off
   *  from a field and lands at one, at plausible field elevation. */
  is_flight: boolean;        // false = ground log (taxi/parked), not a flight
  verified: boolean;         // both endpoints airport+elevation confirmed
  from_airport?: { id: string; n: string; dist_km: number; el: number | null } | null;
  to_airport?: { id: string; n: string; dist_km: number; el: number | null } | null;
}

/** injectable airport resolver (server/airportsIndex.nearestAirport in prod) */
export type NearestAirportFn = (la: number, lo: number) =>
  ({ id: string; n: string; dist_km: number; el: number | null } | null);

/** baro-vs-field tolerance: pressure offsets + reference-point elevation
 *  spread on big fields */
export const QC_FIELD_ELEV_TOLERANCE_M = 350;

/** Pure: does an endpooint fix agree with a field? grounded always agrees;
 *  an altitude agrees when within tolerance of the published elevation;
 *  a field without published elevation can't disagree (null = unverifiable,
 *  not wrong). */
export function endpointAgreesWithField(
  f: ArchivedFix, ap: { el: number | null } | null,
): boolean {
  if (!ap) return false;
  if (f.g || f.al == null) return true;
  if (ap.el == null) return true;
  return Math.abs(f.al - ap.el) <= QC_FIELD_ELEV_TOLERANCE_M;
}

/** Pure: sorted fixes -> trips. Split on hard time gaps and on qualifying
 *  ground dwells (>= dwellSec continuously on-ground/no-altitude, with
 *  airborne fixes before AND after — a parked transponder is not a trip
 *  boundary generator by itself). */
export function splitTrips(
  fixes: ArchivedFix[],
  gapSec = TRIP_GAP_SEC,
  dwellSec = TRIP_GROUND_DWELL_SEC,
  minFixes = TRIP_MIN_FIXES,
  nearest?: NearestAirportFn,
): Trip[] {
  if (!fixes.length) return [];
  const segs: ArchivedFix[][] = [];
  let cur: ArchivedFix[] = [fixes[0]];
  const grounded = (f: ArchivedFix) => !!f.g || f.al == null;
  let dwellStart: number | null = grounded(fixes[0]) ? fixes[0].t : null;
  let sawAirborne = !grounded(fixes[0]);
  for (let i = 1; i < fixes.length; i++) {
    const f = fixes[i];
    const prev = fixes[i - 1];
    let boundary = f.t - prev.t > gapSec;
    if (!boundary && !grounded(f) && dwellStart != null && sawAirborne
        && prev.t - dwellStart >= dwellSec) {
      // airborne again after a sustained ground dwell inside an airborne
      // track: the dwell ends the previous trip; its last fix leads the new
      // one in as the takeoff point (trimToCurrentFlight convention)
      boundary = true;
    }
    if (boundary) {
      segs.push(cur);
      // hard time gap: clean start at f. dwell boundary: the dwell's last
      // fix leads the new trip in as its takeoff point
      cur = f.t - prev.t > gapSec ? [f] : [prev, f];
      sawAirborne = !grounded(f);
      dwellStart = grounded(f) ? f.t : null;
      continue;
    }
    if (grounded(f)) { if (dwellStart == null) dwellStart = f.t; }
    else { dwellStart = null; sawAirborne = true; }
    cur.push(f);
  }
  segs.push(cur);
  const trips: Trip[] = [];
  for (const s of segs) {
    if (s.length < minFixes) continue;
    let maxAlt: number | null = null;
    let minLon = Infinity, minLa = Infinity, maxLon = -Infinity, maxLa = -Infinity;
    const calls = new Set<string>();
    for (const f of s) {
      if (f.al != null && (maxAlt == null || f.al > maxAlt)) maxAlt = f.al;
      if (f.lo < minLon) minLon = f.lo;
      if (f.lo > maxLon) maxLon = f.lo;
      if (f.la < minLa) minLa = f.la;
      if (f.la > maxLa) maxLa = f.la;
      if (f.c) calls.add(f.c);
    }
    const qc = classifyTrip(s);
    // QC-2 airport verification: resolve both endpoints against the field
    // catalog. Only trips whose SHAPE already says the end is low/ground
    // can verify — an endpoint at cruise has no landing to confirm.
    const fromAp = nearest ? nearest(s[0].la, s[0].lo) : null;
    const toAp = nearest ? nearest(s[s.length - 1].la, s[s.length - 1].lo) : null;
    const isFlight = qc.quality !== "taxi_only";
    const verified = isFlight && qc.quality === "complete"
      && endpointAgreesWithField(s[0], fromAp)
      && endpointAgreesWithField(s[s.length - 1], toAp);
    trips.push({
      is_flight: isFlight,
      verified,
      from_airport: fromAp,
      to_airport: toAp,
      start_t: s[0].t,
      end_t: s[s.length - 1].t,
      duration_s: s[s.length - 1].t - s[0].t,
      fixes: s.length,
      from: { la: s[0].la, lo: s[0].lo },
      to: { la: s[s.length - 1].la, lo: s[s.length - 1].lo },
      max_alt_m: maxAlt,
      callsigns: Array.from(calls),
      bbox: [minLon, minLa, maxLon, maxLa],
      quality: qc.quality,
      quality_basis: qc.basis,
    });
  }
  return trips.sort((a, b) => b.start_t - a.start_t); // newest first
}

/** Stream a hex's fixes across the WHOLE raw window (all retained days,
 *  not recentTrackAsync's 48h), optionally time-bounded. Keeps callsign,
 *  altitude and ground flag — the trip splitter needs them. */
export async function fullTrackAsync(
  kind: "aircraft" | "vessels" | "trains", id: string, baseDir: string,
  opts: { fromSec?: number; toSec?: number; maxPoints?: number } = {},
): Promise<ArchivedFix[]> {
  const dir = path.join(baseDir, kind);
  if (!fs.existsSync(dir)) return [];
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return []; }
  const pts: ArchivedFix[] = [];
  for (const f of files) {
    if (!/^\d{4}-\d{2}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
    // hour-file prefilter by the requested window (file hour in UTC)
    if (opts.fromSec || opts.toSec) {
      const hourMs = Date.parse(f.slice(0, 13).replace(/-(\d{2})$/, "T$1:00:00Z"));
      if (Number.isFinite(hourMs)) {
        const h0 = hourMs / 1000, h1 = h0 + 3600;
        if (opts.toSec && h0 > opts.toSec) continue;
        if (opts.fromSec && h1 < opts.fromSec) continue;
      }
    }
    await streamJsonlLines(path.join(dir, f), f.endsWith(".gz"), (line) => {
      if (!line.includes(id)) return; // substring prefilter, then verify
      try {
        const r = JSON.parse(line);
        if (r.i !== id) return;
        if (opts.fromSec && r.t < opts.fromSec) return;
        if (opts.toSec && r.t > opts.toSec) return;
        pts.push({ t: r.t, la: r.la, lo: r.lo, al: r.al ?? null, c: r.c ?? null, g: !!r.g });
      } catch {}
    });
  }
  pts.sort((a, b) => a.t - b.t);
  // t-dedupe (backfill 2026-08-08): the 30s poller and the day-trace
  // backfill both write this hex — a same-second duplicate is the same
  // physical fix; keep the one with altitude when they differ.
  const dedup: ArchivedFix[] = [];
  for (const p of pts) {
    const last = dedup[dedup.length - 1];
    if (last && last.t === p.t) {
      if (last.al == null && p.al != null) dedup[dedup.length - 1] = p;
      continue;
    }
    dedup.push(p);
  }
  const cap = opts.maxPoints ?? 20_000;
  return dedup.length > cap ? dedup.slice(-cap) : dedup;
}

// ── TRIP QUALITY (human directive 2026-08-11: "for it to be one log it
// need a take off airport and a landing one... planes might turn on adsb
// and just taxi around thats not a flight or they might crash we need a
// way to identify these anomaly") ──────────────────────────────────────────
// QC-1 classifies from the track's OWN shape — no external data needed:
// a real flight starts low/ground, climbs, and returns low/ground; a track
// that begins or ends AT ALTITUDE is coverage that started late or a
// signal lost airborne. HONESTY RULE: "signal_lost_airborne" is exactly
// that — receiver coverage ends far more often than aircraft do; this
// module never claims a crash. QC-2 (filed) adds OurAirports (~80k fields
// with elevations, public domain) to name the endpoints and verify field
// elevation.
export type TripQuality =
  | "complete"              // low/ground at both ends with real altitude between
  | "taxi_only"             // never meaningfully above its own low point
  | "partial_start"         // first seen already at altitude (coverage began mid-flight)
  | "signal_lost_airborne"  // last seen at altitude — coverage loss OR anomaly, unresolvable from here
  | "partial_both";         // airborne at both ends (a pass through our coverage)

/** meters above the trip's own minimum that counts as "really airborne" */
export const QC_AIRBORNE_ABOVE_MIN_M = 300;
/** no airport on Earth sits above ~4,400m MSL (La Paz, Daocheng); a trip
 *  whose LOWEST fix is higher than this never came near any possible
 *  ground — its "low point" is cruise, not a landing (test-caught: a
 *  cruise-only pass classified its own minimum as ground) */
export const QC_GROUND_PLAUSIBLE_MAX_M = 4600;

export function classifyTrip(fixes: ArchivedFix[]): { quality: TripQuality; basis: string } {
  const alts = fixes.map((f) => (f.g ? null : f.al)).filter((a): a is number => a != null);
  if (!alts.length) {
    return { quality: "taxi_only", basis: "no airborne altitude ever broadcast — ground movement only" };
  }
  const minAl = Math.min(...alts);
  const maxAl = Math.max(...alts);
  if (maxAl - minAl < QC_AIRBORNE_ABOVE_MIN_M && minAl <= QC_GROUND_PLAUSIBLE_MAX_M) {
    return { quality: "taxi_only", basis: `altitude never rose ${QC_AIRBORNE_ABOVE_MIN_M}m above the track's own low point` };
  }
  const groundPlausible = minAl <= QC_GROUND_PLAUSIBLE_MAX_M;
  const lowAt = (f: ArchivedFix) => !!f.g || f.al == null
    || (groundPlausible && f.al <= minAl + QC_AIRBORNE_ABOVE_MIN_M);
  const startLow = lowAt(fixes[0]);
  const endLow = lowAt(fixes[fixes.length - 1]);
  if (startLow && endLow) return { quality: "complete", basis: "low/ground at both ends with real altitude between" };
  if (!startLow && endLow) return { quality: "partial_start", basis: "first seen already at altitude — coverage began mid-flight" };
  if (startLow && !endLow) {
    return {
      quality: "signal_lost_airborne",
      basis: "last fix at altitude — receiver coverage ends far more often than aircraft do; logged as an anomaly, never asserted as one",
    };
  }
  return { quality: "partial_both", basis: "airborne at both ends — a pass through our coverage window" };
}

/** The honest coverage statement every trips response carries. */
export function tripsCoverage(): { raw_days: number; note: string } {
  return {
    raw_days: RAW_RETENTION_DAYS,
    note: `trips are built from raw archived fixes, retained ${RAW_RETENTION_DAYS} days ` +
      `(older history survives only as per-day coarse polylines without altitude); ` +
      `fix cadence is adaptively thinned 30s-5min, so durations are lower bounds`,
  };
}
