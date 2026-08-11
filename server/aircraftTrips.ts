// Plane-tracking T2 (human directive 2026-08-08): per-plane TRIP history
// from our own archive. Splits a hex's archived fixes into flights and
// serves them honestly bounded: raw fixes survive RAW_RETENTION_DAYS (7),
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
    trips.push({
      start_t: s[0].t,
      end_t: s[s.length - 1].t,
      duration_s: s[s.length - 1].t - s[0].t,
      fixes: s.length,
      from: { la: s[0].la, lo: s[0].lo },
      to: { la: s[s.length - 1].la, lo: s[s.length - 1].lo },
      max_alt_m: maxAlt,
      callsigns: Array.from(calls),
      bbox: [minLon, minLa, maxLon, maxLa],
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
  const cap = opts.maxPoints ?? 20_000;
  return pts.length > cap ? pts.slice(-cap) : pts;
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
