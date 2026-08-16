// TIME MACHINE T-1 — hex-multiplexed WINDOW reader (earth_twin_program.md
// TIME MACHINE v2, human directive 2026-08-11: the Time Machine "just shows
// dots — it should have a time scale … and when on it shows the curtains
// for all planes and the path tracked").
//
// One call answers "every aircraft that moved through THIS viewport in THIS
// time window, with enough per-hex track to draw its curtain" — the payload
// T-2's scrubber selectors and T-3's curtain fleet ride on. Vessels reuse
// the same machinery (T-4 parity, wired in server/routes.ts's `kind` query
// param — no altitude → paths, not curtains).
//
// SCALE S1 LAW (scale_program.md): viewport-bounded, LOD-decimated by zoom,
// hard caps stated honestly in the response ("returned N of M hexes — zoom
// in"). The archive keeps everything; only the RENDER fidelity narrows.
//
// SCAN SHAPE (charter requirement): ONE stream pass per hour file, every
// hex accumulated in that single pass — never per-hex file walks (a window
// read must not be O(hexes) scans). Files stream NEWEST-FIRST under a file
// budget, so a 30-day request over a dense archive degrades to "the newest
// K hours, honestly labeled" instead of an unbounded scan: coverage in the
// result says exactly what was and wasn't scanned.
//
// Read-side rules inherited from fullTrackAsync (aircraftTrips.ts): the
// same-second t-dedupe (altitude-bearing fix wins — the 30s poller and the
// day-trace backfill both write tracked hexes) and the hour-file window
// prefilter. Raw hour files exist for the full RAW_RETENTION_DAYS=30 (the
// day-rollup only consumes files PAST retention), so every reachable window
// is hour-file-shaped.

import fs from "fs";
import path from "path";
import { archiveBaseDir, streamJsonlLines } from "./datacoreArchive";

export interface WindowBBox { w: number; s: number; e: number; n: number }

export interface WindowCaps {
  /** most hexes returned per response (honest counter: hexes_seen). */
  maxHexes: number;
  /** most points kept per hex AFTER decimation (per-hex truncated flag). */
  maxPointsPerHex: number;
  /** total points across the response (stops the scan when hit). */
  maxTotalPoints: number;
  /** most hour files streamed per request, newest-first (coverage honesty). */
  maxFiles: number;
}

export const WINDOW_DEFAULT_CAPS: WindowCaps = {
  maxHexes: 300,
  maxPointsPerHex: 600,
  maxTotalPoints: 60_000,
  maxFiles: 192, // 8 days of hour files; longer windows scan newest-first
};

/** T-2 step selector (earth_twin_program.md TIME MACHINE v2 — the human
 *  directive asked for "mins or hours different thing to choose from" as
 *  its own control, not only an automatic function of zoom). The route's
 *  optional `step` query param must be one of these; readWindow uses it in
 *  place of lodStepSec(zoom) when supplied. 3600s (1h) has no zoom-derived
 *  equivalent (lodStepSec tops out at 900s), so this is additive, not a
 *  reachable-via-zoom shortcut. */
export const WINDOW_STEP_OPTIONS_SEC: number[] = [60, 300, 900, 3600];

/** widest window a single request may ask for (the raw archive's span). */
export const WINDOW_MAX_SPAN_SEC = 30 * 86_400;

/** LOD decimation: minimum seconds between kept points per hex, by zoom
 *  (SCALE S1: more map = less per-entity fidelity; the archive loses
 *  nothing — this is render fidelity only). */
export function lodStepSec(zoom: number): number {
  if (zoom >= 9) return 0;    // street-ish: every archived fix
  if (zoom >= 7) return 60;
  if (zoom >= 5) return 300;
  if (zoom >= 3) return 600;
  return 900;                 // world view
}

export interface WindowHex {
  i: string;                       // icao24 (or mmsi for vessels)
  rg?: string;                     // registration (last seen in window)
  c?: string;                      // callsign/name (last seen in window)
  ty?: string;                     // type code (last seen in window)
  /** [t, la, lo, al|null] per kept point, time-ascending. */
  points: Array<[number, number, number, number | null]>;
  /** points in-window∩bbox for this hex BEFORE decimation/caps. */
  raw_count: number;
  truncated: boolean;
}

export interface WindowResult {
  kind: "aircraft" | "vessels";
  from: number;
  to: number;
  zoom: number;
  step_sec: number;
  hexes: WindowHex[];
  hexes_seen: number;
  total_points: number;
  /** scan honesty: which part of the request was actually streamed. */
  coverage: {
    requested_from: number;
    /** oldest hour actually scanned (== requested_from when complete). */
    scanned_from: number;
    complete: boolean;
    files_scanned: number;
  };
  note?: string;
}

/** lon inside [w,e] with antimeridian support (w > e = the seam window). */
export function lonInBBox(lo: number, w: number, e: number): boolean {
  return w <= e ? lo >= w && lo <= e : lo >= w || lo <= e;
}

/** UTC hour-file basename ("YYYY-MM-DD-HH") for an epoch-seconds hour. */
export function hourName(hourStartSec: number): string {
  return new Date(hourStartSec * 1000).toISOString().slice(0, 13).replace("T", "-");
}

interface Accum {
  pts: Array<[number, number, number, number | null]>;
  rg?: string; c?: string; ty?: string;
  rgT: number; cT: number; tyT: number;
}

/**
 * The window read. Pure I/O over the archive directory — injectable baseDir
 * and caps for tests; no cache here (the route owns TTL caching).
 */
export async function readWindow(opts: {
  kind?: "aircraft" | "vessels";
  bbox: WindowBBox;
  fromSec: number;
  toSec: number;
  zoom: number;
  /** T-2: explicit decimation step (seconds), overriding lodStepSec(zoom)
   *  when provided. Must be one of WINDOW_STEP_OPTIONS_SEC — the route
   *  validates; this function just trusts a finite non-negative value. */
  stepSecOverride?: number;
  baseDir?: string;
  caps?: Partial<WindowCaps>;
}): Promise<WindowResult> {
  const kind = opts.kind ?? "aircraft";
  const caps: WindowCaps = { ...WINDOW_DEFAULT_CAPS, ...(opts.caps || {}) };
  const { bbox } = opts;
  const from = Math.floor(opts.fromSec);
  const to = Math.floor(opts.toSec);
  const step = Number.isFinite(opts.stepSecOverride) && (opts.stepSecOverride as number) >= 0
    ? Math.round(opts.stepSecOverride as number)
    : lodStepSec(opts.zoom);
  const base = opts.baseDir || archiveBaseDir();
  const dir = path.join(base, kind);

  const empty = (note: string): WindowResult => ({
    kind, from, to, zoom: opts.zoom, step_sec: step, hexes: [],
    hexes_seen: 0, total_points: 0,
    coverage: { requested_from: from, scanned_from: from, complete: true, files_scanned: 0 },
    note,
  });
  if (!(to > from)) return empty("empty window (to must be after from)");
  if (!fs.existsSync(dir)) return empty("no archive yet for this kind");

  // candidate hour files inside the window, NEWEST first, budget-capped.
  // Names are derived from the window (not readdir) so the scan order is
  // exact; a missing hour (nothing archived / not yet written) just isn't
  // on disk in either flavor.
  const firstHour = Math.floor(from / 3600) * 3600;
  const lastHour = Math.floor((to - 1) / 3600) * 3600;
  const files: Array<{ fp: string; gz: boolean; hourSec: number }> = [];
  for (let h = lastHour; h >= firstHour; h -= 3600) {
    const nm = hourName(h);
    const plain = path.join(dir, `${nm}.jsonl`);
    const gzp = path.join(dir, `${nm}.jsonl.gz`);
    if (fs.existsSync(plain)) files.push({ fp: plain, gz: false, hourSec: h });
    else if (fs.existsSync(gzp)) files.push({ fp: gzp, gz: true, hourSec: h });
  }

  const acc = new Map<string, Accum>();
  let rawMatched = 0;
  let filesScanned = 0;
  let scannedFrom = to; // narrows downward as hours stream
  let stoppedEarly = false;

  for (const f of files) {
    if (filesScanned >= caps.maxFiles || rawMatched >= caps.maxTotalPoints * 4) {
      stoppedEarly = true;
      break;
    }
    filesScanned++;
    scannedFrom = Math.max(from, f.hourSec);
    await streamJsonlLines(f.fp, f.gz, (line) => {
      let r: any;
      try { r = JSON.parse(line); } catch { return; }
      if (typeof r?.i !== "string" || !Number.isFinite(r.t)) return;
      if (r.t < from || r.t > to) return;
      if (!Number.isFinite(r.la) || !Number.isFinite(r.lo)) return;
      if (r.la < bbox.s || r.la > bbox.n || !lonInBBox(r.lo, bbox.w, bbox.e)) return;
      const a = acc.get(r.i) ?? ((): Accum => {
        const na: Accum = { pts: [], rgT: -1, cT: -1, tyT: -1 };
        acc.set(r.i, na);
        return na;
      })();
      a.pts.push([r.t, r.la, r.lo, r.al ?? null]);
      rawMatched++;
      if (typeof r.rg === "string" && r.t > a.rgT) { a.rg = r.rg; a.rgT = r.t; }
      if (typeof r.c === "string" && r.t > a.cT) { a.c = r.c; a.cT = r.t; }
      if (typeof r.ty === "string" && r.t > a.tyT) { a.ty = r.ty; a.tyT = r.t; }
    });
  }
  if (files.length === 0) scannedFrom = from;

  // per-hex: sort → same-second dedupe (altitude wins, the fullTrackAsync
  // rule) → LOD decimation (last point always kept so the track reaches its
  // end) → per-hex cap (newest kept).
  const hexes: WindowHex[] = [];
  for (const [i, a] of Array.from(acc.entries())) {
    a.pts.sort((p, q) => p[0] - q[0]);
    const dedup: Array<[number, number, number, number | null]> = [];
    for (const p of a.pts) {
      const last = dedup[dedup.length - 1];
      if (last && last[0] === p[0]) {
        if (last[3] == null && p[3] != null) dedup[dedup.length - 1] = p;
        continue;
      }
      dedup.push(p);
    }
    let kept = dedup;
    if (step > 0 && dedup.length > 2) {
      kept = [];
      let lastT = -Infinity;
      for (let k = 0; k < dedup.length; k++) {
        const isLast = k === dedup.length - 1;
        if (isLast || dedup[k][0] - lastT >= step) {
          kept.push(dedup[k]);
          lastT = dedup[k][0];
        }
      }
    }
    let truncated = false;
    if (kept.length > caps.maxPointsPerHex) {
      kept = kept.slice(-caps.maxPointsPerHex);
      truncated = true;
    }
    hexes.push({
      i, rg: a.rg, c: a.c, ty: a.ty,
      points: kept, raw_count: dedup.length, truncated,
    });
  }

  // most-active hexes first; cap count honestly
  hexes.sort((x, y) => y.raw_count - x.raw_count);
  const hexesSeen = hexes.length;
  let returned = hexes.slice(0, caps.maxHexes);

  // total-point cap across the response (drop whole tail hexes, never
  // silently thin an included one further)
  let total = 0;
  const capped: WindowHex[] = [];
  for (const h of returned) {
    if (total + h.points.length > caps.maxTotalPoints && capped.length > 0) break;
    total += h.points.length;
    capped.push(h);
  }
  returned = capped;

  const complete = !stoppedEarly && scannedFrom <= Math.max(from, firstHour);
  const notes: string[] = [];
  const noun = kind === "vessels" ? "vessels" : "aircraft";
  if (hexesSeen > returned.length) {
    notes.push(`returned ${returned.length} of ${hexesSeen} ${noun} in this window — zoom in or narrow the time range`);
  }
  if (!complete) {
    notes.push(`scan budget hit: window scanned back to ${new Date(scannedFrom * 1000).toISOString()} (newest-first), not the full requested range`);
  }

  return {
    kind, from, to, zoom: opts.zoom, step_sec: step,
    hexes: returned,
    hexes_seen: hexesSeen,
    total_points: returned.reduce((s, h) => s + h.points.length, 0),
    coverage: {
      requested_from: from,
      scanned_from: scannedFrom,
      complete,
      files_scanned: filesScanned,
    },
    note: notes.length ? notes.join("; ") : undefined,
  };
}
