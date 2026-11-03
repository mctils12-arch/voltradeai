/**
 * portDwell.ts — per-vessel port dwell analytics computed from OUR OWN AIS
 * position archive (fusion directive 2026-07-04: "highest immediate value,
 * zero new data"). Continuous AIS is primary; imagery verifies/enriches
 * when that pipeline lands.
 *
 * RAW vs SIGNAL boundary: what ships now are RAW STATISTICS — arrival/
 * departure detection per port geofence, dwell distributions, and
 * clearly-labeled anomaly FLAGS (dwell >= 3x the port's own median). The
 * dwell-anomaly SIGNAL ("congestion at port X predicts Y") stays off the
 * surface until ladder gate 2 passes against published port-congestion
 * indices (validation plan in research/open_questions.md).
 *
 * Geofences: the 9 imagery-verified port terminals from datacore/sites
 * (REFERENCE DATA ACCURACY — geofence coordinates require verification;
 * these are the only port coordinates in the repo that have it). Radius
 * default 5km around the verified terminal; overlapping fences (LA/Long
 * Beach are ~5.6km apart) are resolved by nearest-port assignment.
 *
 * Honesty notes baked into the output:
 *  - the archive began 2026-07-03; medians are THIN until weeks accumulate,
 *    so anomaly flags are suppressed for ports with < MIN_VISITS_FOR_ANOMALY
 *    completed visits in the window;
 *  - a visit still in progress at the window edge is right-censored: counted
 *    as "in port now," excluded from the dwell distribution;
 *  - terrestrial AIS: a coverage gap inside a visit longer than maxGapHours
 *    splits it into two visits — dwell figures are lower bounds, never
 *    inflated.
 *
 * Pure module: fs reads only (via shadowFleet.readVesselTracks), baseDir-
 * injectable, no trading imports.
 */
import path from "path";
import { readVesselTracks, foldVesselArchiveAsync } from "./shadowFleet";
import { archiveBaseDir } from "./datacoreArchive";
import {
  readFoldMode, runIncrementalFold, PartialCacheRegistry, FoldStatsTracker, StringInterner,
  histAdd, histMergeInto, histMedian, type CachedPartial, type FoldStats, type SpeedHist, type VesselPt,
} from "./vesselArchiveFold";

export interface PortDef { id: string; name: string; lat: number; lon: number; radius_km: number }

export interface PortVisit {
  mmsi: string; name?: string;
  portId: string;
  firstSeen: number;   // unix sec of first in-fence point
  lastSeen: number;    // unix sec of last in-fence point
  dwellHours: number;
  points: number;
  ongoing: boolean;    // still in port at window edge (right-censored)
}

export interface PortDwellPortStats {
  id: string; name: string; lat: number; lon: number;
  visits_completed: number;
  unique_vessels: number;
  in_port_now: number;
  dwell_median_h: number | null;
  dwell_p90_h: number | null;
  dwell_max_h: number | null;
  anomaly_count: number;
  anomaly_examples: Array<{ mmsi: string; name?: string; dwell_h: number; median_h: number }>;
}

export interface PortDwellStats {
  window_hours: number;
  vessels_seen: number;
  visits_completed: number;
  in_port_now: number;
  anomaly_count: number;
  ports: PortDwellPortStats[];
  caveat: string;
}

const DEFAULT_RADIUS_KM = 5;
const MIN_VISITS_FOR_ANOMALY = 10; // thin-history honesty: no anomaly claims off a tiny median
const ANOMALY_FACTOR = 3;          // human directive: "vessel X in port 3x median"

const kmBetween = (aLat: number, aLon: number, bLat: number, bLon: number) => {
  const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
};

/** Ports from the strategic-sites registry (category "port"), fence radius
 *  attached. Kept as a mapper so routes.ts stays a one-liner and tests can
 *  pin the shape. */
export function portsFromSites(sites: Array<{ id: string; name: string; category: string; lat: number; lon: number }>,
                               radiusKm = DEFAULT_RADIUS_KM): PortDef[] {
  return sites.filter((s) => s.category === "port")
    .map((s) => ({ id: s.id, name: s.name, lat: s.lat, lon: s.lon, radius_km: radiusKm }));
}

/** Nearest port whose fence contains the point, or null. Nearest (not first)
 *  resolves overlapping fences deterministically (LA vs Long Beach). */
export function assignPort(lat: number, lon: number, ports: PortDef[]): string | null {
  let best: string | null = null, bestKm = Infinity;
  for (const p of ports) {
    const d = kmBetween(lat, lon, p.lat, p.lon);
    if (d <= p.radius_km && d < bestKm) { best = p.id; bestKm = d; }
  }
  return best;
}

interface Pt { t: number; la: number; lo: number; v?: number; c?: string }

/**
 * Detect port visits for one vessel track (points sorted by time).
 * A visit is a maximal run of same-port points where consecutive in-fence
 * points are <= maxGapHours apart. Filters: >= minPoints points,
 * span >= minDwellHours, median SOG <= maxMedianKts (keeps moored/anchored
 * ships; drops harbor craft and channel transits). A visit whose last point
 * is within maxGapHours of `now` is ongoing (right-censored).
 */
export function detectVisits(mmsi: string, pts: Pt[], ports: PortDef[], nowSec: number,
                             minDwellHours = 2, maxGapHours = 6,
                             minPoints = 3, maxMedianKts = 3): PortVisit[] {
  const out: PortVisit[] = [];
  let cur: { portId: string; pts: Pt[] } | null = null;
  const flush = () => {
    if (!cur) return;
    const { portId, pts: run } = cur;
    cur = null;
    if (run.length < minPoints) return;
    const spanH = (run[run.length - 1].t - run[0].t) / 3600;
    if (spanH < minDwellHours) return;
    const speeds = run.map((p) => p.v ?? 0).sort((a, b) => a - b);
    if (speeds[Math.floor(speeds.length / 2)] > maxMedianKts) return;
    const ongoing = (nowSec - run[run.length - 1].t) / 3600 <= maxGapHours;
    out.push({
      mmsi, name: run[run.length - 1].c || run[0].c, portId,
      firstSeen: run[0].t, lastSeen: run[run.length - 1].t,
      dwellHours: Math.round(spanH * 10) / 10, points: run.length, ongoing,
    });
  };
  for (const p of pts) {
    const portId = assignPort(p.la, p.lo, ports);
    if (!portId) { flush(); continue; }
    if (cur && (cur.portId !== portId || (p.t - cur.pts[cur.pts.length - 1].t) / 3600 > maxGapHours)) flush();
    if (!cur) cur = { portId, pts: [] };
    cur.pts.push(p);
  }
  flush();
  return out;
}

const quantile = (sorted: number[], q: number): number | null => {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
  return sorted[idx];
};

/** The visit filters detectVisits/VisitDetector share. */
export interface VisitParams { minDwellHours: number; maxGapHours: number; minPoints: number; maxMedianKts: number }
export const DEFAULT_VISIT_PARAMS: VisitParams = { minDwellHours: 2, maxGapHours: 6, minPoints: 3, maxMedianKts: 3 };

/** [A2 2026-11-03] One maximal run of same-port in-fence points — the
 *  detector's `cur` state — in a form that needs no per-point storage:
 *  first/last pushed timestamps (span and the gap test use exactly these,
 *  not min/max), the point count, the first and last names, and the speed
 *  multiset as a histogram (vesselArchiveFold.histMedian yields precisely
 *  `sorted[floor(n/2)]`). Two adjacent runs concatenate exactly, which is
 *  what lets a visit spanning many hour files be carried across per-hour
 *  partials. It also removes the old `pts: {t,v,c}[]` per moored vessel —
 *  the fold's own retained-heap peak (a vessel moored a week at a
 *  full-resolution site held ~10k triples). */
interface Run { portId: string; firstT: number; lastT: number; n: number; firstC?: string; lastC?: string; hist: SpeedHist }
const newRun = (portId: string, p: Pt): Run =>
  ({ portId, firstT: p.t, lastT: p.t, n: 1, firstC: p.c, lastC: p.c, hist: new Map([[p.v ?? 0, 1]]) });
const runAdd = (run: Run, p: Pt): void => { run.lastT = p.t; run.n++; run.lastC = p.c; histAdd(run.hist, p.v ?? 0); };
const copyRun = (r: Run): Run => ({ ...r, hist: new Map(r.hist) });
/** `a` then `b`, as if b's points had been pushed onto a's run. */
const concatRuns = (a: Run, b: Run): Run => {
  const hist = new Map(a.hist);
  histMergeInto(hist, b.hist);
  return { portId: a.portId, firstT: a.firstT, lastT: b.lastT, n: a.n + b.n, firstC: a.firstC, lastC: b.lastC, hist };
};

/** A visit minus the two things only the final fold knows: the vessel's
 *  mmsi (the partial's map key) and `ongoing` (depends on the fold's `now`). */
interface PreVisit { portId: string; name?: string; firstSeen: number; lastSeen: number; dwellHours: number; points: number }

/** The visit filters, exactly as detectVisits applies them: >= minPoints,
 *  span >= minDwellHours, median SOG <= maxMedianKts; name = last point's
 *  name, else the first's. Null = this run is not a port call. */
function runToPreVisit(run: Run, P: VisitParams): PreVisit | null {
  if (run.n < P.minPoints) return null;
  const spanH = (run.lastT - run.firstT) / 3600;
  if (spanH < P.minDwellHours) return null;
  if (histMedian(run.hist, run.n) > P.maxMedianKts) return null;
  return { portId: run.portId, name: run.lastC || run.firstC, firstSeen: run.firstT, lastSeen: run.lastT,
           dwellHours: Math.round(spanH * 10) / 10, points: run.n };
}
function preVisitToVisit(v: PreVisit, mmsi: string, nowSec: number, P: VisitParams): PortVisit {
  return {
    mmsi, name: v.name, portId: v.portId,
    firstSeen: v.firstSeen, lastSeen: v.lastSeen,
    dwellHours: v.dwellHours, points: v.points, ongoing: (nowSec - v.lastSeen) / 3600 <= P.maxGapHours,
  };
}

/** [A2 2026-11-03] What one closed hour file contributes to one vessel's
 *  visit detection, in a form that composes EXACTLY onto the detector state
 *  carried in from earlier hours, for ANY point order:
 *   - `head`: the hour's FIRST in-fence run, kept as an unfiltered RUN —
 *     whether it is a port call depends on the run it may continue;
 *   - `headAtStart`: the vessel's first point of the hour was in a fence
 *     (otherwise that point ended any run carried in);
 *   - `headOpen`: the head run never closed inside the hour (it is also the
 *     tail, carried on to the next hour);
 *   - `middle`: runs that both started and closed inside the hour after the
 *     head — independent of anything outside, so already filtered;
 *   - `tail`: the open run at the end of the hour (when it is not the head).
 *  A vessel with points but no in-fence point in the hour is present in
 *  `vesselIds` only: its carried-in run ends at its first point. */
interface FileVesselState { head: Run; headAtStart: boolean; headOpen: boolean; middle: PreVisit[]; tail: Run | null }

export interface PortDwellPartial extends CachedPartial {
  stampSec: number;
  points: number;
  /** every vessel with a point in this hour, in first-appearance order (ids interned by the owning cache) */
  vesselIds: Int32Array;
  /** only vessels with at least one in-fence point this hour */
  inPort: Map<number, FileVesselState>;
}

const PORT_PARTIAL_BYTES_FIXED = 128;
const PORT_STATE_BYTES = 400;   // FileVesselState + head/tail Runs with small histograms
const PORT_PREVISIT_BYTES = 120;

/** [OOM REPAIR 2026-07-05, supersedes the R5 event-loop fix's memory
 *  profile] Online visit state machine — the exact detectVisits
 *  transitions, fed one point at a time. Retains ONLY the current in-port
 *  run per vessel (as a Run — see above; before A2 2026-11-03 as {t,v,c}
 *  triples); the previous async path materialized the ENTIRE 168h archive
 *  (~7M points) into a Map first, which OOM-crash-looped prod under the
 *  512MB heap cap. Output pinned identical by the async-vs-sync ratchet.
 *  [A2] It is also the per-hour partial's builder (`trackHour`) and its
 *  consumer (`absorbHour`), so the transitions have exactly one home. */
class VisitDetector {
  private cur: Run | null = null;
  readonly visits: PortVisit[] = [];
  // partial-builder bookkeeping (trackHour mode only)
  private sawFirst = false;
  private headAtStart = false;
  private head: Run | null = null;
  private headOpen = false;
  private middle: PreVisit[] = [];
  private runsStarted = 0;
  constructor(private mmsi: string, private ports: PortDef[], private nowSec: number,
              private P: VisitParams = DEFAULT_VISIT_PARAMS, private trackHour = false) {}
  private flush(): void {
    if (!this.cur) return;
    const run = this.cur;
    this.cur = null;
    if (this.trackHour && this.runsStarted === 1) { this.head = run; return; } // the head stays a RUN
    const v = runToPreVisit(run, this.P);
    if (!v) return;
    if (this.trackHour) this.middle.push(v);
    else this.visits.push(preVisitToVisit(v, this.mmsi, this.nowSec, this.P));
  }
  push(p: Pt): void {
    const portId = assignPort(p.la, p.lo, this.ports);
    if (!this.sawFirst) { this.sawFirst = true; this.headAtStart = portId != null; }
    if (!portId) { this.flush(); return; }
    if (this.cur && (this.cur.portId !== portId || (p.t - this.cur.lastT) / 3600 > this.P.maxGapHours)) this.flush();
    if (!this.cur) { this.cur = newRun(portId, p); this.runsStarted++; }
    else runAdd(this.cur, p);
  }
  finish(): PortVisit[] {
    this.flush();
    return this.visits;
  }
  /** [A2] Builder mode: this vessel's contribution for the hour just fed, or
   *  null when it never entered a fence. */
  hourState(): FileVesselState | null {
    if (!this.trackHour) throw new Error("hourState needs a detector built with trackHour");
    if (this.cur) {
      if (this.runsStarted === 1) { this.head = this.cur; this.headOpen = true; this.cur = null; }
      // else: the open run is the tail, left in `cur` and read below
    }
    if (!this.head) return null;
    const tail = this.cur;
    this.cur = null;
    return { head: this.head, headAtStart: this.headAtStart, headOpen: this.headOpen, middle: this.middle, tail };
  }
  /** [A2] Consumer mode: applies one closed hour's state for this vessel
   *  EXACTLY as pushing that hour's points would have. `null` = the vessel
   *  had points this hour but none in a fence. */
  absorbHour(fs: FileVesselState | null): void {
    if (!fs) { this.flush(); return; }
    const carried = this.cur;
    let head: Run;
    if (carried && fs.headAtStart && carried.portId === fs.head.portId &&
        (fs.head.firstT - carried.lastT) / 3600 <= this.P.maxGapHours) {
      head = concatRuns(carried, fs.head);       // the hour's first run continues the carried one
      this.cur = null;
    } else {
      this.flush();                              // the carried run ended (off-fence point, other port, or a gap)
      head = copyRun(fs.head);
    }
    if (fs.headOpen) { this.cur = head; return; }  // still in port at the hour's end
    this.cur = head;
    this.flush();                                  // closed inside the hour: filtered now, with its true start
    for (const v of fs.middle) this.visits.push(preVisitToVisit(v, this.mmsi, this.nowSec, this.P));
    this.cur = fs.tail ? copyRun(fs.tail) : null;
  }
}

/** [A2 2026-11-03] The whole fold's state (one VisitDetector per vessel, in
 *  first-appearance order — aggregateVisits' tie-breaks depend on that
 *  order, so the incremental fold keeps it) — fed either by points (push)
 *  or by cached hour partials (absorbPartial). */
class PortFoldState {
  readonly detectors = new Map<string, VisitDetector>();
  points = 0;
  constructor(private ports: PortDef[], private nowSec: number, private P: VisitParams = DEFAULT_VISIT_PARAMS,
              private trackHour = false) {}
  private detector(mmsi: string): VisitDetector {
    let d = this.detectors.get(mmsi);
    if (!d) { d = new VisitDetector(mmsi, this.ports, this.nowSec, this.P, this.trackHour); this.detectors.set(mmsi, d); }
    return d;
  }
  push(mmsi: string, p: Pt): void {
    this.points++;
    this.detector(mmsi).push(p);
  }
  /** Exact for any point order — no precondition, so this never fails. */
  absorbPartial(partial: PortDwellPartial, interner: StringInterner): boolean {
    const ids = partial.vesselIds;
    for (let k = 0; k < ids.length; k++) {
      const id = ids[k];
      this.detector(interner.str(id)).absorbHour(partial.inPort.get(id) ?? null);
    }
    this.points += partial.points;
    return true;
  }
  /** Builder mode: the partial for the hour this state was fed. */
  exportPartial(stampSec: number, interner: StringInterner): PortDwellPartial {
    const vesselIds = new Int32Array(this.detectors.size);
    const inPort = new Map<number, FileVesselState>();
    let i = 0, preVisits = 0;
    this.detectors.forEach((d, mmsi) => {
      const id = interner.id(mmsi);
      vesselIds[i++] = id;
      const fs = d.hourState();
      if (fs) { inPort.set(id, fs); preVisits += fs.middle.length; }
    });
    return {
      stampSec, points: this.points, vesselIds, inPort,
      bytes: PORT_PARTIAL_BYTES_FIXED + vesselIds.byteLength + inPort.size * PORT_STATE_BYTES + preVisits * PORT_PREVISIT_BYTES,
      remapIds(map) {
        for (let k = 0; k < this.vesselIds.length; k++) this.vesselIds[k] = map(this.vesselIds[k]);
        const next = new Map<number, FileVesselState>();
        this.inPort.forEach((v, k) => next.set(map(k), v));
        this.inPort = next;
      },
    };
  }
  finish(): { visitsByPort: Map<string, PortVisit[]>; vesselsSeen: number; pointsScanned: number } {
    const visitsByPort = new Map<string, PortVisit[]>();
    for (const p of this.ports) visitsByPort.set(p.id, []);
    this.detectors.forEach((d) => {
      for (const v of d.finish()) visitsByPort.get(v.portId)?.push(v);
    });
    return { visitsByPort, vesselsSeen: this.detectors.size, pointsScanned: this.points };
  }
}

/** Shared online fold — extracted from computePortDwellAsync (2026-07-06,
 *  Everything Graph step 2) so the calls_at edge builder can reuse the same
 *  bounded-memory archive pass instead of re-implementing it. Behavior of
 *  computePortDwellAsync is unchanged (same fold, same aggregateVisits
 *  call) — pinned by the existing portDwell tests. `pointsScanned` (added
 *  2026-09-06, diagnosing the portdwell_window diag probe's live timeouts
 *  at hours>=48 -- see computePortDwellAsyncTimed below) is a pure counter
 *  of onPoint invocations, purely additive: both existing callers
 *  (computePortDwellAsync, entityGraph.ts's calls_at builder) destructure
 *  only the fields they already used, so this changes no existing
 *  behavior. */
// ── [A2 2026-11-03] incremental fold wiring ─────────────────────────────────
//
// VOLTRADE_PORTDWELL_FOLD = full | shadow (default) | incremental — see
// vesselArchiveFold.ts's header for the contract. Rollback = set `full`.
// Switch to `incremental` only after 24h of zero `FOLD-MISMATCH portdwell`
// lines (portDwellFoldStats().mismatches stays 0 on /api/health). Governs
// every caller of foldPortVisitsAsync (computePortDwellAsync, the timed diag
// variant, entityGraph's calls_at builder) — they share one partial cache
// per (archive dir, ports) so the second caller's closed hours are free.
export const PORTDWELL_FOLD_ENV = "VOLTRADE_PORTDWELL_FOLD";
const portFoldRegistry = new PartialCacheRegistry<PortDwellPartial>();
const portFoldStats = new FoldStatsTracker();

/** Cheap snapshot for the /api/health diagnostic. */
export function portDwellFoldStats(): FoldStats { return portFoldStats.snapshot(); }
export function _resetPortDwellFoldForTests(): void { portFoldRegistry.clear(); portFoldStats.reset(); }
/** Partials depend on the geofences and the visit filters, so they key the cache. */
function portFoldCacheKey(baseDir: string | undefined, ports: PortDef[]): string {
  return `${path.join(baseDir || archiveBaseDir(), "vessels")}|ports=${JSON.stringify(ports)}|params=${JSON.stringify(DEFAULT_VISIT_PARAMS)}`;
}
/** The live cache for one archive dir + port set — tests poison it to prove
 *  the shadow compare catches a wrong incremental result. */
export function _portDwellFoldCacheForTests(baseDir: string | undefined, ports: PortDef[]) {
  return portFoldRegistry.forKey(portFoldCacheKey(baseDir, ports));
}

export async function foldPortVisitsAsync(ports: PortDef[], windowHours = 168,
                                          baseDir?: string, nowMs?: number):
    Promise<{ visitsByPort: Map<string, PortVisit[]>; vesselsSeen: number; pointsScanned: number }> {
  const now = nowMs ?? Date.now();
  const nowSec = Math.floor(now / 1000);
  const mode = readFoldMode(PORTDWELL_FOLD_ENV);
  if (mode === "full") {
    const t0 = performance.now();
    const state = new PortFoldState(ports, nowSec);
    await foldVesselArchiveAsync(windowHours, (mmsi, p) => state.push(mmsi, p), baseDir, now);
    const r = state.finish();
    portFoldStats.update({ mode, lastFullMs: Math.round(performance.now() - t0), lastRunAt: new Date().toISOString(), ...portFoldRegistry.totals() });
    return r;
  }
  const cache = portFoldRegistry.forKey(portFoldCacheKey(baseDir, ports));
  return runIncrementalFold<PortFoldState, PortDwellPartial, ReturnType<PortFoldState["finish"]>>({
    stream: "portdwell", mode, windowHours, baseDir, nowMs: now, cache, stats: portFoldStats,
    registryTotals: () => portFoldRegistry.totals(),
    newState: () => new PortFoldState(ports, nowSec),
    push: (state, mmsi, p: VesselPt) => state.push(mmsi, p),
    newBuilder: (file, interner) => {
      const b = new PortFoldState(ports, nowSec, DEFAULT_VISIT_PARAMS, true);
      return { push: (mmsi, p) => b.push(mmsi, p), finish: () => b.exportPartial(file.stampSec, interner) };
    },
    absorb: (state, partial, interner) => state.absorbPartial(partial, interner),
    finish: (state) => state.finish(),
  });
}

/** Async variant — the only one routes may use. Online fold, bounded
 *  memory; the sync path stays for tests, pinned by the ratchet. */
export async function computePortDwellAsync(ports: PortDef[], windowHours = 168,
                                            baseDir?: string, nowMs?: number): Promise<PortDwellStats> {
  const { visitsByPort, vesselsSeen } = await foldPortVisitsAsync(ports, windowHours, baseDir, nowMs);
  return aggregateVisits(visitsByPort, ports, vesselsSeen, windowHours);
}

/** [ADDED 2026-09-06] Same computation as computePortDwellAsync, wrapped
 *  with wall-clock timing and a point-count so the portdwell_window diag
 *  probe can report where time actually goes. Built to root-cause a live
 *  finding this session: hours=168 (the weekly-snapshot script's own
 *  window) now reliably exceeds Railway's edge-proxy response timeout
 *  (observed 46-71s before a connection reset/502, at hours=24/48/168
 *  alike -- /api/health stayed fully responsive throughout every probe,
 *  ruling out an event-loop stall or process crash), while the archive
 *  has grown since this path was last exercised successfully
 *  (2026-08-14/2026-09-03/09-04 weekly captures). This does NOT fix the
 *  timeout -- it is deliberately the smallest safe next step: measure
 *  before guessing at a concurrency change to the SHARED, order-sensitive
 *  foldVesselArchiveAsync (also consumed by shadowFleet.ts's dark-vessel
 *  detection and gridStress.ts, both of which depend on its "points
 *  arrive per-vessel in near-chronological order" contract -- see that
 *  function's own header). A future session should call this via a
 *  window short enough to survive the proxy timeout (e.g. hours=24,
 *  confirmed live to complete), read elapsedMs/pointsScanned, and only
 *  then decide whether the bottleneck is CPU-bound parsing (favors
 *  reducing total work) or I/O-bound file-open latency (favors bounded,
 *  order-preserving concurrent prefetch) before changing the shared fold. */
export async function computePortDwellAsyncTimed(ports: PortDef[], windowHours = 168,
                                                  baseDir?: string, nowMs?: number):
    Promise<PortDwellStats & { pointsScanned: number; elapsedMs: number }> {
  const start = Date.now();
  const { visitsByPort, vesselsSeen, pointsScanned } = await foldPortVisitsAsync(ports, windowHours, baseDir, nowMs);
  const stats = aggregateVisits(visitsByPort, ports, vesselsSeen, windowHours);
  return { ...stats, pointsScanned, elapsedMs: Date.now() - start };
}

export function computePortDwell(ports: PortDef[], windowHours = 168,
                                 baseDir?: string, nowMs?: number): PortDwellStats {
  const now = nowMs ?? Date.now();
  const tracks = readVesselTracks(windowHours, baseDir, now);
  return dwellFromTracks(tracks, ports, now, windowHours);
}

function dwellFromTracks(tracks: Map<string, Pt[]>, ports: PortDef[], now: number, windowHours: number): PortDwellStats {
  const nowSec = Math.floor(now / 1000);
  const visitsByPort = new Map<string, PortVisit[]>();
  for (const p of ports) visitsByPort.set(p.id, []);
  for (const [mmsi, pts] of tracks) {
    for (const v of detectVisits(mmsi, pts, ports, nowSec)) {
      visitsByPort.get(v.portId)?.push(v);
    }
  }
  let vesselsSeen = 0;
  for (const _ of tracks) vesselsSeen++;
  return aggregateVisits(visitsByPort, ports, vesselsSeen, windowHours);
}

/** Shared aggregation tail — one engine for the sync (test) and online
 *  (prod) paths so the stats math cannot diverge. */
function aggregateVisits(visitsByPort: Map<string, PortVisit[]>, ports: PortDef[],
                         vesselsSeen: number, windowHours: number): PortDwellStats {
  const portStats: PortDwellPortStats[] = [];
  let totCompleted = 0, totOngoing = 0, totAnomalies = 0;
  for (const p of ports) {
    const visits = visitsByPort.get(p.id) || [];
    const completed = visits.filter((v) => !v.ongoing);
    const ongoing = visits.filter((v) => v.ongoing);
    const dwells = completed.map((v) => v.dwellHours).sort((a, b) => a - b);
    const median = quantile(dwells, 0.5);
    let anomalies: PortDwellPortStats["anomaly_examples"] = [];
    if (median != null && completed.length >= MIN_VISITS_FOR_ANOMALY) {
      anomalies = completed
        .filter((v) => v.dwellHours >= ANOMALY_FACTOR * median)
        .sort((a, b) => b.dwellHours - a.dwellHours)
        .map((v) => ({ mmsi: v.mmsi, name: v.name, dwell_h: v.dwellHours, median_h: median }));
    }
    totCompleted += completed.length;
    totOngoing += ongoing.length;
    totAnomalies += anomalies.length;
    portStats.push({
      id: p.id, name: p.name, lat: p.lat, lon: p.lon,
      visits_completed: completed.length,
      unique_vessels: new Set(visits.map((v) => v.mmsi)).size,
      in_port_now: ongoing.length,
      dwell_median_h: median,
      dwell_p90_h: quantile(dwells, 0.9),
      dwell_max_h: dwells.length ? dwells[dwells.length - 1] : null,
      anomaly_count: anomalies.length,
      anomaly_examples: anomalies.slice(0, 3),
    });
  }

  return {
    window_hours: windowHours,
    vessels_seen: vesselsSeen,
    visits_completed: totCompleted,
    in_port_now: totOngoing,
    anomaly_count: totAnomalies,
    ports: portStats,
    caveat: "RAW statistics from our own terrestrial-AIS archive (began 2026-07-03). " +
            "Dwell figures are lower bounds (coverage gaps split visits); medians are " +
            "thin until weeks of history accumulate, so anomaly flags are suppressed " +
            `below ${MIN_VISITS_FOR_ANOMALY} completed calls per port. The dwell-anomaly ` +
            "SIGNAL is ladder-gated (gate 2 vs published congestion indices; see " +
            "research/open_questions.md).",
  };
}

// ── rollup-based (coarse, months-deep) port presence ────────────────────────
//
// GATE 1 (research/open_questions.md "PORT DWELL ANALYTICS"): the raw-archive
// pipeline above (detectVisits/computePortDwellAsync) can only ever see the
// last RAW_RETENTION_DAYS (30) — the 2026-08-19 GATE 1 session found the
// specific July-2026-vs-Port-of-LA-TEU comparison this root has queued since
// 2026-08-04 "CLOSED AS PERMANENTLY UNATTAINABLE via this data path" for
// exactly that reason, and filed "build a rollup-summary-format reader" as
// the alternative NEXT step (never attempted until this session).
//
// datacoreArchive.ts's rollupOldDaysAsync does not delete data past
// RAW_RETENTION_DAYS — it folds each entity's raw hour-points for that day
// into ONE summary row (`vessels_tracks/<day>.jsonl.gz`: {i, d, n, t0, t1,
// bbox, pl: up to ~50 index-subsampled [lat,lon] points}) and keeps that
// forever. This is real, queryable history reaching back to the archive's
// 2026-07-03 start (confirmed live this session via
// `/api/diag/archive?stream=vessels_tracks&day=2026-07-15`) — it just isn't
// PER-POINT-TIMESTAMPED, so detectVisits's exact-hour dwell computation
// cannot run on it directly.
//
// portPresenceFromRollup trades that precision for reach: a coarse,
// DAY-granularity presence detector instead of an hour-granularity dwell
// one. Same lower-bound honesty posture as detectVisits' own dwell figures,
// just at day resolution instead of hour resolution.
export interface RollupTrackRow {
  i: string; d: string; n: number; t0: number; t1: number;
  bbox: [number, number, number, number]; pl: Array<[number, number]>;
}

export interface RollupPortPresence {
  mmsi: string; portId: string;
  firstDay: string; lastDay: string;
  daysPresent: number;  // count of days in this run with >=1 in-fence sampled point
  ongoing: boolean;      // lastDay is the queried window's own last day (right-censored)
}

function daysBetweenIso(a: string, b: string): number {
  return Math.round((Date.parse(`${b}T00:00:00Z`) - Date.parse(`${a}T00:00:00Z`)) / 86_400_000);
}

/**
 * Coarse port-presence detector over ROLLUP-summary track rows for ONE
 * vessel, already sorted by day ascending (caller's job — this function
 * does not sort, so a multi-vessel caller can group once and reuse the
 * order the archive read already produced).
 *
 * A "presence day" is: at least one of that day's (up to ~50, index-
 * subsampled) polyline points falls inside a port's geofence
 * (`assignPort`). A "call" is a maximal run of CONSECUTIVE calendar days
 * with the SAME vessel present at the SAME port — a single day without a
 * detected in-fence point (whether the vessel truly left, or the day's
 * subsample simply missed the one in-fence point) ends the run. No
 * gap-bridging, matching detectVisits' own conservative posture on the raw
 * side — this is a LOWER BOUND on true dwell length and, symmetrically, can
 * OVER-count call count (one real multi-day call split by a missed day
 * reads as two shorter calls) — both biases are stated here, not corrected,
 * per MEASUREMENT INTEGRITY.
 */
export function portPresenceFromRollup(
  rowsByDayAscending: RollupTrackRow[], ports: PortDef[], lastWindowDay: string,
): RollupPortPresence[] {
  const out: RollupPortPresence[] = [];
  let cur: { mmsi: string; portId: string; firstDay: string; lastDay: string; daysPresent: number } | null = null;
  let prevDay: string | null = null;
  const flush = () => {
    if (!cur) return;
    out.push({ ...cur, ongoing: cur.lastDay === lastWindowDay });
    cur = null;
  };
  for (const row of rowsByDayAscending) {
    let portId: string | null = null;
    for (const [la, lo] of row.pl) { portId = assignPort(la, lo, ports); if (portId) break; }
    const gapDays = prevDay != null ? daysBetweenIso(prevDay, row.d) : 0;
    if (cur && (cur.portId !== portId || gapDays > 1)) flush();
    if (portId) {
      if (!cur) cur = { mmsi: row.i, portId, firstDay: row.d, lastDay: row.d, daysPresent: 1 };
      else { cur.lastDay = row.d; cur.daysPresent++; }
    }
    prevDay = row.d;
  }
  flush();
  return out;
}

/** Aggregates portPresenceFromRollup's per-vessel runs across an entire
 *  fetched window into the same shape of summary computePortDwellAsync
 *  reports (visits_completed/unique_vessels/in_port_now), so a GATE 1
 *  script can compare rollup-derived and raw-derived readings order-of-
 *  magnitude-for-order-of-magnitude. `rowsByVessel` values must each
 *  already be sorted by day ascending (same contract as
 *  portPresenceFromRollup itself). */
export function summarizeRollupPresence(
  rowsByVessel: Map<string, RollupTrackRow[]>, ports: PortDef[], lastWindowDay: string,
): Record<string, { visits_completed: number; unique_vessels: number; in_port_now: number; calls: RollupPortPresence[] }> {
  const byPort: Record<string, { completed: RollupPortPresence[]; ongoing: RollupPortPresence[]; vessels: Set<string> }> = {};
  for (const p of ports) byPort[p.id] = { completed: [], ongoing: [], vessels: new Set() };
  for (const rows of rowsByVessel.values()) {
    for (const run of portPresenceFromRollup(rows, ports, lastWindowDay)) {
      const bucket = byPort[run.portId];
      if (!bucket) continue;
      bucket.vessels.add(run.mmsi);
      (run.ongoing ? bucket.ongoing : bucket.completed).push(run);
    }
  }
  const out: Record<string, { visits_completed: number; unique_vessels: number; in_port_now: number; calls: RollupPortPresence[] }> = {};
  for (const [portId, b] of Object.entries(byPort)) {
    out[portId] = {
      visits_completed: b.completed.length, unique_vessels: b.vessels.size,
      in_port_now: b.ongoing.length, calls: [...b.completed, ...b.ongoing],
    };
  }
  return out;
}
