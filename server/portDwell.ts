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
import { readVesselTracks, foldVesselArchiveAsync } from "./shadowFleet";

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

/** [OOM REPAIR 2026-07-05, supersedes the R5 event-loop fix's memory
 *  profile] Online visit state machine — the exact detectVisits
 *  transitions, fed one point at a time. Retains ONLY the current in-port
 *  run per vessel ({t,v,c} triples — in-port points are a tiny, bounded
 *  subset); the previous async path materialized the ENTIRE 168h archive
 *  (~7M points) into a Map first, which OOM-crash-looped prod under the
 *  512MB heap cap. Output pinned identical by the async-vs-sync ratchet. */
class VisitDetector {
  private cur: { portId: string; pts: Array<{ t: number; v?: number; c?: string }> } | null = null;
  readonly visits: PortVisit[] = [];
  constructor(private mmsi: string, private ports: PortDef[], private nowSec: number,
              private minDwellHours = 2, private maxGapHours = 6,
              private minPoints = 3, private maxMedianKts = 3) {}
  private flush(): void {
    if (!this.cur) return;
    const { portId, pts: run } = this.cur;
    this.cur = null;
    if (run.length < this.minPoints) return;
    const spanH = (run[run.length - 1].t - run[0].t) / 3600;
    if (spanH < this.minDwellHours) return;
    const speeds = run.map((p) => p.v ?? 0).sort((a, b) => a - b);
    if (speeds[Math.floor(speeds.length / 2)] > this.maxMedianKts) return;
    const ongoing = (this.nowSec - run[run.length - 1].t) / 3600 <= this.maxGapHours;
    this.visits.push({
      mmsi: this.mmsi, name: run[run.length - 1].c || run[0].c, portId,
      firstSeen: run[0].t, lastSeen: run[run.length - 1].t,
      dwellHours: Math.round(spanH * 10) / 10, points: run.length, ongoing,
    });
  }
  push(p: Pt): void {
    const portId = assignPort(p.la, p.lo, this.ports);
    if (!portId) { this.flush(); return; }
    if (this.cur && (this.cur.portId !== portId ||
        (p.t - this.cur.pts[this.cur.pts.length - 1].t) / 3600 > this.maxGapHours)) this.flush();
    if (!this.cur) this.cur = { portId, pts: [] };
    this.cur.pts.push({ t: p.t, v: p.v, c: p.c });
  }
  finish(): PortVisit[] {
    this.flush();
    return this.visits;
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
export async function foldPortVisitsAsync(ports: PortDef[], windowHours = 168,
                                          baseDir?: string, nowMs?: number):
    Promise<{ visitsByPort: Map<string, PortVisit[]>; vesselsSeen: number; pointsScanned: number }> {
  const now = nowMs ?? Date.now();
  const nowSec = Math.floor(now / 1000);
  const detectors = new Map<string, VisitDetector>();
  let pointsScanned = 0;
  await foldVesselArchiveAsync(windowHours, (mmsi, p) => {
    pointsScanned++;
    let d = detectors.get(mmsi);
    if (!d) { d = new VisitDetector(mmsi, ports, nowSec); detectors.set(mmsi, d); }
    d.push(p);
  }, baseDir, now);
  const visitsByPort = new Map<string, PortVisit[]>();
  for (const p of ports) visitsByPort.set(p.id, []);
  detectors.forEach((d) => {
    for (const v of d.finish()) visitsByPort.get(v.portId)?.push(v);
  });
  return { visitsByPort, vesselsSeen: detectors.size, pointsScanned };
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
