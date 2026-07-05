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

/** Async variant — the only one routes may use. Online fold, bounded
 *  memory; the sync path stays for tests, pinned by the ratchet. */
export async function computePortDwellAsync(ports: PortDef[], windowHours = 168,
                                            baseDir?: string, nowMs?: number): Promise<PortDwellStats> {
  const now = nowMs ?? Date.now();
  const nowSec = Math.floor(now / 1000);
  const detectors = new Map<string, VisitDetector>();
  await foldVesselArchiveAsync(windowHours, (mmsi, p) => {
    let d = detectors.get(mmsi);
    if (!d) { d = new VisitDetector(mmsi, ports, nowSec); detectors.set(mmsi, d); }
    d.push(p);
  }, baseDir, now);
  const visitsByPort = new Map<string, PortVisit[]>();
  for (const p of ports) visitsByPort.set(p.id, []);
  detectors.forEach((d) => {
    for (const v of d.finish()) visitsByPort.get(v.portId)?.push(v);
  });
  return aggregateVisits(visitsByPort, ports, detectors.size, windowHours);
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
