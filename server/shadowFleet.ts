/**
 * shadowFleet.ts — dark-ship analytics computed from OUR OWN AIS position
 * archive (datacore; ARCHIVE EVERYTHING pays off: this consumes the vessel
 * JSONL that has been accumulating since 2026-07-03).
 *
 * RAW vs SIGNAL boundary (Map v2.2 directive): interpreted per-vessel claims
 * ("this ship is shadow fleet") are SIGNAL-class and stay OFF the surface
 * until ROOT VALIDATION LADDER gate 1+2 pass (validation plan in
 * research/open_questions.md). What ships now are clearly-labeled RAW
 * STATISTICS — event counts with honest caveats:
 *   - a "gap event" is transponder-silent-then-reappeared-far in OUR archive;
 *     terrestrial-AIS coverage loss produces the same pattern (mid-ocean is
 *     dark to us) — that ambiguity is exactly what gate 1 must resolve;
 *   - an "identity candidate" is a heuristic match (a new MMSI appearing near
 *     where another went dark, or a name reappearing under a new MMSI);
 *   - "loitering" is sustained slow movement inside a known STS/transshipment
 *     zone (zones from public reporting, datacore/shadow_zones.json).
 *
 * Pure module: fs reads only, baseDir-injectable, no trading imports.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir } from "./datacoreArchive";

export interface ShadowZone { id: string; name: string; lat: number; lon: number; radius_km: number }

export interface GapEvent {
  mmsi: string; name?: string;
  darkAt: number; reappearAt: number;
  gapHours: number; distanceKm: number;
  from: [number, number]; to: [number, number];
}

export interface ShadowStats {
  window_hours: number;
  vessels_seen: number;
  points_read: number;
  gap_events: number;
  gap_examples: GapEvent[];
  identity_candidates: number;
  loiter_events: number;
  loiter_by_zone: Record<string, number>;
  caveat: string;
}

const kmBetween = (aLat: number, aLon: number, bLat: number, bLon: number) => {
  const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
};

interface Pt { t: number; la: number; lo: number; v?: number; c?: string }

/** Reads every vessel point in the lookback window, grouped per MMSI. */
export function readVesselTracks(windowHours: number, baseDir?: string, nowMs?: number): Map<string, Pt[]> {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "vessels");
  const now = nowMs ?? Date.now();
  const cutoff = Math.floor((now - windowHours * 3600_000) / 1000);
  const tracks = new Map<string, Pt[]>();
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return tracks; }
  for (const f of files) {
    // hour files: YYYY-MM-DD-HH.jsonl(.gz) — skip whole files outside window
    const stamp = Date.parse(`${f.slice(0, 10)}T${f.slice(11, 13)}:00:00Z`);
    if (!Number.isFinite(stamp) || stamp < now - (windowHours + 1) * 3600_000) continue;
    let text: string;
    try {
      const raw = fs.readFileSync(path.join(dir, f));
      text = f.endsWith(".gz") ? zlib.gunzipSync(raw).toString("utf8") : raw.toString("utf8");
    } catch { continue; }
    for (const line of text.split("\n")) {
      if (!line) continue;
      try {
        const p = JSON.parse(line);
        if (p.t < cutoff || p.la == null || p.lo == null || !p.i) continue;
        let arr = tracks.get(p.i);
        if (!arr) { arr = []; tracks.set(p.i, arr); }
        arr.push({ t: p.t, la: p.la, lo: p.lo, v: p.v, c: p.c });
      } catch {}
    }
  }
  for (const arr of tracks.values()) arr.sort((a, b) => a.t - b.t);
  return tracks;
}

export function detectGapEvents(tracks: Map<string, Pt[]>,
                                minGapHours = 6, minDistanceKm = 100): GapEvent[] {
  const out: GapEvent[] = [];
  for (const [mmsi, pts] of tracks) {
    for (let i = 1; i < pts.length; i++) {
      const dtH = (pts[i].t - pts[i - 1].t) / 3600;
      if (dtH < minGapHours) continue;
      const dKm = kmBetween(pts[i - 1].la, pts[i - 1].lo, pts[i].la, pts[i].lo);
      if (dKm < minDistanceKm) continue;
      out.push({
        mmsi, name: pts[i].c || pts[i - 1].c,
        darkAt: pts[i - 1].t, reappearAt: pts[i].t,
        gapHours: Math.round(dtH * 10) / 10, distanceKm: Math.round(dKm),
        from: [pts[i - 1].la, pts[i - 1].lo], to: [pts[i].la, pts[i].lo],
      });
    }
  }
  return out.sort((a, b) => b.gapHours - a.gapHours);
}

export interface TimedPoint { mmsi: string; t: number; la: number; lo: number }

/** [REPAIR 2026-07-13, KNOWN BROKEN #18 root cause] Hull-swap candidate count
 *  for the (lastPts x firstPts) pairing, WITHOUT the O(n^2) all-pairs scan
 *  the brute-force version used (~1.26 BILLION Map.get calls/refresh at
 *  production archive size, ~85-98s of synchronous event-loop time on the
 *  exact 10min refreshShadowStats cadence in routes.ts — the confirmed root
 *  cause of KNOWN BROKEN #18's stalls, growing with vessels_seen over time;
 *  portdwell's own 10-min refresh is unaffected, its VisitDetector fold is
 *  already linear per-point, no all-pairs scan).
 *
 *  Uses a spatial grid over `firstPts`, cell size chosen so every cell is
 *  >= nearKm wide in BOTH dimensions everywhere in the actual dataset:
 *  latCellDeg = nearKm/KM_PER_DEG_LAT (exact); lonCellDeg is widened by
 *  1/cos(maxAbsLat) (using the dataset's own highest absolute latitude, so
 *  it's never too narrow for any point actually present) and snapped to
 *  evenly divide 360 degrees for clean antimeridian wraparound. Since two
 *  points within nearKm of each other cannot differ by more than nearKm in
 *  EITHER the north-south or east-west component (each is a lower bound on
 *  the true great-circle distance), and every cell covers at least nearKm
 *  in both dimensions, any true match is guaranteed to fall in the query
 *  point's own cell or one of its 8 immediate neighbors (with longitude
 *  wraparound at +-180) — a 3x3 lookup can never miss a match, and the
 *  final dtH/kmBetween checks are the SAME exact conditions the brute-force
 *  scan used, so the count is identical, just reached without ever
 *  enumerating physically-distant or out-of-time-window pairs. */
const KM_PER_DEG_LAT = (6371 * Math.PI) / 180; // matches kmBetween's R=6371

export function countHullSwapCandidates(lastPts: TimedPoint[], firstPts: TimedPoint[],
                                 nearKm: number, withinHours: number): number {
  if (firstPts.length === 0 || lastPts.length === 0) return 0;

  let maxAbsLat = 0;
  for (const p of firstPts) if (Math.abs(p.la) > maxAbsLat) maxAbsLat = Math.abs(p.la);
  for (const p of lastPts) if (Math.abs(p.la) > maxAbsLat) maxAbsLat = Math.abs(p.la);
  const latCellDeg = nearKm / KM_PER_DEG_LAT;
  // Floor cos() at a small epsilon so a (realistically never-hit) near-pole
  // point can't blow up the cell width to infinity/NaN.
  const cosFloor = Math.max(Math.cos(Math.min(maxAbsLat, 89) * Math.PI / 180), 0.05);
  const lonBuckets = Math.max(1, Math.ceil(360 / (latCellDeg / cosFloor)));
  const lonCellDeg = 360 / lonBuckets; // evenly divides 360 -> exact wraparound

  const latBucketOf = (la: number) => Math.floor(la / latCellDeg);
  const lonBucketOf = (lo: number) => {
    const norm = ((lo + 180) % 360 + 360) % 360; // normalize to [0, 360)
    return Math.floor(norm / lonCellDeg) % lonBuckets;
  };

  const grid = new Map<string, TimedPoint[]>();
  for (const p of firstPts) {
    const key = `${latBucketOf(p.la)}:${lonBucketOf(p.lo)}`;
    let cell = grid.get(key);
    if (!cell) { cell = []; grid.set(key, cell); }
    cell.push(p);
  }

  let count = 0;
  for (const A of lastPts) {
    const latIdx = latBucketOf(A.la);
    const lonIdx = lonBucketOf(A.lo);
    for (let dLat = -1; dLat <= 1; dLat++) {
      const lonWrapped = ((lonIdx + lonBuckets) % lonBuckets); // reused per dLon below
      for (let dLon = -1; dLon <= 1; dLon++) {
        const cell = grid.get(`${latIdx + dLat}:${(lonWrapped + dLon + lonBuckets) % lonBuckets}`);
        if (!cell) continue;
        for (const B of cell) {
          if (B.mmsi === A.mmsi) continue;
          const dtH = (B.t - A.t) / 3600;
          if (dtH <= 0 || dtH > withinHours) continue;
          if (kmBetween(A.la, A.lo, B.la, B.lo) <= nearKm) count++;
        }
      }
    }
  }
  return count;
}

/** Identity-change candidates: (a) a name seen under two different MMSIs in
 *  the window; (b) an MMSI first-seen within `nearKm` and `withinHours` of
 *  another MMSI's last-seen point (hull-swap heuristic). Conservative counts
 *  only — candidates, never claims. */
export function detectIdentityCandidates(tracks: Map<string, Pt[]>,
                                         nearKm = 20, withinHours = 12): number {
  let count = 0;
  const byName = new Map<string, Set<string>>();
  for (const [mmsi, pts] of tracks) {
    for (const p of pts) {
      if (!p.c) continue;
      let s = byName.get(p.c);
      if (!s) { s = new Set(); byName.set(p.c, s); }
      s.add(mmsi);
    }
  }
  for (const s of byName.values()) if (s.size > 1) count += s.size - 1;

  const lastPts: TimedPoint[] = [];
  const firstPts: TimedPoint[] = [];
  for (const [mmsi, pts] of tracks) {
    const last = pts[pts.length - 1], first = pts[0];
    lastPts.push({ mmsi, t: last.t, la: last.la, lo: last.lo });
    firstPts.push({ mmsi, t: first.t, la: first.la, lo: first.lo });
  }
  count += countHullSwapCandidates(lastPts, firstPts, nearKm, withinHours);
  return count;
}

export function detectLoitering(tracks: Map<string, Pt[]>, zones: ShadowZone[],
                                minHours = 4, maxMedianKts = 2): Record<string, number> {
  const out: Record<string, number> = {};
  for (const z of zones) out[z.id] = 0;
  for (const pts of tracks.values()) {
    for (const z of zones) {
      const inZone = pts.filter((p) => kmBetween(p.la, p.lo, z.lat, z.lon) <= z.radius_km);
      if (inZone.length < 3) continue;
      const spanH = (inZone[inZone.length - 1].t - inZone[0].t) / 3600;
      if (spanH < minHours) continue;
      const speeds = inZone.map((p) => p.v ?? 0).sort((a, b) => a - b);
      const median = speeds[Math.floor(speeds.length / 2)];
      if (median <= maxMedianKts) out[z.id] = (out[z.id] || 0) + 1;
    }
  }
  return out;
}

/** [REPAIR 2026-07-05] Async streaming variant of readVesselTracks — the
 *  synchronous version gunzipSync-ed 72h of archive ON THE REQUEST PATH,
 *  which at current archive size blocked the entire event loop for tens of
 *  seconds (prod probe: 90s timeout then 26s; Railway returned 502/000
 *  because the app could not answer ANY request during the scan). Streams
 *  line-by-line like fleetUtilization/aircraftEntities so the loop keeps
 *  breathing; the sync version stays for tests and small archives. */
export function readVesselTracksAsync(windowHours: number, baseDir?: string, nowMs?: number): Promise<Map<string, Pt[]>> {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "vessels");
  const now = nowMs ?? Date.now();
  const cutoff = Math.floor((now - windowHours * 3600_000) / 1000);
  const tracks = new Map<string, Pt[]>();
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return Promise.resolve(tracks); }
  const wanted = files.filter((f) => {
    const stamp = Date.parse(`${f.slice(0, 10)}T${f.slice(11, 13)}:00:00Z`);
    return Number.isFinite(stamp) && stamp >= now - (windowHours + 1) * 3600_000;
  });
  const readOne = (f: string) => new Promise<void>((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(path.join(dir, f));
      if (f.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (line) => {
        if (!line) return;
        try {
          const p = JSON.parse(line);
          if (p.t < cutoff || p.la == null || p.lo == null || !p.i) return;
          let arr = tracks.get(p.i);
          if (!arr) { arr = []; tracks.set(p.i, arr); }
          arr.push({ t: p.t, la: p.la, lo: p.lo, v: p.v, c: p.c });
        } catch {}
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
    } catch { resolve(); }
  });
  return wanted.reduce((p, f) => p.then(() => readOne(f)), Promise.resolve()).then(() => {
    tracks.forEach((arr) => arr.sort((a, b) => a.t - b.t));
    return tracks;
  });
}

/** [OOM REPAIR 2026-07-05] Generic streaming fold over the vessel archive:
 *  parses each in-window point and hands it to `onPoint` WITHOUT retaining
 *  it. The materializing reader above collects every point of every vessel
 *  into a Map — at the archive's current size (~141MB gz, ~7M points) that
 *  is >800MB of JS objects, and prod OOM-crash-looped every ~61s under the
 *  512MB heap cap the moment the eager boot pollers ran (Railway log:
 *  "JavaScript heap out of memory" ~20s after startup). Async analytics now
 *  fold online with bounded per-vessel state; the materializing reader
 *  stays ONLY for tests and is pinned equal by the async-vs-sync ratchets.
 *  Points arrive per-vessel in near-chronological order (hourly files,
 *  append-ordered); residual disorder is minutes, far under the 2h/6h
 *  analytic thresholds. */
export function foldVesselArchiveAsync(windowHours: number,
                                       onPoint: (mmsi: string, p: Pt) => void,
                                       baseDir?: string, nowMs?: number): Promise<void> {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "vessels");
  const now = nowMs ?? Date.now();
  const cutoff = Math.floor((now - windowHours * 3600_000) / 1000);
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return Promise.resolve(); }
  const wanted = files.filter((f) => {
    const stamp = Date.parse(`${f.slice(0, 10)}T${f.slice(11, 13)}:00:00Z`);
    return Number.isFinite(stamp) && stamp >= now - (windowHours + 1) * 3600_000;
  });
  const readOne = (f: string) => new Promise<void>((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(path.join(dir, f));
      if (f.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (line) => {
        if (!line) return;
        try {
          const p = JSON.parse(line);
          if (p.t < cutoff || p.la == null || p.lo == null || !p.i) return;
          onPoint(p.i, { t: p.t, la: p.la, lo: p.lo, v: p.v, c: p.c });
        } catch {}
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
    } catch { resolve(); }
  });
  return wanted.reduce((p, f) => p.then(() => readOne(f)), Promise.resolve()).then(() => undefined);
}

/** Online shadow-stats aggregator — bounded state per vessel:
 *  last point (gap detection), first point (identity hull-swap), name set
 *  membership, and per-zone in-zone speed runs (in-zone points only — a
 *  tiny subset of the archive). Produces output IDENTICAL to
 *  statsFromTracks on time-ordered input (pinned by the ratchet test). */
export class ShadowAggregator {
  private zones: ShadowZone[];
  private points = 0;
  private prev = new Map<string, Pt>();       // last point per vessel
  private first = new Map<string, Pt>();      // first point per vessel
  private byName = new Map<string, Set<string>>();
  private inZone = new Map<string, { t0: number; t1: number; speeds: number[] }>(); // key mmsi|zone
  private gaps: GapEvent[] = [];
  constructor(zones: ShadowZone[], private minGapHours = 6, private minDistanceKm = 100) {
    this.zones = zones;
  }
  push(mmsi: string, p: Pt): void {
    this.points++;
    if (!this.first.has(mmsi)) this.first.set(mmsi, p);
    const prev = this.prev.get(mmsi);
    if (prev && p.t > prev.t) {
      const dtH = (p.t - prev.t) / 3600;
      if (dtH >= this.minGapHours) {
        const dKm = kmBetween(prev.la, prev.lo, p.la, p.lo);
        if (dKm >= this.minDistanceKm) {
          this.gaps.push({
            mmsi, name: p.c || prev.c,
            darkAt: prev.t, reappearAt: p.t,
            gapHours: Math.round(dtH * 10) / 10, distanceKm: Math.round(dKm),
            from: [prev.la, prev.lo], to: [p.la, p.lo],
          });
        }
      }
    }
    if (!prev || p.t >= prev.t) this.prev.set(mmsi, p);
    if (p.c) {
      let s = this.byName.get(p.c);
      if (!s) { s = new Set(); this.byName.set(p.c, s); }
      s.add(mmsi);
    }
    for (const z of this.zones) {
      if (kmBetween(p.la, p.lo, z.lat, z.lon) > z.radius_km) continue;
      const key = `${mmsi}|${z.id}`;
      const run = this.inZone.get(key);
      if (!run) this.inZone.set(key, { t0: p.t, t1: p.t, speeds: [p.v ?? 0] });
      else { run.t1 = Math.max(run.t1, p.t); run.speeds.push(p.v ?? 0); }
    }
  }
  finish(windowHours: number, minLoiterHours = 4, maxMedianKts = 2): ShadowStats {
    const loiter: Record<string, number> = {};
    for (const z of this.zones) loiter[z.id] = 0;
    this.inZone.forEach((run, key) => {
      if (run.speeds.length < 3) return;
      if ((run.t1 - run.t0) / 3600 < minLoiterHours) return;
      const speeds = run.speeds.sort((a, b) => a - b);
      if (speeds[Math.floor(speeds.length / 2)] > maxMedianKts) return;
      const zid = key.slice(key.indexOf("|") + 1);
      loiter[zid] = (loiter[zid] || 0) + 1;
    });
    let identity = 0;
    this.byName.forEach((s) => { if (s.size > 1) identity += s.size - 1; });
    // [REPAIR 2026-07-13, KNOWN BROKEN #18] was an O(n^2) all-pairs Map.get
    // scan over every vessel seen (~1.26B calls at 35k vessels — the
    // confirmed root cause of the ~10min event-loop stalls). See
    // countHullSwapCandidates's header for the equivalence argument.
    const lastPts: TimedPoint[] = [];
    const firstPts: TimedPoint[] = [];
    this.prev.forEach((p, mmsi) => lastPts.push({ mmsi, t: p.t, la: p.la, lo: p.lo }));
    this.first.forEach((p, mmsi) => firstPts.push({ mmsi, t: p.t, la: p.la, lo: p.lo }));
    identity += countHullSwapCandidates(lastPts, firstPts, 20, 12);
    const gaps = this.gaps.sort((a, b) => b.gapHours - a.gapHours);
    return {
      window_hours: windowHours,
      vessels_seen: this.prev.size,
      points_read: this.points,
      gap_events: gaps.length,
      gap_examples: gaps.slice(0, 5),
      identity_candidates: identity,
      loiter_events: Object.values(loiter).reduce((s, n) => s + n, 0),
      loiter_by_zone: loiter,
      caveat: "RAW statistics from our own terrestrial-AIS archive. A gap can be " +
              "coverage loss, not dark sailing — per-vessel claims are SIGNAL-class " +
              "and gated until validated against documented shadow-fleet vessels " +
              "(ladder gate 1; see research/open_questions.md).",
    };
  }
}

/** Async stats — the only variant routes may use. Online fold, bounded
 *  memory; output pinned identical to the materializing sync path by the
 *  ratchet test. */
export async function computeShadowStatsAsync(zones: ShadowZone[], windowHours = 72,
                                              baseDir?: string, nowMs?: number): Promise<ShadowStats> {
  const agg = new ShadowAggregator(zones);
  await foldVesselArchiveAsync(windowHours, (mmsi, p) => agg.push(mmsi, p), baseDir, nowMs);
  return agg.finish(windowHours);
}

export function computeShadowStats(zones: ShadowZone[], windowHours = 72,
                                   baseDir?: string, nowMs?: number): ShadowStats {
  const tracks = readVesselTracks(windowHours, baseDir, nowMs);
  return statsFromTracks(tracks, zones, windowHours);
}

function statsFromTracks(tracks: Map<string, Pt[]>, zones: ShadowZone[], windowHours: number): ShadowStats {
  let points = 0;
  for (const arr of tracks.values()) points += arr.length;
  const gaps = detectGapEvents(tracks);
  const loiter = detectLoitering(tracks, zones);
  return {
    window_hours: windowHours,
    vessels_seen: tracks.size,
    points_read: points,
    gap_events: gaps.length,
    gap_examples: gaps.slice(0, 5),
    identity_candidates: detectIdentityCandidates(tracks),
    loiter_events: Object.values(loiter).reduce((s, n) => s + n, 0),
    loiter_by_zone: loiter,
    caveat: "RAW statistics from our own terrestrial-AIS archive. A gap can be " +
            "coverage loss, not dark sailing — per-vessel claims are SIGNAL-class " +
            "and gated until validated against documented shadow-fleet vessels " +
            "(ladder gate 1; see research/open_questions.md).",
  };
}
