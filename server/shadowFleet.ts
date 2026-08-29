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

// `st` = AIS ship-type code (numeric, e.g. 80-89 = tanker) when the archive
// line carries one (`archiveVessels`'s `st: p.shiptype ?? undefined` in
// datacoreArchive.ts) — added 2026-08-29 so a tanker-only universe can be
// built from the reader without a second archive pass. Purely additive:
// existing consumers that never read `.st` are unaffected either way.
interface Pt { t: number; la: number; lo: number; v?: number; c?: string; st?: number }

/** Reads every vessel point in the lookback window, grouped per MMSI. */
export function readVesselTracks(windowHours: number, baseDir?: string, nowMs?: number): Map<string, Pt[]> {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "vessels");
  const now = nowMs ?? Date.now();
  const nowSec = Math.floor(now / 1000);
  const cutoff = Math.floor((now - windowHours * 3600_000) / 1000);
  const tracks = new Map<string, Pt[]>();
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return tracks; }
  for (const f of files) {
    // hour files: YYYY-MM-DD-HH.jsonl(.gz) — skip whole files outside window.
    // Both bounds matter: a `now` in the PAST (a historical query against a
    // live, still-growing archive) must not silently absorb every file
    // written after it — see the 2026-08-19 GATE-1 finding this fixes.
    const stamp = Date.parse(`${f.slice(0, 10)}T${f.slice(11, 13)}:00:00Z`);
    if (!Number.isFinite(stamp) || stamp < now - (windowHours + 1) * 3600_000 || stamp > now) continue;
    let text: string;
    try {
      const raw = fs.readFileSync(path.join(dir, f));
      text = f.endsWith(".gz") ? zlib.gunzipSync(raw).toString("utf8") : raw.toString("utf8");
    } catch { continue; }
    for (const line of text.split("\n")) {
      if (!line) continue;
      try {
        const p = JSON.parse(line);
        if (p.t < cutoff || p.t > nowSec || p.la == null || p.lo == null || !p.i) continue;
        let arr = tracks.get(p.i);
        if (!arr) { arr = []; tracks.set(p.i, arr); }
        arr.push({ t: p.t, la: p.la, lo: p.lo, v: p.v, c: p.c, st: p.st });
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

interface Endpoint { mmsi: string; t: number; la: number; lo: number }

/** [PERF REPAIR 2026-07-13, KNOWN BROKEN #18 root cause] Hull-swap candidate
 *  count: is vessel A's last point followed, within `withinHours` and
 *  `nearKm`, by some other vessel B's first point? The naive form compares
 *  every (A,B) pair — O(vessels²) — which at the live archive's scale
 *  (34,895 distinct vessels in the 72h window, 2026-07-13) is ~1.2 BILLION
 *  haversine calls, run synchronously with zero yield points inside a
 *  10-minute setInterval (server/routes.ts refreshShadowStats/
 *  refreshPortDwell's shared computeShadowStatsAsync -> ShadowAggregator.
 *  finish()). That is the actual cause of the recurring ~10-minute,
 *  60-95s-and-growing EVENTLOOP-LAG stalls — the two prior fix attempts on
 *  this item (tmpCleanup async conversion v1.0.291, SQLite WAL mode
 *  v1.0.302) targeted different, ultimately-innocent candidates because
 *  neither session grepped routes.ts's setIntervals, only bot.ts's. Grows
 *  monotonically with the archive because `vessels_seen` in a fixed window
 *  only grows as coverage/history accumulates — matching the observed
 *  growing-magnitude symptom exactly.
 *  Fix: sort firsts by time, binary-search each last's (t, t+window] slice
 *  instead of scanning all N firsts. Same predicate, same output (pinned by
 *  the RATCHET [PERF] test below) — just doesn't do 34,895x more distance
 *  calls than the data requires. */
// Terrestrial AIS traffic packs into a FIXED wall-clock window (windowHours,
// e.g. 72h) regardless of how many vessels the archive has accumulated —
// meaning time-window filtering alone only buys a constant-factor
// (withinHours/windowHours) reduction, not a real complexity fix: density
// (firsts per second) scales WITH vessel count, so the time-window slice
// size scales with N too. A spatial grid is the piece that actually breaks
// the N^2 (real ocean traffic clusters near coastlines/receivers, not
// uniformly over the globe, so per-cell candidate counts stay small as N
// grows). Cell size = nearKm degrees-equivalent; longitude degrees compress
// toward the poles (1 lon-degree ~= 111*cos(lat) km), so the lon-neighbor
// search radius widens with latitude to guarantee a true nearKm-radius
// circle is never under-covered (never a false negative — may scan a few
// extra, harmless, cells instead).
const EARTH_KM_PER_DEG = 111.0;

// KNOWN LIMITATION: cell keys use raw (unwrapped) longitude, so a pair
// straddling the antimeridian (e.g. 179.9 / -179.9, ~11km apart in reality)
// lands in far-apart buckets and is missed. None of this system's tracked
// zones (Gibraltar, Malta, Fujairah, Laconian Gulf) are near the date line;
// accepted as a known gap in a heuristic RAW statistic rather than adding
// dateline-wrap bucket logic for a real-world-irrelevant case.
function countHullSwapCandidates(lasts: Endpoint[], firsts: Endpoint[],
                                 nearKm: number, withinHours: number): number {
  const windowSec = withinHours * 3600;
  const cellDeg = nearKm / EARTH_KM_PER_DEG;
  const grid = new Map<string, { pts: Endpoint[]; times: number[] }>();
  const cellKey = (la: number, lo: number) => `${Math.floor(la / cellDeg)}|${Math.floor(lo / cellDeg)}`;
  for (const f of firsts) {
    const key = cellKey(f.la, f.lo);
    let bucket = grid.get(key);
    if (!bucket) { bucket = { pts: [], times: [] }; grid.set(key, bucket); }
    bucket.pts.push(f);
  }
  grid.forEach((bucket) => {
    bucket.pts.sort((a, b) => a.t - b.t);
    bucket.times = bucket.pts.map((f) => f.t);
  });

  let count = 0;
  for (const last of lasts) {
    const latCell = Math.floor(last.la / cellDeg);
    const lonCell = Math.floor(last.lo / cellDeg);
    // +1 buffer beyond the geometric minimum guards floating-point cell-edge
    // rounding; cos clamped to avoid an unbounded radius at the poles (no
    // real shipping traffic operates there).
    const cosLat = Math.max(Math.cos((last.la * Math.PI) / 180), 0.05);
    const lonRadius = Math.max(1, Math.ceil(1 / cosLat)) + 1;
    const latRadius = 2;
    for (let dLat = -latRadius; dLat <= latRadius; dLat++) {
      for (let dLon = -lonRadius; dLon <= lonRadius; dLon++) {
        const bucket = grid.get(`${latCell + dLat}|${lonCell + dLon}`);
        if (!bucket) continue;
        const lo = upperBoundExclusive(bucket.times, last.t);
        const hi = upperBoundExclusive(bucket.times, last.t + windowSec);
        for (let i = lo; i < hi; i++) {
          const f = bucket.pts[i];
          if (f.mmsi === last.mmsi) continue;
          if (kmBetween(last.la, last.lo, f.la, f.lo) <= nearKm) count++;
        }
      }
    }
  }
  return count;
}

/** First index i such that sorted[i] > value (sorted ascending). */
function upperBoundExclusive(sorted: number[], value: number): number {
  let lo = 0, hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] <= value) lo = mid + 1; else hi = mid;
  }
  return lo;
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

  const lasts: Endpoint[] = [];
  const firsts: Endpoint[] = [];
  for (const [mmsi, pts] of tracks) {
    const f = pts[0], l = pts[pts.length - 1];
    firsts.push({ mmsi, t: f.t, la: f.la, lo: f.lo });
    lasts.push({ mmsi, t: l.t, la: l.la, lo: l.lo });
  }
  count += countHullSwapCandidates(lasts, firsts, nearKm, withinHours);
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
  const nowSec = Math.floor(now / 1000);
  const cutoff = Math.floor((now - windowHours * 3600_000) / 1000);
  const tracks = new Map<string, Pt[]>();
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return Promise.resolve(tracks); }
  const wanted = files.filter((f) => {
    const stamp = Date.parse(`${f.slice(0, 10)}T${f.slice(11, 13)}:00:00Z`);
    return Number.isFinite(stamp) && stamp >= now - (windowHours + 1) * 3600_000 && stamp <= now;
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
          if (p.t < cutoff || p.t > nowSec || p.la == null || p.lo == null || !p.i) return;
          let arr = tracks.get(p.i);
          if (!arr) { arr = []; tracks.set(p.i, arr); }
          arr.push({ t: p.t, la: p.la, lo: p.lo, v: p.v, c: p.c, st: p.st });
        } catch {}
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
      // readline.Interface re-emits a piped-in stream's error on ITSELF too
      // (separate from stream.on("error", ...) above) — unlistened, that
      // crashes the whole process on a truncated/corrupt .gz. See
      // datacoreArchive.ts's streamJsonlLines for the full writeup.
      rl.on("error", () => resolve());
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
  const nowSec = Math.floor(now / 1000);
  const cutoff = Math.floor((now - windowHours * 3600_000) / 1000);
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return Promise.resolve(); }
  const wanted = files.filter((f) => {
    const stamp = Date.parse(`${f.slice(0, 10)}T${f.slice(11, 13)}:00:00Z`);
    return Number.isFinite(stamp) && stamp >= now - (windowHours + 1) * 3600_000 && stamp <= now;
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
          if (p.t < cutoff || p.t > nowSec || p.la == null || p.lo == null || !p.i) return;
          onPoint(p.i, { t: p.t, la: p.la, lo: p.lo, v: p.v, c: p.c, st: p.st });
        } catch {}
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
      // readline.Interface re-emits a piped-in stream's error on ITSELF too
      // (separate from stream.on("error", ...) above) — unlistened, that
      // crashes the whole process on a truncated/corrupt .gz. See
      // datacoreArchive.ts's streamJsonlLines for the full writeup.
      rl.on("error", () => resolve());
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
    const lasts: Endpoint[] = [];
    const firsts: Endpoint[] = [];
    this.prev.forEach((p, mmsi) => lasts.push({ mmsi, t: p.t, la: p.la, lo: p.lo }));
    this.first.forEach((p, mmsi) => firsts.push({ mmsi, t: p.t, la: p.la, lo: p.lo }));
    identity += countHullSwapCandidates(lasts, firsts, 20, 12);
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
