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
import {
  listVesselWindowFiles, streamVesselFile, readFoldMode, runIncrementalFold,
  PartialCacheRegistry, FoldStatsTracker, StringInterner, histAdd, histMergeInto, histMedian,
  type CachedPartial, type FoldStats, type SpeedHist,
} from "./vesselArchiveFold";

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
interface HullSwapMatch { lastMmsi: string; firstMmsi: string }

/** Every (A last-seen, B first-seen) pair within `nearKm`/`withinHours` —
 *  the actual matched identities behind `countHullSwapCandidates`'s count
 *  and `identitySwapMmsis`'s MMSI reduction. Factored out 2026-09-02 so the
 *  two views can never drift against each other, same discipline
 *  `loiteringMatches` already applies to detectLoitering/detectLoiteringMmsis. */
function hullSwapMatches(lasts: Endpoint[], firsts: Endpoint[],
                         nearKm: number, withinHours: number): HullSwapMatch[] {
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

  const out: HullSwapMatch[] = [];
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
          if (kmBetween(last.la, last.lo, f.la, f.lo) <= nearKm) {
            out.push({ lastMmsi: last.mmsi, firstMmsi: f.mmsi });
          }
        }
      }
    }
  }
  return out;
}

function countHullSwapCandidates(lasts: Endpoint[], firsts: Endpoint[],
                                 nearKm: number, withinHours: number): number {
  return hullSwapMatches(lasts, firsts, nearKm, withinHours).length;
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

/** GATE 1 support (2026-09-02, shadow_fleet_maritime identity-swap detector
 *  independent test — see research/open_questions.md's queued NEXT item
 *  from the 2026-09-02 gap/loiter GATE 1 FAIL): the same two predicates
 *  detectIdentityCandidates counts (name reused under >1 MMSI; hull-swap
 *  proximity), but returns the vessel IDENTITIES rather than a scalar
 *  count, so a case-control test can build its candidate set — mirroring
 *  detectLoiteringMmsis's relationship to detectLoitering. For a hull-swap
 *  match BOTH the vessel that went dark and the vessel that appeared in
 *  its place are candidates: either could be the reference-list vessel
 *  (an OFAC-listed hull could be the one going dark under a swapped
 *  identity, or the one reappearing under a new one) and the detector
 *  itself makes no claim about which. */
export function detectIdentitySwapMmsis(tracks: Map<string, Pt[]>,
                                        nearKm = 20, withinHours = 12): Set<string> {
  const out = new Set<string>();
  const byName = new Map<string, Set<string>>();
  for (const [mmsi, pts] of tracks) {
    for (const p of pts) {
      if (!p.c) continue;
      let s = byName.get(p.c);
      if (!s) { s = new Set(); byName.set(p.c, s); }
      s.add(mmsi);
    }
  }
  for (const s of byName.values()) if (s.size > 1) for (const m of s) out.add(m);

  const lasts: Endpoint[] = [];
  const firsts: Endpoint[] = [];
  for (const [mmsi, pts] of tracks) {
    const f = pts[0], l = pts[pts.length - 1];
    firsts.push({ mmsi, t: f.t, la: f.la, lo: f.lo });
    lasts.push({ mmsi, t: l.t, la: l.la, lo: l.lo });
  }
  for (const m of hullSwapMatches(lasts, firsts, nearKm, withinHours)) {
    out.add(m.lastMmsi);
    out.add(m.firstMmsi);
  }
  return out;
}

/** Shared predicate behind detectLoitering/detectLoiteringMmsis — one
 *  (mmsi, zone) entry per vessel-zone pair that clears the sustained-slow-
 *  presence bar. Factored out 2026-09-01 so the per-zone COUNT view
 *  (detectLoitering, the existing RAW-stats surface) and the per-vessel
 *  MMSI view (detectLoiteringMmsis, GATE 1 case-control support) can never
 *  drift against each other — both are pure reductions of this one list. */
function loiteringMatches(tracks: Map<string, Pt[]>, zones: ShadowZone[],
                          minHours: number, maxMedianKts: number): Array<{ mmsi: string; zone: string }> {
  const out: Array<{ mmsi: string; zone: string }> = [];
  for (const [mmsi, pts] of tracks) {
    for (const z of zones) {
      const inZone = pts.filter((p) => kmBetween(p.la, p.lo, z.lat, z.lon) <= z.radius_km);
      if (inZone.length < 3) continue;
      const spanH = (inZone[inZone.length - 1].t - inZone[0].t) / 3600;
      if (spanH < minHours) continue;
      const speeds = inZone.map((p) => p.v ?? 0).sort((a, b) => a - b);
      const median = speeds[Math.floor(speeds.length / 2)];
      if (median <= maxMedianKts) out.push({ mmsi, zone: z.id });
    }
  }
  return out;
}

export function detectLoitering(tracks: Map<string, Pt[]>, zones: ShadowZone[],
                                minHours = 4, maxMedianKts = 2): Record<string, number> {
  const out: Record<string, number> = {};
  for (const z of zones) out[z.id] = 0;
  for (const m of loiteringMatches(tracks, zones, minHours, maxMedianKts)) out[m.zone] = (out[m.zone] || 0) + 1;
  return out;
}

/** GATE 1 support (2026-09-01, shadow_fleet_maritime case-control test):
 *  the same loitering predicate as detectLoitering, but returns the vessel
 *  IDENTITIES rather than per-zone counts, so shadowFleetGate1.ts's
 *  case-control test can build its candidate set. A vessel loitering in
 *  more than one zone appears once. */
export function detectLoiteringMmsis(tracks: Map<string, Pt[]>, zones: ShadowZone[],
                                     minHours = 4, maxMedianKts = 2): Set<string> {
  return new Set(loiteringMatches(tracks, zones, minHours, maxMedianKts).map((m) => m.mmsi));
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
//  [A2 2026-11-03] The file listing and the per-file streamer now live in
//  vesselArchiveFold.ts (listVesselWindowFiles / streamVesselFile) so the
//  incremental fold and this full fold share ONE accept predicate — same
//  name parse, same [now-(W+1)h, now] file filter, same sorted order, same
//  per-point cutoff/now bounds, same resolve-never-reject error posture.
//  Behavior pinned by the existing tests below (truncated gz, past-`now`,
//  `st` threading, async==sync ratchets).
export async function foldVesselArchiveAsync(windowHours: number,
                                             onPoint: (mmsi: string, p: Pt) => void,
                                             baseDir?: string, nowMs?: number): Promise<void> {
  const listing = listVesselWindowFiles(windowHours, baseDir, nowMs);
  for (const f of listing.files) await streamVesselFile(f.fp, listing.cutoff, listing.nowSec, onPoint);
}

/** Online shadow-stats aggregator — bounded state per vessel:
 *  last point (gap detection), first point (identity hull-swap), name set
 *  membership, and per-zone in-zone speed runs (in-zone points only — a
 *  tiny subset of the archive). Produces output IDENTICAL to
 *  statsFromTracks on time-ordered input (pinned by the ratchet test). */
// AIS ship-type codes 80-89 = tanker (ITU-R M.1371 Table 39); the exact
// range research/open_questions.md's SHADOW-FLEET SIGNAL gate-1 plan names
// for building "the tanker-only universe."
export const TANKER_SHIP_TYPE_MIN = 80;
export const TANKER_SHIP_TYPE_MAX = 89;

/** In-zone speed run per (vessel, zone): `t0` = the FIRST in-zone point in
 *  processing order (not the min — mirrors the original `t0: p.t` at
 *  creation), `t1` = max t, `hist` = the multiset of speeds as a histogram
 *  (vesselArchiveFold.histMedian yields exactly `sorted[floor(n/2)]`). */
interface ZoneRun { t0: number; t1: number; n: number; hist: SpeedHist }

/** [A2 2026-11-03] One closed hour file's worth of ShadowAggregator state,
 *  compact enough to keep ~170 of them resident: per vessel a Float64 row
 *  (SHADOW_STRIDE columns, ids interned by the owning PartialCache) rather
 *  than JS objects. Exactly composable onto an aggregator whose every
 *  retained point is OLDER than this file's `tMin` — see absorbPartial. */
export interface ShadowPartial extends CachedPartial {
  stampSec: number;
  tMin: number;
  tMax: number;
  points: number;
  /** vessels in first-appearance order */
  n: number;
  /** per vessel: [mmsiId, firstOrdinal, firstT, firstLa, firstLo, firstNameId|-1,
   *  prevT, prevLa, prevLo, prevNameId|-1, tanker(0|1)] */
  cols: Float64Array;
  /** flattened (mmsiId, nameId) pairs — every distinct name a vessel wore in this hour */
  namePairs: Int32Array;
  zoneRuns: Array<{ mmsiId: number; zone: string; t0: number; t1: number; n: number; hist: SpeedHist }>;
  /** gaps detected INSIDE this hour (empty whenever minGapHours >= 1h) with
   *  the ordinal of the point that triggered each, for exact ordering */
  gaps: Array<{ ordinal: number; gap: GapEvent }>;
}
export const SHADOW_STRIDE = 11;

const SHADOW_PARTIAL_BYTES_FIXED = 128;
const SHADOW_ZONE_RUN_BYTES = 256; // small object + a few-entry Map
const SHADOW_GAP_BYTES = 200;

export class ShadowAggregator {
  private zones: ShadowZone[];
  private points = 0;
  private prev = new Map<string, Pt>();       // last point per vessel
  private first = new Map<string, Pt>();      // first point per vessel
  private byName = new Map<string, Set<string>>();
  private inZone = new Map<string, ZoneRun>(); // key mmsi|zone
  private gaps: GapEvent[] = [];
  /** ordinal (points processed before it) of the point that emitted each gap — parallel to `gaps` */
  private gapOrdinals: number[] = [];
  private tankers = new Set<string>();        // GATE 1 support: MMSIs ever seen with st in [80,89]
  /** min/max t ever pushed — the hour-invariant check when exporting a partial */
  private tMin = Infinity;
  private tMax = -Infinity;
  /** first-appearance ordinal per vessel; only kept when built as a partial builder */
  private firstOrdinal: Map<string, number> | null;
  constructor(zones: ShadowZone[], private minGapHours = 6, private minDistanceKm = 100,
              opts: { trackOrder?: boolean } = {}) {
    this.zones = zones;
    this.firstOrdinal = opts.trackOrder ? new Map() : null;
  }
  push(mmsi: string, p: Pt): void {
    if (!this.first.has(mmsi)) {
      this.first.set(mmsi, p);
      if (this.firstOrdinal) this.firstOrdinal.set(mmsi, this.points);
    }
    if (p.t < this.tMin) this.tMin = p.t;
    if (p.t > this.tMax) this.tMax = p.t;
    if (p.st != null && p.st >= TANKER_SHIP_TYPE_MIN && p.st <= TANKER_SHIP_TYPE_MAX) this.tankers.add(mmsi);
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
          this.gapOrdinals.push(this.points);
        }
      }
    }
    this.points++;
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
      if (!run) this.inZone.set(key, { t0: p.t, t1: p.t, n: 1, hist: new Map([[p.v ?? 0, 1]]) });
      else { run.t1 = Math.max(run.t1, p.t); run.n++; histAdd(run.hist, p.v ?? 0); }
    }
  }
  /** The loiter predicate finish()/gate1Inputs() share: >=3 in-zone points,
   *  span >= minLoiterHours, median speed <= maxMedianKts. */
  private static loiters(run: ZoneRun, minLoiterHours: number, maxMedianKts: number): boolean {
    if (run.n < 3) return false;
    if ((run.t1 - run.t0) / 3600 < minLoiterHours) return false;
    return histMedian(run.hist, run.n) <= maxMedianKts;
  }

  /** [A2 2026-11-03] Exports this aggregator's state as the partial for the
   *  hour file whose points it was fed (and nothing else). Returns null when
   *  any point's timestamp lies outside [stampSec, stampSec+1h): the exact
   *  composition in absorbPartial relies on every point of a later hour
   *  being newer than every point of an earlier one, and the live writer
   *  guarantees it (datacoreArchive.ts stamps `t` and picks the hour file
   *  from the same `now`) — a file that breaks it is streamed live instead
   *  of being cached, so correctness never depends on the guarantee. */
  exportPartial(stampSec: number, interner: StringInterner): ShadowPartial | null {
    if (!this.firstOrdinal) throw new Error("exportPartial needs an aggregator built with trackOrder");
    if (this.points > 0 && (this.tMin < stampSec || this.tMax >= stampSec + 3600)) return null;
    const n = this.first.size;
    const cols = new Float64Array(n * SHADOW_STRIDE);
    let i = 0;
    this.first.forEach((f, mmsi) => {
      const p = this.prev.get(mmsi) as Pt; // set on the same push that set `first`
      const o = i * SHADOW_STRIDE;
      cols[o] = interner.id(mmsi);
      cols[o + 1] = this.firstOrdinal!.get(mmsi) ?? 0;
      cols[o + 2] = f.t; cols[o + 3] = f.la; cols[o + 4] = f.lo; cols[o + 5] = f.c ? interner.id(f.c) : -1;
      cols[o + 6] = p.t; cols[o + 7] = p.la; cols[o + 8] = p.lo; cols[o + 9] = p.c ? interner.id(p.c) : -1;
      cols[o + 10] = this.tankers.has(mmsi) ? 1 : 0;
      i++;
    });
    const pairs: number[] = [];
    this.byName.forEach((set, name) => { const nid = interner.id(name); set.forEach((m) => pairs.push(interner.id(m), nid)); });
    const zoneRuns: ShadowPartial["zoneRuns"] = [];
    this.inZone.forEach((run, key) => {
      const cut = key.indexOf("|");
      zoneRuns.push({ mmsiId: interner.id(key.slice(0, cut)), zone: key.slice(cut + 1), t0: run.t0, t1: run.t1, n: run.n, hist: new Map(run.hist) });
    });
    const gaps = this.gaps.map((gap, k) => ({ ordinal: this.gapOrdinals[k], gap }));
    const namePairs = Int32Array.from(pairs);
    const partial: ShadowPartial = {
      stampSec, tMin: this.tMin, tMax: this.tMax, points: this.points, n, cols, namePairs, zoneRuns, gaps,
      bytes: SHADOW_PARTIAL_BYTES_FIXED + cols.byteLength + namePairs.byteLength +
             zoneRuns.length * SHADOW_ZONE_RUN_BYTES + gaps.length * SHADOW_GAP_BYTES,
      remapIds(map) {
        for (let k = 0; k < this.n; k++) {
          const o = k * SHADOW_STRIDE;
          this.cols[o] = map(this.cols[o]);
          if (this.cols[o + 5] >= 0) this.cols[o + 5] = map(this.cols[o + 5]);
          if (this.cols[o + 9] >= 0) this.cols[o + 9] = map(this.cols[o + 9]);
        }
        for (let k = 0; k < this.namePairs.length; k++) this.namePairs[k] = map(this.namePairs[k]);
        for (const z of this.zoneRuns) z.mmsiId = map(z.mmsiId);
      },
    };
    return partial;
  }

  /** [A2 2026-11-03] Merges a closed hour's partial into this aggregator
   *  EXACTLY as if that hour's points had been pushed in file order after
   *  everything pushed so far. Per vessel: `first` keeps the earliest
   *  (order-based); the hour's first point is gap-checked against the
   *  retained `prev` precisely as push() would have (then the hour's own
   *  tie-latest max becomes `prev`, since all its points are newer); names
   *  and tanker flags are set unions; zone runs keep the earliest `t0`, the
   *  max `t1` and sum their speed histograms; gaps are appended in the
   *  order the sequential fold would have emitted them (by point ordinal
   *  within the hour) so finish()'s stable sort sees the same sequence.
   *  PRECONDITION (checked, not assumed): every retained `prev.t` for a
   *  vessel in the partial is older than the partial's `tMin`. Returns
   *  false when it is violated (e.g. the same hour present as .jsonl AND
   *  .jsonl.gz during compression) — the caller then discards this state
   *  and runs the full fold. */
  absorbPartial(partial: ShadowPartial, interner: StringInterner): boolean {
    const { cols, n } = partial;
    const pending: Array<{ ordinal: number; gap: GapEvent }> = [];
    for (let k = 0; k < n; k++) {
      const o = k * SHADOW_STRIDE;
      const mmsi = interner.str(cols[o]);
      const fT = cols[o + 2], fLa = cols[o + 3], fLo = cols[o + 4];
      const fC = cols[o + 5] >= 0 ? interner.str(cols[o + 5]) : undefined;
      const pC = cols[o + 9] >= 0 ? interner.str(cols[o + 9]) : undefined;
      if (cols[o + 10]) this.tankers.add(mmsi);
      if (!this.first.has(mmsi)) {
        this.first.set(mmsi, { t: fT, la: fLa, lo: fLo, c: fC });
        if (this.firstOrdinal) this.firstOrdinal.set(mmsi, this.points + cols[o + 1]);
      }
      const prev = this.prev.get(mmsi);
      if (prev) {
        if (prev.t >= partial.tMin) return false;
        const dtH = (fT - prev.t) / 3600;
        if (dtH >= this.minGapHours) {
          const dKm = kmBetween(prev.la, prev.lo, fLa, fLo);
          if (dKm >= this.minDistanceKm) {
            pending.push({ ordinal: cols[o + 1], gap: {
              mmsi, name: fC || prev.c,
              darkAt: prev.t, reappearAt: fT,
              gapHours: Math.round(dtH * 10) / 10, distanceKm: Math.round(dKm),
              from: [prev.la, prev.lo], to: [fLa, fLo],
            } });
          }
        }
      }
      this.prev.set(mmsi, { t: cols[o + 6], la: cols[o + 7], lo: cols[o + 8], c: pC });
    }
    const np = partial.namePairs;
    for (let k = 0; k < np.length; k += 2) {
      const name = interner.str(np[k + 1]);
      let s = this.byName.get(name);
      if (!s) { s = new Set(); this.byName.set(name, s); }
      s.add(interner.str(np[k]));
    }
    for (const z of partial.zoneRuns) {
      const key = `${interner.str(z.mmsiId)}|${z.zone}`;
      const run = this.inZone.get(key);
      if (!run) this.inZone.set(key, { t0: z.t0, t1: z.t1, n: z.n, hist: new Map(z.hist) });
      else { run.t1 = Math.max(run.t1, z.t1); run.n += z.n; histMergeInto(run.hist, z.hist); }
    }
    for (const g of partial.gaps) pending.push(g);
    pending.sort((a, b) => a.ordinal - b.ordinal);
    for (const g of pending) { this.gaps.push(g.gap); this.gapOrdinals.push(this.points + g.ordinal); }
    if (partial.points > 0) {
      if (partial.tMin < this.tMin) this.tMin = partial.tMin;
      if (partial.tMax > this.tMax) this.tMax = partial.tMax;
    }
    this.points += partial.points;
    return true;
  }

  finish(windowHours: number, minLoiterHours = 4, maxMedianKts = 2): ShadowStats {
    const loiter: Record<string, number> = {};
    for (const z of this.zones) loiter[z.id] = 0;
    this.inZone.forEach((run, key) => {
      if (!ShadowAggregator.loiters(run, minLoiterHours, maxMedianKts)) return;
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
  /** GATE 1 support (2026-09-01, shadow_fleet_maritime case-control test):
   *  the MMSI-level detector output the online fold already has bounded
   *  state for, PLUS the tanker population — without materializing the
   *  full per-vessel point archive the way readVesselTracks(Async) does
   *  (the exact OOM this class exists to avoid; see the class doc comment
   *  and the 2026-07-05 OOM REPAIR note on foldVesselArchiveAsync). Reuses
   *  the identical loiter predicate `finish()` already computes for the
   *  per-zone COUNT view — this is a second, MMSI-identity reduction of
   *  the same `inZone` state, not a re-derivation. */
  gate1Inputs(minLoiterHours = 4, maxMedianKts = 2): { gapMmsis: string[]; loiterMmsis: string[]; identitySwapMmsis: string[]; tankerPool: string[] } {
    const gapMmsis = Array.from(new Set(this.gaps.map((g) => g.mmsi)));
    const loiterMmsis = new Set<string>();
    this.inZone.forEach((run, key) => {
      if (!ShadowAggregator.loiters(run, minLoiterHours, maxMedianKts)) return;
      loiterMmsis.add(key.slice(0, key.indexOf("|")));
    });
    return {
      gapMmsis, loiterMmsis: Array.from(loiterMmsis),
      identitySwapMmsis: Array.from(this.identitySwapMmsis()),
      tankerPool: Array.from(this.tankers),
    };
  }
  /** GATE 1 support (2026-09-02): the identity-swap-candidate reduction of
   *  this aggregator's already-bounded `byName`/`first`/`prev` state — same
   *  two predicates detectIdentitySwapMmsis computes from a materializing
   *  scan (name reuse, hull-swap proximity), computed here without ever
   *  holding the full per-vessel point archive (the OOM this class exists
   *  to avoid; see the class doc comment). `first`/`prev` are one point per
   *  vessel each, so this stays O(vessels), not O(points), exactly like the
   *  existing gap/loiter reductions above. */
  identitySwapMmsis(nearKm = 20, withinHours = 12): Set<string> {
    const out = new Set<string>();
    this.byName.forEach((s) => { if (s.size > 1) for (const m of s) out.add(m); });
    const lasts: Endpoint[] = [];
    const firsts: Endpoint[] = [];
    this.prev.forEach((p, mmsi) => lasts.push({ mmsi, t: p.t, la: p.la, lo: p.lo }));
    this.first.forEach((p, mmsi) => firsts.push({ mmsi, t: p.t, la: p.la, lo: p.lo }));
    for (const m of hullSwapMatches(lasts, firsts, nearKm, withinHours)) {
      out.add(m.lastMmsi);
      out.add(m.firstMmsi);
    }
    return out;
  }
}

// ── [A2 2026-11-03] incremental fold wiring ─────────────────────────────────
//
// VOLTRADE_SHADOWFLEET_FOLD = full | shadow (default) | incremental — see
// vesselArchiveFold.ts's header for the contract. Rollback = set `full`.
// Switch to `incremental` only after 24h of zero `FOLD-MISMATCH shadowfleet`
// lines (shadowFleetFoldStats().mismatches stays 0 on /api/health).
export const SHADOWFLEET_FOLD_ENV = "VOLTRADE_SHADOWFLEET_FOLD";
const shadowFoldRegistry = new PartialCacheRegistry<ShadowPartial>();
const shadowFoldStats = new FoldStatsTracker();

/** Cheap snapshot for the /api/health diagnostic. */
export function shadowFleetFoldStats(): FoldStats { return shadowFoldStats.snapshot(); }
export function _resetShadowFoldForTests(): void { shadowFoldRegistry.clear(); shadowFoldStats.reset(); }
/** Partials depend on the zones (in-zone runs) and the gap thresholds (the
 *  class defaults, used by every caller of computeShadowStatsAsync), so
 *  they key the cache. */
function shadowFoldCacheKey(baseDir: string | undefined, zones: ShadowZone[]): string {
  return `${path.join(baseDir || archiveBaseDir(), "vessels")}|zones=${JSON.stringify(zones)}`;
}
/** The live cache for one archive dir + zone set — tests poison it to prove
 *  the shadow compare catches a wrong incremental result. */
export function _shadowFoldCacheForTests(baseDir: string | undefined, zones: ShadowZone[]) {
  return shadowFoldRegistry.forKey(shadowFoldCacheKey(baseDir, zones));
}

/** Async stats — the only variant routes may use. Online fold, bounded
 *  memory; output pinned identical to the materializing sync path by the
 *  ratchet test. In shadow/incremental mode the closed hours come from
 *  cached per-hour partials (ShadowAggregator.exportPartial/absorbPartial)
 *  and only the open hour (plus the cutoff-straddling oldest hour) is
 *  streamed each cycle. */
export async function computeShadowStatsAsync(zones: ShadowZone[], windowHours = 72,
                                              baseDir?: string, nowMs?: number): Promise<ShadowStats> {
  const mode = readFoldMode(SHADOWFLEET_FOLD_ENV);
  if (mode === "full") {
    const t0 = performance.now();
    const agg = new ShadowAggregator(zones);
    await foldVesselArchiveAsync(windowHours, (mmsi, p) => agg.push(mmsi, p), baseDir, nowMs);
    const stats = agg.finish(windowHours);
    shadowFoldStats.update({ mode, lastFullMs: Math.round(performance.now() - t0), lastRunAt: new Date().toISOString(), ...shadowFoldRegistry.totals() });
    return stats;
  }
  const cache = shadowFoldRegistry.forKey(shadowFoldCacheKey(baseDir, zones));
  return runIncrementalFold<ShadowAggregator, ShadowPartial, ShadowStats>({
    stream: "shadowfleet", mode, windowHours, baseDir, nowMs, cache, stats: shadowFoldStats,
    registryTotals: () => shadowFoldRegistry.totals(),
    newState: () => new ShadowAggregator(zones),
    push: (agg, mmsi, p) => agg.push(mmsi, p),
    newBuilder: (file, interner) => {
      const b = new ShadowAggregator(zones, undefined, undefined, { trackOrder: true });
      return { push: (mmsi, p) => b.push(mmsi, p), finish: () => b.exportPartial(file.stampSec, interner) };
    },
    absorb: (agg, partial, interner) => agg.absorbPartial(partial, interner),
    finish: (agg) => agg.finish(windowHours),
  });
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
