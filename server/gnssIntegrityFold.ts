/**
 * gnssIntegrityFold.ts — the live GNSS-integrity signal's INCREMENTAL fold
 * (memory fix A2 part b, 2026-11-03, territory T-DATACORE).
 *
 * WHAT IT REPLACES. routes.ts's 10-minute refresh called
 * computeGnssIntegritySignal, which re-reads 21 UTC days of aircraft hour
 * files for BOTH regions every cycle (two parallel even-sampled reads, up
 * to 20,000 rows/day/region) and keeps every sampled row of the window in
 * one array until aggregateGnssIntegrity has pooled it — a transient of up
 * to ~840k parsed row objects six times an hour, for an answer whose 20
 * closed days cannot have changed since the previous cycle.
 *
 * WHAT IS EXACT HERE, AND WHY PER DAY. readArchiveDayEvenSample splits the
 * row budget per file (ceil(limit / files-in-the-day)), so a day's sample is
 * a function of the WHOLE day's file set — a per-hour partial could not
 * reproduce it. A per-DAY partial does: the same reader call over the same
 * files returns the same rows, so a partial (per band x origin cell:
 * n_total, n_zero and the airframe identities — kept as a set so the pooled
 * distinct_airframes is a UNION, never a sum) is cached keyed by the day's
 * file manifest (names + sizes + mtimes). A day whose manifest is unchanged
 * is combined from cache; any change — the open day growing, the ~2h gzip
 * rename of a closed day's last hours, an eviction — re-reads that day
 * only. A partial is cached only when the manifest is identical before and
 * after its read, so a file that moved mid-read is never trusted as stable.
 *
 * MODES (env VOLTRADE_GNSS_FOLD, parsed by foldShadow.ts): `full` =
 * computeGnssIntegritySignal, as before; `shadow` (DEFAULT) = read every day
 * once, fold the SAME rows both ways (pooled = today's algorithm, combined =
 * cached/fresh partials), deep-compare with the timing field excluded, log
 * any difference as FOLD-MISMATCH gnss (console.error + audit row), SERVE
 * THE FULL RESULT; `incremental` = combine only, re-reading only days whose
 * manifest changed. Sharing the bytes between both sides in shadow mode is
 * what makes the comparison exact on a LIVE archive: the open hour file
 * grows between any two separate reads, so two independent reads of it
 * would differ for reasons that are not bugs — and would then either drown
 * real mismatches in noise or force the comparison to be skipped almost
 * every cycle.
 *
 * SWITCH ORDER. shadow accumulates cycles with mismatches == 0
 * (gnssFoldStats feeds the health diagnostic); then VOLTRADE_GNSS_FOLD=
 * incremental drops the per-cycle read to the open day. Rollback at any
 * point: VOLTRADE_GNSS_FOLD=full. The API payload is byte-identical across
 * modes (one packaging function, one cell comparator); nothing here writes
 * to the archive or any snapshot file — the cache is process memory only,
 * bounded to the window (<= 21 days x 2 regions), rebuilt one day at a time
 * after a restart.
 */
import fs from "fs";
import path from "path";
import { performance } from "perf_hooks";
import { archiveBaseDir, type ArchiveOrigin } from "./datacoreArchive";
import { firstDifference, parseFoldMode, type AuditWriter, type FoldMode } from "./foldShadow";
import {
  aggregateGnssIntegrityWithAirframes, compareIntegrityCells, poolGnssIntegrityDays, readGnssIntegrityDay,
  type AltitudeBand, type Bbox, type GnssIntegrityDayRead, type GnssIntegrityWindowResult, type IntegrityCell,
} from "./gnssIntegrityQuery";
import {
  CANDIDATE_BBOX, CONTROL_BBOX, WRITER_LIVE_SINCE, computeGnssIntegritySignal, fillMissingDaysFromDaily, recentDays,
  summarizeGnssWindows, type GnssDailyLookup, type GnssIntegritySignalSummary,
} from "./gnssIntegritySignal";

export type GnssFoldMode = FoldMode;
export const GNSS_FOLD_ENV = "VOLTRADE_GNSS_FOLD";
/** readGnssIntegrityWindow's own per-day row budget — restated here only as
 *  the managed entry's default so a test can lower it. */
const DEFAULT_PER_DAY_LIMIT = 20_000;

export function gnssFoldMode(): GnssFoldMode {
  return parseFoldMode(process.env[GNSS_FOLD_ENV]);
}

// ── per-day partials ────────────────────────────────────────────────────────

/** Compact, collision-free key for one airframe identity, so the cache does
 *  not hold (days x regions x up to 20k) 6-char strings: a lowercase 6-hex
 *  icao24 — archiveAircraft's normal output — becomes its 24-bit integer
 *  (an SMI, no heap object); any other string stays itself; a non-string
 *  `i` (never produced by the writer, kept exact anyway) is tagged so it
 *  can collide with neither. Distinct originals always map to distinct
 *  keys, so a union over keys has exactly the size a union over the
 *  originals has — which is all distinct_airframes is. Live bound with the
 *  control box saturating its 20k/day sample: 42 partials x 20k x 8 B
 *  ~ 7 MB worst case, versus ~5x that as strings. */
export type AirframeKey = number | string;
const HEX6 = /^[0-9a-f]{6}$/;
export function airframeKey(id: unknown): AirframeKey {
  if (typeof id === "string") return HEX6.test(id) ? parseInt(id, 16) : id;
  return `\u0000${typeof id}:${String(id)}`;
}

export interface GnssDayPartialCell {
  band: AltitudeBand;
  origin: ArchiveOrigin;
  n_total: number;
  n_zero: number;
  /** the contributing icao24s (as airframeKey) — unioned, never summed, at combine time */
  airframes: AirframeKey[];
}

export interface GnssDayPartial {
  day: string;
  present: boolean;
  rows_scanned: number;
  truncated: boolean;
  cells: GnssDayPartialCell[];
  /** the file manifest (names + sizes + mtimes) this partial was read from; "" when the day had no files */
  manifest: string;
}

const partialCache = new Map<string, GnssDayPartial>();

const bboxKey = (b: Bbox) => `${b.lamin},${b.lamax},${b.lomin},${b.lomax}`;
const cacheKey = (base: string, bbox: Bbox, day: string) => JSON.stringify([base, bboxKey(bbox), day]);

async function listNames(dir: string): Promise<string[] | null> {
  try {
    return await fs.promises.readdir(dir);
  } catch {
    return null; // no aircraft dir yet: every day is honestly missing, same as the reader's own null
  }
}

/** The day's file manifest, from an already-listed directory (one readdir
 *  per cycle, not one per day). Prefix match + sort mirrors archiveDayFiles'
 *  hour-file branch, so the key covers exactly the files the reader opens. */
async function dayManifest(dir: string, day: string, names: string[] | null): Promise<string> {
  if (!names) return "";
  const parts: string[] = [];
  for (const f of names.filter((n) => n.startsWith(day)).sort()) {
    try {
      const st = await fs.promises.stat(path.join(dir, f));
      parts.push(`${f}:${st.size}:${st.mtimeMs}`);
    } catch {
      parts.push(`${f}:gone`); // renamed/evicted between readdir and stat — still a distinct key
    }
  }
  return parts.join("\n");
}

function partialOf(read: GnssIntegrityDayRead, bbox: Bbox, manifest: string): GnssDayPartial {
  const cells = aggregateGnssIntegrityWithAirframes(read.rows, bbox).map((c) => ({
    band: c.band, origin: c.origin, n_total: c.n_total, n_zero: c.n_zero, airframes: Array.from(c.airframes, airframeKey),
  }));
  return { day: read.day, present: read.present, rows_scanned: read.rows.length, truncated: read.truncated, cells, manifest };
}

/** Combine per-day partials (in window order) into the window result the
 *  pooled read would have produced: counts add, airframe sets union, cells
 *  take the shared comparator's order, days keep their iteration order. */
export function combineGnssDayPartials(partials: GnssDayPartial[]): GnssIntegrityWindowResult {
  const byKey = new Map<string, { band: AltitudeBand; origin: ArchiveOrigin; n_total: number; n_zero: number; airframes: Set<AirframeKey> }>();
  const daysRead: string[] = [];
  const daysMissing: string[] = [];
  let rowsScanned = 0;
  let truncated = false;
  for (const p of partials) {
    if (!p.present) { daysMissing.push(p.day); continue; }
    daysRead.push(p.day);
    rowsScanned += p.rows_scanned;
    if (p.truncated) truncated = true;
    for (const c of p.cells) {
      const key = `${c.band}|${c.origin}`;
      let cur = byKey.get(key);
      if (!cur) { cur = { band: c.band, origin: c.origin, n_total: 0, n_zero: 0, airframes: new Set() }; byKey.set(key, cur); }
      cur.n_total += c.n_total;
      cur.n_zero += c.n_zero;
      for (const a of c.airframes) cur.airframes.add(a);
    }
  }
  const cells: IntegrityCell[] = Array.from(byKey.values())
    .map((c) => ({ band: c.band, origin: c.origin, n_total: c.n_total, n_zero: c.n_zero, distinct_airframes: c.airframes.size }))
    .sort(compareIntegrityCells);
  return { cells, days_read: daysRead, days_missing: daysMissing, rows_scanned: rowsScanned, truncated };
}

// ── stats + audit ───────────────────────────────────────────────────────────

export interface GnssFoldStats {
  mode: GnssFoldMode;
  partialsCached: number;
  lastFullMs: number | null;
  /** incremental mode: the cycle's wall time. shadow mode: an ESTIMATE of what
   *  incremental mode would have cost = read time of the days that were not
   *  cache hits + combine time (every day is read in shadow mode). */
  lastIncrementalMs: number | null;
  mismatches: number;
  lastMismatchAt: string | null;
  lastMismatchDetail: string | null;
  cycles: number;
  lastCycleAt: string | null;
  /** day reads performed last cycle across both regions (incremental: the cache misses) */
  lastCycleDayReads: number;
  /** most sampled rows held at once last cycle — the transient this fold exists to shrink; null in full mode (uninstrumented) */
  lastCyclePeakRows: number | null;
}

function freshStats(): GnssFoldStats {
  return {
    mode: gnssFoldMode(), partialsCached: 0, lastFullMs: null, lastIncrementalMs: null,
    mismatches: 0, lastMismatchAt: null, lastMismatchDetail: null,
    cycles: 0, lastCycleAt: null, lastCycleDayReads: 0, lastCyclePeakRows: null,
  };
}
let stats: GnssFoldStats = freshStats();
let auditWriter: AuditWriter | null = null;

/** Snapshot for the health diagnostic. `mode` is the mode the LAST cycle ran
 *  in (the env's current value before any cycle). */
export function gnssFoldStats(): GnssFoldStats {
  return { ...stats, partialsCached: partialCache.size };
}

/** Wire the persistent audit log (routes.ts passes the same audit_log insert
 *  the compliance tripwire uses). Null unwires. */
export function setGnssFoldAuditWriter(w: AuditWriter | null): void {
  auditWriter = w;
}

function recordMismatch(detail: string): void {
  stats.mismatches++;
  stats.lastMismatchAt = new Date().toISOString();
  stats.lastMismatchDetail = detail;
  console.error(`FOLD-MISMATCH gnss: ${detail}`);
  if (!auditWriter) return;
  try {
    auditWriter("FOLD-MISMATCH", `gnss: ${detail}`.slice(0, 500));
  } catch (e: unknown) {
    console.error("[gnss-fold] audit writer threw:", (e as Error)?.message || e);
  }
}

const roundMs = (t0: number) => Math.round((performance.now() - t0) * 10) / 10;

// ── the fold ────────────────────────────────────────────────────────────────

interface RegionFold {
  combined: GnssIntegrityWindowResult;
  pooled: GnssIntegrityWindowResult | null;
  dayReads: number;
  missReadMs: number;
  peakRows: number;
}

/** One region over the window. `keepRows` (shadow mode) reads EVERY day and
 *  keeps the rows so the pooled result can be built from the very same bytes
 *  the partials were — the cached partial is still what the combine uses
 *  when the manifest matched, because the cache's reuse is the thing under
 *  test. Without `keepRows` (incremental mode) a matching manifest skips the
 *  read entirely and a read day's rows are dropped as soon as its partial
 *  exists, so the transient is one day, not the window. */
async function foldRegion(
  bbox: Bbox, days: string[], base: string, names: string[] | null, perDayLimit: number, keepRows: boolean,
): Promise<RegionFold> {
  const dir = path.join(base, "aircraft");
  const reads: GnssIntegrityDayRead[] = [];
  const partials: GnssDayPartial[] = [];
  let dayReads = 0;
  let missReadMs = 0;
  let peakRows = 0;
  let heldRows = 0;
  for (const day of days) {
    const key = cacheKey(base, bbox, day);
    const manifest = await dayManifest(dir, day, names);
    const cached = partialCache.get(key);
    const hit = cached !== undefined && cached.manifest === manifest;
    if (hit && !keepRows) { partials.push(cached); continue; }
    const t0 = performance.now();
    const read = await readGnssIntegrityDay(day, bbox, base, perDayLimit);
    const readMs = performance.now() - t0;
    dayReads++;
    if (!hit) missReadMs += readMs;
    const fresh = partialOf(read, bbox, manifest);
    // Only a day whose files did not move during the read is cached; and
    // when they did move, both sides of the shadow compare use this read.
    const stable = (await dayManifest(dir, day, await listNames(dir))) === manifest;
    if (stable) partialCache.set(key, fresh); else partialCache.delete(key);
    partials.push(hit && stable ? cached : fresh);
    if (keepRows) {
      reads.push(read);
      heldRows += read.rows.length;
      peakRows = Math.max(peakRows, heldRows);
    } else {
      peakRows = Math.max(peakRows, read.rows.length);
    }
  }
  return {
    combined: combineGnssDayPartials(partials),
    pooled: keepRows ? poolGnssIntegrityDays(reads, bbox) : null,
    dayReads, missReadMs, peakRows,
  };
}

/** Drop cached partials for this base whose day left the window, so the
 *  cache is bounded to (days in window) x (regions) no matter how long the
 *  process runs. */
function pruneCache(base: string, days: string[]): void {
  const keep = new Set(days);
  for (const key of Array.from(partialCache.keys())) {
    const [b, , day] = JSON.parse(key) as [string, string, string];
    if (b === base && !keep.has(day)) partialCache.delete(key);
  }
}

const stripTiming = (s: GnssIntegritySignalSummary) => ({ ...s, generated_at: "" });

export interface GnssFoldRunOpts {
  /** overrides the env flag (tests) */
  mode?: GnssFoldMode;
  /** the reader's per-day row budget (tests exercise truncation with a small one) */
  perDayLimit?: number;
}

/**
 * The managed entry routes.ts's poller calls: same parameters and the same
 * payload as computeGnssIntegritySignal, dispatching on VOLTRADE_GNSS_FOLD.
 */
export async function computeGnssIntegritySignalManaged(
  maxDays = 21, baseDir?: string, now: number = Date.now(), daily?: GnssDailyLookup, opts: GnssFoldRunOpts = {},
): Promise<GnssIntegritySignalSummary> {
  const mode = opts.mode ?? gnssFoldMode();
  const perDayLimit = opts.perDayLimit ?? DEFAULT_PER_DAY_LIMIT;
  const t0 = performance.now();
  stats.mode = mode;
  stats.cycles++;
  stats.lastCycleAt = new Date().toISOString();
  const days = recentDays(maxDays, WRITER_LIVE_SINCE, now);

  if (mode === "full") {
    const out = await computeGnssIntegritySignal(maxDays, baseDir, now, daily, perDayLimit);
    stats.lastFullMs = roundMs(t0);
    stats.lastCycleDayReads = days.length * 2;
    stats.lastCyclePeakRows = null;
    return out;
  }

  const base = baseDir ?? archiveBaseDir();
  const names = await listNames(path.join(base, "aircraft"));
  const shadow = mode === "shadow";
  // Both regions concurrently, as computeGnssIntegritySignal's Promise.all does.
  const [cand, ctrl] = await Promise.all([
    foldRegion(CANDIDATE_BBOX, days, base, names, perDayLimit, shadow),
    foldRegion(CONTROL_BBOX, days, base, names, perDayLimit, shadow),
  ]);
  pruneCache(base, days);
  stats.lastCycleDayReads = cand.dayReads + ctrl.dayReads;
  stats.lastCyclePeakRows = cand.peakRows + ctrl.peakRows; // the two regions overlap in time: the sum bounds the transient

  const tCombine = performance.now();
  const incremental = summarizeGnssWindows(
    fillMissingDaysFromDaily(cand.combined, daily, "candidate"),
    fillMissingDaysFromDaily(ctrl.combined, daily, "control"),
    now,
  );
  const combineMs = performance.now() - tCombine;
  if (!shadow) {
    stats.lastIncrementalMs = roundMs(t0);
    return incremental;
  }

  const full = summarizeGnssWindows(
    fillMissingDaysFromDaily(cand.pooled as GnssIntegrityWindowResult, daily, "candidate"),
    fillMissingDaysFromDaily(ctrl.pooled as GnssIntegrityWindowResult, daily, "control"),
    now,
  );
  stats.lastFullMs = roundMs(t0);
  stats.lastIncrementalMs = Math.round((cand.missReadMs + ctrl.missReadMs + combineMs) * 10) / 10;
  const diff = firstDifference(stripTiming(full), stripTiming(incremental));
  if (diff) recordMismatch(diff);
  return full;
}

// ── test seams ──────────────────────────────────────────────────────────────

export function _resetGnssFoldForTests(): void {
  partialCache.clear();
  stats = freshStats();
  auditWriter = null;
}

export function _gnssFoldCacheForTests(): Map<string, GnssDayPartial> {
  return partialCache;
}

export function _gnssFoldCacheKeyForTests(base: string, bbox: Bbox, day: string): string {
  return cacheKey(base, bbox, day);
}
