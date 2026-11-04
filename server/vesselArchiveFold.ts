/**
 * vesselArchiveFold.ts — shared machinery for INCREMENTAL folds over the
 * vessel position archive (memory fix A2, 2026-11-03; T-DATACORE).
 *
 * WHY. Two 10-minute jobs (shadow-fleet stats, 72h; port-dwell, 168h) each
 * re-stream the whole archive window every cycle: live 2026-11-03 the
 * 168h window is ~10M points (~63s of gunzip+JSON.parse per cycle) and the
 * fold's transient state is the Node heap/RSS peak driver. Yet every hour
 * file except the one still being appended to is IMMUTABLE once its hour
 * ends (datacoreArchive.ts's writer stamps `t` and picks the hour file from
 * the same `now`; compressOldHoursAsync only re-encodes a closed hour as
 * .gz). So: fold each closed hour ONCE into a compact per-hour PARTIAL,
 * cache it, and each cycle combine the cached partials with a live stream
 * of only the still-open hour (and the cutoff-straddling oldest hour).
 *
 * WHAT LIVES HERE (module-agnostic): window listing + closed-hour
 * classification, the per-file streamer (byte-for-byte the same parse and
 * filter as shadowFleet.ts's foldVesselArchiveAsync), the partial cache
 * (per archive dir + fold signature, hard-capped, size/mtime-validated,
 * with an interned string table that is compacted on eviction so it cannot
 * grow with uptime), the fold-mode env flag, a deep diff for the shadow
 * compare, the fold-stats tracker /api/health reads, and ONE generic
 * single-pass orchestrator that runs the full and the incremental fold off
 * the SAME stream of bytes (so a vessel tick appending to the open hour
 * mid-pass can never manufacture a spurious mismatch).
 *
 * WHAT DOES NOT: anything that knows what a partial MEANS. The per-module
 * partial builders/absorbers (what per-vessel state makes a merge exact)
 * live in shadowFleet.ts and portDwell.ts next to the aggregators they
 * mirror, so the math has one home.
 *
 * MODES (env VOLTRADE_<STREAM>_FOLD): "full" = today's behavior exactly;
 * "shadow" (DEFAULT) = run both, deep-compare, log `FOLD-MISMATCH <stream>`
 * on any difference, SERVE THE FULL RESULT; "incremental" = serve the
 * incremental result (switch after 24h of zero mismatches in shadow mode).
 *
 * Pure module: fs reads only, baseDir-injectable, no trading imports.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir } from "./datacoreArchive";

export type FoldMode = "full" | "shadow" | "incremental";
export const FOLD_MODES: readonly FoldMode[] = ["full", "shadow", "incremental"];
export const DEFAULT_FOLD_MODE: FoldMode = "shadow";

const warnedUnknownMode = new Set<string>();

/** Reads the fold mode for one stream from the environment; an unset or
 *  unrecognized value is the DEFAULT (shadow), with a one-time warning for
 *  an unrecognized one so a typo in a deploy variable is visible. */
export function readFoldMode(envVar: string, env: NodeJS.ProcessEnv = process.env): FoldMode {
  const raw = (env[envVar] || "").trim().toLowerCase();
  if (!raw) return DEFAULT_FOLD_MODE;
  if ((FOLD_MODES as readonly string[]).includes(raw)) return raw as FoldMode;
  if (!warnedUnknownMode.has(envVar)) {
    warnedUnknownMode.add(envVar);
    console.warn(`[fold] ${envVar}=${JSON.stringify(env[envVar])} is not one of ${FOLD_MODES.join("|")} — using ${DEFAULT_FOLD_MODE}`);
  }
  return DEFAULT_FOLD_MODE;
}

/** One archived vessel point as the folds see it (same shape as
 *  shadowFleet.ts's Pt; `st` = AIS ship-type code when the line carries one). */
export interface VesselPt { t: number; la: number; lo: number; v?: number; c?: string; st?: number }

// ── window listing ──────────────────────────────────────────────────────────

export interface WindowFile {
  /** file name, e.g. 2026-11-03-10.jsonl or 2026-11-03-10.jsonl.gz */
  name: string;
  fp: string;
  /** the hour the file name encodes, unix seconds */
  stampSec: number;
  /** stat() at listing time; -1 when the stat failed (file vanished) — a
   *  cached partial is valid only while both still match */
  size: number;
  mtimeMs: number;
  /** closed hour (no further appends) whose every point lies inside the
   *  window — the only files a partial may stand in for */
  cacheable: boolean;
}

export interface WindowListing {
  dir: string;
  nowSec: number;
  cutoff: number;
  files: WindowFile[];
}

/** A closed hour is one whose END is at least this far in the past. The
 *  writer and the folds run in one process on one clock, so strictly the
 *  margin could be 0; two minutes costs at most one extra live stream of the
 *  just-closed hour and buys insurance against any clock step. */
export const CLOSED_HOUR_MARGIN_SEC = 120;

/** Lists the vessel hour files foldVesselArchiveAsync would read for this
 *  window (same name parse, same [now-(W+1)h, now] file filter, same sorted
 *  order) and classifies each as cacheable or live. */
export function listVesselWindowFiles(windowHours: number, baseDir?: string, nowMs?: number): WindowListing {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "vessels");
  const now = nowMs ?? Date.now();
  const nowSec = Math.floor(now / 1000);
  const cutoff = Math.floor((now - windowHours * 3600_000) / 1000);
  let names: string[] = [];
  try { names = fs.readdirSync(dir).sort(); } catch (err) {
    // No archive dir yet (fresh volume) — the fold reads nothing, exactly
    // like foldVesselArchiveAsync's own `catch { return Promise.resolve(); }`.
    void err;
    return { dir, nowSec, cutoff, files: [] };
  }
  const files: WindowFile[] = [];
  for (const name of names) {
    const stamp = Date.parse(`${name.slice(0, 10)}T${name.slice(11, 13)}:00:00Z`);
    if (!Number.isFinite(stamp) || stamp < now - (windowHours + 1) * 3600_000 || stamp > now) continue;
    const fp = path.join(dir, name);
    const stampSec = Math.floor(stamp / 1000);
    let size = -1, mtimeMs = -1;
    try { const st = fs.statSync(fp); size = st.size; mtimeMs = st.mtimeMs; } catch (err) { void err; }
    const closed = stampSec + 3600 + CLOSED_HOUR_MARGIN_SEC <= nowSec;
    const insideWindow = stampSec >= cutoff;
    files.push({ name, fp, stampSec, size, mtimeMs, cacheable: size >= 0 && closed && insideWindow });
  }
  return { dir, nowSec, cutoff, files };
}

/** Streams one hour file line by line and hands every in-window point to
 *  `onPoint` — the SAME parse, the SAME accept predicate and the SAME
 *  resolve-never-reject error posture as foldVesselArchiveAsync's readOne,
 *  so a fold built on this sees exactly what the original fold saw.
 *  Resolves true when the file streamed to completion, false when the
 *  stream errored partway (truncated/corrupt gz, vanished file) — a caller
 *  comparing two folds must not count a mismatch on a partial read. */
export function streamVesselFile(fp: string, cutoff: number, nowSec: number,
                                 onPoint: (mmsi: string, p: VesselPt) => void): Promise<{ ok: boolean; badLines: number }> {
  return new Promise((resolve) => {
    let ok = true, badLines = 0;
    const done = () => resolve({ ok, badLines });
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(fp);
      if (fp.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (line) => {
        if (!line) return;
        let parsed: unknown;
        try { parsed = JSON.parse(line); } catch (err) { void err; badLines++; return; }
        if (typeof parsed !== "object" || parsed === null) { badLines++; return; }
        // The archive line's declared shape (datacoreArchive.ts archiveVessels).
        // The comparisons below keep JS semantics on whatever the line holds —
        // identical to the original fold's untyped predicate.
        const p = parsed as { t: number; la: number | null; lo: number | null; i: string; v?: number; c?: string; st?: number };
        if (p.t < cutoff || p.t > nowSec || p.la == null || p.lo == null || !p.i) return;
        onPoint(p.i, { t: p.t, la: p.la, lo: p.lo, v: p.v, c: p.c, st: p.st });
      });
      rl.on("close", done);
      const bail = () => { ok = false; done(); };
      stream.on("error", bail);
      // readline re-emits a piped stream's error on itself — unlistened it
      // crashes the process (see datacoreArchive.ts streamJsonlLines).
      rl.on("error", bail);
    } catch (err) { void err; ok = false; done(); }
  });
}

// ── exact median over a speed histogram ─────────────────────────────────────
//
// The detectors' median is `speeds.sort((a, b) => a - b)[Math.floor(n / 2)]`
// over the run's speeds. Archived speeds are rounded to 0.1 kt, so a run of
// thousands of points has at most a few dozen distinct values: a Map from
// value to count is a far smaller, MERGEABLE representation that yields the
// identical element (the floor(n/2)-th smallest). This is what lets a run
// that spans hours be carried across partials without keeping its points.

export type SpeedHist = Map<number, number>;

export function histAdd(h: SpeedHist, v: number, count = 1): void {
  h.set(v, (h.get(v) ?? 0) + count);
}

export function histMergeInto(into: SpeedHist, from: SpeedHist): void {
  from.forEach((count, v) => histAdd(into, v, count));
}

/** The floor(n/2)-th smallest value (0-based) — exactly `sorted[Math.floor(n/2)]`. */
export function histMedian(h: SpeedHist, n: number): number {
  const keys = Array.from(h.keys()).sort((a, b) => a - b);
  const target = Math.floor(n / 2);
  let seen = 0;
  for (const k of keys) {
    seen += h.get(k) ?? 0;
    if (seen > target) return k;
  }
  return keys[keys.length - 1];
}

// ── string interning (mmsi / vessel names across partials) ──────────────────

/** Maps repeated strings (the same ~60k MMSIs recur hour after hour) to
 *  small integer ids so a partial can hold them in a typed array. One table
 *  per cache; compacted whenever partials are evicted so it never grows
 *  with uptime. */
export class StringInterner {
  private ids = new Map<string, number>();
  private strs: string[] = [];
  private chars = 0;
  id(s: string): number {
    let i = this.ids.get(s);
    if (i === undefined) { i = this.strs.length; this.ids.set(s, i); this.strs.push(s); this.chars += s.length; }
    return i;
  }
  str(i: number): string { return this.strs[i]; }
  get size(): number { return this.strs.length; }
  /** Rough heap footprint: one-byte strings + Map/array slots. */
  bytesEstimate(): number { return this.chars + this.strs.length * 72; }
}

// ── partial cache ───────────────────────────────────────────────────────────

/** What every module's partial must expose to the cache. */
export interface CachedPartial {
  /** rough heap bytes, for the health diagnostic */
  bytes: number;
  /** every interned id this partial references, in place — used when the
   *  cache compacts its string table after evictions */
  remapIds(map: (oldId: number) => number): void;
}

interface CacheEntry<P> { file: WindowFile; partial: P }

/** Per-hour partials for ONE (archive dir, fold signature). Hard-capped;
 *  `get` validates size+mtime so a file that changed under a partial (a
 *  late append, a clock step) is simply re-folded. */
export class PartialCache<P extends CachedPartial> {
  private entries = new Map<string, CacheEntry<P>>();
  /** files whose last build proved uncacheable (e.g. a shadow-fleet hour
   *  file carrying a timestamp outside its own hour) — streamed live every
   *  cycle without rebuilding, until the file itself changes */
  private uncacheable = new Map<string, { size: number; mtimeMs: number }>();
  interner = new StringInterner();
  private evictedSinceCompact = 0;
  /** [REPAIR 2026-11-04] fold cycles currently running on this cache. Live
   *  2026-11-04 (`FOLD-MISMATCH portdwell: vesselsSeen: 60448 vs 60396`):
   *  two callers' cycles overlapped, and one cycle's prune() compacted, i.e.
   *  REPLACED, the string table while the other still held ids minted by the
   *  old one (a builder that captured it, or a partial fetched before an
   *  await and then evicted, so never remapped). Those ids then resolved
   *  against the new table to the wrong MMSI or to undefined (merged
   *  vessels, or a TypeError inside compact itself). While any cycle is in
   *  flight the table is therefore never replaced; the last cycle out
   *  compacts (endCycle). */
  private inFlight = 0;
  /** partials compact() had to drop because they referenced an id the table
   *  does not hold. Stays 0 while cycles use begin/endCycle; anything else
   *  is a bug, logged loudly. */
  compactDrops = 0;

  /** A fold cycle starts using this cache's partials and string table. */
  beginCycle(): void { this.inFlight++; }
  /** The cycle is done; the last one out runs the compaction that any
   *  cycle's evictions deferred. */
  endCycle(): void {
    if (this.inFlight > 0) this.inFlight--;
    if (this.inFlight === 0 && this.evictedSinceCompact > 0) this.compact();
  }
  get cyclesInFlight(): number { return this.inFlight; }

  get(file: WindowFile): P | undefined {
    const e = this.entries.get(file.name);
    if (!e) return undefined;
    if (e.file.size !== file.size || e.file.mtimeMs !== file.mtimeMs) { this.entries.delete(file.name); this.evictedSinceCompact++; return undefined; }
    return e.partial;
  }
  set(file: WindowFile, partial: P): void { this.entries.set(file.name, { file, partial }); }
  /** Unvalidated lookup by file name — for tests that poison a partial to
   *  prove the shadow compare catches a wrong incremental result. */
  peek(name: string): P | undefined { return this.entries.get(name)?.partial; }
  markUncacheable(file: WindowFile): void { this.uncacheable.set(file.name, { size: file.size, mtimeMs: file.mtimeMs }); }
  isUncacheable(file: WindowFile): boolean {
    const u = this.uncacheable.get(file.name);
    return !!u && u.size === file.size && u.mtimeMs === file.mtimeMs;
  }
  get size(): number { return this.entries.size; }
  get uncacheableCount(): number { return this.uncacheable.size; }
  bytesEstimate(): number {
    let b = this.interner.bytesEstimate();
    this.entries.forEach((e) => { b += e.partial.bytes; });
    return b;
  }
  /** Drops partials for files that are no longer cacheable in this window
   *  (fell out of it, or became the cutoff-straddling oldest hour), then the
   *  oldest hours beyond `cap`; uncacheable marks survive while their file
   *  is still listed at all. Compacts the string table if anything was
   *  evicted, unless a fold cycle is in flight: then the last cycle's
   *  endCycle() compacts. Returns the number of partials evicted. */
  prune(cacheableNames: Set<string>, cap: number, listedNames: Set<string> = cacheableNames): number {
    let evicted = 0;
    for (const name of Array.from(this.entries.keys())) {
      if (!cacheableNames.has(name)) { this.entries.delete(name); evicted++; }
    }
    for (const name of Array.from(this.uncacheable.keys())) {
      if (!listedNames.has(name)) this.uncacheable.delete(name);
    }
    if (this.entries.size > cap) {
      const byAge = Array.from(this.entries.values()).sort((a, b) => a.file.stampSec - b.file.stampSec);
      for (const e of byAge.slice(0, this.entries.size - cap)) { this.entries.delete(e.file.name); evicted++; }
    }
    this.evictedSinceCompact += evicted;
    if (this.evictedSinceCompact > 0 && this.inFlight === 0) this.compact();
    return evicted;
  }
  /** Rebuilds the string table from the surviving partials so ids of
   *  evicted hours' vessels are released. O(ids in cache); runs at most
   *  once per cycle and only after an eviction. A partial holding an id the
   *  table does not know is dropped (re-folded next cycle) rather than
   *  half-remapped or allowed to throw out of the cycle. */
  compact(): void {
    const next = new StringInterner();
    const old = this.interner;
    const map = (id: number): number => {
      const s = old.str(id);
      if (s === undefined) throw new Error(`id ${id} is not in the string table (size ${old.size})`);
      return next.id(s);
    };
    for (const [name, e] of Array.from(this.entries)) {
      try { e.partial.remapIds(map); } catch (err) {
        this.entries.delete(name);
        this.compactDrops++;
        console.error(`[fold] compact dropped the partial for ${name}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    this.interner = next;
    this.evictedSinceCompact = 0;
  }
  clear(): void { this.entries.clear(); this.uncacheable.clear(); this.interner = new StringInterner(); this.evictedSinceCompact = 0; }
}

/** Partial caches keyed by archive dir + fold signature (zones/ports/
 *  thresholds), LRU-capped so hermetic tests with many tmp dirs stay
 *  bounded. Production has one dir and one signature per module. */
export class PartialCacheRegistry<P extends CachedPartial> {
  private caches = new Map<string, PartialCache<P>>();
  constructor(private maxKeys = 4) {}
  forKey(key: string): PartialCache<P> {
    let c = this.caches.get(key);
    if (c) { this.caches.delete(key); this.caches.set(key, c); return c; } // refresh LRU position
    c = new PartialCache<P>();
    this.caches.set(key, c);
    while (this.caches.size > this.maxKeys) {
      const oldest = this.caches.keys().next().value;
      if (oldest === undefined) break;
      this.caches.delete(oldest);
    }
    return c;
  }
  totals(): { partialsCached: number; partialBytesEstimate: number; uncacheableFiles: number; compactDrops: number } {
    let partialsCached = 0, partialBytesEstimate = 0, uncacheableFiles = 0, compactDrops = 0;
    this.caches.forEach((c) => {
      partialsCached += c.size; partialBytesEstimate += c.bytesEstimate(); uncacheableFiles += c.uncacheableCount; compactDrops += c.compactDrops;
    });
    return { partialsCached, partialBytesEstimate, uncacheableFiles, compactDrops };
  }
  clear(): void { this.caches.clear(); }
}

// ── deep diff for the shadow compare ────────────────────────────────────────

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v) && !(v instanceof Map) && !(v instanceof Set);
}

function fmt(v: unknown): string {
  if (v === undefined) return "undefined";
  if (v instanceof Map) return `Map(${v.size})`;
  if (v instanceof Set) return `Set(${v.size})`;
  try { const s = JSON.stringify(v); return s.length > 60 ? s.slice(0, 57) + "..." : s; } catch (err) { void err; return String(v); }
}

/** Structural diff of two fold results as compact "path: a vs b" lines
 *  (at most `maxItems`); [] means identical. Arrays are compared IN ORDER
 *  on purpose — every array a fold emits is either explicitly sorted
 *  (gap_examples, anomaly_examples, the per-port visit lists feed a stable
 *  sort) or has a fixed input-defined order (ports), so an order difference
 *  IS an output difference. Maps/objects are compared by key regardless of
 *  insertion order (that order is never observable in the served JSON).
 *  NaN equals NaN. Nothing is excluded here: the folds' result objects
 *  carry no timing fields (generated_at/elapsedMs are added by callers
 *  outside the compared value). */
export function foldDiffSummary(a: unknown, b: unknown, maxItems = 8): string[] {
  const out: string[] = [];
  const walk = (x: unknown, y: unknown, p: string): void => {
    if (out.length >= maxItems) return;
    if (x instanceof Map || y instanceof Map) {
      if (!(x instanceof Map) || !(y instanceof Map)) { out.push(`${p}: ${fmt(x)} vs ${fmt(y)}`); return; }
      const keys = new Set<unknown>([...x.keys(), ...y.keys()]);
      for (const k of keys) {
        if (!x.has(k) || !y.has(k)) { out.push(`${p}[${fmt(k)}]: ${x.has(k) ? "present" : "missing"} vs ${y.has(k) ? "present" : "missing"}`); continue; }
        walk(x.get(k), y.get(k), `${p}[${fmt(k)}]`);
      }
      return;
    }
    if (x instanceof Set || y instanceof Set) {
      if (!(x instanceof Set) || !(y instanceof Set)) { out.push(`${p}: ${fmt(x)} vs ${fmt(y)}`); return; }
      const onlyX = [...x].filter((v) => !y.has(v)), onlyY = [...y].filter((v) => !x.has(v));
      if (onlyX.length || onlyY.length) out.push(`${p}: Set differs (only-a ${onlyX.length}, only-b ${onlyY.length})`);
      return;
    }
    if (Array.isArray(x) || Array.isArray(y)) {
      if (!Array.isArray(x) || !Array.isArray(y)) { out.push(`${p}: ${fmt(x)} vs ${fmt(y)}`); return; }
      if (x.length !== y.length) { out.push(`${p}.length: ${x.length} vs ${y.length}`); }
      const n = Math.min(x.length, y.length);
      for (let i = 0; i < n; i++) walk(x[i], y[i], `${p}[${i}]`);
      return;
    }
    if (isPlainObject(x) || isPlainObject(y)) {
      if (!isPlainObject(x) || !isPlainObject(y)) { out.push(`${p}: ${fmt(x)} vs ${fmt(y)}`); return; }
      const keys = new Set([...Object.keys(x), ...Object.keys(y)]);
      for (const k of Array.from(keys).sort()) walk(x[k], y[k], p ? `${p}.${k}` : k);
      return;
    }
    const same = x === y || (typeof x === "number" && typeof y === "number" && Number.isNaN(x) && Number.isNaN(y));
    if (!same) out.push(`${p || "<root>"}: ${fmt(x)} vs ${fmt(y)}`);
  };
  walk(a, b, "");
  return out;
}

// ── fold stats (read by the /api/health diagnostic) ─────────────────────────

export interface FoldStats {
  mode: FoldMode;
  partialsCached: number;
  partialBytesEstimate: number;
  uncacheableFiles: number;
  lastFullMs: number | null;
  /** incremental mode: the whole pass. shadow mode: an ESTIMATE — the
   *  time of the streams the incremental fold itself needed (open hour,
   *  cutoff-straddling hour, hours not yet cached) plus its merge time,
   *  since both folds are fed from one shared stream there. */
  lastIncrementalMs: number | null;
  mismatches: number;
  lastMismatchAt: string | null;
  matches: number;
  /** compares not attempted: a file read errored partway (both folds then
   *  saw a prefix no later run can reproduce) or the incremental fold hit
   *  a precondition violation and was discarded */
  compareSkipped: number;
  /** cycles where the incremental fold could not stand in for the full
   *  one (a partial's precondition failed — e.g. the same hour present as
   *  both .jsonl and .jsonl.gz during compression, or a file carrying a
   *  timestamp outside its hour) and the full fold ran instead */
  fallbacks: number;
  /** partials PartialCache.compact() dropped because they held an id its
   *  string table did not know (cumulative). Stays 0 unless overlapping
   *  cycles corrupt the cache again — see PartialCache.inFlight. */
  compactDrops: number;
  filesStreamedLastRun: number;
  filesCachedLastRun: number;
  lastRunAt: string | null;
}

export class FoldStatsTracker {
  private s: FoldStats = {
    mode: DEFAULT_FOLD_MODE, partialsCached: 0, partialBytesEstimate: 0, uncacheableFiles: 0,
    lastFullMs: null, lastIncrementalMs: null, mismatches: 0, lastMismatchAt: null, matches: 0,
    compareSkipped: 0, fallbacks: 0, compactDrops: 0, filesStreamedLastRun: 0, filesCachedLastRun: 0, lastRunAt: null,
  };
  snapshot(): FoldStats { return { ...this.s }; }
  /** Mutates the live record — only runIncrementalFold and the module's
   *  own full-mode path call this. */
  update(patch: Partial<FoldStats>): void { Object.assign(this.s, patch); }
  bump(field: "mismatches" | "matches" | "compareSkipped" | "fallbacks"): void { this.s[field]++; }
  reset(): void {
    this.s = {
      mode: DEFAULT_FOLD_MODE, partialsCached: 0, partialBytesEstimate: 0, uncacheableFiles: 0,
      lastFullMs: null, lastIncrementalMs: null, mismatches: 0, lastMismatchAt: null, matches: 0,
      compareSkipped: 0, fallbacks: 0, compactDrops: 0, filesStreamedLastRun: 0, filesCachedLastRun: 0, lastRunAt: null,
    };
  }
}

/** Optional sink for mismatches beyond console.error (e.g. the audit log,
 *  wired by routes.ts if wanted; the pure modules have no audit hook). */
let mismatchHook: ((stream: string, diff: string[]) => void) | null = null;
export function setFoldMismatchHook(fn: ((stream: string, diff: string[]) => void) | null): void { mismatchHook = fn; }

// ── the generic single-pass orchestrator ────────────────────────────────────

/** Builds the partial for ONE cacheable file from its points. `finish`
 *  returns null when the file turned out not to be safely cacheable (the
 *  module's own invariant failed) — the file is then streamed live. */
export interface PartialBuilder<P> {
  push(mmsi: string, p: VesselPt): void;
  finish(): P | null;
}

export interface IncrementalFoldSpec<S, P extends CachedPartial, R> {
  /** log tag: "shadowfleet" | "portdwell" */
  stream: string;
  mode: Exclude<FoldMode, "full">;
  windowHours: number;
  baseDir?: string;
  nowMs?: number;
  cache: PartialCache<P>;
  stats: FoldStatsTracker;
  registryTotals: () => { partialsCached: number; partialBytesEstimate: number; uncacheableFiles: number; compactDrops: number };
  newState(): S;
  push(state: S, mmsi: string, p: VesselPt): void;
  newBuilder(file: WindowFile, interner: StringInterner): PartialBuilder<P>;
  /** merges a cached partial into the running incremental state; false =
   *  the partial's precondition does not hold against the state (the
   *  cycle then falls back to the full fold) */
  absorb(state: S, partial: P, interner: StringInterner): boolean;
  finish(state: S): R;
}

/** Runs one fold cycle in shadow or incremental mode over one stream of
 *  bytes. Returns the result to SERVE (full in shadow mode, incremental in
 *  incremental mode — or the full fold's when the incremental one had to
 *  be discarded). Full mode never comes here: the module's untouched
 *  original path handles it. Cycles on one cache may overlap (several
 *  callers share it); begin/endCycle keep its string table fixed until the
 *  last of them ends. */
export async function runIncrementalFold<S, P extends CachedPartial, R>(spec: IncrementalFoldSpec<S, P, R>): Promise<R> {
  spec.cache.beginCycle();
  try { return await foldCycle(spec); } finally { spec.cache.endCycle(); }
}

async function foldCycle<S, P extends CachedPartial, R>(spec: IncrementalFoldSpec<S, P, R>): Promise<R> {
  const { cache, stats, mode } = spec;
  const listing = listVesselWindowFiles(spec.windowHours, spec.baseDir, spec.nowMs);
  const passStart = performance.now();
  const full = mode === "shadow" ? spec.newState() : null;
  const inc = spec.newState();
  let incExact = true, readErrors = 0, incMs = 0, streamed = 0, fromCache = 0;

  const streamInto = async (file: WindowFile, sink: (mmsi: string, p: VesselPt) => void): Promise<boolean> => {
    streamed++;
    const r = await streamVesselFile(file.fp, listing.cutoff, listing.nowSec, sink);
    if (!r.ok) readErrors++;
    return r.ok;
  };
  const absorbTimed = (partial: P): void => {
    const t = performance.now();
    if (!spec.absorb(inc, partial, cache.interner)) incExact = false;
    incMs += performance.now() - t;
  };

  for (const file of listing.files) {
    const cached = file.cacheable && !cache.isUncacheable(file) ? cache.get(file) : undefined;
    if (cached) {
      fromCache++;
      // The full fold (shadow mode) still needs this file's bytes; the
      // incremental fold stands on the cached partial.
      if (full) await streamInto(file, (m, p) => spec.push(full, m, p));
      absorbTimed(cached);
      continue;
    }
    const t = performance.now();
    if (!file.cacheable || cache.isUncacheable(file)) {
      await streamInto(file, (m, p) => { if (full) spec.push(full, m, p); spec.push(inc, m, p); });
      incMs += performance.now() - t;
      continue;
    }
    // Cacheable and not yet cached: build its partial off the same stream.
    const builder = spec.newBuilder(file, cache.interner);
    const ok = await streamInto(file, (m, p) => { if (full) spec.push(full, m, p); builder.push(m, p); });
    const partial = ok ? builder.finish() : null;
    if (partial) {
      cache.set(file, partial);
      absorbTimed(partial);
    } else {
      // Not safely cacheable (or the read bailed): the incremental state
      // never saw these points, so stream the file again straight into it.
      // A read error also voids this cycle's compare below.
      if (ok) cache.markUncacheable(file);
      await streamInto(file, (m, p) => spec.push(inc, m, p));
    }
    incMs += performance.now() - t;
  }

  cache.prune(new Set(listing.files.filter((f) => f.cacheable).map((f) => f.name)), spec.windowHours + 2,
              new Set(listing.files.map((f) => f.name)));

  const totals = spec.registryTotals();
  const base: Partial<FoldStats> = {
    mode, ...totals, filesStreamedLastRun: streamed, filesCachedLastRun: fromCache,
    lastRunAt: new Date().toISOString(),
  };

  if (mode === "shadow") {
    const fullResult = spec.finish(full as S);
    const passMs = Math.round(performance.now() - passStart); // streams + absorbs + the full finish
    if (!incExact) {
      stats.bump("fallbacks"); stats.bump("compareSkipped");
    } else if (readErrors > 0) {
      stats.bump("compareSkipped");
    } else {
      const t = performance.now();
      const incResult = spec.finish(inc);
      incMs += performance.now() - t;
      const diff = foldDiffSummary(fullResult, incResult);
      if (diff.length) {
        stats.bump("mismatches");
        stats.update({ lastMismatchAt: new Date().toISOString() });
        console.error(`FOLD-MISMATCH ${spec.stream} window=${spec.windowHours}h files=${listing.files.length} cached=${fromCache} :: ${diff.join("; ")}`);
        if (mismatchHook) { try { mismatchHook(spec.stream, diff); } catch (err) { console.error(`[fold] mismatch hook threw: ${String(err)}`); } }
      } else {
        stats.bump("matches");
      }
    }
    stats.update({ ...base, lastFullMs: passMs, lastIncrementalMs: Math.round(incMs) });
    return fullResult;
  }

  // incremental mode
  if (!incExact) {
    stats.bump("fallbacks");
    const passMs = Math.round(performance.now() - passStart);
    const t = performance.now();
    const fb = spec.newState();
    for (const file of listing.files) await streamVesselFile(file.fp, listing.cutoff, listing.nowSec, (m, p) => spec.push(fb, m, p));
    const r = spec.finish(fb);
    stats.update({ ...base, lastFullMs: Math.round(performance.now() - t), lastIncrementalMs: passMs });
    return r;
  }
  const result = spec.finish(inc);
  stats.update({ ...base, lastIncrementalMs: Math.round(performance.now() - passStart) });
  return result;
}
