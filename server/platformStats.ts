/**
 * platformStats.ts — honest, self-updating platform counters for the
 * landing hero (hero-refinements directive 2026-07-05). NOTHING here is
 * hardcoded to today's numbers: layers come from the live registry,
 * streams from registry + what the archive is actually recording, and
 * observations are REAL line counts across the archive (the previous
 * hero summed a `samples` field prod never returned — hence the dash).
 *
 * Counting 80+ MB of JSONL(.gz) per request would be rude; the count runs
 * at most once per TTL window and is cached (stale value served while a
 * refresh runs — stale beats spinner, per the house rule). Since
 * 2026-11-05 it is also incremental (per-file size+mtime cache, persisted
 * beside the archive) and never holds a request: see the REPAIR block below.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir, coldDayFilesHeldInR2, PRIVATE_ARCHIVE_DIRS } from "./datacoreArchive";
import { repoDataPath } from "./repoFiles";
import { registerHeapGauge } from "./heapDiag";

/** Archive dir -> registry layer id, where the names differ. */
export const DIR_LAYER_MAP: Record<string, string> = {
  filings: "insider",
  earnings8k: "earnings",
};
/** Operational streams — recorded, but not data products; never counted.
 *  One set with datacoreArchive's PRIVATE_ARCHIVE_DIRS, so the dirs kept
 *  out of these counters and out of every archive read surface can't drift. */
export const OPERATIONAL_DIRS: ReadonlySet<string> = PRIVATE_ARCHIVE_DIRS;

/** Distinct ingestion streams = every LIVE registry layer (each live layer
 *  is a live feed/surface: weather fields, radar, port dwell, plants, …)
 *  plus archive streams that have no map layer of their own (e.g. option
 *  chains). Grows automatically as layers and archives are added. */
export function streamsRecording(liveLayerIds: string[], archiveDirs: string[]): number {
  const mapped = new Set(liveLayerIds);
  let extra = 0;
  for (const d of archiveDirs) {
    if (OPERATIONAL_DIRS.has(d) || d.endsWith("_tracks")) continue;
    if (!mapped.has(DIR_LAYER_MAP[d] || d)) extra++;
  }
  return liveLayerIds.length + extra;
}

/** An archive file of observations: `*.jsonl` / `*.jsonl.gz`, but never a
 *  "_"-prefixed one — in this archive "_" marks a stream's own bookkeeping
 *  (datacoreArchive.newestDataFile's rule; clinicalTrials' _state.json.gz).
 *  [RULE-REVIEW 2026-11-06, measurement] Before, every *.jsonl(.gz) counted,
 *  so two bookkeeping files were counted as observations on top of the rows
 *  they describe: dtccswaps/_seen_ids_index.jsonl.gz (one dedup ID per line,
 *  the same IDs as the day files' rows — 6,939,910 seen IDs on 2026-11-06)
 *  and finrashortvol/_summary_history.jsonl (one derived summary per day of
 *  rows already counted; 89 lines). Those are the only "_"-prefixed JSONL
 *  files any writer in this repo creates (checked 2026-11-06). */
export function isObservationFile(name: string): boolean {
  return !name.startsWith("_") && (name.endsWith(".jsonl") || name.endsWith(".jsonl.gz"));
}

/** Archive dirs that hold at least one observation file (a dir holding only
 *  bookkeeping records nothing). */
export function listArchiveDirs(base = archiveBaseDir()): string[] {
  try {
    return fs.readdirSync(base, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .filter((n) => {
        try {
          return fs.readdirSync(path.join(base, n)).some(isObservationFile);
        } catch { return false; }
      });
  } catch { return []; }
}

/** Count newline-terminated records in one archive file (gz-aware,
 *  streaming — never loads a file into memory). */
export function countFileLines(fp: string): Promise<number> {
  return new Promise((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(fp);
      if (fp.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      let n = 0;
      rl.on("line", (l) => { if (l.trim()) n++; });
      rl.on("close", () => resolve(n));
      stream.on("error", () => resolve(0));
      // readline.Interface re-emits a piped-in stream's error on ITSELF too
      // (separate from stream.on("error", ...) above) — unlistened, that
      // crashes the whole process on a truncated/corrupt .gz. See
      // datacoreArchive.ts's streamJsonlLines for the full writeup.
      rl.on("error", () => resolve(0));
    } catch { resolve(0); }
  });
}

// ── [REPAIR 2026-11-05] incremental, persisted counting ─────────────────────
// Live (human screenshot 2026-11-05): the landing hero showed only its "—"
// placeholders. The FIRST call after every boot awaited a full recount —
// every line of every archive file, gunzipped (hundreds of MB by now, far
// past the 80 MB this module was sized for) — and the request outlived the
// edge proxy (HTTP 000 after 123 s, measured 2026-11-05 00:31Z), so the
// numbers never arrived on a freshly deployed box, i.e. most of a busy
// deploy day. Two changes: (1) a per-file line count is cached against the
// file's size + mtime (closed hour/day files never change, so only new or
// growing files are counted again) and persisted beside the archive, so a
// boot re-counts what changed, not the whole history; (2) platformStats
// never holds a response longer than FIRST_COUNT_WAIT_MS — it serves the
// persisted total with its own as_of, or null + observations_pending while
// the very first count ever runs.
type FileCount = { size: number; mtimeMs: number; lines: number };
const fileCounts = new Map<string, FileCount>();
let fileCountsBase: string | null = null;
let lineCounter: (fp: string) => Promise<number> = countFileLines;
/** Persisted per-file counts + the last total, a dotfile at the archive root
 *  (every archive reader enumerates directories only). */
export const PLATFORM_STATS_CACHE_FILE = ".platform_stats_cache.json";
/** The longest a request waits for a count when nothing is cached yet. */
export const FIRST_COUNT_WAIT_MS = 2_000;

function useBase(base: string): void {
  if (fileCountsBase !== base) { fileCounts.clear(); fileCountsBase = base; }
}

/** Real total archived records across all non-operational streams (their
 *  observation files only — never a "_" bookkeeping file). A file whose size
 *  and mtime are unchanged since it was last counted is not read again; files
 *  that disappeared (pruned, or evicted by the rolling R2 tier) or stopped
 *  qualifying leave the total. EXCEPTION (2026-11-06): a day file the permanent
 *  R2 tier holds as its only copy (archivePermanentTier.ts: local copy deleted
 *  after a byte-for-byte read-back, never expired) is still archived — it keeps
 *  its last local count, or the tier's own count of that file when this cache
 *  never saw it. Files the tier reports MISSING are not held and not counted. */
export async function countObservations(base = archiveBaseDir()): Promise<number> {
  useBase(base);
  let total = 0;
  const seen = new Set<string>();
  for (const dir of listArchiveDirs(base)) {
    if (OPERATIONAL_DIRS.has(dir) || dir.endsWith("_tracks")) continue;
    const dp = path.join(base, dir);
    let files: string[] = [];
    try { files = fs.readdirSync(dp).filter(isObservationFile); } catch { continue; }
    for (const f of files) {
      const key = `${dir}/${f}`;
      const fp = path.join(dp, f);
      let st: fs.Stats;
      try { st = fs.statSync(fp); } catch { continue; } // vanished between the listing and now: counts 0, as before
      seen.add(key);
      const hit = fileCounts.get(key);
      if (hit && hit.size === st.size && hit.mtimeMs === st.mtimeMs) { total += hit.lines; continue; }
      const lines = await lineCounter(fp);
      fileCounts.set(key, { size: st.size, mtimeMs: st.mtimeMs, lines });
      total += lines;
    }
  }
  for (const f of coldDayFilesHeldInR2()) {
    if (OPERATIONAL_DIRS.has(f.stream) || f.stream.endsWith("_tracks")) continue;
    const key = `${f.stream}/${f.file}`;
    if (seen.has(key)) continue; // still local: counted above
    const hit = fileCounts.get(key);
    const lines = hit ? hit.lines : f.lines;
    if (lines == null) continue; // adopted from a listing, never counted: unknown, not guessed
    if (!hit) fileCounts.set(key, { size: f.bytes, mtimeMs: -1, lines });
    seen.add(key);
    total += lines;
  }
  for (const k of Array.from(fileCounts.keys())) if (!seen.has(k)) fileCounts.delete(k);
  return total;
}

function cacheFilePath(base: string): string { return path.join(base, PLATFORM_STATS_CACHE_FILE); }

/** Seeds the per-file counts from disk; returns the persisted total, or null. Never throws. */
function loadPersistedCounts(base: string): { at: number; observations: number } | null {
  try {
    const raw = JSON.parse(fs.readFileSync(cacheFilePath(base), "utf8"));
    if (raw?.v !== 1 || typeof raw.at !== "number" || typeof raw.observations !== "number"
        || !raw.files || typeof raw.files !== "object") return null;
    useBase(base);
    for (const [k, v] of Object.entries(raw.files as Record<string, unknown>)) {
      if (Array.isArray(v) && v.length === 3 && v.every((x) => typeof x === "number" && Number.isFinite(x))) {
        fileCounts.set(k, { size: v[0], mtimeMs: v[1], lines: v[2] });
      }
    }
    return { at: raw.at, observations: raw.observations };
  } catch {
    return null; // absent (first boot ever) or unreadable: counted from scratch
  }
}

/** Atomic write (tmp + rename) of the per-file counts and the total. Never throws. */
function persistCounts(base: string, at: number, observations: number): void {
  try {
    const files: Record<string, [number, number, number]> = {};
    fileCounts.forEach((c, k) => { files[k] = [c.size, c.mtimeMs, c.lines]; });
    const fp = cacheFilePath(base);
    const tmp = `${fp}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify({ v: 1, at, observations, files }));
    fs.renameSync(tmp, fp);
  } catch (err) {
    console.warn(`[platform-stats] count cache not saved: ${err instanceof Error ? err.message : err}`);
  }
}

/** Heap gauge: one entry per archive file counted (registered beside the state, the A0 convention). */
export function platformStatsCacheStats(): { entries: number } { return { entries: fileCounts.size }; }
registerHeapGauge("platformStats.fileCounts", platformStatsCacheStats);

// ── cached assembly ─────────────────────────────────────────────────────────
const TTL_MS = 10 * 60_000;
let cache: { base: string; at: number; observations: number } | null = null;
let inflight: { base: string; done: Promise<number> } | null = null;

function startCount(base: string): Promise<number> {
  if (inflight && inflight.base === base) return inflight.done;
  const done = countObservations(base).then((n) => {
    const at = Date.now();
    cache = { base, at, observations: n };
    persistCounts(base, at, n);
    return n;
  }).finally(() => {
    if (inflight && inflight.done === done) inflight = null;
  });
  inflight = { base, done };
  return done;
}

/** [REPAIR 2026-07-05, audit defect #9] Sentinel-2 readings are git-side
 *  (session-run pipeline; Railway never runs it) — a stall was invisible.
 *  Surfacing the newest reading date + age here makes staleness a finding
 *  every DAILY health check and dashboard can see. Reads the bundled repo
 *  file; parse failure returns nulls, never throws. */
// [REPAIR 2026-07-07] repoDataPath, not bare cwd: the production image
// has no working-tree datacore/ — this default silently resolved to a
// missing file on Railway, so prod served sentinel2_last_reading:null
// since this surface shipped. The dist/datacore copy (script/build.ts)
// is the runtime source there.
export function sentinel2Freshness(nowMs = Date.now(), fp = repoDataPath("datacore/sentinel2/readings.jsonl")): { last: string | null; age_days: number | null } {
  try {
    const lines = fs.readFileSync(fp, "utf8").trim().split("\n");
    let last: string | null = null;
    for (const line of lines) {
      try {
        const d = JSON.parse(line)?.date;
        if (d && (!last || d > last)) last = d;
      } catch {}
    }
    if (!last) return { last: null, age_days: null };
    return { last, age_days: Math.floor((nowMs - Date.parse(last)) / 86400_000) };
  } catch {
    return { last: null, age_days: null };
  }
}

export async function platformStats(
  layers: Array<{ id: string; status: string }>,
  base = archiveBaseDir(),
  nowMs = Date.now(),
  opts: { firstWaitMs?: number } = {},
): Promise<{ layers_live: number; layers_total: number; streams_recording: number; observations: number | null; observations_as_of: string | null; observations_pending: boolean; sentinel2_last_reading: string | null; sentinel2_reading_age_days: number | null }> {
  const live = layers.filter((l) => l.status === "live").map((l) => l.id);
  // a stream whose day files now sit only in the permanent R2 tier is still recorded
  const dirs = Array.from(new Set([...listArchiveDirs(base), ...coldDayFilesHeldInR2().map((f) => f.stream)]));
  if (!cache || cache.base !== base) {
    const persisted = loadPersistedCounts(base);       // a restart serves the last total at once
    cache = persisted ? { base, ...persisted } : null;
  }
  if (!cache || nowMs - cache.at > TTL_MS) {
    const done = startCount(base);
    if (!cache) {
      // Nothing counted yet, ever: a small archive answers inside the wait;
      // a big one is never allowed to hold the response (the old await did).
      const waitMs = opts.firstWaitMs ?? FIRST_COUNT_WAIT_MS;
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([done, new Promise<void>((resolve) => { timer = setTimeout(resolve, waitMs); })]);
      if (timer) clearTimeout(timer);
    }
  }
  const s2 = sentinel2Freshness(nowMs);
  const counted = cache && cache.base === base ? cache : null;
  return {
    layers_live: live.length,
    layers_total: layers.length,
    streams_recording: streamsRecording(live, dirs),
    observations: counted ? counted.observations : null,
    observations_as_of: counted ? new Date(counted.at).toISOString() : null,
    observations_pending: !!inflight && inflight.base === base,
    sentinel2_last_reading: s2.last,
    sentinel2_reading_age_days: s2.age_days,
  };
}

/** test hook: forget the in-memory state (the persisted file stays, as across a restart) */
export function _resetPlatformStatsCache() { cache = null; inflight = null; fileCounts.clear(); fileCountsBase = null; }
/** test hook: substitute the per-file line counter (null restores countFileLines) */
export function _setLineCounterForTests(fn: ((fp: string) => Promise<number>) | null) { lineCounter = fn ?? countFileLines; }
