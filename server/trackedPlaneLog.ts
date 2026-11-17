// trackedPlaneLog.ts — the PERMANENT, CAPPED flight log of the tracked tails.
//
// OWNER'S INSTRUCTION (2026-11-17, verbatim, about plane data and the hosting
// bill): "for the memory get rid of it to reduce vol or hold it for a short
// time frame and delete. I want to reduce my railway cost per month rn or keep
// full flight logs for watched or favorited planes and if that vol get to big
// have a cap as well but in that order of logic".
// What was decided from it:
//   1. the bulk log of every aircraft in the world is NOT kept
//      (globalAircraft.ts: GLOBAL_SWEEP_ARCHIVE is off unless set);
//   2. the tails on the server's tracked list (trackedPlanes.ts, at most 20)
//      get a FULL log that is never thinned and never expires by age — this
//      module;
//   3. that log has a cap, in bytes, per plane and in total.
//
// WHAT IS WRITTEN. Every position fix of a tracked plane that ANY path
// receives travels over the in-process fix bus (aircraftFixBus.ts): the
// tracked poller, the selected-aircraft lane, the viewport chain, the world
// sweep, the scopes, OpenSky. This module subscribes to the bus and writes
// each distinct fix ONCE:
//   <archive>/aircraft_tracked/<HEX>/YYYY-MM.jsonl      (one file per plane
//                                                        per UTC month)
// The line is the shared archive's own-time line (datacoreArchive
// aircraftFixLine, schema_version 5: t = the fix's own time, ft: 1, pv = the
// provider that served it), so aircraftTrackData.fixFromLine reads both.
// Rules, the same ones the selected-aircraft lane's writer holds:
//   - not at least TRACKED_LOG_NEW_FIX_MIN_MS newer than the last fix written
//     for that aircraft  -> skipped (the same upstream fix arriving twice, or
//     an older one arriving late through a slower path);
//   - older than TRACKED_LOG_MAX_FIX_AGE_MS, or dated more than
//     TRACKED_LOG_MAX_FIX_AHEAD_MS ahead of our clock -> skipped.
// A row that carries no position age is written at OUR receive time, without
// ft, and counted (lines_receive_time_total): the reader labels it.
// NOT written: the 15-minute day-trace backfill (trackedPlanes.ts) — its fixes
// are hours old and stay in the shared hour files only.
// Matching is by ICAO hex. The tracked list is keyed by registration and
// learns each hex from the plane's first live fix; a tail with no hex yet logs
// nothing, and a fix that names such a tail is counted (skipped_hex_unknown).
//
// THE CAP. TRACKED_LOG_MAX_BYTES_PER_PLANE (50 MiB) and
// TRACKED_LOG_MAX_BYTES_TOTAL (500 MiB), bytes on disk (past months are
// gzipped, the current month is not). Over a cap, the OLDEST month file is
// deleted — of that plane for the per-plane cap, across all planes for the
// total — until it is under. The CURRENT UTC month is never deleted: when it
// alone exceeds a cap, writing pauses (counted, visible in the status) until
// the month turns. Removing a tail from the tracked list does NOT delete its
// log; a log with no tracked tail still counts toward the total.
// Free disk under the shared 1 GiB floor, or unreadable: nothing is written
// (fails closed, counted).
//
// NOT in the R2 cold tier and not rolled up: archiveOffload.ts and
// datacoreArchive's compress/rollup passes name their directories
// ("aircraft", "vessels", "trains", "events") and never list this one. This
// module owns its files: gzip of past months, the caps, nothing else deletes.
//
// MEMORY: per tracked plane one number (last written fix time), one small
// pause note and one "seeded" flag — at most TRACKED_CAP (20) of each, pruned
// to the tracked list on every batch — plus a directory listing (one entry per
// plane-month file on disk, re-read every TRACKED_LOG_INVENTORY_MS), the names
// of at most OPENED_MAX files appended to, and a first/last-fix cache of at
// most RANGE_CACHE_MAX files. No fix is held, so no heap gauge is registered.
//
// TRACKED_PLANE_LOG=0 turns the writer off; what is on disk stays readable.

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { pipeline } from "stream/promises";
import { aircraftFixLine, archiveBaseDir, mergedHourBody, streamJsonlLines } from "./datacoreArchive";
import { MIN_FREE_BYTES, readFreeBytes } from "./globalScopes";
import { subscribeFixes, type FixBatch } from "./aircraftFixBus";
import { toSnapRow } from "./globalSnapshot";
import { normalizeReg, type TrackedPlane } from "./trackedPlanes";

const MiB = 1024 * 1024;
export const TRACKED_LOG_DIR = "aircraft_tracked";
export const TRACKED_LOG_ENV = "TRACKED_PLANE_LOG";
export const TRACKED_LOG_MAX_BYTES_PER_PLANE = 50 * MiB;
export const TRACKED_LOG_MAX_BYTES_TOTAL = 500 * MiB;
export const TRACKED_LOG_PER_PLANE_MB_ENV = "TRACKED_PLANE_LOG_MAX_MB_PER_PLANE";
export const TRACKED_LOG_TOTAL_MB_ENV = "TRACKED_PLANE_LOG_MAX_MB_TOTAL";
/** a fix is written only when at least this much newer than the last one written for that aircraft */
export const TRACKED_LOG_NEW_FIX_MIN_MS = 1_000;
/** a fix older than this is not written (the worldwide snapshot refuses it too) */
export const TRACKED_LOG_MAX_FIX_AGE_MS = 10 * 60_000;
/** ... nor one dated more than this ahead of our clock (a wrong provider clock) */
export const TRACKED_LOG_MAX_FIX_AHEAD_MS = 60_000;
/** the shared free-disk floor (1 GiB); under it, or unreadable, nothing is written */
export const TRACKED_LOG_MIN_FREE_BYTES = MIN_FREE_BYTES;
/** a month file is gzipped once its month ended this long ago (a fix can be up
 *  to 10 min old, so nothing is written into a month after its first 10 min) */
export const TRACKED_LOG_GZIP_AFTER_MS = 60 * 60_000;
export const TRACKED_LOG_MAINTAIN_EVERY_MS = 10 * 60_000;
/** a late raw month file is merged into its existing .gz in memory only when
 *  that .gz is at most this big; a bigger one keeps both files (both are read) */
export const TRACKED_LOG_MERGE_MAX_GZ_BYTES = 2 * MiB;
/** how long the directory listing the caps are checked against is trusted */
export const TRACKED_LOG_INVENTORY_MS = 60_000;
const FREE_RECHECK_MS = 60_000;
const RANGE_CACHE_MAX = 256;
const TAIL_READ_BYTES = 8 * 1024;

const HEX_RE = /^[0-9a-f]{6}$/;
const HEX_DIR_RE = /^[0-9A-F]{6}$/;
const MONTH_FILE_RE = /^(\d{4}-\d{2})\.jsonl(\.gz)?$/;
const LINE_T_RE = /^\{"t":(\d+)/;

export const monthOf = (ms: number): string => new Date(ms).toISOString().slice(0, 7);
/** [start, end) of a UTC month "YYYY-MM", ms */
export function monthBoundsMs(month: string): { start: number; end: number } {
  const y = Number(month.slice(0, 4)), m = Number(month.slice(5, 7));
  return { start: Date.UTC(y, m - 1, 1), end: Date.UTC(y, m, 1) };
}
export function trackedLogDir(base: string, hex?: string): string {
  // the literal (not TRACKED_LOG_DIR) on purpose: server/manifests.test.ts
  // scrapes path.join(base, "<dir>") and requires datacore/manifests/<dir>.json
  const root = path.join(base, "aircraft_tracked");
  return hex ? path.join(root, hex.toUpperCase()) : root;
}

/** The kill switch: off only on "0" / "false" / "off" / "no". */
export function trackedLogEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = String(env[TRACKED_LOG_ENV] ?? "1").trim().toLowerCase();
  return !(v === "0" || v === "false" || v === "off" || v === "no");
}

function mbEnv(v: string | undefined, fallbackBytes: number): number {
  const n = v == null || v.trim() === "" ? NaN : Number(v);
  return Number.isFinite(n) && n > 0 ? Math.round(n * MiB) : fallbackBytes;
}
export function trackedLogCaps(env: NodeJS.ProcessEnv = process.env): { perPlane: number; total: number } {
  return {
    perPlane: mbEnv(env[TRACKED_LOG_PER_PLANE_MB_ENV], TRACKED_LOG_MAX_BYTES_PER_PLANE),
    total: mbEnv(env[TRACKED_LOG_TOTAL_MB_ENV], TRACKED_LOG_MAX_BYTES_TOTAL),
  };
}

// ── what is on disk ─────────────────────────────────────────────────────────
export interface LogMonth { month: string; bytes: number; raw_bytes: number; gz_bytes: number }
export interface LogPlane { hex: string; months: LogMonth[]; bytes: number }

/** One plane's month files; null when it has no directory. */
export function scanPlaneLog(base: string, hex: string): LogPlane | null {
  const h = hex.toLowerCase();
  if (!HEX_RE.test(h)) return null;
  const dir = trackedLogDir(base, h);
  let names: string[];
  try { names = fs.readdirSync(dir); } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[tracked-log] listing", h, e instanceof Error ? e.message : String(e));
    return null;
  }
  const byMonth = new Map<string, LogMonth>();
  for (const f of names) {
    const m = MONTH_FILE_RE.exec(f);
    if (!m) continue;
    let size: number;
    try { size = fs.statSync(path.join(dir, f)).size; } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[tracked-log] stat", f, e instanceof Error ? e.message : String(e));
      continue; // gone between the listing and the stat
    }
    const cur = byMonth.get(m[1]) ?? { month: m[1], bytes: 0, raw_bytes: 0, gz_bytes: 0 };
    if (m[2]) cur.gz_bytes += size; else cur.raw_bytes += size;
    cur.bytes += size;
    byMonth.set(m[1], cur);
  }
  const months = Array.from(byMonth.values()).sort((a, b) => a.month.localeCompare(b.month));
  return { hex: h, months, bytes: months.reduce((n, x) => n + x.bytes, 0) };
}

/** Every plane directory and its month files (sync; a few small listings). */
export function scanTrackedLog(base: string): Map<string, LogPlane> {
  const out = new Map<string, LogPlane>();
  let dirs: fs.Dirent[];
  try { dirs = fs.readdirSync(trackedLogDir(base), { withFileTypes: true }); } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[tracked-log] listing:", e instanceof Error ? e.message : String(e));
    return out;
  }
  for (const d of dirs) {
    if (!d.isDirectory() || !HEX_DIR_RE.test(d.name)) continue;
    const plane = scanPlaneLog(base, d.name);
    if (plane) out.set(plane.hex, plane);
  }
  return out;
}

const totalBytes = (inv: Map<string, LogPlane>): number => { let n = 0; for (const p of inv.values()) n += p.bytes; return n; };

// ── first / last fix of a file (cached by size + mtime) ─────────────────────
interface FileRange { size: number; mtimeMs: number; first: number | null; last: number | null }
const rangeCache = new Map<string, FileRange>();

function lineT(line: string | undefined): number | null {
  const m = line ? LINE_T_RE.exec(line) : null;
  return m ? Number(m[1]) : null;
}

/** first and last `t` (epoch s) of one month file. A raw file is written in
 *  time order, so its first and last line are read; a gzip is streamed once
 *  (it does not change afterwards, so the cache answers from then on). */
async function fileRange(fp: string): Promise<{ first: number | null; last: number | null }> {
  let st: fs.Stats;
  try { st = await fs.promises.stat(fp); } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[tracked-log] range stat:", e instanceof Error ? e.message : String(e));
    return { first: null, last: null };
  }
  const hit = rangeCache.get(fp);
  if (hit && hit.size === st.size && hit.mtimeMs === st.mtimeMs) return { first: hit.first, last: hit.last };
  let first: number | null = null, last: number | null = null;
  if (fp.endsWith(".gz")) {
    // min / max of every line's leading "t" (no JSON parse; a merged late line can sit out of order at the end)
    await streamJsonlLines(fp, true, (line) => {
      const t = lineT(line);
      if (t == null) return;
      if (first == null || t < first) first = t;
      if (last == null || t > last) last = t;
    });
  } else {
    const fh = await fs.promises.open(fp, "r");
    try {
      const head = Buffer.alloc(Math.min(TAIL_READ_BYTES, st.size));
      await fh.read(head, 0, head.length, 0);
      first = lineT(head.toString("utf8").split("\n")[0]);
      const tail = Buffer.alloc(Math.min(TAIL_READ_BYTES, st.size));
      await fh.read(tail, 0, tail.length, st.size - tail.length);
      const lines = tail.toString("utf8").split("\n").filter((l) => l.length > 0);
      last = lineT(lines[lines.length - 1]);
    } finally {
      await fh.close();
    }
  }
  rangeCache.delete(fp);
  rangeCache.set(fp, { size: st.size, mtimeMs: st.mtimeMs, first, last });
  while (rangeCache.size > RANGE_CACHE_MAX) {
    const oldest = rangeCache.keys().next().value;
    if (oldest === undefined) break;
    rangeCache.delete(oldest);
  }
  return { first, last };
}

const monthFiles = (base: string, hex: string, m: LogMonth): string[] => {
  const stem = path.join(trackedLogDir(base, hex), m.month);
  return [...(m.gz_bytes ? [`${stem}.jsonl.gz`] : []), ...(m.raw_bytes ? [`${stem}.jsonl`] : [])];
};

export interface PlaneLogSummary {
  months: number; first_month: string | null; last_month: string | null; bytes: number;
  /** epoch s of the oldest / newest fix held; null = no log */
  first_fix_t: number | null; last_fix_t: number | null;
}
export const EMPTY_LOG_SUMMARY: PlaneLogSummary = { months: 0, first_month: null, last_month: null, bytes: 0, first_fix_t: null, last_fix_t: null };

async function summarize(base: string, p: LogPlane | undefined): Promise<PlaneLogSummary> {
  if (!p || !p.months.length) return { ...EMPTY_LOG_SUMMARY };
  const firsts: number[] = [], lasts: number[] = [];
  for (const fp of monthFiles(base, p.hex, p.months[0])) { const r = await fileRange(fp); if (r.first != null) firsts.push(r.first); }
  for (const fp of monthFiles(base, p.hex, p.months[p.months.length - 1])) { const r = await fileRange(fp); if (r.last != null) lasts.push(r.last); }
  return {
    months: p.months.length, first_month: p.months[0].month, last_month: p.months[p.months.length - 1].month, bytes: p.bytes,
    first_fix_t: firsts.length ? Math.min(...firsts) : null, last_fix_t: lasts.length ? Math.max(...lasts) : null,
  };
}

/** What the permanent log holds for one aircraft (reads the disk; no writer needed). */
export async function trackedLogSummary(base: string, hex: string): Promise<PlaneLogSummary> {
  return summarize(base, scanPlaneLog(base, hex) ?? undefined);
}

// ── read-back ───────────────────────────────────────────────────────────────
export interface TrackedLogReadStats {
  months_held: number; months_in_window: number; files_read: number; files_failed: number;
  /** lines that did not parse (a torn last line after a crash): not fixes */
  lines_unparsed: number;
}

/** Stream one aircraft's log lines, oldest month first. The window picks the
 *  month files and, by each line's own `t`, the lines (with `slackSec` either
 *  side: a receive-stamped line can be stamped later than its fix). */
export async function readTrackedLogRows(
  base: string, hex: string,
  opts: { fromSec?: number; toSec?: number; slackSec?: number },
  onRow: (row: Record<string, unknown>) => void,
): Promise<TrackedLogReadStats> {
  const stats: TrackedLogReadStats = { months_held: 0, months_in_window: 0, files_read: 0, files_failed: 0, lines_unparsed: 0 };
  const plane = scanPlaneLog(base, hex);
  if (!plane) return stats;
  const h = plane.hex;
  stats.months_held = plane.months.length;
  const slack = opts.slackSec ?? 600;
  const lo = opts.fromSec == null ? null : opts.fromSec - slack;
  const hi = opts.toSec == null ? null : opts.toSec + slack;
  for (const m of plane.months) {
    const b = monthBoundsMs(m.month);
    if (hi != null && b.start / 1000 > hi) continue;
    if (lo != null && b.end / 1000 < lo) continue;
    stats.months_in_window++;
    for (const fp of monthFiles(base, h, m)) {
      const ok = await streamJsonlLines(fp, fp.endsWith(".gz"), (line) => {
        const t = lineT(line);
        if (t != null && ((lo != null && t < lo) || (hi != null && t > hi))) return;
        let row: Record<string, unknown>;
        try { row = JSON.parse(line) as Record<string, unknown>; } catch (e) { stats.lines_unparsed++; return; }
        if (row.i !== h) return;
        onRow(row);
      });
      if (ok) stats.files_read++; else stats.files_failed++;
    }
  }
  return stats;
}

// ── gzip of a past month (never loses a line) ───────────────────────────────
const gunzipAsync = (b: Buffer): Promise<Buffer> => new Promise((res, rej) => zlib.gunzip(b, (e, out) => (e ? rej(e) : res(out))));
const gzipAsync = (b: Buffer): Promise<Buffer> => new Promise((res, rej) => zlib.gzip(b, (e, out) => (e ? rej(e) : res(out))));

/** the late file's lines that the existing body does not already hold (an
 *  interrupted pass can leave a raw file whose lines are already in the .gz) */
export function lateLinesNotHeld(existing: Buffer, late: Buffer): Buffer {
  const held = new Set(existing.toString("utf8").split("\n"));
  const fresh = late.toString("utf8").split("\n").filter((l) => l.length > 0 && !held.has(l));
  return Buffer.from(fresh.length ? fresh.join("\n") + "\n" : "");
}

export type CompressOutcome = "compressed" | "merged" | "left_beside" | "retry" | "failed";

/** gzip `<dir>/<month>.jsonl` into `<month>.jsonl.gz`. An existing .gz is
 *  MERGED (its lines first, then the late ones it does not hold), as
 *  datacoreArchive.mergedHourBody does for hour files. The gzip is written to
 *  a temp name and renamed; the raw file is removed only in the same tick as
 *  the rename, and only when it did not grow meanwhile. */
export async function compressLogMonth(dir: string, month: string, mergeMaxGzBytes = TRACKED_LOG_MERGE_MAX_GZ_BYTES): Promise<CompressOutcome> {
  const raw = path.join(dir, `${month}.jsonl`);
  const gz = `${raw}.gz`;
  const tmp = `${gz}.tmp`;
  try {
    const before = (await fs.promises.stat(raw)).size;
    const existingSize = await fs.promises.stat(gz).then((st) => st.size, () => null);
    if (existingSize != null) {
      if (existingSize > mergeMaxGzBytes) return "left_beside";
      let existing: Buffer;
      try { existing = await gunzipAsync(await fs.promises.readFile(gz)); } catch (e) {
        console.error(`[tracked-log] ${path.basename(dir)}/${month}: the existing .gz does not gunzip, both files kept:`, e instanceof Error ? e.message : String(e));
        return "left_beside";
      }
      const late = lateLinesNotHeld(existing, await fs.promises.readFile(raw));
      await fs.promises.writeFile(tmp, await gzipAsync(mergedHourBody(existing, late)));
    } else {
      await pipeline(fs.createReadStream(raw), zlib.createGzip(), fs.createWriteStream(tmp));
    }
    // one tick, no await: a write cannot land between the check and the unlink
    if (fs.statSync(raw).size !== before) { fs.unlinkSync(tmp); return "retry"; }
    fs.renameSync(tmp, gz);
    fs.unlinkSync(raw);
    return existingSize != null ? "merged" : "compressed";
  } catch (e) {
    console.error(`[tracked-log] gzip ${path.basename(dir)}/${month}:`, e instanceof Error ? e.message : String(e));
    await fs.promises.unlink(tmp).catch((e2: NodeJS.ErrnoException) => { if (e2?.code !== "ENOENT") console.error("[tracked-log] gzip temp cleanup:", e2?.message || e2); });
    return "failed";
  }
}

// ── the writer ──────────────────────────────────────────────────────────────
const OPENED_MAX = 256;
/** true for a missing or empty file, or one whose last byte is a newline */
function endsWithNewline(fp: string): boolean {
  let fd: number;
  try { fd = fs.openSync(fp, "r"); } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[tracked-log] tail check:", e instanceof Error ? e.message : String(e));
    return true;
  }
  try {
    const size = fs.fstatSync(fd).size;
    if (!size) return true;
    const b = Buffer.alloc(1);
    fs.readSync(fd, b, 0, 1, size - 1);
    return b[0] === 0x0a;
  } finally {
    fs.closeSync(fd);
  }
}

export interface TrackedPlaneLogOptions {
  /** the server's tracked tail list (routes.ts passes the poller's registry) */
  tracked: () => TrackedPlane[];
  baseDir?: string;
  env?: NodeJS.ProcessEnv;
  now?: () => number;
  /** free-space probe (tests inject; default statfs on the archive dir) */
  readFree?: (dir: string) => number | null;
  /** cap overrides in BYTES (tests); default: the env, else the constants */
  maxBytesPerPlane?: number;
  maxBytesTotal?: number;
  minFreeBytes?: number;
}

export type PauseReason = "plane_cap" | "total_cap" | "low_disk";
const PAUSE_TEXT: Record<PauseReason, string> = {
  plane_cap: "this plane's current month alone is at the per-plane cap; writing resumes when the month turns and the old month can go",
  total_cap: "the log is at its total cap and only current-month files are left; writing resumes when the month turns",
  low_disk: "free disk is under the 1 GiB floor (or could not be read): nothing is written",
};

export class TrackedPlaneLog {
  /** hex -> time (ms) of the last fix written; at most TRACKED_CAP entries */
  private last = new Map<string, number>();
  private seeded = new Set<string>();
  private paused = new Map<string, { why: PauseReason; at: number }>();
  /** month files this process has appended to (their last byte was checked once) */
  private opened = new Set<string>();
  private inv: Map<string, LogPlane> | null = null;
  private invAt = 0;
  private freeAt = 0;
  private freeBytes: number | null = null;
  private lowDiskWarned = false;
  private maintaining = false;
  private errorCount = 0;
  readonly counters = {
    fixes_offered_total: 0,            // fixes of a tracked plane (hex known) that reached the decision
    lines_written_total: 0,
    bytes_written_total: 0,
    lines_receive_time_total: 0,       // written at our receive time: the row carried no position age
    skipped_same_fix_total: 0,         // not at least 1 s newer than the last fix written for that aircraft
    skipped_stale_total: 0,            // fix older than TRACKED_LOG_MAX_FIX_AGE_MS
    skipped_future_total: 0,           // fix dated more than TRACKED_LOG_MAX_FIX_AHEAD_MS ahead of our clock
    skipped_low_disk_total: 0,         // free < the shared 1 GiB floor, or unreadable
    skipped_plane_cap_total: 0,        // the plane's current month alone is at the per-plane cap
    skipped_total_cap_total: 0,        // the total cap is reached with only current months left
    skipped_disabled_total: 0,         // TRACKED_PLANE_LOG=0
    skipped_hex_unknown_total: 0,      // a fix naming a tracked tail whose hex is not known yet
    write_errors_total: 0,
    months_deleted_plane_cap_total: 0,
    months_deleted_total_cap_total: 0,
    bytes_deleted_total: 0,
    months_compressed_total: 0,
    months_merged_total: 0,            // a late raw file merged into its existing .gz
    months_left_beside_total: 0,       // a merge that could not be done safely: both files kept, both read
    maintenance_errors_total: 0,
    last_write_at: null as number | null,
    last_error: null as string | null,
  };

  constructor(private opts: TrackedPlaneLogOptions) {}

  private now(): number { return (this.opts.now ?? Date.now)(); }
  get baseDir(): string { return this.opts.baseDir || archiveBaseDir(); }
  get enabled(): boolean { return trackedLogEnabled(this.opts.env ?? process.env); }
  get caps(): { perPlane: number; total: number } {
    const c = trackedLogCaps(this.opts.env ?? process.env);
    return { perPlane: this.opts.maxBytesPerPlane ?? c.perPlane, total: this.opts.maxBytesTotal ?? c.total };
  }
  private get minFree(): number { return this.opts.minFreeBytes ?? TRACKED_LOG_MIN_FREE_BYTES; }

  private noteError(where: string, e: unknown): void {
    this.counters.last_error = `${where}: ${e instanceof Error ? e.message : String(e)}`;
    if (this.errorCount++ % 100 === 0) console.error(`[tracked-log] ${this.counters.last_error} (x${this.errorCount})`);
  }

  private diskAllows(t: number): boolean {
    if (this.freeAt === 0 || t - this.freeAt > FREE_RECHECK_MS) {
      this.freeAt = t;
      this.freeBytes = (this.opts.readFree ?? readFreeBytes)(this.baseDir);
    }
    const ok = this.freeBytes != null && this.freeBytes >= this.minFree; // unreadable = closed
    if (!ok && !this.lowDiskWarned) {
      this.lowDiskWarned = true;
      console.error(`[tracked-log] writes PAUSED: ${this.freeBytes == null ? "free space unreadable (fails closed)" : `${this.freeBytes} bytes free`} < ${this.minFree} floor; skipped fixes are counted`);
    } else if (ok && this.lowDiskWarned) {
      this.lowDiskWarned = false;
      console.error("[tracked-log] disk headroom back above the floor: writes resumed");
    }
    return ok;
  }

  private inventory(t: number, fresh = false): Map<string, LogPlane> {
    if (fresh || !this.inv || t - this.invAt > TRACKED_LOG_INVENTORY_MS) {
      this.inv = scanTrackedLog(this.baseDir);
      this.invAt = t;
    }
    return this.inv;
  }

  /** the time of the last line already in this plane's newest raw month file:
   *  after a restart the first fix is not written a second time */
  private seedLast(hex: string, t: number): void {
    if (this.seeded.has(hex)) return;
    this.seeded.add(hex);
    for (const month of [monthOf(t), monthOf(t - TRACKED_LOG_MAX_FIX_AGE_MS)]) {
      const fp = path.join(trackedLogDir(this.baseDir, hex), `${month}.jsonl`);
      try {
        const size = fs.statSync(fp).size;
        const fd = fs.openSync(fp, "r");
        try {
          const buf = Buffer.alloc(Math.min(TAIL_READ_BYTES, size));
          fs.readSync(fd, buf, 0, buf.length, size - buf.length);
          const lines = buf.toString("utf8").split("\n").filter((l) => l.length > 0);
          const lt = lineT(lines[lines.length - 1]);
          if (lt != null) { this.last.set(hex, Math.max(this.last.get(hex) ?? 0, lt * 1000)); return; }
        } finally {
          fs.closeSync(fd);
        }
      } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") this.noteError("seed", e);
      }
    }
  }

  private deleteMonth(plane: LogPlane, m: LogMonth, why: "plane_cap" | "total_cap"): boolean {
    const stem = path.join(trackedLogDir(this.baseDir, plane.hex), m.month);
    let ok = true;
    for (const fp of [`${stem}.jsonl`, `${stem}.jsonl.gz`]) {
      try { fs.unlinkSync(fp); } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") { ok = false; this.noteError("cap delete", e); }
      }
    }
    if (!ok) return false;
    plane.months = plane.months.filter((x) => x !== m);
    plane.bytes -= m.bytes;
    this.counters.bytes_deleted_total += m.bytes;
    if (why === "plane_cap") this.counters.months_deleted_plane_cap_total++; else this.counters.months_deleted_total_cap_total++;
    console.error(`[tracked-log] cap: deleted ${plane.hex.toUpperCase()}/${m.month} (${m.bytes} bytes; ${why === "plane_cap" ? "this plane is over its cap" : "the log is over its total cap"}; oldest month first, the current month is never deleted)`);
    if (!plane.months.length) {
      try { fs.rmdirSync(trackedLogDir(this.baseDir, plane.hex)); this.inv?.delete(plane.hex); } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOTEMPTY") this.noteError("cap rmdir", e);
      }
    }
    return true;
  }

  /** Delete oldest month files until every plane and the total are under their
   *  caps (`need`: bytes about to be written, for one plane and in all, that
   *  must fit as well). The current UTC month is never deleted. Returns the
   *  months deleted. */
  enforceCaps(nowMs = this.now(), need: { hex: string; planeBytes: number; totalBytes: number } | null = null): number {
    const inv = this.inventory(nowMs);
    const cur = monthOf(nowMs);
    const caps = this.caps;
    const headroom = need ? need.totalBytes : 0;
    let deleted = 0;
    for (const plane of Array.from(inv.values())) {
      const extra = need && plane.hex === need.hex ? need.planeBytes : 0;
      while (plane.bytes + extra > caps.perPlane) {
        const m = plane.months.find((x) => x.month < cur);
        if (!m || !this.deleteMonth(plane, m, "plane_cap")) break;
        deleted++;
      }
    }
    while (totalBytes(inv) + headroom > caps.total) {
      let pick: { plane: LogPlane; m: LogMonth } | null = null;
      for (const plane of inv.values()) {
        const m = plane.months.find((x) => x.month < cur);
        if (m && (!pick || m.month < pick.m.month || (m.month === pick.m.month && plane.bytes > pick.plane.bytes))) pick = { plane, m };
      }
      if (!pick || !this.deleteMonth(pick.plane, pick.m, "total_cap")) break;
      deleted++;
    }
    return deleted;
  }

  /** One batch off the fix bus. Returns the lines written. */
  ingest(b: FixBatch): number {
    if (!b.aircraft.length) return 0;
    const planes = this.opts.tracked();
    if (!planes.length) return 0;
    const tracked = new Set<string>();
    const hexless = new Set<string>();
    for (const tp of planes) {
      const h = String(tp.hex || "").toLowerCase();
      if (HEX_RE.test(h)) tracked.add(h); else hexless.add(tp.reg);
    }
    // memory stays bounded by the tracked list (TRACKED_CAP)
    for (const m of [this.last, this.paused, this.seeded]) {
      if (m.size > tracked.size) for (const h of Array.from(m.keys())) if (!tracked.has(h)) m.delete(h);
    }
    const c = this.counters;
    const t = this.now();
    const caps = this.caps;
    const pending = new Map<string, { hex: string; month: string; lines: string[]; bytes: number }>();
    const pendingByHex = new Map<string, number>();
    let pendingTotal = 0;
    for (const p of b.aircraft) {
      const hex = String(p?.icao24 || "").toLowerCase();
      if (!tracked.has(hex)) {
        if (hexless.size && p?.registration) {
          const reg = normalizeReg(p.registration);
          if (reg && hexless.has(reg)) c.skipped_hex_unknown_total++;
        }
        continue;
      }
      const row = toSnapRow(p, b);
      if (!row) continue;
      c.fixes_offered_total++;
      if (!this.enabled) { c.skipped_disabled_total++; continue; }
      if (t - row.seenAt > TRACKED_LOG_MAX_FIX_AGE_MS) { c.skipped_stale_total++; continue; }
      if (row.seenAt - t > TRACKED_LOG_MAX_FIX_AHEAD_MS) { c.skipped_future_total++; continue; }
      this.seedLast(hex, t);
      const last = this.last.get(hex);
      if (last !== undefined && row.seenAt < last + TRACKED_LOG_NEW_FIX_MIN_MS) { c.skipped_same_fix_total++; continue; }
      if (!this.diskAllows(t)) { c.skipped_low_disk_total++; this.paused.set(hex, { why: "low_disk", at: t }); continue; }
      const ownTime = Number.isFinite(p.seen_at_ms as number) || Number.isFinite(p.seen_pos as number);
      const line = aircraftFixLine({ ...p, icao24: hex, provider: p.provider ?? b.provider, tSec: row.seenAt / 1000 }, ownTime);
      const bytes = Buffer.byteLength(line) + 1;
      const inv = this.inventory(t);
      const mine = (pendingByHex.get(hex) ?? 0) + bytes;
      if ((inv.get(hex)?.bytes ?? 0) + mine > caps.perPlane || totalBytes(inv) + pendingTotal + bytes > caps.total) {
        this.enforceCaps(t, { hex, planeBytes: mine, totalBytes: pendingTotal + bytes });
        if ((inv.get(hex)?.bytes ?? 0) + mine > caps.perPlane) { c.skipped_plane_cap_total++; this.paused.set(hex, { why: "plane_cap", at: t }); continue; }
        if (totalBytes(inv) + pendingTotal + bytes > caps.total) { c.skipped_total_cap_total++; this.paused.set(hex, { why: "total_cap", at: t }); continue; }
      }
      const month = monthOf(row.seenAt);
      const key = `${hex}|${month}`;
      const slot = pending.get(key) ?? { hex, month, lines: [], bytes: 0 };
      slot.lines.push(line); slot.bytes += bytes;
      pending.set(key, slot);
      pendingByHex.set(hex, mine);
      pendingTotal += bytes;
      this.last.set(hex, row.seenAt);
      if (!ownTime) c.lines_receive_time_total++;
    }
    let written = 0;
    for (const slot of pending.values()) {
      const dir = trackedLogDir(this.baseDir, slot.hex);
      const fp = path.join(dir, `${slot.month}.jsonl`);
      try {
        fs.mkdirSync(dir, { recursive: true });
        // first append of this process to this file: a line torn by a crash
        // mid-write (no final newline) must not swallow the next fix
        const lead = this.opened.has(fp) || endsWithNewline(fp) ? "" : "\n";
        if (this.opened.size >= OPENED_MAX) this.opened.clear();
        this.opened.add(fp);
        fs.appendFileSync(fp, lead + slot.lines.join("\n") + "\n");
      } catch (e) {
        // counted, never retried (a retry could write a line twice)
        c.write_errors_total += slot.lines.length;
        this.noteError("append", e);
        continue;
      }
      written += slot.lines.length;
      c.lines_written_total += slot.lines.length;
      c.bytes_written_total += slot.bytes;
      c.last_write_at = t;
      this.paused.delete(slot.hex);
      const inv = this.inventory(t);
      const plane = inv.get(slot.hex) ?? { hex: slot.hex, months: [], bytes: 0 };
      let m = plane.months.find((x) => x.month === slot.month);
      if (!m) { m = { month: slot.month, bytes: 0, raw_bytes: 0, gz_bytes: 0 }; plane.months.push(m); plane.months.sort((a, z) => a.month.localeCompare(z.month)); }
      m.bytes += slot.bytes; m.raw_bytes += slot.bytes; plane.bytes += slot.bytes;
      inv.set(slot.hex, plane);
    }
    return written;
  }

  /** gzip the month files whose month ended over an hour ago (merging a late
   *  raw file into its .gz), then hold the caps. */
  async maintain(nowMs = this.now()): Promise<{ compressed: number; merged: number; deleted: number }> {
    const res = { compressed: 0, merged: 0, deleted: 0 };
    if (this.maintaining) return res;
    this.maintaining = true;
    try {
      for (const plane of scanTrackedLog(this.baseDir).values()) {
        for (const m of plane.months) {
          if (!m.raw_bytes || nowMs - monthBoundsMs(m.month).end < TRACKED_LOG_GZIP_AFTER_MS) continue;
          const out = await compressLogMonth(trackedLogDir(this.baseDir, plane.hex), m.month);
          if (out === "compressed") { res.compressed++; this.counters.months_compressed_total++; }
          else if (out === "merged") { res.merged++; this.counters.months_merged_total++; }
          else if (out === "left_beside") this.counters.months_left_beside_total++;
          else if (out === "failed") this.counters.maintenance_errors_total++;
        }
      }
      this.inventory(nowMs, true);
      res.deleted = this.enforceCaps(nowMs);
    } catch (e) {
      this.counters.maintenance_errors_total++;
      this.noteError("maintain", e);
    } finally {
      this.maintaining = false;
    }
    return res;
  }

  /** Subscribe to the fix bus and run the maintenance pass; returns stop(). */
  start(): () => void {
    const unsubscribe = subscribeFixes((b) => { this.ingest(b); });
    const run = () => { void this.maintain(); };
    const first = setTimeout(run, 60_000);
    const timer = setInterval(run, TRACKED_LOG_MAINTAIN_EVERY_MS);
    first.unref?.();
    timer.unref?.();
    return () => { unsubscribe(); clearTimeout(first); clearInterval(timer); };
  }

  /** The `log` block of every tracked tail, the logs of tails no longer
   *  tracked, and the total against the caps (GET /api/data/aircraft/tracked). */
  async status(nowMs = this.now()): Promise<TrackedLogStatus> {
    const inv = this.inventory(nowMs, true);
    const caps = this.caps;
    const planes = this.opts.tracked();
    const byReg: Record<string, TailLogStatus> = {};
    const trackedHexes = new Set<string>();
    for (const tp of planes) {
      const hex = String(tp.hex || "").toLowerCase();
      const known = HEX_RE.test(hex);
      if (known) trackedHexes.add(hex);
      const s = await summarize(this.baseDir, known ? inv.get(hex) : undefined);
      const pause = known ? this.paused.get(hex) : undefined;
      const why = !this.enabled ? `${TRACKED_LOG_ENV}=0: the writer is off; what is on disk stays readable`
        : !known ? "the hex of this tail is not known yet (it is learned from the plane's first live fix): nothing is logged until then"
          : pause ? PAUSE_TEXT[pause.why] : null;
      byReg[tp.reg] = {
        hex: known ? hex : null, ...s, paused: why != null, why,
        paused_since_t: this.enabled && pause ? Math.round(pause.at / 1000) : null,
        cap_bytes: caps.perPlane,
      };
    }
    const untracked: UntrackedLogStatus[] = [];
    for (const p of inv.values()) {
      if (trackedHexes.has(p.hex)) continue;
      untracked.push({ hex: p.hex, ...(await summarize(this.baseDir, p)) });
    }
    const bytes = totalBytes(inv);
    return {
      enabled: this.enabled,
      by_reg: byReg,
      total: {
        bytes, planes_with_log: inv.size,
        cap_bytes_per_plane: caps.perPlane, cap_bytes_total: caps.total,
        over_total_cap: bytes > caps.total,
        free_bytes: this.freeBytes, min_free_bytes: this.minFree,
        untracked_logs: untracked,
        ...this.counters,
      },
      storage: `<archive>/${TRACKED_LOG_DIR}/<HEX>/YYYY-MM.jsonl, one file per plane per UTC month; a month is gzipped an hour after it ends`,
      rule: `every position fix any path receives for a tracked tail is written once, at the fix's own time, never thinned: skipped when it is not at least ${TRACKED_LOG_NEW_FIX_MIN_MS / 1000} s newer than the last one written for that aircraft, older than ${TRACKED_LOG_MAX_FIX_AGE_MS / 60_000} min, or dated more than ${TRACKED_LOG_MAX_FIX_AHEAD_MS / 1000} s ahead of our clock. The 15-minute day-trace backfill is not in this log`,
      cap_rule: `at most ${caps.perPlane} bytes on disk per plane and ${caps.total} in all. Over a cap the oldest month file goes first (of that plane; across all planes for the total); the current UTC month is never deleted, so a plane whose current month alone is at the cap stops being written until the month turns. Nothing is written while free disk is under ${this.minFree} bytes or unreadable`,
      note: "Removing a tail from the tracked list does NOT delete its log: it stays on disk (total.untracked_logs) and still counts toward the total cap, so it can be deleted by that cap, oldest month first. No age limit applies to this log, and it is not copied to or evicted by the R2 cold tier.",
    };
  }
}

export interface TailLogStatus extends PlaneLogSummary {
  hex: string | null;
  /** nothing is being written for this tail: the writer is off, its hex is
   *  not known, or the LAST fix offered for it was refused (cap / low disk) */
  paused: boolean;
  why: string | null;
  /** epoch s of that last refused fix; null when not paused by a cap or low disk */
  paused_since_t: number | null;
  cap_bytes: number;
}
export interface UntrackedLogStatus extends PlaneLogSummary { hex: string }
export interface TrackedLogStatus {
  enabled: boolean;
  by_reg: Record<string, TailLogStatus>;
  total: {
    bytes: number; planes_with_log: number; cap_bytes_per_plane: number; cap_bytes_total: number; over_total_cap: boolean;
    free_bytes: number | null; min_free_bytes: number; untracked_logs: UntrackedLogStatus[];
  } & TrackedPlaneLog["counters"];
  storage: string; rule: string; cap_rule: string; note: string;
}

/** Boot the log: subscribe to the fix bus, start maintenance (timers unref'd). */
export function startTrackedPlaneLog(opts: TrackedPlaneLogOptions): { log: TrackedPlaneLog; stop: () => void } {
  const log = new TrackedPlaneLog(opts);
  return { log, stop: log.start() };
}

/** test hook: forget cached first/last fix times */
export function _resetTrackedLogRangeCache(): void { rangeCache.clear(); }
