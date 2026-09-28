// archiveOffload.ts — R2 COLD TIER for the raw position archive
// (FLIGHT PROGRAM B2, 2026-09-28; rolling-window requirement from the human
// the same day: "after say 30 days all flight logs get deleted to free up
// storage, like it's rolling so we have X amount of days so you can replay
// back a month and then it gets rid of the day that goes over the amount").
//
// ROLE: R2 is CAPACITY, not longevity. One env knob, REPLAY_RETENTION_DAYS
// (default 30), is the TOTAL replay window across both tiers; nothing raw
// (aircraft/vessel hour files, flight_events/ daily JSONL) survives past it
// anywhere. Local disk is the HOT tier; R2 (a PRIVATE bucket,
// R2_ARCHIVE_BUCKET — never the public tiles bucket) is the cold tier that
// lets the Railway volume shed verified-offloaded days under pressure.
//
//   offload  (every 30 min, <= 48 files/tick, oldest first): each COMPLETED
//            hour file older than 2 days (and each flight_events day file)
//            -> gzip (streamed; .gz files upload as-is) -> PUT
//            archive/<kind>/<YYYY-MM-DD>/<HH>.jsonl.gz -> HEAD size verify
//            -> local manifest entry. Idempotent + resumable: the manifest
//            is the ledger, a changed local file re-uploads.
//   evict    (R2 configured only): a local day may be deleted EARLY only
//            after every hour of it is verified in R2, only when it is older
//            than LOCAL_HOT_DAYS or the volume is under pressure, never
//            younger than HOT_TIER_FLOOR_DAYS, oldest day first. Before the
//            delete it runs the same fold-before-delete chain the 6-hourly
//            rollup runs (fleet weekly + GNSS daily folds, then the per-day
//            <kind>_tracks summary) so the PERMANENT ROLLUPS never lose a day.
//   expire   (daily): R2 objects under archive/<kind>/<date>/ for dates past
//            the replay window are deleted (DeleteObjects batches, bounded);
//            local flight_events day files past it are deleted too. The
//            manifest is reconciled against the R2 listing (adopts objects a
//            lost manifest forgot, drops entries R2 no longer has).
//
// WITHOUT R2 (any of the four env vars missing) behavior is EXACTLY today's:
// local hour files live RAW_RETENTION_DAYS (30, unchanged) and are rolled +
// deleted by datacoreArchive.rollupOldDaysAsync; no eviction, no network.
// The only local action is expiring flight_events/ day files (a new
// directory with no prior behavior) at that same 30 days.
//
// NOT FLIGHT LOGS (never offloaded, never deleted here): the permanent tiny
// rollups — <kind>_tracks/ daily summaries, aircraft_fleet_weekly.json.gz,
// gnss_integrity_daily.json.gz. They feed signals; they are not replay data.

import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import crypto from "crypto";
import type { Express } from "express";
import { archiveBaseDir, RAW_RETENTION_DAYS, rollupDayAsync, type ArchiveKind } from "./datacoreArchive";
import { MIN_FREE_BYTES, readFreeBytes, volumeAllowsWrite } from "./globalScopes";
import { hourName, setColdHourSource, type ColdHourSource } from "./aircraftWindow";
import { createR2Client, errText, r2ConfigFromEnv, type R2Client } from "./r2Client";
import { preserveWeeklyBeforeRollup } from "./fleetUtilization";
import { preserveGnssIntegrityDailyBeforeRollup } from "./gnssIntegrityDaily";

// ── constants ───────────────────────────────────────────────────────────────
const DAY_MS = 86_400_000;
/** Remove a file/dir, reporting instead of throwing. `force` makes an
 *  already-missing path a success. Returns the error text, or null. */
function tryRemove(fp: string, recursive = false): string | null {
  try {
    fs.rmSync(fp, { force: true, recursive });
    return null;
  } catch (e: unknown) {
    return errText(e);
  }
}

export const OFFLOAD_HOUR_KINDS = ["aircraft", "vessels"] as const;
export type OffloadHourKind = (typeof OFFLOAD_HOUR_KINDS)[number];
/** agent C's daily flight-event JSONL directory under the archive base */
export const EVENTS_DIR = "flight_events";
export const R2_PREFIX = "archive";
/** a unit (hour / events day) uploads once it ENDED at least this long ago */
export const OFFLOAD_MIN_AGE_MS = 2 * DAY_MS;
export const OFFLOAD_MAX_UNITS_PER_TICK = 48;
export const OFFLOAD_TICK_MS = 30 * 60_000;
export const OFFLOAD_FIRST_TICK_DELAY_MS = 5 * 60_000;
/** "daily" with slack so a 30-min tick cadence never skips a calendar day */
export const DAILY_JOB_INTERVAL_MS = 23 * 3600_000;
export const DEFAULT_REPLAY_RETENTION_DAYS = 30;
export const MAX_REPLAY_RETENTION_DAYS = 400;
/** No early eviction younger than this: the live gate-2 GNSS-integrity
 *  signal (gnssIntegritySignal.computeGnssIntegritySignal) reads the last 21
 *  UTC days of LOCAL raw aircraft files; evicting inside that window would
 *  silently change a validated signal's input (MEASUREMENT INTEGRITY). 22 =
 *  21 + one day of margin. Lowering it is a [RULE-REVIEW], not a tune. */
export const HOT_TIER_FLOOR_DAYS = 22;
/** evict under pressure while free space is below this — twice the global-
 *  scopes guard, so eviction acts BEFORE the guard pauses global archiving */
export const HOT_TIER_TARGET_FREE_BYTES = 2 * MIN_FREE_BYTES;
export const MAX_EVICTIONS_PER_TICK = 6;
export const R2_DELETE_MAX_KEYS_PER_RUN = 5000;
export const COLD_MAX_HOURS_PER_REQUEST = 24;
export const COLD_CACHE_DEFAULT_MAX_BYTES = 512 * 1024 * 1024;
export const MANIFEST_NAME = "archive_offload_manifest.json";
const CONSECUTIVE_FAILURE_ABORT = 3;
const HOUR_RE = /^(\d{4}-\d{2}-\d{2})-(\d{2})\.jsonl(\.gz)?$/;
const EVENTS_RE = /^((\d{4}-\d{2}-\d{2})[A-Za-z0-9_.-]*?)\.jsonl(\.gz)?$/;

// ── retention config (env) ──────────────────────────────────────────────────
export interface RetentionConfig {
  /** REPLAY_RETENTION_DAYS after clamping — total window across both tiers */
  replayDays: number;
  replayClamped: boolean;
  /** LOCAL_HOT_DAYS after clamping — local days kept before age eviction */
  localHotDays: number;
  localHotClamped: boolean;
}

function intEnv(v: string | undefined): number | null {
  if (v == null || String(v).trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
}

export function retentionConfig(env: NodeJS.ProcessEnv = process.env): RetentionConfig {
  const reqReplay = intEnv(env.REPLAY_RETENTION_DAYS) ?? DEFAULT_REPLAY_RETENTION_DAYS;
  const replayDays = Math.min(MAX_REPLAY_RETENTION_DAYS, Math.max(HOT_TIER_FLOOR_DAYS, reqReplay));
  const hotMax = Math.min(replayDays, RAW_RETENTION_DAYS);
  const reqHot = intEnv(env.LOCAL_HOT_DAYS) ?? hotMax;
  const localHotDays = Math.min(hotMax, Math.max(HOT_TIER_FLOOR_DAYS, reqHot));
  return { replayDays, replayClamped: replayDays !== reqReplay, localHotDays, localHotClamped: localHotDays !== reqHot };
}

/** The replay window actually in force: REPLAY_RETENTION_DAYS once R2 holds
 *  the cold tier, else the unchanged local RAW_RETENTION_DAYS. */
export function effectiveRetentionDays(configured: boolean, rc: RetentionConfig): number {
  return configured ? rc.replayDays : RAW_RETENTION_DAYS;
}

// ── UTC date math (pure) ────────────────────────────────────────────────────
export function utcDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}
export function dayStartMs(date: string): number {
  return Date.parse(`${date}T00:00:00Z`);
}
/** whole UTC calendar days between `date` and today (today = 0) */
export function ageDays(date: string, nowMs: number): number {
  return Math.round((dayStartMs(utcDate(nowMs)) - dayStartMs(date)) / DAY_MS);
}
/** oldest UTC date still inside a `days`-day replay window (today - days). */
export function oldestReplayableDate(nowMs: number, days: number): string {
  return utcDate(dayStartMs(utcDate(nowMs)) - days * DAY_MS);
}
/** rolling boundary: day `days` back is KEPT, day `days + 1` back expires. */
export function isDateExpired(date: string, nowMs: number, days: number): boolean {
  return ageDays(date, nowMs) > days;
}

// ── R2 keys ─────────────────────────────────────────────────────────────────
/** "YYYY-MM-DD-HH" hour-file basename -> archive/<kind>/<date>/<HH>.jsonl.gz */
export function hourKey(kind: string, hourBase: string): string {
  return `${R2_PREFIX}/${kind}/${hourBase.slice(0, 10)}/${hourBase.slice(11, 13)}.jsonl.gz`;
}
export function eventsKey(stem: string, date: string): string {
  return `${R2_PREFIX}/${EVENTS_DIR}/${date}/${stem}.jsonl.gz`;
}
export function dateOfKey(key: string): string | null {
  const m = /^archive\/[a-z_]+\/(\d{4}-\d{2}-\d{2})\//.exec(key);
  return m ? m[1] : null;
}

// ── manifest (local ledger of verified offloads) ─────────────────────────────
export interface ManifestEntry {
  kind: string;
  date: string;
  /** bytes of the R2 object (verified by HEAD) */
  bytes: number;
  /** local source flavor at upload; "adopted" = found in R2 by reconcile */
  src: "gz" | "plain" | "adopted";
  srcBytes: number | null;
  sha256?: string;
  at: number;
}
export interface OffloadManifest {
  version: 1;
  entries: Record<string, ManifestEntry>;
  lastDailyRunAt: number | null;
}

export function loadManifest(base: string): OffloadManifest {
  try {
    const j = JSON.parse(fs.readFileSync(path.join(base, MANIFEST_NAME), "utf8"));
    return {
      version: 1,
      entries: j && typeof j.entries === "object" && j.entries ? j.entries : {},
      lastDailyRunAt: Number.isFinite(j?.lastDailyRunAt) ? j.lastDailyRunAt : null,
    };
  } catch {
    return { version: 1, entries: {}, lastDailyRunAt: null };
  }
}

export function saveManifest(base: string, m: OffloadManifest): void {
  fs.mkdirSync(base, { recursive: true });
  const fp = path.join(base, MANIFEST_NAME);
  const tmp = `${fp}.tmp-${process.pid}`;
  fs.writeFileSync(tmp, JSON.stringify(m));
  fs.renameSync(tmp, fp); // atomic: a crash never leaves a half manifest
}

// ── local unit enumeration ──────────────────────────────────────────────────
export interface LocalUnit {
  kind: string;
  key: string;
  date: string;
  startMs: number;
  endMs: number;
  /** basename(s) on disk for this unit */
  names: string[];
  fp: string;
  flavor: "gz" | "plain";
  size: number;
  /** .jsonl AND .jsonl.gz both present (mid-compress / backfill) — skipped */
  bothFlavors: boolean;
}

function statSize(fp: string): number | null {
  try { const s = fs.statSync(fp); return s.isFile() ? s.size : null; } catch { return null; }
}

export function listLocalHourUnits(base: string, kind: string): LocalUnit[] {
  let names: string[] = [];
  try { names = fs.readdirSync(path.join(base, kind)); } catch { return []; }
  const byHour = new Map<string, string[]>();
  for (const f of names) {
    const m = HOUR_RE.exec(f);
    if (!m) continue;
    const hb = `${m[1]}-${m[2]}`;
    (byHour.get(hb) || byHour.set(hb, []).get(hb)!).push(f);
  }
  const out: LocalUnit[] = [];
  for (const [hb, fs_] of Array.from(byHour.entries())) {
    const gzName = fs_.find((f) => f.endsWith(".gz"));
    const plainName = fs_.find((f) => !f.endsWith(".gz"));
    const pick = gzName ?? plainName!;
    const fp = path.join(base, kind, pick);
    const size = statSize(fp);
    if (size == null) continue;
    const startMs = Date.parse(`${hb.slice(0, 10)}T${hb.slice(11, 13)}:00:00Z`);
    if (!Number.isFinite(startMs)) continue;
    out.push({
      kind, key: hourKey(kind, hb), date: hb.slice(0, 10), startMs, endMs: startMs + 3600_000,
      names: fs_.slice().sort(), fp, flavor: gzName ? "gz" : "plain", size,
      bothFlavors: !!gzName && !!plainName,
    });
  }
  return out.sort((a, b) => a.startMs - b.startMs || a.kind.localeCompare(b.kind));
}

export function listLocalEventUnits(base: string): LocalUnit[] {
  let names: string[] = [];
  try { names = fs.readdirSync(path.join(base, EVENTS_DIR)); } catch { return []; }
  const byStem = new Map<string, string[]>();
  for (const f of names) {
    const m = EVENTS_RE.exec(f);
    if (!m) continue;
    (byStem.get(m[1]) || byStem.set(m[1], []).get(m[1])!).push(f);
  }
  const out: LocalUnit[] = [];
  for (const [stem, fs_] of Array.from(byStem.entries())) {
    const date = stem.slice(0, 10);
    const startMs = dayStartMs(date);
    if (!Number.isFinite(startMs)) continue;
    const gzName = fs_.find((f) => f.endsWith(".gz"));
    const plainName = fs_.find((f) => !f.endsWith(".gz"));
    const pick = gzName ?? plainName!;
    const fp = path.join(base, EVENTS_DIR, pick);
    const size = statSize(fp);
    if (size == null) continue;
    out.push({
      kind: EVENTS_DIR, key: eventsKey(stem, date), date, startMs, endMs: startMs + DAY_MS,
      names: fs_.slice().sort(), fp, flavor: gzName ? "gz" : "plain", size,
      bothFlavors: !!gzName && !!plainName,
    });
  }
  return out.sort((a, b) => a.startMs - b.startMs);
}

/** Already verified in R2 for THIS local content? (a changed file re-uploads)
 *  An "adopted" row (found by the daily listing, not uploaded+HEAD-verified
 *  by this ledger) only counts when the local file is a .gz whose size
 *  equals the listed object — .gz files upload byte-for-byte, so equal size
 *  is an independent confirmation; anything else re-uploads rather than let
 *  an unverified R2 copy license a local delete. */
export function isOffloaded(u: LocalUnit, m: OffloadManifest): boolean {
  const e = m.entries[u.key];
  if (!e) return false;
  if (e.src === "adopted") return u.flavor === "gz" && e.bytes === u.size;
  return e.src === u.flavor && e.srcBytes === u.size;
}

/** Stream a unit into its gzipped upload body + sha256 (the .gz files go up
 *  byte-for-byte; a plain .jsonl is gzipped in chunks — the loop breathes). */
export function gzBodyFor(fp: string, flavor: "gz" | "plain"): Promise<{ body: Buffer; sha256: string }> {
  return new Promise((resolve, reject) => {
    const src = fs.createReadStream(fp);
    const stream: NodeJS.ReadableStream = flavor === "plain" ? src.pipe(zlib.createGzip()) : src;
    const hash = crypto.createHash("sha256");
    const chunks: Buffer[] = [];
    let settled = false;
    const fail = (e: unknown) => { if (!settled) { settled = true; src.destroy(); reject(e); } };
    src.on("error", fail);
    stream.on("error", fail);
    stream.on("data", (c: Buffer) => { hash.update(c); chunks.push(c); });
    stream.on("end", () => { if (!settled) { settled = true; resolve({ body: Buffer.concat(chunks), sha256: hash.digest("hex") }); } });
  });
}

// ── cold-tier reader (implements aircraftWindow.ColdHourSource) ─────────────
export interface ColdSourceOpts {
  client: R2Client;
  getManifest: () => OffloadManifest;
  retentionDays: () => number;
  nowMs?: () => number;
  cacheDir?: string;
  cacheMaxBytes?: number;
  maxHoursPerRequest?: number;
  maxObjectBytes?: number;
}

export interface ColdSourceStats { files: number; bytes: number; maxBytes: number; fetches: number; hits: number; errors: number; removeErrors: number; lastError: string | null }

export function createColdHourSource(o: ColdSourceOpts): ColdHourSource & { stats(): ColdSourceStats; clear(): void } {
  const now = o.nowMs ?? (() => Date.now());
  const cacheDir = o.cacheDir ?? path.join(os.tmpdir(), "voltrade_r2_hour_cache");
  const maxBytes = o.cacheMaxBytes ?? (intEnv(process.env.R2_COLD_CACHE_MAX_BYTES) ?? COLD_CACHE_DEFAULT_MAX_BYTES);
  // the cache dir is ours alone: start clean (a previous container's files
  // are unindexed and would silently eat the byte cap)
  const wipeErr = tryRemove(cacheDir, true);
  if (wipeErr) console.warn(`[archive-offload] could not clear cold cache dir ${cacheDir}: ${wipeErr} — stale files may count against the cap until the next boot`);
  const cache = new Map<string, { fp: string; bytes: number; used: number }>();
  const inflight = new Map<string, Promise<{ fp: string; gz: boolean } | null>>();
  let clock = 0;
  let total = 0;
  const st = { fetches: 0, hits: 0, errors: 0, removeErrors: 0, lastError: null as string | null };

  const evict = (keep: string) => {
    while (total > maxBytes && cache.size > 1) {
      let lruKey: string | null = null;
      let lruUsed = Infinity;
      cache.forEach((v, k) => { if (k !== keep && v.used < lruUsed) { lruUsed = v.used; lruKey = k; } });
      if (lruKey == null) break;
      const v = cache.get(lruKey)!;
      cache.delete(lruKey);
      total -= v.bytes;
      // unlinking a file another request is mid-stream on is safe on POSIX
      fs.promises.unlink(v.fp).catch(() => {});
    }
  };

  return {
    maxHoursPerRequest: o.maxHoursPerRequest ?? COLD_MAX_HOURS_PER_REQUEST,
    locate(kind, hourStartSec) {
      const hb = hourName(hourStartSec);
      if (isDateExpired(hb.slice(0, 10), now(), o.retentionDays())) return "expired";
      if (!o.client.configured) return "absent";
      return o.getManifest().entries[hourKey(kind, hb)] ? "available" : "absent";
    },
    async fetch(kind, hourStartSec, signal) {
      const key = hourKey(kind, hourName(hourStartSec));
      const hit = cache.get(key);
      if (hit && fs.existsSync(hit.fp)) { hit.used = ++clock; st.hits++; return { fp: hit.fp, gz: true }; }
      if (hit) { cache.delete(key); total -= hit.bytes; }
      const running = inflight.get(key);
      if (running) return running;
      const p = (async () => {
        st.fetches++;
        const fp = path.join(cacheDir, key.replace(/\//g, "__"));
        const r = await o.client.getObjectToFile(key, fp, { maxBytes: o.maxObjectBytes, signal });
        if (!r.ok) {
          st.errors++;
          st.lastError = `${key}: ${r.error || "fetch failed"}`;
          return null;
        }
        const bytes = r.bytes ?? 0;
        cache.set(key, { fp, bytes, used: ++clock });
        total += bytes;
        evict(key);
        return { fp, gz: true };
      })().finally(() => inflight.delete(key));
      inflight.set(key, p);
      return p;
    },
    stats: () => ({ files: cache.size, bytes: total, maxBytes, ...st }),
    clear: () => {
      // a cache file that won't delete is harmless (the dir is wiped at boot);
      // count it so a systematically unwritable /tmp is visible in stats
      cache.forEach((v) => { if (tryRemove(v.fp)) st.removeErrors++; });
      cache.clear();
      total = 0;
    },
  };
}

// ── cost model (pure) ───────────────────────────────────────────────────────
/** Cloudflare R2 list prices (2026-09): storage beyond 10 GB-month free,
 *  Class A (PUT/LIST/POST) beyond 1M/month free, Class B (GET/HEAD) beyond
 *  10M/month free; egress is free. The free tier is ACCOUNT-wide, shared
 *  with the public map-tiles bucket (~1.35 GB). Decimal GB. */
export const R2_PRICING = {
  storageUsdPerGbMonth: 0.015,
  classAUsdPerMillion: 4.5,
  classBUsdPerMillion: 0.36,
  freeStorageGb: 10,
  freeClassA: 1_000_000,
  freeClassB: 10_000_000,
  tilesBucketBytes: 1.35e9,
};

export interface CostEstimate {
  storedBytes: number;
  billableGb: number;
  storageUsd: number;
  classAOps: number;
  classBOps: number;
  opsUsd: number;
  totalUsd: number;
}

export function estimateR2MonthlyCost(o: { storedBytes: number; classAOps: number; classBOps: number; otherBucketBytes?: number }): CostEstimate {
  const other = o.otherBucketBytes ?? R2_PRICING.tilesBucketBytes;
  const billableGb = Math.max(0, (o.storedBytes + other) / 1e9 - R2_PRICING.freeStorageGb);
  const storageUsd = billableGb * R2_PRICING.storageUsdPerGbMonth;
  const opsUsd =
    (Math.max(0, o.classAOps - R2_PRICING.freeClassA) / 1e6) * R2_PRICING.classAUsdPerMillion +
    (Math.max(0, o.classBOps - R2_PRICING.freeClassB) / 1e6) * R2_PRICING.classBUsdPerMillion;
  const r = (x: number) => Math.round(x * 10000) / 10000;
  return {
    storedBytes: Math.round(o.storedBytes), billableGb: r(billableGb), storageUsd: r(storageUsd),
    classAOps: Math.round(o.classAOps), classBOps: Math.round(o.classBOps), opsUsd: r(opsUsd),
    totalUsd: r(storageUsd + opsUsd),
  };
}

// ── the service ─────────────────────────────────────────────────────────────
export interface EvictionRecord { kind: string; date: string; files: number; reason: "expired" | "age" | "pressure" }

export interface TickSummary {
  at: number;
  configured: boolean;
  uploaded: number;
  uploadFailed: number;
  bytesUploaded: number;
  pending: number;
  skippedBothFlavors: number;
  evicted: EvictionRecord[];
  evictionNote?: string;
  localEventsExpired: number;
  daily?: { r2Deleted: number; r2DeleteErrors: number; adopted: number; dropped: number };
  errors: string[];
}

export interface OffloadDeps {
  base?: string;
  client?: R2Client;
  env?: NodeJS.ProcessEnv;
  nowMs?: () => number;
  freeBytes?: (dir: string) => number | null;
  /** fold-before-delete hook for aircraft days (fleet weekly + GNSS daily);
   *  called with a cutoff that selects exactly the days being evicted */
  beforeEvictAircraft?: (base: string, cutoffMs: number, nowMs: number) => Promise<void>;
  rollupDay?: (kind: ArchiveKind, day: string, base: string) => Promise<string[] | null>;
  maxUnitsPerTick?: number;
  maxEvictionsPerTick?: number;
  cacheDir?: string;
  cacheMaxBytes?: number;
  log?: (msg: string) => void;
  logError?: (msg: string) => void;
}

/** production fold chain: the SAME two folds the 6-hourly rollup runs first
 *  (server/routes.ts), with a cutoff 1s before the midnight after the newest
 *  evicted day so exactly the evicted days (all older days are gone or in
 *  the same eviction run — eviction is oldest-first contiguous) are folded;
 *  their processedFiles sets make the later RAW_RETENTION_DAYS pass a no-op. */
export async function foldAircraftBeforeEvict(base: string, cutoffMs: number, nowMs: number): Promise<void> {
  const retentionDays = (nowMs - cutoffMs) / DAY_MS;
  await preserveWeeklyBeforeRollup(base, nowMs, retentionDays).catch((e) => console.error("[archive-offload] fleet fold:", e?.message || e));
  await preserveGnssIntegrityDailyBeforeRollup(base, nowMs, retentionDays).catch((e) => console.error("[archive-offload] gnss fold:", e?.message || e));
}

export function createArchiveOffloadService(deps: OffloadDeps = {}) {
  const base = deps.base ?? archiveBaseDir();
  const env = deps.env ?? process.env;
  const client = deps.client ?? createR2Client(r2ConfigFromEnv(env));
  const now = deps.nowMs ?? (() => Date.now());
  const freeBytes = deps.freeBytes ?? readFreeBytes;
  const beforeEvictAircraft = deps.beforeEvictAircraft ?? foldAircraftBeforeEvict;
  const rollupDay = deps.rollupDay ?? ((k: ArchiveKind, d: string, b: string) => rollupDayAsync(k, d, b));
  const maxUnits = deps.maxUnitsPerTick ?? OFFLOAD_MAX_UNITS_PER_TICK;
  const maxEvictions = deps.maxEvictionsPerTick ?? MAX_EVICTIONS_PER_TICK;
  const log = deps.log ?? ((m: string) => console.log(`[archive-offload] ${m}`));
  const logError = deps.logError ?? ((m: string) => console.error(`[archive-offload] ${m}`));
  const rc = () => retentionConfig(env);
  const retentionDays = () => effectiveRetentionDays(client.configured, rc());

  let manifest = loadManifest(base);
  const state = {
    lastRun: null as number | null,
    lastSummary: null as TickSummary | null,
    lastError: null as string | null,
    lastErrorAt: null as number | null,
    bootAt: now(),
    uploadsSinceBoot: 0,
    bytesSinceBoot: 0,
  };
  let inFlight = false;
  let timers: NodeJS.Timeout[] = [];

  const coldSource = createColdHourSource({
    client, getManifest: () => manifest, retentionDays, nowMs: now,
    cacheDir: deps.cacheDir, cacheMaxBytes: deps.cacheMaxBytes,
  });

  const noteError = (s: TickSummary, msg: string) => {
    s.errors.push(msg);
    state.lastError = msg;
    state.lastErrorAt = now();
    logError(msg);
  };

  async function offloadPhase(s: TickSummary): Promise<void> {
    const t = now();
    const days = retentionDays();
    const units = [
      ...OFFLOAD_HOUR_KINDS.flatMap((k) => listLocalHourUnits(base, k)),
      ...listLocalEventUnits(base),
    ].filter((u) => u.endMs <= t - OFFLOAD_MIN_AGE_MS && !isDateExpired(u.date, t, days))
     .sort((a, b) => a.startMs - b.startMs || a.kind.localeCompare(b.kind));
    const todo: LocalUnit[] = [];
    for (const u of units) {
      if (isOffloaded(u, manifest)) continue;
      if (u.bothFlavors) { s.skippedBothFlavors++; continue; }
      todo.push(u);
    }
    s.pending = todo.length;
    let consecutiveFails = 0;
    for (const u of todo.slice(0, maxUnits)) {
      try {
        const { body, sha256 } = await gzBodyFor(u.fp, u.flavor);
        const put = await client.putObject(u.key, body, "application/gzip", undefined, { payloadSha256: sha256 });
        if (!put.ok) throw new Error(`put ${u.key}: ${put.error}`);
        const head = await client.headObject(u.key);
        if (!head.ok || !head.exists || head.size !== body.length) {
          throw new Error(`verify ${u.key}: ${head.error || `R2 size ${head.size ?? "none"} != uploaded ${body.length}`}`);
        }
        manifest.entries[u.key] = { kind: u.kind, date: u.date, bytes: body.length, src: u.flavor, srcBytes: u.size, sha256, at: now() };
        s.uploaded++;
        s.bytesUploaded += body.length;
        s.pending--;
        consecutiveFails = 0;
      } catch (e: unknown) {
        s.uploadFailed++;
        noteError(s, errText(e));
        if (++consecutiveFails >= CONSECUTIVE_FAILURE_ABORT) {
          noteError(s, `offload paused this tick after ${consecutiveFails} consecutive failures`);
          break;
        }
      }
    }
    state.uploadsSinceBoot += s.uploaded;
    state.bytesSinceBoot += s.bytesUploaded;
  }

  /** Which local days may go early, oldest-first contiguous per kind. */
  function evictionCandidates(t: number, underPressure: boolean): Array<{ kind: OffloadHourKind; date: string; units: LocalUnit[]; reason: EvictionRecord["reason"] }> {
    const cfg = rc();
    const out: Array<{ kind: OffloadHourKind; date: string; units: LocalUnit[]; reason: EvictionRecord["reason"] }> = [];
    const rollupCutoff = t - RAW_RETENTION_DAYS * DAY_MS;
    for (const kind of OFFLOAD_HOUR_KINDS) {
      const byDay = new Map<string, LocalUnit[]>();
      for (const u of listLocalHourUnits(base, kind)) (byDay.get(u.date) || byDay.set(u.date, []).get(u.date)!).push(u);
      for (const date of Array.from(byDay.keys()).sort()) {
        // past RAW_RETENTION_DAYS the existing 6-hourly rollup owns the day
        if (dayStartMs(date) < rollupCutoff) continue;
        const age = ageDays(date, t);
        if (age < HOT_TIER_FLOOR_DAYS) break;
        const us = byDay.get(date)!;
        const expired = isDateExpired(date, t, cfg.replayDays);
        const verified = us.every((u) => !u.bothFlavors && isOffloaded(u, manifest));
        const reason: EvictionRecord["reason"] | null =
          expired ? "expired"
          : verified && age > cfg.localHotDays ? "age"
          : verified && underPressure ? "pressure"
          : null;
        if (!reason) break; // contiguity: never evict past an older day that stays
        out.push({ kind, date, units: us, reason });
      }
    }
    return out.sort((a, b) => a.date.localeCompare(b.date) || a.kind.localeCompare(b.kind));
  }

  async function evictPhase(s: TickSummary): Promise<void> {
    const t = now();
    const free = freeBytes(base);
    const underPressure = !volumeAllowsWrite(free, HOT_TIER_TARGET_FREE_BYTES);
    const cands = evictionCandidates(t, underPressure);
    if (!cands.length) {
      if (underPressure) s.evictionNote = `volume under pressure (${free} bytes free) but no day is both >= ${HOT_TIER_FLOOR_DAYS} days old and fully verified in R2 yet`;
      return;
    }
    const blockedKinds = new Set<string>();
    for (const c of cands) {
      if (s.evicted.length >= maxEvictions) break;
      if (blockedKinds.has(c.kind)) continue;
      if (c.reason === "pressure" && volumeAllowsWrite(freeBytes(base), HOT_TIER_TARGET_FREE_BYTES)) continue;
      if (c.kind === "aircraft") {
        await beforeEvictAircraft(base, dayStartMs(c.date) + DAY_MS - 1000, t);
      }
      const rolled = await rollupDay(c.kind, c.date, base);
      if (rolled == null) {
        blockedKinds.add(c.kind);
        noteError(s, `evict ${c.kind}/${c.date}: day rollup failed — local files kept`);
        continue;
      }
      // delete ONLY what the rollup read AND (unless expired) what R2 verified
      const verifiedNames = new Set(c.units.filter((u) => c.reason === "expired" || isOffloaded(u, manifest)).flatMap((u) => u.names));
      const unverified = rolled.filter((f) => !verifiedNames.has(f));
      if (unverified.length) {
        blockedKinds.add(c.kind);
        noteError(s, `evict ${c.kind}/${c.date}: ${unverified.length} file(s) not verified in R2 — local files kept`);
        continue;
      }
      let removed = 0;
      for (const f of rolled) {
        const rmErr = tryRemove(path.join(base, c.kind, f));
        if (rmErr) noteError(s, `evict unlink ${c.kind}/${f}: ${rmErr}`);
        else removed++;
      }
      s.evicted.push({ kind: c.kind, date: c.date, files: removed, reason: c.reason });
      log(`evicted local ${c.kind}/${c.date} (${removed} hour files, reason=${c.reason}; R2 copy ${c.reason === "expired" ? "not needed — past replay window" : "verified"})`);
    }
  }

  /** local flight_events day files past the replay window (every tick; cheap) */
  function expireLocalEvents(s: TickSummary): void {
    const t = now();
    const days = retentionDays();
    for (const u of listLocalEventUnits(base)) {
      if (!isDateExpired(u.date, t, days)) continue;
      for (const f of u.names) {
        const rmErr = tryRemove(path.join(base, EVENTS_DIR, f));
        if (rmErr) noteError(s, `expire ${EVENTS_DIR}/${f}: ${rmErr}`);
        else s.localEventsExpired++;
      }
    }
    if (s.localEventsExpired) log(`expired ${s.localEventsExpired} local ${EVENTS_DIR} file(s) past the ${days}-day replay window`);
  }

  async function dailyPhase(s: TickSummary): Promise<void> {
    const t = now();
    const oldest = oldestReplayableDate(t, retentionDays());
    const daily = { r2Deleted: 0, r2DeleteErrors: 0, adopted: 0, dropped: 0 };
    let budget = R2_DELETE_MAX_KEYS_PER_RUN;
    let allOk = true;
    for (const kind of [...OFFLOAD_HOUR_KINDS, EVENTS_DIR]) {
      const list = await client.listObjects(`${R2_PREFIX}/${kind}/`, { maxPages: 20 });
      if (!list.ok) { allOk = false; noteError(s, `daily list ${kind}: ${list.error}`); continue; }
      const seen = new Set<string>();
      const expired: string[] = [];
      for (const o of list.objects) {
        const d = dateOfKey(o.key);
        if (!d) continue;
        seen.add(o.key);
        if (d < oldest) { expired.push(o.key); continue; }
        const e = manifest.entries[o.key];
        if (!e) {
          manifest.entries[o.key] = { kind, date: d, bytes: o.size, src: "adopted", srcBytes: null, at: t };
          daily.adopted++;
        } else if (e.bytes !== o.size) {
          delete manifest.entries[o.key]; // manifest's claim is wrong: re-upload / re-adopt
          daily.dropped++;
        }
      }
      if (!list.truncated) {
        for (const [k, e] of Object.entries(manifest.entries)) {
          if (e.kind === kind && !seen.has(k)) { delete manifest.entries[k]; daily.dropped++; }
        }
      }
      const batch = expired.slice(0, budget);
      if (batch.length) {
        budget -= batch.length;
        const del = await client.deleteObjects(batch);
        const failed = new Set(del.errors.map((x) => x.key));
        if (!del.ok && !del.errors.length) {
          allOk = false;
          noteError(s, `daily delete ${kind}: ${del.error}`);
        } else {
          for (const k of batch) {
            if (failed.has(k)) continue;
            delete manifest.entries[k];
            daily.r2Deleted++;
          }
          daily.r2DeleteErrors += failed.size;
          if (failed.size) noteError(s, `daily delete ${kind}: ${failed.size} key(s) failed`);
        }
        const dates = Array.from(new Set(batch.map((k) => dateOfKey(k)))).sort();
        log(`R2 rolling delete ${kind}: ${batch.length - failed.size} object(s) from ${dates[0]}..${dates[dates.length - 1]} (older than ${oldest})`);
      }
    }
    // stale manifest rows past the window whose objects are already gone
    for (const [k, e] of Object.entries(manifest.entries)) {
      if (e.date < oldest && allOk) { delete manifest.entries[k]; daily.dropped++; }
    }
    s.daily = daily;
    if (allOk && budget > 0) manifest.lastDailyRunAt = t;
  }

  async function runTick(opts: { forceDaily?: boolean } = {}): Promise<TickSummary | null> {
    if (inFlight) return null;
    inFlight = true;
    const s: TickSummary = {
      at: now(), configured: client.configured, uploaded: 0, uploadFailed: 0, bytesUploaded: 0,
      pending: 0, skippedBothFlavors: 0, evicted: [], localEventsExpired: 0, errors: [],
    };
    try {
      if (client.configured) {
        await offloadPhase(s);
        const dailyDue = opts.forceDaily || manifest.lastDailyRunAt == null || now() - manifest.lastDailyRunAt >= DAILY_JOB_INTERVAL_MS;
        if (dailyDue) await dailyPhase(s);
        try { saveManifest(base, manifest); } catch (e: unknown) { noteError(s, `manifest save: ${errText(e)}`); }
        await evictPhase(s);
      }
      expireLocalEvents(s);
    } catch (e: unknown) {
      noteError(s, `tick: ${errText(e)}`);
    } finally {
      try {
        if (client.configured) saveManifest(base, manifest);
      } catch (e: unknown) {
        noteError(s, `manifest save (final): ${errText(e)}`);
      }
      state.lastRun = s.at;
      state.lastSummary = s;
      inFlight = false;
    }
    return s;
  }

  // ── status (the /api/data/archive/offload-status payload) ─────────────────
  let statusCache: { at: number; data: unknown } | null = null;

  async function localTierStats(): Promise<Record<string, { files: number; bytes: number; oldestDate: string | null; newestDate: string | null }>> {
    const out: Record<string, { files: number; bytes: number; oldestDate: string | null; newestDate: string | null }> = {};
    for (const kind of [...OFFLOAD_HOUR_KINDS, EVENTS_DIR]) {
      const dir = path.join(base, kind);
      let names: string[] = [];
      try { names = (await fs.promises.readdir(dir)).sort(); } catch { names = []; }
      const re = kind === EVENTS_DIR ? EVENTS_RE : HOUR_RE;
      let files = 0, bytes = 0;
      let oldest: string | null = null, newest: string | null = null;
      for (const f of names) {
        if (!re.test(f)) continue;
        try {
          const st = await fs.promises.stat(path.join(dir, f));
          if (!st.isFile()) continue;
          files++; bytes += st.size;
          const d = f.slice(0, 10);
          if (!oldest || d < oldest) oldest = d;
          if (!newest || d > newest) newest = d;
        } catch {
          continue; // file rotated/evicted between readdir and stat — not part of the tier any more
        }
      }
      out[kind] = { files, bytes, oldestDate: oldest, newestDate: newest };
    }
    return out;
  }

  function r2TierStats() {
    const byKind: Record<string, { objects: number; bytes: number; oldestDate: string | null; newestDate: string | null }> = {};
    for (const e of Object.values(manifest.entries)) {
      const k = (byKind[e.kind] ||= { objects: 0, bytes: 0, oldestDate: null, newestDate: null });
      k.objects++; k.bytes += e.bytes;
      if (!k.oldestDate || e.date < k.oldestDate) k.oldestDate = e.date;
      if (!k.newestDate || e.date > k.newestDate) k.newestDate = e.date;
    }
    return byKind;
  }

  /** bytes/day of raw log growth (gz): the R2 ledger's last 7 FULL offloaded
   *  days when it has them, else the local tier's complete days. */
  function bytesPerDayEstimate(local: Record<string, { files: number; bytes: number }>): { bytesPerDay: number; basis: string } {
    const t = now();
    const perDay = new Map<string, number>();
    for (const e of Object.values(manifest.entries)) {
      if (e.kind === EVENTS_DIR || ageDays(e.date, t) < 3) continue;
      perDay.set(e.date, (perDay.get(e.date) || 0) + e.bytes);
    }
    const days = Array.from(perDay.keys()).sort().slice(-7);
    if (days.length >= 3) {
      const sum = days.reduce((a, d) => a + perDay.get(d)!, 0);
      return { bytesPerDay: sum / days.length, basis: `R2 ledger, mean of last ${days.length} offloaded days (aircraft+vessels gz)` };
    }
    let bytes = 0, files = 0;
    for (const k of OFFLOAD_HOUR_KINDS) { bytes += local[k]?.bytes || 0; files += local[k]?.files || 0; }
    // files are per-kind hours: bytes/hour-file x 24 x kinds
    const kindsPresent = OFFLOAD_HOUR_KINDS.filter((k) => (local[k]?.files || 0) > 0).length || 1;
    const bpd = files ? (bytes / files) * 24 * kindsPresent : 0;
    return { bytesPerDay: bpd, basis: "local tier: mean bytes per hour file x 24 x kinds (includes today's still-growing hour)" };
  }

  async function status(): Promise<any> {
    const t = now();
    if (statusCache && t - statusCache.at < 60_000) return statusCache.data;
    const cfg = rc();
    const days = retentionDays();
    const local = await localTierStats();
    const r2 = r2TierStats();
    const hourEntries = Object.values(manifest.entries).filter((e) => e.kind !== EVENTS_DIR);
    const r2Bytes = Object.values(r2).reduce((a, k) => a + k.bytes, 0);
    const r2Objects = Object.values(r2).reduce((a, k) => a + k.objects, 0);
    const localBytes = Object.values(local).reduce((a, k) => a + k.bytes, 0);
    const free = freeBytes(base);
    const est = bytesPerDayEstimate(local);
    const steady = est.bytesPerDay * (days + 1);
    const unitsPerDay = OFFLOAD_HOUR_KINDS.length * 24 + 1;
    const classA = unitsPerDay * 30 + 30 * 3 * 2;     // PUTs + daily LIST/DeleteObjects
    const classB = unitsPerDay * 30 + 30 * 24 * 20;  // HEAD verifies + ~20 cold replay requests/day x 24 hours
    const data = {
      configured: client.configured,
      retentionDays: days,
      retentionSource: client.configured ? "REPLAY_RETENTION_DAYS (both tiers)" : "RAW_RETENTION_DAYS (local only — R2 not configured)",
      replayRetentionDays: cfg.replayDays,
      replayRetentionClamped: cfg.replayClamped,
      localHotDays: cfg.localHotDays,
      hotTierFloorDays: HOT_TIER_FLOOR_DAYS,
      oldestReplayableDate: oldestReplayableDate(t, days),
      lastRun: state.lastRun ? new Date(state.lastRun).toISOString() : null,
      lastDailyRun: manifest.lastDailyRunAt ? new Date(manifest.lastDailyRunAt).toISOString() : null,
      lastRunSummary: state.lastSummary,
      lastError: state.lastError,
      lastErrorAt: state.lastErrorAt ? new Date(state.lastErrorAt).toISOString() : null,
      hoursOffloaded: hourEntries.length,
      bytesOffloaded: r2Bytes,
      uploadsSinceBoot: state.uploadsSinceBoot,
      tiers: {
        local: { bytes: localBytes, byKind: local },
        r2: { bytes: r2Bytes, objects: r2Objects, byKind: r2, source: "local manifest, reconciled against the R2 listing daily" },
      },
      volume: {
        freeBytes: free,
        hotTierTargetFreeBytes: HOT_TIER_TARGET_FREE_BYTES,
        globalScopesGuardBytes: MIN_FREE_BYTES,
        underPressure: !volumeAllowsWrite(free, HOT_TIER_TARGET_FREE_BYTES),
      },
      coldCache: coldSource.stats(),
      estimatedBytesPerDay: Math.round(est.bytesPerDay),
      estimatedMonthlyBytes: Math.round(est.bytesPerDay * 30),
      estimatedSteadyStateBytes: Math.round(steady),
      estimateBasis: est.basis,
      estimatedMonthlyCost: {
        atCurrentGrowth: estimateR2MonthlyCost({ storedBytes: steady, classAOps: classA, classBOps: classB }),
        at10xGrowth: estimateR2MonthlyCost({ storedBytes: steady * 10, classAOps: classA, classBOps: classB }),
        note: "rolling window: storage PLATEAUS at ~(retentionDays+1) days of data instead of growing; free tier (10 GB, 1M Class A, 10M Class B) is account-wide and shared with the ~1.35 GB tiles bucket; R2 egress is free",
      },
      notes: [
        "Permanent rollups (<kind>_tracks daily summaries, aircraft_fleet_weekly.json.gz, gnss_integrity_daily.json.gz) are not flight logs: never offloaded, never deleted by this job.",
        `Local early eviction never touches days younger than ${HOT_TIER_FLOOR_DAYS} (the live GNSS-integrity signal reads 21 days of local raw files) and only deletes a day after every hour of it is verified in R2.`,
      ],
      generated_at: new Date(t).toISOString(),
    };
    statusCache = { at: t, data };
    return data;
  }

  return {
    base,
    client,
    coldSource,
    runTick,
    status,
    getManifest: () => manifest,
    /** test seam: drop the 60s status cache */
    _resetStatusCache: () => { statusCache = null; },
    start(): void {
      if (timers.length) return;
      const first = setTimeout(() => { void runTick(); }, OFFLOAD_FIRST_TICK_DELAY_MS);
      const every = setInterval(() => { void runTick(); }, OFFLOAD_TICK_MS);
      (first as any).unref?.();
      (every as any).unref?.();
      timers = [first, every];
    },
    stop(): void {
      for (const t of timers) { clearTimeout(t); clearInterval(t); }
      timers = [];
    },
  };
}

export type ArchiveOffloadService = ReturnType<typeof createArchiveOffloadService>;

/** Boot wiring: ONE call from server/routes.ts. Registers the Time Machine
 *  cold-tier source, starts the unref'd offload timer, and serves
 *  GET /api/data/archive/offload-status (no secrets: never the keys, the
 *  account id, or the bucket name). */
export function registerArchiveOffloadRoutes(app: Express, deps: OffloadDeps & { startTimers?: boolean } = {}): ArchiveOffloadService {
  const svc = createArchiveOffloadService(deps);
  setColdHourSource(svc.coldSource);
  if (deps.startTimers !== false) svc.start();
  app.get("/api/data/archive/offload-status", async (_req, res) => {
    try { res.json(await svc.status()); } catch (e: unknown) { res.status(500).json({ error: errText(e) || "offload status failed" }); }
  });
  return svc;
}
