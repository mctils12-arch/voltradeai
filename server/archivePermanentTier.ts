// archivePermanentTier.ts — PERMANENT R2 tier for write-once archive day files
// (2026-11-06, human "fix all of this": the Railway volume is short of space).
//
// WHY: live 2026-11-06 the volume had 1,006 MiB free — under the global-scope
// aircraft archiving floor (1 GiB), the price panel's floor (1.25 GiB) and the
// sweep-archive gate (2 GiB). The rolling R2 tier (archiveOffload.ts) already
// keeps the raw aircraft/vessel hour files at their local floors (22 d / 8 d),
// so it could free nothing more. The next-largest thing on the volume that no
// live reader needs locally is the per-entity daily ROLLUPS the raw tier
// leaves behind: vessels_tracks 475 MiB, aircraft_tracks 176 MiB,
// trains_tracks 17 MiB (668 MiB, ~23% of the archive), growing ~5.5 MiB/day.
//
// READERS OF THOSE FILES (grep `_tracks` over server/, scripts/, *.py, client
// at 2026-11-06): only readArchiveDay behind the token-gated /api/diag/archive
// probe (research: scripts/portdwell_gate1_rollup.ts reads vessels_tracks
// through it). platformStats and streamsInventory skip/do not list them; port
// dwell, shadow fleet, GNSS integrity and the Time Machine read RAW hour files
// only. readArchiveDay therefore falls back to this tier (coldDaySource) for a
// day whose file is no longer local, so the research path keeps working.
//
// WHAT (ARCHIVE_PERMANENT_TIER, default ON; "0"/"off"/"false"/"no"/"" = off):
//   offload  each closed `<stream>/<YYYY-MM-DD>.jsonl.gz` (day ended >= 2 d
//            ago, untouched >= 1 h, gzip readable end to end) is PUT byte-for-
//            byte to permanent/<stream>/<file> with its SHA-256 as the signed
//            payload hash (R2 rejects a body that does not match it), then
//            HEAD-verified for size, then recorded in its own manifest
//            (archive_permanent_manifest.json: bytes, sha256, line count).
//            An object already at the key is NEVER overwritten: identical
//            bytes are adopted, different bytes are a recorded conflict.
//   evict    a local file more than PERMANENT_LOCAL_DAYS past its date is
//            deleted only after the R2 object is READ BACK and its SHA-256
//            and length equal the local file's at that moment. Any failure
//            (network, size, hash) deletes nothing.
//   reconcile (daily) lists permanent/<stream>/: adopts objects a lost
//            manifest forgot, and reports any evicted file whose object is
//            gone as MISSING (loud: that would be data loss, not a cleanup).
// NOTHING here ever deletes an R2 object: the permanent/ prefix has no
// retention window (the rolling tier only lists and deletes under archive/).
// A local file that differs from its permanent copy is never uploaded over
// it and never evicted: it stays local and is listed under `conflicts`.
//
// When off: no uploads, no evictions, no listing; reads of already-evicted
// days still come from R2 (an off switch must never make data unreadable).

import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import crypto from "crypto";
import { Writable } from "stream";
import { pipeline } from "stream/promises";
import { errText, type R2Client } from "./r2Client";
import { archiveBaseDir, type ColdDayFileSource } from "./datacoreArchive";

// ── constants ───────────────────────────────────────────────────────────────
/** streams whose closed day files move to the permanent tier — only ones
 *  whose sole reader is the research read path (see the header) */
export const PERMANENT_STREAMS = ["vessels_tracks", "aircraft_tracks", "trains_tracks"] as const;
export const PERMANENT_PREFIX = "permanent";
export const PERMANENT_MANIFEST_NAME = "archive_permanent_manifest.json";
/** local copies stay until their date is MORE than this many UTC days old */
export const PERMANENT_LOCAL_DAYS = 7;
/** a day file is offloaded only once its date is at least this many days old... */
export const PERMANENT_MIN_AGE_DAYS = 2;
/** ...and it has not been modified for this long (no writer mid-write) */
export const PERMANENT_MIN_IDLE_MS = 60 * 60_000;
/** per 30-min tick; the 2026-11-06 backlog (~308 files) clears in ~5 ticks */
export const PERMANENT_MAX_UPLOADS_PER_TICK = 64;
export const PERMANENT_MAX_EVICTIONS_PER_TICK = 64;
/** largest file this tier handles in memory (live day files are 0.1-6 MB) */
export const PERMANENT_MAX_OBJECT_BYTES = 64 * 1024 * 1024;
export const PERMANENT_RECONCILE_INTERVAL_MS = 23 * 3600_000;
export const PERMANENT_CACHE_DEFAULT_MAX_BYTES = 256 * 1024 * 1024;
const CONSECUTIVE_FAILURE_ABORT = 3;
const DAY_MS = 86_400_000;
const FILE_RE = /^(\d{4}-\d{2}-\d{2})\.jsonl\.gz$/;

// ── flag ────────────────────────────────────────────────────────────────────
/** Env var PRESENT wins ("0"/"off"/"false"/"no"/"" disable); absent = ON. */
export function permanentTierFlag(env: NodeJS.ProcessEnv): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.ARCHIVE_PERMANENT_TIER;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: !["0", "off", "false", "no", ""].includes(v) };
}

// ── keys + dates (pure) ─────────────────────────────────────────────────────
export function permanentKey(stream: string, file: string): string {
  return `${PERMANENT_PREFIX}/${stream}/${file}`;
}
function utcDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}
/** whole UTC calendar days between `date` and today (today = 0) */
export function permAgeDays(date: string, nowMs: number): number {
  return Math.round((Date.parse(`${utcDate(nowMs)}T00:00:00Z`) - Date.parse(`${date}T00:00:00Z`)) / DAY_MS);
}

// ── manifest ────────────────────────────────────────────────────────────────
export interface PermanentEntry {
  stream: string;
  file: string;
  date: string;
  /** bytes of the R2 object (HEAD-verified; .gz files upload byte-for-byte) */
  bytes: number;
  /** SHA-256 of those bytes; null only for an object adopted from a listing
   *  that has not been read back yet */
  sha256: string | null;
  /** records (non-empty lines) in the decompressed file, counted while the
   *  gzip was validated; null when adopted from a listing */
  lines: number | null;
  src: "uploaded" | "adopted";
  /** local mtime when uploaded/verified (an unchanged file is not re-hashed) */
  srcMtimeMs: number | null;
  at: number;
  /** when the local copy was deleted (after a read-back match); null = local */
  evictedAt: number | null;
}
export interface PermanentConflict {
  stream: string;
  file: string;
  at: number;
  reason: string;
  localBytes: number | null;
  localSha256: string | null;
}
export interface PermanentManifest {
  version: 1;
  entries: Record<string, PermanentEntry>;
  conflicts: Record<string, PermanentConflict>;
  /** keys whose local copy was evicted but whose object the last listing lacked */
  missing: string[];
  lastReconcileAt: number | null;
}

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

export function loadPermanentManifest(base: string): PermanentManifest {
  try {
    const j: unknown = JSON.parse(fs.readFileSync(path.join(base, PERMANENT_MANIFEST_NAME), "utf8"));
    if (!isRecord(j)) throw new Error("not an object");
    return {
      version: 1,
      entries: isRecord(j.entries) ? (j.entries as Record<string, PermanentEntry>) : {},
      conflicts: isRecord(j.conflicts) ? (j.conflicts as Record<string, PermanentConflict>) : {},
      missing: Array.isArray(j.missing) ? j.missing.filter((k): k is string => typeof k === "string") : [],
      lastReconcileAt: typeof j.lastReconcileAt === "number" ? j.lastReconcileAt : null,
    };
  } catch {
    return { version: 1, entries: {}, conflicts: {}, missing: [], lastReconcileAt: null };
  }
}

export function savePermanentManifest(base: string, m: PermanentManifest): void {
  fs.mkdirSync(base, { recursive: true });
  const fp = path.join(base, PERMANENT_MANIFEST_NAME);
  const tmp = `${fp}.tmp-${process.pid}`;
  fs.writeFileSync(tmp, JSON.stringify(m));
  fs.renameSync(tmp, fp); // atomic: a crash never leaves a half manifest
}

// ── local files ─────────────────────────────────────────────────────────────
export interface PermanentLocalFile {
  stream: string;
  file: string;
  date: string;
  key: string;
  fp: string;
  size: number;
  mtimeMs: number;
}

export function listPermanentLocalFiles(base: string, streams: readonly string[] = PERMANENT_STREAMS): PermanentLocalFile[] {
  const out: PermanentLocalFile[] = [];
  for (const stream of streams) {
    let names: string[] = [];
    try { names = fs.readdirSync(path.join(base, stream)); } catch { continue; }
    for (const file of names) {
      const m = FILE_RE.exec(file);
      if (!m) continue;
      const fp = path.join(base, stream, file);
      let st: fs.Stats;
      try { st = fs.statSync(fp); } catch { continue; } // vanished between readdir and stat
      if (!st.isFile()) continue;
      out.push({ stream, file, date: m[1], key: permanentKey(stream, file), fp, size: st.size, mtimeMs: st.mtimeMs });
    }
  }
  return out.sort((a, b) => a.date.localeCompare(b.date) || a.stream.localeCompare(b.stream));
}

export const sha256Of = (b: Buffer): string => crypto.createHash("sha256").update(b).digest("hex");

/** Gunzip `buf` end to end (streamed, never materialized) and count its
 *  non-empty lines; ok=false when the gzip is unreadable or truncated. */
export async function gzipLineCount(buf: Buffer): Promise<{ ok: boolean; lines: number; error: string | null }> {
  let lines = 0;
  let pending = false; // non-newline bytes since the last newline
  const sink = new Writable({
    write(chunk: Buffer, _enc, cb) {
      // native newline search: O(lines) JS work, not O(bytes)
      let start = 0;
      for (;;) {
        const nl = chunk.indexOf(0x0a, start);
        if (nl < 0) { if (start < chunk.length) pending = true; break; }
        if (nl > start || pending) lines++;
        pending = false;
        start = nl + 1;
      }
      cb();
    },
  });
  try {
    const gunzip = zlib.createGunzip();
    gunzip.end(buf);
    await pipeline(gunzip, sink);
    if (pending) lines++; // a last line without a trailing newline
    return { ok: true, lines, error: null };
  } catch (e: unknown) {
    return { ok: false, lines: 0, error: errText(e) };
  }
}

// ── read-through cache (readArchiveDay's cold source) ───────────────────────
export type { ColdDayFileSource };

export interface PermanentCacheStats { files: number; bytes: number; maxBytes: number; fetches: number; hits: number; errors: number; lastError: string | null }

function createPermanentCache(o: { client: R2Client; cacheDir: string; maxBytes: number; getEntry: (key: string) => PermanentEntry | undefined; logError: (m: string) => void }) {
  // the cache dir is ours alone: start clean (unindexed files would eat the cap)
  try { fs.rmSync(o.cacheDir, { recursive: true, force: true }); } catch (e: unknown) { o.logError(`could not clear permanent cache dir ${o.cacheDir}: ${errText(e)}`); }
  const cache = new Map<string, { fp: string; bytes: number; used: number }>();
  const inflight = new Map<string, Promise<string | null>>();
  let clock = 0;
  let total = 0;
  const st = { fetches: 0, hits: 0, errors: 0, lastError: null as string | null };
  const fail = (msg: string): null => { st.errors++; st.lastError = msg; o.logError(msg); return null; };
  const evict = (keep: string) => {
    while (total > o.maxBytes && cache.size > 1) {
      let lru: string | null = null;
      let lruUsed = Infinity;
      cache.forEach((v, k) => { if (k !== keep && v.used < lruUsed) { lruUsed = v.used; lru = k; } });
      if (lru == null) break;
      const v = cache.get(lru)!;
      cache.delete(lru);
      total -= v.bytes;
      fs.promises.unlink(v.fp).catch((e: unknown) => { o.logError(`permanent cache unlink ${v.fp}: ${errText(e)}`); });
    }
  };
  async function fetchKey(key: string, signal?: AbortSignal): Promise<string | null> {
    const hit = cache.get(key);
    if (hit && fs.existsSync(hit.fp)) { hit.used = ++clock; st.hits++; return hit.fp; }
    if (hit) { cache.delete(key); total -= hit.bytes; }
    const running = inflight.get(key);
    if (running) return running;
    const p = (async () => {
      st.fetches++;
      const e = o.getEntry(key);
      // cache under the original basename so callers report the real file name
      const fp = path.join(o.cacheDir, key.slice(PERMANENT_PREFIX.length + 1));
      const r = await o.client.getObjectToFile(key, fp, { maxBytes: PERMANENT_MAX_OBJECT_BYTES, signal });
      if (!r.ok) return fail(`permanent read ${key}: ${r.error || "fetch failed"}`);
      if (e?.sha256) {
        let got: string;
        try { got = sha256Of(await fs.promises.readFile(fp)); } catch (err: unknown) { return fail(`permanent read ${key}: ${errText(err)}`); }
        if (got !== e.sha256) {
          // never serve bytes that fail their recorded hash
          await fs.promises.unlink(fp).catch((err: unknown) => { o.logError(`permanent cache unlink ${fp}: ${errText(err)}`); });
          return fail(`permanent read ${key}: sha256 ${got} != recorded ${e.sha256} — not served`);
        }
      }
      const bytes = r.bytes ?? 0;
      cache.set(key, { fp, bytes, used: ++clock });
      total += bytes;
      evict(key);
      return fp;
    })().finally(() => inflight.delete(key));
    inflight.set(key, p);
    return p;
  }
  return {
    fetchKey,
    stats: (): PermanentCacheStats => ({ files: cache.size, bytes: total, maxBytes: o.maxBytes, ...st }),
  };
}

// ── the tier ────────────────────────────────────────────────────────────────
export interface PermanentTickSummary {
  enabled: boolean;
  uploaded: number;
  adopted: number;
  bytesUploaded: number;
  uploadFailed: number;
  pendingUploads: number;
  evicted: number;
  bytesEvicted: number;
  evictFailed: number;
  newConflicts: number;
  reconciled: { adopted: number; missing: number; dropped: number } | null;
  errors: string[];
}

export interface PermanentTierDeps {
  /** the archive base (the manifest lives here, on the volume); default archiveBaseDir() */
  base?: string;
  client: R2Client;
  env?: NodeJS.ProcessEnv;
  nowMs?: () => number;
  streams?: readonly string[];
  maxUploadsPerTick?: number;
  maxEvictionsPerTick?: number;
  cacheDir?: string;
  cacheMaxBytes?: number;
  log?: (msg: string) => void;
  logError?: (msg: string) => void;
}

export function createPermanentTier(deps: PermanentTierDeps) {
  const base = deps.base ?? archiveBaseDir();
  const client = deps.client;
  const env = deps.env ?? process.env;
  const now = deps.nowMs ?? (() => Date.now());
  const streams = deps.streams ?? PERMANENT_STREAMS;
  const maxUploads = deps.maxUploadsPerTick ?? PERMANENT_MAX_UPLOADS_PER_TICK;
  const maxEvictions = deps.maxEvictionsPerTick ?? PERMANENT_MAX_EVICTIONS_PER_TICK;
  const log = deps.log ?? ((m: string) => console.log(`[archive-permanent] ${m}`));
  const logError = deps.logError ?? ((m: string) => console.error(`[archive-permanent] ${m}`));
  const enabled = () => permanentTierFlag(env).on;
  let manifest = loadPermanentManifest(base);
  let lastSummary: PermanentTickSummary | null = null;
  let lastRunAt: number | null = null;

  const cache = createPermanentCache({
    client,
    cacheDir: deps.cacheDir ?? path.join(os.tmpdir(), "voltrade_r2_permanent_cache"),
    maxBytes: deps.cacheMaxBytes ?? PERMANENT_CACHE_DEFAULT_MAX_BYTES,
    getEntry: (k) => manifest.entries[k],
    logError,
  });

  const noteError = (s: PermanentTickSummary, msg: string) => { s.errors.push(msg); logError(msg); };
  const conflict = (s: PermanentTickSummary, f: PermanentLocalFile, reason: string, localSha: string | null) => {
    manifest.conflicts[f.key] = { stream: f.stream, file: f.file, at: now(), reason, localBytes: f.size, localSha256: localSha };
    s.newConflicts++;
    noteError(s, `CONFLICT ${f.key}: ${reason} — local file kept, R2 object untouched`);
  };

  /** read a local file whole (bounded) and re-check it did not change meanwhile */
  async function readStable(f: PermanentLocalFile): Promise<Buffer | null> {
    if (f.size > PERMANENT_MAX_OBJECT_BYTES) return null;
    const buf = await fs.promises.readFile(f.fp);
    const st = await fs.promises.stat(f.fp);
    if (st.size !== buf.length || st.mtimeMs !== f.mtimeMs || buf.length !== f.size) return null;
    return buf;
  }

  /** GET the object and compare it to `localSha` + length (the eviction proof) */
  async function readBackMatches(key: string, localSha: string, len: number): Promise<{ ok: boolean; error: string | null; mismatch: boolean }> {
    const g = await client.getObject(key, { maxBytes: PERMANENT_MAX_OBJECT_BYTES });
    if (!g.ok || !g.body) return { ok: false, error: `read-back ${key}: ${g.error || "no body"}`, mismatch: false };
    if (g.body.length !== len) return { ok: false, error: `read-back ${key}: R2 ${g.body.length} bytes != local ${len}`, mismatch: true };
    const got = sha256Of(g.body);
    if (got !== localSha) return { ok: false, error: `read-back ${key}: R2 sha256 ${got} != local ${localSha}`, mismatch: true };
    return { ok: true, error: null, mismatch: false };
  }

  async function offloadPhase(s: PermanentTickSummary, files: PermanentLocalFile[]): Promise<void> {
    const t = now();
    const todo = files.filter((f) =>
      permAgeDays(f.date, t) >= PERMANENT_MIN_AGE_DAYS && t - f.mtimeMs >= PERMANENT_MIN_IDLE_MS && !manifest.conflicts[f.key]);
    let consecutive = 0;
    let attempted = 0;
    for (const f of todo) {
      const e = manifest.entries[f.key];
      if (e && e.evictedAt == null && e.sha256 && e.bytes === f.size && e.srcMtimeMs === f.mtimeMs) continue; // unchanged since verified
      if (attempted >= maxUploads) { s.pendingUploads++; continue; }
      attempted++;
      try {
        const buf = await readStable(f);
        if (!buf) {
          if (f.size > PERMANENT_MAX_OBJECT_BYTES) noteError(s, `${f.key}: ${f.size} bytes is over the ${PERMANENT_MAX_OBJECT_BYTES}-byte cap — kept local, not offloaded`);
          continue; // changed while being read: next tick
        }
        const sha = sha256Of(buf);
        if (e) {
          // known key, local differs from what was recorded (or came back after eviction)
          if (e.sha256 === sha) {
            e.srcMtimeMs = f.mtimeMs;
            if (e.evictedAt != null) { e.evictedAt = null; log(`${f.key} re-appeared locally with identical bytes — eligible for eviction again`); }
            continue;
          }
          if (e.sha256 == null) {
            // adopted from a listing, never read back: verify before trusting it
            const rb = await readBackMatches(f.key, sha, buf.length);
            if (rb.ok) { e.sha256 = sha; e.srcMtimeMs = f.mtimeMs; e.bytes = buf.length; if (e.evictedAt != null) e.evictedAt = null; consecutive = 0; continue; }
            if (rb.mismatch) { conflict(s, f, "an adopted R2 object differs from the local file", sha); continue; }
            throw new Error(rb.error || "read-back failed");
          }
          conflict(s, f, e.evictedAt != null ? "a different local file re-appeared after eviction" : "the local file changed after it was offloaded", sha);
          continue;
        }
        const gz = await gzipLineCount(buf);
        if (!gz.ok) { conflict(s, f, `local gzip is unreadable (${gz.error}) — not offloaded`, sha); continue; }
        const head = await client.headObject(f.key);
        if (!head.ok) throw new Error(`head ${f.key}: ${head.error}`);
        if (head.exists) {
          // never overwrite a permanent object: adopt only an identical one
          const rb = await readBackMatches(f.key, sha, buf.length);
          if (rb.ok) {
            manifest.entries[f.key] = { stream: f.stream, file: f.file, date: f.date, bytes: buf.length, sha256: sha, lines: gz.lines, src: "adopted", srcMtimeMs: f.mtimeMs, at: t, evictedAt: null };
            s.adopted++;
            consecutive = 0;
            continue;
          }
          if (rb.mismatch) { conflict(s, f, "an object with different bytes already exists at the key", sha); continue; }
          throw new Error(rb.error || "read-back failed");
        }
        const put = await client.putObject(f.key, buf, "application/gzip", undefined, { payloadSha256: sha });
        if (!put.ok) throw new Error(`put ${f.key}: ${put.error}`);
        const verify = await client.headObject(f.key);
        if (!verify.ok || !verify.exists || verify.size !== buf.length) {
          throw new Error(`verify ${f.key}: ${verify.error || `R2 size ${verify.size ?? "none"} != uploaded ${buf.length}`}`);
        }
        manifest.entries[f.key] = { stream: f.stream, file: f.file, date: f.date, bytes: buf.length, sha256: sha, lines: gz.lines, src: "uploaded", srcMtimeMs: f.mtimeMs, at: t, evictedAt: null };
        s.uploaded++;
        s.bytesUploaded += buf.length;
        consecutive = 0;
      } catch (err: unknown) {
        s.uploadFailed++;
        noteError(s, errText(err));
        if (++consecutive >= CONSECUTIVE_FAILURE_ABORT) { noteError(s, `offload paused this tick after ${consecutive} consecutive failures`); break; }
      }
    }
  }

  async function evictPhase(s: PermanentTickSummary, files: PermanentLocalFile[]): Promise<void> {
    const t = now();
    let consecutive = 0;
    for (const f of files) {
      if (s.evicted >= maxEvictions) break;
      if (permAgeDays(f.date, t) <= PERMANENT_LOCAL_DAYS) continue;
      const e = manifest.entries[f.key];
      if (!e || e.evictedAt != null || !e.sha256 || manifest.conflicts[f.key]) continue;
      if (e.bytes !== f.size || e.srcMtimeMs !== f.mtimeMs) continue; // changed since verified: the offload phase judges it first
      try {
        const buf = await readStable(f);
        if (!buf) continue;
        const sha = sha256Of(buf);
        if (sha !== e.sha256) { conflict(s, f, "the local file no longer matches its recorded sha256", sha); continue; }
        const rb = await readBackMatches(f.key, sha, buf.length);
        if (!rb.ok) {
          if (rb.mismatch) { conflict(s, f, `R2 read-back does not match the local file (${rb.error})`, sha); continue; }
          throw new Error(rb.error || "read-back failed");
        }
        await fs.promises.unlink(f.fp);
        e.evictedAt = now();
        s.evicted++;
        s.bytesEvicted += f.size;
        consecutive = 0;
        log(`evicted local ${f.stream}/${f.file} (${f.size} bytes; R2 copy read back, sha256 match)`);
      } catch (err: unknown) {
        s.evictFailed++;
        noteError(s, `evict ${f.key}: ${errText(err)} — local file kept`);
        if (++consecutive >= CONSECUTIVE_FAILURE_ABORT) { noteError(s, `eviction paused this tick after ${consecutive} consecutive failures`); break; }
      }
    }
  }

  async function reconcilePhase(s: PermanentTickSummary, files: PermanentLocalFile[]): Promise<void> {
    const t = now();
    const local = new Set(files.map((f) => f.key));
    const r = { adopted: 0, missing: 0, dropped: 0 };
    const missing: string[] = [];
    let allOk = true;
    // a stream that cannot be judged this time keeps its previous MISSING keys
    const keepPrevious = (stream: string) => {
      for (const k of manifest.missing) if (manifest.entries[k]?.stream === stream) missing.push(k);
    };
    for (const stream of streams) {
      const list = await client.listObjects(`${PERMANENT_PREFIX}/${stream}/`, { maxPages: 50 });
      if (!list.ok) { allOk = false; noteError(s, `reconcile list ${stream}: ${list.error}`); keepPrevious(stream); continue; }
      const seen = new Map(list.objects.map((o) => [o.key, o.size] as [string, number]));
      seen.forEach((size, key) => {
        const file = key.slice(`${PERMANENT_PREFIX}/${stream}/`.length);
        const m = FILE_RE.exec(file);
        if (!m || manifest.entries[key]) return;
        manifest.entries[key] = { stream, file, date: m[1], bytes: size, sha256: null, lines: null, src: "adopted", srcMtimeMs: null, at: t, evictedAt: local.has(key) ? null : t };
        r.adopted++;
      });
      if (list.truncated) { keepPrevious(stream); continue; } // a partial listing proves nothing is absent
      for (const [key, e] of Object.entries(manifest.entries)) {
        if (e.stream !== stream) continue;
        const size = seen.get(key);
        if (size === e.bytes) continue;
        if (e.evictedAt != null) {
          missing.push(key); // the ONLY copy is gone or changed: never silently dropped
          noteError(s, `MISSING ${key}: evicted locally ${new Date(e.evictedAt).toISOString()} but R2 ${size == null ? "has no object" : `holds ${size} bytes, not ${e.bytes}`}`);
        } else {
          delete manifest.entries[key]; // local copy still here: re-offload it
          r.dropped++;
        }
      }
    }
    r.missing = missing.length;
    manifest.missing = missing;
    if (allOk) manifest.lastReconcileAt = t;
    s.reconciled = r;
  }

  async function tick(): Promise<PermanentTickSummary> {
    const s: PermanentTickSummary = {
      enabled: enabled(), uploaded: 0, adopted: 0, bytesUploaded: 0, uploadFailed: 0, pendingUploads: 0,
      evicted: 0, bytesEvicted: 0, evictFailed: 0, newConflicts: 0, reconciled: null, errors: [],
    };
    if (!client.configured || !s.enabled) { lastSummary = s; return s; }
    try {
      let files = listPermanentLocalFiles(base, streams);
      if (manifest.lastReconcileAt == null || now() - manifest.lastReconcileAt >= PERMANENT_RECONCILE_INTERVAL_MS) {
        await reconcilePhase(s, files);
      }
      await offloadPhase(s, files);
      try { savePermanentManifest(base, manifest); } catch (e: unknown) { noteError(s, `manifest save: ${errText(e)}`); }
      files = listPermanentLocalFiles(base, streams);
      await evictPhase(s, files);
    } catch (e: unknown) {
      noteError(s, `tick: ${errText(e)}`);
    } finally {
      try { savePermanentManifest(base, manifest); } catch (e: unknown) { noteError(s, `manifest save (final): ${errText(e)}`); }
      lastSummary = s;
      lastRunAt = now();
    }
    if (s.uploaded || s.evicted || s.adopted) log(`tick: ${s.uploaded} uploaded (${s.bytesUploaded} B), ${s.adopted} adopted, ${s.evicted} local copies evicted (${s.bytesEvicted} B), ${s.pendingUploads} pending`);
    return s;
  }

  const coldSource: ColdDayFileSource = {
    // every recorded entry stays readable, whatever the stream list says now:
    // dropping a stream from PERMANENT_STREAMS stops new offloads, never reads
    async resolve(stream, day, signal) {
      if (!client.configured) return [];
      const key = permanentKey(stream, `${day}.jsonl.gz`);
      const e = manifest.entries[key];
      if (!e || manifest.missing.includes(key)) return [];
      const fp = await cache.fetchKey(key, signal);
      return fp ? [fp] : [];
    },
  };

  function status() {
    const flag = permanentTierFlag(env);
    const byStream: Record<string, { r2Objects: number; r2Bytes: number; evictedFiles: number; evictedBytes: number; localFiles: number; localBytes: number; oldestDate: string | null; newestDate: string | null }> = {};
    for (const st of streams) byStream[st] = { r2Objects: 0, r2Bytes: 0, evictedFiles: 0, evictedBytes: 0, localFiles: 0, localBytes: 0, oldestDate: null, newestDate: null };
    for (const e of Object.values(manifest.entries)) {
      const b = byStream[e.stream];
      if (!b) continue;
      b.r2Objects++; b.r2Bytes += e.bytes;
      if (e.evictedAt != null) { b.evictedFiles++; b.evictedBytes += e.bytes; }
      if (!b.oldestDate || e.date < b.oldestDate) b.oldestDate = e.date;
      if (!b.newestDate || e.date > b.newestDate) b.newestDate = e.date;
    }
    for (const f of listPermanentLocalFiles(base, streams)) { byStream[f.stream].localFiles++; byStream[f.stream].localBytes += f.size; }
    const sum = (k: "r2Bytes" | "evictedBytes" | "localBytes" | "r2Objects" | "evictedFiles") => Object.values(byStream).reduce((a, b) => a + b[k], 0);
    return {
      enabled: flag.on,
      flag,
      configured: client.configured,
      streams: [...streams],
      prefix: `${PERMANENT_PREFIX}/<stream>/<YYYY-MM-DD>.jsonl.gz`,
      localDays: PERMANENT_LOCAL_DAYS,
      r2Objects: sum("r2Objects"),
      r2Bytes: sum("r2Bytes"),
      evictedFiles: sum("evictedFiles"),
      /** local bytes this tier has freed (objects whose only copy is now R2) */
      evictedBytes: sum("evictedBytes"),
      localBytes: sum("localBytes"),
      byStream,
      conflicts: Object.entries(manifest.conflicts).map(([key, c]) => ({ key, reason: c.reason, at: new Date(c.at).toISOString(), localBytes: c.localBytes })),
      missing: manifest.missing,
      lastRun: lastRunAt ? new Date(lastRunAt).toISOString() : null,
      lastReconcile: manifest.lastReconcileAt ? new Date(manifest.lastReconcileAt).toISOString() : null,
      lastRunSummary: lastSummary,
      readThroughCache: cache.stats(),
      rule: `A day file is copied to R2 once its date is ${PERMANENT_MIN_AGE_DAYS}+ days old and it has been idle 1 h (SHA-256-signed PUT, HEAD size check); its local copy is deleted only when its date is more than ${PERMANENT_LOCAL_DAYS} days old AND the R2 object read back byte-for-byte matches it. R2 objects under ${PERMANENT_PREFIX}/ are never deleted. /api/diag/archive reads evicted days from R2.`,
    };
  }

  return {
    tick,
    status,
    coldSource,
    getManifest: () => manifest,
    /** test seam: reload the manifest from disk */
    _reload: () => { manifest = loadPermanentManifest(base); },
  };
}

export type PermanentTier = ReturnType<typeof createPermanentTier>;
