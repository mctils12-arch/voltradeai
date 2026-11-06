/**
 * dtccIdsStore.ts — the DTCC swaps dedup ID set, on disk (memory fix A1,
 * 2026-11-03, T-DATACORE).
 *
 * WHY THIS EXISTS. dtccSwaps.ts dedups the vendor's cumulative-from-
 * inception file by Dissemination Identifier across every day ever
 * archived, which until this module meant one resident `Set<string>` of
 * every ID. Live 2026-11-03: 6,555,571 IDs (`total_archived` on
 * /api/data/dtcc-swaps), ~118k new US-underlier rows/day, an estimated
 * 0.4-0.5 GB of Node heap growing ~5-8 MB/day. The compact
 * `_seen_ids_index.jsonl.gz` (2026-09-19) already made rebuilding that Set
 * at boot cheap; the RETAINED Set was the remaining cost. This module keeps
 * the same set in a SQLite file on the Railway volume next to the archive
 * (`<archive>/dtccswaps/_seen_ids.db`, `seen_ids(id TEXT PRIMARY KEY)
 * WITHOUT ROWID`, WAL, synchronous=NORMAL) so a dedup decision is an indexed
 * lookup instead of a heap-resident hash, at ~118k lookups per day.
 *
 * CONTRACT WITH dtccSwaps.ts (the only consumer):
 *  - The compact index stays the ground truth and is never modified here —
 *    READ ONLY. dtccSwaps.ts keeps appending new IDs to it exactly as before
 *    in every mode, so `VOLTRADE_DTCC_IDS_STORE=memory` is always a clean
 *    rollback: the Set rebuilds from the index as it always has.
 *  - Backfill (`ensureReadyAsync` at boot, `ensureReadySync` if a dedup
 *    happens before boot finished): if the DB holds fewer IDs than the
 *    index, the missing IDs are imported FROM the index (streamed, never
 *    the whole index in memory on the async path), the counts are then
 *    re-checked, and a DTCC-IDS-BACKFILL line carries both counts. A
 *    mismatch after import is an ERROR line, never silent.
 *  - An unreadable or absent index falls back to the caller's `rebuildIds`
 *    (dtccSwaps.loadSeenIds: scans the day files and rewrites the index —
 *    the SAME self-heal memory mode has always used), then imports from
 *    that Set.
 *  - No trading imports, no knowledge of rows — bare ID strings only
 *    (SPINOUT-READY DATA LAYER).
 *
 * better-sqlite3 is a fully synchronous binding (server/dbWriteTiming.ts has
 * the evidence trail on the volume's periodic I/O stalls), so every call here
 * is bounded: lookups are chunked IN-lists, inserts are 5,000-row
 * transactions, and the async backfill yields to the event loop between
 * batches. Measured numbers live in dtccIdsStore.test.ts (1M synthetic IDs).
 */
import Database from "better-sqlite3";
import type { Database as DatabaseType, Statement, Transaction } from "better-sqlite3";
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { StringDecoder } from "string_decoder";
import { archiveBaseDir } from "./datacoreArchive";

export type DtccIdsMode = "memory" | "dual" | "sqlite";

/** Env flag. `memory` = the pre-A1 behavior exactly (Set only, no DB ever
 *  opened); `dual` (DEFAULT) = Set AND DB, every dedup decision computed by
 *  both, disagreements logged as DTCC-IDS-MISMATCH, the MEMORY answer served;
 *  `sqlite` = DB only, the Set is never built (the heap win). */
export const DTCC_IDS_STORE_ENV = "VOLTRADE_DTCC_IDS_STORE";
export const DTCC_IDS_DEFAULT_MODE: DtccIdsMode = "dual";
export const DTCC_IDS_DB_FILE = "_seen_ids.db";
/** The archive subdirectory the DTCC stream lives in (dtccSwaps.dtccDir). */
export const DTCC_IDS_DIRNAME = "dtccswaps";

/** Where the store lives when no `dir` is given: the DTCC archive directory
 *  under archiveBaseDir() — the Railway volume (/data/voltrade/...), so the
 *  file survives redeploys (durability ratchet, server/durability.test.ts). */
export function defaultDtccIdsDir(): string {
  return path.join(archiveBaseDir(), DTCC_IDS_DIRNAME);
}
/** Rows per INSERT transaction during backfill and routine archiving. */
export const DTCC_IDS_INSERT_BATCH = 5_000;
/** IDs per `SELECT ... WHERE id IN (...)` lookup statement. */
export const DTCC_IDS_LOOKUP_CHUNK = 500;

const KNOWN_MODES: ReadonlySet<string> = new Set<string>(["memory", "dual", "sqlite"]);
let warnedUnknownMode: string | null = null;

/** Reads the mode from the environment on every call (cheap), so a test can
 *  flip it between calls and a misspelt value is reported once, not never:
 *  an unknown value falls back to the DEFAULT and logs a DTCC-IDS-MODE line. */
export function dtccIdsMode(env: NodeJS.ProcessEnv = process.env): DtccIdsMode {
  const raw = env[DTCC_IDS_STORE_ENV];
  if (raw == null || raw.trim() === "") return DTCC_IDS_DEFAULT_MODE;
  const v = raw.trim().toLowerCase();
  if (KNOWN_MODES.has(v)) return v as DtccIdsMode;
  if (warnedUnknownMode !== v) {
    warnedUnknownMode = v;
    console.error(`[datacore] dtccswaps DTCC-IDS-MODE: unknown ${DTCC_IDS_STORE_ENV}="${raw}" — using "${DTCC_IDS_DEFAULT_MODE}"`);
  }
  return DTCC_IDS_DEFAULT_MODE;
}

export function _resetDtccIdsModeWarningForTests(): void { warnedUnknownMode = null; }

export interface DtccIdsBackfill {
  /** pending = not attempted yet; done = db count == index count after the
   *  (possibly empty) import; mismatch = counts still disagree (error line
   *  logged); failed = the attempt threw (error line logged; retried on the
   *  next ensure*). */
  state: "pending" | "done" | "mismatch" | "failed";
  at: number | null;
  ms: number | null;
  dbBefore: number | null;
  dbAfter: number | null;
  indexCount: number | null;
  imported: number | null;
  /** streamed = async gunzip stream over the compact index; sync = whole
   *  index gunzipped in one buffer (only when a dedup ran before boot's
   *  async pass); rebuilt = index absent/unreadable, ids came from
   *  `rebuildIds` (dtccSwaps.loadSeenIds's day-file scan). */
  source: "streamed" | "sync" | "rebuilt" | null;
  error: string | null;
}

export interface DtccIdsStoreOpts {
  /** The archive's dtccswaps directory (holds the day files, the compact
   *  index and now the DB). Defaults to defaultDtccIdsDir() — under
   *  archiveBaseDir(), i.e. the persistent volume. */
  dir?: string;
  /** Full path of the compact index (`_seen_ids_index.jsonl.gz`). */
  indexPath: string;
  /** Self-heal when the index is absent/unreadable — dtccSwaps.loadSeenIds
   *  (a CompactIdSet since 2026-11-06; any sized iterable of IDs works). */
  rebuildIds?: () => Iterable<string> & { readonly size: number };
  log?: (line: string) => void;
  logError?: (line: string) => void;
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Lines of a gzipped text file, split on "\n" exactly like the
 *  `split("\n")`+`if (line)` read dtccSwaps.readSeenIdsIndex has always done
 *  (no trimming, no "\r" handling — the writer only ever emits "\n"), yielded
 *  one decompressed CHUNK's worth at a time (~800 ids per 16KB zlib chunk):
 *  never the whole text, never a whole-file array, and no per-line promise
 *  (a per-line async generator measured ~3.7µs/line of pure scheduling
 *  overhead at 1M ids — at the live 6.5M that is ~25s of boot CPU for
 *  nothing). */
export async function* gzipLineBatches(file: string): AsyncGenerator<string[]> {
  const decoder = new StringDecoder("utf8");
  const src = fs.createReadStream(file).pipe(zlib.createGunzip());
  let leftover = "";
  for await (const chunk of src) {
    const text = leftover + decoder.write(chunk as Buffer);
    const lines: string[] = [];
    let start = 0;
    for (;;) {
      const nl = text.indexOf("\n", start);
      if (nl === -1) break;
      if (nl > start) lines.push(text.slice(start, nl));
      start = nl + 1;
    }
    leftover = text.slice(start);
    if (lines.length) yield lines;
  }
  leftover += decoder.end();
  if (leftover) yield [leftover];
}

/** Counts the non-empty "\n"-separated lines in `chunk` — the same number
 *  `split("\n").filter(Boolean)` gives — scanning bytes (memchr) and never
 *  materializing a line. `state.lineHasBytes` carries a line split across
 *  chunks; the caller adds 1 at the end if it is still set (an unterminated
 *  last line). */
export function countNonEmptyLines(chunk: Buffer, state: { lineHasBytes: boolean }): number {
  let n = 0;
  let pos = 0;
  for (;;) {
    const nl = chunk.indexOf(10, pos);
    if (nl === -1) {
      if (pos < chunk.length) state.lineHasBytes = true;
      return n;
    }
    if (nl > pos || state.lineHasBytes) n++;
    state.lineHasBytes = false;
    pos = nl + 1;
  }
}

/** Non-empty line count of a gzipped file, streamed (the boot-time index
 *  count: one pass, constant memory). */
export async function countGzipLines(file: string): Promise<number> {
  const src = fs.createReadStream(file).pipe(zlib.createGunzip());
  const state = { lineHasBytes: false };
  let n = 0;
  for await (const chunk of src) n += countNonEmptyLines(chunk as Buffer, state);
  return n + (state.lineHasBytes ? 1 : 0);
}

/** Synchronous twin over an already-gunzipped buffer. */
export function countBufferLines(buf: Buffer): number {
  const state = { lineHasBytes: false };
  const n = countNonEmptyLines(buf, state);
  return n + (state.lineHasBytes ? 1 : 0);
}

/** Synchronous twin of gzipLineBatches over an already-gunzipped buffer:
 *  walks "\n" boundaries in the Buffer and materializes one line at a time,
 *  never a whole-text string nor a split() array. */
export function* bufferLines(buf: Buffer): Generator<string> {
  let start = 0;
  for (;;) {
    const nl = buf.indexOf(10, start);
    if (nl === -1) break;
    if (nl > start) yield buf.toString("utf8", start, nl);
    start = nl + 1;
  }
  if (start < buf.length) yield buf.toString("utf8", start);
}

export class DtccIdsStore {
  readonly path: string;
  readonly journalMode: string;
  /** The connection's `PRAGMA synchronous` as SQLite reports it ("1" =
   *  NORMAL). Per-connection, not persisted — exposed so a test can pin it. */
  readonly synchronous: string;
  backfill: DtccIdsBackfill = {
    state: "pending", at: null, ms: null, dbBefore: null, dbAfter: null,
    indexCount: null, imported: null, source: null, error: null,
  };
  private readonly db: DatabaseType;
  private readonly hasStmt: Statement<[string], number>;
  private readonly countStmt: Statement<[], number>;
  private readonly insertStmt: Statement<[string]>;
  private readonly insertBatchTx: Transaction<(batch: readonly string[]) => number>;
  private readonly inStmts = new Map<number, Statement<string[], string>>();
  private cachedCount = 0;
  private ready = false;
  private closed = false;
  private readonly opts: DtccIdsStoreOpts;
  private readonly log: (line: string) => void;
  private readonly logError: (line: string) => void;

  constructor(opts: DtccIdsStoreOpts) {
    this.opts = opts;
    this.log = opts.log ?? ((line) => console.log(line));
    this.logError = opts.logError ?? ((line) => console.error(line));
    const dir = opts.dir ?? defaultDtccIdsDir();
    fs.mkdirSync(dir, { recursive: true });
    this.path = path.join(dir, DTCC_IDS_DB_FILE);
    // timeout: wait on a lock (a concurrent reader of the volume, e.g. a
    // one-off script) instead of throwing SQLITE_BUSY at the poll.
    this.db = new Database(this.path, { timeout: 5_000 });
    try {
      this.journalMode = String(this.db.pragma("journal_mode = WAL", { simple: true }));
      this.db.pragma("synchronous = NORMAL");
      this.synchronous = String(this.db.pragma("synchronous", { simple: true }));
      this.db.exec("CREATE TABLE IF NOT EXISTS seen_ids (id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID");
      if (Number(this.db.pragma("user_version", { simple: true })) === 0) this.db.pragma("user_version = 1");
      this.hasStmt = this.db.prepare<[string], number>("SELECT 1 FROM seen_ids WHERE id = ?").pluck();
      this.countStmt = this.db.prepare<[], number>("SELECT COUNT(*) FROM seen_ids").pluck();
      this.insertStmt = this.db.prepare<[string]>("INSERT OR IGNORE INTO seen_ids (id) VALUES (?)");
      this.insertBatchTx = this.db.transaction((batch: readonly string[]) => {
        let changes = 0;
        for (const id of batch) changes += this.insertStmt.run(id).changes;
        return changes;
      });
      this.cachedCount = this.recount();
    } catch (e) {
      // Never leak the handle behind a thrown constructor (a read-only
      // volume fails at the first pragma, after the open succeeded).
      this.db.close();
      throw e;
    }
  }

  get isReady(): boolean { return this.ready; }
  get isOpen(): boolean { return !this.closed; }

  /** Number of distinct IDs in the DB. Cached: COUNT(*) walks the whole
   *  b-tree on a WITHOUT ROWID table, so it runs at open/backfill (recount)
   *  and is otherwise advanced by the `changes` each insert reports. */
  count(): number { return this.cachedCount; }

  recount(): number {
    const n = this.countStmt.get();
    this.cachedCount = typeof n === "number" ? n : Number(n ?? 0);
    return this.cachedCount;
  }

  /** Single-ID membership — the per-row lookup form (benchmarked against
   *  missingFrom in the test; the archive path uses missingFrom). */
  has(id: string): boolean {
    return this.hasStmt.get(id) !== undefined;
  }

  /** The subset of `ids` NOT in the DB, via chunked `IN (...)` lookups.
   *  Duplicate inputs are looked up once; the caller filters its rows by
   *  membership in the returned Set, so duplicates within one poll get the
   *  same verdict the in-memory Set gives them (both marked fresh). */
  missingFrom(ids: readonly string[]): Set<string> {
    const missing = new Set<string>();
    if (!ids.length) return missing;
    const unique = Array.from(new Set(ids));
    const found = new Set<string>();
    for (let i = 0; i < unique.length; i += DTCC_IDS_LOOKUP_CHUNK) {
      const chunk = unique.slice(i, i + DTCC_IDS_LOOKUP_CHUNK);
      for (const id of this.inStmt(chunk.length).all(...chunk)) found.add(id);
    }
    for (const id of unique) if (!found.has(id)) missing.add(id);
    return missing;
  }

  private inStmt(n: number): Statement<string[], string> {
    let stmt = this.inStmts.get(n);
    if (!stmt) {
      const marks = new Array<string>(n).fill("?").join(",");
      stmt = this.db.prepare<string[], string>(`SELECT id FROM seen_ids WHERE id IN (${marks})`).pluck();
      this.inStmts.set(n, stmt);
    }
    return stmt;
  }

  /** INSERT OR IGNORE in DTCC_IDS_INSERT_BATCH-row transactions. Returns how
   *  many were genuinely new. */
  insertMany(ids: readonly string[]): number {
    let changes = 0;
    for (let i = 0; i < ids.length; i += DTCC_IDS_INSERT_BATCH) {
      changes += this.insertBatchTx(ids.slice(i, i + DTCC_IDS_INSERT_BATCH));
    }
    this.cachedCount += changes;
    return changes;
  }

  /** Boot path: streamed. Counts the compact index, imports from it only
   *  when the DB is behind, re-counts, logs. Idempotent once ready. */
  async ensureReadyAsync(): Promise<DtccIdsBackfill> {
    if (this.ready) return this.backfill;
    const t0 = Date.now();
    const dbBefore = this.recount();
    try {
      if (!fs.existsSync(this.opts.indexPath)) return this.backfillFromRebuild(dbBefore, t0);
      let indexCount = 0;
      try {
        indexCount = await countGzipLines(this.opts.indexPath);
      } catch (e) {
        this.logError(`[datacore] dtccswaps DTCC-IDS-BACKFILL: compact index unreadable (${errText(e)}) — rebuilding from the day files`);
        return this.backfillFromRebuild(dbBefore, t0);
      }
      let imported = 0;
      if (dbBefore < indexCount) {
        let batch: string[] = [];
        for await (const lines of gzipLineBatches(this.opts.indexPath)) {
          for (const line of lines) {
            batch.push(line);
            if (batch.length >= DTCC_IDS_INSERT_BATCH) {
              imported += this.insertMany(batch);
              batch = [];
              // Let timers/I/O run between batches — the insert itself is a
              // synchronous native call (see module header).
              await new Promise<void>((resolve) => setImmediate(resolve));
            }
          }
        }
        if (batch.length) imported += this.insertMany(batch);
      }
      return this.finishBackfill(dbBefore, indexCount, imported, "streamed", t0);
    } catch (e) {
      return this.failBackfill(dbBefore, e, t0);
    }
  }

  /** Same contract as ensureReadyAsync, synchronous: used when a dedup is
   *  requested before the boot pass ran (direct archiveNewRows callers).
   *  Gunzips the whole index into ONE buffer (external memory, no per-line
   *  array) — the same order of transient today's readSeenIdsIndex pays. */
  ensureReadySync(): DtccIdsBackfill {
    if (this.ready) return this.backfill;
    const t0 = Date.now();
    const dbBefore = this.recount();
    try {
      if (!fs.existsSync(this.opts.indexPath)) return this.backfillFromRebuild(dbBefore, t0);
      let buf: Buffer;
      try {
        buf = zlib.gunzipSync(fs.readFileSync(this.opts.indexPath));
      } catch (e) {
        this.logError(`[datacore] dtccswaps DTCC-IDS-BACKFILL: compact index unreadable (${errText(e)}) — rebuilding from the day files`);
        return this.backfillFromRebuild(dbBefore, t0);
      }
      const indexCount = countBufferLines(buf);
      let imported = 0;
      if (dbBefore < indexCount) {
        let batch: string[] = [];
        for (const line of bufferLines(buf)) {
          batch.push(line);
          if (batch.length >= DTCC_IDS_INSERT_BATCH) { imported += this.insertMany(batch); batch = []; }
        }
        if (batch.length) imported += this.insertMany(batch);
      }
      return this.finishBackfill(dbBefore, indexCount, imported, "sync", t0);
    } catch (e) {
      return this.failBackfill(dbBefore, e, t0);
    }
  }

  private backfillFromRebuild(dbBefore: number, t0: number): DtccIdsBackfill {
    if (!this.opts.rebuildIds) {
      return this.failBackfill(dbBefore, new Error("compact index absent/unreadable and no rebuildIds fallback configured"), t0);
    }
    const ids = this.opts.rebuildIds();
    let imported = 0;
    if (dbBefore < ids.size) {
      // batched straight off the iterator (2026-11-06): the rebuilt set is a
      // CompactIdSet whose ids are decoded on iteration, so never materialize
      // all of them at once — same INSERT OR IGNORE batches, same count
      let batch: string[] = [];
      for (const id of ids) {
        batch.push(id);
        if (batch.length >= DTCC_IDS_INSERT_BATCH) { imported += this.insertMany(batch); batch = []; }
      }
      if (batch.length) imported += this.insertMany(batch);
    }
    return this.finishBackfill(dbBefore, ids.size, imported, "rebuilt", t0);
  }

  private finishBackfill(dbBefore: number, indexCount: number, imported: number,
                         source: DtccIdsBackfill["source"], t0: number): DtccIdsBackfill {
    const dbAfter = this.recount();
    const ms = Date.now() - t0;
    const ok = dbAfter === indexCount;
    this.backfill = {
      state: ok ? "done" : "mismatch", at: Date.now(), ms, dbBefore, dbAfter, indexCount, imported, source, error: null,
    };
    this.ready = true;
    if (ok) {
      this.log(`[datacore] dtccswaps DTCC-IDS-BACKFILL ok: db ${dbBefore} -> ${dbAfter}, index ${indexCount}, imported ${imported} (${source}, ${ms}ms)`);
    } else {
      const hint = dbAfter > indexCount
        ? "the db holds ids the compact index lacks (an earlier index append failed?) — a rollback to memory mode would re-archive those ids"
        : "the compact index has duplicate lines or the import was partial";
      this.logError(`[datacore] dtccswaps DTCC-IDS-BACKFILL-MISMATCH: db ${dbAfter} != index ${indexCount} (db ${dbBefore} before, imported ${imported}, ${source}, ${ms}ms) — ${hint}`);
    }
    return this.backfill;
  }

  private failBackfill(dbBefore: number, e: unknown, t0: number): DtccIdsBackfill {
    const ms = Date.now() - t0;
    this.backfill = {
      state: "failed", at: Date.now(), ms, dbBefore, dbAfter: null, indexCount: null, imported: null, source: null, error: errText(e),
    };
    // NOT ready: the next ensure* retries (a transient volume error must not
    // park the store for the whole process lifetime).
    this.logError(`[datacore] dtccswaps DTCC-IDS-BACKFILL-FAILED: ${errText(e)} (db ${dbBefore}, ${ms}ms) — dedup falls back to the in-memory Set until the next attempt`);
    return this.backfill;
  }

  /** Bytes on disk for the main db file (the WAL is bounded by SQLite's
   *  autocheckpoint and excluded); null if unreadable. */
  dbBytes(): number | null {
    try { return fs.statSync(this.path).size; } catch (e) {
      this.logError(`[datacore] dtccswaps DTCC-IDS-STAT: ${errText(e)}`);
      return null;
    }
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.inStmts.clear();
    this.db.close();
  }
}
