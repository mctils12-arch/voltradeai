/**
 * dtccSwaps.ts — DTCC SBSDR equity total-return-swap dissemination stream
 * (DATA CENSUS census #11, worked up for build 2026-08-22).
 *
 * Source: kgc0418-tdw-data-0.s3.amazonaws.com/sec/eod/
 * SEC_CUMULATIVE_EQUITIES_YYYY_MM_DD.zip — keyless, daily, ~132MB zip /
 * ~1.15GB CSV (probed live 2026-08-22: HTTP 200, single-member zip, header
 * verified). "Cumulative" is CUMULATIVE FROM PROGRAM INCEPTION (event
 * timestamps observed back to 2019-09-12) re-served in full every day —
 * every row is one lifecycle event (NEWT/MODI/TERM/CORR/EROR/REVI) keyed by
 * a unique Dissemination Identifier; nothing is ever removed from the file.
 *
 * VOLUME-BUDGET DECISION (this session, filed per BUILD-FIRST/data_census.md
 * §2 item 2 "needs its own budget decision before build"): the full file is
 * global (every SEC-regulated equity swap, ~2.0M rows/day, dominated by
 * non-US underliers e.g. China A-share TRS). We only trade the US equity
 * universe, so this pipeline ARCHIVES US-UNDERLIER ROWS ONLY (Underlier ID
 * source = CUSIP, or ISIN with a "US" country prefix) — live-measured at
 * ~5.7% of rows (2026-08-22 sample: 112,612 of 1,981,839). Because the
 * upstream file is cumulative, day-over-day NEW volume for the US subset
 * is just net-new Dissemination Identifiers (a few hundred to low
 * thousands typically) — the one-time backfill capture is the only large
 * write (~112k rows, single low-digit-MB gz file). This is a genuine
 * BUILD-FIRST scoping decision, not a technical limitation: the raw file
 * is still fetched in full each day (bandwidth only, never held in memory
 * — see streaming zip reader below) and filtered in flight; nothing about
 * this choice blocks widening the filter later if a non-US thesis needs it.
 *
 * License: SEC-mandated public dissemination (Reg SBSR real-time reporting)
 * — attribution required, informational use.
 *
 * HYPOTHESIS (gate-locked): fresh large TRS notional on small/mid-cap US
 * underliers is a near-zero-retail-competition positioning signal (the
 * Archegos instrument). Gate 1 (this build) = do we parse genuine,
 * well-formed identifiers out of the right columns? DTCC's own file has no
 * separate "ground truth" summary page to diff against (unlike OCC/FINRA),
 * so gate 1 here is TWO INDEPENDENT external checksum standards applied to
 * the same extracted string: ISO 6166 (ISIN) and the CUSIP Global Services
 * mod-10 algorithm, run against the derived CUSIP embedded inside every US
 * ISIN (chars 3-11). A parser reading the wrong column would pass neither
 * checksum at anywhere near 100% (see scripts/dtcc_swaps_gate1.ts).
 * Gate 2 (small/mid-cap notional clustering vs forward returns) is NOT
 * attempted here — blocked on accumulating archive depth, same precedent
 * as every other freshly-archived root in signal_ladder.json.
 *
 * KNOWN LIMITATION (found empirically during the gate-1 probe, 2026-08-22):
 * basket swaps carry a semicolon-delimited LIST of ISINs in one field
 * (Underlier ID source-Leg 1 = "ISIN;ISIN;...") with no reliable per-
 * component country attribution in this file — 0.003% of the US-flagged
 * sample in the probed day. v1 DROPS basket rows entirely (isUsUnderlier
 * returns false for any semicolon-joined source field) rather than guess
 * at US-ness; isBasketField exists only so scripts/dtcc_swaps_gate1.ts can
 * COUNT them for the record, never to archive them under a guessed label.
 * Decomposing baskets into their component legs is a v2 candidate if a
 * thesis ever needs it.
 */
import { Readable } from "stream";
import zlib from "zlib";
import readline from "readline";
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";
import { guardedRefresh } from "./crashSafeRefresh";
import { DtccIdsStore, dtccIdsMode, DTCC_IDS_DB_FILE, type DtccIdsBackfill, type DtccIdsMode } from "./dtccIdsStore";
import { MIN_FREE_BYTES as ARCHIVE_GUARD_FREE_BYTES, readFreeBytes } from "./globalScopes";

const BASE_URL = "https://kgc0418-tdw-data-0.s3.amazonaws.com/sec/eod";

export function dtccUrl(dateYmd: string): string {
  // dateYmd = YYYY_MM_DD (the vendor's own separator)
  return `${BASE_URL}/SEC_CUMULATIVE_EQUITIES_${dateYmd}.zip`;
}

// ── CSV (quote-aware, RFC4180-style doubled-quote escaping) ────────────────

export function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; } else inQuotes = false;
      } else cur += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { out.push(cur); cur = ""; }
      else cur += c;
    }
  }
  out.push(cur);
  return out;
}

/** Columns this pipeline actually depends on — index-looked-up so column
 *  REORDERING is transparent; a RENAMED/REMOVED column throws a clear error
 *  (schema-drift guard) rather than silently reading garbage. */
export const REQUIRED_COLUMNS = [
  "Dissemination Identifier", "Original Dissemination Identifier", "Action type",
  "Event timestamp", "Effective Date", "Asset Class",
  "Notional amount-Leg 1", "Notional currency-Leg 1",
  "Underlier ID-Leg 1", "Underlier ID source-Leg 1", "UPI Underlier Name",
] as const;

export function parseHeader(headerLine: string): Record<string, number> {
  const cols = parseCsvLine(headerLine);
  const idx: Record<string, number> = {};
  cols.forEach((c, i) => { idx[c] = i; });
  const missing = REQUIRED_COLUMNS.filter((c) => !(c in idx));
  if (missing.length) {
    throw new Error(`dtccSwaps: schema drift — missing columns: ${missing.join(", ")}`);
  }
  return idx;
}

// ── Identifier checksums (ROOT VALIDATION LADDER gate 1) ────────────────────

const letterVal = (ch: string) => ch.charCodeAt(0) - 55; // 'A' -> 10 .. 'Z' -> 35

/** ISO 6166 ISIN check-digit (Luhn over the letter-expanded numeral string). */
export function isinCheckDigitValid(isin: string): boolean {
  if (!/^[A-Z]{2}[A-Z0-9]{9}[0-9]$/.test(isin)) return false;
  let digits = "";
  for (const ch of isin) digits += ch >= "0" && ch <= "9" ? ch : String(letterVal(ch));
  let sum = 0, double = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = digits.charCodeAt(i) - 48;
    if (double) { d *= 2; if (d > 9) d -= 9; }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}

/** CUSIP Global Services mod-10 check digit over an 8-char base + 1 check char. */
export function cusipCheckDigitValid(cusip9: string): boolean {
  if (!/^[A-Z0-9*@#]{8}[0-9]$/.test(cusip9)) return false;
  const base = cusip9.slice(0, 8);
  let sum = 0;
  for (let i = 0; i < 8; i++) {
    const ch = base[i];
    let v: number;
    if (ch >= "0" && ch <= "9") v = ch.charCodeAt(0) - 48;
    else if (ch === "*") v = 36;
    else if (ch === "@") v = 37;
    else if (ch === "#") v = 38;
    else v = letterVal(ch);
    if ((i + 1) % 2 === 0) v *= 2;
    sum += Math.floor(v / 10) + (v % 10);
  }
  const check = (10 - (sum % 10)) % 10;
  return String(check) === cusip9[8];
}

/** The CUSIP embedded inside a US ISIN is chars 3-11 (0-indexed 2..10). */
export function cusipFromUsIsin(isin: string): string | null {
  if (!isin.startsWith("US") || isin.length !== 12) return null;
  return isin.slice(2, 11);
}

/** Basket-swap rows carry a semicolon-delimited LIST with no reliable
 *  per-component country attribution — out of scope for v1. Live-probed
 *  2026-08-22 (scripts/dtcc_swaps_gate1.ts): TWO distinct encodings exist
 *  in the wild — most baskets repeat "ISIN;ISIN;..." in the SOURCE field,
 *  but a handful carry a single "ISIN" source with the semicolon-joined
 *  list in the ID field instead (source stays singular). Both must be
 *  checked, or a handful of basket rows leak through disguised as a
 *  single (bad) identifier — see isBasketField, used only to COUNT them
 *  for the gate-1 record, never to archive them under a guessed label. */
export function isUsUnderlier(id: string, source: string): boolean {
  if (source.includes(";") || id.includes(";")) return false;
  return source === "CUSIP" || (source === "ISIN" && id.startsWith("US"));
}

export function isBasketField(id: string, source: string): boolean {
  return source.includes(";") || id.includes(";");
}

// ── Row shape ────────────────────────────────────────────────────────────

export interface DtccSwapRow {
  disseminationId: string;
  originalDisseminationId: string;
  actionType: string;
  eventTimestamp: string;
  effectiveDate: string;
  notionalAmountLeg1: number | null; // null = blank/masked (Dodd-Frank real-time caps mask large notionals — an honest null, never a fabricated 0)
  notionalCurrencyLeg1: string;
  underlierId: string;
  underlierIdSource: string;
  underlierName: string;
}

function parseNotional(raw: string): number | null {
  if (!raw) return null;
  const n = Number(raw.replace(/,/g, ""));
  return Number.isFinite(n) ? n : null;
}

/** Parses one data line into a row, or null if malformed (column-count
 *  mismatch) or the underlier is not a US equity per isUsUnderlier. */
export function parseDtccLine(line: string, idx: Record<string, number>, expectedCols: number): DtccSwapRow | null {
  const c = parseCsvLine(line);
  if (c.length !== expectedCols) return null;
  const source = c[idx["Underlier ID source-Leg 1"]];
  const id = c[idx["Underlier ID-Leg 1"]];
  if (!isUsUnderlier(id, source)) return null;
  return {
    disseminationId: c[idx["Dissemination Identifier"]],
    originalDisseminationId: c[idx["Original Dissemination Identifier"]],
    actionType: c[idx["Action type"]],
    eventTimestamp: c[idx["Event timestamp"]],
    effectiveDate: c[idx["Effective Date"]],
    notionalAmountLeg1: parseNotional(c[idx["Notional amount-Leg 1"]]),
    notionalCurrencyLeg1: c[idx["Notional currency-Leg 1"]],
    underlierId: id,
    underlierIdSource: source,
    underlierName: c[idx["UPI Underlier Name"]],
  };
}

// ── Dependency-free streaming single-member ZIP reader ──────────────────────
// Buffers ONLY the fixed+variable local file header (<64KB); the file body
// (up to ~150MB compressed / ~1.2GB decompressed) is never held in memory —
// it streams local-header-bytes -> inflateRaw -> readline, so peak RSS stays
// bounded regardless of source size. (secFtd.ts's unzipSingleText buffers the
// whole member — fine at its <1MB scale, unsafe at this stream's scale in a
// process that also runs the trading loop — see KEEP THE SYSTEM ALIVE.)

interface ZipLocalHeader { method: number; compressedSize: number; headerLen: number; flags: number; }

export function tryParseLocalHeader(buf: Buffer): ZipLocalHeader | null {
  if (buf.length < 30) return null;
  if (buf.readUInt32LE(0) !== 0x04034b50) throw new Error("dtccSwaps zip: bad local header signature");
  const flags = buf.readUInt16LE(6);
  const method = buf.readUInt16LE(8);
  const compressedSize = buf.readUInt32LE(18);
  const nameLen = buf.readUInt16LE(26);
  const extraLen = buf.readUInt16LE(28);
  const headerLen = 30 + nameLen + extraLen;
  if (buf.length < headerLen) return null;
  if (compressedSize === 0xffffffff) throw new Error("dtccSwaps zip: ZIP64 member not supported");
  if (flags & 0x0008) throw new Error("dtccSwaps zip: streamed (data-descriptor) member not supported");
  return { method, compressedSize, headerLen, flags };
}

type NodeFetchResponse = { ok: boolean; status: number; body: ReadableStream<Uint8Array> | null };
type FetchFn = (url: string, init?: { headers?: Record<string, string> }) => Promise<NodeFetchResponse>;

/** Streams the single CSV member of a DTCC zip line-by-line through onLine,
 *  yielding to the event loop between chunks the whole way (no synchronous
 *  pass over the full decompressed text). Returns summary counts. */
export async function streamDtccZip(
  url: string,
  onLine: (line: string, isHeader: boolean) => void,
  fetchImpl: FetchFn = fetch as any,
): Promise<{ ok: boolean; status: number; lines: number }> {
  const resp = await fetchImpl(url, {
    headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
  });
  if (!resp.ok || !resp.body) return { ok: false, status: resp.status, lines: 0 };

  const source: AsyncIterator<Uint8Array> = (Readable.fromWeb
    ? (Readable.fromWeb(resp.body as any) as any)[Symbol.asyncIterator]()
    : (resp.body as any)[Symbol.asyncIterator]());

  let leftover = Buffer.alloc(0);
  const pull = async (): Promise<Buffer | null> => {
    const { value, done } = await source.next();
    if (done) return null;
    return Buffer.from(value);
  };
  const readAtLeast = async (n: number): Promise<void> => {
    while (leftover.length < n) {
      const chunk = await pull();
      if (!chunk) break;
      leftover = Buffer.concat([leftover, chunk]);
    }
  };

  await readAtLeast(30);
  const fixed = tryParseLocalHeader(leftover);
  if (!fixed) return { ok: false, status: 0, lines: 0 };
  await readAtLeast(fixed.headerLen);
  leftover = leftover.subarray(fixed.headerLen);
  if (fixed.method !== 8 && fixed.method !== 0) {
    throw new Error(`dtccSwaps zip: unsupported compression method ${fixed.method}`);
  }

  let remaining = fixed.compressedSize;
  const bounded = new Readable({
    async read() {
      try {
        if (leftover.length > 0) {
          const take = Math.min(leftover.length, remaining);
          const chunk = leftover.subarray(0, take);
          leftover = leftover.subarray(take);
          remaining -= take;
          this.push(chunk);
          if (remaining <= 0) this.push(null);
          return;
        }
        if (remaining <= 0) { this.push(null); return; }
        const chunk = await pull();
        if (!chunk) { this.push(null); return; }
        const take = Math.min(chunk.length, remaining);
        remaining -= take;
        this.push(chunk.subarray(0, take));
        if (remaining <= 0) this.push(null);
      } catch (e) { this.destroy(e as Error); }
    },
  });

  const decompressed = fixed.method === 8 ? bounded.pipe(zlib.createInflateRaw()) : bounded;
  const rl = readline.createInterface({ input: decompressed, crlfDelay: Infinity });
  let lines = 0;
  let first = true;
  for await (const line of rl) {
    if (!line) continue;
    onLine(line, first);
    first = false;
    lines++;
  }
  return { ok: true, status: resp.status, lines };
}

// ── Archive (append-only JSONL.gz per calendar day observed; dedup by
//    Dissemination Identifier across ALL previously archived days, since
//    the upstream file is cumulative-from-inception and re-serves every
//    prior event daily) ───────────────────────────────────────────────────

function dtccDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "dtccswaps");
}

let seenIds: Set<string> | null = null;

// ── Compact seenIds index (boot-burst fix, KNOWN BROKEN #41 leak-audit
//    NEXT(1), 2026-09-18) ────────────────────────────────────────────────
//
// The leak audit (research/experiments.md 2026-09-18) measured
// `loadSeenIds`'s ORIGINAL full-archive scan at +400-650MB heapUsed
// transient per boot: every archived day file gunzipped, then EVERY line
// JSON.parse'd into a full 10-field DtccSwapRow object just to read one
// field off it (disseminationId), for a Set that already held 1.95M IDs
// in production. This index makes the boot-time read cheap: one flat file
// of bare ID strings, no per-row JSON reconstruction. It does NOT shrink
// the retained Set itself (that's inherent to needing all IDs resident to
// dedup a cumulative-from-inception upstream file) — only the CPU/memory
// burst of rebuilding it at every boot, which the audit named as "the
// biggest single contributor to the boot burst and the only one that
// grows day over day." (2026-11-03, memory fix A1: the retained Set is
// what server/dtccIdsStore.ts moves to disk — see ON-DISK ID STORE below.
// This index stays the ground truth and is appended in every mode.)
const SEEN_IDS_INDEX_FILE = "_seen_ids_index.jsonl.gz";

function seenIdsIndexPath(dir: string): string {
  return path.join(dir, SEEN_IDS_INDEX_FILE);
}

/** Fast path: one ID per line, gzip-compressed. Returns null (not an empty
 *  Set) when the index doesn't exist yet or fails to read, so the caller
 *  can tell "no index" from "index exists and is empty" apart and fall
 *  back to the full scan instead of silently starting from zero. */
function readSeenIdsIndex(dir: string): Set<string> | null {
  const file = seenIdsIndexPath(dir);
  if (!fs.existsSync(file)) return null;
  try {
    const raw = zlib.gunzipSync(fs.readFileSync(file)).toString("utf8");
    const seen = new Set<string>();
    for (const line of raw.split("\n")) {
      if (line) seen.add(line);
    }
    return seen;
  } catch (e) {
    console.error("[datacore] dtccswaps readSeenIdsIndex:", e);
    return null;
  }
}

/** Full rewrite from a Set — used once, right after the legacy full-archive
 *  scan below (first boot after this fix, or any future from-scratch
 *  rebuild). Never called on the routine append path. */
function writeSeenIdsIndex(dir: string, ids: Set<string>): void {
  try {
    fs.mkdirSync(dir, { recursive: true });
    const payload = Array.from(ids).join("\n") + (ids.size ? "\n" : "");
    fs.writeFileSync(seenIdsIndexPath(dir), zlib.gzipSync(payload));
  } catch (e) {
    console.error("[datacore] dtccswaps writeSeenIdsIndex:", e);
  }
}

/** Routine (non-boot) write path: appends newly-archived IDs to the compact
 *  index, same read-modify-write-gzip shape `archiveNewRows` already uses
 *  for its own per-day files, just for this one small file instead of N
 *  full row-object files. Runs only when a poll actually finds fresh rows
 *  (at most once per ~6h poll), never on the boot read path above. */
function appendSeenIdsIndex(dir: string, ids: string[]): void {
  if (!ids.length) return;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const file = seenIdsIndexPath(dir);
    const existing = fs.existsSync(file) ? zlib.gunzipSync(fs.readFileSync(file)).toString("utf8") : "";
    const payload = existing + ids.join("\n") + "\n";
    fs.writeFileSync(file, zlib.gzipSync(payload));
  } catch (e) {
    console.error("[datacore] dtccswaps appendSeenIdsIndex:", e);
  }
}

/** Reconstructs the full seenIds Set — index fast path first; falls back
 *  to the legacy full-archive scan (gunzip + JSON.parse every day file)
 *  only when no index exists yet (this fix's own first boot against an
 *  existing archive, or a from-scratch archive dir), then persists the
 *  index so every later boot takes the fast path instead of re-paying
 *  this cost. The day-file glob is name-anchored to the actual
 *  `<observedDate>.jsonl(.gz)` convention `archiveNewRows` writes (same
 *  pattern fdicBanks.ts's `seedSeen` already uses) so a corrupt/unreadable
 *  index file sitting in the same directory is never itself misread as a
 *  day file during the fallback scan. */
export function loadSeenIds(dir: string): Set<string> {
  const indexed = readSeenIdsIndex(dir);
  if (indexed) return indexed;

  const seen = new Set<string>();
  let malformedLines = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
      const raw = f.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString("utf8")
        : fs.readFileSync(path.join(dir, f), "utf8");
      for (const line of raw.split("\n")) {
        if (!line) continue;
        try { seen.add(JSON.parse(line).disseminationId); }
        catch { malformedLines++; }
      }
    }
  } catch (e) {
    console.error("[datacore] dtccswaps loadSeenIds:", e);
  }
  if (malformedLines) {
    console.error(`[datacore] dtccswaps loadSeenIds: ${malformedLines} malformed archive line(s) skipped`);
  }
  writeSeenIdsIndex(dir, seen);
  return seen;
}

// ── ON-DISK ID STORE (memory fix A1, 2026-11-03) ─────────────────────────
//
// The compact index above made REBUILDING `seenIds` at boot cheap; the
// retained Set itself (6,555,571 IDs live on 2026-11-03, an estimated
// 0.4-0.5 GB of heap, +5-8 MB/day) was the remaining cost. server/
// dtccIdsStore.ts keeps the same set in `<dir>/_seen_ids.db` (SQLite, WAL).
// VOLTRADE_DTCC_IDS_STORE selects the mode: `memory` = the pre-A1 path
// exactly (no DB ever opened); `dual` (DEFAULT) = Set AND DB, every dedup
// decision computed by both, disagreements logged as DTCC-IDS-MISMATCH, the
// MEMORY answer served; `sqlite` = DB only, the Set is never built (the
// heap win — switched to after a mismatch-free dual period). The compact
// index is appended in EVERY mode and never otherwise touched, so `memory`
// is always a clean rollback. A store that cannot open/backfill/insert
// degrades THAT call to the memory path and says so (DTCC-IDS-DB-ERROR):
// the archive never stops over its dedup store (KEEP THE SYSTEM ALIVE).
let idsStore: DtccIdsStore | null = null;
let idsStoreDir: string | null = null;
let idsStoreError: string | null = null;
let idsMismatches = 0;
let idsLastMismatch: { at: number; kind: "decision" | "count"; detail: string } | null = null;
let idsLastLookupMs: number | null = null;
let idsLastLookupRows: number | null = null;

export type DtccAuditSink = (action: string, detail: string) => void;
let idsAuditSink: DtccAuditSink | null = null;
/** Optional audit-log hook for DTCC-IDS-* lines (e.g. bot.ts's `audit`);
 *  console logging happens regardless. Nothing wires it by default. */
export function setDtccIdsAuditSink(sink: DtccAuditSink | null): void { idsAuditSink = sink; }

function idsEmit(action: string, detail: string): void {
  console.error(`[datacore] dtccswaps ${action}: ${detail}`);
  if (idsAuditSink) {
    try { idsAuditSink(action, detail); } catch (e) { console.error("[datacore] dtccswaps audit sink threw:", e); }
  }
}

function idsMismatch(kind: "decision" | "count", detail: string): void {
  idsMismatches++;
  idsLastMismatch = { at: Date.now(), kind, detail };
  idsEmit("DTCC-IDS-MISMATCH", `${kind}: ${detail}`);
}

function closeIdsStore(): void {
  if (idsStore) {
    try { idsStore.close(); } catch (e) { console.error("[datacore] dtccswaps ids store close:", e); }
  }
  idsStore = null; idsStoreDir = null;
}

// ── VOLUME GUARD (2026-11-04) ─────────────────────────────────────────────
// The dedup DB sits on the same Railway volume as the archives. On its first
// live boot (2026-11-04 03:49Z) the 178 MB DB moved the volume's free space
// from ~1,146 MB to 967.9 MB — under globalScopes' MIN_FREE_BYTES (1 GiB), the
// line below which the global-scope aircraft archiver SKIPS writes. An archive
// gap never refills (GOAL priority 1); the DB is DERIVED data, rebuilt from the
// compact index at any time, so it never gets to spend the archives' headroom:
//   - an existing DB is removed (with its -wal/-shm) while free < the guard;
//   - a new DB is created only if free − its estimated size stays ≥ the guard
//     + DTCC_IDS_DISK_MARGIN_BYTES (hysteresis: no create/remove flapping).
// Refused → the call runs the memory path (exactly pre-A1) and says so once
// per state change (DTCC-IDS-DISK-GUARD / -CLEAR). An unreadable free-space
// reading fails OPEN, the same as volumeAllowsWrite (never silently disable).
export const DTCC_IDS_DISK_MARGIN_BYTES = 256 * 1024 * 1024;
/** measured 24.7 B/id on disk (A1) + WAL/page slack */
export const DTCC_IDS_EST_BYTES_PER_ID = 28;
/** used only when no count is known yet (no Set built, no poll cached) */
const DTCC_IDS_EST_FALLBACK_COUNT = 8_000_000;
let freeBytesReader: (dir: string) => number | null = readFreeBytes;
let idsDiskGuard: { active: boolean; reason: string | null; freeBytes: number | null; at: number | null } = { active: false, reason: null, freeBytes: null, at: null };

export function _setDtccFreeBytesReaderForTests(fn: ((dir: string) => number | null) | null): void {
  freeBytesReader = fn ?? readFreeBytes;
}

function removeIdsDbFiles(dir: string): string[] {
  const removed: string[] = [];
  for (const suffix of ["", "-wal", "-shm"]) {
    const fp = path.join(dir, DTCC_IDS_DB_FILE + suffix);
    try {
      fs.unlinkSync(fp);
      removed.push(path.basename(fp));
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error(`[datacore] dtccswaps could not remove ${fp}:`, e);
    }
  }
  return removed;
}

/** true = the store may exist/open here; false = memory path (and an existing
 *  derived DB has been removed to give the archives their headroom back). */
function idsStoreDiskAllowed(dir: string): boolean {
  const free = freeBytesReader(dir);
  let refusal: string | null = null;
  if (free != null) {
    const dbPresent = fs.existsSync(path.join(dir, DTCC_IDS_DB_FILE));
    if (dbPresent) {
      if (free < ARCHIVE_GUARD_FREE_BYTES) {
        if (idsStore && idsStoreDir === dir) closeIdsStore();
        const removed = removeIdsDbFiles(dir);
        refusal = `volume free ${free} B < archive guard ${ARCHIVE_GUARD_FREE_BYTES} B with the derived dedup DB present; removed ${removed.join(", ") || "nothing"} — memory path`;
      }
    } else {
      const ids = seenIds ? seenIds.size : cache ? cache.totalArchived : DTCC_IDS_EST_FALLBACK_COUNT;
      const est = ids * DTCC_IDS_EST_BYTES_PER_ID;
      if (free - est < ARCHIVE_GUARD_FREE_BYTES + DTCC_IDS_DISK_MARGIN_BYTES) {
        refusal = `creating the dedup DB (~${est} B for ${ids} ids) would leave ${free - est} B free, under the archive guard ${ARCHIVE_GUARD_FREE_BYTES} B + margin ${DTCC_IDS_DISK_MARGIN_BYTES} B — memory path`;
      }
    }
  }
  if (refusal) {
    if (!idsDiskGuard.active) idsEmit("DTCC-IDS-DISK-GUARD", refusal);
    idsDiskGuard = { active: true, reason: refusal, freeBytes: free, at: Date.now() };
    idsStoreError = refusal;
    return false;
  }
  if (idsDiskGuard.active) idsEmit("DTCC-IDS-DISK-GUARD-CLEAR", `volume free ${free} B — the dedup DB may be used again`);
  idsDiskGuard = { active: false, reason: null, freeBytes: free, at: Date.now() };
  return true;
}

/** Opens (or re-targets, when a test switches archive dirs) the store.
 *  Returns null — after logging — when the DB cannot be opened, or when the
 *  VOLUME GUARD above refuses it (memory path). */
function openIdsStore(dir: string): DtccIdsStore | null {
  if (!idsStoreDiskAllowed(dir)) return null;
  if (idsStore && idsStoreDir === dir && idsStore.isOpen) return idsStore;
  closeIdsStore();
  try {
    idsStore = new DtccIdsStore({ dir, indexPath: seenIdsIndexPath(dir), rebuildIds: () => loadSeenIds(dir) });
    idsStoreDir = dir;
    idsStoreError = null;
  } catch (e) {
    idsStoreError = e instanceof Error ? e.message : String(e);
    idsEmit("DTCC-IDS-DB-ERROR", `could not open ${path.join(dir, "_seen_ids.db")}: ${idsStoreError} — this call uses the in-memory Set`);
  }
  return idsStore;
}

/** A store ready for dedup decisions (backfilled), or null to degrade this
 *  call to the memory path. Synchronous: used by archiveNewRows itself. */
function readyIdsStoreSync(dir: string): DtccIdsStore | null {
  const store = openIdsStore(dir);
  if (!store) return null;
  if (!store.isReady) store.ensureReadySync();
  return store.backfill.state === "failed" ? null : store;
}

/** Boot path (refreshDtccSwaps): the streamed backfill, awaited BEFORE the
 *  zip stream so the two never overlap in memory. No-op in memory mode. */
export async function ensureDtccIdsReady(baseDir?: string): Promise<void> {
  if (dtccIdsMode() === "memory") return;
  const store = openIdsStore(dtccDir(baseDir));
  if (store && !store.isReady) await store.ensureReadyAsync();
}

/** dual mode: the two dedup verdicts for one poll must agree. Logged, counted,
 *  never silently reconciled — the memory answer is what gets served. */
function compareIdsDecisions(memFresh: DtccSwapRow[], dbFresh: DtccSwapRow[]): void {
  const memIds = new Set(memFresh.map((r) => r.disseminationId));
  const dbIds = new Set(dbFresh.map((r) => r.disseminationId));
  const onlyMem: string[] = [];
  const onlyDb: string[] = [];
  for (const id of memIds) if (!dbIds.has(id)) onlyMem.push(id);
  for (const id of dbIds) if (!memIds.has(id)) onlyDb.push(id);
  if (!onlyMem.length && !onlyDb.length) return;
  idsMismatch("decision",
    `memory says ${memIds.size} fresh, db says ${dbIds.size}; fresh-only-in-memory ${onlyMem.length}` +
    ` (e.g. ${onlyMem.slice(0, 3).join(",") || "-"}), fresh-only-in-db ${onlyDb.length}` +
    ` (e.g. ${onlyDb.slice(0, 3).join(",") || "-"}); serving the memory answer`);
}

/** Archive-wide distinct-ID count for `totalArchived` when nothing was
 *  archived this cycle: the Set's size (memory/dual — the served answer) or
 *  the DB's (sqlite). */
function seenIdsTotal(dir: string): number {
  if (dtccIdsMode() === "sqlite") {
    const store = readyIdsStoreSync(dir);
    if (store) return store.count();
  }
  if (!seenIds) seenIds = loadSeenIds(dir);
  return seenIds.size;
}

export interface DtccIdsStats {
  mode: DtccIdsMode;
  /** Set size, or null when the Set has not been built (sqlite mode's win). */
  memoryCount: number | null;
  /** DB distinct-id count, or null when no DB has been opened (memory mode). */
  dbCount: number | null;
  /** dual-mode disagreements (decision or count) since boot. */
  mismatches: number;
  lastMismatch: { at: number; kind: "decision" | "count"; detail: string } | null;
  /** Wall-clock ms of the DB lookup pass in the last archiveNewRows call. */
  lastLookupMs: number | null;
  lastLookupRows: number | null;
  backfill: DtccIdsBackfill | null;
  dbPath: string | null;
  dbBytes: number | null;
  journalMode: string | null;
  /** Last store open failure (the call degraded to the memory path), if any. */
  storeError: string | null;
  /** VOLUME GUARD: active = the DB is refused/removed for archive headroom. */
  diskGuard: { active: boolean; reason: string | null; freeBytes: number | null; at: number | null };
}

/** For the health diagnostic: how the dedup store is doing right now. */
export function dtccIdsStats(): DtccIdsStats {
  const open = idsStore && idsStore.isOpen ? idsStore : null;
  return {
    mode: dtccIdsMode(),
    memoryCount: seenIds ? seenIds.size : null,
    dbCount: open ? open.count() : null,
    mismatches: idsMismatches,
    lastMismatch: idsLastMismatch,
    lastLookupMs: idsLastLookupMs,
    lastLookupRows: idsLastLookupRows,
    backfill: idsStore ? idsStore.backfill : null,
    dbPath: idsStore ? idsStore.path : null,
    dbBytes: open ? open.dbBytes() : null,
    journalMode: idsStore ? idsStore.journalMode : null,
    storeError: idsStoreError,
    diskGuard: { ...idsDiskGuard },
  };
}

export function _resetDtccForTests(): void {
  seenIds = null; cache = null; refreshing = false;
  closeIdsStore();
  idsStoreError = null; idsMismatches = 0; idsLastMismatch = null; idsLastLookupMs = null; idsLastLookupRows = null;
  idsDiskGuard = { active: false, reason: null, freeBytes: null, at: null };
}

/** Appends only rows whose Dissemination Identifier has never been archived
 *  before, to <archive>/dtccswaps/<observedDate>.jsonl.gz (one file per
 *  calendar day this pipeline RAN, not per event date — events span years).
 *  Returns { newRows, seenTotal }. */
export function archiveNewRows(rows: DtccSwapRow[], observedDate: string, baseDir?: string): { newRows: number; seenTotal: number } {
  const dir = dtccDir(baseDir);
  fs.mkdirSync(dir, { recursive: true });
  const mode = dtccIdsMode();
  // dual/sqlite: the store, or null when it could not open/backfill — then
  // this call runs the memory path (already logged by the store helpers).
  const store = mode === "memory" ? null : readyIdsStoreSync(dir);
  const useMemory = mode !== "sqlite" || !store;
  let memFresh: DtccSwapRow[] | null = null;
  let dbFresh: DtccSwapRow[] | null = null;
  if (useMemory) {
    if (!seenIds) seenIds = loadSeenIds(dir);
    memFresh = rows.filter((r) => !seenIds!.has(r.disseminationId));
  }
  if (store) {
    const t0 = performance.now();
    const missing = store.missingFrom(rows.map((r) => r.disseminationId));
    idsLastLookupMs = performance.now() - t0;
    idsLastLookupRows = rows.length;
    dbFresh = rows.filter((r) => missing.has(r.disseminationId));
  }
  if (memFresh && dbFresh) compareIdsDecisions(memFresh, dbFresh);
  // memory/dual serve the Set's verdict; sqlite serves the DB's.
  const fresh = memFresh ?? dbFresh!;
  const total = (): number => (useMemory ? seenIds!.size : store!.count());
  const dualCountCheck = (): void => {
    // A real COUNT(*) (not the store's running count, which is exact only
    // for writes made through it): dual mode's job is to check the DB's
    // actual state. One b-tree walk per poll, at the 6h cadence.
    if (store && useMemory) {
      const dbCount = store.recount();
      if (seenIds!.size !== dbCount) idsMismatch("count", `set ${seenIds!.size} vs db ${dbCount} after this poll`);
    }
  };
  if (!fresh.length) { dualCountCheck(); return { newRows: 0, seenTotal: total() }; }
  if (seenIds) fresh.forEach((r) => seenIds!.add(r.disseminationId));
  try {
    const file = path.join(dir, `${observedDate}.jsonl.gz`);
    const existing = fs.existsSync(file) ? zlib.gunzipSync(fs.readFileSync(file)).toString("utf8") : "";
    const payload = existing + fresh.map((r) => JSON.stringify(r)).join("\n") + "\n";
    fs.writeFileSync(file, zlib.gzipSync(payload));
  } catch (e) {
    console.error("[datacore] dtccswaps archive:", e);
  }
  const freshIds = fresh.map((r) => r.disseminationId);
  appendSeenIdsIndex(dir, freshIds);
  if (store) {
    // AFTER the day file and the index: a crash between them leaves the db
    // BEHIND the index, which the next boot's backfill heals; the reverse
    // order could record an id whose row never reached the archive.
    try {
      store.insertMany(freshIds);
    } catch (e) {
      idsEmit("DTCC-IDS-DB-ERROR", `insert of ${freshIds.length} ids failed: ${e instanceof Error ? e.message : String(e)} — the next boot's backfill re-imports them from the compact index`);
    }
  }
  dualCountCheck();
  return { newRows: fresh.length, seenTotal: total() };
}

/** Most recently archived day's rows (plain or gz), walked backward up to
 *  `lookbackDays` — used to backfill the live cache when EVERY day in
 *  `refreshDtccSwaps`'s own `DTCC_LOOKBACK_DAYS` vendor-side walk 403s (a
 *  cold boot during a multi-day vendor/network outage), so that outage
 *  doesn't report `warming_up` forever over a real archived day already on
 *  disk. Same pattern as nrcReactorStatus.ts's `readArchivedReactorStatus`/
 *  gridGeneration.ts's `readArchivedGeneration` — the VULNERABLE-queue
 *  entry this closes (research/open_questions.md's cold-cache-no-disk-
 *  backfill table named dtccSwaps.ts as the queue's last remaining entry,
 *  explicitly lower-priority because this module's own live 8-day lookback
 *  already mitigates a single missed poll; this is the disk fallback for
 *  when even that lookback comes up empty). Returns ONE day's rows, not a
 *  merged window, since `topNotionalRows`'s own contract is "this cycle's
 *  file" — merging days would silently turn a single-file top-N into a
 *  multi-day one. `fileDate` is the day-file's own name (the day the
 *  pipeline RAN, not any row's event date — same `observedDate` semantics
 *  `archiveNewRows` writes under), a separate, deliberately-longer default
 *  window than the live walk (14 vs 7) since a real vendor outage lasting
 *  longer than the live lookback is exactly the case this exists for. */
export function readArchivedDtccRows(baseDir?: string, nowMs?: number, lookbackDays = 14): { fileDate: string; rows: DtccSwapRow[] } | null {
  const now = nowMs ?? Date.now();
  const dir = dtccDir(baseDir);
  for (let i = 0; i < lookbackDays; i++) {
    const iso = new Date(now - i * 86400_000).toISOString().slice(0, 10);
    const rows: DtccSwapRow[] = [];
    for (const fp of [path.join(dir, `${iso}.jsonl`), path.join(dir, `${iso}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try {
          const r = JSON.parse(line);
          if (r?.disseminationId) rows.push(r);
        } catch { continue; }
      }
    }
    if (rows.length > 0) return { fileDate: iso, rows };
  }
  return null;
}

// ── Cache + poll ─────────────────────────────────────────────────────────

// How many rows to keep in memory for the /data browsing view — a bounded
// slice of the current cycle's US-underlier rows, never the full ~112k-row
// set (MEMORY LAW: no unbounded selects). 100 is generous for a "largest
// notionals" table without meaningfully growing this poller's footprint
// (each row is a handful of short strings/numbers).
const TOP_ROWS_CAP = 100;

/** Largest-notional US-underlier rows from THIS poll cycle only, sorted
 *  descending. Notional-masked rows (Dodd-Frank caps -> null, never a
 *  fabricated 0) sort after every rated row rather than dropping out or
 *  sorting as if notional were zero. This is a snapshot of one cycle's
 *  observed file, not a running archive-wide ranking — archiveNewRows
 *  already holds the durable history; scanning it for a global top-N would
 *  mean decompressing every prior day's file on every request, which the
 *  MEMORY LAW's "no unbounded selects" forbids for an interactive route. */
export function topNotionalRows(rows: DtccSwapRow[], cap = TOP_ROWS_CAP): DtccSwapRow[] {
  return [...rows]
    .sort((a, b) => {
      if (a.notionalAmountLeg1 == null && b.notionalAmountLeg1 == null) return 0;
      if (a.notionalAmountLeg1 == null) return 1;
      if (b.notionalAmountLeg1 == null) return -1;
      return b.notionalAmountLeg1 - a.notionalAmountLeg1;
    })
    .slice(0, cap);
}

export interface DtccPollResult {
  at: number;
  fileDate: string;
  sourceDate: string;
  usRows: number;
  newRows: number;
  totalArchived: number;
  topRows: DtccSwapRow[];
}

let cache: DtccPollResult | null = null;
let polling = false;

export function latestDtcc(): DtccPollResult | null {
  return cache;
}

function ymdUnderscore(d: Date): string {
  return d.toISOString().slice(0, 10).replace(/-/g, "_");
}

/** How many calendar days back to walk looking for a published file. Found
 *  live 2026-08-23 (production had ZERO archived rows despite the pipeline
 *  passing gate 1 two sessions earlier): the vendor does not keep every
 *  dated zip permanently available under a stable "cumulative" URL as the
 *  original build assumed — GET on the current or immediately-prior day
 *  routinely 403s (S3 AccessDenied, the bucket's stand-in for "no such
 *  key" when the caller lacks ListBucket) while a few days further back
 *  200s. 7 covers a long weekend/holiday stretch generously without
 *  hammering the source. */
const DTCC_LOOKBACK_DAYS = 7;

/** Walks backward from "today" (ET-naive UTC date) trying each calendar
 *  day's dated zip until one is actually published, instead of assuming
 *  "today" is always available and relying on the next 6h poll to
 *  self-heal — that assumption silently never recovers on a day whose
 *  own dated file is never going to exist (a weekend/holiday with no
 *  trading activity to disseminate), which is exactly what left
 *  production's archive directory never created at all. Rows are still
 *  archived under TODAY's calendar day (`observedDate`, unchanged
 *  semantics — "one file per calendar day this pipeline RAN") regardless
 *  of which vendor-dated file supplied them; `sourceDate` records which
 *  one actually did, for honesty (FRESHNESS LAW). */
// Crash-loop guard (KNOWN BROKEN #41 leak-audit NEXT(1)): this refresh runs
// UNCONDITIONALLY at every process boot (bootDtccSwapsPoll below) and
// streams a ~132MB zip + rebuilds/reads the seenIds index — the audit's own
// S2 finding, the single biggest contributor to the measured boot-memory
// burst. If that burst OOM-kills the process before `cache`/the index are
// written, the crash erases the only evidence an attempt was made and the
// very next boot retries the identical expensive fetch immediately — the
// exact bug shape `crashSafeRefresh.ts` was built to close for
// refreshShadowStats/refreshPortDwell (routes.ts). Same 6h cooldown
// convention as those two, matched to this module's own poll cadence.
const DTCC_CRASH_COOLDOWN_MS = 6 * 60 * 60_000;
let refreshing = false;

export async function refreshDtccSwaps(fetchImpl: FetchFn = fetch as any, nowMs?: number, baseDir?: string): Promise<void> {
  if (refreshing) return;
  refreshing = true;
  try {
    const result = await guardedRefresh("dtccswaps", DTCC_CRASH_COOLDOWN_MS, async () => {
      // Dedup store backfill first (dual/sqlite; no-op once done or in memory
      // mode) so its streamed import never overlaps the zip stream below.
      await ensureDtccIdsReady(baseDir);
      const now = new Date(nowMs ?? Date.now());
      let sourceDate = "";
      let foundRows: DtccSwapRow[] = [];
      let foundLines = 0;
      let lastStatus = 0;
      for (let back = 0; back <= DTCC_LOOKBACK_DAYS; back++) {
        const candidate = new Date(now.getTime() - back * 86_400_000);
        const candidateYmd = ymdUnderscore(candidate);
        let idx: Record<string, number> | null = null;
        let expectedCols = 0;
        const rows: DtccSwapRow[] = [];
        const { ok, status, lines } = await streamDtccZip(dtccUrl(candidateYmd), (line, isHeader) => {
          if (isHeader) {
            idx = parseHeader(line);
            expectedCols = parseCsvLine(line).length;
            return;
          }
          if (!idx) return;
          const row = parseDtccLine(line, idx, expectedCols);
          if (row) rows.push(row);
        }, fetchImpl);
        lastStatus = status;
        if (ok) {
          sourceDate = candidate.toISOString().slice(0, 10);
          foundRows = rows;
          foundLines = lines;
          break;
        }
        console.error(`[datacore] dtccswaps ${candidateYmd}: HTTP ${status} / no body${back < DTCC_LOOKBACK_DAYS ? " — trying the prior day" : ""}`);
      }
      if (!sourceDate) {
        console.error(`[datacore] dtccswaps: no published file found across the last ${DTCC_LOOKBACK_DAYS + 1} days (last HTTP ${lastStatus})`);
        if (!cache) {
          const backfill = readArchivedDtccRows(baseDir, nowMs);
          if (backfill) {
            cache = {
              at: Date.now(), fileDate: backfill.fileDate, sourceDate: "",
              usRows: backfill.rows.length, newRows: 0, totalArchived: seenIdsTotal(dtccDir(baseDir)),
              topRows: topNotionalRows(backfill.rows),
            };
            console.log(`[datacore] dtccswaps: live walk found nothing — backfilled from archived ${backfill.fileDate} (${backfill.rows.length} rows)`);
          }
        }
        return;
      }
      const observedDate = now.toISOString().slice(0, 10);
      const { newRows, seenTotal } = archiveNewRows(foundRows, observedDate, baseDir);
      cache = {
        at: Date.now(), fileDate: observedDate, sourceDate, usRows: foundRows.length, newRows,
        totalArchived: seenTotal, topRows: topNotionalRows(foundRows),
      };
      console.log(`[datacore] dtccswaps ${observedDate} (source dated ${sourceDate}): ${foundLines} total lines, ${foundRows.length} US-underlier, ${newRows} new, ${seenTotal} archived total`);
    }, baseDir, nowMs);
    if (!result.ran) {
      console.error(`[datacore] dtccswaps: skipped this cycle (${result.reason}) — ${result.detail}`);
    }
  } catch (e) {
    console.error("[datacore] dtccswaps refresh:", e);
  } finally {
    refreshing = false;
  }
}

/** Daily source, no fixed publish-time guarantee known yet — 6h poll is
 *  self-healing (re-fetching the full cumulative file naturally catches up
 *  on any missed day, since it's cumulative-from-inception). Eager boot. */
export function bootDtccSwapsPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshDtccSwaps().catch((e) => console.error("[datacore] dtccswaps boot:", e?.message || e));
  setInterval(() => { refreshDtccSwaps(); }, intervalMs).unref?.();
}
