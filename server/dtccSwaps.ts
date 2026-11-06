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
import { Readable, pipeline } from "stream";
import { once } from "events";
import { promisify } from "util";
import zlib from "zlib";
import readline from "readline";
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";
import { guardedRefresh } from "./crashSafeRefresh";
import { DtccIdsStore, dtccIdsMode, DTCC_IDS_DB_FILE, countBufferLines, countNonEmptyLines, type DtccIdsBackfill, type DtccIdsMode } from "./dtccIdsStore";
import { CompactIdSet } from "./dtccIdSet";
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

  // 2026-11-05 REPAIR: every exit (done, parse error, network failure) releases
  // the HTTP body. The zip's central directory after the member is never read,
  // so without this the body stayed open until undici's 300 s timeout.
  let bounded: Readable | null = null;
  let rl: readline.Interface | null = null;
  try {
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
    const member = new Readable({
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

    bounded = member;
    let decompressed: Readable = member;
    if (fixed.method === 8) {
      // stream.pipeline, never member.pipe(): .pipe() leaves the member with no
      // 'error' listener, so a connection dying mid-member (undici's
      // TypeError: terminated) was an uncaught exception that restarted the
      // container (live 2026-11-04 05:03Z, 2026-11-05 01:17Z). pipeline hands
      // the error to the inflater, readline re-emits it and the for-await
      // rejects: one failed refresh, caught by refreshDtccSwaps.
      const inflate = zlib.createInflateRaw();
      pipeline(member, inflate, () => { /* surfaced through readline below */ });
      decompressed = inflate;
    }
    rl = readline.createInterface({ input: decompressed, crlfDelay: Infinity });
    let lines = 0;
    let first = true;
    for await (const line of rl) {
      if (!line) continue;
      onLine(line, first);
      first = false;
      lines++;
    }
    return { ok: true, status: resp.status, lines };
  } finally {
    rl?.close();
    bounded?.destroy();
    await Promise.resolve(source.return?.()).catch((e) => {
      console.warn(`[datacore] dtccswaps: could not release the response body: ${e?.message || e}`);
    });
  }
}

// ── Archive (append-only JSONL.gz per calendar day observed; dedup by
//    Dissemination Identifier across ALL previously archived days, since
//    the upstream file is cumulative-from-inception and re-serves every
//    prior event daily) ───────────────────────────────────────────────────

function dtccDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "dtccswaps");
}

/** Every Dissemination Identifier ever archived (the memory path's dedup set).
 *  A CompactIdSet (2026-11-06 dtcc-compact REPAIR): the canonical decimal IDs
 *  — all of the live ones — in a typed-array hash table at ~11.4 B/ID instead
 *  of a Set<string> at 64-76 B/ID; anything else in a plain Set. Exactly the
 *  same membership as a Set<string> (server/dtccIdSet.ts). */
let seenIds: CompactIdSet | null = null;

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
 *  back to the full scan instead of silently starting from zero.
 *  (2026-11-06: the lines are counted first so the set is sized once, then
 *  walked in the buffer — the same lines `toString().split("\n")` + `if
 *  (line)` gave, with no 139 MB string and no 6.94M-element array.) */
function readSeenIdsIndex(dir: string): CompactIdSet | null {
  const file = seenIdsIndexPath(dir);
  if (!fs.existsSync(file)) return null;
  try {
    const raw = zlib.gunzipSync(fs.readFileSync(file));
    const seen = new CompactIdSet(countBufferLines(raw));
    let start = 0;
    for (;;) {
      const nl = raw.indexOf(10, start);
      if (nl === -1) break;
      if (nl > start) seen.addBytes(raw, start, nl);
      start = nl + 1;
    }
    if (start < raw.length) seen.addBytes(raw, start, raw.length);
    return seen;
  } catch (e) {
    console.error("[datacore] dtccswaps readSeenIdsIndex:", e);
    return null;
  }
}

/** Full rewrite from a Set — used once, right after the legacy full-archive
 *  scan below (first boot after this fix, or any future from-scratch
 *  rebuild). Never called on the routine append path. */
function writeSeenIdsIndex(dir: string, ids: CompactIdSet): void {
  try {
    fs.mkdirSync(dir, { recursive: true });
    const payload = Array.from(ids).join("\n") + (ids.size ? "\n" : "");
    fs.writeFileSync(seenIdsIndexPath(dir), zlib.gzipSync(payload));
  } catch (e) {
    console.error("[datacore] dtccswaps writeSeenIdsIndex:", e);
  }
}

/** Routine (non-boot) write path: appends newly-archived IDs to the compact
 *  index as ONE NEW GZIP MEMBER (2026-11-06 dtcc-stall REPAIR). Concatenated
 *  gzip members are a single valid gzip stream (RFC 1952 §2.2; Node's Gunzip
 *  reads every member since v5.9.0 — zlib.gunzipSync, zlib.gunzip and
 *  createGunzip alike), so every reader sees the same one-ID-per-line text.
 *  The old shape — gunzip the whole index, concatenate, re-gzip the whole
 *  index, rewrite it — cost 7.1 s of synchronous zlib at 6.94M IDs (measured
 *  2026-11-06) on every poll that found fresh rows; this costs only the new
 *  IDs (41 ms for 128k). Runs only when a poll actually finds fresh rows. */
function appendSeenIdsIndex(dir: string, ids: string[]): void {
  if (!ids.length) return;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(seenIdsIndexPath(dir), zlib.gzipSync(ids.join("\n") + "\n"));
  } catch (e) {
    indexAppendFailed(e, ids.length);
  }
}

/** Async twin of appendSeenIdsIndex (the refresh path): the same bytes —
 *  zlib.gzip and zlib.gzipSync produce identical output — compressed on the
 *  libuv threadpool instead of the event loop. */
async function appendSeenIdsIndexAsync(dir: string, ids: string[]): Promise<void> {
  if (!ids.length) return;
  try {
    await fs.promises.mkdir(dir, { recursive: true });
    await fs.promises.appendFile(seenIdsIndexPath(dir), await gzipAsync(ids.join("\n") + "\n"));
  } catch (e) {
    indexAppendFailed(e, ids.length);
  }
}

const gzipAsync = promisify(zlib.gzip);

// ── NON-BLOCKING LOAD (2026-11-06 dtcc-stall REPAIR, KEEP ALIVE) ───────────
//
// Live 2026-11-06: an EVENTLOOP-LAG row of 7.7-10.2 s about 5 minutes after
// 26 of that day's 39 boots — the eager boot poll finishing its zip download
// and then running loadSeenIds synchronously (gunzipSync of the 139 MB index,
// split into 6.94M strings, a 6.94M-entry Set). Reproduced locally on a
// synthetic 6.94M-ID index: 8.4 s in one block (gunzip 0.8 s, split 1.7 s,
// Set build 5.0 s). The refresh path now loads through loadSeenIdsAsync
// below: the index is STREAMED (createReadStream -> createGunzip, both under
// stream.pipeline so a read error rejects instead of going unhandled), split
// on "\n" at the byte level chunk by chunk, and the loop gets a turn at least
// every DTCC_ASYNC_SLICE_MS. The new Set is private until it is COMPLETE and
// only then installed as `seenIds`, so a dedup decision never sees a partial
// set; a refresh that arrives while it loads awaits the same load. The sync
// loadSeenIds below stays, unchanged, for direct callers.

/** Longest stretch (ms) the non-blocking DTCC paths run before handing the
 *  event loop a turn. server/eventLoopLag.ts alerts (EVENTLOOP-LAG) at 500 ms;
 *  the gzip work itself runs on the libuv threadpool, off the loop. */
export const DTCC_ASYNC_SLICE_MS = 8;

const yieldToLoop = (): Promise<void> => new Promise<void>((resolve) => setImmediate(resolve));

/** A file's bytes as a stream: plain, or gunzipped with every gzip member
 *  read. stream.pipeline (never .pipe()) so a read error on the file destroys
 *  the gunzip stream with that error and the consumer's for-await rejects —
 *  an unlistened 'error' on the source would otherwise end the process. */
function fileByteStream(file: string, gunzip: boolean): Readable {
  const src = fs.createReadStream(file);
  if (!gunzip) return src;
  const gz = zlib.createGunzip();
  pipeline(src, gz, () => { /* any error has already destroyed gz; the consumer's for-await rejects with it */ });
  return gz;
}

/** Calls onLine(buf, start, end) for each "\n"-separated line of a byte
 *  stream — the same lines `text.split("\n")` + `if (line)` gives on the
 *  decoded text: no trimming, "\r" kept, empty lines skipped, a final
 *  unterminated line kept (0x0A never occurs inside a multi-byte UTF-8
 *  sequence, so splitting the bytes first is exact). The pieces of a line
 *  that straddles chunks are joined once, when its "\n" arrives (each byte is
 *  copied at most once, however long the line). Yields to the loop whenever a
 *  slice has run DTCC_ASYNC_SLICE_MS, even when the stream hands chunks over
 *  back to back through microtasks. Rejects if the stream errors. */
async function forEachLine(input: AsyncIterable<Buffer>, onLine: (buf: Buffer, start: number, end: number) => void): Promise<void> {
  let pending: Buffer[] = [];
  let sliceStart = performance.now();
  for await (const chunk of input) {
    let start = 0;
    for (;;) {
      const nl = chunk.indexOf(10, start);
      if (nl === -1) break;
      if (pending.length) {
        pending.push(chunk.subarray(start, nl));
        const line = Buffer.concat(pending);
        pending = [];
        if (line.length) onLine(line, 0, line.length);
      } else if (nl > start) {
        onLine(chunk, start, nl);
      }
      start = nl + 1;
    }
    if (start < chunk.length) pending.push(chunk.subarray(start));
    if (performance.now() - sliceStart >= DTCC_ASYNC_SLICE_MS) {
      await yieldToLoop();
      sliceStart = performance.now();
    }
  }
  if (pending.length) {
    const line = Buffer.concat(pending);
    if (line.length) onLine(line, 0, line.length);
  }
}

/** Non-empty lines of a gzipped file, streamed — the number countBufferLines
 *  gives the synchronous reader, so both size the set identically. */
async function countIndexLines(file: string): Promise<number> {
  const state = { lineHasBytes: false };
  let n = 0;
  let sliceStart = performance.now();
  for await (const chunk of fileByteStream(file, true)) {
    n += countNonEmptyLines(chunk as Buffer, state);
    if (performance.now() - sliceStart >= DTCC_ASYNC_SLICE_MS) {
      await yieldToLoop();
      sliceStart = performance.now();
    }
  }
  return n + (state.lineHasBytes ? 1 : 0);
}

/** Non-blocking twin of readSeenIdsIndex: same null-vs-Set contract (null =
 *  no index, or it failed to read — the caller falls back to the day-file
 *  scan exactly as before), same lines, streamed. A first streamed pass
 *  counts the lines so the set is allocated once at its final size (no
 *  resize, no second table, during the boot load); the second adds each
 *  line straight from the bytes. */
async function readSeenIdsIndexAsync(dir: string): Promise<CompactIdSet | null> {
  const file = seenIdsIndexPath(dir);
  if (!fs.existsSync(file)) return null;
  try {
    const seen = new CompactIdSet(await countIndexLines(file));
    await forEachLine(fileByteStream(file, true), (buf, start, end) => { seen.addBytes(buf, start, end); });
    return seen;
  } catch (e) {
    console.error("[datacore] dtccswaps readSeenIdsIndex:", e);
    return null;
  }
}

/** Non-blocking twin of writeSeenIdsIndex: the same text (Array.from order,
 *  one ID per line, a trailing "\n" when non-empty) streamed through gzip
 *  into a temp file, then renamed over the index in one step, so a crash
 *  mid-write never leaves a half-written index behind. */
async function writeSeenIdsIndexAsync(dir: string, ids: CompactIdSet): Promise<void> {
  const file = seenIdsIndexPath(dir);
  const tmp = `${file}.rebuild.tmp`;
  try {
    await fs.promises.mkdir(dir, { recursive: true });
    const gz = zlib.createGzip();
    const written = new Promise<void>((resolve, reject) => {
      pipeline(gz, fs.createWriteStream(tmp), (err) => (err ? reject(err) : resolve()));
    });
    let batch: string[] = [];
    for (const id of ids) {
      batch.push(id);
      if (batch.length >= 50_000) {
        if (!gz.write(batch.join("\n") + "\n")) await once(gz, "drain");
        batch = [];
      }
    }
    if (batch.length) gz.write(batch.join("\n") + "\n");
    gz.end();
    await written;
    fs.renameSync(tmp, file);
  } catch (e) {
    console.error("[datacore] dtccswaps writeSeenIdsIndex:", e);
    fs.rm(tmp, { force: true }, (rmErr) => {
      if (rmErr) console.error("[datacore] dtccswaps writeSeenIdsIndex: could not remove the temp file:", rmErr);
    });
  }
}

/** Non-blocking twin of loadSeenIds — the identical set (same members; same
 *  iteration order, since both size and fill it the same way), the identical
 *  fallback (no readable index -> scan the name-anchored day files in readdir
 *  order, stop at the first unreadable one, count malformed lines, persist
 *  the index), with every file streamed and the loop yielded to between
 *  slices. Never touches `seenIds`: the caller installs the result once it
 *  is complete. */
export async function loadSeenIdsAsync(dir: string): Promise<CompactIdSet> {
  const indexed = await readSeenIdsIndexAsync(dir);
  if (indexed) return indexed;

  const seen = new CompactIdSet();
  let malformedLines = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
      await forEachLine(fileByteStream(path.join(dir, f), f.endsWith(".gz")), (buf, start, end) => {
        try { seen.add(JSON.parse(buf.toString("utf8", start, end)).disseminationId); }
        catch { malformedLines++; }
      });
    }
  } catch (e) {
    console.error("[datacore] dtccswaps loadSeenIds:", e);
  }
  if (malformedLines) {
    console.error(`[datacore] dtccswaps loadSeenIds: ${malformedLines} malformed archive line(s) skipped`);
  }
  await writeSeenIdsIndexAsync(dir, seen);
  return seen;
}

/** Bumped by _resetDtccForTests so a load that started before a reset can
 *  never install its Set afterwards. */
let seenIdsEpoch = 0;
let seenIdsLoad: { epoch: number; promise: Promise<void> } | null = null;

/** The memory path's Set, loaded WITHOUT blocking the loop (single flight).
 *  Concurrent callers await the same load; the Set is installed only when
 *  complete, and never over a Set someone loaded synchronously meanwhile
 *  (that one may already hold newer IDs). */
async function ensureSeenIdsLoaded(dir: string): Promise<CompactIdSet> {
  for (;;) {
    const loaded = seenIds;
    if (loaded) return loaded;
    const epoch = seenIdsEpoch;
    let load = seenIdsLoad;
    if (!load || load.epoch !== epoch) {
      const promise = loadSeenIdsAsync(dir).then((built) => {
        if (seenIdsEpoch === epoch && !seenIds) seenIds = built;
      });
      const entry = { epoch, promise };
      const clear = () => { if (seenIdsLoad === entry) seenIdsLoad = null; };
      promise.then(clear, clear);
      seenIdsLoad = entry;
      load = entry;
    }
    await load.promise;
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
export function loadSeenIds(dir: string): CompactIdSet {
  const indexed = readSeenIdsIndex(dir);
  if (indexed) return indexed;

  const seen = new CompactIdSet();
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

// ── ARCHIVE WRITE FAILURES ([REPAIR] 2026-11-06, archive integrity) ────────
// The day file IS the archive; the seen-ID index, the in-memory claim and the
// dedup DB only record what reached it. Before, a failed day-file write was a
// console line and the fresh IDs were still appended to the index (and kept
// in the claim), so those rows were treated as archived forever: a permanent
// gap (CLAUDE.md priority 1: an archive gap never refills), and the volume is
// short of space. Now nothing records them, the in-memory claim is dropped so
// the next poll reloads the set from the index (which never got them) and
// re-archives the rows, and the failure is an audit row (DTCC-ARCHIVE-ERROR).
function archiveWriteFailed(e: unknown, rows: number, file: string): void {
  seenIds = null; seenIdsEpoch++; seenIdsLoad = null; seenIdsMutations++;
  idsEmit("DTCC-ARCHIVE-ERROR", `${rows} new row(s) not written to ${path.basename(file)}: ${e instanceof Error ? e.message : String(e)} — not recorded as seen; the next poll retries them`);
}

/** The rows reached the day file but their IDs did not reach the index: this
 *  process still knows them (its in-memory claim stands), but after a restart
 *  the index lacks them and a later poll archives them again — a duplicate,
 *  never a gap. Loud, so the duplicate window is visible. */
function indexAppendFailed(e: unknown, ids: number): void {
  idsEmit("DTCC-ARCHIVE-ERROR", `seen-ID index append of ${ids} id(s) failed: ${e instanceof Error ? e.message : String(e)} — the rows are archived; after a restart a later poll may archive them again (a duplicate, never a gap)`);
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

/** The refresh path's twin of readyIdsStoreSync: the same store, the same
 *  degrade-to-memory contract, but a still-pending (or earlier failed)
 *  backfill is retried through the STREAMED ensureReadyAsync, never the
 *  whole-index gunzipSync of ensureReadySync. */
async function readyIdsStoreAsync(dir: string): Promise<DtccIdsStore | null> {
  const store = openIdsStore(dir);
  if (!store) return null;
  if (!store.isReady) await store.ensureReadyAsync();
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
 *  the DB's (sqlite). Its one caller is the refresh path's disk backfill, so
 *  (2026-11-06) the Set loads / the store backfills without blocking the
 *  loop; the count itself is unchanged. */
async function seenIdsTotalAsync(dir: string): Promise<number> {
  if (dtccIdsMode() === "sqlite") {
    const store = await readyIdsStoreAsync(dir);
    if (store) return store.count();
  }
  return (await ensureSeenIdsLoaded(dir)).size;
}

export interface DtccIdsStats {
  mode: DtccIdsMode;
  /** Set size, or null when the Set has not been built (sqlite mode's win). */
  memoryCount: number | null;
  /** true while the non-blocking load (loadSeenIdsAsync) is building the Set;
   *  memoryCount stays null until it is complete and installed. */
  memoryLoading: boolean;
  /** Exact bytes of the set's typed-array table (every canonical numeric ID),
   *  or null when the set has not been built. */
  memoryTableBytes: number | null;
  /** Members held in the set's plain-Set fallback (IDs that are not canonical
   *  decimals; 0 for every live ID seen so far), or null when not built. */
  memoryFallbackCount: number | null;
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
    memoryLoading: seenIdsLoad !== null,
    memoryTableBytes: seenIds ? seenIds.tableBytes() : null,
    memoryFallbackCount: seenIds ? seenIds.fallbackSize : null,
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
  seenIdsEpoch++; seenIdsLoad = null;
  closeIdsStore();
  idsStoreError = null; idsMismatches = 0; idsLastMismatch = null; idsLastLookupMs = null; idsLastLookupRows = null;
  idsDiskGuard = { active: false, reason: null, freeBytes: null, at: null };
}

interface FreshDecision {
  /** Rows whose Dissemination Identifier was never archived (the served verdict). */
  fresh: DtccSwapRow[];
  store: DtccIdsStore | null;
  total: () => number;
  dualCountCheck: () => void;
}

/** Bumped every time decideFresh records fresh IDs in the Set: lets a verdict
 *  computed in slices (freshInSlices) prove nothing was claimed meanwhile. */
let seenIdsMutations = 0;

/** The memory verdict computed in slices for the refresh path — exactly
 *  `rows.filter((r) => !set.has(r.disseminationId))` (136.7k lookups ≈ 210 ms
 *  in one go at 6.94M IDs, measured 2026-11-06), the loop yielded to between
 *  slices. Claims nothing: decideFresh records the IDs, in one turn, and only
 *  uses this verdict if the Set is still the same object and unmutated. */
interface SlicedVerdict { set: CompactIdSet; mutations: number; fresh: DtccSwapRow[] }
async function freshInSlices(rows: DtccSwapRow[], set: CompactIdSet): Promise<SlicedVerdict> {
  const mutations = seenIdsMutations;
  const fresh: DtccSwapRow[] = [];
  let sliceStart = performance.now();
  for (let i = 0; i < rows.length; i++) {
    if (!set.has(rows[i].disseminationId)) fresh.push(rows[i]);
    if ((i & 1023) === 1023 && performance.now() - sliceStart >= DTCC_ASYNC_SLICE_MS) {
      await yieldToLoop();
      sliceStart = performance.now();
    }
  }
  return { set, mutations, fresh };
}

/** The dedup decision both archive paths share. The Set update happens in
 *  the same turn as the final verdict, so no other archive call can
 *  interleave between deciding a row is fresh and recording it as seen. A
 *  verdict precomputed in slices is used only when the Set it was computed
 *  against is still installed and nothing was recorded since; otherwise it is
 *  recomputed here, synchronously (never stale). */
function decideFresh(rows: DtccSwapRow[], dir: string, mode: DtccIdsMode, store: DtccIdsStore | null, sliced?: SlicedVerdict): FreshDecision {
  const useMemory = mode !== "sqlite" || !store;
  let memFresh: DtccSwapRow[] | null = null;
  let dbFresh: DtccSwapRow[] | null = null;
  if (useMemory) {
    if (!seenIds) seenIds = loadSeenIds(dir);
    memFresh = sliced && sliced.set === seenIds && sliced.mutations === seenIdsMutations
      ? sliced.fresh
      : rows.filter((r) => !seenIds!.has(r.disseminationId));
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
  if (fresh.length && seenIds) {
    fresh.forEach((r) => seenIds!.add(r.disseminationId));
    seenIdsMutations++;
  }
  return { fresh, store, total, dualCountCheck };
}

function dayFilePath(dir: string, observedDate: string): string {
  return path.join(dir, `${observedDate}.jsonl.gz`);
}

/** AFTER the day file and the index: a crash between them leaves the db
 *  BEHIND the index, which the next boot's backfill heals; the reverse order
 *  could record an id whose row never reached the archive. */
function insertFreshIds(store: DtccIdsStore | null, freshIds: string[]): void {
  if (!store) return;
  try {
    store.insertMany(freshIds);
  } catch (e) {
    idsEmit("DTCC-IDS-DB-ERROR", `insert of ${freshIds.length} ids failed: ${e instanceof Error ? e.message : String(e)} — the next boot's backfill re-imports them from the compact index`);
  }
}

/** Appends only rows whose Dissemination Identifier has never been archived
 *  before, to <archive>/dtccswaps/<observedDate>.jsonl.gz (one file per
 *  calendar day this pipeline RAN, not per event date — events span years).
 *  Returns { newRows, seenTotal }. Synchronous, for direct callers; the
 *  refresh path runs archiveNewRowsAsync (same decision, same bytes on disk,
 *  without blocking the loop). The day file gets the fresh rows as one NEW
 *  gzip member (2026-11-06; it was gunzipped, concatenated and re-gzipped
 *  whole): the same JSONL text for every reader, without re-reading or
 *  rewriting the rows already archived that day. */
export function archiveNewRows(rows: DtccSwapRow[], observedDate: string, baseDir?: string): { newRows: number; seenTotal: number } {
  const dir = dtccDir(baseDir);
  fs.mkdirSync(dir, { recursive: true });
  const mode = dtccIdsMode();
  // dual/sqlite: the store, or null when it could not open/backfill — then
  // this call runs the memory path (already logged by the store helpers).
  const store = mode === "memory" ? null : readyIdsStoreSync(dir);
  const d = decideFresh(rows, dir, mode, store);
  if (!d.fresh.length) { d.dualCountCheck(); return { newRows: 0, seenTotal: d.total() }; }
  const freshIds = d.fresh.map((r) => r.disseminationId);
  try {
    fs.appendFileSync(dayFilePath(dir, observedDate), zlib.gzipSync(d.fresh.map((r) => JSON.stringify(r)).join("\n") + "\n"));
  } catch (e) {
    const seenBefore = d.total() - new Set(freshIds).size;
    archiveWriteFailed(e, d.fresh.length, dayFilePath(dir, observedDate));
    return { newRows: 0, seenTotal: seenBefore };
  }
  appendSeenIdsIndex(dir, freshIds);
  insertFreshIds(d.store, freshIds);
  d.dualCountCheck();
  return { newRows: d.fresh.length, seenTotal: d.total() };
}

/** JSON lines for the day file, serialized in slices (128k rows ≈ 190 ms of
 *  JSON.stringify in one go, measured 2026-11-06). Same text as
 *  `rows.map((r) => JSON.stringify(r)).join("\n") + "\n"`. */
async function jsonlPayloadAsync(rows: DtccSwapRow[]): Promise<string> {
  const parts = new Array<string>(rows.length);
  let sliceStart = performance.now();
  for (let i = 0; i < rows.length; i++) {
    parts[i] = JSON.stringify(rows[i]);
    if ((i & 1023) === 1023 && performance.now() - sliceStart >= DTCC_ASYNC_SLICE_MS) {
      await yieldToLoop();
      sliceStart = performance.now();
    }
  }
  return parts.join("\n") + "\n";
}

/** archiveNewRows for the refresh path (2026-11-06 dtcc-stall REPAIR): the
 *  Set is loaded (or the store backfilled) without blocking the loop and
 *  awaited BEFORE the decision, so a dedup never runs against a partial set;
 *  the decision is the shared synchronous decideFresh; the day file and the
 *  index get exactly the bytes archiveNewRows writes, compressed on the
 *  threadpool. Same order of writes (day file, index, store), same
 *  return value, same log lines. */
export function archiveNewRowsAsync(rows: DtccSwapRow[], observedDate: string, baseDir?: string): Promise<{ newRows: number; seenTotal: number }> {
  // One at a time: the refresh is already single-flight, and chaining keeps
  // any other concurrent caller from interleaving its writes or its total
  // with another call's (the caller still receives its own rejection).
  const run = archiveChain.then(() => archiveNewRowsOnce(rows, observedDate, baseDir));
  archiveChain = run.then(() => undefined, () => undefined);
  return run;
}
let archiveChain: Promise<void> = Promise.resolve();

async function archiveNewRowsOnce(rows: DtccSwapRow[], observedDate: string, baseDir?: string): Promise<{ newRows: number; seenTotal: number }> {
  const dir = dtccDir(baseDir);
  await fs.promises.mkdir(dir, { recursive: true });
  const mode = dtccIdsMode();
  const store = mode === "memory" ? null : await readyIdsStoreAsync(dir);
  let sliced: SlicedVerdict | undefined;
  if (mode !== "sqlite" || !store) {
    const set = await ensureSeenIdsLoaded(dir);
    // room for every row's ID BEFORE the claim, so recording the fresh ones
    // never resizes the table in one block (a resize, when due, runs in slices)
    await set.reserveAsync(rows.length, DTCC_ASYNC_SLICE_MS);
    sliced = await freshInSlices(rows, set);
  }
  const d = decideFresh(rows, dir, mode, store, sliced);
  if (!d.fresh.length) { d.dualCountCheck(); return { newRows: 0, seenTotal: d.total() }; }
  const freshIds = d.fresh.map((r) => r.disseminationId);
  try {
    await fs.promises.appendFile(dayFilePath(dir, observedDate), await gzipAsync(await jsonlPayloadAsync(d.fresh)));
  } catch (e) {
    const seenBefore = d.total() - new Set(freshIds).size;
    archiveWriteFailed(e, d.fresh.length, dayFilePath(dir, observedDate));
    return { newRows: 0, seenTotal: seenBefore };
  }
  await appendSeenIdsIndexAsync(dir, freshIds);
  insertFreshIds(d.store, freshIds);
  d.dualCountCheck();
  return { newRows: d.fresh.length, seenTotal: d.total() };
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
              usRows: backfill.rows.length, newRows: 0, totalArchived: await seenIdsTotalAsync(dtccDir(baseDir)),
              topRows: topNotionalRows(backfill.rows),
            };
            console.log(`[datacore] dtccswaps: live walk found nothing — backfilled from archived ${backfill.fileDate} (${backfill.rows.length} rows)`);
          }
        }
        return;
      }
      const observedDate = now.toISOString().slice(0, 10);
      // async twin: the Set loads (and the day file / index compress) without
      // blocking the loop — the 7.7-10.2 s EVENTLOOP-LAG after each boot
      const { newRows, seenTotal } = await archiveNewRowsAsync(foundRows, observedDate, baseDir);
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
