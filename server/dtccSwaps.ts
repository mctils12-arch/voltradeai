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

type NodeFetchResponse = { ok: boolean; status: number; body: any };
type FetchFn = (url: string, init?: any) => Promise<NodeFetchResponse>;

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

function loadSeenIds(dir: string): Set<string> {
  const seen = new Set<string>();
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl.gz") && !f.endsWith(".jsonl")) continue;
      const raw = f.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString("utf8")
        : fs.readFileSync(path.join(dir, f), "utf8");
      for (const line of raw.split("\n")) {
        if (!line) continue;
        try { seen.add(JSON.parse(line).disseminationId); } catch {}
      }
    }
  } catch {}
  return seen;
}

export function _resetDtccForTests(): void { seenIds = null; }

/** Appends only rows whose Dissemination Identifier has never been archived
 *  before, to <archive>/dtccswaps/<observedDate>.jsonl.gz (one file per
 *  calendar day this pipeline RAN, not per event date — events span years).
 *  Returns { newRows, seenTotal }. */
export function archiveNewRows(rows: DtccSwapRow[], observedDate: string, baseDir?: string): { newRows: number; seenTotal: number } {
  const dir = dtccDir(baseDir);
  if (!seenIds) { fs.mkdirSync(dir, { recursive: true }); seenIds = loadSeenIds(dir); }
  const fresh = rows.filter((r) => !seenIds!.has(r.disseminationId));
  if (!fresh.length) return { newRows: 0, seenTotal: seenIds.size };
  fresh.forEach((r) => seenIds!.add(r.disseminationId));
  try {
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, `${observedDate}.jsonl.gz`);
    const existing = fs.existsSync(file) ? zlib.gunzipSync(fs.readFileSync(file)).toString("utf8") : "";
    const payload = existing + fresh.map((r) => JSON.stringify(r)).join("\n") + "\n";
    fs.writeFileSync(file, zlib.gzipSync(payload));
  } catch (e: any) {
    console.error("[datacore] dtccswaps archive:", e?.message || e);
  }
  return { newRows: fresh.length, seenTotal: seenIds.size };
}

// ── Cache + poll ─────────────────────────────────────────────────────────

export interface DtccPollResult {
  at: number;
  fileDate: string;
  usRows: number;
  newRows: number;
  totalArchived: number;
}

let cache: DtccPollResult | null = null;
let polling = false;

export function latestDtcc(): DtccPollResult | null {
  return cache;
}

function ymdUnderscore(d: Date): string {
  return d.toISOString().slice(0, 10).replace(/-/g, "_");
}

/** Fetches "today" (ET-naive UTC date; the source publishes end-of-day, a
 *  1-day-stale read on any given poll self-heals on the next poll — same
 *  tolerance as every other daily-cadence root in this codebase). */
export async function refreshDtccSwaps(fetchImpl: FetchFn = fetch as any, nowMs?: number, baseDir?: string): Promise<void> {
  try {
    const now = new Date(nowMs ?? Date.now());
    const fileDate = ymdUnderscore(now);
    let idx: Record<string, number> | null = null;
    let expectedCols = 0;
    const rows: DtccSwapRow[] = [];
    const { ok, status, lines } = await streamDtccZip(dtccUrl(fileDate), (line, isHeader) => {
      if (isHeader) {
        idx = parseHeader(line);
        expectedCols = parseCsvLine(line).length;
        return;
      }
      if (!idx) return;
      const row = parseDtccLine(line, idx, expectedCols);
      if (row) rows.push(row);
    });
    if (!ok) {
      console.error(`[datacore] dtccswaps ${fileDate}: HTTP ${status} / no body`);
      return;
    }
    const observedDate = now.toISOString().slice(0, 10);
    const { newRows, seenTotal } = archiveNewRows(rows, observedDate, baseDir);
    cache = { at: Date.now(), fileDate: observedDate, usRows: rows.length, newRows, totalArchived: seenTotal };
    console.log(`[datacore] dtccswaps ${observedDate}: ${lines} total lines, ${rows.length} US-underlier, ${newRows} new, ${seenTotal} archived total`);
  } catch (e: any) {
    console.error("[datacore] dtccswaps refresh:", e?.message || e);
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
