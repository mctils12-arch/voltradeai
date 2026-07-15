/**
 * secMidas.ts — SEC MIDAS individual-security market-structure metrics,
 * quarterly files (DATACORE MAXIMUS census build #10; endpoint discovered
 * live this session via web search + WebFetch — two prior sessions
 * (2026-07-06 census entry, 2026-07-08 experiments.md WHAT DIDN'T WORK)
 * guessed static URL patterns from memory and all 404'd; the real path
 * lives under /files/opa/data/market-structure/metrics-individual-security/,
 * only discoverable from the live downloads page, not guessable).
 *
 * Source: sec.gov individual_security_{YYYY}_q{N}.zip. Public domain (US
 * government work) — resale-safe, same category as secFtd.
 *
 * VERIFIED CONTRACT (live-probed this session, not assumed):
 *  - URL: https://www.sec.gov/files/opa/data/market-structure/
 *    metrics-individual-security/individual_security_{year}_q{quarter}.zip
 *    — confirmed 200 for 2025_q4 (23.5MB zip); 2026_q1/q2 both 404 (not yet
 *    published — MIDAS runs on a multi-quarter lag, unlike FTD's 2.5-4.5wk).
 *  - Zip contains exactly one CSV member named `q{N}_{YYYY}_all.csv` plus a
 *    README.txt (kept upstream, not archived here — its content is folded
 *    into this docstring and the manifest).
 *  - CSV header (verified exact against the live q4_2025 file):
 *    Date,Security,Ticker,McapRank,TurnRank,VolatilityRank,PriceRank,
 *    LitVol('000),OrderVol('000),Hidden,TradesForHidden,HiddenVol('000),
 *    TradeVolForHidden('000),Cancels,LitTrades,OddLots,TradesForOddLots,
 *    OddLotVol('000),TradeVolForOddLots('000) — 19 fields, one row per
 *    (date, ticker) for every trading day in the quarter.
 *  - RANK SCALE DIFFERS BY SECURITY KIND (verified by full-file tabulation,
 *    not documented anywhere in the README): Security="ETF" rows are
 *    ranked in QUARTILES (McapRank/TurnRank/VolatilityRank/PriceRank in
 *    1-4); Security="Stock" rows are ranked in DECILES (1-10). Rank 1 =
 *    smallest/lowest bucket in both cases (spot-checked: AAME, a
 *    micro-cap, carries McapRank=2; A/Agilent, a mega-cap, carries
 *    McapRank=10). Never compare an ETF rank to a Stock rank directly.
 *  - A small fraction of Stock rows (121 of 235,165 in q4_2025, all four
 *    rank columns blank together) carry no rank at all — new/thin listings
 *    presumably below the ranking model's minimum data bar. Stored as
 *    null, never coerced to 0 or a guessed rank.
 *  - No official checksum exists for this format (unlike FTD's trailer
 *    lines) — integrity is enforced here by (a) exact header match and
 *    (b) a malformed-row-rate guard: a corrupted/truncated download is
 *    the realistic failure mode, and a healthy file's every non-header
 *    line splits into exactly 19 comma fields.
 *  - Default UA gets treated the same as every other sec.gov endpoint in
 *    this repo — SEC-etiquette UA sent regardless of the file being
 *    static (matches secFtd/edgarForm4 convention, not proven necessary
 *    for this specific path but zero cost to include).
 *
 * SIGNAL FRAMING (gate-locked, filed in open_questions.md): cross-sectional
 * cancel-to-trade / hidden-rate / odd-lot-rate on LOW-McapRank (small-cap)
 * Stock rows is a candidate HFT-colonization FILTER for EDGE DOCTRINE #2
 * ("fish where whales can't") — which capacity-constrained small caps are
 * already crowded by fast money vs. which are genuinely under-arbitraged.
 * Nothing here is a signal yet: this module is DATA-LADDER GATE 1 ONLY.
 *
 * VOLUME NOTE: each quarter is ~533k rows (q4_2025 measured exactly),
 * ~20MB gz archived. Only the newest MIDAS_LOOKBACK_QUARTERS are polled —
 * full 2013-2025 history (~50 quarters, ~1GB gz est.) is a volume-budget
 * decision, filed as a future backfill task, never the boot path (same
 * precedent as secFtd/finraQuery history backfills).
 *
 * MEMORY NOTE (deviation from the secFtd template, stated explicitly):
 * secFtd's refreshFtd() walks ALL candidate periods in one poll call
 * because each half-month file is small (~58k rows). A MIDAS quarter is
 * ~9x bigger — parsing several 68MB CSVs in one event-loop turn risks the
 * daemon's 1GB RSS self-kill ceiling. refreshMidas() therefore PROBES
 * newest-first through every unarchived candidate (cheap: a 404/not-yet-
 * published response downloads nothing) but ARCHIVES (parses + gzips) AT
 * MOST ONE period per call, returning the instant one succeeds; remaining
 * backlog is picked up on the next poll interval.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };

function midasUrl(year: number, quarter: number): string {
  return `https://www.sec.gov/files/opa/data/market-structure/metrics-individual-security/individual_security_${year}_q${quarter}.zip`;
}

export interface MidasRow {
  date: string;              // ISO trading date
  kind: "Stock" | "ETF";
  ticker: string;
  mcapRank: number | null;   // decile (Stock, 1-10) or quartile (ETF, 1-4); 1=smallest
  turnRank: number | null;
  volatilityRank: number | null;
  priceRank: number | null;
  litVol: number;            // '000 shares, lit (non-hidden) trade volume
  orderVol: number;          // '000 shares, add-order volume
  hidden: number;            // trade count against hidden orders
  tradesForHidden: number;   // trade count from exchanges that report hidden
  hiddenVol: number;         // '000 shares
  tradeVolForHidden: number; // '000 shares
  cancels: number;
  litTrades: number;
  oddLots: number;
  tradesForOddLots: number;
  oddLotVol: number;         // '000 shares
  tradeVolForOddLots: number; // '000 shares
  rt: string;                 // our capture date
}

// ── Minimal single-member ZIP reader (stored + deflate), CSV variant ────────
// Independent of secFtd.ts's .txt-targeted reader (build-don't-buy applies
// to internal dependencies too — this is a ~40-line function, not worth a
// cross-territory shared-module refactor for one caller).

export function unzipSingleCsv(buf: Buffer): string {
  let eocd = -1;
  const scanFrom = Math.max(0, buf.length - 65557);
  for (let i = buf.length - 22; i >= scanFrom; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("zip: EOCD not found");
  const entries = buf.readUInt16LE(eocd + 10);
  let off = buf.readUInt32LE(eocd + 16);
  for (let e = 0; e < entries; e++) {
    if (buf.readUInt32LE(off) !== 0x02014b50) throw new Error("zip: bad central directory entry");
    const method = buf.readUInt16LE(off + 10);
    const csize = buf.readUInt32LE(off + 20);
    const fnLen = buf.readUInt16LE(off + 28);
    const extraLen = buf.readUInt16LE(off + 30);
    const commentLen = buf.readUInt16LE(off + 32);
    const localOff = buf.readUInt32LE(off + 42);
    const name = buf.toString("utf8", off + 46, off + 46 + fnLen);
    if (name.toLowerCase().endsWith(".csv")) {
      if (buf.readUInt32LE(localOff) !== 0x04034b50) throw new Error("zip: bad local header");
      const lfnLen = buf.readUInt16LE(localOff + 26);
      const lextraLen = buf.readUInt16LE(localOff + 28);
      const dataStart = localOff + 30 + lfnLen + lextraLen;
      const data = buf.subarray(dataStart, dataStart + csize);
      if (method === 0) return data.toString("utf8");
      if (method === 8) return zlib.inflateRawSync(data).toString("utf8");
      throw new Error(`zip: unsupported compression method ${method}`);
    }
    off += 46 + fnLen + extraLen + commentLen;
  }
  throw new Error("zip: no .csv member");
}

// ── Parser (malformed-row-rate guarded — no official checksum exists) ──────

const HEADER = "Date,Security,Ticker,McapRank,TurnRank,VolatilityRank,PriceRank,LitVol('000),OrderVol('000),Hidden,TradesForHidden,HiddenVol('000),TradeVolForHidden('000),Cancels,LitTrades,OddLots,TradesForOddLots,OddLotVol('000),TradeVolForOddLots('000)";
const MAX_MALFORMED_RATE = 0.01; // >1% bad rows = treat as a corrupted/truncated download

function num(s: string): number {
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : 0;
}
function rank(s: string): number | null {
  if (s === "") return null;
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : null;
}

/** Parse one quarter's CSV. Returns [] when the header is wrong or the
 *  malformed-row rate exceeds the guard (refusing, not silently degrading). */
export function parseMidas(text: string, rt: string): MidasRow[] {
  if (!text) return [];
  const lines = text.split(/\r?\n/).filter((l) => l.length);
  if (!lines.length || lines[0].trim() !== HEADER) {
    console.error(`[datacore] secmidas: unexpected header: ${(lines[0] || "").slice(0, 100)}`);
    return [];
  }
  const body = lines.slice(1);
  const out: MidasRow[] = [];
  let malformed = 0;
  for (const line of body) {
    const p = line.split(",");
    if (p.length !== 19 || (p[1] !== "Stock" && p[1] !== "ETF") || !/^\d{8}$/.test(p[0])) {
      malformed++;
      continue;
    }
    out.push({
      date: `${p[0].slice(0, 4)}-${p[0].slice(4, 6)}-${p[0].slice(6, 8)}`,
      kind: p[1] as "Stock" | "ETF",
      ticker: p[2],
      mcapRank: rank(p[3]),
      turnRank: rank(p[4]),
      volatilityRank: rank(p[5]),
      priceRank: rank(p[6]),
      litVol: num(p[7]),
      orderVol: num(p[8]),
      hidden: num(p[9]),
      tradesForHidden: num(p[10]),
      hiddenVol: num(p[11]),
      tradeVolForHidden: num(p[12]),
      cancels: num(p[13]),
      litTrades: num(p[14]),
      oddLots: num(p[15]),
      tradesForOddLots: num(p[16]),
      oddLotVol: num(p[17]),
      tradeVolForOddLots: num(p[18]),
      rt,
    });
  }
  if (body.length > 0 && malformed / body.length > MAX_MALFORMED_RATE) {
    console.error(`[datacore] secmidas: ${malformed}/${body.length} malformed rows (>${MAX_MALFORMED_RATE * 100}%) — refusing file`);
    return [];
  }
  return out;
}

// ── Periods (quarterly publication units) ───────────────────────────────────

export const MIDAS_LOOKBACK_QUARTERS = 6;

/** Newest-first candidate "{year}q{quarter}" strings, current quarter first.
 *  The newest 1-2 typically 404 until MIDAS's own multi-quarter publish lag
 *  clears — an expected honest no-op, not an error. */
export function candidateQuarters(nowMs: number, count = MIDAS_LOOKBACK_QUARTERS): string[] {
  const out: string[] = [];
  const d = new Date(nowMs);
  let y = d.getUTCFullYear();
  let q = Math.floor(d.getUTCMonth() / 3) + 1;
  while (out.length < count) {
    out.push(`${y}q${q}`);
    q--;
    if (q === 0) { q = 4; y--; }
  }
  return out;
}

function parseQuarterKey(period: string): { year: number; quarter: number } {
  const m = period.match(/^(\d{4})q([1-4])$/);
  if (!m) throw new Error(`bad period ${period}`);
  return { year: parseInt(m[1], 10), quarter: parseInt(m[2], 10) };
}

// ── Fetch ─────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; arrayBuffer(): Promise<ArrayBuffer> }>;

/** Fetch one quarter's rows. [] = not published yet (honest no-op);
 *  null = transport error (retry next poll). */
export async function fetchMidasPeriod(
  period: string,
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
): Promise<MidasRow[] | null> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const { year, quarter } = parseQuarterKey(period);
  const url = midasUrl(year, quarter);
  try {
    const r = await fetchImpl(url, { headers: SEC_UA, signal: AbortSignal.timeout(120000) as any });
    if (r.status === 404) return [];
    if (!r.ok) {
      console.error(`[datacore] secmidas ${period} ${url} -> ${r.status}`);
      return null;
    }
    const buf = Buffer.from(await r.arrayBuffer());
    return parseMidas(unzipSingleCsv(buf), rt);
  } catch (e: any) {
    console.error(`[datacore] secmidas ${period}:`, e?.message || e);
    return null;
  }
}

// ── Archive (period-level dedup; one gz file per quarter) ──────────────────

const archivedQuarters = new Set<string>();
let seeded = false;

function midasDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "secmidas");
}

function seedSeen(baseDir?: string): void {
  if (seeded) return;
  try {
    for (const f of fs.readdirSync(midasDir(baseDir))) {
      const m = f.match(/^(\d{4}q[1-4])\.jsonl\.gz$/);
      if (m) archivedQuarters.add(m[1]);
    }
  } catch {}
  seeded = true;
}

export function isMidasPeriodArchived(period: string, baseDir?: string): boolean {
  seedSeen(baseDir);
  return archivedQuarters.has(period);
}

/** ~533k rows/quarter (q4_2025 measured) -> gz-on-write (~20MB). */
export function archiveMidasPeriod(period: string, rows: MidasRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  seedSeen(baseDir);
  if (archivedQuarters.has(period)) return 0;
  try {
    const dir = midasDir(baseDir);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${period}.jsonl.gz`),
      zlib.gzipSync(rows.map((r) => JSON.stringify(r)).join("\n") + "\n"));
    archivedQuarters.add(period);
    return rows.length;
  } catch (e: any) {
    console.error(`[datacore] secmidas archive ${period}:`, e?.message || e);
    return 0;
  }
}

/**
 * OUTAGE FIX (2026-07-15, site down — root cause): the sync archive above
 * builds `rows.map(JSON.stringify).join("\n")` as ONE ~189MB string (probe-
 * measured live) before gzipSync — with cons-string flattening and the
 * Buffer copy, boot heap blows the production 512MB cap the FIRST time a
 * not-yet-archived quarter publishes. The volume's dedup guard hid this for
 * weeks: every boot skipped the write until SEC published the next quarter,
 * then EVERY boot OOM'd = a Railway crash loop (502 "Application failed to
 * respond"). This variant writes the identical bytes through a gzip STREAM
 * in row batches with backpressure — peak memory is O(batch), not O(file).
 * refreshMidas awaits this; the sync variant stays for tests/back-compat.
 */
export async function archiveMidasPeriodStreamed(
  period: string,
  rows: MidasRow[],
  baseDir?: string,
  batchRows = 5000,
): Promise<number> {
  if (!rows.length) return 0;
  seedSeen(baseDir);
  if (archivedQuarters.has(period)) return 0;
  const dir = midasDir(baseDir);
  const fp = path.join(dir, `${period}.jsonl.gz`);
  try {
    fs.mkdirSync(dir, { recursive: true });
    const gz = zlib.createGzip();
    const out = fs.createWriteStream(fp);
    const done = new Promise<void>((resolve, reject) => {
      out.on("finish", resolve);
      out.on("error", reject);
      gz.on("error", reject);
    });
    gz.pipe(out);
    for (let i = 0; i < rows.length; i += batchRows) {
      const chunk = rows.slice(i, i + batchRows).map((r) => JSON.stringify(r)).join("\n") + "\n";
      if (!gz.write(chunk)) {
        await new Promise<void>((resolve) => gz.once("drain", resolve)); // backpressure
      }
    }
    gz.end();
    await done;
    archivedQuarters.add(period);
    return rows.length;
  } catch (e: any) {
    console.error(`[datacore] secmidas archive (streamed) ${period}:`, e?.message || e);
    try { fs.unlinkSync(fp); } catch {} // drop a partial file — never a corrupt archive
    return 0;
  }
}

export function readMidasPeriod(period: string, baseDir?: string): MidasRow[] {
  try {
    const text = zlib.gunzipSync(fs.readFileSync(path.join(midasDir(baseDir), `${period}.jsonl.gz`))).toString("utf8");
    const out: MidasRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  } catch { return []; }
}

export function _resetMidasForTests(): void {
  archivedQuarters.clear();
  seeded = false;
  cache = null;
  polling = false;
}

// ── Summary (poll-time only; small-cap colonization watchlist) ─────────────
// RAW, not a signal: sorts by the source's own published ratios (no model,
// no interpretation) on the newest date in the newest archived quarter,
// restricted to Stock rows (ETF ranks are on a different scale, never
// mixed in) in the bottom two McapRank deciles (smallest ~20% by SEC's own
// ranking) with enough activity for the ratios to be meaningful.

export const MIDAS_TOP_CAP = 20;
export const MIDAS_MIN_TRADES_FOR_HIDDEN = 20; // denominator floor for hiddenRatePct
export const MIDAS_SMALLCAP_MAX_RANK = 2;      // McapRank <= 2 of 10 (Stock scale only)

export interface MidasWatchRow {
  ticker: string;
  mcapRank: number;
  cancelToTrade: number | null;
  hiddenRatePct: number | null;
  oddLotRatePct: number | null;
}

export interface MidasSummary {
  period: string;
  kind_counts: { stock: number; etf: number };
  newest_date: string;
  rows: number;
  smallcap_watch: MidasWatchRow[];
  smallcap_max_rank: number;
  min_trades_for_hidden: number;
  top_cap: number;
}

export function summarizeMidas(period: string, rows: MidasRow[]): MidasSummary | null {
  if (!rows.length) return null;
  const dates = new Set(rows.map((r) => r.date));
  const newest = Array.from(dates).sort().pop()!;
  const stockCount = rows.reduce((a, r) => a + (r.kind === "Stock" ? 1 : 0), 0);
  const watch = rows
    .filter((r) => r.date === newest && r.kind === "Stock" && r.mcapRank != null && r.mcapRank <= MIDAS_SMALLCAP_MAX_RANK
      && r.tradesForHidden >= MIDAS_MIN_TRADES_FOR_HIDDEN)
    .map((r) => ({
      ticker: r.ticker,
      mcapRank: r.mcapRank as number,
      cancelToTrade: r.litTrades > 0 ? r.cancels / r.litTrades : null,
      hiddenRatePct: r.tradesForHidden > 0 ? (100 * r.hidden) / r.tradesForHidden : null,
      oddLotRatePct: r.tradesForOddLots > 0 ? (100 * r.oddLots) / r.tradesForOddLots : null,
    }))
    .filter((r) => r.cancelToTrade != null)
    .sort((a, b) => (b.cancelToTrade as number) - (a.cancelToTrade as number))
    .slice(0, MIDAS_TOP_CAP);
  return {
    period,
    kind_counts: { stock: stockCount, etf: rows.length - stockCount },
    newest_date: newest,
    rows: rows.length,
    smallcap_watch: watch,
    smallcap_max_rank: MIDAS_SMALLCAP_MAX_RANK,
    min_trades_for_hidden: MIDAS_MIN_TRADES_FOR_HIDDEN,
    top_cap: MIDAS_TOP_CAP,
  };
}

let cache: { at: number; summary: MidasSummary } | null = null;
let polling = false;

export function latestMidas() {
  return cache;
}

/** Fetches AND ARCHIVES AT MOST ONE new quarter per call (see MEMORY NOTE
 *  in the module docstring) — walks candidateQuarters newest-first, skips
 *  anything already archived (cheap), stops at the first genuinely new
 *  period whether it publishes rows or 404s honestly-empty. */
export async function refreshMidas(
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
  baseDir?: string,
): Promise<void> {
  const now = nowMs ?? Date.now();
  try {
    seedSeen(baseDir);
    const candidates = candidateQuarters(now);
    // Refresh the cache from the newest already-archived period first,
    // regardless of whether a new fetch happens below.
    const newestArchived = candidates.find((p) => archivedQuarters.has(p));
    if (newestArchived && cache?.summary.period !== newestArchived) {
      const s = summarizeMidas(newestArchived, readMidasPeriod(newestArchived, baseDir));
      if (s) cache = { at: Date.now(), summary: s };
    }
    // Probe candidates newest-first. 404s ("not published yet") and
    // transport errors are cheap — no large payload is downloaded — so
    // any number of them may be probed in one call, catching up through
    // MIDAS's multi-quarter publish lag immediately. The memory guard is
    // specifically: archive (parse + gzip a ~68MB CSV) AT MOST ONCE per
    // call, so stop the instant one period actually publishes rows.
    for (const period of candidates) {
      if (archivedQuarters.has(period)) continue;
      const rows = await fetchMidasPeriod(period, fetchImpl, now);
      if (rows === null || rows.length === 0) continue; // retry-later or not published — keep probing older
      await archiveMidasPeriodStreamed(period, rows, baseDir); // OUTAGE FIX: O(batch) memory, not a 189MB string
      const s = summarizeMidas(period, rows);
      if (s && (!cache || period >= cache.summary.period)) cache = { at: Date.now(), summary: s };
      return;
    }
  } catch (e: any) {
    console.error("[datacore] secmidas refresh:", e?.message || e);
  }
}

/** Quarterly-cadence source with a multi-quarter publish lag — a daily poll
 *  is already generous (12x cheaper than secFtd's 12h, appropriate for a
 *  slower-moving source). Eager boot. */
export function bootMidasPoll(intervalMs = 24 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshMidas().catch((e) => console.error("[datacore] secmidas boot:", e?.message || e));
  setInterval(() => { refreshMidas(); }, intervalMs).unref?.();
}
