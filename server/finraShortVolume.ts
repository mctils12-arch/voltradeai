/**
 * finraShortVolume.ts — FINRA daily consolidated short-sale volume
 * (BUILD ORDER 5 #1, filed + probed 2026-07-05).
 *
 * Source: cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt —
 * keyless pipe-delimited daily file (~12.2K symbols), published each
 * trading day; weekend/holiday URLs 403 (a valid "not published", not an
 * error). FINRA publishes these files for free use with attribution
 * ("FINRA Reg SHO daily short sale volume"). Format verified live
 * 2026-07-05: header Date|Symbol|ShortVolume|ShortExemptVolume|
 * TotalVolume|Market, fractional share counts, bare row-count trailer.
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * short-volume-ratio extremes and multi-day deltas precede reversals in
 * small caps; joins 13F clusters + Form 4 for a squeeze-candidate
 * screen. Nothing is claimed until ladder gate 2.
 *
 * INTEGRITY: the trailer row count must equal parsed data rows or the
 * file is refused (protects the archive from truncated downloads).
 * DEDUP is at DATE level — the file is atomic and final once published,
 * and per-row keys (12K/day x 40d seed) would waste ~50MB in the
 * RSS-capped process. If FINRA ever reposts a corrected file we keep
 * the first version; noted in the manifest.
 *
 * OTC/ORF FACILITY ADDED 2026-07-27 (scheduled-routine PRODUCT session):
 * CNMS ("Consolidated NMS") covers exchange-LISTED securities only —
 * FINRA's own file catalog (developer.finra.org, live-fetched this
 * session) confirms CNMS "combines exchange-listed securities reported
 * to TRFs and the ADF," and lists a SEPARATE file, FORFshvol{YYYYMMDD}.txt
 * ("ORF" = Over-the-Counter Reporting Facility), for non-exchange-listed
 * OTC equities. Root-caused this session: server/settlementStress.ts's
 * composite joins finrathreshold (server/finraQuery.ts's `thresholdList`
 * dataset — OTC-only by FINRA's own schema: "OTC Regulation SHO and Rule
 * 4320 Threshold Securities", confirmed via the live FINRA metadata
 * endpoint) against CNMS-only short volume — two DISJOINT populations by
 * regulatory design (OTC threshold names vs. NMS-exchange-listed short
 * volume), so the composite's 3-way join was structurally guaranteed to
 * find zero overlap forever, not "rare" or gate-2-blocked-on-time. Live-
 * verified: FORFshvol20260625.txt (same header/format, `Market` field
 * value "O") contains every one of that day's finrathreshold symbols
 * (CHLSY, DSFIY, KRKNF, ...) that CNMS was missing. `readOrfArchivedDay`/
 * `fetchOrfShortVolDay`/`archiveOrfShortVolDay` below are the fix: same
 * parser (format is identical), separate archive dir + separate boot
 * poll, so CNMS's existing consumers (squeeze-screen, symbol lookback,
 * market-wide trend) are untouched. Full trace in experiments.md.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface ShortVolRow {
  date: string;        // YYYY-MM-DD trade date
  symbol: string;
  short_vol: number;
  short_exempt_vol: number;
  total_vol: number;
  market: string;      // FINRA facility codes as published, e.g. "B,Q,N"
  rt: string;          // as-seen UTC date
}

const FILE_URL = (yyyymmdd: string) =>
  `https://cdn.finra.org/equity/regsho/daily/CNMSshvol${yyyymmdd}.txt`;

// ORF = FINRA's Over-the-Counter Reporting Facility file — the OTC/
// non-exchange-listed counterpart to CNMS (see the OTC/ORF module
// comment above). Same pipe-delimited schema; `parseShortVol` handles
// both — verified live against FORFshvol20260624/20260625.txt this
// session (header byte-identical to CNMS's).
const ORF_FILE_URL = (yyyymmdd: string) =>
  `https://cdn.finra.org/equity/regsho/daily/FORFshvol${yyyymmdd}.txt`;

const num = (v: string): number | null => {
  if (v == null || v === "") return null;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
};

/** Parse a CNMS file. Returns [] when the header is missing/unexpected or
 *  the trailer row count disagrees with parsed rows (truncation guard). */
export function parseShortVol(text: string, rt: string): ShortVolRow[] {
  if (!text) return [];
  const lines = text.split("\n").map((l) => l.trim()).filter((l) => l.length);
  if (lines.length < 2) return [];
  const header = lines[0].split("|").map((h) => h.trim());
  const idx: Record<string, number> = {};
  header.forEach((h, i) => { idx[h.toUpperCase()] = i; });
  if (idx.DATE == null || idx.SYMBOL == null || idx.SHORTVOLUME == null || idx.TOTALVOLUME == null) return [];
  const out: ShortVolRow[] = [];
  let trailer: number | null = null;
  for (const line of lines.slice(1)) {
    const parts = line.split("|");
    if (parts.length === 1) {                 // bare row-count trailer
      trailer = num(parts[0]);
      continue;
    }
    const rawDate = parts[idx.DATE];
    const symbol = parts[idx.SYMBOL];
    if (!symbol || !/^\d{8}$/.test(rawDate || "")) continue;
    const sv = num(parts[idx.SHORTVOLUME]);
    const tv = num(parts[idx.TOTALVOLUME]);
    if (sv == null || tv == null) continue;
    out.push({
      date: `${rawDate.slice(0, 4)}-${rawDate.slice(4, 6)}-${rawDate.slice(6, 8)}`,
      symbol,
      short_vol: sv,
      short_exempt_vol: idx.SHORTEXEMPTVOLUME != null ? (num(parts[idx.SHORTEXEMPTVOLUME]) ?? 0) : 0,
      total_vol: tv,
      market: idx.MARKET != null ? (parts[idx.MARKET] || "") : "",
      rt,
    });
  }
  if (trailer != null && trailer !== out.length) {
    console.error(`[datacore] finrashortvol trailer says ${trailer} rows, parsed ${out.length} — refusing truncated file`);
    return [];
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

/** Fetch one trade date's file. Returns [] on 403/404 (weekend/holiday —
 *  valid "not published") and null on transport errors (retry next poll). */
export async function fetchShortVolDay(
  yyyymmdd: string,
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
): Promise<ShortVolRow[] | null> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  try {
    const r = await fetchImpl(FILE_URL(yyyymmdd), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (r.status === 403 || r.status === 404) return []; // not a trading day / not yet published
    if (!r.ok) {
      console.error(`[datacore] finrashortvol ${yyyymmdd} -> ${r.status}`);
      return null;
    }
    return parseShortVol(await r.text(), rt);
  } catch (e: any) {
    console.error(`[datacore] finrashortvol ${yyyymmdd}:`, e?.message || e);
    return null;
  }
}

/** ORF (OTC facility) counterpart to fetchShortVolDay — same contract
 *  (403/404 = valid non-trading-day, null = transport error). */
export async function fetchOrfShortVolDay(
  yyyymmdd: string,
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
): Promise<ShortVolRow[] | null> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  try {
    const r = await fetchImpl(ORF_FILE_URL(yyyymmdd), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (r.status === 403 || r.status === 404) return [];
    if (!r.ok) {
      console.error(`[datacore] finrashortvolotc ${yyyymmdd} -> ${r.status}`);
      return null;
    }
    return parseShortVol(await r.text(), rt);
  } catch (e: any) {
    console.error(`[datacore] finrashortvolotc ${yyyymmdd}:`, e?.message || e);
    return null;
  }
}

// ── Archive (date-level dedup; one JSONL day-file per TRADE date) ───────────

const archivedDates = new Set<string>();
let seeded = false;
const orfArchivedDates = new Set<string>();
let orfSeeded = false;

function svDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "finrashortvol");
}

function otcDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "finrashortvolotc");
}

function seedSeenInto(set: Set<string>, dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/);
      if (m) set.add(m[1]);
    }
  } catch {}
}

function seedSeen(dir: string): void {
  seedSeenInto(archivedDates, dir);
}

/** Archive one day's rows under the TRADE date. Returns rows written
 *  (0 if the date is already archived or rows are empty). */
export function archiveShortVolDay(rows: ShortVolRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  const dir = svDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const date = rows[0].date;
  if (archivedDates.has(date)) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${date}.jsonl`),
                     rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    archivedDates.add(date);
    return rows.length;
  } catch (e: any) {
    console.error("[datacore] finrashortvol archive:", e?.message || e);
    return 0;
  }
}

/** ORF counterpart to archiveShortVolDay — separate dir/dedup set so a
 *  symbol appearing in neither, either, or (never in practice — the two
 *  facilities cover disjoint listed/OTC populations) both files never
 *  collides with CNMS's own archive. */
export function archiveOrfShortVolDay(rows: ShortVolRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  const dir = otcDir(baseDir);
  if (!orfSeeded) { seedSeenInto(orfArchivedDates, dir); orfSeeded = true; }
  const date = rows[0].date;
  if (orfArchivedDates.has(date)) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${date}.jsonl`),
                     rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    orfArchivedDates.add(date);
    return rows.length;
  } catch (e: any) {
    console.error("[datacore] finrashortvolotc archive:", e?.message || e);
    return 0;
  }
}

/** Read one ORF-archived day back (plain or gz). Poller-context only,
 *  same shape as readArchivedDay. */
export function readOrfArchivedDay(iso: string, baseDir?: string): ShortVolRow[] {
  const dir = otcDir(baseDir);
  for (const fp of [path.join(dir, `${iso}.jsonl`), path.join(dir, `${iso}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: ShortVolRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  }
  return [];
}

/** One shallow-lookback refresh pass for the ORF/OTC facility — mirrors
 *  refreshShortVol's newest-first loop, deliberately WITHOUT a deep-
 *  backfill path (that stays a follow-up per the 2026-07-05 CNMS
 *  emergency-off volume-fill incident this module's own history
 *  documents — ship thin first, backfill is its own future PR/decision,
 *  same precedent as every other census stream in this codebase). */
export async function refreshOrfShortVol(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                         lookbackDays = 10, baseDir?: string): Promise<void> {
  const now = nowMs ?? Date.now();
  try {
    if (!orfSeeded) { seedSeenInto(orfArchivedDates, otcDir(baseDir)); orfSeeded = true; }
    for (let i = 0; i < lookbackDays; i++) {
      const iso = new Date(now - i * 86400_000).toISOString().slice(0, 10);
      if (orfArchivedDates.has(iso)) continue;
      const yyyymmdd = iso.replace(/-/g, "");
      const rows = await fetchOrfShortVolDay(yyyymmdd, fetchImpl, now);
      if (rows === null || rows.length === 0) continue;
      archiveOrfShortVolDay(rows, baseDir);
      if (now - Date.parse(iso) >= 2 * 86400_000) {
        try {
          const fp = path.join(otcDir(baseDir), `${iso}.jsonl`);
          if (fs.existsSync(fp)) {
            fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
            fs.unlinkSync(fp);
          }
        } catch {}
      }
      await new Promise((res) => setTimeout(res, 300)); // polite spacing
    }
  } catch (e: any) {
    console.error("[datacore] finrashortvolotc refresh:", e?.message || e);
  }
}

let orfPolling = false;

/** Same 6h cadence as bootShortVolPoll (files publish evenings ET). */
export function bootOrfShortVolPoll(intervalMs = 6 * 60 * 60_000): void {
  if (orfPolling) return;
  orfPolling = true;
  refreshOrfShortVol().catch((e) => console.error("[datacore] finrashortvolotc boot:", e?.message || e));
  setInterval(() => { refreshOrfShortVol(); }, intervalMs).unref?.();
}

function gzTargets(dir: string, now: number): string[] {
  try {
    return fs.readdirSync(dir)
      .filter((f) => f.endsWith(".jsonl") && now - Date.parse(f.slice(0, 10)) >= 2 * 86400_000)
      .map((f) => path.join(dir, f));
  } catch { return []; }
}

function gzOne(fp: string): void {
  fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
  fs.unlinkSync(fp);
}

export function gzipOldShortVolDays(baseDir?: string, nowMs?: number): number {
  let n = 0;
  for (const fp of gzTargets(svDir(baseDir), nowMs ?? Date.now())) {
    try { gzOne(fp); n++; } catch {}
  }
  return n;
}

/** Poller-path variant: yields the event loop between files — after a
 *  deep backfill ~500 files gzip in one sweep, and a fully synchronous
 *  loop would block the loop for tens of seconds (the shadowstats
 *  lesson, same class). */
export async function gzipOldShortVolDaysAsync(baseDir?: string, nowMs?: number): Promise<number> {
  let n = 0;
  for (const fp of gzTargets(svDir(baseDir), nowMs ?? Date.now())) {
    try { gzOne(fp); n++; } catch {}
    await new Promise((res) => setImmediate(res));
  }
  return n;
}

// ── Summary cache (computed at poll time — NEVER on the request path) ───────

export interface ShortVolSummary {
  date: string;
  symbols: number;
  agg_short_ratio: number | null;      // sum(short)/sum(total) across all rows
  top_ratio: Array<{ symbol: string; short_ratio: number; total_vol: number; market: string }>;
  floor_total_vol: number;             // stated selection floor for top_ratio
  top_cap: number;                     // stated cap
}

export const TOP_CAP = 30;
export const FLOOR_TOTAL_VOL = 500_000;

export function summarize(rows: ShortVolRow[]): ShortVolSummary | null {
  if (!rows.length) return null;
  let sSum = 0, tSum = 0;
  const eligible: Array<{ symbol: string; short_ratio: number; total_vol: number; market: string }> = [];
  for (const r of rows) {
    sSum += r.short_vol;
    tSum += r.total_vol;
    if (r.total_vol >= FLOOR_TOTAL_VOL && r.total_vol > 0) {
      eligible.push({
        symbol: r.symbol,
        short_ratio: Math.round((r.short_vol / r.total_vol) * 10000) / 10000,
        total_vol: Math.round(r.total_vol),
        market: r.market,
      });
    }
  }
  eligible.sort((a, b) => b.short_ratio - a.short_ratio);
  return {
    date: rows[0].date,
    symbols: rows.length,
    agg_short_ratio: tSum > 0 ? Math.round((sSum / tSum) * 10000) / 10000 : null,
    top_ratio: eligible.slice(0, TOP_CAP),
    floor_total_vol: FLOOR_TOTAL_VOL,
    top_cap: TOP_CAP,
  };
}

let cache: { at: number; summary: ShortVolSummary } | null = null;
let polling = false;

export function latestShortVol() {
  return cache;
}

// ── Market-wide summary-history trend (tiny append-only log — NOT the
// per-day 12K-row archive) ──────────────────────────────────────────────
//
// The per-day archive is ~1.5MB uncompressed; a "market-wide ratio over
// time" chart only needs one float per day. Recomputing that from the
// full archive on every request would repeat the "materialize the whole
// archive on a hot path" mistake that caused the 2026-07-05 OOM incident
// (see experiments.md v1.0.143) — so the daily summarize() already
// computed in refreshShortVol appends one small line here instead, and
// the history route reads only this file, never the day archives, for
// the market-wide trend.

const summaryHistorySeen = new Set<string>();
let summaryHistorySeeded = false;

function summaryHistoryPath(baseDir?: string): string {
  return path.join(svDir(baseDir), "_summary_history.jsonl");
}

function seedSummaryHistorySeen(baseDir?: string): void {
  try {
    const text = fs.readFileSync(summaryHistoryPath(baseDir), "utf8");
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { summaryHistorySeen.add(JSON.parse(line).date); } catch {}
    }
  } catch {}
}

/** Append one day's {date, symbols, agg_short_ratio} — a no-op if that
 *  date is already recorded (idempotent across repeated polls/restarts). */
export function appendSummaryHistoryEntry(summary: ShortVolSummary, baseDir?: string): void {
  if (!summaryHistorySeeded) { seedSummaryHistorySeen(baseDir); summaryHistorySeeded = true; }
  if (summaryHistorySeen.has(summary.date)) return;
  try {
    fs.mkdirSync(svDir(baseDir), { recursive: true });
    fs.appendFileSync(summaryHistoryPath(baseDir), JSON.stringify({
      date: summary.date, symbols: summary.symbols, agg_short_ratio: summary.agg_short_ratio,
    }) + "\n");
    summaryHistorySeen.add(summary.date);
  } catch (e: any) {
    console.error("[datacore] finrashortvol summary history append:", e?.message || e);
  }
}

export interface ShortVolTrendPoint { date: string; symbols: number; agg_short_ratio: number | null; }

/** Last `days` entries (ascending by date) from the small trend log —
 *  bounded file read, no day-archive access. */
export function readSummaryHistory(days: number, baseDir?: string): ShortVolTrendPoint[] {
  let text: string;
  try {
    text = fs.readFileSync(summaryHistoryPath(baseDir), "utf8");
  } catch { return []; }
  const rows: ShortVolTrendPoint[] = [];
  for (const line of text.split("\n")) {
    if (!line) continue;
    try { rows.push(JSON.parse(line)); } catch {}
  }
  rows.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return rows.slice(-days);
}

// ── Per-symbol lookback (reads the day-archive directly — the deep
// backfill already holds 2+ years; bounded to <=90 trading days per
// request, same cap convention as insider/earnings history routes, and
// one day is read+discarded at a time so peak memory is one day's rows,
// never the full lookback window at once) ──────────────────────────────

/** Archived trading dates, newest first (jsonl or jsonl.gz; the small
 *  summary-history/backfill-marker files never match this pattern). */
export function listArchivedDates(baseDir?: string, limit = 90): string[] {
  let files: string[];
  try { files = fs.readdirSync(svDir(baseDir)); } catch { return []; }
  return files
    .map((f) => f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/))
    .filter((m): m is RegExpMatchArray => !!m)
    .map((m) => m[1])
    .sort()
    .reverse()
    .slice(0, limit);
}

export interface ShortVolSymbolPoint {
  date: string; short_vol: number; total_vol: number; short_ratio: number | null; market: string;
}

/** One symbol's short-volume ratio across its last `days` archived
 *  trading dates, ascending by date. A date with no row for the symbol
 *  (delisted/not traded that day) is honestly omitted, never zero-filled. */
export function lookupSymbolHistory(symbol: string, days: number, baseDir?: string): ShortVolSymbolPoint[] {
  const sym = symbol.trim().toUpperCase();
  const dates = listArchivedDates(baseDir, days);
  const out: ShortVolSymbolPoint[] = [];
  for (const iso of dates) {
    const rows = readArchivedDay(iso, baseDir);
    const r = rows.find((x) => x.symbol.toUpperCase() === sym);
    if (r) {
      out.push({
        date: r.date, short_vol: r.short_vol, total_vol: r.total_vol,
        short_ratio: r.total_vol > 0 ? Math.round((r.short_vol / r.total_vol) * 10000) / 10000 : null,
        market: r.market,
      });
    }
  }
  out.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return out;
}

/** Read one archived day back (plain or gz). Poller-context only —
 *  a single ~1.5MB file, same budget as the other streams' seedSeen. */
export function readArchivedDay(iso: string, baseDir?: string): ShortVolRow[] {
  const dir = svDir(baseDir);
  for (const fp of [path.join(dir, `${iso}.jsonl`), path.join(dir, `${iso}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: ShortVolRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  }
  return [];
}

/** Try the last `lookbackDays` calendar days newest-first; archive any
 *  unarchived trading day; cache the newest day's summary. A restart with
 *  the newest day already on disk rebuilds the cache FROM the archive
 *  instead of serving warming_up until the next publish. */
export async function refreshShortVol(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                      lookbackDays = 7, baseDir?: string): Promise<void> {
  const now = nowMs ?? Date.now();
  try {
    if (!seeded) { seedSeen(svDir(baseDir)); seeded = true; }
    let newestSummarized = false;
    for (let i = 0; i < lookbackDays; i++) {
      const d = new Date(now - i * 86400_000);
      const iso = d.toISOString().slice(0, 10);
      const yyyymmdd = iso.replace(/-/g, "");
      if (archivedDates.has(iso)) {
        if (!newestSummarized) {
          if (cache?.summary.date !== iso) {
            const s = summarize(readArchivedDay(iso, baseDir));
            if (s) { cache = { at: Date.now(), summary: s }; appendSummaryHistoryEntry(s, baseDir); }
          }
          newestSummarized = true; // newest archived day handled either way
        }
        continue;
      }
      const rows = await fetchShortVolDay(yyyymmdd, fetchImpl, now);
      if (rows === null || rows.length === 0) continue; // transport error or non-trading day
      archiveShortVolDay(rows, baseDir);
      // gz-eligible days compress IMMEDIATELY — a deep pass must never
      // hold hundreds of days uncompressed on the volume while it runs
      if (now - Date.parse(iso) >= 2 * 86400_000) {
        try {
          const fp = path.join(svDir(baseDir), `${iso}.jsonl`);
          if (fs.existsSync(fp)) {
            fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
            fs.unlinkSync(fp);
          }
        } catch {}
      }
      if (!newestSummarized) {
        const s = summarize(rows);
        if (s) { cache = { at: Date.now(), summary: s }; appendSummaryHistoryEntry(s, baseDir); newestSummarized = true; }
      }
      await new Promise((res) => setTimeout(res, 300)); // polite spacing
    }
    await gzipOldShortVolDaysAsync(baseDir, now);
  } catch (e: any) {
    console.error("[datacore] finrashortvol refresh:", e?.message || e);
  }
}

/** Dated CNMS files persist for years — history someone else recorded
 *  is history we can still capture (accumulation substitutes for
 *  purchase, EDGE DOCTRINE). Backfill target ~2 years; ~500 trading-day
 *  files x ~150KB gz on the volume. Gate-2 (short-ratio extremes vs
 *  forward returns) needs exactly this depth. */
export const DEEP_BACKFILL_DAYS = 750; // calendar days (~500 trading days)

export function countArchivedDays(baseDir?: string): number {
  try {
    return fs.readdirSync(svDir(baseDir)).filter((f) => /^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)).length;
  } catch { return 0; }
}

const doneMarker = (baseDir?: string) => path.join(svDir(baseDir), "backfill_done.json");

/** EMERGENCY DEFAULT-OFF (2026-07-05, ~30 min after v1.0.138 deployed):
 *  prod entered a ~60s crash-restart loop immediately after the deep
 *  backfill shipped. Leading theory: a full pass writes ~750MB of
 *  UN-GZIPPED day-files (gzip only ran after a complete pass, and no
 *  pass ever completed) — plausibly filling the Railway volume, which
 *  makes the bot's periodic state writes crash Node. Until volume
 *  capacity is verified, the deep backfill requires the explicit
 *  FINRA_DEEP_BACKFILL=1 env opt-in, and when it does run it gzips
 *  EACH day as it lands (~75MB total, 10x smaller, no full-pass gap). */
export function deepBackfillEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.FINRA_DEEP_BACKFILL === "1";
}

export async function deepBackfillIfSparse(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                           baseDir?: string, env: NodeJS.ProcessEnv = process.env): Promise<void> {
  if (!deepBackfillEnabled(env)) return;
  if (fs.existsSync(doneMarker(baseDir))) return;
  const have = countArchivedDays(baseDir);
  console.log(`[datacore] finrashortvol deep backfill: archive has ${have} day-files — fetching ~${DEEP_BACKFILL_DAYS} calendar days`);
  await refreshShortVol(fetchImpl, nowMs, DEEP_BACKFILL_DAYS, baseDir);
  try {
    fs.writeFileSync(doneMarker(baseDir), JSON.stringify({
      done_rt: new Date().toISOString(),
      day_files: countArchivedDays(baseDir),
      calendar_days: DEEP_BACKFILL_DAYS,
    }));
  } catch {}
  console.log(`[datacore] finrashortvol deep backfill done: ${countArchivedDays(baseDir)} day-files`);
}

/** Files publish evenings after the trading day — poll every 6h so a
 *  publish is picked up same-day without hammering the CDN. Eager boot
 *  per KNOWN BROKEN #9; sparse archives trigger the one-shot deep
 *  backfill after the current data is up. */
export function bootShortVolPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  // .catch: an unhandled rejection here would crash the process
  refreshShortVol().then(() => deepBackfillIfSparse())
    .catch((e) => console.error("[datacore] finrashortvol boot:", e?.message || e));
  setInterval(() => { refreshShortVol(); }, intervalMs).unref?.();
}
