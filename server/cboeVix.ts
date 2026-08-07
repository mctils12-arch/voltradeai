/**
 * cboeVix.ts — Cboe volatility-index term structure (VIX1D/VIX9D/VIX/
 * VIX3M/VIX6M) + VVIX, archived daily (EDGE DOCTRINE #1/#3 build-first
 * pipeline, scheduled-routine session 2026-08-07).
 *
 * WHY THIS AND NOT THE ORIGINALLY-CATALOGUED "cboe_daily_stats" TARGET:
 * the census (research/data_census.md #6) named CBOE daily put/call
 * ratios + a "VX curve" as the candidate. Live-probed this session:
 * (a) cdn.cboe.com/resources/options/volume_and_call_put_ratios/*.csv
 * (equitypc/indexpcarchive/pcratioarchive/totalpc) all return HTTP 200
 * but every one carries `Last-Modified: Fri, 30 Oct 2020` — a FROZEN
 * legacy archive, not a live feed (verified via `curl -I`); a nightly
 * poller would poll a dead file forever. (b) the live per-expiry VX
 * futures settlement page is not a simple curve CSV — it resolves to
 * `soq_vxs_<date>.csv`, which turned out to be the SPXW variance-strip
 * inputs behind the VIX calculation (a wide options-surface file), not
 * a VX contract curve — too large a scope shift for this session and a
 * different, harder root. NEITHER is what got built.
 * WHAT DID: probing cdn.cboe.com/api/global/us_indices/daily_prices/
 * <TICKER>_History.csv for the VIX family found six tickers (VIX1D,
 * VIX9D, VIX, VIX3M, VIX6M, VVIX) that are keyless, HTTP 200, and
 * genuinely live (Last-Modified within the same session's UTC day).
 * GATE 1 (DATA) cross-check this session: CBOE's own VIX_History.csv
 * close matched FRED's independently-published VIXCLS series exactly
 * for 2026-08-03/04/05 (15.86/16.50/15.81 both sources, both dates).
 * This module archives the full term structure + two derived ratios
 * daily. RAW / regime-feature framing only — no trading logic reads
 * this yet; any predictive claim about the ratios stays gate-locked
 * pending SIGNAL-layer validation (ROOT VALIDATION LADDER, CLAUDE.md).
 * Cross-system note: bot_engine.py's regime classifier today derives
 * its own vol proxy from a VXX ETF ratio (KNOWN BROKEN #20) — this
 * archive is a candidate cleaner regime input for a FUTURE gated
 * session, not wired in this one.
 *
 * License: Cboe informational use (page footer text, verbatim);
 * redistribution of the raw series needs Cboe permission — bot-internal
 * archive + gated derived signals OK, same posture as occVolume.ts.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices";

// VIX1D inception (first published trading day) — the window where all
// six series overlap, so every archived record is fully populated
// (never null-padded for a tenor that simply didn't exist yet).
export const CBOE_VIX_FLOOR_DATE = "2022-05-16";

export const OHLC_TENORS = ["VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M"] as const;
export type OhlcTenor = typeof OHLC_TENORS[number];
export const CLOSE_ONLY_TENORS = ["VVIX"] as const;

export interface CboeVixDay {
  date: string;               // YYYY-MM-DD
  vix1d: number | null;
  vix9d: number | null;
  vix: number | null;
  vix3m: number | null;
  vix6m: number | null;
  vvix: number | null;
  vix_vix3m_ratio: number | null;  // vix/vix3m: <1 normal contango, >=1 backwardation (stress)
  vix9d_vix_ratio: number | null;  // vix9d/vix: front-end stress relative to the 30d level
}

/** "08/06/2026" -> "2026-08-06"; null on garbage. */
export function normalizeMdY(v: string): string | null {
  const m = v.trim().match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (!m) return null;
  return `${m[3]}-${m[1]}-${m[2]}`;
}

/** DATE,OPEN,HIGH,LOW,CLOSE -> Map<date, close> (only CLOSE is used downstream). */
export function parseOhlcCsv(csv: string): Map<string, number> {
  const out = new Map<string, number>();
  const lines = csv.split(/\r?\n/);
  if (!lines.length || !lines[0].toUpperCase().startsWith("DATE,")) return out;
  for (const line of lines.slice(1)) {
    if (!line.trim()) continue;
    const c = line.split(",");
    if (c.length < 5) continue;
    const date = normalizeMdY(c[0]);
    const close = parseFloat(c[4]);
    if (!date || !Number.isFinite(close)) continue;
    out.set(date, close);
  }
  return out;
}

/** DATE,VVIX -> Map<date, value>. */
export function parseCloseOnlyCsv(csv: string): Map<string, number> {
  const out = new Map<string, number>();
  const lines = csv.split(/\r?\n/);
  if (!lines.length || !lines[0].toUpperCase().startsWith("DATE,")) return out;
  for (const line of lines.slice(1)) {
    if (!line.trim()) continue;
    const c = line.split(",");
    if (c.length < 2) continue;
    const date = normalizeMdY(c[0]);
    const v = parseFloat(c[1]);
    if (!date || !Number.isFinite(v)) continue;
    out.set(date, v);
  }
  return out;
}

/** Merges the six per-tenor series into per-date records, floored at
 *  CBOE_VIX_FLOOR_DATE so every record is fully populated (no nulls). */
export function mergeTenors(
  series: Record<OhlcTenor | "VVIX", Map<string, number>>,
  floorDate = CBOE_VIX_FLOOR_DATE,
): CboeVixDay[] {
  const dates = new Set<string>();
  Object.values(series).forEach((m) => Array.from(m.keys()).forEach((d) => dates.add(d)));
  const out: CboeVixDay[] = [];
  for (const date of Array.from(dates).sort()) {
    if (date < floorDate) continue;
    const vix1d = series.VIX1D.get(date) ?? null;
    const vix9d = series.VIX9D.get(date) ?? null;
    const vix = series.VIX.get(date) ?? null;
    const vix3m = series.VIX3M.get(date) ?? null;
    const vix6m = series.VIX6M.get(date) ?? null;
    const vvix = series.VVIX.get(date) ?? null;
    if (vix1d == null || vix9d == null || vix == null || vix3m == null || vix6m == null || vvix == null) {
      continue; // partial day (a publish gap on one tenor) — skip rather than fabricate
    }
    out.push({
      date, vix1d, vix9d, vix, vix3m, vix6m, vvix,
      vix_vix3m_ratio: Math.round((vix / vix3m) * 10000) / 10000,
      vix9d_vix_ratio: Math.round((vix9d / vix) * 10000) / 10000,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export function seriesUrl(ticker: string): string {
  return `${BASE_URL}/${ticker}_History.csv`;
}

async function fetchCsv(ticker: string, fetchImpl: FetchFn): Promise<string | null> {
  try {
    const r = await fetchImpl(seriesUrl(ticker), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] cboevix ${ticker} -> HTTP ${r.status}`);
      return null;
    }
    return await r.text();
  } catch (e: any) {
    console.error(`[datacore] cboevix ${ticker}:`, e?.message || e);
    return null;
  }
}

export async function fetchAllTenors(fetchImpl: FetchFn = fetch as any):
    Promise<Record<OhlcTenor | "VVIX", Map<string, number>>> {
  const out: any = {};
  for (const t of OHLC_TENORS) {
    const csv = await fetchCsv(t, fetchImpl);
    out[t] = csv ? parseOhlcCsv(csv) : new Map();
  }
  for (const t of CLOSE_ONLY_TENORS) {
    const csv = await fetchCsv(t, fetchImpl);
    out[t] = csv ? parseCloseOnlyCsv(csv) : new Map();
  }
  return out;
}

// ── Archive (one JSONL file per YEAR — payload is ~200 bytes/day, so a
//    per-day file scheme would be thousands of tiny files for no benefit;
//    day-level dedup scanning the year file keeps re-polls cheap) ──────────

const archivedDates = new Set<string>();
let seeded = false;

function vixDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "cboevix");
}

function yearFile(dir: string, date: string, gz: boolean): string {
  return path.join(dir, `${date.slice(0, 4)}.jsonl${gz ? ".gz" : ""}`);
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4})\.jsonl(\.gz)?$/);
      if (!m) continue;
      const fp = path.join(dir, f);
      const text = m[2] ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8") : fs.readFileSync(fp, "utf8");
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { archivedDates.add(JSON.parse(line).date); } catch {}
      }
    }
  } catch {}
}

export function isDateArchived(date: string, baseDir?: string): boolean {
  if (!seeded) { seedSeen(vixDir(baseDir)); seeded = true; }
  return archivedDates.has(date);
}

export function _resetCboeVixForTests(): void {
  archivedDates.clear();
  seeded = false;
  cache = null;
}

/** Appends new days to their year file (plain JSONL — current year stays
 *  uncompressed for easy tailing; prior years get gzipped by the same
 *  rollover sweep other datacore modules use). Returns count archived. */
export function archiveCboeVixDays(days: CboeVixDay[], baseDir?: string, nowMs?: number): number {
  if (!days.length) return 0;
  const dir = vixDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = days.filter((d) => !archivedDates.has(d.date));
  if (!fresh.length) return 0;
  const currentYear = String(new Date(nowMs ?? Date.now()).getUTCFullYear());
  try {
    fs.mkdirSync(dir, { recursive: true });
    const byYear = new Map<string, CboeVixDay[]>();
    fresh.forEach((d) => {
      const y = d.date.slice(0, 4);
      const arr = byYear.get(y) || [];
      arr.push(d);
      byYear.set(y, arr);
    });
    byYear.forEach((rows, year) => {
      // a past year may already be gzipped (rare — only via a late backfill
      // day for a year that already rolled over); reopen it plain so the
      // append lands, then let the next gzipPastYearFiles sweep re-gzip it
      const gzPath = yearFile(dir, `${year}-01-01`, true);
      const plainPath = yearFile(dir, `${year}-01-01`, false);
      if (year !== currentYear && fs.existsSync(gzPath) && !fs.existsSync(plainPath)) {
        fs.writeFileSync(plainPath, zlib.gunzipSync(fs.readFileSync(gzPath)));
        fs.unlinkSync(gzPath);
      }
      fs.appendFileSync(plainPath, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    });
    fresh.forEach((d) => archivedDates.add(d.date));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] cboevix archive:", e?.message || e);
    return 0;
  }
}

/** Gzips prior-year files once the calendar year has rolled over (mirrors
 *  wikiAttention.ts's gzipOldAttentionDays; this stream's day-files are
 *  yearly, so the cutoff is "not the current year" rather than an age). */
export function gzipPastYearFiles(baseDir?: string, nowMs?: number): number {
  const dir = vixDir(baseDir);
  const currentYear = String(new Date(nowMs ?? Date.now()).getUTCFullYear());
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4})\.jsonl$/);
      if (!m || m[1] === currentYear) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; latest: CboeVixDay; recent: CboeVixDay[] } | null = null;
let polling = false;

export function latestCboeVix() {
  return cache;
}

/** Fetches all six series, archives every not-yet-seen fully-populated
 *  day, and caches the latest day + a short recent window for a simple
 *  term-structure sparkline consumer. */
export async function refreshCboeVix(fetchImpl: FetchFn = fetch as any, baseDir?: string, nowMs?: number): Promise<void> {
  try {
    const series = await fetchAllTenors(fetchImpl);
    const days = mergeTenors(series);
    if (!days.length) return;
    archiveCboeVixDays(days, baseDir, nowMs);
    const latest = days[days.length - 1];
    const recent = days.slice(-30);
    if (!cache || latest.date >= cache.latest.date) {
      cache = { at: Date.now(), latest, recent };
    }
    gzipPastYearFiles(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] cboevix refresh:", e?.message || e);
  }
}

/** Daily source, updates by CBOE's own close of business; 4h poll mirrors
 *  occVolume.ts's cadence for a daily-published series. Eager boot per
 *  KNOWN BROKEN #9 (connect/fetch eagerly at boot, don't wait for a cycle). */
export function bootCboeVixPoll(intervalMs = 4 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshCboeVix().catch((e) => console.error("[datacore] cboevix boot:", e?.message || e));
  setInterval(() => { refreshCboeVix(); }, intervalMs).unref?.();
}
