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

// ── Archive (date-level dedup; one JSONL day-file per TRADE date) ───────────

const archivedDates = new Set<string>();
let seeded = false;

function svDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "finrashortvol");
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/);
      if (m) archivedDates.add(m[1]);
    }
  } catch {}
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
            if (s) cache = { at: Date.now(), summary: s };
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
        if (s) { cache = { at: Date.now(), summary: s }; newestSummarized = true; }
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
