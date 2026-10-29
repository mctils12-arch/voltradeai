/**
 * squeezeGex.ts — SqueezeMetrics dealer-gamma (GEX) / dark-pool index (DIX)
 * daily archive, plus the pre-registered trailing-252 percentile rank R.
 * Edge P1 (research/edge_preregistration.md § P1; gate-2 PASS 2026-10-28,
 * research/edge_p1_dealer_gamma_result.md). T-DATACORE shadow-logging
 * stream named in the pre-registration: date, price, dix, gex, fetched_at.
 *
 * INTERNAL ONLY. DIX.csv's licence terms are unverified and GATE 1 (DATA)
 * is still open (GEX is a proprietary model estimate; our own
 * Black-Scholes cross-check needs open-interest history we do not hold
 * yet). So: no /data route, no /api/v1 surface, no raw rows leaving the
 * server — the diag probe returns aggregates only (latest date, R,
 * counts). RAW OVERLAY vs SIGNAL: this is neither on /data today.
 *
 * R is bit-for-bit the registered definition: percentile rank of G_t in
 * the trailing 252 rows ending at t, inclusive, (below + 0.5*ties_excl_self)
 * / 251; null for the first 251 rows. Mirrors trailing_pct_rank() in
 * scripts/edge_p1_dealer_gamma.py (tests pin the same fixtures).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export const SQUEEZE_URL = "https://squeezemetrics.com/monitor/static/DIX.csv";
export const RANK_WINDOW = 252;

export interface SqueezeRow {
  date: string;       // YYYY-MM-DD, the SPX session the values describe
  price: number;      // SPX close
  dix: number;
  gex: number;        // $ gamma per 1% move (estimate)
  fetched_at: string; // ISO time this row was first archived
}

/** `date,price,dix,gex` -> rows sorted by date; malformed lines skipped,
 *  never filled. Duplicate dates keep the last occurrence. */
export function parseSqueezeCsv(csv: string, fetchedAt = new Date().toISOString()): SqueezeRow[] {
  const lines = csv.split(/\r?\n/);
  if (!lines.length || !lines[0].trim().toLowerCase().startsWith("date,")) return [];
  const byDate = new Map<string, SqueezeRow>();
  for (const line of lines.slice(1)) {
    if (!line.trim()) continue;
    const c = line.split(",");
    if (c.length < 4 || !/^\d{4}-\d{2}-\d{2}$/.test(c[0])) continue;
    const price = parseFloat(c[1]), dix = parseFloat(c[2]), gex = parseFloat(c[3]);
    if (![price, dix, gex].every(Number.isFinite)) continue;
    byDate.set(c[0], { date: c[0], price, dix, gex, fetched_at: fetchedAt });
  }
  return Array.from(byDate.values()).sort((a, b) => a.date.localeCompare(b.date));
}

/** Trailing percentile rank, uses only values[<= t]. null until a full window. */
export function trailingPctRank(values: number[], window = RANK_WINDOW): (number | null)[] {
  const out: (number | null)[] = [];
  for (let t = 0; t < values.length; t++) {
    if (t < window - 1) { out.push(null); continue; }
    const v = values[t];
    let below = 0, ties = 0;
    for (let i = t - window + 1; i <= t; i++) {
      if (values[i] < v) below++;
      else if (values[i] === v) ties++;
    }
    out.push((below + 0.5 * (ties - 1)) / (window - 1));
  }
  return out;
}

// ── Archive: one plain JSONL per year, day-level dedup ───────────────────────

function dir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "squeeze_gex");
}
function yearFile(d: string, year: string, gz: boolean): string {
  return path.join(d, `${year}.jsonl${gz ? ".gz" : ""}`);
}

function readYear(d: string, year: string): SqueezeRow[] {
  let text: string | null = null;
  try {
    const plain = yearFile(d, year, false), gz = yearFile(d, year, true);
    if (fs.existsSync(plain)) text = fs.readFileSync(plain, "utf8");
    else if (fs.existsSync(gz)) text = zlib.gunzipSync(fs.readFileSync(gz)).toString("utf8");
  } catch { return []; }
  const out: SqueezeRow[] = [];
  for (const line of (text || "").split("\n")) {
    if (!line) continue;
    try { out.push(JSON.parse(line)); } catch { continue; }
  }
  return out;
}

/** Every archived row, ascending by date. */
export function readAllSqueezeRows(baseDir?: string): SqueezeRow[] {
  const d = dir(baseDir);
  let years: string[] = [];
  try {
    years = Array.from(new Set(fs.readdirSync(d).map((f) => f.match(/^(\d{4})\.jsonl(\.gz)?$/)?.[1]).filter(Boolean) as string[])).sort();
  } catch { return []; }
  return years.flatMap((y) => readYear(d, y)).sort((a, b) => a.date.localeCompare(b.date));
}

/** Appends rows whose date is not yet archived. Returns the count written. */
export function archiveSqueezeRows(rows: SqueezeRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  const d = dir(baseDir);
  try {
    fs.mkdirSync(d, { recursive: true });
    const seen = new Set(readAllSqueezeRows(baseDir).map((r) => r.date));
    const fresh = rows.filter((r) => !seen.has(r.date));
    if (!fresh.length) return 0;
    const byYear = new Map<string, SqueezeRow[]>();
    fresh.forEach((r) => byYear.set(r.date.slice(0, 4), [...(byYear.get(r.date.slice(0, 4)) || []), r]));
    byYear.forEach((rs, year) => {
      const gz = yearFile(d, year, true), plain = yearFile(d, year, false);
      if (fs.existsSync(gz) && !fs.existsSync(plain)) {   // late row for a rolled-over year
        fs.writeFileSync(plain, zlib.gunzipSync(fs.readFileSync(gz)));
        fs.unlinkSync(gz);
      }
      fs.appendFileSync(plain, rs.map((r) => JSON.stringify(r)).join("\n") + "\n");
    });
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] squeeze_gex archive:", e?.message || e);
    return 0;
  }
}

/** Gzips every year file but the current one. */
export function gzipPastSqueezeYears(baseDir?: string, nowMs = Date.now()): number {
  const d = dir(baseDir), cur = String(new Date(nowMs).getUTCFullYear());
  let n = 0;
  try {
    for (const f of fs.readdirSync(d)) {
      const m = f.match(/^(\d{4})\.jsonl$/);
      if (!m || m[1] === cur) continue;
      const fp = path.join(d, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Aggregate-only report (diag probe) ───────────────────────────────────────

export interface SqueezeReport {
  probe: "squeeze_gex";
  rows: number;
  first_date: string | null;
  latest_date: string | null;
  latest_age_days: number | null;
  latest_gex_negative: boolean | null;
  latest_rank: number | null;      // R_t; null inside the 251-row burn-in
  rank_rows: number;               // rows with a defined R
  fetched_at_latest: string | null;
  last_refresh_ok: boolean | null;
  note: string;
}

export function squeezeReport(baseDir?: string, nowMs = Date.now()): SqueezeReport {
  const rows = readAllSqueezeRows(baseDir);
  const ranks = trailingPctRank(rows.map((r) => r.gex));
  const last = rows[rows.length - 1];
  return {
    probe: "squeeze_gex",
    rows: rows.length,
    first_date: rows[0]?.date ?? null,
    latest_date: last?.date ?? null,
    latest_age_days: last ? Math.floor((nowMs - Date.parse(last.date + "T00:00:00Z")) / 86_400_000) : null,
    latest_gex_negative: last ? last.gex < 0 : null,
    latest_rank: ranks.length ? ranks[ranks.length - 1] : null,
    rank_rows: ranks.filter((r) => r != null).length,
    fetched_at_latest: last?.fetched_at ?? null,
    last_refresh_ok: lastRefreshOk,
    note: "aggregates only; internal shadow stream, licence unverified, gate 1 open — not a /data surface and not read by any trading path",
  };
}

// ── Fetch + poll ─────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
let lastRefreshOk: boolean | null = null;
let polling = false;

export function _resetSqueezeForTests(): void { lastRefreshOk = null; polling = false; }

export async function refreshSqueezeGex(fetchImpl: FetchFn = fetch as any, baseDir?: string, nowMs = Date.now()): Promise<number> {
  try {
    const r = await fetchImpl(SQUEEZE_URL, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) { console.error(`[datacore] squeeze_gex -> HTTP ${r.status}`); lastRefreshOk = false; return 0; }
    const rows = parseSqueezeCsv(await r.text(), new Date(nowMs).toISOString());
    if (!rows.length) { console.error("[datacore] squeeze_gex: no parseable rows"); lastRefreshOk = false; return 0; }
    const n = archiveSqueezeRows(rows, baseDir);
    gzipPastSqueezeYears(baseDir, nowMs);
    lastRefreshOk = true;
    return n;
  } catch (e: any) {
    console.error("[datacore] squeeze_gex refresh:", e?.message || e);
    lastRefreshOk = false;
    return 0;
  }
}

/** Source publishes after the close; a 6h poll lands the new row within hours. */
export function bootSqueezeGexPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshSqueezeGex().catch((e) => console.error("[datacore] squeeze_gex boot:", e?.message || e));
  setInterval(() => { refreshSqueezeGex(); }, intervalMs).unref?.();
}
