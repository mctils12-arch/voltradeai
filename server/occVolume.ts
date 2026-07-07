/**
 * occVolume.ts — OCC daily options cleared volume by trade origin
 * (DATACORE MAXIMUS census #1, filed + worked up 2026-07-06).
 *
 * Source: marketdata.theocc.com/volume-query — keyless CSV, probed
 * live 2026-07-06. ⚠️ ARCHIVE-NOW ROOT: OCC serves a ROLLING 2-YEAR
 * window (verified: older dates return the verbatim body "Report date
 * cannot be prior to  2 years" — double space theirs); data ages off
 * the back daily, so every day not archived is permanently lost.
 * Accumulation substitutes for purchase (EDGE DOCTRINE #1).
 *
 * Verified shape (CRLF): quantity,underlying,symbol,actype,porc,
 * exchange,actdate — actype C=customer/F=firm/M=market-maker, porc
 * P/C, symbol is the option root (numeric prefixes = adjusted
 * series), 18 exchange codes, actdate MM/DD/YYYY. ~153k rows /
 * ~5MB raw / ~700KB gz per day. GOTCHAS (all verified): every error
 * arrives as HTTP 200 + plain text (detect by body, never status);
 * weekends = "No record(s) found"; unpublished = "Report date cannot
 * be greater than ..."; per-symbol variant has a trailing comma;
 * quantity counts EACH CLEARING SIDE (a C row and its matching M row
 * both carry the full qty — summing actypes double-counts cleared
 * volume; divide by 2 for totals).
 *
 * License: OCC informational use; redistribution needs OCC
 * permission — bot-internal + gated derived signals OK; raw resale
 * needs review (filed).
 *
 * HYPOTHESIS (gate-locked — archive + RAW only): customer-vs-
 * market-maker put/call BY TICKER (the retired ISEE's superior
 * successor) is uncrowded on small caps. Gate 1 = one day's totals vs
 * OCC's published daily volume page; gate 2 = customer-opening P/C
 * extremes vs forward returns, discounted for the census breadth.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const BASE_URL = "https://marketdata.theocc.com/volume-query";

export interface OccRow {
  date: string;              // YYYY-MM-DD (normalized from MM/DD/YYYY)
  underlying: string;
  symbol: string;            // option root (numeric prefix = adjusted series)
  actype: string;            // C | F | M
  porc: string;              // P | C
  exchange: string;
  qty: number;               // per clearing side — see double-count note
}

/** Verbatim error-body classification (all arrive as HTTP 200). */
export type OccBodyKind = "data" | "no_records" | "not_yet_published" | "aged_out" | "unknown_error";

export function classifyBody(body: string): OccBodyKind {
  const t = body.trim();
  if (t.startsWith("quantity,")) return "data";
  if (t === "No record(s) found") return "no_records";
  if (t.startsWith("Report date cannot be greater than")) return "not_yet_published";
  if (t.startsWith("Report date cannot be prior to")) return "aged_out";
  return "unknown_error";
}

/** "07/02/2026" -> "2026-07-02"; null on garbage. */
export function normalizeActDate(v: string): string | null {
  const m = v.trim().match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (!m) return null;
  return `${m[3]}-${m[1]}-${m[2]}`;
}

/** CSV -> rows. Handles CRLF and the per-symbol variant's trailing comma. */
export function parseOcc(csv: string): OccRow[] {
  const lines = csv.split(/\r?\n/);
  if (!lines.length || !lines[0].startsWith("quantity,")) return [];
  const out: OccRow[] = [];
  for (const line of lines.slice(1)) {
    if (!line.trim()) continue;
    const c = line.replace(/,\s*$/, "").split(",");
    if (c.length !== 7) continue;
    const qty = parseInt(c[0], 10);
    const date = normalizeActDate(c[6]);
    if (!Number.isFinite(qty) || !date || !c[1] || !c[3] || !c[4]) continue;
    out.push({
      date, underlying: c[1].trim(), symbol: c[2].trim(),
      actype: c[3].trim(), porc: c[4].trim(), exchange: c[5].trim(), qty,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export function occUrl(dateYmd: string): string {
  // dateYmd = YYYYMMDD
  return `${BASE_URL}?reportDate=${dateYmd}&format=csv&volumeQueryType=O&symbolType=ALL&symbol=ALL&reportType=D&accountType=ALL&productKind=ALL&porc=BOTH`;
}

export async function fetchOccDay(dateYmd: string, fetchImpl: FetchFn = fetch as any):
    Promise<{ kind: OccBodyKind; rows: OccRow[] }> {
  try {
    const r = await fetchImpl(occUrl(dateYmd), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(60000) as any,
    });
    const body = await r.text();
    if (!r.ok) {
      console.error(`[datacore] occvolume ${dateYmd} -> HTTP ${r.status}`);
      return { kind: "unknown_error", rows: [] };
    }
    const kind = classifyBody(body);
    return { kind, rows: kind === "data" ? parseOcc(body) : [] };
  } catch (e: any) {
    console.error(`[datacore] occvolume ${dateYmd}:`, e?.message || e);
    return { kind: "unknown_error", rows: [] };
  }
}

// ── Archive (one gz CSV-equivalent JSONL per REPORT date; day-level dedup) ──

const archivedDays = new Set<string>();
let seeded = false;

function occDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "occvolume");
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/);
      if (m) archivedDays.add(m[1]);
    }
  } catch {}
}

export function isDayArchived(day: string, baseDir?: string): boolean {
  if (!seeded) { seedSeen(occDir(baseDir)); seeded = true; }
  return archivedDays.has(day);
}

/** Test isolation only (alpaca_feed._reset_for_tests precedent) — the
 *  module-level dedup Set spans one process by design in prod. */
export function _resetOccForTests(): void {
  archivedDays.clear();
  seeded = false;
  cache = null;
}

/** Writes one report day (all rows) gzipped-on-write — at ~153k rows/day
 *  the plain file would be ~18MB JSONL; gz lands ~1MB. Day-level dedup:
 *  a report date is final when published. */
export function archiveOccDay(rows: OccRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  const dir = occDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const day = rows[0].date;
  if (archivedDays.has(day)) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const payload = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
    fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(payload));
    archivedDays.add(day);
    return rows.length;
  } catch (e: any) {
    console.error("[datacore] occvolume archive:", e?.message || e);
    return 0;
  }
}

// ── Cache + poll ────────────────────────────────────────────────────────────

export interface UnderlyingStat {
  underlying: string;
  cust_put: number;          // customer-side qty (per clearing side, NOT halved —
  cust_call: number;         // ratios are side-consistent so P/C math is valid)
  mm_put: number;
  mm_call: number;
  total_cleared: number;     // all-rows qty / 2 (double-count corrected)
}

let cache: { at: number; report_date: string; top: UnderlyingStat[]; underlyings: number } | null = null;
let polling = false;

export function latestOcc() {
  return cache;
}

export function aggregateOcc(rows: OccRow[], topN = 50): { top: UnderlyingStat[]; underlyings: number } {
  const by = new Map<string, UnderlyingStat>();
  let allQty = 0;
  for (const r of rows) {
    allQty += r.qty;
    let s = by.get(r.underlying);
    if (!s) {
      s = { underlying: r.underlying, cust_put: 0, cust_call: 0, mm_put: 0, mm_call: 0, total_cleared: 0 };
      by.set(r.underlying, s);
    }
    s.total_cleared += r.qty;
    if (r.actype === "C") {
      if (r.porc === "P") s.cust_put += r.qty; else s.cust_call += r.qty;
    } else if (r.actype === "M") {
      if (r.porc === "P") s.mm_put += r.qty; else s.mm_call += r.qty;
    }
  }
  const top = Array.from(by.values());
  // quantity counts each clearing side — halve totals for cleared volume
  top.forEach((s) => { s.total_cleared = Math.round(s.total_cleared / 2); });
  top.sort((a, b) => b.total_cleared - a.total_cleared);
  return { top: top.slice(0, topN), underlyings: by.size };
}

function recentTradingDaysYmd(nowMs: number, n = 5): string[] {
  const out: string[] = [];
  const d = new Date(nowMs);
  while (out.length < n) {
    const dow = d.getUTCDay();
    if (dow !== 0 && dow !== 6) out.push(d.toISOString().slice(0, 10).replace(/-/g, ""));
    d.setUTCDate(d.getUTCDate() - 1);
  }
  return out;
}

/** Fetches the most recent unarchived report days (holiday/not-yet-published
 *  bodies are classified no-ops, never parsed as data). */
export async function refreshOcc(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                 baseDir?: string): Promise<void> {
  try {
    const now = nowMs ?? Date.now();
    for (const ymd of recentTradingDaysYmd(now)) {
      const day = `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}`;
      if (isDayArchived(day, baseDir)) continue;
      const { kind, rows } = await fetchOccDay(ymd, fetchImpl);
      if (kind === "data" && rows.length) {
        archiveOccDay(rows, baseDir);
        if (!cache || day > cache.report_date) {
          const agg = aggregateOcc(rows);
          cache = { at: Date.now(), report_date: day, top: agg.top, underlyings: agg.underlyings };
        }
      } else if (kind === "unknown_error") {
        console.error(`[datacore] occvolume ${day}: unclassified body — investigate`);
      }
      // no_records (weekend/holiday) and not_yet_published are honest no-ops
    }
  } catch (e: any) {
    console.error("[datacore] occvolume refresh:", e?.message || e);
  }
}

// ── Deep backfill (env-gated OFF — R8 lesson; volume budget → Mike) ─────────
// The OCC window is a HARD rolling 2 years (verified error body): data
// ages off the BACK EDGE DAILY, so every unbackfilled day is permanently
// lost. Full capture ≈ 500 trading days x ~1MB gz ≈ 500MB on the volume
// (prod archive was 0.25GB total when this shipped) — Mike confirms
// capacity, then sets OCC_DEEP_BACKFILL=1. Files gz on write via
// archiveOccDay, so nothing accumulates uncompressed mid-pass.

export const OCC_BACKFILL_CALENDAR_DAYS = 730; // the verified window

// HUMAN-APPROVED DEFAULT-ON (2026-07-07): Mike confirmed volume headroom
// (~2GB of 5GB used; backfill adds ~500MB gz) and instructed enabling.
// Default-on with an explicit opt-out (OCC_DEEP_BACKFILL=0) — the R8
// caution was about UNAPPROVED backfills; the done-marker still limits
// this to a single pass ever.
export function occDeepBackfillEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.OCC_DEEP_BACKFILL !== "0";
}

const backfillMarker = (baseDir?: string) => path.join(occDir(baseDir), "backfill_done.json");

export async function occDeepBackfillIfEnabled(
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
  baseDir?: string,
  env: NodeJS.ProcessEnv = process.env,
  calendarDays = OCC_BACKFILL_CALENDAR_DAYS,
  spacingMs = 1500, // politeness: ~5MB/request
): Promise<void> {
  if (!occDeepBackfillEnabled(env)) return;
  if (fs.existsSync(backfillMarker(baseDir))) return;
  const now = nowMs ?? Date.now();
  console.log(`[datacore] occvolume deep backfill: walking ${calendarDays} calendar days (rolling-window rescue)`);
  let fetched = 0, skipped = 0;
  // oldest-first: the back edge is what the purge is eating — rescue it first
  for (let i = calendarDays; i >= 0; i--) {
    const d = new Date(now - i * 86400_000);
    const dow = d.getUTCDay();
    if (dow === 0 || dow === 6) continue;
    const day = d.toISOString().slice(0, 10);
    if (isDayArchived(day, baseDir)) { skipped++; continue; }
    const { kind, rows } = await fetchOccDay(day.replace(/-/g, ""), fetchImpl);
    if (kind === "data" && rows.length) { archiveOccDay(rows, baseDir); fetched++; }
    // no_records / not_yet_published / aged_out are honest no-ops
    await new Promise((res) => setTimeout(res, spacingMs));
  }
  try {
    fs.writeFileSync(backfillMarker(baseDir), JSON.stringify({
      done_rt: new Date().toISOString(), days_fetched: fetched, days_already_archived: skipped,
      calendar_days: calendarDays,
    }));
  } catch {}
  console.log(`[datacore] occvolume deep backfill done: ${fetched} days fetched, ${skipped} already archived`);
}

/** Daily source published overnight ET — 4h poll retries "not yet
 *  published" until day T lands, and backfills brief gaps via the
 *  5-trading-day window. Eager boot per KNOWN BROKEN #9. Deep backfill
 *  (rolling-window rescue) only on explicit env opt-in, after current
 *  data is up. */
export function bootOccPoll(intervalMs = 4 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshOcc().then(() => occDeepBackfillIfEnabled())
    .catch((e) => console.error("[datacore] occvolume boot:", e?.message || e));
  setInterval(() => { refreshOcc(); }, intervalMs).unref?.();
}
