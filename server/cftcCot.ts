/**
 * cftcCot.ts — CFTC Commitments of Traders, disaggregated futures-only
 * (BUILD ORDER 5 #2, filed + probed 2026-07-05).
 *
 * Source: CFTC Public Reporting Socrata API, dataset 72hh-3qpy
 * (publicreporting.cftc.gov) — keyless JSON with NAMED FIELDS, chosen
 * over the flat f_disagg.txt file precisely because that file is
 * HEADERLESS positional CSV (query-shape honesty: never parse by
 * position when a named-field source exists). Field names verified live
 * 2026-07-05 including their quirks (e.g. `swap__positions_short_all`
 * has a double underscore; several fields drop the `_all` suffix) — the
 * FIELD constant below encodes them exactly.
 *
 * US government work, public domain; attribution "CFTC Commitments of
 * Traders (disaggregated)". Weekly release (Tuesday data, Friday
 * ~15:30 ET publish); ~274 contract markets per week.
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * managed-money net-positioning extremes (percentile vs trailing
 * history) mean-revert in commodity-linked ETFs; joins the EIA
 * petroleum/natgas inventory work. Accumulation substitutes for
 * purchase: vendors sell exactly this series recorded over time.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const DATASET_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json";
/** ~274 markets/week live; headroom for growth without pagination. */
export const WEEK_ROW_LIMIT = 1200;

/** Exact Socrata field names (verified live 2026-07-05 — quirks and all). */
const FIELD = {
  market: "market_and_exchange_names",
  reportDate: "report_date_as_yyyy_mm_dd",
  code: "cftc_contract_market_code",
  commodity: "commodity_name",
  oi: "open_interest_all",
  prodLong: "prod_merc_positions_long",
  prodShort: "prod_merc_positions_short",
  swapLong: "swap_positions_long_all",
  swapShort: "swap__positions_short_all",   // double underscore in the source
  swapSpread: "swap__positions_spread_all", // double underscore in the source
  mmLong: "m_money_positions_long_all",
  mmShort: "m_money_positions_short_all",
  mmSpread: "m_money_positions_spread",
  otherLong: "other_rept_positions_long",
  otherShort: "other_rept_positions_short",
  otherSpread: "other_rept_positions_spread",
} as const;

export interface CotRow {
  report_date: string;      // YYYY-MM-DD (Tuesday as-of date)
  market: string;           // market + exchange as published
  code: string;             // CFTC contract market code
  commodity: string | null;
  open_interest: number | null;
  prod_merc_long: number | null;
  prod_merc_short: number | null;
  swap_long: number | null;
  swap_short: number | null;
  swap_spread: number | null;
  m_money_long: number | null;
  m_money_short: number | null;
  m_money_spread: number | null;
  other_rept_long: number | null;
  other_rept_short: number | null;
  other_rept_spread: number | null;
  rt: string;               // as-seen UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

/** Socrata JSON rows -> CotRows for the NEWEST report date in the batch
 *  (a fetch ordered DESC can straddle two weeks at the boundary — keep
 *  one coherent week, never mix vintages in a day-file). */
export function parseCot(json: any, rt: string): CotRow[] {
  if (!Array.isArray(json) || !json.length) return [];
  const dateOf = (r: any) => String(r?.[FIELD.reportDate] || "").slice(0, 10);
  const newest = json.reduce((mx: string, r: any) => (dateOf(r) > mx ? dateOf(r) : mx), "");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(newest)) return [];
  const out: CotRow[] = [];
  for (const r of json) {
    if (dateOf(r) !== newest) continue;
    const market = r?.[FIELD.market];
    const code = r?.[FIELD.code];
    if (!market || !code) continue;
    out.push({
      report_date: newest,
      market: String(market),
      code: String(code).trim(),
      commodity: r[FIELD.commodity] != null ? String(r[FIELD.commodity]).trim() : null,
      open_interest: num(r[FIELD.oi]),
      prod_merc_long: num(r[FIELD.prodLong]),
      prod_merc_short: num(r[FIELD.prodShort]),
      swap_long: num(r[FIELD.swapLong]),
      swap_short: num(r[FIELD.swapShort]),
      swap_spread: num(r[FIELD.swapSpread]),
      m_money_long: num(r[FIELD.mmLong]),
      m_money_short: num(r[FIELD.mmShort]),
      m_money_spread: num(r[FIELD.mmSpread]),
      other_rept_long: num(r[FIELD.otherLong]),
      other_rept_short: num(r[FIELD.otherShort]),
      other_rept_spread: num(r[FIELD.otherSpread]),
      rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchLatestCot(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<CotRow[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const url = `${DATASET_URL}?$limit=${WEEK_ROW_LIMIT}&$order=${encodeURIComponent("report_date_as_yyyy_mm_dd DESC")}`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] cftccot -> ${r.status}`);
      return [];
    }
    return parseCot(JSON.parse(await r.text()), rt);
  } catch (e: any) {
    console.error("[datacore] cftccot:", e?.message || e);
    return [];
  }
}

// ── Archive (one JSONL file per REPORT date; week-level dedup) ──────────────

const archivedWeeks = new Set<string>();
let seeded = false;

function cotDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "cftccot");
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/);
      if (m) archivedWeeks.add(m[1]);
    }
  } catch {}
}

export function archiveCotWeek(rows: CotRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  const dir = cotDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const week = rows[0].report_date;
  if (archivedWeeks.has(week)) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${week}.jsonl`),
                     rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    archivedWeeks.add(week);
    return rows.length;
  } catch (e: any) {
    console.error("[datacore] cftccot archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldCotWeeks(baseDir?: string, nowMs?: number): number {
  const dir = cotDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 9 * 86400_000) continue; // report stays plain until superseded
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; report_date: string; rows: CotRow[] } | null = null;
let polling = false;

export function latestCot() {
  return cache;
}

export function readArchivedWeek(week: string, baseDir?: string): CotRow[] {
  const dir = cotDir(baseDir);
  for (const fp of [path.join(dir, `${week}.jsonl`), path.join(dir, `${week}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: CotRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  }
  return [];
}

/** On restart with the newest week already archived, rebuild the cache
 *  from disk (same honesty rule as finraShortVolume). */
export async function refreshCot(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                 baseDir?: string): Promise<void> {
  try {
    if (!seeded) { seedSeen(cotDir(baseDir)); seeded = true; }
    const rows = await fetchLatestCot(fetchImpl, nowMs);
    if (rows.length) {
      archiveCotWeek(rows, baseDir);
      if (cache?.report_date !== rows[0].report_date) {
        cache = { at: Date.now(), report_date: rows[0].report_date, rows };
      }
    } else if (!cache && archivedWeeks.size) {
      const newest = Array.from(archivedWeeks).sort().pop()!;
      const archived = readArchivedWeek(newest, baseDir);
      if (archived.length) cache = { at: Date.now(), report_date: newest, rows: archived };
    }
    gzipOldCotWeeks(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] cftccot refresh:", e?.message || e);
  }
}

/** Weekly source (Friday ~15:30 ET publish) — 12h poll makes off-days a
 *  dedup no-op while catching the publish within half a day. Eager boot
 *  per KNOWN BROKEN #9. */
export function bootCotPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshCot();
  setInterval(() => { refreshCot(); }, intervalMs).unref?.();
}
