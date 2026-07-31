/**
 * cftcTff.ts — CFTC Traders in Financial Futures, futures-only
 * (BUILD ORDER 6 #1, filed + probed 2026-07-06).
 *
 * Source: CFTC Public Reporting Socrata API, dataset gpe5-46if
 * (publicreporting.cftc.gov) — keyless JSON with NAMED FIELDS, the
 * financial-futures sibling of the live disaggregated-COT stream
 * (cftcCot.ts, 72hh-3qpy). TFF covers ES/NQ/rates/FX-class markets with
 * FINANCIAL trader categories: dealer/intermediary, asset manager,
 * leveraged money, other reportables. Field names verified live
 * 2026-07-06 including their quirks (dealer_* keeps the `_all` suffix;
 * asset_mgr_* and lev_money_* drop it) — FIELD below encodes them
 * exactly.
 *
 * US government work, public domain; attribution "CFTC Traders in
 * Financial Futures". Weekly release (Tuesday data, Friday ~15:30 ET
 * publish); ~44 markets per week, history to 2006 on the dataset.
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * leveraged-money net-positioning extremes mean-revert in index
 * futures; dealer positioning is the informed side. Joins the COT
 * commodities archive for cross-asset positioning reads.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { partitionValid, type QualityIssue } from "./dataQuality";

const DATASET_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json";
/** ~44 markets/week live; headroom for growth without pagination. */
export const TFF_WEEK_ROW_LIMIT = 400;

/** Exact Socrata field names (verified live 2026-07-06 — quirks and all). */
const FIELD = {
  market: "market_and_exchange_names",
  reportDate: "report_date_as_yyyy_mm_dd",
  code: "cftc_contract_market_code",
  commodity: "commodity_name",
  oi: "open_interest_all",
  dealerLong: "dealer_positions_long_all",   // dealer keeps _all …
  dealerShort: "dealer_positions_short_all",
  dealerSpread: "dealer_positions_spread_all",
  amLong: "asset_mgr_positions_long",        // … asset_mgr drops it …
  amShort: "asset_mgr_positions_short",
  amSpread: "asset_mgr_positions_spread",
  levLong: "lev_money_positions_long",       // … and so does lev_money
  levShort: "lev_money_positions_short",
  levSpread: "lev_money_positions_spread",
  otherLong: "other_rept_positions_long",
  otherShort: "other_rept_positions_short",
  otherSpread: "other_rept_positions_spread",
  nonreptLong: "nonrept_positions_long_all",
  nonreptShort: "nonrept_positions_short_all",
  totLong: "tot_rept_positions_long_all",
  totShort: "tot_rept_positions_short",
} as const;

export interface TffRow {
  report_date: string;      // YYYY-MM-DD (Tuesday as-of date)
  market: string;           // market + exchange as published
  code: string;             // CFTC contract market code
  commodity: string | null;
  open_interest: number | null;
  dealer_long: number | null;
  dealer_short: number | null;
  dealer_spread: number | null;
  asset_mgr_long: number | null;
  asset_mgr_short: number | null;
  asset_mgr_spread: number | null;
  lev_money_long: number | null;
  lev_money_short: number | null;
  lev_money_spread: number | null;
  other_rept_long: number | null;
  other_rept_short: number | null;
  other_rept_spread: number | null;
  nonrept_long: number | null;
  nonrept_short: number | null;
  rt: string;               // as-seen UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

// ── GATE 1 (DATA): accounting-identity validation ───────────────────────────
// Same discipline as the passed Legacy-COT gate 1 (cftc_cot.py's
// validate_record: verify a record against CFTC's own accounting identities
// before trusting it — catches field-mapping bugs and upstream corruption
// deterministically, no external ground truth needed beyond the report's own
// arithmetic), adapted for TFF's four financial trader categories (each with
// its own spread field, counted on BOTH the long and short leg — verified
// against a real live record 2026-07-31, see cftcTff.test.ts):
//
//   dealer_l+dealer_sp + asset_mgr_l+asset_mgr_sp + lev_money_l+lev_money_sp
//     + other_rept_l+other_rept_sp        == tot_rept_positions_long_all
//   (mirror on the short side, same spread values reused per leg)
//   tot_rept_long + nonrept_long          == open_interest_all
//   tot_rept_short + nonrept_short        == open_interest_all
const toNum = (v: any): number => {
  const n = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(n) ? n : 0;
};
const IDENTITY_TOLERANCE = 5; // rare report-revision rounding artifact, mirrors cftc_cot.py

/** Validates ONE RAW Socrata row (pre-parseTff field mapping) against CFTC's
 *  own accounting identities. Pure, per-record — no archive/network state. */
export function tffAccountingIssues(raw: any): QualityIssue[] {
  const issues: QualityIssue[] = [];
  const oi = toNum(raw?.[FIELD.oi]);
  if (oi <= 0) {
    issues.push({ field: FIELD.oi, rule: "min", detail: "open interest zero/missing" });
    return issues; // downstream identities are meaningless without a real OI denominator
  }
  const dl = toNum(raw?.[FIELD.dealerLong]), ds = toNum(raw?.[FIELD.dealerShort]), dsp = toNum(raw?.[FIELD.dealerSpread]);
  const al = toNum(raw?.[FIELD.amLong]), as_ = toNum(raw?.[FIELD.amShort]), asp = toNum(raw?.[FIELD.amSpread]);
  const ll = toNum(raw?.[FIELD.levLong]), ls = toNum(raw?.[FIELD.levShort]), lsp = toNum(raw?.[FIELD.levSpread]);
  const ol = toNum(raw?.[FIELD.otherLong]), os_ = toNum(raw?.[FIELD.otherShort]), osp = toNum(raw?.[FIELD.otherSpread]);
  const tl = toNum(raw?.[FIELD.totLong]), ts = toNum(raw?.[FIELD.totShort]);
  const nl = toNum(raw?.[FIELD.nonreptLong]), ns = toNum(raw?.[FIELD.nonreptShort]);

  const calcLong = dl + dsp + al + asp + ll + lsp + ol + osp;
  const calcShort = ds + dsp + as_ + asp + ls + lsp + os_ + osp;
  const checks: Array<[number, string, string]> = [
    [Math.abs(calcLong - tl), FIELD.totLong, `computed ${calcLong} vs reported ${tl}`],
    [Math.abs(calcShort - ts), FIELD.totShort, `computed ${calcShort} vs reported ${ts}`],
    [Math.abs(tl + nl - oi), FIELD.oi, `long total+nonrept ${tl + nl} vs OI ${oi}`],
    [Math.abs(ts + ns - oi), FIELD.oi, `short total+nonrept ${ts + ns} vs OI ${oi}`],
  ];
  for (const [delta, field, detail] of checks) {
    if (delta > IDENTITY_TOLERANCE) issues.push({ field, rule: "identity", detail });
  }
  return issues;
}

/** Socrata JSON rows -> TffRows for the NEWEST report date in the batch
 *  (a fetch ordered DESC can straddle two weeks at the boundary — keep
 *  one coherent week, never mix vintages in a day-file). */
export function parseTff(json: any, rt: string): TffRow[] {
  if (!Array.isArray(json) || !json.length) return [];
  const dateOf = (r: any) => String(r?.[FIELD.reportDate] || "").slice(0, 10);
  const newest = json.reduce((mx: string, r: any) => (dateOf(r) > mx ? dateOf(r) : mx), "");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(newest)) return [];
  const out: TffRow[] = [];
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
      dealer_long: num(r[FIELD.dealerLong]),
      dealer_short: num(r[FIELD.dealerShort]),
      dealer_spread: num(r[FIELD.dealerSpread]),
      asset_mgr_long: num(r[FIELD.amLong]),
      asset_mgr_short: num(r[FIELD.amShort]),
      asset_mgr_spread: num(r[FIELD.amSpread]),
      lev_money_long: num(r[FIELD.levLong]),
      lev_money_short: num(r[FIELD.levShort]),
      lev_money_spread: num(r[FIELD.levSpread]),
      other_rept_long: num(r[FIELD.otherLong]),
      other_rept_short: num(r[FIELD.otherShort]),
      other_rept_spread: num(r[FIELD.otherSpread]),
      nonrept_long: num(r[FIELD.nonreptLong]),
      nonrept_short: num(r[FIELD.nonreptShort]),
      rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchLatestTff(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<TffRow[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const url = `${DATASET_URL}?$limit=${TFF_WEEK_ROW_LIMIT}&$order=${encodeURIComponent("report_date_as_yyyy_mm_dd DESC")}`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] cftctff -> ${r.status}`);
      return [];
    }
    const json = JSON.parse(await r.text());
    if (!Array.isArray(json)) return [];
    // GATE 1: quarantine any row that fails CFTC's own accounting identities
    // before it ever reaches the archive — never let field-mapping bugs or
    // upstream corruption become "trusted" raw history.
    const part = partitionValid(json, tffAccountingIssues);
    if (part.suspect.length) {
      const sample = part.suspect[0].issues.map((i) => `${i.field}:${i.rule}`).join(",");
      console.error(`[datacore] cftctff gate1-reject ${part.suspect.length}/${json.length} (e.g. ${sample})`);
    }
    return parseTff(part.clean, rt);
  } catch (e: any) {
    console.error("[datacore] cftctff:", e?.message || e);
    return [];
  }
}

// ── Archive (one JSONL file per REPORT date; week-level dedup) ──────────────

const archivedWeeks = new Set<string>();
let seeded = false;

function tffDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "cftctff");
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/);
      if (m) archivedWeeks.add(m[1]);
    }
  } catch {}
}

export function archiveTffWeek(rows: TffRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  const dir = tffDir(baseDir);
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
    console.error("[datacore] cftctff archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldTffWeeks(baseDir?: string, nowMs?: number): number {
  const dir = tffDir(baseDir);
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

let cache: { at: number; report_date: string; rows: TffRow[] } | null = null;
let polling = false;

export function latestTff() {
  return cache;
}

export function readArchivedTffWeek(week: string, baseDir?: string): TffRow[] {
  const dir = tffDir(baseDir);
  for (const fp of [path.join(dir, `${week}.jsonl`), path.join(dir, `${week}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: TffRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  }
  return [];
}

/** On restart with the newest week already archived, rebuild the cache
 *  from disk (same honesty rule as cftcCot/finraShortVolume). */
export async function refreshTff(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                 baseDir?: string): Promise<void> {
  try {
    if (!seeded) { seedSeen(tffDir(baseDir)); seeded = true; }
    const rows = await fetchLatestTff(fetchImpl, nowMs);
    if (rows.length) {
      archiveTffWeek(rows, baseDir);
      if (cache?.report_date !== rows[0].report_date) {
        cache = { at: Date.now(), report_date: rows[0].report_date, rows };
      }
    } else if (!cache && archivedWeeks.size) {
      const newest = Array.from(archivedWeeks).sort().pop()!;
      const archived = readArchivedTffWeek(newest, baseDir);
      if (archived.length) cache = { at: Date.now(), report_date: newest, rows: archived };
    }
    gzipOldTffWeeks(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] cftctff refresh:", e?.message || e);
  }
}

/** Weekly source (Friday ~15:30 ET publish) — 12h poll makes off-days a
 *  dedup no-op while catching the publish within half a day. Eager boot
 *  per KNOWN BROKEN #9. */
export function bootTffPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshTff();
  setInterval(() => { refreshTff(); }, intervalMs).unref?.();
}
