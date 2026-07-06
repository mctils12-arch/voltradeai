/**
 * treasuryDts.ts — Treasury Daily Statement, deposits & withdrawals of
 * operating cash (BUILD ORDER 6 #2, filed + probed 2026-07-06).
 *
 * Source: Treasury FiscalData API (api.fiscaldata.treasury.gov),
 * endpoint v1/accounting/dts/deposits_withdrawals_operating_cash —
 * fully keyless JSON. Field names verified live 2026-07-06
 * (record_date, account_type, transaction_type Deposits|Withdrawals,
 * transaction_catg, transaction_today_amt/mtd/fytd in $ MILLIONS,
 * src_line_nbr). ~90 rows per business day; history to 2005-10-03
 * (473k rows on the dataset). Published each business day for the
 * prior business day (~1-2 day lag). NOTE: square brackets in the
 * page[size] param must be URL-encoded — the API 400s otherwise.
 *
 * US Treasury open data, public domain; attribution "U.S. Treasury
 * Fiscal Data — Daily Treasury Statement".
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * withheld income/employment tax deposit categories form a DAILY
 * payroll nowcast weeks ahead of BLS releases; corporate-tax and FUTA
 * category deltas nowcast macro turns (regime input). Gate 1 =
 * reconcile monthly sums vs MTS/FRED federal receipts; gate 2 =
 * withheld-tax YoY growth vs payroll-surprise dates.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const DATASET_URL =
  "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/deposits_withdrawals_operating_cash";
/** ~90 rows/business day live; 500 covers a full day plus boundary spillover. */
export const DTS_FETCH_LIMIT = 500;

export interface DtsRow {
  record_date: string;        // YYYY-MM-DD (business day the statement covers)
  account_type: string;       // e.g. "Treasury General Account (TGA)"
  transaction_type: string;   // "Deposits" | "Withdrawals"
  category: string;           // transaction_catg verbatim (withheld taxes, FUTA, ...)
  today_amt: number | null;   // $ millions
  mtd_amt: number | null;     // $ millions
  fytd_amt: number | null;    // $ millions
  src_line: number | null;    // source line number (stable ordering within a table)
  rt: string;                 // as-seen UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "" || v === "null") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

/** FiscalData JSON -> DtsRows for the NEWEST record_date in the batch
 *  (a DESC fetch straddles two business days at the boundary — keep one
 *  coherent day, never mix statements in a day-file). */
export function parseDts(json: any, rt: string): DtsRow[] {
  const data = json?.data;
  if (!Array.isArray(data) || !data.length) return [];
  const dateOf = (r: any) => String(r?.record_date || "").slice(0, 10);
  const newest = data.reduce((mx: string, r: any) => (dateOf(r) > mx ? dateOf(r) : mx), "");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(newest)) return [];
  const out: DtsRow[] = [];
  for (const r of data) {
    if (dateOf(r) !== newest) continue;
    const catg = r?.transaction_catg;
    const ttype = r?.transaction_type;
    if (!catg || !ttype) continue;
    out.push({
      record_date: newest,
      account_type: String(r.account_type ?? ""),
      transaction_type: String(ttype),
      category: String(catg),
      today_amt: num(r.transaction_today_amt),
      mtd_amt: num(r.transaction_mtd_amt),
      fytd_amt: num(r.transaction_fytd_amt),
      src_line: num(r.src_line_nbr),
      rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchLatestDts(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<DtsRow[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  // brackets MUST be encoded: page%5Bsize%5D (the API 400s on raw [])
  const url = `${DATASET_URL}?sort=-record_date&page%5Bsize%5D=${DTS_FETCH_LIMIT}`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] treasurydts -> ${r.status}`);
      return [];
    }
    return parseDts(JSON.parse(await r.text()), rt);
  } catch (e: any) {
    console.error("[datacore] treasurydts:", e?.message || e);
    return [];
  }
}

// ── Archive (one JSONL file per statement date; day-level dedup) ────────────

const archivedDays = new Set<string>();
let seeded = false;

function dtsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "treasurydts");
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/);
      if (m) archivedDays.add(m[1]);
    }
  } catch {}
}

export function archiveDtsDay(rows: DtsRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  const dir = dtsDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const day = rows[0].record_date;
  if (archivedDays.has(day)) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${day}.jsonl`),
                     rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    archivedDays.add(day);
    return rows.length;
  } catch (e: any) {
    console.error("[datacore] treasurydts archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldDtsDays(baseDir?: string, nowMs?: number): number {
  const dir = dtsDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 4 * 86400_000) continue; // statement final once published; margin for the lag
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; record_date: string; rows: DtsRow[] } | null = null;
let polling = false;

export function latestDts() {
  return cache;
}

export function readArchivedDtsDay(day: string, baseDir?: string): DtsRow[] {
  const dir = dtsDir(baseDir);
  for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: DtsRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  }
  return [];
}

/** On restart with the newest day already archived, rebuild the cache
 *  from disk (same honesty rule as cftcCot/cftcTff/finraShortVolume). */
export async function refreshDts(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                 baseDir?: string): Promise<void> {
  try {
    if (!seeded) { seedSeen(dtsDir(baseDir)); seeded = true; }
    const rows = await fetchLatestDts(fetchImpl, nowMs);
    if (rows.length) {
      archiveDtsDay(rows, baseDir);
      if (cache?.record_date !== rows[0].record_date) {
        cache = { at: Date.now(), record_date: rows[0].record_date, rows };
      }
    } else if (!cache && archivedDays.size) {
      const newest = Array.from(archivedDays).sort().pop()!;
      const archived = readArchivedDtsDay(newest, baseDir);
      if (archived.length) cache = { at: Date.now(), record_date: newest, rows: archived };
    }
    gzipOldDtsDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] treasurydts refresh:", e?.message || e);
  }
}

/** Daily source (published each business day ~afternoon ET for the
 *  prior business day) — 6h poll catches the publish same day while
 *  weekends/holidays are dedup no-ops. Eager boot per KNOWN BROKEN #9. */
export function bootDtsPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshDts();
  setInterval(() => { refreshDts(); }, intervalMs).unref?.();
}
