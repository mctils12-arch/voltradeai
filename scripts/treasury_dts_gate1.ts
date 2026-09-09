/**
 * treasury_dts_gate1.ts — GATE 1 (DATA) for the treasury_daily_statement
 * root: does our TGA-deposits reading track Treasury's own INDEPENDENTLY
 * published Monthly Treasury Statement (MTS) Total Receipts?
 *
 * The build order's own stated ladder path (server/treasuryDts.ts's file
 * header, BUILD ORDER 6 #2) has named this test since 2026-07-06 and it
 * had never been run (datacore/signal_ladder.json's own note: "no gate-1
 * run found in the record").
 *
 * WHY THIS IS A REAL RECONCILIATION, NOT A CIRCULAR ONE: the two
 * datasets are DIFFERENT FiscalData products, compiled by different
 * internal processes on different cadences —
 * deposits_withdrawals_operating_cash is a daily cash ledger of every
 * TGA transaction; mts_table_1 is Treasury's own monthly SUMMARY report
 * (published separately, with its own review process). Comparing them
 * is the same class of check as the fred_macro_series/un_comtrade
 * gate-1 precedent: our own parse of a granular feed vs. the issuing
 * agency's own separately-compiled rollup of the same underlying
 * activity.
 *
 * PASS BAR (stated before running the correlation, REASONING STANDARD
 * #10): Pearson r >= 0.85 between sumTgaDepositsExDebt (the real
 * production function, imported not reimplemented) evaluated on the
 * last business day of each calendar month, and MTS's own published
 * "current month gross receipts" for that same month, over EVERY
 * complete month both series can report (both fiscal years in the
 * latest MTS report — no cherry-picked window). The ONE exclusion
 * (Public Debt Cash Issues) is decided a priori on ordinary federal-
 * budget accounting grounds — debt issuance/rollover is financing, not
 * a receipt — not tuned after seeing the correlation. An initial naive
 * sum (no exclusions) was investigated and found to double-count via
 * the API's own subtotal row (a mechanical bug, not a hypothesis
 * search) before this one exclusion was decided; no second exclusion
 * was tried after that.
 *
 * KNOWN, EXPECTED, NOT A FAILURE MODE: the ratio will not be ~1.0. TGA
 * deposits ex-debt still include some categories unified-budget
 * "receipts" nets out or classifies differently (this session measured
 * a stable ~1.2x wedge, ratio stdev ~0.09 across 22 months) — the gate
 * asks whether the ARCHIVE TRACKS REALITY (correlation), not whether
 * the two levels are numerically identical (see un_comtrade_gate1.py
 * for a case where the ladder DOES require near-1.0, a different
 * accounting relationship).
 *
 * Usage: npx tsx scripts/treasury_dts_gate1.ts
 * Prints a JSON verdict to stdout. Result goes in research/experiments.md
 * + datacore/signal_ladder.json — this script touches no runtime state.
 */
import { parseDts, sumTgaDepositsExDebt, type DtsRow } from "../server/treasuryDts";

const DTS_URL =
  "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/deposits_withdrawals_operating_cash";
const MTS_URL =
  "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_1";
const PASS_R = 0.85;
const MIN_MONTHS = 6;

const MONTH_NUM: Record<string, number> = {
  October: 10, November: 11, December: 12, January: 1, February: 2, March: 3,
  April: 4, May: 5, June: 6, July: 7, August: 8, September: 9,
};

async function fetchJson(url: string): Promise<any> {
  const r = await fetch(url, { signal: AbortSignal.timeout(30000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return JSON.parse(await r.text());
}

/** The newest DTS record_date on or before a calendar month's last day,
 *  returned only if it actually falls IN that month (else null — a
 *  month with no business days recorded yet is skipped, not guessed). */
async function lastBusinessDayOfMonth(yyyyMm: string): Promise<string | null> {
  const [y, m] = yyyyMm.split("-").map(Number);
  const lastDay = new Date(Date.UTC(y, m, 0)).toISOString().slice(0, 10);
  const url = `${DTS_URL}?filter=record_date:lte:${lastDay}&sort=-record_date&page%5Bsize%5D=1`;
  const j = await fetchJson(url);
  const d = j?.data?.[0]?.record_date;
  return typeof d === "string" && d.startsWith(yyyyMm) ? d : null;
}

async function dtsDepositsForDay(day: string): Promise<DtsRow[]> {
  const url = `${DTS_URL}?filter=record_date:eq:${day}&page%5Bsize%5D=250`;
  return parseDts(await fetchJson(url), day);
}

interface MtsMonth { month: string; receiptsUsd: number; }

/** One mts_table_1 row, verbatim FiscalData field names (values are
 *  strings, including the literal "null", as the API returns them). */
interface MtsRawRow {
  parent_id: string | null;
  classification_id: string;
  classification_desc: string;
  current_month_gross_rcpt_amt: string | null;
}

/** Walks mts_table_1's FY-parent/month-child rows for every fiscal year
 *  present in the latest published report (usually the current FY plus
 *  the prior FY's comparative column — this doubles the sample size for
 *  free, not by choice of window). */
async function mtsMonthlyReceipts(): Promise<MtsMonth[]> {
  const latest = await fetchJson(`${MTS_URL}?sort=-record_date&page%5Bsize%5D=1`);
  const recordDate = latest?.data?.[0]?.record_date;
  if (!recordDate) return [];
  const all = await fetchJson(`${MTS_URL}?filter=record_date:eq:${recordDate}&page%5Bsize%5D=40&sort=src_line_nbr`);
  const rows: MtsRawRow[] = all?.data || [];
  const fyParents = rows.filter((r) => r.parent_id === "null" || r.parent_id == null);
  const out: MtsMonth[] = [];
  for (const fy of fyParents) {
    const fyNum = parseInt(String(fy.classification_desc).replace(/\D/g, ""), 10);
    if (!fyNum) continue;
    for (const r of rows) {
      if (r.parent_id !== fy.classification_id) continue;
      const monthName = r.classification_desc;
      if (!(monthName in MONTH_NUM)) continue; // "Year-to-Date" rows excluded
      const amt = r.current_month_gross_rcpt_amt;
      if (amt == null || amt === "null") continue;
      const mnum = MONTH_NUM[monthName];
      const calYear = mnum >= 10 ? fyNum - 1 : fyNum; // FY starts in October of (FY-1)
      out.push({ month: `${calYear}-${String(mnum).padStart(2, "0")}`, receiptsUsd: parseFloat(amt) });
    }
  }
  // A month can appear in two adjacent FY reports' comparative columns; keep one.
  const dedup = new Map<string, MtsMonth>();
  for (const m of out) dedup.set(m.month, m);
  return Array.from(dedup.values());
}

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  const cov = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / n;
  const sx = Math.sqrt(xs.reduce((a, x) => a + (x - mx) ** 2, 0) / n);
  const sy = Math.sqrt(ys.reduce((a, y) => a + (y - my) ** 2, 0) / n);
  return cov / (sx * sy);
}

async function main() {
  const mts = await mtsMonthlyReceipts();
  if (!mts.length) {
    console.log(JSON.stringify({ verdict: "ERROR", reason: "mtsMonthlyReceipts returned zero months" }, null, 2));
    process.exitCode = 1;
    return;
  }
  const points: { month: string; dtsUsd: number; mtsUsd: number; ratio: number }[] = [];
  for (const { month, receiptsUsd } of mts.sort((a, b) => a.month.localeCompare(b.month))) {
    const day = await lastBusinessDayOfMonth(month);
    if (!day) continue;
    const rows = await dtsDepositsForDay(day);
    if (!rows.length) continue;
    const dtsUsd = sumTgaDepositsExDebt(rows) * 1_000_000; // rows are in $ millions
    points.push({ month, dtsUsd, mtsUsd: receiptsUsd, ratio: dtsUsd / receiptsUsd });
  }

  if (points.length < MIN_MONTHS) {
    console.log(JSON.stringify({ verdict: "ERROR", reason: `only ${points.length} reconcilable months, need >= ${MIN_MONTHS}`, points }, null, 2));
    process.exitCode = 1;
    return;
  }

  const r = pearson(points.map((p) => p.dtsUsd), points.map((p) => p.mtsUsd));
  const ratios = points.map((p) => p.ratio);
  const ratioMean = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  const ratioStdev = Math.sqrt(ratios.reduce((a, x) => a + (x - ratioMean) ** 2, 0) / ratios.length);
  const verdict = r >= PASS_R ? "PASS" : "FAIL";

  console.log(JSON.stringify({
    verdict,
    passBar: PASS_R,
    n: points.length,
    pearson_r: r,
    ratio_mean: ratioMean,
    ratio_stdev: ratioStdev,
    ratio_min: Math.min(...ratios),
    ratio_max: Math.max(...ratios),
    points,
  }, null, 2));
  process.exitCode = verdict === "PASS" ? 0 : 1;
}

main();
