// tiles3dBudget — cost guard for Google Photorealistic 3D Tiles (worldview-globe
// G3). The BILLABLE unit is the ROOT request (`GET tile.googleapis.com/v1/3dtiles/
// root.json`): each one starts a session (Enterprise SKU) that buys >=3h of FREE
// child-tile fetches. Free tier = 1,000 root requests / month, then $6.00/1,000.
// The runaway vector is page loads / reloads / React remounts each issuing a fresh
// root request. This module is the PURE, TESTABLE guard — same discipline as
// scripts/runpod_budget.py for GPU: authorize before spending, ledger every root,
// refuse past a hard cap so 3D Tiles can never run away.
//
// It does NOT fetch Google or hold the key — the proxy route does that and MUST
// call authorizeRoot() first and recordRoot() on success. Child-tile fetches are
// FREE within the session window and are NOT ledgered here.
import fs from "node:fs";
import path from "node:path";

export const FREE_ROOTS_PER_MONTH = 1000;
// Daily cap keeps us inside the free tier by construction: 1000/mo ≈ 33/day.
// A day that needs more is almost certainly a remount/reload loop, not real use.
export const DAILY_ROOT_CAP = 33;
// Hard monthly backstop at the free-tier ceiling (never silently roll into paid).
export const MONTHLY_ROOT_CAP = FREE_ROOTS_PER_MONTH;

export interface RootRow { day: string; month: string; at: string; note?: string }

const LEDGER = path.join(process.cwd(), "datacore", "tiles3d", "root_ledger.jsonl");

/** UTC calendar day "YYYY-MM-DD" for a timestamp (ms). Pure. */
export function utcDay(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}
/** UTC month "YYYY-MM" for a timestamp (ms). Pure. */
export function utcMonth(ms: number): string {
  return new Date(ms).toISOString().slice(0, 7);
}

/** Count ledgered root requests for a given UTC day. Pure. */
export function countForDay(rows: RootRow[], day: string): number {
  return rows.reduce((n, r) => (r && r.day === day ? n + 1 : n), 0);
}
/** Count ledgered root requests for a given UTC month. Pure. */
export function countForMonth(rows: RootRow[], month: string): number {
  return rows.reduce((n, r) => (r && r.month === month ? n + 1 : n), 0);
}

export interface RootDecision {
  authorized: boolean;
  reason: "ok" | "daily_cap" | "monthly_cap";
  detail: string;
  today: number;
  month: number;
  daily_remaining: number;
  monthly_remaining: number;
  day: string;
}

/**
 * Decide whether a NEW root request (= a billable 3D-tiles session) may be made.
 * Refuses at the daily cap (remount/reload runaway guard) and at the monthly
 * free-tier ceiling (never silently spend). Pure — the caller passes the ledger
 * rows + now(ms).
 */
export function authorizeRoot(
  rows: RootRow[],
  nowMs: number,
  opts: { dailyCap?: number; monthlyCap?: number } = {},
): RootDecision {
  const dailyCap = opts.dailyCap ?? DAILY_ROOT_CAP;
  const monthlyCap = opts.monthlyCap ?? MONTHLY_ROOT_CAP;
  const day = utcDay(nowMs);
  const month = utcMonth(nowMs);
  const today = countForDay(rows, day);
  const monthN = countForMonth(rows, month);
  const daily_remaining = Math.max(0, dailyCap - today);
  const monthly_remaining = Math.max(0, monthlyCap - monthN);
  const base = { today, month: monthN, daily_remaining, monthly_remaining, day };
  if (monthN >= monthlyCap) {
    return { authorized: false, reason: "monthly_cap",
      detail: `monthly free-tier ceiling reached (${monthN}/${monthlyCap} root requests) — 3D Tiles paused until next month or a raised cap`, ...base };
  }
  if (today >= dailyCap) {
    return { authorized: false, reason: "daily_cap",
      detail: `daily root-request cap reached (${today}/${dailyCap}) — likely a reload/remount loop; 3D Tiles paused until tomorrow UTC`, ...base };
  }
  return { authorized: true, reason: "ok",
    detail: `authorized — ${today}/${dailyCap} today, ${monthN}/${monthlyCap} this month`, ...base };
}

/** Build the row for a successful root request. Pure. */
export function makeRootRow(nowMs: number, note?: string): RootRow {
  return { day: utcDay(nowMs), month: utcMonth(nowMs), at: new Date(nowMs).toISOString(), note };
}

// ---- thin file I/O (append-only JSONL; git keeps vintages) -----------------
export function loadLedger(file = LEDGER): RootRow[] {
  try {
    if (!fs.existsSync(file)) return [];
    return fs.readFileSync(file, "utf8").split("\n").filter(Boolean).map((l) => {
      try { return JSON.parse(l) as RootRow; } catch { return null; }
    }).filter(Boolean) as RootRow[];
  } catch { return []; }
}

export function recordRoot(nowMs: number, note?: string, file = LEDGER): RootRow {
  const row = makeRootRow(nowMs, note);
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.appendFileSync(file, JSON.stringify(row) + "\n");
  } catch { /* ledger write best-effort — never block the response on disk */ }
  return row;
}

/** Status summary for a /status route or health surface. */
export function budgetStatus(rows: RootRow[], nowMs: number) {
  const d = authorizeRoot(rows, nowMs);
  return {
    daily_cap: DAILY_ROOT_CAP, monthly_cap: MONTHLY_ROOT_CAP,
    today: d.today, month: d.month,
    daily_remaining: d.daily_remaining, monthly_remaining: d.monthly_remaining,
    would_authorize: d.authorized, day: d.day,
  };
}
