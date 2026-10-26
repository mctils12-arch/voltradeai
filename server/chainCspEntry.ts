// chainCspEntry.ts — per underlying per archived day, the cash-secured put
// options_execution._select_sell_put would have SOLD, read off the options
// chain archive — research test 2, gate 3 ([RULE-REVIEW] measurement,
// 2026-10-26). The entry half of the manager-rule simulation.
//
// WHY A SEPARATE ENTRY TABLE: the chain archive snapshots each day's
// ROTATING CSP-candidate list plus the positions open at snapshot time, so a
// single name has no run of consecutive archived days (NVDA: 2 days in three
// months, 133 live fills) and the same contract cannot be marked to market
// daily from the archive. Gate 3 therefore splits: this module derives, per
// name-day, the contract the bot would have sold with its archived quote and
// greeks (one row per underlying-day, cached like iv30), and
// scripts/csp_manager_rule_sim.py marks each entry forward from the
// underlying's bars with a LABELLED model mark. Nothing here is a price
// claim: the feed is indicative and so labelled on every row.
//
// SELECTION (each constant pinned to the Python source by chainCspEntry.test.ts):
//  - puts only, expiry DTE within [CSP_SIM_DTE_MIN, CSP_SIM_DTE_MAX] — the live
//    fetch floor CSP_ENTRY_MIN_DTE = max(T1_TARGET_DTE_MIN, DTE_EXIT_THRESHOLD+1)
//    and _fetch_option_chain's ceiling of now + 50 days;
//  - liquidity as options_execution._is_liquid: bid > 0 and ask > 0,
//    (ask − bid) / mid ≤ MAX_SPREAD_PCT, mid ≥ 0.10, and OI ≥ MIN_OPEN_INTEREST
//    OR both quote sizes ≥ 10. The archive carries no daily VOLUME, so
//    _is_liquid's "volume ≥ MIN_OPTION_VOLUME" branch cannot be applied here —
//    a contract that would have passed on volume alone is MISSED (stated;
//    counted in nLiquid vs nPuts);
//  - OTM/ATM as |delta| ≤ 0.5: the live filter is strike ≤ spot, and the
//    archive carries no spot; a put with |delta| ≤ 0.5 is at or out of the
//    money by definition. Rows without a usable delta are skipped (the live
//    strike-distance fallback needs spot);
//  - pick: the put whose delta is nearest CSP_SIM_TARGET_DELTA; ties → higher OI.
// SPOT PROXY: `spotImplied` is the strike at which put |delta| crosses 0.5
// within the chosen expiry, linearly interpolated — a labelled proxy the
// simulation uses only as a cross-check against the underlying's real bar.
//
// DERIVED CACHE: a day's rows are written once under
// optionchains/derived/csp_entry/<day>.json — only for days strictly before
// the current UTC day. Source files are never modified.

import fs from "fs";
import path from "path";
import { parseOcc, chainDir, type ChainRecord } from "./optionsChainArchive";
import { archiveBaseDir } from "./datacoreArchive";
import { readChainDay } from "./chainIv";

export const CSP_SIM_TARGET_DELTA = -0.30;   // options_execution._select_sell_put target_delta
export const CSP_SIM_DTE_MIN = 30;           // options_execution.CSP_ENTRY_MIN_DTE
export const CSP_SIM_DTE_MAX = 50;           // _fetch_option_chain: max_exp = now + 50 days
export const CSP_SIM_MIN_OI = 200;           // options_execution.MIN_OPEN_INTEREST
export const CSP_SIM_MAX_SPREAD_PCT = 0.10;  // options_execution.MAX_SPREAD_PCT
export const CSP_SIM_MIN_MID = 0.10;         // _is_liquid price floor
export const CSP_SIM_MIN_QUOTE_SIZE = 10;    // _is_liquid last-resort size test
/** uncached days computed per probe call — the rest is reported as pending */
export const CSP_ENTRY_MAX_COMPUTE_PER_CALL = 15;

export type CspEntryReason = "ok" | "no_puts" | "no_liquid" | "no_delta" | "no_otm";

export interface CspEntryRow {
  day: string;
  u: string;
  occ: string | null;
  exp: string | null;
  dte: number | null;
  strike: number | null;
  bid: number | null;
  ask: number | null;
  mid: number | null;
  spreadPct: number | null;
  iv: number | null;
  delta: number | null;
  oi: number | null;
  /** strike where put |delta| crosses 0.5 in the chosen expiry (interpolated); a labelled proxy, not a price */
  spotImplied: number | null;
  /** puts seen for the underlying inside the DTE window */
  nPuts: number;
  /** of which passed the liquidity test */
  nLiquid: number;
  reason: CspEntryReason;
  feed: "indicative";
}

const DAY_MS = 86_400_000;
const r2 = (x: number) => Math.round(x * 100) / 100;
const r4 = (x: number) => Math.round(x * 10_000) / 10_000;

/** options_execution._is_liquid, minus the daily-volume branch the archive cannot supply. */
export function isLiquid(r: Pick<ChainRecord, "bid" | "ask" | "oi" | "bsz" | "asz">): boolean {
  const bid = r.bid ?? 0, ask = r.ask ?? 0;
  if (!(bid > 0) || !(ask > 0)) return false;
  const mid = (bid + ask) / 2;
  if ((ask - bid) / mid > CSP_SIM_MAX_SPREAD_PCT) return false;
  if (mid < CSP_SIM_MIN_MID) return false;
  if ((r.oi ?? 0) >= CSP_SIM_MIN_OI) return true;
  return (r.bsz ?? 0) >= CSP_SIM_MIN_QUOTE_SIZE && (r.asz ?? 0) >= CSP_SIM_MIN_QUOTE_SIZE;
}

interface PutRow { r: ChainRecord; exp: number; dte: number; strike: number }

function putsInWindow(records: readonly ChainRecord[], dayMs: number): PutRow[] {
  const out: PutRow[] = [];
  for (const r of records) {
    const occ = parseOcc(r.s);
    if (!occ || occ.type !== "P") continue;
    const dte = Math.round((occ.exp - dayMs) / DAY_MS);
    if (dte < CSP_SIM_DTE_MIN || dte > CSP_SIM_DTE_MAX) continue;
    out.push({ r, exp: occ.exp, dte, strike: occ.strike });
  }
  return out;
}

const usableDelta = (d: number | null): d is number => d !== null && Number.isFinite(d) && d < 0;

/** The strike at which put |delta| crosses 0.5 among `puts` of one expiry, or null without a bracket. */
export function impliedSpot(puts: readonly PutRow[]): number | null {
  const pts = puts
    .filter((p) => usableDelta(p.r.delta))
    .map((p) => ({ k: p.strike, a: Math.abs(p.r.delta as number) }))
    .sort((x, y) => x.k - y.k);
  for (const p of pts) if (p.a === 0.5) return r2(p.k);
  let lo: { k: number; a: number } | null = null;
  for (const p of pts) {
    if (p.a < 0.5) { lo = p; continue; }
    if (lo && p.a > lo.a) return r2(lo.k + ((0.5 - lo.a) / (p.a - lo.a)) * (p.k - lo.k));
    return null;
  }
  return null;
}

function emptyRow(day: string, u: string, nPuts: number, nLiquid: number, reason: CspEntryReason): CspEntryRow {
  return {
    day, u, occ: null, exp: null, dte: null, strike: null, bid: null, ask: null, mid: null, spreadPct: null,
    iv: null, delta: null, oi: null, spotImplied: null, nPuts, nLiquid, reason, feed: "indicative",
  };
}

/** The entry _select_sell_put would make for ONE underlying's records on `dayMs`. */
export function cspEntryFor(day: string, u: string, records: readonly ChainRecord[], dayMs: number): CspEntryRow {
  const puts = putsInWindow(records, dayMs);
  if (!puts.length) return emptyRow(day, u, 0, 0, "no_puts");
  const liquid = puts.filter((p) => isLiquid(p.r));
  if (!liquid.length) return emptyRow(day, u, puts.length, 0, "no_liquid");
  const withDelta = liquid.filter((p) => usableDelta(p.r.delta));
  if (!withDelta.length) return emptyRow(day, u, puts.length, liquid.length, "no_delta");
  const otm = withDelta.filter((p) => Math.abs(p.r.delta as number) <= 0.5);
  if (!otm.length) return emptyRow(day, u, puts.length, liquid.length, "no_otm");
  otm.sort((a, b) => {
    const da = Math.abs((a.r.delta as number) - CSP_SIM_TARGET_DELTA), db = Math.abs((b.r.delta as number) - CSP_SIM_TARGET_DELTA);
    return da !== db ? da - db : (b.r.oi ?? 0) - (a.r.oi ?? 0);
  });
  const best = otm[0];
  const bid = best.r.bid as number, ask = best.r.ask as number, mid = (bid + ask) / 2;
  return {
    day, u, occ: best.r.s, exp: new Date(best.exp).toISOString().slice(0, 10), dte: best.dte, strike: r2(best.strike),
    bid: r4(bid), ask: r4(ask), mid: r4(mid), spreadPct: r4((ask - bid) / mid),
    iv: best.r.iv === null ? null : r4(best.r.iv), delta: r4(best.r.delta as number), oi: best.r.oi,
    spotImplied: impliedSpot(puts.filter((p) => p.exp === best.exp)),
    nPuts: puts.length, nLiquid: liquid.length, reason: "ok", feed: "indicative",
  };
}

/** One row per underlying present in the day's records, sorted by underlying. */
export function cspEntryForDay(records: readonly ChainRecord[], day: string): CspEntryRow[] {
  const dayMs = Date.parse(`${day}T00:00:00Z`);
  const byU = new Map<string, ChainRecord[]>();
  for (const r of records) {
    const u = String(r.u || "").toUpperCase();
    if (!u) continue;
    const arr = byU.get(u);
    if (arr) arr.push(r); else byU.set(u, [r]);
  }
  const rows: CspEntryRow[] = [];
  for (const [u, recs] of byU) rows.push(cspEntryFor(day, u, recs, dayMs));
  return rows.sort((a, b) => a.u.localeCompare(b.u));
}

// ── derived cache (same shape and rules as chainIv.iv30ForDays) ───────────

export function derivedDir(baseDir?: string): string {
  return path.join(chainDir(baseDir ?? archiveBaseDir()), "derived", "csp_entry");
}

function readDerived(fp: string): CspEntryRow[] | null {
  if (!fs.existsSync(fp)) return null;
  try {
    const v: unknown = JSON.parse(fs.readFileSync(fp, "utf8"));
    return Array.isArray(v) ? (v as CspEntryRow[]) : null;
  } catch (e) {
    console.warn(`[chainCspEntry] derived file unreadable, recomputing ${fp}:`, e instanceof Error ? e.message : String(e));
    return null;
  }
}

const utcDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/** Entry rows for `days`, from the derived cache where a day is already
 *  derived, else computed from the archive — at most `maxCompute` uncached
 *  days per call (the rest come back in `pending`). A finished day (strictly
 *  before the current UTC day) is cached write-once. */
export async function cspEntryForDays(
  days: readonly string[], baseDir?: string, nowMs = Date.now(), maxCompute = CSP_ENTRY_MAX_COMPUTE_PER_CALL,
): Promise<{ rows: CspEntryRow[]; computed: string[]; cached: string[]; pending: string[] }> {
  const dir = derivedDir(baseDir);
  const today = utcDay(nowMs);
  const rows: CspEntryRow[] = [];
  const computed: string[] = [], cached: string[] = [], pending: string[] = [];
  for (const day of days) {
    const fp = path.join(dir, `${day}.json`);
    const derived = readDerived(fp);
    if (derived) { rows.push(...derived); cached.push(day); continue; }
    if (computed.length >= maxCompute) { pending.push(day); continue; }
    const dayRows = cspEntryForDay(await readChainDay(day, baseDir), day);
    rows.push(...dayRows);
    computed.push(day);
    if (day < today) {
      try { fs.mkdirSync(dir, { recursive: true }); fs.writeFileSync(fp, JSON.stringify(dayRows)); }
      catch (e) { console.error(`[chainCspEntry] cache write failed for ${day}:`, e instanceof Error ? e.message : String(e)); }
    }
  }
  return { rows, computed, cached, pending };
}
