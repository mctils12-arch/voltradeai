// chainVs.ts — per-underlying option-implied volatility SPREAD (Cremers–
// Weinbaum), derived from the options chain archive (server/optionsChainArchive.ts)
// — Edge P3 shadow stream, research/edge_preregistration.md ([PIPELINE]
// 2026-10-30). Raw material for ladder gate 1 and, after ~52 weeks, gate 2;
// nothing here is read by a trading path.
//
// SIGNAL (as pre-registered): VS(u,d) = mean over matched pairs j of
// (iv_call_j − iv_put_j), a pair being a call and a put with the SAME strike
// and expiry, both iv present and > 0, expiry VS_MIN_DTE..VS_MAX_DTE, call
// delta inside VS_CALL_DELTA_BAND. Needs ≥ VS_MIN_PAIRS pairs, else null —
// never guessed. Positive = calls rich vs puts. Unit: IV as a fraction
// (0.01 = one vol point), same as chainIv.
//
// DESCRIPTIVE COLUMNS (logged, never tested — the XZZ smirk was rejected as a
// signal in the pre-registration): atmIv = mean call iv over pairs whose call
// delta is in VS_ATM_BAND; smirk = mean put iv with put delta in
// VS_SMIRK_PUT_BAND minus atmIv (Xing–Zhang–Zhao OTM-put minus ATM-call).
//
// GATE-1 RISK, MEASURED NOT ASSUMED: Alpaca's indicative option quotes are
// derived values, and a derivation that imposes put–call parity would make VS
// ≈ 0 by construction. `vsGate1Summary` reports, per day, the cross-sectional
// sd of VS and the share of days under the pre-registered kill bar (sd under
// 0.5 vol points on ≥ 80% of days kills the DATASET, not the idea). It is a
// summary of this module's own output; the real-quote rank-correlation check
// (second kill clause) needs an external sample and stays open.
//
// DAY LABEL: `day` is the chain file's label (the ET trading date since the
// 2026-10-21 capture fix; older files carry the one-session shift documented
// in audit 2026-10-27 §2.4). It is carried as-is, not re-derived.
//
// DERIVED CACHE: a day's rows are written once under
// optionchains/derived/vs/<day>.json — only for days strictly before the
// current UTC day (same rule as chainIv / chainCspEntry). Source files are
// never modified.

import fs from "fs";
import path from "path";
import { parseOcc, chainDir, type ChainRecord } from "./optionsChainArchive";
import { archiveBaseDir } from "./datacoreArchive";
import { readChainDay } from "./chainIv";

export const VS_MIN_DTE = 14;
export const VS_MAX_DTE = 60;
export const VS_CALL_DELTA_BAND: readonly [number, number] = [0.25, 0.75];
export const VS_MIN_PAIRS = 3;
export const VS_ATM_BAND: readonly [number, number] = [0.45, 0.55];
export const VS_SMIRK_PUT_BAND: readonly [number, number] = [0.125, 0.375]; // |put delta|
/** pre-registered gate-1 kill bar: cross-sectional sd of VS under this, on ≥ VS_KILL_DAY_SHARE of days */
export const VS_KILL_SD = 0.005;
export const VS_KILL_DAY_SHARE = 0.8;
/** a day needs at least this many underlyings with a VS to count toward the sd */
export const VS_MIN_NAMES_FOR_SD = 10;
/** uncached days computed per probe call — the rest is reported as pending */
export const VS_MAX_COMPUTE_PER_CALL = 15;

export interface VsRow {
  day: string;
  u: string;
  vs: number | null;
  nPairs: number;
  atmIv: number | null;
  smirk: number | null;
}

const DAY_MS = 86_400_000;
const r4 = (x: number) => Math.round(x * 10_000) / 10_000;
const mean = (a: readonly number[]) => a.reduce((s, v) => s + v, 0) / a.length;

/** VS row for ONE underlying's records on `dayMs` (the day's UTC midnight).
 *  Duplicate rows (an archive retry) collapse: one row per (expiry, strike, side). */
export function vsFor(day: string, u: string, records: readonly ChainRecord[], dayMs: number): VsRow {
  const legs = new Map<string, { exp: number; call?: ChainRecord; put?: ChainRecord }>();
  for (const r of records) {
    if (r.iv === null || !(r.iv > 0) || r.delta === null || !Number.isFinite(r.delta)) continue;
    const occ = parseOcc(r.s);
    if (!occ) continue;
    const dte = Math.round((occ.exp - dayMs) / DAY_MS);
    if (dte < VS_MIN_DTE || dte > VS_MAX_DTE) continue;
    if ((r.delta > 0) !== (occ.type === "C")) continue; // a call with negative delta (or put with positive) is a bad row
    const key = `${occ.exp}|${occ.strike}`;
    let leg = legs.get(key);
    if (!leg) { leg = { exp: occ.exp }; legs.set(key, leg); }
    if (occ.type === "C") leg.call = r; else leg.put = r;
  }
  const spreads: number[] = [], atm: number[] = [], otmPuts: number[] = [];
  const [cLo, cHi] = VS_CALL_DELTA_BAND, [aLo, aHi] = VS_ATM_BAND, [pLo, pHi] = VS_SMIRK_PUT_BAND;
  for (const leg of legs.values()) {
    const c = leg.call, p = leg.put;
    if (c && c.delta! >= aLo && c.delta! <= aHi) atm.push(c.iv!);
    if (p && Math.abs(p.delta!) >= pLo && Math.abs(p.delta!) <= pHi) otmPuts.push(p.iv!);
    if (c && p && c.delta! >= cLo && c.delta! <= cHi) spreads.push(c.iv! - p.iv!);
  }
  const atmIv = atm.length ? mean(atm) : null;
  return {
    day, u,
    vs: spreads.length >= VS_MIN_PAIRS ? r4(mean(spreads)) : null,
    nPairs: spreads.length,
    atmIv: atmIv === null ? null : r4(atmIv),
    smirk: atmIv !== null && otmPuts.length ? r4(mean(otmPuts) - atmIv) : null,
  };
}

/** One row per underlying present in the day's records, sorted by underlying. */
export function vsForDay(records: readonly ChainRecord[], day: string): VsRow[] {
  const dayMs = Date.parse(`${day}T00:00:00Z`);
  const byU = new Map<string, ChainRecord[]>();
  for (const r of records) {
    const u = String(r.u || "").toUpperCase();
    if (!u) continue;
    const arr = byU.get(u);
    if (arr) arr.push(r); else byU.set(u, [r]);
  }
  const rows: VsRow[] = [];
  for (const [u, recs] of byU) rows.push(vsFor(day, u, recs, dayMs));
  return rows.sort((a, b) => a.u.localeCompare(b.u));
}

export interface VsGate1Summary {
  days: number;
  /** days with ≥ VS_MIN_NAMES_FOR_SD names carrying a VS */
  daysScored: number;
  daysUnderKillSd: number;
  shareUnderKillSd: number | null;
  /** true when the pre-registered dataset-kill condition holds (needs ≥ 1 scored day) */
  datasetKill: boolean;
  medianCrossSectionSd: number | null;
  perDay: Array<{ day: string; names: number; sd: number | null; meanVs: number | null }>;
}

/** Cross-sectional sd of VS per day and the pre-registered gate-1 kill check. */
export function vsGate1Summary(rows: readonly VsRow[]): VsGate1Summary {
  const byDay = new Map<string, number[]>();
  for (const r of rows) {
    const arr = byDay.get(r.day) ?? [];
    if (r.vs !== null) arr.push(r.vs);
    byDay.set(r.day, arr);
  }
  const perDay = [...byDay.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([day, vals]) => {
    if (vals.length < VS_MIN_NAMES_FOR_SD) return { day, names: vals.length, sd: null, meanVs: vals.length ? r4(mean(vals)) : null };
    const m = mean(vals);
    const sd = Math.sqrt(vals.reduce((s, v) => s + (v - m) * (v - m), 0) / (vals.length - 1));
    return { day, names: vals.length, sd: r4(sd), meanVs: r4(m) };
  });
  const scored = perDay.filter((d) => d.sd !== null) as Array<{ sd: number }>;
  const under = scored.filter((d) => d.sd < VS_KILL_SD).length;
  const sds = scored.map((d) => d.sd).sort((a, b) => a - b);
  const share = scored.length ? under / scored.length : null;
  return {
    days: perDay.length, daysScored: scored.length, daysUnderKillSd: under,
    shareUnderKillSd: share === null ? null : r4(share),
    datasetKill: share !== null && share >= VS_KILL_DAY_SHARE,
    medianCrossSectionSd: sds.length ? sds[Math.floor((sds.length - 1) / 2)] : null,
    perDay,
  };
}

// ── derived cache (same shape and rules as chainIv.iv30ForDays) ───────────

export function derivedDir(baseDir?: string): string {
  return path.join(chainDir(baseDir ?? archiveBaseDir()), "derived", "vs");
}

function readDerived(fp: string): VsRow[] | null {
  if (!fs.existsSync(fp)) return null;
  try {
    const v: unknown = JSON.parse(fs.readFileSync(fp, "utf8"));
    return Array.isArray(v) ? (v as VsRow[]) : null;
  } catch (e) {
    console.warn(`[chainVs] derived file unreadable, recomputing ${fp}:`, e instanceof Error ? e.message : String(e));
    return null;
  }
}

const utcDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/** VS rows for `days`, from the derived cache where a day is already
 *  derived, else computed from the archive — at most `maxCompute` uncached
 *  days per call (the rest come back in `pending`). A finished day (strictly
 *  before the current UTC day) is cached write-once. */
export async function vsForDays(
  days: readonly string[], baseDir?: string, nowMs = Date.now(), maxCompute = VS_MAX_COMPUTE_PER_CALL,
): Promise<{ rows: VsRow[]; computed: string[]; cached: string[]; pending: string[] }> {
  const dir = derivedDir(baseDir);
  const today = utcDay(nowMs);
  const rows: VsRow[] = [];
  const computed: string[] = [], cached: string[] = [], pending: string[] = [];
  for (const day of days) {
    const fp = path.join(dir, `${day}.json`);
    const derived = readDerived(fp);
    if (derived) { rows.push(...derived); cached.push(day); continue; }
    if (computed.length >= maxCompute) { pending.push(day); continue; }
    const dayRows = vsForDay(await readChainDay(day, baseDir), day);
    rows.push(...dayRows);
    computed.push(day);
    if (day < today) {
      try { fs.mkdirSync(dir, { recursive: true }); fs.writeFileSync(fp, JSON.stringify(dayRows)); }
      catch (e) { console.error(`[chainVs] cache write failed for ${day}:`, e instanceof Error ? e.message : String(e)); }
    }
  }
  return { rows, computed, cached, pending };
}
