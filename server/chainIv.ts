// chainIv.ts — per-underlying ATM 30-day implied volatility, derived from the
// options chain archive (server/optionsChainArchive.ts) — research test 2,
// step 1 ([RULE-REVIEW] measurement, 2026-10-20).
//
// research/bot_logic_research_2026-10-20.md Q2 asks whether single-name
// short premium carries any variance risk premium at all (Driessen–Maenhout–
// Vilkov: zero individual VRP not rejected for most S&P 500 names) while the
// index does. Gate 2 (SIGNAL) needs, per underlying per day, the 30-day
// implied vol to compare with the variance subsequently realized. The chain
// archive has stored every contract's IV and delta once per trading day
// since 2026-07-04 (indicative feed — labeled, never dropped); this derives
// the series from it, server-side, so a session never has to pull 30k
// contract rows per day through the diag probe (which caps at 5,000 rows).
//
// METHOD (stated so it can be audited): ATM per expiry = the call with delta
// nearest +0.5 and the put with delta nearest −0.5, both inside
// IV30_DELTA_BAND, averaged (one side alone is accepted); expiries inside
// IV30_MIN_DTE are ignored (pin risk, stale quotes). IV30 = interpolation in
// TOTAL VARIANCE (σ²·T) between the expiries bracketing 30 calendar days;
// with no bracket, the nearest expiry inside IV30_NEAREST_DTE is used and
// labeled "nearest"; otherwise null — never guessed. Delta-ATM needs no
// spot, which the archive does not carry.
//
// DERIVED CACHE: a day's result is written once under
// optionchains/derived/iv30/<day>.json — only for days strictly before the
// current UTC day (a day file can still be appended by a retry until its
// day is over). Source files are never modified.

import fs from "fs";
import path from "path";
import { parseOcc, chainDir, type ChainRecord } from "./optionsChainArchive";
import { archiveDayFiles, streamJsonlLines } from "./datacoreArchive";

export const IV30_TARGET_DAYS = 30;
export const IV30_DELTA_BAND: readonly [number, number] = [0.35, 0.65];
export const IV30_NEAREST_DTE: readonly [number, number] = [20, 45];
export const IV30_MIN_DTE = 5;
export const IV30_MAX_DTE = 120;
/** uncached days computed per probe call — the rest is reported as pending */
export const IV30_MAX_COMPUTE_PER_CALL = 15;

export interface ExpiryAtm { dte: number; iv: number; n: number; call: number | null; put: number | null }
export interface Iv30Row {
  day: string;
  u: string;
  iv30: number | null;
  method: "interp" | "nearest" | null;
  dteLo: number | null;
  dteHi: number | null;
  /** in-band contracts seen for the underlying that day */
  n: number;
  expiries: number;
}

const DAY_MS = 86_400_000;
const r4 = (x: number) => Math.round(x * 10_000) / 10_000;

/** ATM IV per expiry (keyed by DTE) for ONE underlying's records on `dayMs`
 *  (the day's UTC midnight). Duplicate rows (an archive retry) are harmless:
 *  "nearest to 0.5" is idempotent. */
export function atmByExpiry(records: readonly ChainRecord[], dayMs: number): Map<number, ExpiryAtm> {
  const [lo, hi] = IV30_DELTA_BAND;
  const best = new Map<number, { call: [number, number] | null; put: [number, number] | null; n: number }>();
  for (const r of records) {
    if (r.iv === null || !(r.iv > 0) || r.delta === null || !Number.isFinite(r.delta)) continue;
    const occ = parseOcc(r.s);
    if (!occ) continue;
    const dte = Math.round((occ.exp - dayMs) / DAY_MS);
    if (dte < IV30_MIN_DTE || dte > IV30_MAX_DTE) continue;
    const a = Math.abs(r.delta);
    if (a < lo || a > hi) continue;
    const side = r.delta > 0 ? "call" : "put";
    if ((side === "call") !== (occ.type === "C")) continue; // a call with negative delta is a bad row
    let e = best.get(dte);
    if (!e) { e = { call: null, put: null, n: 0 }; best.set(dte, e); }
    e.n++;
    const dist = Math.abs(a - 0.5);
    const cur = e[side];
    if (!cur || dist < cur[0]) e[side] = [dist, r.iv];
  }
  const out = new Map<number, ExpiryAtm>();
  for (const [dte, e] of best) {
    const ivs = [e.call?.[1], e.put?.[1]].filter((v): v is number => typeof v === "number");
    if (!ivs.length) continue;
    out.set(dte, { dte, iv: ivs.reduce((s, v) => s + v, 0) / ivs.length, n: e.n, call: e.call?.[1] ?? null, put: e.put?.[1] ?? null });
  }
  return out;
}

/** 30-day IV from the per-expiry ATM points: total-variance interpolation
 *  between the bracketing expiries, else the nearest inside IV30_NEAREST_DTE. */
export function iv30From(expiries: ReadonlyMap<number, ExpiryAtm>): Pick<Iv30Row, "iv30" | "method" | "dteLo" | "dteHi"> {
  const dtes = [...expiries.keys()].sort((a, b) => a - b);
  const T = IV30_TARGET_DAYS;
  const none = { iv30: null, method: null, dteLo: null, dteHi: null } as const;
  if (!dtes.length) return none;
  const exact = expiries.get(T);
  if (exact) return { iv30: r4(exact.iv), method: "interp", dteLo: T, dteHi: T };
  const lo = dtes.filter((d) => d < T).pop();
  const hi = dtes.find((d) => d > T);
  if (lo !== undefined && hi !== undefined) {
    const a = expiries.get(lo)!, b = expiries.get(hi)!;
    const vLo = a.iv * a.iv * lo, vHi = b.iv * b.iv * hi;
    const v30 = vLo + (vHi - vLo) * ((T - lo) / (hi - lo));
    return { iv30: r4(Math.sqrt(Math.max(v30, 0) / T)), method: "interp", dteLo: lo, dteHi: hi };
  }
  const only = lo ?? hi!;
  const [nLo, nHi] = IV30_NEAREST_DTE;
  if (only >= nLo && only <= nHi) return { iv30: r4(expiries.get(only)!.iv), method: "nearest", dteLo: only, dteHi: only };
  return none;
}

/** One row per underlying present in the day's records, sorted by underlying. */
export function atmIv30ForDay(records: readonly ChainRecord[], day: string): Iv30Row[] {
  const dayMs = Date.parse(`${day}T00:00:00Z`);
  const byU = new Map<string, ChainRecord[]>();
  for (const r of records) {
    const u = String(r.u || "").toUpperCase();
    if (!u) continue;
    const arr = byU.get(u);
    if (arr) arr.push(r); else byU.set(u, [r]);
  }
  const rows: Iv30Row[] = [];
  for (const [u, recs] of byU) {
    const ex = atmByExpiry(recs, dayMs);
    let n = 0;
    for (const e of ex.values()) n += e.n;
    rows.push({ day, u, ...iv30From(ex), n, expiries: ex.size });
  }
  return rows.sort((a, b) => a.u.localeCompare(b.u));
}

// ── archive reading + derived cache ────────────────────────────────────────

export function derivedDir(baseDir?: string): string {
  return path.join(chainDir(baseDir), "derived", "iv30");
}

/** Days with a chain day-file, ascending ("YYYY-MM-DD"). */
export function listChainDays(baseDir?: string): string[] {
  let names: string[] = [];
  try { names = fs.readdirSync(chainDir(baseDir)); } catch { return []; }
  const days = new Set<string>();
  for (const f of names) {
    const m = /^(\d{4}-\d{2}-\d{2})\.jsonl(?:\.gz)?$/.exec(f);
    if (m) days.add(m[1]);
  }
  return [...days].sort();
}

/** The day's records (every file for the day, .jsonl or .gz), minimally validated. */
export async function readChainDay(day: string, baseDir?: string): Promise<ChainRecord[]> {
  const out: ChainRecord[] = [];
  for (const fp of archiveDayFiles(chainDir(baseDir), day)) {
    await streamJsonlLines(fp, fp.endsWith(".gz"), (line) => {
      let r: any;
      try { r = JSON.parse(line); } catch { return; }
      if (r && typeof r === "object" && typeof r.s === "string" && typeof r.u === "string") out.push(r as ChainRecord);
    });
  }
  return out;
}

const utcDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/** IV30 rows for `days`, from the derived cache where a day is already
 *  derived, else computed from the archive — at most `maxCompute` uncached
 *  days per call (the rest come back in `pending`). A finished day (strictly
 *  before the current UTC day) is cached write-once. */
export async function iv30ForDays(
  days: readonly string[], baseDir?: string, nowMs = Date.now(), maxCompute = IV30_MAX_COMPUTE_PER_CALL,
): Promise<{ rows: Iv30Row[]; computed: string[]; cached: string[]; pending: string[] }> {
  const dir = derivedDir(baseDir);
  const today = utcDay(nowMs);
  const rows: Iv30Row[] = [];
  const computed: string[] = [], cached: string[] = [], pending: string[] = [];
  for (const day of days) {
    const fp = path.join(dir, `${day}.json`);
    if (fs.existsSync(fp)) {
      try { rows.push(...(JSON.parse(fs.readFileSync(fp, "utf8")) as Iv30Row[])); cached.push(day); continue; } catch { /* recompute below */ }
    }
    if (computed.length >= maxCompute) { pending.push(day); continue; }
    const dayRows = atmIv30ForDay(await readChainDay(day, baseDir), day);
    rows.push(...dayRows);
    computed.push(day);
    if (day < today) {
      try { fs.mkdirSync(dir, { recursive: true }); fs.writeFileSync(fp, JSON.stringify(dayRows)); } catch (e: any) { console.error(`[chainIv] cache write failed for ${day}:`, e?.message || e); }
    }
  }
  return { rows, computed, cached, pending };
}
