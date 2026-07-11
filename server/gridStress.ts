/**
 * gridStress.ts — TX/ERCOT grid-stress DESCRIPTIVE dashboard surface
 * (GRID VISION A1 gate-2 FAIL path, research/grid_vision_products.md
 * "A1 gate-2 RESULT" section; research/experiments.md 2026-07-07 v1.0.190).
 *
 * WHY THIS EXISTS, NOT A SIGNAL: the v0 stress index (demand percentile x
 * forecast strain x weather extremity) was gate-2 tested against two
 * pre-stated outcome definitions and VOIDED on both — see
 * scripts/grid_stress_gate2.py + datacore/gridvision/gate2_result.json.
 * The design's own FAIL path (locked before computation) says the index
 * "demotes to a DESCRIPTIVE dashboard surface labeled non-predictive" —
 * still a product, honestly framed, per Amendment 5c ("premium
 * presentation of wrong numbers is fraud with good typography"). This
 * module ships exactly that: three raw descriptive ingredients plus an
 * EQUAL-WEIGHTED composite (deliberately NOT the gate-2 fitted weights,
 * which were fit against an outcome variable that turned out to be wrong
 * — reusing them here would smuggle a voided claim back in under a new
 * name). No field in this module's output may be read as predictive,
 * tradable, or sellable; `predictive: false` ships on every response.
 *
 * Data sources (both already recording for other reasons — no new
 * collection here, this is a pure read + join, same boundary rule as
 * entityGraph.ts): the griddemand archive (server/gridDemand.ts, ERCO
 * rows only) for demand + forecast strain, and the git-versioned CPC TX
 * degree-day archive (datacore/cpc/degree_days.json) for weather
 * extremity. Percentiles are computed against the FULL same-calendar-
 * month history currently on disk — this is a live descriptive reading,
 * not a train/valid backtest split (that split is gate-2's job, already
 * run and already voided).
 *
 * Memory: streams the griddemand archive file-by-file (readline, gz-
 * aware) exactly like shadowFleet.ts's foldVesselArchiveAsync — the OOM
 * lesson from that repair applies here too. Retained state is bounded to
 * one small {peak, strainSum, strainCount} record per ARCHIVE DAY (a few
 * thousand entries even at full depth), never per-row.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir } from "./datacoreArchive";
import { gridDemandEnabled, RESPONDENTS } from "./gridDemand";
import degreeDaysJson from "../datacore/cpc/degree_days.json";

const RESPONDENT = "ERCO"; // the one region with a per-region weather join (TX CPC degree days)
const MIN_SAMPLE_DAYS = 5; // below this, a same-month percentile is honestly withheld, not guessed

/** Human labels for the EIA-930 respondent codes we surface. */
export const REGION_LABEL: Record<string, string> = {
  US48: "United States (48)", ERCO: "Texas (ERCOT)", CISO: "California (CAISO)",
  MISO: "Midcontinent (MISO)", PJM: "Mid-Atlantic (PJM)", SWPP: "Central (SPP)",
  NYIS: "New York (NYISO)", ISNE: "New England (ISO-NE)", FPL: "Florida (FPL)",
  SE: "Southeast", NW: "Northwest", SW: "Southwest",
};

export interface DailyDemandAgg {
  peak: number | null;
  strainSum: number;
  strainCount: number;
}

/** Streaming fold over the griddemand archive for a SET of respondents, in a
 *  single pass. Returns respondent -> (day -> agg). One file = one UTC
 *  observation day (archiveDemand groups by o.period day), so per-period
 *  pairing never crosses a file boundary. Memory stays bounded to one small
 *  agg record per (respondent, day) — never per-row (the shadowFleet OOM
 *  lesson). Passing no set folds every configured RESPONDENT. */
export async function foldRegionsDailyAsync(
  respondents: readonly string[] = RESPONDENTS, baseDir?: string,
): Promise<Map<string, Map<string, DailyDemandAgg>>> {
  const want = new Set(respondents);
  const dir = path.join(baseDir || archiveBaseDir(), "griddemand");
  const result = new Map<string, Map<string, DailyDemandAgg>>();
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)).sort();
  } catch {
    return result;
  }
  const readOne = (f: string) => new Promise<void>((resolve) => {
    const day = f.slice(0, 10);
    // per respondent for THIS day: period -> {d, df}
    const perResp = new Map<string, Map<string, { d?: number; df?: number }>>();
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(path.join(dir, f));
      if (f.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (line) => {
        if (!line) return;
        try {
          const o = JSON.parse(line);
          if (!want.has(o.respondent) || o.mwh == null) return;
          const type = o.type || "D";
          if (type !== "D" && type !== "DF") return;
          let pp = perResp.get(o.respondent);
          if (!pp) { pp = new Map(); perResp.set(o.respondent, pp); }
          let e = pp.get(o.period);
          if (!e) { e = {}; pp.set(o.period, e); }
          if (type === "D") e.d = o.mwh; else e.df = o.mwh;
        } catch { /* skip malformed line */ }
      });
      rl.on("close", () => {
        perResp.forEach((pp, resp) => {
          let peak: number | null = null;
          let strainSum = 0, strainCount = 0;
          pp.forEach((e) => {
            if (e.d != null) {
              if (peak === null || e.d > peak) peak = e.d;
              if (e.df) { strainSum += (e.d - e.df) / e.df; strainCount++; }
            }
          });
          if (peak !== null) {
            let m = result.get(resp);
            if (!m) { m = new Map(); result.set(resp, m); }
            m.set(day, { peak, strainSum, strainCount });
          }
        });
        resolve();
      });
      stream.on("error", () => resolve());
    } catch {
      resolve();
    }
  });
  await files.reduce((p, f) => p.then(() => readOne(f)), Promise.resolve());
  return result;
}

/** ERCO-only daily fold (backward-compatible wrapper over the general fold). */
export async function foldErcoDailyAsync(baseDir?: string): Promise<Map<string, DailyDemandAgg>> {
  return (await foldRegionsDailyAsync([RESPONDENT], baseDir)).get(RESPONDENT) ?? new Map();
}

/** TX HDD+CDD per date from the committed CPC archive (2016-01-01+). */
export function loadTxDegreeDays(json: any = degreeDaysJson): Map<string, number> {
  const out = new Map<string, number>();
  const axes = json?.axes || {};
  const series = json?.series || {};
  for (const kind of ["H", "C"]) {
    const dates: string[] = axes[kind] || [];
    const vals: Array<number | null> = series[`TX|${kind}`] || [];
    dates.forEach((dt, i) => {
      const v = vals[i];
      if (v != null) out.set(dt, (out.get(dt) || 0) + v);
    });
  }
  return out;
}

/** Fraction of sortedVals <= x, i.e. same convention as
 *  scripts/grid_stress_gate2.py's pct_rank — 0..1. */
export function pctRank(sortedVals: number[], x: number): number | null {
  if (!sortedVals.length) return null;
  let lo = 0, hi = sortedVals.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedVals[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo / sortedVals.length;
}

const monthOf = (iso: string) => Number(iso.slice(5, 7));

/** Same-calendar-month percentile of `value` at `targetIso`, against every
 *  OTHER day in `values` sharing that month (any year available). Returns
 *  {percentile 0-100, sampleDays} — percentile null when sampleDays is too
 *  thin to mean anything (MIN_SAMPLE_DAYS), never a guessed number. */
function monthPercentile(values: Map<string, number>, targetIso: string):
    { percentile: number | null; sampleDays: number } {
  const m = monthOf(targetIso);
  const peers = Array.from(values.entries())
    .filter(([iso]) => iso !== targetIso && monthOf(iso) === m)
    .map(([, v]) => v)
    .sort((a, b) => a - b);
  const x = values.get(targetIso);
  if (x == null || peers.length < MIN_SAMPLE_DAYS) return { percentile: null, sampleDays: peers.length };
  const r = pctRank(peers, x);
  return { percentile: r == null ? null : Math.round(r * 100), sampleDays: peers.length };
}

export interface GridStressReading {
  date: string;
  respondent: string;
  demand_peak_mw: number | null;
  demand_percentile: number | null;
  demand_sample_days: number;
  forecast_strain_pct: number | null;
  strain_percentile: number | null;
  strain_sample_days: number;
  weather_degree_days: number | null;
  weather_percentile: number | null;
  weather_sample_days: number;
  composite_percentile: number | null;
}

/** Picks the most recent day with a complete ERCO peak reading, walking
 *  back from "yesterday UTC" (today's file is still accumulating) up to
 *  `lookbackDays` — a short EIA publication gap is normal, not a bug. */
function pickTargetDay(daily: Map<string, DailyDemandAgg>, nowMs: number, lookbackDays = 5): string | null {
  for (let i = 1; i <= lookbackDays; i++) {
    const iso = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    if (daily.has(iso)) return iso;
  }
  return null;
}

/** Build one region's descriptive reading from its daily agg map. `dd` is the
 *  per-region weather (TX degree days) or null where we have no weather join —
 *  in that case weather + composite are honestly null (never guessed), and the
 *  region still shows demand percentile + forecast strain. */
export function readingFromDaily(
  respondent: string, daily: Map<string, DailyDemandAgg>,
  dd: Map<string, number> | null, nowMs: number,
): GridStressReading | null {
  const targetIso = pickTargetDay(daily, nowMs);
  if (!targetIso) return null;
  const target = daily.get(targetIso)!;
  const peakByDay = new Map<string, number>();
  const strainByDay = new Map<string, number>();
  daily.forEach((v, iso) => {
    if (v.peak != null) peakByDay.set(iso, v.peak);
    if (v.strainCount > 0) strainByDay.set(iso, v.strainSum / v.strainCount);
  });
  const targetStrain = target.strainCount > 0 ? target.strainSum / target.strainCount : null;
  const targetDd = dd ? (dd.get(targetIso) ?? null) : null;

  const demandPct = monthPercentile(peakByDay, targetIso);
  const strainPct = targetStrain == null
    ? { percentile: null, sampleDays: 0 }
    : monthPercentile(strainByDay, targetIso);
  const weatherPct = (dd && targetDd != null) ? monthPercentile(dd, targetIso) : { percentile: null, sampleDays: 0 };

  // Composite = equal-weighted mean of the AVAILABLE descriptive percentiles
  // (deliberately NOT the voided gate-2 fitted weights). ERCO has all three;
  // regions without a weather join composite over demand+strain only.
  const parts = [demandPct.percentile, strainPct.percentile, weatherPct.percentile].filter(
    (p): p is number => p != null,
  );
  const composite = parts.length >= 2 ? Math.round(parts.reduce((a, b) => a + b, 0) / parts.length) : null;

  return {
    date: targetIso,
    respondent,
    demand_peak_mw: target.peak,
    demand_percentile: demandPct.percentile,
    demand_sample_days: demandPct.sampleDays,
    forecast_strain_pct: targetStrain == null ? null : Math.round(targetStrain * 1000) / 10,
    strain_percentile: strainPct.percentile,
    strain_sample_days: strainPct.sampleDays,
    weather_degree_days: targetDd,
    weather_percentile: weatherPct.percentile,
    weather_sample_days: weatherPct.sampleDays,
    composite_percentile: composite,
  };
}

export async function computeGridStressReading(baseDir?: string, nowMs?: number):
    Promise<{ reading: GridStressReading | null; history_depth_days: number }> {
  const now = nowMs ?? Date.now();
  const daily = await foldErcoDailyAsync(baseDir);
  const reading = readingFromDaily(RESPONDENT, daily, loadTxDegreeDays(), now);
  return { reading, history_depth_days: daily.size };
}

/** All-regions descriptive readings in a single archive pass. ERCO carries the
 *  TX weather join; every other region reports demand percentile + forecast
 *  strain only (weather/composite honestly null). Descriptive, non-predictive
 *  for every region — same banner ERCOT already shows. */
export async function computeAllRegionsStress(baseDir?: string, nowMs?: number):
    Promise<{ regions: GridStressReading[]; history_depth_days: number }> {
  const now = nowMs ?? Date.now();
  const byResp = await foldRegionsDailyAsync(RESPONDENTS, baseDir);
  const txDd = loadTxDegreeDays();
  const regions: GridStressReading[] = [];
  let depth = 0;
  for (const resp of RESPONDENTS) {
    const daily = byResp.get(resp);
    if (!daily || daily.size === 0) continue;
    depth = Math.max(depth, daily.size);
    const r = readingFromDaily(resp, daily, resp === RESPONDENT ? txDd : null, now);
    if (r) regions.push(r);
  }
  return { regions, history_depth_days: depth };
}

// ── Cache + poll (mirrors gridDemand.ts's pattern; a full archive fold is
// too heavy for the request path — recomputed periodically instead) ────────

export function gridStressEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return gridDemandEnabled(env); // same EIA-930 archive dependency
}

interface GridStressCache {
  at: number;
  reading: GridStressReading | null;     // ERCO (backward compat for the existing view)
  regions: GridStressReading[];          // all regions incl. ERCO (national overview)
  history_depth_days: number;
}

let cache: GridStressCache | null = null;
let polling = false;

export function latestGridStress(): GridStressCache | null {
  return cache;
}

export async function refreshGridStress(baseDir?: string, nowMs?: number): Promise<void> {
  try {
    if (!gridStressEnabled()) return;
    const { regions, history_depth_days } = await computeAllRegionsStress(baseDir, nowMs);
    const reading = regions.find((r) => r.respondent === RESPONDENT) ?? null;
    cache = { at: Date.now(), reading, regions, history_depth_days };
  } catch (e: any) {
    console.error("[datacore] gridstress refresh:", e?.message || e);
  }
}

/** 6h cadence: inputs (daily archive files, weekly CPC refresh) change far
 *  slower than that — no reason to re-fold the whole archive more often. */
export function bootGridStressPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshGridStress().catch((e) => console.error("[datacore] gridstress boot:", e?.message || e));
  setInterval(() => { refreshGridStress(); }, intervalMs).unref?.();
}
