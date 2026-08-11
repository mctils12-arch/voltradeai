/**
 * finra_shortvol_gate2_retest.ts — RETEST of the finra_short_volume root's
 * GATE 2 (SIGNAL) hypothesis, per the follow-up pre-registered by the
 * 2026-08-06 first-pass run (research/open_questions.md's FINRA DAILY
 * SHORT-SALE VOLUME entry, UPDATE 2026-08-06).
 *
 * WHY A RETEST, NOT A NEW ROOT: the first pass (scripts/finra_shortvol_
 * gate2.ts) used a fixed-size 3-way HIGH_SHORT/NEUTRAL/LOW_SHORT bucket
 * design and required monotonic ordering across all three. It found a
 * significant +20d HIGH-LOW spread (day-clustered t=2.279 > crit=2.131)
 * but FAILED the ordering requirement: NEUTRAL underperformed LOW_SHORT at
 * both horizons — a U-shape, not a gradient. The diagnosed cause: NEUTRAL
 * was a fixed middle-ranked bucket that may over-index on "boring middle"
 * names unrelated to short pressure, not a genuine random-entry baseline.
 * That entry pre-registered two fixes for a future retest: (a) test
 * HIGH_SHORT against a full-population baseline instead of a 3-way
 * ordering design, and (b) regime-split. This script is that retest —
 * same root, same sample window (for direct regime-texture comparability,
 * Reasoning Standard #2), genuinely different statistical design (not a
 * fished variant of the killed one).
 *
 * DESIGN CHANGE FROM THE FIRST PASS (pre-registered here, BEFORE running,
 * Reasoning Standard #10):
 *
 *   - POPULATION BASELINE, NOT A NEUTRAL BUCKET. Every day's qualifying
 *     universe (total_vol >= FLOOR_TOTAL_VOL, same floor as the first
 *     pass — reused from server/finraShortVolume.ts, not redefined) is
 *     split into HIGH_SHORT (top BUCKET_SIZE=40 by short_ratio, same size
 *     as the first pass — imported from finra_shortvol_gate2.ts, not
 *     redefined) and "the rest" (everything else that met the liquidity
 *     floor). HONEST LIMITATION: fetching a full-population forward-return
 *     series for every qualifying ticker (typically 1,000+/day at this
 *     floor) is not network-feasible in a research-script sandbox run, so
 *     the population mean is estimated from a SEEDED RANDOM SAMPLE of
 *     POPULATION_SAMPLE_SIZE=40 tickers drawn from "the rest" (same
 *     per-day network cost order as the first pass's NEUTRAL bucket, but
 *     now a genuine random draw across the FULL remaining range instead of
 *     a fixed middle-ranked slice — this is the actual fix for the
 *     diagnosed "boring middle" composition confound, not a cosmetic
 *     rename). The RNG seed (RNG_SEED below) is fixed and stated before
 *     any data is fetched — one draw per day, in day order, no re-rolling.
 *   - NO MONOTONIC-ORDERING REQUIREMENT. With two groups (HIGH_SHORT vs.
 *     population-sample) instead of three, "ordering" isn't a meaningful
 *     concept — the pass bar is a single two-tailed day-clustered t-test on
 *     the HIGH-vs-population spread, same statistical machinery
 *     (scripts/statsUtils.ts's clusterMeanTTest, reused not reimplemented)
 *     as the first pass, at the same pre-declared more-theory-relevant
 *     horizon (+20d). +5d reported informationally only, same as first pass.
 *   - DIRECTION: still stated as genuinely ambiguous (see the first pass's
 *     header for the two competing literatures) — two-tailed test, no
 *     directional assumption.
 *   - REGIME SPLIT: each sample day gets a real 5-level regime label
 *     (PANIC/BEAR/CAUTION/NEUTRAL/BULL) computed by backtest_v2.py's own
 *     regime_series() — the EXACT function the production backtest engine
 *     uses (imported live via a python3 subprocess call, not reimplemented
 *     in TS — a second copy of the VXX/SPY threshold logic would be a
 *     third place these thresholds could drift, which regime_util.py's own
 *     header already names as the historical bug this repo fixed once).
 *     This directly answers the first pass's own open question ("Jan-Apr
 *     2026 window was Sunday-fungible only if verified against roughly
 *     BULL/CAUTION texture, not re-verified for this run") with real
 *     labels instead of an assumption. Regime-subgroup t-tests (when any
 *     label has >=5 sample days) are reported as INFORMATIONAL ONLY, not
 *     part of the formal pass bar — with n=16 days total, per-regime
 *     subgroups are too thin to trust as a second confirmatory test
 *     (Reasoning Standard #4); they exist to describe the window's texture,
 *     not to fish for a subgroup that passes when the whole-sample test
 *     doesn't.
 *
 * PRE-REGISTERED PASS BAR (single test, stated before running): day-
 * clustered two-tailed |t| > tCrit005(df) on the +20d HIGH-SHORT-minus-
 * population-sample spread. PRIOR: same order as the first pass's own
 * (~35%), NOT re-elevated by the first pass's near-miss — Reasoning
 * Standard #4 says a near-miss on a now-abandoned design does not raise
 * the prior for a materially different design.
 *
 * Session-run: `npx tsx scripts/finra_shortvol_gate2_retest.ts [--dry]`
 * --dry stops after the FINRA fetch + population/regime data shaping,
 * before the Yahoo price fetch, to size network cost first.
 * Result goes in research/experiments.md + datacore/signal_ladder.json,
 * never into any runtime path — this script touches no production state.
 */
import { pathToFileURL } from "url";
import { execFileSync } from "child_process";
import { fetchShortVolDay, FLOOR_TOTAL_VOL, type ShortVolRow } from "../server/finraShortVolume";
import { weeklySampleDays, ymdCompact, fetchYahooDaily, toSeries, fwdReturn, type Series } from "./occ_volume_gate2";
import { SAMPLE_START, SAMPLE_WEEKS, BUCKET_SIZE, HORIZONS } from "./finra_shortvol_gate2";
import { clusterMeanTTest, tCrit005, survivesAtCrit005 } from "./statsUtils";

export const POPULATION_SAMPLE_SIZE = 40;
export const RNG_SEED = 20260811; // today's session date — fixed before any data was fetched
const DRY = process.argv.includes("--dry");

export interface ShortRatioRow { ticker: string; short_ratio: number; total_vol: number; }
export interface DaySplit { day: string; qualifying: ShortRatioRow[]; highShort: ShortRatioRow[]; }

/** Split one day's FINRA rows into the qualifying universe (liquidity-
 *  floored) and its top-ranked HIGH_SHORT bucket. Pure function, no I/O —
 *  unit-testable on synthetic rows. */
export function splitPopulationAndTop(day: string, rows: ShortVolRow[]): DaySplit {
  const qualifying: ShortRatioRow[] = rows
    .filter((r) => r.total_vol >= FLOOR_TOTAL_VOL)
    .map((r) => ({ ticker: r.symbol, short_ratio: r.short_vol / r.total_vol, total_vol: r.total_vol }))
    .sort((a, b) => b.short_ratio - a.short_ratio); // descending: most heavily shorted first

  const n = qualifying.length;
  const bucketSize = Math.max(1, Math.min(BUCKET_SIZE, Math.floor(n / 3)));
  const highShort = qualifying.slice(0, bucketSize);
  return { day, qualifying, highShort };
}

// Deterministic seeded PRNG (mulberry32) — reproducible, not re-rollable,
// so the population sample can't be quietly re-drawn to chase a result.
export function mulberry32(seed: number): () => number {
  let a = seed;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleWithoutReplacement<T>(arr: T[], k: number, rng: () => number): T[] {
  const pool = arr.slice();
  const out: T[] = [];
  const take = Math.min(k, pool.length);
  for (let i = 0; i < take; i++) {
    const idx = Math.floor(rng() * pool.length);
    out.push(pool[idx]);
    pool.splice(idx, 1);
  }
  return out;
}

export function computeSpread(popRets: number[], topRets: number[]): { popMean: number; topMean: number; spread: number } | null {
  if (!popRets.length || !topRets.length) return null;
  const popMean = popRets.reduce((a, b) => a + b, 0) / popRets.length;
  const topMean = topRets.reduce((a, b) => a + b, 0) / topRets.length;
  return { popMean, topMean, spread: topMean - popMean };
}

// ── Regime bridge: reuse backtest_v2.py's own regime_series() verbatim via
// a subprocess call, instead of reimplementing the VXX/SPY thresholds a
// third time in TypeScript (regime_util.py's own header names that exact
// duplication as the historical bug this repo fixed once already). ──────
export interface RegimeSeriesResult { labels: string[]; quality: string; }

export function regimeLabelsFor(
  spy: { date: string[]; close: number[] },
  vxx: { date: string[]; close: number[] },
): RegimeSeriesResult {
  const py = `
import sys, json
from backtest_v2 import regime_series
data = json.load(sys.stdin)
labels, quality = regime_series(data["spy"], data["vxx"])
print(json.dumps({"labels": labels, "quality": quality}))
`;
  const out = execFileSync("python3", ["-c", py], {
    input: JSON.stringify({ spy, vxx }),
    maxBuffer: 32 * 1024 * 1024,
  }).toString();
  const lastLine = out.trim().split("\n").pop()!;
  return JSON.parse(lastLine);
}

async function main() {
  const days = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  console.log(`finra_shortvol_gate2_retest ${DRY ? "(--dry calibration run)" : "(full run)"}: ${days.length} weekly sample days (same window as the first pass), FLOOR_TOTAL_VOL=${FLOOR_TOTAL_VOL}, BUCKET_SIZE=${BUCKET_SIZE}, POPULATION_SAMPLE_SIZE=${POPULATION_SAMPLE_SIZE}, RNG_SEED=${RNG_SEED}`);

  const rng = mulberry32(RNG_SEED);
  const splits: DaySplit[] = [];
  let skippedDays = 0;
  for (const day of days) {
    const rows = await fetchShortVolDay(ymdCompact(day));
    if (rows === null || rows.length === 0) {
      console.log(`  ${day}: skipped (no data / transport error) — honest no-op`);
      skippedDays++;
      continue;
    }
    const split = splitPopulationAndTop(day, rows);
    splits.push(split);
    console.log(`  ${split.day}: qualifying=${split.qualifying.length} highShort=${split.highShort.length}`);
  }
  console.log(`\ndays_with_data=${splits.length}/${days.length} (skipped=${skippedDays})`);

  // Draw the population sample per day, sequentially in day order, from
  // "the rest" (qualifying minus the HIGH_SHORT names themselves) — one
  // deterministic pass through the seeded RNG, no re-rolling.
  const daySamples = new Map<string, ShortRatioRow[]>();
  for (const s of splits) {
    const highTickers = new Set(s.highShort.map((r) => r.ticker));
    const rest = s.qualifying.filter((r) => !highTickers.has(r.ticker));
    daySamples.set(s.day, sampleWithoutReplacement(rest, POPULATION_SAMPLE_SIZE, rng));
  }

  const tickerSet = new Set<string>();
  for (const s of splits) {
    for (const r of s.highShort) tickerSet.add(r.ticker);
    for (const r of daySamples.get(s.day) || []) tickerSet.add(r.ticker);
  }
  console.log(`unique tickers needed for price lookup=${tickerSet.size}`);

  if (DRY) {
    console.log(`\n--dry: stopping before Yahoo/regime fetches.`);
    return;
  }
  if (splits.length === 0) {
    console.log(`\nno sample days had data — aborting.`);
    return;
  }

  // ── Regime labels: fetch SPY + VXX with enough trailing history for a
  // real 200-day MA (backtest_v2.regime_series needs it for spy_above_200d/
  // spy_below_200_days), then hand off to the python bridge. ─────────────
  const firstDaySec = new Date(splits[0].day + "T00:00:00Z").getTime() / 1000;
  const lastDaySec = new Date(splits[splits.length - 1].day + "T00:00:00Z").getTime() / 1000;
  const regimeLookbackSec = 450 * 86400; // ~450 calendar days for a stable 200-trading-day MA
  const regimeStartSec = firstDaySec - regimeLookbackSec;
  const regimeEndSec = lastDaySec + 5 * 86400;

  const spySeries = toSeries(await fetchYahooDaily("SPY", regimeStartSec, regimeEndSec));
  const vxxSeries = toSeries(await fetchYahooDaily("VXX", regimeStartSec, regimeEndSec));
  console.log(`\nregime inputs: SPY ${spySeries.dates.length} days, VXX ${vxxSeries.dates.length} days (${spySeries.dates[0]}..${spySeries.dates[spySeries.dates.length - 1]})`);

  const regimeResult = regimeLabelsFor(
    { date: spySeries.dates, close: spySeries.closes },
    { date: vxxSeries.dates, close: vxxSeries.closes },
  );
  console.log(`regime_series data_quality=${regimeResult.quality}`);
  const regimeByDate = new Map(spySeries.dates.map((d, i) => [d, regimeResult.labels[i]]));

  const regimeForDay = new Map<string, string>();
  for (const s of splits) {
    const label = regimeByDate.get(s.day);
    regimeForDay.set(s.day, label ?? "UNKNOWN");
    if (!label) console.log(`  ${s.day}: no SPY regime label found (not a trading day in the fetched series?) — marked UNKNOWN`);
  }
  const regimeCounts: Record<string, number> = {};
  for (const label of regimeForDay.values()) regimeCounts[label] = (regimeCounts[label] || 0) + 1;
  console.log(`\nregime composition across ${splits.length} sample days: ${JSON.stringify(regimeCounts)}`);

  // ── Price fetch for every ticker in every day's HIGH_SHORT + population
  // sample (same batching pattern as the first pass / occ_volume_gate2). ─
  const firstSec = firstDaySec - 5 * 86400;
  const lastSec = lastDaySec + 45 * 86400;
  const seriesCache = new Map<string, Series>();
  let fetched = 0, failed = 0;
  const tickers = Array.from(tickerSet);
  const CONCURRENCY = 8;
  for (let i = 0; i < tickers.length; i += CONCURRENCY) {
    const batch = tickers.slice(i, i + CONCURRENCY);
    const results = await Promise.all(batch.map(async (t) => {
      try { return { t, map: await fetchYahooDaily(t, firstSec, lastSec) }; }
      catch { return { t, map: null as Map<string, number> | null }; }
    }));
    for (const { t, map } of results) {
      if (map) { seriesCache.set(t, toSeries(map)); fetched++; } else { failed++; }
    }
    if ((i + CONCURRENCY) % 80 === 0 || i + CONCURRENCY >= tickers.length) {
      console.log(`  priced ${Math.min(i + CONCURRENCY, tickers.length)}/${tickers.length} tickers (fetched=${fetched} failed=${failed})`);
    }
    await new Promise((res) => setTimeout(res, 250));
  }
  console.log(`\nprice fetch done: fetched=${fetched} failed=${failed}/${tickers.length}`);

  // ── Per-day, per-horizon spread (HIGH_SHORT mean fwd return minus
  // population-sample mean fwd return), day is the cluster. ─────────────
  interface DaySpreadRow { day: string; regime: string; spread: number; }
  const dayLevelSpreads: Record<number, DaySpreadRow[]> = { 5: [], 20: [] };

  for (const s of splits) {
    const sample = daySamples.get(s.day) || [];
    for (const h of HORIZONS) {
      const rets = (rows: ShortRatioRow[]) => {
        const out: number[] = [];
        for (const r of rows) {
          const series = seriesCache.get(r.ticker);
          if (!series) continue;
          const ret = fwdReturn(series, s.day, h);
          if (ret != null) out.push(ret);
        }
        return out;
      };
      const spreadResult = computeSpread(rets(sample), rets(s.highShort));
      if (spreadResult) {
        dayLevelSpreads[h].push({ day: s.day, regime: regimeForDay.get(s.day) || "UNKNOWN", spread: spreadResult.spread });
      }
    }
  }

  console.log(`\n=== RESULTS (HIGH_SHORT vs. population-random-sample, day-clustered) ===`);
  const clustered: Record<number, ReturnType<typeof clusterMeanTTest>> = {} as any;
  for (const h of HORIZONS) {
    const spreads = dayLevelSpreads[h].map((r) => r.spread);
    const result = clusterMeanTTest(spreads);
    clustered[h] = result;
    const crit = tCrit005(result.df);
    console.log(`+${h}d horizon (n_days=${result.n}, df=${result.df}, crit=${crit.toFixed(3)}): day-clustered mean spread=${(result.mean * 100).toFixed(3)}%  t=${result.t.toFixed(3)}  SURVIVES=${survivesAtCrit005(result)}`);
  }

  console.log(`\n=== REGIME SPLIT (informational only — n=16 total days, subgroups are NOT a second confirmatory test) ===`);
  const byRegime = new Map<string, DaySpreadRow[]>();
  for (const r of dayLevelSpreads[20]) {
    if (!byRegime.has(r.regime)) byRegime.set(r.regime, []);
    byRegime.get(r.regime)!.push(r);
  }
  for (const [label, rows] of byRegime) {
    const result = clusterMeanTTest(rows.map((r) => r.spread));
    const note = rows.length >= 5 ? "" : " (n<5, too thin to interpret even informationally)";
    console.log(`  ${label}: n_days=${rows.length} +20d mean spread=${(result.mean * 100).toFixed(3)}% t=${result.t.toFixed(3)}${note}`);
  }

  const t20 = clustered[20];
  const verdict = survivesAtCrit005(t20) ? "PASS" : "FAIL";
  console.log(`\nVERDICT (pre-stated bar: day-clustered |t20|>${tCrit005(t20.df).toFixed(3)}, two-tailed, single test): ${verdict}`);
  console.log(`clustered_t_20d=${t20.t.toFixed(3)} df=${t20.df} n_days=${t20.n}`);
}

// Entrypoint guard (ESM has no require.main) — same fix pattern as the
// first-pass script needed after the 2026-08-05 entrypoint-guard bug.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
