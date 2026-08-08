/**
 * finra_shortvol_gate2_retest.ts — GATE 2 (SIGNAL layer) RETEST for the
 * finra_short_volume root, running the FOLLOW-UP HYPOTHESIS the first-pass
 * run pre-registered (research/open_questions.md, FINRA DAILY SHORT-SALE
 * VOLUME entry, "UPDATE 2026-08-06" block) rather than re-fishing a new
 * design.
 *
 * WHY A RETEST, NOT A NEW HYPOTHESIS: finra_shortvol_gate2.ts (2026-08-06)
 * pre-registered a 3-bucket monotonic-ordering design (HIGH_SHORT >
 * NEUTRAL > LOW_SHORT, or the reverse) and FAILED it — NEUTRAL (the
 * middle-rank-by-short-ratio bucket) underperformed BOTH extremes at both
 * horizons, a U-shape. That run's own filed follow-up flagged the likely
 * cause: the middle-RANK bucket is not a fair "what would random entry
 * have returned" control (Reasoning Standard #3) — rank-based selection
 * can itself carry a composition confound ("boring middle" names in this
 * specific window) distinct from short-pressure. This script keeps the
 * SIGNAL side of the original test completely UNCHANGED (identical
 * HIGH_SHORT definition, identical sample window, identical liquidity
 * floor) and changes exactly the two things the follow-up pre-registered:
 *   (a) replace the rank-based NEUTRAL control with a random-entry
 *       baseline sample drawn from the full qualifying population
 *       (Reasoning Standard #3's actual "random entry" framing, not a
 *       literal every-ticker fetch — see NETWORK COST NOTE below);
 *   (b) regime-split the result (Reasoning Standard #2) using a simple,
 *       stated-upfront SPY-trend classifier — NOT the live bot's 5-level
 *       Markov regime classifier (out of scope to replicate in a
 *       standalone script; this is a coarser, explicitly-labeled proxy).
 *
 * NETWORK COST NOTE (Reasoning Standard #6-adjacent, an honesty caveat
 * not a fudge): the FLOOR_TOTAL_VOL qualifying population is ~2,000-2,400
 * tickers/day (confirmed via --dry run of the original script). A literal
 * full-population fetch across 16 sample days would be several thousand
 * unique Yahoo price series — likely feasible but far more network cost
 * than any prior gate-2 script in this codebase attempted (occ_volume_
 * gate2.ts and the original finra_shortvol_gate2.ts both explicitly
 * bounded cost to ~1,500 unique tickers). Rather than invent a new
 * unbounded cost budget mid-session, this script uses a RANDOM sample —
 * the open_questions.md follow-up itself wrote the bar as "FULL-
 * POPULATION mean/random-entry baseline", treating the two as equivalent
 * operationalizations of the same base-rate control. A uniform random
 * sample is an UNBIASED estimator of the population mean regardless of
 * sample size (only the standard error shrinks with more draws), so this
 * is a legitimate substitution, not a weaker test — stated here so a
 * future session can raise BASELINE_SIZE if it wants a tighter CI.
 *
 * OPERATIONALIZATION (pre-registered, stated BEFORE running):
 *   - HIGH_SHORT: UNCHANGED from finra_shortvol_gate2.ts — top BUCKET_SIZE
 *     (40) qualifying tickers by short_ratio (short_vol/total_vol),
 *     qualifying = total_vol >= FLOOR_TOTAL_VOL (imported, not redefined).
 *   - BASELINE: BASELINE_SIZE (80 — 2x HIGH_SHORT, still network-bounded)
 *     tickers drawn UNIFORMLY AT RANDOM (no replacement) from the
 *     qualifying population EXCLUDING the HIGH_SHORT names (keeps the two
 *     buckets disjoint, same discipline as the original LOW_SHORT/NEUTRAL
 *     construction). Randomness is SEEDED (RANDOM_SEED, a fixed constant
 *     chosen from today's date before any data was looked at) via a
 *     per-day derived seed, so a re-run reproduces the identical baseline
 *     — this is not "keep re-rolling until it looks right" fishing.
 *   - Forward return: identical to the original script — close-to-close
 *     from the FINRA trade date to +5/+20 TRADING days later (Yahoo
 *     adjusted close), reusing fetchYahooDaily/toSeries/fwdReturn.
 *   - Sample: IDENTICAL window — 16 weekly Wednesdays, 2026-01-07 through
 *     2026-04-22. This is a methodology correction on the SAME period,
 *     not a new out-of-sample window — a true disjoint replication (same
 *     two-stage discipline the OCC/CFTC-TFF candidates used: fix
 *     methodology first, then test a disjoint window) is the explicit
 *     NEXT step if this passes, not attempted here.
 *   - STATISTICAL METHOD: day-clustered t-test (clusterMeanTTest, reused
 *     verbatim from statsUtils.ts) on the per-day (HIGH_SHORT mean -
 *     BASELINE mean) spread — same anti-fishing discipline as the
 *     original script, not a naive pooled per-row test.
 *   - REGIME CLASSIFICATION (simple, stated upfront): SPY trailing
 *     20-trading-day return as of each sample day. UP = >=0, DOWN = <0.
 *     This is a coarse two-way proxy, NOT the live bot's 5-level Markov
 *     regime classifier (regime_util.classify_regime_5level) — replicating
 *     that machinery in a standalone TS research script is out of scope;
 *     stated as a scope simplification per the follow-up's own language
 *     ("verified against roughly BULL/CAUTION texture").
 *   - PASS bar (pre-stated, falsifiable, PRIMARY test — no fishing):
 *     day-clustered |t| on the POOLED (all 16 days, both regimes) +20d
 *     HIGH_SHORT-baseline spread > tCrit005(df). Direction is NOT assumed
 *     (two-tailed), matching the original script's stated ambiguity
 *     between the "informed shorts" and "squeeze reversal" literatures.
 *   - Regime split is DIAGNOSTIC/EXPLORATORY ONLY, not a second pass/fail
 *     gate: with only 16 total days, an UP/DOWN split leaves ~6-10 days
 *     per regime — too few clusters for a reliable independent bar. It is
 *     reported for DIRECTIONAL CONSISTENCY (does the pooled result hold
 *     up in both slices, or is it entirely carried by one regime — which
 *     would itself be the finding, per Reasoning Standard #2) rather than
 *     scored pass/fail. Stated here BEFORE running so a "regime X passes,
 *     regime Y doesn't" result cannot later be spun as if regime X alone
 *     had been the pre-registered bar.
 *   - PRIOR stated before running: ~30% chance of a clean POOLED pass
 *     (lower than the original script's 35% prior — the first pass's
 *     result was itself disappointing/ambiguous, and a fair random
 *     baseline removes whatever composition effect inflated the original
 *     HIGH-LOW spread's significance). A null result here would mean the
 *     original run's HIGH-LOW significance was substantially a NEUTRAL-
 *     bucket artifact, not evidence FOR the pooled test failing to find
 *     anything at all — that distinction is itself informative and will
 *     be stated plainly in the write-up either way.
 *
 * Session-run: `npx tsx scripts/finra_shortvol_gate2_retest.ts [--dry]`
 * --dry skips the Yahoo price fetch and only prints bucket sizes, to size
 * network cost before committing to the full run.
 * Result goes in research/experiments.md + datacore/signal_ladder.json,
 * never into any production code path — this script touches no runtime
 * state.
 */
import { pathToFileURL } from "url";
import { fetchShortVolDay, FLOOR_TOTAL_VOL, type ShortVolRow } from "../server/finraShortVolume";
import { weeklySampleDays, ymdCompact, fetchYahooDaily, toSeries, fwdReturn, type Series } from "./occ_volume_gate2";
import { clusterMeanTTest, tCrit005, survivesAtCrit005 } from "./statsUtils";

export const BUCKET_SIZE = 40;
export const BASELINE_SIZE = 80;
export const HORIZONS = [5, 20] as const;
export const SAMPLE_START = "2026-01-07";
export const SAMPLE_WEEKS = 16;
export const RANDOM_SEED = 20260808; // chosen from today's date, before any data was inspected
const DRY = process.argv.includes("--dry");

export interface ShortRatioRow { ticker: string; short_ratio: number; total_vol: number; }
export interface RetestDayBuckets {
  day: string;
  qualifying: number;
  highShort: ShortRatioRow[];
  baseline: ShortRatioRow[];
}

// mulberry32 — small deterministic PRNG so the "random" baseline draw is
// reproducible across reruns (no Math.random(), which would make this
// script's own output non-reproducible and therefore un-auditable).
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Derive a per-day seed from RANDOM_SEED + the day string so each sample
// day gets an independent-looking but fully reproducible draw.
export function daySeed(day: string, baseSeed = RANDOM_SEED): number {
  let h = baseSeed >>> 0;
  for (let i = 0; i < day.length; i++) {
    h = (Math.imul(h, 31) + day.charCodeAt(i)) >>> 0;
  }
  return h;
}

/** Fisher-Yates using a seeded PRNG — deterministic, unbiased shuffle. */
export function seededShuffle<T>(arr: T[], rng: () => number): T[] {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Bucket one day's FINRA rows: HIGH_SHORT unchanged from the original
 *  script; BASELINE is a seeded random sample of the remaining qualifying
 *  population (disjoint from HIGH_SHORT). Pure function, no I/O. */
export function bucketDayRetest(day: string, rows: ShortVolRow[]): RetestDayBuckets {
  const qualifying: ShortRatioRow[] = rows
    .filter((r) => r.total_vol >= FLOOR_TOTAL_VOL)
    .map((r) => ({ ticker: r.symbol, short_ratio: r.short_vol / r.total_vol, total_vol: r.total_vol }))
    .sort((a, b) => b.short_ratio - a.short_ratio); // descending: highest short ratio first

  const n = qualifying.length;
  const highSize = Math.min(BUCKET_SIZE, Math.floor(n / 3) || n);
  const highShort = qualifying.slice(0, highSize);
  const highTickers = new Set(highShort.map((r) => r.ticker));
  const remainder = qualifying.filter((r) => !highTickers.has(r.ticker));

  const baselineSize = Math.min(BASELINE_SIZE, remainder.length);
  const rng = mulberry32(daySeed(day));
  const baseline = seededShuffle(remainder, rng).slice(0, baselineSize);

  return { day, qualifying: n, highShort, baseline };
}

export type RegimeLabel = "UP" | "DOWN";

/** Classify a sample day by SPY's trailing 20-trading-day return as of
 *  that day. Coarse two-way proxy — see header for why this is not the
 *  live bot's 5-level regime classifier. */
export function classifyRegime(spySeries: Series, day: string): RegimeLabel | null {
  const idx = spySeries.indexOf.get(day);
  if (idx == null || idx < 20) return null;
  const now = spySeries.closes[idx];
  const then = spySeries.closes[idx - 20];
  if (!Number.isFinite(now) || !Number.isFinite(then) || then === 0) return null;
  return now / then - 1 >= 0 ? "UP" : "DOWN";
}

async function main() {
  const days = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  console.log(`finra_shortvol_gate2_retest ${DRY ? "(--dry calibration run)" : "(full run)"}: ${days.length} weekly sample days, FLOOR_TOTAL_VOL=${FLOOR_TOTAL_VOL}, BUCKET_SIZE=${BUCKET_SIZE}, BASELINE_SIZE=${BASELINE_SIZE}, RANDOM_SEED=${RANDOM_SEED}`);

  const dayBuckets: RetestDayBuckets[] = [];
  let skippedDays = 0;
  for (const day of days) {
    const rows = await fetchShortVolDay(ymdCompact(day));
    if (rows === null || rows.length === 0) {
      console.log(`  ${day}: skipped (no data / transport error) — honest no-op`);
      skippedDays++;
      continue;
    }
    const b = bucketDayRetest(day, rows);
    dayBuckets.push(b);
    console.log(`  ${b.day}: qualifying=${b.qualifying} high=${b.highShort.length} baseline=${b.baseline.length}`);
    await new Promise((res) => setTimeout(res, 400));
  }

  const tickerSet = new Set<string>();
  for (const b of dayBuckets) {
    for (const r of [...b.highShort, ...b.baseline]) tickerSet.add(r.ticker);
  }
  tickerSet.add("SPY");
  console.log(`\ndays_with_data=${dayBuckets.length}/${days.length} (skipped=${skippedDays})`);
  console.log(`unique tickers needed for price lookup (incl. SPY)=${tickerSet.size}`);

  if (DRY) {
    console.log(`\n--dry: stopping before Yahoo fetch (${tickerSet.size} price series would be requested).`);
    return;
  }

  const firstDay = new Date(dayBuckets[0].day + "T00:00:00Z").getTime() / 1000;
  const lastDay = new Date(dayBuckets[dayBuckets.length - 1].day + "T00:00:00Z").getTime() / 1000;
  const startSec = firstDay - 60 * 86400; // extra lookback for SPY's 20d-trailing regime calc
  const endSec = lastDay + 45 * 86400;

  const seriesCache = new Map<string, Series>();
  let fetched = 0, failed = 0;
  const tickers = Array.from(tickerSet);
  const CONCURRENCY = 8;
  for (let i = 0; i < tickers.length; i += CONCURRENCY) {
    const batch = tickers.slice(i, i + CONCURRENCY);
    const results = await Promise.all(batch.map(async (t) => {
      try { return { t, map: await fetchYahooDaily(t, startSec, endSec) }; }
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

  const spySeries = seriesCache.get("SPY");
  if (!spySeries) {
    console.error("FATAL: SPY series unavailable — cannot regime-classify. Aborting.");
    process.exit(1);
  }

  type DayRow = { day: string; regime: RegimeLabel | null; high: Record<number, number | null>; baseline: Record<number, number | null>; spread: Record<number, number | null> };
  const dayRows: DayRow[] = [];

  for (const b of dayBuckets) {
    const regime = classifyRegime(spySeries, b.day);
    const row: DayRow = { day: b.day, regime, high: {}, baseline: {}, spread: {} };
    for (const h of HORIZONS) {
      const meanRet = (rows: ShortRatioRow[]): number | null => {
        const rets: number[] = [];
        for (const r of rows) {
          const s = seriesCache.get(r.ticker);
          if (!s) continue;
          const ret = fwdReturn(s, b.day, h);
          if (ret == null) continue;
          rets.push(ret);
        }
        return rets.length ? rets.reduce((a, x) => a + x, 0) / rets.length : null;
      };
      const highMean = meanRet(b.highShort);
      const baseMean = meanRet(b.baseline);
      row.high[h] = highMean;
      row.baseline[h] = baseMean;
      row.spread[h] = highMean != null && baseMean != null ? highMean - baseMean : null;
    }
    dayRows.push(row);
  }

  console.log(`\n=== RESULTS (HIGH_SHORT vs random-entry baseline, day-clustered) ===`);
  console.log(`regime labels: ${dayRows.map((r) => `${r.day}=${r.regime ?? "NA"}`).join(", ")}`);

  const runTest = (rows: DayRow[], label: string) => {
    console.log(`\n-- ${label} (n_days=${rows.length}) --`);
    for (const h of HORIZONS) {
      const spreads = rows.map((r) => r.spread[h]).filter((x): x is number => x != null);
      if (spreads.length < 2) {
        console.log(`  +${h}d: insufficient data (n=${spreads.length})`);
        continue;
      }
      const result = clusterMeanTTest(spreads);
      const crit = tCrit005(result.df);
      const survives = survivesAtCrit005(result);
      console.log(`  +${h}d: n=${result.n} df=${result.df} mean_spread=${(result.mean * 100).toFixed(3)}% t=${result.t.toFixed(3)} crit=${crit.toFixed(3)} SURVIVES=${survives}`);
    }
  };

  runTest(dayRows, "POOLED (primary pre-registered test)");
  runTest(dayRows.filter((r) => r.regime === "UP"), "REGIME=UP (SPY trailing 20d >= 0) — diagnostic only");
  runTest(dayRows.filter((r) => r.regime === "DOWN"), "REGIME=DOWN (SPY trailing 20d < 0) — diagnostic only");

  const pooled20 = dayRows.map((r) => r.spread[20]).filter((x): x is number => x != null);
  const pooledResult = clusterMeanTTest(pooled20);
  const verdict = survivesAtCrit005(pooledResult) ? "PASS" : "FAIL/INCONCLUSIVE";
  console.log(`\nVERDICT (pre-stated bar: pooled day-clustered |t20| > ${tCrit005(pooledResult.df).toFixed(3)}): ${verdict}`);
  console.log(`pooled_mean_spread_20d=${(pooledResult.mean * 100).toFixed(3)}% t20=${pooledResult.t.toFixed(3)} df=${pooledResult.df}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
