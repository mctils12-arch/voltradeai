/**
 * finra_shortvol_gate2_retest.ts — the pre-registered FOLLOW-UP to
 * finra_shortvol_gate2.ts (2026-08-06, FAIL against a monotonic-ordering
 * bar). This is NOT a new hypothesis: it is the exact retest that FAIL's
 * own writeup pre-registered, filed in
 * research/open_questions.md "FINRA DAILY SHORT-SALE VOLUME UPDATE
 * 2026-08-06" and left unclaimed across four later sessions
 * (research/experiments.md:2880, 2996, 3173, 48428, 48668, 48915-48920,
 * 49063).
 *
 * WHAT THE FIRST RUN FOUND: a 3-bucket (HIGH_SHORT/NEUTRAL/LOW_SHORT) design
 * required monotonic ordering on both +5d and +20d. NEUTRAL underperformed
 * LOW_SHORT at BOTH horizons (a U-shape), even though the HIGH-LOW spread
 * alone cleared significance at +20d (day-clustered t=2.279 > crit=2.131).
 * The pre-registered follow-up named two specific, separate changes — this
 * script makes EXACTLY those two changes and nothing else:
 *
 *   (a) test HIGH_SHORT against a FULL-POPULATION baseline instead of a
 *       3-bucket ordering design that assumes monotonicity;
 *   (b) regime-split before trusting the raw HIGH-LOW contrast.
 *
 * Everything else — the universe (FLOOR_TOTAL_VOL), the HIGH_SHORT bucket
 * definition (BUCKET_SIZE=40, same rank construction), the sample window
 * (16 weekly Wednesdays, 2026-01-07..2026-04-22), and the forward-return
 * definition (+5d/+20d Yahoo adjusted close) — is IMPORTED from
 * finra_shortvol_gate2.ts, not retyped, so this run cannot silently drift
 * from "the same test, two ingredients changed" into "a different test"
 * (Reasoning Standard #10 discipline: change one thing, know which thing
 * you changed).
 *
 * PRE-REGISTERED CRITERIA (stated BEFORE running).
 *
 * (a) POPULATION BASELINE. The first run's NEUTRAL bucket was 40 tickers
 *     centered on the MIDDLE of the ranked short_ratio distribution — a
 *     rank-selected band, not a random/representative sample, and the
 *     follow-up hypothesis's own suspicion is that this band shares a
 *     composition confound with the extremes it doesn't. The true "what
 *     would random entry have returned" (Reasoning Standard #3) baseline is
 *     the full qualifying population (~2,100+ names/day at FLOOR_TOTAL_VOL,
 *     live-probed this session) — but pricing every qualifying name every
 *     day is 10-15x the first run's network cost for a session script.
 *     Compromise, stated up front as a scope limitation, not discovered
 *     after the fact: a SYSTEMATIC sample of up to POP_SAMPLE_SIZE=200
 *     qualifying tickers per day, selected by sorting the day's qualifying
 *     tickers ALPHABETICALLY BY SYMBOL (a key with zero relationship to
 *     short_ratio) and taking every Nth at a fixed stride — deterministic
 *     and reproducible (no RNG), and critically UNCORRELATED with the
 *     metric under test, unlike a middle-rank band. This directly answers
 *     part (a): the comparison is now against a spread-across-the-whole-
 *     distribution proxy, not a single rank-adjacent band.
 * (b) REGIME SPLIT. vxx_ratio (VXX close / 30-TRADING-day VXX avg,
 *     strictly prior days) and spy_vs_ma50 (SPY close / 50-TRADING-day SPY
 *     MA, strictly prior days) are computed per sample day from Yahoo daily
 *     closes, mirroring ml_model_v2.py's live calculation verbatim
 *     (ml_model_v2.py:439-455 — same 30/50 trading-day windows, same
 *     "strictly before today" indexing) so the labels mean the same thing
 *     as the production regime the bot actually trades under. Classified
 *     into bear/neutral/bull via regime_util.py's classify_regime()
 *     thresholds (BEAR_VXX_THRESHOLD=1.15, BEAR_SPY_THRESHOLD=0.94,
 *     BULL_VXX_THRESHOLD=0.95, BULL_SPY_THRESHOLD=0.98 — mirrored as local
 *     constants, not re-derived, since this is a TS research script and
 *     regime_util.py is Python; values pinned in this script's own test).
 * STATISTICAL METHOD: same day-clustered t-test as the first run (collapse
 *     to one HIGH_SHORT-minus-population spread per report day, t-test
 *     across day-level spreads via scripts/statsUtils.ts) — not a naive
 *     pooled per-row test, for the same reason the first run already
 *     applied that lesson from its own start.
 * DIRECTION: still genuinely ambiguous (unchanged from the first run's own
 *     reasoning: informed-shorts literature predicts HIGH_SHORT underperforms,
 *     the wishlist's squeeze framing predicts the opposite, and this is
 *     neither instrument) — two-tailed test, no directional assumption.
 * PASS BAR (pre-stated, single test, no fishing): day-clustered |t| >
 *     tCrit005 on the +20d HIGH_SHORT-minus-population spread (the more
 *     theory-relevant horizon, matching the first run's own choice) AND the
 *     SIGN of the mean +20d spread does not flip across any regime bucket
 *     that has >=MIN_REGIME_DAYS=3 sample days. The sign-stability clause
 *     directly operationalizes the follow-up's own suspicion — a genuine
 *     standalone edge should not reverse direction depending on market
 *     regime; a spread that flips sign regime-to-regime is a composition/
 *     regime confound, exactly the failure mode the first run's U-shape
 *     hinted at. +5d is reported informationally only (not part of the
 *     bar), matching the first run's own +20d-is-primary choice.
 * PRIOR (stated before running): the population mean should sit close to
 *     the first run's NEUTRAL value (both are middle-of-distribution
 *     proxies), so P(overall +20d t still clears significance) ~60%; but
 *     P(sign is stable across every regime bucket with n>=3 days) ~40%
 *     given the first run's own U-shape already hinted at regime/
 *     composition sensitivity. Composite P(clean PASS) ~30-35% — this
 *     retest is expected to SHARPEN the diagnosis of the anomaly, not
 *     necessarily resolve it to a clean signal.
 *
 * Session-run: `npx tsx scripts/finra_shortvol_gate2_retest.ts [--dry]`
 * --dry skips the Yahoo price fetch and only prints bucket/sample sizes,
 * to size network cost before committing to the full run.
 * Result goes in research/experiments.md + datacore/signal_ladder.json,
 * never into any runtime path — this script touches no production state.
 */
import { pathToFileURL } from "url";
import { fetchShortVolDay, FLOOR_TOTAL_VOL, type ShortVolRow } from "../server/finraShortVolume";
import {
  bucketDay,
  BUCKET_SIZE,
  HORIZONS,
  SAMPLE_START,
  SAMPLE_WEEKS,
  type ShortRatioRow,
} from "./finra_shortvol_gate2";
import { weeklySampleDays, ymdCompact, fetchYahooDaily, toSeries, fwdReturn, meanStd, type Series } from "./occ_volume_gate2";
import { clusterMeanTTest, tCrit005, survivesAtCrit005 } from "./statsUtils";

export const POP_SAMPLE_SIZE = 200;
export const MIN_REGIME_DAYS = 3;
const DRY = process.argv.includes("--dry");

// Mirrored verbatim from regime_util.py's classify_regime() thresholds —
// the canonical Python source. Pinned in this file's own test against the
// published values, not re-derived.
const BEAR_VXX_THRESHOLD = 1.15;
const BEAR_SPY_THRESHOLD = 0.94;
const BULL_VXX_THRESHOLD = 0.95;
const BULL_SPY_THRESHOLD = 0.98;

export type Regime = "bear" | "neutral" | "bull";

/** Systematic (deterministic, metric-independent) sample of a day's
 *  qualifying population — the "what would broad random entry have
 *  returned" proxy (Reasoning Standard #3), bounded to POP_SAMPLE_SIZE for
 *  network cost. Sorted by TICKER SYMBOL, not short_ratio, so selection
 *  carries zero correlation with the metric under test — unlike a
 *  rank-centered band. Pure function, no I/O. */
export function populationSample(rows: ShortVolRow[]): { qualifying: number; sampled: ShortRatioRow[] } {
  const qualifying: ShortRatioRow[] = rows
    .filter((r) => r.total_vol >= FLOOR_TOTAL_VOL)
    .map((r) => ({ ticker: r.symbol, short_ratio: r.short_vol / r.total_vol, total_vol: r.total_vol }))
    .sort((a, b) => a.ticker.localeCompare(b.ticker));

  const n = qualifying.length;
  if (n === 0) return { qualifying: 0, sampled: [] };
  const stride = Math.max(1, Math.floor(n / POP_SAMPLE_SIZE));
  const sampled: ShortRatioRow[] = [];
  for (let i = 0; i < n && sampled.length < POP_SAMPLE_SIZE; i += stride) sampled.push(qualifying[i]);
  return { qualifying: n, sampled };
}

/** Trailing average over `lookback` TRADING days STRICTLY BEFORE `day`
 *  (excludes day itself) — mirrors ml_model_v2.py:439-455's vxx_hist30/
 *  spy_hist50 indexing exactly (idx-j-1 for j in 0..lookback-1). */
function trailingAvg(series: Series, day: string, lookback: number): number | null {
  const idx = series.indexOf.get(day);
  if (idx == null) return null;
  const vals: number[] = [];
  for (let j = 0; j < lookback; j++) {
    const i = idx - j - 1;
    if (i < 0) break;
    vals.push(series.closes[i]);
  }
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

/** bear/neutral/bull per regime_util.py's classify_regime() thresholds,
 *  fed by vxx_ratio/spy_vs_ma50 computed the same way ml_model_v2.py
 *  computes them live. Returns null when either series lacks enough
 *  history for `day` (honest no-op, not a guessed default). */
export function classifyRegime(day: string, spySeries: Series, vxxSeries: Series): Regime | null {
  const spyIdx = spySeries.indexOf.get(day);
  const vxxIdx = vxxSeries.indexOf.get(day);
  if (spyIdx == null || vxxIdx == null) return null;
  const spyClose = spySeries.closes[spyIdx];
  const vxxClose = vxxSeries.closes[vxxIdx];
  const spyMa50 = trailingAvg(spySeries, day, 50);
  const vxxAvg30 = trailingAvg(vxxSeries, day, 30);
  if (spyMa50 == null || vxxAvg30 == null || spyMa50 <= 0 || vxxAvg30 <= 0) return null;
  const vxxRatio = vxxClose / vxxAvg30;
  const spyVsMa50 = spyClose / spyMa50;
  if (vxxRatio >= BEAR_VXX_THRESHOLD || spyVsMa50 < BEAR_SPY_THRESHOLD) return "bear";
  if (vxxRatio <= BULL_VXX_THRESHOLD && spyVsMa50 >= BULL_SPY_THRESHOLD) return "bull";
  return "neutral";
}

async function main() {
  const days = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  console.log(`finra_shortvol_gate2_retest ${DRY ? "(--dry calibration run)" : "(full run)"}: ${days.length} weekly sample days, FLOOR_TOTAL_VOL=${FLOOR_TOTAL_VOL}, BUCKET_SIZE=${BUCKET_SIZE}, POP_SAMPLE_SIZE=${POP_SAMPLE_SIZE}`);

  const highByDay: { day: string; high: ShortRatioRow[]; pop: ShortRatioRow[] }[] = [];
  let skippedDays = 0;
  for (const day of days) {
    const rows = await fetchShortVolDay(ymdCompact(day));
    if (rows === null || rows.length === 0) {
      console.log(`  ${day}: skipped (no data / transport error) — honest no-op`);
      skippedDays++;
      continue;
    }
    const b = bucketDay(day, rows);
    const pop = populationSample(rows);
    highByDay.push({ day, high: b.highShort, pop: pop.sampled });
    console.log(`  ${day}: qualifying=${pop.qualifying} highShort=${b.highShort.length} popSample=${pop.sampled.length}`);
    await new Promise((res) => setTimeout(res, 400)); // polite spacing vs FINRA's CDN
  }
  console.log(`\ndays_with_data=${highByDay.length}/${days.length} (skipped=${skippedDays})`);

  const tickerSet = new Set<string>();
  for (const d of highByDay) for (const r of [...d.high, ...d.pop]) tickerSet.add(r.ticker);
  console.log(`unique tickers needed for price lookup=${tickerSet.size}`);

  if (DRY) {
    console.log(`\n--dry: stopping before Yahoo fetch.`);
    return;
  }

  const firstDay = new Date(highByDay[0].day + "T00:00:00Z").getTime() / 1000;
  const lastDay = new Date(highByDay[highByDay.length - 1].day + "T00:00:00Z").getTime() / 1000;
  // Extra lookback for the 50-trading-day SPY MA / 30-trading-day VXX avg —
  // ~120 calendar days comfortably covers 50 trading days plus weekends/holidays.
  const startSec = firstDay - 130 * 86400;
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
    if ((i + CONCURRENCY) % 200 === 0 || i + CONCURRENCY >= tickers.length) {
      console.log(`  priced ${Math.min(i + CONCURRENCY, tickers.length)}/${tickers.length} tickers (fetched=${fetched} failed=${failed})`);
    }
    await new Promise((res) => setTimeout(res, 250));
  }
  console.log(`\nprice fetch done: fetched=${fetched} failed=${failed}/${tickers.length}`);

  // SPY + VXX for regime classification — same startSec buffer as above.
  const spyMap = await fetchYahooDaily("SPY", startSec, endSec);
  const vxxMap = await fetchYahooDaily("VXX", startSec, endSec);
  const spySeries = toSeries(spyMap);
  const vxxSeries = toSeries(vxxMap);

  const dayLevelSpreads: Record<number, { day: string; regime: Regime | null; high: number; pop: number; spread: number }[]> = { 5: [], 20: [] };

  for (const d of highByDay) {
    const regime = classifyRegime(d.day, spySeries, vxxSeries);
    for (const h of HORIZONS) {
      const meanFwd = (rows: ShortRatioRow[]): number | null => {
        const rets: number[] = [];
        for (const r of rows) {
          const series = seriesCache.get(r.ticker);
          if (!series) continue;
          const ret = fwdReturn(series, d.day, h);
          if (ret == null) continue;
          rets.push(ret);
        }
        return rets.length ? rets.reduce((a, x) => a + x, 0) / rets.length : null;
      };
      const highMean = meanFwd(d.high);
      const popMean = meanFwd(d.pop);
      if (highMean != null && popMean != null) {
        dayLevelSpreads[h].push({ day: d.day, regime, high: highMean, pop: popMean, spread: highMean - popMean });
      }
    }
  }

  console.log(`\n=== RESULTS (HIGH_SHORT vs population-sample, day-clustered) ===`);
  const clustered: Record<number, ReturnType<typeof clusterMeanTTest>> = {} as any;
  for (const h of HORIZONS) {
    const rows = dayLevelSpreads[h];
    const result = clusterMeanTTest(rows.map((r) => r.spread));
    clustered[h] = result;
    const crit = tCrit005(result.df);
    console.log(`\n+${h}d horizon (n_days=${result.n}, df=${result.df}, crit=${crit.toFixed(3)}):`);
    console.log(`  day-clustered mean spread (HIGH-POP): ${(result.mean * 100).toFixed(3)}%  t=${result.t.toFixed(3)}  SURVIVES=${survivesAtCrit005(result)}`);

    const byRegime = new Map<string, number[]>();
    for (const r of rows) {
      const key = r.regime ?? "unclassified";
      if (!byRegime.has(key)) byRegime.set(key, []);
      byRegime.get(key)!.push(r.spread);
    }
    for (const [regime, spreads] of byRegime) {
      const stats = meanStd(spreads);
      console.log(`    regime=${regime}: n_days=${stats.n} mean_spread=${(stats.mean * 100).toFixed(3)}%`);
    }
  }

  const rows20 = dayLevelSpreads[20];
  const byRegime20 = new Map<string, number[]>();
  for (const r of rows20) {
    const key = r.regime ?? "unclassified";
    if (!byRegime20.has(key)) byRegime20.set(key, []);
    byRegime20.get(key)!.push(r.spread);
  }
  let signStable = true;
  const signs: string[] = [];
  for (const [regime, spreads] of byRegime20) {
    if (spreads.length < MIN_REGIME_DAYS) continue;
    const mean = spreads.reduce((a, b) => a + b, 0) / spreads.length;
    signs.push(`${regime}(n=${spreads.length}): ${mean >= 0 ? "+" : "-"}`);
  }
  const overallSign = clustered[20].mean >= 0;
  for (const [regime, spreads] of byRegime20) {
    if (spreads.length < MIN_REGIME_DAYS) continue;
    const mean = spreads.reduce((a, b) => a + b, 0) / spreads.length;
    if ((mean >= 0) !== overallSign) signStable = false;
  }

  const survives20 = survivesAtCrit005(clustered[20]);
  const verdict = survives20 && signStable ? "PASS" : "FAIL/INCONCLUSIVE";
  console.log(`\nsign by regime (n>=${MIN_REGIME_DAYS} days only): ${signs.join(", ") || "no regime had enough days"}`);
  console.log(`VERDICT (pre-stated bar: day-clustered |t20|>${tCrit005(clustered[20].df).toFixed(3)} AND sign stable across regimes with n>=${MIN_REGIME_DAYS} days): ${verdict}`);
  console.log(`survives_t20=${survives20} sign_stable=${signStable}`);
}

// Entrypoint guard (ESM has no require.main) — same pattern as
// finra_shortvol_gate2.ts's own header note on why this matters for a
// module other scripts might import from.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
