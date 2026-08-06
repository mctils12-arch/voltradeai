/**
 * finra_shortvol_gate2.ts — GATE 2 (SIGNAL layer) for the finra_short_volume
 * root: do daily short-sale-volume ratio EXTREMES carry forward-return
 * information, measured with NO trading involved?
 *
 * ROOT CONTEXT: gate 1 (DATA) passed 2026-07-05 (100.000% sum-of-parts
 * identity across 12,240 symbols vs FINRA's own 4 independent per-facility
 * files — datacore/signal_ladder.json). The standalone hypothesis has sat
 * unattempted at gate 2 since: research/wishlist.md's BUILD ORDER 5 #1
 * states it verbatim — "short-volume-ratio extremes and multi-day deltas
 * precede reversals in small caps ... PRIOR: P(gate-2 pass) ~35% —
 * short-ratio signals are well-mined in large caps; the residual edge, if
 * any, is in the illiquid tail (EDGE DOCTRINE #2)." This script runs that
 * exact standalone test (the 13F/Form4-join extension named in the same
 * sentence is a SEPARATE, later hypothesis — Form4 clustering's own gate 2
 * was independently KILLED 2026-07-22, so that join is out of scope here,
 * same reasoning the MIDAS module's 2026-07-28 update already applied to
 * itself).
 *
 * PRE-REGISTERED CRITERIA (stated BEFORE running, Reasoning Standard #10).
 *
 * OPERATIONALIZATION (one definition, not fished across variants):
 *   - Signal = short_vol/total_vol per (symbol, trade date), FINRA's own
 *     daily consolidated CNMS file (exchange-listed only — the ORF/OTC
 *     facility is a separate, newer stream with far less history, out of
 *     scope for this run).
 *   - Universe per day: symbols with total_vol >= FLOOR_TOTAL_VOL, REUSING
 *     the exact liquidity floor server/finraShortVolume.ts's own summarize()
 *     already uses for its top_ratio view (500,000 shares/day) — not a new
 *     number invented for this script (READ BEFORE WRITE: checked the
 *     existing module before picking a floor). HONEST LIMITATION: this is
 *     a LIQUIDITY floor, not a market-cap ceiling — the FINRA file carries
 *     no market-cap field, so "small caps" in the wishlist hypothesis text
 *     is approximated by "actively-traded but not mega-cap-only" via the
 *     bucket construction below, not a true cap screen. Stated up front,
 *     not discovered after the fact.
 *   - Buckets per day (rank by short_ratio, symmetric so bucket SIZE, not
 *     selection, is the only free parameter — same construction as the
 *     occ_options_volume precedent this script's methodology is modeled
 *     on): FIXED size BUCKET_SIZE=40 (same network-cost bound as
 *     occ_volume_gate2.ts, chosen before any price data was looked at):
 *       HIGH_SHORT = top 40 by short_ratio (most heavily shorted that day)
 *       LOW_SHORT  = bottom 40 by short_ratio (least shorted, still liquid)
 *       NEUTRAL    = 40 tickers centered on the middle of the qualifying
 *         range — the base-rate control (Reasoning Standard #3).
 *   - Forward return: close-to-close from the FINRA trade date to +5 and
 *     +20 TRADING days later (Yahoo adjusted close, reusing
 *     occ_volume_gate2.ts's fetchYahooDaily/toSeries/fwdReturn verbatim —
 *     not reimplemented). Already-realized history — no lookahead risk
 *     (Reasoning Standard #7).
 *   - Sample: 16 weekly Wednesdays, 2026-01-07 through 2026-04-22 (verified
 *     via weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS) — occ_volume_gate2.ts's
 *     own header text says "through 2026-04-15" for the identical
 *     SAMPLE_START/SAMPLE_WEEKS pair, which is off by one week; not fixed
 *     here, out of scope for this PR, noted so this script's own dates
 *     are stated correctly) — THE IDENTICAL WINDOW occ_volume_gate2.ts
 *     used (same values, deliberately
 *     not imported from that file since this is a conceptually distinct
 *     root — matching its values is a deliberate choice for comparable
 *     regime texture across BULL/CAUTION/NEUTRAL per Reasoning Standard #2,
 *     not an accident). Ends >100 calendar days before this script's run
 *     date (2026-08-06) so +20-trading-day outcomes are fully realized.
 *     FINRA's CNMS files are fetched LIVE per sample date directly from
 *     cdn.finra.org (dated files persist for years — live-verified this
 *     session that 2026-01-07 still returns 200), NOT read from the
 *     Railway volume's own deep-backfill archive (this script has no
 *     access to that volume from a fresh session container) — a different
 *     data-access path than the production poller, same end result.
 *   - STATISTICAL METHOD: day-CLUSTERED t-test from the start (collapse to
 *     one HIGH-LOW spread per report day, t-test across the ~16 day-level
 *     spreads, df-correct critical value via scripts/statsUtils.ts) — NOT
 *     a naive pooled per-row Welch t. This applies the lesson the
 *     occ_options_volume candidate only learned on its SECOND gate-2 pass
 *     (2026-08-03 finding: pooled t overstates significance when the same
 *     tickers recur across days and every name in a day's bucket shares
 *     that day's market-wide move) from the FIRST pass here, per Reasoning
 *     Standard #4's instruction to distrust naive pooled results.
 *   - DIRECTION — stated as genuinely AMBIGUOUS before running, not
 *     assumed: the academic short-interest literature (Asquith/Pathak/
 *     Ritter 2005 and successors) finds HIGH short interest -> LOWER
 *     subsequent returns ("shorts are informed"), which would predict
 *     HIGH_SHORT < LOW_SHORT here. The wishlist's own hypothesis text
 *     instead frames this as "extremes ... precede reversals" (a squeeze
 *     framing), which would predict HIGH_SHORT > LOW_SHORT. These two
 *     narratives point OPPOSITE directions, and this dataset (daily
 *     EXECUTION-volume ratio, not published total short INTEREST/days-to-
 *     cover) is not the same instrument either literature was built on —
 *     so this script pre-registers a TWO-TAILED test on the HIGH-LOW
 *     spread (no directional assumption), with a monotonic-ordering
 *     requirement in EITHER consistent direction (HIGH>NEUTRAL>LOW on
 *     both horizons, OR LOW>NEUTRAL>HIGH on both horizons) as the
 *     structure check, exactly analogous to the OCC precedent's ordering
 *     bar but without assuming which direction wins.
 *   - PASS bar (pre-stated, falsifiable, single test — no multiple-
 *     hypothesis fishing): monotonic ordering (in either direction, see
 *     above) on BOTH horizons AND day-clustered |t| > tCrit005(df) on the
 *     +20d HIGH-LOW spread (the more theory-relevant horizon — multi-day
 *     short pressure, not next-day noise, matching the wishlist's own
 *     "multi-day deltas" framing).
 *   - PRIOR stated before running (wishlist.md's own number, not
 *     recomputed): ~35% chance of a clean pass on both horizons in EITHER
 *     direction combined; a null result is itself informative (it would
 *     mean daily execution-volume short ratio alone, without a 13F/Form4
 *     join, carries no standalone signal in this universe).
 *
 * Session-run: `npx tsx scripts/finra_shortvol_gate2.ts [--dry]`
 * --dry skips the Yahoo price fetch/return computation and only prints
 * bucket sizes + unique-ticker counts, to size network cost before
 * committing to the full run.
 * Result goes in research/experiments.md + datacore/signal_ladder.json,
 * never into any runtime path — this script touches no production state.
 */
import { pathToFileURL } from "url";
import { fetchShortVolDay, type ShortVolRow, FLOOR_TOTAL_VOL } from "../server/finraShortVolume";
import { weeklySampleDays, ymdCompact, fetchYahooDaily, toSeries, fwdReturn, meanStd, type Series } from "./occ_volume_gate2";
import { clusterMeanTTest, tCrit005, survivesAtCrit005 } from "./statsUtils";

export const BUCKET_SIZE = 40;
export const HORIZONS = [5, 20] as const;
export const SAMPLE_START = "2026-01-07"; // Wednesday — same window as occ_volume_gate2.ts, see header
export const SAMPLE_WEEKS = 16;
const DRY = process.argv.includes("--dry");

export interface ShortRatioRow { ticker: string; short_ratio: number; total_vol: number; }

export interface DayBuckets {
  day: string;
  qualifying: number;
  highShort: ShortRatioRow[];
  lowShort: ShortRatioRow[];
  neutral: ShortRatioRow[];
}

/** Bucket one day's FINRA rows by short_ratio. Pure function, no I/O — the
 *  liquidity floor and bucket size are the module's own documented
 *  constants (FLOOR_TOTAL_VOL imported, not redefined; BUCKET_SIZE fixed
 *  above), so this is fully unit-testable on synthetic rows. */
export function bucketDay(day: string, rows: ShortVolRow[]): DayBuckets {
  const qualifying: ShortRatioRow[] = rows
    .filter((r) => r.total_vol >= FLOOR_TOTAL_VOL)
    .map((r) => ({ ticker: r.symbol, short_ratio: r.short_vol / r.total_vol, total_vol: r.total_vol }))
    .sort((a, b) => a.short_ratio - b.short_ratio); // ascending: lowest short ratio first

  const n = qualifying.length;
  const bucketSize = Math.max(1, Math.min(BUCKET_SIZE, Math.floor(n / 3)));
  const lowShort = qualifying.slice(0, bucketSize);
  const highShort = qualifying.slice(n - bucketSize);
  const midStart = Math.floor((n - bucketSize) / 2);
  const neutral = qualifying.slice(midStart, midStart + bucketSize);

  return { day, qualifying: n, highShort, lowShort, neutral };
}

async function main() {
  const days = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  console.log(`finra_shortvol_gate2 ${DRY ? "(--dry calibration run)" : "(full run)"}: ${days.length} weekly sample days, FLOOR_TOTAL_VOL=${FLOOR_TOTAL_VOL}, BUCKET_SIZE=${BUCKET_SIZE}`);

  const dayBuckets: DayBuckets[] = [];
  let skippedDays = 0;
  for (const day of days) {
    const rows = await fetchShortVolDay(ymdCompact(day));
    if (rows === null || rows.length === 0) {
      console.log(`  ${day}: skipped (no data / transport error) — honest no-op`);
      skippedDays++;
      continue;
    }
    const b = bucketDay(day, rows);
    dayBuckets.push(b);
    console.log(`  ${b.day}: qualifying=${b.qualifying} bucketSize=${b.highShort.length}`);
    await new Promise((res) => setTimeout(res, 400)); // polite spacing vs FINRA's CDN
  }

  const tickerSet = new Set<string>();
  let totalObs = 0;
  for (const b of dayBuckets) {
    for (const r of [...b.highShort, ...b.lowShort, ...b.neutral]) {
      tickerSet.add(r.ticker);
      totalObs++;
    }
  }
  console.log(`\ndays_with_data=${dayBuckets.length}/${days.length} (skipped=${skippedDays})`);
  console.log(`total bucket observations=${totalObs}, unique tickers needed for price lookup=${tickerSet.size}`);

  if (DRY) {
    console.log(`\n--dry: stopping before Yahoo fetch (${tickerSet.size} price series would be requested).`);
    return;
  }

  const firstDay = new Date(dayBuckets[0].day + "T00:00:00Z").getTime() / 1000;
  const lastDay = new Date(dayBuckets[dayBuckets.length - 1].day + "T00:00:00Z").getTime() / 1000;
  const startSec = firstDay - 5 * 86400;
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

  // Day-level (cluster-level) mean forward return per bucket per horizon —
  // the CLUSTER is the report day, per statsUtils.ts's rationale.
  const dayLevelSpreads: Record<number, { day: string; high: number; low: number; neutral: number; spread: number }[]> = { 5: [], 20: [] };
  const pooled: Record<string, Record<number, number[]>> = {
    highShort: { 5: [], 20: [] }, lowShort: { 5: [], 20: [] }, neutral: { 5: [], 20: [] },
  };

  for (const b of dayBuckets) {
    for (const h of HORIZONS) {
      const bucketMean = (rows: ShortRatioRow[], bucketName: "highShort" | "lowShort" | "neutral"): number | null => {
        const rets: number[] = [];
        for (const r of rows) {
          const series = seriesCache.get(r.ticker);
          if (!series) continue;
          const ret = fwdReturn(series, b.day, h);
          if (ret == null) continue;
          rets.push(ret);
          pooled[bucketName][h].push(ret);
        }
        return rets.length ? rets.reduce((a, x) => a + x, 0) / rets.length : null;
      };
      const highMean = bucketMean(b.highShort, "highShort");
      const lowMean = bucketMean(b.lowShort, "lowShort");
      const neutralMean = bucketMean(b.neutral, "neutral");
      if (highMean != null && lowMean != null && neutralMean != null) {
        dayLevelSpreads[h].push({ day: b.day, high: highMean, low: lowMean, neutral: neutralMean, spread: highMean - lowMean });
      }
    }
  }

  console.log(`\n=== RESULTS (daily short-ratio, day-clustered) ===`);
  const clustered: Record<number, ReturnType<typeof clusterMeanTTest>> = {} as any;
  for (const h of HORIZONS) {
    const rows = dayLevelSpreads[h];
    const spreads = rows.map((r) => r.spread);
    const result = clusterMeanTTest(spreads);
    clustered[h] = result;
    const crit = tCrit005(result.df);
    const survives = survivesAtCrit005(result);
    const highPooled = meanStd(pooled.highShort[h]);
    const lowPooled = meanStd(pooled.lowShort[h]);
    const neuPooled = meanStd(pooled.neutral[h]);
    console.log(`\n+${h}d horizon (n_days=${result.n}, df=${result.df}, crit=${crit.toFixed(3)}):`);
    console.log(`  HIGH_SHORT (pooled, informational only): n=${highPooled.n} mean=${(highPooled.mean * 100).toFixed(3)}%`);
    console.log(`  NEUTRAL    (pooled, informational only): n=${neuPooled.n} mean=${(neuPooled.mean * 100).toFixed(3)}%`);
    console.log(`  LOW_SHORT  (pooled, informational only): n=${lowPooled.n} mean=${(lowPooled.mean * 100).toFixed(3)}%`);
    console.log(`  day-clustered mean spread (HIGH-LOW): ${(result.mean * 100).toFixed(3)}%  t=${result.t.toFixed(3)}  SURVIVES=${survives}`);
  }

  const rows5 = dayLevelSpreads[5], rows20 = dayLevelSpreads[20];
  const meanOf = (rows: typeof rows5, key: "high" | "low" | "neutral") =>
    rows.length ? rows.reduce((a, r) => a + r[key], 0) / rows.length : NaN;
  const high5 = meanOf(rows5, "high"), low5 = meanOf(rows5, "low"), neu5 = meanOf(rows5, "neutral");
  const high20 = meanOf(rows20, "high"), low20 = meanOf(rows20, "low"), neu20 = meanOf(rows20, "neutral");
  const orderedHighDominant = high5 > neu5 && neu5 > low5 && high20 > neu20 && neu20 > low20;
  const orderedLowDominant = low5 > neu5 && neu5 > high5 && low20 > neu20 && neu20 > high20;
  const ordered = orderedHighDominant || orderedLowDominant;
  const t20 = clustered[20].t;
  const verdict = ordered && survivesAtCrit005(clustered[20]) ? "PASS" : "FAIL/INCONCLUSIVE";
  console.log(`\nVERDICT (pre-stated bar: monotonic ordering in EITHER consistent direction on both horizons AND day-clustered |t20|>${tCrit005(clustered[20].df).toFixed(3)}): ${verdict}`);
  console.log(`ordered_high_dominant=${orderedHighDominant} ordered_low_dominant=${orderedLowDominant} clustered_t_20d=${t20.toFixed(3)} df=${clustered[20].df}`);
}

// Entrypoint guard (ESM has no require.main) — mirrors the fix
// occ_volume_gate2.ts's sibling scripts needed after the entrypoint-guard
// bug found 2026-08-05: without this, a future script importing bucketDay/
// SAMPLE_START from this file would ALSO re-trigger this file's own full
// FINRA+Yahoo fetch as an unwanted import side effect.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
