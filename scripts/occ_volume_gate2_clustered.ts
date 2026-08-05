/**
 * occ_volume_gate2_clustered.ts — day-clustered re-test of the OCC
 * REVERSAL candidate filed in research/open_questions.md on 2026-08-03
 * ("[HYPOTHESIS] Customer put-heavy OCC skew outperforms customer
 * call-heavy skew — reversed from the naive ISEE reading").
 *
 * THIS IS LADDER-PATH STEP (1) OF 3 for that filed candidate, not a new
 * gate-2 run of the already-killed occ_options_volume root (that root's
 * signal_ladder.json status stays `killed` regardless of this result — the
 * hypothesis under test here is the REVERSED one, still pre-gate2_pass).
 *
 * WHY THIS SCRIPT EXISTS (do not just re-read occ_volume_gate2.ts's
 * printed Welch t): that script's pooled Welch t-test (naive-pooled t =
 * -2.304 on the +5d spread, per the 2026-08-03 finding) treats every
 * (day,ticker) row as an independent draw. It is not — the same tickers
 * recur across the 16 sample weeks, and every name in one day's bucket
 * shares that day's market-wide move. True independent information
 * content is closer to 16 day-clusters than ~620 i.i.d. rows, so the
 * pooled t overstates significance (Reasoning Standard #4). This script
 * fixes that by the "collapse to cluster means" method (scripts/
 * statsUtils.ts's clusterMeanTTest): for each report day, compute the
 * day-level mean forward return for the CALL_SKEW and PUT_SKEW buckets,
 * take the day-level (CALL-PUT) spread, and t-test ACROSS those ~16
 * day-level spreads (df = days_with_data - 1) instead of across ~620 rows.
 *
 * IDENTICAL METHODOLOGY, NOT A NEW VARIANT (do not re-fish): same FLOOR,
 * BUCKET_SIZE, HORIZONS, SAMPLE_START, SAMPLE_WEEKS as occ_volume_gate2.ts
 * (imported directly, not redefined, so they cannot drift). Only the
 * statistical test changes — pooled-row Welch t -> day-clustered t.
 *
 * PRE-STATED NEXT STEP (per the 2026-08-03 filed hypothesis's ladder
 * path): "if |t_clustered| collapses well below [the df-correct critical
 * value], this candidate is ALSO dead and should be marked killed too,
 * not left open indefinitely." This script computes exactly that number
 * and prints the verdict against the CORRECT df-adjusted critical value
 * (scripts/statsUtils.ts's tCrit005), not a flat |t|>2 heuristic — with
 * only ~16 clusters the true 5% critical value is materially higher than
 * 2 (e.g. 2.131 at df=15), so a flat threshold would have been too lax.
 *
 * Session-run: `npx tsx scripts/occ_volume_gate2_clustered.ts`
 * Result goes in research/open_questions.md + research/experiments.md,
 * never into any runtime path or signal_ladder.json gate-2 status (this
 * is pre-registration data for a still-unconfirmed candidate, not a
 * ladder promotion) — this script touches no production state.
 *
 * `runClusteredDayTest(days, label)` below is exported so LADDER PATH
 * STEP (2) (a disjoint out-of-sample window, scripts/
 * occ_volume_gate2_clustered_disjoint.ts) can reuse the identical
 * fetch/bucket/cluster-test logic against a different day list without
 * redefining it — same pattern as cftc_tff_tlt_disjoint_replication.py's
 * reuse of cftc_tff_gate2_test.py's machinery. main() below still calls
 * it with the original SAMPLE_START/SAMPLE_WEEKS days; behavior for the
 * original run is unchanged.
 */
import { pathToFileURL } from "url";
import { fetchOccDay, aggregateOcc } from "../server/occVolume";
import {
  FLOOR, BUCKET_SIZE, HORIZONS, SAMPLE_START, SAMPLE_WEEKS,
  weeklySampleDays, ymdCompact, bucketDay, fetchYahooDaily, toSeries,
  fwdReturn, meanStd, type DayBuckets, type Series,
} from "./occ_volume_gate2";
import { clusterMeanTTest, tCrit005, survivesAtCrit005, type ClusterTTestResult } from "./statsUtils";

export interface ClusteredHorizonResult {
  horizon: number;
  result: ClusterTTestResult;
  crit: number;
  survives: boolean;
  dayDetail: { day: string; callMean: number; putMean: number; spread: number }[];
}

export async function runClusteredDayTest(days: string[], label: string): Promise<ClusteredHorizonResult[]> {
  console.log(`${label}: ${days.length} weekly sample days, FLOOR=${FLOOR}, BUCKET_SIZE=${BUCKET_SIZE} (identical to occ_volume_gate2.ts)`);

  const dayBuckets: DayBuckets[] = [];
  for (const day of days) {
    const { kind, rows } = await fetchOccDay(ymdCompact(day));
    if (kind !== "data" || !rows.length) {
      console.log(`  ${day}: skipped (${kind}) — honest no-op`);
      continue;
    }
    const { top } = aggregateOcc(rows, 999_999);
    const b = bucketDay(rows[0].date, top);
    dayBuckets.push(b);
    console.log(`  ${b.day}: qualifying=${b.qualifying} bucketSize=${b.callSkew.length}`);
    await new Promise((res) => setTimeout(res, 500));
  }

  const tickerSet = new Set<string>();
  for (const b of dayBuckets) {
    for (const r of [...b.callSkew, ...b.putSkew, ...b.neutral]) tickerSet.add(r.ticker);
  }
  console.log(`\ndays_with_data=${dayBuckets.length}/${days.length}, unique tickers needed=${tickerSet.size}`);

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

  // Day-level means (NOT pooled rows) per bucket per horizon.
  console.log(`\n=== DAY-CLUSTERED RESULTS (cluster = report day, n=${dayBuckets.length}) ===`);
  const horizonResults: ClusteredHorizonResult[] = [];
  for (const h of HORIZONS) {
    const daySpreads: number[] = [];
    const dayDetail: { day: string; callMean: number; putMean: number; spread: number }[] = [];
    for (const b of dayBuckets) {
      const callRets: number[] = [];
      const putRets: number[] = [];
      for (const r of b.callSkew) {
        const s = seriesCache.get(r.ticker);
        const ret = s ? fwdReturn(s, b.day, h) : null;
        if (ret != null) callRets.push(ret);
      }
      for (const r of b.putSkew) {
        const s = seriesCache.get(r.ticker);
        const ret = s ? fwdReturn(s, b.day, h) : null;
        if (ret != null) putRets.push(ret);
      }
      if (!callRets.length || !putRets.length) continue; // honest no-op: day contributes no cluster
      const callMean = meanStd(callRets).mean;
      const putMean = meanStd(putRets).mean;
      daySpreads.push(callMean - putMean);
      dayDetail.push({ day: b.day, callMean, putMean, spread: callMean - putMean });
    }

    const result = clusterMeanTTest(daySpreads);
    const crit = tCrit005(result.df);
    const survives = survivesAtCrit005(result);

    console.log(`\n+${h}d horizon (n_days=${result.n}, df=${result.df}):`);
    for (const d of dayDetail) {
      console.log(`  ${d.day}: CALL=${(d.callMean * 100).toFixed(3)}% PUT=${(d.putMean * 100).toFixed(3)}% spread=${(d.spread * 100).toFixed(3)}%`);
    }
    console.log(`  day-clustered mean spread (CALL-PUT)=${(result.mean * 100).toFixed(3)}%  t_clustered=${result.t.toFixed(3)}  df-correct critical(0.05,two-tailed)=${crit.toFixed(3)}`);
    console.log(`  SURVIVES clustering at 5%: ${survives}`);
    horizonResults.push({ horizon: h, result, crit, survives, dayDetail });
  }

  return horizonResults;
}

async function main() {
  const days = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  const horizonResults = await runClusteredDayTest(days, "occ_volume_gate2_clustered");
  const allDead = horizonResults.every((r) => !r.survives);
  console.log(`\nVERDICT: if both horizons show SURVIVES=false, the reversal candidate is dead at ladder-path step (1) — mark it killed in the same terms as occ_options_volume, per the pre-stated next step in open_questions.md. If either horizon survives, proceed to ladder-path step (2): disjoint out-of-sample window. (this run: allDead=${allDead})`);
}

// Entrypoint guard (ESM has no require.main) — see occ_volume_gate2.ts's
// identical guard for why: occ_volume_gate2_clustered_disjoint.ts imports
// runClusteredDayTest from this file and must not trigger this file's own
// original-window main() as an import side effect.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
