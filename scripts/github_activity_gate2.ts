/**
 * github_activity_gate2.ts — GATE 2 (SIGNAL) readiness check + exploratory
 * run for the github_org_engineering_momentum root (server/githubOrgActivity.ts):
 * does a develop-in-public company's WEEK-OVER-WEEK GitHub velocity
 * (merged PRs + commits) carry forward-return information, measured with
 * no trading involved (CLAUDE.md's ROOT VALIDATION LADDER gate-2
 * definition)?
 *
 * WHY NOW: this root's own ladder note (datacore/signal_ladder.json,
 * last_update_date 2026-09-08) recorded the weekly archiver going dark
 * for 8 straight days and left GATE 2 explicitly "unstarted" pending a
 * root-cause. Live-checked this session (2026-09-28, curl against
 * production): the archive has in fact recovered and now holds 10
 * CONSECUTIVE clean weeks (2026-07-20 .. 2026-09-27, all 15 watchlist
 * orgs reporting every week, /api/data/github-activity/history) — the
 * 2026-09-08 stall was transient and is not reproducing. That is the
 * first point in this root's life where a real, if thin, gate-2 attempt
 * is possible at all.
 *
 * OPERATIONALIZATION (pre-registered before any price data is fetched,
 * Reasoning Standard #10 — one definition, not fished across variants):
 *   - Weekly total per org = mergedPRs + commits (the same two fields the
 *     archiver's own RAW display already sums; uniqueActorsSample is
 *     capped/bot-sampling-only and excluded, per the module's own honesty
 *     note).
 *   - Signal = week-over-week PERCENT change in that total, computed per
 *     ticker against its OWN prior week (not a raw count delta) —
 *     required because the 15-org panel spans a >20x scale range
 *     (PagerDuty vs Cloudflare/Datadog); ranking by raw count would just
 *     re-derive "biggest org" every week, never "accelerating org".
 *   - Cross-sectional ranking PER WEEK (mirrors occ_volume_gate2.ts's own
 *     bucket convention): sort the panel's valid pct-changes that week,
 *     TOP = fastest-accelerating third, BOTTOM = fastest-decelerating
 *     third, MID = the remaining middle third (base-rate control,
 *     Reasoning Standard #3). With a 15-org panel this is a clean 5/5/5
 *     split; a week where fewer than MIN_PANEL_FOR_WEEK orgs have a valid
 *     delta is dropped entirely rather than bucketed unevenly.
 *   - Entry date = the latest Yahoo-adjusted-close trading day on or
 *     before that week's weekEnd date (weeks end mid-archive-cycle, not
 *     always a trading day). Forward return = close-to-close at +5 and
 *     +20 TRADING days later (fetchYahooDaily/toSeries/fwdReturn reused
 *     verbatim from occ_volume_gate2.ts — no lookahead, Reasoning
 *     Standard #7: only entry dates whose forward window is fully in the
 *     past are ever scored).
 *   - Cluster = the WEEK, not the (week,ticker) row — the same fix
 *     statsUtils.ts's clusterMeanTTest was built for (occ_options_volume
 *     2026-08-03 gate-2 kill): every ticker in one week's TOP/BOTTOM
 *     bucket shares that week's market-wide move, so the raw row is not
 *     an independent observation. Cluster value = that week's
 *     (TOP mean +5d return) - (BOTTOM mean +5d return); the gating test
 *     is a one-sample cluster-mean t-test on those per-week spreads
 *     (clusterMeanTTest + survivesAtCrit005, exact df-correct critical
 *     value, not a flat |t|>2 heuristic).
 *   - +20d horizon: informational only, never gating — the archive is
 *     far too young for enough weeks to clear +20 trading days
 *     (~28 calendar days) in the past.
 *
 * PRE-REGISTERED READINESS BAR (stated before this session read any
 * number): MIN_CLUSTERS = 15 usable (+5d-realized) weeks. This mirrors
 * the ~15-30-cluster minimums this repo's other cluster-mean gate-2
 * scripts (occ_volume_gate2/finra_shortvol_gate2/settlement_stress_gate2)
 * already use as their own floor for a cluster-mean t-test to have any
 * realistic power — picked by that precedent, not backfit to whatever
 * count this run happens to produce. Below that bar this script reports
 * READY/WAITING only, never PASS/FAIL (Reasoning Standard #4 — a result
 * from too few clusters is not evidence in either direction).
 *
 * SOBER PRIOR (restated from server/githubOrgActivity.ts's own header,
 * unchanged by this script): expect real structure for maybe a third of
 * this panel (genuine develop-in-public names) and noise for the rest;
 * a null or weak pooled result does not by itself falsify the hypothesis
 * for the develop-in-public subset — that per-ticker breakdown is
 * printed alongside the pooled verdict for exactly that reason, but is
 * NOT itself gating (Reasoning Standard #4 — slicing 15 tickers after
 * the fact and picking whichever look good would be multiple-hypothesis
 * fishing).
 *
 * Session-run: `npx tsx scripts/github_activity_gate2.ts [--dry] [prodBaseUrl]`
 * --dry skips the live fetch (production RAW history endpoint + Yahoo)
 * and only prints the pure-function behavior exercised by
 * github_activity_gate2.test.ts. No DIAG_TOKEN needed — the RAW history
 * endpoint this pulls from is public. Touches no runtime path; result
 * goes in research/experiments.md + datacore/signal_ladder.json.
 */
import { pathToFileURL } from "url";
import { WATCHLIST } from "../server/githubOrgActivity";
import { fetchYahooDaily, toSeries, fwdReturn, type Series } from "./occ_volume_gate2";
import { clusterMeanTTest, survivesAtCrit005 } from "./statsUtils";

export const MIN_CLUSTERS = 15;
export const MIN_PANEL_FOR_WEEK = 12; // out of 15 — allow a couple of missing orgs before dropping the week

export interface OrgWeek {
  ticker: string;
  weekStart: string;
  weekEnd: string;
  total: number | null; // mergedPRs + commits, null if either field missing
}

export function toWeeklyTotal(rec: { weekStart: string; weekEnd: string; mergedPRs: number | null; commits: number | null }): OrgWeek {
  const total = rec.mergedPRs != null && rec.commits != null ? rec.mergedPRs + rec.commits : null;
  return { ticker: "", weekStart: rec.weekStart, weekEnd: rec.weekEnd, total };
}

export interface WeekDelta {
  ticker: string;
  weekEnd: string;
  pctChange: number;
}

/** Week-over-week percent change per ticker, computed against each
 *  ticker's own prior archived week only (never cross-ticker). Pure
 *  function over already-fetched per-ticker series (weeks sorted
 *  ascending by weekStart, one series per ticker). */
export function computeWeeklyDeltas(seriesByTicker: Map<string, OrgWeek[]>): WeekDelta[] {
  const out: WeekDelta[] = [];
  for (const [ticker, weeks] of seriesByTicker) {
    for (let i = 1; i < weeks.length; i++) {
      const prev = weeks[i - 1].total;
      const cur = weeks[i].total;
      if (prev == null || cur == null || prev === 0) continue;
      out.push({ ticker, weekEnd: weeks[i].weekEnd, pctChange: (cur - prev) / prev });
    }
  }
  return out;
}

export interface WeekBucket {
  weekEnd: string;
  top: string[];
  mid: string[];
  bottom: string[];
}

/** Cross-sectional TOP/MID/BOTTOM terciles for one week's deltas. Returns
 *  null if fewer than MIN_PANEL_FOR_WEEK tickers have a valid delta that
 *  week (dropped entirely, never bucketed unevenly). */
export function bucketWeek(weekEnd: string, deltas: WeekDelta[]): WeekBucket | null {
  const rows = deltas.filter((d) => d.weekEnd === weekEnd);
  if (rows.length < MIN_PANEL_FOR_WEEK) return null;
  const sorted = [...rows].sort((a, b) => b.pctChange - a.pctChange);
  const third = Math.floor(sorted.length / 3);
  return {
    weekEnd,
    top: sorted.slice(0, third).map((r) => r.ticker),
    mid: sorted.slice(third, sorted.length - third).map((r) => r.ticker),
    bottom: sorted.slice(sorted.length - third).map((r) => r.ticker),
  };
}

/** Latest date in `series` on or before `onOrBefore` (ISO yyyy-mm-dd
 *  string compare is safe since both are zero-padded). Null if the
 *  series has no date that early. */
export function nearestTradingDayOnOrBefore(series: Series, onOrBefore: string): string | null {
  let best: string | null = null;
  for (const d of series.dates) {
    if (d <= onOrBefore && (best == null || d > best)) best = d;
  }
  return best;
}

export interface WeekResult {
  weekEnd: string;
  topMean5: number | null;
  midMean5: number | null;
  bottomMean5: number | null;
  spread5: number | null; // top - bottom, this week's cluster value
  topMean20: number | null;
  midMean20: number | null;
  bottomMean20: number | null;
}

export function meanOf(xs: (number | null)[]): number | null {
  const clean = xs.filter((x): x is number => x != null);
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

const DRY = process.argv.includes("--dry");
const BASE = process.argv.find((a, i) => i >= 2 && !a.startsWith("--")) || process.env.VOLTRADE_PROD_URL || "https://voltradeai.com";

async function fetchTickerHistory(ticker: string): Promise<OrgWeek[]> {
  const url = `${BASE}/api/data/github-activity/history?ticker=${encodeURIComponent(ticker)}&weeks=90`;
  const r = await fetch(url, { signal: AbortSignal.timeout(20000) as unknown as AbortSignal });
  if (!r.ok) throw new Error(`${ticker}: HTTP ${r.status}`);
  const body = await r.json();
  const series = Array.isArray(body?.series) ? body.series : [];
  return series
    .map((rec: any) => ({ ...toWeeklyTotal(rec), ticker }))
    .sort((a: OrgWeek, b: OrgWeek) => a.weekStart.localeCompare(b.weekStart));
}

async function main() {
  if (DRY) {
    console.log("--dry: skipping live fetch. Pure functions (toWeeklyTotal/computeWeeklyDeltas/bucketWeek/nearestTradingDayOnOrBefore) are exercised by github_activity_gate2.test.ts.");
    return;
  }

  const seriesByTicker = new Map<string, OrgWeek[]>();
  for (const org of WATCHLIST) {
    try {
      seriesByTicker.set(org.ticker, await fetchTickerHistory(org.ticker));
    } catch (e: any) {
      console.log(`  ${org.ticker}: history fetch failed (${e?.message || e}), excluded from panel`);
    }
    await new Promise((res) => setTimeout(res, 150));
  }

  const deltas = computeWeeklyDeltas(seriesByTicker);
  const weekEnds = Array.from(new Set(deltas.map((d) => d.weekEnd))).sort();
  console.log(`panel: ${seriesByTicker.size}/${WATCHLIST.length} tickers fetched, ${deltas.length} weekly deltas across ${weekEnds.length} candidate weeks`);

  const buckets = weekEnds.map((w) => bucketWeek(w, deltas)).filter((b): b is WeekBucket => b != null);
  console.log(`${buckets.length} weeks clear MIN_PANEL_FOR_WEEK=${MIN_PANEL_FOR_WEEK} (dropped ${weekEnds.length - buckets.length})`);

  // One Yahoo range fetch per ticker, spanning the full panel + buffer.
  const allDates = Array.from(seriesByTicker.values()).flat().map((w) => w.weekEnd).sort();
  const startSec = new Date(`${allDates[0]}T00:00:00Z`).getTime() / 1000 - 5 * 86400;
  const endSec = Date.now() / 1000 + 1 * 86400;
  const priceSeries = new Map<string, Series>();
  for (const org of WATCHLIST) {
    if (!seriesByTicker.has(org.ticker)) continue;
    try {
      priceSeries.set(org.ticker, toSeries(await fetchYahooDaily(org.ticker, startSec, endSec)));
    } catch (e: any) {
      console.log(`  ${org.ticker}: Yahoo price fetch failed (${e?.message || e}), excluded from pricing`);
    }
    await new Promise((res) => setTimeout(res, 200));
  }

  function bucketFwd(tickers: string[], weekEnd: string, horizon: number): (number | null)[] {
    return tickers.map((t) => {
      const s = priceSeries.get(t);
      if (!s) return null;
      const entry = nearestTradingDayOnOrBefore(s, weekEnd);
      return entry ? fwdReturn(s, entry, horizon) : null;
    });
  }

  const results: WeekResult[] = buckets.map((b) => {
    const top5 = bucketFwd(b.top, b.weekEnd, 5);
    const mid5 = bucketFwd(b.mid, b.weekEnd, 5);
    const bot5 = bucketFwd(b.bottom, b.weekEnd, 5);
    const top20 = bucketFwd(b.top, b.weekEnd, 20);
    const mid20 = bucketFwd(b.mid, b.weekEnd, 20);
    const bot20 = bucketFwd(b.bottom, b.weekEnd, 20);
    const topMean5 = meanOf(top5), midMean5 = meanOf(mid5), bottomMean5 = meanOf(bot5);
    return {
      weekEnd: b.weekEnd,
      topMean5, midMean5, bottomMean5,
      spread5: topMean5 != null && bottomMean5 != null ? topMean5 - bottomMean5 : null,
      topMean20: meanOf(top20), midMean20: meanOf(mid20), bottomMean20: meanOf(bot20),
    };
  });

  console.log(`\nper-week buckets (+5d realized only):`);
  for (const r of results) {
    const fmt = (x: number | null) => (x == null ? "n/a" : `${(x * 100).toFixed(2)}%`);
    console.log(`  ${r.weekEnd}: TOP=${fmt(r.topMean5)} MID=${fmt(r.midMean5)} BOTTOM=${fmt(r.bottomMean5)} spread=${fmt(r.spread5)}`);
  }

  const usable5 = results.filter((r) => r.spread5 != null);
  const ready = usable5.length >= MIN_CLUSTERS;

  console.log(`\n=== GATE 2 READINESS (+5d, cluster = week) ===`);
  console.log(`${usable5.length} usable weeks vs MIN_CLUSTERS=${MIN_CLUSTERS}`);
  console.log(ready ? "READY" : "WAITING", `— not enough realized +5d weeks yet to render a PASS/FAIL verdict (Reasoning Standard #4)`);

  const spreads = usable5.map((r) => r.spread5 as number);
  const clusterTest = spreads.length ? clusterMeanTTest(spreads) : null;
  const pooledTop = meanOf(usable5.map((r) => r.topMean5));
  const pooledMid = meanOf(usable5.map((r) => r.midMean5));
  const pooledBottom = meanOf(usable5.map((r) => r.bottomMean5));
  const orderingHolds = pooledTop != null && pooledMid != null && pooledBottom != null && pooledTop > pooledMid && pooledMid > pooledBottom;
  const significant = clusterTest ? survivesAtCrit005(clusterTest) : false;
  const verdict = ready ? (orderingHolds && significant ? "PASS" : "FAIL") : "WAITING";

  console.log(`\n=== EXPLORATORY (non-gating at n=${usable5.length} until READY) ===`);
  console.log(`pooled TOP mean +5d: ${pooledTop != null ? (pooledTop * 100).toFixed(2) + "%" : "n/a"}`);
  console.log(`pooled MID mean +5d: ${pooledMid != null ? (pooledMid * 100).toFixed(2) + "%" : "n/a"}`);
  console.log(`pooled BOTTOM mean +5d: ${pooledBottom != null ? (pooledBottom * 100).toFixed(2) + "%" : "n/a"}`);
  console.log(`ordering TOP>MID>BOTTOM holds: ${orderingHolds}`);
  if (clusterTest) {
    console.log(`cluster-mean t-test on weekly spreads: n=${clusterTest.n} mean=${(clusterTest.mean * 100).toFixed(2)}% t=${clusterTest.t.toFixed(3)} df=${clusterTest.df} survivesAtCrit005=${significant}`);
  }

  console.log(JSON.stringify({
    verdict,
    n_clusters: usable5.length,
    min_clusters: MIN_CLUSTERS,
    pooledTop5: pooledTop, pooledMid5: pooledMid, pooledBottom5: pooledBottom,
    orderingHolds, clusterTest, results,
  }, null, 2));

  process.exitCode = verdict === "PASS" || verdict === "WAITING" ? 0 : 1;
}

// Entrypoint guard — importing this file's pure functions from a test
// must never re-trigger the live network fetch (same fix class as
// settlement_stress_gate2.ts's own header note).
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
