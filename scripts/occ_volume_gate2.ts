/**
 * occ_volume_gate2.ts — GATE 2 (SIGNAL layer) for the occ_options_volume
 * root: does customer options put/call positioning BY TICKER carry
 * forward-return information, measured with NO trading involved?
 *
 * PRE-REGISTERED CRITERIA (stated BEFORE running, Reasoning Standard #10).
 *
 * OPERATIONALIZATION (one definition, not fished across variants):
 *   - Signal = CUSTOMER-ONLY call/put skew per underlying per day:
 *       skew = (cust_call - cust_put) / (cust_call + cust_put)  in [-1, 1]
 *     Customer-only (actype "C"), excluding market-maker/firm rows, matches
 *     the retired ISEE sentiment index's own methodology (ISEE used
 *     customer-only volume specifically because MM/firm flow is mechanical
 *     hedging, not sentiment) — this is why the root's hypothesis text
 *     calls it "the retired ISEE's successor," not a literal customer-
 *     minus-MM difference.
 *   - HONESTY CAVEAT (stated before running, lowers the prior): OCC's public
 *     volume-query feed has NO buy/sell direction and NO opening/closing
 *     distinction — unlike the real ISEE, which used opening-buy volume
 *     only. "Customer put volume" here mixes new put buyers (bearish),
 *     put sellers closing out, cash-secured-put writers (bullish premium
 *     sellers), and covering buyback — all four blended into one number.
 *     This is a structurally CRUDER, noisier proxy than true ISEE. PRIOR:
 *     expect a weak or null effect on this raw feed; a real edge would be
 *     stronger on an opening-only feed we don't have access to.
 *   - Universe per day: underlyings with cust_put + cust_call >= FLOOR
 *     contracts (noise floor — calibrated once against LIVE VOLUME on a
 *     sample day, never against the outcome: 2026-04-01 had 4,348
 *     underlyings, median cust volume 128 contracts; a first pass at
 *     FLOOR=1000 still needed ~1,100 Yahoo price series/day — too many
 *     for a session-run script to fetch within a reasonable wall time —
 *     so FLOOR was raised to 8,000 purely to bound NETWORK COST, landing
 *     ~450-500 qualifying/day, still well above single-name noise and
 *     still reaching into real small/mid-caps per EDGE DOCTRINE #2, not
 *     only mega-caps).
 *   - Buckets per day (rank by skew, symmetric around the middle so bucket
 *     SIZE, not selection, is the only free parameter): FIXED size
 *     BUCKET_SIZE=40 (not a decile fraction — also a network-cost bound,
 *     chosen before any price data was looked at):
 *       CALL_SKEW = top 40 by skew (most call-heavy)
 *       PUT_SKEW  = bottom 40 by skew (most put-heavy)
 *       NEUTRAL   = 40 tickers centered on the middle of the qualifying
 *         range — the base-rate control (Reasoning Standard #3: "what
 *         would random entry with the same universe/holding period have
 *         returned"). A full-universe base rate would need a Yahoo call
 *         for every qualifying name (~450-500/day) — a size-matched
 *         neutral-skew band gives the same base-rate function at bounded
 *         network cost, stated here as a scope limitation, not hidden.
 *   - Forward return: close-to-close from the OCC report day to +5 and
 *     +20 TRADING days later (Yahoo adjusted close). Already-realized
 *     history — no lookahead risk (Reasoning Standard #7): we are
 *     measuring past outcomes, not predicting live.
 *   - Sample: 16 weekly Wednesdays, 2026-01-07 through 2026-04-15 (within
 *     OCC's rolling 2-year window; ends >100 calendar days before this
 *     script's run date so +20-trading-day outcomes are fully realized).
 *     Weekly spacing bounds total OCC fetches to 16 full-day CSVs
 *     (~5MB each) while still spanning a full quarter across BULL/CAUTION/
 *     NEUTRAL-ish regime texture per Reasoning Standard #2.
 *   - PASS bar (pre-stated, falsifiable, single test — no multiple-
 *     hypothesis fishing): CALL_SKEW mean forward return > NEUTRAL mean >
 *     PUT_SKEW mean, on BOTH horizons, with |Welch t| > 2 on the +20d
 *     spread (CALL_SKEW vs PUT_SKEW; the more theory-relevant horizon
 *     since options positioning reflects short/medium conviction, not
 *     next-day noise). Given the honesty caveat above, ANY pass here is
 *     "promising, needs an opening-only refinement + true out-of-sample
 *     replication before being trusted" per Reasoning Standard #4 — this
 *     single run is not a promotion to LOGIC (gate 3).
 *   - PRIOR stated before running: ~30% chance of a clean pass on both
 *     horizons, given the direction/open-close dilution above; a partial
 *     or null result is the more likely honest outcome and is itself a
 *     useful finding (it would mean the total-volume feed alone can't
 *     carry the ISEE-style signal, sharpening what a future opening-only
 *     data source would need to add).
 *
 * Session-run: `npx tsx scripts/occ_volume_gate2.ts [--dry]`
 * --dry skips the Yahoo price fetch/return computation and only prints
 * bucket sizes + unique-ticker counts, to size the network cost before
 * committing to the full run (a calibration step on VOLUME, never on
 * the outcome — the skew/bucket/floor definitions above are fixed before
 * any price data is looked at).
 * Result goes in research/experiments.md + datacore/signal_ladder.json,
 * never into any runtime path — this script touches no production state.
 */
import { fetchOccDay, aggregateOcc, type UnderlyingStat } from "../server/occVolume";

const FLOOR = 8000;
const BUCKET_SIZE = 40;
const HORIZONS = [5, 20] as const;
const SAMPLE_START = "2026-01-07"; // Wednesday
const SAMPLE_WEEKS = 16;
const DRY = process.argv.includes("--dry");

function weeklySampleDays(startYmd: string, weeks: number): string[] {
  const [y, m, d] = startYmd.split("-").map(Number);
  const out: string[] = [];
  const cur = new Date(Date.UTC(y, m - 1, d));
  for (let i = 0; i < weeks; i++) {
    out.push(cur.toISOString().slice(0, 10));
    cur.setUTCDate(cur.getUTCDate() + 7);
  }
  return out;
}

function ymdCompact(iso: string): string {
  return iso.replace(/-/g, "");
}

interface SkewRow {
  ticker: string;
  skew: number;
  volume: number;
}

interface DayBuckets {
  day: string;
  qualifying: number;
  callSkew: SkewRow[];
  putSkew: SkewRow[];
  neutral: SkewRow[];
}

function bucketDay(day: string, stats: UnderlyingStat[]): DayBuckets {
  const qualifying: SkewRow[] = stats
    .filter((s) => s.cust_put + s.cust_call >= FLOOR)
    .map((s) => ({
      ticker: s.underlying,
      volume: s.cust_put + s.cust_call,
      skew: (s.cust_call - s.cust_put) / (s.cust_call + s.cust_put),
    }))
    .sort((a, b) => a.skew - b.skew); // ascending: most put-heavy first

  const n = qualifying.length;
  const bucketSize = Math.max(1, Math.min(BUCKET_SIZE, Math.floor(n / 3)));
  const putSkew = qualifying.slice(0, bucketSize);
  const callSkew = qualifying.slice(n - bucketSize);
  const midStart = Math.floor((n - bucketSize) / 2);
  const neutral = qualifying.slice(midStart, midStart + bucketSize);

  return { day, qualifying: n, callSkew, putSkew, neutral };
}

// ── Yahoo daily adjusted close, one range fetch per ticker ─────────────────

async function fetchYahooDaily(symbol: string, startSec: number, endSec: number): Promise<Map<string, number>> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}` +
    `?period1=${startSec}&period2=${endSec}&interval=1d&events=div%2Csplit`;
  let lastErr: any;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(url, {
        headers: { "User-Agent": "Mozilla/5.0 (voltradeai-datacore-research/1.0)" },
        signal: AbortSignal.timeout(20000) as any,
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j: any = await r.json();
      const res = j?.chart?.result?.[0];
      if (!res) throw new Error("no chart result");
      const stamps: number[] = res.timestamp || [];
      const q = res.indicators?.quote?.[0] || {};
      const adjArr: number[] | undefined = res.indicators?.adjclose?.[0]?.adjclose;
      const out = new Map<string, number>();
      for (let i = 0; i < stamps.length; i++) {
        const close = q.close?.[i];
        if (close == null) continue;
        const adj = adjArr?.[i] ?? close;
        const date = new Date(stamps[i] * 1000).toISOString().slice(0, 10);
        out.set(date, adj);
      }
      return out;
    } catch (e: any) {
      lastErr = e;
      await new Promise((res) => setTimeout(res, 1200 * (attempt + 1)));
    }
  }
  throw lastErr;
}

interface Series { dates: string[]; closes: number[]; indexOf: Map<string, number>; }

function toSeries(map: Map<string, number>): Series {
  const dates = Array.from(map.keys()).sort();
  const closes = dates.map((d) => map.get(d)!);
  const indexOf = new Map(dates.map((d, i) => [d, i]));
  return { dates, closes, indexOf };
}

function fwdReturn(series: Series, entryDate: string, horizonTradingDays: number): number | null {
  const idx = series.indexOf.get(entryDate);
  if (idx == null) return null;
  const fwdIdx = idx + horizonTradingDays;
  if (fwdIdx >= series.dates.length) return null;
  const entry = series.closes[idx];
  const fwd = series.closes[fwdIdx];
  if (!entry || !Number.isFinite(entry) || !Number.isFinite(fwd)) return null;
  return fwd / entry - 1;
}

function meanStd(arr: number[]): { mean: number; std: number; median: number; n: number } {
  const n = arr.length;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  const variance = n > 1 ? arr.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : NaN;
  return { mean, std: Math.sqrt(variance), median, n };
}

function welchT(a: number[], b: number[]): number {
  const sa = meanStd(a), sb = meanStd(b);
  const se = Math.sqrt((sa.std ** 2) / sa.n + (sb.std ** 2) / sb.n);
  if (se === 0) return NaN;
  return (sa.mean - sb.mean) / se;
}

async function main() {
  const days = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  console.log(`occ_volume_gate2 ${DRY ? "(--dry calibration run)" : "(full run)"}: ${days.length} weekly sample days, FLOOR=${FLOOR}, BUCKET_SIZE=${BUCKET_SIZE}`);

  const dayBuckets: DayBuckets[] = [];
  let skippedDays = 0;
  for (const day of days) {
    const { kind, rows } = await fetchOccDay(ymdCompact(day));
    if (kind !== "data" || !rows.length) {
      console.log(`  ${day}: skipped (${kind}) — honest no-op`);
      skippedDays++;
      continue;
    }
    const { top } = aggregateOcc(rows, 999_999);
    const b = bucketDay(rows[0].date, top);
    dayBuckets.push(b);
    console.log(`  ${b.day}: qualifying=${b.qualifying} bucketSize=${b.callSkew.length}`);
    await new Promise((res) => setTimeout(res, 500));
  }

  const tickerSet = new Set<string>();
  let totalObs = 0;
  for (const b of dayBuckets) {
    for (const r of [...b.callSkew, ...b.putSkew, ...b.neutral]) {
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

  // one Yahoo range fetch per unique ticker: earliest sample day -> latest + 45 calendar days buffer
  const firstDay = new Date(dayBuckets[0].day + "T00:00:00Z").getTime() / 1000;
  const lastDay = new Date(dayBuckets[dayBuckets.length - 1].day + "T00:00:00Z").getTime() / 1000;
  const startSec = firstDay - 5 * 86400;
  const endSec = lastDay + 45 * 86400;

  // Batched concurrency (politeness via small batches + inter-batch delay,
  // not per-request serialization — 852+ unique tickers serially at Yahoo's
  // observed ~0.5-1s/request would run 15+ minutes; BATCH keeps total wall
  // time bounded while still respecting the same host).
  const seriesCache = new Map<string, Series>();
  let fetched = 0, failed = 0;
  const tickers = Array.from(tickerSet);
  const CONCURRENCY = 8;
  for (let i = 0; i < tickers.length; i += CONCURRENCY) {
    const batch = tickers.slice(i, i + CONCURRENCY);
    const results = await Promise.all(batch.map(async (t) => {
      try {
        const map = await fetchYahooDaily(t, startSec, endSec);
        return { t, map };
      } catch {
        return { t, map: null };
      }
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

  const returns: Record<string, Record<number, number[]>> = {
    callSkew: { 5: [], 20: [] },
    putSkew: { 5: [], 20: [] },
    neutral: { 5: [], 20: [] },
  };
  const lookupMisses: Record<string, number> = { callSkew: 0, putSkew: 0, neutral: 0 };

  for (const b of dayBuckets) {
    for (const [bucketName, rows] of Object.entries({ callSkew: b.callSkew, putSkew: b.putSkew, neutral: b.neutral })) {
      for (const r of rows) {
        const series = seriesCache.get(r.ticker);
        if (!series) { lookupMisses[bucketName]++; continue; }
        for (const h of HORIZONS) {
          const ret = fwdReturn(series, b.day, h);
          if (ret == null) { continue; }
          returns[bucketName][h].push(ret);
        }
      }
    }
  }

  console.log(`\nlookup misses (ticker had no price series at all): ${JSON.stringify(lookupMisses)}`);
  console.log(`\n=== RESULTS (customer-only call/put skew, forward returns) ===`);
  for (const h of HORIZONS) {
    const call = meanStd(returns.callSkew[h]);
    const put = meanStd(returns.putSkew[h]);
    const neu = meanStd(returns.neutral[h]);
    const t = welchT(returns.callSkew[h], returns.putSkew[h]);
    console.log(`\n+${h}d horizon:`);
    console.log(`  CALL_SKEW: n=${call.n} mean=${(call.mean * 100).toFixed(3)}% median=${(call.median * 100).toFixed(3)}% std=${(call.std * 100).toFixed(2)}%`);
    console.log(`  NEUTRAL:   n=${neu.n} mean=${(neu.mean * 100).toFixed(3)}% median=${(neu.median * 100).toFixed(3)}% std=${(neu.std * 100).toFixed(2)}% (base rate)`);
    console.log(`  PUT_SKEW:  n=${put.n} mean=${(put.mean * 100).toFixed(3)}% median=${(put.median * 100).toFixed(3)}% std=${(put.std * 100).toFixed(2)}%`);
    console.log(`  spread (CALL-PUT): ${((call.mean - put.mean) * 100).toFixed(3)}%   Welch t=${t.toFixed(3)}`);
  }

  const call20 = meanStd(returns.callSkew[20]).mean, put20 = meanStd(returns.putSkew[20]).mean, neu20 = meanStd(returns.neutral[20]).mean;
  const call5 = meanStd(returns.callSkew[5]).mean, put5 = meanStd(returns.putSkew[5]).mean, neu5 = meanStd(returns.neutral[5]).mean;
  const t20 = welchT(returns.callSkew[20], returns.putSkew[20]);
  const orderedBoth = call5 > neu5 && neu5 > put5 && call20 > neu20 && neu20 > put20;
  const verdict = orderedBoth && Math.abs(t20) > 2 ? "PASS" : "FAIL/INCONCLUSIVE";
  console.log(`\nVERDICT (pre-stated bar: CALL>NEUTRAL>PUT on both horizons AND |t20|>2): ${verdict}`);
  console.log(`ordered_both_horizons=${orderedBoth} welch_t_20d=${t20.toFixed(3)}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
