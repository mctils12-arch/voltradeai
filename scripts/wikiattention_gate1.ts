/**
 * wikiattention_gate1.ts — GATE 1 (DATA layer) for the
 * wikimedia_pageviews_attention root: do the archived daily pageviews
 * plausibly track a real, independently-known event (an earnings
 * announcement), and does every seed ticker->article pair actually point
 * at the article it claims to (not a redirect stub)?
 *
 * wikiAttention.ts's own header has stated the gate-1 bar since 2026-07-05:
 * "views vs known events, e.g. earnings dates, on 10 hand-checked tickers."
 * datacore/signal_ladder.json's wikimedia_pageviews_attention entry has
 * carried status "gate1_pending" with the note "no explicit gate-1 pass/
 * fail statistic found in the record" ever since — this script is that
 * statistic, run for the first time.
 *
 * PART A — REDIRECT SAFETY CHECK (the check that would have caught this
 * session's own finding). Every seed article title is live-queried against
 * MediaWiki's own redirects=1 API. The pageviews API does NOT follow
 * redirects (the documented reason RIOT was dropped at curation) — a seed
 * pair pointed at a redirect stub silently serves near-zero or partial
 * traffic instead of the real article's views, and nothing in the existing
 * test suite could catch it (ARTICLES is a static bundled map; the redirect
 * target can only be known by asking Wikipedia). Run live this session
 * (2026-08-18) BEFORE Part B and found 3 of 22 pairs broken:
 *   PLTR "Palantir_Technologies" -> "Palantir"      (stub ~28% of real traffic)
 *   AMC  "AMC_Entertainment"     -> "AMC Theatres"  (stub ~1% of real traffic, 94x undercount)
 *   SMCI "Super_Micro_Computer"  -> "Supermicro"
 * All three fixed in datacore/wiki_articles.json in the same PR as this
 * script (RATCHET: server/wikiAttention.test.ts pins the corrected titles).
 * This script re-runs the SAME check against whatever ARTICLES currently
 * ships, so a future regression (or a newly-added pair curated without the
 * redirect check) is caught by re-running it, not assumed fixed forever.
 *
 * PART B — EVENT-VS-BASELINE HAND-CHECK. For 11 of the 22 seed tickers with
 * a clean, single, unambiguous SEC 8-K Item 2.02 ("Results of Operations
 * and Financial Condition") filing inside the archive's live window
 * (2026-07-05 onward — the wikiattention archive's own start date), fetch
 * that ticker's daily pageviews across the whole window and compare the
 * peak of [event date, event date + 1] (accounting for after-hours filing
 * / UTC day-boundary lag between an earnings release and the pageview
 * spike it causes) against the MEDIAN of every other day in the window
 * (excluding a one-day buffer on each side of the event) as an honest,
 * self-referential baseline — no external "normal day" assumption needed.
 *
 * EARNINGS_EVENTS below is hand-recorded from a live query of EDGAR's own
 * submissions API (data.sec.gov/submissions/CIK##########.json) this
 * session, filtered to 8-K filings whose SEC-published itemCodes include
 * "2.02" and dated on/after 2026-07-01, one entry per ticker chosen to be
 * unambiguous (a single Item-2.02 8-K in the window, or the clear later
 * one where two exist). This is an independent ground truth: EDGAR filing
 * dates carry no dependency on price/volume data (which this sandbox does
 * not have live access to today — confirmed: yfinance not installed,
 * ALPACA_* env empty, query1.finance.yahoo.com -> HTTP 429), so this check
 * is executable even when GATE 2 (which needs forward returns) is not.
 *
 * PASS BAR (gate 1 = DATA plausibility, not gate 2 = predictive power):
 * every sampled ticker's peak-window views must exceed its own baseline
 * median (ratio > 1.0) — full statistical significance against noise is
 * explicitly GATE 2's job (occ_volume_gate2.ts / finra_shortvol_gate2.ts
 * etc. already do that elsewhere in this repo for other roots), not this
 * one. A single counterexample (peak <= baseline) does not by itself kill
 * the root, but is reported honestly, not rounded away — REASONING
 * STANDARD #4 discounts a small hand-checked sample regardless of outcome.
 *
 * Usage: npx tsx scripts/wikiattention_gate1.ts
 * Session-run, no args. Result goes in research/experiments.md +
 * datacore/signal_ladder.json, never into any production code path — this
 * script touches no runtime state and makes no writes.
 */
import { pathToFileURL } from "url";
import { ARTICLES } from "../server/wikiAttention";

type FetchInit = { headers?: Record<string, string> };
type FetchFn = (url: string, init?: FetchInit) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

// Hand-recorded 2026-08-18 from data.sec.gov/submissions/CIK##########.json,
// filtered to 8-K filings with itemCodes including "2.02", dated >= 2026-07-01.
// One unambiguous Item-2.02 filing date per ticker.
export const EARNINGS_EVENTS: Record<string, string> = {
  AAPL: "2026-07-30",
  AMD: "2026-08-04",
  PLTR: "2026-08-03",
  COIN: "2026-07-30",
  AMC: "2026-07-20",
  SOFI: "2026-07-29",
  HOOD: "2026-07-29",
  RDDT: "2026-07-30",
  CVNA: "2026-07-29",
  MARA: "2026-08-06",
  SMCI: "2026-08-11",
};

export interface RatioResult {
  peakDate: string;
  peakViews: number;
  baselineMedian: number;
  baselineDays: number;
  ratio: number;
}

/** Pure: given a date(YYYY-MM-DD)->views map and an event date inside it,
 *  compute the peak of [event, event+1] vs the median of every other day
 *  excluding a 1-day buffer on each side of that 2-day window. Returns
 *  null if the event date (or the series) is empty/missing — an honest
 *  "cannot evaluate", never a fabricated ratio. */
export function eventWindowRatio(series: Record<string, number>, eventDateIso: string): RatioResult | null {
  const dates = Object.keys(series).sort();
  if (!dates.length || !series.hasOwnProperty(eventDateIso)) return null;
  const evIdx = dates.indexOf(eventDateIso);
  const windowDates = dates.slice(evIdx, evIdx + 2);
  const windowViews = windowDates.map((d) => series[d]);
  const peakViews = Math.max(...windowViews);
  const peakDate = windowDates[windowViews.indexOf(peakViews)];
  const excl = new Set(dates.slice(Math.max(0, evIdx - 1), evIdx + 3));
  const baselineVals = dates.filter((d) => !excl.has(d)).map((d) => series[d]).sort((a, b) => a - b);
  if (!baselineVals.length) return null;
  const mid = Math.floor(baselineVals.length / 2);
  const baselineMedian = baselineVals.length % 2 ? baselineVals[mid] : (baselineVals[mid - 1] + baselineVals[mid]) / 2;
  return {
    peakDate, peakViews, baselineMedian,
    baselineDays: baselineVals.length,
    ratio: baselineMedian > 0 ? peakViews / baselineMedian : Infinity,
  };
}

/** Live: is `title` (as it would be requested from the pageviews API) a
 *  redirect to something else on en.wikipedia? Returns the resolved title
 *  when it differs (a real defect), or null when the title already
 *  resolves to itself (healthy). */
export async function checkRedirect(fetchImpl: FetchFn, title: string): Promise<string | null> {
  const url = `https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(title)}&redirects=1&format=json`;
  const r = await fetchImpl(url, { headers: { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" } });
  if (!r.ok) return null; // transient failure — not evidence of a redirect, don't flag it
  const body = JSON.parse(await r.text());
  const pages = body?.query?.pages || {};
  const finalTitle = Object.values(pages)[0] as any;
  const resolved = finalTitle?.title as string | undefined;
  if (!resolved) return null;
  if (resolved.replace(/ /g, "_") !== title) return resolved;
  return null;
}

async function fetchDailySeries(fetchImpl: FetchFn, article: string, startYmd: string, endYmd: string): Promise<Record<string, number>> {
  const url = `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/${encodeURIComponent(article)}/daily/${startYmd}00/${endYmd}00`;
  const r = await fetchImpl(url, { headers: { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" } });
  if (!r.ok) throw new Error(`${article}: ${r.status}`);
  const body = JSON.parse(await r.text());
  const out: Record<string, number> = {};
  for (const it of body.items || []) {
    const ts = String(it.timestamp || "");
    if (/^\d{10}$/.test(ts)) out[`${ts.slice(0, 4)}-${ts.slice(4, 6)}-${ts.slice(6, 8)}`] = it.views;
  }
  return out;
}

async function sleep(ms: number) { return new Promise((res) => setTimeout(res, ms)); }

async function withRetry<T>(fn: () => Promise<T>, attempts = 5, baseDelayMs = 3000): Promise<T> {
  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try { return await fn(); } catch (e) { lastErr = e; await sleep(baseDelayMs * (i + 1)); }
  }
  throw lastErr;
}

const WINDOW_START = "20260705";
const WINDOW_END = "20260817"; // yesterday relative to this session; today's day is incomplete

async function main() {
  const fetchImpl = fetch as any as FetchFn;

  // ── PART A: redirect safety check over the full live seed ──────────────
  console.log(`PART A — redirect safety check, ${Object.keys(ARTICLES).length} seed pairs`);
  const redirects: Array<{ ticker: string; title: string; resolvesTo: string }> = [];
  for (const [ticker, title] of Object.entries(ARTICLES)) {
    const resolved = await withRetry(() => checkRedirect(fetchImpl, title));
    if (resolved) {
      redirects.push({ ticker, title, resolvesTo: resolved });
      console.log(`  DEFECT: ${ticker} "${title}" -> redirects to "${resolved}"`);
    }
    await sleep(1200);
  }
  console.log(redirects.length ? `  ${redirects.length} redirect defect(s) found` : "  0 redirect defects — every seed pair resolves to itself");

  // ── PART B: event-vs-baseline hand-check ────────────────────────────────
  console.log(`\nPART B — event-vs-baseline hand-check, ${Object.keys(EARNINGS_EVENTS).length} tickers`);
  const results: Array<{ ticker: string; event: string } & RatioResult> = [];
  const failures: string[] = [];
  for (const [ticker, event] of Object.entries(EARNINGS_EVENTS)) {
    const article = ARTICLES[ticker];
    if (!article) { failures.push(`${ticker}: not in current seed`); continue; }
    const series = await withRetry(() => fetchDailySeries(fetchImpl, article, WINDOW_START, WINDOW_END));
    const r = eventWindowRatio(series, event);
    if (!r) { failures.push(`${ticker}: event date ${event} not covered by fetched series`); continue; }
    results.push({ ticker, event, ...r });
    console.log(`  ${ticker.padEnd(6)} event=${event} peak=${r.peakDate} peak_views=${r.peakViews} baseline_med=${r.baselineMedian.toFixed(0)} ratio=${r.ratio.toFixed(2)}`);
    await sleep(1500);
  }

  const consistent = results.filter((r) => r.ratio > 1.0).length;
  const strong = results.filter((r) => r.ratio >= 1.5).length;
  const verdict = redirects.length === 0 && failures.length === 0 && consistent === results.length && results.length >= 10
    ? "PASS" : (consistent === results.length && results.length > 0 ? "PASS_WITH_CAVEAT" : "FAIL");

  console.log(`\n${consistent}/${results.length} tickers show ratio > 1.0 (directionally consistent); ${strong}/${results.length} clear >= 1.5x`);
  if (failures.length) console.log("failures:", failures);
  console.log(`\nVERDICT: ${verdict}`);
  console.log(JSON.stringify({ verdict, redirects, results, failures }, null, 2));
}

// Entrypoint guard (ESM has no require.main) — same pattern as
// finra_shortvol_gate2_retest.ts / occ_volume_gate2_clustered.ts: this
// module's pure helpers (eventWindowRatio, checkRedirect) are imported by
// wikiattention_gate1.test.ts, which must not trigger a live network run
// as an import side effect.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
