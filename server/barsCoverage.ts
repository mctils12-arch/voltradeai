/**
 * barsCoverage.ts — READ-ONLY probe: do Alpaca's free daily stock bars reach
 * back to 2010, and do they include DELISTED tickers? (human order
 * 2026-10-28: "Check whether Alpaca's free bars include delisted tickers since
 * 2010. If they do, build the price panel from them. No paid data.")
 *
 * WHY: the deferred strategy-level backtest (docs/regime/FINDINGS.md Section
 * 10) and the Lazy Prices gate 2 (research/wishlist.md, 2026-10-18 entry) both
 * need a point-in-time, survivorship-free daily price panel. A panel of
 * today's survivors flatters every strategy that avoids later failures. The
 * wishlist's build-first step (1) is exactly this free probe; the sandbox has
 * no Alpaca keys, so it runs in production behind the diag token.
 *
 * WHAT IS PROBED (stock market-data API v2, GET only):
 *  - GET /v2/stocks/{sym}/bars?timeframe=1Day&adjustment=all&start=2010-01-01
 *    &end={yesterday}&limit=10000&feed={sip|iex}[&asof={date}]
 *      * SPY / AAPL / IWM on both feeds: the EARLIEST bar answers "does
 *        history reach 2010 or start ~2016?".
 *      * ~17 known delisted / acquired / bankrupt / renamed tickers
 *        (2012-2023), three times on sip: with the API's default symbol
 *        mapping (asof = today), with `asof` = a date while the old company
 *        still traded, and with `asof=-` (mapping disabled: the raw ticker
 *        string as traded). The three expose TICKER REUSE (S = Sprint until
 *        2020, then SentinelOne from 2021; DELL = old Dell until 2013, new
 *        Dell from 2018) and decide the panel builder's design: one batched
 *        multi-symbol pass works only if a mode that needs no per-symbol date
 *        (default or `-`) returns the delisted history. iex is tried only
 *        where sip@asof is refused.
 *      * one continuously listed control (SBUX).
 *    `end` is yesterday because the free plan refuses the most recent 15
 *    minutes of SIP data.
 *  - GET /v2/assets?status=inactive|active&asset_class=us_equity (trading API):
 *    counts only, a few inactive symbols as examples, how many inactive
 *    symbols are ALSO active today (a ticker-reuse indicator), and whether each
 *    probed delisted ticker appears in the inactive list. The asset object has
 *    no delisting date — first/last bar dates are the only dating available.
 *
 * WHAT LEAVES: per series {symbol, feed, asof, http, verdict, a vendor-message
 * CLASS token, bar count, first/last bar date, first/last close}, the summary
 * and the asset counts. Never a URL, a response body, a raw vendor message or
 * a key. Keys ride in headers only.
 *
 * RATE LIMITS: alpaca_rate_limiter.py (FROZEN) is a per-process Python bucket;
 * Node's Alpaca data calls (fetchDailyBarsRange, optionsEntitlements) have no
 * shared bucket, so this module paces itself: at most BARS_COVERAGE_CONCURRENCY
 * requests in flight, starts spaced >= BARS_COVERAGE_MIN_INTERVAL_MS apart,
 * ~63 requests per run (well under the 200/min quota the bot shares), a hard
 * BARS_COVERAGE_BUDGET_MS wall-clock budget (calls not started in time are
 * reported "skipped"/"budget_exhausted"), and a 6-hour result cache with
 * in-flight de-duplication so repeated diag polls cost nothing.
 *
 * Pure module apart from the injected fetch (node:test safe).
 */
import { verdictFor, type Verdict, type FetchLike } from "./optionsEntitlements";

export type Feed = "sip" | "iex";

export interface DelistedCase {
  symbol: string;
  /** Approximate last regular-session trade on the primary listing (± a few trading days). */
  event_date: string;
  event: "acquired" | "bankrupt" | "renamed" | "went_private" | "seized";
  /** A date while the OLD company traded under the symbol — the `asof` mapping date. */
  asof: string;
  /** Set when the symbol was later reissued to a different company. */
  reused_from?: string;
}

/** Fixed, pre-registered list (dates from public deal/delisting records). */
export const DELISTED_CASES: readonly DelistedCase[] = [
  { symbol: "MHS", event_date: "2012-04-02", event: "acquired", asof: "2012-03-01" },        // Medco -> Express Scripts
  { symbol: "HNZ", event_date: "2013-06-06", event: "acquired", asof: "2013-05-01" },        // Heinz -> Berkshire/3G
  { symbol: "DELL", event_date: "2013-10-29", event: "went_private", asof: "2013-09-03", reused_from: "2018-12-28" }, // DELL reissued to Dell Technologies
  { symbol: "RSH", event_date: "2015-02-02", event: "bankrupt", asof: "2015-01-02" },        // RadioShack, NYSE delisting
  { symbol: "EMC", event_date: "2016-09-06", event: "acquired", asof: "2016-08-01" },        // EMC -> Dell
  { symbol: "LNKD", event_date: "2016-12-07", event: "acquired", asof: "2016-11-01" },       // LinkedIn -> Microsoft
  { symbol: "YHOO", event_date: "2017-06-16", event: "renamed", asof: "2017-05-01" },        // Yahoo -> Altaba (AABA)
  { symbol: "MON", event_date: "2018-06-06", event: "acquired", asof: "2018-05-01" },        // Monsanto -> Bayer
  { symbol: "CELG", event_date: "2019-11-20", event: "acquired", asof: "2019-10-01" },       // Celgene -> BMS
  { symbol: "S", event_date: "2020-03-31", event: "acquired", asof: "2020-03-02", reused_from: "2021-06-30" }, // Sprint -> T-Mobile; S reissued to SentinelOne
  { symbol: "XLNX", event_date: "2022-02-11", event: "acquired", asof: "2022-01-03" },       // Xilinx -> AMD
  { symbol: "CTXS", event_date: "2022-09-29", event: "acquired", asof: "2022-08-01" },       // Citrix -> Vista/Elliott
  { symbol: "TWTR", event_date: "2022-10-27", event: "went_private", asof: "2022-09-01" },   // Twitter
  { symbol: "SIVB", event_date: "2023-03-09", event: "seized", asof: "2023-02-01" },         // SVB, halted 03-10
  { symbol: "FRC", event_date: "2023-04-28", event: "seized", asof: "2023-04-03" },          // First Republic
  { symbol: "BBBY", event_date: "2023-05-02", event: "bankrupt", asof: "2023-03-01" },       // Bed Bath & Beyond, Nasdaq delisting
  { symbol: "ATVI", event_date: "2023-10-12", event: "acquired", asof: "2023-09-01" },       // Activision -> Microsoft
];
export const INDEX_SYMBOLS = ["SPY", "AAPL", "IWM"] as const;
export const CONTROL_SYMBOL = "SBUX";
export const COVERAGE_START = "2010-01-01";
/** asof value that disables Alpaca's symbol mapping (raw ticker as traded). */
export const ASOF_RAW = "-";
/** A delisted series "reaches its event" when its last bar is within this many days of event_date. */
export const EVENT_TOLERANCE_DAYS = 10;

export const BARS_COVERAGE_CACHE_MS = 6 * 60 * 60 * 1000;
export const BARS_COVERAGE_BUDGET_MS = 18_000;
export const BARS_COVERAGE_CONCURRENCY = 4;
export const BARS_COVERAGE_MIN_INTERVAL_MS = 200;
const CALL_TIMEOUT_MS = 8_000;
const MIN_TIME_TO_START_MS = 1_000;
const SYMBOL_RE = /^[A-Z][A-Z.]{0,9}$/;
const YMD_RE = /^\d{4}-\d{2}-\d{2}$/;

export interface SeriesResult {
  id: string;
  symbol: string;
  feed: Feed;
  /** null = API default mapping (today). */
  asof: string | null;
  http: number | null;
  verdict: Verdict;
  /** Coarse class of the vendor's error message (never the message itself). */
  vendor_msg: string | null;
  bars: number | null;
  first_date: string | null;
  last_date: string | null;
  first_close: number | null;
  last_close: number | null;
  /** A next_page_token came back: `bars` is a lower bound. */
  truncated: boolean | null;
  note?: string;
}

export interface DelistedRow {
  symbol: string;
  event: DelistedCase["event"];
  event_date: string;
  asof: string;
  reused_from: string | null;
  in_inactive_assets: boolean | null;
  in_active_assets: boolean | null;
  /** Best series for the old company: sip@asof, else iex@asof. */
  best_feed: Feed | null;
  bars: number | null;
  first_date: string | null;
  last_date: string | null;
  last_minus_event_days: number | null;
  reaches_event: boolean;
  /** sip with default (today) mapping — differs from the asof read when the symbol was reused. */
  default_asof_bars: number | null;
  default_asof_first_date: string | null;
  default_asof_last_date: string | null;
  /** sip with asof=- (no mapping: the ticker string as traded, old and new company concatenated). */
  raw_symbol_bars: number | null;
  raw_symbol_first_date: string | null;
  raw_symbol_last_date: string | null;
}

export interface AssetsSummary {
  http_inactive: number | null;
  http_active: number | null;
  inactive_count: number | null;
  active_count: number | null;
  inactive_examples: string[];
  /** Inactive symbols that are also an active symbol today — ticker reuse indicator. */
  inactive_symbols_also_active: number | null;
  probed_delisted_in_inactive: number | null;
}

export interface BarsCoverageReport {
  probe: "bars_coverage";
  checked_at: string;
  keys: { alpaca: boolean };
  request: { start: string; end: string; timeframe: "1Day"; adjustment: "all" };
  series: SeriesResult[];
  delisted: DelistedRow[];
  assets: AssetsSummary;
  summary: {
    earliest_bar: { sip: string | null; iex: string | null };
    history_reaches_2010: { sip: boolean | null; iex: boolean | null };
    delisted_probed: number;
    delisted_with_bars: number;
    delisted_reaching_event: number;
    /** Same, for the two date-free modes a batched builder could use. */
    delisted_reaching_event_default_asof: number;
    delisted_reaching_event_raw_symbol: number;
    /** Events before the earliest index bar are invisible by construction. */
    delisted_before_history_start: number;
    calls_made: number;
    calls_skipped: number;
    elapsed_ms: number;
  };
}

export interface BarsCoverageConfig {
  env: Record<string, string | undefined>;
  fetch: FetchLike;
  now?: Date;
  tradingBase?: string;
  dataBase?: string;
  /** Test hooks. */
  clock?: () => number;
  minIntervalMs?: number;
  concurrency?: number;
  budgetMs?: number;
  cases?: readonly DelistedCase[];
}

const ymd = (d: Date) => d.toISOString().slice(0, 10);
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const numOrNull = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const dateOf = (t: unknown): string | null => {
  const s = typeof t === "string" ? t.slice(0, 10) : "";
  return YMD_RE.test(s) ? s : null;
};
const daysBetween = (a: string, b: string) => Math.round((Date.parse(`${a}T00:00:00Z`) - Date.parse(`${b}T00:00:00Z`)) / 86_400_000);

/** Map Alpaca's error message to a fixed token — the raw text never leaves. */
export function vendorMsgClass(body: unknown): string | null {
  const m = isObj(body) && typeof body.message === "string" ? body.message : "";
  if (!m) return null;
  if (/recent\s+sip/i.test(m)) return "recent_sip_not_permitted";
  if (/invalid\s+feed/i.test(m)) return "invalid_feed";
  if (/subscription/i.test(m)) return "subscription_not_permitted";
  if (/invalid\s+symbol|unknown\s+symbol|not\s+found/i.test(m)) return "symbol_not_found";
  if (/too\s+many|rate\s+limit/i.test(m)) return "rate_limited";
  if (/forbidden|unauthori[sz]ed/i.test(m)) return "forbidden";
  if (/invalid|parse/i.test(m)) return "invalid_request";
  return "other";
}

/** Reduce a single-symbol bars body to counts, endpoints and first/last close. */
export function summarizeBars(body: unknown): Pick<SeriesResult, "bars" | "first_date" | "last_date" | "first_close" | "last_close" | "truncated"> {
  const arr = isObj(body) && Array.isArray(body.bars) ? body.bars.filter(isObj) : [];
  const truncated = isObj(body) && typeof body.next_page_token === "string" && body.next_page_token.length > 0;
  if (!arr.length) return { bars: 0, first_date: null, last_date: null, first_close: null, last_close: null, truncated };
  const first = arr[0], last = arr[arr.length - 1];
  return {
    bars: arr.length,
    first_date: dateOf(first.t),
    last_date: dateOf(last.t),
    first_close: numOrNull(first.c),
    last_close: numOrNull(last.c),
    truncated,
  };
}

type Job<T> = (remainingMs: number) => Promise<T>;

/** Run jobs with a concurrency cap, spaced starts and a wall-clock budget. */
export async function runPaced<T>(
  jobs: Job<T>[], opts: { concurrency: number; minIntervalMs: number; budgetMs: number; clock: () => number },
): Promise<T[]> {
  const out: T[] = new Array(jobs.length);
  const t0 = opts.clock();
  let next = 0;
  let nextStartAt = t0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= jobs.length) return;
      const wait = nextStartAt - opts.clock();
      nextStartAt = Math.max(nextStartAt, opts.clock()) + opts.minIntervalMs;
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      out[i] = await jobs[i](opts.budgetMs - (opts.clock() - t0));
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(opts.concurrency, jobs.length)) }, worker));
  return out;
}

interface CallOutcome { http: number | null; body: unknown; note?: string }

async function getJson(fetchFn: FetchLike, url: string, headers: Record<string, string>, timeoutMs: number): Promise<CallOutcome> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), Math.max(1, timeoutMs));
  try {
    const r = await fetchFn(url, { method: "GET", headers, signal: ctl.signal });
    try {
      return { http: r.status, body: await r.json() };
    } catch (e) {
      // Non-JSON body: the status is the answer.
      return { http: r.status, body: null, note: "non_json_body" };
    }
  } catch (e) {
    return { http: 0, body: null, note: ctl.signal.aborted ? "timeout" : "network_error" };
  } finally {
    clearTimeout(timer);
  }
}

interface SeriesSpec { symbol: string; feed: Feed; asof: string | null }

function skippedSeries(s: SeriesSpec, note: string): SeriesResult {
  return {
    id: seriesId(s), symbol: s.symbol, feed: s.feed, asof: s.asof, http: null, verdict: "skipped",
    vendor_msg: null, bars: null, first_date: null, last_date: null, first_close: null, last_close: null,
    truncated: null, note,
  };
}

export const seriesId = (s: SeriesSpec) => `${s.symbol}|${s.feed}|${s.asof ?? "default"}`;

function seriesJob(cfg: BarsCoverageConfig, headers: Record<string, string>, end: string, s: SeriesSpec): Job<SeriesResult> {
  return async (remainingMs) => {
    if (remainingMs < MIN_TIME_TO_START_MS) return skippedSeries(s, "budget_exhausted");
    const data = cfg.dataBase || "https://data.alpaca.markets";
    const q = new URLSearchParams({
      timeframe: "1Day", adjustment: "all", start: COVERAGE_START, end, limit: "10000", feed: s.feed,
    });
    if (s.asof) q.set("asof", s.asof);
    const url = `${data}/v2/stocks/${encodeURIComponent(s.symbol)}/bars?${q.toString()}`;
    const r = await getJson(cfg.fetch, url, headers, Math.min(CALL_TIMEOUT_MS, remainingMs));
    const ok = r.http === 200;
    const sum = ok ? summarizeBars(r.body) : { bars: null, first_date: null, last_date: null, first_close: null, last_close: null, truncated: null };
    const res: SeriesResult = {
      id: seriesId(s), symbol: s.symbol, feed: s.feed, asof: s.asof, http: r.http,
      verdict: verdictFor(r.http, null), vendor_msg: ok ? null : vendorMsgClass(r.body), ...sum,
    };
    if (r.note) res.note = r.note;
    return res;
  };
}

interface AssetsOutcome { http: number | null; symbols: string[] | null; note?: string }

function assetsJob(cfg: BarsCoverageConfig, headers: Record<string, string>, status: "active" | "inactive"): Job<AssetsOutcome> {
  return async (remainingMs) => {
    if (remainingMs < MIN_TIME_TO_START_MS) return { http: null, symbols: null, note: "budget_exhausted" };
    const trading = cfg.tradingBase || "https://paper-api.alpaca.markets";
    const r = await getJson(cfg.fetch, `${trading}/v2/assets?status=${status}&asset_class=us_equity`, headers,
      Math.min(CALL_TIMEOUT_MS, remainingMs));
    if (r.http !== 200 || !Array.isArray(r.body)) return { http: r.http, symbols: null, note: r.note };
    const symbols = r.body.filter(isObj).map((a) => String(a.symbol ?? "")).filter((x) => SYMBOL_RE.test(x));
    return { http: r.http, symbols };
  };
}

/** Run the probe once. */
export async function runBarsCoverage(cfg: BarsCoverageConfig): Promise<BarsCoverageReport> {
  const clock = cfg.clock ?? Date.now;
  const t0 = clock();
  const now = cfg.now ?? new Date();
  const end = ymd(new Date(now.getTime() - 86_400_000));
  const cases = cfg.cases ?? DELISTED_CASES;
  const key = cfg.env.ALPACA_KEY || "", secret = cfg.env.ALPACA_SECRET || "";
  const pace = {
    concurrency: cfg.concurrency ?? BARS_COVERAGE_CONCURRENCY,
    minIntervalMs: cfg.minIntervalMs ?? BARS_COVERAGE_MIN_INTERVAL_MS,
    clock,
  };
  const budget = cfg.budgetMs ?? BARS_COVERAGE_BUDGET_MS;

  const phase1: SeriesSpec[] = [
    ...INDEX_SYMBOLS.flatMap((symbol) => (["sip", "iex"] as Feed[]).map((feed) => ({ symbol, feed, asof: null }))),
    ...cases.flatMap((c) => [c.asof, null, ASOF_RAW].map((asof) => ({ symbol: c.symbol, feed: "sip" as Feed, asof }))),
    { symbol: CONTROL_SYMBOL, feed: "sip", asof: null }, { symbol: CONTROL_SYMBOL, feed: "iex", asof: null },
  ];

  let series: SeriesResult[];
  let inactive: AssetsOutcome, active: AssetsOutcome;
  let calls = 0;
  if (!key || !secret) {
    series = phase1.map((s) => skippedSeries(s, "no_key"));
    inactive = { http: null, symbols: null, note: "no_key" };
    active = { http: null, symbols: null, note: "no_key" };
  } else {
    const headers = { "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret };
    const assetJobs: Job<AssetsOutcome>[] = [assetsJob(cfg, headers, "inactive"), assetsJob(cfg, headers, "active")];
    const seriesJobs = phase1.map((s) => seriesJob(cfg, headers, end, s));
    // One paced pool for both kinds so the request-rate bound holds across them.
    const all = await runPaced<AssetsOutcome | SeriesResult>(
      [...assetJobs, ...seriesJobs], { ...pace, budgetMs: budget },
    );
    [inactive, active] = all.slice(0, 2) as AssetsOutcome[];
    series = all.slice(2) as SeriesResult[];
    // Phase 2: iex at the dated mapping, only where sip was refused.
    const byId = new Map(series.map((s) => [s.id, s]));
    const phase2: SeriesSpec[] = cases
      .filter((c) => { const r = byId.get(seriesId({ symbol: c.symbol, feed: "sip", asof: c.asof })); return !!r && r.http !== 200 && r.verdict !== "skipped"; })
      .map((c) => ({ symbol: c.symbol, feed: "iex" as Feed, asof: c.asof }));
    if (phase2.length) {
      const left = budget - (clock() - t0);
      series.push(...await runPaced(phase2.map((s) => seriesJob(cfg, headers, end, s)), { ...pace, budgetMs: left }));
    }
    calls = series.filter((s) => s.verdict !== "skipped").length
      + [inactive, active].filter((a) => a.note !== "budget_exhausted").length;
  }

  const byId = new Map(series.map((s) => [s.id, s]));
  const inactiveSet = inactive.symbols ? new Set(inactive.symbols) : null;
  const activeSet = active.symbols ? new Set(active.symbols) : null;

  const earliest = (feed: Feed): string | null => {
    const ds = INDEX_SYMBOLS.map((sym) => byId.get(seriesId({ symbol: sym, feed, asof: null }))?.first_date)
      .filter((d): d is string => !!d).sort();
    return ds[0] ?? null;
  };
  const earliestBar = { sip: earliest("sip"), iex: earliest("iex") };
  const historyStart = [earliestBar.sip, earliestBar.iex].filter((d): d is string => !!d).sort()[0] ?? null;

  const delisted: DelistedRow[] = cases.map((c) => {
    const sipAsof = byId.get(seriesId({ symbol: c.symbol, feed: "sip", asof: c.asof }));
    const iexAsof = byId.get(seriesId({ symbol: c.symbol, feed: "iex", asof: c.asof }));
    const def = byId.get(seriesId({ symbol: c.symbol, feed: "sip", asof: null }));
    const raw = byId.get(seriesId({ symbol: c.symbol, feed: "sip", asof: ASOF_RAW }));
    const best = sipAsof && sipAsof.http === 200 ? sipAsof : iexAsof && iexAsof.http === 200 ? iexAsof : null;
    const lastMinus = best?.last_date ? daysBetween(best.last_date, c.event_date) : null;
    return {
      symbol: c.symbol, event: c.event, event_date: c.event_date, asof: c.asof, reused_from: c.reused_from ?? null,
      in_inactive_assets: inactiveSet ? inactiveSet.has(c.symbol) : null,
      in_active_assets: activeSet ? activeSet.has(c.symbol) : null,
      best_feed: best ? best.feed : null,
      bars: best ? best.bars : null,
      first_date: best ? best.first_date : null,
      last_date: best ? best.last_date : null,
      last_minus_event_days: lastMinus,
      reaches_event: lastMinus !== null && Math.abs(lastMinus) <= EVENT_TOLERANCE_DAYS && (best?.bars ?? 0) > 0,
      default_asof_bars: def && def.http === 200 ? def.bars : null,
      default_asof_first_date: def ? def.first_date : null,
      default_asof_last_date: def ? def.last_date : null,
      raw_symbol_bars: raw && raw.http === 200 ? raw.bars : null,
      raw_symbol_first_date: raw ? raw.first_date : null,
      raw_symbol_last_date: raw ? raw.last_date : null,
    };
  });
  /** Does a date-free series span the old company's event? It must start on or
   *  before the event and run to within tolerance of it (a reused symbol's
   *  series runs past it; a gap-free check of the event window is the builder's job). */
  const coversEvent = (c: DelistedCase, s: SeriesResult | undefined) =>
    !!s && s.http === 200 && !!s.bars && !!s.first_date && !!s.last_date
    && s.first_date <= c.event_date && daysBetween(s.last_date, c.event_date) >= -EVENT_TOLERANCE_DAYS;

  let alsoActive: number | null = null;
  if (inactiveSet && activeSet) {
    alsoActive = 0;
    for (const s of inactiveSet) if (activeSet.has(s)) alsoActive++;
  }

  return {
    probe: "bars_coverage",
    checked_at: now.toISOString(),
    keys: { alpaca: !!(key && secret) },
    request: { start: COVERAGE_START, end, timeframe: "1Day", adjustment: "all" },
    series,
    delisted,
    assets: {
      http_inactive: inactive.http,
      http_active: active.http,
      inactive_count: inactive.symbols ? inactive.symbols.length : null,
      active_count: active.symbols ? active.symbols.length : null,
      inactive_examples: inactive.symbols ? [...inactive.symbols].sort().slice(0, 8) : [],
      inactive_symbols_also_active: alsoActive,
      probed_delisted_in_inactive: inactiveSet ? cases.filter((c) => inactiveSet.has(c.symbol)).length : null,
    },
    summary: {
      earliest_bar: earliestBar,
      history_reaches_2010: {
        sip: earliestBar.sip ? earliestBar.sip < "2010-02-01" : null,
        iex: earliestBar.iex ? earliestBar.iex < "2010-02-01" : null,
      },
      delisted_probed: cases.length,
      delisted_with_bars: delisted.filter((d) => (d.bars ?? 0) > 0).length,
      delisted_reaching_event: delisted.filter((d) => d.reaches_event).length,
      delisted_reaching_event_default_asof: cases.filter((c) => coversEvent(c, byId.get(seriesId({ symbol: c.symbol, feed: "sip", asof: null })))).length,
      delisted_reaching_event_raw_symbol: cases.filter((c) => coversEvent(c, byId.get(seriesId({ symbol: c.symbol, feed: "sip", asof: ASOF_RAW })))).length,
      delisted_before_history_start: historyStart ? cases.filter((c) => c.event_date < historyStart).length : 0,
      calls_made: calls,
      calls_skipped: series.filter((s) => s.verdict === "skipped").length,
      elapsed_ms: Math.max(0, clock() - t0),
    },
  };
}

let _cache: { at: number; report: BarsCoverageReport } | null = null;
let _inflight: Promise<BarsCoverageReport> | null = null;

/** Cached wrapper for the diag route: the answer changes only when Alpaca's
 *  coverage or the subscription changes. `fresh` bypasses the cache; a run
 *  already in flight is shared, never doubled. */
export async function cachedBarsCoverage(
  cfg: BarsCoverageConfig, opts: { fresh?: boolean; nowMs?: number } = {},
): Promise<BarsCoverageReport & { cached: boolean }> {
  const nowMs = opts.nowMs ?? Date.now();
  if (!opts.fresh && _cache && nowMs - _cache.at < BARS_COVERAGE_CACHE_MS) return { ..._cache.report, cached: true };
  if (!_inflight) {
    _inflight = runBarsCoverage(cfg).finally(() => { _inflight = null; });
  }
  const report = await _inflight;
  _cache = { at: nowMs, report };
  return { ...report, cached: false };
}

export function _resetBarsCoverageCacheForTests(): void {
  _cache = null;
  _inflight = null;
}
