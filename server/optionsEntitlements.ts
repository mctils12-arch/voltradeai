/**
 * optionsEntitlements.ts — READ-ONLY entitlement check for historical
 * options data (Phase 6 part A, human-directed 2026-10-27).
 *
 * QUESTION IT ANSWERS: what do the production Alpaca and Polygon keys
 * actually entitle us to for options TRADES, QUOTES, BARS and OPEN
 * INTEREST? The repo history (research/wishlist.md 2026-07-25 entry,
 * open_questions.md KNOWN BROKEN #25, alpaca_feed.options_feed) proves only
 * one thing: Alpaca rejected feed=opra with HTTP 403 "subscription does not
 * permit querying OPRA data" on 2026-07-25. Nothing in the repo has ever
 * called a Polygon options endpoint, so the Polygon options tier is
 * unknown, and the audit (2026-10-27, §2.4) found that no diag probe
 * surfaces either fact. This module makes ONE minimal GET per endpoint and
 * reports only the HTTP status, a verdict, and which expected FIELD NAMES
 * exist in the first record. It never returns a price, a size, a symbol,
 * a URL, a response body, or a key.
 *
 * WHAT IS PROBED (vendor docs read 2026-10-27):
 *  Alpaca Trading API  GET /v2/options/contracts            (open_interest + open_interest_date
 *                                                            are schema fields — free OI source?)
 *  Alpaca Market Data  GET /v1beta1/options/snapshots/{u}  feed=opra vs feed=indicative
 *                      GET /v1beta1/options/trades          historical trades (docs: history from 2024-02)
 *                      GET /v1beta1/options/bars            historical daily bars
 *                      (no historical options QUOTES endpoint exists at Alpaca — reported as
 *                       not_offered without a call)
 *  Polygon (Massive)   GET /v3/reference/options/contracts  (Basic tier)
 *                      GET /v3/snapshot/options/{u}         carries open_interest (Starter+ per the
 *                                                            vendor knowledge base; OI is latest-only,
 *                                                            no historical OI anywhere at the vendor)
 *                      GET /v3/trades/{optionTicker}        (Developer+)
 *                      GET /v3/quotes/{optionTicker}        (Advanced+)
 *                      GET /v2/aggs/ticker/{optionTicker}/range/1/day/... (Basic: EOD aggregates)
 *
 * SAFETY POSTURE:
 *  - GET only. No order path is referenced or reachable from this module.
 *  - Keys travel ONLY in request headers (Polygon via `Authorization:
 *    Bearer`, never the `apiKey=` query string), so no URL built here ever
 *    contains a secret. The result never echoes a URL.
 *  - A contract symbol is needed to address the trades/quotes/bars
 *    endpoints; it is read from the contracts response, validated against
 *    the OCC pattern, used for the follow-up calls, and dropped.
 *  - Polygon Basic is 5 calls/min: this module makes at most 5 Polygon
 *    calls per run, sequentially, and callers cache the result
 *    (OPTIONS_ENTITLEMENTS_CACHE_MS). HTTP 429 is reported as inconclusive,
 *    never as "not entitled".
 *
 * Pure module apart from the injected fetch (node:test safe).
 */

export type Verdict =
  | "entitled"
  | "entitled_delayed"
  | "unauthorized_key"
  | "not_entitled"
  | "not_found"
  | "bad_request"
  | "rate_limited_inconclusive"
  | "network_error_inconclusive"
  | "http_other"
  | "skipped"
  | "not_offered";

export interface EndpointResult {
  id: string;
  provider: "alpaca" | "polygon";
  /** Template path only — no ticker, no query values, never a key. */
  endpoint: string;
  http: number | null;
  verdict: Verdict;
  /** Polygon's own body `status` token (OK / DELAYED / NOT_AUTHORIZED …), whitelisted shape. */
  vendor_status: string | null;
  /** Did the 200 body carry at least one record? null when not 200. */
  rows: boolean | null;
  /** Expected field name -> present (non-null) in the first record. */
  fields: Record<string, boolean>;
  note?: string;
}

export interface EntitlementsReport {
  probe: "options_entitlements";
  checked_at: string;
  underlying: string;
  keys: {
    alpaca: boolean;
    /** Which env name(s) carry a Polygon key — the audit found Python reads
     *  POLYGON_KEY while the server reads POLYGON_API_KEY. */
    polygon_env: "POLYGON_API_KEY" | "POLYGON_KEY" | "both" | "none";
    /** When both names are set: do they hold the same value? (boolean only) */
    polygon_keys_match: boolean | null;
  };
  results: EndpointResult[];
  summary: {
    historical_trades: { alpaca: Verdict; polygon: Verdict };
    historical_quotes: { alpaca: Verdict; polygon: Verdict };
    historical_bars: { alpaca: Verdict; polygon: Verdict };
    open_interest: {
      alpaca_contracts: boolean | null;
      alpaca_snapshot_indicative: boolean | null;
      alpaca_snapshot_opra: boolean | null;
      polygon_snapshot: boolean | null;
    };
    opra_realtime: Verdict;
  };
}

export interface FetchResponseLike {
  status: number;
  json(): Promise<unknown>;
}
export type FetchLike = (
  url: string,
  init: { method: "GET"; headers: Record<string, string>; signal?: AbortSignal },
) => Promise<FetchResponseLike>;

export interface EntitlementsConfig {
  env: Record<string, string | undefined>;
  fetch: FetchLike;
  now?: Date;
  /** Trading API host (paper by default — same host bot.ts uses). */
  tradingBase?: string;
  dataBase?: string;
  polygonBase?: string;
  underlying?: string;
  timeoutMs?: number;
}

export const OPTIONS_ENTITLEMENTS_CACHE_MS = 10 * 60 * 1000;
const DEFAULT_TIMEOUT_MS = 8000;
const OCC_RE = /^[A-Z]{1,6}\d{6}[CP]\d{8}$/;
const VENDOR_STATUS_RE = /^[A-Z_]{2,24}$/;

const ymd = (d: Date) => d.toISOString().slice(0, 10);
const addDays = (d: Date, n: number) => new Date(d.getTime() + n * 86_400_000);

export function verdictFor(http: number | null, vendorStatus: string | null): Verdict {
  if (http === null || http === 0) return "network_error_inconclusive";
  if (http === 200) return vendorStatus === "DELAYED" ? "entitled_delayed" : "entitled";
  if (http === 401) return "unauthorized_key";
  if (http === 403) return "not_entitled";
  if (http === 404) return "not_found";
  if (http === 400 || http === 422) return "bad_request";
  if (http === 429) return "rate_limited_inconclusive";
  return "http_other";
}

function isObj(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

/** First record of the shapes the probed endpoints return. */
export function firstRecord(body: unknown): Record<string, unknown> | null {
  if (!isObj(body)) return null;
  // Polygon v3: { results: [...] } (or a single object for some snapshots)
  if (Array.isArray(body.results)) return isObj(body.results[0]) ? body.results[0] : null;
  if (isObj(body.results)) return body.results;
  // Alpaca trading: { option_contracts: [...] }
  if (Array.isArray(body.option_contracts)) return isObj(body.option_contracts[0]) ? body.option_contracts[0] : null;
  // Alpaca market data: { snapshots: {SYM: {...}} } | { trades: {SYM: [...]} } | { bars: {SYM: [...]} }
  for (const k of ["snapshots", "trades", "bars"]) {
    const m = body[k];
    if (isObj(m)) {
      const first = Object.values(m)[0];
      if (Array.isArray(first)) return isObj(first[0]) ? first[0] : null;
      if (isObj(first)) return first;
      return null;
    }
  }
  return null;
}

/** OCC root symbol of the first contract (Alpaca `symbol`, Polygon `ticker` "O:…"). */
export function contractSymbolOf(body: unknown): string | null {
  const rec = firstRecord(body);
  if (!rec) return null;
  const raw = String(rec.symbol ?? rec.ticker ?? "").replace(/^O:/, "");
  return OCC_RE.test(raw) ? raw : null;
}

function presence(rec: Record<string, unknown> | null, fields: string[]): Record<string, boolean> {
  const out: Record<string, boolean> = {};
  for (const f of fields) out[f] = !!rec && rec[f] !== undefined && rec[f] !== null && rec[f] !== "";
  return out;
}

interface CallSpec {
  id: string;
  provider: "alpaca" | "polygon";
  endpoint: string;
  url: string;
  headers: Record<string, string>;
  fields: string[];
}

async function call(cfg: EntitlementsConfig, spec: CallSpec): Promise<{ result: EndpointResult; body: unknown }> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  let http: number | null = null;
  let body: unknown = null;
  let note: string | undefined;
  try {
    const r = await cfg.fetch(spec.url, { method: "GET", headers: spec.headers, signal: ctl.signal });
    http = r.status;
    try {
      body = await r.json();
    } catch (e) {
      // Non-JSON body: the status alone is the answer; fields stay false.
      note = "non_json_body";
    }
  } catch (e) {
    http = 0;
    note = ctl.signal.aborted ? "timeout" : "network_error";
  } finally {
    clearTimeout(timer);
  }
  const vs = isObj(body) && typeof body.status === "string" && VENDOR_STATUS_RE.test(body.status) ? body.status : null;
  const rec = http === 200 ? firstRecord(body) : null;
  const result: EndpointResult = {
    id: spec.id,
    provider: spec.provider,
    endpoint: spec.endpoint,
    http,
    verdict: verdictFor(http, vs),
    vendor_status: vs,
    rows: http === 200 ? rec !== null : null,
    fields: presence(rec, spec.fields),
  };
  if (note) result.note = note;
  return { result, body };
}

function skipped(id: string, provider: "alpaca" | "polygon", endpoint: string, fields: string[], note: string): EndpointResult {
  return { id, provider, endpoint, http: null, verdict: "skipped", vendor_status: null, rows: null, fields: presence(null, fields), note };
}

const F = {
  alpacaContracts: ["symbol", "open_interest", "open_interest_date", "close_price", "close_price_date"],
  alpacaSnapshot: ["latestQuote", "latestTrade", "greeks", "impliedVolatility", "openInterest", "dailyBar"],
  alpacaTrades: ["t", "p", "s", "x", "c"],
  alpacaBars: ["t", "o", "h", "l", "c", "v", "n", "vw"],
  polyContracts: ["ticker", "expiration_date", "strike_price", "contract_type"],
  polySnapshot: ["open_interest", "greeks", "implied_volatility", "last_quote", "last_trade", "day", "underlying_asset"],
  polyTrades: ["price", "size", "sip_timestamp", "exchange", "conditions"],
  polyQuotes: ["bid_price", "ask_price", "bid_size", "ask_size", "sip_timestamp"],
  polyAggs: ["o", "h", "l", "c", "v", "t"],
};

async function runAlpaca(cfg: EntitlementsConfig, u: string, now: Date): Promise<{ results: EndpointResult[]; occ: string | null }> {
  const key = cfg.env.ALPACA_KEY || "", secret = cfg.env.ALPACA_SECRET || "";
  const trading = cfg.tradingBase || "https://paper-api.alpaca.markets";
  const data = cfg.dataBase || "https://data.alpaca.markets";
  const P = "alpaca" as const;
  const contractsEp = "GET /v2/options/contracts?underlying_symbols={u}&limit=1";
  const snapEp = (feed: string) => `GET /v1beta1/options/snapshots/{u}?feed=${feed}&limit=1`;
  const tradesEp = "GET /v1beta1/options/trades?symbols={occ}&start={-30d}&limit=1";
  const barsEp = "GET /v1beta1/options/bars?symbols={occ}&timeframe=1Day&start={-30d}&limit=1";
  const quotesHist: EndpointResult = {
    id: "alpaca_quotes_historical", provider: P, endpoint: "(none — Alpaca serves latest options quotes only)",
    http: null, verdict: "not_offered", vendor_status: null, rows: null, fields: {},
    note: "Alpaca market-data docs list options bars, trades, latest quotes/trades and snapshots; no historical options quotes endpoint",
  };
  if (!key || !secret) {
    return {
      occ: null,
      results: [
        skipped("alpaca_contracts", P, contractsEp, F.alpacaContracts, "no_key"),
        skipped("alpaca_snapshot_opra", P, snapEp("opra"), F.alpacaSnapshot, "no_key"),
        skipped("alpaca_snapshot_indicative", P, snapEp("indicative"), F.alpacaSnapshot, "no_key"),
        skipped("alpaca_trades_historical", P, tradesEp, F.alpacaTrades, "no_key"),
        skipped("alpaca_bars_historical", P, barsEp, F.alpacaBars, "no_key"),
        quotesHist,
      ],
    };
  }
  const headers = { "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret };
  const eu = encodeURIComponent(u);
  const out: EndpointResult[] = [];
  // A monthly-ish expiry 20-60 days out: an established contract that has
  // existed for the 30-day trades/bars window.
  const contracts = await call(cfg, {
    id: "alpaca_contracts", provider: P, endpoint: contractsEp, headers, fields: F.alpacaContracts,
    url: `${trading}/v2/options/contracts?underlying_symbols=${eu}&expiration_date_gte=${ymd(addDays(now, 20))}&expiration_date_lte=${ymd(addDays(now, 60))}&limit=1`,
  });
  out.push(contracts.result);
  const opra = await call(cfg, {
    id: "alpaca_snapshot_opra", provider: P, endpoint: snapEp("opra"), headers, fields: F.alpacaSnapshot,
    url: `${data}/v1beta1/options/snapshots/${eu}?feed=opra&limit=1`,
  });
  out.push(opra.result);
  const ind = await call(cfg, {
    id: "alpaca_snapshot_indicative", provider: P, endpoint: snapEp("indicative"), headers, fields: F.alpacaSnapshot,
    url: `${data}/v1beta1/options/snapshots/${eu}?feed=indicative&limit=1`,
  });
  out.push(ind.result);
  const occ = contractSymbolOf(contracts.body) || contractSymbolOf(ind.body);
  if (!occ) {
    out.push(skipped("alpaca_trades_historical", P, tradesEp, F.alpacaTrades, "no_contract_symbol"));
    out.push(skipped("alpaca_bars_historical", P, barsEp, F.alpacaBars, "no_contract_symbol"));
  } else {
    const start = ymd(addDays(now, -30));
    out.push((await call(cfg, {
      id: "alpaca_trades_historical", provider: P, endpoint: tradesEp, headers, fields: F.alpacaTrades,
      url: `${data}/v1beta1/options/trades?symbols=${encodeURIComponent(occ)}&start=${start}&limit=1`,
    })).result);
    out.push((await call(cfg, {
      id: "alpaca_bars_historical", provider: P, endpoint: barsEp, headers, fields: F.alpacaBars,
      url: `${data}/v1beta1/options/bars?symbols=${encodeURIComponent(occ)}&timeframe=1Day&start=${start}&limit=1`,
    })).result);
  }
  out.push(quotesHist);
  return { results: out, occ };
}

async function runPolygon(cfg: EntitlementsConfig, u: string, now: Date, fallbackOcc: string | null): Promise<EndpointResult[]> {
  const key = cfg.env.POLYGON_API_KEY || cfg.env.POLYGON_KEY || "";
  const base = cfg.polygonBase || "https://api.polygon.io";
  const P = "polygon" as const;
  const eps = {
    contracts: "GET /v3/reference/options/contracts?underlying_ticker={u}&limit=1",
    snapshot: "GET /v3/snapshot/options/{u}?limit=1",
    trades: "GET /v3/trades/{optionTicker}?limit=1",
    quotes: "GET /v3/quotes/{optionTicker}?limit=1",
    aggs: "GET /v2/aggs/ticker/{optionTicker}/range/1/day/{-30d}/{today}?limit=1",
  };
  if (!key) {
    return [
      skipped("polygon_contracts", P, eps.contracts, F.polyContracts, "no_key"),
      skipped("polygon_snapshot_chain", P, eps.snapshot, F.polySnapshot, "no_key"),
      skipped("polygon_trades", P, eps.trades, F.polyTrades, "no_key"),
      skipped("polygon_quotes", P, eps.quotes, F.polyQuotes, "no_key"),
      skipped("polygon_aggs_daily", P, eps.aggs, F.polyAggs, "no_key"),
    ];
  }
  const headers = { Authorization: `Bearer ${key}` };
  const eu = encodeURIComponent(u);
  const out: EndpointResult[] = [];
  const contracts = await call(cfg, {
    id: "polygon_contracts", provider: P, endpoint: eps.contracts, headers, fields: F.polyContracts,
    url: `${base}/v3/reference/options/contracts?underlying_ticker=${eu}&expiration_date.gte=${ymd(addDays(now, 20))}&expiration_date.lte=${ymd(addDays(now, 60))}&limit=1`,
  });
  out.push(contracts.result);
  out.push((await call(cfg, {
    id: "polygon_snapshot_chain", provider: P, endpoint: eps.snapshot, headers, fields: F.polySnapshot,
    url: `${base}/v3/snapshot/options/${eu}?limit=1`,
  })).result);
  const occ = contractSymbolOf(contracts.body) || fallbackOcc;
  if (!occ) {
    out.push(skipped("polygon_trades", P, eps.trades, F.polyTrades, "no_contract_symbol"));
    out.push(skipped("polygon_quotes", P, eps.quotes, F.polyQuotes, "no_contract_symbol"));
    out.push(skipped("polygon_aggs_daily", P, eps.aggs, F.polyAggs, "no_contract_symbol"));
    return out;
  }
  const t = encodeURIComponent(`O:${occ}`);
  out.push((await call(cfg, {
    id: "polygon_trades", provider: P, endpoint: eps.trades, headers, fields: F.polyTrades,
    url: `${base}/v3/trades/${t}?limit=1`,
  })).result);
  out.push((await call(cfg, {
    id: "polygon_quotes", provider: P, endpoint: eps.quotes, headers, fields: F.polyQuotes,
    url: `${base}/v3/quotes/${t}?limit=1`,
  })).result);
  out.push((await call(cfg, {
    id: "polygon_aggs_daily", provider: P, endpoint: eps.aggs, headers, fields: F.polyAggs,
    url: `${base}/v2/aggs/ticker/${t}/range/1/day/${ymd(addDays(now, -30))}/${ymd(now)}?limit=1`,
  })).result);
  return out;
}

function oiOf(results: EndpointResult[], id: string, field: string): boolean | null {
  const r = results.find((x) => x.id === id);
  if (!r || r.http !== 200 || !r.rows) return null;
  return !!r.fields[field];
}

const verdictOf = (results: EndpointResult[], id: string): Verdict =>
  results.find((x) => x.id === id)?.verdict ?? "skipped";

/** Run every probe once. Alpaca runs first so its contract symbol can address
 *  Polygon's per-contract endpoints if Polygon's own contracts call fails. */
export async function runOptionsEntitlements(cfg: EntitlementsConfig): Promise<EntitlementsReport> {
  const now = cfg.now ?? new Date();
  const u = (cfg.underlying || "SPY").toUpperCase().replace(/[^A-Z]/g, "").slice(0, 6) || "SPY";
  const a = cfg.env.POLYGON_API_KEY || "", b = cfg.env.POLYGON_KEY || "";
  const polygonEnv = a && b ? "both" : a ? "POLYGON_API_KEY" : b ? "POLYGON_KEY" : "none";

  // The contract symbol travels only in memory between the two groups; it is
  // never written into an EndpointResult.
  const alpaca = await runAlpaca(cfg, u, now);
  const alpacaResults = alpaca.results;
  const polygonResults = await runPolygon(cfg, u, now, alpaca.occ);
  const results = [...alpacaResults, ...polygonResults];

  return {
    probe: "options_entitlements",
    checked_at: now.toISOString(),
    underlying: u,
    keys: {
      alpaca: !!(cfg.env.ALPACA_KEY && cfg.env.ALPACA_SECRET),
      polygon_env: polygonEnv,
      polygon_keys_match: a && b ? a === b : null,
    },
    results,
    summary: {
      historical_trades: { alpaca: verdictOf(results, "alpaca_trades_historical"), polygon: verdictOf(results, "polygon_trades") },
      historical_quotes: { alpaca: verdictOf(results, "alpaca_quotes_historical"), polygon: verdictOf(results, "polygon_quotes") },
      historical_bars: { alpaca: verdictOf(results, "alpaca_bars_historical"), polygon: verdictOf(results, "polygon_aggs_daily") },
      open_interest: {
        alpaca_contracts: oiOf(results, "alpaca_contracts", "open_interest"),
        alpaca_snapshot_indicative: oiOf(results, "alpaca_snapshot_indicative", "openInterest"),
        alpaca_snapshot_opra: oiOf(results, "alpaca_snapshot_opra", "openInterest"),
        polygon_snapshot: oiOf(results, "polygon_snapshot_chain", "open_interest"),
      },
      opra_realtime: verdictOf(results, "alpaca_snapshot_opra"),
    },
  };
}

let _cache: { at: number; report: EntitlementsReport } | null = null;

/** Cached wrapper for the diag route: Polygon Basic allows 5 calls/min, and
 *  the answer changes only when a subscription changes. `fresh` bypasses. */
export async function cachedOptionsEntitlements(
  cfg: EntitlementsConfig, opts: { fresh?: boolean; nowMs?: number } = {},
): Promise<EntitlementsReport & { cached: boolean }> {
  const nowMs = opts.nowMs ?? Date.now();
  if (!opts.fresh && _cache && nowMs - _cache.at < OPTIONS_ENTITLEMENTS_CACHE_MS) {
    return { ..._cache.report, cached: true };
  }
  const report = await runOptionsEntitlements(cfg);
  _cache = { at: nowMs, report };
  return { ...report, cached: false };
}

export function _resetOptionsEntitlementsCacheForTests(): void {
  _cache = null;
}
