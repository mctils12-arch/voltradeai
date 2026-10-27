// optionsEntitlements — READ-ONLY options-data entitlement probe (Phase 6
// part A, 2026-10-27). Run: npx tsx --test server/optionsEntitlements.test.ts
// Every test uses a mock fetch: no network, no real keys.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  runOptionsEntitlements, cachedOptionsEntitlements, _resetOptionsEntitlementsCacheForTests,
  verdictFor, firstRecord, contractSymbolOf, type FetchLike,
} from "./optionsEntitlements";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// Dummy test-only credentials — chosen to be greppable in the output.
const ENV = {
  ALPACA_KEY: "TESTALPACAKEYID0001",
  ALPACA_SECRET: "TESTALPACASECRET0002",
  POLYGON_API_KEY: "TESTPOLYGONKEY0003",
};
const NOW = new Date("2026-10-27T18:00:00Z");
const OCC = "SPY261218P00500000";

interface Seen { url: string; method: string; headers: Record<string, string> }

/** Mock fetch routed by URL substring → [status, body]. */
function mockFetch(routes: Array<[RegExp, number, unknown]>, seen: Seen[] = []): FetchLike {
  return async (url, init) => {
    seen.push({ url, method: init.method, headers: init.headers });
    for (const [re, status, body] of routes) {
      if (re.test(url)) return { status, json: async () => body };
    }
    return { status: 404, json: async () => ({ message: "no route" }) };
  };
}

// Realistic-shaped bodies carrying real-looking DATA the probe must never echo.
const ALPACA_CONTRACTS = { option_contracts: [{ symbol: OCC, open_interest: "12345", open_interest_date: "2026-10-26", close_price: "4.56", close_price_date: "2026-10-26" }] };
const ALPACA_SNAP_IND = { snapshots: { [OCC]: { latestQuote: { bp: 4.5, ap: 4.6 }, latestTrade: { p: 4.55 }, greeks: { delta: -0.2 }, impliedVolatility: 0.21 } } };
const ALPACA_TRADES = { trades: { [OCC]: [{ t: "2026-10-01T15:00:00Z", p: 7.77, s: 3, x: "C", c: ["I"] }] } };
const ALPACA_BARS = { bars: { [OCC]: [{ t: "2026-10-01T04:00:00Z", o: 7.1, h: 8.2, l: 6.9, c: 7.7, v: 99, n: 12, vw: 7.5 }] } };
const POLY_CONTRACTS = { status: "OK", results: [{ ticker: `O:${OCC}`, expiration_date: "2026-12-18", strike_price: 500, contract_type: "put" }] };
const POLY_SNAPSHOT = { status: "OK", results: [{ open_interest: 6789, greeks: { delta: -0.2 }, implied_volatility: 0.2, day: { close: 4.4 }, underlying_asset: { price: 580.12 } }] };
const POLY_FORBIDDEN = { status: "NOT_AUTHORIZED", message: "You are not entitled to this data. Please upgrade your plan at https://polygon.io/pricing" };
const POLY_AGGS = { status: "DELAYED", results: [{ o: 1, h: 2, l: 0.5, c: 1.5, v: 10, t: 1 }] };

const FULL_ROUTES: Array<[RegExp, number, unknown]> = [
  [/paper-api\.alpaca\.markets\/v2\/options\/contracts/, 200, ALPACA_CONTRACTS],
  [/options\/snapshots\/SPY\?feed=opra/, 403, { message: "subscription does not permit querying OPRA data" }],
  [/options\/snapshots\/SPY\?feed=indicative/, 200, ALPACA_SNAP_IND],
  [/v1beta1\/options\/trades/, 200, ALPACA_TRADES],
  [/v1beta1\/options\/bars/, 200, ALPACA_BARS],
  [/api\.polygon\.io\/v3\/reference\/options\/contracts/, 200, POLY_CONTRACTS],
  [/api\.polygon\.io\/v3\/snapshot\/options\/SPY/, 200, POLY_SNAPSHOT],
  [/api\.polygon\.io\/v3\/trades\//, 403, POLY_FORBIDDEN],
  [/api\.polygon\.io\/v3\/quotes\//, 403, POLY_FORBIDDEN],
  [/api\.polygon\.io\/v2\/aggs\/ticker\//, 200, POLY_AGGS],
];

test("verdictFor: status codes map to verdicts; 429 and network errors are inconclusive, never 'not entitled'", () => {
  assert.equal(verdictFor(200, null), "entitled");
  assert.equal(verdictFor(200, "DELAYED"), "entitled_delayed");
  assert.equal(verdictFor(401, null), "unauthorized_key");
  assert.equal(verdictFor(403, "NOT_AUTHORIZED"), "not_entitled");
  assert.equal(verdictFor(404, null), "not_found");
  assert.equal(verdictFor(422, null), "bad_request");
  assert.equal(verdictFor(429, null), "rate_limited_inconclusive");
  assert.equal(verdictFor(0, null), "network_error_inconclusive");
  assert.equal(verdictFor(null, null), "network_error_inconclusive");
  assert.equal(verdictFor(502, null), "http_other");
});

test("firstRecord / contractSymbolOf: every probed response shape; only OCC-valid symbols survive", () => {
  assert.equal(firstRecord(ALPACA_CONTRACTS)?.open_interest, "12345");
  assert.equal(firstRecord(ALPACA_SNAP_IND)?.impliedVolatility, 0.21);
  assert.equal(firstRecord(ALPACA_TRADES)?.p, 7.77);
  assert.equal(firstRecord(ALPACA_BARS)?.vw, 7.5);
  assert.equal(firstRecord(POLY_SNAPSHOT)?.open_interest, 6789);
  assert.equal(firstRecord({ trades: {} }), null);
  assert.equal(firstRecord({ results: [] }), null);
  assert.equal(firstRecord("nope"), null);
  assert.equal(contractSymbolOf(ALPACA_CONTRACTS), OCC);
  assert.equal(contractSymbolOf(POLY_CONTRACTS), OCC, "Polygon's O: prefix is stripped");
  assert.equal(contractSymbolOf({ option_contracts: [{ symbol: "../../v2/account" }] }), null, "a non-OCC string is never used to build a URL");
});

test("full run: verdicts, field presence and the summary answer the human's question", async () => {
  const seen: Seen[] = [];
  const r = await runOptionsEntitlements({ env: ENV, fetch: mockFetch(FULL_ROUTES, seen), now: NOW });
  const by = Object.fromEntries(r.results.map((x) => [x.id, x]));
  assert.equal(by.alpaca_snapshot_opra.verdict, "not_entitled");
  assert.equal(by.alpaca_snapshot_indicative.verdict, "entitled");
  assert.equal(by.alpaca_snapshot_indicative.fields.openInterest, false, "indicative snapshot carries no OI (prod finding, audit §2.4)");
  assert.equal(by.alpaca_snapshot_indicative.fields.impliedVolatility, true);
  assert.equal(by.alpaca_contracts.fields.open_interest, true);
  assert.equal(by.alpaca_trades_historical.verdict, "entitled");
  assert.equal(by.alpaca_trades_historical.rows, true);
  assert.equal(by.alpaca_quotes_historical.verdict, "not_offered");
  assert.equal(by.polygon_trades.verdict, "not_entitled");
  assert.equal(by.polygon_trades.vendor_status, "NOT_AUTHORIZED");
  assert.equal(by.polygon_aggs_daily.verdict, "entitled_delayed");
  assert.deepEqual(r.summary.open_interest, {
    alpaca_contracts: true, alpaca_snapshot_indicative: false, alpaca_snapshot_opra: null, polygon_snapshot: true,
  });
  assert.equal(r.summary.opra_realtime, "not_entitled");
  assert.deepEqual(r.summary.historical_quotes, { alpaca: "not_offered", polygon: "not_entitled" });
  assert.deepEqual(r.keys, { alpaca: true, polygon_env: "POLYGON_API_KEY", polygon_keys_match: null });
  // 5 Alpaca + 5 Polygon calls; Polygon stays within Basic's 5 calls/min.
  assert.equal(seen.filter((s) => s.url.includes("polygon.io")).length, 5);
  assert.equal(seen.length, 10);
});

test("NEVER DATA, NEVER KEYS: no price, size, symbol, URL, vendor message or key reaches the report", async () => {
  const r = await runOptionsEntitlements({ env: ENV, fetch: mockFetch(FULL_ROUTES), now: NOW });
  const flat = JSON.stringify(r);
  for (const leak of [
    ENV.ALPACA_KEY, ENV.ALPACA_SECRET, ENV.POLYGON_API_KEY, OCC, "12345", "6789", "7.77", "4.56", "580.12",
    "upgrade your plan", "OPRA data", "https://", "apiKey", "Bearer",
  ]) assert.ok(!flat.includes(leak), `report leaked ${leak}`);
  // The report survives the diag sanitizer unchanged (nothing in it looks secret-ish).
  assert.deepEqual(sanitizeDiag(r), r);
});

test("keys travel only in headers: no URL ever carries a key; every call is a GET; no order path", async () => {
  const seen: Seen[] = [];
  await runOptionsEntitlements({ env: { ...ENV, POLYGON_KEY: ENV.POLYGON_API_KEY }, fetch: mockFetch(FULL_ROUTES, seen), now: NOW });
  for (const s of seen) {
    assert.equal(s.method, "GET");
    for (const k of [ENV.ALPACA_KEY, ENV.ALPACA_SECRET, ENV.POLYGON_API_KEY]) assert.ok(!s.url.includes(k), "key in URL");
    assert.ok(!/apiKey=/i.test(s.url));
    assert.ok(!s.url.includes("/orders"), "the probe must never touch an order endpoint");
    if (s.url.includes("polygon.io")) assert.equal(s.headers.Authorization, `Bearer ${ENV.POLYGON_API_KEY}`);
    else assert.equal(s.headers["APCA-API-KEY-ID"], ENV.ALPACA_KEY);
  }
  const src = fs.readFileSync(path.join(here, "optionsEntitlements.ts"), "utf8");
  assert.ok(!/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src), "module must be GET-only");
  assert.ok(!src.includes("/orders"), "module must not reference any order path");
});

test("no keys: every network probe is skipped with note no_key and zero calls are made", async () => {
  const seen: Seen[] = [];
  const r = await runOptionsEntitlements({ env: {}, fetch: mockFetch(FULL_ROUTES, seen), now: NOW });
  assert.equal(seen.length, 0);
  for (const x of r.results) assert.ok(x.verdict === "skipped" || x.verdict === "not_offered", `${x.id} ran without a key`);
  assert.deepEqual(r.keys, { alpaca: false, polygon_env: "none", polygon_keys_match: null });
});

test("env-name split: POLYGON_KEY alone is used; both names set reports whether they match (boolean only)", async () => {
  const seen: Seen[] = [];
  const r1 = await runOptionsEntitlements({ env: { POLYGON_KEY: "TESTONLYPYKEY" }, fetch: mockFetch(FULL_ROUTES, seen), now: NOW });
  assert.equal(r1.keys.polygon_env, "POLYGON_KEY");
  assert.ok(seen.some((s) => s.headers.Authorization === "Bearer TESTONLYPYKEY"));
  const r2 = await runOptionsEntitlements({ env: { POLYGON_KEY: "A1", POLYGON_API_KEY: "B2" }, fetch: mockFetch(FULL_ROUTES), now: NOW });
  assert.deepEqual([r2.keys.polygon_env, r2.keys.polygon_keys_match], ["both", false]);
});

test("Polygon contracts failing: per-contract Polygon probes are addressed with Alpaca's contract symbol", async () => {
  const seen: Seen[] = [];
  const routes = FULL_ROUTES.map(([re, st, b]) => (String(re).includes("reference") ? [re, 403, POLY_FORBIDDEN] : [re, st, b])) as Array<[RegExp, number, unknown]>;
  const r = await runOptionsEntitlements({ env: ENV, fetch: mockFetch(routes, seen), now: NOW });
  assert.ok(seen.some((s) => s.url.includes(`/v3/trades/${encodeURIComponent(`O:${OCC}`)}`)));
  assert.equal(r.results.find((x) => x.id === "polygon_contracts")?.verdict, "not_entitled");
});

test("no contract symbol anywhere: per-contract probes are skipped, not called with a guess", async () => {
  const seen: Seen[] = [];
  const routes: Array<[RegExp, number, unknown]> = [[/./, 403, POLY_FORBIDDEN]];
  const r = await runOptionsEntitlements({ env: ENV, fetch: mockFetch(routes, seen), now: NOW });
  for (const id of ["alpaca_trades_historical", "alpaca_bars_historical", "polygon_trades", "polygon_quotes", "polygon_aggs_daily"]) {
    const x = r.results.find((y) => y.id === id)!;
    assert.equal(x.verdict, "skipped");
    assert.equal(x.note, "no_contract_symbol");
  }
  assert.equal(seen.length, 5, "3 Alpaca + 2 Polygon symbol-free calls only");
});

test("transport failure, timeout, 429 and non-JSON bodies are reported as inconclusive / status-only, never thrown", async () => {
  const throwing: FetchLike = async () => { throw new Error("ECONNRESET TESTPOLYGONKEY0003"); };
  const r = await runOptionsEntitlements({ env: ENV, fetch: throwing, now: NOW });
  const net = r.results.filter((x) => x.http === 0);
  assert.ok(net.length >= 5);
  for (const x of net) assert.equal(x.verdict, "network_error_inconclusive");
  assert.ok(!JSON.stringify(r).includes("ECONNRESET"), "error text (which can echo secrets) never reaches the report");

  const hanging: FetchLike = (_u, init) => new Promise((_res, rej) => init.signal?.addEventListener("abort", () => rej(new Error("aborted"))));
  const r2 = await runOptionsEntitlements({ env: { ALPACA_KEY: "a", ALPACA_SECRET: "b" }, fetch: hanging, now: NOW, timeoutMs: 20 });
  assert.equal(r2.results.find((x) => x.id === "alpaca_snapshot_opra")?.note, "timeout");

  const notJson: FetchLike = async () => ({ status: 429, json: async () => { throw new SyntaxError("Unexpected token <"); } });
  const r3 = await runOptionsEntitlements({ env: ENV, fetch: notJson, now: NOW });
  const x = r3.results.find((y) => y.id === "polygon_snapshot_chain")!;
  assert.equal(x.verdict, "rate_limited_inconclusive");
  assert.equal(x.note, "non_json_body");
});

test("underlying param is sanitized before it reaches a URL", async () => {
  const seen: Seen[] = [];
  const r = await runOptionsEntitlements({ env: ENV, fetch: mockFetch(FULL_ROUTES, seen), now: NOW, underlying: "iwm/../../v2/account?x=" });
  assert.equal(r.underlying, "IWMVAC");
  for (const s of seen) assert.ok(!s.url.includes("../"));
});

test("cache: second call within 10 min makes no network calls; fresh bypasses", async () => {
  _resetOptionsEntitlementsCacheForTests();
  const seen: Seen[] = [];
  const cfg = { env: ENV, fetch: mockFetch(FULL_ROUTES, seen), now: NOW };
  const a = await cachedOptionsEntitlements(cfg, { nowMs: 1_000_000 });
  assert.equal(a.cached, false);
  const n = seen.length;
  const b = await cachedOptionsEntitlements(cfg, { nowMs: 1_000_000 + 60_000 });
  assert.equal(b.cached, true);
  assert.equal(seen.length, n);
  await cachedOptionsEntitlements(cfg, { nowMs: 1_000_000 + 60_000, fresh: true });
  assert.equal(seen.length, 2 * n);
  _resetOptionsEntitlementsCacheForTests();
});

test("wiring: probe is whitelisted in diag.ts and served by bot.ts through the sanitizer", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("options_entitlements"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "options_entitlements": {');
  assert.ok(i > 0, "case missing in bot.ts");
  const block = bot.slice(i, i + 1400);
  assert.ok(block.includes("cachedOptionsEntitlements("));
  assert.ok(block.includes("res.json(sanitizeDiag(report))"));
});
