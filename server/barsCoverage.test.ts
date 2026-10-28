// barsCoverage — READ-ONLY probe of Alpaca's daily-bar history depth and
// delisted-ticker coverage (2026-10-28). Run: npx tsx --test server/barsCoverage.test.ts
// Every test uses a mock fetch: no network, no real keys.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  runBarsCoverage, cachedBarsCoverage, _resetBarsCoverageCacheForTests, runPaced,
  summarizeBars, vendorMsgClass, seriesId, DELISTED_CASES, INDEX_SYMBOLS, CONTROL_SYMBOL, COVERAGE_START,
  type DelistedCase,
} from "./barsCoverage";
import type { FetchLike } from "./optionsEntitlements";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const ENV = { ALPACA_KEY: "TESTALPACAKEYID0001", ALPACA_SECRET: "TESTALPACASECRET0002" };
const NOW = new Date("2026-10-28T18:00:00Z");
const FAST = { minIntervalMs: 0 };

interface Seen { url: string; method: string; headers: Record<string, string> }
type Route = (u: URL) => [number, unknown] | null;

function mockFetch(routes: Route[], seen: Seen[] = []): FetchLike {
  return async (url, init) => {
    seen.push({ url, method: init.method, headers: init.headers });
    const u = new URL(url);
    for (const r of routes) {
      const hit = r(u);
      if (hit) return { status: hit[0], json: async () => hit[1] };
    }
    return { status: 404, json: async () => ({ message: "no route" }) };
  };
}

const bar = (t: string, c: number) => ({ t: `${t}T04:00:00Z`, o: c, h: c, l: c, c, v: 1000, n: 10, vw: c });
const bars = (...xs: Array<[string, number]>) => ({ bars: xs.map(([t, c]) => bar(t, c)), symbol: "X", next_page_token: null });

// Two small cases: one covered delisting, one reused symbol, one sip-refused (iex fallback).
const CASES: DelistedCase[] = [
  { symbol: "TWTR", event_date: "2022-10-27", event: "went_private", asof: "2022-09-01" },
  { symbol: "S", event_date: "2020-03-31", event: "acquired", asof: "2020-03-02", reused_from: "2021-06-30" },
  { symbol: "HNZ", event_date: "2013-06-06", event: "acquired", asof: "2013-05-01" },
];

const ROUTES: Route[] = [
  (u) => (u.pathname === "/v2/assets" && u.searchParams.get("status") === "inactive"
    ? [200, [{ symbol: "TWTR", name: "Twitter Inc", status: "inactive" }, { symbol: "S", name: "Sprint" }, { symbol: "HNZ" }, { symbol: "ZZZQ" }, { symbol: "bad sym!" }]] : null),
  (u) => (u.pathname === "/v2/assets" && u.searchParams.get("status") === "active"
    ? [200, [{ symbol: "SPY" }, { symbol: "S" }, { symbol: "AAPL" }]] : null),
  (u) => {
    const m = /^\/v2\/stocks\/([A-Z.]+)\/bars$/.exec(u.pathname);
    if (!m) return null;
    const sym = m[1], feed = u.searchParams.get("feed"), asof = u.searchParams.get("asof");
    if (sym === "SPY") return [200, feed === "sip" ? bars(["2016-01-04", 201.0], ["2026-10-27", 580.12]) : bars(["2020-07-27", 321.0], ["2026-10-27", 580.12])];
    if (sym === "AAPL" || sym === "IWM" || sym === "SBUX") return [200, bars(["2016-01-04", 26.0], ["2026-10-27", 230.0])];
    if (sym === "TWTR") return [200, asof ? bars(["2016-01-04", 22.0], ["2022-10-27", 53.7]) : { bars: null, next_page_token: null }];
    // asof=- concatenates every company that traded under the string (Sprint, then SentinelOne).
    if (sym === "S" && asof === "-") return [200, bars(["2016-01-04", 3.6], ["2020-03-31", 8.62], ["2021-06-30", 42.5], ["2026-10-27", 17.0])];
    if (sym === "S") return [200, asof ? bars(["2016-01-04", 3.6], ["2020-03-31", 8.62]) : bars(["2021-06-30", 42.5], ["2026-10-27", 17.0])];
    if (sym === "HNZ") return feed === "sip" ? [403, { message: "subscription does not permit querying recent SIP data" }] : [200, { bars: [] }];
    return null;
  },
];

const run = (extra: Partial<Parameters<typeof runBarsCoverage>[0]> = {}, seen: Seen[] = []) =>
  runBarsCoverage({ env: ENV, fetch: mockFetch(ROUTES, seen), now: NOW, cases: CASES, ...FAST, ...extra });

test("whitelisted: bars_coverage is a diag probe with a case in bot.ts using the cached, sanitized wrapper", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("bars_coverage"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const start = bot.indexOf('case "bars_coverage":');
  assert.ok(start > 0);
  const body = bot.slice(start, bot.indexOf("case ", start + 10));
  assert.ok(body.includes("cachedBarsCoverage(") && body.includes("sanitizeDiag(report)"));
  assert.ok(body.includes("tradingBase: ALPACA_BASE"), "assets read uses the same (paper) trading host as the bot");
});

test("vendorMsgClass / summarizeBars: fixed tokens only; empty and null bars count as zero", () => {
  assert.equal(vendorMsgClass({ message: "subscription does not permit querying recent SIP data" }), "recent_sip_not_permitted");
  assert.equal(vendorMsgClass({ message: "invalid feed: delayed_sip" }), "invalid_feed");
  assert.equal(vendorMsgClass({ message: "too many requests" }), "rate_limited");
  assert.equal(vendorMsgClass({ message: "something else https://x" }), "other");
  assert.equal(vendorMsgClass(null), null);
  assert.deepEqual(summarizeBars({ bars: null }), { bars: 0, first_date: null, last_date: null, first_close: null, last_close: null, truncated: false });
  const s = summarizeBars({ bars: [bar("2016-01-04", 1.5), bar("2016-01-05", 2.5)], next_page_token: "abc" });
  assert.deepEqual(s, { bars: 2, first_date: "2016-01-04", last_date: "2016-01-05", first_close: 1.5, last_close: 2.5, truncated: true });
});

test("request shape: daily, adjustment=all, from 2010, end = yesterday, feed set, asof only on dated reads", async () => {
  const seen: Seen[] = [];
  await run({}, seen);
  const barCalls = seen.map((s) => new URL(s.url)).filter((u) => u.pathname.endsWith("/bars"));
  assert.ok(barCalls.length > 0);
  for (const u of barCalls) {
    assert.equal(u.host, "data.alpaca.markets");
    assert.equal(u.searchParams.get("timeframe"), "1Day");
    assert.equal(u.searchParams.get("adjustment"), "all");
    assert.equal(u.searchParams.get("start"), COVERAGE_START);
    assert.equal(u.searchParams.get("end"), "2026-10-27", "free plan refuses the latest SIP minutes: end is yesterday");
    assert.ok(["sip", "iex"].includes(String(u.searchParams.get("feed"))));
  }
  const twtr = barCalls.filter((u) => u.pathname.includes("/TWTR/"));
  assert.deepEqual(twtr.map((u) => String(u.searchParams.get("asof"))).sort(), ["-", "2022-09-01", "null"]);
  // Index symbols on both feeds; control on both; delisted on sip twice; iex only where sip was refused.
  const count = (sym: string, feed: string) => barCalls.filter((u) => u.pathname.includes(`/${sym}/`) && u.searchParams.get("feed") === feed).length;
  for (const sym of INDEX_SYMBOLS) { assert.equal(count(sym, "sip"), 1); assert.equal(count(sym, "iex"), 1); }
  assert.equal(count(CONTROL_SYMBOL, "iex"), 1);
  assert.equal(count("TWTR", "iex"), 0, "sip answered: no iex retry");
  assert.equal(count("HNZ", "iex"), 1, "sip refused: one iex read at the dated mapping");
});

test("report answers the question: history start, delisted coverage, reuse visible, assets counts", async () => {
  const r = await run();
  assert.deepEqual(r.summary.earliest_bar, { sip: "2016-01-04", iex: "2016-01-04" });
  assert.deepEqual(r.summary.history_reaches_2010, { sip: false, iex: false });
  const by = Object.fromEntries(r.delisted.map((d) => [d.symbol, d]));
  assert.equal(by.TWTR.reaches_event, true);
  assert.equal(by.TWTR.last_minus_event_days, 0);
  assert.equal(by.TWTR.default_asof_bars, 0, "today's mapping finds nothing for a retired symbol");
  assert.equal(by.S.last_date, "2020-03-31", "dated asof returns the OLD company (Sprint)");
  assert.equal(by.S.default_asof_first_date, "2021-06-30", "default asof returns the NEW company (reuse)");
  assert.equal(by.S.in_active_assets, true);
  assert.equal(by.S.raw_symbol_bars, 4, "asof=- returns old and new company under one string");
  assert.equal(by.TWTR.raw_symbol_last_date, "2022-10-27");
  // Date-free modes a batched builder could use: default mapping misses TWTR
  // (null bars) and S (starts 2021, after the event); asof=- covers both.
  assert.equal(r.summary.delisted_reaching_event_default_asof, 0);
  assert.equal(r.summary.delisted_reaching_event_raw_symbol, 2);
  assert.equal(by.HNZ.best_feed, "iex");
  assert.equal(by.HNZ.bars, 0);
  assert.equal(by.HNZ.reaches_event, false);
  assert.equal(r.summary.delisted_with_bars, 2);
  assert.equal(r.summary.delisted_reaching_event, 2);
  assert.equal(r.summary.delisted_before_history_start, 1, "HNZ (2013) predates a 2016 history start");
  const hnzSip = r.series.find((s) => s.id === seriesId({ symbol: "HNZ", feed: "sip", asof: "2013-05-01" }));
  assert.equal(hnzSip?.verdict, "not_entitled");
  assert.equal(hnzSip?.vendor_msg, "recent_sip_not_permitted");
  assert.deepEqual(r.assets, {
    http_inactive: 200, http_active: 200, inactive_count: 4, active_count: 3,
    inactive_examples: ["HNZ", "S", "TWTR", "ZZZQ"], inactive_symbols_also_active: 1, probed_delisted_in_inactive: 3,
  });
  assert.equal(r.keys.alpaca, true);
});

test("NEVER BODIES, NEVER KEYS: no URL, company name, vendor message or key reaches the report; sanitizer is a no-op", async () => {
  const r = await run();
  const flat = JSON.stringify(r);
  for (const leak of [ENV.ALPACA_KEY, ENV.ALPACA_SECRET, "https://", "data.alpaca.markets", "Twitter Inc", "Sprint", "subscription does not", "bad sym", "APCA"]) {
    assert.ok(!flat.includes(leak), `report leaked ${leak}`);
  }
  assert.deepEqual(sanitizeDiag(r), r);
});

test("GET only, keys only in headers, never an order path; source has no write verbs", async () => {
  const seen: Seen[] = [];
  await run({}, seen);
  for (const s of seen) {
    assert.equal(s.method, "GET");
    for (const k of [ENV.ALPACA_KEY, ENV.ALPACA_SECRET]) assert.ok(!s.url.includes(k), "key in URL");
    assert.equal(s.headers["APCA-API-KEY-ID"], ENV.ALPACA_KEY);
    assert.ok(/\/v2\/(stocks\/[A-Z.]+\/bars|assets)\?/.test(s.url), `unexpected endpoint ${s.url}`);
    assert.ok(!s.url.includes("/orders"));
  }
  const src = fs.readFileSync(path.join(here, "barsCoverage.ts"), "utf8");
  assert.ok(!/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src), "module must be GET-only");
  assert.ok(!src.includes("/orders"), "module must not reference any order path");
});

test("no keys: no network call, every series skipped", async () => {
  const seen: Seen[] = [];
  const r = await runBarsCoverage({ env: {}, fetch: mockFetch(ROUTES, seen), now: NOW, cases: CASES, ...FAST });
  assert.equal(seen.length, 0);
  assert.ok(r.series.every((s) => s.verdict === "skipped" && s.note === "no_key"));
  assert.equal(r.keys.alpaca, false);
  assert.equal(r.summary.calls_made, 0);
  assert.equal(r.assets.inactive_count, null);
});

test("budget: calls that cannot start in time are skipped, not fired", async () => {
  let t = 0;
  const seen: Seen[] = [];
  const slow: FetchLike = async (url, init) => { t += 5_000; return mockFetch(ROUTES, seen)(url, init); };
  const r = await runBarsCoverage({ env: ENV, fetch: slow, now: NOW, cases: CASES, ...FAST, concurrency: 1, budgetMs: 12_000, clock: () => t });
  assert.ok(seen.length <= 3, `fired ${seen.length} calls past the budget`);
  assert.ok(r.summary.calls_skipped > 0);
  assert.ok(r.series.some((s) => s.note === "budget_exhausted"));
});

test("runPaced: concurrency cap and spaced starts hold", async () => {
  let inFlight = 0, peak = 0;
  const starts: number[] = [];
  const t0 = Date.now();
  const jobs = Array.from({ length: 6 }, () => async () => {
    starts.push(Date.now() - t0); inFlight++; peak = Math.max(peak, inFlight);
    await new Promise((r) => setTimeout(r, 5)); inFlight--; return 1;
  });
  const out = await runPaced(jobs, { concurrency: 2, minIntervalMs: 15, budgetMs: 10_000, clock: Date.now });
  assert.deepEqual(out, [1, 1, 1, 1, 1, 1]);
  assert.ok(peak <= 2);
  assert.ok(starts[5] >= 60, `starts not spaced: ${starts.join(",")}`);
});

test("cache: second call is served from cache; fresh bypasses; concurrent calls share one run", async () => {
  _resetBarsCoverageCacheForTests();
  const seen: Seen[] = [];
  const cfg = { env: ENV, fetch: mockFetch(ROUTES, seen), now: NOW, cases: CASES, ...FAST };
  const [a, b] = await Promise.all([cachedBarsCoverage(cfg, { nowMs: 1 }), cachedBarsCoverage(cfg, { nowMs: 1 })]);
  const once = seen.length;
  assert.equal(a.cached, false); assert.equal(b.cached, false);
  const c = await cachedBarsCoverage(cfg, { nowMs: 2 });
  assert.equal(c.cached, true);
  assert.equal(seen.length, once, "concurrent first calls must not double the API load");
  await cachedBarsCoverage(cfg, { nowMs: 3, fresh: true });
  assert.equal(seen.length, once * 2);
  _resetBarsCoverageCacheForTests();
});

test("pre-registered delisted list: ~17 real cases spanning 2011-2023, valid dates, asof before event, reuse cases marked", () => {
  assert.ok(DELISTED_CASES.length >= 15);
  const syms = new Set(DELISTED_CASES.map((c) => c.symbol));
  assert.equal(syms.size, DELISTED_CASES.length);
  for (const c of DELISTED_CASES) {
    assert.match(c.event_date, /^\d{4}-\d{2}-\d{2}$/);
    assert.ok(c.asof < c.event_date, `${c.symbol}: asof must precede the event`);
    if (c.reused_from) assert.ok(c.reused_from > c.event_date);
  }
  assert.ok(DELISTED_CASES.some((c) => c.event_date < "2016-01-01"), "must test pre-2016 delistings");
  assert.ok(DELISTED_CASES.filter((c) => c.reused_from).length >= 2, "must test ticker reuse");
});
