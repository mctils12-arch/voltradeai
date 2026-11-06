// The Trade cockpit's look-up runs ONE Python process on the server (2026-11-06,
// keep the system alive: the server's heap is ~2.7 GB and the container
// OOM-restarted tonight). Pinned two ways: the request model (pure), and the
// page's wiring — TradePage fetches only through useAnalyze, and each heavy
// body (research, insights, the ETF builder) mounts only for a press made on
// the symbol open now. The harness (visual_check.mjs, the Trade cockpit block)
// checks the same on the wire: the requests a look-up actually makes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { HEAVY_LOADS, NO_REQUESTS, isRequested, requestLoad, symbolFetches } from "./heavyLoads";

const here = path.dirname(new URL(import.meta.url).pathname);
const read = (rel: string) => readFileSync(path.resolve(here, rel), "utf8");
const page = read("./TradePage.tsx");
const pythonRoutes = (sym: string | null, r = NO_REQUESTS) => symbolFetches(sym, r).filter((x) => x.python).map((x) => x.route);

test("a look-up alone runs one Python-backed fetch, /api/analyze; research, insights and the ETF builder wait for their press", () => {
  assert.deepEqual(pythonRoutes("SPY"), ["/api/analyze/SPY"]);
  assert.deepEqual(symbolFetches("SPY", NO_REQUESTS).filter((x) => !x.python).map((x) => x.route), ["/api/market/news?ticker=SPY"], "the only other read: the News filter, one Alpaca HTTP call");
  for (const what of HEAVY_LOADS) assert.equal(isRequested(NO_REQUESTS, what, "SPY"), false, `${what} does not load on a look-up`);
  assert.deepEqual(symbolFetches(null, NO_REQUESTS), [], "no symbol, no symbol reads");
});

test("a press loads that one panel, for the symbol open at the press; a new look-up does not inherit it", () => {
  const r = requestLoad(NO_REQUESTS, "research", "SPY");
  assert.equal(isRequested(r, "research", "SPY"), true);
  assert.equal(isRequested(r, "insights", "SPY"), false, "a research press loads nothing else");
  assert.deepEqual(pythonRoutes("SPY", r), ["/api/analyze/SPY", "/api/research/SPY"]);
  assert.equal(isRequested(r, "research", "QQQ"), false, "the press was for SPY");
  assert.deepEqual(pythonRoutes("QQQ", r), ["/api/analyze/QQQ"]);
  const all = HEAVY_LOADS.reduce((acc, w) => requestLoad(acc, w, "SPY"), NO_REQUESTS);
  assert.deepEqual(pythonRoutes("SPY", all), ["/api/analyze/SPY", "/api/research/SPY", "/api/insights/SPY", "/api/etf/SPY"]);
  assert.equal(isRequested(all, "insights", null), false, "nothing loads without a symbol");
});

test("TradePage fetches only through useAnalyze and mounts each heavy body only for a press on the open symbol", () => {
  assert.equal((page.match(/useAnalyze\(/g) ?? []).length, 1, "one analyze query, shared by the strip, Options and Stock");
  assert.doesNotMatch(page, /useQuery|apiRequest\(|fetch\(/, "no other fetch in the page itself");
  const gated: [string, RegExp][] = [
    ["ResearchBody", /isRequested\(heavy, "research", sym\) \? <ResearchBody ticker=\{sym\} \/> : onRequest\("research", `Load research for \$\{sym\}`/],
    ["InsightsView smart-money", /isRequested\(heavy, "insights", sym\) \? <InsightsView ticker=\{sym\} filter="smart-money" \/> : onRequest\("insights", `Load smart money for \$\{sym\}`/],
    ["InsightsView structure", /isRequested\(heavy, "insights", sym\) \? <InsightsView ticker=\{sym\} filter="structure" \/> : onRequest\("insights", `Load structure for \$\{sym\}`/],
    ["ETFBuilderView", /isRequested\(heavy, "etf", sym\) \? <ETFBuilderView ticker=\{sym\} \/> : onRequest\("etf", `Load ETF holdings for \$\{sym\}`/],
  ];
  for (const [what, re] of gated) assert.match(page, re, `${what} waits for its press`);
  // every mount of a heavy body is one of the gated ones above
  assert.equal((page.match(/<ResearchBody\b/g) ?? []).length, 1);
  assert.equal((page.match(/<InsightsView\b/g) ?? []).length, 2);
  assert.equal((page.match(/<ETFBuilderView\b/g) ?? []).length, 1);
  assert.match(page, /useEffect\(\(\) => \{ setHeavy\(NO_REQUESTS\); \}, \[sym\]\);/, "a new look-up clears every press");
  assert.match(page, /const loadNow = useCallback\(\(what: HeavyLoad\) => \{ if \(sym\) setHeavy\(\(h\) => requestLoad\(h, what, sym\)\); \}, \[sym\]\);/, "a press is for the symbol open now");
  assert.doesNotMatch(page, /etfOpen/, "the ETF builder no longer stays open across look-ups");
  assert.match(page, /ageMs: id === "options" \|\| id === "stock" \? ageMs : null,/, "Smart Money and Structure do not show the analyze query's age as theirs");
});

test("the heavy bodies fetch only when mounted with a symbol, and Smart Money + Structure share one query", () => {
  const research = read("../pages/research.tsx");
  const insights = read("../pages/InsightsView.tsx");
  const etf = read("../pages/ETFBuilderView.tsx");
  assert.match(research, /queryKey: \["\/api\/research", ticker\],[\s\S]{0,400}?enabled: !!ticker,/);
  assert.match(insights, /queryKey: \["\/api\/insights", ticker\],[\s\S]{0,400}?enabled: !!ticker,/, "one key per symbol whatever the filter: one server run fills both panels");
  assert.match(etf, /queryKey: \["\/api\/etf", ticker\],[\s\S]{0,400}?enabled: !!ticker,/);
});
