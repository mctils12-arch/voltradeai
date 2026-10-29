// Source-text pins for the 2026-10-29 Trade BODY split: every Trade feature
// is a body component (no page header, no own ticker input, no page-width
// wrapper) that the cockpit places inside a panel, while the default page
// exports stay as thin header + input + body wrappers. Pinned so the next
// session cannot quietly grow a second implementation (home.nav.test.ts
// pins the cockpit side: it must use THESE bodies).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const read = (rel: string) => readFileSync(path.resolve(here, rel), "utf8");

const PAGES = {
  analyze: read("./analyze.tsx"),
  scanner: read("./scanner.tsx"),
  news: read("./news.tsx"),
  watchlist: read("./watchlist.tsx"),
  research: read("./research.tsx"),
  planner: read("./planner.tsx"),
  taxes: read("./taxes.tsx"),
  insights: read("./InsightsView.tsx"),
  etf: read("./ETFBuilderView.tsx"),
};

test("analyze.tsx exports the data hook, its data type, the render pieces and the section meta", () => {
  const a = PAGES.analyze;
  assert.match(a, /export function useAnalyze\(ticker: string \| null/, "useAnalyze(ticker)");
  assert.match(a, /return useQuery<AnalyzeData>\(\{\s*queryKey: \["\/api\/analyze", ticker\]/, "same queryKey as before");
  assert.match(a, /enabled: !!ticker && enabled,/, "enabled only with a symbol");
  assert.match(a, /staleTime: 30000,/, "same staleTime");
  assert.match(a, /export interface AnalyzeData \{/, "AnalyzeData is exported");
  for (const piece of [
    "LivePriceBar", "VolMetricsPanel", "RecommendationCard", "EdgeFactorsPanel", "SentimentPanel",
    "EarningsIntelPanel", "FundamentalsPanel", "ValuationPanel", "SpreadCard", "VolSurfaceChart",
    "ScoreBar", "MetricCard", "AnalyzeEmptyState",
  ]) {
    assert.match(a, new RegExp(`\\nexport function ${piece}\\(`), `${piece} is exported`);
  }
  assert.match(a, /export const SECTION_META: Record<AnalyzeSection, AnalyzeSectionMeta> = \{/, "SECTION_META is module-level and exported");
  assert.match(a, /export type AnalyzeSection = /, "AnalyzeSection stays exported");
  assert.match(a, /export function AnalyzeEmptyState\(\{ section \}: \{ section: AnalyzeSection \}\)/, "the empty state takes the section");
});

test("the Analyze page is the same page: it consumes the hook, the meta and the empty state", () => {
  const a = PAGES.analyze;
  assert.match(a, /export default function AnalyzePage\(/, "default export stays");
  assert.match(a, /useAnalyze\(ticker, \{ enabled: subTab !== "etf-builder" \}\)/, "ETF Builder still skips /api/analyze");
  assert.match(a, /\{!ticker && !isLoading && <AnalyzeEmptyState section=\{subTab\} \/>\}/, "the page renders the shared empty state");
  assert.match(a, /const currentMeta = SECTION_META\[subTab\];/, "the badge reads the shared table");
  assert.equal((a.match(/useQuery</g) ?? []).length, 1, "one useQuery left in analyze.tsx (the hook)");
});

test("the duplicate Market Scanner inside analyze.tsx is gone; scanner.tsx is the one implementation", () => {
  const a = PAGES.analyze;
  assert.doesNotMatch(a, /MarketScanner/, "no MarketScanner in analyze.tsx");
  assert.doesNotMatch(a, /ScanProgressBar|ScanResponse|ScanProgressResponse|interface ScanResult/, "its private helpers went with it");
  assert.doesNotMatch(a, /\/api\/scan/, "analyze.tsx no longer calls /api/scan");
  assert.match(PAGES.scanner, /queryKey: \["\/api\/market\/scanner"\]/, "scanner.tsx still owns the scanner feed");
});

test("every Trade feature has a body export with the agreed props", () => {
  assert.match(PAGES.scanner, /export function ScannerBody\(\{ onSelectTicker, compact \}: \{ onSelectTicker: \(ticker: string\) => void; compact\?: boolean \}\)/);
  assert.match(PAGES.news, /export function NewsBody\(\{ ticker, onSelectTicker, showFilter \}: \{\s*ticker: string \| null;\s*onSelectTicker: \(ticker: string\) => void;\s*showFilter\?: boolean;\s*\}\)/);
  assert.match(PAGES.watchlist, /export function WatchlistBody\(\{ onSelectTicker, authenticated \}: \{ onSelectTicker: \(ticker: string\) => void; authenticated\?: boolean \}\)/);
  assert.match(PAGES.research, /export function ResearchBody\(\{ ticker \}: \{ ticker: string \| null \}\)/);
  assert.match(PAGES.planner, /export function PlannerBody\(\{ ticker \}: \{ ticker: string \| null \}\)/);
  assert.match(PAGES.taxes, /export function TaxesBody\(\)/);
});

test("bodies own their data and have no page header, no page-width wrapper", () => {
  const bodyOf = (src: string, name: string) => {
    const start = src.indexOf(`export function ${name}(`);
    assert.ok(start >= 0, `${name} defined`);
    const next = src.indexOf("\nexport default function", start);
    return src.slice(start, next < 0 ? undefined : next);
  };
  const bodies: [string, string, RegExp][] = [
    [PAGES.scanner, "ScannerBody", /useQuery<MarketSnapshotResponse>/],
    [PAGES.news, "NewsBody", /useQuery<NewsResponse>/],
    [PAGES.watchlist, "WatchlistBody", /queryKey: \["\/api\/watchlist"\]/],
    [PAGES.research, "ResearchBody", /queryKey: \["\/api\/research", ticker\]/],
    [PAGES.planner, "PlannerBody", /apiRequest\("POST", "\/api\/plan", payload\)/],
    [PAGES.taxes, "TaxesBody", /apiRequest\("POST", "\/api\/tax\/estimate", payload\)/],
  ];
  for (const [src, name, fetchPin] of bodies) {
    const body = bodyOf(src, name);
    assert.match(body, fetchPin, `${name} fetches its own data`);
    assert.doesNotMatch(body, /<h1\b/, `${name} has no page title`);
    assert.doesNotMatch(body, /maxWidth: (960|1000)\b/, `${name} has no page-width cap`);
    assert.doesNotMatch(body, /margin: "0 auto"/, `${name} does not center itself`);
  }
});

test("the host's symbol is a prop: bodies take it, they do not read the page input", () => {
  assert.match(PAGES.news, /const activeFilter = showFilter \? debouncedTicker : \(ticker \?\? ""\);/, "news filters by the prop unless its own filter is shown");
  assert.match(PAGES.news, /\{showFilter && \(/, "the typed filter renders only on request");
  assert.match(PAGES.news, /\}, 600\);/, "the typed filter keeps its debounce");
  assert.match(PAGES.planner, /const symbol = ticker \?\? ownTicker;/, "planner uses the prop, else its own field");
  assert.match(PAGES.planner, /\{ticker === null && \(/, "the ticker field renders only without a host symbol");
  assert.match(PAGES.planner, /setActiveTicker\(t\);/, "a built plan still writes the symbol back");
  assert.match(PAGES.scanner, /\{!compact && \(/, "compact drops the sector heatmap block");
});

test("the default page exports still exist and are header + body wrappers", () => {
  assert.match(PAGES.scanner, /export default function ScannerPage\(/);
  assert.match(PAGES.scanner, /<ScannerBody onSelectTicker=\{onSelectTicker\} \/>/);
  assert.match(PAGES.news, /export default function NewsPage\(/);
  assert.match(PAGES.news, /<NewsBody ticker=\{seed \|\| null\} onSelectTicker=\{onSelectTicker\} showFilter \/>/);
  assert.match(PAGES.news, /getActiveTicker\(\) \?\? ""/, "the page still seeds from the active symbol");
  assert.match(PAGES.watchlist, /export default function WatchlistPage\(/);
  assert.match(PAGES.watchlist, /<WatchlistBody onSelectTicker=\{onSelectTicker\} authenticated=\{authenticated\} \/>/);
  assert.match(PAGES.research, /export default function ResearchPage\(/);
  assert.match(PAGES.research, /<ResearchBody ticker=\{ticker \|\| null\} \/>/);
  assert.match(PAGES.research, /useIsFetching\(\{ queryKey: \["\/api\/research", ticker\] \}\)/, "the page button reads the body's query state without a second fetch");
  assert.match(PAGES.planner, /export default function PlannerPage\(/);
  assert.match(PAGES.planner, /<PlannerBody ticker=\{null\} \/>/, "the page keeps its own ticker field");
  assert.match(PAGES.taxes, /export default function TaxesPage\(/);
  assert.match(PAGES.taxes, /<TaxesBody \/>/);
  // The page-width caps moved up into the wrappers.
  assert.match(PAGES.planner, /maxWidth: 1000 \}\}>\s*<div>/);
  assert.match(PAGES.taxes, /maxWidth: 1000 \}\}>\s*<div>/);
  assert.match(PAGES.research, /maxWidth: 960, margin: "0 auto"/);
});

test("InsightsView and ETFBuilderView already are bodies: ticker prop, no page header", () => {
  assert.match(PAGES.insights, /export default function InsightsView\(\{ ticker, filter = "all" \}: InsightsViewProps\)/);
  assert.match(PAGES.insights, /export type InsightsFilter = "all" \| "smart-money" \| "structure";/);
  assert.match(PAGES.insights, /export interface InsightsViewProps \{/);
  assert.doesNotMatch(PAGES.insights, /<h1\b/);
  assert.match(PAGES.etf, /export default function ETFBuilderView\(\{ ticker \}: ETFBuilderViewProps = \{\}\)/);
  assert.match(PAGES.etf, /export interface ETFBuilderViewProps \{/);
  assert.doesNotMatch(PAGES.etf, /<h1\b/);
});

test("existing test ids survive the split", () => {
  for (const id of ["section-switcher", "form-ticker", "input-ticker", "button-analyze", "text-error", "section-results",
    "text-company", "text-ticker", "text-spot", "text-price-change", "text-vrp-signal", "spreads-empty",
    "section-live-price", "section-vol-metrics", "section-recommendation", "section-edge-factors",
    "section-sentiment", "section-earnings-intel", "section-fundamentals", "section-valuation"]) {
    assert.ok(PAGES.analyze.includes(`data-testid="${id}"`), `analyze keeps data-testid=${id}`);
  }
  assert.ok(PAGES.insights.includes('data-testid="section-insights"'));
  assert.ok(PAGES.etf.includes('data-testid="section-etf-builder"'));
});
