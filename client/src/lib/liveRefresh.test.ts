// liveRefresh.test.ts — price queries re-read while their page is open
// (2026-11-11; human: tickers on the new pages stayed at their start-of-day
// values), and the analyze header says how fresh its price is.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { LIVE_ANALYSIS, LIVE_QUOTES, LIVE_SCANNER, ON_FOCUS, priceAsOfLabel } from "./liveRefresh";

const pages = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "pages");
const src = (f: string) => fs.readFileSync(path.join(pages, f), "utf8");

test("every policy refreshes on focus; the intervals scale with the server's cost", () => {
  for (const p of [LIVE_QUOTES, LIVE_SCANNER, LIVE_ANALYSIS, ON_FOCUS]) assert.equal(p.refetchOnWindowFocus, true);
  assert.equal(LIVE_QUOTES.refetchInterval, 60_000);
  assert.equal(LIVE_SCANNER.refetchInterval, 120_000);
  assert.equal(LIVE_ANALYSIS.refetchInterval, 300_000);
  assert.equal("refetchInterval" in ON_FOCUS, false, "heavy reports refresh on focus only");
  for (const p of [LIVE_QUOTES, LIVE_SCANNER, LIVE_ANALYSIS]) assert.equal("refetchIntervalInBackground" in p, false, "a hidden tab never polls");
});

test("each price query on the Trade page and the Ticker Dossier uses a policy", () => {
  assert.match(src("watchlist.tsx"), /\.\.\.LIVE_QUOTES/);
  assert.match(src("scanner.tsx"), /\.\.\.LIVE_SCANNER/);
  assert.match(src("analyze.tsx"), /staleTime: 30000,\n\s+\.\.\.LIVE_ANALYSIS/);
  const d = src("tickerDossier.tsx");
  assert.match(d, /\/api\/ticker-dossier\/\$\{sym\}`, 125_000\), staleTime: 4 \* 60_000, retry: 1, \.\.\.LIVE_ANALYSIS \}/);
  assert.match(d, /\/api\/bot\/lookup\/\$\{sym\}`, 120_000\), staleTime: 60_000, retry: 0, \.\.\.ON_FOCUS \}/);
  for (const f of ["InsightsView.tsx", "ETFBuilderView.tsx", "research.tsx"]) assert.match(src(f), /\.\.\.ON_FOCUS/, f);
});

test("the as-of label: live time in ET for a session in progress, the date for a close, nothing when unknown", () => {
  assert.equal(priceAsOfLabel({ price_asof: "2026-11-11T20:58:01.5Z", price_is_partial_session: true }), "live · 3:58 PM ET");
  assert.equal(priceAsOfLabel({ price_asof: "2026-11-10", price_is_partial_session: false }), "close · Nov 10");
  assert.equal(priceAsOfLabel({ price_asof: null }), "");
  assert.equal(priceAsOfLabel({}), "");
  assert.equal(priceAsOfLabel(null), "");
  assert.equal(priceAsOfLabel({ price_asof: "not a time", price_is_partial_session: true }), "");
});

test("the analyze header renders the label next to the price", () => {
  const a = src("analyze.tsx");
  assert.match(a, /data-testid="text-price-asof"/);
  assert.match(a, /priceAsOfLabel\(data\)/);
});

test("the Trade page's hero tile states the price's own as-of when the server gives one", () => {
  const t = fs.readFileSync(path.join(pages, "..", "trade", "TradePage.tsx"), "utf8");
  assert.match(t, /priceAsOfLabel\(data\) \|\| `\$\{data\.data_source/);
});

