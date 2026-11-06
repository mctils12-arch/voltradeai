// Source-text pins for the 2026-10-29 Trade-world change (human: "trade data
// forecast brain ai engine, what do we need, it seems like a lot … I want
// this moving map like this but more modern as well for some of the screens
// … if you look for a ticker symbol and then you might want the stock vs.
// options info on it, and trade information"; on the first preview: "it
// needs to look like the bot UI but have the functionality of what's there
// and no duplicates").
//
// Pinned here so the next session cannot quietly re-grow the nav, move the
// map back into one page, re-route the old pages, or drop what they showed:
//   0. the owner reaches the bot page, Dossier, Forecast and Brain through ONE
//      drop-down (ENGINE_MENU), not four direct buttons, and every one of them
//      stays a TABS entry, so deep links and the phone More sheet keep working;
//   1. the moving map (DataWorldMap) is mounted once by the shell for the
//      Trade world (TradeBackdrop) and no longer by analyze.tsx;
//   2. every Trade tab renders the one cockpit page (trade/TradePage) on the
//      bot page's engine, with the bodies (tradeBodies.test.ts), one look-up
//      box and one active symbol (lib/activeTicker);
//   3. "#/analyze/<section>/<SYMBOL>" is a shareable link to one symbol;
//   4. what the old Analyze and Scanner pages showed is still on the page
//      (2026-11-06, the port onto tonight's tip): the live price bar, why no
//      spread is ranked, the options disclaimer, the sector heatmap when the
//      screener has room, and a look-up box in a zoomed tool tab (the old
//      Research page had its own input).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const read = (rel: string) => readFileSync(path.resolve(here, rel), "utf8");
const home = read("./home.tsx");
const analyze = read("./analyze.tsx");
const trade = read("../trade/TradePage.tsx");
const css = read("../index.css");

test("one owner-only AI Engine drop-down carries the bot page, dossier, forecast and brain", () => {
  assert.match(home, /const ENGINE_MENU: NavMenu = \{\s*id: "engine", label: "AI Engine"/);
  for (const tab of ["bot", "dossier", "forecast", "brain"]) {
    assert.match(home, new RegExp(`key: "engine-[a-z]+"[^\\n]*action: \\{ kind: "tab", tab: "${tab}" \\}`), `ENGINE_MENU has a leaf for ${tab}`);
  }
  assert.match(home, /\(isOwner \? \[\.\.\.NAV_MENUS, ENGINE_MENU\] : NAV_MENUS\)\.map\(/, "the engine menu renders only for the owner");
  assert.match(home, /engine: new Set<TabId>\(\["bot", "dossier", "forecast", "brain"\]\)/, "MENU_TABS highlights the engine menu for its tabs");
});

test("the four standalone owner buttons are gone from the desktop nav", () => {
  for (const tab of ["forecast", "brain", "dossier", "bot"]) {
    assert.doesNotMatch(home, new RegExp(`className=\\{\`tab-btn \\$\\{activeTab === "${tab}"`), `no direct tab-btn for ${tab}`);
  }
});

test("every owner page is still a TABS entry (deep links + phone More sheet)", () => {
  for (const id of ["forecast", "brain", "dossier", "bot"]) {
    assert.match(home, new RegExp(`\\{ id: "${id}",[^\\n]*requiresOwner: true`), `${id} stays in TABS`);
  }
});

test("the moving map is the Trade world's shared backdrop, mounted once by the shell", () => {
  assert.match(home, /import TradeBackdrop from "@\/components\/TradeBackdrop"/);
  assert.match(home, /const isTradeTab = MENU_TABS\.trade\.has\(activeTab\)/);
  assert.match(home, /\{isTradeTab && <TradeBackdrop \/>\}/);
  assert.doesNotMatch(analyze, /<DataWorldMap/, "analyze.tsx no longer mounts the map itself");
  assert.doesNotMatch(analyze, /import\("@\/components\/DataWorldMap"\)/, "analyze.tsx no longer lazy-loads the map");
  assert.match(analyze, /setTradeBackdrop\(\{ isLoading, hasData: !!data && !data\.error, ticker: ticker \|\| null \}\)/, "analyze reports its state to the backdrop store");
  assert.match(analyze, /resetTradeBackdrop\(\)/, "analyze resets the backdrop on unmount");
});

test("every Trade tab renders the one cockpit page; the old pages are no longer routed", () => {
  assert.match(home, /import TradePage from "@\/trade\/TradePage"/);
  assert.match(home, /\{isTradeTab && \(\s*<TradePage tab=\{activeTab as [^}]+\} section=\{analyzeSection\} authenticated=\{authenticated\} isOwner=\{!!isOwner\} \/>/);
  for (const page of ["AnalyzePage", "ScannerPage", "ResearchPage", "NewsPage", "WatchlistPage", "PlannerPage", "TaxesPage"]) {
    assert.doesNotMatch(home, new RegExp(`<${page}\\b`), `${page} is not rendered by the shell any more`);
  }
  assert.match(home, /useSyncExternalStore\(subscribeActiveTicker, getActiveTicker, getActiveTicker\)/, "the shell still reads the active symbol for the URL");
  assert.equal((trade.match(/<LookupBox\b/g) ?? []).length, 1, "one look-up box on the cockpit");
  for (const body of ["ScannerBody", "NewsBody", "WatchlistBody", "ResearchBody", "PlannerBody", "TaxesBody", "InsightsView", "ETFBuilderView", "useAnalyze"]) {
    assert.ok(trade.includes(body), `the cockpit uses ${body} (no second implementation)`);
  }
  assert.match(trade, /className="vt-bot-page" data-world="trade"/, "same page root as the bot cockpit");
  assert.match(trade, /useCockpitLayout</, "same layout engine as the bot cockpit");
  assert.match(trade, /<PanelBoundary resetKey=\{sym \?\? ""\}>\{body\(id\)\}<\/PanelBoundary>/, "one panel's render error stays in its panel");
});

test("#/analyze/<section>/<SYMBOL> is a shareable link to one symbol's view", () => {
  assert.match(home, /function tickerFromHash\(hash: string\): string \| null/);
  assert.match(home, /const sym = tickerFromHash\(window\.location\.hash\);\s*\n\s*if \(sym\) setActiveTicker\(sym\);/, "the initial hash seeds the active symbol");
  assert.match(home, /sym \? `#\/analyze\/\$\{analyzeSection\}\/\$\{sym\}` : `#\/analyze\/\$\{analyzeSection\}`/, "the hash sync writes the symbol back");
  assert.match(home, /\}, \[activeTab, analyzeSection, activeTicker\]\);/, "a new symbol updates the URL");
});

test("what the old Analyze and Scanner pages showed is still on the cockpit (2026-11-06)", () => {
  assert.match(trade, /<LivePriceBar f=\{data\.fundamentals\} spot=\{data\.spot\} high52=\{data\.high_52\} low52=\{data\.low_52\} \/><FundamentalsPanel/, "the live price bar (day range, volume, market cap, beta) leads Stock · Fundamentals");
  assert.match(trade, /const m = spreadsEmptyMessage\(data\.ticker, !!data\.vol_surface\?\.length\);/, "no ranked spread says why (no live quotes, or nothing buildable)");
  assert.doesNotMatch(trade, /no spread cleared the bar/, "never a blanket 'none cleared the bar'");
  assert.match(trade, /For educational purposes only\. Not financial advice\. Options trading involves substantial risk of loss\./, "the options disclaimer the Analyze page carried");
  assert.match(trade, /<ScannerBody onSelectTicker=\{pick\} compact=\{!roomy\("screener"\)\} \/>/, "the sector heatmap comes back whenever the screener has room");
  assert.match(trade, /const roomy = \(id: TradePanelId\) => L\.flags\.zoomed === id \|\| L\.flags\.floating\.includes\(id\);/);
  assert.match(trade, /gridTemplateRows: "auto minmax\(0, 1fr\)"/, "the zoomed view keeps the strip at its natural height, as the bot page does");
  assert.match(trade, /const status = id === "options" && !tpl\.scrolls && !L\.flags\.zoomed \? lookupBox : undefined;/, "the Options header carries the box only when nothing is zoomed");
  assert.match(trade, /Esc returns<\/span><span style=\{\{ marginLeft: "auto" \}\}>\{lookupBox\}<\/span><\/div>/, "a zoomed tool tab (Research's Stock panel) keeps the look-up box in its bar");
});

test("the Trade page's CSS fits its own panels: six strip tiles on their own tracks, one news column in the left slot", () => {
  assert.match(css, /\.vt-bot-page\[data-world="trade"\] \.vt-bot-kpi \{ grid-template-columns: 1\.35fr 1\.25fr 1fr 1\.45fr 1\.05fr 1\.1fr; \}/);
  assert.match(css, /\.vt-bot-page\[data-world="trade"\] \.vt-bot-kpi \.vt-bot-panel:last-child \{ grid-column: auto; \}/, "no full-width sixth tile below 1180 px");
  assert.match(css, /@media \(min-width: 1180px\) \{ \.vt-bot-page\[data-world="trade"\] \.vt-bot-area-left \.news-grid \{ grid-template-columns: minmax\(0, 1fr\); \} \}/, "the 330-380 px left column's news cards in one column");
  assert.match(css, /\.vt-bot-page\[data-world="trade"\] \.vt-lookup-box input \{ width: 176px; max-width: 52vw; \}/, "the one look-up box fits its prompt at the 16 px touch font");
  const tiles = (trade.match(/<Tile\b/g) ?? []).length;
  assert.equal(tiles, 6, "the strip the six tracks are for");
});
