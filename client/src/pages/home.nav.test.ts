// Source-text pins for the 2026-10-29 Trade-world change (human: "trade data
// forecast brain ai engine, what do we need, it seems like a lot … I want
// this moving map like this but more modern as well for some of the screens
// … if you look for a ticker symbol and then you might want the stock vs.
// options info on it, and trade information").
//
// Pinned here so the next session cannot quietly re-grow the nav or move
// the map back into one page:
//   1. the owner reaches the cockpit, Forecast, Brain and Dossier through ONE
//      drop-down (ENGINE_MENU), not four direct buttons;
//   2. every one of those stays a TABS entry, so deep links and the phone
//      More sheet keep working;
//   3. the moving map (DataWorldMap) is mounted once by the shell for the
//      Trade world (TradeBackdrop) and no longer by analyze.tsx;
//   4. the active symbol follows the user: the shell shows a ticker bar and
//      Analyze, Research, Planner and News seed from lib/activeTicker.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const read = (rel: string) => readFileSync(path.resolve(here, rel), "utf8");
const home = read("./home.tsx");
const analyze = read("./analyze.tsx");

test("one owner-only AI Engine drop-down carries cockpit, dossier, forecast and brain", () => {
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
  const trade = read("../trade/TradePage.tsx");
  assert.equal((trade.match(/<LookupBox\b/g) ?? []).length, 1, "one look-up box on the cockpit");
  for (const body of ["ScannerBody", "NewsBody", "WatchlistBody", "ResearchBody", "PlannerBody", "TaxesBody", "InsightsView", "ETFBuilderView", "useAnalyze"]) {
    assert.ok(trade.includes(body), `the cockpit uses ${body} (no second implementation)`);
  }
  assert.match(trade, /className="vt-bot-page" data-world="trade"/, "same page root as the bot cockpit");
  assert.match(trade, /useCockpitLayout</, "same layout engine as the bot cockpit");
});

test("#/analyze/<section>/<SYMBOL> is a shareable link to one symbol's view", () => {
  assert.match(home, /function tickerFromHash\(hash: string\): string \| null/);
  assert.match(home, /const sym = tickerFromHash\(window\.location\.hash\);\s*\n\s*if \(sym\) setActiveTicker\(sym\);/, "the initial hash seeds the active symbol");
  assert.match(home, /sym \? `#\/analyze\/\$\{analyzeSection\}\/\$\{sym\}` : `#\/analyze\/\$\{analyzeSection\}`/, "the hash sync writes the symbol back");
  assert.match(home, /\}, \[activeTab, analyzeSection, activeTicker\]\);/, "a new symbol updates the URL");
});
