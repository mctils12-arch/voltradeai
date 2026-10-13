// POS-KILL SCOPE FIX 2026-10-13 — the ITEM 13 per-position kill in
// `syncMonitoredPositions()` (server/bot.ts, -25% for every position) was
// a stock-style percent stop, but it was evaluated for us_option positions
// too. For a just-sold short put Alpaca's unrealized_plpc is measured against
// the premium, so the bid/ask spread alone reads <= -25% and POS-KILL bought
// the contract back at market within seconds (live: DRAM -54.7% / ~-$35 in
// 27s; T/VZ/WMB/TXG), after which T1 re-sold it. options_manager.py owns
// option exits (2x-credit loss limit, MIN_HOLD_MINUTES, limit orders).
//
// Fix: options are not judged by the stock -25% stop, but they KEEP a
// per-position backstop at option-scaled thresholds (sold -150%, bought -75%):
// options_manager only runs inside scan_market, which returns early during
// the portfolio DD halt, so a blanket exemption would leave short options
// with no automatic loss exit exactly when markets are stressed. Between the
// stock threshold and the backstop: one POS-KILL-OPTION-HOLD audit line per
// position per day (its own type: POS-KILL-SKIP stays the duplicate-guard
// line). The kill branch itself stays byte-for-byte the pre-fix code: only
// POSITION_KILL_LOSS_PCT became per-position (STOCK_KILL_LOSS_PCT for stocks,
// the option backstop for options), so the three existing tests that slice
// that branch by its `if (pnlPct <= POSITION_KILL_LOSS_PCT) {` opener keep
// slicing the kill branch and nothing else. The pure helpers are executed;
// the branch structure is checked statically (the repo's style for
// syncMonitoredPositions).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

function extractFn(signature: string, plainSignature: string): string {
  const start = bot.indexOf(signature);
  assert.ok(start > 0, `${signature} not found in bot.ts`);
  const end = bot.indexOf("\n}", start);
  return bot.slice(start, end + 2).replace(signature, plainSignature);
}

// Executes the real helpers from bot.ts (both are pure — no closures over
// bot.ts state).
function loadExemptFn(): (sym: string, assetClass?: string) => boolean {
  const src =
    extractFn("function isOptionSymbol(ticker: string): boolean {", "function isOptionSymbol(ticker) {") +
    "\n" +
    extractFn(
      "function isPosKillExemptOption(symbol: string, assetClass?: string): boolean {",
      "function isPosKillExemptOption(symbol, assetClass) {",
    );
  return new Function(`${src}\nreturn isPosKillExemptOption;`)();
}

function loadThresholdFn(): (isOpt: boolean, side: "long" | "short", stock: number, shortOpt: number, longOpt: number) => number {
  const src = extractFn(
    'function posKillThresholdPct(isOption: boolean, side: "long" | "short", stockPct: number,\n                             optionShortPct: number, optionLongPct: number): number {',
    "function posKillThresholdPct(isOption, side, stockPct, optionShortPct, optionLongPct) {",
  );
  return new Function(`${src}\nreturn posKillThresholdPct;`)();
}

// The constants the ITEM 13 block actually uses, read from bot.ts.
function constFromBot(name: string): number {
  const m = bot.match(new RegExp(`const ${name} = (-?[0-9.]+);`));
  assert.ok(m, `${name} not found in bot.ts`);
  return parseFloat(m![1]);
}

// The ITEM 13 block: from the per-position kill constants to the stale-state
// read that follows the warn branch.
const item13 = slice("const STOCK_KILL_LOSS_PCT = -25.0;", "// Read stop state for this ticker");

test("option positions are classified exempt from the stock POS-KILL; stocks are not", () => {
  const exempt = loadExemptFn();
  // The live incident symbols (short puts) — by OCC shape alone and by asset_class.
  for (const sym of ["DRAM261120P00025000", "T261120P00027000", "VZ261120P00040000", "WMB261120P00055000", "TXG261120P00020000"]) {
    assert.equal(exempt(sym), true, `${sym} (OCC shape, no asset_class) must be exempt`);
    assert.equal(exempt(sym, "us_option"), true, `${sym} with asset_class us_option must be exempt`);
  }
  // asset_class alone is sufficient even if the symbol were short/odd.
  assert.equal(exempt("XYZ", "us_option"), true, "explicit us_option asset_class must be exempt regardless of symbol shape");
  // Stocks keep the -25% kill.
  for (const sym of ["AAPL", "DRAM", "T", "VZ", "WMB", "TXG", "BRK.B", "NVDA"]) {
    assert.equal(exempt(sym), false, `${sym} (no asset_class) must remain subject to POS-KILL`);
    assert.equal(exempt(sym, "us_equity"), false, `${sym} us_equity must remain subject to POS-KILL`);
  }
});

test("option kill thresholds: spread noise never kills, a blow-up always does (live incident numbers)", () => {
  const th = loadThresholdFn();
  const STOCK = constFromBot("STOCK_KILL_LOSS_PCT");
  const SHORT = constFromBot("OPTION_SHORT_KILL_LOSS_PCT");
  const LONG = constFromBot("OPTION_LONG_KILL_LOSS_PCT");
  assert.equal(STOCK, -25, "the stock threshold is unchanged");
  assert.equal(th(false, "long", STOCK, SHORT, LONG), STOCK, "stocks keep the stock threshold");
  assert.equal(th(false, "short", STOCK, SHORT, LONG), STOCK);
  const shortTh = th(true, "short", STOCK, SHORT, LONG);
  const longTh = th(true, "long", STOCK, SHORT, LONG);
  // The live round-trip: DRAM short put read -54.7% from the spread alone and was killed in 27s.
  assert.ok(-54.7 > shortTh, "a fresh short put at -54.7% (spread noise, live DRAM) must NOT be killed");
  // options_manager stops a sold option at 2x credit (-100%); the backstop sits past it.
  assert.ok(shortTh < -100, "short-option backstop must sit past options_manager's 2x-credit stop");
  assert.ok(-160 <= shortTh, "a short option at 2.6x credit (-160%) must be killed — the backstop holds during the Python DD halt");
  assert.ok(-80 <= longTh, "a long option down 80% must be killed by its backstop");
  assert.ok(longTh < -50 && longTh > -100, "long-option backstop sits past options_manager's 50% stop and before total loss");
});

const KILL_OPENER = "if (pnlPct <= POSITION_KILL_LOSS_PCT) {";
const SKIP_OPENER = "if (isOptionPos && pnlPct <= STOCK_KILL_LOSS_PCT && pnlPct > POSITION_KILL_LOSS_PCT) {";

test("the ITEM 13 block: the kill threshold is per-position and the option skip runs before the kill", () => {
  assert.ok(item13.includes("const isOptionPos = isPosKillExemptOption(ticker, pos.asset_class);"), "options must be detected via isPosKillExemptOption(ticker, pos.asset_class)");
  assert.ok(
    item13.includes("const POSITION_KILL_LOSS_PCT = posKillThresholdPct(isOptionPos, side, STOCK_KILL_LOSS_PCT,"),
    "the kill threshold must come from posKillThresholdPct (stock rule for stocks, option backstop for options)",
  );
  const skipAt = item13.indexOf(SKIP_OPENER);
  const killAt = item13.indexOf(KILL_OPENER);
  assert.ok(skipAt > 0, "an option between the stock rule and its backstop must reach the audit-only branch");
  assert.ok(killAt > skipAt, "the audit-only option branch must come before the kill branch");
  assert.ok(
    item13.slice(skipAt, killAt).trimEnd().endsWith("}"),
    "the option skip must be its own closed statement, not a branch that swallows the kill/warn chain",
  );
});

test("the kill-branch opener appears exactly once, so slice-based kill tests read the kill branch only", () => {
  // posKillDuplicateOrderGuard / posKillStaleOrderTracking /
  // exitFillRecordingContract all slice from this opener to the warn branch.
  // A second copy (e.g. inside the option skip) would shift their slice onto
  // the skip and let its POS-KILL-SKIP line satisfy their assertions.
  assert.equal(bot.split(KILL_OPENER).length - 1, 1, "exactly one kill-branch opener in bot.ts");
  const killBranch = slice(KILL_OPENER, "} else if (pnlPct <= POSITION_WARN_LOSS_PCT) {");
  assert.ok(!killBranch.includes("exits owned by options_manager"), "the option skip line must not sit inside the kill branch");
  assert.ok(!killBranch.includes("_posKillOptionSkipDay"), "the option skip dedupe must not sit inside the kill branch");
});

test("the option skip branch never submits an order and audits at most once per position per day", () => {
  const exemptBranch = slice(SKIP_OPENER, KILL_OPENER);
  assert.ok(!/alpaca\(/.test(exemptBranch), "the option branch must not call Alpaca (no liquidation, no order POST)");
  assert.ok(!/openOrders\.push/.test(exemptBranch), "the option branch must not register any order");
  assert.ok(!/continue;/.test(exemptBranch), "the skip must not bypass the rest of the loop (POS-WARN, stop state) for the option");
  assert.ok(!/audit\("POS-KILL-SKIP"/.test(exemptBranch), "POS-KILL-SKIP is reserved for the duplicate-liquidation guard; the option hold has its own type");
  assert.ok(/audit\("POS-KILL-OPTION-HOLD"/.test(exemptBranch), "an option past the stock threshold must leave a POS-KILL-OPTION-HOLD audit line");
  assert.ok(/options_manager/.test(exemptBranch), "the skip line must name options_manager as the exit owner");
  assert.ok(
    /if \(skipLog\[ticker\] !== skipDay\) \{[\s\S]*?skipLog\[ticker\] = skipDay;[\s\S]*?audit\("POS-KILL-OPTION-HOLD"/.test(exemptBranch),
    "the audit must be gated on a per-ticker per-day key (the loop runs every ~45s)",
  );
  assert.ok(/toISOString\(\)\.split\("T"\)\[0\]/.test(exemptBranch), "the dedupe key must be a calendar day, not a timestamp");
});

test("the kill branch is unchanged: liquidation POST, order params and duplicate guard still present", () => {
  assert.ok(/const STOCK_KILL_LOSS_PCT = -25\.0;/.test(item13), "the stock -25% threshold must be unchanged");
  const killBranch = slice(KILL_OPENER, "} else if (pnlPct <= POSITION_WARN_LOSS_PCT) {");
  assert.ok(/audit\("POS-KILL", /.test(killBranch), "a kill must still log POS-KILL");
  assert.ok(
    /const killOrderResult = await alpaca\("\/v2\/orders", \{\s*method: "POST",/.test(killBranch),
    "kills must still liquidate via the unchanged order POST",
  );
  assert.ok(/getOrderParams\(current, 'stop_loss'\)/.test(killBranch), "liquidation order params must be unchanged");
  assert.ok(/DUPLICATE LIQUIDATION GUARD/.test(killBranch), "the duplicate liquidation guard must still run before any kill");
});
