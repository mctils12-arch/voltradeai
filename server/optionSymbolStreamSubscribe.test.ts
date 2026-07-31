// REPAIR (found via live /api/diag/audit?type=STREAM-ERROR this session):
// the position monitor's WebSocket resubscribe logic added OCC option
// symbols (e.g. "AAL260828P00014000", from Alpaca's own `pos.symbol` field
// on an options position) into the same "subscribe"/"unsubscribe bars"
// message sent to the EQUITY market-data stream
// (wss://stream.data.alpaca.markets/v2/iex, see wsPositionFeed.test.ts).
// Alpaca rejects that entire batched message with `{code:400, msg:"invalid
// syntax"}` because option symbols aren't valid subscribable bars symbols
// on that feed — and because the message is batched, ANY legitimate equity
// ticker bundled in the same call silently failed to get real-time
// WS-driven exit monitoring too, falling back to the slower Tier-1 ~30s
// poll. Live evidence: /api/diag/audit?type=STREAM-ERROR showed
// "code=400 msg=invalid syntax" recurring dozens of times across
// 2026-07-30/07-31, each one immediately preceded by a POS-MONITOR
// "Subscribed to <OCC-symbol...>" audit entry. Option positions still need
// full monitoring (POS-KILL/POS-WARN/TIME-EXIT all operate on
// `activeTickers`/`monitoredPositions` directly, unaffected) — they just
// can't ride the equity bars WebSocket, which doesn't carry option data
// anyway. Fix: a shared `isOptionSymbol()` guard (reusing the existing OCC
// shape check from `/api/bot/bars/:ticker`) at both subscribe call sites.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function fnBlock(startMarker: string, endMarkers: string[]): string {
  const start = bot.indexOf(startMarker);
  assert.ok(start > 0, `${startMarker} not found in bot.ts`);
  let end = -1;
  for (const marker of endMarkers) {
    const idx = bot.indexOf(marker, start + startMarker.length);
    if (idx > start && (end === -1 || idx < end)) end = idx;
  }
  assert.ok(end > start, `end of ${startMarker} not found`);
  return bot.slice(start, end);
}

// Extract and actually execute isOptionSymbol — it's a pure function (no
// closures over bot.ts state), so behavioral testing is cheap and exact.
function loadIsOptionSymbol(): (t: string) => boolean {
  const start = bot.indexOf("function isOptionSymbol(ticker: string): boolean {");
  assert.ok(start > 0, "isOptionSymbol() not found in bot.ts");
  const end = bot.indexOf("\n}", start);
  const src = bot.slice(start, end + 2)
    .replace("function isOptionSymbol(ticker: string): boolean {", "function isOptionSymbol(ticker) {");
  const factory = new Function(`${src}\nreturn isOptionSymbol;`);
  return factory();
}

test("isOptionSymbol classifies OCC option symbols as options, plain tickers as not", () => {
  const isOptionSymbol = loadIsOptionSymbol();
  for (const sym of ["AAL260828P00014000", "AAL260918P00014000", "CIFR260918P00020000", "XLK260424P00149000"]) {
    assert.equal(isOptionSymbol(sym), true, `${sym} should be detected as an option symbol`);
  }
  for (const sym of ["AAPL", "IREN", "KWEB", "SPY", "BRK.B", "vxus"]) {
    assert.equal(isOptionSymbol(sym), false, `${sym} should NOT be detected as an option symbol`);
  }
});

test("syncMonitoredPositions excludes option symbols from the bulk WS bars subscribe", () => {
  const block = fnBlock("async function syncMonitoredPositions()", ["\n  async function", "\n  function addPositionToMonitor"]);
  assert.ok(block.includes("toSubscribe.push"), "toSubscribe.push not found in syncMonitoredPositions");
  const pushIdx = block.indexOf("toSubscribe.push");
  const guardLine = block.slice(Math.max(0, pushIdx - 250), pushIdx);
  assert.ok(
    /!isOptionSymbol\(ticker\)/.test(guardLine),
    "the bulk resubscribe guard must exclude option symbols — otherwise Alpaca's v2/iex stream rejects the whole batched subscribe message with code=400 'invalid syntax', silently starving any equity tickers bundled in the same call"
  );
});

test("addPositionToMonitor excludes option symbols from its single-ticker WS subscribe", () => {
  const block = fnBlock("function addPositionToMonitor(ticker: string, side: 'long' | 'short', entryPrice: number, qty: number) {", ["\n  function removePositionFromMonitor", "\n  function "]);
  assert.ok(
    /if \(!streamSet\.has\(ticker\) && !positionSubscribedTickers\.has\(ticker\) && !isOptionSymbol\(ticker\)\)/.test(block),
    "addPositionToMonitor must not subscribe an option symbol to the equity bars WebSocket"
  );
});

test("/api/bot/bars/:ticker reuses the shared isOptionSymbol helper (no duplicated regex)", () => {
  const start = bot.indexOf('app.get("/api/bot/bars/:ticker"');
  assert.ok(start > 0, "/api/bot/bars/:ticker route not found");
  const end = bot.indexOf("\n  });", start);
  const block = bot.slice(start, end);
  assert.ok(block.includes("isOptionSymbol(tickerStr)"), "route should call the shared isOptionSymbol helper, not its own inline regex");
});

test("exactly one isOptionSymbol declaration exists (single source of truth)", () => {
  const matches = bot.match(/function isOptionSymbol\(/g) || [];
  assert.equal(matches.length, 1, "expected exactly one isOptionSymbol function declaration");
});
