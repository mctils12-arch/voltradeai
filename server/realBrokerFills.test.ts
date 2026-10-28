// [RULE-REVIEW] 2026-10-27 — REAL BROKER FILLS (audit §7.1 / §7.2.5).
// Static pins on bot.ts: no entry-feedback record may carry a synthetic
// fill_price again. Before this change the regular-hours path wrote
// filled_avg_price x (1 + random haircut) as fill_price, the morning queue
// wrote trade.price x (1 + random haircut) and never read a fill, and the
// ETF path wrote the UNDERLYING's price. Now every entry record goes through
// recordEntryFill(buildEntryFillPayload(...)) with fillPrice/qty taken from
// confirmedEntryFill(<Alpaca order object>).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("no random haircut is computed inline or written as fill_price in bot.ts", () => {
  assert.ok(!/realisticFillPrice|mFillPrice/.test(bot), "the synthetic fill_price variables must be gone");
  assert.ok(!/(slippagePct|mSlip) = 0\.\d+ \+ Math\.random/.test(bot),
    "the volume-tier draw lives in entryFill.syntheticSlippagePct (comparison field only)");
  assert.ok(!/fill_price:\s*(realisticFillPrice|mFillPrice|trade\.price)/.test(bot));
});

test("every entry-record payload takes fillPrice/qty from confirmedEntryFill(<order>)", () => {
  const calls = bot.match(/recordEntryFill\(buildEntryFillPayload\(\{[\s\S]*?\}\)\)/g) || [];
  assert.equal(calls.length, 2, `expected 2 recordEntryFill sites (settleEntryFill + resolvePendingEntryFill), found ${calls.length}`);
  for (const c of calls) {
    assert.ok(/qty:\s*fill\.qty/.test(c) && /fillPrice:\s*fill\.price/.test(c), c);
  }
  const fills = bot.match(/const fill = confirmedEntryFill\((final|order)\);/g) || [];
  assert.equal(fills.length, 2);
});

test("the only direct track_fill RPC sites are recordExitFill and recordEntryFill", () => {
  const rpc = bot.match(/"track_fill",/g) || [];
  assert.equal(rpc.length, 2, `found ${rpc.length} direct "track_fill" pythonCall sites`);
});

test("all three entry paths hand their order to settleEntryFill", () => {
  assert.ok(/settleEntryFill\(pending\.orderId, orderStatus, entryArgs\)/.test(bot), "regular-hours batch confirmation");
  assert.ok(/settleEntryFill\(order\.id, order, \{[\s\S]*?session: "morning_queue"/.test(bot), "morning queue");
  assert.ok(/settleEntryFill\(etfOrderResult\.id, etfOrderResult, \{[\s\S]*?instrument: "etf"/.test(bot), "ETF path");
});

test("the sweep resolves pending entry fills before dropping any tracked order", () => {
  const start = bot.indexOf("async function sweepStaleOrders() {");
  const end = bot.indexOf("async function replaceIfBetter(", start);
  const sweep = bot.slice(start, end);
  const n = (sweep.match(/resolvePendingEntryFill\((stale|tracked)\)/g) || []).length;
  assert.equal(n, 3, "stale-cancel success, stale-cancel error, and reconciliation paths");
  assert.ok(/if \(!\(await resolvePendingEntryFill\(tracked\)\)\) continue;[\s\S]*?openOrders\.splice\(i, 1\)/.test(sweep));
});

test("resolvePendingEntryFill keeps a still-working order tracked and records nothing for an unfilled one", () => {
  const start = bot.indexOf("async function resolvePendingEntryFill(");
  const fn = bot.slice(start, bot.indexOf("function settleEntryFill(", start));
  assert.ok(/if \(!isTerminalOrderStatus\(final\?\.status\)\) return false;/.test(fn));
  assert.ok(/if \(fill\) \{\s*recordEntryFill\(/.test(fn));
  assert.ok(/catch \{\s*return false;\s*\}/.test(fn));
});
