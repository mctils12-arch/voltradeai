// R19 (KNOWN BROKEN follow-up, discovered 2026-07-09 via /api/diag/audit +
// /api/diag/orders live queries): the real-time WS position monitor —
// checkPositionOnTick, the ONLY caller of the stop-loss/take-profit/
// trailing-stop exit path per bot.ts's own comment ("Position monitoring...
// moved to the WebSocket handler for real-time, event-driven exits") — was
// hardcoded to wss://stream.data.alpaca.markets/v2/sip, the same paid
// entitlement tier whose 2026-07-06 rejection required alpaca_feed.py's
// data_feed() resolver for every REST call site (wishlist.md #9). That fix
// never reached this file: TypeScript, outside the Python-only ratchet in
// test_alpaca_feed.py's TestNoHardcodedFeeds (`if not f.endswith(".py")`).
// Live evidence (this session, /api/diag/audit type=STREAM/WS-EXIT/
// STARTUP, limit=500): across 19 container restarts spanning 22+ hours,
// "Real-time feed live" (subscription success) and "WS-EXIT" (any exit
// fired by this path) both had ZERO occurrences, while bot_engine.py's
// manage_positions() repeatedly flagged trailing_stop conditions via
// "POS-MONITOR-SYNC: bot_engine flagged X (trailing_stop) — WS monitor
// should handle" that bot.ts's tier1Reflex deliberately does not execute
// itself (by design, it defers entirely to this WS path). Net effect: for
// as long as SIP entitlement was rejected (or this connection otherwise
// failed to subscribe), regular positions' stop-loss/take-profit/
// trailing-stop exits were not firing in real time, silently.
//
// These tests pin the fix: switch to /v2/iex (no entitlement dependency —
// this stream is consumed only for `close` price on already-owned tickers,
// never for volume-based candidate discovery, so IEX's volume undercount
// — the reason alpaca_feed.py rejects iex for REST scanning — does not
// apply here) and add audit visibility to the two previously-silent
// failure points (a rejection frame, and every disconnect/reconnect) so a
// future failure of this path is visible on the next occurrence instead of
// requiring a live WS trace to notice.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function startStreamingFn(): string {
  const start = bot.indexOf("function startStreaming()");
  assert.ok(start > 0, "startStreaming() not found in bot.ts");
  const end = bot.indexOf("\n  function stopStreaming()", start);
  assert.ok(end > start, "startStreaming() body not found (stopStreaming marker missing)");
  return bot.slice(start, end);
}

test("position-monitor stream connects to /v2/iex, not the entitlement-gated /v2/sip", () => {
  const block = startStreamingFn();
  assert.ok(
    block.includes('new WebSocket("wss://stream.data.alpaca.markets/v2/iex")'),
    "must connect to /v2/iex — no paid entitlement, so this safety-critical path can't be silently starved again"
  );
  assert.ok(
    !/new WebSocket\(["']wss:\/\/stream\.data\.alpaca\.markets\/v2\/sip["']\)/.test(block),
    "must not hardcode /v2/sip — that entitlement can be rejected (2026-07-06 incident) with zero effect on this connection succeeding at the TCP/open level"
  );
});

test("a stream-level error frame ({T:'error'}) is audited, not silently dropped", () => {
  const block = startStreamingFn();
  assert.ok(block.includes('item.T === "error"'), "must explicitly handle Alpaca's error frame type");
  assert.ok(
    block.includes('audit("STREAM-ERROR"'),
    "error frames must reach the persisted audit log — this is the exact failure mode (auth/subscribe rejected) that left zero trail before this fix"
  );
});

test("every socket close is audited before the reconnect timer fires", () => {
  const block = startStreamingFn();
  const closeStart = block.indexOf('ws.on("close"');
  assert.ok(closeStart > 0, 'ws.on("close", ...) handler not found');
  const closeEnd = block.indexOf("\n    });", closeStart);
  const closeBlock = block.slice(closeStart, closeEnd);
  assert.ok(closeBlock.includes('audit("STREAM-DISCONNECT"'), "close handler must audit before scheduling reconnect");
  assert.ok(closeBlock.includes("wasConnected"), "must distinguish a blip (was subscribed) from a connection that never subscribed at all");
  assert.ok(closeBlock.includes("setTimeout"), "reconnect scheduling must be unchanged — this is an additive visibility fix, not a behavior change");
});

test("STREAM-ERROR stays the shared audit type for both frame-level and socket-level failures", () => {
  // Guards against a future edit fragmenting the two failure classes into
  // differently-named audit types that a dashboard/alert would need to
  // track separately for no reason — both mean "the safety-critical feed
  // is not delivering bars right now."
  const matches = bot.match(/audit\("STREAM-ERROR"/g) || [];
  assert.equal(matches.length, 2, "expected exactly 2 STREAM-ERROR audit call sites (frame-level {T:'error'} + socket-level ws.on('error'))");
});
