import { test } from "node:test";
import assert from "node:assert/strict";
import { eventOf, liveEvents, flashes, flashAttr, EVENT_TYPES, FEED_MAX, FLASH_MS } from "./live";

const row = (time: string, type: string, message: string) => ({ time, type, message });

test("eventOf: a decision row becomes an event with its kind, ticker and loudness; scans, signals and unknown types do not", () => {
  const t = eventOf(row("2026-10-21T13:30:21Z", "MORNING-TRADE", "KWEB: bought 120 @ 31.42 (score 75)"));
  assert.ok(t);
  assert.equal(t.kind, "trade"); assert.equal(t.ticker, "KWEB"); assert.equal(t.title, "trade · KWEB"); assert.equal(t.loud, true);
  assert.equal(t.detail, "bought 120 @ 31.42 (score 75)");
  const r = eventOf(row("2026-10-21T13:38:59Z", "OPTIONS-SLOT-FULL", "TSLA: options slots full (6/6)"));
  assert.ok(r); assert.equal(r.kind, "refusal"); assert.equal(r.loud, false); assert.equal(r.title, "refused · TSLA");
  const q = eventOf(row("2026-10-21T13:33:36Z", "QUEUE", "TSLA queued for market open (score 74.2)"));
  assert.ok(q); assert.equal(q.kind, "queue"); assert.equal(q.ticker, "TSLA");
  const k = eventOf(row("2026-10-21T10:02:00Z", "DRAWDOWN-KILL", "MAX DRAWDOWN KILL SWITCH: Equity $99008 is -10.6% below peak $110727"));
  assert.ok(k); assert.equal(k.kind, "halt"); assert.equal(k.loud, true);
  assert.equal(eventOf(row("2026-10-21T13:39:59Z", "TIER2", "Starting intelligence scan...")), null);
  assert.equal(eventOf(row("2026-10-21T13:31:00Z", "STREAM-SIGNAL", "NFLX: 503.0x vol, +1.58% bar (bullish)")), null);
  assert.equal(eventOf(row("2026-10-21T13:34:36Z", "DIAGNOSTIC", "[MEDIUM] Low win rate")), null);
  assert.equal(eventOf(row("not a time", "TRADE", "NVDA: bought")), null);
  assert.ok(Object.values(EVENT_TYPES).every((s) => typeof s.loud === "boolean"));
});

test("liveEvents: only rows newer than the last look, newest first, capped at the feed length", () => {
  const rows = [
    row("2026-10-21T13:30:21Z", "MORNING-TRADE", "KWEB: bought"),
    row("2026-10-21T13:33:36Z", "QUEUE", "TSLA queued for market open"),
    row("2026-10-21T13:33:36Z", "QUEUE", "VXUS queued for market open"),
    row("2026-10-21T13:38:59Z", "OPTIONS-SLOT-FULL", "TSLA: options slots full (6/6)"),
    row("2026-10-21T13:39:59Z", "TIER2", "Starting scan"),
  ];
  const since = Date.parse("2026-10-21T13:31:00Z");
  const ev = liveEvents(rows, since);
  assert.deepEqual(ev.map((e) => [e.kind, e.ticker]), [["refusal", "TSLA"], ["queue", "TSLA"], ["queue", "VXUS"]]);
  assert.equal(liveEvents(rows, Date.parse("2026-10-21T13:40:00Z")).length, 0);
  const many = Array.from({ length: 12 }, (_, i) => row(`2026-10-21T13:${String(10 + i).padStart(2, "0")}:00Z`, "QUEUE", `T${i} queued`));
  assert.equal(liveEvents(many, 0).length, FEED_MAX);
  assert.equal(liveEvents(many, 0)[0].ticker, "T11");
});

test("flashes: a changed number flashes with its direction, a changed string flashes, unchanged and vanished values do not, old flashes expire", () => {
  const now = 1_000_000;
  const first = flashes(null, { equity: 100, state: "idle" }, {}, now);
  assert.deepEqual(first, {});
  const f = flashes({ equity: 100, state: "idle", pnl: 5 }, { equity: 101, state: "scanning", pnl: 5 }, {}, now);
  assert.deepEqual(f, { equity: { dir: "up", at: now }, state: { dir: "changed", at: now } });
  const down = flashes({ equity: 101 }, { equity: 99 }, {}, now);
  assert.equal(down.equity?.dir, "down");
  // appearing counts, vanishing does not
  assert.equal(flashes({ x: null }, { x: 3 }, {}, now).x?.dir, "changed");
  assert.equal(flashes({ x: 3 }, { x: null }, {}, now).x, undefined);
  // a live flash is carried, an expired one dropped
  const carried = flashes({ a: 1 }, { a: 1 }, { a: { dir: "up", at: now - FLASH_MS + 100 }, b: { dir: "down", at: now - FLASH_MS - 1 } }, now);
  assert.deepEqual(carried, { a: { dir: "up", at: now - FLASH_MS + 100 } });
  assert.equal(flashAttr({ dir: "up", at: now - 10 }, now), "up");
  assert.equal(flashAttr({ dir: "up", at: now - FLASH_MS }, now), undefined);
  assert.equal(flashAttr(undefined, now), undefined);
});
