import { test } from "node:test";
import assert from "node:assert/strict";
import { deriveWatch, scanAgeMs } from "./watch";

const scan = {
  timestamp: "2026-10-21T14:00:00", scanned: 12138, filtered: 54, deep_analyzed: 5, slots_available: 2,
  new_trades: [{ ticker: "kweb", score: 75, side: "sell", action_label: "SELL COVERED CALL", price: 24.5 }],
  sizing_blocks: [{ ticker: "AMD", reason: "negative expectancy", score: 74.3 }, { ticker: "ZZZ", reason: "no cash", score: 60 }],
  top_10: [
    { ticker: "KWEB", score: 75, side: "sell", action_label: "SELL COVERED CALL", price: 24.54, change_pct: -1.2, reasons: ["a", "b"], rank: 1 },
    { ticker: "AMD", score: 74.3, reasons: ["m"] },
    { ticker: "NVDA", score: 70, side: "buy", price: 900 },
  ],
};


test("each top-10 row is marked by what the trade loop did with it", () => {
  const w = deriveWatch(scan as never);
  assert.deepEqual(w.rows.map((r) => [r.ticker, r.status]), [["KWEB", "taken"], ["AMD", "blocked"], ["NVDA", "not taken"], ["ZZZ", "blocked"]]);
  assert.equal(w.rows[1].blockReason, "negative expectancy");
  assert.equal(w.rows[0].action, "SELL COVERED CALL");
  assert.equal(w.rows[2].action, "BUY");
});
test("refusals outside the top 10 are appended and no ticker repeats", () => {
  const w = deriveWatch(scan as never);
  assert.equal(new Set(w.rows.map((r) => r.ticker)).size, w.rows.length);
  assert.equal(w.rows.find((r) => r.ticker === "ZZZ")?.score, 60);
});
test("the summary is carried and nothing is invented for missing fields", () => {
  const w = deriveWatch(scan as never);
  assert.equal(w.summary.scanned, 12138); assert.equal(w.summary.filtered, 54); assert.equal(w.summary.deepAnalyzed, 5); assert.equal(w.summary.slots, 2);
  assert.equal(w.rows[2].changePct, null);
  assert.ok(!("stop" in w.rows[0]) && !("target" in w.rows[0]));
});
test("no scan yet (the route's message body) and undefined are empty", () => {
  assert.equal(deriveWatch({ message: "No scan run yet." } as never).empty, true);
  assert.equal(deriveWatch(undefined).empty, true);
});
test("scan age reads a naive timestamp as UTC and clamps the future to 0", () => {
  const w = deriveWatch(scan as never);
  assert.equal(scanAgeMs(w, Date.parse("2026-10-21T14:10:00Z")), 600_000);
  assert.equal(scanAgeMs(w, Date.parse("2026-10-21T13:00:00Z")), 0);
  assert.equal(scanAgeMs(deriveWatch(undefined), 0), null);
});
