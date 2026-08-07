import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { finiteOrNull, fmt, sanitizePosition, SANITIZED_FIELDS } from "./tradeChartGuards";

// ── the crash class (repair 2026-08-06) ─────────────────────────────────────
// Alpaca returns null pnl/price fields for options legs without quotes and
// for halted names; server parseFloat → NaN serializes back to null; the
// unguarded .toFixed calls in TradeChart then threw and the app-root
// ErrorBoundary took down the WHOLE bot dashboard.

test("finiteOrNull: null, undefined, NaN, Infinity and junk all become null", () => {
  for (const v of [null, undefined, NaN, Infinity, -Infinity, "abc", {}, ""]) {
    assert.equal(finiteOrNull(v), null, String(v));
  }
});

test("finiteOrNull: finite numbers pass through, numeric strings coerce", () => {
  assert.equal(finiteOrNull(3.14), 3.14);
  assert.equal(finiteOrNull(0), 0);
  assert.equal(finiteOrNull(-2), -2);
  assert.equal(finiteOrNull("4.5"), 4.5); // server sometimes stringifies
});

test("fmt: em-dash for unknown — never a fabricated 0.00", () => {
  assert.equal(fmt(null), "—");
  assert.equal(fmt(12.3456), "12.35");
  assert.equal(fmt(1.2, 1), "1.2");
});

test("sanitizePosition: a null-field position renders without throwing", () => {
  // the exact field shape that crashed the dashboard: options leg, no quote
  const raw = {
    ticker: "QQQ260918P00400000", qty: 2, side: "long", phase: 1, daysHeld: 3,
    entryPrice: 4.2, currentPrice: null, marketValue: null,
    pnl: null, pnlPct: NaN, stopPrice: undefined,
    takeProfitPrice: null, rMultiple: "NaN", highestPnl: Infinity,
  } as Record<string, unknown>;
  const p = sanitizePosition(raw);
  assert.equal(p.entryPrice, 4.2);
  for (const k of ["currentPrice", "pnl", "pnlPct", "stopPrice", "takeProfitPrice", "rMultiple", "highestPnl"] as const) {
    assert.equal(p[k], null, k);
  }
  // every sanitized field is now safe for `x != null && x.toFixed(...)`
  for (const k of SANITIZED_FIELDS) {
    const v = p[k];
    assert.ok(v == null || Number.isFinite(v), k);
  }
  // non-numeric fields pass through untouched
  assert.equal((p as Record<string, unknown>).ticker, "QQQ260918P00400000");
  assert.equal((p as Record<string, unknown>).phase, 1);
});

test("sanitizePosition: a fully-populated position is unchanged", () => {
  const raw = {
    ticker: "AAPL", qty: 10, side: "long", phase: 2, daysHeld: 5,
    entryPrice: 150, currentPrice: 155.5, marketValue: 1555,
    pnl: 55, pnlPct: 3.67, stopPrice: 145,
    takeProfitPrice: 165, rMultiple: 1.1, highestPnl: 60,
  };
  const p = sanitizePosition(raw);
  assert.equal(p.pnl, 55);
  assert.equal(p.takeProfitPrice, 165);
  assert.equal(p.stopPrice, 145);
});

// ── source ratchet (test_alpaca_feed precedent) ─────────────────────────────
// A refactor of TradeChart.tsx must keep (1) the single sanitation pass and
// (2) the null guards on every unconditional createPriceLine call — casts
// like `entryPrice as number` do NOT guard and are what shipped the crash.

test("TradeChart.tsx routes the raw position through sanitizePosition", () => {
  const src = readFileSync(new URL("../components/TradeChart.tsx", import.meta.url), "utf-8");
  assert.match(src, /sanitizePosition\(rawPosition\)/,
    "the sanitation pass was removed — null API fields will crash the dashboard again");
});

test("TradeChart.tsx guards every price line behind a null check", () => {
  const src = readFileSync(new URL("../components/TradeChart.tsx", import.meta.url), "utf-8");
  for (const field of ["entryPrice", "stopPrice", "currentPrice"]) {
    assert.match(src, new RegExp(String.raw`if \(position\.${field} != null\)`),
      `createPriceLine(${field}) lost its null guard`);
  }
  assert.doesNotMatch(src, /position\.\w+ as number/,
    "an `as number` cast reappeared — casts do not guard nulls into createPriceLine");
});
