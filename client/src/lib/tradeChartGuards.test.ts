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
// (2) the null guards on every drawn level — casts like `entryPrice as
// number` do NOT guard and are what shipped the crash. (The chart moved
// from components/TradeChart.tsx to client/src/bot/chart/ on 2026-10-21;
// the contract moved with it.)

test("the chart panel routes the raw position through sanitizePosition before any level is derived", () => {
  // 2026-10-21: the library chart (components/TradeChart.tsx) is gone; the
  // same contract now lives in the Bot page's own chart — ChartBody
  // (client/src/bot/panels.tsx) sanitizes the selected position before
  // levels.ts reads a single numeric field from it.
  const src = readFileSync(new URL("../bot/panels.tsx", import.meta.url), "utf-8");
  assert.match(src, /sanitizePosition\(sel\.position\)/,
    "the sanitation pass was removed — null API fields will crash the dashboard again");
});

test("every drawn level is guarded by a finiteness check, never a cast", () => {
  // draw.ts skips a level whose price is not finite (lines and gutter
  // pills alike); levels.ts only emits the entry level from a finite entry
  // price and checks the strike before an option overlay uses it.
  const draw = readFileSync(new URL("../bot/chart/draw.ts", import.meta.url), "utf-8");
  assert.ok((draw.match(/Number\.isFinite\(l\.price\)/g) ?? []).length >= 2,
    "draw.ts lost the finite-price guard on its level loops");
  const levels = readFileSync(new URL("../bot/chart/levels.ts", import.meta.url), "utf-8");
  assert.match(levels, /entry !== null && Number\.isFinite\(entry\)/,
    "levels.ts lost the finite-entry guard before the entry level");
  assert.match(levels, /Number\.isFinite\(m\.strike \?\? NaN\)/,
    "levels.ts lost the strike check before the option overlay");
  for (const file of ["../bot/panels.tsx", "../bot/chart/draw.ts", "../bot/chart/BotChart.tsx"]) {
    const src = readFileSync(new URL(file, import.meta.url), "utf-8");
    assert.doesNotMatch(src, /position\.\w+ as number/,
      `${file}: an \`as number\` cast reappeared — casts do not guard nulls`);
  }
});
