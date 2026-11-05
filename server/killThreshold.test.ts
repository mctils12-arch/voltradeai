// [RULE-REVIEW] 2026-11-05: the Node max-drawdown kill moves from -10% to -20%
// from peak (prior value -10). Live 2026-11-05: the -10% line latched at 15:46Z
// on a real (explained) loss, equity $89,937 vs peak $99,962, freezing the scan
// and the floor until a recovery above -8% or the 2-trading-day paper re-baseline.
// Over the floor study's 1993-2026 sample a -10% kill froze the volatility-
// managed floor 37 times (15 since 2013); at -20% it fires 5 times, only in
// real crashes, behind bot_engine's 18% DD halt. These tests pin the new line
// and what it does to that live latch.
// Run: npx tsx --test server/killThreshold.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { evaluateDrawdown } from "./drawdownGuard";
import { evaluateAutoResume, type AutoResumeInput } from "./killSwitchAutoResume";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

const PEAK = 99962.18;              // the live peak behind the 2026-11-05 latch
const t = (iso: string) => Date.parse(iso);

function live(over: Partial<AutoResumeInput> = {}): AutoResumeInput {
  return {
    nowMs: t("2026-11-06T14:31:00Z"),          // Fri 09:31 ET, market open
    paper: true,
    killSwitch: true,
    latchReason: "drawdown-kill (account route)",
    latchedAtMs: t("2026-11-05T15:46:34Z"),
    firstObservedAtMs: t("2026-11-06T14:30:00Z"),
    lastAutoResumeAtMs: null,
    marketOpen: true,
    account: { equity: "89968.03", status: "ACTIVE", trading_blocked: false, account_blocked: false },
    // a REAL loss: the book explains most of the drop (live: the evaluator said
    // "UNRECONCILED ... drawdown -9.96% not above -8.00%"), so the DATA_ANOMALY
    // fast path is out and only RECOVERED / PAPER_REBASE can resume
    positions: [
      { symbol: "QQQ", qty: "24", side: "long", avg_entry_price: "696.53", unrealized_pl: "-7100.00", asset_class: "us_equity" },
      { symbol: "SMH", qty: "11", side: "long", avg_entry_price: "522.85", unrealized_pl: "-2400.00", asset_class: "us_equity" },
    ],
    fills: { rows: [], truncated: false },
    equityPeak: PEAK,
    maxDrawdownPct: -20,
    ...over,
  };
}

test("bot.ts: the kill line is -20% from peak, and the resume message reads the configured line", () => {
  assert.ok(/maxDrawdownPct: -20,/.test(bot));
  assert.ok(!/maxDrawdownPct: -10,/.test(bot), "the old -10% line is gone");
  assert.ok(bot.includes("the ${state.maxDrawdownPct}% kill stays armed from the new baseline"),
    "the re-baseline audit line states the configured threshold, not a hardcoded -10%");
});

test("a -10% drawdown no longer trips the kill; -20% still does", () => {
  assert.equal(evaluateDrawdown("89937.00", PEAK, -20).kill, false);
  assert.equal(evaluateDrawdown("89937.00", PEAK, -10).kill, true, "premise: the live trip equity crossed the old line");
  assert.equal(evaluateDrawdown(String(PEAK * 0.799), PEAK, -20).kill, true);
});

test("the live latch clears through RECOVERED under -20% (peak kept), after the debounce", () => {
  let prev = { consecutiveOk: 0 };
  const out = [];
  for (let i = 0; i < 3; i++) {
    const d = evaluateAutoResume(live({ nowMs: t("2026-11-06T14:31:00Z") + i * 45_000 }), prev);
    out.push(d);
    prev = { consecutiveOk: d.consecutiveOk };
  }
  assert.deepEqual(out.map((d) => d.resume), [false, false, true], "three consecutive OK evaluations, then resume");
  const last = out[2];
  assert.equal(last.basis, "RECOVERED");
  assert.equal(last.rebaselinePeakTo, null, "a real peak is kept; the -20% line stays armed from it");
});

test("under the old -10% line the same latch would not clear (it waited for -8% or the 2-day rebase)", () => {
  let prev = { consecutiveOk: 0 };
  let last;
  for (let i = 0; i < 3; i++) {
    last = evaluateAutoResume(live({ maxDrawdownPct: -10, nowMs: t("2026-11-06T14:31:00Z") + i * 45_000 }), prev);
    prev = { consecutiveOk: last.consecutiveOk };
  }
  assert.equal(last!.resume, false);
});
