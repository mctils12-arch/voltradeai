// floorTracking.test.ts — the volatility-managed floor's watch list, compiled
// ([RULE-REVIEW] measurement, 2026-11-12). Pins every number's definition
// against arithmetic written out here, the pre-registered trigger against the
// study's own text, and the honesty rules: a missing input is null with a
// reason, the entry session is in no window. Since 2026-11-22 the drawdown
// leg has its limit (research/floor_study_drawdown_limit_2026-11-22.json) and
// is evaluated: it reads "not met" only of a full 63-session window with
// every close on record, and "met" as soon as the fall is past the limit.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  floorTracking, floorOrders, floorTrackingMessage, parseFloorTrackingConfig,
  FLOOR_TRACKING_MODE, FLOOR_TRACKING_TRIGGER_TEXT, FLOOR_TRACKING_LIMIT_PP, FLOOR_TRACKING_WINDOW_SESSIONS,
  FLOOR_TRACKING_LEVERED_MULTIPLE, FLOOR_TRACKING_MV_CHECK_TOL, FLOOR_TRACKING_STUDY_REBALANCES, FLOOR_TRACKING_STUDY_SESSIONS,
  FLOOR_TRACKING_DEFINITIONS, FLOOR_TRACKING_FILLS_LISTED,
  FLOOR_DRAWDOWN_LIMIT_PCT, FLOOR_DRAWDOWN_WINDOW_SESSIONS, FLOOR_DRAWDOWN_LIMIT_SOURCE, largestFall,
  type FloorTrackingBars, type FloorTrackingConfig, type FloorTrackingInputs, type FloorTrackingReport, type FloorTrackingSession,
} from "./floorTracking";
import { sanitizeDiag } from "./diag";
import { DAILY_LINE_KEEP, type DailyLine, type PositionLite } from "./dailyLine";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");

const CFG: FloorTrackingConfig = { mode: "vol_managed_spy", base: "SPY", levered: "SSO", cash: "BOXX", capitalPct: 0.85, band: 0.1 };
const CAL = { holidays: new Set<string>(), halfDays: new Set<string>() };
const DAY_MS = 86_400_000;

function line(day: string, o: { targetX?: number | null; lastEquity?: number | null; floorMv?: number | null; mode?: string; close?: string } = {}): DailyLine {
  return {
    version: 1, day, written_at: `${day}T21:21:00.000Z`, session_close: o.close ?? "16:00",
    account_vs_spy: { day: { last_equity: o.lastEquity === undefined ? 100_000 : o.lastEquity } },
    floor: { mode: o.mode ?? "vol_managed_spy", target_x: o.targetX === undefined ? null : o.targetX, floor_mv: o.floorMv === undefined ? null : o.floorMv },
  } as unknown as DailyLine;
}
function bars(closes: Record<string, number>, readAtMs: number): FloorTrackingBars {
  return { bars: Object.entries(closes).map(([date, close]) => ({ date, close })), read_at_ms: readAtMs };
}
function order(id: string, symbol: string, side: string, qty: number, price: number | null, filledAt: string | null, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id, symbol, side, type: "limit", qty: String(qty), filled_qty: price === null ? "0" : String(qty), filled_avg_price: price === null ? null : String(price),
    status: price === null ? "canceled" : "filled", submitted_at: filledAt ?? "2026-11-06T14:47:57Z", filled_at: filledAt, ...extra,
  };
}
const held = (rows: Record<string, number>): PositionLite[] =>
  Object.entries(rows).map(([symbol, qty]) => ({ symbol, qty, marketValue: null, unrealizedPl: null, option: false }));
const r4 = (x: number) => Math.round(x * 10_000) / 10_000;
const r2 = (x: number) => Math.round(x * 100) / 100;
const near = (got: number | null, want: number, tol = 1e-4) => {
  assert.ok(got !== null && Math.abs(got - want) <= tol, `got ${got}, want ${want}`);
};
const byDay = (r: FloorTrackingReport, day: string): FloorTrackingSession => {
  const s = r.sessions.find((x) => x.day === day);
  assert.ok(s, `no session row for ${day}`);
  return s;
};
function weekdays(start: string, n: number): string[] {
  const out: string[] = [];
  for (let t = Date.parse(`${start}T00:00:00Z`); out.length < n; t += DAY_MS) {
    const dow = new Date(t).getUTCDay();
    if (dow !== 0 && dow !== 6) out.push(new Date(t).toISOString().slice(0, 10));
  }
  return out;
}

// ── the live record (read 2026-11-12 through the daily_line, orders, positions-detail and reconstruct_pnl probes) ──
// Targets, prior-close equities and floor market values are the stored daily lines' own. The two orders are the
// floor's only fills: 82 SPY at 740.12 (09:47 ET) and 110 BOXX at 118.93 (09:51 ET) on 2026-11-06. The closes are
// the bars the report reads (Alpaca daily bars, adjustment=split, feed iex), as the reconstruct_pnl probe returned them.
const SPY = { "2026-11-04": 766.73, "2026-11-05": 733.43, "2026-11-06": 735.4, "2026-11-09": 752.72, "2026-11-10": 749.93, "2026-11-11": 759.8 };
const BOXX = { "2026-11-04": 118.87, "2026-11-05": 118.9, "2026-11-06": 118.95, "2026-11-09": 118.97, "2026-11-10": 118.99, "2026-11-11": 118.99 };
const NOW = Date.parse("2026-11-12T18:00:00Z");
const LIVE_LINES = [
  line("2026-11-05", { targetX: null, lastEquity: 93_623.46, floorMv: 0 }),
  line("2026-11-06", { targetX: 0.6835, lastEquity: 89_888.9, floorMv: 60_347.08 }),
  line("2026-11-09", { targetX: 0.6888, lastEquity: 89_160.63, floorMv: 61_721.4 }),
  line("2026-11-10", { targetX: 0.6225, lastEquity: 90_582.83, floorMv: 61_583.64 }),
  line("2026-11-11", { targetX: 0.6225, lastEquity: 90_363.79, floorMv: 62_330.66 }),
];
const LIVE_ORDERS = [
  order("4bfaf255-21f5-41be-bd81-a26f0b3e233f", "SPY", "buy", 82, 740.12, "2026-11-06T14:47:58.976943Z"),
  order("fe4fcbcf-e5e3-428c-a4bb-07cacf5b1926", "BOXX", "buy", 110, 118.93, "2026-11-06T14:51:00.342235Z"),
  order("875a84df-66ba-4f8b-baa9-b9c73a7220d1", "GLD", "sell", 71, 370.71, "2026-11-06T14:48:00.081331Z"),
];
function live(over: Partial<FloorTrackingInputs> = {}): FloorTrackingInputs {
  return {
    nowMs: NOW, cfg: CFG, lines: LIVE_LINES, bars: { base: bars(SPY, NOW), levered: bars({}, NOW), cash: bars(BOXX, NOW) },
    positions: held({ SPY: 82, BOXX: 110 }), orders: LIVE_ORDERS, ordersFromMs: Date.parse("2026-10-26T00:00:00Z"), calendar: CAL, errors: {}, ...over,
  };
}

test("each session: floor_pct, rule_pct and tracking_pp are the stated arithmetic on the stored records", () => {
  const r = floorTracking(live());
  assert.equal(r.status, "ok");
  assert.equal(r.reason, null);
  assert.deepEqual(r.sessions.map((s) => s.day), ["2026-11-11", "2026-11-10", "2026-11-09", "2026-11-06"], "newest first; 11-05 is before the floor");
  assert.equal(r.inputs.sessions_before_floor, 1, "the 11-05 line: no SPY or SSO held, none traded");
  assert.deepEqual([r.inputs.lines, r.inputs.first_line, r.inputs.last_line, r.inputs.orders, r.inputs.fills], [5, "2026-11-05", "2026-11-11", 2, 2], "GLD is not a floor ticker here");

  // 11-09: 82 shares held from the 11-06 close; target 0.6888 of equity = 0.6888 / 0.85 of capital
  const d9 = byDay(r, "2026-11-09");
  const capital = 0.85 * 89_160.63;
  const spy = 752.72 / 735.4 - 1;
  const floorPct = ((82 * (752.72 - 735.4)) / capital) * 100;
  const rulePct = (0.6888 / 0.85) * spy * 100;
  near(d9.capital, capital, 0.01);
  near(d9.spy_pct, spy * 100);
  near(d9.target_exposure, 0.6888 / 0.85);
  near(d9.floor_pnl, 82 * (752.72 - 735.4), 0.01);
  near(d9.rule_pnl, 0.6888 * 89_160.63 * spy, 0.01);
  near(d9.floor_pct, floorPct);
  near(d9.rule_pct, rulePct);
  near(d9.tracking_pp, floorPct - rulePct);
  assert.ok(d9.tracking_pp !== null && d9.tracking_pp < 0 && d9.tracking_pp > -0.05, "a few hundredths of a point: the floor sat just under its target");
  // the split: entering the session at 0.7956x against a 0.8104x target explains it all; nothing traded
  const heldX = (82 * 735.4) / capital;
  near(d9.held_exposure, heldX);
  near(d9.drift, Math.abs(0.6888 / 0.85 - heldX));
  near(d9.exposure_pp, (heldX - 0.6888 / 0.85) * spy * 100);
  near(d9.other_pp, 0, 2e-4);
  assert.equal(d9.entry, false);
  assert.equal(d9.rebalance, false);
  assert.equal(d9.reason, null);
  assert.deepEqual(d9.core, { orders: 0, filled: 0, dollars: 0 });
  assert.deepEqual(d9.mv_check, { rebuilt: r2(82 * 752.72), line: 61_721.4, diff_pct: r4((Math.abs(82 * 752.72 - 61_721.4) / (82 * 752.72)) * 100), ok: true });
  const spyLeg = d9.legs.find((l) => l.role === "base");
  assert.deepEqual(spyLeg, { role: "base", ticker: "SPY", shares_prev: 82, close_prev: 735.4, close: 752.72, fills_count: 0, fills: [], pnl: r2(82 * (752.72 - 735.4)), reason: null });
  // the cash fund is measured the same way and kept OUT of floor_pct
  near(d9.cash_fund_pnl, 110 * (118.97 - 118.95), 0.01);
  assert.equal(byDay(r, "2026-11-11").cash_fund_pnl, 0, "BOXX closed 11-11 where it closed 11-10");
  near(d9.cash_fund_pct, ((110 * (118.97 - 118.95)) / capital) * 100);

  // a down day and an up day, same arithmetic
  const d10 = byDay(r, "2026-11-10"), d11 = byDay(r, "2026-11-11");
  near(d10.tracking_pp, ((82 * (749.93 - 752.72)) / (0.85 * 90_582.83)) * 100 - (0.6225 / 0.85) * (749.93 / 752.72 - 1) * 100);
  near(d11.tracking_pp, ((82 * (759.8 - 749.93)) / (0.85 * 90_363.79)) * 100 - (0.6225 / 0.85) * (759.8 / 749.93 - 1) * 100);
  assert.ok(d10.tracking_pp !== null && d10.tracking_pp < 0, "above target on a down day");
  assert.ok(d11.tracking_pp !== null && d11.tracking_pp > 0, "above target on an up day");
  assert.equal(r.latest?.day, "2026-11-11");
  for (const s of [d9, d10, d11]) near((s.exposure_pp ?? NaN) + (s.other_pp ?? NaN), s.tracking_pp ?? NaN, 2e-4);
});

test("the entry session is shown with its shortfall and is in no window", () => {
  const r = floorTracking(live());
  const d6 = byDay(r, "2026-11-06");
  assert.equal(d6.entry, true);
  assert.equal(d6.tracking_pp, null, "no close-to-close floor return: the floor held nothing at the 11-05 close");
  assert.match(d6.reason ?? "", /^entry session: the floor held no base or levered shares at the previous close/);
  assert.equal(d6.held_exposure, 0);
  assert.equal(d6.rebalance, false, "building the position is not a rebalance");
  assert.deepEqual(d6.core, { orders: 1, filled: 1, dollars: r2(82 * 740.12) });
  assert.deepEqual(d6.cash_fund, { orders: 1, filled: 1, dollars: r2(110 * 118.93) });
  const leg = d6.legs.find((l) => l.role === "base");
  assert.deepEqual(leg?.fills, [{ side: "buy", qty: 82, price: 740.12, at: "2026-11-06T14:47:58.976Z", order_id: "4bfaf255-21f5-41be-bd81-a26f0b3e233f" }]);
  assert.equal(leg?.shares_prev, 0);
  // from the fill to the close against the target held from the previous close
  const capital = 0.85 * 89_888.9;
  const floorPnl = 82 * (735.4 - 740.12);
  const rulePnl = 0.6835 * 89_888.9 * (735.4 / 733.43 - 1);
  assert.ok(r.entry);
  assert.equal(r.entry.day, "2026-11-06");
  assert.equal(r.entry.first_fill_at, "2026-11-06T14:47:58.976Z");
  near(r.entry.floor_pnl, floorPnl, 0.01);
  near(r.entry.rule_pnl, rulePnl, 0.01);
  near(r.entry.shortfall_pp, ((floorPnl - rulePnl) / capital) * 100);
  assert.ok(r.entry.shortfall_pp !== null && r.entry.shortfall_pp < -0.7 && r.entry.shortfall_pp > -0.75, "-0.72 pp on the stored records");
  // ... and none of it reaches a window, the running total or the turnover summaries
  const want = (1 + (byDay(r, "2026-11-09").floor_pct ?? 0) / 100) * (1 + (byDay(r, "2026-11-10").floor_pct ?? 0) / 100) * (1 + (byDay(r, "2026-11-11").floor_pct ?? 0) / 100)
    - (1 + (byDay(r, "2026-11-09").rule_pct ?? 0) / 100) * (1 + (byDay(r, "2026-11-10").rule_pct ?? 0) / 100) * (1 + (byDay(r, "2026-11-11").rule_pct ?? 0) / 100);
  near(r.stored.tracking_pp, want * 100, 2e-4);
  assert.deepEqual([r.stored.sessions, r.stored.from, r.stored.to], [3, "2026-11-09", "2026-11-11"]);
  assert.deepEqual([r.rolling.sessions, r.rolling.window_sessions, r.rolling.complete, r.rolling.missing], [3, 20, false, []], "the entry is not a missing session");
  assert.equal(r.rolling.tracking_pp, r.stored.tracking_pp);
  assert.deepEqual([r.turnover.window.sessions, r.turnover.window.core.dollars, r.turnover.window.cash_fund.filled, r.turnover.window.rebalance_sessions], [3, 0, 0, 0]);
});

test("the rollback trigger: too little history is said plainly, and the drawdown leg is never 'not met' on a partial window", () => {
  const t = floorTracking(live()).rollback_trigger;
  assert.equal(t.text, FLOOR_TRACKING_TRIGGER_TEXT);
  assert.equal(t.status, "undetermined");
  assert.equal(t.tracking.status, "not_enough_history");
  assert.deepEqual([t.tracking.limit_pp, t.tracking.window_sessions, t.tracking.sessions_measured, t.tracking.windows_evaluated, t.tracking.worst], [3, 20, 3, 0, null]);
  assert.equal(t.tracking.reason, "3 measured sessions, a window needs 20 in a row");
  assert.ok(t.tracking.so_far && t.tracking.so_far.sessions === 3 && t.tracking.so_far.beyond_limit === false);
  // the account's closes on the stored records: 11-05 89,888.90 (the 11-06 line's last_equity), 11-06 89,160.63,
  // 11-09 90,582.83, 11-10 90,363.79; the 11-11 close is on no line of this fixture (the next line would carry it)
  const d = t.drawdown;
  assert.deepEqual([d.status, d.limit_pct, d.window_sessions, d.sessions_measured], ["not_enough_history", -14.64, 63, 4]);
  assert.equal(d.drawdown_pct, r2((89_160.63 / 89_888.9 - 1) * 100));
  assert.equal(d.drawdown_pct, -0.81);
  assert.deepEqual([d.peak, d.trough], [{ day: "2026-11-05", equity: 89_888.9 }, { day: "2026-11-06", equity: 89_160.63 }]);
  assert.deepEqual([d.from, d.to, d.missing, d.beyond_limit], ["2026-11-05", "2026-11-11", ["2026-11-11"], false]);
  assert.match(d.reason ?? "", /^1 session with no closing equity on record \(2026-11-11\): a fall through it is not seen$/);
  assert.match(t.summary, /^tracking: 3 measured sessions, a window needs 20 in a row; [+-]0\.\d\dpp so far; drawdown -0\.81% in 4 of 63 sessions \(limit -14\.64%; 1 unread\)$/);
});

// ── a long run: one position held at 0.80x of capital, SPY +1% a session, the target set away from it ──
// The account's equity grows with the position, so the floor enters every session at 0.80x.
function run(n: number, targetCapital: number, o: { dropLine?: string; flat?: boolean } = {}): { inputs: FloorTrackingInputs; days: string[]; closes: number[]; shares: number; capital: number[] } {
  const days = weekdays("2026-11-05", n + 1);           // days[0] is the close before the first session
  const closes = days.map((_, k) => 500 * (o.flat ? 1 : 1.01 ** k));
  const shares = (0.8 * 0.85 * 100_000) / closes[0];
  const equity = days.map((_, k) => (k === 0 ? 0 : 100_000 * (closes[k - 1] / closes[0])));   // at the previous close
  const spy: Record<string, number> = {};
  days.forEach((d, k) => { spy[d] = closes[k]; });
  const lines = days.map((d, k) => line(d, { targetX: targetCapital * 0.85, lastEquity: equity[k], floorMv: shares * closes[k] }))
    .slice(1).filter((l) => l.day !== o.dropLine);
  const now = Date.parse(`${days[n]}T23:00:00Z`);
  return {
    inputs: {
      nowMs: now, cfg: CFG, lines, bars: { base: bars(spy, now), levered: null, cash: null }, positions: held({ SPY: shares }),
      orders: [], ordersFromMs: Date.parse(`${days[0]}T00:00:00Z`) - 10 * DAY_MS, calendar: CAL, errors: {},
    },
    days, closes, shares, capital: equity.map((e) => 0.85 * e),
  };
}
function windowPp(x: ReturnType<typeof run>, target: number, from: number, to: number): number {
  let f = 1, r = 1;
  for (let k = from; k <= to; k++) {
    f *= 1 + (x.shares * (x.closes[k] - x.closes[k - 1])) / x.capital[k];
    r *= 1 + target * (x.closes[k] / x.closes[k - 1] - 1);
  }
  return (f - r) * 100;
}

test("rollback trigger MET: a 20-session window more than 3 pp from the rule, with the worst window named", () => {
  const x = run(25, 1.0);                                // held 0.8x against a 1.0x target, SPY +1% a session
  const r = floorTracking(x.inputs);
  assert.equal(r.sessions.length, 25);
  near(byDay(r, x.days[7]).tracking_pp, (0.8 - 1.0) * 1, 1e-4);
  const first = windowPp(x, 1.0, 1, 20), last = windowPp(x, 1.0, 6, 25);
  near(first, (1.008 ** 20 - 1.01 ** 20) * 100, 1e-9);   // 17.28% against 22.02%: -4.74 pp
  assert.ok(first < -FLOOR_TRACKING_LIMIT_PP);
  near(byDay(r, x.days[20]).window_pp, first, 2e-4);
  near(byDay(r, x.days[25]).window_pp, last, 2e-4);
  assert.equal(byDay(r, x.days[19]).window_pp, null, "19 sessions are not a window");
  const t = r.rollback_trigger;
  assert.equal(t.status, "met");
  assert.equal(t.tracking.status, "met");
  assert.equal(t.tracking.windows_evaluated, 6, "sessions 20..25 each end a window");
  assert.deepEqual([t.tracking.worst?.from, t.tracking.worst?.to], [x.days[1], x.days[20]]);
  near(t.tracking.worst?.tracking_pp ?? null, first, 2e-4);
  assert.equal(t.tracking.so_far, null);
  assert.match(t.summary, /^tracking BEYOND 3 pp over 20 sessions \(worst -4\.74pp, /);
  assert.match(r.message, /^ROLLBACK TRIGGER MET: tracking BEYOND 3 pp over 20 sessions/);
  assert.deepEqual([r.rolling.sessions, r.rolling.complete, r.rolling.missing], [20, true, []]);
  near(r.rolling.tracking_pp, last, 2e-4);
  near(r.stored.tracking_pp, windowPp(x, 1.0, 1, 25), 2e-4);
  assert.equal(r.stored.sessions, 25);
});

test("tracking inside the limit: the leg reads not_met, the trigger stays undetermined (the drawdown window is not full)", () => {
  const x = run(25, 0.85);                               // 0.05x under target: about -1 pp over 20 sessions
  const r = floorTracking(x.inputs);
  const t = r.rollback_trigger;
  assert.equal(t.tracking.status, "not_met");
  assert.ok(Math.abs(t.tracking.worst?.tracking_pp ?? 99) < FLOOR_TRACKING_LIMIT_PP);
  assert.equal(t.status, "undetermined", "one leg inside its limit does not clear a two-leg trigger");
  assert.match(t.summary, /^tracking within 3 pp over 20 sessions \(worst -\d\.\d\dpp, .*; 6 windows\); drawdown 0\.00% in 25 of 63 sessions \(limit -14\.64%; 1 unread\)$/);
  assert.equal(t.drawdown.status, "not_enough_history");
  assert.doesNotMatch(r.message, /ROLLBACK TRIGGER MET/);
  assert.match(r.message, /rollback trigger undetermined: tracking within 3 pp over 20 sessions/);
});

test("the limit is strict and two-sided: exactly 3 pp is not beyond it, 3.0002 pp is, and a floor ahead of its rule counts too", () => {
  // SPY flat except one session: the whole window is that session's tracking, set exactly
  const mk = (gapPp: number) => {
    const x = run(21, 0.8, { flat: true });
    const day = x.days[5], prev = x.days[4];
    const closes: Record<string, number> = {};
    x.days.forEach((d, k) => { closes[d] = k < 5 ? 500 : 550; });                // +10% on `day`, flat before and after
    const target = 0.8 + gapPp / 10;                                               // tracking = (0.8 - target) x 10% = -gapPp
    const lines = x.days.slice(1).map((d) => line(d, { targetX: (d === day ? target : 0.8) * 0.85, lastEquity: 100_000, floorMv: x.shares * closes[d] }));
    const r = floorTracking({ ...x.inputs, lines, bars: { base: bars(closes, x.inputs.nowMs), levered: null, cash: null } });
    assert.equal(byDay(r, prev).tracking_pp, 0);
    return r.rollback_trigger.tracking;
  };
  const at = mk(3), over = mk(3.0002), under = mk(-3.5);
  assert.equal(at.worst?.tracking_pp, -3, "the window is that one session's tracking");
  assert.equal(at.status, "not_met", "> 3 pp, not >= 3 pp");
  assert.deepEqual([over.status, over.worst?.tracking_pp], ["met", -3.0002]);
  assert.deepEqual([under.status, under.worst?.tracking_pp], ["met", 3.5], "a floor ahead of its rule by more than the limit is off its rule too");
});

test("a session with no daily line is unmeasured, and no window is stitched across it", () => {
  const x = run(30, 1.0, { dropLine: weekdays("2026-11-05", 31)[12] });
  const r = floorTracking(x.inputs);
  const gap = byDay(r, x.days[12]);
  assert.equal(gap.tracking_pp, null);
  assert.equal(gap.capital, null);
  assert.equal(gap.reason, "no daily line is stored for this session (no recorded target or prior-close equity)");
  near(gap.floor_pnl, x.shares * (x.closes[12] - x.closes[11]), 0.01);        // the dollars are still known
  // 11 measured, the gap, 18 measured: never 20 in a row
  assert.equal(r.rollback_trigger.tracking.sessions_measured, 29);
  assert.equal(r.rollback_trigger.tracking.status, "not_evaluable");
  assert.equal(r.rollback_trigger.tracking.windows_evaluated, 0);
  assert.equal(r.rollback_trigger.tracking.reason, "no 20 sessions in a row all have a tracking number");
  assert.equal(r.rollback_trigger.status, "undetermined");
  assert.ok(r.rollback_trigger.tracking.so_far?.beyond_limit, "20 measured sessions already past 3 pp: flagged, though no full window exists");
  assert.equal(r.sessions.every((s) => s.window_pp === null), true);
  assert.deepEqual([r.rolling.sessions, r.rolling.complete, r.rolling.missing], [19, false, [x.days[12]]]);
  assert.match(r.message, /last 19 of 20 sessions -\d\.\d\dpp \(1 unmeasured\)/);
});

// ── a rebalance day with the levered fund ───────────────────────────────────
function rebalance(): FloorTrackingInputs {
  const now = Date.parse("2026-11-12T23:00:00Z");
  return {
    nowMs: now, cfg: CFG,
    lines: [
      line("2026-11-10", { targetX: 0.85, lastEquity: 60_000, floorMv: 50_000 }),
      line("2026-11-11", { targetX: 1.02, lastEquity: 60_000, floorMv: 80 * 505 + 40 * 95.9 }),   // 1.2x of capital
      line("2026-11-12", { targetX: 1.02, lastEquity: 60_000, floorMv: 80 * 500 + 40 * 94 }),
    ],
    bars: {
      base: bars({ "2026-11-09": 495, "2026-11-10": 500, "2026-11-11": 505, "2026-11-12": 500 }, now),
      levered: bars({ "2026-11-10": 94, "2026-11-11": 95.9, "2026-11-12": 94 }, now), cash: bars({}, now),
    },
    positions: held({ SPY: 80, SSO: 40 }),
    orders: [
      order("o-sso", "SSO", "buy", 40, 95, "2026-11-11T15:00:00Z"),
      order("o-spy", "SPY", "sell", 20, 502, "2026-11-11T15:01:00Z"),
      order("o-dead", "SPY", "sell", 20, null, null, { submitted_at: "2026-11-11T14:50:00Z" }),
    ],
    ordersFromMs: Date.parse("2026-10-30T00:00:00Z"), calendar: CAL, errors: {},
  };
}

test("a rebalance: fills are marked to the close, the levered fund counts twice in the exposure and once in the P&L", () => {
  const r = floorTracking(rebalance());
  const capital = 0.85 * 60_000;
  const d = byDay(r, "2026-11-11");
  // SPY: 100 held x +5, and 20 sold at 502 against a 505 close; SSO: 40 bought at 95 against a 95.9 close
  const spyPnl = 100 * (505 - 500) - 20 * (505 - 502), ssoPnl = 40 * (95.9 - 95);
  near(d.floor_pnl, spyPnl + ssoPnl, 0.01);
  near(d.floor_pct, ((spyPnl + ssoPnl) / capital) * 100);
  near(d.target_exposure, 1.2);
  near(d.rule_pct, 1.2 * 1);
  near(d.tracking_pp, ((spyPnl + ssoPnl) / capital) * 100 - 1.2);
  near(d.held_exposure, (100 * 500) / capital);
  near(d.exposure_pp, ((100 * 500) / capital - 1.2) * 1, 2e-4);
  near(d.other_pp, (d.tracking_pp ?? NaN) - (d.exposure_pp ?? NaN), 2e-4);
  assert.equal(d.rebalance, true);
  assert.deepEqual(d.core, { orders: 3, filled: 2, dollars: 40 * 95 + 20 * 502 }, "the cancelled order was sent, not filled");
  assert.deepEqual(d.legs.map((l) => [l.ticker, l.shares_prev, l.fills_count, l.fills[0]?.side ?? null]), [["SPY", 100, 1, "sell"], ["SSO", 0, 1, "buy"], ["BOXX", 0, 0, null]]);
  assert.equal(d.mv_check?.ok, true);
  // the next session holds 80 SPY and 40 SSO: 2x on the levered fund in the exposure, its own price move in the P&L
  const n = byDay(r, "2026-11-12");
  near(n.held_exposure, (80 * 505 + FLOOR_TRACKING_LEVERED_MULTIPLE * 40 * 95.9) / capital);
  near(n.floor_pnl, 80 * (500 - 505) + 40 * (94 - 95.9), 0.01);
  near(n.tracking_pp, ((80 * (500 - 505) + 40 * (94 - 95.9)) / capital) * 100 - 1.2 * (500 / 505 - 1) * 100);
  assert.equal(n.rebalance, false);
  // the first stored session began holding 100 SPY: positions before the fills are rebuilt backwards from today's
  near(byDay(r, "2026-11-10").floor_pnl, 100 * (500 - 495), 0.01);
  assert.equal(r.entry, null);
});

test("turnover: orders, dollars and rebalance sessions per window, beside the study's pace; the band counts", () => {
  const r = floorTracking(rebalance());
  const capital = 0.85 * 60_000;
  const w = r.turnover.window;
  assert.deepEqual([w.sessions, w.from, w.to], [3, "2026-11-10", "2026-11-12"]);
  assert.deepEqual(w.core, { orders: 3, filled: 2, dollars: 13_840 });
  near(w.core_turnover_pct, (13_840 / capital) * 100);
  assert.deepEqual([w.rebalance_sessions, w.longest_rebalance_run], [1, 1]);
  // 11-11 began 0.22x from its new target: a rebalance the band called for. 11-12 began 0.74x of capital in SPY plus
  // 2 x 0.075x in SSO = 0.94x against 1.2x, outside the band, and nothing traded: counted.
  assert.equal(w.rebalances_inside_band, 0);
  assert.equal(w.outside_band_no_trade, 1);
  assert.deepEqual(r.turnover.stored, w);
  assert.deepEqual(r.turnover.study, {
    rebalances: 800, sessions: 8499, per_window: r2((800 / 8499) * 20),
    source: "research/floor_study_2026-11-05.json (exposure.V1_rebalances over window.sessions)",
  });
  assert.equal(r.turnover.study.per_window, 1.88);
  assert.match(r.message, /turnover: 1 rebalance session in 3 \(study pace 1\.9 per 20\), \$13840 core traded$/);
  // rebalances on consecutive sessions, and one inside the band (the watch list's "band too tight for live marks")
  const i = rebalance();
  (i.orders as unknown[]).push(order("o-spy2", "SPY", "sell", 1, 501, "2026-11-12T16:00:00Z"));
  i.positions = held({ SPY: 79, SSO: 40 });
  i.lines = [i.lines[0], i.lines[1], line("2026-11-12", { targetX: 0.85 * 0.95, lastEquity: 60_000, floorMv: 79 * 500 + 40 * 94 })];
  const two = floorTracking(i).turnover.window;
  assert.deepEqual([two.rebalance_sessions, two.longest_rebalance_run, two.rebalances_inside_band, two.outside_band_no_trade], [2, 2, 1, 0]);
  // without a band nothing is counted against it
  const noBand = floorTracking({ ...rebalance(), cfg: { ...CFG, band: null } }).turnover.window;
  assert.deepEqual([noBand.rebalances_inside_band, noBand.outside_band_no_trade], [null, null]);
});

test("fills after the last stored session's close are pending, and a half-day close splits the day at 13:00 ET", () => {
  const i = rebalance();
  (i.orders as unknown[]).push(order("o-late", "SPY", "buy", 5, 499, "2026-11-13T15:00:00Z"));
  i.positions = held({ SPY: 85, SSO: 40 });
  const r = floorTracking(i);
  assert.deepEqual(r.turnover.pending.core, { orders: 1, filled: 1, dollars: 5 * 499 });
  assert.deepEqual(r.turnover.pending.cash_fund, { orders: 0, filled: 0, dollars: 0 });
  near(byDay(r, "2026-11-12").floor_pnl, 80 * (500 - 505) + 40 * (94 - 95.9), 0.01);   // the stored sessions are unchanged by it
  assert.equal(byDay(r, "2026-11-12").mv_check?.ok, true);
  // 2026-11-11 as a half day: the 10:00 ET SSO fill stays in it, an 18:30Z (13:30 ET) SPY fill falls in the next session
  const h = rebalance();
  h.orders = [order("o-sso", "SSO", "buy", 40, 95, "2026-11-11T15:00:00Z"), order("o-spy", "SPY", "sell", 20, 502, "2026-11-11T18:30:00Z")];
  h.lines = [h.lines[0], line("2026-11-11", { targetX: 1.02, lastEquity: 60_000, floorMv: 100 * 505 + 40 * 95.9, close: "13:00" }), h.lines[2]];
  const half = floorTracking(h);
  assert.deepEqual(byDay(half, "2026-11-11").legs.map((l) => l.fills_count), [0, 1, 0]);
  assert.deepEqual(byDay(half, "2026-11-12").legs.map((l) => l.fills_count), [1, 0, 0]);
  near(byDay(half, "2026-11-12").floor_pnl, 100 * (500 - 505) - 20 * (500 - 502) + 40 * (94 - 95.9), 0.01);
  // a fill stamped at the close's own instant is that session's last fill, counted once
  const edge = rebalance();
  edge.orders = [order("o-sso", "SSO", "buy", 40, 95, "2026-11-11T15:00:00Z"), order("o-spy", "SPY", "sell", 20, 502, "2026-11-11T21:00:00.000Z")];
  const atClose = floorTracking(edge);
  assert.deepEqual(byDay(atClose, "2026-11-11").legs.map((l) => l.fills_count), [1, 1, 0]);
  assert.deepEqual(byDay(atClose, "2026-11-12").legs.map((l) => [l.shares_prev, l.fills_count]), [[80, 0], [40, 0], [0, 0]]);
  near(byDay(atClose, "2026-11-11").floor_pnl, 100 * (505 - 500) - 20 * (505 - 502) + 40 * (95.9 - 95), 0.01);
  // the same split from the calendar when that session has no line
  const c = rebalance();
  c.orders = h.orders;
  c.lines = [c.lines[0], c.lines[2]];
  c.calendar = { holidays: new Set<string>(), halfDays: new Set(["2026-11-11"]) };
  assert.deepEqual(byDay(floorTracking(c), "2026-11-12").legs.map((l) => l.fills_count), [1, 0, 0]);
});

test("nothing is inferred: each missing input voids its session with the reason", () => {
  // a bar read before its session closed is not a close
  const early = floorTracking(live({ bars: { base: bars(SPY, Date.parse("2026-11-11T20:30:00Z")), levered: bars({}, NOW), cash: bars(BOXX, NOW) } }));
  assert.equal(byDay(early, "2026-11-11").tracking_pp, null);
  assert.match(byDay(early, "2026-11-11").reason ?? "", /the SPY bar for 2026-11-11 was read before that session closed/);
  assert.ok(byDay(early, "2026-11-10").tracking_pp !== null, "the sessions closed before the read are unaffected");
  assert.equal(early.latest?.day, "2026-11-10");
  // the daily line recorded no target / no prior-close equity / another mode
  const lines = (k: number, l: DailyLine) => LIVE_LINES.map((x, j) => (j === k ? l : x));
  const noTarget = byDay(floorTracking(live({ lines: lines(3, line("2026-11-10", { targetX: null, lastEquity: 90_582.83, floorMv: 61_583.64 })) })), "2026-11-10");
  assert.equal(noTarget.tracking_pp, null);
  assert.equal(noTarget.rule_pct, null);
  near(noTarget.floor_pct, ((82 * (749.93 - 752.72)) / (0.85 * 90_582.83)) * 100);
  assert.equal(noTarget.reason, "the daily line recorded no floor target (the floor's state was stale or held no plan)");
  const noEquity = byDay(floorTracking(live({ lines: lines(3, line("2026-11-10", { targetX: 0.6225, lastEquity: null, floorMv: 61_583.64 })) })), "2026-11-10");
  assert.deepEqual([noEquity.tracking_pp, noEquity.capital, noEquity.floor_pct], [null, null, null]);
  assert.equal(noEquity.reason, "the daily line recorded no prior-close equity");
  const basket = byDay(floorTracking(live({ lines: lines(3, line("2026-11-10", { targetX: 0.6225, lastEquity: 90_582.83, mode: "basket" })) })), "2026-11-10");
  assert.equal(basket.tracking_pp, null);
  assert.match(basket.reason ?? "", /the floor's mode on this session was "basket"/);
  // the rebuilt value is not what the daily line read (a split, a missed fill): void, with both numbers
  const off = byDay(floorTracking(live({ lines: lines(3, line("2026-11-10", { targetX: 0.6225, lastEquity: 90_582.83, floorMv: 61_583.64 * 2 })) })), "2026-11-10");
  assert.equal(off.tracking_pp, null);
  assert.equal(off.mv_check?.ok, false);
  assert.match(off.reason ?? "", /the core's value at the 2026-11-10 close rebuilt from the orders \(\$61494\) is not the daily line's reading \(\$123167\)/);
  const ok5 = byDay(floorTracking(live({ lines: lines(3, line("2026-11-10", { targetX: 0.6225, lastEquity: 90_582.83, floorMv: 82 * 749.93 * (1 + FLOOR_TRACKING_MV_CHECK_TOL * 0.9) })) })), "2026-11-10");
  assert.ok(ok5.tracking_pp !== null && ok5.mv_check?.ok, "inside the tolerance: after-hours marks, not a different position");
  // the orders do not explain today's shares
  const negative = floorTracking(live({ positions: held({ SPY: 40, BOXX: 110 }) }));
  assert.match(byDay(negative, "2026-11-06").reason ?? "", /SPY: the orders do not explain the shares held now \(a rebuilt position is negative\)/);
  assert.equal(byDay(negative, "2026-11-09").mv_check?.ok, false, "after the fill the rebuilt 40 shares are not what the daily line read");
  assert.equal(negative.sessions.every((s) => s.tracking_pp === null), true, "so no session is measured on a position the records disagree on");
  assert.equal(negative.latest, null);
  // the orders read does not reach back to the session
  const short = floorTracking(live({ ordersFromMs: Date.parse("2026-11-10T00:00:00Z") }));
  assert.match(byDay(short, "2026-11-09").reason ?? "", /the orders read does not reach back to the 2026-11-06 close/);
  assert.ok(byDay(short, "2026-11-11").tracking_pp !== null);
  // a held levered fund with no close
  const lev = rebalance();
  lev.bars.levered = null;
  const noLev = floorTracking(lev);
  assert.equal(byDay(noLev, "2026-11-12").tracking_pp, null);
  assert.match(byDay(noLev, "2026-11-12").reason ?? "", /SSO daily closes are unreadable/);
  assert.ok(byDay(noLev, "2026-11-10").tracking_pp !== null, "a session that neither held nor traded it needs no close for it");
  // the closes read skip a session a line is stored for: the next row would span two sessions, so it is void
  const { "2026-11-10": _dropped, ...spyGap } = SPY;
  const gap = floorTracking(live({ bars: { base: bars(spyGap, NOW), levered: bars({}, NOW), cash: bars(BOXX, NOW) } }));
  assert.equal(gap.inputs.errors.lines_without_bar, "no SPY bar for the stored line(s) of 2026-11-10: not measured");
  assert.equal(gap.sessions.some((x) => x.day === "2026-11-10"), false);
  assert.equal(byDay(gap, "2026-11-11").prev_day, "2026-11-09");
  assert.equal(byDay(gap, "2026-11-11").tracking_pp, null);
  assert.match(byDay(gap, "2026-11-11").reason ?? "", /^the SPY closes read skip the 2026-11-10 session, so 2026-11-09 is not the close before 2026-11-11/);
  assert.equal(gap.latest?.day, "2026-11-09");
  assert.equal(_dropped, 749.93);
  // the cash fund's own gap is a note, never a reason to void the core
  const noBoxx = floorTracking(live({ bars: { base: bars(SPY, NOW), levered: bars({}, NOW), cash: null } }));
  assert.ok(byDay(noBoxx, "2026-11-09").tracking_pp !== null);
  assert.equal(byDay(noBoxx, "2026-11-09").cash_fund_pnl, null);
  assert.equal(byDay(noBoxx, "2026-11-09").reason, "cash fund: BOXX daily closes are unreadable");
});

test("unavailable and not applicable say why, with every block present and every number null", () => {
  const cases: Array<[Partial<FloorTrackingInputs>, FloorTrackingReport["status"], RegExp]> = [
    [{ cfg: null, errors: { config: "Python mutex timeout" } }, "unavailable", /^the floor's config is unreadable \(Python mutex timeout\)$/],
    [{ cfg: { ...CFG, mode: "basket" } }, "not_applicable", /^FLOOR_MODE is "basket": this report is defined for "vol_managed_spy" only$/],
    [{ lines: [] }, "unavailable", /^no daily line is stored yet/],
    [{ lines: [], errors: { lines: "the daily line file is unreadable (bad json)" } }, "unavailable", /^the daily lines are unreadable/],
    [{ bars: { base: null, levered: null, cash: null }, errors: { base_bars: "alpaca bars SPY -> 500" } }, "unavailable", /^SPY daily closes are unreadable \(alpaca bars SPY -> 500\)$/],
    [{ bars: { base: bars({ "2026-11-06": 735.4, "2026-11-09": 752.72 }, NOW), levered: null, cash: null } }, "unavailable", /^the SPY closes read do not reach back before the first stored line \(2026-11-05\)$/],
    [{ positions: null }, "unavailable", /^positions are unreadable/],
    [{ orders: { message: "forbidden" } }, "unavailable", /^the floor's orders are unreadable/],
    [{ orders: [...LIVE_ORDERS, order("x", "SPY", "buy", 1, 700, null, { submitted_at: null })] }, "unavailable", /^1 filled floor order\(s\) carry no readable time, side or price/],
  ];
  for (const [over, status, why] of cases) {
    const r = floorTracking(live(over));
    assert.equal(r.status, status, String(why));
    assert.match(r.reason ?? "", why);
    assert.deepEqual([r.sessions, r.latest, r.entry, r.stored.tracking_pp, r.rolling.tracking_pp], [[], null, null, null, null]);
    assert.equal(r.rollback_trigger.status, "undetermined");
    assert.equal(r.rollback_trigger.tracking.status, "not_evaluable");
    assert.equal(r.rollback_trigger.text, FLOOR_TRACKING_TRIGGER_TEXT);
    assert.equal(r.message, `floor tracking ${status === "not_applicable" ? "not applicable" : "unavailable"}: ${r.reason}`.slice(0, 490));
  }
  // a floor that never held or traded its core in the stored sessions: ok, empty, and it says so
  const idle = floorTracking(live({ positions: held({}), orders: [] }));
  assert.equal(idle.status, "ok");
  assert.equal(idle.reason, "the floor held no SPY or SSO and filled no order in the stored sessions (2026-11-05 to 2026-11-11)");
  assert.equal(idle.inputs.sessions_before_floor, 5);
  assert.match(idle.message, /^no session measured yet \(the floor held no SPY or SSO/);
});

test("floorOrders: fills by ticker, a part-filled cancelled order placed at its submission, an unreadable fill counted", () => {
  const book = floorOrders([
    order("a", "spy", "buy", 10, 700, "2026-11-06T15:00:00Z"),
    order("b", "SSO", "sell_short", 3, 95, "2026-11-06T14:00:00Z"),
    order("c", "SPY", "sell", 4, 701, null, { status: "canceled", submitted_at: "2026-11-06T16:00:00Z" }),
    order("d", "SPY", "buy", 5, null, null, { submitted_at: "2026-11-06T16:30:00Z" }),
    order("e", "QQQ", "buy", 9, 600, "2026-11-06T15:00:00Z"),
    order("f", "SPY", "hold", 1, 700, "2026-11-06T15:00:00Z"),
    null, "junk",
  ], new Set(["SPY", "SSO"]));
  assert.ok(book);
  assert.deepEqual(book.fills.map((f) => [f.order_id, f.symbol, f.dir, f.qty, f.price, f.time_basis]), [
    ["b", "SSO", -1, 3, 95, "filled_at"], ["a", "SPY", 1, 10, 700, "filled_at"], ["c", "SPY", -1, 4, 701, "submitted_at"],
  ]);
  assert.equal(book.submitted.length, 5, "every order sent in the floor's tickers, filled or not");
  assert.equal(book.unplaced, 1, "a filled order with no usable side");
  assert.equal(floorOrders({ message: "forbidden" }, new Set(["SPY"])), null);
  assert.equal(floorOrders(null, new Set(["SPY"])), null);
});

test("parseFloorTrackingConfig: system_config's values as they are, never a default", () => {
  const raw = { FLOOR_MODE: "vol_managed_spy", FLOOR_VM_TICKER: "spy", FLOOR_VM_LEVERED_TICKER: "SSO", FLOOR_VM_CASH_TICKER: " boxx ", FLOOR_VM_CAPITAL_PCT: 0.85, FLOOR_VM_BAND: 0.1, FLOOR_TICKER: "QQQ" };
  assert.deepEqual(parseFloorTrackingConfig(raw), { cfg: { mode: "vol_managed_spy", base: "SPY", levered: "SSO", cash: "BOXX", capitalPct: 0.85, band: 0.1 }, error: null });
  assert.equal(parseFloorTrackingConfig({ ...raw, FLOOR_VM_CASH_TICKER: "SPY" }).cfg?.cash, null, "a fund that is a core ticker is not a cash fund");
  assert.equal(parseFloorTrackingConfig({ ...raw, FLOOR_VM_CASH_TICKER: undefined }).cfg?.cash, null);
  assert.equal(parseFloorTrackingConfig({ ...raw, FLOOR_VM_BAND: undefined }).cfg?.band, null);
  assert.equal(parseFloorTrackingConfig({ ...raw, FLOOR_MODE: "basket" }).cfg?.mode, "basket");
  for (const [bad, why] of [
    [{ ...raw, FLOOR_MODE: undefined }, /no FLOOR_MODE/], [{ ...raw, FLOOR_VM_TICKER: "" }, /no FLOOR_VM_TICKER/],
    [{ ...raw, FLOOR_VM_LEVERED_TICKER: "SPY" }, /are both SPY/], [{ ...raw, FLOOR_VM_CAPITAL_PCT: undefined }, /no FLOOR_VM_CAPITAL_PCT in \(0, 1\]/],
    [{ ...raw, FLOOR_VM_CAPITAL_PCT: 0 }, /FLOOR_VM_CAPITAL_PCT/], [{ ...raw, FLOOR_VM_CAPITAL_PCT: 1.2 }, /FLOOR_VM_CAPITAL_PCT/],
    [{ error: "No module named system_config" }, /No module named system_config/], [null, /no FLOOR_MODE/], ["x", /no FLOOR_MODE/],
  ] as Array<[unknown, RegExp]>) {
    const p = parseFloorTrackingConfig(bad);
    assert.equal(p.cfg, null);
    assert.match(p.error ?? "", why);
  }
});

test("the config read bot.ts runs returns what Python's own system_config holds", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const m = /const FLOOR_TRACKING_CFG_CMD = `python3 -c "([^`]+)"`;/.exec(bot);
  assert.ok(m, "FLOOR_TRACKING_CFG_CMD not found in bot.ts");
  const py = (code: string) => execFileSync("python3", ["-c", code], { cwd: repo, encoding: "utf8" }).trim().split("\n").pop() ?? "";
  const parsed = parseFloorTrackingConfig(JSON.parse(py(m[1])));
  assert.equal(parsed.error, null);
  const own = JSON.parse(py("import json; from system_config import BASE_CONFIG as c; print(json.dumps([c['FLOOR_MODE'], c['FLOOR_VM_TICKER'], c['FLOOR_VM_LEVERED_TICKER'], c['FLOOR_VM_CASH_TICKER'], c['FLOOR_VM_CAPITAL_PCT'], c['FLOOR_VM_BAND']]))"));
  assert.deepEqual([parsed.cfg?.mode, parsed.cfg?.base, parsed.cfg?.levered, parsed.cfg?.cash, parsed.cfg?.capitalPct, parsed.cfg?.band], own);
  // the mode this report is defined for is the engine's own name for the volatility-managed floor (read from the
  // engine, not from the configured value: the rollback to "basket" must not need this test changed)
  assert.ok(fs.readFileSync(path.join(repo, "bot_engine.py"), "utf8").includes(`BASE_CONFIG.get("FLOOR_MODE", "basket") == "${FLOOR_TRACKING_MODE}"`));
});

test("the trigger is the study's own sentence, its two numbers are the code's, and the study's pace is the results file's", () => {
  const study = fs.readFileSync(path.join(repo, "research", "floor_study_2026-11-05.md"), "utf8");
  assert.ok(study.includes(FLOOR_TRACKING_TRIGGER_TEXT), "FLOOR_TRACKING_TRIGGER_TEXT is not in research/floor_study_2026-11-05.md word for word");
  const m = /> (\d+(?:\.\d+)?) pp over any (\d+) sessions/.exec(FLOOR_TRACKING_TRIGGER_TEXT);
  assert.ok(m);
  assert.deepEqual([Number(m[1]), Number(m[2])], [FLOOR_TRACKING_LIMIT_PP, FLOOR_TRACKING_WINDOW_SESSIONS]);
  const results = JSON.parse(fs.readFileSync(path.join(repo, "research", "floor_study_2026-11-05.json"), "utf8"));
  assert.deepEqual([results.exposure.V1_rebalances, results.window.sessions], [FLOOR_TRACKING_STUDY_REBALANCES, FLOOR_TRACKING_STUDY_SESSIONS]);
  // the study's own results file holds no drawdown percentile and stays as published; the second leg's limit was
  // computed from the study's code on 2026-11-22 and lives in its own file, which the leg's two numbers are read from
  assert.doesNotMatch(JSON.stringify(results), /percentile|p95|drawdown_3m|3m_dd/i);
  const limit = JSON.parse(fs.readFileSync(path.join(repo, "research", "floor_study_drawdown_limit_2026-11-22.json"), "utf8"));
  assert.deepEqual([limit.V1_volmanaged.p95_pct, limit.V1_volmanaged.window_sessions], [FLOOR_DRAWDOWN_LIMIT_PCT, FLOOR_DRAWDOWN_WINDOW_SESSIONS]);
  assert.equal(limit.reproduction.reproduces_study, true);
  assert.deepEqual(limit.window, results.window);
  assert.ok(FLOOR_DRAWDOWN_LIMIT_SOURCE.startsWith("research/floor_study_drawdown_limit_2026-11-22.json"));
  assert.match(FLOOR_TRACKING_TRIGGER_TEXT, /95th-percentile 3-month drawdown$/);
  assert.ok(DAILY_LINE_KEEP >= FLOOR_DRAWDOWN_WINDOW_SESSIONS + 1, "a 63-session window needs 64 stored lines: each line carries the close before its session");
  const vm = fs.readFileSync(path.join(repo, "floor_vol_managed.py"), "utf8");
  assert.equal(Number(/^LEVERAGE_OF_LEVERED = ([\d.]+)/m.exec(vm)?.[1]), FLOOR_TRACKING_LEVERED_MULTIPLE);
});

test("the report passes the diag sanitizer untouched, and the module stays pure", () => {
  for (const r of [floorTracking(live()), floorTracking(rebalance()), floorTracking(run(60, 1.0).inputs)]) {
    assert.deepEqual(sanitizeDiag(r), JSON.parse(JSON.stringify(r)), "nothing redacted, truncated or cut by depth");
    assert.ok(r.message.length <= 490);
    assert.equal(r.message, floorTrackingMessage(r));
    assert.equal(r.definitions, FLOOR_TRACKING_DEFINITIONS);
  }
  for (const k of ["capital", "target_exposure", "held_exposure", "spy_pct", "rule_pct", "floor_pct", "tracking_pp", "exposure_pp", "other_pp", "cash_fund", "window", "entry", "turnover", "account_drawdown"]) {
    assert.ok(FLOOR_TRACKING_DEFINITIONS[k]?.length > 20, `no definition for ${k}`);
  }
  assert.equal(floorTracking(run(60, 1.0).inputs).sessions.length, 60, "the stored history at its longest fits the sanitizer's 200-row cap");
  const src = fs.readFileSync(path.join(here, "floorTracking.ts"), "utf8");
  for (const banned of ["node:fs", "from \"fs\"", "fetch(", "alpaca(", "process.env", "/v2/", "Date.now()"]) {
    assert.ok(!src.includes(banned), `floorTracking.ts must stay pure (found ${banned})`);
  }
  assert.match(floorTracking(live()).message, /^2026-11-11 floor \+1\.05% vs rule \+0\.96% \(0\.73x of SPY \+1\.32%\): tracking \+0\.09pp \| last 3 of 20 sessions \+0\.03pp \| entry 11-06 -0\.72pp \(in no window\) \| rollback trigger undetermined: tracking: 3 measured sessions, a window needs 20 in a row; [+-]0\.\d\dpp so far; drawdown -0\.81% in 4 of 63 sessions \(limit -14\.64%; 1 unread\) \| turnover: 0 rebalance sessions in 3 \(study pace 1\.9 per 20\), \$0 core traded$/);
  assert.equal(FLOOR_TRACKING_FILLS_LISTED, 20);
});

// ── the trigger's second leg: the account's drawdown against the backtest's 95th-percentile 3-month drawdown ──
// A flat SPY run (tracking 0 every session) whose account equity is set here: closes[k] is the account's close of
// days[k]; line k carries closes[k-1] as its last_equity, and the newest line reads its own close.
function equityRun(closes: number[], o: { noLastClose?: boolean; dropPrior?: number } = {}): FloorTrackingInputs {
  const n = closes.length - 1;
  const x = run(n, 0.8, { flat: true });
  const lines = x.inputs.lines.map((l, i) => {
    const k = i + 1;                                     // lines[i] is the session of days[k]
    const day: Record<string, unknown> = { last_equity: o.dropPrior === k ? null : closes[k - 1] };
    if (k === n && !o.noLastClose) { day.equity = closes[k] + 37; day.equity_end = closes[k]; }
    return { ...l, account_vs_spy: { day } } as unknown as DailyLine;
  });
  return { ...x.inputs, lines };
}

test("largestFall: the largest fall from a high to a LATER low, with the two closes it runs between", () => {
  const pts = [100, 110, 99, 105, 120, 108].map((equity, k) => ({ day: `d${k}`, equity }));
  const f = largestFall(pts);
  assert.ok(f);
  near(f.pct, (99 / 110 - 1) * 100, 1e-9);               // -10% from 110, not the later -10% from 120 (the first is kept on a tie)
  assert.deepEqual([f.peak.day, f.trough.day], ["d1", "d2"]);
  assert.equal(largestFall([{ day: "a", equity: 1 }, { day: "b", equity: 2 }])?.pct, 0, "never falls");
  near(largestFall([{ day: "a", equity: 100 }, { day: "b", equity: 50 }, { day: "c", equity: 200 }, { day: "d", equity: 150 }])?.pct ?? null, -50, 1e-9);
  assert.equal(largestFall([{ day: "a", equity: 1 }]), null);
});

test("drawdown leg: a partial window inside the limit is 'not enough history', with the fall, its two closes and the newest line's own close reading", () => {
  // 100,000 -> 104,000 -> 95,680 (-8.00% from the high) -> 97,000
  const r = floorTracking(equityRun([100_000, 104_000, 99_000, 95_680, 97_000]));
  const d = r.rollback_trigger.drawdown;
  assert.deepEqual([d.status, d.sessions_measured, d.drawdown_pct, d.beyond_limit, d.missing], ["not_enough_history", 4, -8, false, []]);
  assert.deepEqual([d.peak?.equity, d.trough?.equity], [104_000, 95_680]);
  assert.equal(d.reason, "4 sessions since the floor entered, a window needs 63");
  assert.equal(d.limit_source, FLOOR_DRAWDOWN_LIMIT_SOURCE);
  assert.equal(r.rollback_trigger.status, "undetermined");
  assert.match(r.rollback_trigger.summary, /; drawdown -8\.00% in 4 of 63 sessions \(limit -14\.64%\)$/);
  // the newest session's close is its line's close reading (equity_end), not the after-hours read beside it
  const last = floorTracking(equityRun([100_000, 104_000, 99_000, 95_680, 80_000])).rollback_trigger.drawdown;
  assert.deepEqual([last.trough?.equity, last.drawdown_pct], [80_000, r2((80_000 / 104_000 - 1) * 100)]);
});

test("drawdown leg MET: a fall past the limit trips the trigger at once, on a partial window, whatever the tracking leg says", () => {
  // the limit is strict: -14.64% exactly is not beyond it
  const at = floorTracking(equityRun([100_000, 85_360, 85_360])).rollback_trigger;
  assert.deepEqual([at.drawdown.drawdown_pct, at.drawdown.status, at.status], [-14.64, "not_enough_history", "undetermined"]);
  const r = floorTracking(equityRun([100_000, 101_000, 90_000, 86_000, 88_000]));
  const t = r.rollback_trigger;
  assert.equal(t.drawdown.drawdown_pct, r2((86_000 / 101_000 - 1) * 100));
  assert.ok((t.drawdown.drawdown_pct ?? 0) < FLOOR_DRAWDOWN_LIMIT_PCT);
  assert.deepEqual([t.drawdown.status, t.drawdown.beyond_limit, t.drawdown.reason], ["met", true, null]);
  assert.equal(t.tracking.status, "not_enough_history");
  assert.equal(t.status, "met", "either leg met is the trigger met");
  assert.match(t.summary, /; account drawdown -14\.85% BEYOND -14\.64% \(\d{4}-\d\d-\d\d to \d{4}-\d\d-\d\d\)$/);
  assert.match(r.message, /^ROLLBACK TRIGGER MET: tracking: /);
  assert.ok(r.message.length <= 490);
});

test("drawdown leg NOT MET needs a full 63-session window with every close on record; then both legs inside their limits clear the trigger", () => {
  const closes = (n: number) => Array.from({ length: n + 1 }, (_, k) => 100_000 - (k % 7 === 3 ? 4_000 : 0));   // dips of 4%, never more
  const short = floorTracking(equityRun(closes(62))).rollback_trigger;
  assert.deepEqual([short.drawdown.status, short.drawdown.sessions_measured, short.status], ["not_enough_history", 62, "undetermined"]);
  const full = floorTracking(equityRun(closes(63)));
  const t = full.rollback_trigger;
  assert.deepEqual([t.drawdown.status, t.drawdown.sessions_measured, t.drawdown.drawdown_pct, t.drawdown.reason], ["not_met", 63, -4, null]);
  assert.equal(t.tracking.status, "not_met");
  assert.equal(t.status, "not_met");
  assert.match(t.summary, /; drawdown -4\.00% within -14\.64% over 63 sessions$/);
  assert.match(full.message, /rollback trigger not_met: /);
  // one close on no line: a fall through it would not be seen, so the leg does not say "not met"
  const gap = floorTracking(equityRun(closes(63), { dropPrior: 30 })).rollback_trigger;
  assert.equal(gap.drawdown.status, "not_enough_history");
  assert.equal(gap.drawdown.missing.length, 1);
  assert.equal(gap.status, "undetermined");
  // the newest line with no close reading: the same
  assert.equal(floorTracking(equityRun(closes(63), { noLastClose: true })).rollback_trigger.drawdown.status, "not_enough_history");
});

test("drawdown leg: only the newest 63 sessions count, and a store of DAILY_LINE_KEEP lines is enough for a full window", () => {
  // a -30% fall that ended more than 63 sessions ago is outside the window
  const old = [100_000, 70_000, ...Array.from({ length: 63 }, () => 70_000)];
  const t = floorTracking(equityRun(old)).rollback_trigger.drawdown;
  assert.deepEqual([t.sessions_measured, t.drawdown_pct, t.status], [63, 0, "not_met"]);
  assert.equal(floorTracking(equityRun(Array.from({ length: DAILY_LINE_KEEP + 1 }, () => 100_000))).rollback_trigger.drawdown.status, "not_met");
});

test("drawdown leg: with nothing measured it carries its limit and says why it is not evaluated", () => {
  const d = floorTracking(live({ lines: [] })).rollback_trigger.drawdown;
  assert.deepEqual([d.status, d.limit_pct, d.drawdown_pct, d.sessions_measured], ["not_evaluable", FLOOR_DRAWDOWN_LIMIT_PCT, null, 0]);
  assert.ok(d.reason && d.reason.length > 10);
});
