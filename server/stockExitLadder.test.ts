// shared/stockExitLadder.ts against the code that sells stock (KNOWN BROKEN
// #50a, 2026-11-12). The Book's stock levels must be the ladder the engine
// executes, so the mirror is checked three ways:
//
//   1. NUMBERS. Every constant equals live_exit_policy.py's (the stock exit
//      policy of record, read by running Python), and the server/bot.ts
//      expression each one copies still carries that number.
//   2. BEHAVIOUR. The level-building block of syncMonitoredPositions and the
//      decision blocks of checkPositionOnTick are cut out of server/bot.ts and
//      RUN as written (the house pattern: posKillMarkGuard.test.ts,
//      taxAware.test.ts). Over 4,000 seeded stop states the ladder holds the
//      engine's own state, and every price it reports is the price at which
//      the engine's own test flips.
//   3. SCOPE. POS-KILL's stock threshold is the backstop; floor tickers are
//      never laddered; a scale-out ends the tick before the exit tests.
//
// bot.ts is not edited to import the shared numbers: its literals are pinned
// verbatim by test_backtest_v2_exit_model.py and its order paths are frozen.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import {
  stockLadder, regimeClass, phaseFromPeakR, stopWidthPct, takeProfitPct, ladderDaysHeld, pricePct,
  RISK_ATR_MULT, P1_STOP_ATR_MULT, P1_STOP_MIN_PCT, P1_STOP_MAX_PCT, PHASE2_R, PHASE3_R, PHASE4_R, SCALE1_R, SCALE2_R, TRAIL4_ATR_MULT,
  P2_TRAIL_MIN_PCT, P2_TRAIL_ATR_MULT, P3_TRAIL_MIN_PCT, P3_TRAIL_ATR_MULT, TP_MIN_PCT, TP_ATR_MULT, TP_MAX_PCT,
  TIME_STOP_DAYS, TIME_STOP_BAND_PCT, ATR_DEFAULT_PCT, STOCK_KILL_LOSS_PCT,
  type LadderRow, type Side, type StockLadder,
} from "../shared/stockExitLadder";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..");
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const policy = fs.readFileSync(path.join(repo, "live_exit_policy.py"), "utf8");
/** a number the way bot.ts writes it: one decimal ("8.0"), a whole count as is ("7") */
const lit = (n: number, decimals = 1): string => n.toFixed(decimals);

// ── 1. NUMBERS ───────────────────────────────────────────────────────────────

test("every ladder number equals live_exit_policy.py's, read by running the policy module", () => {
  const names = ["RISK_ATR_MULT", "P1_STOP_ATR_MULT", "P1_STOP_MIN_PCT", "P1_STOP_MAX_PCT", "SCALE1_R", "SCALE2_R", "TRAIL4_ATR_MULT",
    "P2_TRAIL_MIN_PCT", "P2_TRAIL_ATR_MULT", "P3_TRAIL_MIN_PCT", "P3_TRAIL_ATR_MULT", "TP_MIN_PCT", "TP_ATR_MULT", "TP_MAX_PCT",
    "TIME_STOP_DAYS", "TIME_STOP_BAND_PCT", "ATR_DEFAULT_PCT"];
  const script = [
    "import json, live_exit_policy as l",
    `names = ${JSON.stringify(names)}`,
    "labels = ['BULL', 'NEUTRAL_BULL', 'NEUTRAL', 'CAUTION', 'BEAR', 'PANIC', 'bull', '']",
    "vol = []",
    "for atr in (0.4, 0.75, 1.5, 2.5, 4.0, 4.5, 7.0):",
    "    for cls in ('bullish', 'bearish', 'neutral'):",
    "        p = l.LivePosition(100.0, atr, cls, 30)",
    "        vol.append({'atr': atr, 'cls': cls, 'risk': p.risk, 'p1_stop': p.p1_stop, 'tp': p.tp, 'scale2_r': p.scale2_r, 'trail4': p.trail4, 'third': p.third_qty})",
    "print(json.dumps({'consts': {n: getattr(l, n) for n in names}, 'regime': {r: l.regime_class(r) for r in labels}, 'none': l.regime_class(None), 'vol': vol}))",
  ].join("\n");
  const py = JSON.parse(execFileSync("python3", ["-c", script], { cwd: repo }).toString()) as {
    consts: Record<string, number | Record<string, number>>; regime: Record<string, string>; none: string;
    vol: Array<{ atr: number; cls: "bullish" | "bearish" | "neutral"; risk: number; p1_stop: number; tp: number; scale2_r: number; trail4: number; third: number }>;
  };
  const mine: Record<string, number | Record<string, number>> = {
    RISK_ATR_MULT, P1_STOP_ATR_MULT, P1_STOP_MIN_PCT, P1_STOP_MAX_PCT, SCALE1_R, SCALE2_R, TRAIL4_ATR_MULT,
    P2_TRAIL_MIN_PCT, P2_TRAIL_ATR_MULT, P3_TRAIL_MIN_PCT, P3_TRAIL_ATR_MULT, TP_MIN_PCT, TP_ATR_MULT, TP_MAX_PCT,
    TIME_STOP_DAYS, TIME_STOP_BAND_PCT, ATR_DEFAULT_PCT,
  };
  assert.deepEqual(Object.keys(mine).sort(), [...names].sort(), "every policy constant has its twin here");
  for (const n of names) assert.deepEqual(mine[n], py.consts[n], `${n}: shared/stockExitLadder.ts ${JSON.stringify(mine[n])} vs live_exit_policy.py ${JSON.stringify(py.consts[n])}`);
  // the regime classes, label by label (exact labels: a lowercase or empty label is neutral in both)
  for (const [label, cls] of Object.entries(py.regime)) assert.equal(regimeClass(label), cls, `regime ${JSON.stringify(label)}`);
  assert.equal(regimeClass(null), py.none);
  assert.equal(regimeClass(undefined), py.none);
  // the policy's own position object, for a spread of ATRs and regimes: 1R, the phase-1 stop, the target, the thirds
  assert.equal(py.vol.length, 21);
  for (const v of py.vol) {
    assert.equal(v.atr * RISK_ATR_MULT, v.risk, `1R at ATR ${v.atr}`);
    assert.equal(stopWidthPct(1, v.atr, v.cls), v.p1_stop, `phase-1 stop at ATR ${v.atr}`);
    assert.equal(takeProfitPct(v.atr), v.tp, `target at ATR ${v.atr}`);
    assert.equal(SCALE2_R[v.cls], v.scale2_r);
    assert.equal(TRAIL4_ATR_MULT[v.cls], v.trail4);
    assert.equal(stockLadder({ row: { current_stop_pct: v.atr }, side: "long", entry: 100, current: 100, qty: 30, pnlPct: 0, nowMs: 0 }).thirdQty, v.third);
  }
  // the phase thresholds and the trails are inline in the policy's tick()
  assert.ok(policy.includes(`if self.peak_pnl >= ${lit(PHASE4_R)} * self.risk - _TOL and self.phase < 4:`), "phase 4 at 3R");
  assert.ok(policy.includes(`elif self.peak_pnl >= ${lit(PHASE3_R)} * self.risk - _TOL and self.phase < 3:`), "phase 3 at 2R");
  assert.ok(policy.includes(`elif self.peak_pnl >= ${lit(PHASE2_R)} * self.risk - _TOL and self.phase < 2:`), "phase 2 at 1R");
  assert.ok(policy.includes("return self.remaining_qty > 1 and self.third_qty >= 1"), "the policy's own scale-out gate");
  assert.ok(policy.includes("if reason is None and pnl >= self.tp - _TOL and self.phase < 3 and self.scales >= 2:"), "the policy's own target gate");
});

test("each number is still the one server/bot.ts executes (the expression it copies carries it)", () => {
  const has = (literal: string) => assert.ok(bot.includes(literal), `server/bot.ts no longer contains ${JSON.stringify(literal)} — the ladder the engine executes changed; change shared/stockExitLadder.ts (and live_exit_policy.py) with it`);
  // syncMonitoredPositions: the levels the tick path is fed
  has(`const atrPct = ps.current_stop_pct ? ps.current_stop_pct : ${lit(ATR_DEFAULT_PCT)};`);
  has(`const initialRiskPct = ps.initial_risk_pct || atrPct * ${lit(RISK_ATR_MULT)};`);
  has("const highestPnl = Math.max(ps.highest_pnl || 0, pnlPct);");
  has(`const phase = ps.phase || (peakR >= ${PHASE4_R} ? 4 : peakR >= ${PHASE3_R} ? 3 : peakR >= ${PHASE2_R} ? 2 : 1);`);
  has(`else stopPct = Math.max(${lit(P1_STOP_MIN_PCT)}, Math.min(atrPct * ${lit(P1_STOP_ATR_MULT)}, ${lit(P1_STOP_MAX_PCT)}));`);
  has(`const tpPct = phase < 3 ? Math.max(${lit(TP_MIN_PCT)}, Math.min(atrPct * ${lit(TP_ATR_MULT)}, ${lit(TP_MAX_PCT)})) : 999;`);
  has("const highestPrice = Math.max(ps.highest_price || current, current);");
  has("const preScaleOut = scalesCompleted === 0;");
  has('entryDate: ps.entry_date || new Date().toISOString().split("T")[0],');
  has('const qty = Math.abs(parseInt(pos.qty || "0"));');
  // checkPositionOnTick: the tests each tick runs
  has(`if (peakR >= ${PHASE4_R} && pos.phase < 4) pos.phase = 4;`);
  has(`else if (peakR >= ${PHASE3_R} && pos.phase < 3) pos.phase = 3;`);
  has(`else if (peakR >= ${PHASE2_R} && pos.phase < 2) pos.phase = 2;`);
  has('const isBullish = regime === "BULL" || regime === "NEUTRAL_BULL";');
  has('const isBearish = regime === "BEAR" || regime === "PANIC" || regime === "CAUTION";');
  has(`const scaleOut1R = ${lit(SCALE1_R)};`);
  has(`const scaleOut2R = isBullish ? ${lit(SCALE2_R.bullish)} : (isBearish ? ${lit(SCALE2_R.bearish)} : ${lit(SCALE2_R.neutral)});`);
  has(`const trailAtrMult = isBullish ? ${lit(TRAIL4_ATR_MULT.bullish)} : (isBearish ? ${lit(TRAIL4_ATR_MULT.bearish)} : ${lit(TRAIL4_ATR_MULT.neutral)});`);
  has(`const atr = pos.atrPct || ${lit(ATR_DEFAULT_PCT)};`);
  has("if (pos.scalesCompleted < 2 && pos.remainingQty > 1) {");
  has("const thirdQty = Math.floor(pos.originalQty / 3);");
  has(`stopPct = Math.max(${lit(P3_TRAIL_MIN_PCT)}, atr * ${lit(P3_TRAIL_ATR_MULT)});`);
  has(`stopPct = Math.max(${lit(P2_TRAIL_MIN_PCT)}, atr * ${lit(P2_TRAIL_ATR_MULT)});`);
  has("const drawdownFromHighPrice = ((pos.highestPrice - currentPrice) / pos.highestPrice) * 100;");
  has("const drawdownFromPeak = pos.highestPnl - pnlPct;");
  has("if (!shouldStop && breakevenFloor && pnlPct < 0) {");
  has("if (pnlPct <= -stopPct) {");
  has("if (!shouldStop && pnlPct >= pos.tpPct && pos.phase < 3 && pos.scalesCompleted >= 2) {");
  has("const daysHeld = Math.floor((Date.now() - new Date(pos.entryDate).getTime()) / 86400000);");
  has(`if (daysHeld >= ${TIME_STOP_DAYS} && Math.abs(pnlPct) < ${lit(TIME_STOP_BAND_PCT)}) {`);
  // POS-KILL: the stock backstop
  has(`const STOCK_KILL_LOSS_PCT = ${lit(STOCK_KILL_LOSS_PCT)};`);
});

// ── 2. BEHAVIOUR: the engine's own code, run as written ─────────────────────

const NOW = Date.parse("2026-11-12T15:00:00Z");
class FakeDate extends Date {
  constructor(v?: number | string | Date) { if (v === undefined) super(NOW); else super(v); }
  static now(): number { return NOW; }
}
function between(from: string, to: string, at: number): string {
  const a = bot.indexOf(from, at);
  const b = bot.indexOf(to, a + from.length);
  assert.ok(a >= at && b > a, `markers not found in server/bot.ts: ${from} … ${to}`);
  return bot.slice(a, b);
}
const syncAt = bot.indexOf("async function syncMonitoredPositions()");
const tickAt = bot.indexOf("async function checkPositionOnTick(");
assert.ok(syncAt > 0 && tickAt > syncAt, "syncMonitoredPositions / checkPositionOnTick not found");
/** the stop-state read and the monitored position it builds */
const BUILD = between("// Read stop state for this ticker", "// Preserve pendingExit flag from in-memory state", syncAt);
/** P&L, high-water marks, phase, regime thresholds and the scale-out test */
const TICK_STATE = between("const entry = pos.entryPrice;", "// ── TAX HOLD", tickAt);
/** the stop level, the stop / breakeven tests, the take-profit and the time stop */
const TICK_EXITS = between("// ── STOP LEVEL CALCULATION", "if (!shouldStop && (shouldTP || shouldTimeStop) && taxHold) {", tickAt);

interface EnginePos {
  ticker: string; side: Side; entryPrice: number; qty: number; phase: number; initialRiskPct: number; highestPnl: number; currentStopPct: number;
  tpPct: number; entryDate: string; lastCheckedPrice: number; originalQty: number; remainingQty: number; scalesCompleted: number;
  breakevenActive: boolean; regime: string; atrPct: number; highestPrice: number;
}
interface EngineTick { pnlPct: number; shouldScaleOut: boolean; scaleOutQty: number; shouldStop: boolean; stopType: string; stopReason: string; shouldTP: boolean; shouldTimeStop: boolean }
const engine = new Function("Date", `${transformSync(`
function engineBuild(stopState, ticker, side, entry, current, pnlPct, qty) {
  const monitoredPositions = {};
  const audit = () => undefined;
  ${BUILD}
  return monitoredPositions[ticker];
}
function engineTick(pos, currentPrice) {
  ${TICK_STATE}
  ${TICK_EXITS}
  return { pnlPct, shouldScaleOut, scaleOutQty, shouldStop, stopType, stopReason, shouldTP, shouldTimeStop };
}
`, { loader: "ts" }).code}\nreturn { engineBuild, engineTick };`)(FakeDate) as {
  engineBuild: (stopState: Record<string, LadderRow>, ticker: string, side: Side, entry: number, current: number, pnlPct: number, qty: number) => EnginePos;
  engineTick: (pos: EnginePos, currentPrice: number) => EngineTick | undefined;
};
/** the engine's tests at `price`, on a copy of the position (a tick moves its high-water marks and phase) */
const at = (pos: EnginePos, price: number): EngineTick => {
  const out = engine.engineTick({ ...pos }, price);
  assert.ok(out, `the tick did not run at ${price}`);
  return out;
};
/** the tick's own P&L arithmetic */
const pnlAt = (entry: number, price: number, side: Side): number => (side === "long" ? ((price - entry) / entry) * 100 : ((entry - price) / entry) * 100);

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Case { row: LadderRow | undefined; side: Side; entry: number; current: number; qty: number; pnl: number; label: string }
function makeCase(rand: () => number, k: number): Case {
  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)];
  const side: Side = rand() < 0.15 ? "short" : "long";
  const entry = pick([4.37, 20, 100, 741.32]);
  const qty = pick([1, 2, 3, 4, 30, 388]);
  const pnlTarget = pick([-30, -9, -8.01, -7.99, -4.5, -3.01, -1.2, -0.01, 0, 0.5, 1.9, 2.0, 3, 6, 9.5, 12, 20, 45]);
  const current = pricePct(entry, pnlTarget, side);
  const pnl = pnlAt(entry, current, side);
  if (rand() < 0.04) return { row: undefined, side, entry, current, qty, pnl, label: `#${k} no stop state` };
  const atr = pick([undefined, 0, 0.6, 0.9, 1.2, 2.5, 3.9, 4.0, 4.1, 7.0, 11]);
  const atrEff = atr || 2.0;
  const risk = pick([undefined, undefined, atrEff * 2, 5.0]);
  const riskEff = risk || atrEff * 2;
  const peakR = pick([undefined, 0, 0.5, 0.999, 1, 1.6, 2, 2.4, 3, 4.2]);
  const highest_pnl = peakR === undefined ? undefined : peakR * riskEff;
  const highest_price = highest_pnl === undefined || rand() < 0.2 ? undefined : pricePct(entry, highest_pnl, "long");
  const scales = pick([0, 0, 0, 1, 2]);
  const row: LadderRow = {
    current_stop_pct: atr, initial_risk_pct: risk, highest_pnl, highest_price,
    phase: pick([undefined, undefined, 1, 2, 3, 4]),
    regime: pick([undefined, "BULL", "NEUTRAL_BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC", "bull"]),
    scales_completed: scales,
    original_qty: scales ? pick([undefined, qty, qty + Math.max(1, Math.floor(qty / 2)), 3]) : pick([undefined, qty, 999]),
    remaining_qty: scales ? pick([undefined, qty, qty + 5, 1]) : pick([undefined, qty, 1]),
    breakeven_active: scales ? rand() < 0.8 : rand() < 0.1,
    entry_date: pick([undefined, "2026-11-12", "2026-11-09", "2026-11-06", "2026-11-05", "2026-11-03", "2026-10-01"]),
  };
  return { row, side, entry, current, qty, pnl, label: `#${k} ${side} ${JSON.stringify(row)} entry ${entry} now ${current} (${pnl.toFixed(3)}%) x${qty}` };
}
const EPS = 1e-7;
/** prices from -40 % to +400 % P&L (the engine ignores a price that is not positive) */
const sweep = (c: Case): number[] => {
  const out: number[] = [];
  for (let p = -40; p <= 400; p += p < 60 ? 2.37 : 23.7) { const price = pricePct(c.entry, p, c.side); if (price > 0) out.push(price); }
  return out;
};
/** a price a hair on the profit side (+1) or the loss side (−1) of `price` */
const nudge = (price: number, side: Side, toward: 1 | -1): number => price * (1 + (side === "long" ? toward : -toward) * EPS);

test("the ladder is the engine's own state and tests: 4,000 seeded stop states against syncMonitoredPositions + checkPositionOnTick as written", () => {
  const rand = mulberry32(50);
  const seen = { phase: new Set<number>(), stopFrom: new Set<string>(), stopHit: 0, scale1: 0, scale2: 0, noScale: 0, tpArmed: 0, tpNone: 0, timeStop: 0, shortP4: 0, noRow: 0 };
  for (let k = 0; k < 4000; k++) {
    const c = makeCase(rand, k);
    const pos = engine.engineBuild(c.row ? { T: c.row } : {}, "T", c.side, c.entry, c.current, c.pnl, c.qty);
    const now = engine.engineTick(pos, c.current); // moves pos's high-water marks and phase, as the next tick does
    assert.ok(now, c.label);
    const L: StockLadder = stockLadder({ row: c.row, side: c.side, entry: c.entry, current: c.current, qty: c.qty, pnlPct: c.pnl, nowMs: NOW });
    if (!c.row) seen.noRow++;

    // the state the engine holds
    assert.equal(L.atrPct, pos.atrPct, `atr ${c.label}`);
    assert.equal(L.riskPct, pos.initialRiskPct, `1R ${c.label}`);
    assert.equal(L.pnlPct, now.pnlPct, `P&L ${c.label}`);
    assert.equal(L.highestPnl, pos.highestPnl, `peak P&L ${c.label}`);
    assert.equal(L.highestPrice, pos.highestPrice, `high ${c.label}`);
    assert.equal(L.phase, pos.phase, `phase ${c.label}`);
    assert.equal(L.originalQty, pos.originalQty, `original qty ${c.label}`);
    assert.equal(L.remainingQty, pos.remainingQty, `remaining qty ${c.label}`);
    assert.equal(L.scalesCompleted, pos.scalesCompleted, `scale-outs ${c.label}`);
    assert.equal(L.breakevenActive, pos.breakevenActive, `breakeven ${c.label}`);
    seen.phase.add(L.phase);

    // the stop: met now exactly when the engine's test is, and its price is where that test flips
    const shortP4 = c.side === "short" && L.phase >= 4; // the engine's phase-4 test is written for a long (module header)
    if (shortP4) seen.shortP4++;
    else {
      assert.equal(L.stopHit, now.shouldStop, `stop met now ${c.label}`);
      if (L.stopHit) seen.stopHit++;
      seen.stopFrom.add(L.stopFrom);
      assert.ok(L.stopPrice !== null && L.stopPrice > 0, `stop price ${c.label}`);
      assert.equal(at(pos, nudge(L.stopPrice, c.side, -1)).shouldStop, true, `a hair past the stop sells ${c.label}`);
      assert.equal(at(pos, nudge(L.stopPrice, c.side, 1)).shouldStop, false, `a hair inside the stop holds ${c.label}`);
      const past = at(pos, nudge(L.stopPrice, c.side, -1));
      assert.equal(past.stopType, L.phase >= 2 ? "trailing_stop" : "stop_loss", `stop kind ${c.label}`);
      if (L.stopFrom === "breakeven") assert.match(past.stopReason, /^WS BREAKEVEN STOP/, c.label);
      if (L.stopFrom === "entry") assert.ok(past.stopReason.includes(`Phase 1 stop at -${L.stopPct.toFixed(1)}%`), `${past.stopReason} ${c.label}`);
    }

    // the scale-outs: the next third sells a hair above its price, not a hair below; a lot that cannot split never does
    const next = L.scalesCompleted === 0 ? L.scaleOuts[0] : L.scalesCompleted === 1 ? L.scaleOuts[1] : null;
    assert.equal(L.scaleOuts[0].done, pos.scalesCompleted >= 1, c.label);
    assert.equal(L.scaleOuts[1].done, pos.scalesCompleted >= 2, c.label);
    if (next && next.applicable) {
      const up = at(pos, nudge(next.price, c.side, 1));
      assert.equal(up.shouldScaleOut, true, `third ${next.n} sells at its price ${c.label}`);
      assert.equal(up.scaleOutQty, L.thirdQty, `third ${next.n} size ${c.label}`);
      assert.equal(at(pos, nudge(next.price, c.side, -1)).shouldScaleOut, false, `third ${next.n} waits below its price ${c.label}`);
      if (next.n === 1) seen.scale1++; else seen.scale2++;
    } else if (next) {
      for (const p of sweep(c)) assert.equal(at(pos, p).shouldScaleOut, false, `a lot that cannot split never scales out ${c.label}`);
      seen.noScale++;
    }
    assert.equal(now.shouldScaleOut, !!next && next.applicable && L.rMultiple !== null && L.rMultiple >= next.rMultiple, `third due now ${c.label}`);

    // the target: where it exists it fires a hair above its price once both thirds are out; where it does not, no price fires it
    if (L.takeProfitPrice !== null && L.takeProfitPct !== null) {
      assert.ok(L.phase < 3, c.label);
      const up = at(pos, nudge(L.takeProfitPrice, c.side, 1));
      if (!up.shouldStop) assert.equal(up.shouldTP, L.takeProfitArmed, `target at its price ${c.label}`);
      assert.equal(at(pos, nudge(L.takeProfitPrice, c.side, -1)).shouldTP, false, `no target below its price ${c.label}`);
      if (L.takeProfitArmed) seen.tpArmed++;
    } else {
      for (const p of sweep(c)) assert.equal(at(pos, p).shouldTP, false, `no target at ${p} ${c.label}`);
      // also with both thirds out: phase 3+ has none, and a target at or beyond +2R is never reached below phase 3
      const out2 = { ...pos, scalesCompleted: 2 };
      if (L.phase >= 3 || takeProfitPct(L.atrPct) >= PHASE3_R * L.riskPct) {
        for (const p of sweep(c)) assert.equal(at(out2, p).shouldTP, false, `no target even after both thirds at ${p} ${c.label}`);
      }
      seen.tpNone++;
    }

    // the time stop: armed from day 7 of the entry date, and it fires inside the band when nothing else does
    if (!now.shouldStop && !now.shouldTP) {
      assert.equal(now.shouldTimeStop, L.timeStopArmed && Math.abs(c.pnl) < TIME_STOP_BAND_PCT, `time stop ${c.label}`);
      if (now.shouldTimeStop) seen.timeStop++;
    }
    assert.equal(L.daysHeld, ladderDaysHeld(c.row?.entry_date, NOW), c.label);
  }
  // the sample reaches every branch it claims to check
  assert.deepEqual([...seen.phase].sort(), [1, 2, 3, 4]);
  assert.deepEqual([...seen.stopFrom].sort(), ["breakeven", "entry", "peak_pnl", "peak_price"]);
  for (const [what, n] of Object.entries({ stopHit: seen.stopHit, scale1: seen.scale1, scale2: seen.scale2, noScale: seen.noScale, tpArmed: seen.tpArmed, tpNone: seen.tpNone, timeStop: seen.timeStop, shortP4: seen.shortP4, noRow: seen.noRow })) {
    assert.ok(n >= 20, `the sample reaches ${what} only ${n} times`);
  }
});

// ── 3. SCOPE ────────────────────────────────────────────────────────────────

test("a scale-out ends the tick before the exit tests; floor tickers and options never reach the ladder", () => {
  const tick = bot.slice(tickAt, bot.indexOf("function startStreaming(", tickAt));
  const scaleReturn = tick.indexOf("return; // Don't evaluate full exit on the same tick as a scale-out");
  assert.ok(scaleReturn > tick.indexOf("// ── TAX HOLD") && scaleReturn < tick.indexOf("// ── STOP LEVEL CALCULATION"), "the third is sold first; the stop is tested on the next tick");
  const guard = tick.slice(0, tick.indexOf("const entry = pos.entryPrice;"));
  assert.ok(guard.includes("if (pos.positionType && pos.positionType !== 'stock') {"), "option positions leave the tick before the ladder");
  // the sync skips floor / leg tickers before it builds a monitored position or judges POS-KILL
  const loopAt = bot.indexOf("for (const pos of positions) {", syncAt);
  const skipAt = bot.indexOf("if (FLOOR_AND_LEG_TICKERS.has(ticker)) continue;", loopAt);
  assert.ok(loopAt > syncAt && skipAt > loopAt && skipAt < bot.indexOf("const STOCK_KILL_LOSS_PCT", loopAt), "floor tickers: no ladder, no POS-KILL");
});

test("the backstop is POS-KILL's stock threshold: -25 % from entry, on the broker's own P&L, for either side", () => {
  assert.equal(STOCK_KILL_LOSS_PCT, -25);
  // the per-position threshold the kill branch compares against is the stock constant for a stock
  const sig = 'function posKillThresholdPct(isOption: boolean, side: "long" | "short", stockPct: number,';
  const fnAt = bot.indexOf(sig);
  assert.ok(fnAt > 0, "posKillThresholdPct not found");
  const fnSrc = bot.slice(fnAt, bot.indexOf("\n}\n", fnAt) + 3);
  const threshold = new Function(`${transformSync(fnSrc, { loader: "ts" }).code}\nreturn posKillThresholdPct;`)() as (o: boolean, s: Side, stock: number, optShort: number, optLong: number) => number;
  assert.equal(threshold(false, "long", STOCK_KILL_LOSS_PCT, -150, -75), STOCK_KILL_LOSS_PCT);
  assert.equal(threshold(false, "short", STOCK_KILL_LOSS_PCT, -150, -75), STOCK_KILL_LOSS_PCT);
  assert.ok(bot.includes("const POSITION_KILL_LOSS_PCT = posKillThresholdPct(isOptionPos, side, STOCK_KILL_LOSS_PCT,"));
  assert.ok(bot.includes("if (pnlPct <= POSITION_KILL_LOSS_PCT) {"));
  assert.ok(bot.includes('? posKillObserveBreach(posKillPendingBreach, ticker, pnlPct <= POSITION_KILL_LOSS_PCT, pnlPct, Date.now())'), "confirmed on a second valid reading");
  const long = stockLadder({ row: { current_stop_pct: 2 }, side: "long", entry: 80, current: 80, qty: 10, pnlPct: 0, nowMs: NOW });
  assert.equal(long.backstopPct, 25);
  assert.equal(long.backstopPrice, 60);
  const short = stockLadder({ row: { current_stop_pct: 2 }, side: "short", entry: 80, current: 80, qty: 10, pnlPct: 0, nowMs: NOW });
  assert.equal(short.backstopPrice, 100);
  // every phase: a position three R up still carries it
  const p4 = stockLadder({ row: { current_stop_pct: 2, highest_pnl: 14, highest_price: 91.2, scales_completed: 2, original_qty: 30, remaining_qty: 10, breakeven_active: true }, side: "long", entry: 80, current: 90, qty: 10, pnlPct: 12.5, nowMs: NOW });
  assert.equal(p4.phase, 4);
  assert.equal(p4.backstopPrice, 60);
});

// ── hand-worked ladders (every number below is arithmetic on the constants) ──

test("phase by phase, by hand: the stop, its basis and where it sits", () => {
  const base = { side: "long" as const, entry: 100, qty: 30, nowMs: NOW };
  // phase 1, ATR 2.5 %: stop 2 × 2.5 = 5 % under entry; 1R = 5 %
  const p1 = stockLadder({ ...base, row: { current_stop_pct: 2.5 }, current: 100.4, pnlPct: 0.4 });
  assert.deepEqual([p1.phase, p1.stopPct, p1.stopFrom, p1.stopPrice, p1.riskPct, p1.stopHit], [1, 5, "entry", 95, 5, false]);
  // the clamps: ATR 0.6 % → 1.5 % (not 1.2); ATR 7 % → 8 % (not 14, and not manage_positions' 10)
  assert.equal(stockLadder({ ...base, row: { current_stop_pct: 0.6 }, current: 100, pnlPct: 0 }).stopPct, 1.5);
  assert.equal(stockLadder({ ...base, row: { current_stop_pct: 7 }, current: 100, pnlPct: 0 }).stopPct, 8);
  assert.equal(stockLadder({ ...base, row: { current_stop_pct: 4 }, current: 100, pnlPct: 0 }).stopPct, 8);
  assert.equal(stockLadder({ ...base, row: { current_stop_pct: 3.9 }, current: 100, pnlPct: 0 }).stopPct, 7.8);
  // phase 2 (peak 6 % = 1.5R at 1R 4 %): trails max(1.5, 1.5 × 2) = 3 % under the peak P&L → +3 % → 103
  const p2 = stockLadder({ ...base, row: { current_stop_pct: 2, initial_risk_pct: 4, highest_pnl: 6, highest_price: 106 }, current: 104, pnlPct: 4 });
  assert.deepEqual([p2.phase, p2.stopPct, p2.stopFrom, p2.stopPrice, p2.stopHit], [2, 3, "peak_pnl", 103, false]);
  // phase 3 (peak 9 % = 2.25R): trails max(1, 2) = 2 % under the peak P&L → +7 % → 107
  const p3 = stockLadder({ ...base, row: { current_stop_pct: 2, initial_risk_pct: 4, highest_pnl: 9, highest_price: 109 }, current: 108, pnlPct: 8 });
  assert.deepEqual([p3.phase, p3.stopPct, p3.stopFrom, p3.stopPrice, p3.takeProfitPct], [3, 2, "peak_pnl", 107, null]);
  // phase 4 (peak 14 % = 3.5R), three regimes: trail = ATR × 3 / 2 / 1.5 under the highest PRICE (114)
  const row4 = { current_stop_pct: 2, initial_risk_pct: 4, highest_pnl: 14, highest_price: 114 };
  const bull = stockLadder({ ...base, row: { ...row4, regime: "BULL" }, current: 112, pnlPct: 12 });
  assert.deepEqual([bull.phase, bull.stopPct, bull.stopFrom], [4, 6, "peak_price"]);
  assert.ok(Math.abs((bull.stopPrice ?? 0) - 114 * 0.94) < 1e-9);
  assert.equal(stockLadder({ ...base, row: { ...row4, regime: "NEUTRAL" }, current: 112, pnlPct: 12 }).stopPct, 4);
  assert.equal(stockLadder({ ...base, row: { ...row4, regime: "PANIC" }, current: 112, pnlPct: 12 }).stopPct, 3);
  // no regime scalar anywhere: the same row under every label differs only in the phase-4 multiple
  for (const regime of ["BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC"]) {
    assert.equal(stockLadder({ ...base, row: { current_stop_pct: 2.5, regime }, current: 100, pnlPct: 0 }).stopPct, 5, `phase 1 in ${regime}`);
    assert.equal(stockLadder({ ...base, row: { current_stop_pct: 2, initial_risk_pct: 4, highest_pnl: 6, regime }, current: 104, pnlPct: 4 }).stopPct, 3, `phase 2 in ${regime}`);
    assert.equal(stockLadder({ ...base, row: { current_stop_pct: 2, initial_risk_pct: 4, highest_pnl: 9, regime }, current: 108, pnlPct: 8 }).stopPct, 2, `phase 3 in ${regime}`);
  }
  // breakeven (after the first third): a trail that would sit under entry is lifted to entry, in phase 2+ only
  const be = stockLadder({ ...base, row: { current_stop_pct: 3, initial_risk_pct: 4, highest_pnl: 4.1, scales_completed: 1, original_qty: 30, remaining_qty: 20, breakeven_active: true }, qty: 20, current: 102, pnlPct: 2 });
  assert.deepEqual([be.phase, be.stopPct, be.stopFrom, be.stopPrice], [2, 4.5, "breakeven", 100]);
  const beP1 = stockLadder({ ...base, row: { current_stop_pct: 2.5, breakeven_active: true }, current: 100.4, pnlPct: 0.4 });
  assert.deepEqual([beP1.phase, beP1.stopFrom, beP1.stopPrice], [1, "entry", 95]);
  // the stored phase is kept when it is ahead of the peak; the live price can move it up, never down
  assert.equal(stockLadder({ ...base, row: { current_stop_pct: 2, initial_risk_pct: 4, highest_pnl: 3.99, phase: 2 }, current: 103, pnlPct: 3 }).phase, 2);
  assert.equal(stockLadder({ ...base, row: { current_stop_pct: 2, initial_risk_pct: 4, highest_pnl: 0.9, phase: 1 }, current: 104.2, pnlPct: 4.2 }).phase, 2);
  assert.deepEqual([0.99, 1, 1.99, 2, 2.99, 3, 9].map(phaseFromPeakR), [1, 2, 2, 3, 3, 4, 4]);
});

test("by hand: thirds, target, time stop and the engine's defaults", () => {
  const base = { side: "long" as const, entry: 100, current: 100, pnlPct: 0, nowMs: NOW };
  // thirds: floor(lot / 3) shares at +1R and at the regime's second threshold; 1 or 2 shares never split
  const lot30 = stockLadder({ ...base, qty: 30, row: { current_stop_pct: 2.5, regime: "BEAR" } });
  assert.deepEqual(lot30.scaleOuts.map((s) => [s.n, s.rMultiple, s.pct, s.price, s.done, s.applicable]), [[1, 1, 5, 105, false, true], [2, 1.5, 7.5, 107.5, false, true]]);
  assert.equal(lot30.thirdQty, 10);
  assert.deepEqual(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 2.5, regime: "NEUTRAL_BULL" } }).scaleOuts.map((s) => [s.rMultiple, s.pct]), [[1, 5], [3, 15]]);
  for (const [qty, third, can] of [[1, 0, false], [2, 0, false], [3, 1, true], [4, 1, true], [5, 1, true], [6, 2, true]] as const) {
    const l = stockLadder({ ...base, qty, row: { current_stop_pct: 2.5 } });
    assert.deepEqual([l.thirdQty, l.scaleOuts[0].applicable, l.scaleOuts[1].applicable], [third, can, can], `lot of ${qty}`);
    // without both thirds the target never arms, so such a lot has none
    assert.equal(l.takeProfitPct, can ? 7.5 : null, `target for a lot of ${qty}`);
  }
  // before the first third the live share count IS the position (a scale-in tranche grows it); fractions are dropped
  const grown = stockLadder({ ...base, qty: 45.9, row: { current_stop_pct: 2.5, original_qty: 30, remaining_qty: 30, scales_completed: 0 } });
  assert.deepEqual([grown.originalQty, grown.remainingQty, grown.thirdQty], [45, 45, 15]);
  // after it, the recorded lot stands, and the shares left are capped at what the broker holds
  const after = stockLadder({ ...base, qty: 18, row: { current_stop_pct: 2.5, original_qty: 30, remaining_qty: 20, scales_completed: 1, highest_pnl: 5.2, breakeven_active: true } });
  assert.deepEqual([after.originalQty, after.remainingQty, after.thirdQty, after.scaleOuts[0].done, after.scaleOuts[1].applicable], [30, 18, 10, true, true]);
  const last = stockLadder({ ...base, qty: 1, row: { current_stop_pct: 2.5, original_qty: 3, remaining_qty: 1, scales_completed: 1, highest_pnl: 5.2, breakeven_active: true } });
  assert.deepEqual([last.scaleOuts[0].applicable, last.scaleOuts[1].applicable, last.takeProfitPct], [true, false, null], "one share left: no second third, no target");
  // target: max(4, min(3 × ATR, 15)); gone in phase 3+; none at or beyond +2R
  assert.deepEqual([1.5, 2.5, 6].map((atr) => stockLadder({ ...base, qty: 30, row: { current_stop_pct: atr } }).takeProfitPct), [4.5, 7.5, 15]);
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 0.8 } }).takeProfitPct, null, "ATR 0.8 %: the 4 % target is beyond +2R (3.2 %)");
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 1 } }).takeProfitPct, null, "ATR 1 %: the 4 % target IS +2R, where phase 3 begins");
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 1.01 } }).takeProfitPct, 4);
  const armed = stockLadder({ ...base, qty: 10, row: { current_stop_pct: 2.5, original_qty: 30, remaining_qty: 10, scales_completed: 2, highest_pnl: 8, phase: 2 }, current: 107, pnlPct: 7 });
  assert.deepEqual([armed.takeProfitArmed, armed.takeProfitPct, armed.takeProfitPrice], [true, 7.5, 107.5]);
  // time stop: whole days from the entry date; no date → the engine counts from today, so it cannot arm
  assert.deepEqual(["2026-11-12", "2026-11-06", "2026-11-05", "2026-11-03"].map((d) => ladderDaysHeld(d, NOW)), [0, 6, 7, 9]);
  assert.equal(ladderDaysHeld(undefined, NOW), null);
  assert.equal(ladderDaysHeld("not a date", NOW), null);
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 2.5, entry_date: "2026-11-05" } }).timeStopArmed, true);
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 2.5, entry_date: "2026-11-06" } }).timeStopArmed, false);
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 2.5 } }).timeStopArmed, false);
  // no stop state, or one without an ATR: the engine's defaults (ATR 2 %, 1R 4 %, neutral, phase from the live P&L)
  const none = stockLadder({ ...base, qty: 30, row: null });
  assert.deepEqual([none.atrPct, none.atrDefaulted, none.riskPct, none.stopPct, none.stopPrice, none.takeProfitPct, none.scaleOut2R], [2, true, 4, 4, 96, 6, 2]);
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 0 } }).atrDefaulted, true);
  assert.equal(stockLadder({ ...base, qty: 30, row: { current_stop_pct: 2.5 } }).atrDefaulted, false);
});
