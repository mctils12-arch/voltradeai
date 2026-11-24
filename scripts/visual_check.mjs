#!/usr/bin/env node
/**
 * visual_check.mjs — the DESIGN.md enforcement harness.
 *
 * Renders key pages headless at the three canonical widths (390 / 768 /
 * 1440), saves screenshots to .visual/, and runs mechanical layout checks.
 * Deterministic: serves the built client (dist/public) with ALL /api/*
 * routes mocked from fixtures — no live backend, no external data APIs
 * (map base tiles still load from the tile CDN; that's the documented
 * scoped exception and only affects pixels, not checks).
 *
 * Usage:
 *   npm run build              # produce dist/public first
 *   node scripts/visual_check.mjs [--soft] [--page data]
 *
 * --soft: report failures but exit 0 (baseline mode).
 * Per CLAUDE.md promotion rule 6, PRs touching client/ must include this
 * run and review the screenshots against DESIGN.md before opening.
 */
import { createServer } from "http";
import { readFileSync, existsSync, mkdirSync, writeFileSync, readdirSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import pngjs from "pngjs";
const { PNG } = pngjs;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DIST = path.join(ROOT, "dist", "public");
const OUT = path.join(ROOT, ".visual");
const SOFT = process.argv.includes("--soft");
// DEVICE ENVELOPE D1 (scale_program.md, human directive 2026-08-07):
// `--demand` runs the per-layer demand matrix instead of the visual
// battery — each layer measured in isolation at emulated CPU tiers,
// ledger written to .visual/demand_profile.json (diffable across
// versions). Measurement tool, not a gate: it never fails the build.
const DEMAND = process.argv.includes("--demand");

const WIDTHS = [
  { w: 390, h: 844, label: "phone", touch: true },
  { w: 768, h: 1024, label: "tablet", touch: true },
  { w: 1440, h: 900, label: "desktop", touch: false },
];
// A page entry may add widths of its own (`widths`): the cockpit pages
// (2026-10-29) also render at the human's 1920px monitor, where the ask was made.
const WIDE = [{ w: 1920, h: 1080, label: "wide", touch: false }];
// COCKPIT LAYOUT pages (2026-10-29) and what their strip and grid must show:
// tiles in the KPI strip, grid panels sharing a row at 1440 / 1920.
const COCKPIT_PAGES = new Map([
  // (innerText carries the eyebrow's CSS uppercase, hence /i)
  // cols = distinct panel left edges (a 2-wide + 1-wide row is 3 columns)
  // forecast (Part C, 2026-11-04): the DAILY view is the default; `horizon` presses the Weekly switch and
  // requires the weekly strip, then presses Daily and requires the daily strip back
  ["forecast", { tiles: 5, cols1440: 3, cols1920: 4, first: /^Calls for Nov 4 9 1 index · 3 positions · 5 candidates · 2 intraday$/i,
    horizon: { weekly: /^Calls this week 8 1 index · 3 positions · 4 candidates$/i } }],
  ["brain", { tiles: 6, cols1440: 2, cols1920: 3, first: /^All parts 11 /i, filter: { status: "shadow", cards: 8 } }],
  ["dossier", { tiles: 6, cols1440: 3, cols1920: 4, first: /^Price · 5 sessions \$27\.42\+5\.2% 5d last close \$27\.09/i }],
  ["dossierempty", { tiles: 0, cols1440: 0, cols1920: 0, empty: true }],
]);

// Pages under test: name -> app hash route
// Pages under test: map pages get the full map/perf/self-see battery;
// non-map pages get layout + interaction checks only.
/** the weekday before an ET date (YYYY-MM-DD): the fixtures' previous session (holidays aside — a fixture) */
function prevWeekday(day) {
  let d = new Date(`${day}T12:00:00Z`);
  do d = new Date(d.getTime() - 86_400_000); while (d.getUTCDay() === 0 || d.getUTCDay() === 6);
  return d.toISOString().slice(0, 10);
}
/** server/todayTile.ts's body from an equity, its prior close and SPY's move (null + a reason when SPY is missing) —
 *  during a session (phase "session", 2026-11-06 fields: the broker clock said open; the change runs to equity now) */
function todayFixture(equityNow, prevClose, spyPct, now, ago, spyReason) {
  const r2 = (x) => Math.round(x * 100) / 100;
  const day = new Date(now).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const pct = r2(((equityNow - prevClose) / prevClose) * 100);
  return { day, phase: "session", when: "during the session", phase_source: "broker clock", prev_day: prevWeekday(day),
    equity_now: equityNow, equity_prev_close: prevClose, prev_close_source: "last_equity", equity_end: equityNow, end_basis: "now",
    pnl_usd: r2(equityNow - prevClose), pnl_pct: pct, pnl_reason: null,
    spy_pct_today: spyPct, spy_asof: spyPct === null ? null : ago(4), spy_bar_date: spyPct === null ? null : day, spy_reason: spyReason, spy_final: false,
    excess_pct: spyPct === null ? null : r2(pct - spyPct), asof: new Date(now).toISOString(),
    source: "Alpaca paper account: equity now vs last_equity (the prior session's 16:00 ET close) · SPY: today's daily bar vs the prior close, price only, no dividends, like the paper account (Alpaca bars, adjustment=split, feed iex)" };
}
/** BETWEEN SESSIONS ([REPAIR] 2026-11-06, server/todayTile.ts): the pre-market tile — the last session (the weekday
 *  before the run's ET date), from the close before it (its daily line's record) to its own close (Alpaca has rolled
 *  last_equity overnight), SPY's final bar; on the idle fixture's account (equity now 99,655.03, −$57 since that close). */
function todayBetweenFixture(now) {
  const day = prevWeekday(new Date(now).toLocaleDateString("en-CA", { timeZone: "America/New_York" }));
  return { day, phase: "between", when: "pre-market", phase_source: "broker clock", prev_day: prevWeekday(day),
    equity_now: 99655.03, equity_prev_close: 100423.31, prev_close_source: "daily_line", equity_end: 99712.4, end_basis: "close",
    pnl_usd: -710.91, pnl_pct: -0.71, pnl_reason: null,
    spy_pct_today: -0.4, spy_asof: new Date(now - 12 * 60_000).toISOString(), spy_bar_date: day, spy_reason: null, spy_final: true,
    excess_pct: -0.31, asof: new Date(now).toISOString(),
    source: "Between sessions, the last session: Alpaca paper account equity now, or that session's own 16:00 ET close once Alpaca has rolled last_equity to it, vs the close before it (last_equity while Alpaca still holds it, else the session's daily line's record of it) · SPY: that session's daily bar vs the prior close, price only, no dividends, like the paper account (Alpaca bars, adjustment=split, feed iex)" };
}
function botFixtures({ latched, between = false }) {
  const now = Date.now();
  const ago = (min) => new Date(now - min * 60_000).toISOString();
  const row = (min, type, message) => ({ time: ago(min), type, action: type, message, detail: message });
  const audit = [
    row(0.5, "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"),
    row(1.2, "TIER2", "Starting intelligence scan..."),
    row(1.2, "TIER2", "Starting scan (interval: 5min based on market time)"),
    row(1.1, "SIZE-BLOCK", "AMD [stocks]: Negative expectancy for strategy 'stocks' (WR=28%); refusing to size (score 74.3)"),
    row(1.1, "CORR-CAP", "NVDA: sector Semiconductors already holds 2 positions (cap 2)"),
    row(1.1, "SKIP", "PFE: score 58 < threshold 65"),
    row(1.1, "QUEUE", "KWEB queued for market open (score 75)"),
    row(1.1, "QUEUE", "VXUS queued for market open (score 75)"),
    row(6, "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"), row(7, "TIER2", "Starting scan (interval: 5min based on market time)"),
    row(12, "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"), row(13, "TIER2", "Starting scan (interval: 5min based on market time)"),
    row(18, "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"), row(19, "TIER2", "Starting scan (interval: 5min based on market time)"),
    row(24, "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"), row(25, "TIER2", "Starting scan (interval: 5min based on market time)"),
    row(44, "TIER3", "ML model fresh (9h old) — skipping retrain"), row(44, "TIER3", "Starting strategic scan..."),
    row(45, "TIER3-DIAG", "System health: warning — 1 issues: [MEDIUM] ml_model: Low win rate: 27.8% over 36 trades"),
    row(61, "POS-WARN", "CRWV261120P00055000: down -43.7% (warn threshold -15%)"),
    row(62, "EVENTLOOP-LAG", "Node event loop stalled ~5151ms past its 2000ms tick — check for a coincident TIER2-ERROR/STREAM-DISCONNECT (KNOWN BROKEN #18)"),
    row(104, "TIER3", "ML model fresh (8h old) — skipping retrain"),
    row(105, "STREAM", "Real-time feed live — 43 tickers"), row(105, "POS-MONITOR", "Initialized — tracking 6 positions via WebSocket"),
    row(105, "SYSTEM", "Mode: PAPER TRADING (Alpaca)"), row(105, "STARTUP", "Server boot — code_version 1.0.1096, pid 1"),
    row(105.2, "SHUTDOWN", "All open orders cancelled"), row(105.2, "SHUTDOWN", "Received SIGTERM — cancelling open orders and shutting down..."),
    row(140, "EMAIL", "Alert sent: Trading Loop Still Dark — Resume Decision Needed"),
    row(164, "KILL-AUTO-RESUME-EVAL", "EVALUATED UNRECONCILED (0/3): UNRECONCILED (71 closed qty with a pre-window cost basis): drawdown -9.65% not above -8.00%; paper re-baseline after 13 market hours (2.8 so far)"),
    row(165, "LEARN", "Win rate 0% (20 post-deploy trades) — shifting weight toward VRP/squeeze (momentum: 24%, VRP: 27%)"),
    row(165, "POS-KILL-OPTION-HOLD", "CRWV261120P00055000: option down -78.1% vs premium — under the stock -25% rule but above the option backstop -150%; exits owned by options_manager"),
    row(165, "EMAIL", "Alert sent: MAX DRAWDOWN — Trading Stopped"),
    row(165, "DRAWDOWN-KILL", "MAX DRAWDOWN KILL SWITCH: Equity $99008 is -10.6% below peak $110727"),
    row(167, "SHUTDOWN", "All open orders cancelled"), row(167, "STARTUP", "Server boot — code_version 1.0.1096, pid 1"),
  ].filter((r) => latched || !/DRAWDOWN-KILL|KILL-AUTO|EMAIL|POS-KILL/.test(r.type));
  const pos = (ticker, qty, entry, cur, mv, pnl, pnlPct, stop, tp, isOption = false, extra = {}) => ({ ticker, qty, side: "long", entryPrice: entry, currentPrice: cur, marketValue: mv, pnl, pnlPct, stopPrice: stop, takeProfitPrice: tp, phase: 1, rMultiple: 0, highestPnl: 0, daysHeld: 3, asset_class: isOption ? "us_option" : "us_equity", isOption, ...extra });
  // the exit ladder the engine executes per position (shared/positionLevels.ts
  // shape; since 2026-11-12, KNOWN BROKEN #50a, the mirror of bot.ts's
  // checkPositionOnTick with POS-KILL's −25 % as the backstop on every phase):
  // QQQ phase 1 with both thirds ahead, SMH phase 2 after its first third
  // with the breakeven stop on, KWEB past its stop (the engine has no
  // short-hold grace: the stop reads "hit" and sells on the next 1-minute
  // bar), VXUS/GLD not yet recorded (the honest path), option legs with
  // their metadata
  const lv = (o) => ({ recorded: true, side: "long", phase: 1, regime: "NEUTRAL", atrPct: 1.5, riskPct: 3, pnlPct: -2.42, rMultiple: -0.81, peakR: 0.2, highestPnl: 0.6, highestPrice: 745.8, stopPct: 3, stopPrice: 719.08, stopFrom: "entry", stopHit: false, breakevenActive: false, hardFloorPct: 25, hardFloorPrice: 555.99, takeProfitPct: 4.5, takeProfitPrice: 774.68, takeProfitArmed: false, scaleOuts: [{ n: 1, rMultiple: 1, pct: 3, price: 763.56, done: false, applicable: true }, { n: 2, rMultiple: 2, pct: 6, price: 785.8, done: false, applicable: true }], trailAtrMult: 2, scaleOut2R: 2, originalQty: 61, remainingQty: 61, scalesCompleted: 0, daysHeld: 3, heldMinutes: 4320, timeStop: { day: 7, withinPct: 2, armed: false }, ...o });
  // option legs as /api/bot/positions labels them: the engine's own strategy names (options_manager state),
  // "unknown" for a leg the Node tier dispatcher opened (its state never records one) — the honest path
  const optMeta = (underlyingTicker, strike, strategy = "sell_cash_secured_put", optionType = "PUT", daysToExpiry = 30, expiry = "2026-11-20") => ({ optionMeta: { underlyingTicker, expiry, strike, optionType, strategy, setup: null, daysToExpiry }, levels: null });
  const positions = [
    pos("QQQ", 61, 741.32, 723.36, 44124.96, -1095.52, -2.42, 719.08, 774.68, false, { levels: lv({}), dayPct: -2.4 }),
    pos("SMH", 20, 551.43, 551.49, 11029.7, 1.15, 0.01, 551.43, 576.24, false, { phase: 2, rMultiple: 0.004, highestPnl: 3.4, daysHeld: 9, levels: lv({ phase: 2, atrPct: 1.6, riskPct: 3.2, pnlPct: 0.01, rMultiple: 0.004, peakR: 1.06, highestPnl: 3.4, highestPrice: 570.2, stopPct: 2.4, stopPrice: 551.43, stopFrom: "breakeven", breakevenActive: true, hardFloorPrice: 413.57, takeProfitPct: 4.8, takeProfitPrice: 576.24, scaleOuts: [{ n: 1, rMultiple: 1, pct: 3.2, price: 569.08, done: true, applicable: true }, { n: 2, rMultiple: 2, pct: 6.4, price: 586.72, done: false, applicable: true }], originalQty: 30, remainingQty: 20, scalesCompleted: 1, daysHeld: 9, heldMinutes: 12960, timeStop: { day: 7, withinPct: 2, armed: true } }) }),
    pos("KWEB", 388, 26.53, 24.54, 9521.52, -770.8, -7.49, 25.47, 27.59, false, { daysHeld: 2, levels: lv({ atrPct: 2, riskPct: 4, pnlPct: -7.49, rMultiple: -1.87, peakR: 0.1, highestPnl: 0.4, highestPrice: 26.64, stopPct: 4, stopPrice: 25.47, stopHit: true, hardFloorPrice: 19.9, takeProfitPct: 6, takeProfitPrice: 28.12, scaleOuts: [{ n: 1, rMultiple: 1, pct: 4, price: 27.59, done: false, applicable: true }, { n: 2, rMultiple: 2, pct: 8, price: 28.65, done: false, applicable: true }], originalQty: 388, remainingQty: 388, daysHeld: 2, heldMinutes: 2880 }) }),
    pos("VXUS", 150, 71.2, 71.4, 10710, 30, 0.28, null, null, false, { levels: null, dayPct: 0.3 }), pos("GLD", 21, 381.05, 380.11, 7982.31, -19.69, -0.25, null, null, false, { levels: null, dayPct: -0.4 }),
    pos("CRWV261120P00055000", -1, 2.47, 2.67, -267, -20, -8.1, null, null, true, optMeta("CRWV", 55)), pos("TEM261120P00070000", -1, 1.4, 1.96, -196, -56, -40.1, null, null, true, optMeta("TEM", 70, "unknown")), pos("PFE261120P00027000", -1, 0.48, 0.21, -21, 27, 56.25, null, null, true, optMeta("PFE", 27, "csp_normal_market")),
    // a bought call up 300 % (the 2026-11-02 question), and the two legs of one credit spread the manager closes as one unit
    pos("GOOGL261219C00280000", 1, 3.2, 12.8, 1280, 960, 300, null, null, true, { ...optMeta("GOOGL", 280, null, "CALL", 47, "2026-12-19"), dayPct: 8.4 }),
    pos("META261120P00600000", -1, 4.1, 3.2, -320, 90, 22, null, null, true, optMeta("META", 600, "bull_put_credit_spread", "PUT", 18)), pos("META261120P00580000", 1, 1.9, 1.3, 130, -60, -31.6, null, null, true, optMeta("META", 580, "bull_put_credit_spread", "PUT", 18)),
  ];
  // /api/bot/bars/<any symbol> (prefix fixture): 300 hourly bars, a
  // deterministic walk around QQQ's fixture price, ending now
  const bars = (() => {
    const nowS = Math.floor(now / 1000);
    let seed = 11, p = 736;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const out = [];
    for (let k = 299; k >= 0; k--) {
      const o = p, c = o * (1 + (rnd() - 0.5) * 0.011);
      const h = Math.max(o, c) * (1 + rnd() * 0.0035), l = Math.min(o, c) * (1 - rnd() * 0.0035);
      out.push({ time: nowS - k * 3600, open: +o.toFixed(2), high: +h.toFixed(2), low: +l.toFixed(2), close: +c.toFixed(2), volume: Math.round(1e5 + rnd() * 4e5) });
      p = c;
    }
    return out;
  })();
  const curve = Array.from({ length: 30 }, (_, k) => { const d = new Date(now - (29 - k) * 86400_000); const v = [101068, 101988, 102661, 102560, 102560, 104467, 104964, 104103, 103929, 104390, 104306, 104306, 103811, 103997, 104064, 104239, 104883, 104879, 104879, 105593, 105774, 104847, 103182, 103563, 103553, 103553, 102593, 102181, 99810, 99655][k]; return { date: d.toISOString().slice(0, 10), value: v, pnl: 0 }; });
  // GET /api/bot/scoreboard (server/scoreboard.ts shape): one row per weekday
  // since the account's 2026-05-06 start, a deterministic walk for both legs
  // (fixture numbers, real shape) — one missed snapshot (a row spanning two
  // sessions) and today's row provisional, so both markers render
  const scoreboard = (() => {
    let seed = 7;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const days = [];
    for (let t = Date.parse("2026-05-06T00:00:00Z"); t <= now; t += 86400_000) { const d = new Date(t); if (d.getUTCDay() % 6) days.push(d.toISOString().slice(0, 10)); }
    let eq = 106742.68, px = 730.13;
    const pts = days.map((date, k) => { if (k) { eq *= 1 + (rnd() - 0.52) * 0.012; px *= 1 + (rnd() - 0.47) * 0.009; } return { date, eq, px }; });
    const r2 = (x) => Math.round(x * 100) / 100;
    const kept = pts.filter((_, k) => k !== pts.length - 9);
    const rows = [];
    let beat = 0, peakB = 0, peakS = 0, ddB = 0, ddS = 0;
    for (let k = 0; k < kept.length; k++) {
      const b = kept[k];
      peakB = Math.max(peakB, b.eq); peakS = Math.max(peakS, b.px); ddB = Math.min(ddB, (b.eq / peakB - 1) * 100); ddS = Math.min(ddS, (b.px / peakS - 1) * 100);
      if (!k) continue;
      const a = kept[k - 1], bot = (b.eq / a.eq - 1) * 100, spy = (b.px / a.px - 1) * 100;
      const bc = (b.eq / kept[0].eq - 1) * 100, sc = (b.px / kept[0].px - 1) * 100;
      if (bot > spy) beat++;
      rows.push({ date: b.date, prevDate: a.date, sessions: days.indexOf(b.date) - days.indexOf(a.date), botPct: r2(bot), spyPct: r2(spy), diffPp: r2(bot - spy), botCumPct: r2(bc), spyCumPct: r2(sc), cumDiffPp: r2(bc - sc), provisional: k === kept.length - 1 });
    }
    const last = rows[rows.length - 1];
    // the AFTER-TAX series (server/afterTaxSeries.ts shape, 2026-11-06): SPY's gain since the start taxed once at the
    // short-term 24 %; the account's realized-ledger tax ramping to 0.35 % of the start equity over its first 40
    // sessions (wash-sale add-backs) — fixture numbers, real shape; oldest first, the start at zero
    const atPoints = [{ date: kept[0].date, botPct: 0, spyPct: 0, botTaxPct: 0, spyTaxPct: 0, botAfterTaxPct: 0, spyAfterTaxPct: 0, gapPp: 0, spyTerm: "short" },
      ...rows.map((r, k) => { const bt = r2(Math.min(0.35, (0.35 * (k + 1)) / 40)), st = r2(Math.max(0, r.spyCumPct) * 0.24); const ba = r2(r.botCumPct - bt), sa = r2(r.spyCumPct - st); return { date: r.date, botPct: r.botCumPct, spyPct: r.spyCumPct, botTaxPct: bt, spyTaxPct: st, botAfterTaxPct: ba, spyAfterTaxPct: sa, gapPp: r2(ba - sa), spyTerm: "short" }; })];
    const afterTax = { rates: { shortTerm: 0.24, longTerm: 0.15, state: 0, niit: 0 }, from: kept[0].date, to: last.date, equityStart: 106742.68, points: atPoints,
      notes: ["each point: both sold at that day's close — an estimate at the rates shown; the paper account pays no tax"] };
    return { afterTax, afterTaxReason: null, rows: rows.reverse(), summary: { from: kept[0].date, to: last.date, botCumPct: last.botCumPct, spyCumPct: last.spyCumPct, cumDiffPp: last.cumDiffPp, daysBeat: beat, daysCompared: rows.length, gapRows: 1, botMaxDdPct: r2(ddB), spyMaxDdPct: r2(ddS) },
      barsThrough: last.date, curveThrough: last.date, fetchedAt: new Date(now).toISOString(), computedAt: new Date(now).toISOString(),
      sources: { equity: "bot equity curve (Alpaca account portfolio_value, daily snapshot after the close, UTC date)", spy: "SPY daily closes, price return: SPY without dividends, because the paper account receives none (2026-11-05) (Alpaca bars, adjustment=split, feed=iex)" },
      cashFlows: { journals: 0, transfers: 0, since: "2026-05-07T00:00:00.000Z" } };
  })();
  const feats = ["momentum_1m", "momentum_3m", "rsi_14", "price_vs_52w_high", "volume_ratio", "ewma_vol", "atr_pct", "vrp", "iv_rank_proxy", "vwap_position", "adx", "range_pct", "vxx_ratio", "spy_vs_ma50", "markov_state", "regime_score", "sector_momentum", "cross_sec_rank", "earnings_surprise", "put_call_proxy", "vol_of_vol", "frac_diff_price", "idiosyncratic_ret", "change_pct_today", "above_ma10", "trend_strength", "intel_score", "news_sentiment", "insider_signal", "iv_rank_stock", "days_to_earnings", "credit_spread", "market_breadth", "put_call_ratio"];
  const constant = new Set(["price_vs_52w_high", "ewma_vol", "range_pct", "vxx_ratio", "markov_state", "cross_sec_rank", "earnings_surprise", "intel_score", "news_sentiment", "insider_signal", "days_to_earnings"]);
  const features = feats.map((name, k) => constant.has(name) ? { name, n: 300, mode: name === "ewma_vol" ? 2 : name === "vxx_ratio" ? 1 : name === "days_to_earnings" || name === "cross_sec_rank" ? 0.5 : 0, mode_fraction: 1, variance: 0, constant: true, last: name === "ewma_vol" ? 2 : name === "vxx_ratio" ? 1 : 0 } : { name, n: 300, mode: 0, mode_fraction: 0.05 + (k % 7) * 0.04, variance: 1.2 + k, constant: false, last: Math.round(((k * 7) % 13) * 10) / 10 });
  const q = (a5, a20, n5, n20) => ({ "+5d": { n: n5, win_rate: a5, mean_return_pct: -0.9 }, "+20d": { n: n20, win_rate: a20, mean_return_pct: -1.4 } });
  const insight = {
    generated_at: new Date(now).toISOString(),
    model: { exists: true, age_hours: 9.7, fallback_at_hours: 24, calibrator_newer_than_model: false },
    blend: { rule_weight: 0.72, ml_weight: 0.28, updated: ago(300) },
    performance: { total_trades: 36, win_rate: 27.8, avg_win: 9.08, avg_loss: -5.16, recent_win_rate_20: 20 },
    recent: { window: 300, traced: 120, model_types: { lightgbm_neutral: 118, fallback_rules: 2, "not recorded": 180 }, regime_used: { neutral: 120 }, gated: 41, blended: 120, errors: 0,
      win_probability: { n: 118, p10: 0.41, p50: 0.52, p90: 0.63 }, features, constant_features: [...constant], feature_count: 34, regime_labels: { NEUTRAL_BULL: 300 }, vxx_ratio: { last: 1.0, min: 1.0, max: 1.0 }, last_timestamp: ago(0.5), last_score: 75 },
    bands: { labeling: { method: "path_dependent_v2", win_pct: 2, stop_pct: -4 }, score_quintile_edges: [48.15, 56.592, 66.478, 77.4],
      score_quintile: { Q1: q(56.2, 83.3, 3275, 450), Q2: q(52.5, 76.5, 2873, 792), Q3: q(48.1, 57.4, 3167, 1373), Q4: q(56.8, 54.5, 3103, 1719), Q5: q(42.5, 33.2, 3664, 2369) },
      decision: { taken: q(51.5, 42.3, 6938, 4031), rejected_score: q(51.8, 70.8, 8098, 2031), rejected_other: q(52.2, 69, 602, 342), rejected_heat: q(24.1, 36.6, 428, 290) },
      decision_x_band: { "taken|<-10": q(79.7, 79.4, 1863, 383), "taken|10..20": q(38.9, 36.5, 3130, 2420), "taken|20..30": q(44.8, 42.3, 1945, 1228) } },
  };
  // ── OVERVIEW routes (2026-11-06; the contracts the backend agents build to): the decision journal, loss attribution,
  // the institutional flow model, the tax filter and the signal audit — fixture numbers, real shapes. The latched page
  // has them all (the QQQ / SMH / KWEB / CRWV / PFE / GOOGL / META positions explained, VXUS / GLD / TEM never seen by
  // the journal — the honest fallback); the idle page has an EMPTY journal, no attribution route (404 → the fill-ledger
  // fallback, still computing there), no tax route and the OLD per-ticker signal list.
  const H = 60, D = 24 * 60;
  const decisions = [
    { ts: ago(3 * D + 40), ticker: "QQQ", action: "buy", instrument: "etf", chosen: "61 QQQ shares for the ETF floor", symbol: "QQQ",
      rejected: [{ instrument: "QQQ Dec calls", reason: "IV rank 74: the calls price a bigger move than QQQ has been making, so shares keep more of the expected gain" }, { instrument: "SPY", reason: "a lower expected return over the same 20 days, with the same short-term tax" }],
      stages: { regime: "NEUTRAL: VIX 17.4 in contango, SPY above its 200-day — the full stock budget", calendar: "no holiday or index rebalance in the next 5 sessions", flows: "pension rebalancing is expected to sell stocks into month-end, so the first buy is one third", sector: "technology momentum ranks 2nd of 11 and is not crowded", selection: "3-month momentum +9.8%; the top holdings beat their last three quarters", options: "dealer gamma is positive (dips get bought); calls are priced rich", tax: "expected +3.1% over 20 days → +2.4% after short-term tax, against SPY's +0.9% — the edge survives" },
      sizing: { qty: 61, price: 741.32, notional: 45220.52, pct_equity: 44.9, reasoning: "the floor's target weight at a full risk budget" },
      summary: "Bought QQQ shares for the ETF floor: tech momentum in a calm, rising market, and the edge survives short-term tax. Calls were passed over because they cost more than the move they need." },
    { ts: ago(9 * D), ticker: "SMH", action: "buy", instrument: "shares", chosen: "30 SMH shares", rejected: [{ instrument: "NVDA shares", reason: "one name is three times the risk of the semiconductor basket for the same signal" }],
      stages: { regime: "NEUTRAL — the full stock budget", sector: "semiconductors lead on 1- and 3-month momentum", tax: "+4.0% expected → +3.0% after short-term tax vs SPY's +1.1%" }, sizing: "$16,543 — the full 4% target in one buy (the regime was calm)",
      summary: "Bought SMH: semiconductors lead the market and the basket spreads the single-name risk." },
    { ts: ago(2 * D + 3 * H), ticker: "SMH", action: "scale_out", summary: "Sold the first third at +1R ($569.08) — the plan's first profit target; the stop moved to breakeven." },
    { ts: ago(2 * D + 30), ticker: "KWEB", action: "buy", instrument: "shares", chosen: "388 KWEB shares",
      rejected: [{ instrument: "KWEB calls", reason: "too little open interest at the strikes the setup needs — the spread would cost about 9% of the premium" }],
      stages: { selection: "China internet down 7.5% in a month with selling exhausted (RSI 26)", tax: "short holding period: +2.1% expected → +1.6% after tax vs SPY's +0.3%" }, sizing: "$10,294 — a 2% risk budget at a 4% stop",
      summary: "Bought KWEB on a mean-reversion setup: down 7.5% in a month with the selling exhausted. Shares over calls because the calls barely trade." },
    { ts: ago(3 * D + 2 * H), ticker: "CRWV", action: "buy", instrument: "cash-secured put", chosen: "CRWV261120P00055000",
      rejected: [{ instrument: "CRWV shares", reason: "the put collects $247 now and only buys the stock if it falls below $55" }, { instrument: "TEM $70 put", reason: "TEM reports before expiry — a sold option is closed ahead of earnings" }],
      stages: { options: "implied volatility far above what CRWV has been moving (IV rank 91): the premium is rich", tax: "a sold put's premium is short-term income: $247 → $188 after tax, against SPY's $41 on the same cash" },
      sizing: "1 contract — $5,500 of cash set aside to buy 100 shares at $55",
      summary: "Sold a CRWV $55 put for $247: the premium is rich and the bot would be glad to own CRWV at $52.53." },
    { ts: ago(3 * D + 5 * H), ticker: "PFE", action: "buy", instrument: "cash-secured put", chosen: "PFE261120P00027000", rejected: [],
      summary: "Sold a PFE $27 put for $48: a dividend payer near its 52-week low with steady implied volatility." },
    { ts: ago(3 * D + 6 * H), ticker: "GOOGL", action: "buy", instrument: "call", chosen: "GOOGL261219C00280000",
      rejected: [{ instrument: "GOOGL shares", reason: "100 shares tie up $27,000 for the same upside" }],
      stages: { options: "calls are cheap (IV rank 18) going into the cloud-earnings momentum", tax: "a 47-day hold is short-term either way" }, sizing: "1 contract, $320 — the most it can lose",
      summary: "Bought one GOOGL $280 call for $320: a cheap bet on the cloud momentum; the most it can lose is the premium." },
    { ts: ago(3 * D + 7 * H), ticker: "META", action: "buy", instrument: "bull put credit spread", chosen: "sell META $600 put / buy $580 put",
      rejected: [{ instrument: "a naked $600 put", reason: "the bought $580 put caps the loss at $1,780 for $60 of the credit" }],
      summary: "Opened a META bull put spread for a $220 credit: the stock holds above $600 support and the spread caps the loss at $1,780." },
    { ts: ago(2 * D + 4 * H), ticker: "TSLA", action: "sell", instrument: "shares", summary: "Sold TSLA: the stop at $412.30 was hit after the trade gave back its first target." },
    { ts: ago(3 * H), ticker: "NVDA", action: "skip", instrument: "shares", rejected: [{ instrument: "NVDA shares", reason: "tax filter: 0.38% after short-term tax, below SPY's 0.60% over the same 10 days" }],
      summary: "Skipped NVDA: after short-term tax the expected gain is less than simply holding SPY." },
    { ts: ago(H), ticker: "AMD", action: "skip", instrument: "shares", rejected: [{ instrument: "AMD shares", reason: "the single-stock sleeve's record is negative (win rate 28%), so the sizer refuses new risk there" }],
      summary: "Skipped AMD: the stock sleeve is losing, so the sizer refuses new single-stock risk." },
  ];
  const attribution = (days) => {
    const k = days <= 7 ? 0.2 : days >= 90 ? 2.2 : 1;
    const m = (v) => Math.round(v * k * 100) / 100;
    return { window: { days, from: new Date(now - days * 86_400_000).toISOString().slice(0, 10), to: new Date(now).toISOString().slice(0, 10) }, total: m(-4210.5),
      bySignal: [{ signal: "VRP (sold premium)", pnl: m(-3120), n: 22, reason: "sold puts into a falling market; three were closed at twice their credit" }, { signal: "momentum", pnl: m(-1400.5), n: 6, reason: "entries came late, after most of the move" }, { signal: "mean reversion", pnl: m(310), n: 9, reason: "bought dips that bounced" }],
      bySector: [{ sector: "Technology", pnl: m(-2600), n: 14, reason: "the CRWV and TEM puts" }, { sector: "Consumer", pnl: m(-1730.5), n: 8, reason: "TSLA stopped out twice" }, { sector: "Energy", pnl: m(120), n: 3, reason: null }],
      byInstrument: [{ instrument: "sold puts", pnl: m(-3010), n: 18, reason: "the losses came from the credit side, not from shares" }, { instrument: "shares", pnl: m(-1200.5), n: 11, reason: "two stops, one target" }],
      byTiming: [{ timing: "first 15 minutes", pnl: m(-1900), n: 7, reason: "orders in the opening auction filled at the worst prices of the day" }, { timing: "held under a day", pnl: m(-1610.5), n: 12, reason: "quick exits paid the spread twice" }, { timing: "held over a week", pnl: m(-700), n: 9, reason: null }],
      notes: ["realized + open P&L, attributed by the decision journal's own tags (fixture numbers)"], asOf: ago(4) };
  };
  const flows = { asOf: ago(30), horizonDays: 10,
    equity: { pressure: -0.42, direction: "sell", confidence: 0.6, drivers: [{ name: "pension rebalancing", value: -0.3, reason: "stocks ran ahead of bonds this quarter, so target-weight funds sell the excess into month-end" }, { name: "fund tax-loss selling", value: -0.12, reason: "October fiscal year-ends: funds sell this year's losers" }] },
    sectors: [{ sector: "Technology", etf: "XLK", pressure: -0.5, direction: "sell", drivers: [{ name: "crowding", value: 1, reason: "the most crowded sector" }] }, { sector: "Utilities", etf: "XLU", pressure: 0.3, direction: "buy", drivers: [] }, { sector: "Consumer staples", etf: "XLP", pressure: 0.12, direction: "buy", drivers: [] }],
    calendar: [{ date: new Date(now + 14 * 86_400_000).toISOString().slice(0, 10), event: "monthly options expiry", effect: "dealer hedging pins the index" }, { date: new Date(now + 42 * 86_400_000).toISOString().slice(0, 10), event: "Nasdaq-100 annual reconstitution", effect: "index funds rebalance at the close" }],
    notes: ["a model estimate from published target allocations and fund calendars"],
    label: "ESTIMATE", computedAt: ago(30), ageDays: 0, stale: false,
    floor: { multiplier: 0.94, applied: true, reason: "expected net selling into month-end: the floor holds 6% under its target until it passes" },
    sessions: [0, 1, 2, 3, 4].map((k) => ({ date: new Date(now + k * 86_400_000).toISOString().slice(0, 10), pressure: [-0.42, -0.35, -0.2, 0.05, 0.12][k], phase: k < 3 ? "month-end" : null })) };
  // the flows route before tier 3 has computed its first estimate (server/flowsReport.ts's own empty body)
  const flowsNotYet = { asOf: null, horizonDays: 5, equity: { pressure: 0, direction: "neutral", confidence: 0, drivers: [] }, sectors: [], calendar: [], label: "ESTIMATE", computedAt: null, ageDays: null, stale: true, floor: null, sessions: [],
    notes: ["No flows estimate: the daily estimate has not been computed yet. Tier 3 computes it once per completed session."] };
  // /api/bot/regime with the regime agent's risk_budget block (the pipeline's first stage)
  const regime = { composite: "RISK_OFF", legacy_label: "NEUTRAL",
    risk_budget: { budget: 0.56, raw: 0.6, band: "reduced", summary: "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) −0.21; breadth 38% −0.12",
      drivers: [{ name: "vix_ts", label: "VIX term structure", value: 1.04, score: 0.2, weight: 0.4, cut: -0.21, short: "inverted", reason: "VIX above VIX3M: hedging demand is high, which tends to precede wider swings" },
        { name: "breadth", label: "Breadth", value: 38, score: 0.4, weight: 0.3, cut: -0.12, short: "narrow", reason: "only 38% of S&P 500 names are above their 50-day average" },
        { name: "trend", label: "SPY trend", value: 1.01, score: 1, weight: 0.3, cut: 0, short: "up", reason: "SPY is above its 200-day average" }],
      missing: [], as_of: new Date(now - 86_400_000).toISOString().slice(0, 10), age_market_days: 1, stale: false, enabled: true, applied: true, consumers: ["tier_exposure_cap"],
      history: Array.from({ length: 12 }, (_, k) => ({ as_of: new Date(now - (12 - k) * 86_400_000).toISOString().slice(0, 10), budget: [0.92, 0.9, 0.88, 0.81, 0.74, 0.71, 0.66, 0.6, 0.58, 0.61, 0.57, 0.56][k], band: k < 3 ? "full" : "reduced" })),
      note: "applied: the tier engine's room above the passive floor (MAX_TOTAL_EXPOSURE) is multiplied by the budget" } };
  const tax = { rates: { shortTerm: 0.24, longTerm: 0.15 },
    decisions: [
      { ts: ago(3 * H), ticker: "NVDA", side: "buy", allowed: false, expectedPct: 0.5, afterTaxPct: 0.38, spyAfterTaxPct: 0.6, holdingDays: 10, reason: "0.38% after short-term tax is below SPY's 0.60% over the same 10 days" },
      { ts: ago(3 * D + 40), ticker: "QQQ", side: "buy", allowed: true, expectedPct: 3.1, afterTaxPct: 2.36, spyAfterTaxPct: 0.9, holdingDays: 20, reason: "keeps 2.36% after tax against SPY's 0.90%" },
      { ts: ago(2 * D + 30), ticker: "KWEB", side: "buy", allowed: true, expectedPct: 2.1, afterTaxPct: 1.6, spyAfterTaxPct: 0.3, holdingDays: 8, reason: "keeps 1.60% after tax against SPY's 0.30%" },
    ],
    harvest: [{ ticker: "CRWV", unrealized: -420, replacement: "IGV", washSaleUntil: new Date(now + 31 * 86_400_000).toISOString().slice(0, 10), reason: "sell the loss and hold the software exposure through IGV until the wash-sale window closes" }],
    ltApproaching: [] };
  const signalsReport = { asOf: ago(90), signals: [
    { name: "vrp", verdict: "keep", rank_ic: 0.061, n: 1900, reason: "positive out of sample in all three regimes" },
    { name: "mean_reversion", verdict: "keep", rank_ic: 0.034, n: 1900, reason: "works in NEUTRAL and BEAR tapes" },
    { name: "momentum_1m", verdict: "noise", rank_ic: 0.004, n: 1900, reason: "indistinguishable from zero out of sample" },
    { name: "news_sentiment", verdict: "noise", rank_ic: 0.001, n: 1900, reason: "constant at serve time for most candidates" },
    { name: "deep_score", verdict: "hurts", rank_ic: -0.08, n: 20000, reason: "anti-predictive: the top score fifth wins least (open question, 2026-10-26)" },
    { name: "insider_signal", verdict: "unclear", n: 120, reason: "too few samples to judge" },
  ], notes: ["rank IC of each input against the 5-day forward return, walk-forward"] };
  const nextClose = new Date(now + 2 * 3600_000 + 2 * 60_000).toISOString();
  return {
    "/api/auth/me": { authenticated: true, isOwner: true, user: { email: "owner@example.com", tier: "pro", role: "owner" } },
    "/api/bot/status": { active: !latched, killSwitch: latched, dailyLossLimit: 3, mode: "paper", equityPeak: 110727.04, maxDrawdownPct: 10, minScoreThreshold: 65, autoResume: latched ? { status: "UNRECONCILED", drawdownPct: -9.65, resumeAbovePct: -8, marketHoursSoFar: 2.8, marketHoursRequired: 13, reconciled: 0, required: 3 } : null, unreadNotifications: 0, circuitBreakerActive: false, consecutiveStopLosses: 0,
      // ENTRY SHADOW MODE (2026-10-28, server/entryShadow.ts): both sleeves shadowed. The idle
      // moment carries four would-be entries (one Python-side with no size, one ETF instrument);
      // the latched page none — the designed "no would-be entries yet today" state.
      entry_shadow: { stocks: true, options: true, source: latched ? "config" : "env" },
      shadow_entries_today: latched ? { day: new Date(now).toLocaleDateString("en-CA", { timeZone: "America/New_York" }), count: 0, entries: [] } : { day: new Date(now).toLocaleDateString("en-CA", { timeZone: "America/New_York" }), count: 4, entries: [
        { time: ago(3), sleeve: "options", ticker: "SVXY", side: null, notional: null, source: "bot_engine.third_leg", instrument: null },
        { time: ago(9), sleeve: "options", ticker: "AMD", side: "sell", notional: 15500, source: "tier_dispatcher.SELL_CSP", instrument: "AMD261120P00155000" },
        { time: ago(21), sleeve: "stocks", ticker: "NVDA", side: "buy", notional: 4210.5, source: "executeTrades.etf", instrument: "NVDL" },
        { time: ago(40), sleeve: "stocks", ticker: "KWEB", side: "buy", notional: 2980, source: "executeMorningQueue", instrument: null },
      ] } },
    // TODAY tile (B1, 2026-11-03; server/todayTile.ts shape): the fixture's own equity against its prior close — latched
    // −$251 · −0.25 % against SPY +0.25 % (the benchmark fixture's spyLast) = −0.50 pp; idle −$768 · −0.77 % with no SPY
    // series yet, the designed null state (the live fixture server moves equity_now with equity from the third poll on)
    "/api/bot/account": between
      // between sessions (2026-11-06): pre-market, Alpaca has rolled last_equity to the last session's close
      ? { equity: 99655.03, lastEquity: 99712.4, cash: 17601.75, buyingPower: 175355.48, dailyPnL: -57.37, dailyPnLPct: -0.06, equityPeak: 110727.04, drawdownPct: -10.0, today: todayBetweenFixture(now) }
      : { equity: latched ? 100172.07 : 99655.03, lastEquity: 100423.31, cash: 17601.75, buyingPower: 175355.48, dailyPnL: latched ? -251.24 : -155.28, dailyPnLPct: latched ? -0.25 : -0.16, equityPeak: 110727.04, drawdownPct: latched ? -9.53 : -10.0,
      today: todayFixture(latched ? 100172.07 : 99655.03, 100423.31, latched ? 0.25 : null, now, ago, latched ? null : "no SPY bars fetched yet") },
    "/api/bot/positions": positions,
    "/api/bot/bars": { bars, ticker: "QQQ", timeframe: "1Hour" },
    "/api/bot/audit": audit,
    // the bot-log number (36 trigger-price records, frozen since 2026-10-14, 351 orphan exits dropped) and, on the latched
    // page, the BROKER's expectancy per closed lot ([RULE-REVIEW] 2026-10-29, server/edgeLedger.ts shape; realistic negative
    // numbers — n 1,902 active lots, −3.1 % per lot, last close today); the idle page's sleeve ledger has not loaded (broker null)
    "/api/bot/performance": { totalTrades: 36, totalFills: 55, winRate: 27.8, avgGain: 9.08, avgLoss: -5.16, totalPnlPct: -14.1, profitFactor: 0.68, byStrategy: {}, recentTrades: [], equityCurve: curve.slice().reverse(), realisticPnlPct: -15.2, avgSlippagePct: 0.12,
      lastClosedAt: "2026-10-14T19:59:02.123456", excluded: { orphanExits: 351, openEntries: 20 }, basis: "trigger price, not fill",
      broker: latched ? (() => {
        const row = (n, wins, avgWinPct, avgLossPct, expectancyPct, expectancyUsd, realizedUsd, profitFactor, lastCloseAt, unpriced) => ({ n, wins, losses: n - wins, winRate: Math.round((wins / n) * 10000) / 100, avgWinPct, avgLossPct, expectancyPct, expectancyUsd, realizedUsd, profitFactor, firstCloseAt: "2026-05-07T14:31:04.000Z", lastCloseAt, unpriced });
        const active = row(1902, 760, 24.1, -21.2, -3.1, -6.76, -12863, 0.62, ago(31), 71);
        const options = row(1724, 690, 25.4, -22.6, -3.4, -7.17, -12364, 0.6, ago(31), 0);
        const stocks = row(178, 70, 6.2, -5.9, -1.1, -2.8, -499, 0.81, "2026-10-14T19:59:02.000Z", 71);
        const floor = row(96, 61, 3.1, -2.4, 1.1, 16.19, 1554, 2.1, ago(190), 0);
        // B2 (2026-11-03, shared/expectancy.ts shape): the SAME rows with payoff and breakeven derived from them, so the
        // table's "options + stocks" total IS the tile's headline by construction; the VRP leg has closed no lot (live:
        // SVXY 0 lots since May 7); all = active + floor. The unpriced 71 are the stocks sleeve's.
        const r2 = (v) => Math.round(v * 100) / 100;
        const sl = (r) => ({ n: r.n, wins: r.wins, losses: r.losses, winRate: r.winRate, avgWinPct: r.avgWinPct, avgLossPct: r.avgLossPct,
          payoffRatio: r.avgWinPct !== null && r.avgLossPct ? r2(r.avgWinPct / Math.abs(r.avgLossPct)) : null,
          breakevenWinRate: r.avgWinPct !== null && r.avgLossPct !== null ? r2((Math.abs(r.avgLossPct) / (r.avgWinPct + Math.abs(r.avgLossPct))) * 100) : null,
          expectancyPct: r.expectancyPct, realizedUsd: r.realizedUsd, lastCloseAt: r.lastCloseAt, unpriced: r.unpriced });
        const vrp = { n: 0, wins: 0, losses: 0, winRate: null, avgWinPct: null, avgLossPct: null, payoffRatio: null, breakevenWinRate: null, expectancyPct: null, realizedUsd: 0, lastCloseAt: null, unpriced: 0 };
        const all = sl(row(1998, 821, 22.54, -20.64, -2.9, -5.66, -11309, 0.66, ago(31), 71));
        return { windowStart: "2026-05-07T00:00:00.000Z",
          active, options, stocks, floor,
          bySleeve: { options: sl(options), floor: sl(floor), stocks: sl(stocks), vrp, active: sl(active), all, unclassified: 0 },
          asOf: ago(4), source: "alpaca fills via sleeve ledger", cached_s: 240, stale: false, lastError: null };
      })() : null,
      brokerReason: latched ? null : "sleeve ledger not loaded yet; the next poll fills it" },
    "/api/bot/market-status": between
      // pre-market: the open is 3 h 40 min away (05:50 ET)
      ? { isOpen: false, nextOpen: new Date(now + 220 * 60_000).toISOString(), nextClose: new Date(now + 610 * 60_000).toISOString(), timestamp: new Date(now).toISOString() }
      : { isOpen: true, nextOpen: new Date(now + 20 * 3600_000).toISOString(), nextClose, timestamp: new Date(now).toISOString() },
    "/api/bot/calendar": { earlyCloses: [], holidays: [], tomorrowStatus: "open" },
    "/api/ml/status": { model_exists: true, enabled: true, model_age_hours: 9.7, last_accuracy: 58, last_samples: 10000, last_status: "trained" },
    "/api/bot/insight": insight,
    "/api/bot/last-scan": { timestamp: ago(0.5), scanned: 12138, filtered: 54, deep_analyzed: 5, slots_available: 0, new_trades: [{ ticker: "KWEB", score: 75, side: "sell", action_label: "SELL COVERED CALL", trade_type: "options", reasons: ["388 shares held, no short call", "Momentum 1m -7.5%", "VRP +4.1"], rules_score: 75, ml_score_raw: null, price: 24.54 }, { ticker: "VXUS", score: 75, side: "sell", action_label: "SELL COVERED CALL", trade_type: "options", reasons: ["150 shares held, no short call"], rules_score: 75, ml_score_raw: null, price: 71.4 }], top_10: [
        { ticker: "KWEB", score: 75, price: 24.54, change_pct: -1.2, side: "sell", action_label: "SELL COVERED CALL", reasons: ["388 shares held, no short call"], rank: 1, sector: "China internet", sector_clash: false, factors: { momentum: 42, mean_reversion: 71, vrp: 88, squeeze: 35, volume: 60 } },
        { ticker: "NVDA", score: 73.4, price: 184.2, change_pct: 2.1, side: "buy", action_label: "BUY", reasons: ["Momentum 1m +9.8%", "VRP +3.2", "Volume 1.8x"], rank: 2, sector: "Semiconductors", sector_clash: true, factors: { momentum: 86, mean_reversion: 30, vrp: 64, squeeze: 48, volume: 77 } },
        { ticker: "AMD", score: 74.3, price: 162.7, change_pct: 1.4, side: "buy", action_label: "BUY", reasons: ["Momentum 3m +22%", "Squeeze setup"], rank: 3, sector: "Semiconductors", sector_clash: true, factors: { momentum: 80, mean_reversion: 25, vrp: 51, squeeze: 83, volume: 66 } },
        { ticker: "XLE", score: 66.1, price: 91.3, change_pct: -0.6, side: "buy", action_label: "BUY", reasons: ["Mean reversion: RSI 28", "Sector momentum"], rank: 4, sector: "Energy", sector_clash: false, factors: { momentum: 38, mean_reversion: 82, vrp: 45, squeeze: 20, volume: 55 } },
        { ticker: "PFE", score: 58, price: 27.1, change_pct: 0.4, side: "buy", action_label: "WATCH", reasons: ["Dividend support"], rank: 5, sector: "Pharma", sector_clash: false, factors: { momentum: 22, mean_reversion: 61, vrp: 40, squeeze: 15, volume: 33 } },
      ], sizing_blocks: [{ ticker: "AMD", reason: "Negative expectancy for strategy 'stocks' (WR=28%); refusing to size", score: 74.3, strategy: "stocks" }], scan_phase_times: [{ phase: "universe_load", duration_sec: 0.93 }, { phase: "snapshot_fetch", duration_sec: 3.76 }, { phase: "quick_scan", duration_sec: 0.11 }, { phase: "deep_score", duration_sec: 41.44 }, { phase: "step6a_trade_loop_and_covered_calls", duration_sec: 0.39 }, { phase: "step7_manage_positions", duration_sec: 0.04 }, { phase: "step8b_options_manager", duration_sec: 0.3 }, { phase: "step9_third_leg", duration_sec: 0.12 }, { phase: "step9b_convexity_overlay", duration_sec: 0.31 }, { phase: "step10_spy_floor", duration_sec: 0.14 }] },
    // the real server shape (routes.ts): { orders } — a bare [] here hid the
    // Orders-panel shape bug (audit §8.2 D4) from this harness.
    "/api/orders/open": { orders: [{ id: "ord-1", symbol: "NVDA", side: "buy", qty: "12", type: "limit", limit_price: "181.20", status: "new", submitted_at: ago(4) }] },
    // SELF-CHECK + ALERTS ([PRODUCT] 2026-11-20, client/src/bot/selfCheck.tsx): the latched page has three alerts
    // (two unread) and a degraded self-check that shrinks size and lifts the score bar; the idle page an empty
    // alert list and no diagnostics route (a designed sentence)
    "/api/bot/notifications": latched ? [
      { time: ago(3), type: "alert", message: "Daily loss: -3.1%. Trading halted.", read: false },
      { time: ago(22), type: "trade", message: "OPTIONS: SELL 1x AAL $11.5 PUT (2026-12-31)", read: false },
      { time: ago(190), type: "system", message: "Bot activated — scanning for opportunities", read: true },
    ] : [],
    ...(latched ? { "/api/bot/diagnostics": { report: { timestamp: ago(1), overall_status: "degraded",
      problems: [
        { system: "cache", severity: "high", message: "Critical data sources stale: ['macro', 'insider']", auto_fix: "reduce_position_size", fix_params: { multiplier: 0.7 } },
        { system: "ml_model", severity: "medium", message: "Low win rate: 36% over 58 trades", auto_fix: "increase_score_threshold", fix_params: { min_score: 75 } },
      ],
      warnings: [{ system: "disk", message: "Cache using 410MB of 500MB limit" }], auto_fixes: ["Removed 12 expired cache files"], recommendations: [] },
      auto_fix: { position_size_multiplier: 0.7, min_score_threshold: 75, should_pause: false, force_retrain: false, problems_summary: "[HIGH] Critical data sources stale" } } } : {}),
    // SAFETY LIMITS ([PRODUCT] 2026-11-20, client/src/bot/limitsView.ts; the shape of the live answer): on the latched
    // page this loop's kill is tripped, the day's worst loss is 80% of the way to its limit, the streak is 2 of 5 and
    // the engine's halt measures from a different peak (said); the idle page has no route (a designed sentence)
    ...(latched ? { "/api/bot/safety-limits": { via: "daemon",
      node: { killSwitch: true, reason: "Max drawdown -10.0% from peak", equityPeak: 110727.04, maxDrawdownPct: -10, equity: 100172.07, drawdownPct: -9.53 },
      engine: { checked_at: ago(0), equity: 100172.07, state_read: true,
        kill_switch: { killed: false, reason: "", killed_at: null, manual_kill_file: false, peak_equity: 110727.04, drawdown_pct: -9.53, kill_equity: 88581.63, resume_equity: 105190.69 },
        engine_halt: { halted: false, reason: "", halted_at: null, peak_equity: 110000, drawdown_pct: -8.93 },
        streak: { losses: 2, last_loss_at: "2026-11-19T15:01:00" }, day: { worst_pct: -2.41, date: "2026-11-20", is_today: true },
        limits: { kill_pct: -20, daily_loss_pct: -3, streak_losses: 5, streak_pause_hours: 24, vxx_ratio_stop: 1.4, min_free_bp_pct: 10, correlated_group_cap_pct: 60, position_stop_pct: -25, position_warn_pct: -15,
          resume_recovery_pct: 95, resume_cooldown_hours: 4, engine_halt_pct: -18, engine_halt_enabled: true, engine_halt_resume_within_pct: 5, engine_halt_resume_regimes: ["BULL", "NEUTRAL"] },
        recent_events: [{ at: "2026-11-06T15:32:44.341096", reason: "DRAWDOWN", equity: 89370.65, drawdown_pct: -20.02 }] } } } : {}),
    // EVIDENCE ([PRODUCT] 2026-11-20, client/src/bot/evidence.tsx; real shapes, taken from the live probes): the
    // latched page has all six reads; the idle page none of the routes (a server that predates them — each block
    // says so in a designed sentence)
    ...(latched ? evidenceFixtures({ now, ago }) : {}),
    // OVERVIEW routes (2026-11-06, see above): the journal (empty on the idle page), attribution by the request's
    // ?days= (404 on the idle page: the fill-ledger fallback), flows, tax (absent on the idle page) and signals
    // (the old per-ticker list on the idle page)
    "/api/bot/decisions": latched ? decisions : [],
    "/api/bot/attribution": latched ? (req) => attribution(Number(new URL(req.url || "/", "http://fixture").searchParams.get("days")) || 30) : { __status: 404, __body: { error: "Not found" } },
    "/api/bot/flows": latched ? flows : flowsNotYet,
    ...(latched ? { "/api/bot/regime": regime } : {}),
    ...(latched ? { "/api/bot/tax": tax } : {}),
    "/api/bot/signals": latched ? signalsReport : [{ ticker: "SPY", action: "HOLD", confidence: 50, reason: "", type: "neutral", time: ago(30) }],
    // look-up (2026-10-22): what the scan would do with NVDA right now — a name down big on bad news that scores on mean
    // reversion, stopped at admission (prefix fixture: /api/bot/lookup/<symbol>)
    "/api/bot/lookup": { symbol: "NVDA", at: new Date(now).toISOString(), path: "snapshot → quick pass → deep_score → gates → chain", errors: {}, timings: { total_s: 6.4 },
      quick: { price: 183.2, change_pct: -6.8, volume: 61_000_000, dollar_volume: 11_175_200_000, quick_score: 48.6, above_open: false },
      deep: { score: 58.2, side: "buy", action_label: "BUY", sector: "Semiconductors", ml_win_prob: 0.44, reasons: ["Mean reversion: RSI 24 oversold", "Volume 2.6× average", "News sentiment −0.6 (guidance cut)"], factors: { momentum: -18, mean_reversion: 31, vrp: 4, squeeze: 2, volume: 2.6, sentiment: -9 } },
      regime: { label: "NEUTRAL", vxx_ratio: 1.02, spy_vs_ma50: 1.01, spy_below_200_days: 0, min_score: 65 }, held: false,
      decision: { stopped_at: "admission", verdict: "stopped at admission", gates: [
        { gate: "universe", pass: true, why: "in the scan's universe" }, { gate: "regime", pass: true, why: "NEUTRAL" }, { gate: "cooldown", pass: true, why: "no recent stop" },
        { gate: "admission", pass: false, why: "score 58.2 vs MIN_SCORE 65" }, { gate: "sector", pass: false, why: "sector / correlation cap reached (max 3/sector)" }, { gate: "sizing", pass: true, why: "40 sh · $7,300" } ] },
      chain: { contracts: 412, atm_iv: 61.2, iv_rank: 88, window_days: "21–45", expiries: ["2026-11-21", "2026-12-19"],
        csp: { occ: "NVDA261121P00170000", strike: 170, exp_date: "2026-11-21", days_out: 30, bid: 2.4, ask: 2.6, delta: -0.22, delta_real: true, iv: 0.61, oi: 12_400, premium_pct: 1.31 },
        cc: { occ: "NVDA261121C00200000", strike: 200, exp_date: "2026-11-21", days_out: 30, bid: 1.9, ask: 2.0, delta: 0.25, delta_real: true, iv: 0.58, oi: 9_800, premium_pct: 1.04 },
        scanner_setup: null } },
    // the look-up box now opens the ticker dossier (2026-10-29): prefix fixture /api/ticker-dossier/<symbol>
    "/api/ticker-dossier": dossierFixture("NVDA", 183.2),
    // the vs-SPY comparison (latched page) / no SPY series yet (idle page). 2026-11-06: the latched window follows the
    // scoreboard's walk (it was the live 2026-10-20 reading before), so the vs-SPY tile, the daily scoreboard and the
    // overview's after-tax chart tell one story; "sold now" = the chart line's last point plus the tax selling the open
    // positions would add (0.12 % of the start equity) — fixture numbers, real shape
    "/api/bot/benchmark": latched
      ? (() => {
          const sm = scoreboard.summary, at = scoreboard.afterTax.points[scoreboard.afterTax.points.length - 1];
          const days = Math.round((Date.parse(sm.to) - Date.parse(sm.from)) / 86_400_000);
          const r2 = (v) => Math.round(v * 100) / 100;
          const acct = r2(at.botAfterTaxPct - 0.12);
          return { since: { from: sm.from, to: sm.to, days, accountPct: sm.botCumPct, spyPct: sm.spyCumPct, gapPp: sm.cumDiffPp }, d30: { from: "2026-09-20", to: "2026-10-20", days: 30, accountPct: -3.11, spyPct: 0.88, gapPp: -3.99 }, spyLast: { date: new Date(now).toLocaleDateString("en-CA", { timeZone: "America/New_York" }), pct: 0.25 }, barsThrough: sm.to, fetchedAt: new Date(now).toISOString(), adjustment: "split", basis: "price return: SPY without dividends, because the paper account receives none (2026-11-05)",
            // the after-tax leg ([RULE-REVIEW] 2026-10-22, server/afterTaxBenchmark.ts): both sold at the walk's last day at the default rates
            afterTax: { rates: { shortTerm: 0.24, longTerm: 0.15, state: 0, niit: 0 }, window: { from: sm.from, to: sm.to, days }, spy: { pct: sm.spyCumPct, term: "short", taxPct: at.spyTaxPct, afterTaxPct: at.spyAfterTaxPct }, account: { pct: sm.botCumPct, taxPct: r2(at.botTaxPct + 0.12), afterTaxPct: acct, equityStart: 106742.68, realizedTax: r2(at.botTaxPct * 1067.4268), liquidationTax: r2((at.botTaxPct + 0.12) * 1067.4268), openPnl: 534.2, openTaxIfSold: 128.09 }, gapPp: r2(acct - at.spyAfterTaxPct),
              notes: [`both sold at ${sm.to}: an estimate at the rates shown — the paper account pays no tax`, "SPY: its price return, without dividends like the paper account, taxed once at the short-term rate; a real holder would also collect the dividends, taxed each year at the qualified rate, so leaving them out flatters the account by about yield × (1 − qualified rate)", "account: the ledger's estimated tax on the trades closed in the window, plus what selling every open position would add — open P&L taxed as short-term (the conservative term) and netted the way the ledger nets, so open losses reduce the tax"] } };
        })()
      : { since: null, d30: null, spyLast: null, barsThrough: null, fetchedAt: null },
    // the daily scoreboard (fixture walk above) / not enough history on the idle page
    "/api/bot/scoreboard": latched ? scoreboard : { rows: [], summary: null, barsThrough: null, curveThrough: null, fetchedAt: null, cashFlows: null },
    "/api/bot/daily-line": dailyLineFixture({ latched, now }),
    // P&L by sleeve (fixture numbers, real shape) / still computing on the idle page
    "/api/bot/sleeves": latched
      ? { since: "2026-05-07T00:00:00.000Z", truncated: false, malformedFills: 0, fetchedAt: new Date(now).toISOString(), firstFill: "2026-03-12T14:02:11.000Z", lastFill: new Date(now).toISOString(), realizedTotal: -2961.4, unrealizedTotal: -4408.9, rows: [
          { sleeve: "options", fills: 412, symbols: 118, realized: -1532.1, unpricedCloses: 0, unrealized: -96, openPositions: 6, byMonth: { "2026-10": -516.4 } },
          { sleeve: "floor", fills: 690, symbols: 7, realized: 93.2, unpricedCloses: 0, unrealized: -4398.2, openPositions: 5, byMonth: { "2026-10": 93.2 } },
          { sleeve: "stocks", fills: 236, symbols: 61, realized: -1522.5, unpricedCloses: 3, unrealized: 85.3, openPositions: 2, byMonth: { "2026-08": -1340.2 } },
        ], optionsUnderlyings: 118, optionsByUnderlying: [
          { root: "CRWV", fills: 14, realized: -612.4, byMonth: { "2026-08": -414.4, "2026-10": -198 }, open: -42.5, openPositions: 1 },
          { root: "TSLA", fills: 8, realized: -66, byMonth: { "2026-10": -66 }, open: 0, openPositions: 0 },
          { root: "META", fills: 8, realized: -70, byMonth: { "2026-10": -70 }, open: 0, openPositions: 0 },
          { root: "GOOGL", fills: 8, realized: -84, byMonth: { "2026-10": -84 }, open: 0, openPositions: 0 },
          { root: "BAC", fills: 6, realized: -78, byMonth: { "2026-10": -78 }, open: -17, openPositions: 1 },
          { root: "XLF", fills: 12, realized: 161.2, byMonth: { "2026-09": 161.2 }, open: 0, openPositions: 0 },
        ], nonTrade: { total: 4712.3, count: 9, nonCashRows: 0, groups: { dividends: 412.3, interest: 0, fees: 0, journals: 4300, optionEvents: 0, corporate: 0, transfers: 0, other: 0 },
          byMonth: { "2026-05": 4300, "2026-08": 312.3, "2026-09": 100 }, types: [{ type: "JNLC", group: "journals", count: 1, net: 4300, byMonth: { "2026-05": 4300 } }, { type: "DIV", group: "dividends", count: 8, net: 412.3, byMonth: { "2026-08": 312.3, "2026-09": 100 } }], first: "2026-05-05", last: "2026-09-15" },
        reconciliation: { equityFrom: "2026-05-06", equityTo: "2026-10-20", equityStart: 106743, equityEnd: 99373, equityChange: -7370, realized: -2961.4, unrealized: -4408.9, nonTrade: 4712.3, explained: -2658, residual: -4712 },
        windowStart: "2026-05-07T00:00:00.000Z", preWindowFills: 212,
        optionsByHold: [{ bucket: "<1h", events: 96, realized: -1104.3 }, { bucket: "<1d", events: 40, realized: -212 }, { bucket: "1–7d", events: 110, realized: 318.5 }, { bucket: "8–30d", events: 150, realized: -534.3 }, { bucket: ">30d", events: 12, realized: 0 }, { bucket: "unknown", events: 4, realized: 0 }],
        // the same trades in a TAXABLE account (server/taxLedger.ts, [RULE-REVIEW] 2026-10-21): a net loss, no tax, a carryforward
        tax: { rates: { shortTerm: 0.24, longTerm: 0.15, state: 0, niit: 0 }, windowStart: "2026-05-07T00:00:00.000Z",
          byTerm: [{ term: "short", events: 380, gains: 6210.5, losses: -9050.5, net: -2840, washDisallowed: 612.8 }, { term: "long", events: 0, gains: 0, losses: 0, net: 0, washDisallowed: 0 }, { term: "unknown", events: 4, gains: 0, losses: -121.4, net: -121.4, washDisallowed: 0 }],
          byYear: [{ year: 2026, short: -2840, long: 0, unknown: -121.4, washDisallowed: 612.8, estimatedTax: 0 }],
          washSales: { count: 23, disallowed: 612.8, examples: [{ symbol: "NVDA", closedAt: "2026-08-12T14:31:00.000Z", loss: -188.2, disallowed: 188.2, replacementQty: 20, replacedAt: "2026-08-13T13:45:00.000Z", term: "short" }] },
          netting: { shortNet: -2348.6, longNet: 0, shortAfterNetting: -2348.6, longAfterNetting: 0, carryforward: -2348.6 },
          estimatedTax: { shortTerm: 0, longTerm: 0, niit: 0, total: 0 },
          afterTax: { realizedNet: -2961.4, taxDrag: 0, afterTaxNet: -2961.4 },
          notes: ["estimate at the assumed rates shown; the paper account pays no tax"] } }
      : { pending: true },
  };
}

// Ticker dossier (2026-10-29, human-asked): GET /api/ticker-dossier/<SYM>
// (ticker_dossier.py). An RKLB-like space name: live market/sector/stock
// layers, the range bands with backtest + a thin live ledger, no direction
// call (not in the weekly set — the honest coin-flip state), a trade plan,
// options with implied vs model, events across three credibility tiers.
/** GET /api/bot/daily-line ([PRODUCT] 2026-11-06, server/dailyLine.ts dailyLineBody): the lines as the job writes them,
 *  newest first. Latched page: ten sessions (the weekdays before the run's ET date) — one written partial (its orders
 *  unreadable, so no option total), one whose option window opens at an older line (a line in between was missed);
 *  their account days end at the close reading ([RULE-REVIEW] 2026-11-06) except one that fell back to the equity read
 *  when written (marked "*"), and the newest carries its close check of the line before it.
 *  Idle page: the one line the live server held on 2026-11-06 (no previous line, so its option total is not
 *  computable — the designed note; written before the close reading, so its account cell is marked "*"). Both: today's
 *  line due after the close, 2 h 22 min from now (the fixture's close is 2 h 02 min away). */
/** The owner's evidence routes ([PRODUCT] 2026-11-20; server/bot.ts OWNER VIEWS OF THE EVIDENCE PROBES): the closed
 *  lots (12, so "show 12" has something to show), the order history with a two-step price ladder, the base holding's
 *  tracking report, the rule counterfactuals, the regime engine against the old rule, and the option quote feed. */
function evidenceFixtures({ now, ago }) {
  const iso = (minAgo) => new Date(now - minAgo * 60_000).toISOString();
  const dayAgo = (d) => new Date(now - d * 86_400_000).toISOString().slice(0, 10);
  const minAgoOf = (isoTime) => (now - Date.parse(isoTime)) / 60_000;
  const lot = (i, symbol, option, side, qty, entry, exit) => { const mult = option ? 100 : 1; const basis = +(qty * entry * mult).toFixed(2); const pnl = +((exit - entry) * qty * mult * (side === "long" ? 1 : -1)).toFixed(2); return { symbol, option, side, qty, closeDir: side === "short" ? 1 : -1, openedAt: i === 5 ? null : iso(1440 * (i + 3)), closedAt: iso(1440 * i + 95), basis, pnl, returnPct: +((pnl / basis) * 100).toFixed(2), entryPrice: entry, exitPrice: exit }; };
  const lots = [
    lot(11, "IREN261204C00023000", true, "short", 3, 2.61, 2.64), lot(10, "IREN261204P00021000", true, "short", 3, 2.75, 2.72), lot(9, "QQQ", false, "long", 14, 612.4, 598.15),
    lot(8, "SMH", false, "long", 22, 301.2, 296.84), lot(7, "KWEB", false, "long", 388, 25.1, 24.54), lot(6, "VXUS", false, "long", 120, 71.3, 70.92), lot(5, "GLD", false, "long", 12, 377.47, 370.68),
    lot(4, "T261218P00022000", true, "short", 2, 0.44, 0.21), lot(3, "JD261218P00025000", true, "short", 1, 0.62, 0.9), lot(2, "NVDA261120C00190000", true, "long", 1, 3.6, 5.05),
    lot(1, "AMD", false, "long", 40, 162.7, 158.2), lot(0, "CMCSA261218P00024000", true, "short", 1, 0.79, 0.38),
  ];
  const ord = (id, symbol, cls, side, type, qty, limit, status, minAgo, filledPx = null, endAfterMin = null) => ({ id, symbol, asset_class: cls, side, type, qty, notional: null, filled_qty: status === "filled" ? qty : 0, filled_avg_price: filledPx, limit_price: limit, status, submitted_at: iso(minAgo), filled_at: status === "filled" ? iso(minAgo - (endAfterMin ?? 0)) : null, canceled_at: status === "canceled" ? iso(minAgo - (endAfterMin ?? 12)) : null });
  const orders = [
    ord("eo-1", "NVDA", "us_equity", "buy", "limit", 12, 181.2, "new", 4),
    // the price ladder: three orders for one contract on ONE ET day (fixed UTC clock times two days back, so a run
    // near midnight ET cannot split the day): 0.29 cancelled, 0.30 cancelled, 0.31 filled
    ord("eo-2", "AAL261231P00011500", "us_option", "sell", "limit", 1, 0.31, "filled", minAgoOf(`${dayAgo(2)}T15:54:00Z`), 0.31, 2),
    ord("eo-3", "AAL261231P00011500", "us_option", "sell", "limit", 1, 0.3, "canceled", minAgoOf(`${dayAgo(2)}T15:32:00Z`), null, 20),
    ord("eo-4", "AAL261231P00011500", "us_option", "sell", "limit", 1, 0.29, "canceled", minAgoOf(`${dayAgo(2)}T15:10:00Z`), null, 20),
    ord("eo-5", "CMCSA261218P00024000", "us_option", "buy", "limit", 1, 0.38, "filled", 1440 * 3 + 20, 0.38, 0),
    ord("eo-6", "SPY", "us_equity", "buy", "limit", 82, 740.5, "filled", 1440 * 13, 740.13, 0),
    ord("eo-7", "VXUS", "us_equity", "sell", "market", 120, null, "filled", 1440 * 14, null, 0),
  ];
  const sess = (d, spy, target, held, reb = false) => { const rule = +(target * spy).toFixed(4), floor = +(held * spy).toFixed(4); return { day: dayAgo(d), prev_day: dayAgo(d + 1), entry: false, spy_pct: spy, capital: 75472.55, target_exposure: target, held_exposure: held, drift: +Math.abs(target - held).toFixed(4), rule_pnl: 0, rule_pct: rule, floor_pnl: 0, floor_pct: floor, tracking_pp: +(floor - rule).toFixed(4), exposure_pp: +(floor - rule).toFixed(4), other_pp: 0, rebalance: reb, legs: [], reason: null }; };
  const sessions = [sess(1, -1.984, 0.7039, 0.7935), sess(2, -0.284, 0.6992, 0.7941), sess(3, 0.612, 0.7104, 0.7938), sess(4, -0.905, 0.7421, 0.7933), sess(5, 0.33, 0.7688, 0.7929)];
  const cell = (labeled, mean, baseMean, baseN, pending, sized = 0, prevention = 0) => ({ labeled, pending, unlabelable: 0, win_rate_pct: labeled ? 50 : undefined, ...(mean === null ? {} : { return_pct: { mean, median: mean } }), base_scanned_same_days: baseMean === null ? { n: 0 } : { n: baseN, win_rate_pct: 55, mean_return_pct: baseMean }, dollars: { estimate: true, positions: labeled, sized_positions: sized, cut_dollars: 0, prevention_pnl_dollars: prevention } });
  const day = (d, sel, opt, t1) => ({ date: dayAgo(d), sizing: t1 ? { tier1: { n: 14, applied: 14, stale: 0, mean_legacy_multiplier: t1, mean_engine_multiplier: 0.5 } } : {}, selection: sel, options: opt, label: { n: 26, differs: 26, applied: 26, pairs: { "NEUTRAL>BULL": 26 } }, composites: { CALM_BULL: 300 } });
  return {
    "/api/bot/lots": { count: 2141, truncated: true, since: "2026-05-07T00:00:00.000Z", fetchedAt: iso(6), pending: false, lastError: null, rows: lots },
    "/api/bot/order-history": { count: orders.length, orders },
    "/api/bot/floor-tracking": { version: 1, generated_at: iso(3), status: "ok", reason: null, mode: "vol_managed_spy", config: { base: "SPY", levered: "SSO", cash_fund: "BOXX", capital_pct: 0.85, band: 0.1 },
      definitions: { tracking_pp: "floor_pct - rule_pct, in percentage points of the floor's capital" },
      entry: { day: dayAgo(14), shortfall_pp: -0.7225, note: "floor_pnl runs from the fills to the close" }, latest: sessions[0],
      rolling: { sessions: 9, from: dayAgo(11), to: dayAgo(1), floor_pct: -2.1184, rule_pct: -1.6541, tracking_pp: -0.4644, window_sessions: 20, complete: false, missing: [] },
      rollback_trigger: { text: "rollback trigger: live floor tracking vs its own rule > 3 pp over any 20 sessions, or account drawdown beyond the backtest's 95th-percentile 3-month drawdown", source: "research/floor_study_2026-11-05.md, Pre-registration", status: "undetermined", summary: "tracking: 9 measured sessions, a window needs 20 in a row; -0.46pp so far; drawdown leg not evaluated (the study recorded no limit)", tracking: { status: "not_enough_history", limit_pp: 3, window_sessions: 20, sessions_measured: 9 } },
      turnover: { window: { sessions: 9, core: { orders: 0, filled: 0, dollars: 0 }, cash_fund: { orders: 0, filled: 0, dollars: 0 }, rebalance_sessions: 0, outside_band_no_trade: 0 } },
      sessions, message: "fixture" },
    "/api/bot/rule-counterfactuals": { via: "daemon", generated_at: iso(8), min_n: 5, log: { records: 20000, first_day: "2026-09-03", last_day: dayAgo(0), gate_rows: 503, taken_rows: 3555 },
      dollar_basis: "ESTIMATE. cut dollars x the stock price-path label's return (long the underlying, +2% target / -4% stop / N-day timeout). Not option P&L.",
      limits: ["Outcomes are shadow_portfolio's path labels of the UNDERLYING stock, long. A blocked cash-secured put is bullish, so the sign is right; the size of its P&L is not.", "Records from before the 2026-11-12 change carry no tier and no size.", "A third limit the page does not print."],
      gates: {
        exposure_ceiling: { what: "the regime's MAX_TOTAL_EXPOSURE", rows: 378, name_days: 45, names: 28, groups: [
          { tier: "1", action: "dropped", rows: 375, name_days: 45, names: 28, first_day: "2026-10-29", last_day: dayAgo(3), "+5d": cell(14, -1, -0.623, 131, 19) },
          { tier: "1", action: "scaled", rows: 3, name_days: 3, names: 3, first_day: dayAgo(3), last_day: dayAgo(3), "+5d": cell(0, null, null, 0, 3) }] },
        daily_loss: { what: "the tier engine's daily loss limit", rows: 68, name_days: 15, names: 14, groups: [{ tier: "1", action: "dropped", rows: 68, name_days: 15, names: 14, first_day: "2026-10-29", last_day: "2026-11-05", "+5d": cell(13, -2.154, -2.026, 41, 0) }] },
        drawdown: { what: "the tier engine's portfolio drawdown kill", rows: 0, name_days: 0, names: 0, groups: [] },
        cash_reserve: { what: "the regime's cash reserve", rows: 57, name_days: 16, names: 15, groups: [
          { tier: "scanner", action: "dropped", rows: 21, name_days: 9, names: 9, first_day: "2026-11-09", last_day: "2026-11-13", "+5d": cell(8, 1.5, -1, 20, 1, 4, -312.4) },
          { tier: "unrecorded", action: "dropped", rows: 36, name_days: 7, names: 6, first_day: "2026-11-06", last_day: dayAgo(2), "+5d": cell(3, 2, 0.838, 32, 4) }] },
      } },
    "/api/bot/regime-counterfactual": { days: 14, since: dayAgo(13), rows: 3112, note: "legacy = what the pre-engine system would have done at the same decision point; engine = what the regime engine did.",
      by_day: [
        day(2, { n: 32, differs: 6, decisions: { block: 6, take: 26 }, blocks_by_guard: { knife: 6, rs: 1 }, size_cut: 0 }, { n: 15, differs: 9, decisions: { take: 15 }, blocks_by_guard: {}, size_cut: 9 }, 0.8607),
        day(1, { n: 18, differs: 9, decisions: { take: 9, block: 9 }, blocks_by_guard: { knife: 2, rs: 9 }, size_cut: 0 }, { n: 10, differs: 10, decisions: { take: 10 }, blocks_by_guard: {}, size_cut: 10 }, 0.6643),
        day(0, { n: 0, differs: 0, decisions: {}, blocks_by_guard: {}, size_cut: 0 }, { n: 0, differs: 0, decisions: {}, blocks_by_guard: {}, size_cut: 0 }, null),
      ] },
    "/api/bot/options-feed": { feed: "indicative", opra: "not_entitled", indicative: "entitled", checked_at: iso(40) },
  };
}

function dailyLineFixture({ latched, now }) {
  const r2 = (x) => Math.round(x * 100) / 100;
  const et = (ms) => new Date(ms).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const days = [];
  for (let d = new Date(`${et(now)}T12:00:00Z`); days.length < (latched ? 11 : 2);) {
    d = new Date(d.getTime() - 86_400_000);
    if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) days.push(d.toISOString().slice(0, 10));
  }
  // 16:22 ET on the day (the job's first tick after the close + 20 min), in EST or EDT
  const at1622 = (day) => { const est = Date.parse(`${day}T21:22:00Z`); return new Date(new Date(est).toLocaleString("en-US", { timeZone: "America/New_York", hour: "2-digit", hour12: false }) === "17" ? est - 3_600_000 : est).toISOString(); };
  const basis = "account: Alpaca equity vs last_equity (no dividends in the paper account) · SPY: PRICE-ONLY daily closes (Alpaca bars, adjustment=split, feed iex)";
  const basisClose = "account: Alpaca equity at the close (the last account read before the session's close) vs last_equity, the prior close (no dividends in the paper account) · SPY: PRICE-ONLY daily closes (Alpaca bars, adjustment=split, feed iex)";
  const basisWriteTime = "account: Alpaca equity when the line was written (no usable close reading: after-hours marks included) vs last_equity, the prior close (no dividends in the paper account) · SPY: PRICE-ONLY daily closes (Alpaca bars, adjustment=split, feed iex)";
  // 15:59:52 ET on the day (the close reading's last tick), in EST or EDT
  const at1559 = (day) => new Date(Date.parse(at1622(day)) - (22 * 60 + 8) * 1000).toISOString();
  const ACCT = [-4.07, 0.42, -0.95, 1.12, -0.31, 0.08, -1.64, 0.55, -0.22, 0.73];
  const SPYD = [-4.34, 0.61, -0.71, 0.98, -0.12, 0.21, -1.25, 0.47, -0.05, 0.66];
  const X = [0.8512, 0.8833, 0.9102, 0.8711, 0.9404, 0.9611, 0.8226, 0.9015, 0.9333, 0.9488];
  const TOT = [-120.4, 85.2, -310.65, null, 42, -18.3, -455.9, 130.1, -60, 25.5];
  const FP = [-40.3, -12.5, -88.1, 0, -6.2, -9.9, -102.4, -15, -20.2, -4.4];
  const FN = [[2, 5], [3, 3], [4, 6], [0, 0], [1, 2], [2, 2], [5, 9], [2, 3], [1, 4], [1, 1]];
  const HOLDS = [[1, 1, 1], [0, 0, 0], [2, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0], [3, 1, 2], [0, 0, 0], [1, 0, 0], [0, 0, 0]];
  const line = (i) => {
    const day = days[i];
    const live = !latched;
    const total = live ? null : TOT[i];
    const [attributed, fills] = live ? [0, 3] : FN[i];
    const [single, group, posKill] = live ? [0, 0, 0] : HOLDS[i];
    return {
      version: 1, day, written_at: at1622(day), session_close: "16:00",
      // the live line predates the close reading (no source); the latched lines end at it, but for one that fell back
      account_vs_spy: live
        ? { basis, day: { account_pct: ACCT[i], spy_pct: SPYD[i], gap_pp: r2(ACCT[i] - SPYD[i]), equity: 90000, last_equity: r2(90000 / (1 + ACCT[i] / 100)), spy_bar_date: day, reason: null }, since: null, d30: null, since_requested: "2026-05-06", windows_reason: null }
        : { basis: i === 5 ? basisWriteTime : basisClose,
          day: { account_pct: ACCT[i], spy_pct: SPYD[i], gap_pp: r2(ACCT[i] - SPYD[i]), equity: 90000, last_equity: r2(90000 / (1 + ACCT[i] / 100)), spy_bar_date: day, reason: null,
            account_source: i === 5 ? "write_time" : "close", equity_end: 90000, equity_end_at: i === 5 ? at1622(day) : at1559(day), close_reason: i === 5 ? `no close reading for ${day}` : null },
          since: null, d30: null, since_requested: "2026-05-06", windows_reason: null },
      floor: { mode: "vol_managed_spy", exposure_x: live ? 0 : X[i], basis: "(SPY + 2 x SSO market value) / equity", base_mv: 0, levered_mv: 0, floor_mv: 0, legacy_mv: 26603.7, target_x: live ? null : r2(X[i] + 0.05), target_at: live ? null : at1622(day), reason: null },
      options: {
        window: { from: at1622(days[i + 1]), to: at1622(day), from_line: live ? null : i === 6 ? days[i + 2] ?? null : days[i + 1], single_session: live ? true : i !== 6 },
        realized: live ? 948 : 0, realized_lots: 3, unrealized_start: live ? null : 0, unrealized_end: -18, closed_without_fill: [], total,
        fills: { fills, attributed, cost: -FP[i], pnl: live ? 0 : FP[i], unattributed: { count: fills - attributed, no_row: fills - attributed, no_mid: 0, mleg_no_leg_fills: 0, notional: 0 }, worst: [], ledger_rows: 0, ledger_malformed: 0 },
        direction: total === null ? null : r2(total - FP[i]), contracts_end: [],
        reason: live ? "no previous line within 7 days to start from (the option total starts with the next line)" : i === 3 ? "the window's orders are unreadable (fills not joined)" : null,
      },
      opening_window: { total: single + group + posKill, manager_single_leg: single, manager_group: group, pos_kill: posKill, pos_kill_tickers: posKill ? ["AMD"] : [], keys: [], reason: null },
      missing: !live && i === 3 ? ["orders: the Alpaca orders page timed out"] : [],
      // the newest latched line's check of the line before it against Alpaca's rolled last_equity (hover only)
      close_check: live ? undefined : i === 0
        ? { day: days[1], line_pct: ACCT[1], line_equity: 90000, line_source: "close", alpaca_close: 89982, alpaca_pct: r2(ACCT[1] - 0.02), error_pp: 0.02, error_usd: 18, balance_asof: days[1], reason: null }
        : null,
      message: "",
    };
  };
  const lines = days.slice(0, latched ? 10 : 1).map((_, i) => line(i));
  // today's line is due at 16:20 ET (the close + 20 min); a run after that hour keeps the designed "after the close" state
  // by placing the due time 2 h 22 min ahead (the fixture's close is 2 h 02 min away)
  const due1620 = Date.parse(at1622(et(now))) - 2 * 60_000;
  return { count: lines.length, latest: lines[0], recent: lines, lastAttempt: null,
    schedule: { day: et(now), session: true, due_at: new Date(now < due1620 ? due1620 : now + (2 * 60 + 22) * 60_000).toISOString() }, keep: 60, error: null };
}
function dossierFixture(sym = "RKLB", px = 27.42, opts = {}) {
  const now = Date.now();
  const iso = (msAgo) => new Date(now - msAgo).toISOString();
  const day = (d) => new Date(now - d * 86400_000).toISOString().slice(0, 10);
  const band = (s) => { const r = (z) => +(px * Math.exp(z * s)).toFixed(2); return { lo68: r(-1), hi68: r(1), lo90: r(-1.6449), hi90: r(1.6449), move68_pct: +((Math.exp(s) - 1) * 100).toFixed(2), move90_pct: +((Math.exp(1.6449 * s) - 1) * 100).toFixed(2) }; };
  const s1 = 0.0452, s5 = 0.0981;
  // measured event-reaction history (research/event_reaction_study.md, 2026-10-29) — history, never a flag
  const STUDY = { path: "research/event_reaction_study.md", date: "2026-10-29", window: "2016–2026", caveat: "survivor panel (delisted names missing); history, not a prediction" };
  const H101 = { label: "8-K Item 1.01 (material agreement)", n: 42956, car01_pct: 0.14, t01: 2.3, drift: "no reliable day-0–1 move; later drift failed the robustness and cost checks" };
  const trend = (a, b, c, d) => ({ label: a > 0 && b > 0 ? "up" : a < 0 && b < 0 ? "down" : "mixed", sma20: px * 0.97, sma50: px * 0.9, sma200: px * 0.72, vs_sma50_pct: 11.1, vs_sma200_pct: 38.6, ret_5d_pct: a, ret_20d_pct: b, ret_60d_pct: c, ret_250d_pct: d });
  const touch = (t, sp) => ({ p_target_first: t, p_stop_first: sp, p_neither: +(1 - t - sp).toFixed(4), paths: 4000, steps_per_session: 78, seed: 20261029, drift_daily: 0 });
  const stop = +(px * Math.exp(-1.6449 * s5)).toFixed(4), tgt = +(px * Math.exp(s5)).toFixed(4), str = +(px * Math.exp(1.6449 * s5)).toFixed(4);
  const dec = (k, lo, hi, n, pos, mean) => ({ decile: k, lo, hi, count: n, d5: { n, pos_rate: pos, mean_pct: mean, rule_win_rate: pos - 0.12 }, d20: { n: Math.round(n * 0.7), pos_rate: pos + 0.01, mean_pct: mean * 2.4, rule_win_rate: null }, decisions: { rejected_score: Math.round(n * 0.8), taken: Math.round(n * 0.05) } });
  return {
    symbol: sym, generated_at: iso(95_000), errors: opts.optionsDown ? { options: "HTTPError: 403 options feed" } : {}, timings: { bars_s: 1.4, ranges_s: 0.3, options_s: 2.1, events_s: 1.2, total_s: 5.6 }, via: "daemon", research_not_advice: true,
    credibility_tiers: { 1: "primary — the issuer's own legal filing (SEC EDGAR)", 2: "wire / major outlet — newsroom with editorial standards, or an issuer press release carried by a wire", 3: "aggregator / financial blog — rewrites, summaries, opinion", 4: "social / unclassified — unverified or unknown source" },
    market: { status: "live", source: "regime engine (regime_engine.py persisted state, gating index SPY)", as_of: day(1), reason: null, composite: "RISK_ON", legacy_label: "BULL", exposure: 1.0, exposure_continuous: 0.94, trend: "UP", fast: "UP", vol: "NORMAL", breadth: "STRONG", confidence: 0.82, days_in_regime: 37, panic: false, allowed: ["momentum", "dip_buy"],
      stress: { vix: 16.4, vix_vix3m: 0.88, vix_vix3m_inverted: false, vrp_points: 3.1 }, spy: { close: 682.1, trend: trend(0.8, 2.1, 5.6, 17.9) } },
    sector: { status: "live", source: "static industry map (ticker_dossier.INDUSTRY_MAP)", as_of: day(1), reason: null, key: "space", label: "Space / aerospace", etf: "UFO", tried: ["UFO", "ARKX", "ITA"], trend: trend(2.4, 6.8, 14.2, 41.0), rs_vs_spy_20d: 4.7, rs_vs_spy_60d: 8.6, stock_vs_sector_20d: 9.3, stock_vs_sector_60d: 21.4 },
    stock: { status: "live", source: "Alpaca daily bars (split-adjusted) · dossier_ranges dr-har-ewma-v1", as_of: day(1), reason: null, price: px, price_day: day(0), price_is_partial_session: true, base_close: +(px / 1.012).toFixed(2), base_date: day(1),
      trend: trend(5.2, 16.1, 35.6, 212.4), realized_vol_20d_pct: 71, realized_vol_60d_pct: 78, high_52w: +(px * 1.08).toFixed(2), low_52w: +(px * 0.31).toFixed(2), vs_spy_20d: 14.0, vs_spy_60d: 30.0, sessions: 792,
      ranges: { "1": { h: 1, model: "har", fallback_reason: null, sigma_pct: s1 * 100, bands: band(s1), backtest: { h: 1, n: 250, in68: 178, in90: 221, rate68: 0.712, rate90: 0.884, target68: 0.6827, target90: 0.9, overlapping: false, models_used: { har: 250, ewma: 0 } } },
                "5": { h: 5, model: "har", fallback_reason: null, sigma_pct: s5 * 100, bands: band(s5), backtest: { h: 5, n: 250, in68: 171, in90: 219, rate68: 0.684, rate90: 0.876, target68: 0.6827, target90: 0.9, overlapping: true, models_used: { har: 250, ewma: 0 } } } },
      expected_move: { sigma_1d_pct: 4.52, median_abs_move_pct: 3.1, ratio_vs_median: 1.46, sigma_pctile_1y: 0.41, lookback_sessions: 252 },
      range_method: "68% and 90% bands of a zero-drift lognormal around the last close, sigma from a HAR-RV model on daily squared returns (EWMA lambda 0.94 when HAR is unavailable or implausible). Daily returns are fat-tailed: the backtest and the live ledger show how often the bands actually held." },
    bot_guards: { flags: { knife: 0, dip: 0, panic: 0, rs: 0, breadth: 0 }, reasons: {}, would_block: [], composite: "RISK_ON", as_of: day(1), source: "regime_engine.selection_verdict (read-only)" },
    direction: { status: "none", call: null, statement: "No validated direction signal for this ticker — over a day or a week, direction is close to a coin flip.", model_version: "fc-logit-v1", source: "weekly forecast screen (forecast.py, SHADOW, experimental)",
      model_accuracy: { n: 84, thin: false, hits: 45, hit_rate: 0.5357, brier: 0.2471, brier_skill_vs_coinflip: 0.0116, in_range_rate: 0.69 } },
    bot_track: { n: 11840, thin: false, source: "shadow portfolio (every scanned candidate, all decisions; +5d/+20d outcomes, nightly backfill)", base: { d5: { n: 11840, pos_rate: 0.507, mean_pct: 0.18, rule_win_rate: 0.39 }, d20: { n: 8210, pos_rate: 0.523, mean_pct: 0.61, rule_win_rate: null } },
      buckets: [dec(1, 20, 41, 1184, 0.49, -0.12), dec(2, 41, 47, 1184, 0.5, 0.05), dec(3, 47, 51, 1184, 0.5, 0.1), dec(4, 51, 54, 1184, 0.51, 0.14), dec(5, 54, 57, 1184, 0.5, 0.2), dec(6, 57, 60, 1184, 0.51, 0.22), dec(7, 60, 63, 1184, 0.52, 0.27), dec(8, 63, 66, 1184, 0.51, 0.3), dec(9, 66, 71, 1184, 0.52, 0.33), dec(10, 71, 92, 1184, 0.53, 0.41)] },
    plan: { entry_ref: px, horizon_sessions: 5, sigma_h: s5, stop, target: tgt, stretch_target: str, risk_per_share: +(px - stop).toFixed(4), reward_per_share: +(tgt - px).toFixed(4), reward_risk: +((tgt - px) / (px - stop)).toFixed(3), stretch_reward_risk: +((str - px) / (px - stop)).toFixed(3),
      stop_pct: +((stop / px - 1) * 100).toFixed(2), target_pct: +((tgt / px - 1) * 100).toFixed(2), stretch_pct: +((str / px - 1) * 100).toFixed(2), touch: touch(0.3612, 0.1083), touch_stretch: touch(0.0962, 0.1271),
      rules: { stop: "the 90% band's lower edge for the next 5 sessions, from the entry reference price — outside ~95% of driftless 5-session closes", target: "the 68% band's upper edge for the next 5 sessions (stretch: the 90% band's upper edge)", probabilities: "Monte Carlo of 4,000 driftless paths at 5-minute steps on the fitted daily vol; no drift assumed (no validated direction signal)" } },
    options: opts.optionsDown ? { status: "unavailable", source: "Alpaca options chain", as_of: null, reason: "HTTPError: 403 options feed", earnings: { status: "unavailable", reason: "no Finnhub key configured" } } : { status: "live", source: "Alpaca options chain (indicative feed; options_scanner)", as_of: iso(95_000), reason: null, contracts: 684, atm_iv: 82.4, atm_iv_30d: 76.9, front: { exp_date: day(-8), days_out: 8 },
      implied_move_1d_pct: 5.33, implied_move_5d_pct: 12.31, model_move_5d_pct: 10.31, model_move_1d_pct: 4.62, implied_vs_model_5d: 1.18, skew_put_minus_call_pts: 3.4, skew_strikes: "OTM puts at 88–95% of spot vs OTM calls at 105–112%, expiry nearest 30 days",
      earnings: { status: "live", date: day(-9), hour: "amc", source: "Finnhub earnings calendar" } },
    events: { status: "live", source: "SEC EDGAR submissions + Alpaca news", as_of: iso(95_000), reason: null, form4_count: 7, window_days: 60,
      history_note: "Event facts are raw. Where a filing type was measured (event-reaction study, 2016–2026), its average day-0–1 move is shown as history: every measured type was priced in on day 0–1 with no reliable drift after, so nothing here says an event is 'not yet priced in'. Lawsuits, other 8-K items and news were not measured.",
      event_study: STUDY, event_history: { insider_cluster: { label: "insider purchase clusters (≥2 officers/directors buying within 10 days)", n: 10090, car01_pct: 2.16, t01: 21.8, drift: "no reliable drift after (days 2–20 +0.8%, below the bar)" }, "8k_1.01": H101, "8k_4.02": { label: "8-K Item 4.02 (restatement)", n: 715, car01_pct: -1.51, t01: -3.4, drift: "no reliable drift after (the sample can only rule out drifts above ~3–4%)" } },
      items: [
        { kind: "news", title: `${sym} awarded launch contract for a constellation deployment`, time: iso(26 * 3600_000), date: day(1), source: "benzinga", tier: 3, url: "https://example.com/a", material: true },
        { kind: "filing", form: "8-K", title: "Entry into a material definitive agreement", items: [{ item: "1.01", label: "Entry into a material definitive agreement" }, { item: "9.01", label: "Financial statements and exhibits" }], time: iso(3 * 86400_000), date: day(3), source: "SEC EDGAR", tier: 1, url: "https://www.sec.gov/", material: true, history: { key: "8k_1.01", ...H101, study_path: STUDY.path, study_date: STUDY.date, study_window: STUDY.window, study_caveat: STUDY.caveat } },
        { kind: "news", title: `${sym} shares rise as analysts lift targets`, time: iso(4 * 86400_000), date: day(4), source: "Reuters", tier: 2, url: "https://example.com/b", material: true },
        { kind: "filing", form: "4", title: "Insider transaction (Form 4)", items: [], time: iso(6 * 86400_000), date: day(6), source: "SEC EDGAR", tier: 1, url: "https://www.sec.gov/", material: false },
        { kind: "filing", form: "424B5", title: "Prospectus supplement (offering)", items: [], time: iso(20 * 86400_000), date: day(20), source: "SEC EDGAR", tier: 1, url: "https://www.sec.gov/", material: true },
      ] },
    insiders: { windowDays: 30, filings: 7, buys: 0, sells: 5, buyShares: 0, sellShares: 412000, rows: [], source: "SEC EDGAR Form 4 archive", note: "Raw insider transactions. Form 4 buy-clustering failed ladder gate 2 in this system — not a signal." },
    ledger: { model_version: "dr-har-ewma-v1", open: 14, min_n: 30,
      all: { range: { "1": { n: 41, in68: 30, in90: 37, rate68: 0.7317, rate90: 0.9024 }, "5": { n: 12, in68: 8, in90: 11, rate68: 0.6667, rate90: 0.9167 } }, plan: { n: 9, target_first: 4, stop_first: 1, neither: 4, ambiguous: 0, target_first_rate: 0.4444, stop_first_rate: 0.1111, neither_rate: 0.4444, ambiguous_rate: 0 } },
      ticker: { range: { "1": { n: 2, in68: 1, in90: 2, rate68: 0.5, rate90: 1 }, "5": { n: 0, in68: 0, in90: 0, rate68: null, rate90: null } }, plan: { n: 0, target_first: 0, stop_first: 0, neither: 0, ambiguous: 0, target_first_rate: null, stop_first_rate: null, neither_rate: null, ambiguous_rate: null } } },
  };
}

// Weekly forecast screen (2026-10-27, SHADOW): GET /api/forecast in forecast.py
// report()'s real shape — this week's open calls (SPY, three positions, four
// candidates; two on chain IV30) and six earlier weeks resolved, so the track
// record, the live calibration dots and the week table all render. Fixture
// numbers only; deterministic.
function forecastFixture() {
  const rnd = (() => { let s = 7; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();
  const mk = (week, base, end, sym, group, spot, pUp, sigma, iv, score) => {
    const p = Math.round(pUp * 10000) / 10000;
    return { id: `${week}:${sym}`, week_key: week, symbol: sym, group, made_at: `${base}T20:21:00+00:00`, base_date: base, horizon_end: end,
      spot, p_up: p, direction: p >= 0.5 ? "up" : "down", p_call: Math.max(p, 1 - p),
      range_lo: Math.round(spot * Math.exp(-sigma) * 100) / 100, range_hi: Math.round(spot * Math.exp(sigma) * 100) / 100, band_coverage: 0.6827,
      sigma_h: sigma, vol_source: iv ? "chain_iv30" : "realized_20d", vol_annual: iv ?? Math.round(sigma / Math.sqrt(5) * Math.sqrt(252) * 1e4) / 1e4,
      features: { mom5_z: 0.4, mom20_z: 0.9, rev1_z: -0.3, spy_trend_z: 1.1 }, model_version: "fc-logit-v1", score };
  };
  const cur = [
    ["SPY", "index", 671.3, 0.561, 0.0201, 0.142], ["XLE", "position", 91.42, 0.472, 0.0312, null], ["KWEB", "position", 24.54, 0.538, 0.0455, 0.33],
    ["VXUS", "position", 71.4, 0.553, 0.0208, null], ["NVDA", "candidate", 184.2, 0.612, 0.0611, null], ["AMD", "candidate", 162.7, 0.587, 0.0702, null],
    ["PFE", "candidate", 27.1, 0.447, 0.0298, null], ["CRWV", "candidate", 98.6, 0.524, 0.1043, null],
  ].map(([s, g, spot, p, sig, iv]) => mk("2026-W44", "2026-10-23", "2026-10-30", s, g, spot, p, sig, iv, null));
  const past = [];
  const weeks = [["2026-W38", "2026-09-11", "2026-09-18"], ["2026-W39", "2026-09-18", "2026-09-25"], ["2026-W40", "2026-09-25", "2026-10-02"], ["2026-W41", "2026-10-02", "2026-10-09"], ["2026-W42", "2026-10-09", "2026-10-16"], ["2026-W43", "2026-10-16", "2026-10-23"]];
  for (const [w, base, end] of weeks) for (const [s, g] of [["SPY", "index"], ["XLE", "position"], ["KWEB", "position"], ["VXUS", "position"], ["NVDA", "candidate"], ["AMD", "candidate"], ["XLF", "candidate"]]) {
    const p = 0.38 + rnd() * 0.3, up = rnd() < p ? 1 : 0, sig = 0.02 + rnd() * 0.05, ret = (up ? 1 : -1) * rnd() * sig * 1.4;
    const f = mk(w, base, end, s, g, 100, p, sig, null, null);
    f.score = { kind: "score", id: f.id, status: "resolved", resolved_date: end, realized_close: Math.round(100 * Math.exp(ret) * 100) / 100, realized_ret_pct: Math.round((Math.exp(ret) - 1) * 1e4) / 100,
      outcome_up: up, hit: (f.direction === "up") === (up === 1) ? 1 : 0, in_range: Math.abs(ret) <= sig ? 1 : 0, brier: Math.round((p - up) ** 2 * 1e6) / 1e6 };
    past.push(f);
  }
  const acc = (rows) => { const n = rows.length, hits = rows.reduce((a, f) => a + f.score.hit, 0), inr = rows.reduce((a, f) => a + f.score.in_range, 0), b = n ? Math.round(rows.reduce((a, f) => a + f.score.brier, 0) / n * 1e5) / 1e5 : null;
    return { n, thin: n < 30, hits, hit_rate: n ? Math.round(hits / n * 1e4) / 1e4 : null, brier: b, brier_coinflip: 0.25, brier_skill_vs_coinflip: b === null ? null : Math.round((1 - b / 0.25) * 1e4) / 1e4, in_range: inr, in_range_rate: n ? Math.round(inr / n * 1e4) / 1e4 : null, in_range_target: 0.6827, up_rate: n ? Math.round(rows.reduce((a, f) => a + f.score.outcome_up, 0) / n * 1e4) / 1e4 : null }; };
  const edges = [0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1];
  const bins = (pairs) => edges.slice(0, -1).map((lo, k) => { const inb = pairs.filter(([p]) => p >= lo && p < edges[k + 1]); const n = inb.length;
    return { lo, hi: edges[k + 1], n, mean_p: n ? Math.round(inb.reduce((a, [p]) => a + p, 0) / n * 1e4) / 1e4 : null, freq: n ? Math.round(inb.reduce((a, [, o]) => a + o, 0) / n * 1e4) / 1e4 : null }; });
  const holdout = []; for (let k = 0; k < 640; k++) { const p = 0.42 + rnd() * 0.22; holdout.push([p, rnd() < p + (rnd() - 0.5) * 0.06 ? 1 : 0]); }
  const a = acc(past);
  const byGroup = Object.fromEntries(["index", "position", "candidate"].map((g) => [g, acc(past.filter((f) => f.group === g))]));
  return {
    generated_at: new Date(Date.now() - 40_000).toISOString(), model_version: "fc-logit-v1", experimental: true, shadow: true, horizon_days: 5, band_coverage: 0.6827,
    latest: { week_key: "2026-W44", made_at: new Date(Date.now() - 2 * 86400_000).toISOString(), base_date: "2026-10-23", horizon_end: "2026-10-30", on_time: true, model_version: "fc-logit-v1",
      model: { coef: [0.1412, -0.0331, 0.0198, -0.0254, 0.0412], features: ["mom5_z", "mom20_z", "rev1_z", "spy_trend_z"], n_samples: 3120, first_date: "2023-10-20", last_date: "2026-10-16",
        validation: { split_date: "2025-12-04", n_train: 2160, n_test: 940, train_base_rate: 0.5532, test_base_rate: 0.5617, brier: 0.24511, brier_climatology: 0.24622, brier_skill: 0.0045, log_loss: 0.68321, hit_rate: 0.5574, calibration: bins(holdout) } },
      train_tickers: ["AMD", "CRWV", "DIA", "IWM", "KWEB", "NVDA", "PFE", "QQQ", "SPY", "VXUS", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
      n_forecasts: cur.length, skipped: [], candidates_available: true,
      candidates_scan_at: new Date(Date.now() - 2 * 86400_000 - 3600_000).toISOString(), iv_day: "2026-10-23", iv_names: 2,
      forecasts: cur.map(({ features, ...f }) => f) },
    accuracy: { ...a, on_time: a, by_group: byGroup, by_vol_source: { chain_iv30: acc([]), realized_20d: a } },
    calibration: { edges, bins: bins(past.map((f) => [f.p_up, f.score.outcome_up])) },
    weeks: weeks.slice().reverse().map(([w, base, end]) => { const rs = past.filter((f) => f.week_key === w); const x = acc(rs);
      return { week_key: w, made_at: `${base}T20:21:00+00:00`, base_date: base, horizon_end: end, on_time: true, model_version: "fc-logit-v1", n: rs.length, resolved: x.n, hits: x.hits, hit_rate: x.hit_rate, brier: x.brier, in_range_rate: x.in_range_rate }; }),
    open: cur.length,
    method: "P(up) = ridge logistic regression on vol-scaled 5- and 20-session momentum, 1-session reversal and SPY's distance from its 50-day average, fit weekly on 3 years of daily bars (15 index/sector ETFs + the week's names, one sample per 5 sessions), validated on a time-ordered 30% holdout. Range = 68% band of a zero-drift lognormal from chain IV30 when archived, else 20-session realized vol.",
    caveats: ["Experimental model, shadow only: nothing trades on these calls.", "Training names exist today (survivorship bias); edges at a 1-week horizon are small, so most probabilities sit near the base rate — read the holdout Brier skill before trusting any call.", "Accuracy figures under 30 resolved calls are marked thin."],
    via: "daemon",
    daily: forecastDailyFixture(),
  };
}

// Daily horizon (Part C, 2026-11-04): forecast.py daily_report()'s real shape —
// the calls for the next close (SPY on VIX, three positions, five candidates,
// two of them made intraday at the live price), each with its live band
// mid-session, and 25 earlier sessions resolved so the daily track record, the
// calibration dot and the by-session table render. Fixture numbers only.
function forecastDailyFixture() {
  const rnd = (() => { let s = 11; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();
  const BASE = 0.5452;
  const now = Date.now();
  const mk = (session, base, sym, group, spot, q1, src, timing, f, score, live) => {
    const p = Math.round((0.5 + (BASE - 0.5) * Math.sqrt(f)) * 10000) / 10000, q = Math.round(q1 * Math.sqrt(f) * 1e6) / 1e6;
    return { id: `D:${session}:${sym}`, horizon: "daily", session, symbol: sym, group, made_at: new Date(now - (timing === "pre_open" ? 15 : 2) * 3600_000).toISOString(),
      timing, on_time: timing === "pre_open", base_date: base, base_close: spot, spot: timing === "pre_open" ? spot : Math.round(spot * 1.004 * 100) / 100, horizon_end: session,
      p_up: p, p_base: p, direction: p >= 0.5 ? "up" : "down", p_call: Math.max(p, 1 - p), base_rate: BASE, direction_model: "base-rate-v1",
      remaining_fraction: f, q, q_1d: q1, q_clim: Math.round(q1 * 0.93 * Math.sqrt(f) * 1e6) / 1e6,
      range_lo: Math.round(spot * Math.exp(-q) * 100) / 100, range_hi: Math.round(spot * Math.exp(q) * 100) / 100, band_coverage: 0.6827,
      range_source: src, range_model_version: "fc-range-har-v1", iv: src === "har" ? null : (src === "har+vix" ? 0.2083 : 0.41), iv_day: src === "har" ? null : base,
      model_version: "fc-daily-v1", score, live };
  };
  const liveOf = (spot, q1, f) => { const px = Math.round(spot * (1 + (rnd() - 0.45) * 0.012) * 100) / 100, q = q1 * Math.sqrt(f);
    return { price: px, price_day: "2026-11-04", at: new Date(now - 6 * 60_000).toISOString(), remaining_fraction: f, q: Math.round(q * 1e6) / 1e6,
      lo: Math.round(px * Math.exp(-q) * 100) / 100, hi: Math.round(px * Math.exp(q) * 100) / 100, vs_spot_pct: Math.round((px / spot - 1) * 1e5) / 1e3 }; };
  const F_NOW = 0.41;
  const cur = [
    ["SPY", "index", 754.35, 0.0079, "har+vix", "pre_open"], ["XLE", "position", 91.42, 0.0121, "har", "pre_open"], ["KWEB", "position", 24.54, 0.0189, "har+chain_iv30", "pre_open"],
    ["VXUS", "position", 71.4, 0.0074, "har", "pre_open"], ["NVDA", "candidate", 206.69, 0.0228, "har+chain_iv30", "pre_open"], ["AMD", "candidate", 162.7, 0.0251, "har", "pre_open"],
    ["RKLB", "candidate", 61.2, 0.0402, "har", "pre_open"], ["PFE", "candidate", 27.1, 0.0109, "har", "intraday"], ["CRWV", "candidate", 98.6, 0.0388, "har", "intraday"],
  ].map(([s, g, spot, q1, src, timing]) => mk("2026-11-04", "2026-11-03", s, g, spot, q1, src, timing, timing === "pre_open" ? 1 : 0.62, null, liveOf(spot, q1, F_NOW)));
  const past = [];
  const sessions = [];
  const d0 = new Date("2026-11-03T12:00:00Z");
  for (let k = 0, day = new Date(d0); sessions.length < 25; k++) {
    if (day.getUTCDay() !== 0 && day.getUTCDay() !== 6) sessions.push(day.toISOString().slice(0, 10));
    day = new Date(day.getTime() - 86400_000);
  }
  for (let i = 0; i < sessions.length; i++) {
    const s = sessions[i], base = sessions[i + 1] ?? "2026-10-07";
    for (const [sym, g, q1, src] of [["SPY", "index", 0.0081, "har+vix"], ["XLE", "position", 0.0124, "har"], ["KWEB", "position", 0.0192, "har+chain_iv30"], ["VXUS", "position", 0.0076, "har"], ["NVDA", "candidate", 0.0231, "har+chain_iv30"], ["AMD", "candidate", 0.0249, "har"], ["XLF", "candidate", 0.0098, "har"]]) {
      const y = (rnd() - 0.5) * q1 * 3.1, up = y > 0 ? 1 : 0, inr = Math.abs(y) <= q1 ? 1 : 0;
      const isv = 2 * q1 + (2 / 0.3173) * Math.max(Math.abs(y) - q1, 0), isc = 2 * q1 * 0.93 + (2 / 0.3173) * Math.max(Math.abs(y) - q1 * 0.93, 0) * 1.25;
      const f = mk(s, base, sym, g, 100, q1, src, "pre_open", 1, null, null);
      f.score = { kind: "score", horizon: "daily", id: f.id, status: "resolved", resolved_date: s, realized_close: Math.round(100 * Math.exp(y) * 100) / 100,
        realized_ret_pct: Math.round((Math.exp(y) - 1) * 1e4) / 100, outcome_up: up, hit: up, in_range: inr,
        brier: Math.round((f.p_up - up) ** 2 * 1e6) / 1e6, brier_base: Math.round((f.p_up - up) ** 2 * 1e6) / 1e6,
        interval_score: Math.round(isv * 100 * 1e5) / 1e5, interval_score_clim: Math.round(isc * 100 * 1e5) / 1e5 };
      past.push(f);
    }
  }
  const acc = (rows) => {
    const n = rows.length; const sum = (fn) => rows.reduce((a, f) => a + fn(f), 0);
    const b = n ? Math.round(sum((f) => f.score.brier) / n * 1e5) / 1e5 : null;
    const isv = n ? sum((f) => f.score.interval_score) / n : null, isc = n ? sum((f) => f.score.interval_score_clim) / n : null;
    return { n, thin: n < 30, hits: sum((f) => f.score.hit), hit_rate: n ? Math.round(sum((f) => f.score.hit) / n * 1e4) / 1e4 : null, up_rate: n ? Math.round(sum((f) => f.score.outcome_up) / n * 1e4) / 1e4 : null,
      brier: b, brier_base_rate: b, brier_coinflip: 0.25, bss_vs_base_rate: n ? 0 : null, bss_vs_coinflip: b === null ? null : Math.round((1 - b / 0.25) * 1e4) / 1e4,
      in_range: sum((f) => f.score.in_range), in_range_rate: n ? Math.round(sum((f) => f.score.in_range) / n * 1e4) / 1e4 : null, in_range_target: 0.6827,
      interval_score_pct: isv === null ? null : Math.round(isv * 1e5) / 1e5, interval_score_clim_pct: isc === null ? null : Math.round(isc * 1e5) / 1e5,
      range_skill_vs_constant: isv === null ? null : Math.round((1 - isv / isc) * 1e4) / 1e4, range_skill_n: n, width_pct: n ? Math.round(sum((f) => 200 * f.q) / n * 1e4) / 1e4 : null };
  };
  const edges = [0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1];
  const bins = (pairs) => edges.slice(0, -1).map((lo, k) => { const inb = pairs.filter(([p]) => p >= lo && p < edges[k + 1]); const n = inb.length;
    return { lo, hi: edges[k + 1], n, mean_p: n ? Math.round(inb.reduce((a, [p]) => a + p, 0) / n * 1e4) / 1e4 : null, freq: n ? Math.round(inb.reduce((a, [, o]) => a + o, 0) / n * 1e4) / 1e4 : null }; });
  const a = acc(past);
  return {
    model_version: "fc-daily-v1", range_model_version: "fc-range-har-v1", direction_model: "base-rate-v1", direction_features: [], horizon: "next session close", band_coverage: 0.6827,
    latest: { session: "2026-11-04", made_at: new Date(now - 15 * 3600_000).toISOString(), base_date: "2026-11-03", timing: "pre_open", on_time: true,
      model_version: "fc-daily-v1", range_model_version: "fc-range-har-v1", direction_model: "base-rate-v1", direction_features: [],
      base_rate: { p: BASE, n: 11340, first: "2023-10-26", last: "2026-11-03", basket: ["SPY", "QQQ"] }, vix: { value: 20.83, day: "2026-11-03" }, vix_used: true,
      iv_day: "2026-11-03", n_forecasts: cur.length, skipped: [], candidates_available: true, candidates_scan_at: new Date(now - 16 * 3600_000).toISOString(),
      range_sources: { "har+vix": 1, "har+chain_iv30": 2, har: 6 }, forecasts: cur },
    live: { session: "2026-11-04", made_at: new Date(now - 6 * 60_000).toISOString(), et_minutes: 731 },
    accuracy: { ...a, by_timing: { pre_open: a, intraday: acc([]) }, by_group: { index: acc(past.filter((f) => f.group === "index")), position: acc(past.filter((f) => f.group === "position")), candidate: acc(past.filter((f) => f.group === "candidate")) },
      by_range_source: { "har+vix": acc(past.filter((f) => f.range_source === "har+vix")), "har+chain_iv30": acc(past.filter((f) => f.range_source === "har+chain_iv30")), har: acc(past.filter((f) => f.range_source === "har")) } },
    calibration: { edges, bins: bins(past.map((f) => [f.p_up, f.score.outcome_up])) },
    sessions: sessions.map((s) => { const rs = past.filter((f) => f.session === s); const x = acc(rs);
      return { session: s, n: rs.length, resolved: x.n, hits: x.hits, hit_rate: x.hit_rate, brier: x.brier, in_range_rate: x.in_range_rate, interval_score_pct: x.interval_score_pct, range_skill_vs_constant: x.range_skill_vs_constant }; }),
    open: cur.length,
    study: { range: { doc: "research/forecast_daily_study_2026-11-04.md", holdout: "2019-01-02 → 2026-11-02", spy_coverage: 0.7081, spy_skill_vs_constant: 0.0793, spy_skill_vs_current: 0.0604, iv_names_coverage: 0.6838, iv_names_skill_vs_constant: 0.0445, names_coverage: 0.6579, names_skill_vs_constant: 0.0386 },
      direction: { doc: "research/forecast_direction_study_2026-11-04.md", holdout: "2019-01-02 → 2026-11-02", tests: 14, passed: 0, spy_base_rate: 0.5508, spy_brier_base_rate: 0.24767, spy_skill_vs_coinflip: 0.0093 } },
    method: "Direction: P(up) = the point-in-time base rate (fixture).",
    caveats: ["Experimental model, shadow only: nothing trades on these calls."],
  };
}

// Brain page (2026-10-29): GET /api/bot/brain in server/brain.ts brainBody()'s
// real shape — components across every kind and status, each short-term kind
// (bucket vs SPY, source IC with its BH verdict, not-measured), long-term doc
// evidence and gaps, all three agreement flags, forward selection (nothing
// selected + insufficient arm) and an ablation table. Fixture numbers only.
function brainFixture() {
  const now = Date.now();
  const iso = (ms) => new Date(now - ms).toISOString();
  const evAt = iso(4 * 60_000);
  const noShort = (reason) => ({ measured: false, reason, what: "—", source: "brain_evidence.py over the shadow log", as_of: evAt, sign: "insufficient", sign_note: "not measured" });
  const svShort = (name, n, ic, ci, q, verdict, reason, sign) => ({ measured: true, what: `rank IC of ${name} vs the +5d outcome`, metric: "ic", n, n_days: 74, ic, ci95: ci, q_bh: q, verdict, verdict_reason: reason, outcome: "+5d", window: { first: "2026-06-02", last: "2026-10-21" }, source: "source_validation.py (source_validation_v2 (2026-10-27)), primary outcome +5d, BH FDR 0.10", as_of: evAt, sign, sign_note: `${verdict}: ${reason}` });
  const bucketShort = (label, bucket, n, mean, ci, ex, exci, sign, note) => ({ measured: true, what: label, metric: "mean_return", bucket, outcome: "xa (exit-aligned: the live exit ladder, long stock)", n, n_days: 61, mean_pct: mean, hit_rate: 0.44, ci95: ci, vs_spy: { n, mean_excess_pct: ex, ci95: exci, beat_rate: 0.41 }, horizons: { xa: { n, mean_pct: mean }, "+5d": { n: Math.round(n * 0.95), mean_pct: mean / 2 }, "+20d": { n: Math.round(n * 0.8), mean_pct: mean * 1.4 } }, window: { first: "2026-06-03", last: "2026-10-02" }, source: "brain_evidence.py over the shadow log", as_of: evAt, sign, sign_note: note });
  const noLong = { measured: false, reason: "Not yet measured: no multi-year test of the stock score or its inputs exists — no point-in-time universe (docs/regime/FINDINGS.md); a 2016+ walk-forward panel harness is not built yet.", items: [], sign: "insufficient", sign_note: "no long-horizon test exists" };
  const c = (id, name, kind, group, does, status, status_source, detail, short_term, long_term, flag, note) => ({ id, name, kind, group, does, inventory_ref: null, status, status_source, detail, short_term, long_term, agreement: { flag, note } });
  const components = [
    c("stock_entries", "Stock entries", "sleeve", "Sleeves", "New stock/ETF positions from the scan (executeTrades, morning queue, Tier-3).", "shadow", "config", 'VOLTRADE_SHADOW_ENTRIES="stocks,options" (config): would-be entries are logged, no order',
      bucketShort("would-be stock entries (shadow) — or taken, when live", "shadow_mode_entry:stocks", 18, 0.42, null, 0.1, null, "insufficient", "n=18 (< 30 or no clustered CI)"), noLong, "insufficient", "short-term (live shadow) evidence is missing or too thin"),
    c("floor_basket", "Floor basket (QQQ/SMH/KWEB/VXUS)", "sleeve", "Sleeves", "Regime-scheduled passive core: 70% BULL / 90% NEUTRAL / 35% CAUTION / 0% BEAR-PANIC.", "live", "config", "FLOOR_BASKET_ENABLED=true; never gated by entry shadow (entry_shadow.py)",
      noShort("Not measured here: the sleeve ledger reports dollars, not a daily return series vs same-exposure SPY (research/floor_basket_decision.md)."),
      { measured: true, items: [{ id: "floor_composition", title: "Floor basket vs same-exposure SPY (simulation)", source_file: "research/floor_basket_decision.md", date: "2026-10-27", horizon: "2014-11 → 2026-10, 2,990 days, 5 bp/side", finding: "Basket CAGR 7.82%, max DD -21.6%, Sharpe 0.733 ± 0.291; Sharpe minus same-exposure SPY -0.109 [95% -0.375, +0.156].", compared_to: "SPY held at the floor's own exposure", sign: "null", variants_tried: 4 }, { id: "floor_schedule", title: "The floor's regime exposure schedule (simulation)", source_file: "research/floor_basket_decision.md", date: "2026-10-27", horizon: "2014-11 → 2026-10", finding: "SPY at the floor's regime-driven exposure made 7.41%/yr vs 13.58% buy-and-hold at Sharpe 0.843 vs 0.816.", compared_to: "buy-and-hold SPY", sign: "negative" }], sign: "null", sign_note: "vs same-exposure SPY" }, "insufficient", "short-term (live shadow) evidence is missing or too thin"),
    c("convexity", "QQQ-put convexity overlay", "sleeve", "Sleeves", "Far-OTM QQQ puts, 60 DTE, rolled at 21 DTE, 2–4% of equity.", "shadow", "config", "runs every scan (CONVEXITY_OVERLAY.enabled=false is never read); buy-to-open is logged, not ordered (options sleeve shadowed)",
      noShort("Not measured: option P&L (no options simulator)."), { measured: true, items: [{ id: "convexity_10yr", title: "QQQ-put convexity overlay, 10-year backtest", source_file: "system_config.py", date: "2026-04-17", horizon: "10 years", finding: "The overlay bled $307K-$508K in annual premium against 1-2 crash payoffs that did not recoup it.", compared_to: "no overlay", sign: "negative" }], sign: "negative", sign_note: "vs no overlay" }, "insufficient", "short-term (live shadow) evidence is missing or too thin"),
    c("ml_v2", "ML v2 (LightGBM) blend", "model", "Models", "Win probability blended into the score at ~0.28–0.40 weight.", "shadow", "config", "computed every scan and logged; nothing it touches is ordered while both entry sleeves are shadowed",
      { ...svShort("ml_only_score", 912, 0.012, [-0.031, 0.054], undefined, undefined, "", "null"), verdict: undefined, sign_note: "IC 95% CI includes 0 (reference row, outside the BH family)" },
      { measured: true, items: [{ id: "ml_v2_honest", title: "ML v2 procedure with the CV leak removed", source_file: "research/ml_v3_eval.md", date: "2026-10-27", horizon: "2023-02-02 → 2026-10-16, 97 tickers", finding: "Reported AUC 0.658 falls to 0.515 with date folds and an honest embargo — the reported skill is the leak.", compared_to: "a constant base-rate forecast", sign: "null" }], sign: "null", sign_note: "vs base rate" }, "agree", "both say no measurable effect"),
    c("strategy_weights", "\"LEARN\" strategy-weight loop", "model", "Models", "Shifts momentum/VRP weights from recent win rate — written and audited, never read.", "off", "code", "server/bot.ts strategyWeights is write-only: the score uses hard-coded weights (bot_engine.py)",
      noShort("Off (dead code): reads nothing, changes nothing."), { measured: false, reason: "Not measured (dead).", items: [], sign: "insufficient", sign_note: "no long-horizon test exists" }, "insufficient", "neither horizon has a usable test"),
    c("min_score", "Minimum score gate", "rule", "Entry gates", "Adaptive MIN_SCORE 63/63/67/75 (Node floor 65): candidates below it are rejected.", "shadow", "config", 'VOLTRADE_SHADOW_ENTRIES="stocks,options" (config): would-be entries are logged, no order',
      bucketShort("candidates it rejected", "rejected_score", 1240, -0.61, [-0.92, -0.33], -0.74, [-1.05, -0.44], "positive", "blocked names did worse (excess vs SPY CI < 0): the rule earns its keep"), noLong, "insufficient", "no long-horizon test exists"),
    c("sel_knife", "Regime selection guard: knife", "rule", "Regime engine", "Blocks names below 75% of their 52-week high or at a new 20-day low.", "shadow", "config", "gate on, but stock entries are shadowed: it blocks would-be entries only",
      bucketShort("candidates it blocked (or would have)", "rejected_regime_knife", 64, 0.88, [0.21, 1.6], 0.71, [0.08, 1.33], "negative", "blocked names did better (excess vs SPY CI > 0): the rule blocks winners"),
      { measured: false, reason: "Not yet measured in our data: the guards cite published research, but the stock sleeve cannot be backtested here; first forward read is due 2026-11-16.", items: [], sign: "insufficient", sign_note: "no long-horizon test exists" }, "insufficient", "no long-horizon test exists"),
    c("regime_sizing", "Regime exposure multiplier (sizing)", "rule", "Regime engine", "One exposure multiplier replaces the VIX scalar, the stress multiplier and the Tier-1 VXX ladder.", "shadow", "config", "sleeves stock, options, tier1 (VOLTRADE_REGIME_SIZING, config); nothing it touches is ordered while both entry sleeves are shadowed",
      noShort("Not measured from the shadow log: outcomes are per-candidate returns, not position sizes."), { measured: true, items: [{ id: "regime_backtest", title: "Regime-layer backtest (index level, 12 arms)", source_file: "research/regime_backtest_2026-10-28.md", date: "2026-10-28", horizon: "2007-08-28 → 2026-10-26, SPY", finding: "engine_quantized: CAGR 8.34%, max DD -20.5%, Sharpe 0.76 vs the old schedule 4.30% / -20.2% / 0.56; buy-and-hold 11.05% / -55.2% / 0.63.", compared_to: "the regime schedule it replaced (current_proxy)", sign: "positive", variants_tried: 12 }], sign: "positive", sign_note: "vs current_proxy" }, "insufficient", "short-term (live shadow) evidence is missing or too thin"),
    c("news", "Headline sentiment", "input", "News & social", "Alpaca news keyword score: +10/+5/-6/-12 points on the stock score.", "shadow", "config", "computed every scan and logged; nothing it touches is ordered while both entry sleeves are shadowed",
      svShort("adj.news", 912, -0.087, [-0.139, -0.035], 0.012, "FAIL_INVERTED", "BH-significant NEGATIVE IC: the + this adjustment adds predicts WORSE outcomes", "negative"), noLong, "insufficient", "no long-horizon test exists"),
    c("alt", "Alt-data composite", "input", "Alternative data", "Wikipedia attention, short-volume proxy, FTD, GDELT, FRED flags: ±20 points.", "shadow", "config", "computed every scan and logged; nothing it touches is ordered while both entry sleeves are shadowed",
      svShort("adj.alt", 912, 0.011, [-0.04, 0.063], 0.81, "FAIL", "no BH-significant positive IC above the floor", "null"), noLong, "insufficient", "no long-horizon test exists"),
    c("google_trends", "Google Trends", "input", "News & social", "Would weight 35% of the social score; hard-coded to 0 (rate limits).", "off", "code", "disabled in social_data.py (signal fixed at 0); its 35% weight slot still exists",
      noShort("Off: constant 0, nothing to measure."), { measured: false, reason: "Not measured (off).", items: [], sign: "insufficient", sign_note: "no long-horizon test exists" }, "insufficient", "neither horizon has a usable test"),
  ];
  const counts = { live: 0, shadow: 0, off: 0 };
  for (const x of components) counts[x.status] += 1;
  return {
    generated_at: iso(3_000), evidence_generated_at: evAt, evidence_via: "daemon", evidence_error: null, evidence_cache_age_s: 240,
    explainer: "Short-term = live shadow outcomes (what the bot's own candidates did after it scored them); long-term = multi-year tests. A component is promoted only when both agree.",
    rule: "Live beats backtest when they disagree: a component is promoted only when short-term (live shadow) and long-term (multi-year) evidence agree; when they disagree, the live result wins and the gap is itself the next question.",
    status_counts: counts,
    switches: { config_source: "python", entry_shadow: { stocks: true, options: true, value: "stocks,options", source: "config" }, regime: { VOLTRADE_REGIME_ENGINE: { value: "1", source: "config" } } },
    short_term: { window: { first: "2026-04-20", last: "2026-10-28" }, records: 19874, spy: { ok: true, first_day: "2026-04-13", last_day: "2026-10-28" }, primary_outcome: "xa = exit-aligned ruler (the live exit ladder) for buckets; +5d path ruler for source ICs (source_validation's pre-registered primary)", note: "One first-seen record per (ticker, ET entry day). Long-stock returns of the scored name — never option P&L. 95% CIs are day-clustered bootstraps." },
    long_term: { sources: [{ file: "research/regime_backtest_2026-10-28.md", date: "2026-10-28" }, { file: "research/ml_v3_eval.md", date: "2026-10-27" }, { file: "research/floor_basket_decision.md", date: "2026-10-27" }], gaps: ["No portfolio walk-forward harness on a 2016+ point-in-time panel exists yet: the stock score, its inputs and the stock sleeve have no multi-year test.", "No options simulator: options entries, CSP guards and the convexity overlay have no option-P&L test (KB #48).", "Kill-switch thresholds have no counterfactual logging."] },
    components,
    combinations: {
      short_term: {
        available: true, reason: null, as_of: evAt, source: "source_validation.py forward selection (day-split fit/select/holdout; each step's p-value Bonferroni-multiplied by the candidates tried) + ablation", family: "Benjamini-Hochberg across 52 sources at FDR 0.1",
        summary: { PASS: [], FAIL_INVERTED: ["adj.news"], FAIL: ["adj.alt"], INSUFFICIENT: [], CONSTANT: [] },
        arms: [
          { arm: "features", label: "ML features only (all deduped rows)", status: "NOTHING_SELECTED", reason: "no source added significant out-of-sample IC on the SELECT block", n_rows: 884, candidates: 25, variants_tried: 25, selected: [], weights: null, steps: [{ candidate: "feat.price_vs_52w_high", select_ic: 0.061, gain: 0.061, se: 0.034, p_bonferroni: 0.906, tried: 25, accepted: false }], holdout_selected: null, holdout_live_score: { n: 221, n_days: 19, ic: -0.044, ci95: [-0.12, 0.03] }, holdout_vs_live_score: null, holdout_window: { first: "2026-09-12", last: "2026-10-21" } },
          { arm: "breakdown", label: "score adjustments + features (rows with a logged breakdown)", status: "INSUFFICIENT", reason: "212 rows < MIN_SELECT_ROWS=300", n_rows: 212, candidates: 39, variants_tried: 0, selected: [], weights: null, steps: [], holdout_selected: null, holdout_live_score: null, holdout_vs_live_score: null, holdout_window: null },
        ],
        ablation: { status: "OK", reason: null, n_rows: 341, baseline_inverted: true, un_inverts: true, improvement_significant: false, rows: [
          { name: "baseline", removed: [], n: 341, ic: -0.052, ci95: [-0.11, 0.004], ic_change_vs_baseline: null, ic_change_se: null, q5_minus_q1_pct: -0.67, status: null },
          { name: "drop_unvalidated", removed: ["intel", "alt", "social", "news", "fundamental"], n: 341, ic: 0.018, ci95: [-0.04, 0.07], ic_change_vs_baseline: 0.07, ic_change_se: 0.041, q5_minus_q1_pct: 0.12, status: null },
          { name: "drop_news", removed: ["news"], n: 341, ic: -0.021, ci95: [-0.08, 0.035], ic_change_vs_baseline: 0.031, ic_change_se: 0.02, q5_minus_q1_pct: -0.31, status: null },
        ] },
      },
      long_term: [
        { id: "regime_arms", title: "Regime-layer arms (SPY, full window)", source_file: "research/regime_backtest_2026-10-28.md", date: "2026-10-28", horizon: "2007-08-28 → 2026-10-26", variants_tried: 12, discount: "No deflated Sharpe in the source: 12 arms on one history, so the best arm's edge is optimistic.", columns: ["Arm", "CAGR", "Max DD", "Sharpe"], rows: [["vol_target_15", "10.13%", "-34.7%", "0.78"], ["engine_quantized (live)", "8.34%", "-20.5%", "0.76"], ["buy_and_hold", "11.05%", "-55.2%", "0.63"], ["current_proxy (old)", "4.30%", "-20.2%", "0.56"]] },
        { id: "floor_arms", title: "Floor composition arms", source_file: "research/floor_basket_decision.md", date: "2026-10-27", horizon: "2014-11 → 2026-10", variants_tried: 4, discount: "Deflated-Sharpe bar for 4 arms is 0.08 — cleared only by qqq_only.", columns: ["Arm", "CAGR", "Max DD", "Sharpe ± SE"], rows: [["qqq_only", "9.76%", "-23.6%", "0.838 ± 0.291"], ["basket (live)", "7.82%", "-21.6%", "0.733 ± 0.291"]] },
      ],
    },
  };
}

// TRADE WORLD fixtures (2026-10-29): real public API responses captured from
// voltradeai.com that day, trimmed, so the screenshots show the moving-map
// backdrop with loaded cards rather than empty states.
const tradeFx = (name) => JSON.parse(readFileSync(path.join(ROOT, "scripts", "fixtures", name), "utf8"));
const OWNER_ME = { authenticated: true, isOwner: true, user: { email: "owner@example.com", tier: "pro", role: "owner" } };
const TRADE_LOADED = { "/api/analyze/SPY": tradeFx("trade_analyze_spy.json"), "/api/research/SPY": tradeFx("trade_research_spy.json"), "/api/insights/SPY": tradeFx("trade_insights_spy.json"), "/api/market/scanner": tradeFx("trade_scanner.json"), "/api/market/news": tradeFx("trade_news.json"), "/api/watchlist": tradeFx("trade_watchlist_empty.json") };

// A truncated filing-history page and its older continuation, in the shape
// server/filingHistoryPage.ts serves (window / covered / truncated / next).
// First request (no `to=`): 2 of 30 days. `?from=…&to=2026-11-13`: the rest.
// The Form 4 page asks by FILED day (2026-11-18): the first page covers
// Mon 11-16 → Wed 11-18 and continues at the Friday before (the live shape,
// read from production 2026-11-18), with one day complete against EDGAR's
// index, one partial and one not yet checked on each page.
// The 13F page asks the same way since 2026-11-21 (a 45-day window, cap 500).
function filedDayForm4History(req, form4, kind = "form4") {
  const older = /[?&]filed_to=2026-11-13/.test(req.url || "");
  const is13f = kind === "13f";
  const from = is13f ? "2026-10-05" : "2026-10-20";
  // a 13F row carries a 10-character filed day on production; a Form 4 row a timestamp
  const mk = (i, d) => (is13f ? { ...form4(i, d), filedAt: d } : form4(i, d));
  const filings = older
    ? [mk(3, "2026-11-13"), mk(4, "2026-11-06"), mk(5, "2026-10-20")]
    : [mk(0, "2026-11-18"), mk(1, "2026-11-17"), mk(2, "2026-11-16")];
  const day = (d, n, index, complete) => ({ day: d, filings: n, omitted: 0, edgar_index: index, complete, checked_at: index == null ? null : "2026-11-18T07:15:41.235Z" });
  return {
    kind: "raw", source: is13f ? "SEC EDGAR (13F-HR) — accumulated archive (began 2026-07-05) (fixture)" : "SEC EDGAR (Form 4) — accumulated archive (began 2026-07-04) (fixture)",
    ...(is13f ? { focused_cap: 250 } : {}),
    window_by: "filed_day", days: null,
    window: older ? { from, to: "2026-11-13" } : { from, to: "2026-11-18" },
    covered: older ? { from, to: "2026-11-13" } : { from: "2026-11-16", to: "2026-11-18" },
    truncated: !older, next: older ? null : `?filed_from=${from}&filed_to=2026-11-13`,
    max_filings: is13f ? 500 : 2000, days_scanned: 3, archive_start: is13f ? "2026-07-05" : "2026-07-04", malformed_lines: 0, archive_days_read: 3, omitted: 0,
    filed_days: older
      ? [day("2026-11-13", 1, 1, true), day("2026-11-06", 1, 515, false), day("2026-10-20", 1, null, null)]
      : [day("2026-11-18", 1, null, null), day("2026-11-17", 1, 1, true), day("2026-11-16", 1, 624, false)],
    complete_days: 1, count: filings.length, filings,
  };
}

// FILED DAYS AT REAL VOLUME (2026-11-18, item (l)): the shape production
// answered at 18:10Z that day — three filed days of 94 / 1,047 / 624 filings
// holding 264 / 2,800 / 1,333 transaction rows — built here, not copied. The
// page used to show the first 500 rows of it (today and 236 rows of
// yesterday); every day must now be a line that opens.
const VOLUME_DAYS = [["2026-11-18", 94, 264, null], ["2026-11-17", 1047, 2800, 1047], ["2026-11-16", 624, 1333, 624]];
function volumeForm4History() {
  const filings = [];
  let i = 0;
  for (const [day, nFilings, nRows] of VOLUME_DAYS) {
    for (let k = 0; k < nFilings; k++, i++) {
      // rows spread over the day's filings: base each, the remainder on the first ones
      const tx = Math.floor(nRows / nFilings) + (k < nRows % nFilings ? 1 : 0);
      filings.push({
        accession: `0000000000-26-${String(100000 + i)}`, filedAt: `${day}T21:10:00.000Z`,
        issuerName: `FIXTURE ISSUER ${i} HOLDINGS INC`, issuerTradingSymbol: `F${i % 900}`,
        indexUrl: `https://www.sec.gov/Archives/edgar/data/${1000 + i}/`,
        owners: [{ cik: String(i), name: `Fixture Insider ${i}`, isDirector: i % 2 === 0, isOfficer: i % 2 === 1, isTenPercentOwner: false, officerTitle: i % 2 === 1 ? "Chief Financial Officer" : null }],
        transactions: Array.from({ length: tx }, (_, t) => ({
          table: "nonDerivative", kind: (i + t) % 10 === 0 ? "open_market_buy" : (i + t) % 10 === 1 ? "open_market_sale" : "award_grant",
          shares: 1000 + ((i * 37 + t * 11) % 9000), pricePerShare: 12.5 + (i % 300), transactionDate: day,
        })),
      });
    }
  }
  return {
    kind: "raw", source: "SEC EDGAR (Form 4) — accumulated archive (began 2026-07-04) (fixture)",
    window_by: "filed_day", days: null, window: { from: "2026-10-20", to: "2026-11-18" }, covered: { from: "2026-11-16", to: "2026-11-18" },
    truncated: true, next: "?filed_from=2026-10-20&filed_to=2026-11-13",
    max_filings: 2000, days_scanned: 3, archive_start: "2026-07-04", malformed_lines: 0, archive_days_read: 5, omitted: 312,
    filed_days: VOLUME_DAYS.map(([day, n, , index]) => ({ day, filings: n, omitted: 0, edgar_index: index, complete: index == null ? null : true, checked_at: index == null ? null : "2026-11-18T07:15:41.235Z" })),
    complete_days: 2, count: filings.length, filings,
  };
}

function pagedFilingHistory(req, kind) {
  const older = /[?&]to=2026-11-13/.test(req.url || "");
  const window = older ? { from: "2026-10-17", to: "2026-11-13" } : { from: "2026-10-17", to: "2026-11-15" };
  const covered = older ? { from: "2026-10-17", to: "2026-11-13" } : { from: "2026-11-14", to: "2026-11-15" };
  const form4 = (i, day) => ({
    accession: `0000000000-26-${String(i).padStart(6, "0")}`, filedAt: `${day}T21:10:00.000Z`,
    issuerName: `FIXTURE ISSUER ${i} INC`, issuerTradingSymbol: `FX${i}`,
    indexUrl: `https://www.sec.gov/Archives/edgar/data/${1000 + i}/`,
    owners: [{ cik: String(i), name: `Fixture Insider ${i}`, isDirector: i % 2 === 0, isOfficer: i % 2 === 1, isTenPercentOwner: false, officerTitle: i % 2 === 1 ? "CFO" : null }],
    transactions: [{ table: "nonDerivative", kind: i % 3 === 0 ? "open_market_buy" : "open_market_sale", shares: 1000 * (i + 1), pricePerShare: 12.5 + i, transactionDate: day }],
  });
  const f13 = (i, day) => ({
    accession: `0000000000-26-${String(500 + i).padStart(6, "0")}`, filedAt: `${day}T00:00:00.000Z`, cik: String(2000 + i),
    indexUrl: `https://www.sec.gov/Archives/edgar/data/${2000 + i}/`,
    managerCik: String(2000 + i), managerName: `Fixture Capital ${i} LP`, periodOfReport: "2026-09-30", submissionType: "13F-HR",
    entryTotal: 20 + i, valueTotal: 150000000 * (i + 1), otherManagersCount: 0, holdingsOmitted: false,
    holdings: [{ issuer: `FIXTURE HOLDING ${i} CORP`, titleOfClass: "COM", cusip: `00000000${i}`, value: 9000000 * (i + 1), shares: 100000 * (i + 1), sharesType: "SH", putCall: null, discretion: "SOLE" }],
  });
  const e8k = (i, day) => ({
    accession: `0000000000-26-${String(900 + i).padStart(6, "0")}`, cik: String(3000 + i), companyName: `FIXTURE RESULTS ${i} INC`,
    filedAt: day, itemCodes: i % 2 ? ["2.02", "9.01"] : ["2.02"],
    indexUrl: `https://www.sec.gov/Archives/edgar/data/${3000 + i}/`, exhibitUrl: `https://www.sec.gov/Archives/edgar/data/${3000 + i}/ex99.htm`,
    text: `Fixture Results ${i} Inc today reported quarterly results and updated its full-year guidance.`, textLength: 92, truncated: false,
  });
  if (kind === "form4" && /[?&]filed_(from|to)=/.test(req.url || "")) return filedDayForm4History(req, form4);
  if (kind === "13f" && /[?&]filed_(from|to)=/.test(req.url || "")) return filedDayForm4History(req, f13, "13f");
  const make = kind === "form4" ? form4 : kind === "earnings" ? e8k : f13;
  const filings = older
    ? [make(3, "2026-11-13"), make(4, "2026-11-06"), make(5, "2026-10-20")]
    : [make(0, "2026-11-15"), make(1, "2026-11-14"), make(2, "2026-11-14")];
  return {
    kind: "raw", source: kind === "form4" ? "SEC EDGAR (Form 4) — accumulated archive (began 2026-07-04) (fixture)" : kind === "earnings" ? "SEC EDGAR (8-K Item 2.02 / Exhibit 99) — accumulated archive (began 2026-07-04) (fixture)" : "SEC EDGAR (13F-HR) — accumulated archive (began 2026-07-05) (fixture)",
    ...(kind === "13f" ? { focused_cap: 250 } : {}),
    days: older ? null : 30, window, covered, truncated: !older, next: older ? null : "?from=2026-10-17&to=2026-11-13",
    max_filings: kind === "form4" ? 2000 : 500, days_scanned: older ? 28 : 2, archive_start: kind === "13f" ? "2026-07-05" : "2026-07-04",
    ...(kind === "earnings" ? { text_omitted: false } : {}),
    malformed_lines: 0, count: filings.length, filings,
  };
}

const PAGES = {
  // TRADE WORLD (2026-10-29, human: "I want this moving map like this but
  // more modern as well for some of the screens … it needs to look like the
  // bot UI but have the functionality of what's there and no duplicates"):
  // every Trade tab renders the one Trade cockpit (client/src/trade/) over the
  // moving world map (TradeBackdrop); a tool tab (scanner, news, research,
  // watchlist, planner, taxes) zooms its panel, and
  // "#/analyze/<section>/<SYMBOL>" opens a symbol loaded. Visitor view by
  // default; analyzeowner renders the owner's view (the dossier jump).
  analyze: { route: "/app#/analyze/options", map: false },
  // the loaded cockpit: every panel's endpoint answered with a real capture, so
  // the render shows content, not nine "loading" frames
  analyzeloaded: { route: "/app#/analyze/options/SPY", map: false, fixtures: TRADE_LOADED },
  analyzeowner: { route: "/app#/analyze/options/SPY", map: false, fixtures: { "/api/auth/me": OWNER_ME, ...TRADE_LOADED } },
  scanner: { route: "/app#/scanner", map: false, fixtures: { "/api/market/scanner": tradeFx("trade_scanner.json") } },
  news: { route: "/app#/news", map: false, fixtures: { "/api/market/news": tradeFx("trade_news.json") } },
  // the Research tab: the harness types SPY into the look-up box (real captures answer it)
  research: { route: "/app#/research", map: false, fixtures: TRADE_LOADED },
  watchlist: { route: "/app#/watchlist", map: false },
  planner: { route: "/app#/planner", map: false },
  taxes: { route: "/app#/taxes", map: false },
  data: { route: "/app#/data", map: true },
  // Streams inventory overlay (Phase 4, 2026-07-06) — a page-wide surface,
  // registered here per the Phase 5 ratchet: every new view passes the
  // harness at 390/768/1440 before it ships. Non-map battery (the overlay
  // covers the map).
  streams: { route: "/app#/data/streams", map: false },
  // Data quality dashboard (MAP V2 ROADMAP R6(b), 2026-07-30) — same Phase 5
  // ratchet rule as streams above: every new view passes the harness at
  // 390/768/1440 before it ships.
  // The Archive checks section (2026-11-20) reads the fire archive's
  // completeness ledger: one day of each status. A function, so the last run
  // is always 2 h 10 min before the page reads it.
  quality: { route: "/app#/data/quality", map: false, fixtures: {
    // Archive files + off-site copy (batch C, second part): 14 archives so
    // the "show all" control exists; one whose newest file is 18 days old
    // (the fires stall of 2026-11-20), one dated ahead, one with a marker
    // file as its oldest name, one moved off-site whole. Dates are relative
    // to the run so the ages never drift.
    "/api/data/archive/stats": () => {
      const day = (back) => new Date(Date.now() - back * 86400_000).toISOString().slice(0, 10);
      const k = (files, bytes, oldest, newest) => ({ files, bytes, oldest, newest });
      return { base: "/data/voltrade/datacore_archive", totalBytes: 2012622084, kinds: {
        dtccswaps: k(93, 514568833, "2026-08-23.jsonl.gz", `${day(0)}.jsonl.gz`),
        vessels: k(187, 348035232, `${day(7)}-00.jsonl.gz`, `${day(0)}-18.jsonl`),
        aircraft: k(523, 272372657, `${day(21)}-00.jsonl.gz`, `${day(0)}-18.jsonl`),
        secmidas: k(5, 149626356, "2025q2.jsonl.gz", "2026q2.jsonl.gz"),
        finrashortvol: k(612, 105016630, "2024-06-17.jsonl.gz", `${day(1)}.jsonl`),
        optionchains: k(99, 96581068, ".last_run_day", `${day(1)}.jsonl`),
        flight_plan_versions: k(92, 77530701, `${day(4)}-23.jsonl.gz`, `${day(0)}-18.jsonl`),
        buoys: k(32, 50091491, `${day(31)}.jsonl.gz`, `${day(0)}.jsonl`),
        filings13f: k(125, 45158310, "2026-07-05.jsonl.gz", `${day(0)}.jsonl`),
        fires: k(14, 15478920, `${day(31)}.jsonl.gz`, `${day(18)}.jsonl.gz`),
        eudayaheadprices: k(121, 986441, "2026-07-25.jsonl.gz", `${day(-2)}.jsonl`),
        aircraft_tracked: k(3, 755590, "A2605B/2026-11.jsonl", "ABDB6B/2026-11.jsonl"),
        cropconditions: k(17, 3841, "2026-07-06.jsonl.gz", "2026-10-26.jsonl.gz"),
        aircraft_tracks: k(0, 0, null, null),
      } };
    },
    "/api/data/archive/offload-status": () => ({
      configured: true, retentionDays: 30, oldestReplayableDate: "2026-10-21",
      lastRun: new Date(Date.now() - 9 * 60_000).toISOString(),
      lastRunSummary: { uploaded: 2, uploadFailed: 0, pending: 0, permanent: { uploaded: 0, uploadFailed: 0, pendingUploads: 0 } },
      lastError: null, lastErrorAt: null,
      tiers: { r2: { bytes: 1686760740, objects: 1386, byKind: {
        aircraft: { objects: 690, bytes: 318328602, oldestDate: "2026-10-21", newestDate: "2026-11-18" },
        vessels: { objects: 690, bytes: 1368429871, oldestDate: "2026-10-21", newestDate: "2026-11-18" },
      } } },
      volume: { freeBytes: 1878728704, hotTierTargetFreeBytes: 2147483648, underPressure: true },
      permanentTier: {
        enabled: true, r2Objects: 1197, r2Bytes: 1541902410, conflicts: [], missing: [],
        lastReconcile: new Date(Date.now() - 219 * 60_000).toISOString(),
        byStream: {
          aircraft_tracks: { r2Objects: 119, r2Bytes: 202020693, localFiles: 0, oldestDate: "2026-07-03", newestDate: "2026-10-29" },
          fires: { r2Objects: 116, r2Bytes: 203889327, localFiles: 14, oldestDate: "2026-07-04", newestDate: "2026-11-02" },
        },
      },
    }),
    "/api/data/fires/completeness": () => ({
      kind: "raw", enabled: true, job: "fires_completeness_daily",
      last_run_at: new Date(Date.now() - 130 * 60_000).toISOString(),
      refilled_days: 2, no_rows_days: 1, failed_days: 0,
      live_poll_missed: { days: 1, upstream_rows: 75878, added: 281, share: 0.0037 },
      days: [
        { day: "2026-11-20", status: "complete", source: "VIIRS_NOAA20_NRT", upstream_rows: 75878, held_before: 75597, added: 281, attempts: 1, error: null },
        { day: "2026-11-19", status: "refilled", source: "VIIRS_NOAA20_NRT", upstream_rows: 70100, held_before: 0, added: 70100, attempts: 1, error: null },
        { day: "2026-11-03", status: "refilled", source: "VIIRS_NOAA20_NRT", upstream_rows: 64000, held_before: 0, added: 64000, attempts: 1, error: null },
        { day: "2026-10-27", status: "no_rows", source: null, upstream_rows: 0, held_before: 0, added: 0, attempts: 3, error: null },
      ],
    }),
  } },
  // Signal-strength dashboard (MAP V2 ROADMAP R6(a), 2026-07-30) — same
  // Phase 5 ratchet rule as streams/quality above.
  signals: { route: "/app#/data/signals", map: false },
  // Signals PRODUCT hub (PLATFORM INTEGRATION PROGRAM — the nav's "Signals"
  // leaf, live from this PR): gate-2-passed roots as products. Same Phase 5
  // ratchet rule as the dashboards above.
  validatedsignals: { route: "/app#/data/validated-signals", map: false },
  // Pipeline-health dashboard (MAP V2 ROADMAP R6(c), 2026-07-31) — same
  // Phase 5 ratchet rule as streams/quality/signals above; closes the R6
  // dashboard trio.
  pipelinehealth: { route: "/app#/data/pipeline-health", map: false },
  // Grid-stress descriptive reading overlay (GRID VISION A1 gate-2 FAIL
  // path, 2026-07-07) — same Phase 5 ratchet rule as streams above.
  gridstress: { route: "/app#/data/grid-stress", map: false },
  // Methane repeat-detection hotspots (gate-2(b) of the GEM METHANE-PLUME
  // × EXTRACTION-REGISTRY PROXIMITY hypothesis, 2026-07-20) — same Phase 5
  // ratchet rule as streams/gridstress above.
  methanehotspots: { route: "/app#/data/methane-hotspots", map: false },
  // Crop conditions — USDA NASS weekly ratings (2026-08-04) — same Phase 5
  // ratchet rule as streams/gridstress/methanehotspots above.
  cropconditions: { route: "/app#/data/crop-conditions", map: false },
  // App Store rankings — consumer-app watchlist (2026-08-05) — same
  // Phase 5 ratchet rule as streams/gridstress/cropconditions above.
  appstorerankings: { route: "/app#/data/appstore-rankings", map: false },
  // GitHub org engineering-momentum watchlist (2026-08-05) — same
  // Phase 5 ratchet rule as streams/gridstress/appstorerankings above.
  githubactivity: { route: "/app#/data/github-activity", map: false },
  // Cboe VIX term structure (2026-08-07) — same Phase 5 ratchet rule as
  // streams/gridstress/githubactivity above.
  vixtermstructure: { route: "/app#/data/vix-term-structure", map: false },
  // JODI world oil closing-stock levels (2026-08-21) — same Phase 5
  // ratchet rule as streams/vixtermstructure above.
  jodioilstocks: { route: "/app#/data/jodi-oil-stocks", map: false },
  // European macro cluster — ECB/Eurostat/Bundesbank regime input
  // (2026-08-08) — same Phase 5 ratchet rule as streams/vixtermstructure
  // above.
  eumacro: { route: "/app#/data/eu-macro", map: false },
  // Institutional 13F-HR holdings (2026-08-08) — same Phase 5 ratchet rule
  // as streams/vixtermstructure/eumacro above.
  filings13f: { route: "/app#/data/filings13f", map: false },
  // US macro regime cluster — FRED, 28 public series (2026-08-08) — same
  // Phase 5 ratchet rule as streams/eumacro/filings13f above.
  fredmacro: { route: "/app#/data/fred-macro", map: false },
  // NRC daily reactor status — sortable per-plant table (2026-08-09,
  // closes the "sec_form4_bulk_archive / nrc_outage_reports still without
  // a standalone /data page" queue item — NRC was the one still genuinely
  // missing) — same Phase 5 ratchet rule as streams/fredmacro above.
  nrcreactorstatus: { route: "/app#/data/nrc-reactor-status", map: false },
  // GNSS integrity anomaly signal — first live gate2_pass SIGNAL detail
  // page (2026-08-16) — same Phase 5 ratchet rule as streams/nrcreactorstatus
  // above.
  gnssintegrity: { route: "/app#/data/gnss-integrity", map: false },
  // EU power markets — ENTSO-E load/generation-mix/day-ahead-price, 8
  // bidding zones (2026-08-16) — same Phase 5 ratchet rule as
  // streams/gnssintegrity above.
  eupower: { route: "/app#/data/eu-power", map: false },
  // SEC CNS fails-to-deliver, half-month files (2026-08-16) — same Phase 5
  // ratchet rule as streams/eupower above.
  secftd: { route: "/app#/data/ftd", map: false },
  // Corporate/LLC aircraft fleet utilization — same Phase 5 ratchet rule as
  // streams/secftd above.
  fleetutilization: { route: "/app#/data/fleet-utilization", map: false },
  // EIA-930 hourly electric grid demand — same Phase 5 ratchet rule as
  // fleetutilization/streams/secftd above.
  griddemand: { route: "/app#/data/grid-demand", map: false },
  // EIA-930 hourly net generation by fuel type — griddemand's sibling
  // series — same Phase 5 ratchet rule as griddemand/fleetutilization above.
  gridgeneration: { route: "/app#/data/grid-generation", map: false },
  // OCC daily options cleared volume by trade origin — same Phase 5
  // ratchet rule as griddemand/fleetutilization/streams above.
  occvolume: { route: "/app#/data/occ-volume", map: false },
  // Drought severity — U.S. Drought Monitor weekly (2026-08-20) — same
  // Phase 5 ratchet rule as occvolume/griddemand/fleetutilization above.
  drought: { route: "/app#/data/drought", map: false },
  // Facilities near active fires — NASA FIRMS × strategic-sites cross-tie
  // (2026-08-20) — same Phase 5 ratchet rule as drought/occvolume above.
  firesnearfacilities: { route: "/app#/data/fires-near-facilities", map: false },
  // [REPAIR 2026-11-20] the same panel after a fires poll that found no
  // detection on any satellite (the live state from 2026-11-03 to 11-20,
  // when the panel read "warming up"): it says what NASA was asked and when,
  // and that an empty answer is not "no fires nearby".
  firesnodata: { route: "/app#/data/fires-near-facilities", map: false, fixtures: {
    "/api/data/fires-near-facilities": () => ({
      enabled: true, no_detections: true, facilities: [],
      note: "the last fires poll returned no detections, so there is nothing to check against the facilities",
      poll: {
        last_attempt_at: new Date(Date.now() - 12 * 60_000).toISOString(), last_rows_at: null, source_used: null, polls_without_rows: 3,
        sources: [
          { source: "VIIRS_NOAA20_NRT", rows: 0, error: null },
          { source: "VIIRS_NOAA21_NRT", rows: null, error: "FIRMS 500" },
          { source: "VIIRS_SNPP_NRT", rows: 0, error: null },
        ],
      },
    }),
  } },
  // Air quality at strategic sites — Google Air Quality current conditions
  // (2026-08-20) — same Phase 5 ratchet rule as firesnearfacilities above.
  airquality: { route: "/app#/data/air-quality", map: false },
  // US port imports — Census FT920 monthly port-level values (2026-08-20) —
  // same Phase 5 ratchet rule as airquality above.
  portimports: { route: "/app#/data/imports", map: false },
  // UI coverage batch C (2026-11-20): two routes that had no panel. The
  // history charts added to cropconditions / appstorerankings /
  // githubactivity in the same batch are checked on those pages.
  plantsunderalerts: { route: "/app#/data/plants-under-alerts", map: false },
  railtraffic: { route: "/app#/data/rail-traffic", map: false },
  // GDELT media events near tracked facilities — same Phase 5 ratchet rule as
  // portimports above.
  facilityevents: { route: "/app#/data/facility-events", map: false },
  // CFTC Traders in Financial Futures — same Phase 5 ratchet rule as
  // streams/secftd above.
  tff: { route: "/app#/data/tff", map: false },
  // US Treasury auctions — same Phase 5 ratchet rule as streams/tff above.
  treasuryauctions: { route: "/app#/data/treasury-auctions", map: false },
  // Treasury Daily Statement — same Phase 5 ratchet rule as streams/
  // treasuryauctions above.
  treasurydts: { route: "/app#/data/treasury-dts", map: false },
  // FDA binary events — same Phase 5 ratchet rule as streams/treasurydts
  // above.
  fdaevents: { route: "/app#/data/fda-events", map: false },
  // ClinicalTrials.gov date drift — same Phase 5 ratchet rule as streams/
  // fdaevents above.
  clinicaltrials: { route: "/app#/data/clinical-trials", map: false },
  // NHTSA vehicle complaints — same Phase 5 ratchet rule as
  // streams/fdaevents above.
  vehiclecomplaints: { route: "/app#/data/vehicle-complaints", map: false },
  // FDIC bank failures — same Phase 5 ratchet rule as
  // streams/vehiclecomplaints above.
  bankfailures: { route: "/app#/data/bank-failures", map: false },
  // Federal contract awards (USAspending) — same Phase 5 ratchet rule as
  // bankfailures/vehiclecomplaints above; closes the "shipped-data-no-
  // client-page" gap the 2026-08-25 /api/v1 mirror sweep flagged (session
  // #27, research/experiments.md).
  contracts: { route: "/app#/data/contracts", map: false },
  // Port dwell & transit ranked table — closes the last remaining
  // gate1_pass-root-with-no-dedicated-page gap (2026-09-16 session's own
  // NEXT); same Phase 5 ratchet rule as contracts/bankfailures above.
  // Fixture already exists at /api/data/portdwell (used by the map layer).
  portdwellview: { route: "/app#/data/port-dwell", map: false },
  // Attention proxy (Wikipedia pageviews) — existing page, brought under
  // the Phase 5 ratchet this session (GATE 1 pass + honesty-note addition,
  // 2026-08-18) alongside the other /data detail views above.
  attention: { route: "/app#/data/attention", map: false },
  // Attention spike -> forward volume SIGNAL (2026-09-05) — the second
  // live gate2_pass SIGNAL detail page, after gnssintegrity above — same
  // Phase 5 ratchet rule.
  wikiattentionsignal: { route: "/app#/data/wiki-attention-signal", map: false },
  // EPA CAMD power-plant utilization — same Phase 5 ratchet rule as
  // contracts/attention above; closes the "shipped-data-no-client-page"
  // gap for plant_operations (map layer only since 2026-07-20).
  plantoperations: { route: "/app#/data/plant-operations", map: false },
  // FINRA consolidated short interest + Reg SHO threshold list (2026-08-28)
  // — same Phase 5 ratchet rule as plantoperations/contracts above; closes
  // the last "shipped-data-no-client-page" gap in the FINRA Query API
  // cluster (short-volume/threshold/ATS already had client views —
  // consolidatedShortInterest did not).
  shortinterest: { route: "/app#/data/short-interest", map: false },
  // USA bilateral goods trade — UN Comtrade (2026-09-07) — same Phase 5
  // ratchet rule as shortinterest/plantoperations above; closes the
  // "shipped-data-no-client-page" gap for un_comtrade_bilateral_trade
  // (API-only since the same-day PIPELINE session that built the archive).
  uncomtrade: { route: "/app#/data/un-comtrade", map: false },
  // COVERAGE GAP CLOSED (2026-09-10, scheduled-routine PRODUCT session):
  // these 8 full overlay views (each already has a real "Full X view
  // (#/data/Y) — same overlay pattern" hash-driven route in datamap.tsx,
  // several shipped as far back as the very first /data build) had NEVER
  // been added here — this harness has never once rendered them at any
  // width, so PROMOTION RULE 6 was silently unmet for all 8, discovered by
  // diffing datamap.tsx's hash-route useState initializers against this
  // map programmatically. dtcc-swaps is the freshest instance (shipped
  // 2026-08-23, three weeks unverified) — see server/visualPagesWiring.test.ts,
  // the new ratchet this session added so this class can't recur silently.
  filings: { route: "/app#/data/filings", map: false },
  // 2026-11-15: the same two filing pages on a TRUNCATED history page (the
  // production shape in a heavy EDGAR week): the header must state the days
  // loaded, and "load older days" must walk the server's `next` to the rest.
  filingspaged: { route: "/app#/data/filings", map: false, fixtures: { "/api/data/insider/history": (req) => pagedFilingHistory(req, "form4") } },
  filingsvolume: { route: "/app#/data/filings", map: false, fixtures: { "/api/data/insider/history": () => volumeForm4History() } },
  filings13fpaged: { route: "/app#/data/filings13f", map: false, fixtures: { "/api/data/filings13f/history": (req) => pagedFilingHistory(req, "13f") } },
  earnings: { route: "/app#/data/earnings", map: false },
  earningspaged: { route: "/app#/data/earnings", map: false, fixtures: { "/api/data/earnings-language/history": (req) => pagedFilingHistory(req, "earnings") } },
  shortvolume: { route: "/app#/data/short-volume", map: false },
  cot: { route: "/app#/data/cot", map: false },
  graph: { route: "/app#/data/graph", map: false },
  atssummary: { route: "/app#/data/ats-summary", map: false },
  midas: { route: "/app#/data/midas", map: false },
  dtccswaps: { route: "/app#/data/dtcc-swaps", map: false },
  // FMCSA Out-of-Service orders — motor-carrier enforcement log
  // (2026-09-28) — same Phase 5 ratchet rule as dtccswaps/uncomtrade above;
  // closes the "shipped-data-no-client-page" gap the same-day archive
  // session left open (research/open_questions.md's own filed NEXT(3)).
  fmcsaoos: { route: "/app#/data/fmcsa-oos", map: false },
  xbrlfundamentals: { route: "/app#/data/xbrl-fundamentals", map: false },
  developers: { route: "/developers", map: false },
  // Self-serve preview key management (PLATFORM P3, 2026-07-11) — same
  // Phase 5 ratchet rule as streams/gridstress above. /api/auth/me's
  // fixture defaults to authenticated:false, so this exercises the
  // logged-out "log in to manage keys" state — the most common visitor.
  apikeys: { route: "/apikeys", map: false },
  // Contact + Careers (2026-10-13): the site's public email addresses were
  // consolidated to hello@voltradeai.com; registered so the one-card Contact
  // layout passes 390/768/1440 like every other page.
  contact: { route: "/contact", map: false },
  careers: { route: "/careers", map: false },
  // Owner signups & visits (2026-10-13, human-directed) — same Phase 5
  // ratchet rule. The page takes its state from /api/admin/site-report
  // alone, so the harness renders the populated owner view (fixture
  // below); signed-out / not-owner are designed copy on the same page.
  admin: { route: "/admin", map: false },
  // Bot page v2 (2026-10-16, research/bot_cockpit_design_brief_v2.md): the
  // owner's one-page instrument. Two renders of the same route — the kill
  // switch latched (incident takes the centre) and a normal pre-market
  // moment with two candidates queued — so both the incident element and
  // the decision chain pass 390/768/1440. Numbers are the 2026-10-15
  // production readings; audit times are relative to the run so the day
  // timeline draws them.
  // also at 1920 since the Today tile (2026-11-03): the one width where the KPI strip is a single row of seven
  bot: { route: "/app#/bot", map: false, widths: WIDE, fixtures: botFixtures({ latched: true }) },
  botidle: { route: "/app#/bot", map: false, fixtures: botFixtures({ latched: false }) },
  // BETWEEN SESSIONS ([REPAIR] 2026-11-06): the same idle page pre-market — the market closed, the Today tile naming
  // the last session ("Thu Nov 5 session"), close to close, SPY's final bar. Its own battery below (the Today tile).
  botnight: { route: "/app#/bot", map: false, fixtures: botFixtures({ latched: false, between: true }) },
  // POP-OUT PANE (slice D, 2026-10-21): one panel popped out into its own
  // browser window — the same app at /app#/bot/pane/<id>, rendered alone
  // (no shell), the window IS the panel. Same fixtures as `bot`.
  botpane: { route: "/app#/bot/pane/chart", map: false, fixtures: botFixtures({ latched: true }) },
  // Weekly forecast screen (2026-10-27, SHADOW) — owner tab; one open week +
  // six resolved weeks (fixture above), so every section renders.
  forecast: { route: "/app#/forecast", map: false, widths: WIDE, fixtures: { "/api/auth/me": { authenticated: true, isOwner: true, user: { email: "owner@example.com", tier: "pro", role: "owner" } }, "/api/forecast": forecastFixture() } },
  // EXPIRED SESSION (field bug 2026-10-28): the owner's cached /api/auth/me
  // still says signed in, but the 24h session is gone — /api/forecast answers
  // 401 "Not authenticated" exactly as server/auth.ts does, and from then on
  // /api/auth/me answers signed-out (the server deleted the session row; a
  // per-context cookie models that). Must render the sign-in view with the
  // expiry notice, never "Forecast service unavailable".
  forecastexpired: { route: "/app#/forecast", map: false, fixtures: {
    "/api/auth/me": (req) => (/vt_fx_expired=1/.test(req.headers.cookie || "")
      ? { authenticated: false }
      : { authenticated: true, isOwner: true, user: { email: "owner@example.com", tier: "pro", role: "owner" } }),
    "/api/forecast": { __status: 401, __setCookie: "vt_fx_expired=1; Path=/", __body: { error: "Not authenticated" } },
  } },
  // Brain page (2026-10-29) — owner tab; every status, panel kind and agreement flag (fixture above).
  brain: { route: "/app#/brain", map: false, widths: WIDE, fixtures: { "/api/auth/me": { authenticated: true, isOwner: true, user: { email: "owner@example.com", tier: "pro", role: "owner" } }, "/api/bot/brain": brainFixture() } },
  // Ticker dossier (2026-10-29, human-asked) — owner tab opened from the AI
  // Engine's look-up box; an RKLB-like space name with every layer live, plus
  // the look-up card folded into its stock layer (fixtures above).
  dossier: { route: "/app#/dossier/RKLB", map: false, widths: WIDE, fixtures: { "/api/auth/me": { authenticated: true, isOwner: true, user: { email: "owner@example.com", tier: "pro", role: "owner" } }, "/api/ticker-dossier": dossierFixture("RKLB", 27.42),
    "/api/bot/lookup": { ...botFixtures({ latched: true })["/api/bot/lookup"], symbol: "RKLB", quick: { price: 27.42, change_pct: 1.2, volume: 21_000_000, dollar_volume: 575_820_000, quick_score: 31.2, above_open: true }, deep: { score: 58.2, side: "buy", action_label: "BUY", sector: "Unknown", reasons: ["Momentum 1m +16%", "Volume 1.4× average"], factors: { momentum: 22, mean_reversion: -6, vrp: 3, squeeze: 4, volume: 1.4, sentiment: 2 } } } } },
  // the empty state (no symbol yet) and a name whose options feed is down
  dossierempty: { route: "/app#/dossier", map: false, widths: WIDE, fixtures: { "/api/auth/me": { authenticated: true, isOwner: true, user: { email: "owner@example.com", tier: "pro", role: "owner" } } } },
  landing: { route: "/", map: false },
  // MULTI-SEGMENT ROUTE REGRESSION (found 2026-07-11 PLATFORM P3 session,
  // fixed this PR): vite.config.ts's base:"./" baked a relative script src
  // that only resolved correctly for single-segment routes; any 2+-segment
  // route (this one included) rendered a blank white screen in production
  // with no pageerror. Kept in PAGES permanently as the ratchet for that
  // class of bug — see the "app actually mounted" check in CHECKS_SNIPPET,
  // which is what actually catches it (a blank page has no overflow and no
  // interactive elements, so the pre-existing checks alone passed it clean).
  newsletterissue: { route: "/newsletter/test-issue", map: false },
};
const only = process.argv.includes("--page")
  ? process.argv[process.argv.indexOf("--page") + 1]
  : null;

// ── deterministic API fixtures ─────────────────────────────────────────────
// Per-page fixture overrides (bot page v2, 2026-10-16): a PAGES entry may
// carry `fixtures` — a map of route → body consulted BEFORE the global map
// while that page renders. The bot page is owner-gated, so its entry
// overrides /api/auth/me with the owner identity (fixture person only).
let PAGE_FIXTURES = {};
const FIXTURES = {
  "/api/auth/me": { authenticated: false },
  // /admin owner report (server/siteReport.ts shape). Fixture people only.
  "/api/admin/site-report": {
    generatedAt: "2026-10-13T12:00:00.000Z",
    accounts: [
      { email: "owner@example.com", joinedAt: "2026-04-01T00:00:00.000Z", role: "owner", tier: "free", subscriptionStatus: null, isOwner: true, activeApiKeys: 0, apiKeyLastUsedAt: null, signedInLast24h: true },
      { email: "ada.lovelace@example.com", joinedAt: "2026-08-02T09:30:00.000Z", role: "user", tier: "free", subscriptionStatus: null, isOwner: false, activeApiKeys: 1, apiKeyLastUsedAt: 1791900000000, signedInLast24h: false },
      { email: "grace.hopper.fixture@example.org", joinedAt: "2026-09-14T17:05:00.000Z", role: "user", tier: "pro", subscriptionStatus: "active", isOwner: false, activeApiKeys: 0, apiKeyLastUsedAt: null, signedInLast24h: true },
    ],
    waitlist: [{ at: 1783236000000, email: "dev.fixture@example.net", source: "developers" }],
    visits: {
      source: "history", since: "2026-10-09T00:02:00.000Z", totalPageViews: 412, uniquesToday: 37, browserUniquesToday: 22,
      uniquesByDay: { "2026-10-09": 18, "2026-10-10": 25, "2026-10-11": 12, "2026-10-12": 41, "2026-10-13": 37 },
      topPages: [{ route: "/", count: 260 }, { route: "/app", count: 94 }, { route: "/developers", count: 41 }, { route: "/pricing", count: 17 }],
      days: [
        { day: "2026-10-09", views: 52, uniques: 18, browserViews: 30, browserUniques: 9, capped: false },
        { day: "2026-10-10", views: 77, uniques: 25, browserViews: 51, browserUniques: 14, capped: false },
        { day: "2026-10-11", views: 31, uniques: 12, browserViews: 15, browserUniques: 6, capped: false },
        { day: "2026-10-12", views: 140, uniques: 41, browserViews: 96, browserUniques: 26, capped: false },
        { day: "2026-10-13", views: 112, uniques: 37, browserViews: 70, browserUniques: 22, capped: false },
      ],
    },
    notes: ["fixture"],
  },
  "/api/data/layers": {
    layers: [
      { id: "imagery", name: "Satellite imagery", kind: "raw", status: "live", group: "base", costTier: "light", source: "Esri World Imagery", description: "Base imagery." },
      { id: "terrain", name: "Terrain (hillshade)", kind: "raw", status: "live", group: "base", costTier: "moderate", source: "Mapterhorn (© Mapterhorn)", description: "Global hillshade, off by default." },
      { id: "seafloor", name: "Seafloor (drain the ocean)", kind: "raw", status: "live", field: true, group: "base", costTier: "moderate", source: "NOAA ETOPO1 via Terrain Tiles on AWS", description: "Depth-tinted ocean-floor relief overlay, off by default." },
      { id: "weather", name: "Weather radar (US)", kind: "raw", status: "live", field: true, group: "base", costTier: "heavy", source: "NOAA nowCOAST (public domain)", description: "US radar mosaic, off by default." },
      { id: "weather_temp", name: "Temperature (global)", kind: "raw", status: "live", field: true, group: "base", costTier: "heavy", source: "OpenWeatherMap (© OpenWeatherMap)", description: "Global temp field, off by default." },
      { id: "weather_wind", name: "Wind (global)", kind: "raw", status: "live", field: true, group: "base", costTier: "heavy", source: "OpenWeatherMap (© OpenWeatherMap)", description: "Global wind field, off by default." },
      { id: "aircraft", name: "Live aircraft (ADS-B)", kind: "raw", status: "live", group: "live", costTier: "heavy", source: "adsb.lol/airplanes.live", description: "Live aircraft." },
      { id: "vessels", name: "Live vessels (AIS)", kind: "raw", status: "awaiting_key", group: "live", costTier: "moderate", source: "aisstream.io", description: "Needs AISSTREAM_KEY." },
      { id: "trains", name: "Live trains (rail)", kind: "raw", status: "live", group: "live", costTier: "light", source: "Digitraffic FI + Entur NO", description: "FI+NO launch coverage." },
      { id: "sites", name: "Strategic sites", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "datacore/sites", description: "Reference sites." },
      { id: "powerplants", name: "US power plants", kind: "raw", status: "live", group: "facilities", costTier: "moderate", source: "WRI GPPD (CC BY 4.0)", description: "US plants by fuel." },
      // [REPAIR R15 2026-07-07] powergrid was absent here, so the
      // toggle-consistency battery never clicked it — and the missing
      // LAYER_GROUP entry (permanent "reload to enable") shipped unseen.
      // Every toggleable registry layer must appear in this fixture;
      // the wiring ratchet (layersWiring.test.ts) pins the source side.
      { id: "powergrid", name: "Power grid (TX pilot)", kind: "raw", status: "live", group: "facilities", costTier: "moderate", source: "OpenStreetMap power features (© OpenStreetMap contributors, ODbL)", description: "TX pilot vector tiles." },
      // [REPAIR 2026-07-25] These 9 were shipped end-to-end (client symbol
      // layer + server route) across several prior sessions but never added
      // here — so the self-see/toggle-consistency/legend-parity batteries
      // above have never actually exercised them (filed as open debt across
      // four session log entries before this fix). Every toggleable
      // registry layer must appear in this fixture; see the powergrid note
      // above for the precedent this repeats.
      { id: "nukefacilities", name: "Nuclear facilities (fuel cycle)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "Wikidata (CC0 1.0), curated", description: "67 fuel-cycle/production facilities, building-trefoil symbols by category." },
      { id: "faa_airports", name: "FAA airport status", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "FAA National Airspace System Status (public domain)", description: "Ground stops/delays/closures, rolling snapshot." },
      { id: "flight_restrictions", name: "Flight restrictions (FAA)", kind: "raw", status: "live", group: "live", costTier: "light", source: "FAA SWIM FNS NOTAM (fixture)", description: "Flight restrictions in force or starting within 6 hours, as received." },
      // [REPAIR, found alongside this session's NRC dashboard page]: this
      // toggleable registry layer (datacore/layers.json) was missing from
      // this fixture, same class of gap the R15 (2026-07-07) and
      // 2026-07-25 fixes closed for 9 other layers — the toggle-
      // consistency/self-see/legend-parity batteries never exercised it.
      { id: "nrc_reactor_status", name: "US nuclear reactor status (NRC, daily)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "U.S. NRC daily Power Reactor Status Reports (nrc.gov, public domain, keyless)", description: "Percent-of-rated-thermal-power per operating unit, joined onto the nuclear-plant registry." },
      { id: "border_waits", name: "CBP land-border wait times", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "CBP Border Wait Times (public domain)", description: "Hourly wait times at US land ports of entry." },
      // [REPAIR, found this session, same class of gap as the R15
      // (2026-07-07) and 2026-07-25 fixes above]: `lng_shipyards` and
      // `steel_raw_materials` (datacore/layers.json) shipped end-to-end
      // (client symbol/choropleth layer + server route) but were never
      // added here — so the self-see/toggle-consistency/legend-parity
      // batteries never exercised either, closing the visual-verification
      // gap the 2026-09-24 fifth/seventh sessions' own experiments.md
      // entries flagged as NOT DONE. Every toggleable registry layer must
      // appear in this fixture; see the powergrid note above.
      { id: "lng_shipyards", name: "LNG carrier shipyards (GEM)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "Global Energy Monitor — Global LNG Carrier Tracker (CC BY 4.0)", description: "32 shipbuilding yards, aggregated carrier counts by lifecycle status and country." },
      { id: "steel_raw_materials", name: "Steel raw materials (met coal & iron ore, GEM)", kind: "raw", status: "live", field: true, group: "facilities", costTier: "light", source: "Global Energy Monitor — Production-Consumption of Met Coal & Iron Ore by Steel Industry (CC BY 4.0)", description: "Country-level met-coal/iron-ore balance sheet, choropleth-filled by iron ore mined (ttpa)." },
      { id: "nucleartests", name: "Nuclear tests 1945–1998 (time machine)", kind: "raw", status: "live", group: "hazards", costTier: "light", source: "SIPRI / Johnston archive nuclear explosions catalog", description: "2,027 located tests, emplacement symbols, year-slider scrub." },
      { id: "radiation", name: "Ambient radiation (gamma)", kind: "raw", status: "live", group: "hazards", costTier: "light", source: "BfS · Health Canada · STUK/FMI · EPA RadNet", description: "Observed gamma dose-rate monitors, four national networks." },
      { id: "nukeaccidents", name: "Nuclear accidents & incidents", kind: "raw", status: "live", group: "hazards", costTier: "light", source: "Wikidata (CC0 1.0), curated", description: "46 accident/incident sites 1949–2024, INES-tinted." },
      { id: "pfas", name: "PFAS drinking-water detections (EPA UCMR 5)", kind: "raw", status: "live", group: "hazards", costTier: "moderate", source: "U.S. EPA UCMR 5, public domain", description: "3,417 water systems with a detected PFAS compound." },
      { id: "cancerrates", name: "County cancer rates (NCI State Cancer Profiles)", kind: "raw", status: "live", field: true, group: "hazards", costTier: "moderate", source: "National Cancer Institute, State Cancer Profiles, public domain", description: "County-level incidence/mortality choropleth, 3,143 US counties." },
      { id: "methane_plumes", name: "Methane plumes (GEM GMET)", kind: "raw", status: "live", group: "environmental", costTier: "moderate", source: "Global Energy Monitor GMET (CC BY 4.0)", description: "Satellite methane-plume detections, nearest-asset match." },
      { id: "coal_mine_features", name: "Coal mine boundaries & infrastructure (GEM)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "Global Energy Monitor (CC BY 4.0)", description: "Mine boundary polygons + point infrastructure features." },
      // [REPAIR, found 2026-09-25 while verifying the immediately-prior
      // session's own claim that 5 layers were fixture-absent]: only these
      // 4 GEM point layers were actually missing — `coal_mine_features`
      // above was already present (that prior entry's list was stale/
      // wrong, caught here by grepping the fixture directly rather than
      // trusting the log). Same class of gap as the R15 (2026-07-07) and
      // 2026-07-25/2026-09-24 fixes above: shipped end-to-end (client
      // symbol layer + server route) across several prior sessions but
      // never added here, so the self-see/toggle-consistency/legend-
      // parity batteries never exercised them. Every toggleable registry
      // layer must appear in this fixture; see the powergrid note above.
      { id: "coal_terminals", name: "Coal terminals (GEM)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "Global Energy Monitor — Global Coal Terminals Tracker (CC BY 4.0)", description: "521 port coal-handling terminals, symbol by terminal role, colour by lifecycle status." },
      { id: "iron_ore_mines", name: "Iron ore mines (GEM)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "Global Energy Monitor — Global Iron Ore Mines Tracker (CC BY 4.0)", description: "949 iron ore mines worldwide, colour by lifecycle status." },
      { id: "iron_steel_plants", name: "Iron & steel plants (GEM)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "Global Energy Monitor — Global Iron and Steel Tracker (CC BY 4.0)", description: "1,293 iron and steel plants worldwide, colour by primary production technology; furnace-unit detail in the popup." },
      { id: "chemicals", name: "Chemical plants (GEM)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "Global Energy Monitor — Global Chemicals Inventory (CC BY 4.0)", description: "868 chemical plants worldwide, colour by primary feedstock family." },
      // [REPAIR 2026-09-27] Same class of gap as the R15 (2026-07-07),
      // 2026-07-25, 2026-09-24, and 2026-09-25 fixes above: comparing every
      // `id` in datacore/layers.json (256) against every `id:` in this file
      // (116) found 140 missing — ~118 were per-state/country
      // `powergrid_<region>` variants of the single already-fixtured
      // `powergrid` tile layer (legitimately excluded, same reasoning as
      // the powergrid note above: one generic tile layer represents its
      // data-partitioned variants, not each one separately). The remaining
      // 22 were genuinely distinct layers shipped end-to-end (client toggle
      // + server route) but never added here. Of those 22, these 12 have a
      // real `/api/data/*` (or `/api/v1/data/*`) JSON route the client
      // fetches and toggle on — the standard fixture-driven pattern — so
      // they get a row here (4 of the 8 needing a NEW data fixture below;
      // ats_summary/contracts/plant_operations/midas already had one,
      // shipped for their own dedicated `#/data/*` full-page views before
      // this fix, but were never added to THIS array, so the self-see/
      // toggle-consistency/legend-parity batteries never exercised them
      // either). The other 10 (celestial_paths, daynight, orbital_sats,
      // floodzones, firetemp, so2, places, seafloor_confidence,
      // submarine_cables, timezones) do NOT fit this pattern — see the
      // dedicated comment after this array for why each is excluded.
      { id: "ats_summary", name: "ATS / OTC venue volume (FINRA)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "FINRA Query API — weeklySummary/monthlySummary/blocksSummary (no API key required)", description: "FINRA-precomputed ATS-vs-OTC per-symbol volume leaderboards (weekly) and OTC per-symbol volume (monthly), plus per-venue block-trading ranks." },
      { id: "boundaries_admin1", name: "State / province borders", kind: "raw", status: "live", group: "base", costTier: "light", source: "Natural Earth 1:50m admin-1 (public domain) — compiled into datacore", description: "State and province boundary lines (first-level subdivisions, worldwide) as a thin reference overlay." },
      { id: "contracts", name: "Federal contract awards (USAspending)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "USAspending.gov, U.S. Department of the Treasury (US government work, public domain, keyless, no API key required)", description: "Contracts A-D with |transaction amount| >= $25K, ticker-matched where a recipient resolves via exact-name or FPDS parent lookup." },
      { id: "grid_generation", name: "Electric grid generation by fuel type (EIA-930)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "EIA-930 Hourly Electric Grid Monitor (public domain, EIA_API_KEY-gated)", description: "Hourly net generation (MWh) by fuel source per balancing authority, ~1-2h publication lag." },
      { id: "meteors", name: "Large meteors (NASA)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "NASA/JPL CNEOS fireball API (US Government sensors) — public domain, self-archived", description: "Big meteors that exploded in the atmosphere (bolides), detected by US Government sensors and published by NASA/JPL CNEOS after the fact." },
      { id: "midas", name: "Market microstructure (SEC MIDAS)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "SEC MIDAS individual-security market-structure metrics, quarterly files (public domain, no API key required)", description: "SEC-published per-(date, ticker) lit/hidden/odd-lot/cancel-to-trade metrics; rank scale differs by kind." },
      { id: "military_installations", name: "Military installations", kind: "raw", status: "live", group: "facilities", costTier: "heavy", source: "US DoD open data, OpenStreetMap contributors (© OpenStreetMap contributors, ODbL), and cited government publications", description: "Officially published installation locations (~3,024 named installations); static reference geography, never joined to live tracking layers." },
      { id: "plant_operations", name: "EPA CAMD plant operations (ground truth, TX pilot)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "U.S. EPA Clean Air Markets Division (CAMD) Continuous Emissions Monitoring, unit-level daily (public domain, 40 CFR Part 75)", description: "Direct plant-utilization ground truth from EPA's own unit-level Continuous Emissions Monitoring reporting; v1 pilot scope Texas only, quarterly cadence." },
      { id: "quakehistory", name: "Earthquakes 1900–now (time machine)", kind: "raw", status: "live", group: "hazards", costTier: "moderate", source: "USGS ANSS Comprehensive Catalog (ComCat), public domain", description: "Every catalogued M6.0+ earthquake since 1900, shares the history time bar with the nuclear-tests layer." },
      { id: "shadowstats", name: "Dark-ship statistics", kind: "raw", status: "live", group: "filings", costTier: "light", source: "Derived from our own AIS archive (terrestrial coverage)", description: "Event COUNTS from our own vessel archive: AIS gap events, identity-change candidates, loitering in known STS/transshipment zones." },
      { id: "superfund", name: "Superfund sites (EPA NPL)", kind: "raw", status: "live", group: "hazards", costTier: "light", source: "U.S. EPA Superfund National Priorities List (NPL), SEMS — public domain", description: "EPA National Priorities List Superfund sites (~1,840): the nation's most contaminated cleanup sites, colored by NPL status." },
      { id: "waterviolators", name: "Water violators (EPA CWA)", kind: "raw", status: "live", group: "hazards", costTier: "moderate", source: "U.S. EPA ECHO — Clean Water Act compliance records (public domain)", description: "Active facilities with more than 8 of the last 12 quarters in Clean Water Act noncompliance (~25k)." },
      // freshness (Phase 5, three of the five fixture health states so the
      // visual harness actually exercises the chip's color/label variants):
      { id: "insider", name: "Insider transactions (Form 4)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "SEC EDGAR", description: "Recent Form 4 filings as filed.", freshness: { stream: "filings", health: "live", age_hours: 0.4, health_note: "newest file 0.4h old" } },
      { id: "earnings", name: "Earnings language (8-K)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "SEC EDGAR", description: "As-filed 8-K Item 2.02 results/guidance releases." },
      { id: "shortvol", name: "Short-sale volume (FINRA)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "FINRA Reg SHO", description: "Daily consolidated short-marked execution volume per symbol — a flow proxy, not short interest.", freshness: { stream: "finrashortvol", health: "stale", age_hours: 411.2, health_note: "newest file 411.2h old — exceeds cadence-derived threshold" } },
      { id: "attention", name: "Attention proxy (Wikipedia pageviews)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "Wikimedia pageviews API", description: "Daily article pageviews for a curated ticker seed — an attention proxy, not a signal." },
      { id: "cot", name: "Commitments of Traders (CFTC, disaggregated)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "CFTC Public Reporting Socrata API", description: "Weekly futures-only positioning by trader category — a positioning proxy, not a signal.", freshness: { stream: "cftccot", health: "recent", age_hours: 122.9, health_note: "newest file 122.9h old (within cadence)" } },
      { id: "tff", name: "Traders in Financial Futures (CFTC, futures-only)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "CFTC Public Reporting Socrata API, futures-only", description: "Weekly financial-futures positioning by trader category — a positioning proxy, not a signal." },
      { id: "treasury_auctions", name: "US Treasury auctions", kind: "raw", status: "live", group: "filings", costTier: "light", source: "TreasuryDirect TA_WS, securities/auctioned", description: "Primary-market bid-to-cover + bidder-class shares — a stress proxy, not a signal." },
      { id: "treasury_dts", name: "Treasury Daily Statement", kind: "raw", status: "live", group: "filings", costTier: "light", source: "U.S. Treasury Fiscal Data API, deposits_withdrawals_operating_cash", description: "Daily operating-cash deposit/withdrawal line items — a cash-flow proxy, not a signal." },
      { id: "fda_events", name: "FDA binary events", kind: "raw", status: "live", group: "filings", costTier: "light", source: "openFDA drugsfda API + Federal Register API", description: "Drug approvals + advisory-committee meeting notices — a catalyst-timing proxy, not a signal." },
      { id: "clinical_trials", name: "Clinical-trial date drift", kind: "raw", status: "live", group: "filings", costTier: "light", source: "ClinicalTrials.gov API v2", description: "Changes in sponsor-reported primary completion dates — identity-tagged where a ticker matches, not a signal." },
      { id: "vehicle_complaints", name: "Vehicle complaints", kind: "raw", status: "live", group: "filings", costTier: "light", source: "NHTSA Office of Defects Investigation complaints API", description: "Per-vehicle complaint counts (crash/fire-flagged) over a curated ticker-mapped watchlist — a complaint-velocity proxy, not a signal." },
      { id: "bank_failures", name: "Bank failures", kind: "raw", status: "live", group: "filings", costTier: "light", source: "FDIC Bank Data API", description: "Most recent US bank failure/assistance events — a regional-bank-stress proxy, not a signal." },
      { id: "portdwell", name: "Port dwell (arrivals/departures)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "Own AIS archive + verified port geofences", description: "Per-port dwell stats; lower bounds; anomaly SIGNAL gate-2 locked." },
      { id: "secftd", name: "Fails-to-deliver (SEC CNS)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "SEC CNS fails-to-deliver, half-month files (public domain, no API key required)", description: "Trailer-checksummed aggregate net fail balances per settlement date — a level, not a daily flow; raw spikes alone are a crowded signal." },
      { id: "fleet_utilization", name: "Corporate/LLC aircraft fleet utilization", kind: "raw", status: "live", group: "filings", costTier: "light", source: "Own aircraft position archive x FAA Aircraft Registry entity spine", description: "Weekly flight-session counts and airborne hours per registered owner — GATE 1 join accuracy passed, GATE 2 not attempted, nothing here is a signal." },
      { id: "grid_demand", name: "Electric grid demand (EIA-930)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "EIA-930 Hourly Electric Grid Monitor (public domain, EIA_API_KEY-gated)", description: "Hourly demand/day-ahead forecast per balancing authority — GATE 1 revision stability passed, GATE 2 demand-residual-vs-industrial-returns attempted and killed, nothing here is a signal." },
      { id: "occ_volume", name: "Options cleared volume (OCC daily)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "The Options Clearing Corporation (OCC) daily cleared volume (public, keyless)", description: "Top underlyings by daily cleared options volume, customer vs market-maker put/call split — GATE 1 day-total accuracy passed, GATE 2 customer call/put skew attempted and killed (reversed direction), nothing here is a signal." },
      { id: "graph", name: "Everything Graph", kind: "raw", status: "live", group: "graph", costTier: "light", source: "Own join over Form 4 + entity_map + AIS port-dwell archive", description: "Entity search across insiders, facilities, and vessels. RAW join with provenance, no predictive claim." },
      { id: "fires", name: "Active fires (VIIRS)", kind: "raw", status: "awaiting_key", group: "environmental", costTier: "moderate", source: "NASA FIRMS / LANCE", description: "Needs NASA_FIRMS_MAP_KEY." },
      { id: "nightlights", name: "Night lights radiance (GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — VIIRS/SNPP Day/Night Band", description: "Daily radiance imagery, dated (defaults to yesterday)." },
      { id: "aerosol", name: "Aerosol optical depth (GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — MODIS Combined Value-Added AOD", description: "Daily aerosol optical depth, dated (defaults to yesterday)." },
      { id: "vegetation", name: "Vegetation health (NDVI, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — VIIRS/SNPP NDVI 8-day", description: "Vegetation greenness (NDVI) 8-day composite, dated (defaults to yesterday)." },
      { id: "soilmoisture", name: "Soil moisture (SMAP, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — SMAP L4 root-zone", description: "Root-zone soil moisture, dated (~6-day lag, defaults ~7 days back)." },
      { id: "no2", name: "Nitrogen dioxide (TROPOMI, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — Sentinel-5P/TROPOMI NO₂", description: "Tropospheric NO₂ column, dated (defaults to yesterday)." },
      { id: "floods", name: "Flood / water extent (MODIS, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — MODIS Terra+Aqua 3-day flood composite", description: "Rolling 3-day flood/water extent composite, dated (defaults to yesterday)." },
      { id: "biomass", name: "Aboveground biomass density (GEDI, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — GEDI L4B aboveground biomass density", description: "Static mission-life mean (2019-04 to 2023-03), no scrubber." },
      { id: "rivergauges", name: "River gauges (barge corridor)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "USGS NWIS (public domain)", description: "Live stage/discharge at 14 barge-corridor gauges." },
      { id: "alerts", name: "Severe weather alerts (NWS)", kind: "raw", status: "live", group: "environmental", costTier: "moderate", source: "National Weather Service (public domain)", description: "Active NWS warnings/watches, colored by severity." },
      { id: "spaceweather", name: "Space weather (NOAA SWPC)", kind: "raw", status: "live", group: "environmental", costTier: "moderate", source: "NOAA Space Weather Prediction Center (public domain)", description: "Aurora oval (OVATION model FORECAST) + observed Kp / R-S-G scales / solar wind." },
      { id: "earthquakes", name: "Earthquakes (USGS)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "USGS Earthquake Hazards Program (public domain)", description: "Real-time M2.5+ seismic events, sized/colored by magnitude." },
      { id: "volcanoes", name: "Volcano alert levels (USGS/GVP)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "USGS Volcano Hazards Program + Global Volcanism Program, Smithsonian Institution", description: "Currently-elevated volcanoes, USGS alert level, coordinates joined live from GVP." },
      { id: "buoys", name: "Ocean buoys (NDBC)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "NOAA National Data Buoy Center (public domain)", description: "Latest wave/wind/pressure readings, ~889 stations worldwide." },
      { id: "surfacewater", name: "Surface water (1984–2021)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "EC JRC/Google GSW v2021", description: "Static water occurrence, off by default." },
      { id: "forest", name: "Forest cover (2020)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "EC JRC GFC2020 via GFW", description: "Static forest extent, off by default." },
      { id: "boundaries", name: "Country borders", kind: "raw", status: "live", group: "base", costTier: "light", source: "Natural Earth 1:110m (public domain)", description: "Reference borders, off by default." },
      { id: "tank_fill", name: "Tank-fill % (Sentinel-2)", kind: "signal", status: "planned", group: "signals", costTier: "light", source: "Copernicus", description: "Gate-2 locked." },
    ],
  },
  // [REPAIR 2026-09-27, continued] The 2026-09-27 fixture-coverage sweep
  // (see the comment above the 12 rows just added) also checked these 10
  // remaining registry ids and found each one is toggled/rendered by a
  // GENUINELY DIFFERENT mechanism than "client fetches /api/data/<id> and
  // the harness mocks that route" — so no fixture is added for any of
  // them, per this file's own rule that a fake fixture is never forced
  // just to fill a row:
  //   - celestial_paths, daynight: pure client-side computed ephemeris
  //     (astronomy-engine / Meeus series). No fetch of any kind — nothing
  //     for this harness to mock.
  //   - orbital_sats: client-fetches CelesTrak directly in the browser
  //     (ORBITAL program's DATA-PATH SPLIT, CLAUDE.md KNOWN STATE) — an
  //     external domain, which this harness's `page.route("**/*", ...)`
  //     aborts outright (only 127.0.0.1 continues). No local route exists
  //     to fixture.
  //   - floodzones (FEMA hazards.fema.gov), firetemp/so2 (NASA GIBS
  //     WMTS): the client adds a MapLibre raster *tile* source pointing
  //     straight at the provider's own public tile service — never a
  //     fetch() to one of our /api/data/* JSON routes. Also aborted by
  //     the same external-domain rule; no JSON shape to fixture.
  //   - places, submarine_cables, timezones, seafloor_confidence: served
  //     from REAL static assets committed to the repo (client/public/
  //     tiles/places.pmtiles, client/public/cables/submarine_cables.json,
  //     client/public/tz/timezone_lines.json, client/public/tiles/
  //     seafloor_*_mariana.pmtiles) via non-/api paths. This harness's own
  //     local server (below) serves any non-"/api/" path from the real
  //     built DIST, not a fixture — so these already exercise genuine,
  //     correct data with zero mocking needed, but they are not the
  //     "/api/data/*route + fixture" pattern this array's precedent
  //     comments (powergrid etc.) describe, so they are left out of this
  //     specific fixture-coverage sweep rather than added on a different
  //     rationale than the other 12.
  // W3 TIME SCRUBBER (server/queryEngine.ts querySnapshot) — a small,
  // deterministic replay payload so the panel's status line + rendered
  // points are real content, not the generic "{}" fallback.
  "/api/data/snapshot": {
    kind: "raw", layer: "aircraft", mode: "position",
    requested_at: "2026-07-08T14:00:00.000Z", bucket_at: "2026-07-08T14:00:00.000Z",
    data: [
      { id: "fx1", lat: 36.1, lon: -97.0 },
      { id: "fx2", lat: 35.8, lon: -96.6 },
      { id: "fx3", lat: 36.4, lon: -97.3 },
    ],
    count: 3, count_before_viewport: 3, count_dropped_offscreen: 0,
    viewport_filtered: true, capped: false,
    freshness: "2026-07-08T14:52:00.000Z",
    provenance: "own position archive (fixture)",
    window: { min_iso: "2026-07-01T14:00:00.000Z", max_iso: "2026-07-08T14:00:00.000Z", days: 7 },
    note: "fixture snapshot for the visual harness",
  },
  // 10,000 synthetic aircraft — the DESIGN.md performance budget says map
  // interactions stay smooth at 10k+ features; the fixture proves the
  // rendering path at that scale (deterministic pseudo-random spread).
  "/api/data/aircraft": (() => {
    const aircraft = [];
    let seed = 42;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const types = ["B738", "A320", "C172", "AT76", "EC35", null];
    for (let i = 0; i < 10000; i++) {
      aircraft.push({
        icao24: "fx" + i.toString(16), callsign: "TST" + i, origin_country: "US",
        lon: -125 + rnd() * 58, lat: 25 + rnd() * 24,
        altitude_m: i % 17 === 0 ? null : Math.round(rnd() * 12000),
        on_ground: i % 23 === 0,
        velocity_ms: Math.round(rnd() * 250),
        heading: Math.round(rnd() * 359),
        type: types[i % types.length], category: null,
      });
    }
    return { source: "fixture", kind: "raw", time: 1, count: aircraft.length, aircraft };
  })(),
  "/api/data/vessels": { enabled: false, reason: "AISSTREAM_KEY not set (fixture)", vessels: [] },
  "/api/data/fires": { enabled: false, kind: "raw", reason: "NASA_FIRMS_MAP_KEY not set (fixture)", fires: [] },
  "/api/data/alerts": {
    kind: "raw", source: "National Weather Service (fixture)", time: 1, count: 2, zone_only: 3,
    alerts: [
      { id: "urn:oid:fixture.tornado", event: "Tornado Warning", severity: "Extreme", area: "Payne, OK", ends: "2026-07-05T13:00:00-05:00",
        rings: [[[-97.2, 35.8], [-96.6, 35.8], [-96.6, 36.2], [-97.2, 36.2], [-97.2, 35.8]]] },
      { id: "urn:oid:fixture.flood", event: "Flood Watch", severity: "Moderate", area: "St. Louis Metro, MO", ends: null,
        rings: [[[-90.6, 38.4], [-89.9, 38.4], [-89.9, 38.9], [-90.6, 38.9], [-90.6, 38.4]]] },
    ],
  },
  "/api/data/spaceweather": {
    kind: "raw", source: "NOAA SWPC (fixture)", time: 1,
    note: "fixture: observed Kp/scales/wind + OVATION forecast cells",
    kp_recent: [
      { t: "2026-07-29T06:00:00", kp: 1.67, a: 6, n: 8, rt: "2026-07-29" },
      { t: "2026-07-29T09:00:00", kp: 3.33, a: 18, n: 8, rt: "2026-07-29" },
    ],
    scales: {
      current: { kind: "current", date: "2026-07-29", time: "12:37:00", r: "0", s: "0", g: "1", rMinorProb: null, rMajorProb: null, sProb: null },
      forecast: [{ kind: "forecast", date: "2026-07-30", time: "00:00:00", r: null, s: null, g: "0", rMinorProb: 30, rMajorProb: 1, sProb: 1 }],
    },
    wind: { speedKms: 329, speedAt: "2026-07-29T12:32:00Z", btNt: 4, bzNt: 1, magAt: "2026-07-29T12:34:00Z" },
    aurora: {
      obs: "2026-07-29T12:30:00Z", forecast: "2026-07-29T13:51:00Z", max: 42, aggDeg: 2, minVal: 2,
      cells: [[-160, 62, 12], [-158, 62, 25], [-156, 64, 42], [-100, 60, 8], [20, 68, 15], [-150, -72, 30]],
    },
    alerts_recent: [
      { id: "K05W:2026-07-27 02:11:00.000", productId: "K05W", issued: "2026-07-27 02:11:00.000", code: "WARK05", serial: 2101, title: "WARNING: Geomagnetic K-index of 5 expected", message: "fixture", rt: "2026-07-29" },
    ],
  },
  "/api/data/rivergauges": {
    kind: "raw", source: "USGS Water Services (fixture)", time: 1, count: 2,
    gauges: [
      { site: "07010000", name: "Mississippi River at St. Louis, MO", param: "00065", d: "2026-07-05T00:15:00.000-05:00", v: 15.13, q: "P", lat: 38.62889, lon: -90.17972 },
      { site: "07032000", name: "Mississippi River at Memphis, TN", param: "00060", d: "2026-07-05T00:15:00.000-06:00", v: 512000, q: "P", lat: 35.12278, lon: -90.0775 },
    ],
  },
  "/api/data/earthquakes": {
    kind: "raw", source: "USGS Earthquake Hazards Program (fixture)", time: 1, count: 2,
    quakes: [
      { id: "fx0001", mag: 4.6, place: "12 km SW of Fixture City, CA", lat: 34.02, lon: -118.5, depth: 8.2, time: 1, updated: 1, tsunami: false, sig: 312, net: "ci", magType: "ml", type: "earthquake", status: "reviewed", url: "https://earthquake.usgs.gov/earthquakes/eventpage/fx0001" },
      { id: "fx0002", mag: 6.3, place: "Offshore Fixture Trench", lat: 38.3, lon: -123.1, depth: 22.0, time: 1, updated: 1, tsunami: true, sig: 820, net: "us", magType: "mww", type: "earthquake", status: "reviewed", url: "https://earthquake.usgs.gov/earthquakes/eventpage/fx0002" },
    ],
  },
  "/api/data/volcanoes": {
    kind: "raw", source: "USGS Volcano Hazards Program + Global Volcanism Program, Smithsonian Institution (fixture)", time: 1, count: 2,
    volcanoes: [
      { vnum: "fx0001", name: "Fixture Sitkin", obs: "avo", obsFullname: "Alaska Volcano Observatory", alertLevel: "WATCH", colorCode: "ORANGE", noticeId: "fx-notice-1", noticeUrl: "https://volcanoes.usgs.gov/hans-public/notice/fx-notice-1", sentUtc: "2026-08-01 19:17:44", sentUnixtime: 1, lat: 52.076, lon: -176.13, elevationM: 1740, country: "United States", rt: "2026-08-02" },
      { vnum: "fx0002", name: "Fixture Kilauea", obs: "hvo", obsFullname: "Hawaiian Volcano Observatory", alertLevel: "ADVISORY", colorCode: "YELLOW", noticeId: "fx-notice-2", noticeUrl: "https://volcanoes.usgs.gov/hans-public/notice/fx-notice-2", sentUtc: "2026-08-01 19:07:02", sentUnixtime: 2, lat: 19.421, lon: -155.287, elevationM: 1222, country: "United States", rt: "2026-08-02" },
    ],
  },
  "/api/data/buoys": {
    kind: "raw", source: "NOAA National Data Buoy Center (fixture)", time: 1, count: 2,
    buoys: [
      { station: "46042", lat: 36.79, lon: -122.4, time: 1, windDir: 300, windSpeed: 7.2, gust: 9.1, waveHeight: 2.1, dominantPeriod: 11, avgPeriod: 8.4, waveDir: 285, pressure: 1015.2, pressureTendency: -0.4, airTemp: 14.3, waterTemp: 13.8, dewpoint: 10.1, visibility: null, tide: null },
      { station: "44013", lat: 42.35, lon: -70.65, time: 1, windDir: 210, windSpeed: 5.1, gust: 6.8, waveHeight: null, dominantPeriod: null, avgPeriod: null, waveDir: null, pressure: 1009.8, pressureTendency: 0.2, airTemp: 18.6, waterTemp: 17.9, dewpoint: 15.0, visibility: null, tide: null },
    ],
  },
  "/api/data/sites": {
    kind: "raw",
    categories: {
      tank_farm: { label: "Crude storage", color: "#fbb24c" },
      steel_mill: { label: "Steel mills", color: "#ff5a6e" },
      port: { label: "Ports", color: "#4ade80" },
    },
    sites: [
      { id: "cushing", name: "Cushing Oil Hub", category: "tank_farm", lat: 35.985, lon: -96.767, operator: "Multiple", relevance: "WTI delivery point; Sentinel-2 tank-fill signal ground (EIA benchmark).", note: "" },
      { id: "port_la", name: "Port of Los Angeles", category: "port", lat: 33.74, lon: -118.272, operator: "POLA", relevance: "#1 US container port.", note: "" },
    ],
  },
  "/api/data/insider": {
    kind: "raw", source: "SEC EDGAR (Form 4)", time: 1, count: 2,
    filings: [
      {
        issuerName: "CYPHERPUNK TECHNOLOGIES INC.", issuerTradingSymbol: "CYPH",
        owners: [{ cik: "1645967", name: "Richard Christian M", isDirector: true, isOfficer: false, isTenPercentOwner: false, officerTitle: null }],
        transactions: [{ table: "derivative", kind: "award_grant", shares: 75000, pricePerShare: 0, transactionDate: "2026-07-01" }],
      },
      {
        issuerName: "STRATUS PROPERTIES INC", issuerTradingSymbol: "STRS",
        owners: [{ cik: "1317904", name: "Oasis Management Co Ltd.", isDirector: false, isOfficer: false, isTenPercentOwner: true, officerTitle: null }],
        transactions: [{ table: "nonDerivative", kind: "open_market_sale", shares: 10000, pricePerShare: 28.9, transactionDate: "2026-06-30" }],
      },
    ],
  },
  "/api/health": { status: "ok", checks: {} },
  "/api/data/boundaries": {
    kind: "raw", source: "Natural Earth 1:110m admin-0 (public domain, fixture)",
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { name: "United States of America", iso3: "USA" },
        geometry: { type: "LineString", coordinates: [[-125, 49], [-66, 49], [-66, 25], [-125, 25], [-125, 49]] } },
    ],
  },
  // [REPAIR 2026-09-27] boundaries_admin1 fixture — see the note above the
  // fixture-coverage sweep's 12 new `layers` rows. Response shape is
  // `{kind, source, ...(the compiled FeatureCollection)}` per server/
  // routes.ts's `/api/data/boundaries_admin1` handler, same
  // spread-a-static-GeoJSON pattern as /api/data/boundaries just above.
  "/api/data/boundaries_admin1": {
    kind: "raw", source: "Natural Earth 1:50m admin-1 (public domain, fixture)",
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { adm0: "United States of America", iso3: "USA", min_zoom: 3 },
        geometry: { type: "LineString", coordinates: [[-104, 49], [-104, 41], [-96, 41]] } },
      { type: "Feature", properties: { adm0: "Canada", iso3: "CAN", min_zoom: 3 },
        geometry: { type: "LineString", coordinates: [[-110, 60], [-110, 49], [-102, 49]] } },
    ],
  },
  "/api/data/weather/global/status": { status: "ok", note: "fixture: key active" },
  "/api/data/platform/stats": { layers_live: 18, layers_total: 19, streams_recording: 20, observations: 1284550, observations_as_of: "2026-07-05T00:00:00Z" },
  "/api/data/weather/grid": {
    kind: "raw", source: "OpenWeatherMap current-weather point samples (fixture)",
    note: "sampled grid — one observation per ~310 km; arrows/labels never denser than the data",
    spacing_km: 310, sampled: 6,
    points: [
      { la: 30, lo: -110, tc: 31.5, wd: 200, ws: 6.2 }, { la: 30, lo: -95, tc: 29.0, wd: 180, ws: 4.0 },
      { la: 30, lo: -80, tc: 27.4, wd: 150, ws: 5.1 }, { la: 42, lo: -110, tc: 22.1, wd: 250, ws: 8.8 },
      { la: 42, lo: -95, tc: 24.7, wd: 230, ws: 7.3 }, { la: 42, lo: -80, tc: 21.9, wd: 270, ws: 9.9 },
    ],
  },
  "/api/v1/meta": {
    version: "v1",
    auth: "x-api-key header (or ?api_key=). Keys are invite-only during the preview — join the waitlist on /developers.",
    endpoints: [
      { path: "/api/v1/tracks/:kind/:id", params: "kind; id; ?hours<=168", desc: "Recent position track from our archive." },
      { path: "/api/v1/stats/portdwell", params: "-", desc: "Per-port dwell statistics.", preview: "/api/data/portdwell" },
      { path: "/api/v1/stats/shadow", params: "-", desc: "Dark-ship RAW statistics.", preview: "/api/data/shadowstats" },
      { path: "/api/v1/stats/archive", params: "-", desc: "Archive growth metadata.", preview: "/api/data/archive/stats" },
      { path: "/api/v1/graph", params: "?entity=<ticker|MMSI|CIK|facility id>&hops<=3 (omit entity for counts-only)", desc: "Everything Graph v1 — entity neighborhood join.", preview: "/api/data/graph" },
      { path: "/api/v1/meta", params: "-", desc: "This document.", preview: "/api/v1/meta" },
    ],
    coming_gated: ["entity timelines (Graph v1)", "tank-fill readings (gate 2)"],
    limits: {
      dev: { perMinute: 60, perDay: 10000 },
      pro: { perMinute: 600, perDay: 100000 },
      enterprise: { perMinute: 3000, perDay: 1000000 },
    },
    license_marks: {
      "tracks/aircraft": { license: "ODbL 1.0 share-alike (fixture)", attribution: "adsb.lol", resell: "share-alike" },
      "stats/archive": { license: "VolTradeAI operational metadata", attribution: "VolTradeAI", resell: "ok" },
    },
    disclaimer: "Data as-is; not for safety-of-life use (fixture).",
  },
  "/api/data/archive/stats": { kinds: { aircraft: { days: 2, samples: 96 }, vessels: { days: 2, samples: 96 } }, totalBytes: 12582912 },
  // Backs the newsletterissue PAGES entry (multi-segment route regression,
  // see vite.config.ts's base:"/" fix) — matched via startsWith prefix, so
  // any /api/newsletter/issues/:slug request gets this single-issue shape.
  "/api/newsletter/issues": {
    slug: "test-issue", title: "Fixture Issue", subtitle: "Harness fixture",
    date: "2026-07-11", excerpt: "Fixture excerpt.", html: "<p>Fixture body.</p>",
  },
  "/api/data/portdwell": {
    kind: "raw", source: "Derived from our own AIS position archive (fixture)",
    window_hours: 168, vessels_seen: 240, visits_completed: 23, in_port_now: 7, anomaly_count: 1,
    ports: [
      { id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, visits_completed: 12, unique_vessels: 11, in_port_now: 4, dwell_median_h: 22.5, dwell_p90_h: 41, dwell_max_h: 78, anomaly_count: 1, anomaly_examples: [{ mmsi: "366999707", name: "FIXTURE STAR", dwell_h: 78, median_h: 22.5 }] },
      { id: "port_lb", name: "Port of Long Beach", lat: 33.7515, lon: -118.213, visits_completed: 8, unique_vessels: 8, in_port_now: 2, dwell_median_h: 19, dwell_p90_h: 30, dwell_max_h: 33, anomaly_count: 0, anomaly_examples: [] },
      { id: "port_savannah", name: "Port of Savannah (Garden City)", lat: 32.129, lon: -81.144, visits_completed: 3, unique_vessels: 3, in_port_now: 1, dwell_median_h: 15, dwell_p90_h: 18, dwell_max_h: 18, anomaly_count: 0, anomaly_examples: [] },
    ],
    caveat: "RAW statistics; dwell figures are lower bounds; anomaly flags suppressed on thin history (fixture).",
  },
  "/api/data/graph": {
    kind: "raw", built_at: 1,
    counts: { nodes: 42, edges: 65, company: 9, person: 14, facility: 16, vessel: 3, insider_of: 18, operates: 12, calls_at: 31, owns: 4 },
    caveat: "RAW graph — every edge asserts a relationship with provenance (fixture).",
    note: "pass ?entity=<ticker|MMSI|CIK|facility id>&hops=1 for a neighborhood query",
  },
  // W5 ENTITY DOSSIER — click-a-feature card enrichment. Non-empty so any
  // page load that happens to click an entity exercises the real render
  // path (companyNode/insider/contracts/nearest_sites all populated), not
  // just an empty-state early-return.
  "/api/data/dossier": {
    kind: "raw", built_at: 1,
    query: { entity: "cushing_hub", lat: 35.94, lon: -96.75, hops: 2 },
    identity: { id: "facility:site:cushing_hub", type: "facility", label: "Cushing Oil Hub", attrs: { lat: 35.94, lon: -96.75 } },
    graph: {
      nodes: [{ id: "company:ENB", type: "company", label: "ENB", attrs: {} }],
      edges: [
        { type: "operates", from: "company:ENB", to: "facility:site:cushing_hub", confidence: "high", attrs: {} },
        { type: "insider_of", from: "person:0000123", to: "company:ENB", confidence: "high", attrs: { visit_count: 0 } },
      ],
    },
    contracts: [{ r: "Enbridge Inc", tkr: "ENB", amt: 1250000, ag: "Department of Energy", rt: "2026-07-01" }],
    contracts_capped: false,
    nearest_sites: [{ id: "la_port", name: "Port of LA", category: "port", km: 1500 }],
    caveat: "RAW composition — fixture.",
  },
  // Grid-stress descriptive reading (GRID VISION A1 gate-2 FAIL path,
  // 2026-07-07) — descriptive-only, predictive:false always present.
  "/api/data/grid-stress": {
    kind: "descriptive",
    source: "VoltradeAI derived composite (EIA-930 ERCOT demand/forecast + NOAA CPC TX degree days) (fixture)",
    attribution: "EIA-930 Hourly Electric Grid Monitor; NOAA Climate Prediction Center",
    predictive: false,
    validation_status: "GATE 2 NOT PASSED (2026-07-07) — two outcome designs voided on their own spot-validation rules (fixture)",
    note: "Descriptive-only TX/ERCOT reading; equal-weighted composite, deliberately not the voided gate-2 fitted weights (fixture).",
    time: 1, history_depth_days: 412,
    reading: {
      date: "2026-07-06", respondent: "ERCO",
      demand_peak_mw: 68500, demand_percentile: 82, demand_sample_days: 41,
      forecast_strain_pct: 1.4, strain_percentile: 61, strain_sample_days: 41,
      weather_degree_days: 27, weather_percentile: 74, weather_sample_days: 62,
      composite_percentile: 72,
    },
  },
  // Crop conditions — USDA NASS weekly ratings (2026-08-04, RAW display).
  "/api/data/crop-conditions": {
    kind: "raw",
    source: "USDA NASS QuickStats — weekly crop condition ratings (US government data) (fixture)",
    attribution: "USDA National Agricultural Statistics Service (QuickStats)",
    time: "2026-08-03T00:00:00.000Z",
    latest_week: "2026-08-02",
    count: 10,
    note: "national weekly CONDITION ratings for corn + soybeans (fixture)",
    rows: [
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT VERY POOR", pct: 4, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT POOR", pct: 10, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT FAIR", pct: 25, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT GOOD", pct: 48, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT EXCELLENT", pct: 13, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT VERY POOR", pct: 2, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT POOR", pct: 7, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT FAIR", pct: 28, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT GOOD", pct: 52, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT EXCELLENT", pct: 11, rt: "2026-08-03" },
    ],
  },
  // Drought severity — U.S. Drought Monitor weekly (2026-08-20, RAW
  // display). Two AOIs each with a recent + ~4-week-prior reading so the
  // page's DSCI-delta computation and both severity-band colors exercise.
  "/api/data/drought": {
    kind: "raw",
    source: "U.S. Drought Monitor data services — free with credit (fixture)",
    attribution: "U.S. Drought Monitor (NDMC/USDA/NOAA)",
    time: 1787171940919,
    count: 4,
    note: "weekly cumulative D0-D4 area %; dsci is the USDM's own published index (DERIVED here from the percentages) (fixture)",
    drought: [
      { aoi: "CONUS", map_date: "2026-08-11", none: 26.38, d0: 73.62, d1: 50.38, d2: 29.5, d3: 10.27, d4: 1.04, dsci: 164.81, valid_start: "2026-08-11", valid_end: "2026-08-17", rt: "2026-08-19" },
      { aoi: "CONUS", map_date: "2026-07-14", none: 33.1, d0: 66.9, d1: 41.2, d2: 21.4, d3: 6.8, d4: 0.5, dsci: 136.8, valid_start: "2026-07-14", valid_end: "2026-07-20", rt: "2026-08-19" },
      { aoi: "IA", map_date: "2026-08-11", none: 18.5, d0: 81.5, d1: 62.3, d2: 38.9, d3: 15.2, d4: 2.1, dsci: 200.0, valid_start: "2026-08-11", valid_end: "2026-08-17", rt: "2026-08-19" },
      { aoi: "IA", map_date: "2026-07-14", none: 24.0, d0: 76.0, d1: 55.0, d2: 30.0, d3: 9.0, d4: 0.0, dsci: 170.0, valid_start: "2026-07-14", valid_end: "2026-07-20", rt: "2026-08-19" },
    ],
  },
  // Facilities near active fires — NASA FIRMS × strategic-sites cross-tie
  // (2026-08-20, RAW display). Two facilities so the table and both null-
  // vs-populated max_frp cases exercise.
  "/api/data/fires-near-facilities": {
    enabled: true,
    kind: "raw",
    as_of: 1787230128775,
    radius_km: 25,
    fires_checked: 24286,
    facilities_with_fires: 2,
    facilities: [
      { id: "port_houston", name: "Port of Houston (Barbours Cut)", lat: 29.677, lon: -95.006, category: "port", operator: "Port Houston", fires_within: 3, nearest_km: 7.3, max_frp: 20.01 },
      { id: "port_seattle", name: "Port of Seattle", lat: 47.582, lon: -122.3474, category: "port", operator: "NWSA", fires_within: 2, nearest_km: 1.6, max_frp: null },
    ],
    note: "RAW cross-tie: strategic assets with active VIIRS 375m fire detections nearby (NRT ~3h). No predictive claim — the fires-near-assets → insurer/utility hypothesis is validation-gated. Not for safety-of-life use. (fixture)",
    source: "NASA FIRMS/LANCE (VIIRS_SNPP_NRT) × datacore/sites strategic facilities",
  },
  // Air quality at strategic sites (2026-08-20, RAW display). Three sites so
  // the fixture exercises the cases that can misrender: a provider colour with
  // an OMITTED channel (usa_epa green-only), a full three-channel colour, and a
  // reading whose two indexes disagree on the dominant pollutant. Values are
  // the live 2026-08-20 shapes.
  "/api/data/air-quality": {
    kind: "raw",
    source: "Google Maps Platform Air Quality API (current conditions) (fixture)",
    attribution: "Air quality data © Google",
    time: 1787241252746,
    awaiting_enable: false,
    note: "latest current-conditions reading per strategic site; universal + US EPA AQI, PM2.5/NO2; archived over time (Google exposes only 30d history). (fixture)",
    budget: {
      monthly_free_tier: 5000, daily_call_budget: 150, poll_interval_hours: 3,
      site_count: 16, sites_this_cycle: 16, rotating_subset: false, projected_calls_per_day: 128,
    },
    count: 3,
    sites: [
      {
        siteId: "cushing_enbridge", dateTime: "2026-08-20T13:00:00Z",
        uaqi: 71, uaqi_category: "Good air quality", uaqi_dominant: "pm10",
        epa_aqi: 41, epa_category: "Good air quality", epa_dominant: "pm25",
        pm25: 12.27, pm25_units: "MICROGRAMS_PER_CUBIC_METER",
        no2: 7.84, no2_units: "PARTS_PER_BILLION",
        indexes: [
          { code: "uaqi", displayName: "Universal AQI", aqi: 71, color: { red: 0.49803922, green: 0.8039216, blue: 0.2 }, category: "Good air quality", dominantPollutant: "pm10" },
          { code: "usa_epa", displayName: "AQI (US)", aqi: 41, color: { green: 0.89411765 }, category: "Good air quality", dominantPollutant: "pm25" },
        ],
        rt: "2026-08-20",
      },
      {
        siteId: "port_oakland", dateTime: "2026-08-20T13:00:00Z",
        uaqi: 83, uaqi_category: "Excellent air quality", uaqi_dominant: "pm10",
        epa_aqi: 32, epa_category: "Good air quality", epa_dominant: "pm25",
        pm25: 2.91, pm25_units: "MICROGRAMS_PER_CUBIC_METER",
        no2: 5.7, no2_units: "PARTS_PER_BILLION",
        indexes: [
          { code: "uaqi", displayName: "Universal AQI", aqi: 83, color: { red: 0.2, green: 0.8, blue: 0.4 }, category: "Excellent air quality", dominantPollutant: "pm10" },
          { code: "usa_epa", displayName: "AQI (US)", aqi: 32, color: { green: 0.89411765 }, category: "Good air quality", dominantPollutant: "pm25" },
        ],
        rt: "2026-08-20",
      },
      {
        siteId: "port_savannah", dateTime: "2026-08-20T13:00:00Z",
        uaqi: 46, uaqi_category: "Moderate air quality", uaqi_dominant: "pm25",
        epa_aqi: 62, epa_category: "Moderate air quality", epa_dominant: "pm25",
        pm25: 25.75, pm25_units: "MICROGRAMS_PER_CUBIC_METER",
        no2: 5.01, no2_units: "PARTS_PER_BILLION",
        indexes: [
          { code: "uaqi", displayName: "Universal AQI", aqi: 46, color: { red: 1, green: 0.87, blue: 0.2 }, category: "Moderate air quality", dominantPollutant: "pm25" },
          { code: "usa_epa", displayName: "AQI (US)", aqi: 62, color: { red: 1, green: 1 }, category: "Moderate air quality", dominantPollutant: "pm25" },
        ],
        rt: "2026-08-20",
      },
    ],
    issues: {},
  },
  // US port imports — Census FT920 (2026-08-20, RAW display). Shaped to
  // exercise every case that can misrender, using live 2026-08-20 values: the
  // national aggregate row (port code "-", which must NOT appear in the port
  // table), a seaport with real containerized figures, a land-border port
  // publishing a genuine 0 for containerized cargo, a port present only in the
  // newest month (undefined month-over-month change), and a port Census
  // publishes no name for. The per-port rows sum EXACTLY to the aggregate in
  // both months, so the page's live reconciliation line renders its pass state.
  "/api/data/imports": {
    kind: "raw",
    source: "US Census Bureau intltrade (imports/porths, FT920 monthly) — public domain (fixture)",
    attribution: "U.S. Census Bureau (USA Trade Online / FT920)",
    time: 1787232272476,
    count: 8,
    note: "monthly port-level import values incl. containerized (~45-day lag); revisions append as new vintages in the archive (fixture)",
    imports: [
      { port: "-", port_name: "TOTAL FOR ALL PORTS", month: "2026-06", gen_val: 47466425097, cnt_val: 21049766472, cnt_wgt: 3940035515, rt: "2026-08-20" },
      { port: "-", port_name: "TOTAL FOR ALL PORTS", month: "2026-05", gen_val: 44883911953, cnt_val: 19026611264, cnt_wgt: 3665704530, rt: "2026-08-20" },
      { port: "2704", port_name: "LOS ANGELES, CA", month: "2026-06", gen_val: 22917449056, cnt_val: 21049766472, cnt_wgt: 3940035515, rt: "2026-08-20" },
      { port: "2704", port_name: "LOS ANGELES, CA", month: "2026-05", gen_val: 20595308379, cnt_val: 19026611264, cnt_wgt: 3665704530, rt: "2026-08-20" },
      { port: "2304", port_name: "LAREDO, TX", month: "2026-06", gen_val: 24201376152, cnt_val: 0, cnt_wgt: 0, rt: "2026-08-20" },
      { port: "2304", port_name: "LAREDO, TX", month: "2026-05", gen_val: 24288603574, cnt_val: 0, cnt_wgt: 0, rt: "2026-08-20" },
      { port: "3013", port_name: "GREAT FALLS, MT", month: "2026-06", gen_val: 270289178, cnt_val: 0, cnt_wgt: 0, rt: "2026-08-20" },
      { port: "1791", port_name: null, month: "2026-06", gen_val: 77310711, cnt_val: 0, cnt_wgt: 0, rt: "2026-08-20" },
    ],
  },
  // GDELT media events near tracked facilities (2026-08-20, RAW display).
  // Shaped from the live 2026-08-20 payload to exercise every case that can
  // misrender: one article producing TWO event rows (so the row count and the
  // incident count differ on screen), one article geocoded into TWO different
  // facilities' boxes, a genuinely POSITIVE article tone under a coercion code
  // (the live sake-bar article at port_oakland — the feed's false-positive
  // rate made visible), and one code absent from our transcribed decode table
  // so the "not in our decode table" branch renders instead of a guessed
  // label. Every Goldstein value matches the published CAMEO constant for its
  // code, so the page's live check renders its pass state.
  "/api/data/facility-events": {
    kind: "raw",
    source: "The GDELT Project (https://www.gdeltproject.org/) — unrest/strike mentions near tracked facilities (fixture)",
    attribution: "The GDELT Project",
    time: 1787256837023,
    count: 6,
    note: "media event MENTIONS, geo approximate — verification prompts for our own sensors, not confirmations (fixture)",
    events: [
      { id: "1319205286", day: "20260820", code: "192", root: "19", gold: -9.5, tone: -4.56310679611651, mentions: 7, lat: 34.0339, lon: -118.205, site: "port_la", url: "https://laist.com/news/climate-environment/lineage-faces-key-deadline-today-to-remove-food-from-boyle-heights-warehouse", rt: "2026-08-20T19:13:56.905Z" },
      { id: "1319205287", day: "20260820", code: "192", root: "19", gold: -9.5, tone: -4.56310679611651, mentions: 3, lat: 34.0522, lon: -118.244, site: "port_la", url: "https://laist.com/news/climate-environment/lineage-faces-key-deadline-today-to-remove-food-from-boyle-heights-warehouse", rt: "2026-08-20T19:13:56.905Z" },
      { id: "1319210101", day: "20260820", code: "173", root: "17", gold: -5, tone: -7.51252086811353, mentions: 8, lat: 32.7765, lon: -79.9311, site: "port_charleston", url: "https://www.christianpost.com/news/acna-court-finds-breakaway-bishop-guilty-of-misconduct.html", rt: "2026-08-20T19:43:56.910Z" },
      { id: "1319210102", day: "20260820", code: "141", root: "14", gold: -6.5, tone: -7.51252086811353, mentions: 5, lat: 32.0809, lon: -81.0912, site: "port_savannah", url: "https://www.christianpost.com/news/acna-court-finds-breakaway-bishop-guilty-of-misconduct.html", rt: "2026-08-20T19:43:56.910Z" },
      { id: "1319209440", day: "20260820", code: "172", root: "17", gold: -5, tone: 7.34463276836158, mentions: 8, lat: 37.8044, lon: -122.2712, site: "port_oakland", url: "https://heritageradionetwork.org/episode/dokidoki-innovative-sake-shochu-cocktails-in-japanese-tachinomi-vibes/", rt: "2026-08-20T19:58:56.918Z" },
      { id: "1319211777", day: "20260820", code: "1799", root: "17", gold: -7.4, tone: -6.1, mentions: 2, lat: 40.7834, lon: -73.9662, site: "port_nynj", url: "https://www.example-news.test/a-code-our-table-predates.html", rt: "2026-08-20T20:13:56.923Z" },
    ],
  },
  // ── UI coverage batch C (2026-11-20) ──────────────────────────────────────
  // History behind three panels. Each fixture carries the case that can
  // misrender: a week / day with no reading (must be a gap, never zero).
  "/api/data/crop-conditions/history": {
    kind: "raw", weeks: 52,
    source: "USDA NASS QuickStats — weekly crop condition ratings (fixture)",
    trend: [
      { week_ending: "2026-09-20", corn: { EXCELLENT: 14, GOOD: 46, FAIR: 26, POOR: 10, "VERY POOR": 4 }, soybeans: { EXCELLENT: 12, GOOD: 48, FAIR: 27, POOR: 9, "VERY POOR": 4 } },
      { week_ending: "2026-09-27", corn: { EXCELLENT: 13, GOOD: 45, FAIR: 27, POOR: 11, "VERY POOR": 4 }, soybeans: { EXCELLENT: 11, GOOD: 48, FAIR: 28, POOR: 9, "VERY POOR": 4 } },
      { week_ending: "2026-10-04", corn: { EXCELLENT: 13, GOOD: 44, FAIR: 27, POOR: 11, "VERY POOR": 5 }, soybeans: { EXCELLENT: 11, GOOD: 47, FAIR: 28, POOR: 10, "VERY POOR": 4 } },
      { week_ending: "2026-10-11", corn: { EXCELLENT: 13, GOOD: 44, FAIR: 26, POOR: 12, "VERY POOR": 5 }, soybeans: {} },
      { week_ending: "2026-10-18", corn: { EXCELLENT: 13, GOOD: 43, FAIR: 27, POOR: 12, "VERY POOR": 5 }, soybeans: {} },
    ],
    note: "(fixture)",
  },
  "/api/data/appstore-rankings/history": {
    kind: "raw", days: 30,
    trend: [
      { date: "2026-08-01", ranked_slots: 22, total_slots: 48, total_rating_count: 159313303 },
      { date: "2026-08-02", ranked_slots: 21, total_slots: 48, total_rating_count: 159401220 },
      { date: "2026-08-03", ranked_slots: 0, total_slots: 0, total_rating_count: null },
      { date: "2026-08-04", ranked_slots: 20, total_slots: 48, total_rating_count: 159580771 },
      { date: "2026-08-05", ranked_slots: 18, total_slots: 48, total_rating_count: 159660012 },
    ],
    note: "(fixture)",
  },
  "/api/data/github-activity/history": {
    kind: "raw", weeks: 26,
    trend: [
      { weekStart: "2026-07-06", weekEnd: "2026-07-12", orgs_reporting: 14, total_merged_prs: 4870, total_commits: 5510 },
      { weekStart: "2026-07-13", weekEnd: "2026-07-19", orgs_reporting: 15, total_merged_prs: 5012, total_commits: null },
      { weekStart: "2026-07-20", weekEnd: "2026-07-26", orgs_reporting: 15, total_merged_prs: 5118, total_commits: 5834 },
      { weekStart: "2026-07-27", weekEnd: "2026-08-02", orgs_reporting: 15, total_merged_prs: 5240, total_commits: 6010 },
    ],
    note: "(fixture)",
  },
  // Plants under NWS warnings: one warning with more plants than the list
  // opens on (the "show the other N" control), a plant with no stated
  // capacity (server writes 0), and one warning with no end time.
  "/api/data/plants-under-alerts": {
    kind: "raw", as_of: Date.now() - 4 * 60000,
    alerts_checked: 10, zone_only_excluded: 421, plants_checked: 14428, alerts_with_plants: 2,
    alerts: [
      {
        id: "urn:fixture:1", event: "Flood Warning", severity: "Severe", area: "Brevard, FL", ends: new Date(Date.now() + 95 * 60000).toISOString(),
        plants: [
          { name: "Cape Canaveral", fuel: "gas", capacity_mw: 1295, operator: "Florida Power & Light Co" },
          { name: "Oleander Power Project LP", fuel: "gas", capacity_mw: 994.4, operator: "Oleander Holdings LLC" },
          { name: "Indian River", fuel: "gas", capacity_mw: 609, operator: "Orlando Utilities Comm" },
          { name: "Indian River Plant", fuel: "gas", capacity_mw: 342.8, operator: "Orlando Utilities Comm" },
          { name: "Discovery Solar Center", fuel: "solar", capacity_mw: 74.5, operator: "Florida Power & Light Co" },
          { name: "Space Coast Next Gen Solar Energy", fuel: "solar", capacity_mw: 10, operator: "Florida Power & Light Co" },
          { name: "Brevard", fuel: "other", capacity_mw: 9.6, operator: "Brevard Energy  LLC" },
          { name: "Fixture Landfill Gas", fuel: "other", capacity_mw: 4.8, operator: null },
          { name: "Fixture Rooftop Solar One", fuel: "solar", capacity_mw: 1.2, operator: "Fixture Solar LLC" },
          { name: "Fixture Unrated Unit", fuel: "oil", capacity_mw: 0, operator: "Fixture Municipal Utility" },
        ],
        plant_count: 10, total_capacity_mw: 3341, capacity_by_fuel: { gas: 3241.2000000000003, solar: 85.7, other: 14.399999999999999, oil: 0 },
      },
      {
        id: "urn:fixture:2", event: "Flood Warning", severity: "Severe", area: "Seminole, FL", ends: null,
        plants: [
          { name: "Sanford", fuel: "gas", capacity_mw: 2377.8, operator: "Florida Power & Light Co" },
          { name: "DeBary", fuel: "oil", capacity_mw: 748, operator: "Duke Energy Florida  LLC" },
        ],
        plant_count: 2, total_capacity_mw: 3126, capacity_by_fuel: { gas: 2377.8, oil: 748 },
      },
    ],
    note: "RAW cross-tie (fixture)",
    source: "NWS api.weather.gov active warnings (public domain) × WRI Global Power Plant DB (CC BY 4.0) (fixture)",
  },
  // Rail carloads: a commodity with no prior week (the change is a dash).
  "/api/data/rail-traffic": {
    kind: "raw", predictive: false,
    source: "STB EP724 consolidated rail service data (public domain) (fixture)",
    attribution: "Surface Transportation Board (EP 724 rail service data)",
    archiveBuiltAt: "2026-11-02T13:39:52.923615+00:00",
    latestWeek: "2026-10-28", priorWeek: "2026-10-21", weeksArchived: 501, railroadsReporting: 6,
    systemLatestWeekCarloads: 741292, systemIntermodalLatestWeekCarloads: 372100, systemNonIntermodalLatestWeekCarloads: 369192,
    commodities: [
      { variable: "Containers", latestWeekCarloads: 362683, priorWeekCarloads: 361708, weekOverWeekDeltaPct: 0.3, isIntermodal: true },
      { variable: "Coal", latestWeekCarloads: 70695, priorWeekCarloads: 70042, weekOverWeekDeltaPct: 0.9, isIntermodal: false },
      { variable: "Non Metallic Minerals", latestWeekCarloads: 3805, priorWeekCarloads: 4425, weekOverWeekDeltaPct: -14, isIntermodal: false },
      { variable: "Farm Products except Grain", latestWeekCarloads: 1315, priorWeekCarloads: null, weekOverWeekDeltaPct: null, isIntermodal: false },
    ],
    railroads: [
      { railroad: "BNSF", latestWeekCarloads: 205978 }, { railroad: "UP", latestWeekCarloads: 178728 },
      { railroad: "NS", latestWeekCarloads: 144281 }, { railroad: "CSX", latestWeekCarloads: 135944 },
      { railroad: "CN", latestWeekCarloads: 46333 }, { railroad: "CPKC", latestWeekCarloads: 30028 },
    ],
    note: "GATE 1 (data) PASSED 2026-09-27. GATE 2 (signal) REJECTED 2026-09-28. Shown here as RAW, self-reported weekly carload counts only, no predictive claim. (fixture)",
  },
  // App Store rankings — consumer-app watchlist (2026-08-05, RAW display).
  "/api/data/appstore-rankings": {
    kind: "raw",
    source: "Apple marketingtools RSS top-free/top-grossing charts (US/GB/CA) + iTunes Lookup rating counts (US) (fixture)",
    attribution: "Apple Inc.",
    time: 1785928979602,
    count: 6,
    note: "GATE 1 (DATA) only — no signal validated yet (fixture).",
    records: [
      { t: "rank", ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", storefront: "us", chart: "top-free", rank: 65, rt: "2026-08-05" },
      { t: "rank", ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", storefront: "us", chart: "top-grossing", rank: 22, rt: "2026-08-05" },
      { t: "rank", ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", storefront: "us", chart: "top-free", rank: null, rt: "2026-08-05" },
      { t: "rank", ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", storefront: "us", chart: "top-grossing", rank: null, rt: "2026-08-05" },
      { t: "rating", ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", avgRating: 4.72, ratingCount: 5357702, version: "7.133.0", rt: "2026-08-05" },
      { t: "rating", ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", avgRating: 3.9, ratingCount: 412300, version: "6.2.0", rt: "2026-08-05" },
    ],
  },
  // GitHub org engineering-momentum — develop-in-public watchlist
  // (2026-08-05, RAW display).
  "/api/data/github-activity": {
    kind: "raw",
    source: "GitHub REST Search API (search/issues, search/commits) — keyless (fixture)",
    attribution: "GitHub, Inc. (public repository activity, aggregated)",
    time: 1785928979602,
    count: 4,
    note: "GATE 1 (DATA) only — no signal validated yet (fixture).",
    records: [
      { t: "org_week", key: "2026-07-27|mongodb", ticker: "MDB", org: "mongodb", weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: 41, commits: 118, uniqueActorsSample: 37, actorSampleCapped: true, rt: "2026-08-03" },
      { t: "org_week", key: "2026-07-27|PagerDuty", ticker: "PD", org: "PagerDuty", weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: 12, commits: 29, uniqueActorsSample: 9, actorSampleCapped: false, rt: "2026-08-03" },
      { t: "org_week", key: "2026-07-27|jfrog", ticker: "FROG", org: "jfrog", weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: 6, commits: null, uniqueActorsSample: null, actorSampleCapped: false, rt: "2026-08-03" },
      { t: "org_week", key: "2026-07-20|mongodb", ticker: "MDB", org: "mongodb", weekStart: "2026-07-20", weekEnd: "2026-07-26", mergedPRs: 35, commits: 102, uniqueActorsSample: 33, actorSampleCapped: true, rt: "2026-07-27" },
    ],
  },
  // Cboe VIX term structure (2026-08-07, RAW display).
  "/api/data/vix-term-structure": {
    kind: "raw",
    source: "Cboe Global Markets — VIX1D/VIX9D/VIX/VIX3M/VIX6M/VVIX daily close (fixture)",
    attribution: "Cboe Global Markets volatility indices",
    time: 1786102149033,
    latest: {
      date: "2026-08-06", vix1d: 12.55, vix9d: 12.66, vix: 15.15, vix3m: 18.69, vix6m: 20.91, vvix: 88.72,
      vix_vix3m_ratio: 0.8106, vix9d_vix_ratio: 0.8356,
    },
    recent: [
      { date: "2026-08-05", vix1d: 13.02, vix9d: 13.4, vix: 15.81, vix3m: 19.02, vix6m: 21.2, vvix: 89.5, vix_vix3m_ratio: 0.8313, vix9d_vix_ratio: 0.8477 },
      { date: "2026-08-06", vix1d: 12.55, vix9d: 12.66, vix: 15.15, vix3m: 18.69, vix6m: 20.91, vvix: 88.72, vix_vix3m_ratio: 0.8106, vix9d_vix_ratio: 0.8356 },
    ],
    note: "GATE 1 (DATA) only — ratios are plain arithmetic, no predictive claim (fixture).",
  },
  // JODI world oil closing-stock levels (2026-08-21, RAW display; GATE 2
  // signal killed 2026-08-06 — the note field restates that verbatim).
  "/api/data/jodi-oil-stocks": {
    kind: "raw",
    predictive: false,
    source: "JODI Oil World Primary database (fixture)",
    attribution: "JODI Oil World Database (Joint Organisations Data Initiative)",
    license: "free with acknowledgment (fixture)",
    product: "TOTCRUDE",
    archiveLatestPeriod: "2026-04",
    seriesCount: 350,
    countriesReporting: 2,
    rows: [
      { area: "SA", name: "Saudi Arabia", period: "2026-03", levelKbbl: 260000, assessmentCode: "3", priorPeriod: "2026-02", priorLevelKbbl: 258500, deltaKbbl: 1500, deltaPct: 0.6 },
      { area: "US", name: "United States", period: "2026-03", levelKbbl: 415000, assessmentCode: "3", priorPeriod: "2026-02", priorLevelKbbl: 420000, deltaKbbl: -5000, deltaPct: -1.2 },
    ],
    note: "GATE 1 (data) PASSED — reconciles against EIA within 1.2%. GATE 2 (signal) KILLED 2026-08-06 — " +
      "no significant forward-return signal found. Shown here as RAW, self-reported closing-stock levels " +
      "only — no predictive claim (fixture).",
  },
  // FINRA consolidated short interest + Reg SHO threshold list (2026-08-28,
  // RAW display; DISTINCT from /api/data/short-volume above — settlement
  // POSITIONS, not daily execution flow).
  "/api/data/short-interest": {
    kind: "raw",
    source: "FINRA Query API — consolidated short interest (semi-monthly) + Reg SHO threshold list (daily) (fixture)",
    attribution: "FINRA Query API",
    time: 1786102149033,
    settlement_date: "2026-08-15",
    si_records: 2,
    note: "SHORT INTEREST (positions, ~T+9 semi-monthly publish) — distinct from /api/data/short-volume (daily flow) (fixture).",
    si: {
      settlement_date: "2026-08-15",
      records: 2,
      top_days_to_cover: [
        { symbol: "GME", name: "GameStop Corp", days_to_cover: 6.2, short_qty: 42_500_000, adv: 6_850_000 },
        { symbol: "AMC", name: "AMC Entertainment Holdings", days_to_cover: 4.1, short_qty: 118_000_000, adv: 28_800_000 },
      ],
      top_change_pct: [
        { symbol: "GME", name: "GameStop Corp", change_pct: 12.4, short_qty: 42_500_000 },
        { symbol: "AMC", name: "AMC Entertainment Holdings", change_pct: -3.8, short_qty: 118_000_000 },
      ],
      adv_floor: 100_000,
      position_floor: 1_000_000,
      top_cap: 15,
    },
    threshold: {
      trade_date: "2026-08-27",
      count: 2,
      symbols: [
        { symbol: "GME", name: "GameStop Corp", market: "NYSE" },
        { symbol: "AMC", name: "AMC Entertainment Holdings", market: "NYSE" },
      ],
    },
  },
  // USA bilateral goods trade — UN Comtrade (2026-09-07, RAW display; GATE 1
  // DATA passed same day, GATE 2 not attempted — the note field restates
  // that verbatim, same convention as jodi-oil-stocks above).
  "/api/data/un-comtrade": {
    kind: "raw",
    predictive: false,
    source: "UN Comtrade Database (fixture)",
    attribution: "UN Comtrade Database, https://comtradeapi.un.org",
    license: "free with citation (fixture)",
    reporter: "USA",
    cmdCode: "TOTAL",
    archiveLatestPeriod: "202606",
    partnerCount: 2,
    rows: [
      { partnerCode: 156, partnerName: "China", period: "202606", importsCifUsd: 39566206514, exportsFobUsd: 12558691242, tradeBalanceUsd: -27007515272, priorPeriod: "202605", priorImportsCifUsd: 41200000000, importsDeltaPct: -4.0 },
      { partnerCode: 124, partnerName: "Canada", period: "202606", importsCifUsd: 36191199138, exportsFobUsd: 35403162277, tradeBalanceUsd: -788036861, priorPeriod: "202605", priorImportsCifUsd: 35800000000, importsDeltaPct: 1.1 },
    ],
    note: "GATE 1 (data) PASSED 2026-09-07 — every partner's CIF import value reconciles against FRED's " +
      "independent Census-Bureau customs-basis import series within a stable CIF/customs-basis offset band. " +
      "GATE 2 (signal) not attempted — shown here as RAW, self-reported bilateral trade levels only, no " +
      "predictive claim (fixture).",
  },
  "/api/data/eu-macro": {
    kind: "raw",
    source: "European macro cluster — ECB Data Portal + Eurostat + Deutsche Bundesbank (fixture)",
    attribution: "per-series (each series carries its required attribution string)",
    time: 1786102149033,
    note: "REGIME INPUT feed (never a direct signal). Sources revise in place (fixture).",
    series: [
      { key: "ECB_EURUSD", source: "ecb", label: "EUR/USD reference rate", cadence: "daily", unit: "USD", attribution: "Source: ECB statistics.",
        latest: { d: "2026-08-06", v: 1.0852 }, prev: { d: "2026-08-05", v: 1.0839 },
        history: [{ d: "2026-08-04", v: 1.0821 }, { d: "2026-08-05", v: 1.0839 }, { d: "2026-08-06", v: 1.0852 }] },
      { key: "ECB_ESTR", source: "ecb", label: "€STR (euro short-term rate)", cadence: "daily", unit: "%", attribution: "Source: ECB statistics.",
        latest: { d: "2026-08-06", v: 2.9 }, prev: { d: "2026-08-05", v: 2.9 },
        history: [{ d: "2026-08-04", v: 2.9 }, { d: "2026-08-05", v: 2.9 }, { d: "2026-08-06", v: 2.9 }] },
      { key: "ECB_BS_TOTAL", source: "ecb", label: "Eurosystem balance sheet (total assets)", cadence: "weekly", unit: "EUR millions", attribution: "Source: ECB statistics.",
        latest: { d: "2026-08-01", v: 6412000 }, prev: { d: "2026-07-25", v: 6398000 },
        history: [{ d: "2026-07-18", v: 6385000 }, { d: "2026-07-25", v: 6398000 }, { d: "2026-08-01", v: 6412000 }] },
      { key: "EU_INDPROD", source: "eurostat", label: "EA20 industrial production (SCA, 2021=100)", cadence: "monthly", unit: "index", attribution: "Source: Eurostat, sts_inpr_m",
        latest: { d: "2026-06-01", v: 101.3 }, prev: { d: "2026-05-01", v: 100.9 },
        history: [{ d: "2026-04-01", v: 100.4 }, { d: "2026-05-01", v: 100.9 }, { d: "2026-06-01", v: 101.3 }] },
      { key: "DE_BUND10Y", source: "bbk", label: "10Y Bund yield (Svensson, listed Federal securities)", cadence: "daily", unit: "%", attribution: "Source: Deutsche Bundesbank",
        latest: { d: "2026-08-06", v: 2.42 }, prev: { d: "2026-08-05", v: 2.4 },
        history: [{ d: "2026-08-04", v: 2.38 }, { d: "2026-08-05", v: 2.4 }, { d: "2026-08-06", v: 2.42 }] },
    ],
  },
  // EU power markets — ENTSO-E load/generation-mix/day-ahead-price, 8
  // bidding zones (2026-08-16, RAW display, euPower.tsx). Fixture covers 2
  // zones so the harness exercises the tile grid, the genmix zone picker,
  // and a negative-price point.
  "/api/data/eu-load": {
    kind: "raw",
    source: "ENTSO-E Transparency Platform (actual total load, A65/A16) (fixture)",
    attribution: "ENTSO-E Transparency Platform",
    time: "2026-08-16T12:00",
    count: 2,
    note: "realised total load in MW per bidding zone, ~1-2h publication lag (fixture).",
    zones: [
      { zone: "DE_LU", latest_ts: "2026-08-16T11:00", latest_mw: 58210, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 51200, window_max_mw: 61840, window_mean_mw: 56430 },
      { zone: "FR", latest_ts: "2026-08-16T11:00", latest_mw: 47120, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 41300, window_max_mw: 52900, window_mean_mw: 46810 },
    ],
    issues: {},
  },
  "/api/data/eu-generation-mix": {
    kind: "raw",
    source: "ENTSO-E Transparency Platform (actual generation per type, A75/A16) (fixture)",
    attribution: "ENTSO-E Transparency Platform",
    time: "2026-08-16T12:00",
    count: 4,
    note: "realised generation in MW per bidding zone x fuel/technology type (fixture).",
    zones: [
      { zone: "DE_LU", psr: "B16", psr_name: "Solar", latest_ts: "2026-08-16T11:00", latest_mw: 18400, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 0, window_max_mw: 24100, window_mean_mw: 9800 },
      { zone: "DE_LU", psr: "B19", psr_name: "Wind Onshore", latest_ts: "2026-08-16T11:00", latest_mw: 12100, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 3200, window_max_mw: 19800, window_mean_mw: 11400 },
      { zone: "FR", psr: "B14", psr_name: "Nuclear", latest_ts: "2026-08-16T11:00", latest_mw: 31200, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 29800, window_max_mw: 32100, window_mean_mw: 30900 },
      { zone: "FR", psr: "B12", psr_name: "Hydro Water Reservoir", latest_ts: "2026-08-16T11:00", latest_mw: 4100, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 1800, window_max_mw: 6200, window_mean_mw: 3900 },
    ],
    issues: {},
  },
  "/api/data/eu-day-ahead-prices": {
    kind: "raw",
    source: "ENTSO-E Transparency Platform (day-ahead auction clearing price, A44) (fixture)",
    attribution: "ENTSO-E Transparency Platform",
    time: "2026-08-16T12:00",
    count: 2,
    note: "day-ahead auction clearing price per bidding zone, currency/unit as published; negative prices are real (fixture).",
    zones: [
      { zone: "DE_LU", latest_ts: "2026-08-16T13:00", latest_price: 62.4, currency: "EUR", unit: "MWH", resolution: "PT60M",
        points_in_window: 72, window_min_price: -8.2, window_max_price: 118.5, window_mean_price: 54.1, negative_price_points: 3 },
      { zone: "FR", latest_ts: "2026-08-16T13:00", latest_price: 58.9, currency: "EUR", unit: "MWH", resolution: "PT60M",
        points_in_window: 72, window_min_price: 12.1, window_max_price: 104.2, window_mean_price: 51.7, negative_price_points: 0 },
    ],
    issues: {},
  },
  // SEC CNS fails-to-deliver, half-month files (2026-08-16, RAW display,
  // secFtd.tsx). Fixture covers 2 top-fail rows so the harness exercises
  // the leaderboard table and the null-price ("." source value) branch.
  "/api/data/ftd": {
    kind: "raw",
    source: "SEC CNS fails-to-deliver, half-month files (public domain) (fixture)",
    attribution: "U.S. Securities and Exchange Commission (CNS fails-to-deliver)",
    time: "2026-08-16T12:00:00.000Z",
    note: "aggregate net fail BALANCES per settlement date (a level, not a daily flow); published on a 2.5-4.5 week lag (fixture).",
    summary: {
      period: "202607b",
      settlement_dates: 11,
      newest_date: "2026-07-31",
      rows: 58328,
      top_fails: [
        { symbol: "JUCY", name: "FIXTURE JUICY CORP", qty: 4823110, price: 3.42 },
        { symbol: "SNDQ", name: "FIXTURE SANDQUEST INC", qty: 2110540, price: null },
      ],
      qty_floor: 100000,
      top_cap: 15,
    },
  },
  // Corporate/LLC aircraft fleet utilization (2026-08-19, RAW display,
  // fleetUtilization.tsx). Fixture covers 2 owners so the harness exercises
  // the multi-week aggregation, the trustee-registered badge, and both
  // resolution-basis branches (callsign-resolved vs FAA registrant).
  "/api/data/fleet-utilization": {
    kind: "derived",
    source: "voltradeai aircraft position archive x FAA entity spine (fixture)",
    attribution: "FAA Aircraft Registry (identity); own archive (positions)",
    as_of: "2026-08-19T12:00:00.000Z",
    owners_total: 2,
    count: 2,
    note: "owners are FAA registrants (trustee/leasing entities hide beneficial owners); airborne hours are LOWER BOUNDS under adaptive archive sampling; weeks without coverage are absent, not zero; single-airframe registrants excluded from this payload (top= caps at 200) (fixture).",
    owners: [
      { owner: "FIXTURE AVIATION LLC", group: "FIXTURE AVIATION LLC", resolution: { callsign: 3, registrant: 0 }, trustee_airframes: 0, registrant_type: "llc", n_airframes: 3,
        weekly: { "2026-08-04": { f: 6, h: 14.2 }, "2026-08-11": { f: 5, h: 11.8 } } },
      { owner: "FIXTURE TRUST CO TRUSTEE", group: "FIXTURE TRUST CO TRUSTEE", resolution: { callsign: 0, registrant: 2 }, trustee_airframes: 2, registrant_type: "corporation", n_airframes: 2,
        weekly: { "2026-08-11": { f: 2, h: 4.5 } } },
    ],
  },
  // EIA-930 hourly electric grid demand (2026-08-19, RAW display,
  // gridDemand.tsx). Fixture covers 3 respondents so the harness exercises
  // descending-by-demand sort, a null day-ahead-forecast branch (strain "—"),
  // and a negative-strain branch alongside a positive one.
  "/api/data/grid-demand": {
    kind: "raw",
    source: "EIA-930 Hourly Electric Grid Monitor (public domain) (fixture)",
    attribution: "EIA-930 Hourly Electric Grid Monitor",
    time: 1786207476120,
    count: 3,
    note: "hourly demand (MWh) for US48 + major balancing authorities, ~1-2h publication lag; latest_forecast_mwh is EIA's day-ahead forecast FOR THE SAME HOUR as latest_mwh (null when not yet published); industrial-activity nowcast signals stay gate-locked until ladder validation (fixture).",
    respondents: [
      { respondent: "US48", latest_period: "2026-08-19T14", latest_mwh: 512340, latest_forecast_mwh: 498210, hours_in_window: 48 },
      { respondent: "ERCO", latest_period: "2026-08-19T14", latest_mwh: 68210, latest_forecast_mwh: 71500, hours_in_window: 48 },
      { respondent: "NW", latest_period: "2026-08-19T13", latest_mwh: 21040, latest_forecast_mwh: null, hours_in_window: 47 },
    ],
  },
  // OCC daily options cleared volume by trade origin (2026-08-19, RAW
  // display, occVolume.tsx). Fixture covers 3 underlyings so the harness
  // exercises a call-skewed row, a put-skewed row, and a zero-customer-
  // volume row (skew "—").
  "/api/data/occ-volume": {
    kind: "raw",
    source: "The Options Clearing Corporation (OCC) daily cleared volume (fixture)",
    attribution: "The Options Clearing Corporation (OCC) daily volume",
    time: 1786207476120,
    report_date: "2026-08-18",
    underlyings: 3,
    count: 3,
    note: "top underlyings by cleared volume with customer/market-maker put-call splits (qty counts each clearing side; totals halved); source keeps only a rolling 2-year window — this archive is the durable copy; origin-split signals stay gate-locked until ladder validation (fixture).",
    top: [
      { underlying: "SPY", cust_put: 40000, cust_call: 120000, mm_put: 210000, mm_call: 190000, total_cleared: 560000 },
      { underlying: "QQQ", cust_put: 95000, cust_call: 21000, mm_put: 88000, mm_call: 71000, total_cleared: 275000 },
      { underlying: "IWM", cust_put: 0, cust_call: 0, mm_put: 15000, mm_call: 12000, total_cleared: 27000 },
    ],
  },
  // US macro regime cluster — FRED, 28 public series (2026-08-08, RAW
  // display). Fixture covers one series per category so the harness
  // exercises every fmtVal unit branch (%, index, claims, thousands,
  // $M/$B, $/bbl) and the category-header grouping.
  "/api/data/macro": {
    kind: "raw",
    source: "FRED API — Source: FRED, Federal Reserve Bank of St. Louis (fixture)",
    attribution: "Source: FRED, Federal Reserve Bank of St. Louis",
    time: 1786207476120,
    note: "Values as currently published by FRED (revisions overwrite here; our archive keeps every vintage as-seen).",
    series: [
      { id: "DGS10", label: "10-Year Treasury", cadence: "daily", unit: "%",
        latest: { d: "2026-08-06", v: 4.69 }, prev: { d: "2026-08-05", v: 4.63 },
        history: [{ d: "2026-08-04", v: 4.61 }, { d: "2026-08-05", v: 4.63 }, { d: "2026-08-06", v: 4.69 }] },
      { id: "STLFSI4", label: "St. Louis Fed Stress Index", cadence: "weekly", unit: "index",
        latest: { d: "2026-07-31", v: -0.5063 }, prev: { d: "2026-07-24", v: -0.481 },
        history: [{ d: "2026-07-17", v: -0.45 }, { d: "2026-07-24", v: -0.481 }, { d: "2026-07-31", v: -0.5063 }] },
      { id: "ICSA", label: "Initial Jobless Claims", cadence: "weekly", unit: "claims",
        latest: { d: "2026-08-01", v: 199000 }, prev: { d: "2026-07-25", v: 203000 },
        history: [{ d: "2026-07-18", v: 208000 }, { d: "2026-07-25", v: 203000 }, { d: "2026-08-01", v: 199000 }] },
      { id: "CPIAUCSL", label: "CPI (All Items)", cadence: "monthly", unit: "index",
        latest: { d: "2026-06-01", v: 332.568 }, prev: { d: "2026-05-01", v: 331.9 },
        history: [{ d: "2026-04-01", v: 331.2 }, { d: "2026-05-01", v: 331.9 }, { d: "2026-06-01", v: 332.568 }] },
      { id: "HOUST", label: "Housing Starts", cadence: "monthly", unit: "thousands",
        latest: { d: "2026-06-01", v: 1427 }, prev: { d: "2026-05-01", v: 1389 },
        history: [{ d: "2026-04-01", v: 1352 }, { d: "2026-05-01", v: 1389 }, { d: "2026-06-01", v: 1427 }] },
      { id: "M2SL", label: "M2 Money Stock", cadence: "monthly", unit: "$B",
        latest: { d: "2026-06-01", v: 23155.2 }, prev: { d: "2026-05-01", v: 23081.4 },
        history: [{ d: "2026-04-01", v: 23002.1 }, { d: "2026-05-01", v: 23081.4 }, { d: "2026-06-01", v: 23155.2 }] },
      { id: "WALCL", label: "Fed Balance Sheet", cadence: "weekly", unit: "$M",
        latest: { d: "2026-08-05", v: 6748567 }, prev: { d: "2026-07-29", v: 6739210 },
        history: [{ d: "2026-07-22", v: 6731500 }, { d: "2026-07-29", v: 6739210 }, { d: "2026-08-05", v: 6748567 }] },
      { id: "DCOILWTICO", label: "WTI Crude Spot", cadence: "daily", unit: "$/bbl",
        latest: { d: "2026-08-03", v: 81.96 }, prev: { d: "2026-07-31", v: 80.4 },
        history: [{ d: "2026-07-29", v: 79.8 }, { d: "2026-07-31", v: 80.4 }, { d: "2026-08-03", v: 81.96 }] },
    ],
  },
  // Institutional 13F-HR holdings (2026-08-08, RAW display). Two filings:
  // one focused (holdings present) and one over the FOCUSED_MAX_HOLDINGS
  // cap (summary-only) so the harness exercises both card states.
  "/api/data/filings13f/history": {
    kind: "raw",
    source: "SEC EDGAR (13F-HR) — accumulated archive (began 2026-07-05) (fixture)",
    days: 45,
    count: 2,
    focused_cap: 250,
    filings: [
      {
        accession: "0001234567-26-000123", filedAt: "2026-08-05T00:00:00.000Z", cik: "0001234567",
        indexUrl: "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000123/",
        managerCik: "0001234567", managerName: "Example Capital Management LLC",
        periodOfReport: "2026-06-30", submissionType: "13F-HR",
        entryTotal: 42, valueTotal: 812500000, otherManagersCount: 0,
        holdingsOmitted: false,
        holdings: [
          { issuer: "EXAMPLE SMALLCAP CORP", titleOfClass: "COM", cusip: "123456789", value: 45000000, shares: 900000, sharesType: "SH", putCall: null, discretion: "SOLE" },
          { issuer: "SAMPLE MICROCAP INC", titleOfClass: "COM", cusip: "987654321", value: 22000000, shares: 500000, sharesType: "SH", putCall: null, discretion: "SOLE" },
        ],
      },
      {
        accession: "0009876543-26-000456", filedAt: "2026-08-04T00:00:00.000Z", cik: "0009876543",
        indexUrl: "https://www.sec.gov/Archives/edgar/data/9876543/000987654326000456/",
        managerCik: "0009876543", managerName: "Mega Institutional Advisors Inc",
        periodOfReport: "2026-06-30", submissionType: "13F-HR",
        entryTotal: 3800, valueTotal: 412000000000, otherManagersCount: 2,
        holdingsOmitted: true, holdings: null,
      },
    ],
  },
  // Data quality dashboard (MAP V2 ROADMAP R6(b), 2026-07-30) — one entry
  // per health bucket + a multi-kind archive so the bar-chart rendering
  // path is exercised at 390px.
  "/api/data/quality-dashboard": {
    generated_at: "2026-07-30T00:00:00.000Z",
    streams_warming_up: false,
    streams: { total: 42, live: 30, recent: 8, stale: 3, no_data: 1 },
    archive: {
      total_bytes: 4831838208,
      total_files: 5120,
      kinds: [
        { kind: "aircraft", files: 2100, bytes: 2147483648 },
        { kind: "vessels", files: 1800, bytes: 1073741824 },
        { kind: "filings", files: 620, bytes: 536870912 },
        { kind: "fires", files: 300, bytes: 268435456 },
        { kind: "trains", files: 210, bytes: 134217728 },
        { kind: "earthquakes", files: 90, bytes: 67108864 },
      ],
    },
    verification: {
      sites: { verified: 16, total: 16, method: "Esri World Imagery crosshair check, scripts/site_verify.py (2026-07-03) (fixture)" },
      ports: { verified: 9, total: 9, method: "subset of the imagery-verified strategic sites (category = port) (fixture)" },
      plants: { verified: 100, total: 9833, method: "top-100-by-MW plants imagery-verified against Esri World Imagery (fixture)" },
    },
  },
  // Signal-strength dashboard (MAP V2 ROADMAP R6(a), 2026-07-30) — one root
  // per status class so the funnel bars, category chips, and badge colors
  // all render at 390px.
  "/api/data/signal-ladder": {
    generated_at: "2026-07-30T00:00:00.000Z",
    compiled: "2026-07-30",
    sources: ["research/experiments.md (fixture)", "research/open_questions.md (fixture)"],
    roots: [
      { id: "fx_raw", name: "Fixture raw overlay", category: "environmental", status: "raw_only", current_gate: 0, last_update_date: "2026-07-08", note: "Display-only overlay, no predictive claim (fixture).", source_ref: "experiments.md:1 (fixture)" },
      { id: "fx_g1p", name: "Fixture gate-1 pending", category: "macro", status: "gate1_pending", current_gate: 0, last_update_date: "2026-07-06", note: "Archiver shipped, gate-1 criteria stated, no run result yet (fixture).", source_ref: "experiments.md:2 (fixture)" },
      { id: "fx_g1pass", name: "Fixture gate-1 pass", category: "macro", status: "gate1_pass", current_gate: 1, last_update_date: "2026-07-05", note: "Prod values exact-matched the published export (fixture).", source_ref: "experiments.md:3 (fixture)" },
      // gate1_pass + detail_route (2026-09-16) — exercises the honest
      // "view live data" label (never "signal") for a DATA-tier root that
      // already has a dedicated RAW /data page but hasn't passed gate 2 yet
      // — see server/signalLadder.ts's detailRouteLabel().
      { id: "fx_g1pass_detail", name: "Fixture gate-1 pass with live RAW-DATA page", category: "macro", status: "gate1_pass", current_gate: 1, last_update_date: "2026-09-16", note: "Gate 1 pass with a dedicated live RAW-DATA page, gate 2 not attempted (fixture).", source_ref: "experiments.md:3b (fixture)", detail_route: "#/data/fred-macro", detail_route_label: "view live data →" },
      { id: "fx_g1fail", name: "Fixture gate-1 fail", category: "commodities", status: "gate1_fail", current_gate: 1, last_update_date: "2026-07-05", note: "Both sensor designs dead at gate 1 (fixture).", source_ref: "experiments.md:4 (fixture)" },
      { id: "fx_g2p", name: "Fixture gate-2 pending", category: "government_spending", status: "gate2_pending", current_gate: 2, last_update_date: "2026-07-26", note: "Gate 1 passed, first gate-2 run inconclusive (fixture).", source_ref: "experiments.md:5 (fixture)" },
      { id: "fx_killed", name: "Fixture killed root", category: "insider_trading", status: "killed", current_gate: 2, last_update_date: "2026-07-22", note: "Gate 2 kill in both directions (fixture).", source_ref: "experiments.md:6 (fixture)" },
      // gate2_pass + detail_route (2026-08-16) — exercises the "view live
      // signal" generic link (signalLadder.tsx), first real use being
      // gnss_integrity_adsb's #/data/gnss-integrity page.
      { id: "fx_g2pass_detail", name: "Fixture gate-2 pass with live detail page", category: "geopolitical_intelligence", status: "gate2_pass", current_gate: 2, last_update_date: "2026-08-16", note: "Gate 2 pass with a dedicated live signal page (fixture).", source_ref: "experiments.md:7 (fixture)", detail_route: "#/data/gnss-integrity", detail_route_label: "view live signal →" },
    ],
    summary: {
      total: 8,
      by_status: { raw_only: 1, gate1_pending: 1, gate1_pass: 2, gate1_fail: 1, gate2_pending: 1, killed: 1, gate2_pass: 1 },
      by_category: { environmental: 1, macro: 3, commodities: 1, government_spending: 1, insider_trading: 1, geopolitical_intelligence: 1 },
      gate_counts: [{ gate: 0, count: 2 }, { gate: 1, count: 3 }, { gate: 2, count: 3 }, { gate: 3, count: 0 }, { gate: 4, count: 0 }, { gate: 5, count: 0 }],
      killed_count: 1,
      raw_only_count: 1,
      furthest_gate_reached: 2,
    },
  },
  // Pipeline-health dashboard (MAP V2 ROADMAP R6(c), 2026-07-31) — a mixed
  // 24h timeline (mostly ok, a few degraded bars) plus non-zero per-check
  // degraded counts so both the uptime tiles and the "what's degrading"
  // bar-chart rendering paths are exercised at 390px.
  "/api/data/pipeline-health-dashboard": {
    generated_at: "2026-07-31T00:00:00.000Z",
    warming_up: false,
    // Scheduled jobs ([PRODUCT] 2026-11-21, client/src/lib/jobsView.ts): one row per state the section can draw:
    // a failed weekly test with its cause, a run open past its limit, on time, running, not due, switched off,
    // and the longest label in the registry so the wrap is exercised at 390px.
    jobs: (() => {
      const now = Date.now();
      const at = (minAgo) => new Date(now - minAgo * 60000).toISOString();
      const j = (id, label, over) => ({ id, label, status: "ok", schedule: "every 15m (04:00-20:00 ET), grace 20m", last_start_at: at(12), last_finish_at: at(10), last_status: "ok", running: false, disabled: false, recent_errors: 0, error_class: null, ...over });
      return { evaluated_at: at(0), verdict: "degraded", counts: { ok: 3, late: 1, missing: 0, errored: 1, not_due: 2 }, jobs: [
        j("tier2_scan", "Tier 2 scan (adaptive 1-15 min)", {}),
        j("tier3_strategic", "Tier 3 strategic (hourly: retrain check, manipulation, diagnostics, COT, Form 4, port dwell)", { schedule: "every 1h (always), grace 15m", last_start_at: at(40), last_finish_at: at(39) }),
        j("options_chain_archive", "Options chain archive (trading days, after 16:15 ET)", { status: "not_due", schedule: "daily (trading days) due 16:15 ET, grace 90m", last_start_at: at(1400), last_finish_at: at(1390) }),
        j("earnings_qoq_weekly", "8-K earnings-language QoQ delta gate-2 test (weekly, Sun 18:00-19:00 ET)", { status: "errored", schedule: "every 168h (always), grace 26h", last_start_at: at(7200), last_finish_at: at(7199), last_status: "error", recent_errors: 1, error_class: "nonzero_exit" }),
        j("price_panel_build", "Price panel build (2016+ daily bars incl. delisted, 20:15-04:00 ET)", { status: "late", schedule: "every 24h (20:15-04:00 ET), grace 2h", last_start_at: at(600), last_finish_at: null, last_status: "running", running: true }),
        j("fires_completeness_daily", "Fire-detection archive completeness pass (NASA FIRMS dated request, from 06:00 UTC)", { schedule: "daily due 06:00 UTC, grace 120m", last_start_at: at(3), last_finish_at: null, last_status: "running", running: true }),
        j("github_gate2_weekly", "GitHub org engineering-momentum gate-2 test (weekly, Sun 19:00-20:00 ET)", { status: "not_due", disabled: true, schedule: "every 168h (always), grace 26h", last_start_at: null, last_finish_at: null, last_status: null }),
      ] };
    })(),
    // SEC filing pollers ([PRODUCT] 2026-11-24, client/src/lib/edgarPollView.ts): one row per state the section
    // draws: a poll that read nothing (with its first failure), one that read in part, one that read its whole page
    // with filings skipped.
    edgar: (() => {
      const now = Date.now();
      const at = (minAgo) => new Date(now - minAgo * 60000).toISOString();
      return { read_at: at(0), booted_at: at(570), poll_interval_min: 15, streams: [
        { stream: "filings", polls_ok: 38, polls_failed: 0, consecutive_failures: 0, last_archive_write_at: at(18),
          documents: { at: at(4), candidates: 25, returned: 22, skipped: 0, failed: 3, first_error: "submission text -> 404", via_submission: 5 } },
        { stream: "filings13f", polls_ok: 36, polls_failed: 2, consecutive_failures: 2, last_archive_write_at: at(960),
          documents: { at: at(4), candidates: 40, returned: 0, skipped: 0, failed: 40, first_error: "index.json -> 404" } },
        { stream: "earnings8k", polls_ok: 38, polls_failed: 0, consecutive_failures: 0, last_archive_write_at: at(18),
          documents: { at: at(4), candidates: 15, returned: 9, skipped: 6, failed: 0, first_error: null } },
      ] };
    })(),
    window_24h: {
      window_hours: 24, sample_count: 288, uptime_pct: 97.6,
      current: {
        t: 1783200000, status: "ok", database_ok: true, alpaca_ok: true, python_ok: true,
        scanner_ok: true, scanner_consecutive_failures: 0, licensing_ok: true,
        bot_status: "active", bot_liveness_dark: false,
      },
      degraded_counts: { database: 0, alpaca: 2, python: 0, scanner: 5, licensing: 0, bot_liveness_dark: 0 },
      timeline: Array.from({ length: 48 }, (_, i) => ({ t: 1783200000 - (47 - i) * 1800, status: i === 10 || i === 11 || i === 30 ? "degraded" : "ok" })),
    },
    window_7d: {
      window_hours: 168, sample_count: 2016, uptime_pct: 99.1,
      current: {
        t: 1783200000, status: "ok", database_ok: true, alpaca_ok: true, python_ok: true,
        scanner_ok: true, scanner_consecutive_failures: 0, licensing_ok: true,
        bot_status: "active", bot_liveness_dark: false,
      },
      degraded_counts: { database: 0, alpaca: 3, python: 0, scanner: 15, licensing: 0, bot_liveness_dark: 1 },
      timeline: Array.from({ length: 100 }, (_, i) => ({ t: 1783200000 - (99 - i) * 6048, status: i % 33 === 0 ? "degraded" : "ok" })),
    },
  },
  // Streams inventory (Phase 4) — one row per health state so the card
  // battery exercises every chip/peek/no-data rendering path at 390px.
  // FAA live feeds on the Streams inventory ([PRODUCT] 2026-11-20, client/src/lib/swimFeeds.ts): the shape production
  // answers with — flight plans receiving with drops, traffic management connected but silent for 20 minutes, notices
  // disconnected with an error, terminal data deliberately not read, terminal weather not set up (3 settings)
  "/api/data/swim/products": (() => {
    const now = Date.now();
    const f = (over) => ({ env: { configured: true, missing: [] }, consumer: "x", connected: true, messagesReceived: 0, messagesProcessed: 0, processErrors: 0, droppedOverflow: 0, reconnects: 0, stallRestarts: 0, lastMessageAt: null, lastError: null, hourly: [], shapes: null, ...over });
    return { at: now, note: "fixture", products: {
      SFDPS: f({ consumer: "flight plans (swimSfdps.ts)", messagesReceived: 18113, messagesProcessed: 17506, droppedOverflow: 607, lastMessageAt: now - 2000 }),
      TFMS: f({ consumer: "capture (shape profile + samples)", messagesReceived: 21624, messagesProcessed: 21624, lastMessageAt: now - 20 * 60_000 }),
      STDDS: f({ consumer: "configured-only", connected: false }),
      NOTAM: f({ consumer: "capture (shape profile + samples)", connected: false, messagesReceived: 32, messagesProcessed: 30, processErrors: 2, reconnects: 1, lastMessageAt: now - 2 * 3600_000, lastError: "broker closed the session" }),
      ITWS: f({ env: { configured: false, missing: ["SWIM_ITWS_URL", "SWIM_ITWS_VPN", "SWIM_ITWS_QUEUE"] }, consumer: "configured-only", connected: false }),
    } };
  })(),
  "/api/data/streams": {
    time: 1, count: 4,
    streams: [
      {
        stream: "occvolume", source: "OCC volume-query (marketdata.theocc.com) — keyless CSV (fixture)", attribution: "The Options Clearing Corporation (OCC) daily volume",
        license: "OCC informational use", started: "2026-07-06", cadence: "daily source, published overnight ET; 4h poll",
        hypothesis: "customer-vs-market-maker put/call by ticker (gate-locked)", storage: "one gzipped JSONL per report date", written_by: "server/occVolume.ts",
        files: 3, bytes: 3145728, newest_file: "2026-07-02.jsonl.gz", age_hours: 6.2, records_newest_file: 153181,
        latest_peek: '{"date":"2026-07-02","underlying":"SPY","symbol":"SPY","actype":"M","porc":"P","exchange":"CBOE","qty":120}', peek_note: null,
        health: "live", health_note: "newest file 6.2h old",
      },
      {
        stream: "cropconditions", source: "USDA NASS QuickStats weekly crop condition ratings (fixture)", attribution: "USDA National Agricultural Statistics Service",
        license: "US government data", started: "2026-07-06", cadence: "weekly, Mondays 4pm ET",
        hypothesis: "condition-change vs grain price moves (gate 1 pending)", storage: "one JSONL per observation week", written_by: "server/cropConditions.ts",
        files: 1, bytes: 4096, newest_file: "2026-07-05.jsonl", age_hours: 96.0, records_newest_file: 10,
        latest_peek: '{"week":"2026-07-05","commodity":"CORN","item":"GOOD","value":41}', peek_note: null,
        health: "recent", health_note: "newest file 96.0h old (within cadence ~240h)",
      },
      {
        stream: "railep724", source: "STB EP724 weekly rail performance (fixture)", attribution: "Surface Transportation Board",
        license: "US government data", started: "2026-07-05", cadence: "weekly, session-run pipeline",
        hypothesis: "carload trends lead reported volumes", storage: "weekly JSONL artifacts", written_by: "scripts/stb_rail.py (session-run)",
        files: 2, bytes: 51200, newest_file: "2026-06-20.jsonl", age_hours: 384.0, records_newest_file: 88,
        latest_peek: null, peek_note: "peek skipped: newest file 9.1MB > 8MB cap",
        health: "stale", health_note: "newest file 384.0h old — exceeds cadence-derived threshold ~240h",
      },
      {
        stream: "griddemand", source: "EIA-930 hourly demand by balancing authority (fixture)", attribution: "U.S. Energy Information Administration",
        license: "US government data", started: "2026-07-06", cadence: "hourly source, 2h poll",
        hypothesis: "regional demand anomalies vs utility/industrial names", storage: "one JSONL per observation day", written_by: "server/gridDemand.ts",
        files: 0, bytes: 0, newest_file: null, age_hours: null, records_newest_file: null,
        latest_peek: null, peek_note: "no files",
        health: "no-data", health_note: "no archive files yet — key-gated, warming up, or first write pending",
      },
    ],
  },
  "/api/data/earnings-language": {
    kind: "raw", source: "SEC EDGAR (8-K Item 2.02 / Exhibit 99)", time: 1, count: 1,
    filings: [
      {
        accession: "0001-26-9", cik: "9", companyName: "FIXTURE INDUSTRIES INC",
        filedAt: "2026-07-05", itemCodes: ["2.02"],
        indexUrl: "https://www.sec.gov/x/9/", exhibitUrl: "https://www.sec.gov/x/9/ex99.htm",
        text: "Fixture Industries today reported second-quarter results ahead of guidance.", textLength: 76, truncated: false,
      },
    ],
  },
  "/api/data/earnings-language/history": {
    kind: "raw", source: "SEC EDGAR (8-K Item 2.02 / Exhibit 99) — accumulated archive", days: 30, count: 2,
    filings: [
      {
        accession: "0001-26-9", cik: "9", companyName: "FIXTURE INDUSTRIES INC",
        filedAt: "2026-07-05", itemCodes: ["2.02"],
        indexUrl: "https://www.sec.gov/x/9/", exhibitUrl: "https://www.sec.gov/x/9/ex99.htm",
        text: "Fixture Industries today reported second-quarter results ahead of guidance.", textLength: 76, truncated: false,
      },
      {
        accession: "0001-26-10", cik: "10", companyName: "SAMPLE HOLDINGS CORP",
        filedAt: "2026-07-04", itemCodes: ["2.02", "9.01"],
        indexUrl: "https://www.sec.gov/x/10/", exhibitUrl: "https://www.sec.gov/x/10/ex99.htm",
        text: "Sample Holdings Corp announced full-year guidance and a share buyback program.", textLength: 79, truncated: false,
      },
    ],
  },
  "/api/data/short-volume": {
    kind: "raw", source: "FINRA Reg SHO daily short sale volume (CNMS consolidated file)", time: 1,
    summary: {
      date: "2026-07-02", symbols: 12240, agg_short_ratio: 0.4633,
      top_ratio: [
        { symbol: "JUCY", short_ratio: 0.9964, total_vol: 911633, market: "Q" },
        { symbol: "AGGH", short_ratio: 0.9909, total_vol: 1968769, market: "Q,N" },
      ],
      floor_total_vol: 500000, top_cap: 30,
    },
  },
  "/api/data/short-volume/history": {
    kind: "raw", source: "FINRA Reg SHO daily short sale volume (CNMS consolidated file)", days: 60,
    today: {
      date: "2026-07-02", symbols: 12240, agg_short_ratio: 0.4633,
      top_ratio: [{ symbol: "JUCY", short_ratio: 0.9964, total_vol: 911633, market: "Q" }],
      floor_total_vol: 500000, top_cap: 30,
    },
    trend: [
      { date: "2026-07-01", symbols: 12200, agg_short_ratio: 0.4610 },
      { date: "2026-07-02", symbols: 12240, agg_short_ratio: 0.4633 },
    ],
  },
  // [REPAIR 2026-07-25] the 9 endpoint fixtures below back the 9 layer
  // descriptors added above — real response shapes copied from server/
  // routes.ts's actual handlers, one point each (enough to exercise the
  // symbol layer + legend-parity + detail card, not a rendering-scale test).
  "/api/data/pfas": {
    kind: "raw", predictive: false, source: "U.S. EPA UCMR 5 (fixture)",
    note: "fixture", geocode_source: "EPA service-area centroid", built_at: "2026-07-01T00:00:00Z",
    totals: { systems: 1 }, health: { suspect: 0, freshness: "public domain" },
    systems: [
      { pwsid: "FX0000001", name: "Fixture Water System", lat: 39.1, lon: -94.6, n_analytes_detected: 2,
        population_served: 12000, detections: [{ contaminant: "PFOA", max_value: 12.4, units: "ng/L", n_detections: 3, last_detected: "2024-11-01" }] },
    ],
  },
  "/api/data/cancer-rates": {
    kind: "raw", predictive: false, source: "NCI State Cancer Profiles (fixture)",
    caveat: "COUNTY-LEVEL AGGREGATE STATISTIC ONLY (fixture)", built_at: "2026-08-11T00:00:00Z",
    gate1: { passed: true }, county_count: 2, health: { suspect: 0, freshness: "public domain" },
    geojson: {
      type: "FeatureCollection",
      features: [
        { type: "Feature", id: "20177", properties: { fips: "20177", county: "Fixture County", state: "Kansas", has_data: true, incidence_rate: 450.0, incidence_trend: "stable", mortality_rate: 145.0, mortality_trend: "falling" },
          geometry: { type: "Polygon", coordinates: [[[-95.8, 39.0], [-95.6, 39.0], [-95.6, 39.2], [-95.8, 39.2], [-95.8, 39.0]]] } },
        { type: "Feature", id: "20179", properties: { fips: "20179", county: "No-Data Fixture County", state: "Kansas", has_data: false, incidence_rate: null, incidence_trend: null, mortality_rate: null, mortality_trend: null },
          geometry: { type: "Polygon", coordinates: [[[-95.6, 39.0], [-95.4, 39.0], [-95.4, 39.2], [-95.6, 39.2], [-95.6, 39.0]]] } },
      ],
    },
  },
  "/api/data/radiation": {
    kind: "raw", predictive: false, source: "BfS · Health Canada · STUK/FMI · EPA RadNet (fixture)",
    time: 1, networks: { "bfs-de": 1, "radnet-us": 1 },
    health: { source: "fixture" },
    stations: [
      { name: "Fixture DE Station", network: "bfs-de", lat: 50.1, lon: 8.7, value: 0.12, unit: "uSv/h", time: "2026-07-25T00:00:00Z", approx: false },
      { name: "Fixture City, KS", network: "radnet-us", lat: 39.0, lon: -95.7, value: 14.2, unit: "cpm", time: "2026-07-25T00:00:00Z", approx: true, rkm: 6 },
    ],
  },
  "/api/data/nukefacilities": {
    kind: "raw", predictive: false, source: "Wikidata (CC0 1.0), curated (fixture)", count: 1,
    facilities: [{ n: "Fixture Enrichment Plant", cat: "Enrichment plant", country: "United States", qid: "Q0", lat: 43.5, lon: -112.0 }],
  },
  "/api/data/nucleartests": {
    kind: "raw", predictive: false, source: "SIPRI / Johnston archive (fixture)", count: 1, quarantined: 0,
    tests: [{ n: "FIXTURE SHOT", c: "USA", y: 1962, d: "1962-07-09", kt: 1.4, t: "AIRDROP", p: "WE", r: "Fixture Test Site", lat: 37.2, lon: -116.0 }],
  },
  "/api/data/nukeaccidents": {
    kind: "raw", predictive: false, source: "Wikidata (CC0 1.0), curated (fixture)", count: 1,
    events: [{ n: "Fixture Reactor Incident", d: "1979-03-28", loc: "Fixture, PA", ines: 5, qid: "Q0", lat: 40.2, lon: -76.7 }],
  },
  "/api/data/methane-plumes": {
    kind: "raw", predictive: false, source: "Global Energy Monitor GMET (fixture)",
    count: 1, matchedCount: 1, ambiguousCount: 0,
    plumes: [{
      name: "Fixture plume detection", infrastructureType: "Oil and gas", country: "United States",
      observedAt: "2026-06-01T00:00:00Z", provider: "GHGSat", instrument: "GHGSat-C", emissionsKgHr: 340.5,
      lat: 31.9, lon: -102.1,
      nearestAsset: { kind: "oil_gas_extraction", id: "A1", name: "Fixture Field", distanceKm: 0.4, operator: "Fixture Operator LLC" },
    }],
  },
  // [REPAIR, added this session] matching client/src/lib/mapIcons.ts's
  // LngShipyard consumer (datamap.tsx ~line 11104): d.shipyards must be a
  // real array (an unmocked "{}" fallback makes `!Array.isArray(d.shipyards)`
  // throw and the layer render "error" forever) — four shipyards across
  // three named country tiers + "other" so the legend-parity battery
  // exercises every LNG_SHIPYARD_COUNTRY_LABEL entry, not just one.
  // GEM point-layer route fixtures (ported 2026-10-21 from stale PR #1174,
  // whose `layers` rows had already landed via the 2026-09-25 repair above
  // but whose route fixtures never did): two rows per layer spanning
  // distinct symbol/colour categories so toggle-consistency and
  // legend-parity exercise more than one branch, same discipline as the
  // lng_shipyards / steel_raw_materials fixtures below. The iron-steel rows
  // carry the `units` array main's route returns today (furnace-level
  // detail keyed from steel_units.json; [] when none joined).
  "/api/data/coal-terminals": {
    kind: "raw", predictive: false, source: "Global Energy Monitor — Global Coal Terminals Tracker (fixture)",
    release: "fixture", count: 2,
    terminals: [
      { id: "FX1", terminalId: "FXT1", name: "Fixture Export Terminal", parentPort: "Fixture Port", status: "Operating",
        typeClass: "exports", typeRaw: "Exports", productType: "Thermal coal", capacityMt: 12.5,
        owner: "Fixture Coal Co", country: "Australia", region: "Oceania", startYear: "1998", retiredYear: null,
        locationAccuracy: "Exact", wiki: null, lat: -23.5, lon: 151.9 },
      { id: "FX2", terminalId: "FXT2", name: "Fixture Import Terminal", parentPort: null, status: "Proposed",
        typeClass: "imports", typeRaw: "Imports", productType: "Thermal coal", capacityMt: null,
        owner: "Fixture Import Co", country: "India", region: "Asia", startYear: null, retiredYear: null,
        locationAccuracy: "Approximate", wiki: null, lat: 19.0, lon: 72.8 },
    ],
  },
  "/api/data/iron-ore-mines": {
    kind: "raw", predictive: false, source: "Global Energy Monitor — Global Iron Ore Mines Tracker (fixture)",
    release: "fixture", count: 2,
    mines: [
      { id: "FXM1", name: "Fixture Ore Mine", status: "operating", statusRaw: "Operating", country: "Brazil",
        region: "South America", coordinateAccuracy: "Exact", production2024Kt: 5000, designCapacityKt: 6000,
        totalReservesKt: 200000, totalResourceKt: 500000, startDate: "1985", stopDate: null,
        owner: "Fixture Mining Co", parent: "Fixture Holdings", wiki: null, lat: -20.0, lon: -43.5 },
      { id: "FXM2", name: "Fixture Proposed Mine", status: "proposed", statusRaw: "Proposed", country: "Guinea",
        region: "Africa", coordinateAccuracy: "Approximate", production2024Kt: null, designCapacityKt: 3000,
        totalReservesKt: 800000, totalResourceKt: null, startDate: null, stopDate: null,
        owner: "Fixture Exploration Co", parent: null, wiki: null, lat: 10.0, lon: -12.5 },
    ],
  },
  "/api/data/iron-steel-plants": {
    kind: "raw", predictive: false, source: "Global Energy Monitor — Global Iron and Steel Tracker (fixture)",
    release: "fixture", count: 2,
    plants: [
      { id: "FXS1", name: "Fixture Integrated Steel Plant", technology: "bf_bof", technologyRaw: "Blast Furnace/Basic Oxygen Furnace",
        categorySteelProduct: "Crude", country: "China", region: "Asia", municipality: null, subnationalUnit: null, coordinateAccuracy: "Exact", workforceSize: 8000,
        startDate: "1970", retiredDate: null, idledDate: null, owner: "Fixture Steel Group", parent: null,
        soeStatus: "State-owned", wiki: null, lat: 39.9, lon: 116.4,
        units: [
          { plantId: "FXS1", unitId: "FXU1", name: "BOF 1", furnaceType: "bof", status: "operating", capacityTtpa: 2400, startDate: "1970", retiredDate: null, manufacturer: null },
          { plantId: "FXS1", unitId: "FXU2", name: "Open hearth 1", furnaceType: "open_hearth", status: "retired", capacityTtpa: 600, startDate: "1970", retiredDate: "1995", manufacturer: null },
        ] },
      { id: "FXS2", name: "Fixture Mini-mill", technology: "eaf", technologyRaw: "Electric Arc Furnace",
        categorySteelProduct: "Finished rolled", country: "United States", region: "North America", municipality: null, subnationalUnit: null, coordinateAccuracy: "Exact",
        workforceSize: 450, startDate: "2005", retiredDate: null, idledDate: null, owner: "Fixture Mini-Mill LLC",
        parent: null, soeStatus: null, wiki: null, lat: 33.7, lon: -84.4,
        units: [{ plantId: "FXS2", unitId: "FXU3", name: "EAF 1", furnaceType: "eaf", status: "operating", capacityTtpa: 900, startDate: "2005", retiredDate: null, manufacturer: "Fixture Furnaces" }] },
    ],
  },
  "/api/data/chemicals": {
    kind: "raw", predictive: false, source: "Global Energy Monitor — Global Chemicals Inventory (fixture)",
    release: "fixture", count: 2,
    plants: [
      { id: "FXC1", name: "Fixture Ammonia Plant", feedstockFamily: "natural_gas", feedstockRaw: "Natural Gas",
        primaryProducts: "Ammonia", secondaryProducts: "Urea", country: "United States", region: "North America",
        coordinateAccuracy: "Exact", owner: "Fixture Chemical Co", wiki: null, lat: 29.7, lon: -95.4 },
      { id: "FXC2", name: "Fixture Refinery Petrochemical Complex", feedstockFamily: "petroleum", feedstockRaw: "Petroleum Liquid",
        primaryProducts: "Ethylene", secondaryProducts: "Propylene", country: "Germany", region: "Europe",
        coordinateAccuracy: "Exact", owner: "Fixture Petro AG", wiki: null, lat: 51.5, lon: 6.5 },
    ],
  },
  "/api/data/lng-shipyards": {
    kind: "raw", predictive: false,
    source: "Global Energy Monitor — Global LNG Carrier Tracker (by shipyard) (fixture)",
    attribution: "Global Energy Monitor (fixture)", license: "CC BY 4.0", release: "2026-Q2 (fixture)",
    note: "LNG carrier SHIPBUILDING YARDS, not vessel positions (fixture).",
    count: 4, totalCarriers: 47,
    shipyards: [
      { id: "fixture-korea", shipbuilder: "Fixture Korea Heavy Industries", country: "South Korea", lat: 35.1, lon: 129.0, coordinateAccuracy: "city", carrierCount: 22, activeCount: 18, onOrderCount: 4, proposedCount: 0, otherCount: 0, totalCapacityCbm: 3960000, knownCapacityCount: 22 },
      { id: "fixture-china", shipbuilder: "Fixture China Shipbuilding Yard", country: "China", lat: 31.2, lon: 121.5, coordinateAccuracy: "city", carrierCount: 12, activeCount: 9, onOrderCount: 3, proposedCount: 0, otherCount: 0, totalCapacityCbm: 2040000, knownCapacityCount: 12 },
      { id: "fixture-japan", shipbuilder: "Fixture Japan Marine United", country: "Japan", lat: 34.4, lon: 132.5, coordinateAccuracy: "city", carrierCount: 8, activeCount: 8, onOrderCount: 0, proposedCount: 0, otherCount: 0, totalCapacityCbm: 1360000, knownCapacityCount: 8 },
      { id: "fixture-other", shipbuilder: "Fixture Nordic Yards", country: "Finland", lat: 60.2, lon: 25.0, coordinateAccuracy: "city", carrierCount: 5, activeCount: 3, onOrderCount: 1, proposedCount: 1, otherCount: 0, totalCapacityCbm: 850000, knownCapacityCount: 5 },
    ],
  },
  // [REPAIR, added this session] matching client/src/lib/mapIcons.ts's
  // consumer (datamap.tsx ~line 11192): `!d.geojson?.features?.length`
  // throws on an unmocked "{}" fallback, so the layer never rendered under
  // this harness at all. Three features exercise the three visually
  // distinct fill states the choropleth's own paint expression switches on
  // (server/gemSteelRawMaterials.ts's joinCountryChoropleth): has_data:false
  // (no GEM record), a real reported 0 (has_data:true, iron_ore_mined_ttpa:0),
  // and a positive value inside the ramp's top stop.
  "/api/data/steel-raw-materials": {
    kind: "raw", predictive: false,
    source: "Global Energy Monitor — Production-Consumption of Met Coal & Iron Ore by Steel Industry (fixture)",
    attribution: "Global Energy Monitor (fixture)", license: "CC BY 4.0", release: "2026-Q2 (fixture)",
    note: "National met-coal/iron-ore balance sheet, one row per country (fixture).",
    count: 2, matched_countries: 2, unmatched_countries: 1,
    balances: [
      { country: "Fixture Ore Nation", metCoalMinedTtpa: 1200, ironOreMinedTtpa: 250000, metCoalConsumedPigIronTtpa: 900, ironOreConsumedPigIronTtpa: 180000, ironOreConsumedDriTtpa: 20000, ironOreConsumedTotalTtpa: 200000, pigIronProducedTtpa: 110000, driProducedTtpa: 8000 },
      { country: "Fixture Zero Nation", metCoalMinedTtpa: 0, ironOreMinedTtpa: 0, metCoalConsumedPigIronTtpa: 0, ironOreConsumedPigIronTtpa: 0, ironOreConsumedDriTtpa: 0, ironOreConsumedTotalTtpa: 0, pigIronProducedTtpa: 0, driProducedTtpa: 0 },
    ],
    geojson: {
      type: "FeatureCollection",
      features: [
        { type: "Feature", properties: { name: "Fixture Ore Nation", iso3: "FX1", has_data: true, met_coal_mined_ttpa: 1200, iron_ore_mined_ttpa: 250000, met_coal_consumed_pig_iron_ttpa: 900, iron_ore_consumed_pig_iron_ttpa: 180000, iron_ore_consumed_dri_ttpa: 20000, iron_ore_consumed_total_ttpa: 200000, pig_iron_produced_ttpa: 110000, dri_produced_ttpa: 8000 },
          geometry: { type: "Polygon", coordinates: [[[10.0, 45.0], [15.0, 45.0], [15.0, 50.0], [10.0, 50.0], [10.0, 45.0]]] } },
        { type: "Feature", properties: { name: "Fixture Zero Nation", iso3: "FX2", has_data: true, met_coal_mined_ttpa: 0, iron_ore_mined_ttpa: 0, met_coal_consumed_pig_iron_ttpa: 0, iron_ore_consumed_pig_iron_ttpa: 0, iron_ore_consumed_dri_ttpa: 0, iron_ore_consumed_total_ttpa: 0, pig_iron_produced_ttpa: 0, dri_produced_ttpa: 0 },
          geometry: { type: "Polygon", coordinates: [[[-70.0, -10.0], [-65.0, -10.0], [-65.0, -5.0], [-70.0, -5.0], [-70.0, -10.0]]] } },
        { type: "Feature", properties: { name: "Fixture No-Record Nation", iso3: "FX3", has_data: false, met_coal_mined_ttpa: null, iron_ore_mined_ttpa: null, met_coal_consumed_pig_iron_ttpa: null, iron_ore_consumed_pig_iron_ttpa: null, iron_ore_consumed_dri_ttpa: null, iron_ore_consumed_total_ttpa: null, pig_iron_produced_ttpa: null, dri_produced_ttpa: null },
          geometry: { type: "Polygon", coordinates: [[[100.0, 20.0], [110.0, 20.0], [110.0, 28.0], [100.0, 28.0], [100.0, 20.0]]] } },
      ],
    },
  },
  "/api/data/coal-mine-features": {
    kind: "raw", predictive: false, source: "Global Energy Monitor (fixture)", count: 1, release: "fixture",
    features: [{
      mineName: "Fixture Mine", category: "ventilation", coalGrade: "thermal", country: "United States",
      owners: "Fixture Coal Co", mineId: "FX1", dataSourceDate: "2025-01-01",
      geometry: { type: "Point", coordinates: [-81.5, 37.8] },
    }],
  },
  "/api/data/airport-status": {
    kind: "raw", source: "FAA National Airspace System Status (fixture)", time: 1, update_time: "2026-07-25T00:00:00Z", count: 1,
    events: [{ airport: "ATL", type: "Ground Delay Program", reason: "Volume", avg: "31 minutes", max: "1 hour 5 minutes", min: "16 minutes", trend: "Increasing", direction: "Arrival", end_time: "18:00 UTC", reopen: null, update_time: "2026-07-25T00:00:00Z" }],
  },
  "/api/data/swim/notam/tfrs": (() => {
    const now = Date.now(), H = 3600_000;
    const circle = (id, lat, lon, text, qcode, startIn) => ({
      id, series: null, number: null, year: "2026", type: "N", classification: qcode ? "INTL" : "FDC", location: id.split(":")[0], qcode,
      issued: now - H, start: now + startIn * H, end: now + 8 * H, minFl: 0, maxFl: 180, center: [lat, lon], radiusNm: 30, text, scenario: null,
      geometry: { kind: "circle", center: [lat, lon], radiusNm: 30, approximate: true },
    });
    return {
      at: now, stats: { notams: 400, tfrs: 9, byClassification: {}, parseErrors: 0 },
      kept: { saved_every_s: 60, last_saved_at: now - 30_000, last_save: "ok", restored: null },
      feed: { connected: true, last_message_at: now - 20_000 },
      source: "FAA SWIM FNS NOTAM (fixture)",
      tfrs: [
        circle("KZDC:FX1", 38.9, -77.0, "TEMPORARY FLIGHT RESTRICTIONS PURSUANT TO 14 CFR SECTION 91.141", null, -1),
        circle("KZLA:FX2", 34.2, -118.2, "TEMPORARY FLIGHT RESTRICTIONS PURSUANT TO 14 CFR SECTION 91.137(A)(2) WILDFIRE", null, -1),
        circle("KZJX:FX3", 28.5, -80.6, "PURSUANT TO 14 CFR SECTION 91.143 SPACE OPERATIONS", null, 2),
        circle("KZAU:FX4", 41.9, -87.6, "PURSUANT TO 14 CFR SECTION 91.145 AIR SHOW", null, -1),
        circle("LDZL:FX5", 45.8, 15.95, "TEMPORARY RESERVED AREA ACTIVATED", "QRTCA", -1),
        { ...circle("KZDV:FX6", 39.5, -105.0, "TEMPORARY FLIGHT RESTRICTIONS IN EFFECT", null, -1),
          geometry: { kind: "polygon", rings: [[[39.2, -105.4], [39.2, -104.6], [39.8, -104.6], [39.8, -105.4], [39.2, -105.4]]] } },
      ],
    };
  })(),
  "/api/data/border-waits": {
    kind: "raw", source: "CBP Border Wait Times (fixture)", time: 1, count: 1,
    waits: [{ port_number: "010401", border: "Canada", lane: "POV", status: "Open", delay_min: 15, lanes_open: 2, update_time: "2026-07-25T00:00:00Z" }],
  },
  // Methane hotspots (gate-2(b), 2026-07-20) — descriptive stat, not a
  // signal (predictive:false; note states the honest gate-2(c)/(d) gap).
  "/api/data/methane-plumes/asset-stats": {
    kind: "raw", predictive: false,
    source: "Global Energy Monitor — Methane Emitters Tracker (GMET) × Oil & Gas Extraction Tracker / Global Coal Mine Tracker, CC BY 4.0",
    note: "Repeat satellite methane-plume detections grouped by nearest catalogued GEM asset (within 2km, ambiguous matches excluded). "
      + "A count/rate of repeat detections is NOT an emissions signal on its own — no base rate or disclosed-intensity comparison exists yet (fixture).",
    count: 2,
    assetStats: [
      {
        assetId: "A1", kind: "coal_mine", name: "Sample Mine", operator: null, owner: "Sample Coal Co", parent: "Sample Holdings",
        detectionCount: 5, firstObservedAt: "2025-01-03T00:00:00Z", lastObservedAt: "2026-06-30T00:00:00Z",
        spanDays: 543, ratePerYear: 3.36, meanEmissionsKgHr: 812.4,
      },
      {
        assetId: "A2", kind: "oil_gas_extraction", name: "Sample Field", operator: "Sample Operator LLC", owner: null, parent: null,
        detectionCount: 1, firstObservedAt: "2026-04-11T00:00:00Z", lastObservedAt: "2026-04-11T00:00:00Z",
        spanDays: null, ratePerYear: null, meanEmissionsKgHr: 214.0,
      },
    ],
  },
  "/api/data/attention": {
    kind: "raw", source: "Wikimedia pageviews API (en.wikipedia, all-access, agent=user)", time: 1,
    date: "2026-07-02", seed_size: 23, count: 2,
    tickers: [
      { ticker: "NVDA", article: "Nvidia", views: 10883 },
      { ticker: "GME", article: "GameStop", views: 4211 },
    ],
  },
  "/api/data/attention/history": {
    kind: "raw", source: "Wikimedia pageviews API (en.wikipedia, all-access, agent=user)", days: 60,
    today: {
      date: "2026-07-02", seed_size: 23, count: 2,
      tickers: [
        { ticker: "NVDA", article: "Nvidia", views: 10883 },
        { ticker: "GME", article: "GameStop", views: 4211 },
      ],
    },
    trend: [
      { date: "2026-07-01", tickers: 23, total_views: 98210 },
      { date: "2026-07-02", tickers: 23, total_views: 101422 },
    ],
  },
  // CFTC Traders in Financial Futures, futures-only (2026-08-16, RAW
  // display, tff.tsx). Fixture covers 2 markets so the harness exercises
  // the ranked table's lev-money/dealer-net columns.
  "/api/data/tff": {
    kind: "raw", source: "CFTC Traders in Financial Futures, futures-only (public domain) (fixture)",
    attribution: "CFTC Traders in Financial Futures", time: "2026-08-16T12:00:00.000Z",
    report_date: "2026-08-11", count: 2,
    note: "weekly financial-futures positioning by trader category (fixture)",
    markets: [
      { report_date: "2026-08-11", market: "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", code: "13874A", commodity: "STOCK INDICES",
        open_interest: 2450000, dealer_long: 210000, dealer_short: 340000, asset_mgr_long: 620000, asset_mgr_short: 410000,
        lev_money_long: 380000, lev_money_short: 710000, other_rept_long: 150000, other_rept_short: 120000 },
      { report_date: "2026-08-11", market: "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE", code: "043602", commodity: "TREASURY",
        open_interest: 3980000, dealer_long: 540000, dealer_short: 610000, asset_mgr_long: 890000, asset_mgr_short: 720000,
        lev_money_long: 1150000, lev_money_short: 940000, other_rept_long: 260000, other_rept_short: 210000 },
    ],
  },
  // US Treasury auctions (2026-08-17, RAW display, treasuryAuctions.tsx).
  // Fixture covers a Bill (discount rate) and a Note (yield) so the harness
  // exercises both rate-kind branches of the table.
  "/api/data/treasury-auctions": {
    kind: "raw", source: "TreasuryDirect TA_WS, securities/auctioned (public domain) (fixture)",
    attribution: "U.S. Department of the Treasury (TreasuryDirect)", time: "2026-08-17T12:00:00.000Z",
    count: 2, note: "results-complete auctions from the last 30 days (fixture)",
    auctions: [
      { cusip: "912797ML0", auction_date: "2026-08-14", issue_date: "2026-08-19", maturity_date: "2026-11-19",
        type: "Bill", term: "13-Week", reopening: false, bid_to_cover: 2.87, high_yield: null, high_discount_rate: 4.212,
        total_accepted: 76000000000, competitive_accepted: 71000000000, primary_dealer_accepted: 38000000000,
        direct_accepted: 5000000000, indirect_accepted: 28000000000, dealer_take: 0.5352 },
      { cusip: "91282CKQ1", auction_date: "2026-08-12", issue_date: "2026-08-15", maturity_date: "2033-08-15",
        type: "Note", term: "7-Year", reopening: false, bid_to_cover: 2.41, high_yield: 4.087, high_discount_rate: null,
        total_accepted: 44000000000, competitive_accepted: 42000000000, primary_dealer_accepted: 15000000000,
        direct_accepted: 9000000000, indirect_accepted: 18000000000, dealer_take: 0.3571 },
    ],
  },
  // Treasury Daily Statement (2026-08-17, RAW display, treasuryDts.tsx).
  // Fixture covers one withheld-tax Deposits line (the hypothesis
  // highlight) and one Withdrawals line so the harness exercises both
  // sections and the highlighted-row styling.
  "/api/data/dts": {
    kind: "raw", source: "U.S. Treasury Fiscal Data — Daily Treasury Statement, Table II (public domain) (fixture)",
    attribution: "U.S. Treasury Fiscal Data — Daily Treasury Statement", time: "2026-08-17T12:00:00.000Z",
    record_date: "2026-08-14", count: 3,
    note: "daily TGA deposits/withdrawals by category, $ millions as published (fixture)",
    lines: [
      { record_date: "2026-08-14", account_type: "Federal Reserve Account", transaction_type: "Deposits",
        category: "Withheld Income and Employment Taxes", today_amt: 18420, mtd_amt: 142300, fytd_amt: 1820400,
        src_line: 2, rt: "2026-08-14" },
      { record_date: "2026-08-14", account_type: "Federal Reserve Account", transaction_type: "Withdrawals",
        category: "Department of Defense - Military Active Duty Pay", today_amt: 3210, mtd_amt: 28100, fytd_amt: 341200,
        src_line: 14, rt: "2026-08-14" },
      // the real shape of a roll-up row: FiscalData publishes the literal
      // STRING "null" as its category (2026-11-19: the page printed the word)
      { record_date: "2026-08-14", account_type: "Treasury General Account Total Deposits", transaction_type: "Deposits",
        category: "null", today_amt: 24450, mtd_amt: 299146, fytd_amt: 4953493,
        src_line: 30, rt: "2026-08-14" },
    ],
  },
  // FDA binary events (2026-08-17, RAW display, fdaEvents.tsx). Fixture
  // covers one approval, one parsed-date AdCom notice, and one
  // unparsed-date AdCom notice so the harness exercises both sections
  // and both dateConf badge states.
  "/api/data/fda-events": {
    kind: "raw", source: "openFDA (drugsfda approvals) + Federal Register (FDA advisory-committee notices) (fixture)",
    attribution: "openFDA (U.S. FDA); Federal Register (U.S. National Archives)", time: "2026-08-17T12:00:00.000Z",
    count: 3, note: "No PDUFA target dates — legally unavailable free (21 CFR 314.430); adcom dates carry parse confidence (fixture)",
    events: [
      { t: "approval", key: "NDA123456|001|2026-08-10", appl: "NDA123456", date: "2026-08-10",
        sponsor: "Example Therapeutics Inc.", brand: "Exampla", committee: null, title: null, url: null,
        pub: null, dateConf: null, rt: "2026-08-10" },
      { t: "adcom", key: "2026-12345", appl: "BLA987654", date: "2026-09-05",
        sponsor: "Sample Biopharma", brand: null, committee: "Oncologic Drugs Advisory Committee",
        title: "Sample Biopharma; BLA 987654; Meeting of the Oncologic Drugs Advisory Committee",
        url: "https://www.federalregister.gov/documents/2026/08/01/2026-12345", pub: "2026-08-01",
        dateConf: "parsed", rt: "2026-08-15" },
      { t: "adcom", key: "2026-12399", appl: null, date: null,
        sponsor: null, brand: null, committee: "Antimicrobial Drugs Advisory Committee",
        title: "Notice of Meeting of the Antimicrobial Drugs Advisory Committee",
        url: "https://www.federalregister.gov/documents/2026/08/03/2026-12399", pub: "2026-08-03",
        dateConf: "unparsed", rt: "2026-08-16" },
    ],
  },
  // ClinicalTrials.gov date drift (2026-10-20, RAW display, clinicalTrials.tsx).
  // Fixture covers a date slip with a ticker, an improvement without one, a
  // month-only (coarse) date, a status change and a new trial so both
  // sections and every badge state render.
  "/api/data/clinical-trials": {
    kind: "raw", source: "ClinicalTrials.gov API v2 (fixture)",
    attribution: "ClinicalTrials.gov (U.S. National Library of Medicine)", time: 1792441277485, tracked: 3242,
    count: 5, note: "fixture",
    events: [
      { t: "pcd_shift", key: "pcd_shift|NCT00000001|2026-12>2027-06", nct: "NCT00000001", sponsor: "Example Biosciences, Inc.",
        phase: "PHASE2", status: "RECRUITING", prevStatus: null, pcd: "2027-06-01", prevPcd: "2026-12-01", shiftDays: 182,
        coarse: true, title: "Fixture trial A", cond: "Fixture", enroll: 120, rt: "2026-10-19", ticker: "EXBI", cik: "0000000001" },
      { t: "pcd_shift", key: "pcd_shift|NCT00000002|2027-03-15>2027-02-01", nct: "NCT00000002", sponsor: "Sample Pharma Co",
        phase: "PHASE3", status: "ACTIVE_NOT_RECRUITING", prevStatus: null, pcd: "2027-02-01", prevPcd: "2027-03-15", shiftDays: -42,
        coarse: false, title: "Fixture trial B", cond: "Fixture", enroll: 800, rt: "2026-10-18", ticker: null, cik: null },
      { t: "status", key: "status|NCT00000003|RECRUITING>TERMINATED", nct: "NCT00000003", sponsor: "Example Biosciences, Inc.",
        phase: "PHASE2", status: "TERMINATED", prevStatus: "RECRUITING", pcd: "2027-01-01", prevPcd: null, shiftDays: null,
        coarse: false, title: "Fixture trial C", cond: "Fixture", enroll: 60, rt: "2026-10-19", ticker: "EXBI", cik: "0000000001" },
      { t: "new", key: "new|NCT00000004|2026-10-19", nct: "NCT00000004", sponsor: "Hansoh BioMedical R&D Company",
        phase: "PHASE2", status: "RECRUITING", prevStatus: null, pcd: "2027-03-01", prevPcd: null, shiftDays: null,
        coarse: false, title: "Fixture trial D", cond: "Fixture", enroll: 390, rt: "2026-10-19", ticker: null, cik: null },
      { t: "entered_window", key: "entered_window|NCT00000005|2026-10-17", nct: "NCT00000005", sponsor: "Sample Pharma Co",
        phase: "PHASE2/PHASE3", status: "ACTIVE_NOT_RECRUITING", prevStatus: null, pcd: "2027-10-01", prevPcd: null, shiftDays: null,
        coarse: true, title: "Fixture trial E", cond: "Fixture", enroll: 300, rt: "2026-10-17", ticker: null, cik: null },
    ],
  },
  // NHTSA vehicle complaints (2026-08-18, RAW display, vehicleComplaints.tsx).
  // Fixture covers two tickers so the harness exercises the ticker-grouping
  // + per-model table, and one model with a fire flag so the fire column
  // renders a nonzero value.
  "/api/data/vehicle-complaints": {
    kind: "raw", source: "NHTSA ODI complaints API (fixture)",
    attribution: "NHTSA Office of Defects Investigation", time: "2026-08-18T12:00:00.000Z",
    count: 3, note: "per-vehicle complaint stats over a CURATED ticker-mapped watchlist (fixture)",
    vehicles: [
      { ticker: "TSLA", make: "tesla", model: "model 3", model_year: 2024,
        total_complaints: 207, crash_count: 22, fire_count: 0, newest_filed: "2026-08-12" },
      { ticker: "TSLA", make: "tesla", model: "cybertruck", model_year: 2024,
        total_complaints: 168, crash_count: 22, fire_count: 2, newest_filed: "2026-08-02" },
      { ticker: "F", make: "ford", model: "bronco", model_year: 2024,
        total_complaints: 72, crash_count: 4, fire_count: 3, newest_filed: "2026-08-10" },
    ],
  },
  // FDIC bank failures (2026-08-18, RAW display, bankFailures.tsx).
  // Fixture covers one estimated-cost failure and one FDIC-assisted merger
  // with cost still null so the harness exercises the never-coerce-to-zero
  // honesty rule on the "Est. DIF cost" column.
  "/api/data/bank-failures": {
    kind: "raw", source: "FDIC Bank Data API — bank failures (fixture)",
    attribution: "FDIC Bank Data API", time: "2026-08-18T12:00:00.000Z",
    count: 2, note: "most recent US bank failures/assistance events, amounts in $ thousands as published (fixture)",
    failures: [
      { cert: 25796, name: "Community Bank and Trust - West Georgia", fail_date: "2026-05-01",
        city_state: "LAGRANGE, GA", state: "GA", charter_class: "NM", restype: "FAILURE",
        assets_k: 305716, deposits_k: 296420, cost_k: null, fund: "DIF", rt: "2026-08-18" },
      { cert: 57488, name: "Pulaski Savings Bank", fail_date: "2026-01-30",
        city_state: "CHICAGO, IL", state: "IL", charter_class: "NM", restype: "FAILURE",
        assets_k: 41200, deposits_k: 39800, cost_k: 19647, fund: "DIF", rt: "2026-08-18" },
    ],
  },
  // USAspending federal contract awards (2026-08-26, RAW display,
  // usaspendingContracts.tsx). Fixture covers one ticker-matched award, one
  // unmatched recipient (tkr:null, excluded by the "ticker-matched only"
  // default filter), and one negative deobligation so the harness exercises
  // both the never-guess-a-ticker rule and the signed-amount formatting.
  "/api/data/contracts": {
    kind: "raw", source: "USAspending.gov, U.S. Department of the Treasury (fixture)",
    attribution: "Source: USAspending.gov, U.S. Department of the Treasury",
    time: "2026-08-26T12:00:00.000Z", count: 3,
    note: "action dates are signature dates — DoD/USACE publish ~90 days late; rt is the as-seen date",
    contracts: [
      { aid: "CONT_AWD_FIXTURE1", piid: "FA123", ad: "2026-05-20", amt: 4200000, r: "LOCKHEED MARTIN CORP",
        pn: null, tkr: "LMT", ag: "Department of Defense", sub: "Department of the Air Force",
        desc: "Sustainment support services", rt: "2026-08-26" },
      { aid: "CONT_AWD_FIXTURE2", piid: "GS456", ad: "2026-08-15", amt: 88000, r: "RIVERBEND LOCAL CONSTRUCTION LLC",
        pn: null, tkr: null, ag: "General Services Administration", sub: "Public Buildings Service",
        desc: "Facility repair, region 4", rt: "2026-08-26" },
      { aid: "CONT_AWD_FIXTURE3", piid: "N789", ad: "2026-07-01", amt: -150000, r: "HUNTINGTON INGALLS INDUSTRIES",
        pn: null, tkr: "HII", ag: "Department of Defense", sub: "Department of the Navy",
        desc: "Deobligation, contract modification 0003", rt: "2026-08-26" },
    ],
  },
  "/api/data/cot": {
    kind: "raw", source: "CFTC Commitments of Traders, disaggregated futures-only (public domain)", time: 1,
    report_date: "2026-06-23", count: 2,
    markets: [
      { code: "067651", market: "CRUDE OIL - NYMEX", commodity: "CRUDE OIL", open_interest: 428305, m_money_long: 68457, m_money_short: 138890 },
      { code: "088691", market: "GOLD - COMMODITY EXCHANGE INC.", commodity: "GOLD", open_interest: 512300, m_money_long: 91200, m_money_short: 40100 },
    ],
  },
  "/api/data/cot/history": {
    kind: "raw", source: "CFTC Commitments of Traders, disaggregated futures-only (public domain)", weeks: 52,
    today: {
      report_date: "2026-06-23",
      rows: [
        { code: "067651", market: "CRUDE OIL - NYMEX", commodity: "CRUDE OIL", open_interest: 428305, m_money_long: 68457, m_money_short: 138890 },
      ],
    },
    trend: [
      { report_date: "2026-06-16", markets: 274, total_open_interest: 18320000 },
      { report_date: "2026-06-23", markets: 274, total_open_interest: 18455000 },
    ],
  },
  "/api/data/insider/history": {
    kind: "raw", source: "SEC EDGAR (Form 4) — accumulated archive", days: 30, count: 2,
    filings: [
      {
        accession: "0001-26-1", filedAt: "2026-07-03", indexUrl: "https://www.sec.gov/x/1/",
        issuerName: "CYPHERPUNK TECHNOLOGIES INC.", issuerTradingSymbol: "CYPH",
        owners: [{ cik: "1", name: "Richard Christian M", isDirector: true, isOfficer: false, isTenPercentOwner: false, officerTitle: null }],
        transactions: [{ table: "derivative", kind: "award_grant", shares: 75000, pricePerShare: 0, transactionDate: "2026-07-01" }],
      },
      {
        accession: "0001-26-2", filedAt: "2026-07-02", indexUrl: "https://www.sec.gov/x/2/",
        issuerName: "STRATUS PROPERTIES INC", issuerTradingSymbol: "STRS",
        owners: [{ cik: "2", name: "Oasis Management Co Ltd.", isDirector: false, isOfficer: false, isTenPercentOwner: true, officerTitle: null }],
        transactions: [{ table: "nonDerivative", kind: "open_market_sale", shares: 10000, pricePerShare: 28.9, transactionDate: "2026-06-30" }],
      },
    ],
  },
  "/api/data/trains": {
    kind: "raw", source: "Digitraffic Finland (CC BY 4.0) + Entur Norway (NLOD)",
    time: 1, coverage: "FI + NO (launch)", count: 3,
    sources: [
      { key: "digitraffic", country: "FI", status: "ok", count: 2 },
      { key: "entur", country: "NO", status: "ok", count: 1 },
    ],
    trains: [
      { id: "FI-62-2026-07-04", country: "FI", lat: 63.7632, lon: 27.3121, speed_kmh: 139, bearing: null, label: "Train 62", ts: 1 },
      { id: "FI-104-2026-07-04", country: "FI", lat: 62.5036, lon: 29.8569, speed_kmh: 129, bearing: null, label: "Train 104", ts: 1 },
      { id: "NO-71-12", country: "NO", lat: 59.741, lon: 10.2016, speed_kmh: 108, bearing: 45, label: "FLY1", ts: 1 },
    ],
  },
  // The real compiled dataset (repo file, deterministic): ~9.8k plants
  // exercises the clustering path at full production scale in the perf
  // window, on top of the 10k-aircraft budget check.
  "/api/data/powerplants": {
    kind: "raw",
    ...JSON.parse(readFileSync(path.join(ROOT, "datacore", "powerplants", "us_power_plants.json"), "utf8")),
  },
  // NRC daily reactor status full view (2026-08-09) — one plant per status
  // tier (full/reduced/outage/unknown) so the filter chips, worst-first
  // sort, and per-unit expand-on-click all get exercised at 390px; a
  // multi-unit plant (Beaver Valley) exercises the units-breakdown table.
  "/api/data/nrc-reactor-status": {
    kind: "raw",
    source: "U.S. NRC Power Reactor Status Reports (public domain, US government data) (fixture)",
    attribution: "U.S. Nuclear Regulatory Commission (NRC)",
    time: 1, date: "2026-08-09", count: 5,
    note: "daily percent-of-rated-thermal-power per operating unit (NRC unit granularity, not plant); outage-adjacent signals stay gate-locked until ladder validation",
    rows: [],
    plantCount: 4,
    plants: [
      { idx: 1, name: "Beaver Valley", lat: 40.622, lon: -80.434, mw: 1836, owner: "Energy Harbor",
        units: [{ unit: "Beaver Valley 1", power: 100 }, { unit: "Beaver Valley 2", power: 100 }],
        avgPower: 100, status: "full" },
      { idx: 2, name: "Palo Verde", lat: 33.3881, lon: -112.8646, mw: 3937, owner: "Arizona Public Service",
        units: [{ unit: "Palo Verde 1", power: 55 }],
        avgPower: 55, status: "reduced" },
      { idx: 3, name: "Diablo Canyon", lat: 35.2116, lon: -120.8551, mw: 2256, owner: "PG&E",
        units: [{ unit: "Diablo Canyon 1", power: 0 }],
        avgPower: 0, status: "outage" },
      { idx: 4, name: "Waterford 3", lat: 29.9942, lon: -90.4696, mw: 1250, owner: "Entergy",
        units: [{ unit: "Waterford 3", power: null }],
        avgPower: null, status: "unknown" },
    ],
  },
  // GNSS integrity anomaly signal (2026-08-16) — the first live gate2_pass
  // SIGNAL detail page. Fixture mirrors the real shape from
  // server/gnssIntegritySignal.ts's computeGnssIntegritySignal(), with one
  // band of each elevated/not-elevated state so both table badge colors
  // render at every canonical width.
  // Signals PRODUCT catalog (server/signalProducts.ts buildSignalProducts()).
  // Two products (one live, one warming up — both dot colors render), one
  // in-validation row with a truncated note, and the five gate definitions.
  "/api/data/signal-products": {
    generated_at: "2026-11-11T00:00:00.000Z",
    ladder_compiled: "2026-11-10",
    products: [
      {
        root_id: "gnss_integrity_adsb", product: "GNSS interference — Baltic corridor",
        measures: "Per-altitude-band GPS position-integrity degradation over the Gdańsk–Bornholm–Gotland corridor (fixture text).",
        validated_channel: "Elevation at cruise and mid altitude only; gate 1 is PARTIAL (fixture text).",
        cadence: "Rolling 21-day window, recomputed every 10 minutes (fixture text).",
        data_route: "/api/data/gnss-integrity-signal", v1_path: "/api/v1/data/gnss-integrity-signal", detail_route: "#/data/gnss-integrity",
        license_key: "data/gnss-integrity-signal",
        name: "GPS-interference-from-ADS-B (GNSS integrity field passthrough)", category: "geopolitical_intelligence",
        status: "gate2_pass", current_gate: 2, last_update_date: "2026-09-26", source_ref: "experiments.md:1 (fixture)",
        tradeable: false,
        license: { resell: "share-alike", attribution: "VolTradeAI datacore over adsb.lol (ODbL 1.0)", note: "Fixture license note." },
        live: { state: "live", generated_at: "2026-11-10T23:48:00.000Z", headline: "PASS — 2/2 expected bands elevated (cruise, mid) over 21 archive days" },
      },
      {
        root_id: "wikimedia_pageviews_attention", product: "Attention spike → forward volume",
        measures: "A Wikipedia pageview spike (z ≥ 2.0 against a trailing baseline) on a 23-ticker seed (fixture text).",
        validated_channel: "Small/mid-cap tickers: elevated forward VOLUME at 1/3/5 days, net of a same-day 8-K; no price claim (fixture text).",
        cadence: "Computed per request from the daily pageviews archive (fixture text).",
        data_route: "/api/data/wiki-attention-signal", v1_path: "/api/v1/data/wiki-attention-signal", detail_route: "#/data/wiki-attention-signal",
        license_key: "data/wiki-attention-signal",
        name: "Wikimedia pageviews (retail-attention proxy, pytrends replacement)", category: "other",
        status: "gate2_pass", current_gate: 2, last_update_date: "2026-09-28", source_ref: "experiments.md:2 (fixture)",
        tradeable: false,
        license: { resell: "ok", attribution: "Wikimedia pageviews API (Wikimedia Foundation, CC0)", note: "Fixture license note." },
        live: { state: "warming_up", generated_at: null, headline: "first archive scan in progress" },
      },
    ],
    unwired: [],
    in_validation: [
      { id: "sec_8k_earnings_language", name: "SEC 8-K Item 2.02 earnings-language tone signal", category: "other", status: "gate2_pending", current_gate: 2, last_update_date: "2026-10-02", note: "Gate-2 90-day re-run: mechanical PASS at 1/3/5d but FRAGILE across months, so not promoted; QoQ language-delta feature untested (fixture text)…" },
    ],
    gate_definitions: [
      { gate: 1, name: "DATA", test: "the reading is verified against an external truth source (fixture)" },
      { gate: 2, name: "SIGNAL", test: "predictive power is measured statistically, out of sample (fixture)" },
      { gate: 3, name: "LOGIC", test: "entry/exit rules are backtested by ablation (fixture)" },
      { gate: 4, name: "SIZING", test: "results compared against equal-weight (fixture)" },
      { gate: 5, name: "EXECUTION", test: "the slippage gap isolates fill quality (fixture)" },
    ],
    honesty: "A signal is listed here only after ladder gate 2. None has passed gate 3, so nothing here is a trade recommendation (fixture text).",
  },
  "/api/data/gnss-integrity-signal": {
    kind: "signal", root_id: "gnss_integrity_adsb", generated_at: "2026-08-16T00:00:00.000Z",
    gate: { current_gate: 2, status: "gate2_pass" }, verdict: "PASS",
    bands: [
      { band: "cruise", candidate_k: 15, candidate_n: 414, control_rate: 0.00147, expected_under_null: 0.61, p_value: 0.000001, elevated: true, expected_to_elevate: true },
      { band: "mid", candidate_k: 12, candidate_n: 277, control_rate: 0.00272, expected_under_null: 0.75, p_value: 0.000001, elevated: true, expected_to_elevate: true },
      { band: "low", candidate_k: 7, candidate_n: 903, control_rate: 0.02214, expected_under_null: 20.0, p_value: 0.46, elevated: false, expected_to_elevate: false },
    ],
    region: {
      candidate_label: "Baltic corridor (Gdańsk–Bornholm–Gotland)", candidate_bbox: [53, 60, 17, 24],
      control_label: "NY + Paris control region", control_bbox: [35, 55, -80, 10],
    },
    freshness: {
      writer_live_since: "2026-08-11",
      candidate: { days_read: ["2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"], days_missing: [], rows_scanned: 1613, truncated: false },
      control: { days_read: ["2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"], days_missing: [], rows_scanned: 32944, truncated: false },
    },
    methodology_note: "Broadcast-origin ADS-B rows only. One-tailed exact binomial test per altitude band (fixture text).",
    caveats: [
      "GATE 1 is PARTIAL, not a full pass — DTU Space's Bornholm RF station corroborates the phenomenon/region, not the exact sample dates (fixture text).",
      "Small, growing sample — the archive has carried integrity fields only since 2026-08-11 (fixture text).",
      "Not tradeable — this is GATE 2 statistical discrimination, not a trading signal (fixture text).",
    ],
    license: { source: "adsb.lol (ODbL 1.0) + adsb.fi/airplanes.live (non-commercial fallbacks)", note: "Fixture license note." },
  },
  // Wikimedia attention spike -> forward volume signal (2026-09-05) — the
  // second live gate2_pass SIGNAL detail page. Fixture mirrors the real
  // shape from server/wikiAttentionSignal.ts's computeWikiAttentionSignal(),
  // with one spiking row and one normal row so both status badge colors
  // render, plus a mix of complete/incomplete baselines and a null-z row.
  "/api/data/wiki-attention-signal": {
    kind: "signal", root_id: "wikimedia_pageviews_attention", generated_at: "2026-09-05T00:00:00.000Z",
    gate: { current_gate: 2, status: "gate2_pass", channel: "trading_volume_elevation" },
    z_threshold: 2.0, trailing_window_days: 90, min_baseline_days: 20,
    tickers: [
      { ticker: "GME", article: "GameStop", cap_tier: "small_mid", latest_date: "2026-09-04", current_views: 42000, baseline_mean: 8200.5, baseline_days: 62, baseline_complete: false, z_score: 3.41, spike: true },
      { ticker: "PLTR", article: "Palantir", cap_tier: "small_mid", latest_date: "2026-09-04", current_views: 6100, baseline_mean: 5850.2, baseline_days: 62, baseline_complete: false, z_score: 0.62, spike: false },
      { ticker: "NVDA", article: "Nvidia", cap_tier: "mega", latest_date: "2026-09-04", current_views: 15200, baseline_mean: 14980.0, baseline_days: 62, baseline_complete: false, z_score: 0.18, spike: false },
      { ticker: "ACHR", article: "Archer_Aviation", cap_tier: "small_mid", latest_date: null, current_views: null, baseline_mean: null, baseline_days: 0, baseline_complete: false, z_score: null, spike: false },
    ],
    spike_count: 1,
    validated_effect: {
      study_date: "2026-09-04", bonferroni_alpha: 0.005,
      small_mid: [
        { horizon_days: 1, mean_ratio: 1.279, baseline_ratio: 1.039, p_value: 0.0001 },
        { horizon_days: 3, mean_ratio: 1.226, baseline_ratio: 1.053, p_value: 0.0001 },
        { horizon_days: 5, mean_ratio: 1.195, baseline_ratio: 1.063, p_value: 0.0014 },
      ],
      mega: [
        { horizon_days: 1, mean_ratio: 1.119, baseline_ratio: 0.995, p_value: 0.0083 },
        { horizon_days: 3, mean_ratio: 1.100, baseline_ratio: 0.997, p_value: 0.0050 },
        { horizon_days: 5, mean_ratio: 1.101, baseline_ratio: 0.998, p_value: 0.0023 },
      ],
    },
    methodology_note: "Live board: each seed ticker's latest archived daily pageview count, z-scored against its own trailing baseline (fixture text).",
    caveats: [
      "This live board does NOT re-check today's spikes against EDGAR for a same-day-or-prior-day 8-K (fixture text).",
      "Realized volatility showed NO significant elevation at any horizon in the validated study (fixture text).",
      "GATE 3 (a backtested, cost-net entry/exit rule) has not been attempted for this root (fixture text).",
    ],
    license: { source: "Wikimedia pageviews API (CC0/CC-BY) + SEC EDGAR (public domain, offline study only)", note: "Fixture license note." },
  },
  // EPA CAMD power-plant utilization full view — three facilities of
  // varying unit count/fuel mix so the table renders multiple rows at
  // every canonical width; shape mirrors server/epaCamd.ts's
  // aggregateByFacility() output exactly (facilities pre-sorted by
  // sumGrossLoad descending).
  "/api/data/plant-operations": {
    kind: "raw", predictive: false,
    source: "U.S. EPA Clean Air Markets Division (CAMD) — Continuous Emissions Monitoring, unit-level daily (fixture)",
    attribution: "U.S. EPA Clean Air Markets Division (CAMD)",
    state: "TX",
    note: "Per-facility rollup (sum grossLoad MW-days, sum operating hours) of the newest archived quarter (fixture text).",
    key_mode: "shared api.data.gov DEMO_KEY (rate-limited)",
    time: 1, year: 2026, quarter: 2, unit_days: 9009,
    facilities: [
      { facilityId: 1, facilityName: "W A Parish", unitCount: 4, sumOpTime: 5824.5, sumGrossLoad: 612340.2, primaryFuelInfo: "Coal", lat: 29.479, lon: -95.638, ownerOperator: "NRG Texas" },
      { facilityId: 2, facilityName: "Comanche Peak", unitCount: 2, sumOpTime: 4368.0, sumGrossLoad: 421980.7, primaryFuelInfo: "Nuclear", lat: 32.298, lon: -97.785, ownerOperator: "Vistra" },
      { facilityId: 3, facilityName: "Barney M Davis", unitCount: 1, sumOpTime: 980.3, sumGrossLoad: 58210.4, primaryFuelInfo: "Natural Gas", lat: 27.634, lon: -97.322, ownerOperator: "Talen Energy" },
    ],
  },
  // COVERAGE GAP CLOSED (2026-09-10) — three fixtures for the newly-added
  // PAGES entries above that had none before (filings/earnings/short-volume/
  // cot/graph already had fixtures shipped with their own PAGES entries
  // elsewhere; these three never got a PAGES entry OR a fixture, so they
  // fell through to the generic 200 {} default and were fully unexercised).
  "/api/data/ats-summary": {
    kind: "raw", source: "FINRA Query API — ATS/OTC venue volume (fixture)",
    attribution: "FINRA Query API", time: 1,
    note: "Fixture: monthly cut has no ATS-vs-OTC venue split.",
    weekly: {
      week_start: "2026-09-01", tiers_covered: ["T1", "OTCE"], records: 4820,
      composition: { ATS: 0.41, OTC: 0.59 },
      top_ats_by_symbol: [{ symbol: "FXA", name: "Fixture Alpha Corp", shares: 1204500, trades: 3820, notional: 60225000 }],
      top_otc_by_symbol: [{ symbol: "FXB", name: "Fixture Beta Inc", shares: 980200, trades: 2910, notional: 41208400 }],
    },
    monthly: {
      month_start: "2026-08-01", tiers_covered: ["T1", "T2", "OTCE"], records: 19340,
      composition: { ATS: 0.38, OTC: 0.62 },
      top_otc_by_symbol: [{ symbol: "FXB", name: "Fixture Beta Inc", shares: 4102000, trades: 11820, notional: 172284000 }],
    },
    blocks: {
      month_start: "2026-08-01", records: 640,
      top_venues_by_block_volume: [
        { mpid: "FXV1", name: "Fixture Venue One", block_quantity: 820000, block_count: 41, average_block_size: 20000, ats_share_percent: 62.4, ats_block_share_rank: 1 },
      ],
    },
  },
  "/api/data/microstructure": {
    kind: "raw", source: "SEC MIDAS market-structure metrics (fixture)",
    attribution: "SEC MIDAS", note: "Fixture: small-cap HFT-colonization filter watchlist, RAW display only.",
    summary: {
      period: "2026-09-08", kind_counts: { stock: 4200, etf: 380 },
      newest_date: "2026-09-08", rows: 4580,
      smallcap_watch: [
        { ticker: "FXC", mcapRank: 1840, cancelToTrade: 12.4, hiddenRatePct: 8.1, oddLotRatePct: 3.2 },
        { ticker: "FXD", mcapRank: 2010, cancelToTrade: 21.7, hiddenRatePct: null, oddLotRatePct: 5.9 },
      ],
      smallcap_max_rank: 3000, min_trades_for_hidden: 100, top_cap: 500,
    },
  },
  // [REPAIR 2026-09-27] 8 new fixtures for the fixture-coverage sweep above
  // (ats-summary/contracts/plant-operations/microstructure already existed
  // for their own dedicated #/data/* pages — only these 8 were missing a
  // data fixture entirely, so their toggle would have fallen through to
  // the generic 200 "{}" default). Shapes copied from each route handler
  // in server/routes.ts (and, for grid-generation, the RespondentGeneration
  // Stat shape in server/gridGeneration.ts) — field names verified against
  // the client's own strict checks (Array.isArray(...) gates, property
  // reads in the popup/detail builders), not guessed.
  "/api/data/grid-generation": {
    kind: "raw", predictive: false,
    source: "EIA-930 Hourly Electric Grid Monitor (public domain) (fixture)",
    attribution: "EIA-930 Hourly Electric Grid Monitor",
    time: 1, count: 2,
    note: "Fixture: hourly net generation (MWh) by fuel type; storage fuel types legitimately read negative while charging.",
    respondents: [
      { respondent: "ERCO", latest_period: "2026-09-27T11", total_mwh: 58400.2, hours_in_window: 24,
        fuel_mix: [{ fueltype: "WND", latest_mwh: 21400.5 }, { fueltype: "NG", latest_mwh: 18200.0 }, { fueltype: "SUN", latest_mwh: 9800.7 }] },
      { respondent: "US48", latest_period: "2026-09-27T11", total_mwh: 412300.9, hours_in_window: 24,
        fuel_mix: [{ fueltype: "NG", latest_mwh: 168400.0 }, { fueltype: "NUC", latest_mwh: 78200.3 }, { fueltype: "BAT", latest_mwh: -1200.4 }] },
    ],
  },
  "/api/data/meteors": {
    kind: "raw", source: "NASA/JPL CNEOS fireball data (US Government sensors) — public domain (fixture)",
    attribution: "NASA/JPL CNEOS", fetched_at: "2026-09-27T00:00:00.000Z", last_error: null,
    count: 2, with_direction: 1,
    note: "Fixture: bolides — large meteors that exploded in the atmosphere, published after the fact.",
    events: [
      { t: 1758931200, date: "2026-09-27", la: 41.2, lo: -95.9, e: 8.4, imp: 8.4, alt: 34.1, vel: 18.6, hdg: 245 },
      { t: 1758585600, date: "2026-09-23", la: -12.4, lo: 132.1, e: 1.2, imp: 1.2, alt: null, vel: null, hdg: null },
    ],
  },
  "/api/data/military_installations": {
    kind: "raw", predictive: false,
    source: "US DoD open data, OpenStreetMap contributors (ODbL), and cited government publications (fixture)",
    attribution: "© OpenStreetMap contributors (ODbL); US DoD open data (public domain); cited government publications",
    banner: "Officially published installation locations. Reference geography only — not operational information (fixture text).",
    no_cross_ties: true, count: 2,
    installations: [
      { name: "Fixture Fort Alpha", operator_nation: "United States of America", branch: "Army", type: "base", status: "active",
        geometry: { type: "Point", coordinates: [-97.6, 38.4] }, centroid: [-97.6, 38.4] },
      { name: "Fixture Naval Station Beta", operator_nation: "United States of America", branch: "Navy", type: "base", status: "active",
        geometry: { type: "Point", coordinates: [-76.3, 36.9] }, centroid: [-76.3, 36.9] },
    ],
  },
  "/api/data/quakehistory": {
    kind: "raw", predictive: false, source: "USGS ANSS Comprehensive Catalog (ComCat), public domain (fixture)",
    attribution: "USGS ANSS ComCat", count: 2, min_year: 1960, max_year: 2026,
    quakes: [
      { lon: -73.05, lat: -38.29, m: 9.5, y: 1960, dep: 33, pl: "Valdivia, Chile", d: "1960-05-22" },
      { lon: 143.05, lat: 38.3, m: 9.1, y: 2011, dep: 29, pl: "Near the east coast of Honshu, Japan", d: "2011-03-11" },
    ],
  },
  // Shape verified against server/shadowFleet.ts's `ShadowStats` interface
  // and server/routes.ts's refreshShadowStats() (not guessed): the client's
  // own l.id === "shadowstats" render block (client/src/pages/datamap.tsx)
  // does an unguarded `Object.entries(shadowStats.loiter_by_zone)` whenever
  // loiter_events > 0 — the field this fixture originally omitted, which
  // crashed the whole app (uncaught TypeError inside the layers-panel
  // render, caught only by the top-level ErrorBoundary) the instant the
  // panel was opened with this layer's fixture wired in. Found live via a
  // VT_DEBUG_CONSOLE=1 run of this harness, not assumed.
  "/api/data/shadowstats": {
    kind: "raw", source: "Derived from our own AIS position archive (terrestrial coverage; began 2026-07-03) (fixture)",
    zones: [{ id: "fixture_sts_zone", name: "Fixture STS Zone" }],
    generated_at: "2026-09-27T00:00:00.000Z",
    window_hours: 72, vessels_seen: 1204, points_read: 48200,
    gap_events: 18, gap_examples: [
      { mmsi: "412345678", name: "FIXTURE VESSEL", darkAt: 1758900000, reappearAt: 1758921600,
        gapHours: 6, distanceKm: 42.3, from: [53.9, 12.1], to: [54.1, 12.6] },
    ],
    identity_candidates: 3, loiter_events: 41,
    loiter_by_zone: { fixture_sts_zone: 41 },
    caveat: "RAW statistics from our own terrestrial-AIS archive. A gap can be innocent (out of receiver range) or evasive (deliberate AIS shutoff) — this count does not distinguish them (fixture).",
  },
  "/api/data/superfund": {
    kind: "raw", predictive: false,
    source: "EPA Superfund National Priorities List (NPL) — SEMS, U.S. EPA (public domain) (fixture)",
    attribution: "U.S. EPA Superfund (SEMS/NPL)",
    note: "Fixture: factual site records — location/status/HRS score as EPA publishes them, not a risk claim.",
    time: 1, health: { source: "EPA Superfund NPL (SEMS), public domain", clean: 2, suspect: 0, freshness: "live", source_date: "2026-09-27" },
    sites: [
      { name: "Fixture Solvent Recovery Site", epa_id: "FXD980000001", hrs_score: 52.3, status: "NPL Site", state: "NJ", city: "Fixtureville", county: "Fixture County", lat: 40.7, lon: -74.2, listed: "1986-06-10" },
      { name: "Fixture Former Smelter", epa_id: "FXD980000002", hrs_score: null, status: "Deleted NPL Site", state: "PA", city: "Sample City", county: null, lat: 40.0, lon: -76.5, listed: "1998-09-29" },
    ],
  },
  "/api/data/waterviolators": {
    kind: "raw", predictive: false,
    source: "EPA ECHO — Clean Water Act compliance records (public domain) (fixture)",
    attribution: "U.S. EPA ECHO / NPDES",
    time: 1, health: { source: "EPA ECHO Clean Water Act compliance (public domain)", clean: 2, suspect: 0, freshness: "live", source_date: "2026-09-27" },
    violators: [
      { name: "Fixture Chemical Plant", id: "FX0012345", city: "Fixtureville", state: "TX", lat: 29.8, lon: -95.3, permit: "Individual", snc: "Effluent violation", qtrs: 10, actions: 2 },
      { name: "Fixture Paper Mill", id: "FX0067890", city: "Sample Town", state: "GA", lat: 33.0, lon: -83.5, permit: "General", snc: "Report violation", qtrs: 9, actions: null },
    ],
  },
  "/api/data/dtcc-swaps": {
    kind: "raw", source: "DTCC SBSDR equity total-return-swap dissemination (fixture)",
    attribution: "DTCC SBSDR", time: 1,
    file_date: "2026-09-09", source_date: "2026-09-09",
    us_underlier_rows_today: 118420, new_rows_archived: 612, total_archived: 214880,
    note: "Fixture: US-underlier scope only; basket-swap rows excluded from this view (see server/dtccSwaps.ts).",
    top_rows: [
      {
        dissemination_id: "FX00000001", action_type: "NEWT", event_timestamp: "2026-09-09T14:22:00Z", effective_date: "2026-09-10",
        notional_amount: 48200000, notional_currency: "USD", underlier_id: "US0000000FX", underlier_id_source: "ISIN",
        underlier_name: "Fixture Small-Cap Corp",
      },
      {
        dissemination_id: "FX00000002", action_type: "MODI", event_timestamp: "2026-09-09T12:05:00Z", effective_date: "2026-09-10",
        notional_amount: null, notional_currency: "USD", underlier_id: "037833100", underlier_id_source: "CUSIP",
        underlier_name: "Fixture Mid-Cap Holdings",
      },
    ],
  },
  "/api/data/xbrl-fundamentals": {
    kind: "raw", source: "SEC EDGAR XBRL frames API (fixture)", attribution: "U.S. Securities and Exchange Commission, EDGAR",
    note: "Fixture: numbers are the values companies tagged in their own filings, as SEC served them on our nightly read (see server/xbrlFrames.ts).",
    warming_up: false, nights: 3,
    concepts: [
      { tag: "Revenues", label: "Revenue", unit: "USD", kind: "duration" },
      { tag: "NetIncomeLoss", label: "Net income", unit: "USD", kind: "duration" },
      { tag: "Assets", label: "Total assets", unit: "USD", kind: "instant" },
      { tag: "EntityCommonStockSharesOutstanding", label: "Shares outstanding", unit: "shares", kind: "instant" },
    ],
    last_run: {
      day: "2026-11-21", started: "2026-11-21T07:10:00.000Z", finished: "2026-11-21T07:11:00.000Z", status: "ok", detail: "fixture",
      frames: { roster: 96, read: 89, not_published: 7, failed: 0 }, rows: { first: 412, revised: 2, dropped: 0, refiled: 300, backfill: 0 }, companies: 7606,
    },
    coverage: [
      { tag: "Revenues", period: "CY2026Q3", state: "read", held: 1509, first: 60, revised: 0, dropped: 0, backfill: false },
      { tag: "NetIncomeLoss", period: "CY2026Q4", state: "not_published", held: 0, first: 0, revised: 0, dropped: 0, backfill: false },
      { tag: "NetIncomeLoss", period: "CY2026Q3", state: "read", held: 4618, first: 212, revised: 2, dropped: 0, backfill: false },
      { tag: "Assets", period: "CY2026Q3I", state: "read", held: 4952, first: 90, revised: 0, dropped: 0, backfill: false },
      { tag: "EntityCommonStockSharesOutstanding", period: "CY2026Q3I", state: "read", held: 4197, first: 50, revised: 0, dropped: 0, backfill: false },
    ],
    revisions: [
      { seen: "2026-11-21T07:10:00.000Z", tag: "NetIncomeLoss", period: "CY2025Q3", cik: 811830, name: "FIXTURE HOLDINGS INTERNATIONAL CORPORATION", ticker: "FIXT", end: "2025-09-30", val: 456000000, prev_val: 455000000, unit: "USD", filing: "https://www.sec.gov/", prev_filing: "https://www.sec.gov/" },
      { seen: "2026-11-21T07:10:00.000Z", tag: "NetIncomeLoss", period: "CY2026Q2", cik: 1234567, name: "FIXTURE MICRO CO", ticker: null, end: "2026-06-30", val: -3531000, prev_val: -2900000, unit: "USD", filing: "https://www.sec.gov/", prev_filing: null },
    ],
  },
  "/api/data/fmcsa-oos": {
    kind: "raw", source: "FMCSA Out-of-Service Orders (data.transportation.gov, public domain) (fixture)",
    attribution: "FMCSA Out-of-Service Orders", time: 1, count: 2,
    note: "Fixture: trailing-45-day window of motor-carrier out-of-service orders (see server/fmcsaOutOfService.ts).",
    orders: [
      { dot_number: "4539029", legal_name: "FIXTURE TRUCKING & REPAIR LLC", oos_date: "2026-09-26", oos_reason: "New Entrant Revoked - Refusal of Audit/No Contact", status: "ACTIVE", rescind_date: null },
      { dot_number: "1234567", legal_name: "FIXTURE FREIGHT CO", oos_date: "2026-09-01", oos_reason: "Imminent Hazard", status: "INACTIVE", rescind_date: "2026-09-10" },
    ],
  },
};

// FLIGHT PROGRAM B1 (#1207, 2026-09-28) moved ZOOMED-OUT views of the live
// aircraft layer to /api/data/aircraft/global (compact fields+rows payload,
// client/src/lib/air/globalFeed.ts adaptGlobalPayload). The harness starts
// zoomed out and had no fixture for it, so every aircraft check (fields-on,
// data-richness, trail) went red on main with "no aircraft rendered" —
// a fixture gap, not a product bug (prod serves thousands of rows). The
// global fixture re-encodes the SAME 10k synthetic aircraft, so both feeds
// show identical planes and the 9,500-feature richness floor still bites.
// `at` is fixed and every seenAt equals it: fresh rows, deterministic runs.
FIXTURES["/api/data/aircraft/global"] = (() => {
  const at = 1790000000000;
  const fields = ["hex", "lon", "lat", "altFt", "gsKt", "trk", "callsign", "type", "seenAt", "cat", "gnd", "reg", "src"];
  const rows = FIXTURES["/api/data/aircraft"].aircraft.map((a) => [
    a.icao24, a.lon, a.lat,
    a.altitude_m == null ? null : Math.round(a.altitude_m / 0.3048),
    a.velocity_ms == null ? null : Math.round(a.velocity_ms / 0.5144),
    a.heading, a.callsign, a.type, at, a.category, a.on_ground ? 1 : 0, "", "adsblol",
  ]);
  return { at, count: rows.length, full: true, scope: "world", fields, rows,
           coverage: { opensky: { enabled: false } }, honesty: "fixture worldwide snapshot for the visual harness" };
})();

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json", ".woff2": "font/woff2", ".ico": "image/x-icon" };

// Deterministic weather tile standing in for the proxy's OUTPUT (alpha-
// amplified, per owmTiles.ts): a colored gradient at the strength production
// serves. The amplification itself is unit-tested against a real captured
// prod tile; this fixture exercises the CLIENT half — mount, blend, opacity.
const WX_TILE_PNG = (() => {
  const png = new PNG({ width: 256, height: 256 });
  for (let y = 0; y < 256; y++) {
    for (let x = 0; x < 256; x++) {
      const i = (y * 256 + x) * 4;
      png.data[i] = Math.round(40 + x * 0.8);
      png.data[i + 1] = 60;
      png.data[i + 2] = Math.round(200 - y * 0.6);
      png.data[i + 3] = 190; // amplified-output strength (cap 230 upstream)
    }
  }
  return PNG.sync.write(png);
})();

// Mean per-channel pixel difference between two PNG buffers — the same
// metric verify_weather_prod.mjs uses for the prod pixel proof.
function pngMeanDiff(a, b) {
  const A = PNG.sync.read(a), B = PNG.sync.read(b);
  if (A.width !== B.width || A.height !== B.height) return 255;
  let sum = 0; const n = A.width * A.height;
  for (let i = 0; i < n * 4; i += 4) {
    sum += Math.abs(A.data[i] - B.data[i]) + Math.abs(A.data[i + 1] - B.data[i + 1]) + Math.abs(A.data[i + 2] - B.data[i + 2]);
  }
  return sum / (n * 3);
}

// Trail-refresh ratchet state: counts /api/data/track/* hits so the battery
// can assert the popup re-pulls while open (module scope — batteries read it).
let trackCalls = 0;

function startServer() {
  return new Promise((resolve) => {
    // LIVE CHANGE (slice F, 2026-10-21): the bot page must react to the bot
    // without a reload. From the THIRD poll on, every /api/bot/audit answer
    // carries a fresh MORNING-TRADE row (2 s old) and every /api/bot/account
    // answer a higher equity, so the Now tile's latest-decision line, the
    // Book's pulse and the equity flash have something to react to while
    // the battery watches (polls run every 10 s; see the bot battery).
    const liveHits = {};
    const srv = createServer((req, res) => {
      const [u, qs] = (req.url || "/").split("?");
      if ((u === "/api/bot/audit" || u === "/api/bot/account") && PAGE_FIXTURES[u]) {
        liveHits[u] = (liveHits[u] || 0) + 1;
        const n = liveHits[u];
        if (n >= 3) {
          res.writeHead(200, { "content-type": "application/json" });
          const base = PAGE_FIXTURES[u];
          if (u === "/api/bot/audit") return res.end(JSON.stringify([{ time: new Date(Date.now() - 2000).toISOString(), type: "MORNING-TRADE", message: `KWEB: bought 120 @ 31.42 (score 75) — live fixture ${n}` }, ...base]));
          const eq = Number(base.equity) + 12.5 * (n - 2);
          const t = base.today;
          const today = !t ? undefined : t.phase === "between"
            // between sessions the change runs close to close: only equity now (the hover's drift) moves
            ? { ...t, equity_now: eq, asof: new Date().toISOString() }
            : todayFixture(eq, t.equity_prev_close, t.spy_pct_today, Date.now(), (min) => new Date(Date.now() - min * 60_000).toISOString(), t.spy_reason);
          return res.end(JSON.stringify({ ...base, equity: eq, ...(today ? { today } : {}) }));
        }
      }
      if (u.startsWith("/api/data/wxtile/")) {
        res.writeHead(200, { "content-type": "image/png" });
        return res.end(WX_TILE_PNG);
      }
      // TIME MACHINE v2 T-2 window fixture (2026-08-13): checked BEFORE the
      // generic FIXTURES prefix match below, which would otherwise treat
      // "/api/data/aircraft/window" as a sub-path of the "/api/data/aircraft"
      // key (u.startsWith(k + "/")) and hand TimeScrubber.tsx the raw
      // /api/data/aircraft point-list shape instead of a WindowResult — no
      // `.hexes`, no `.to`, a real crash the first live run of this harness
      // caught (`w.hexes is not iterable`). A dedicated small window payload
      // exercises the real hex/points/coverage shape the client reads.
      if (u.startsWith("/api/data/aircraft/window")) {
        const params = new URLSearchParams(qs || "");
        const to = parseInt(params.get("to") || "0", 10) || Math.floor(Date.now() / 1000);
        const from = parseInt(params.get("from") || String(to - 86400), 10) || to - 86400;
        const step = parseInt(params.get("step") || "300", 10) || 300;
        let seed = 7;
        const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
        const hexes = [];
        for (let i = 0; i < 40; i++) {
          const baseLat = 25 + rnd() * 24, baseLon = -125 + rnd() * 58;
          const n = 5 + Math.floor(rnd() * 8);
          const points = [];
          for (let k = 0; k < n; k++) {
            const t = from + Math.floor(((to - from) * k) / Math.max(1, n - 1));
            points.push([t, baseLat + k * 0.05, baseLon + k * 0.05, Math.round(rnd() * 11000)]);
          }
          hexes.push({ i: "fx" + i.toString(16), rg: "N" + i + "FX", c: "TST" + i, ty: "B738", points, raw_count: points.length, truncated: false });
        }
        // FLIGHT PROGRAM replay (2026-09-28): the server's close-approach
        // list (server/closeApproach.ts shape) so the panel's list renders,
        // and fx0/fx1 re-shaped into REAL converging 60 s tracks around the
        // listed instant so a focused approach draws actual curtains.
        const basis = "separation below 5 nm / 1,000 ft per our recorded ADS-B data — not an official loss-of-separation report";
        const tA = from + Math.floor((to - from) * 0.4);
        const conv = (dir, altFt) => {
          const pts = [];
          for (let s = -1800; s <= 1800; s += 60) {
            const nm = (450 / 3600) * s; // 450 kt through the crossing
            pts.push(dir === "east"
              ? [tA + s, 38.2, -104.1 + nm / (60 * Math.cos(38.2 * Math.PI / 180)), Math.round(altFt * 0.3048)]
              : [tA + s, 38.2 - 1.8 / 60 + nm / 60, -104.1, Math.round(altFt * 0.3048)]);
          }
          return pts;
        };
        hexes[0].points = conv("east", 34000); hexes[0].raw_count = hexes[0].points.length;
        hexes[1].points = conv("north", 34400); hexes[1].raw_count = hexes[1].points.length;
        const closeApproaches = [
          { a: "fx0", b: "fx1", ca: "TST0", cb: "TST1", t: tA * 1000, horizNm: 1.8, vertFt: 400, confidence: "high", basis, lat: 38.2, lon: -104.1, altAFt: 34000, altBFt: 34400 },
          { a: "fx2", b: "fx3", ca: "TST2", cb: "TST3", t: (from + Math.floor((to - from) * 0.7)) * 1000, horizNm: 4.3, vertFt: 900, confidence: "medium", basis, lat: 41.5, lon: -95.3, altAFt: 28000, altBFt: 28900 },
        ];
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({
          kind: "aircraft", from, to, zoom: 5, step_sec: step,
          hexes, hexes_seen: hexes.length,
          total_points: hexes.reduce((s, h) => s + h.points.length, 0),
          coverage: { requested_from: from, scanned_from: from, complete: true, files_scanned: 1 },
          closeApproaches,
          closeApproachesMeta: { method: "fixture", thresholds: { horiz_nm: 5, vert_ft: 1000, fix_window_sec: 90, low_alt_ft: 2000 }, evaluated_hexes: hexes.length, excluded_non_icao: 0, found: closeApproaches.length, returned: closeApproaches.length, capped: false, partial_scan: false },
        }));
      }
      // STATEFUL track fixture (trail-refresh ratchet, [REPAIR 2026-07-05]):
      // each call returns one MORE point so a live-refreshing trail visibly
      // grows across the popup's 30s interval; a static-snapshot regression
      // shows identical coordinates on every read.
      // 2026-07-20 (flight-track handoff): points now carry `al` altitudes
      // (climb profile) so the curtain + altitude line + ALTITUDE/TIME
      // profile panel actually render under the harness — an altitude-less
      // fixture exercised none of the 3D track path.
      if (u.startsWith("/api/data/track/")) {
        trackCalls++;
        const nowSec = Math.floor(Date.now() / 1000);
        const points = [];
        for (let k = 0; k <= 2 + trackCalls; k++) {
          points.push({
            t: nowSec - (2 + trackCalls - k) * 60,
            la: 36 + k * 0.05, lo: -97 + k * 0.05,
            al: 1500 + k * 400, // steady climb — exercises the ramp min→max
          });
        }
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({ kind: "aircraft", id: u.split("/").pop(), points, count: points.length }));
      }
      // /developers live-query widget (graph entity/hops "try it" input):
      // exercise the real ?entity= neighborhood path instead of always
      // falling through to the counts-only FIXTURES["/api/data/graph"]
      // shape, so the harness screenshot actually shows the widget's real
      // result state, not just its idle/counts state.
      if (u === "/api/data/graph" && (qs || "").includes("entity=")) {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({
          kind: "raw", built_at: 1, entity: "company:STLD", hops: 1,
          caveat: "RAW graph — every edge asserts a relationship with provenance (fixture).",
          nodes: [
            { id: "company:STLD", type: "company", label: "STLD", attrs: { parent: "Fixture Steel Co.", tickers: ["STLD"] } },
            { id: "facility:site:fixture_mill", type: "facility", label: "Fixture Steel Mill", attrs: { kind: "site", category: "steel_mill", lat: 41.37, lon: -84.9, operator: "Fixture Steel Co. (STLD)" } },
          ],
          edges: [{ type: "operates", from: "company:STLD", to: "facility:site:fixture_mill", confidence: "high", attrs: {} }],
        }));
      }
      // exact match wins before prefix match — otherwise /api/data/insider
      // shadows /api/data/insider/history
      const pfx = Object.keys(PAGE_FIXTURES).find((k) => u === k) ||
                  Object.keys(PAGE_FIXTURES).find((k) => u.startsWith(k + "/"));
      if (pfx) {
        // a fixture may be a function of the request, or carry its own
        // status/cookie ({ __status, __setCookie, __body }) — the expired-
        // session page answers 401 the way server/auth.ts does
        let body = PAGE_FIXTURES[pfx];
        if (typeof body === "function") body = body(req);
        const head = { "content-type": "application/json" };
        let status = 200;
        if (body && typeof body === "object" && "__status" in body) {
          status = body.__status;
          if (body.__setCookie) head["set-cookie"] = body.__setCookie;
          body = body.__body;
        }
        res.writeHead(status, head);
        return res.end(JSON.stringify(body));
      }
      const fx = Object.keys(FIXTURES).find((k) => u === k) ||
                 Object.keys(FIXTURES).find((k) => u.startsWith(k + "/"));
      if (fx) {
        res.writeHead(200, { "content-type": "application/json" });
        // Exercise the real ?since= delta path: an unchanged snapshot returns
        // {unchanged:true} with no payload — the client must skip setData.
        // (This also makes the perf window measure PAN smoothness, not
        // redundant 10k-feature re-uploads.)
        if (fx === "/api/data/aircraft" && (qs || "").includes("since=1")) {
          return res.end(JSON.stringify({ unchanged: true, time: 1, count: FIXTURES[fx].count }));
        }
        return res.end(JSON.stringify(FIXTURES[fx]));
      }
      if (u.startsWith("/api/")) {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end("{}");
      }
      let fp = path.join(DIST, u === "/" || u === "/app" ? "index.html" : u);
      if (!existsSync(fp)) fp = path.join(DIST, "index.html"); // SPA fallback
      try {
        const body = readFileSync(fp);
        res.writeHead(200, { "content-type": MIME[path.extname(fp)] || "application/octet-stream" });
        res.end(body);
      } catch {
        res.writeHead(404); res.end("nf");
      }
    });
    srv.listen(0, "127.0.0.1", () => resolve(srv));
  });
}

// ── mechanical checks (run in-page) ────────────────────────────────────────
const CHECKS_SNIPPET = (width, touch, mapPage = true) => `(() => {
  const out = { failures: [], warnings: [], info: {} };
  const vw = window.innerWidth, vh = window.innerHeight;

  // 0. the app actually mounted (found missing 2026-07-11: a blank white
  // screen — e.g. from the vite base:"./" multi-segment routing bug — has
  // no horizontal overflow and no interactive elements, so every check
  // below this one passed clean on a totally dead page. #root is React's
  // mount point (client/index.html); a script that never executed leaves
  // it permanently empty.
  // 0a. a table cell that is nothing but an unformatted empty value
  // (2026-11-19: the Treasury statement's Category column printed the word
  // "null" on production at every width; no check looked at cell text).
  {
    const bare = [...document.querySelectorAll('td, th')]
      .filter((c) => /^(null|undefined|NaN)$/.test((c.textContent || '').trim()));
    if (bare.length) out.failures.push(bare.length + ' table cell(s) show a bare unformatted value: [' + bare[0].textContent.trim() + '] in the row [' + ((bare[0].closest('tr') || bare[0]).textContent || '').trim().slice(0, 120) + ']');
  }

  const root = document.getElementById('root');
  if (!root || root.childElementCount === 0) {
    out.failures.push("app did not mount: #root has no children (blank page — script failed to load/execute)");
  } else if (root.querySelector('[data-vt-boot]')) {
    // 2026-07-20: index.html now ships an inline boot splash inside #root
    // (slow-connection fix), so "has children" no longer proves the app ran —
    // the splash still present at check time means the bundle never executed.
    out.failures.push("app did not mount: boot splash still present (script failed to load/execute)");
  }

  // 1. no horizontal overflow caused by the page
  if (document.documentElement.scrollWidth > vw + 1) {
    out.failures.push("horizontal overflow: scrollWidth " + document.documentElement.scrollWidth + " > viewport " + vw);
  }

  // 2. map root fills its viewport region (marker: [data-vt-map]) — map pages only
  const map = document.querySelector('[data-vt-map]');
  if (!${mapPage}) {
    // non-map page: no map assertions
  } else if (!map) {
    out.failures.push("[data-vt-map] marker missing — map root not found");
  } else {
    const r = map.getBoundingClientRect();
    out.info.map = { x: r.x, y: r.y, w: r.width, h: r.height };
    if (r.width < vw * 0.98) out.failures.push("map width " + Math.round(r.width) + " < 98% of viewport " + vw);
    if (Math.abs(vh - r.bottom) > 70) out.failures.push("map bottom " + Math.round(r.bottom) + " leaves >70px gap to viewport " + vh + " (letterboxing)");
    if (r.height < vh * 0.6) out.failures.push("map height " + Math.round(r.height) + " < 60% of viewport " + vh);
    // 3. no permanent overlay covering >40% of the map
    const mapArea = r.width * r.height;
    document.querySelectorAll('body *').forEach((el) => {
      if (el === map || map.contains(el) && el.tagName === 'CANVAS') return;
      const s = getComputedStyle(el);
      if ((s.position === 'absolute' || s.position === 'fixed') && s.visibility !== 'hidden' && s.display !== 'none') {
        const b = el.getBoundingClientRect();
        const ix = Math.max(0, Math.min(b.right, r.right) - Math.max(b.left, r.left));
        const iy = Math.max(0, Math.min(b.bottom, r.bottom) - Math.max(b.top, r.top));
        const cover = (ix * iy) / mapArea;
        const bg = s.backgroundColor;
        const opaque = bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent';
        if (cover > 0.4 && opaque && !el.closest('canvas')) {
          out.failures.push("overlay covers " + Math.round(cover * 100) + "% of map: <" + el.tagName.toLowerCase() + " class='" + (el.className || "").toString().slice(0, 40) + "'>");
        }
      }
    });
  }

  // 4. interactive elements: visible, unclipped, touch-sized on phone
  const MIN = ${touch} ? 44 : 24;
  document.querySelectorAll('button, [role="button"], input[type="checkbox"], a[href]').forEach((el) => {
    const b = el.getBoundingClientRect();
    if (b.width === 0 || b.height === 0) return; // hidden is fine
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none') return;
    if (b.right > vw + 1 || b.bottom > vh + 1 || b.left < -1 || b.top < -1) {
      // allow elements that scroll with content below the fold on non-immersive pages
      if (b.top < vh) out.warnings.push("clipped control: " + (el.getAttribute('aria-label') || el.textContent || el.tagName).toString().trim().slice(0, 30));
    }
    if (${touch} && b.width > 0 && (b.width < MIN || b.height < MIN)) {
      const label = (el.getAttribute('aria-label') || el.textContent || el.tagName).toString().trim().slice(0, 30);
      // checkbox inputs may be visually replaced by a larger styled parent
      const p = el.parentElement ? el.parentElement.getBoundingClientRect() : b;
      if (Math.max(p.width, b.width) < MIN || Math.max(p.height, b.height) < MIN) {
        out.warnings.push("touch target < ${44}px: '" + label + "' (" + Math.round(b.width) + "x" + Math.round(b.height) + ")");
      }
    }
  });
  return out;
})()`;

// ── main ───────────────────────────────────────────────────────────────────
// ── DEVICE ENVELOPE D1: per-layer demand matrix (scale_program.md) ─────────
// Boots the map with ALL layers off (the harness's own vt-layers-all-off
// escape), samples a baseline window, then toggles each measured layer ON
// alone, samples, toggles it back OFF — at each emulated CPU tier. Fixture
// data only (deterministic, diffable); SwiftShader numbers are relative
// regression signals, never on-device budgets.
const DEMAND_TIERS = [
  { key: "full", cpu: 1 },
  { key: "throttled4x", cpu: 4 }, // the weak-laptop stand-in (work-PC class)
];
const DEMAND_LAYERS = [
  // default-on set + the cost-budget battery's real toggleables — every id
  // verified wired via clickSwitchVerified; fixture-empty layers still
  // measure (renderedFeatures says how much data backed the number)
  "aircraft", "places", "powerplants", "trains", "sites",
  "terrain", "weather", "weather_temp", "weather_wind",
  "rivergauges", "alerts", "surfacewater", "forest", "boundaries",
];
const DEMAND_SAMPLE_MS = 3000;

async function demandSample(page) {
  return page.evaluate(async (windowMs) => {
    const map = window.__vtMap;
    const heap = () => {
      try { return performance.memory ? performance.memory.usedJSHeapSize / 1048576 : null; }
      catch { return null; }
    };
    const netKB = () => {
      try {
        return performance.getEntriesByType("resource")
          .reduce((s, e) => s + (e.transferSize || 0), 0) / 1024;
      } catch { return null; }
    };
    let longTasks = 0;
    let obs = null;
    try {
      obs = new PerformanceObserver((l) => { longTasks += l.getEntries().length; });
      obs.observe({ entryTypes: ["longtask"] });
    } catch {}
    // camera wiggle so the window measures RENDER demand, not an idle map
    try { map?.easeTo({ center: [-97.6, 38.4], zoom: map.getZoom(), duration: windowMs * 0.9 }); } catch {}
    const deltas = [];
    let last = performance.now();
    const t0 = last;
    await new Promise((res) => {
      const tick = (t) => {
        deltas.push(t - last);
        last = t;
        if (t - t0 >= windowMs) return res();
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    try { obs?.disconnect(); } catch {}
    deltas.sort((a, b) => a - b);
    const q = (p) => Math.round(deltas[Math.min(deltas.length - 1, Math.floor(deltas.length * p))] * 10) / 10;
    let rendered = null;
    try { rendered = map ? map.queryRenderedFeatures().length : null; } catch {}
    return {
      medianMs: deltas.length ? q(0.5) : null,
      p95Ms: deltas.length ? q(0.95) : null,
      heapMB: heap(), netKB: netKB(), longTasks, renderedFeatures: rendered,
    };
  }, DEMAND_SAMPLE_MS);
}

async function bootDemandPage(browser, port, cpu) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.route("**/*", (route) => {
    const u = route.request().url();
    if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
    return route.abort();
  });
  // all-off boot: the baseline is the empty map, same escape the
  // zero-cost battery uses
  await page.addInitScript(() => { try { sessionStorage.setItem("vt-layers-all-off", "1"); } catch {} });
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
  await page.goto(`http://127.0.0.1:${port}/app#/data`, { waitUntil: "load", timeout: 30000 });
  for (let i = 0; i < 60; i++) {
    if (await page.evaluate(() => !document.querySelector(".vt-map-skeleton"))) break;
    await page.waitForTimeout(250);
  }
  await page.waitForTimeout(2000);
  await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
  await page.waitForTimeout(300);
  for (let round = 0; round < 6; round++) {
    const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
    if (!(await btn.count())) break;
    await btn.click().catch(() => {});
    await page.waitForTimeout(100);
  }
  const flip = async (id, on) => {
    const sw = page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first();
    for (let attempt = 0; attempt < 3; attempt++) {
      const cur = await sw.getAttribute("aria-checked").catch(() => null);
      if (cur === String(on)) return true;
      await page.evaluate((lid) => {
        document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
      }, id).catch(() => {});
      await sw.click({ timeout: 3000 }).catch(() => {});
      for (let i = 0; i < 8; i++) {
        await page.waitForTimeout(150);
        if ((await sw.getAttribute("aria-checked").catch(() => null)) === String(on)) return true;
      }
    }
    return false;
  };
  return { ctx, page, flip };
}

async function runDemandMatrix(browser, port) {
  const { demandCell, buildLedger } = await import("./demandProfile.mjs");
  const version = JSON.parse(readFileSync(path.join(ROOT, "package.json"), "utf-8")).version;
  const cells = [];
  const skipped = [];
  for (const tier of DEMAND_TIERS) {
    // SELF-HEALING (first live run 2026-08-08: SwiftShader's renderer died
    // 4 layers in — the matrix meets the weak-machine failure mode it
    // measures): the browser context is rebooted on death and the sweep
    // continues from the next layer; the death itself is recorded per
    // layer, which is demand data in its own right.
    let sess = await bootDemandPage(browser, port, tier.cpu);
    let baseline = await demandSample(sess.page);
    console.log(`[demand] tier=${tier.key} baseline median=${baseline.medianMs}ms p95=${baseline.p95Ms}ms heap=${baseline.heapMB?.toFixed?.(0) ?? "n/a"}MB`);
    for (const id of DEMAND_LAYERS) {
      try {
        if (!(await sess.flip(id, true))) {
          skipped.push(`${id}@${tier.key}: toggle never landed`);
          console.log(`[demand] SKIP ${id} @ ${tier.key} — toggle not wired/visible`);
          continue;
        }
        await sess.page.waitForTimeout(1500); // settle: fetch + first paint
        const s = await demandSample(sess.page);
        cells.push(demandCell(id, tier.key, baseline, s));
        console.log(`[demand] ${id} @ ${tier.key}: median=${s.medianMs}ms (base ${baseline.medianMs}) p95=${s.p95Ms}ms heapΔ=${s.heapMB != null && baseline.heapMB != null ? (s.heapMB - baseline.heapMB).toFixed(1) : "n/a"}MB feats=${s.renderedFeatures}`);
        if (!(await sess.flip(id, false))) skipped.push(`${id}@${tier.key}: restore-to-off failed — later cells may be contaminated`);
        await sess.page.waitForTimeout(600);
      } catch (e) {
        const msg = String(e?.message || e).split("\n")[0];
        skipped.push(`${id}@${tier.key}: browser died mid-cycle (${msg}) — rebooted, continued`);
        console.log(`[demand] DIED at ${id} @ ${tier.key}: ${msg} — rebooting context`);
        try { await sess.ctx.close(); } catch {}
        sess = await bootDemandPage(browser, port, tier.cpu);
        baseline = await demandSample(sess.page);
        console.log(`[demand] tier=${tier.key} re-baseline median=${baseline.medianMs}ms heap=${baseline.heapMB?.toFixed?.(0) ?? "n/a"}MB`);
      }
    }
    try { await sess.ctx.close(); } catch {}
  }
  const ledger = buildLedger(version, cells, {
    layers: DEMAND_LAYERS, skipped,
    sampleSeconds: DEMAND_SAMPLE_MS / 1000,
    tiers: DEMAND_TIERS.map((t) => t.key),
    renderer: "SwiftShader (headless) — relative regression signal only",
  });
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, "demand_profile.json"), JSON.stringify(ledger, null, 1));
  console.log(`\n[demand] ledger written: .visual/demand_profile.json (${cells.length} cells, ${skipped.length} skipped)`);
  if (skipped.length) console.log("[demand] bounded coverage — skipped: " + skipped.join("; "));
  const top = ledger.cells.slice(0, 5).map((c) => `${c.layer}@${c.tier} +${(c.medianMs - c.baselineMedianMs).toFixed(1)}ms`).join(", ");
  console.log("[demand] most demanding: " + top);
}

async function main() {
  if (!existsSync(DIST)) {
    console.error("dist/public missing — run `npm run build` first");
    process.exit(1);
  }
  const { chromium } = await import("playwright");
  const exePath = "/opt/pw-browsers/chromium";
  // Software WebGL (SwiftShader) — maplibre-gl requires a GL context, and
  // headless containers have no GPU. Without these flags the map never
  // fires "load" and every screenshot is a skeleton.
  const GL_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ];
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: GL_ARGS });
  } catch {
    browser = await chromium.launch({
      headless: true,
      executablePath: existsSync(exePath) ? exePath : undefined,
      args: GL_ARGS,
    });
  }
  const srv = await startServer();
  const port = srv.address().port;
  mkdirSync(OUT, { recursive: true });

  if (DEMAND) {
    // measurement mode: matrix only, never gates, exit 0 unless it crashed
    try { await runDemandMatrix(browser, port); }
    finally { await browser.close(); srv.close(); }
    return;
  }

  const results = [];
  for (const [name, cfg] of Object.entries(PAGES)) {
    const route = cfg.route;
    if (only && only !== name) continue;
    PAGE_FIXTURES = cfg.fixtures || {};
    for (const vp of (cfg.widths ? [...WIDTHS, ...cfg.widths] : WIDTHS)) {
      const ctx = await browser.newContext({
        viewport: { width: vp.w, height: vp.h },
        hasTouch: vp.touch,
        isMobile: vp.touch && vp.w < 640,
        // dsf 1: SwiftShader rasterizes in software — dsf 2 quadruples the
        // pixel surface at 1440 and turns the perf guard into a pixel-count
        // test instead of a feature-count test. Real-DPR acceptance is the
        // human's S24 (real GPU).
        deviceScaleFactor: 1,
      });
      const page = await ctx.newPage();
      // Full determinism: only the fixture server is reachable. External
      // requests (tile CDN, fonts) abort instantly — layouts and overlays are
      // what we verify, and aborted tiles let maplibre settle immediately
      // instead of hanging on proxy connection-resets.
      await page.route("**/*", (route) => {
        const u = route.request().url();
        if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
        return route.abort();
      });
      const errors = [];
      page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
      // every /api/* request this render makes, in order (path + query): the Trade cockpit's checks read what a
      // look-up actually fetches (2026-11-06)
      const apiLog = [];
      page.on("request", (r) => { const q = new URL(r.url()); if (q.pathname.startsWith("/api/")) apiLog.push(q.pathname + q.search); });
      const t0 = Date.now();
      await page.goto(`http://127.0.0.1:${port}${route}`, { waitUntil: "load", timeout: 30000 });
      // TTI: skeleton gone = page interactive (DESIGN.md: <3s on device;
      // headless SwiftShader is far slower — hard regression guard at 12s).
      let tti = null;
      for (let i = 0; i < 60; i++) {
        const gone = await page.evaluate(() => !document.querySelector(".vt-map-skeleton"));
        if (gone) { tti = Date.now() - t0; break; }
        await page.waitForTimeout(250);
      }
      await page.waitForTimeout(2500); // let overlay layers mount
      // PERF BUDGET: drive pans through the __vtMap hook while sampling rAF
      // frame deltas. Software-GL thresholds are regression guards, not the
      // on-device budget (that's DESIGN.md's number).
      // BEST-OF-3 (PROGRAM_STATE.md Q25, 2026-08-28): a single measured pass
      // sits inside its own noise band — two runs of the IDENTICAL commit
      // were observed to fail at DIFFERENT widths (768 median 217>200, then
      // 1440 p95 367>350), and prior p95 on this page measured 283/317/383/
      // 467ms across otherwise-unchanged runs. That spread comes from
      // environmental hitches (GC pause, host contention under SwiftShader)
      // landing inside one ~2.6s measured window, not from the code. Per
      // Q25's own instruction ("measure the spread, set thresholds outside
      // it (or take best-of-N) — do NOT simply raise the numbers"): the gate
      // now runs THREE independent measured passes and takes the MEDIAN of
      // the three per-pass medians/p95s. A single unlucky pass no longer
      // decides the verdict; a regression that holds across all three still
      // fails the gate exactly as before. Thresholds are unchanged — this
      // narrows what they're being compared against, not the bar itself.
      const perf = !cfg.map ? {} : await page.evaluate(async () => {
        const map = window.__vtMap;
        if (!map) return { error: "__vtMap hook missing" };
        const c = map.getCenter();
        const runPans = async (record) => {
          const deltas = [];
          let last = performance.now();
          let sampling = true;
          const tick = (t) => { deltas.push(t - last); last = t; if (sampling) requestAnimationFrame(tick); };
          if (record) requestAnimationFrame(tick);
          for (const [dx, dy] of [[8, 3], [-14, -5], [10, 4], [-4, -2]]) {
            map.easeTo({ center: [c.lng + dx, c.lat + dy], duration: 600 });
            await new Promise(r => setTimeout(r, 650));
          }
          sampling = false;
          await new Promise(r => setTimeout(r, 60));
          return deltas.filter(d => d > 0).sort((a, b) => a - b);
        };
        const q = (sorted, f) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * f))] || 0;
        const medianOf3 = (arr) => arr.slice().sort((a, b) => a - b)[1];
        await runPans(false);            // warm-up: first-pan upload hitches
        const passes = [];
        for (let i = 0; i < 3; i++) {
          const sorted = await runPans(true);   // measured window (warm)
          passes.push({ frames: sorted.length, median: Math.round(q(sorted, 0.5)),
                        p95: Math.round(q(sorted, 0.95)), max: Math.round(sorted[sorted.length - 1] || 0) });
        }
        let rendered = 0;
        // decimation split (perf 3/3): full layer above z4.5, rank-filtered
        // twin below — sample whichever is active at the current zoom
        try { rendered = map.queryRenderedFeatures({ layers: ["aircraft-sym", "aircraft-sym-lo"].filter((l) => map.getLayer(l)) }).length; } catch {}
        return { frames: passes.reduce((s, p) => s + p.frames, 0),
                 median: medianOf3(passes.map((p) => p.median)),
                 p95: medianOf3(passes.map((p) => p.p95)),
                 max: Math.max(...passes.map((p) => p.max)),
                 passes, renderedAircraft: rendered };
      });
      const shot = path.join(OUT, `${name}-${vp.w}.png`);
      await page.screenshot({ path: shot });
      const checks = await page.evaluate(CHECKS_SNIPPET(vp.w, vp.touch, cfg.map));
      checks.failures.push(...errors);

      // ── BOT PAGE CONTROLS (2026-10-20, field report: "dock and zoom don't
      // work" on a panel opened as a window). Every panel control is driven
      // with real pointer events (locator.click / tap) and its effect read
      // back from the DOM: hold, pin, open-as-window, the window's dock /
      // zoom / close buttons, the ghost slot, Esc, title-bar drag, the
      // zoomed view's way back. Runs on both bot renders at every width.
      // ── POP-OUT PANE (slice D, 2026-10-21): /app#/bot/pane/chart is the
      // chart ALONE in its own browser window — no shell nav, the panel fills
      // the window, the chart paints there, a "?" opens its card inside THIS
      // window (the directive: "it should function even if it's detached"),
      // "Back to the page" is the one way-back control and pin is gone.
      // ── EXPIRED SESSION (field bug 2026-10-28): a 401 from an app call
      // opens the sign-in view with the expiry notice; the forecast page's
      // outage banner (and the raw 401 text) must never show.
      // ── FILING HISTORY COVERAGE (2026-11-15): on a truncated history page
      // the header states the archive days loaded (never "last N days"), and
      // the footer control walks the server's `next` until the window is whole.
      if (name === "filingspaged" || name === "filings13fpaged" || name === "earningspaged") {
        // the earnings page lists release cards, the two filing pages table rows
        const rowCount = () => name === "earningspaged"
          ? page.locator(".vt-earnings-list > li").count()
          : page.locator(".vt-filings-table").first().locator("tbody tr:not(.vt-filings-dayrow):not(.vt-filings-morerow)").count();
        // DAYS OPEN AND CLOSE (2026-11-18): a filed day's rows are on the page once its line is opened
        const openClosedDays = async () => {
          const closed = page.locator('.vt-filings-dayrow[data-open="0"] .vt-filings-daybtn:not([disabled])');
          for (let n = await closed.count(); n > 0; n = await closed.count()) { await closed.first().click(); await page.waitForTimeout(60); }
        };
        const countWord = name === "earningspaged" ? /\b3 releases\b/ : /\b3 filings\b/;
        const fail = (m) => checks.failures.push("filing-coverage: " + m);
        // the Form 4 page reads by filed day (2026-11-18) and so does the 13F page (2026-11-21); the earnings page by archive day
        const byFiled = name === "filingspaged" || name === "filings13fpaged";
        const is13f = name === "filings13fpaged";
        const want = is13f
          ? { before: /2026-11-16 → 2026-11-18 · 3 of the 45 days asked for/, after: /2026-10-05 → 2026-11-18 · all 45 days/, next: /history\?filed_from=2026-10-05&filed_to=2026-11-13$/ }
          : byFiled
          ? { before: /2026-11-16 → 2026-11-18 · 3 of the 30 days asked for/, after: /2026-10-20 → 2026-11-18 · all 30 days/, next: /history\?filed_from=2026-10-20&filed_to=2026-11-13$/ }
          : { before: /2026-11-14 → 2026-11-15 · 2 of the 30 days asked for/, after: /2026-10-17 → 2026-11-15 · all 30 days/, next: /history\?from=2026-10-17&to=2026-11-13$/ };
        const cov = page.locator("[data-vt-coverage]");
        const before = (await cov.count()) ? await cov.first().innerText() : "";
        if (!want.before.test(before)) fail(`header does not state the loaded days ("${before}")`);
        // FILED DAYS (2026-11-18): each filed day is a line above its rows saying how it compares with EDGAR's index
        const dayLines = async () => page.locator(".vt-filings-dayrow").evaluateAll((els) => els.map((el) => ({
          day: el.getAttribute("data-vt-filed-day"), state: el.getAttribute("data-state"), text: (el.textContent || "").replace(/\s+/g, " ").trim(),
          left: el.getBoundingClientRect().left, right: el.getBoundingClientRect().right,
        })));
        if (byFiled) {
          if (!apiLog.some((u) => (is13f ? /filings13f\/history\?filed_from=\d{4}-\d{2}-\d{2}$/ : /insider\/history\?filed_from=\d{4}-\d{2}-\d{2}$/).test(u))) fail(`the first page was not asked for by filed day (${apiLog.filter((u) => u.includes("history")).join(" , ")})`);
          if (!/\(days filed\)/.test(await page.locator(".vt-filings-sub").first().innerText())) fail("header does not say its days are filed days");
          const sum = page.locator("[data-vt-filed-summary]");
          const sumText = (await sum.count()) ? await sum.first().innerText() : "";
          if (sumText !== "1 of 3 filed days complete against EDGAR's index · 1 partial · 1 not yet checked") fail(`filed-day summary reads "${sumText}"`);
          const lines = await dayLines();
          // a Form 4 day holds transaction rows beside its filings; a 13F row IS a filing, so its line counts once
          const expect = [
            ["2026-11-18", "sample", "Filed 2026-11-18 · 1 filing · sample · not yet checked against EDGAR's index · 1 row"],
            ["2026-11-17", "complete", "Filed 2026-11-17 · 1 filing · complete · all 1 on EDGAR's index · 1 row"],
            ["2026-11-16", "partial", "Filed 2026-11-16 · 1 filing · partial · 1 of the 624 on EDGAR's index · 1 row"],
          ].map(([d, st, t]) => [d, st, is13f ? t.replace(/ · 1 row$/, "") : t]);
          // only the newest day starts open; the other two open from their own line
          const openNow = await page.locator(".vt-filings-dayrow").evaluateAll((els) => els.map((el) => el.getAttribute("data-open")).join(""));
          if (openNow !== "100") fail(`expected only the newest filed day open at first, got "${openNow}"`);
          if ((await rowCount()) !== 1) fail(`expected 1 row on screen before a day is opened, found ${await rowCount()}`);
          await openClosedDays();
          if (lines.length !== 3) fail(`expected 3 filed-day lines, found ${lines.length}`);
          expect.forEach(([day, state, text], i) => {
            const l = lines[i];
            if (!l || l.day !== day || l.state !== state || l.text !== text) fail(`filed-day line ${i} is ${JSON.stringify(l)}, expected ${day} / ${state} / "${text}"`);
            else if (l.left < -0.5 || l.right > vp.w + 0.5) fail(`filed-day line ${day} overflows the viewport (${Math.round(l.left)}..${Math.round(l.right)} of ${vp.w})`);
          });
        } else if (await page.locator(".vt-filings-dayrow").count()) fail("an archive-day page shows filed-day lines");
        const head = await page.locator(".vt-filings-sub").first().innerText();
        if (/last \d+ days/.test(head)) fail(`header still claims "last N days": ${head}`);
        if (!countWord.test(head)) fail(`header filing count is not the loaded count: ${head}`);
        const btn = page.locator("[data-vt-load-older]");
        if (!(await btn.count()) || !(await btn.first().isVisible())) fail("no 'load older days' control on a truncated page");
        else {
          const box = await btn.first().boundingBox();
          if (box && (box.x < 0 || box.x + box.width > vp.w + 0.5)) fail(`load-older control overflows the viewport (${Math.round(box.x)}..${Math.round(box.x + box.width)} of ${vp.w})`);
          if (box && vp.touch && box.height < 44) fail(`load-older control is ${Math.round(box.height)}px tall on touch (<44)`);
          const rows0 = await rowCount();
          await btn.first().click();
          await page.waitForTimeout(600);
          const after = (await cov.count()) ? await cov.first().innerText() : "";
          if (!want.after.test(after)) fail(`after loading older days the header reads "${after}"`);
          if (await page.locator("[data-vt-load-older]").count()) fail("load-older control still shown after the window is whole");
          if (byFiled) {
            // older days arrive closed (their lines are the index); the days already open stay open
            const openAfter = await page.locator(".vt-filings-dayrow").evaluateAll((els) => els.map((el) => el.getAttribute("data-open")).join(""));
            if (openAfter !== "111000") fail(`after loading older days the open days are "${openAfter}", expected the three loaded first`);
            await openClosedDays();
          }
          const rows1 = await rowCount();
          if (rows1 !== rows0 + 3) fail(`older rows not appended (${rows0} -> ${rows1}, expected +3)`);
          if (byFiled) {
            const got = (await dayLines()).map((l) => `${l.day}:${l.state}`).join(" ");
            const all = "2026-11-18:sample 2026-11-17:complete 2026-11-16:partial 2026-11-13:complete 2026-11-06:partial 2026-10-20:sample";
            if (got !== all) fail(`after loading older days the filed-day lines are "${got}"`);
            const sumAfter = await page.locator("[data-vt-filed-summary]").first().innerText().catch(() => "");
            if (sumAfter !== "2 of 6 filed days complete against EDGAR's index · 2 partial · 2 not yet checked") fail(`after loading older days the summary reads "${sumAfter}"`);
          }
          if (!apiLog.some((u) => want.next.test(u))) fail(`the older page was not fetched through the server's next query (${apiLog.filter((u) => u.includes("history")).join(" , ")})`);
          await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-older.png`) });
        }
      }
      // ── FILED DAYS AT REAL VOLUME (2026-11-18, item (l)): 4,397 rows in
      // three filed days. Every day is a line; one day's first page of rows
      // is on screen; the rest is a tap away and no filter is needed.
      if (name === "filingsvolume") {
        const fail = (m) => checks.failures.push("filing-days: " + m);
        const rowCount = () => page.locator(".vt-filings-table tbody tr:not(.vt-filings-dayrow):not(.vt-filings-morerow)").count();
        const lines = () => page.locator(".vt-filings-dayrow").evaluateAll((els) => els.map((el) => ({
          day: el.getAttribute("data-vt-filed-day"), open: el.getAttribute("data-open"),
          rows: (el.querySelector("[data-vt-day-rows]")?.textContent || "").replace(/\s+/g, " ").trim(),
          expanded: el.querySelector(".vt-filings-daybtn")?.getAttribute("aria-expanded"),
          left: el.getBoundingClientRect().left, right: el.getBoundingClientRect().right, h: el.getBoundingClientRect().height,
        })));
        const l0 = await lines();
        const got0 = l0.map((l) => `${l.day}:${l.open}:${l.rows}`).join(" | ");
        if (got0 !== "2026-11-18:1:· 264 rows | 2026-11-17:0:· 2,800 rows | 2026-11-16:0:· 1,333 rows") fail(`day lines read "${got0}"`);
        if (l0.some((l) => l.expanded !== (l.open === "1" ? "true" : "false"))) fail("a day line's aria-expanded disagrees with its state");
        for (const l of l0) {
          if (l.left < -0.5 || l.right > vp.w + 0.5) fail(`day line ${l.day} overflows the viewport (${Math.round(l.left)}..${Math.round(l.right)} of ${vp.w})`);
          if (vp.touch && l.h < 44) fail(`day line ${l.day} is ${Math.round(l.h)}px tall on touch (<44)`);
        }
        if ((await rowCount()) !== 25) fail(`expected the newest day's first 25 rows on screen, found ${await rowCount()}`);
        if (/showing first/.test(await page.locator(".vt-filings-tablewrap").first().innerText())) fail("the 500-row cap message is back on a filed-day page");
        // item (l)'s check: the list is under 20 screens before anything is opened, and every day's line is in it
        const listH = await page.locator(".vt-filings-table").first().evaluate((el) => el.getBoundingClientRect().height);
        if (listH > 20 * vp.h) fail(`the unopened list is ${Math.round(listH)}px, over 20 screens of ${vp.h}px`);
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-first.png`) });
        // "show more" adds a page of the open day
        const more = page.locator('[data-vt-day-more="2026-11-18"]');
        const moreText = (await more.count()) ? (await more.first().innerText()).replace(/\s+/g, " ").trim() : "";
        if (moreText !== "Show 200 more · 25 of 264 rows shown") fail(`show-more control reads "${moreText}"`);
        else {
          const box = await more.first().boundingBox();
          if (box && (box.x < 0 || box.x + box.width > vp.w + 0.5)) fail(`show-more control overflows the viewport (${Math.round(box.x)}..${Math.round(box.x + box.width)} of ${vp.w})`);
          if (box && vp.touch && box.height < 44) fail(`show-more control is ${Math.round(box.height)}px tall on touch (<44)`);
          await more.first().click();
          await page.waitForTimeout(150);
          if ((await rowCount()) !== 225) fail(`after "show more" expected 225 rows, found ${await rowCount()}`);
          const again = (await more.first().innerText().catch(() => "")).replace(/\s+/g, " ").trim();
          if (again !== "Show the last 39 · 225 of 264 rows shown") fail(`second show-more reads "${again}"`);
        }
        // closing the open day and opening the oldest: the day the old cap never reached
        await page.locator('.vt-filings-dayrow[data-vt-filed-day="2026-11-18"] .vt-filings-daybtn').click();
        await page.locator('.vt-filings-dayrow[data-vt-filed-day="2026-11-16"] .vt-filings-daybtn').click();
        await page.waitForTimeout(150);
        const l1 = (await lines()).map((l) => `${l.day}:${l.open}`).join(" ");
        if (l1 !== "2026-11-18:0 2026-11-17:0 2026-11-16:1") fail(`after closing the newest day and opening the oldest the lines are "${l1}"`);
        if ((await rowCount()) !== 25) fail(`the oldest day opened with ${await rowCount()} rows, expected its first 25`);
        // a filter keeps every day's line and says what matched
        await page.locator('.vt-filings-filter[role="tab"]', { hasText: "Buys" }).first().click();
        await page.waitForTimeout(200);
        const l2 = await lines();
        if (l2.length !== 3) fail(`a filter left ${l2.length} day lines, expected 3`);
        if (l2.some((l) => !/matching rows?$|no rows match the filter$/.test(l.rows))) fail(`under a filter the day lines read "${l2.map((l) => l.rows).join(" | ")}"`);
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-opened.png`) });
      }
      // ── UI COVERAGE BATCH C (2026-11-20): the history charts on three
      // panels and the two new panels. Each check reads the rendered words
      // against the fixture: a missing reading is a stated gap, never a zero.
      if (["cropconditions", "appstorerankings", "githubactivity"].includes(name)) {
        const fail = (m) => checks.failures.push("history charts: " + m);
        await page.waitForSelector("[data-vt-hist]", { timeout: 8000 }).catch(() => {});
        const cards = await page.$$eval("[data-vt-hist]", (els) => els.map((e) => ({
          title: e.querySelector(".vt-hist-title")?.textContent || "",
          last: e.querySelector("[data-vt-hist-last]")?.textContent || "",
          count: e.querySelector("[data-vt-hist-count]")?.textContent || "",
          paths: e.querySelectorAll("svg path").length,
          empty: !!e.querySelector("[data-vt-hist-empty]"),
          w: e.getBoundingClientRect().width, sw: e.scrollWidth, cw: e.clientWidth,
        })));
        const want = {
          cropconditions: [
            { title: "Corn", last: "56% · 2026-10-18", count: "5 weeks", paths: 1 },
            { title: "Soybeans", last: "58% · 2026-10-04", count: "3 weeks · 2 with no reading, left as gaps", paths: 1 },
          ],
          appstorerankings: [
            { title: "Chart slots with a watchlist app ranked", last: "38% · 2026-08-05", count: "4 days · 1 with no reading, left as gaps", paths: 2 },
            { title: "Total US rating count, all watchlist apps", last: "159.7M · 2026-08-05", count: "4 days · 1 with no reading, left as gaps", paths: 2 },
          ],
          githubactivity: [
            { title: "Merged pull requests", last: "5,240 · 2026-07-27", count: "4 weeks", paths: 1 },
            { title: "Commits", last: "6,010 · 2026-07-27", count: "3 weeks · 1 with no reading, left as gaps", paths: 1 },
          ],
        }[name];
        if (cards.length !== want.length) fail(`${cards.length} charts rendered, expected ${want.length}`);
        want.forEach((w, i) => {
          const c = cards[i]; if (!c) return;
          for (const k of ["title", "last", "count", "paths"]) if (c[k] !== w[k]) fail(`chart ${i + 1} ${k} reads "${c[k]}", expected "${w[k]}"`);
          if (c.empty) fail(`chart ${i + 1} shows the not-enough-history sentence`);
          if (c.sw > c.cw + 1) fail(`chart ${i + 1} overflows its card (${c.sw} > ${c.cw})`);
        });
        if (name === "githubactivity") {
          const note = await page.$$eval(".vt-hist-note", (els) => els.map((e) => e.textContent));
          if (!note.every((t) => /15 in the newest\); 1 week in this window had a different count/.test(t || ""))) fail(`coverage note reads "${note[0]}"`);
        }
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-history.png`), fullPage: true });
      }
      if (name === "plantsunderalerts") {
        const fail = (m) => checks.failures.push("plants under alerts: " + m);
        await page.waitForSelector("[data-vt-pua-alert]", { timeout: 8000 }).catch(() => {});
        const blocks = page.locator("[data-vt-pua-alert]");
        if ((await blocks.count()) !== 2) fail(`${await blocks.count()} warnings rendered, expected 2`);
        else {
          const age = await page.locator("[data-vt-pua-age]").innerText();
          if (!/^warnings read [45] min ago$/.test(age)) fail(`age reads "${age}"`);
          const scope = await page.locator("[data-vt-pua-scope]").innerText();
          if (!/10 warnings with a polygon checked against 14,428 US plants · 2 cover at least one plant · 421 alerts are issued for a forecast zone with no polygon and could not be checked/.test(scope)) fail(`scope line reads "${scope}"`);
          const sum = await blocks.nth(0).locator("[data-vt-pua-sum]").innerText();
          if (sum !== "10 plants inside the polygon · 3,341 MW nameplate · gas 3,241 MW · solar 85.7 MW · other 14.4 MW") fail(`first summary reads "${sum}"`);
          const meta = await blocks.nth(0).locator(".vt-pua-alert-meta").innerText();
          if (!/^Severe · ends in 1 h 3[45] min$/.test(meta)) fail(`first warning's end reads "${meta}"`);
          const meta2 = await blocks.nth(1).locator(".vt-pua-alert-meta").innerText();
          if (meta2 !== "Severe · no end time stated") fail(`second warning's end reads "${meta2}"`);
          const rows = () => blocks.nth(0).locator("tbody tr").count();
          if ((await rows()) !== 8) fail(`first warning opens on ${await rows()} plants, expected 8`);
          const more = blocks.nth(0).locator("[data-vt-pua-more]");
          if ((await more.innerText()) !== "Show the other 2 plants (smaller)") fail(`more control reads "${await more.innerText()}"`);
          const box = await more.boundingBox();
          if (box && box.height < 44) fail(`more control is ${box.height}px tall`);
          await more.click();
          if ((await rows()) !== 10) fail(`after opening, ${await rows()} plants, expected 10`);
          const lastCap = await blocks.nth(0).locator("tbody tr").nth(9).locator("td").nth(3).innerText();
          if (!/not stated/.test(lastCap)) fail(`a plant with no stated capacity reads "${lastCap}"`);
          const wraps = await page.$$eval(".vt-tablewrap-flow", (els) => els.map((e) => ({ sh: e.scrollHeight, ch: e.clientHeight })));
          if (wraps.some((w) => w.sh > w.ch + 1)) fail(`a plant table is clipped vertically: ${JSON.stringify(wraps)}`);
          if (vp.w < 640) {
            const lbl = await blocks.nth(0).locator("tbody tr td:last-child").first().evaluate((td) => getComputedStyle(td, "::before").content);
            if (!/Nameplate capacity/.test(lbl)) fail(`on a phone the capacity cell's label reads ${lbl}`);
          }
          if ((await blocks.nth(1).locator("[data-vt-pua-more]").count()) !== 0) fail("a two-plant warning offers a more control");
          await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-opened.png`), fullPage: true });
        }
      }
      if (name === "quality") {
        const fail = (m) => checks.failures.push("data quality, archive checks: " + m);
        await page.waitForSelector("[data-vt-quality-checks] li", { timeout: 8000 }).catch(() => {});
        const lines = await page.$$eval("[data-vt-quality-checks] li", (els) => els.map((e) => e.textContent));
        const want = [
          /^Checked against NASA's record by date 2 h 1[01] min ago\.$/,
          /^Over 1 finished day the live recording had missed 281 of the 75,878 detections NASA holds for those dates \(0\.37%\); they were added\.$/,
          /^2 days with no recording \(2026-11-03 to 2026-11-19\) were refilled afterwards from NASA's answer for the date: 134,100 detections\. They are NASA's record as of the refill, not a live recording\.$/,
          /^1 day \(2026-10-27\) stays empty: NASA returned nothing for the date from any satellite\.$/,
        ];
        if (lines.length !== want.length) fail(`${lines.length} sentences, expected ${want.length}: ${JSON.stringify(lines)}`);
        else want.forEach((re, i) => { if (!re.test(lines[i] || "")) fail(`sentence ${i + 1} reads "${lines[i]}"`); });
        const over = await page.$eval("[data-vt-quality-checks]", (el) => el.scrollWidth > el.clientWidth + 1).catch(() => false);
        if (over) fail("the section scrolls sideways");
        // archive files + off-site copy (batch C, second part)
        const fa = (m) => checks.failures.push("data quality, archive files: " + m);
        await page.waitForSelector("[data-vt-quality-files] tbody tr", { timeout: 8000 }).catch(() => {});
        const rowsOf = () => page.$$eval("[data-vt-quality-files] tbody tr", (trs) => trs.map((tr) => [...tr.querySelectorAll("td")].map((td) => td.textContent)));
        let ar = await rowsOf();
        if (ar.length !== 12) fa(`${ar.length} archives listed before "show all", expected 12`);
        else {
          if (ar[0][0] !== "dtccswaps" || ar[0][2] !== "514.6 MB" || !/\(today\)$/.test(ar[0][4])) fa(`the largest archive reads "${ar[0].join(" | ")}"`);
          const fires = ar.find((r) => r[0] === "fires");
          if (!fires || !/\(18 d ago\)$/.test(fires[4]) || !/^116 day files kept for good, 2026-07-04 to 2026-11-02$/.test(fires[5])) fa(`the archive whose newest file is 18 days old reads "${(fires || []).join(" | ")}"`);
          const oc = ar.find((r) => r[0] === "optionchains");
          if (!oc || oc[3] !== "—") fa(`a marker file is shown as a first file: "${(oc || []).join(" | ")}"`);
          const ac = ar.find((r) => r[0] === "aircraft");
          if (!ac || ac[5] !== "690 files in the replay window, 2026-10-21 to 2026-11-18") fa(`the replay-window copy reads "${(ac || [])[5]}"`);
        }
        const moreA = page.locator("[data-vt-archive-more]");
        if ((await moreA.innerText().catch(() => "")) !== "Show all 14 archives") fa(`the show-all control reads "${await moreA.innerText().catch(() => "")}"`);
        const mb = await moreA.boundingBox().catch(() => null);
        if (mb && mb.height < 44) fa(`the show-all control is ${mb.height}px tall`);
        await moreA.click().catch(() => {});
        ar = await rowsOf();
        if (ar.length !== 14) fa(`${ar.length} archives after "show all", expected 14`);
        else {
          const ahead = ar.find((r) => r[0] === "eudayaheadprices");
          if (!ahead || !/\(dated ahead\)$/.test(ahead[4])) fa(`a day-ahead file reads "${(ahead || [])[4]}"`);
          const last = ar[13];
          if (last[0] !== "aircraft_tracks" || last[1] !== "0" || last[2] !== "—" || last[4] !== "—" || last[5] !== "119 day files kept for good, 2026-07-03 to 2026-10-29") fa(`an archive moved off-site whole reads "${last.join(" | ")}"`);
          const tracked = ar.find((r) => r[0] === "aircraft_tracked");
          if (!tracked || tracked[4] !== "2026-11") fa(`a per-month file reads "${(tracked || [])[4]}"`);
        }
        const wrapsA = await page.$$eval("[data-vt-quality-files] .vt-tablewrap-flow", (els) => els.map((e) => ({ sh: e.scrollHeight, ch: e.clientHeight })));
        if (wrapsA.some((w) => w.sh > w.ch + 1)) fa(`the table is clipped vertically: ${JSON.stringify(wrapsA)}`);
        if (vp.w < 640) {
          const lbl = await page.$eval("[data-vt-quality-files] tbody tr td:last-child", (td) => getComputedStyle(td, "::before").content).catch(() => "");
          if (!/Off-site/.test(lbl)) fa(`on a phone the off-site cell's label reads ${lbl}`);
        }
        const off = await page.$$eval("[data-vt-quality-offsite] li", (els) => els.map((e) => e.textContent));
        const wantOff = [
          /^Last copy run (9|10) min ago: 2 files copied, 0 failed, 0 waiting\.$/,
          /^Held off-site: 3\.2 GB in 2,583 files\. 1\.5 GB is day files that are never deleted; 1\.7 GB is the last 30 days of hourly position logs, replayable from 2026-10-21\.$/,
          /^Compared with the off-site listing 3 h (39|40) min ago: 0 files differ from the copy, 0 missing off-site\.$/,
          /^The volume has 1\.9 GB free; the server aims to keep 2\.1 GB free\. It is under that target, so days already verified off-site are removed from the volume first\.$/,
        ];
        if (off.length !== wantOff.length) fa(`${off.length} off-site sentences, expected ${wantOff.length}: ${JSON.stringify(off)}`);
        else wantOff.forEach((re, i) => { if (!re.test(off[i] || "")) fa(`off-site sentence ${i + 1} reads "${off[i]}"`); });
        await page.locator("[data-vt-quality-files]").scrollIntoViewIfNeeded().catch(() => {});
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-files.png`) });
        await page.locator("[data-vt-quality-offsite]").scrollIntoViewIfNeeded().catch(() => {});
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-offsite.png`) });
        await page.locator("[data-vt-quality-checks]").scrollIntoViewIfNeeded().catch(() => {});
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-checks.png`) });
      }
      // ── SCHEDULED JOBS on Pipeline health ([PRODUCT] 2026-11-21; `jobs` in /api/data/pipeline-health-dashboard):
      // failing rows first with their cause in words, the chip never cut, no sideways scroll at any width
      if (name === "pipelinehealth") {
        const fail = (m) => checks.failures.push("pipeline health, scheduled jobs: " + m);
        await page.waitForSelector("[data-vt-jobs] .vt-job", { timeout: 8000 }).catch(() => {});
        const rows = await page.$$eval("[data-vt-jobs] .vt-job", (els) => els.map((e) => ({
          tone: e.getAttribute("data-tone"),
          chip: e.querySelector(".vt-streams-chip")?.textContent || "",
          cause: e.querySelector(".vt-job-cause")?.textContent || "",
          meta: e.querySelector(".vt-job-meta")?.textContent || "",
          cut: (() => { const c = e.querySelector(".vt-streams-chip"); return !c || c.scrollWidth > c.clientWidth + 1 || c.getBoundingClientRect().right > e.getBoundingClientRect().right + 1; })(),
        })));
        if (rows.length !== 7) fail(`${rows.length} rows, expected 7`);
        else {
          const chips = rows.map((r) => r.chip).join(" | ");
          if (chips !== "last run failed | late | on time | on time | running | not due now | switched off") fail(`order or states read "${chips}"`);
          if (rows[0].cause !== "The program exited with an error.") fail(`the failed job's cause reads "${rows[0].cause}"`);
          if (!/^Last started 5 d ago; it failed\. Schedule: every 168h \(always\), grace 26h\.$/.test(rows[0].meta)) fail(`the failed job's last run reads "${rows[0].meta}"`);
          if (rows[1].cause !== "The run has been going longer than this job is allowed.") fail(`the stuck job's cause reads "${rows[1].cause}"`);
          if (rows[6].meta !== "No run is recorded. Schedule: every 168h (always), grace 26h.") fail(`the switched-off job reads "${rows[6].meta}"`);
          if (rows.some((r) => r.cut)) fail("a state chip is cut or runs past its row");
          if (rows.slice(2).some((r) => r.cause)) fail("a job that is not failing shows a cause");
        }
        const sub = await page.locator("[data-vt-jobs] .vt-quality-section-sub").innerText().catch(() => "");
        if (!/^7 jobs: 3 on time, 1 not due now, 1 switched off, 2 failing · read \d+ s ago$/.test(sub)) fail(`the summary reads "${sub}"`);
        const over = await page.$eval("[data-vt-jobs]", (el) => el.scrollWidth > el.clientWidth + 1).catch(() => true);
        if (over) fail("the section scrolls sideways");
        await page.locator("[data-vt-jobs]").scrollIntoViewIfNeeded().catch(() => {});
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-jobs.png`) });
      }
      // ── SEC FILING POLLERS on Pipeline health ([PRODUCT] 2026-11-24; `edgar` in /api/data/pipeline-health-dashboard):
      // the failing poller first with what it could not read, every row's data age said, the chip never cut, no
      // sideways scroll at any width
      if (name === "pipelinehealth") {
        const fail = (m) => checks.failures.push("pipeline health, SEC filing pollers: " + m);
        await page.waitForSelector("[data-vt-edgar] .vt-job", { timeout: 8000 }).catch(() => {});
        const rows = await page.$$eval("[data-vt-edgar] .vt-job", (els) => els.map((e) => ({
          name: e.querySelector(".vt-job-name")?.textContent || "",
          tone: e.getAttribute("data-tone"),
          chip: e.querySelector(".vt-streams-chip")?.textContent || "",
          cause: e.querySelector(".vt-job-cause")?.textContent || "",
          meta: e.querySelector(".vt-job-meta")?.textContent || "",
          cut: (() => { const c = e.querySelector(".vt-streams-chip"); return !c || c.scrollWidth > c.clientWidth + 1 || c.getBoundingClientRect().right > e.getBoundingClientRect().right + 1; })(),
        })));
        if (rows.length !== 3) fail(`${rows.length} rows, expected 3`);
        else {
          const said = rows.map((r) => `${r.name} = ${r.chip}`).join(" | ");
          if (said !== "13F: institutional holdings = failing | Form 4: insider trades = read in part | 8-K: earnings releases = reading") fail(`order or states read "${said}"`);
          if (rows[0].cause !== "The latest poll could not read 40 of 40 filings. First failure: index.json -> 404.") fail(`the failing poller's cause reads "${rows[0].cause}"`);
          if (rows[1].cause !== "The latest poll could not read 3 of 25 filings. First failure: submission text -> 404.") fail(`the part-read poller's cause reads "${rows[1].cause}"`);
          // [PIPELINE 2026-11-24, (ai2)] filings read from the submission text are said in the row's line
          if (rows[1].meta !== "Latest poll 4 min ago: 25 filings asked for, 22 read (5 from the submission text: EDGAR had not yet published their listings). Last new filing archived 18 min ago. Since the server started 10 h ago: 38 polls read, 0 failed.") fail(`the part-read poller's line reads "${rows[1].meta}"`);
          if (rows[2].cause) fail(`a poller that read everything shows a cause: "${rows[2].cause}"`);
          if (rows[2].meta !== "Latest poll 4 min ago: 15 filings asked for, 9 read, 6 had no press-release exhibit. Last new filing archived 18 min ago. Since the server started 10 h ago: 38 polls read, 0 failed.") fail(`the reading poller's line reads "${rows[2].meta}"`);
          if (!rows.every((r) => /^Latest poll 4 min ago: /.test(r.meta) && /Last new filing archived \d+ (min|h) ago\./.test(r.meta))) fail("a row does not say how old its poll and its last archived filing are");
          if (rows.some((r) => r.cut)) fail("a state chip is cut or runs past its row");
        }
        const sub = await page.locator("[data-vt-edgar] .vt-quality-section-sub").innerText().catch(() => "");
        if (!/^3 pollers, every 15 min: 1 failing, 1 read in part, 1 reading · read \d+ s ago$/.test(sub)) fail(`the summary reads "${sub}"`);
        const over = await page.$eval("[data-vt-edgar]", (el) => el.scrollWidth > el.clientWidth + 1).catch(() => true);
        if (over) fail("the section scrolls sideways");
        await page.locator("[data-vt-edgar]").scrollIntoViewIfNeeded().catch(() => {});
        await page.screenshot({ path: path.join(OUT, `${name}-${vp.w}-edgar.png`) });
      }
      if (name === "firesnodata") {
        const fail = (m) => checks.failures.push("fires, no detections: " + m);
        const txt = await page.locator("[data-vt-fnf-nodata]").innerText({ timeout: 8000 }).catch(() => "");
        if (!/^NASA FIRMS returned no detections from any satellite \(asked 1[123] min ago\): NOAA-20 no detections · NOAA-21 request failed · Suomi NPP no detections\. With no detections there is nothing to check against the facilities; this is not a reading of "no fires nearby"\.$/.test(txt)) fail(`the panel reads "${txt}"`);
        const body = await page.evaluate(() => document.body.innerText);
        if (/warming up/i.test(body)) fail("the panel still says it is warming up after a poll has run");
        if ((await page.locator(".vt-filings-table").count()) !== 0) fail("a facilities table is drawn with no detections to check");
      }
      // ── FAA LIVE FEEDS on the Streams inventory ([PRODUCT] 2026-11-20; GET /api/data/swim/products): one card per
      // feed under its own heading, apart from the archive list: the state chip, the newest message's age, what arrived
      // and what was lost (a dropped or unparsed message is always said), what the feed is used for; a missing setting is
      // counted, never named; the archive count above is unchanged by the block.
      if (name === "streams") {
        const fail = (m) => checks.failures.push("faa feeds: " + m);
        await page.waitForSelector("[data-vt-faa-feeds] [data-vt-faa-feed]", { timeout: 8000 }).catch(() => {});
        const ff = await page.evaluate(() => {
          const txt = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
          const sec = document.querySelector("[data-vt-faa-feeds]");
          if (!sec) return null;
          return { state: sec.getAttribute("data-state"), say: txt(sec.querySelector("[data-vt-faa-feeds-say]")), text: txt(sec),
            cards: [...sec.querySelectorAll("[data-vt-faa-feed]")].map((c) => ({ id: c.getAttribute("data-vt-faa-feed"), state: c.getAttribute("data-state"), chip: txt(c.querySelector(".vt-streams-chip")), age: txt(c.querySelector("[data-vt-feed-age]")), meta: txt(c.querySelector("[data-vt-feed-counts]")), headH: Math.round(c.querySelector(".vt-streams-cardhead").getBoundingClientRect().height), attr: [...c.querySelectorAll(".vt-streams-attr")].map(txt), over: Math.round(c.getBoundingClientRect().right - window.innerWidth) })),
            archiveCards: document.querySelectorAll(".vt-streams-list > .vt-streams-card:not([data-vt-faa-feed])").length, sub: txt(document.querySelector(".vt-filings-head .vt-filings-sub")),
            pageOver: Math.max(0, document.documentElement.scrollWidth - window.innerWidth) };
        });
        if (!ff) fail("no FAA live feeds block on the Streams inventory");
        else {
          console.log(`  faa feeds ${vp.w}px: ${ff.state} · "${ff.say}"`);
          if (ff.state !== "ready") fail(`the block reads "${ff.state}"`);
          if (ff.say !== "1 of 3 FAA feeds the server reads are receiving · traffic management quiet, notices to airmen disconnected · 2 more not read") fail(`the line reads "${ff.say}"`);
          if (JSON.stringify(ff.cards.map((c) => [c.id, c.state, c.chip])) !== JSON.stringify([["SFDPS", "receiving", "receiving"], ["TFMS", "quiet", "quiet"], ["STDDS", "not_consumed", "not read"], ["NOTAM", "disconnected", "disconnected"], ["ITWS", "not_set_up", "not set up"]])) fail(`the cards read ${JSON.stringify(ff.cards.map((c) => [c.id, c.state, c.chip]))}`);
          const by = Object.fromEntries(ff.cards.map((c) => [c.id, c]));
          if (by.SFDPS && (by.SFDPS.meta !== "18,113 messages since the server started · 17,506 handled · 607 dropped under load" || !/^last message \d+ (s|min) ago$/.test(by.SFDPS.age))) fail(`flight plans reads "${by.SFDPS.meta}" / "${by.SFDPS.age}"`);
          // the fixture's clock is fixed when the harness starts: by a later width the ages have grown by the run's own minutes
          if (by.TFMS && !/^last message 2\d min ago$/.test(by.TFMS.age)) fail(`a feed silent for 20 minutes reads "${by.TFMS.age}"`);
          if (by.NOTAM && (by.NOTAM.meta !== "32 messages since the server started · 30 handled · 2 failed to parse · 1 reconnect" || !by.NOTAM.attr.some((a) => a === "Last error: broker closed the session"))) fail(`the disconnected feed reads "${by.NOTAM.meta}" / ${JSON.stringify(by.NOTAM.attr)}`);
          if (by.ITWS && (by.ITWS.meta !== "3 settings missing on the server" || by.ITWS.age !== "")) fail(`the feed that is not set up reads "${by.ITWS.meta}" / "${by.ITWS.age}"`);
          if (/SWIM_ITWS_/.test(ff.text)) fail("a server setting's NAME is printed on the page (only the count may be)");
          if (!/These are live connections, not archives/.test(ff.text) || !/starts again when the server restarts/.test(ff.text)) fail("the block does not say the feeds are not archives and that counts restart");
          if (!/still in\s+force are saved, each minute/.test(ff.text) || /nothing here is a file/.test(ff.text)) fail("the block does not say which lists are saved across a restart (or still says nothing is a file on the volume)");
          if (/NaN|undefined|\bnull\b|Infinity/.test(ff.text)) fail(`a bare NaN / undefined / null in "${ff.text.slice(0, 200)}"`);
          if (ff.archiveCards !== 4 || !/^4 archived streams/.test(ff.sub)) fail(`the archive list shows ${ff.archiveCards} cards under "${ff.sub}" (the feeds must not be counted as archives)`);
          // first run, 390px: the name broke into three lines beside the chip and the age; the head is now one or two lines
          if (ff.cards.some((c) => c.headH > 60)) fail(`a feed card's head is ${Math.max(...ff.cards.map((c) => c.headH))}px tall (the name is breaking beside the chip)`);
          if (ff.cards.some((c) => c.over > 1) || ff.pageOver > 1) fail(`a feed card leaves the window (page overflow ${ff.pageOver}px)`);
          await page.locator("[data-vt-faa-feeds]").scrollIntoViewIfNeeded().catch(() => {});
          await page.locator("[data-vt-faa-feeds]").screenshot({ path: path.join(OUT, `streams-faafeeds-${vp.w}.png`), animations: "disabled" }).catch(() => {});
        }
      }
      if (name === "railtraffic") {
        const fail = (m) => checks.failures.push("rail carloads: " + m);
        await page.waitForSelector("[data-vt-rail-commodities]", { timeout: 8000 }).catch(() => {});
        const total = await page.locator("[data-vt-rail-total]").innerText().catch(() => "");
        if (total !== "741,292") fail(`system total reads "${total}"`);
        const age = await page.locator("[data-vt-rail-age]").innerText().catch(() => "");
        if (!/^Newest week in our archive: 2026-10-28 \(\d+ weeks? ago\) · archive rebuilt 2026-11-02 · 501 weeks held · 6 railroads reported this week\. The archive is rebuilt/.test(age)) fail(`age line reads "${age}"`);
        const cells = await page.$$eval("[data-vt-rail-commodities] tbody tr", (trs) => trs.map((tr) => [...tr.querySelectorAll("td")].map((td) => td.textContent)));
        if (cells.length !== 4) fail(`${cells.length} commodity rows, expected 4`);
        else {
          if (cells[0][0] !== "Containers (intermodal)" || cells[0][3] !== "+0.3%") fail(`first row reads "${cells[0].join(" | ")}"`);
          if (cells[2][3] !== "-14.0%") fail(`a fall reads "${cells[2][3]}"`);
          if (cells[3][2] !== "—" || cells[3][3] !== "—") fail(`a commodity with no prior week reads "${cells[3].join(" | ")}"`);
        }
        // two tables on one panel: neither may be squeezed into a scroll box
        // (first run, 390px: each showed one row), and on a phone the last
        // cell keeps its label
        const wraps = await page.$$eval(".vt-tablewrap-flow", (els) => els.map((e) => ({ sh: e.scrollHeight, ch: e.clientHeight })));
        if (wraps.length !== 2 || wraps.some((w) => w.sh > w.ch + 1)) fail(`a table is clipped vertically: ${JSON.stringify(wraps)}`);
        if (vp.w < 640) {
          const lbl = await page.$eval("[data-vt-rail-commodities] tbody tr td:last-child", (td) => getComputedStyle(td, "::before").content);
          if (!/Change/.test(lbl)) fail(`on a phone the change cell's label reads ${lbl}`);
        }
        const rr = await page.locator("[data-vt-rail-railroads] tbody tr").count();
        if (rr !== 6) fail(`${rr} railroad rows, expected 6`);
        const txt = await page.evaluate(() => document.body.innerText);
        if (!/GATE 2 \(signal\) REJECTED/.test(txt)) fail("the rejected signal test is not stated on the page");
      }
      if (name === "forecastexpired") {
        const fail = (m) => checks.failures.push("session-expired: " + m);
        const notice = page.locator("[data-vt-session-expired]");
        if (!(await notice.count()) || !(await notice.first().isVisible())) fail("sign-in view with the expiry notice did not render");
        else if (!/session expired/i.test(await notice.first().innerText())) fail("expiry notice text missing");
        if (!(await page.locator('input[type="password"]').count())) fail("no sign-in form");
        const txt = await page.evaluate(() => document.body.innerText);
        if (/service unavailable/i.test(txt)) fail("outage banner rendered for a 401");
        if (/Not authenticated/.test(txt)) fail("raw 401 body shown to the user");
        if (!/#\/forecast/.test(page.url())) fail(`hash left the forecast tab (${page.url()}) — sign-in would not return there`);
      }
      if (name === "botpane") {
        const fail = (m) => checks.failures.push("bot-pane: " + m);
        const ids = await page.$$eval("section.vt-bot-panel[data-panel]", (els) => els.map((e) => e.getAttribute("data-panel")));
        if (ids.length !== 1 || ids[0] !== "chart") fail(`expected the chart panel alone, got [${ids.join(", ")}]`);
        if (await page.locator(".tab-nav").count()) fail("the app shell's nav rendered inside the pane");
        const fill = await page.evaluate(() => { const el = document.querySelector("section.vt-bot-panel"); if (!el) return 0; return el.getBoundingClientRect().height / innerHeight; });
        if (fill < 0.85) fail(`the panel fills ${Math.round(fill * 100)}% of the window height`);
        const title = await page.title();
        if (!title.includes("Chart")) fail(`window title is "${title}"`);
        if (!(await page.locator('section.vt-bot-panel .vt-bot-ctl button[aria-label="Back to the page"]').count())) fail("no 'Back to the page' control");
        if (await page.locator('section.vt-bot-panel .vt-bot-ctl button[aria-label="Pin this panel"]').count()) fail("pin control shown in a pane (it has no layout to pin to)");
        let info = null;
        for (let i = 0; i < 24; i++) {
          info = await page.evaluate(() => {
            const c = document.querySelector("section.vt-bot-panel canvas");
            if (!c) return { present: false };
            const r = c.getBoundingClientRect();
            try {
              const img = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
              let painted = 0, total = 0;
              for (let k = 3; k < img.length; k += 4 * 7) { total++; if (img[k] > 0) painted++; }
              return { present: true, w: Math.round(r.width), h: Math.round(r.height), painted, total };
            } catch (e) { return { present: true, w: Math.round(r.width), h: Math.round(r.height), error: String(e) }; }
          });
          if (info.present && info.total && info.painted / info.total >= 0.02) break;
          await page.waitForTimeout(250);
        }
        if (!info || !info.present) fail("no canvas in the pane");
        else if (info.error) fail(`the pane's canvas is unreadable (${info.error})`);
        else if (!info.total || info.painted / info.total < 0.02) fail(`the pane's chart has not painted (${info.w}×${info.h})`);
        const glyph = page.locator("button.vt-help").first();
        if (!(await glyph.count())) fail("no help glyph in the pane");
        else {
          if (vp.touch) await glyph.tap(); else await glyph.click();
          await page.waitForTimeout(120);
          const card = await page.evaluate(() => { const c = document.querySelector(".vt-help-card"); if (!c) return null; const r = c.getBoundingClientRect(); return { inside: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight, text: (c.textContent || "").trim().length }; });
          if (!card) fail("the help glyph opened no card in the pane"); else if (!card.inside) fail("the pane's help card leaves its window");
          await page.keyboard.press("Escape");
        }
        // KNOWN BROKEN #50a (2026-11-12): the chart draws the ladder the engine executes. The fixture's KWEB sits
        // past its stop (last 24.54, stop 25.47), so its chip must show the stop as hit — never as waiting — with
        // the note that the engine sells on its next 1-minute bar, and POS-KILL's −25 % backstop in the legend.
        try {
          const chip = page.locator(".vt-chart-chip", { hasText: "KWEB" }).first();
          if (!(await chip.count())) fail("no KWEB chip in the chart's symbol strip");
          else {
            await chip.scrollIntoViewIfNeeded();
            if (vp.touch) await chip.tap({ timeout: 5000 }); else await chip.click({ timeout: 5000 });
            let lg = null;
            for (let i = 0; i < 40; i++) {
              lg = await page.evaluate(() => {
                const box = document.querySelector("section.vt-bot-panel .vt-chart-legend");
                return box ? { text: (box.textContent || "").trim(), notes: [...box.querySelectorAll(".vt-chart-note")].map((n) => (n.textContent || "").trim()), over: box.scrollWidth - box.clientWidth, right: Math.round(box.getBoundingClientRect().right) } : null;
              });
              if (lg && /25\.47/.test(lg.text)) break;
              await page.waitForTimeout(100);
            }
            await page.waitForTimeout(400);
            await page.screenshot({ path: path.join(OUT, `${name}-kweb-${vp.w}.png`), animations: "disabled" });
            if (!lg || !/25\.47/.test(lg.text)) fail("the KWEB chip did not put KWEB's levels in the legend");
            else {
              if (!/stop\??25\.47 · 4\.00% · stop · from entry · hit/.test(lg.text)) fail(`KWEB's stop is not marked hit in the legend — "${lg.text.slice(0, 220)}"`);
              if (!lg.notes.some((n) => n.startsWith("the last price is past the stop: the engine sells on its next 1-minute bar"))) fail(`no note says the engine sells on its next 1-minute bar (notes: ${JSON.stringify(lg.notes)})`);
              if (!/backstop\??19\.90 · −25%/.test(lg.text)) fail(`KWEB's legend has no −25% backstop — "${lg.text.slice(0, 220)}"`);
              if (/waits|grace to day|hard floor|minimum hold|exit now only/.test(lg.text)) fail("KWEB's legend still shows a rule the engine does not execute");
              if (lg.over > 1 || lg.right > vp.w + 1) fail(`the legend overflows with the hit note (scroll ${lg.over}px, right edge ${lg.right}px in a ${vp.w}px window)`);
            }
          }
        } catch (e) {
          fail(`KWEB chip: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
        }
      }
      // ── TRADE COCKPIT (2026-11-06, the port of PR #1556): a tool tab zooms
      // its panel and keeps the one look-up box in the zoomed bar (Research's
      // Stock panel needs a symbol; the old page had its own input). With a
      // symbol loaded, every Trade feature is one panel on the page (no second
      // scanner, no second look-up box), the Stock panel's Fundamentals tab
      // opens the live price bar the old Analyze page carried, and only the
      // owner's view offers the dossier jump.
      // HEAVY LOADS (2026-11-06, keep the system alive): a look-up runs ONE
      // Python-backed fetch, /api/analyze; research (python -m alphadesk),
      // insights (insights.py) and the ETF builder each wait for their
      // "Load … for <SYM>" press, read from the requests the render made.
      if (name === "research") {
        const fail = (m) => checks.failures.push("trade-cockpit: " + m);
        const zoomed = await page.$$eval('[data-world="trade"] section.vt-bot-panel[data-panel]', (els) => els.map((e) => e.getAttribute("data-panel")));
        if (zoomed.join(",") !== "stock") fail(`the Research tab shows [${zoomed.join(", ")}], expected the Stock panel alone`);
        const boxes = await page.locator('[data-world="trade"] .vt-lookup-box input').count();
        if (boxes !== 1) fail(`the zoomed Research tab has ${boxes} look-up boxes, expected 1`);
        try {
          const n = (p) => apiLog.filter((x) => x === p).length;
          const box = page.locator('[data-world="trade"] .vt-lookup-box input').first();
          await box.fill("SPY");
          await box.press("Enter");
          for (let i = 0; i < 40 && !n("/api/analyze/SPY"); i++) await page.waitForTimeout(100);
          await page.waitForTimeout(400);
          if (n("/api/analyze/SPY") !== 1) fail(`the look-up fetched /api/analyze/SPY ${n("/api/analyze/SPY")} times, expected once`);
          for (const p of ["/api/research/", "/api/insights/", "/api/etf/"]) if (apiLog.some((x) => x.startsWith(p))) fail(`the look-up alone fetched ${p} (it must wait for its press)`);
          const load = page.locator('section.vt-bot-panel[data-panel="stock"] [data-load="research"] button', { hasText: "Load research for SPY" }).first();
          if (!(await load.count())) fail("no 'Load research for SPY' control after the look-up");
          else {
            if (vp.touch) await load.tap(); else await load.click();
            for (let i = 0; i < 40 && !n("/api/research/SPY"); i++) await page.waitForTimeout(100);
            if (n("/api/research/SPY") !== 1) fail(`the press fetched /api/research/SPY ${n("/api/research/SPY")} times, expected once`);
            let report = false;
            for (let i = 0; i < 30 && !report; i++) { report = /conviction/.test(await page.locator('section.vt-bot-panel[data-panel="stock"]').innerText()); if (!report) await page.waitForTimeout(100); }
            if (!report) fail("the research report did not render after the press");
            await page.screenshot({ path: path.join(OUT, `${name}-loaded-${vp.w}.png`), animations: "disabled" });
          }
        } catch (e) {
          fail(`driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
        }
      }
      // ── AI ENGINE MENU (2026-11-06, the port of PR #1556's nav): the owner
      // reaches the bot page, Dossier, Forecast and Brain through one drop-down
      // beside Trade and Data — no direct owner buttons left in the desktop
      // nav; the menu opens on hover with its four leaves inside the viewport.
      if (name === "analyzeowner") {
        const fail = (m) => checks.failures.push("engine-menu: " + m);
        const direct = await page.locator(".tab-nav-tabs > button.tab-btn").count();
        if (direct) fail(`${direct} direct owner button(s) left in the desktop nav`);
        const btn = page.locator(".tab-nav-tabs .analyze-dropdown-wrapper > button.tab-btn", { hasText: "AI Engine" }).first();
        if (!vp.touch) {
          if (!(await btn.count()) || !(await btn.isVisible())) fail("no AI Engine menu for the owner");
          else {
            await btn.hover();
            await page.waitForTimeout(150);
            const items = await page.$$eval(".nav-dropdown-menu [role='menuitem'] .analyze-dropdown-label", (els) => els.map((e) => (e.textContent || "").trim()));
            if (items.join(",") !== "Bot,Dossier,Forecast,Brain") fail(`the AI Engine menu reads [${items.join(", ")}]`);
            const inside = await page.evaluate(() => { const m = document.querySelector(".nav-dropdown-menu"); if (!m) return false; const r = m.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth && r.bottom <= innerHeight; });
            if (!inside) fail("the AI Engine menu leaves the viewport");
            await page.screenshot({ path: path.join(OUT, `${name}-enginemenu-${vp.w}.png`), animations: "disabled" });
            await page.mouse.move(2, 600);
            await page.waitForTimeout(250);
          }
        }
      }
      if (name === "analyzeloaded" || name === "analyzeowner") {
        const fail = (m) => checks.failures.push("trade-cockpit: " + m);
        try {
          for (let i = 0; i < 40; i++) { if (await page.locator('section.vt-bot-panel[data-panel="stock"]').count()) break; await page.waitForTimeout(100); }
          const ids = await page.$$eval('[data-world="trade"] section.vt-bot-panel[data-panel]', (els) => els.map((e) => e.getAttribute("data-panel")).sort());
          const want = ["etf", "news", "options", "plan", "screener", "smart", "stock", "structure", "watchlist"];
          if (ids.join(",") !== want.join(",")) fail(`panels [${ids.join(", ")}], expected [${want.join(", ")}]`);
          if ((await page.locator('[data-world="trade"] .vt-lookup-box input').count()) !== 1) fail("expected exactly one look-up box");
          // the page opened on SPY: ONE Python-backed fetch; the heavy panels wait for their press
          const n = (p) => apiLog.filter((x) => x === p).length;
          if (n("/api/analyze/SPY") !== 1) fail(`the look-up fetched /api/analyze/SPY ${n("/api/analyze/SPY")} times, expected once`);
          for (const p of ["/api/research/", "/api/insights/", "/api/etf/"]) if (apiLog.some((x) => x.startsWith(p))) fail(`the look-up alone fetched ${p} (it must wait for its press)`);
          // one press on Smart Money's control: insights fetched once, Smart Money and Structure both filled
          const loadInsights = page.locator('section.vt-bot-panel[data-panel="smart"] [data-load="insights"] button').first();
          if (!(await loadInsights.count())) fail("Smart Money has no load control");
          else {
            if (vp.touch) await loadInsights.tap(); else await loadInsights.click();
            for (let i = 0; i < 40 && !n("/api/insights/SPY"); i++) await page.waitForTimeout(100);
            if (n("/api/insights/SPY") !== 1) fail(`the press fetched /api/insights/SPY ${n("/api/insights/SPY")} times, expected once`);
            let filled = 0;
            for (let i = 0; i < 30 && filled < 2; i++) { filled = await page.locator('section.vt-bot-panel[data-panel="smart"] [data-testid="section-insights"], section.vt-bot-panel[data-panel="structure"] [data-testid="section-insights"]').count(); if (filled < 2) await page.waitForTimeout(100); }
            if (filled !== 2) fail(`one press filled ${filled} of Smart Money and Structure`);
            if (apiLog.some((x) => x.startsWith("/api/research/") || x.startsWith("/api/etf/"))) fail("the insights press loaded another heavy panel");
          }
          const tab = page.locator('section.vt-bot-panel[data-panel="stock"] .vt-bot-tab', { hasText: "Fundamentals" }).first();
          if (vp.touch) await tab.tap(); else await tab.click();
          let bar = false;
          for (let i = 0; i < 20 && !bar; i++) { bar = (await page.locator('section.vt-bot-panel[data-panel="stock"] [data-testid="section-live-price"]').count()) > 0; if (!bar) await page.waitForTimeout(100); }
          if (!bar) fail("Stock · Fundamentals has no live price bar");
          if (!(await page.locator('section.vt-bot-panel[data-panel="stock"] [data-testid="section-fundamentals"]').count())) fail("Stock · Fundamentals has no fundamentals panel");
          await page.screenshot({ path: path.join(OUT, `${name}-fundamentals-${vp.w}.png`), animations: "disabled" });
          const dossier = await page.locator("a.vt-trade-dossier").count();
          if (name === "analyzeowner" && dossier !== 1) fail("the owner's view has no dossier jump");
          if (name === "analyzeloaded" && dossier !== 0) fail("a visitor sees the owner's dossier jump");
        } catch (e) {
          fail(`driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
        }
      }
      // ── TICKER DOSSIER (2026-10-29, human-asked): the range band is the
      // hero, every layer carries a status pill and provenance, the trade
      // plan draws its ladder and sizes from a typed dollar risk, and the
      // look-up card sits in the stock layer. Full-page shot for review.
      if (name === "dossier" || name === "dossierempty") {
        const fail = (m) => checks.failures.push("dossier: " + m);
        try {
          if (name === "dossier") {
            for (let i = 0; i < 60; i++) { await page.waitForTimeout(100); if (await page.locator(".vt-ds-hero .vt-ds-bandviz").count()) break; }
            const st = await page.evaluate(() => {
              const q = (sel) => document.querySelectorAll(sel).length;
              const vis = (el) => { if (!el) return false; const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
              return { bands: q(".vt-ds-hero .vt-ds-bandviz"), b68: vis(document.querySelector(".vt-ds-b68")), px: q(".vt-ds-bpx"), pills: [...document.querySelectorAll(".vt-ds-status")].map((e) => e.getAttribute("data-s")),
                prov: q(".vt-ds-prov"), ladder: q(".vt-ds-ladder .vt-ds-lline"), card: !!document.querySelector("[data-vt-dossier] .vt-lookup[data-symbol='RKLB']"),
                dir: (document.querySelector(".vt-ds-dirword")?.textContent || "").trim(), advice: /Research, not advice/.test(document.body.textContent || ""), tiers: q(".vt-ds-tier"),
                decile: /n=1184: up after 5 sessions/.test(document.body.textContent || ""), coin: /coin flip/.test(document.body.textContent || ""),
                history: [...document.querySelectorAll(".vt-ds-ev [data-history]")].map((e) => e.textContent || ""), notPriced: /possibly not priced in|small reaction|UNVALIDATED/i.test(document.querySelector(".vt-ds-evs")?.textContent || "") };
            });
            if (st.bands !== 2 || !st.b68) fail(`range hero: ${st.bands} bands drawn`);
            if (st.px !== 2) fail("the live price is not marked on both bands");
            if (st.pills.length !== 5 || st.pills.some((p) => p !== "live")) fail(`layer pills read ${st.pills.join(",")}`);
            if (st.prov < 5) fail(`only ${st.prov} provenance lines`);
            if (st.ladder !== 4) fail(`trade-plan ladder has ${st.ladder} lines`);
            if (!st.card) fail("the look-up card is not in the stock layer");
            if (st.dir !== "no validated direction" || !st.coin) fail(`direction reads "${st.dir}" without the coin-flip statement`);
            if (!st.decile) fail("the bot's score-band record is missing (lookup score 58.2 -> decile 6, n=1184)");
            if (!st.advice) fail("no 'Research, not advice' line");
            if (st.tiers < 3) fail(`only ${st.tiers} credibility-tier chips`);
            // event-reaction study (2026-10-29): measured types show history with n; nothing is flagged "not priced in"
            if (st.history.length !== 1 || !/^history Historically \(2016–2026, n=42,956\)/.test(st.history[0])) fail(`event history lines read ${JSON.stringify(st.history)}`);
            if (st.notPriced) fail("an event still carries a 'not priced in' / reaction flag");
            const risk = page.locator("#vt-ds-risk");
            await risk.fill("250"); await page.waitForTimeout(100);
            const sized = await page.evaluate(() => (document.querySelector(".vt-ds-size .vt-ds-m")?.textContent || ""));
            if (!/^\d+ shares · /.test(sized)) fail(`size helper reads "${sized}"`);
            if (vp.touch) {
              const small = await page.evaluate(() => [...document.querySelectorAll(".vt-ds button, .vt-ds input, .vt-ds a, .vt-ds summary")].filter((e) => !e.closest(".vt-lookup") && !e.classList.contains("vt-help")).filter((e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.height < 44; }).map((e) => e.className || e.tagName).slice(0, 5));
              if (small.length) fail(`touch targets under 44px: ${small.join(", ")}`);
            }
          } else {
            const st = await page.evaluate(() => ({ state: !!document.querySelector(".vt-ds-state"), input: !!document.querySelector(".vt-ds-search input") }));
            if (!st.state || !st.input) fail("the empty state or the search box is missing");
          }
          await page.screenshot({ path: path.join(OUT, `${name}-full-${vp.w}.png`), fullPage: true, animations: "disabled" });
        } catch (e) {
          fail(`driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
        }
      }
      // ── COCKPIT LAYOUT (2026-10-29, human-asked on a 1920 monitor: "fill
      // the page … how the UI looks on the bot instead of the strip").
      // Forecast, Brain and the dossier render as the AI Engine does: the page
      // fills the content width (the shell's 920px reading column was 59% of
      // 1440), a KPI strip across the top with every tile in one row, then
      // columns of panels — at 1440 the grid runs 3 columns (2 for the Brain's
      // wide evidence cards) and at least 3 panels share a row (the strip
      // always), at 1920 one column more, at 768 two, at 390 a single column —
      // nothing past the right edge. The first tile carries the page's own
      // headline number from the fixture (every number is the page's), and
      // the Brain's tiles still filter the cards. The dossier's empty state
      // has no data: it proves the shell, the search and the state alone.
      if (COCKPIT_PAGES.has(name)) {
        const fail = (m) => checks.failures.push("cockpit: " + m);
        const expect = COCKPIT_PAGES.get(name);
        try {
          for (let i = 0; i < 60; i++) { if (await page.locator(".vt-ck-grid > .vt-ck-panel, .vt-ck > .vt-ds-state, .vt-ck > .vt-fc-state").count()) break; await page.waitForTimeout(100); }
          const read = () => page.evaluate(() => {
            const shell = document.querySelector(".vt-ck");
            if (!shell) return null;
            const box = (el) => el.getBoundingClientRect();
            const rowMax = (els) => { const m = new Map(); for (const el of els) { const r = box(el); if (r.width <= 0 || r.height <= 0) continue; const k = Math.round(r.top); m.set(k, (m.get(k) || 0) + 1); } return Math.max(0, ...m.values()); };
            const panels = [...document.querySelectorAll(".vt-ck-grid > .vt-ck-panel, .vt-ck-grid > .vt-ds-side > .vt-ck-panel")];
            const tiles = [...document.querySelectorAll(".vt-ck-kpi > .vt-ck-tile")];
            const cols = new Set(panels.map((p) => Math.round(box(p).left))).size;
            // a wide table inside an overflow-x: auto wrapper (the by-week / ablation tables) scrolls within it — only the wrapper must stay inside
            const scroller = (el) => { for (let p = el.parentElement; p && p !== shell; p = p.parentElement) { const ox = getComputedStyle(p).overflowX; if (ox === "auto" || ox === "scroll" || ox === "hidden") return p; } return null; };
            const over = [...shell.querySelectorAll("*")].filter((el) => { const r = box(el); if (r.width <= 0 || r.right <= innerWidth + 1 || getComputedStyle(el).position === "fixed") return false; const s = scroller(el); return !(s && box(s).right <= innerWidth + 1); }).map((el) => (el.className || el.tagName).toString().slice(0, 40)).slice(0, 3);
            const cards = [...document.querySelectorAll(".vt-ck-grid > .vt-br-card")];
            return { fill: box(shell).width / innerWidth, panels: panels.length, panelRow: rowMax(panels), cols, tiles: tiles.length, tileRow: rowMax(tiles), over,
              tileText: tiles.map((t) => (t.innerText || "").replace(/\s+/g, " ").trim()), state: !!document.querySelector(".vt-ck > .vt-ds-state, .vt-ck > .vt-fc-state"),
              cards: cards.length, cardStatus: [...new Set(cards.map((c) => c.getAttribute("data-status")))], h1: !!shell.querySelector("h1"), h2s: shell.querySelectorAll("h2").length };
          });
          const lay = await read();
          if (!lay) fail("no .vt-ck cockpit shell rendered");
          else {
            if (lay.fill < 0.92) fail(`the page fills ${Math.round(lay.fill * 100)}% of the ${vp.w}px viewport (needs ≥ 92%; the old column was 59%)`);
            if (lay.over.length) fail(`past the right edge of the viewport: ${lay.over.join(", ")}`);
            if (!lay.h1) fail("no h1");
            if (lay.tiles !== expect.tiles) fail(`KPI strip has ${lay.tiles} tiles (expected ${expect.tiles})`);
            if (expect.first && !expect.first.test(lay.tileText[0] || "")) fail(`first KPI tile reads "${lay.tileText[0]}"`);
            if (lay.tiles && lay.tileRow < Math.min(lay.tiles, vp.w >= 640 ? 3 : 2)) fail(`KPI strip: only ${lay.tileRow} of ${lay.tiles} tiles share a row at ${vp.w}`);
            if (expect.empty) {
              if (!lay.state || lay.panels) fail(`empty state: state ${lay.state}, ${lay.panels} panels`);
            } else {
              if (!lay.panels || lay.h2s < 3) fail(`${lay.panels} grid panels, ${lay.h2s} h2s`);
              const need = vp.w >= 1920 ? expect.cols1920 : vp.w >= 1440 ? expect.cols1440 : vp.w >= 768 ? 2 : 1;
              if (vp.w >= 768) {
                if (lay.cols < need) fail(`the grid runs ${lay.cols} columns at ${vp.w} (expected ≥ ${need}: columns, not a strip)`);
                if (lay.panelRow < 2) fail(`no two grid panels share a row at ${vp.w}`);
                if (vp.w >= 1440 && Math.max(lay.panelRow, lay.tileRow) < 3) fail(`fewer than 3 panels share a row at ${vp.w}`);
              } else if (lay.cols !== 1 || lay.panelRow !== 1) fail(`phone: ${lay.cols} columns, ${lay.panelRow} panels on one row (expected a single column)`);
            }
            // the forecast's Daily / Weekly switch (Part C): Weekly shows the weekly strip, Daily brings the daily strip back
            if (expect.horizon) {
              const press = async (h) => { const b = page.locator(`button[data-vt-fc-horizon="${h}"]`); if (vp.touch) await b.tap(); else await b.click(); await page.waitForTimeout(150); };
              if (!(await page.locator('button[data-vt-fc-horizon="weekly"]').count())) fail("no Weekly horizon switch");
              else {
                await press("weekly");
                const w = await read();
                if (!w || !expect.horizon.weekly.test(w.tileText[0] || "")) fail(`weekly view: first KPI tile reads "${w?.tileText[0]}"`);
                if (!w || w.tiles !== expect.tiles || !w.panels) fail(`weekly view: ${w?.tiles} tiles, ${w?.panels} panels`);
                if (w && w.over.length) fail(`weekly view: past the right edge: ${w.over.join(", ")}`);
                await page.screenshot({ path: path.join(OUT, `${name}-weekly-full-${vp.w}.png`), fullPage: true, animations: "disabled" });
                await press("daily");
                const d = await read();
                if (!d || !expect.first.test(d.tileText[0] || "")) fail(`daily view did not come back: first tile "${d?.tileText[0]}"`);
              }
            }
            // the Brain's status tiles are the filters: Shadow shows only shadow cards, All parts brings every card back
            if (expect.filter) {
              const tile = page.locator(`.vt-ck-kpi > button.vt-ck-tile[data-status="${expect.filter.status}"]`);
              if (!(await tile.count())) fail(`no "${expect.filter.status}" filter tile`);
              else {
                if (vp.touch) await tile.tap(); else await tile.click();
                await page.waitForTimeout(150);
                const f = await read();
                if (!f || f.cards !== expect.filter.cards || f.cardStatus.join() !== expect.filter.status) fail(`filter "${expect.filter.status}": ${f?.cards} cards, status ${f?.cardStatus.join(",")}`);
                const all = page.locator('.vt-ck-kpi > button.vt-ck-tile[data-status="all"]');
                if (vp.touch) await all.tap(); else await all.click();
                await page.waitForTimeout(150);
                const back = await read();
                if (!back || back.cards !== lay.cards) fail(`"All parts" did not restore the ${lay.cards} cards (${back?.cards})`);
              }
            }
          }
          if (name === "forecast" || name === "brain") await page.screenshot({ path: path.join(OUT, `${name}-full-${vp.w}.png`), fullPage: true, animations: "disabled" });
        } catch (e) {
          fail(`driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
        }
      }
      if (name === "bot" || name === "botidle") {
        const fail = (m) => checks.failures.push("bot-ui: " + m);
        const winCount = () => page.locator(".vt-bot-win").count();
        const ctl = (id, label) => page.locator(`section.vt-bot-panel[data-panel="${id}"] .vt-bot-ctl button[aria-label="${label}"]`).first();
        const press = async (loc) => { if (vp.touch) await loc.tap(); else await loc.click(); };
        const settle = () => page.waitForTimeout(250);
        // ── OVERVIEW (2026-11-06, human refocus spec §8: "simplify; larger main chart: portfolio vs SPY, after tax; per
        // pick a plain-English card: why bought, why alternatives were rejected, current P&L; a clear 'why are we
        // losing money' panel"). The page opens on the overview: the view switch, the after-tax chart vs SPY (the
        // largest panel on the desktop, the first one in the phone stack), the pick cards, the attribution with the
        // flow / tax / signal sections, and the essentials (issues, book, orders) — and none of the cockpit's engine
        // panels. Every number is checked against this page's fixtures; the latched page has every route, the idle
        // page an empty journal, no attribution or tax route and the old signal list (each a designed sentence).
        // Then the switch opens the cockpit, where every check below runs as before.
        {
          const ovFail = (m) => checks.failures.push("overview: " + m);
          const pressQ = async (loc) => { if (vp.touch) await loc.tap({ timeout: 5000 }); else await loc.click({ timeout: 5000 }); };
          const sbFx = PAGE_FIXTURES["/api/bot/scoreboard"];
          const atPts = sbFx && sbFx.afterTax ? sbFx.afterTax.points : null;
          const pctW = (v) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(2)}%`;
          try {
            for (let i = 0; i < 40; i++) { if (await page.locator('.vt-perf[data-state="ready"], .vt-perf[data-state="empty"]').count()) break; await page.waitForTimeout(100); }
            for (let i = 0; i < 40; i++) { if (await page.locator('.vt-why[data-source]').count() && !(await page.locator('.vt-why [aria-busy="true"]').count())) break; await page.waitForTimeout(100); }
            const ov = await page.evaluate(() => {
              const q = (sel) => document.querySelector(sel);
              const r = (el) => (el ? el.getBoundingClientRect() : null);
              const ids = [...document.querySelectorAll("section.vt-bot-panel[data-panel]")].map((e) => e.getAttribute("data-panel"));
              const panelText = [...document.querySelectorAll('section.vt-bot-panel[data-panel="perf"], section.vt-bot-panel[data-panel="why"], section.vt-bot-panel[data-panel="picks"]')].map((e) => e.textContent || "").join(" ");
              const perf = q('section.vt-bot-panel[data-panel="perf"]'), svg = q(".vt-perf-plot svg"), plot = q(".vt-perf-plot");
              const pathPts = (k) => { const d = q(`.vt-perf-plot path[data-line="${k}"]`)?.getAttribute("d") || ""; return d ? d.split(/[ML]/).filter((x) => x.trim()).length : 0; };
              const cols = ["left", "perf", "why", "picks", "daily"].map((a) => {
                const col = q(`.vt-bot-grid[data-mode="overview"] .vt-bot-area-${a}`);
                if (!col) return { a, missing: true };
                const ps = [...col.querySelectorAll("section.vt-bot-panel[data-panel]")];
                const cr = col.getBoundingClientRect();
                const painted = ps.reduce((sum, x) => sum + x.getBoundingClientRect().height, 0) + 8 * Math.max(0, ps.length - 1);
                return { a, missing: false, ids: ps.map((x) => x.getAttribute("data-panel")), hole: ps.length ? cr.height - painted : null };
              });
              const grid = q('.vt-bot-grid[data-mode="overview"]');
              const smallTouch = [...document.querySelectorAll(".vt-bot-modebar button:not(.vt-help), .vt-pick-hd, .vt-why-win button, .vt-why-sh > button:not(.vt-help):not(:disabled), .vt-picks-hd .vt-bot-wbtn")].filter((b) => { const br = b.getBoundingClientRect(); return br.width > 0 && br.height > 0 && br.height < 44; }).map((b) => (b.textContent || b.className).toString().trim().slice(0, 24));
              return {
                mode: q(".vt-bot-page")?.getAttribute("data-mode"), ids,
                switchBtns: [...document.querySelectorAll(".vt-bot-modebar button[data-view]")].map((b) => [b.getAttribute("data-view"), b.getAttribute("aria-pressed")]),
                chromeOk: ["perf", "why", "picks", "daily"].every((id) => ["Hold this panel", "Pin this panel", "Open as a window"].every((l) => !!q(`section.vt-bot-panel[data-panel="${id}"] .vt-bot-ctl button[aria-label="${l}"]`))),
                bad: (panelText.match(/\bNaN\b|\bundefined\b|Infinity|\bnull\b/) || [])[0] || null,
                perf: { state: q(".vt-perf")?.getAttribute("data-state"), basis: q(".vt-perf")?.getAttribute("data-basis"), figs: [...document.querySelectorAll(".vt-perf-fig .v")].map((e) => (e.textContent || "").trim()), say: (q(".vt-perf-say")?.textContent || "").trim(), src: (q(".vt-perf-src")?.textContent || "").trim(), empty: (q(".vt-perf")?.textContent || "").trim(),
                  rect: r(perf) && { w: r(perf).width, h: r(perf).height, area: r(perf).width * r(perf).height }, svg: r(svg) && { w: r(svg).width, h: r(svg).height }, plot: r(plot) && { w: r(plot).width, h: r(plot).height },
                  bot: pathPts("bot"), spy: pathPts("spy"), pre: pathPts("bot-pre"), dd: pathPts("spy-dd"), rings: document.querySelectorAll(".vt-perf-plot circle[data-sold]").length, bands: document.querySelectorAll(".vt-perf-band").length },
                largest: [...document.querySelectorAll("section.vt-bot-panel[data-panel]")].map((e) => { const b = e.getBoundingClientRect(); return [e.getAttribute("data-panel"), b.width * b.height]; }).sort((x, y) => y[1] - x[1])[0]?.[0] ?? null,
                picks: { journal: q(".vt-picks")?.getAttribute("data-journal"), hd: (q(".vt-picks-hd")?.textContent || "").trim(), cards: [...document.querySelectorAll(".vt-pick")].map((c) => ({ key: c.getAttribute("data-key"), src: c.getAttribute("data-source"), title: (c.querySelector(".sym")?.textContent || "").trim(), pl: (c.querySelector(".pl")?.textContent || "").trim(), why: (c.querySelector(".vt-pick-why")?.textContent || "").trim() })),
                  closed: [...document.querySelectorAll('.vt-pick-mini[data-kind="closed"] .vt-pick-mrow b')].map((b) => b.textContent), passed: [...document.querySelectorAll('.vt-pick-mini[data-kind="passed"] .vt-pick-mrow')].map((x) => (x.textContent || "").trim()) },
                why: { source: q(".vt-why")?.getAttribute("data-source"), say: (q(".vt-why-say")?.textContent || "").trim(), groups: [...document.querySelectorAll(".vt-why-g")].map((g) => g.getAttribute("data-group")), firstRow: (q('.vt-why-g[data-group="signal"] .vt-why-row .l')?.textContent || "").trim(), text: (q(".vt-why")?.textContent || "").trim(),
                  secs: Object.fromEntries([...document.querySelectorAll(".vt-why-sec")].map((x) => [x.getAttribute("data-sec"), (x.querySelector(".vt-why-ss")?.textContent || "").trim()])), win: [...document.querySelectorAll(".vt-why-win button")].map((b) => [b.textContent, b.getAttribute("aria-pressed")]) },
                cols, gridOver: grid ? grid.scrollWidth - grid.clientWidth : 0, smallTouch, vw: innerWidth,
              };
            });
            console.log(`  overview ${vp.w}px: mode ${ov.mode} · panels ${ov.ids.join(",")} · chart ${ov.perf.state}/${ov.perf.basis ?? "-"} ${ov.perf.rect ? `${Math.round(ov.perf.rect.w)}×${Math.round(ov.perf.rect.h)}` : "-"} · ${ov.picks.cards.length} cards · why ${ov.why.source}`);
            if (ov.mode !== "overview") ovFail(`the page opened on "${ov.mode}", not the overview`);
            if (JSON.stringify(ov.switchBtns) !== JSON.stringify([["overview", "true"], ["cockpit", "false"]])) ovFail(`the view switch reads ${JSON.stringify(ov.switchBtns)}`);
            // re-pinned 2026-11-06 ([PRODUCT] the daily line card): + daily (was "book,issues,orders,perf,picks,why")
            if ([...ov.ids].sort().join(",") !== "book,daily,issues,orders,perf,picks,why") ovFail(`the overview shows the panels ${ov.ids.join(",")} (expected perf, why, picks, daily, issues, book, orders — no engine panel)`);
            if (!ov.chromeOk) ovFail("an overview panel lacks the hold / pin / window controls");
            if (ov.bad) ovFail(`a bare "${ov.bad}" reached the overview panels`);
            if (ov.gridOver > 1) ovFail(`the overview grid overflows sideways by ${ov.gridOver}px`);
            if (vp.touch && ov.smallTouch.length) ovFail(`touch targets under 44px: ${ov.smallTouch.join(", ")}`);
            // the stack (phone / tablet): the chart is the first panel; the desktop grid: the chart spans the centre and right columns and is the largest panel
            if (vp.w < 1180) {
              if (ov.ids[0] !== "perf") ovFail(`the phone / tablet stack starts with "${ov.ids[0]}", not the chart`);
              if (ov.perf.state === "ready" && (!ov.perf.plot || ov.perf.plot.h < 240)) ovFail(`the chart's plot is ${ov.perf.plot ? Math.round(ov.perf.plot.h) : 0}px tall on the stack (≥ 240)`);
            } else {
              for (const c of ov.cols) {
                if (c.missing) ovFail(`no ${c.a} area in the overview grid`);
                else if (c.hole !== null && c.hole > 12) ovFail(`the ${c.a} area has ${Math.round(c.hole)}px of hole (${c.ids.join(" · ")})`);
              }
              const left = ov.cols.find((c) => c.a === "left");
              if (left && !left.missing && left.ids.join(",") !== "issues,book,orders") ovFail(`the left column holds ${left.ids.join(",")}`);
              if (!ov.perf.rect || ov.perf.rect.w < ov.vw * 0.6) ovFail(`the chart panel is ${ov.perf.rect ? Math.round(ov.perf.rect.w) : 0}px wide (≥ 60% of ${ov.vw})`);
              if (!ov.perf.rect || ov.perf.rect.h < 300) ovFail(`the chart panel is ${ov.perf.rect ? Math.round(ov.perf.rect.h) : 0}px tall (≥ 300)`);
              if (ov.largest !== "perf") ovFail(`the largest panel is "${ov.largest}", not the chart`);
            }
            // the overview as first drawn (the viewport), before any control is touched
            await page.screenshot({ path: path.join(OUT, `${name}-overview-${vp.w}.png`), animations: "disabled" });
            if (name === "bot") {
              const lastPt = atPts ? atPts[atPts.length - 1] : null;
              if (!lastPt) ovFail("the bot fixture carries no after-tax series");
              else {
                if (ov.perf.state !== "ready" || ov.perf.basis !== "after-tax") ovFail(`the chart is "${ov.perf.state}" / "${ov.perf.basis}" with the after-tax series present`);
                if (ov.perf.bot !== atPts.length || ov.perf.spy !== atPts.length) ovFail(`the after-tax lines carry ${ov.perf.bot} / ${ov.perf.spy} points (the series has ${atPts.length})`);
                if (ov.perf.pre !== atPts.length || ov.perf.dd !== atPts.length) ovFail(`the before-tax / drawdown lines carry ${ov.perf.pre} / ${ov.perf.dd} points`);
                if (ov.perf.rings !== 2) ovFail(`${ov.perf.rings} "sold now" rings (expected the tile's two)`);
                const want = [pctW(lastPt.botAfterTaxPct), pctW(lastPt.spyAfterTaxPct)];
                if (ov.perf.figs[0] !== want[0] || ov.perf.figs[1] !== want[1]) ovFail(`the chart's figures read ${JSON.stringify(ov.perf.figs)} (the series' last point: ${JSON.stringify(want)})`);
                if (!/^After tax, the portfolio is ((up|down) \d+\.\d%|flat) since May 6; SPY is ((up|down) \d+\.\d%|flat)\. The bot is ((ahead of|behind) SPY by \d+\.\d points|level with SPY)\.$/.test(ov.perf.say)) ovFail(`the chart's sentence reads "${ov.perf.say}"`);
                if (!ov.perf.src.includes(`SPY: ${sbFx.sources.spy}`) || !/trading days since May 6/.test(ov.perf.src) || !/an estimate at 24 % short-term \/ 15 % long-term/.test(ov.perf.src)) ovFail(`the chart's provenance reads "${ov.perf.src.slice(0, 200)}"`);
              }
              // the cards: one per position, the META spread's two legs one card; seven explained by the journal, three honestly not
              const cards = ov.picks.cards;
              if (cards.length !== 10) ovFail(`${cards.length} pick cards (expected 10: five share lots, four single legs, one two-leg spread)`);
              if (cards.filter((c) => c.src === "journal").length !== 7) ovFail(`${cards.filter((c) => c.src === "journal").length} cards carry a journal reason (expected 7)`);
              if (!/7 of 10 with a recorded reason/.test(ov.picks.hd)) ovFail(`the picks header reads "${ov.picks.hd}"`);
              const qqq = cards.find((c) => c.title === "QQQ");
              if (!qqq || !/^Bought QQQ shares for the ETF floor: tech momentum/.test(qqq.why) || !/^−2\.4%/.test(qqq.pl)) ovFail(`the QQQ card reads ${JSON.stringify(qqq)}`);
              const vxus = cards.find((c) => c.title === "VXUS");
              if (!vxus || vxus.src !== "none" || !/^No recorded reason/.test(vxus.why)) ovFail(`the VXUS card (no journal row) reads ${JSON.stringify(vxus)}`);
              if (!cards.some((c) => c.title === "META bull put credit spread" && /^net/.test(c.pl))) ovFail("the META spread is not one card with its net P&L");
              if (ov.picks.closed.join(",") !== "TSLA") ovFail(`recently closed reads ${ov.picks.closed.join(",")}`);
              if (ov.picks.passed.length !== 2 || !/^NVDA/.test(ov.picks.passed[1] || ov.picks.passed[0])) ovFail(`passed on reads ${JSON.stringify(ov.picks.passed)}`);
              // expand QQQ: what it passed over and why, the stages, the size, and the engine's exit plan
              const qHd = page.locator('.vt-pick[data-key="QQQ"] .vt-pick-hd');
              if (await qHd.count()) {
                await qHd.scrollIntoViewIfNeeded(); await pressQ(qHd); await settle();
                const more = await page.evaluate(() => (document.querySelector('.vt-pick[data-key="QQQ"] .vt-pick-more')?.textContent || ""));
                if (!/Passed over/.test(more) || !/IV rank 74: the calls price a bigger move/.test(more)) ovFail("the expanded QQQ card does not say what it passed over and why");
                if (!/Market regime — NEUTRAL: VIX 17\.4/.test(more) || !/Tax — expected \+3\.1% over 20 days/.test(more)) ovFail("the expanded QQQ card does not walk the decision's stages");
                if (!/61 at \$741\.32 = \$45,221 \(44\.9% of equity\) — the floor's target weight at a full risk budget/.test(more)) ovFail("the expanded QQQ card does not give its size in words");
                if (!/Exit plan/.test(more) || !/stop \$719\.08 from entry/.test(more)) ovFail("the expanded QQQ card does not give the engine's exit plan");
                // KNOWN BROKEN #50a (2026-11-12): the plan is the ladder the engine executes — POS-KILL's −25 % backstop, no hard floor, nothing waiting
                if (!/backstop \$555\.99 \(−25%\)/.test(more)) ovFail(`the expanded QQQ card's exit plan does not show the −25% backstop — "${more.slice(more.indexOf("Exit plan"), more.indexOf("Exit plan") + 220)}"`);
                if (/floor \$|waits|grace to day|minimum hold/.test(more.slice(more.indexOf("Exit plan")))) ovFail("the expanded QQQ card's exit plan still shows a rule the engine does not execute (hard floor / a waiting stop)");
                if (!/decision journal · decided/.test(more)) ovFail("the expanded QQQ card does not say where its reason came from");
              } else ovFail("no QQQ card header to expand");
              const smh = page.locator('.vt-pick[data-key="SMH"] .vt-pick-hd');
              if (await smh.count()) {
                await pressQ(smh); await settle();
                const more = await page.evaluate(() => (document.querySelector('.vt-pick[data-key="SMH"] .vt-pick-more')?.textContent || ""));
                if (!/Since then/.test(more) || !/scale out .* — Sold the first third at \+1R/.test(more)) ovFail("the SMH card does not show its scale-out since the entry");
              }
              // why: the journal's attribution, worst first, four ways; the window control re-asks the route
              if (ov.why.source !== "journal") ovFail(`the Why panel's source is "${ov.why.source}"`);
              if (!/^The bot lost \$4,211 over the last 30 days\. Biggest drag: VRP \(sold premium\) \(signal\) −\$3,120 over 22 trades — sold puts into a falling market/.test(ov.why.say)) ovFail(`the Why panel's lead reads "${ov.why.say}"`);
              if (ov.why.groups.join(",") !== "signal,sector,instrument,timing") ovFail(`the attribution groups are ${ov.why.groups.join(",")}`);
              if (ov.why.firstRow !== "VRP (sold premium)") ovFail(`the worst signal row is "${ov.why.firstRow}"`);
              if (JSON.stringify(ov.why.win) !== JSON.stringify([["7d", "false"], ["30d", "true"], ["90d", "false"]])) ovFail(`the window control reads ${JSON.stringify(ov.why.win)}`);
              if (!/^Next 10 days: big funds are expected to be net sellers of stocks \(pressure −0\.42, confidence 60%\) — stocks ran ahead of bonds.* Flowing in: XLU, XLP; out: XLK\. The floor runs at ×0\.94 of its target because of it\.$/.test(ov.why.secs.flows || "")) ovFail(`the flows line reads "${ov.why.secs.flows}"`);
              if (ov.why.secs.regime !== "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) −0.21; breadth 38% −0.12. Applied to: the tier engine's exposure cap.") ovFail(`the regime line reads "${ov.why.secs.regime}"`);
              // re-pinned 2026-11-20 ([PRODUCT] the evidence sections): + base, rules after the pipeline's four (was "regime,flows,tax,signals")
              if (Object.keys(ov.why.secs).join(",") !== "regime,flows,tax,signals,base,rules") ovFail(`the context sections are ${Object.keys(ov.why.secs).join(",")} (expected the pipeline's order: regime, flows, tax, signals; then base, rules)`);
              if (ov.why.secs.tax !== "Tax filter: 1 of 3 trades checked in the last 7 days was skipped — the edge did not survive short-term tax (24%) against holding SPY. 1 loss could be harvested.") ovFail(`the tax line reads "${ov.why.secs.tax}"`);
              if (ov.why.secs.signals !== "Of 6 signals measured: 2 add return, 2 look like noise, 1 hurt, 1 unclear.") ovFail(`the signals line reads "${ov.why.secs.signals}"`);
              const w7 = page.locator(".vt-why-win button", { hasText: /^7d$/ });
              if (await w7.count()) {
                await w7.scrollIntoViewIfNeeded(); await pressQ(w7);
                let say = "";
                for (let i = 0; i < 40; i++) { say = await page.evaluate(() => (document.querySelector(".vt-why-say")?.textContent || "").trim()); if (/last 7 days/.test(say)) break; await page.waitForTimeout(100); }
                if (!/^The bot lost \$842 over the last 7 days\./.test(say)) ovFail(`after 7d the lead reads "${say}" (the 7-day fixture lost $842)`);
                const w30 = page.locator(".vt-why-win button", { hasText: /^30d$/ });
                if (await w30.count()) { await pressQ(w30); await settle(); }
              }
              // each section opens to its detail
              const fl = page.locator('.vt-why-sec[data-sec="flows"] .vt-why-sh > button:not(.vt-help)');
              if (await fl.count()) {
                await fl.scrollIntoViewIfNeeded(); await pressQ(fl); await settle();
                const det = await page.evaluate(() => (document.querySelector('.vt-why-sec[data-sec="flows"] .vt-why-sb')?.textContent || ""));
                if (!/pension rebalancing/.test(det) || !/XLU in/.test(det) || !/an estimate, not a measured flow/.test(det)) ovFail(`the flows detail reads "${det.slice(0, 160)}"`);
              }
              const rg = page.locator('.vt-why-sec[data-sec="regime"] .vt-why-sh > button:not(.vt-help)');
              if (await rg.count()) {
                await rg.scrollIntoViewIfNeeded(); await pressQ(rg); await settle();
                const det = await page.evaluate(() => { const b = document.querySelector('.vt-why-sec[data-sec="regime"] .vt-why-sb'); return { text: b?.textContent || "", bars: b ? b.querySelectorAll(".vt-why-bars i").length : 0 }; });
                if (!/VIX term structure −0\.21 — VIX above VIX3M/.test(det.text) || !/reduced · 0\.56/.test(det.text) || det.bars !== 12) ovFail(`the regime detail reads "${det.text.slice(0, 160)}" with ${det.bars} budget bars`);
              }
              // the chart answers a pointer: the readout names the day and both lines
              const svgLoc = page.locator(".vt-perf-plot svg").first();
              const box = await svgLoc.boundingBox();
              if (box) {
                await svgLoc.scrollIntoViewIfNeeded();
                const b2 = await svgLoc.boundingBox();
                if (vp.touch) await page.touchscreen.tap(b2.x + b2.width * 0.55, b2.y + b2.height * 0.4);
                else await page.mouse.move(b2.x + b2.width * 0.55, b2.y + b2.height * 0.4);
                await page.waitForTimeout(150);
                const rd = await page.evaluate(() => (document.querySelector(".vt-perf-read")?.textContent || ""));
                if (!/^[A-Z][a-z]{2} \d{1,2}portfolio [−+]?\d+\.\d\d% · before tax/.test(rd) || !/SPY [−+]?\d+\.\d\d%/.test(rd) || !/gap [−+]?\d+\.\d\d pt/.test(rd)) ovFail(`the chart's readout reads "${rd.slice(0, 160)}"`);
                if (!vp.touch) await page.mouse.move(2, 2);
              }
            } else {
              // idle: no history yet, an empty journal, no attribution or tax route, the old signal list — each designed
              if (ov.perf.state !== "empty" || !/not enough history yet/.test(ov.perf.empty)) ovFail(`the idle chart reads "${ov.perf.state}": "${ov.perf.empty.slice(0, 120)}"`);
              if (ov.picks.cards.length !== 10 || ov.picks.cards.some((c) => c.src !== "none")) ovFail("with an empty journal every card must say no reason was recorded");
              if (!/decision journal · 0 decisions/.test(ov.picks.hd)) ovFail(`the idle picks header reads "${ov.picks.hd}"`);
              if (ov.why.source !== "none" || !/loss attribution \(the decision journal\) is not on this server yet · the fill ledger is still computing/.test(ov.why.text)) ovFail(`the idle Why panel reads "${ov.why.text.slice(0, 200)}"`);
              if (ov.why.secs.tax !== "the tax filter is not on this server yet") ovFail(`the idle tax line reads "${ov.why.secs.tax}"`);
              if (!/old per-ticker feed/.test(ov.why.secs.signals || "")) ovFail(`the idle signals line reads "${ov.why.secs.signals}"`);
              if (ov.why.secs.flows !== "No flows estimate: the daily estimate has not been computed yet. Tier 3 computes it once per completed session.") ovFail(`the idle flows line (no estimate yet) reads "${ov.why.secs.flows}"`);
              if (ov.why.secs.regime !== "the regime risk budget is not on this server yet") ovFail(`the idle regime line reads "${ov.why.secs.regime}"`);
            }
            // ── EVIDENCE BLOCKS ([PRODUCT] 2026-11-20, UI coverage batch A; client/src/bot/evidence.tsx over six owner
            // routes): the closed trades and their CSV under the Book, the order history with each price-ladder step in
            // the Orders panel, the base holding against its own rule and what each blocking rule prevented as sections
            // of the Why panel, the engine-vs-old-rule line in the regime section, and the option-quote-feed line under
            // the Book's options. Every cell is checked against this page's fixtures; the idle page has none of the
            // routes, and each block must say so in words. Nothing may leak NaN / undefined / null or widen the page.
            {
              const evFail = (m) => checks.failures.push("evidence: " + m);
              const M = "−";
              const lotsFx = PAGE_FIXTURES["/api/bot/lots"], ordFx = PAGE_FIXTURES["/api/bot/order-history"], floorFx = PAGE_FIXTURES["/api/bot/floor-tracking"], rulesFx = PAGE_FIXTURES["/api/bot/rule-counterfactuals"];
              for (let i = 0; i < 40; i++) { if (await page.locator('.vt-ev[data-ev="lots"]:not([data-state="loading"])').count()) break; await page.waitForTimeout(100); }
              const read = () => page.evaluate(() => {
                const txt = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
                const lots = document.querySelector('.vt-ev[data-ev="lots"]');
                const orders = document.querySelector('.vt-ev[data-ev="orders"]');
                const wrapOver = (el) => (el ? [...el.querySelectorAll(".vt-exp-wrap")].map((w) => Math.round(w.getBoundingClientRect().right - window.innerWidth)).filter((v) => v > 1) : []);
                return {
                  lots: lots ? { state: lots.getAttribute("data-state"), text: txt(lots), rows: [...lots.querySelectorAll("tbody tr")].map((r) => [...r.querySelectorAll("td")].map(txt)), help: !!lots.querySelector('button.vt-help[data-help="closedLots"]'), btns: [...lots.querySelectorAll(".vt-ev-act .vt-bot-wbtn")].map(txt), wrapOver: wrapOver(lots), inBook: !!lots.closest('section.vt-bot-panel[data-panel="book"]') } : null,
                  orders: orders ? { state: orders.getAttribute("data-state"), text: txt(orders), rows: [...orders.querySelectorAll(".vt-ev-or")].map((r) => ({ sym: txt(r.querySelector(".sym")), ask: txt(r.querySelector(".ask")), out: txt(r.querySelector(".out")), step: txt(r.querySelector(".step")), over: Math.round(r.getBoundingClientRect().right - window.innerWidth) })), help: !!orders.querySelector('button.vt-help[data-help="orderHistory"]') } : null,
                  feed: (() => { const f = document.querySelector('[data-ev="options-feed"]'); return f ? { feed: f.getAttribute("data-feed"), text: txt(f), inBook: !!f.closest('section.vt-bot-panel[data-panel="book"]') } : null; })(),
                  openBtn: !!document.querySelector('[data-ev="orders-open"]'),
                  pageOver: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
                };
              });
              // the two Why sections' sentences, once their routes have answered (never the "Reading…" placeholder)
              const evSecs = async () => { let m = {}; for (let i = 0; i < 50; i++) { m = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll(".vt-why-sec")].map((x) => [x.getAttribute("data-sec"), (x.querySelector(".vt-why-ss")?.textContent || "").trim()]))); if (!/^Reading /.test(m.base || "") && !/^Reading /.test(m.rules || "")) break; await page.waitForTimeout(100); } return m; };
              const evShot = async (sel, tag) => { try { const el = page.locator(sel).first(); if (await el.count()) { await el.scrollIntoViewIfNeeded(); await el.screenshot({ path: path.join(OUT, `${name}-evidence-${tag}-${vp.w}.png`), animations: "disabled" }); } } catch (e) { console.log(`  (evidence screenshot ${tag} skipped: ${String(e.message || e).slice(0, 80)})`); } };
              await evShot('.vt-ev[data-ev="lots"]', "lots");
              let ev = await read();
              if (!ev.lots) evFail("no closed-trades block on the page");
              else if (name === "bot") {
                const want = [...lotsFx.rows].sort((a, b) => Date.parse(b.closedAt) - Date.parse(a.closedAt));
                console.log(`  evidence ${vp.w}px: lots ${ev.lots.state} · ${ev.lots.rows.length} rows · orders ${ev.orders ? ev.orders.state + " " + ev.orders.rows.length : "folded"} · feed ${ev.feed?.feed ?? "none"}`);
                if (ev.lots.state !== "ready" || ev.lots.rows.length !== 8) evFail(`the closed-trades block reads "${ev.lots.state}" with ${ev.lots.rows.length} rows (the route served ${want.length}; 8 show first)`);
                if (!ev.lots.inBook) evFail("the closed-trades block is not inside the Book panel");
                if (!ev.lots.help) evFail("no help glyph on the closed-trades block");
                if (/NaN|undefined|\bnull\b|Infinity/.test(ev.lots.text)) evFail(`a bare NaN / undefined / null in "${ev.lots.text.slice(0, 200)}"`);
                if (ev.lots.wrapOver.length) evFail(`the closed-trades table leaves the window by ${ev.lots.wrapOver.join(", ")}px (its wrapper must scroll instead)`);
                // the newest lot: CMCSA put sold at 0.79, bought back at 0.38 — +$41.00, +51.9 %
                const r0 = ev.lots.rows[0] || [];
                if (r0[0] !== "CMCSA 24P 18-Dec" || r0[1] !== "+$41.00" || r0[2] !== "+51.9%" || r0[5] !== "$0.79" || r0[6] !== "$0.38" || r0[7] !== "sold 1 ct, bought back") evFail(`the newest lot reads ${JSON.stringify(r0)}`);
                // a losing long: AMD 40 sh 162.70 -> 158.20 = -$180.00, -2.77 %
                const amd = ev.lots.rows.find((r) => r[0] === "AMD") || [];
                if (amd[7] !== "bought 40, sold" || amd[1] !== `${M}$180.00` || amd[2] !== `${M}2.8%`) evFail(`the AMD lot reads ${JSON.stringify(amd)}`);
                if (!/the newest 8 of 2,141 closed lots since May 7/.test(ev.lots.text)) evFail(`the closed-trades caption reads "${ev.lots.text.slice(-220)}"`);
                if (JSON.stringify(ev.lots.btns) !== JSON.stringify(["show 12", "download CSV · every lot", "bot log CSV"])) evFail(`the closed-trades controls read ${JSON.stringify(ev.lots.btns)}`);
                // "show 12" shows the rest; the lot whose opening fill predates the history says so with a dash
                const more = page.locator('.vt-ev[data-ev="lots"] .vt-ev-act button', { hasText: /^show 12$/ });
                if (await more.count()) {
                  await more.scrollIntoViewIfNeeded(); await pressQ(more); await settle();
                  const ev2 = await read();
                  if (ev2.lots.rows.length !== 12) evFail(`after "show 12" the table has ${ev2.lots.rows.length} rows`);
                  const gld = ev2.lots.rows.find((r) => r[0] === "GLD") || [];
                  if (gld[4] !== "—") evFail(`the lot opened before the fetched history shows held "${gld[4]}" (a dash: unknown, never 0)`);
                  if (ev2.pageOver > 1) evFail(`the page scrolls sideways by ${ev2.pageOver}px with 12 lots shown`);
                  const less = page.locator('.vt-ev[data-ev="lots"] .vt-ev-act button', { hasText: /^show 8$/ });
                  if (await less.count()) { await pressQ(less); await settle(); }
                }
                // the quote-feed line sits in the Book because the fixture book holds options
                if (!ev.feed || ev.feed.feed !== "indicative" || !/^option quotes are INDICATIVE: Alpaca's indicative feed/.test(ev.feed.text) || !ev.feed.inBook) evFail(`the option-quote-feed line reads ${JSON.stringify(ev.feed)}`);
                // the order history: in the stack it follows the working orders; on the desktop the one-line slot keeps
                // its height and the header's "history" control opens the panel
                if (vp.w >= 1180) {
                  if (ev.orders) evFail("the desktop Orders slot shows the history inline (it must stay one line)");
                  if (!ev.openBtn) evFail('no "history" control on the desktop Orders panel');
                  else { const ob = page.locator('[data-ev="orders-open"]').first(); await ob.scrollIntoViewIfNeeded(); await pressQ(ob); await settle(); for (let i = 0; i < 30; i++) { if (await page.locator('.vt-ev[data-ev="orders"][data-state="ready"]').count()) break; await page.waitForTimeout(100); } ev = await read(); }
                }
                if (!ev.orders || ev.orders.state !== "ready") evFail(`the order history reads "${ev.orders?.state ?? "absent"}"`);
                else {
                  if (ev.orders.rows.length !== ordFx.orders.length) evFail(`${ev.orders.rows.length} order rows (the route served ${ordFx.orders.length})`);
                  if (!ev.orders.help) evFail("no help glyph on the order history");
                  if (/NaN|undefined|\bnull\b|Infinity/.test(ev.orders.text)) evFail(`a bare NaN / undefined / null in "${ev.orders.text.slice(0, 200)}"`);
                  const aal = ev.orders.rows.filter((r) => r.sym === "AAL 11.5P 31-Dec");
                  // the ladder, newest first: filled at 0.31 (step 3), then the two cancelled steps before it
                  if (JSON.stringify(aal.map((r) => [r.ask, r.out, r.step])) !== JSON.stringify([["sell 1 ct · limit $0.31", "filled at $0.31", "step 3 of 3 · +$0.01"], ["sell 1 ct · limit $0.30", "cancelled after 20m, unfilled", "step 2 of 3 · +$0.01"], ["sell 1 ct · limit $0.29", "cancelled after 20m, unfilled", "step 1 of 3"]])) evFail(`the AAL ladder reads ${JSON.stringify(aal)}`);
                  const nv = ev.orders.rows.find((r) => r.sym === "NVDA");
                  if (!nv || nv.out !== "working" || nv.step !== "") evFail(`the working NVDA order reads ${JSON.stringify(nv)}`);
                  const vx = ev.orders.rows.find((r) => r.sym === "VXUS");
                  if (!vx || vx.ask !== "sell 120 · market" || vx.out !== "filled") evFail(`the market order with no fill price reads ${JSON.stringify(vx)} (it must say "filled", never a made-up price)`);
                  if (!/7 orders since [A-Z][a-z]{2} \d{1,2} · 4 filled · 3 not/.test(ev.orders.text)) evFail(`the order-history caption reads "${ev.orders.text.slice(-260)}"`);
                  if (ev.orders.rows.some((r) => r.over > 1)) evFail("an order row leaves the window");
                }
                await evShot('.vt-ev[data-ev="orders"]', "orders");
                if (vp.w >= 1180) { await page.keyboard.press("Escape"); await settle(); }
                // the Why panel's two evidence sections: the sentence, then the table behind it
                const secs = await evSecs();
                const baseSay = secs.base || "";
                const l = floorFx.latest;
                const pc = (v) => `${v < 0 ? M : v > 0 ? "+" : ""}${Math.abs(v).toFixed(2)}%`;
                if (baseSay !== `${new Date(l.day + "T12:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" })}: the SPY holding ${pc(l.floor_pct)} against ${pc(l.rule_pct)} for its rule (0.70× of SPY ${M}1.98%). It went in holding 0.79× against a target of 0.70×: inside the 0.10× band, where the rule does not trade. Over the last 9 sessions: ${M}0.46 pt against the rule.`) evFail(`the base-holding line reads "${baseSay}"`);
                if (secs.rules !== `3 rules blocked 76 name-days since Sep 3. Best measured: exposure cap (tier 1 puts, dropped): the 14 judged so far returned ${M}1.0% on average over 5 trading days, worse than the ${M}0.6% of everything scanned those days.`) evFail(`the rules line reads "${secs.rules}"`);
                const openSec = async (id) => { const b = page.locator(`.vt-why-sec[data-sec="${id}"] .vt-why-sh > button:not(.vt-help)`); if (!(await b.count())) return null; if ((await b.getAttribute("aria-expanded")) !== "true") { await b.scrollIntoViewIfNeeded(); await pressQ(b); await settle(); }
                  return page.evaluate((sid) => { const sb = document.querySelector(`.vt-why-sec[data-sec="${sid}"] .vt-why-sb`); const txt = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim(); return sb ? { text: txt(sb), rows: [...sb.querySelectorAll("tbody tr")].map((r) => [...r.querySelectorAll("td")].map(txt)), over: [...sb.querySelectorAll(".vt-exp-wrap")].map((w) => Math.round(w.getBoundingClientRect().right - window.innerWidth)).filter((v) => v > 1), cf: txt(sb.querySelector('[data-ev="regime-cf"]')), trigger: sb.querySelector("[data-trigger]")?.getAttribute("data-trigger") || null } : null; }, id); };
                const bs = await openSec("base");
                if (!bs) evFail("the base-holding section does not open");
                else {
                  if (bs.rows.length !== floorFx.sessions.length) evFail(`${bs.rows.length} session rows (the report has ${floorFx.sessions.length})`);
                  const s0 = floorFx.sessions[0], row0 = bs.rows[0] || [];
                  if (row0[1] !== pc(s0.floor_pct) || row0[2] !== pc(s0.rule_pct) || row0[3] !== `${M}0.18 pt` || row0[4] !== pc(s0.spy_pct) || row0[5] !== "0.70×" || row0[6] !== "0.79×") evFail(`the newest session row reads ${JSON.stringify(row0)}`);
                  if (bs.trigger !== "wait" || !/Rollback line not yet testable: tracking: 9 measured sessions/.test(bs.text)) evFail(`the rollback line reads "${bs.text.slice(0, 200)}" (${bs.trigger})`);
                  if (!/0 orders sent, 0 filled/.test(bs.text) || !/against the rule that day, from entering after the open/.test(bs.text)) evFail("the base-holding detail lacks its trading or entry line");
                  if (/NaN|undefined|\bnull\b|Infinity/.test(bs.text)) evFail(`a bare NaN / undefined / null in the base-holding detail: "${bs.text.slice(0, 200)}"`);
                  if (bs.over.length) evFail(`the base-holding table leaves the window by ${bs.over.join(", ")}px`);
                }
                await evShot('.vt-why-sec[data-sec="base"]', "base");
                const rs = await openSec("rules");
                if (!rs) evFail("the rules section does not open");
                else {
                  const groups = Object.values(rulesFx.gates).reduce((a, g) => a + g.groups.length, 0);
                  if (rs.rows.length !== groups) evFail(`${rs.rows.length} rule rows (the report has ${groups} groups)`);
                  const top = rs.rows[0] || [];
                  if (!/^Exposure cap\s*tier 1 puts, dropped$/.test(top[0] || "") || top[1] !== `${M}1.0%` || !(top[2] || "").startsWith(`${M}0.6%`) || top[3] !== "worse than the rest" || top[4] !== "14" || top[5] !== "45" || top[6] !== "not sized") evFail(`the best-measured rule row reads ${JSON.stringify(top)}`);
                  const sized = rs.rows.find((r) => /scanner picks, dropped/.test(r[0])) || [];
                  if (sized[6] !== `${M}$312` || sized[3] !== "better than the rest") evFail(`the sized cash-reserve row reads ${JSON.stringify(sized)}`);
                  if (!/Never fired in the logged period: drawdown kill \(tier engine\)\./.test(rs.text)) evFail("the rule that never fired is not named");
                  if (!/It is not a statistical test/.test(rs.text) || !/ESTIMATE/.test(rs.text)) evFail("the rules detail lacks its estimate / not-a-test caveat");
                  if (/A third limit the page does not print/.test(rs.text)) evFail("more than two of the report's limits are printed");
                  if (/NaN|undefined|\bnull\b|Infinity/.test(rs.text)) evFail(`a bare NaN / undefined / null in the rules detail: "${rs.text.slice(0, 200)}"`);
                  if (rs.over.length) evFail(`the rules table leaves the window by ${rs.over.join(", ")}px`);
                }
                await evShot('.vt-why-sec[data-sec="rules"]', "rules");
                const rg2 = await openSec("regime");
                if (!rg2 || rg2.cf !== "Against the old rule, the last 14 days: 34 of 75 entry decisions came out differently; 15 entries blocked (weak against the market 10, falling knife 8; one entry can trip several); 19 option entries cut in size; tier 1 sized at 0.50× against 0.66× under the old rule on " + new Date(PAGE_FIXTURES["/api/bot/regime-counterfactual"].by_day[1].date + "T12:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" }) + ". Counts only: what the blocked names went on to do is not in this report.") evFail(`the engine-vs-old-rule line reads "${rg2?.cf}"`);
                const after = await read();
                if (after.pageOver > 1) evFail(`the page scrolls sideways by ${after.pageOver}px with the evidence sections open`);
              } else {
                // idle: a server without the routes — each block is a designed sentence, the page is whole
                if (ev.lots.state !== "absent" || !/the closed-lot ledger is not on this server yet/.test(ev.lots.text)) evFail(`without the route the closed-trades block reads "${ev.lots.state}": "${ev.lots.text.slice(0, 160)}"`);
                if (ev.feed) evFail(`without the route an option-quote-feed line is shown: ${JSON.stringify(ev.feed)}`);
                const secs = await evSecs();
                if (secs.base !== "the base holding's tracking report is not on this server yet") evFail(`the idle base-holding line reads "${secs.base}"`);
                if (secs.rules !== "the rule report is not on this server yet") evFail(`the idle rules line reads "${secs.rules}"`);
                if (vp.w < 1180 && (!ev.orders || ev.orders.state !== "absent" || !/the order history is not on this server yet/.test(ev.orders.text))) evFail(`without the route the order history reads "${ev.orders?.state}": "${(ev.orders?.text || "").slice(0, 160)}"`);
                if (ev.pageOver > 1) evFail(`the idle page scrolls sideways by ${ev.pageOver}px`);
              }
            }
            // ── SELF-CHECK + ALERTS ([PRODUCT] 2026-11-20; client/src/bot/selfCheck.tsx over GET /api/bot/diagnostics and
            // GET /api/bot/notifications): two one-line blocks under the Issues list. The line states the count of
            // problems and warnings and what they set (size multiplier, score bar); opened, each finding names its
            // system, its severity and what it changes, and says when the loop is still on a different score bar. The
            // alerts line counts the unread and quotes the newest; opening the list marks them read (a POST). The idle
            // page has no diagnostics route and an empty alert list: each is a sentence.
            {
              const scFail = (m) => checks.failures.push("self-check: " + m);
              const scShot = async (sel, tag) => { try { const el = page.locator(sel).first(); if (await el.count()) { await el.scrollIntoViewIfNeeded(); await el.screenshot({ path: path.join(OUT, `${name}-evidence-${tag}-${vp.w}.png`), animations: "disabled" }); } } catch (e) { console.log(`  (self-check screenshot ${tag} skipped: ${String(e.message || e).slice(0, 80)})`); } };
              for (let i = 0; i < 50; i++) { const t = await page.evaluate(() => (document.querySelector('.vt-sc[data-sc="check"] .vt-sc-s')?.textContent || "")); if (t && !/^Running /.test(t)) break; await page.waitForTimeout(100); }
              const readSc = () => page.evaluate(() => {
                const txt = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
                const one = (id) => { const el = document.querySelector(`.vt-sc[data-sc="${id}"]`); if (!el) return null; const btn = el.querySelector(".vt-sc-h > button:not(.vt-help)"); const r = btn?.getBoundingClientRect();
                  return { title: txt(btn?.querySelector(".t")), say: txt(el.querySelector(".vt-sc-s")), cls: el.querySelector(".vt-sc-s")?.className || "", open: el.getAttribute("data-open") === "true", body: txt(el.querySelector(".vt-sc-b")), btnH: r ? Math.round(r.height) : 0, disabled: !!btn?.disabled,
                    inIssues: !!el.closest('section.vt-bot-panel[data-panel="issues"]'), help: !!el.querySelector("button.vt-help"), problems: [...el.querySelectorAll('.vt-sc-l[data-list="problems"] li')].map(txt), warnings: [...el.querySelectorAll('.vt-sc-l[data-list="warnings"] li')].map(txt),
                    alerts: [...el.querySelectorAll(".vt-sc-ar")].map((r2) => ({ k: txt(r2.querySelector(".k")), m: txt(r2.querySelector(".m")), unread: r2.getAttribute("data-unread") === "true", over: Math.round(r2.getBoundingClientRect().right - window.innerWidth) })) }; };
                return { check: one("check"), alerts: one("alerts"), pageOver: Math.max(0, document.documentElement.scrollWidth - window.innerWidth) };
              });
              const openSc = async (id) => { const b = page.locator(`.vt-sc[data-sc="${id}"] .vt-sc-h > button:not(.vt-help)`).first(); await b.scrollIntoViewIfNeeded(); await pressQ(b); await settle(); };
              let sc = await readSc();
              if (!sc.check || !sc.alerts) scFail(`the Issues panel lacks the ${!sc.check ? "self-check" : "alerts"} block`);
              else {
                console.log(`  self-check ${vp.w}px: "${sc.check.say.slice(0, 90)}" · alerts "${sc.alerts.title}"`);
                if (!sc.check.inIssues || !sc.alerts.inIssues) scFail("the blocks are not inside the Issues panel");
                if (!sc.check.help || !sc.alerts.help) scFail("a block lacks its help glyph");
                if (name === "bot") {
                  if (sc.check.say !== "Self-check: 2 problems, 1 warning · new positions at ×0.70 of normal size · score bar 75") scFail(`the self-check line reads "${sc.check.say}"`);
                  if (!/vt-bot-r/.test(sc.check.cls)) scFail(`a high-severity finding is not shown in the alarm colour (${sc.check.cls})`);
                  if (vp.touch && (sc.check.btnH < 44 || sc.alerts.btnH < 44)) scFail(`touch targets under 44px: self-check ${sc.check.btnH}px, alerts ${sc.alerts.btnH}px`);
                  await openSc("check");
                  sc = await readSc();
                  if (!sc.check.open || sc.check.problems.length !== 2 || sc.check.warnings.length !== 1) scFail(`opened, the self-check lists ${sc.check.problems.length} problems and ${sc.check.warnings.length} warnings (2 and 1 served)`);
                  if (sc.check.problems[0] !== "Data freshnesshigh Critical data sources stale: ['macro', 'insider'] → new positions are sized smaller") scFail(`the first finding reads "${sc.check.problems[0]}"`);
                  if (sc.check.problems[1] !== "Modelmedium Low win rate: 36% over 58 trades → only higher-scoring picks are taken") scFail(`the second finding reads "${sc.check.problems[1]}"`);
                  if (sc.check.warnings[0] !== "Disk Cache using 410MB of 500MB limit") scFail(`the warning reads "${sc.check.warnings[0]}"`);
                  // the fixture's loop is on a score bar of 65 while the fresh check says 75: the difference is said
                  if (!/The loop is using a score bar of 65 until it next applies the check\./.test(sc.check.body)) scFail("the applied-score-bar note is missing");
                  if (!/Housekeeping it did: Removed 12 expired cache files/.test(sc.check.body) || !/a fresh run, not the loop's last one/.test(sc.check.body)) scFail(`the self-check detail lacks its housekeeping or source line: "${sc.check.body.slice(-220)}"`);
                  if (/NaN|undefined|\bnull\b|Infinity/.test(sc.check.body + sc.check.say)) scFail("a bare NaN / undefined / null in the self-check");
                  await scShot('.vt-sc[data-sc="check"]', "selfcheck");
                  if (sc.alerts.title !== "Alerts · 2 new" || !/^3 alerts · 2 new · newest \d+[smh] ago: Daily loss: -3\.1%\. Trading halted\.$/.test(sc.alerts.say)) scFail(`the alerts block reads "${sc.alerts.title}" / "${sc.alerts.say}"`);
                  if (!/vt-bot-o/.test(sc.alerts.cls)) scFail(`an unread alert is not shown in the warning colour (${sc.alerts.cls})`);
                  let posted = 0;
                  const onReq = (rq) => { if (rq.method() === "POST" && /\/api\/bot\/notifications\/read$/.test(new URL(rq.url()).pathname)) posted += 1; };
                  page.on("request", onReq);
                  await openSc("alerts");
                  await page.waitForTimeout(400);
                  page.off("request", onReq);
                  sc = await readSc();
                  if (posted !== 1) scFail(`opening the alerts sent ${posted} mark-read request(s) (exactly 1)`);
                  if (JSON.stringify(sc.alerts.alerts.map((r) => [r.k, r.m, r.unread])) !== JSON.stringify([["alert", "Daily loss: -3.1%. Trading halted.", true], ["trade", "OPTIONS: SELL 1x AAL $11.5 PUT (2026-12-31)", true], ["system", "Bot activated — scanning for opportunities", false]])) scFail(`the alert rows read ${JSON.stringify(sc.alerts.alerts)}`);
                  if (sc.alerts.alerts.some((r) => r.over > 1)) scFail("an alert row leaves the window");
                  if (!/cleared by a restart or a deploy/.test(sc.alerts.body)) scFail("the alerts detail does not say the list is cleared by a restart");
                  await scShot('.vt-sc[data-sc="alerts"]', "alerts");
                  if (sc.pageOver > 1) scFail(`the page scrolls sideways by ${sc.pageOver}px with both blocks open`);
                  await openSc("alerts"); await openSc("check");
                } else {
                  if (sc.check.say !== "the bot's self-check is not on this server yet" || !sc.check.disabled) scFail(`without the route the self-check reads "${sc.check.say}" (disabled ${sc.check.disabled})`);
                  if (sc.alerts.title !== "Alerts" || sc.alerts.say !== "No alerts since the server last restarted." || !sc.alerts.disabled) scFail(`with an empty list the alerts block reads "${sc.alerts.title}" / "${sc.alerts.say}" (disabled ${sc.alerts.disabled})`);
                }
              }
            }
            // ── SAFETY LIMITS ([PRODUCT] 2026-11-20; client/src/bot/selfCheck.tsx SafetyLimits over GET
            // /api/bot/safety-limits): one line under the Issues list naming what is tripped or near and the three
            // headline readings; opened, one row per guard: its name, a bar of the share of its limit used, the guard's
            // own reading and limit, what it does at the limit; then the limits with no reading, the kill history and
            // the source line. The bars are proportional to the served figures; nothing leaves the panel or the window.
            {
              const lmFail = (m) => checks.failures.push("safety limits: " + m);
              const lmSel = '.vt-sc[data-sc="limits"]';
              for (let i = 0; i < 50; i++) { const t = await page.evaluate((s) => (document.querySelector(`${s} .vt-sc-s`)?.textContent || ""), lmSel); if (t && !/^Reading /.test(t)) break; await page.waitForTimeout(100); }
              const readLm = () => page.evaluate((s) => {
                const txt = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
                const el = document.querySelector(s);
                if (!el) return null;
                const btn = el.querySelector(".vt-sc-h > button:not(.vt-help)"); const br = btn?.getBoundingClientRect();
                const panel = el.closest('section.vt-bot-panel[data-panel="issues"]'); const pr = panel?.getBoundingClientRect();
                return { title: txt(btn?.querySelector(".t")), say: txt(el.querySelector(".vt-sc-s")), cls: el.querySelector(".vt-sc-s")?.className || "", open: el.getAttribute("data-open") === "true", disabled: !!btn?.disabled,
                  btnH: br ? Math.round(br.height) : 0, inIssues: !!panel, help: !!el.querySelector("button.vt-help"), body: txt(el.querySelector(".vt-sc-b")),
                  rows: [...el.querySelectorAll(".vt-sc-lim")].map((r) => { const bar = r.querySelector(".bar"), fill = r.querySelector(".bar i"); const rr = r.getBoundingClientRect();
                    return { id: r.getAttribute("data-limit"), state: r.getAttribute("data-state"), n: txt(r.querySelector(".n")), u: txt(r.querySelector(".u")), r: txt(r.querySelector(".r")), e: txt(r.querySelector(".e")),
                      fill: bar && fill ? Math.round(fill.getBoundingClientRect().width / Math.max(1, bar.getBoundingClientRect().width) * 100) : -1, barH: bar ? Math.round(bar.getBoundingClientRect().height) : 0,
                      over: Math.round(rr.right - window.innerWidth), outOfPanel: pr ? Math.round(rr.right - pr.right) : 0, uColor: getComputedStyle(r.querySelector(".u")).color, nColor: getComputedStyle(r.querySelector(".n")).color,
                      nameCut: r.querySelector(".n").scrollWidth > r.querySelector(".n").clientWidth + 1, tColor: r.querySelector(".r .t") ? getComputedStyle(r.querySelector(".r .t")).color : null }; }),
                  others: [...el.querySelectorAll('.vt-sc-l[data-list="others"] li')].map(txt), peaks: txt(el.querySelector('[data-lim="peaks"]')), events: txt(el.querySelector('[data-lim="events"]')), src: txt(el.querySelector(".vt-sc-src")),
                  pageOver: Math.max(0, document.documentElement.scrollWidth - window.innerWidth) };
              }, lmSel);
              let lm = await readLm();
              if (!lm) lmFail("the Issues panel lacks the safety-limits block");
              else {
                console.log(`  safety limits ${vp.w}px: "${lm.say.slice(0, 110)}"`);
                if (!lm.inIssues) lmFail("the block is not inside the Issues panel");
                if (!lm.help || lm.title !== "Safety limits") lmFail(`the block's head reads "${lm.title}" (help glyph ${lm.help})`);
                if (name === "bot") {
                  if (lm.say !== "Tripped: drawdown kill · drawdown −9.53% of −10% · worst today −2.41% of −3% · losing streak 2 of 5") lmFail(`the line reads "${lm.say}"`);
                  if (!/vt-bot-r/.test(lm.cls)) lmFail(`a tripped guard is not shown in the alarm colour (${lm.cls})`);
                  if (vp.touch && lm.btnH < 44) lmFail(`the control is ${lm.btnH}px tall on touch (44)`);
                  const lmBtn = page.locator(`${lmSel} .vt-sc-h > button:not(.vt-help)`).first();
                  await lmBtn.scrollIntoViewIfNeeded(); await pressQ(lmBtn); await settle();
                  lm = await readLm();
                  const want = [
                    ["kill", "hit", "Drawdown kill", "tripped", "−9.53% from its $110,727 peak · limit −10% ($99,654) · Max drawdown -10.0% from peak", 95],
                    ["engineKill", "ok", "Engine kill switch", "48% of the way", "−9.53% from its $110,727 peak · limit −20% ($88,582)", 48],
                    ["engineHalt", "ok", "Engine drawdown halt", "50% of the way", "−8.93% from its $110,000 peak · limit −18% ($90,200)", 50],
                    ["day", "near", "Daily loss", "80% of the way", "worst today −2.41% · limit −3%", 80],
                    ["streak", "ok", "Losing streak", "40% of the way", "2 losses in a row (last 2026-11-19) · limit 5 in a row", 40],
                  ];
                  if (!lm.open || lm.rows.length !== want.length) lmFail(`opened, the block lists ${lm.rows.length} guards (${want.length} served)`);
                  else want.forEach((w, i) => {
                    const r = lm.rows[i];
                    if (r.id !== w[0] || r.state !== w[1] || r.n !== w[2] || r.u !== w[3] || r.r !== w[4]) lmFail(`guard ${i + 1} reads ${JSON.stringify([r.id, r.state, r.n, r.u, r.r])}`);
                    // the bar is the share of the limit used, drawn from the served reading (±2 points of rounding)
                    if (Math.abs(r.fill - w[5]) > 2) lmFail(`${w[2]}: the bar is ${r.fill}% full for ${w[5]}% of the limit`);
                    if (r.barH < 3) lmFail(`${w[2]}: the bar is ${r.barH}px tall`);
                    if (r.over > 1 || r.outOfPanel > 1) lmFail(`${w[2]}: the row leaves the ${r.over > 1 ? "window" : "panel"} by ${Math.max(r.over, r.outOfPanel)}px`);
                    if (!/^at the limit: /.test(r.e)) lmFail(`${w[2]}: no line for what the guard does ("${r.e}")`);
                    // first run, 1440px: "Engine drawdown halt" was cut to "Engine drawdown …" in the narrow column
                    if (r.nameCut) lmFail(`${w[2]}: the guard's name is cut with an ellipsis`);
                  });
                  if (lm.rows.length === want.length) {
                    if (lm.rows[0].e !== "at the limit: no new entries · protective exits keep running (stops, option exits; the floor may only reduce) · open orders cancelled") lmFail(`the kill's consequence reads "${lm.rows[0].e}" (it must be the shared halt wording)`);
                    if (lm.rows[3].e !== "at the limit: blocks new entries for the rest of the day · closes nothing") lmFail(`the daily limit's consequence reads "${lm.rows[3].e}"`);
                    // first run: the tripped guard's own reason was drawn in the body's bold white; it is the alarm colour
                    if (lm.rows[0].tColor !== lm.rows[0].uColor) lmFail(`the tripped guard's reason is ${lm.rows[0].tColor}, its "tripped" label ${lm.rows[0].uColor}`);
                    // state is carried by colour as well as words: the near and tripped rows differ from a calm one
                    if (lm.rows[0].uColor === lm.rows[1].uColor || lm.rows[3].uColor === lm.rows[1].uColor || lm.rows[0].uColor === lm.rows[3].uColor) lmFail(`tripped / near / calm share a colour: ${lm.rows[0].uColor} / ${lm.rows[3].uColor} / ${lm.rows[1].uColor}`);
                  }
                  if (lm.others.length !== 4 || lm.others[0] !== "Volatility stop: tiers 2 to 4 take no new entries while the VXX ratio is 1.40 or higher.") lmFail(`the limits without a reading read ${JSON.stringify(lm.others)}`);
                  if (lm.peaks !== "The drawdown guards disagree by 0.6 points: each measures from its own peak.") lmFail(`the differing-peaks note reads "${lm.peaks}"`);
                  if (lm.events !== "Kill switch history: 2026-11-06 drawdown kill at $89,371 (−20.02%)") lmFail(`the history line reads "${lm.events}"`);
                  if (!/^each guard's own record, measured against one account value \(\$100,172\) · read \d+[smh] ago · the limits are settings: shown here, never changed from this page$/.test(lm.src)) lmFail(`the source line reads "${lm.src}"`);
                  if (/NaN|undefined|\bnull\b|Infinity/.test(lm.body + lm.say)) lmFail("a bare NaN / undefined / null in the block");
                  if (lm.pageOver > 1) lmFail(`the page scrolls sideways by ${lm.pageOver}px with the block open`);
                  try { const el = page.locator(lmSel).first(); await el.scrollIntoViewIfNeeded(); await el.screenshot({ path: path.join(OUT, `${name}-evidence-limits-${vp.w}.png`), animations: "disabled" }); } catch { /* the assertions above are the gate */ }
                  await pressQ(lmBtn); await settle();
                  if ((await readLm()).open) lmFail("the block does not close again");
                } else if (lm.say !== "the safety-limits read is not on this server yet" || !lm.disabled) lmFail(`without the route the block reads "${lm.say}" (disabled ${lm.disabled})`);
              }
            }
            // ── DAILY LINE CARD ([PRODUCT] 2026-11-06; server/dailyLine.ts via GET /api/bot/daily-line, overview.tsx
            // DailyLineBody): one row per stored line, newest first, every cell the line's own field — the account's day,
            // SPY's, the gap in points (the gap must equal account − SPY as printed), the floor, the option total with its
            // fills / direction split, the 09:30 holds; how old the newest line is and where today's line stands above the
            // table; a partial line carries its chip; the newest line's missing pieces are spelled out on the card (no hover
            // on touch). The phone folds each row onto two lines; nothing is cut, nothing leaks NaN / undefined / null.
            {
              const dlFx = PAGE_FIXTURES["/api/bot/daily-line"];
              for (let i = 0; i < 40; i++) { if (await page.locator('.vt-dl[data-state="ready"], .vt-dl[data-state="empty"], .vt-dl[data-state="error"]').count()) break; await page.waitForTimeout(100); }
              const dl = await page.evaluate(() => {
                const sec = document.querySelector('section.vt-bot-panel[data-panel="daily"]');
                const card = document.querySelector(".vt-dl");
                if (!sec || !card) return null;
                const rows = [...card.querySelectorAll(".vt-dl-r:not(.vt-dl-h)")].map((r) => {
                  const cell = (c) => r.querySelector(`.${c}`);
                  const top = (c) => Math.round(cell(c)?.getBoundingClientRect().top ?? -1);
                  return { day: r.getAttribute("data-day"), partial: r.getAttribute("data-partial") === "true", basis: cell("a")?.getAttribute("data-basis") || null,
                    d: (cell("d")?.textContent || "").trim(), a: (cell("a")?.textContent || "").trim(), s: (cell("s")?.textContent || "").trim(), g: (cell("g")?.textContent || "").trim(),
                    f: (cell("f")?.textContent || "").trim(), o: (cell("o")?.textContent || "").trim(), b: (cell("b")?.textContent || "").trim(),
                    twoLines: top("f") > top("d") + 4, cut: [...r.querySelectorAll(":scope > span:not(.o)")].filter((x) => x.scrollWidth - x.clientWidth > 1).map((x) => x.textContent) };
                });
                return { state: card.getAttribute("data-state"), text: (card.textContent || "").replace(/\s+/g, " ").trim(), hd: (card.querySelector(".vt-dl-hd")?.textContent || "").replace(/\s+/g, " ").trim(),
                  today: card.querySelector("[data-today]")?.getAttribute("data-today") || null, notes: [...card.querySelectorAll(".vt-dl-notes > span")].map((x) => (x.textContent || "").trim()),
                  rows, help: !!sec.querySelector('button.vt-help[data-help="dailyLine"]'), over: Math.max(0, sec.scrollHeight - sec.clientHeight), headCut: [...card.querySelectorAll(".vt-dl-h > span")].filter((x) => x.scrollWidth - x.clientWidth > 1).map((x) => x.textContent) };
              });
              const dlFail = (m) => checks.failures.push("daily-line: " + m);
              if (!dl) dlFail("no daily line card on the overview");
              else {
                console.log(`  daily line ${vp.w}px: ${dl.state} · ${dl.rows.length} rows · "${dl.hd.slice(0, 140)}"`);
                const want = (dlFx?.recent ?? []).slice(0, 10);
                if (dl.state !== "ready") dlFail(`the card reads "${dl.state}" with ${want.length} lines served`);
                if (dl.rows.length !== want.length) dlFail(`${dl.rows.length} rows (the route served ${want.length})`);
                if (/NaN|undefined|\bnull\b|Infinity/.test(dl.text)) dlFail(`a bare NaN / undefined / null in "${dl.text.slice(0, 200)}"`);
                if (!dl.help) dlFail("no help glyph on the card");
                if (dl.over > 1) dlFail(`the panel clips ${dl.over}px of its own content`);
                if (dl.headCut.length) dlFail(`a column header is cut short: ${dl.headCut.join(", ")}`);
                const M = "\u2212";
                const pct = (v) => `${v < 0 ? M : v > 0 ? "+" : ""}${Math.abs(v).toFixed(2)}%`;
                const pp = (v) => `${v < 0 ? M : v > 0 ? "+" : ""}${Math.abs(v).toFixed(2)} pp`;
                const num = (t) => Number(String(t).replace(/\u2212/g, "-").replace(/[%+*]|\s*pp$/g, ""));
                // [RULE-REVIEW] 2026-11-06: a line without a source predates the close reading — its day ended when written
                const writeTimeOf = (fx) => fx.account_vs_spy.day.account_source !== "close";
                dl.rows.forEach((r, i) => {
                  const fx = want[i];
                  if (!fx) return;
                  if (r.day !== fx.day) dlFail(`row ${i} is ${r.day} (the route's ${i}th newest is ${fx.day})`);
                  const vs = fx.account_vs_spy.day;
                  const acctWant = `${pct(vs.account_pct)}${writeTimeOf(fx) ? "*" : ""}`;
                  if (r.a !== acctWant || r.s !== pct(vs.spy_pct) || r.g !== pp(vs.gap_pp)) dlFail(`${r.day} reads ${r.a} / ${r.s} / ${r.g} (the line: ${acctWant} / ${pct(vs.spy_pct)} / ${pp(vs.gap_pp)})`);
                  if (r.basis !== (writeTimeOf(fx) ? "write_time" : "close")) dlFail(`${r.day}: the account cell's basis is "${r.basis}" (the line: ${vs.account_source ?? "none, so write_time"})`);
                  if (Math.abs(num(r.a) - num(r.s) - num(r.g)) > 0.011) dlFail(`${r.day}: ${r.a} − ${r.s} is not ${r.g}`);
                  if (r.f !== `${fx.floor.exposure_x.toFixed(2)}×${fx.floor.target_x !== null ? ` → ${fx.floor.target_x.toFixed(2)}×` : ""}`) dlFail(`${r.day}: the floor reads "${r.f}"`);
                  if (fx.options.total === null ? !/^n\/a fills/.test(r.o) : !/^[−+]?\$[\d,]+ fills [−+]?\$[\d,]+ · direction [−+]?\$[\d,]+ \(\d+\/\d+ priced\)$/.test(r.o)) dlFail(`${r.day}: the options cell reads "${r.o}"`);
                  if (r.b !== String(fx.opening_window.total)) dlFail(`${r.day}: the 09:30 holds read "${r.b}" (the line: ${fx.opening_window.total})`);
                  if (r.partial !== (fx.missing.length > 0)) dlFail(`${r.day}: the partial chip is ${r.partial ? "on" : "off"} for ${fx.missing.length} missing input(s)`);
                  if (r.cut.length) dlFail(`${r.day}: a cell is cut short (${r.cut.join(", ")})`);
                  if (r.twoLines !== (vp.w < 640)) dlFail(`${r.day}: the row is on ${r.twoLines ? "two lines" : "one line"} at ${vp.w}px (the phone folds it, nothing else does)`);
                });
                if (!/^newest [A-Z][a-z]{2} [A-Z][a-z]{2} \d{1,2} · written \d\d:\d\d ET, \d+[smhd] ago/.test(dl.hd)) dlFail(`the header does not say which line is newest and how old it is — "${dl.hd}"`);
                if (!/Today's line is written after the close, from \d\d:\d\d ET/.test(dl.hd) || dl.today !== "ok") dlFail(`the header does not say where today's line stands — "${dl.hd}"`);
                if (!/Account = equity at the close \(the last read before the bell\) vs the prior close \(Alpaca last_equity\)/.test(dl.text) || !/\(approximate\)/.test(dl.text) || !/direction = the rest \(an estimate\)/.test(dl.text)) dlFail("the card does not label what is an estimate");
                const starClause = /\* = equity when the line was written \(no close reading; after-hours marks can move it\)/.test(dl.text);
                if (starClause !== want.some(writeTimeOf)) dlFail(`the "*" footnote is ${starClause ? "shown" : "missing"} with ${want.filter(writeTimeOf).length} write-time line(s) shown`);
                if (name === "bot") {
                  if (dl.notes.length) dlFail(`the newest line is complete, yet the card notes: ${dl.notes.join(" | ")}`);
                  if (!dl.rows.some((r) => r.partial)) dlFail("the partial line carries no chip");
                } else if (!dl.notes.some((t) => /options total n\/a — no previous line within 7 days/.test(t))) dlFail(`the newest line's missing option total is not spelled out — ${JSON.stringify(dl.notes)}`);
              }
            }
            await page.evaluate(() => document.querySelector(".vt-bot-stack")?.scrollTo(0, 0));
          } catch (e) {
            ovFail(`driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 400)}`);
            console.log(`  overview ${vp.w}px: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
          }
          try {
            // the switch opens the cockpit: every engine panel, as before; the checks below run there
            const ck = page.locator('.vt-bot-modebar button[data-view="cockpit"]').first();
            if (!(await ck.count())) ovFail("no Cockpit control");
            else {
              await pressQ(ck);
              for (let i = 0; i < 40; i++) { if (await page.locator('.vt-bot-page[data-mode="cockpit"] section.vt-bot-panel[data-panel="watch"]').count()) break; await page.waitForTimeout(100); }
              await page.waitForTimeout(500);
              const ckIds = await page.$$eval("section.vt-bot-panel[data-panel]", (els) => els.map((e) => e.getAttribute("data-panel")));
              for (const id of ["stages", "issues", "watch", "model", "chart", "book", "orders", "timeline"]) if (!ckIds.includes(id)) ovFail(`the cockpit lacks its ${id} panel`);
              if (ckIds.some((id) => id === "perf" || id === "why" || id === "picks" || id === "daily")) ovFail(`the cockpit shows an overview panel (${ckIds.join(",")})`);
              const pressed = await page.evaluate(() => document.querySelector('.vt-bot-modebar button[data-view="cockpit"]')?.getAttribute("aria-pressed"));
              if (pressed !== "true") ovFail("the Cockpit control does not read pressed in the cockpit");
              const saved = await page.evaluate(() => { try { return localStorage.getItem("vt-bot-mode"); } catch { return "storage unavailable"; } });
              if (saved !== "cockpit") ovFail(`the view is not remembered (localStorage vt-bot-mode = ${saved})`);
              await page.screenshot({ path: path.join(OUT, `${name}-cockpit-${vp.w}.png`), animations: "disabled" });
            }
          } catch (e) {
            ovFail(`cockpit switch: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 400)}`);
          }
        }
        const panels = await page.$$eval("section.vt-bot-panel[data-panel]", (els) => [...new Set(els.map((e) => e.getAttribute("data-panel")))]);
        if (!panels.length) fail("no panels rendered");
        // POP-OUT (slice D, 2026-10-21): the chart's "own window" control
        // opens /app#/bot/pane/chart as a named popup. window.open is stubbed:
        // a blocked popup (null) leaves the panel docked and says so; an
        // opened one turns the slot into a ghost whose "bring back" closes
        // the window and restores the panel. Not offered on touch.
        if (!vp.touch) {
          await page.evaluate(() => { window.__vtOpened = []; window.__vtWin = null; window.open = (u, n, f) => { window.__vtOpened.push({ u: String(u), n, f }); return window.__vtWin; }; });
          const popBtn = ctl("chart", "Open in its own browser window");
          if (!(await popBtn.count())) fail("chart: no 'Open in its own browser window' control");
          else {
            await press(popBtn); await settle();
            if (!(await page.locator(".vt-bot-popnote").count())) fail("pop-out: a blocked popup shows no note");
            const opened = await page.evaluate(() => window.__vtOpened);
            if (opened.length !== 1 || !opened[0].u.endsWith("#/bot/pane/chart") || opened[0].n !== "vt-pane-chart" || !/popup=yes,width=\d+,height=\d+/.test(opened[0].f)) fail(`pop-out: window.open got ${JSON.stringify(opened)}`);
            if ((await page.locator('section.vt-bot-panel[data-panel="chart"]').count()) !== 1) fail("pop-out: a blocked popup removed the panel");
            await page.evaluate(() => { window.__vtWin = { closed: false, focus() {}, close() { this.closed = true; } }; });
            await press(popBtn); await settle();
            if (await page.locator('section.vt-bot-panel[data-panel="chart"]').count()) fail("pop-out: the panel stayed docked after opening its window");
            const ghost = page.locator('.vt-bot-ghost[data-popped="true"]');
            if (!(await ghost.count())) fail("pop-out: no ghost slot for the popped panel");
            const back = ghost.locator("button").first();
            if (!(await back.count())) fail("pop-out: the ghost has no bring-back control");
            else { await press(back); await settle(); }
            if ((await page.locator('section.vt-bot-panel[data-panel="chart"]').count()) !== 1) fail("pop-out: bring back did not restore the panel");
            if (!(await page.evaluate(() => !!(window.__vtWin && window.__vtWin.closed)))) fail("pop-out: bring back did not close the window");
            await page.evaluate(() => { const b = document.querySelector(".vt-bot-popnote button"); if (b) b.click(); });
          }
        }
        // every docked panel has a real height (a collapsed one is invisible
        // and its controls sit under the next panel)
        for (const [id, h] of await page.$$eval("section.vt-bot-panel[data-panel]", (els) => els.map((e) => [e.getAttribute("data-panel"), Math.round(e.getBoundingClientRect().height)]))) {
          if (h < 40) fail(`${id}: panel is ${h}px tall`);
        }
        // ── KPI STRIP ([RULE-REVIEW] 2026-10-29, the Expectancy tile): the
        // first tile's headline is the BROKER expectancy per closed lot (the
        // sleeve ledger's fill walk, all sleeves the bot trades) and the bot-
        // log number sits under it with its staleness and exclusions. The
        // strip may not grow for it: its height is pinned per width to the
        // pre-change readings (the fixed 86 px grid row at the desktop tier,
        // content height in the stack) and no tile may clip its own content.
        // Re-pin only with a deliberate layout change, never to pass.
        {
          // the pre-change readings (2026-10-29 baseline): the idle page's first row at 768 sits on the 72 px tile minimum;
          // its 390 pair row read 319 before and 310 after — the win-rate tile's "—" (no broker ledger yet) takes one line
          // where "27.8% / 36.2%" wrapped to two; nothing grew
          // RE-PINNED 2026-11-03 for the TODAY tile (B1): seven tiles — 1440 stays the l tier's one 86 px row (six tiles:
          // the Session tile folds into the Now tile there), 1920 is the xl tier's 88 px row of seven, 768 three rows
          // (the Now tile spans the third), 390 four rows (Now spans the fourth). The stack readings are the post-change
          // measurements.
          const KPI_STRIP_PX = name === "bot" ? { 390: 427, 768: 254, 1440: 86, 1920: 88 } : { 390: 419, 768: 244, 1440: 86 };
          const kpi = await page.evaluate(() => {
            const strip = document.querySelector(".vt-bot-kpi");
            if (!strip) return null;
            const tiles = [...strip.querySelectorAll(":scope > .vt-bot-panel")].map((t) => ({ w: Math.round(t.getBoundingClientRect().width), h: Math.round(t.getBoundingClientRect().height), over: Math.max(0, t.scrollHeight - t.clientHeight) }));
            const first = strip.querySelector(":scope > .vt-bot-panel");
            const second = strip.querySelectorAll(":scope > .vt-bot-panel")[1];
            const lines = first ? [...first.querySelectorAll(".vt-kpi-edge-cap > span, .vt-kpi-edge-log > span")].map((s) => ({ text: s.textContent || "", cut: s.scrollWidth - s.clientWidth > 1 })) : [];
            const eb = first ? first.querySelector(".vt-bot-eb") : null;
            return { h: Math.round(strip.getBoundingClientRect().height), tiles, first: first ? first.textContent || "" : "", second: second ? second.textContent || "" : "", eyebrow: eb ? eb.textContent || "" : "", eyebrowLines: eb ? Math.round(eb.getBoundingClientRect().height / 13.2) : 0, lines };
          });
          if (!kpi) fail("kpi: no strip rendered");
          else {
            console.log(`  kpi strip ${vp.w}px: ${kpi.h}px (tiles w×h ${kpi.tiles.map((t) => `${t.w}×${t.h}`).join(" ")}; eyebrow ${kpi.eyebrowLines} line(s))`);
            const want = KPI_STRIP_PX[vp.w];
            if (want && Math.abs(kpi.h - want) > 1) fail(`kpi: the strip is ${kpi.h}px tall at ${vp.w} (pinned ${want}px — the Expectancy tile may not grow the strip)`);
            // the two tiles this change touches may not clip their own content (the vs-SPY tile's 768 overflow predates it)
            kpi.tiles.slice(0, 2).forEach((t, i) => { if (t.over > 1) fail(`kpi: tile ${i + 1} clips ${t.over}px of its own content`); });
            // the Expectancy tile: broker headline, its own n / last close, and the demoted bot-log number with its exclusions —
            // every caption line whole (an ellipsis would hide a fact, and the title is no help on touch)
            if (!/^Expectancy \/ closed lot · broker/.test(kpi.eyebrow)) fail(`kpi: the first tile's eyebrow reads "${kpi.eyebrow}"`);
            for (const l of kpi.lines) if (l.cut) fail(`kpi: a caption line is cut short — "${l.text}"`);
            if (kpi.lines.length !== 4) fail(`kpi: ${kpi.lines.length} caption lines (expected n / last close + the two bot-log lines)`);
            if (!/bot log −1\.20% · n 36 · since Oct 14\s*excludes 351 unattributed exits/.test(kpi.first)) fail(`kpi: the bot-log line is missing or not from the fields — "${kpi.first.slice(0, 160)}"`);
            if (name === "bot") {
              if (!/−3\.10%/.test(kpi.first)) fail(`kpi: the broker expectancy (fixture −3.1 %) is not the headline — "${kpi.first.slice(0, 120)}"`);
              if (!/n 1,902\s*last close [A-Z][a-z]{2} \d{1,2}/.test(kpi.first)) fail(`kpi: no "n · last close" caption for the broker number — "${kpi.first.slice(0, 160)}"`);
              if (!/payoff \d+\.\d{2} · pf \d+\.\d{2}/.test(kpi.second)) fail(`kpi: the win-rate tile shows no payoff / pf from the broker figures — "${kpi.second.slice(0, 120)}"`);
              if (!/40\.0% \/ 46\.8%/.test(kpi.second)) fail(`kpi: the win-rate tile is not the broker's 40.0 % against its 46.8 % breakeven — "${kpi.second.slice(0, 120)}"`);
            } else {
              // broker null (the ledger not loaded yet): a dash and the warming-up words, never the stale bot-log number as the headline
              if (!/broker ledger\s*warming up/.test(kpi.first)) fail(`kpi: broker null shows no "warming up" state — "${kpi.first.slice(0, 160)}"`);
              if (/[−+]\d+\.\d{2}%/.test(kpi.first.replace(/bot log −1\.20%[\s\S]*$/, ""))) fail(`kpi: broker null still shows a percent headline — "${kpi.first.slice(0, 120)}"`);
              if (!/broker ledger warming up/.test(kpi.second)) fail(`kpi: the win-rate tile shows a number without the broker ledger — "${kpi.second.slice(0, 120)}"`);
            }
          }
        }
        // ── TODAY TILE (B1, human-approved 2026-11-03; server/todayTile.ts,
        // client/src/bot/today.ts): equity now against the prior close in $ and
        // %, SPY's move on the same ET day with its own as-of clock, and the
        // difference in points as the headline — every line from the server's
        // fields. The three printed numbers must ADD UP (the fixture server
        // moves equity from the third poll on and its hit counter spans the
        // widths, so the change's sign and size are not fixed: the check reads
        // the numbers it finds and verifies the arithmetic, never one frozen value);
        // the prior close → now path must agree with the change; the hover
        // carries the provenance. Idle page: no SPY series yet, so the headline
        // is a dash and the SPY line carries the reason — never a zero, never
        // NaN. The tile may not clip or cut its own content at any width.
        {
          const td = await page.evaluate(() => {
            const t = document.querySelector('.vt-bot-kpi .vt-bot-panel[data-kpi="today"]');
            if (!t) return null;
            const text = (t.textContent || "").replace(/\s+/g, " ").trim();
            const lines = [...t.querySelectorAll(".vt-kpi-today-ln > span")].map((s) => ({ text: s.textContent || "", cut: s.scrollWidth - s.clientWidth > 1 }));
            const r = t.getBoundingClientRect();
            const eb = t.querySelector(".vt-kpi-today-eb");
            return { text, lines, over: Math.max(0, t.scrollHeight - t.clientHeight), w: Math.round(r.width), h: Math.round(r.height), help: !!t.querySelector('button.vt-help[data-help="today"]'), title: t.querySelector(".vt-kpi-today")?.getAttribute("title") || "", headline: t.querySelector(".vt-bot-kpiv")?.textContent || "",
              phase: t.getAttribute("data-phase"), eb: (eb?.innerText || "").replace(/\s+/g, " ").trim(), ebCut: eb ? eb.scrollWidth - eb.clientWidth > 1 : true };
          });
          if (!td) fail("today: no tile in the KPI strip");
          else {
            console.log(`  today tile ${vp.w}px: ${td.w}×${td.h} — eyebrow "${td.eb}" — "${td.text}"`);
            // BETWEEN SESSIONS (2026-11-06): these two pages are in-session fixtures — the eyebrow stays "Today"
            if (td.phase !== "session") fail(`today: the in-session fixture's tile reads phase "${td.phase}"`);
            if (!/^today · vs spy\??$/i.test(td.eb)) fail(`today: the eyebrow reads "${td.eb}" during a session (expected "Today · vs SPY")`);
            if (td.ebCut) fail(`today: the eyebrow is cut short — "${td.eb}"`);
            if (/NaN|undefined|null/.test(td.text)) fail(`today: a bare NaN / undefined / null in "${td.text}"`);
            if (td.over > 1) fail(`today: the tile clips ${td.over}px of its own content`);
            for (const l of td.lines) if (l.cut) fail(`today: a line is cut short — "${l.text}"`);
            if (td.lines.length !== 3) fail(`today: ${td.lines.length} caption lines (expected the change, SPY, and prior → now)`);
            if (!td.help) fail("today: no help glyph");
            if (!/last_equity/.test(td.title) || !/price only, no dividends/.test(td.title)) fail(`today: the hover carries no provenance — "${td.title.slice(0, 120)}"`);
            const num = (s) => Number(s.replace(/−/g, "-").replace(/[$,%]/g, ""));
            const chg = /([−+]?\$[\d,]+) · ([−+]?\d+\.\d\d%)/.exec(td.text);
            const path = /prior (\$[\d,]+) → (\$[\d,]+)/.exec(td.text);
            if (!chg) fail(`today: no "$ · %" change line — "${td.text}"`);
            if (!path) fail(`today: no "prior $ → $" line — "${td.text}"`);
            if (chg && path && Math.abs(num(path[2]) - num(path[1]) - num(chg[1])) > 1) fail(`today: prior → now (${path[1]} → ${path[2]}) does not agree with the change ${chg[1]}`);
            if (chg && path && Math.abs(num(chg[1]) / num(path[1]) * 100 - num(chg[2])) > 0.011) fail(`today: ${chg[1]} on ${path[1]} is not ${chg[2]}`);
            if (name === "bot") {
              const spy = /SPY (\+0\.25%) · (\d\d:\d\d) ET/.exec(td.text);
              if (!spy) fail(`today: SPY's move (fixture +0.25 %) and its own as-of clock are missing — "${td.text}"`);
              const pp = /^([−+]\d+\.\d\d|0\.00) pp$/.exec(td.headline);
              if (!pp) fail(`today: the headline "${td.headline}" is not a signed difference in pp (or exactly 0.00 pp, unsigned)`);
              if (chg && spy && pp && Math.abs(num(chg[2]) - 0.25 - num(pp[1])) > 0.011) fail(`today: ${chg[2]} − SPY +0.25% is not the headline ${td.headline}`);
            } else {
              if (td.headline !== "—") fail(`today: the idle page (no SPY series) shows "${td.headline}" as the headline, not a dash`);
              if (!/SPY — · no bars yet/.test(td.text)) fail(`today: the idle page shows no reason for the missing SPY side — "${td.text}"`);
              if (!/no SPY bars fetched yet/.test(td.title)) fail(`today: the hover does not carry the server's full reason — "${td.title.slice(0, 160)}"`);
              if (!chg) fail("today: the account side must still read on the idle page");
            }
          }
        }
        // ── SESSION FOLD (2026-11-03, with the Today tile): at the l tier (1180–1919)
        // the Session tile is not rendered and its line lives in the Now tile — the
        // strip stays one 86 px row of six; everywhere else the Session tile
        // shows and the Now tile carries no session line. Self-see: the session
        // is reachable at every width, in exactly one place.
        {
          const sf = await page.evaluate(() => {
            const tile = document.querySelector(".vt-bot-kpi .vt-kpi-session");
            const line = document.querySelector(".vt-bot-kpi .vt-kpi-now-session");
            const vis = (el) => !!el && el.getBoundingClientRect().height > 0;
            return { tile: vis(tile), line: vis(line), lineText: line ? (line.textContent || "").trim() : "", lineCut: line ? line.scrollWidth - line.clientWidth > 1 : false, tiles: document.querySelectorAll(".vt-bot-kpi > .vt-bot-panel").length };
          });
          const folded = vp.w >= 1180 && vp.w < 1920;
          if (sf.tiles !== (folded ? 6 : 7)) fail(`session fold: ${sf.tiles} tiles in the strip (expected ${folded ? "six — the Session tile is not rendered at the l tier" : "seven"})`);
          if (sf.tile === folded) fail(`session fold: the Session tile is ${sf.tile ? "visible" : "hidden"} at ${vp.w}`);
          if (sf.line !== folded) fail(`session fold: the Now tile's session line is ${sf.line ? "visible" : "hidden"} at ${vp.w}`);
          if (folded) {
            if (!/^(RTH · closes in|closed · opens in) \d/.test(sf.lineText)) fail(`session fold: the folded line reads "${sf.lineText}"`);
            if (sf.lineCut) fail(`session fold: the folded line is cut short — "${sf.lineText}"`);
          }
        }
        // ── EXPECTANCY BY SLEEVE (B2, [PRODUCT] 2026-11-03, client/src/bot/expectancyTable.tsx): the Book carries the
        // Expectancy tile's number split by sleeve — six rows (options, stocks, their total = the headline, ETF
        // floor, VRP leg, all lots) × seven columns from the broker block's bySleeve. The total row's expectancy IS
        // the tile's headline; no cell is ever cut (the wrapper scrolls sideways when the panel is narrower than
        // the table and the page never overflows); a sleeve below zero over n ≥ 30 wears the red dot and is the
        // FIRST issue on the page with its numbers; the broker-null page shows the designed warming state and
        // flags nothing it cannot measure.
        {
          const ex = await page.evaluate(() => {
            const root = document.querySelector('section.vt-bot-panel[data-panel="book"] .vt-exp');
            if (!root) return null;
            const wrap = root.querySelector(".vt-exp-wrap");
            const table = root.querySelector(".vt-exp-t");
            const rows = [...root.querySelectorAll("tbody tr")].map((tr) => ({
              g: tr.getAttribute("data-group"), total: tr.getAttribute("data-total") === "true", flag: tr.getAttribute("data-flag"),
              cells: [...tr.querySelectorAll("td")].map((td) => (td.textContent || "").trim()),
              cut: [...tr.querySelectorAll("td")].some((td) => td.scrollWidth - td.clientWidth > 1),
              h: Math.round(tr.getBoundingClientRect().height),
            }));
            const head = [...root.querySelectorAll("thead th")].map((th) => (th.textContent || "").trim().replace(/\?$/, "").trim());
            const rr = root.getBoundingClientRect();
            const expCell = root.querySelector("tbody tr td:nth-child(2)");
            const issues = [...document.querySelectorAll('section.vt-bot-panel[data-panel="issues"] .vt-bot-issue')].map((it) => it.textContent || "");
            return { state: root.getAttribute("data-state"), text: root.textContent || "", head, rows, wrapW: wrap ? wrap.clientWidth : 0, tableW: table ? table.scrollWidth : 0,
              scrolls: wrap ? wrap.scrollWidth > wrap.clientWidth + 1 : false, overflowX: wrap ? getComputedStyle(wrap).overflowX : "", inside: rr.left >= -0.5 && rr.right <= innerWidth + 0.5,
              expRight: expCell ? expCell.getBoundingClientRect().right : null, wrapRight: wrap ? wrap.getBoundingClientRect().right : 0,
              src: root.querySelector(".vt-exp-src")?.textContent || "", issues };
          });
          // a progress line as the KPI block prints its strip height: the verdicts print at the very end, so a run the
          // sandbox's memory cgroup kills later still leaves this block's result per width in the log
          const exFailsBefore = checks.failures.length;
          if (!ex) fail("expectancy: no by-sleeve block in the Book panel");
          else {
            if (!/^Expectancy by sleeve/.test(ex.text.replace(/\?/g, ""))) fail(`expectancy: the block's eyebrow reads "${ex.text.slice(0, 60)}"`);
            if (!ex.inside) fail("expectancy: the block leaves the viewport sideways");
            if (name === "bot") {
              if (ex.state !== "ready") fail(`expectancy: state "${ex.state}" with the broker block present`);
              if (ex.head.join("|") !== "sleeve|expectancy|n|win|avg win|avg loss|payoff|breakeven") fail(`expectancy: header reads ${JSON.stringify(ex.head)}`);
              if (ex.rows.map((r) => r.g).join("|") !== "options|stocks|active|floor|vrp|all") fail(`expectancy: rows are ${ex.rows.map((r) => r.g).join("|")}`);
              const by = Object.fromEntries(ex.rows.map((r) => [r.g, r]));
              // the total row IS the tile's headline (fixture −3.1 %, n 1,902), and says so; expectancy is the first
              // column so the phone shows it without a scroll
              if (by.active && !(by.active.cells[1] === "−3.10%" && by.active.cells[2] === "1,902" && /headline/.test(by.active.cells[0]))) fail(`expectancy: the options + stocks row is not the tile's headline — ${JSON.stringify(by.active?.cells)}`);
              // every column from the fixture's own numbers: expectancy, n, win rate against breakeven, payoff, breakeven
              if (by.options && by.options.cells.join("|") !== "options|−3.40%|1,724|40.0%|+25.40%|−22.60%|1.12|47.1%") fail(`expectancy: options row reads ${JSON.stringify(by.options.cells)}`);
              if (by.floor && by.floor.cells.join("|") !== "ETF floor|+1.10%|96|63.5%|+3.10%|−2.40%|1.29|43.6%") fail(`expectancy: floor row reads ${JSON.stringify(by.floor.cells)}`);
              // a sleeve with no closed lot prints dashes (with the reason on hover), never NaN or Infinity
              if (by.vrp && by.vrp.cells.join("|") !== "VRP leg|—|0|—|—|—|—|—") fail(`expectancy: VRP row reads ${JSON.stringify(by.vrp.cells)}`);
              // on the phone the expectancy column is inside the wrapper's first view, never only behind the scroll
              if (vp.touch && ex.expRight !== null && ex.expRight > ex.wrapRight + 0.5) fail(`expectancy: the expectancy column (right edge ${Math.round(ex.expRight)}px) sits behind the sideways scroll on the phone (wrapper right ${Math.round(ex.wrapRight)}px)`);
              if (/NaN|Infinity/.test(ex.text)) fail("expectancy: a NaN or Infinity reached the page");
              // the flag: options (−3.4 % over 1,724) and stocks (−1.1 % over 178) wear it, the floor (+1.1 %) and the totals never
              if (!(by.options?.flag === "neg" && by.stocks?.flag === "neg" && !by.floor?.flag && !by.vrp?.flag && !by.active?.flag && !by.all?.flag)) fail(`expectancy: flags are ${JSON.stringify(ex.rows.map((r) => [r.g, r.flag]))}`);
              // no cell cut, and when the table is wider than the panel the wrapper scrolls instead of the page
              for (const r of ex.rows) if (r.cut) fail(`expectancy: a cell of the ${r.g} row is cut short`);
              if (ex.tableW > ex.wrapW + 1 && !(ex.scrolls && ex.overflowX === "auto")) fail(`expectancy: the table (${ex.tableW}px) outgrows its wrapper (${ex.wrapW}px) without scrolling`);
              // the phone tier (DESIGN.md's ≤ 639 px breakpoint, where the CSS enlarges the rows); the tablet keeps the 22 px density of the Book's tape
              if (vp.w <= 639) for (const r of ex.rows) if (r.h < 26) fail(`expectancy: a ${r.h}px row on the phone (rows read at ≥ 26 px)`);
              // provenance and freshness under the table: window, last close, lots, cache time, source
              if (!/since May 7 · last close [A-Z][a-z]{2} \d{1,2} · 1,998 lots · 71 unpriced excluded · as of \d{2}:\d{2}:\d{2} ET · alpaca fills via sleeve ledger/.test(ex.src)) fail(`expectancy: the caption reads "${ex.src}"`);
              // the TOP issue: first in the Issues panel, ahead of the latched kill switch, naming both sleeves with their numbers
              if (!ex.issues.length) fail("expectancy: no issues rendered on the latched page");
              else {
                if (!/^Negative expectancy · options, stocks/.test(ex.issues[0])) fail(`expectancy: the first issue is "${ex.issues[0].slice(0, 80)}"`);
                if (!/options −3\.40% over 1,724 lots \(win 40\.0% vs breakeven 47\.1%\) · stocks −1\.10% over 178 lots/.test(ex.issues[0])) fail(`expectancy: the issue's numbers are not the table's — "${ex.issues[0].slice(0, 240)}"`);
                if (!/open book/.test(ex.issues[0])) fail("expectancy: the issue offers no way to the Book");
                if (!ex.issues.some((t) => /^Kill switch latched/.test(t))) fail("expectancy: the kill latch issue is gone");
              }
            } else {
              // botidle: the broker block is null — the designed warming state, and no sleeve flagged on data that is not there
              if (ex.state !== "warming") fail(`expectancy: state "${ex.state}" without the broker block`);
              if (!/warming up|not loaded yet/.test(ex.text)) fail(`expectancy: no warming-up words without the broker block — "${ex.text.slice(0, 120)}"`);
              if (ex.rows.length) fail("expectancy: rows rendered without the broker block");
              if (ex.issues.some((t) => /Negative expectancy/.test(t))) fail("expectancy: the broker-null page flags a sleeve it cannot measure");
            }
            // the block sits below the fold of the Book (the tape and the P&L block come first): its own
            // screenshot per width, scrolled into view, so the review sees it (DESIGN.md self-see rule)
            const exEl = page.locator('section.vt-bot-panel[data-panel="book"] .vt-exp');
            if (await exEl.count()) { await exEl.scrollIntoViewIfNeeded(); await page.waitForTimeout(100); await exEl.screenshot({ path: path.join(OUT, `${name}-expectancy-${vp.w}.png`), animations: "disabled" }); }
          }
          const exFails = checks.failures.slice(exFailsBefore);
          console.log(`  expectancy ${vp.w}px: state ${ex?.state ?? "none"} · ${ex?.rows.length ?? 0} rows · table ${ex?.tableW ?? 0}px in ${ex?.wrapW ?? 0}px${ex?.scrolls ? " (scrolls)" : ""} · ${exFails.length ? `${exFails.length} failure(s): ${exFails.join(" | ")}` : "ok"}`);
        }
        // CHART (2026-10-21, our own canvas engine): the chart panel holds a
        // canvas with real size that has actually painted — the regression
        // the old library chart failed (a 320 px chart under a ~205 px
        // panel's hidden overflow, "never loads"). Sampled alpha > 0 on at
        // least 2 % of pixels; polled up to 6 s for the bars fetch + frame.
        {
          let info = null;
          for (let i = 0; i < 24; i++) {
            info = await page.evaluate(() => {
              const c = document.querySelector('section.vt-bot-panel[data-panel="chart"] canvas');
              if (!c) return { present: false };
              const r = c.getBoundingClientRect();
              try {
                const img = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
                let painted = 0, total = 0;
                for (let k = 3; k < img.length; k += 4 * 7) { total++; if (img[k] > 0) painted++; }
                return { present: true, w: Math.round(r.width), h: Math.round(r.height), painted, total };
              } catch (e) { return { present: true, w: Math.round(r.width), h: Math.round(r.height), error: String(e) }; }
            });
            if (info.present && info.total && info.painted / info.total >= 0.02) break;
            await page.waitForTimeout(250);
          }
          if (!info || !info.present) fail("chart: no canvas in the chart panel");
          else if (info.error) fail(`chart: canvas unreadable (${info.error})`);
          else if (info.w < 200 || info.h < 120) fail(`chart: canvas is ${info.w}×${info.h}px`);
          else if (!info.total || info.painted / info.total < 0.02) fail(`chart: canvas blank (${info.painted}/${info.total} sampled pixels painted)`);
          // HELP (slice C, 2026-10-21): every "?" opens one card inside the
          // viewport that never covers its own anchor; Escape closes it.
          // Sampled: the first eight glyphs on the page.
          {
            const glyphs = page.locator("button.vt-help");
            const total = await glyphs.count();
            if (total < 20) fail(`help: only ${total} "?" glyphs on the page`);
            const sample = Math.min(8, total);
            for (let g = 0; g < sample; g++) {
              const btn = glyphs.nth(g);
              await btn.scrollIntoViewIfNeeded();
              await press(btn); await page.waitForTimeout(120);
              const info = await page.evaluate(() => {
                const c = document.querySelector(".vt-help-card");
                const b = document.querySelector('button.vt-help[aria-expanded="true"]');
                if (!c || !b) return { open: false };
                const r = c.getBoundingClientRect(), a = b.getBoundingClientRect();
                const inside = r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight;
                const overlaps = !(r.right <= a.left || r.left >= a.right || r.bottom <= a.top || r.top >= a.bottom);
                return { open: true, inside, overlaps, text: (c.textContent || "").trim().length };
              });
              if (!info.open) fail(`help: glyph ${g} opened no card`);
              else { if (!info.inside) fail(`help: glyph ${g}'s card leaves the viewport`); if (info.overlaps) fail(`help: glyph ${g}'s card covers its anchor`); if (info.text < 40) fail(`help: glyph ${g}'s card is empty`); }
              // Escape closes the card; wait for the close instead of assuming 80 ms (at 1440 the heavier page missed that twice, 2026-10-22)
              await page.keyboard.press("Escape");
              let closed = false;
              for (let w = 0; w < 12 && !closed; w++) { await page.waitForTimeout(50); closed = (await page.locator(".vt-help-card").count()) === 0; }
              if (!closed) { fail(`help: Escape did not close glyph ${g}'s card within 600 ms`); await page.mouse.click(2, 2); await page.waitForTimeout(80); }
            }
          }
          const chips = await page.locator('section.vt-bot-panel[data-panel="chart"] .vt-chart-chip').count();
          if (chips < 2) fail(`chart: symbol strip shows ${chips} chips for 8 positions`);
          const legend = await page.locator('section.vt-bot-panel[data-panel="chart"] .vt-chart-legend .k').count();
          if (!legend) fail("chart: no legend row (phase / R / scale-outs)");
          // KNOWN BROKEN #50a (2026-11-12): a share lot's legend names the levels the engine executes — the −25 %
          // backstop (POS-KILL), not manage_positions' hard floor, and no stop that "waits"
          {
            const lg = await page.evaluate(() => {
              const box = document.querySelector('section.vt-bot-panel[data-panel="chart"] .vt-chart-legend');
              return { keys: [...(box?.querySelectorAll('.k') || [])].map((k) => (k.textContent || '').replace(/\?$/, '').trim()), text: (box?.textContent || '').trim() };
            });
            if (lg.keys.includes("stop")) {
              if (!lg.keys.includes("backstop")) fail(`chart: a share lot's legend has no backstop row (keys: ${lg.keys.join(", ")})`);
              if (!/backstop\??[\d,.]+ · −25%/.test(lg.text)) fail(`chart: the legend's backstop is not −25% from entry — "${lg.text.slice(0, 260)}"`);
            }
            if (lg.keys.includes("floor") || /waits|hard floor|grace to day|minimum hold/.test(lg.text)) fail(`chart: the legend still shows a rule the engine does not execute — "${lg.text.slice(0, 260)}"`);
          }
        }
        for (const id of panels) {
          try {
            // hold + pin toggle and report it
            for (const label of ["Hold this panel", "Pin this panel"]) {
              const b = ctl(id, label);
              if (!(await b.count())) { fail(`${id}: '${label}' missing`); continue; }
              await press(b); await settle();
              const after = label === "Hold this panel" ? ctl(id, "Resume this panel") : b;
              if ((await after.getAttribute("aria-pressed")) !== "true") fail(`${id}: '${label}' did not toggle on`);
              await press(after); await settle();
              if ((await ctl(id, label).getAttribute("aria-pressed")) !== "false") fail(`${id}: '${label}' did not toggle off`);
            }
            const open = async () => { await press(ctl(id, "Open as a window")); await settle(); return (await winCount()) === 1; };
            const win = page.locator(".vt-bot-win");
            // window → dock button
            if (!(await open())) { fail(`${id}: 'Open as a window' opened ${await winCount()} windows`); continue; }
            await press(win.locator("button.vt-bot-wbtn", { hasText: "dock" })); await settle();
            if (await winCount()) { fail(`${id}: window 'dock' button did nothing`); await page.keyboard.press("Escape"); await settle(); }
            // window → close (x)
            if (await open()) {
              await press(win.locator('button[aria-label="Close window"]')); await settle();
              if (await winCount()) { fail(`${id}: window close button did nothing`); await page.keyboard.press("Escape"); await settle(); }
            }
            // window → zoom → back to the page
            if (await open()) {
              await press(win.locator("button.vt-bot-wbtn", { hasText: "zoom" })); await settle();
              const back = page.locator("button.vt-bot-wbtn", { hasText: "back to the page" });
              if (!(await back.count()) || (await winCount())) fail(`${id}: window 'zoom' button did not zoom`);
              if (await back.count()) { await press(back); await settle(); if (await back.count()) fail(`${id}: 'back to the page' did not leave the zoom`); }
              if (await winCount()) { await page.keyboard.press("Escape"); await settle(); }
            }
            // window → ghost slot docks it (when the window does not cover
            // its own ghost — it opens centred, over the centre column)
            if (await open()) {
              const ghost = page.locator(".vt-bot-ghost").first();
              await ghost.scrollIntoViewIfNeeded();
              const reachable = await ghost.evaluate((g) => { const r = g.getBoundingClientRect(); const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); return !!hit && g.contains(hit); });
              if (reachable) {
                await press(ghost); await settle();
                if (await winCount()) fail(`${id}: clicking the ghost slot did not dock`);
              }
              if (await winCount()) { await page.keyboard.press("Escape"); await settle(); }
            }
            // window → Esc docks it (the title bar promises "Esc docks"),
            // with focus wherever the click left it
            if (await open()) {
              await page.keyboard.press("Escape"); await settle();
              if (await winCount()) {
                fail(`${id}: Esc did not dock the window`);
                await press(win.locator("button.vt-bot-wbtn", { hasText: "dock" })); await settle();
              }
            }
            // window → drag the title bar
            if (!vp.touch && (await open())) {
              const bar = win.locator(".vt-bot-wh");
              const b0 = await win.boundingBox();
              const hb = await bar.boundingBox();
              if (b0 && hb) {
                await page.mouse.move(hb.x + 40, hb.y + hb.height / 2);
                await page.mouse.down();
                await page.mouse.move(hb.x + 40 - 60, hb.y + hb.height / 2 + 40, { steps: 6 });
                await page.mouse.up();
                await settle();
                const b1 = await win.boundingBox();
                if (!b1 || Math.abs(b1.x - (b0.x - 60)) > 14 || Math.abs(b1.y - (b0.y + 40)) > 14) fail(`${id}: dragging the title bar did not move the window (${JSON.stringify(b0)} -> ${JSON.stringify(b1)})`);
              }
              await page.keyboard.press("Escape"); await settle();
              if (await winCount()) { await press(win.locator('button[aria-label="Close window"]')); await settle(); }
            }
            if (await winCount()) fail(`${id}: a window was left open after the battery`);
          } catch (e) {
            fail(`${id}: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 700)}`);
          }
        }
        // ── CHART READOUT ([REPAIR] 2026-10-22): the readout row carries no
        // browser tooltip (a native title used to open over the strip and the
        // legend — "text covering text"). The hover-tag de-overlap itself is
        // canvas geometry, pinned by draw.test.ts (gutterYs); the crosshair
        // path is exercised below by moving over the plot.
        {
          const titled = await page.evaluate(() => [...document.querySelectorAll('.vt-chart-readout [title], .vt-chart-tf [title]')].length);
          if (titled) fail(`chart: ${titled} readout control(s) still carry a browser tooltip`);
          if (!vp.touch) {
            const box = await page.locator('.vt-chart canvas').first().boundingBox();
            if (box) { await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.4); await page.waitForTimeout(150); await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.5); await page.waitForTimeout(150); await page.mouse.move(2, 2); }
          }
        }
        // ── 1-MINUTE TIMEFRAME (2026-10-22, human-asked): the timeframe row
        // offers 1m first of five, and selecting it keeps the chart painted
        // (the fixture answers any timeframe).
        {
          const tfs = await page.evaluate(() => [...document.querySelectorAll('.vt-chart-tf button')].map((b) => (b.textContent || '').trim()));
          if (tfs[0] !== "1m" || tfs.length !== 5) fail(`chart: timeframe row reads ${tfs.join(" ")} (expected 1m first of five)`);
          const oneMin = page.locator('.vt-chart-tf button', { hasText: /^1m$/ }).first();
          if (await oneMin.count()) {
            try {
              await press(oneMin); await page.waitForTimeout(700);
              const after = await page.evaluate(() => {
                const pressed = document.querySelector('.vt-chart-tf button[aria-pressed="true"]');
                const canvas = document.querySelector('.vt-chart canvas');
                const readout = (document.querySelector('.vt-chart-readout')?.textContent || '');
                return { pressed: (pressed?.textContent || '').trim(), canvas: !!canvas && canvas.width > 0, readout };
              });
              if (after.pressed !== "1m") fail(`chart: 1m did not take (pressed: ${after.pressed})`);
              if (!after.canvas) fail("chart: no canvas after switching to 1m");
              if (!/O \d/.test(after.readout)) fail(`chart: the readout shows no bar after switching to 1m: "${after.readout.slice(0, 60)}"`);
              const back = page.locator('.vt-chart-tf button', { hasText: /^1h$/ }).first();
              if (await back.count()) { await press(back); await page.waitForTimeout(300); }
            } catch (e) {
              fail(`chart: 1m driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
            }
          }
        }
        // ── BOOK TAPE (2026-10-22, human-asked): the Book reads like a tape —
        // a header row names the columns, every row has five cells, last
        // carries its dollar sign, day is its own cell, and the P&L cell
        // carries a dollar amount beside the percent at desktop widths (the
        // phone row hides the dollars to keep its width).
        {
          const bk = await page.evaluate(() => {
            const panel = document.querySelector('section.vt-bot-panel[data-panel="book"]');
            if (!panel) return null;
            const head = panel.querySelector('.vt-bot-thead');
            const rows = [...panel.querySelectorAll('.vt-bot-trow[role="button"]')]; // position rows; the sleeve summary reuses the row class
            return {
              head: head ? [...head.children].map((c) => (c.textContent || '').trim().replace(/\?$/, '').trim()) : null,
              rows: rows.length,
              fiveCells: rows.every((r) => r.querySelectorAll('.sym, .q, .px, .dy, .pl').length === 5),
              dollarLast: rows.every((r) => /^\$\d|^·$/.test((r.querySelector('.px')?.textContent || '').trim())),
              units: rows.every((r) => /(sh|ct)$/.test((r.querySelector('.q')?.textContent || '').trim())),
              dayCells: rows.filter((r) => /[−+]\d+\.\d%$/.test((r.querySelector('.dy')?.textContent || '').trim())).length,
              usd: rows.filter((r) => /\$\d/.test(r.querySelector('.pl .usd')?.textContent || '')).length,
              usdShown: rows.some((r) => { const u = r.querySelector('.pl .usd'); return !!u && getComputedStyle(u).display !== 'none'; }),
              panelW: panel.getBoundingClientRect().width,
              symW: rows.length ? rows[0].querySelector('.sym').getBoundingClientRect().width : null,
              // 2026-11-02 (human: "what type of option … when would you sell, is that shown"): lines 3–4
              optRows: rows.filter((r) => /ct$/.test((r.querySelector('.q')?.textContent || '').trim())).length,
              whatLines: rows.filter((r) => /^(bought|sold) (call|put)/.test((r.querySelector('.sub--what')?.textContent || '').trim())).length,
              exitLines: rows.filter((r) => /^exit/.test((r.querySelector('.sub--exit')?.textContent || '').trim())).length,
              clipped: rows.filter((r) => { const e = r.querySelector('.sub--exit'); return !!e && e.scrollWidth > e.clientWidth + 1; }).length,
              // KNOWN BROKEN #50a (2026-11-12): the share rows' exit lines, and what each flags as already hit
              stockExits: rows.filter((r) => /sh$/.test((r.querySelector('.q')?.textContent || '').trim())).map((r) => (r.querySelector('.sub--exit')?.textContent || '').trim()),
              stockHits: rows.filter((r) => /sh$/.test((r.querySelector('.q')?.textContent || '').trim())).flatMap((r) => [...r.querySelectorAll('.sub--exit b.hit')].map((b) => (b.textContent || '').trim())),
            };
          });
          if (bk === null) fail("book: no Book panel");
          else {
            if (!bk.head || bk.head.join(" ") !== "sym qty last day P&L") fail(`book: header reads "${(bk.head || []).join(" ")}"`);
            if (bk.rows && !bk.fiveCells) fail("book: a row lacks one of its five cells");
            if (bk.rows && !bk.dollarLast) fail("book: a last price has no dollar sign");
            if (bk.rows && !bk.units) fail("book: a quantity has no unit (sh / ct)");
            if (bk.rows && bk.dayCells < 1) fail("book: no row shows today's move although the fixture carries it");
            if (bk.rows && bk.usd < 1) fail("book: no P&L cell carries a dollar amount");
            // the dollars follow the Book's OWN width (container query): shown from 380 px, hidden below so the symbol keeps its room
            if (bk.rows && bk.panelW >= 380 && !bk.usdShown) fail(`book: the dollar P&L is hidden in a ${Math.round(bk.panelW)} px Book`);
            if (bk.rows && bk.panelW < 380 && bk.usdShown) fail(`book: the dollar P&L shows in a ${Math.round(bk.panelW)} px Book and squeezes the symbol`);
            if (bk.symW !== null && bk.symW < 60) fail(`book: the symbol column is ${Math.round(bk.symW)} px — unreadable`);
            if (bk.rows && bk.exitLines !== bk.rows) fail(`book: ${bk.rows - bk.exitLines} row(s) do not say when the engine exits`);
            if (bk.rows && bk.whatLines !== bk.optRows) fail(`book: ${bk.optRows - bk.whatLines} option row(s) do not say bought/sold call/put`);
            if (bk.clipped) fail(`book: ${bk.clipped} exit line(s) clipped — a rule is hidden`);
            // KNOWN BROKEN #50a (2026-11-12): a share row shows the ladder the engine executes. Recorded rows carry
            // POS-KILL's −25 % backstop; none shows manage_positions' hard floor or a stop that "waits"; the fixture's
            // KWEB sits past its stop, so its stop is flagged hit (the engine sells on its next 1-minute bar).
            const recordedStock = bk.stockExits.filter((t) => /^exit\s*stop \$/.test(t));
            if (recordedStock.length < 3) fail(`book: ${recordedStock.length} share rows carry a stop (the fixture records three: QQQ, SMH, KWEB)`);
            const noBackstop = recordedStock.filter((t) => !/ · backstop \$[\d,.]+ \(−25%\)/.test(t));
            if (noBackstop.length) fail(`book: ${noBackstop.length} recorded share row(s) do not show the −25% backstop — "${noBackstop[0].slice(0, 140)}"`);
            const stale = bk.stockExits.filter((t) => /floor \$|waits|grace to day|minimum hold|exit now only/.test(t));
            if (stale.length) fail(`book: a share row still shows a rule the engine does not execute — "${stale[0].slice(0, 160)}"`);
            if (!bk.stockHits.includes("stop $25.47 from entry · hit")) fail(`book: the KWEB row (last $24.54, stop $25.47) does not flag its stop as hit — flagged: ${JSON.stringify(bk.stockHits)}`);
          }
        }
        // ── DEAD SPACE (2026-10-22, human-approved): at the desktop tiers no
        // column leaves a blank — the last panel of each column reaches the
        // column's bottom (≤ 12 px), the Book lives in the left column and
        // the chart is at least 70 % of the right column.
        if (!vp.touch && vp.width >= 1180) {
          const ds = await page.evaluate(() => {
            const cols = ["left", "centre", "right"].map((a) => {
              const col = document.querySelector(`.vt-bot-area-${a}`);
              if (!col) return { a, missing: true };
              const panels = [...col.querySelectorAll('section.vt-bot-panel[data-panel]')];
              const cr = col.getBoundingClientRect();
              const bottom = Math.max(...panels.map((p) => p.getBoundingClientRect().bottom));
              // a hole BETWEEN panels: the painted panel heights plus the 8 px gaps must add up to the column
              const painted = panels.reduce((s, p) => s + p.getBoundingClientRect().height, 0) + 8 * Math.max(0, panels.length - 1);
              return { a, missing: false, n: panels.length, gap: panels.length ? cr.bottom - bottom : null, hole: panels.length ? cr.height - painted : null, ids: panels.map((p) => p.getAttribute('data-panel')) };
            });
            const book = !!document.querySelector('.vt-bot-area-left section.vt-bot-panel[data-panel="book"]');
            const chart = document.querySelector('.vt-bot-area-right section.vt-bot-panel[data-panel="chart"]');
            const right = document.querySelector('.vt-bot-area-right');
            return { cols, book, chartFrac: chart && right ? chart.getBoundingClientRect().height / right.getBoundingClientRect().height : null };
          });
          for (const c of ds.cols) {
            if (c.missing) fail(`dead space: no ${c.a} column`);
            else if (c.gap !== null && c.gap > 12) fail(`dead space: the ${c.a} column leaves ${Math.round(c.gap)} px blank under ${c.ids[c.ids.length - 1]}`);
            else if (c.hole !== null && c.hole > 12) fail(`dead space: the ${c.a} column has ${Math.round(c.hole)} px of hole between its panels (${c.ids.join(" · ")})`);
          }
          if (!ds.book) fail("dead space: the Book is not in the left column");
          if (ds.chartFrac !== null && ds.chartFrac < 0.7) fail(`dead space: the chart is only ${Math.round(ds.chartFrac * 100)} % of the right column`);
        }
        // ── LOOK-UP -> TICKER DOSSIER (2026-10-22 look-up; 2026-10-29 the
        // same box opens the dossier, human-asked): "/" focuses the box in
        // the Watch header; type a symbol, Enter -> #/dossier/NVDA, where the
        // look-up card (folded into the dossier's stock layer) still names
        // the verdict, the six gates in the scan's order and the factors.
        // The fixture answers NVDA stopped at admission. Then back to #/bot
        // for the checks below.
        {
          const box = page.locator('.vt-lookup-box input').first();
          if (!(await box.count())) fail("lookup: no box in the Watch header");
          else {
            try {
              if (!vp.touch) {
                await page.mouse.click(2, 2); await page.keyboard.press("/"); await page.waitForTimeout(80);
                const focused = await page.evaluate(() => !!document.activeElement && !!document.activeElement.closest('.vt-lookup-box'));
                if (!focused) fail("lookup: '/' did not focus the box");
                await page.keyboard.press("Escape"); await page.mouse.click(2, 2);
              }
              await box.fill("nvda"); await box.press("Enter");
              let card = null;
              for (let i = 0; i < 60 && !card; i++) {
                await page.waitForTimeout(100);
                card = await page.evaluate(() => {
                  if (!/^#\/dossier\/NVDA$/.test(location.hash) || !document.querySelector('[data-vt-dossier]')) return null;
                  const c = document.querySelector('[data-vt-dossier] .vt-lookup[data-symbol]');
                  if (!c || c.getAttribute('data-busy')) return null;
                  return { sym: c.getAttribute('data-symbol'), verdict: (c.querySelector('.verdict')?.textContent || '').trim(), gates: [...c.querySelectorAll('.vt-lookup-gates span[data-status]')].map((g) => g.getAttribute('data-status')), bars: c.querySelectorAll('.vt-lookup-bars > span').length, chain: /ATM IV/.test(c.textContent || ''), hero: !!document.querySelector('.vt-ds-hero .vt-ds-bandviz') };
                });
              }
              await page.screenshot({ path: path.join(OUT, `${name}-lookup-${vp.w}.png`), animations: "disabled" });
              if (!card) fail("lookup: Enter did not open the dossier with the look-up card within 6 s");
              else {
                if (card.sym !== "NVDA") fail(`lookup: the card is for ${card.sym}`);
                if (!/stopped at admission/.test(card.verdict)) fail(`lookup: the verdict reads "${card.verdict}"`);
                if (card.gates.join(",") !== "pass,pass,pass,fail,skipped,skipped") fail(`lookup: the gates read ${card.gates.join(",")}`);
                if (card.bars < 4) fail(`lookup: only ${card.bars} factor bars`);
                if (!card.chain) fail("lookup: no chain line");
                if (!card.hero) fail("lookup: the dossier opened without its range band");
              }
            } catch (e) {
              fail(`lookup: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
            }
            // back to the AI Engine for the checks that follow
            await page.evaluate(() => { location.hash = "#/bot"; });
            for (let i = 0; i < 40; i++) { await page.waitForTimeout(100); if (await page.locator('.vt-lookup-box input').count()) break; }
            await page.waitForTimeout(300);
          }
        }
        // ── DAILY vs SPY SCOREBOARD (2026-10-28, human-ordered): one row per
        // trading day from /api/bot/scoreboard. Phone/tablet: a panel in the
        // stack; desktop: no grid slot (the columns are full) — the vs-SPY
        // tile's "daily" control zooms it. Latched fixture: 30 rows visible of
        // the walk, the summary and the provenance line; idle fixture: the
        // designed "not enough history" state, no table, never zeros.
        {
          try {
            const opener = page.locator(".vt-score-open").first();
            const zoomed = (await opener.count()) > 0;
            if (zoomed) { await press(opener); await settle(); }
            const panel = page.locator('section.vt-bot-panel[data-panel="score"]').first();
            if (!(await panel.count())) fail(`scoreboard: no panel${zoomed ? " after the vs-SPY tile's 'daily' control" : " in the stack"}`);
            else {
              await panel.scrollIntoViewIfNeeded();
              let sb = null;
              for (let i = 0; i < 40; i++) {
                sb = await page.evaluate(() => {
                  const p = document.querySelector('section.vt-bot-panel[data-panel="score"]');
                  if (!p) return null;
                  const pb = p.querySelector(".vt-bot-pb");
                  return { rows: p.querySelectorAll(".vt-score-t tbody tr").length, hero: !!p.querySelector(".vt-score-hero"), src: /price-only daily closes, no dividends/.test(p.querySelector(".vt-score-src")?.textContent || ""), skel: !!p.querySelector('[aria-busy="true"]'), empty: !!p.querySelector(".vt-bot-empty"), h: Math.round(p.getBoundingClientRect().height), over: pb ? pb.scrollWidth - pb.clientWidth : 0 };
                });
                if (sb && !sb.skel) break;
                await page.waitForTimeout(100);
              }
              // the panel as drawn, for the PR's visual review (.visual/<page>-score-<w>.png)
              await page.screenshot({ path: path.join(OUT, `${name}-score-${vp.w}.png`), animations: "disabled" });
              if (!sb || sb.skel) fail("scoreboard: still a skeleton after 4 s");
              else {
                if (sb.h < 120) fail(`scoreboard: panel is ${sb.h}px tall`);
                if (sb.over > 1) fail(`scoreboard: the table overflows its panel by ${sb.over}px`);
                if (name === "bot") {
                  if (sb.rows !== 30) fail(`scoreboard: ${sb.rows} rows visible (expected the last 30 days)`);
                  if (!sb.hero) fail("scoreboard: no since-start summary");
                  if (!sb.src) fail("scoreboard: no provenance line");
                } else if (sb.rows || !sb.empty) fail(`scoreboard: the empty fixture shows ${sb.rows} rows${sb.empty ? "" : " and no designed empty state"}`);
              }
            }
            if (zoomed) {
              const back = page.locator("button.vt-bot-wbtn", { hasText: "back to the page" });
              if (await back.count()) { await press(back); await settle(); } else fail("scoreboard: no way back from the zoomed scoreboard");
            } else await page.evaluate(() => document.querySelector(".vt-bot-stack")?.scrollTo(0, 0));
          } catch (e) {
            fail(`scoreboard: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
          }
        }
        // ── ENTRY SHADOW MODE (2026-10-28, human directive): new stock + options
        // entries are logged, not traded. The Now tile's eyebrow carries an
        // inline chip that must not grow its line; it opens the Orders panel
        // zoomed, where the sentence, the source and today's would-be entries
        // render without overflow (idle fixture: 4 rows; latched: the designed
        // empty state). Desktop: the Orders header carries a count chip and the
        // one-line slot does not grow. Phone/tablet: the block sits in the stack.
        {
          try {
            const chip = page.locator('.vt-bot-kpi .vt-shadow-chip[data-shadow-chip="short"]').first();
            if (!(await chip.count())) fail("shadow: no 'entries: shadow' chip in the Now tile");
            else {
              // the Now eyebrow keeps the height of a chip-less KPI eyebrow (Equity's — the Session tile is hidden at the l tier since the Today tile, 2026-11-03)
              const eb = await page.evaluate(() => { const c = document.querySelector('.vt-bot-kpi .vt-shadow-chip[data-shadow-chip="short"]'); const e = c?.closest(".vt-bot-eb"); const ref = [...document.querySelectorAll(".vt-bot-kpi .vt-bot-eb")].find((x) => /^Equity/i.test((x.textContent || "").trim())); const r = c?.getBoundingClientRect(); const er = e?.getBoundingClientRect(); return r && er && ref ? { inside: r.left >= 0 && r.right <= innerWidth + 0.5, ebH: er.height, refH: ref.getBoundingClientRect().height, title: c.getAttribute("title") || "" } : null; });
              if (!eb) fail("shadow: the chip is not inside the Now eyebrow");
              else {
                if (!eb.inside) fail("shadow: the Now chip leaves the viewport");
                if (eb.ebH > eb.refH + 0.5) fail(`shadow: the chip grew the Now eyebrow to ${eb.ebH.toFixed(1)}px (Equity's is ${eb.refH.toFixed(1)}px)`);
                if (!/logged, not traded/.test(eb.title)) fail("shadow: the chip's title does not say 'logged, not traded'");
              }
              const tpl = await page.evaluate(() => !!document.querySelector(".vt-bot-stack"));
              if (tpl) {
                const inStack = await page.evaluate(() => !!document.querySelector('section.vt-bot-panel[data-panel="orders"] .vt-shadow[data-shadow="on"]'));
                if (!inStack) fail("shadow: no shadow block in the stacked Orders panel");
              } else {
                // the Orders header keeps the height of a chip-less panel header (the Book's)
                const hd = await page.evaluate(() => { const c = document.querySelector('section.vt-bot-panel[data-panel="orders"] .vt-bot-ph .vt-shadow-chip[data-shadow-chip="count"]'); const ph = c?.closest(".vt-bot-ph"); const ref = document.querySelector('section.vt-bot-panel[data-panel="book"] .vt-bot-ph'); return c && ph && ref ? { h: ph.getBoundingClientRect().height, refH: ref.getBoundingClientRect().height, clipped: c.scrollWidth - c.clientWidth > 1, block: !!document.querySelector('section.vt-bot-panel[data-panel="orders"] .vt-shadow') } : null; });
                if (!hd) fail("shadow: no count chip in the desktop Orders header");
                else { if (hd.h > hd.refH + 0.5) fail(`shadow: the Orders header grew to ${hd.h.toFixed(1)}px (the Book's is ${hd.refH.toFixed(1)}px)`); if (hd.clipped) fail("shadow: the Orders header chip is clipped"); if (hd.block) fail("shadow: the desktop grid's one-line Orders slot carries the full block"); }
              }
              await press(chip); await settle();
              // wait for the block AND for the zoom's FLIP spring to settle (no transform left on the slot)
              let sh = null;
              for (let i = 0; i < 40; i++) {
                sh = await page.evaluate(() => {
                  const p = document.querySelector('section.vt-bot-panel[data-panel="orders"]');
                  const b = p?.querySelector('.vt-shadow[data-shadow="on"]');
                  if (!p || !b || (p.parentElement && p.parentElement.style.transform)) return null;
                  const pb = p.querySelector(".vt-bot-pb"); const br = b.getBoundingClientRect();
                  const rowsOver = [...b.querySelectorAll(".vt-bot-trow")].some((r) => r.scrollWidth - r.clientWidth > 1);
                  return { text: b.textContent || "", rows: b.querySelectorAll(".vt-shadow-list .vt-bot-trow").length, empty: !!b.querySelector(".vt-shadow-empty"), over: pb ? pb.scrollWidth - pb.clientWidth : 0, rowsOver, inside: br.left >= 0 && br.right <= innerWidth + 0.5 };
                });
                if (sh) break;
                await page.waitForTimeout(100);
              }
              await page.screenshot({ path: path.join(OUT, `${name}-shadow-${vp.w}.png`), animations: "disabled" });
              if (!sh) fail("shadow: the chip did not open the Orders panel with the shadow block");
              else {
                if (!/New entries: SHADOW — stocks · options \(logged, not traded\)\. Floor \+ exits live\./.test(sh.text)) fail("shadow: the block does not carry the headline");
                if (sh.over > 1) fail(`shadow: the Orders body overflows by ${sh.over}px`);
                if (sh.rowsOver) fail("shadow: a would-be entry row overflows");
                if (!sh.inside) fail("shadow: the block leaves the viewport");
                if (name === "botidle") {
                  if (sh.rows !== 4) fail(`shadow: ${sh.rows} would-be rows (expected 4)`);
                  if (!/size not reported/.test(sh.text)) fail("shadow: an unsized entry is not labelled 'size not reported'");
                  if (!/VOLTRADE_SHADOW_ENTRIES/.test(sh.text)) fail("shadow: the env override is not named");
                } else if (sh.rows || !sh.empty || !/no would-be entries yet today/.test(sh.text)) fail("shadow: the empty day shows no designed empty state");
              }
              const back = page.locator("button.vt-bot-wbtn", { hasText: "back to the page" });
              if (await back.count()) { await press(back); await settle(); } else fail("shadow: no way back from the zoomed Orders panel");
              await page.evaluate(() => document.querySelector(".vt-bot-stack")?.scrollTo(0, 0));
            }
          } catch (e) {
            fail(`shadow: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
          }
        }
        // ── AFTER-TAX LEG ([RULE-REVIEW] 2026-10-22): the latched page's
        // benchmark fixture carries the after-tax comparison (both legs sold
        // at the window's end); the vs-SPY tile must show it as its own line
        // with a help glyph and the liquidation framing on hover. The idle
        // page has no SPY series, so it must show no such line.
        {
          const at = await page.evaluate(() => { const el = document.querySelector(".vt-bot-aftertax"); return el ? { text: el.textContent || "", help: !!el.querySelector("button.vt-help"), title: el.getAttribute("title") || "" } : null; });
          if (name === "bot") {
            if (!at) fail("after-tax: the vs-SPY tile shows no after-tax line for the latched fixture");
            else {
              if (!/after tax [−+-]?\d/.test(at.text)) fail(`after-tax: the line reads "${at.text}"`);
              if (!at.help) fail("after-tax: the line has no help glyph");
              if (!/both sold/.test(at.title)) fail("after-tax: the hover carries no liquidation framing");
            }
          } else if (at) fail("after-tax: the idle page (no SPY series) shows an after-tax line");
        }
        // ── LIVE CHANGE (slice F, 2026-10-21): the fixture server's third
        // and later polls carry a fresh trade row and a moving equity (see
        // liveHits above). The page must react without a reload: the Now
        // tile shows the trade as its latest decision, the Book panel pulses
        // for it, and the equity value flashes. Watched for up to 30 s
        // (polls run every 10 s; a flash lasts 1.4 s, a pulse 10 s).
        {
          let seen = { ev: false, pulse: false, flash: false };
          for (let i = 0; i < 300 && !(seen.ev && seen.pulse && seen.flash); i++) {
            const s = await page.evaluate(() => ({
              ev: !!document.querySelector('.vt-live-ev[data-kind="trade"]'),
              pulse: !!document.querySelector('section.vt-bot-panel[data-panel="book"][data-pulse="trade"]'),
              flash: !!document.querySelector('.vt-bot-kpi [data-flash]'),
            }));
            seen = { ev: seen.ev || s.ev, pulse: seen.pulse || s.pulse, flash: seen.flash || s.flash };
            await page.waitForTimeout(100);
          }
          if (!seen.ev) fail("live: the fresh trade never reached the Now tile's latest-decision line");
          if (!seen.pulse) fail("live: the Book panel never pulsed for the fresh trade");
          if (!seen.flash) fail("live: no KPI value flashed while the equity moved");
        }
        // ── OVERVIEW review shots (2026-11-06): back to the overview, the first card and the regime / flows sections
        // open, then each overview panel on its own (.visual/<page>-<panel>-<w>.png). LAST on purpose: a panel taller
        // than the viewport makes Playwright resize the viewport for the shot, after which Chromium reports
        // (pointer: coarse) false — every touch-sized control a later check measured would shrink.
        try {
          const back = page.locator('.vt-bot-modebar button[data-view="overview"]').first();
          if (!(await back.count())) fail("overview shots: no Overview control in the cockpit");
          else {
            if (vp.touch) await back.tap({ timeout: 5000 }); else await back.click({ timeout: 5000 });
            for (let i = 0; i < 40; i++) { if (await page.locator('.vt-bot-page[data-mode="overview"] section.vt-bot-panel[data-panel="picks"]').count()) break; await page.waitForTimeout(100); }
            await page.waitForTimeout(400);
            for (const sel of ['.vt-pick[data-source="journal"] .vt-pick-hd', '.vt-why-sec[data-sec="regime"] .vt-why-sh > button:not(.vt-help)', '.vt-why-sec[data-sec="flows"] .vt-why-sh > button:not(.vt-help)']) {
              const b = page.locator(sel).first();
              if ((await b.count()) && (await b.isEnabled())) { await b.scrollIntoViewIfNeeded(); if (vp.touch) await b.tap({ timeout: 5000 }); else await b.click({ timeout: 5000 }); await page.waitForTimeout(150); }
            }
            // viewport-only shots (a clip inside the viewport, or the viewport itself): the page is a fixed layer with its
            // own scroller, so an element shot taller than the viewport would resize the viewport and draw the shell over it
            // + "book" (2026-11-12, KNOWN BROKEN #50a): the share rows' exit lines are a review surface of their own
            for (const id of ["perf", "why", "picks", "daily", "book"]) {
              const el = page.locator(`section.vt-bot-panel[data-panel="${id}"]`).first();
              if (!(await el.count())) continue;
              await el.evaluate((e) => e.scrollIntoView({ block: "start" }));
              await page.waitForTimeout(150);
              const bb = await el.boundingBox();
              if (bb && bb.y >= 0 && bb.y + bb.height <= vp.h) {
                await page.screenshot({ path: path.join(OUT, `${name}-${id}-${vp.w}.png`), clip: { x: bb.x, y: bb.y, width: bb.width, height: bb.height }, animations: "disabled" });
              } else {
                await page.screenshot({ path: path.join(OUT, `${name}-${id}-${vp.w}.png`), animations: "disabled" });
                await page.evaluate((dy) => { const sc = document.querySelector(".vt-bot-stack") || document.querySelector('.vt-bot-grid[data-mode="overview"]'); if (sc) sc.scrollBy(0, dy); }, vp.h - 160);
                await page.waitForTimeout(150);
                await page.screenshot({ path: path.join(OUT, `${name}-${id}-${vp.w}-2.png`), animations: "disabled" });
              }
            }
          }
        } catch (e) {
          fail(`overview shots: driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
        }
      }

      // ── TODAY TILE BETWEEN SESSIONS ([REPAIR] 2026-11-06; server/todayTile.ts, client/src/bot/today.ts): the
      // pre-market page. The eyebrow names the last session ("Thu Nov 5 session"; the phone and the folded l-tier strip
      // drop the word "session"), never "Today", and is not cut; the change runs close to close, from the close before
      // that session (its daily line's record) to its own close, and the printed numbers ADD UP (end − prior = the
      // change; the change ÷ prior = its percent; percent − SPY = the headline); SPY's note says "close" (its final
      // bar); the last line names both days ("Wed $X → Thu $Y"; the phone keeps the numbers); the hover says when the
      // read fell and where every number came from, and calls nothing "today". The tile never clips its content.
      if (name === "botnight") {
        const fail = (m) => checks.failures.push("today-between: " + m);
        try {
          for (let i = 0; i < 40; i++) { if (await page.locator('.vt-bot-kpi .vt-bot-panel[data-kpi="today"][data-phase="between"]').count()) break; await page.waitForTimeout(100); }
          const fx = PAGE_FIXTURES["/api/bot/account"] && PAGE_FIXTURES["/api/bot/account"].today;
          const td = await page.evaluate(() => {
            const t = document.querySelector('.vt-bot-kpi .vt-bot-panel[data-kpi="today"]');
            if (!t) return null;
            const eb = t.querySelector(".vt-kpi-today-eb");
            const lines = [...t.querySelectorAll(".vt-kpi-today-ln > span")].map((x) => ({ text: (x.textContent || "").trim(), shown: (x.innerText || "").replace(/\s+/g, " ").trim(), cut: x.scrollWidth - x.clientWidth > 1 }));
            const r = t.getBoundingClientRect();
            return { phase: t.getAttribute("data-phase"), text: (t.textContent || "").replace(/\s+/g, " ").trim(), eb: (eb?.innerText || "").replace(/\s+/g, " ").trim(), ebCut: eb ? eb.scrollWidth - eb.clientWidth > 1 : true,
              lines, over: Math.max(0, t.scrollHeight - t.clientHeight), w: Math.round(r.width), h: Math.round(r.height),
              title: t.querySelector(".vt-kpi-today")?.getAttribute("title") || "", headline: (t.querySelector(".vt-bot-kpiv")?.textContent || "").trim() };
          });
          if (!td) fail("no Today tile in the KPI strip");
          else if (!fx || fx.phase !== "between") fail("the botnight fixture carries no between-sessions today block");
          else {
            console.log(`  today (between) ${vp.w}px: ${td.w}×${td.h} — eyebrow "${td.eb}" — "${td.text}"`);
            const M = "\u2212";
            const words = (day) => { const d = new Date(`${day}T12:00:00Z`); return { wd: d.toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" }), mon: d.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" }), dd: d.getUTCDate() }; };
            const cur = words(fx.day), prev = words(fx.prev_day);
            const money = (v) => `$${Math.round(Math.abs(v)).toLocaleString("en-US")}`;
            const signed = (v, suffix) => `${v < 0 ? M : v > 0 ? "+" : ""}${Math.abs(v).toFixed(2)}${suffix}`;
            const folded = vp.w >= 1180 && vp.w < 1920;
            const wordShown = vp.w >= 640 && !folded;
            if (td.phase !== "between") fail(`the tile reads phase "${td.phase}"`);
            const ebWant = new RegExp(`^${cur.wd} ${cur.mon} ${cur.dd}${wordShown ? " session" : ""} · vs SPY\\??$`, "i");
            if (!ebWant.test(td.eb)) fail(`the eyebrow reads "${td.eb}" (expected "${cur.wd} ${cur.mon} ${cur.dd}${wordShown ? " session" : ""} · vs SPY")`);
            if (/today/i.test(td.eb)) fail(`the eyebrow still says today between sessions — "${td.eb}"`);
            if (td.ebCut) fail(`the eyebrow is cut short — "${td.eb}"`);
            if (/NaN|undefined|null/.test(td.text)) fail(`a bare NaN / undefined / null in "${td.text}"`);
            if (td.over > 1) fail(`the tile clips ${td.over}px of its own content`);
            if (td.lines.length !== 3) fail(`${td.lines.length} caption lines (expected the change, SPY, and prior → close)`);
            for (const l of td.lines) if (l.cut) fail(`a line is cut short — "${l.text}"`);
            const chgWant = `${fx.pnl_usd < 0 ? M : "+"}${money(fx.pnl_usd)} · ${signed(fx.pnl_pct, "%")}`;
            if (td.lines[0]?.text !== chgWant) fail(`the change line reads "${td.lines[0]?.text}" (expected "${chgWant}")`);
            const spyWant = `SPY ${signed(fx.spy_pct_today, "%")} · close`;
            if (td.lines[1]?.text !== spyWant) fail(`the SPY line reads "${td.lines[1]?.text}" (expected "${spyWant}": the final bar)`);
            const pathWant = `${prev.wd} ${money(fx.equity_prev_close)} → ${cur.wd} ${money(fx.equity_end)}`;
            if (td.lines[2]?.text !== pathWant) fail(`the last line reads "${td.lines[2]?.text}" (expected "${pathWant}")`);
            // the phone and the folded strip drop the first day, never the second ("→ Thu": the session's close, not equity now)
            const shownWant = vp.w < 640 || folded ? `${money(fx.equity_prev_close)} → ${cur.wd} ${money(fx.equity_end)}` : pathWant;
            if (td.lines[2]?.shown !== shownWant) fail(`the last line shows "${td.lines[2]?.shown}" at ${vp.w}px (expected "${shownWant}")`);
            if (td.headline !== signed(fx.excess_pct, " pp")) fail(`the headline reads "${td.headline}" (expected "${signed(fx.excess_pct, " pp")}")`);
            // the arithmetic, from the printed numbers
            const num = (x) => Number(String(x).replace(/\u2212/g, "-").replace(/[$,%+]/g, "").replace(/ pp$/, ""));
            const m = /([−+]?\$[\d,]+) · ([−+]?\d+\.\d\d)%/.exec(td.lines[0]?.text || "");
            const pm = /(\$[\d,]+) → \S+ (\$[\d,]+)/.exec(td.lines[2]?.text || "");
            const sm = /SPY ([−+]?\d+\.\d\d)%/.exec(td.lines[1]?.text || "");
            if (!m || !pm || !sm) fail(`the printed numbers could not be read — "${td.text}"`);
            else {
              if (Math.abs(num(pm[2]) - num(pm[1]) - num(m[1])) > 1) fail(`${pm[1]} → ${pm[2]} does not agree with the change ${m[1]}`);
              if (Math.abs((num(m[1]) / num(pm[1])) * 100 - num(m[2])) > 0.011) fail(`${m[1]} on ${pm[1]} is not ${m[2]}%`);
              if (Math.abs(num(m[2]) - num(sm[1]) - num(td.headline)) > 0.011) fail(`${m[2]}% − SPY ${sm[1]}% is not the headline ${td.headline}`);
            }
            // the hover: which session, when the read fell, where each number came from — and nothing called today
            if (!new RegExp(`^${cur.wd} ${cur.mon} ${cur.dd} session \\(${fx.day} ET, shown pre-market\\)`).test(td.title)) fail(`the hover does not name the session and the moment — "${td.title.slice(0, 160)}"`);
            if (!/last_equity/.test(td.title) || !/price only, no dividends/.test(td.title)) fail(`the hover carries no provenance — "${td.title.slice(0, 160)}"`);
            if (!/after the close: the final bar/.test(td.title)) fail("the hover does not say SPY's bar is the final close");
            if (!/daily line recorded Alpaca's last_equity/.test(td.title)) fail("the hover does not say the prior close is the daily line's record");
            if (!/since that close \(after-hours and pre-market marks\)/.test(td.title)) fail("the hover does not carry equity now and the drift since the close");
            if (/\btoday\b/i.test(td.title)) fail(`the hover calls something today between sessions — "${td.title.slice(0, 200)}"`);
            if (!/session: pre-market, by the broker clock/.test(td.title)) fail("the hover does not say what decided the session");
          }
          const strip = page.locator(".vt-bot-kpi").first();
          if (await strip.count()) {
            await strip.evaluate((e) => e.scrollIntoView({ block: "start" }));
            await page.waitForTimeout(150);
            const bb = await strip.boundingBox();
            if (bb && bb.y >= 0 && bb.y + bb.height <= vp.h) await page.screenshot({ path: path.join(OUT, `${name}-today-${vp.w}.png`), clip: { x: bb.x, y: bb.y, width: bb.width, height: bb.height }, animations: "disabled" });
            else await page.screenshot({ path: path.join(OUT, `${name}-today-${vp.w}.png`), animations: "disabled" });
          }
        } catch (e) {
          fail(`driver error — ${String(e?.message || e).replace(/\s+/g, " ").slice(0, 300)}`);
        }
      }

      // ── IMAGERY CAPTURE-DATE RATCHET (Phase 3a/5, 2026-07-07): the
      // DESIGN.md imagery-honesty rule requires the capture-date chip on
      // the map whenever the imagery base is on. External Esri calls
      // abort in the harness, so the chip must show a designed honest
      // state ("checking…"/"unknown") — its ABSENCE is the failure.
      if (cfg.map) {
        try {
          const chip = await page.waitForSelector('[data-testid="imagery-date"]', { timeout: 5000 }).catch(() => null);
          if (!chip) {
            checks.failures.push("imagery-date: combined status bar absent with imagery base on (data-testid=imagery-date) — DESIGN.md imagery-honesty rule");
          } else {
            // combined bottom status bar (2026-07-22): "<scale> · z<zoom> ·
            // <ISO date | date n/a | …>" — must carry a scale unit, a zoom,
            // and a designed capture-date state (never a fabricated date).
            const txt = (await chip.textContent()) || "";
            if (!/\b(mi|km|ft|m)\b/.test(txt) || !/z\d/.test(txt) || !/\d{4}-\d{2}-\d{2}|date n\/a|…/.test(txt)) {
              checks.failures.push(`imagery-date: status bar not a designed state (scale·zoom·date): '${txt.slice(0, 60)}'`);
            }
          }
        } catch (e) {
          checks.failures.push("imagery-date: driver error — " + (e?.message || e));
        }
      }

      // ── W1 GLOBE MODE (console charter, 2026-07-07): the /data map
      // defaults to MapLibre's native globe projection; [data-vt-globe]
      // flips to flat (mercator) and back, persisting the choice in
      // localStorage. Asserts the default, the round-trip, and the
      // persisted preference; screenshots both modes at a low zoom where
      // the sphere silhouette is unmistakable (PR evidence — a black
      // square or a flat rectangle in the "globe" shot is a review fail).
      try {
        if (!cfg.map) throw { skip: true };
        const globeBtn = page.locator("[data-vt-globe]");
        if (!(await globeBtn.count())) {
          checks.failures.push("globe: [data-vt-globe] projection toggle missing");
          throw { skip: true };
        }
        const projOf = () => page.evaluate(() => {
          try { return (window.__vtMap.getProjection?.() || {}).type || "mercator"; } catch { return "probe-error"; }
        });
        const p0 = await projOf();
        if (p0 !== "globe") checks.failures.push(`globe: default projection '${p0}' != 'globe' (charter default)`);
        // low-zoom evidence shots: same camera for globe and flat
        const cam = await page.evaluate(() => {
          const m = window.__vtMap; const c = m.getCenter();
          return { lng: c.lng, lat: c.lat, zoom: m.getZoom() };
        });
        await page.evaluate(() => window.__vtMap.jumpTo({ center: [-96.77, 30], zoom: 1.3 }));
        await page.waitForTimeout(900);
        await page.screenshot({ path: path.join(OUT, `${name}-globe-${vp.w}.png`) });
        await globeBtn.click();
        await page.waitForTimeout(700);
        const p1 = await projOf();
        const pref1 = await page.evaluate(() => { try { return localStorage.getItem("vt-map-globe"); } catch { return null; } });
        if (p1 !== "mercator") checks.failures.push(`globe: after toggle projection '${p1}' != 'mercator'`);
        if (pref1 !== "0") checks.failures.push(`globe: flat choice not persisted (vt-map-globe='${pref1}')`);
        const pressed1 = await globeBtn.getAttribute("aria-pressed");
        if (pressed1 !== "false") checks.failures.push(`globe: toggle aria-pressed '${pressed1}' desynced from flat state`);
        await page.screenshot({ path: path.join(OUT, `${name}-flat-${vp.w}.png`) });
        await globeBtn.click();
        await page.waitForTimeout(700);
        const p2 = await projOf();
        if (p2 !== "globe") checks.failures.push(`globe: toggle back gave '${p2}' != 'globe'`);
        // restore the default camera for the batteries that follow
        await page.evaluate((c) => window.__vtMap.jumpTo({ center: [c.lng, c.lat], zoom: c.zoom }), cam);
        await page.waitForTimeout(500);
      } catch (e) {
        if (!e?.skip) checks.failures.push("globe: driver error — " + (e?.message || e));
      }
      // ── W6 ANALYST PANE (console charter, 2026-07-07): [data-vt-analyst]
      // opens the chat panel; the panel must sit fully inside the viewport
      // at every width (SELF-SEE rule — no overflow past the viewport
      // bottom), its input + send control must be reachable and
      // hit-testable, and close must work. At side-panel widths (>=640)
      // the open panel may not occlude any map control. HONESTY
      // CONSTRAINT: this battery NEVER sends a question — the pane fires
      // POST /api/analyst only on an explicit user send, and the battery
      // asserts exactly zero analyst calls (no faked API responses).
      let analystPosts = 0;
      const onAnalystReq = (r) => {
        try { if (new URL(r.url()).pathname === "/api/analyst") analystPosts++; } catch {}
      };
      try {
        if (!cfg.map) throw { skip: true };
        page.on("request", onAnalystReq);
        const aBtn = page.locator("[data-vt-analyst]");
        if (!(await aBtn.count())) {
          checks.failures.push("analyst: [data-vt-analyst] control missing");
          throw { skip: true };
        }
        await aBtn.click();
        // the pane is a lazy chunk: wait past the Suspense fallback for the
        // real input, not just the panel shell
        const aPanel = await page.waitForSelector("[data-vt-analyst-panel]", { timeout: 8000 }).catch(() => null);
        const aInput = await page.waitForSelector("[data-vt-analyst-input]", { timeout: 8000 }).catch(() => null);
        if (!aPanel) {
          checks.failures.push("analyst: panel did not open from its own control");
        } else {
          if (!aInput) checks.failures.push("analyst: input never appeared (lazy chunk failed to load?)");
          // settle the pane's 0.18s entry animation + one composited paint
          // BEFORE geometry checks and the evidence screenshot (2026-07-07
          // flake: at 1440 the DOM checks passed but the capture composited
          // before the panel's first stable paint — an empty-map screenshot
          // with a green battery is worthless as PR evidence)
          await page.waitForTimeout(450);
          const aChecks = await page.evaluate(() => {
            const fails = [];
            const p = document.querySelector("[data-vt-analyst-panel]");
            if (!p) return ["analyst: panel vanished mid-check"];
            const r = p.getBoundingClientRect();
            if (r.bottom > innerHeight + 1) fails.push(`analyst: panel bottom ${Math.round(r.bottom)} past viewport ${innerHeight} — SELF-SEE`);
            if (r.top < -1 || r.left < -1 || r.right > innerWidth + 1) {
              fails.push(`analyst: panel outside viewport (l=${Math.round(r.left)} t=${Math.round(r.top)} r=${Math.round(r.right)})`);
            }
            for (const [label, sel] of [["input", "[data-vt-analyst-input]"], ["send", "[data-vt-analyst-send]"]]) {
              const el = document.querySelector(sel);
              if (!el) { fails.push(`analyst: ${label} control missing`); continue; }
              const b = el.getBoundingClientRect();
              if (b.width < 4 || b.height < 4) { fails.push(`analyst: ${label} has no size`); continue; }
              if (b.bottom > innerHeight + 1 || b.top < -1) { fails.push(`analyst: ${label} not inside the viewport`); continue; }
              const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
              if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                fails.push(`analyst: ${label} covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'>`);
              }
            }
            // side-panel widths: the open pane may not occlude map controls
            // (phone is a bottom sheet — transient, closable, same accepted
            // behavior as the detail-card sheet over the zoom cluster)
            if (innerWidth >= 640) {
              for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
                const el = document.querySelector(sel);
                if (!el) { fails.push(`analyst: map control ${sel} missing with pane open`); continue; }
                const b = el.getBoundingClientRect();
                const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
                if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                  fails.push(`analyst: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with pane open`);
                }
              }
            }
            return fails;
          });
          checks.failures.push(...aChecks);
          // input is editable — fill/verify/clear, never submit
          if (aInput) {
            await page.fill("[data-vt-analyst-input]", "harness reachability probe");
            const v = await page.inputValue("[data-vt-analyst-input]");
            if (v !== "harness reachability probe") checks.failures.push("analyst: input not editable");
            await page.fill("[data-vt-analyst-input]", "");
          }
          // animations:"disabled" fast-forwards the entry fade at capture —
          // under software GL the composited frame otherwise races the
          // animation (2026-07-07: blank/ghosted panel shots with green
          // DOM checks). Capture-time only; the page itself is untouched.
          await page.screenshot({ path: path.join(OUT, `${name}-analyst-${vp.w}.png`), animations: "disabled" });
          // close via the panel's own close control, then re-assert gone
          await page.click('[data-vt-analyst-panel] [aria-label="Close analyst"]', { timeout: 2000 })
            .catch(() => checks.failures.push("analyst: close control unclickable"));
          await page.waitForTimeout(250);
          if (await page.locator("[data-vt-analyst-panel]").count()) {
            checks.failures.push("analyst: panel still open after its close control");
          }
        }
        if (analystPosts > 0) {
          checks.failures.push(`analyst: battery fired ${analystPosts} POST /api/analyst — the battery must never send a question`);
        }
      } catch (e) {
        if (!e?.skip) checks.failures.push("analyst: driver error — " + (e?.message || e));
      } finally {
        page.off("request", onAnalystReq);
      }
      // ── W3 TIME SCRUBBER (console charter): [data-vt-timescrub] opens the
      // replay panel. Unlike the analyst pane, opening this panel is EXPECTED
      // to fire a read-only GET on open — the battery asserts at least one
      // fires, catching a silent-fetch regression, the mirror image of the
      // analyst's never-fires assertion. TIME MACHINE v2 T-2 (2026-08-13):
      // the default layer (aircraft) now fetches ONE window
      // (/api/data/aircraft/window) instead of a per-instant snapshot; every
      // other layer is unchanged, so both paths count.
      let snapshotGets = 0;
      const onSnapshotReq = (r) => {
        try {
          const p = new URL(r.url()).pathname;
          if (p === "/api/data/snapshot" || p === "/api/data/aircraft/window") snapshotGets++;
        } catch {}
      };
      try {
        if (!cfg.map) throw { skip: true };
        page.on("request", onSnapshotReq);
        const tsBtn = page.locator("[data-vt-timescrub]");
        if (!(await tsBtn.count())) {
          checks.failures.push("timescrub: [data-vt-timescrub] control missing");
          throw { skip: true };
        }
        await tsBtn.click();
        const tsPanel = await page.waitForSelector("[data-vt-timescrub-panel]", { timeout: 8000 }).catch(() => null);
        const tsSlider = await page.waitForSelector("[data-vt-timescrub-slider]", { timeout: 8000 }).catch(() => null);
        if (!tsPanel) {
          checks.failures.push("timescrub: panel did not open from its own control");
        } else {
          if (!tsSlider) checks.failures.push("timescrub: slider never appeared (lazy chunk failed to load?)");
          await page.waitForTimeout(600); // entry animation
          // The close-approach toggle renders only once the aircraft window
          // fetch has landed and been processed. A fixed 600 ms was enough on
          // an idle machine and not on a loaded one (2026-11-16: two harness
          // runs side by side each failed the four close-approach checks
          // below; either one alone passed), so wait for the control itself,
          // bounded. If it never appears the check below still fails.
          await page.waitForSelector("[data-vt-timescrub-ca-toggle]", { state: "attached", timeout: 8000 }).catch(() => null);
          // FLIGHT PROGRAM replay: the close-approach list is a collapsed
          // disclosure by default — open it through its own control (DOM
          // click: the check below is about the list, not the toggle's
          // hit-test, which the Legend card can overlap at desktop widths)
          await page.$eval("[data-vt-timescrub-ca-toggle]", (el) => el.click())
            .catch(() => checks.failures.push("timescrub: close-approach toggle missing for an aircraft window"));
          await page.waitForTimeout(250);
          const tsChecks = await page.evaluate(() => {
            const fails = [];
            const p = document.querySelector("[data-vt-timescrub-panel]");
            if (!p) return ["timescrub: panel vanished mid-check"];
            const r = p.getBoundingClientRect();
            if (r.bottom > innerHeight + 1) fails.push(`timescrub: panel bottom ${Math.round(r.bottom)} past viewport ${innerHeight} — SELF-SEE`);
            if (r.top < -1 || r.left < -1 || r.right > innerWidth + 1) {
              fails.push(`timescrub: panel outside viewport (l=${Math.round(r.left)} t=${Math.round(r.top)} r=${Math.round(r.right)})`);
            }
            // FLIGHT PROGRAM replay: the fixture window carries close
            // approaches — the list must render with reachable rows (the
            // panel scrolls internally; a row may sit below the fold only
            // while the panel itself is scrollable)
            const caRows = p.querySelectorAll(".vt-timescrub-ca-row");
            if (!p.querySelector("[data-vt-timescrub-ca]")) fails.push("timescrub: close-approach section missing for an aircraft window");
            else if (caRows.length < 1) fails.push("timescrub: close-approach list rendered no rows for a fixture with 2");
            else {
              const rr = caRows[0].getBoundingClientRect();
              if (rr.width < 4 || rr.height < 30) fails.push("timescrub: close-approach row has no usable size");
              if (rr.bottom > r.bottom + 1 && p.scrollHeight <= p.clientHeight + 2) fails.push("timescrub: close-approach row below the panel with no internal scroll");
            }
            for (const [label, sel] of [["layer select", "[data-vt-timescrub-layer]"], ["slider", "[data-vt-timescrub-slider]"], ["play", "[data-vt-timescrub-play]"]]) {
              const el = document.querySelector(sel);
              if (!el) { fails.push(`timescrub: ${label} control missing`); continue; }
              const b = el.getBoundingClientRect();
              if (b.width < 4 || b.height < 4) { fails.push(`timescrub: ${label} has no size`); continue; }
              if (b.bottom > innerHeight + 1 || b.top < -1) { fails.push(`timescrub: ${label} not inside the viewport`); continue; }
              const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
              if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                fails.push(`timescrub: ${label} covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'>`);
              }
            }
            if (innerWidth >= 640) {
              for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
                const el = document.querySelector(sel);
                if (!el) { fails.push(`timescrub: map control ${sel} missing with panel open`); continue; }
                const b = el.getBoundingClientRect();
                const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
                if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                  fails.push(`timescrub: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with panel open`);
                }
              }
            }
            return fails;
          });
          checks.failures.push(...tsChecks);
          await page.screenshot({ path: path.join(OUT, `${name}-timescrub-${vp.w}.png`), animations: "disabled" });
          // FLIGHT PROGRAM replay: the fleet GL layer must have compiled and
          // drawn under SwiftShader (unit tests only see the GLSL as strings),
          // and focusing an approach must fly to the pair and draw its
          // curtains (fixture fx0/fx1 = real converging 60 s tracks).
          const fleet0 = await page.evaluate(() => {
            const d = window.__vtFleetReplay;
            return d ? { failed: d.failed(), counts: d.counts(), draws: d.draws() } : null;
          });
          if (!fleet0) checks.failures.push("timescrub: fleet replay diagnostics hook (window.__vtFleetReplay) missing");
          else if (fleet0.failed || fleet0.draws.failStreak > 0) checks.failures.push(`timescrub: fleet replay GL layer failing (shader compile/link or draw error) ${JSON.stringify(fleet0.draws)}`);
          else if (fleet0.counts.heads < 1) checks.failures.push(`timescrub: fleet replay drew no heads at the window end (${JSON.stringify(fleet0.counts)})`);
          else if (fleet0.draws.drawnFrames < 1) checks.failures.push("timescrub: fleet replay has heads but completed no draw");
          const cam0 = await page.evaluate(() => {
            const m = window.__vtMap;
            if (!m) return null;
            const c = m.getCenter();
            return { center: [c.lng, c.lat], zoom: m.getZoom(), pitch: m.getPitch(), bearing: m.getBearing() };
          });
          await page.$eval(".vt-timescrub-ca-row", (el) => el.click())
            .catch(() => checks.failures.push("timescrub: no close-approach row to focus"));
          await page.waitForTimeout(2600); // camera flight (1.4 s) + LOD tick + batched rebuild
          const fleet1 = await page.evaluate(() => {
            const d = window.__vtFleetReplay;
            return d ? { failed: d.failed(), counts: d.counts(), draws: d.draws(), lod: d.lod() } : null;
          });
          if (fleet1 && (fleet1.failed || fleet1.draws.failStreak > 0)) {
            checks.failures.push(`timescrub: fleet replay GL layer failing after focusing an approach ${JSON.stringify(fleet1.draws)}`);
          } else if (fleet1 && fleet1.counts.full < 1) {
            checks.failures.push(`timescrub: focused close approach drew no curtain geometry (${JSON.stringify(fleet1)})`);
          }
          await page.screenshot({ path: path.join(OUT, `${name}-timescrub-focus-${vp.w}.png`), animations: "disabled" });
          // restore the camera so the checks after this block see the page
          // exactly as before the focus flight
          if (cam0) await page.evaluate((c) => window.__vtMap?.jumpTo(c), cam0).catch(() => {});
          await page.click('[data-vt-timescrub-panel] [aria-label="Close time machine"]', { timeout: 2000 })
            .catch(() => checks.failures.push("timescrub: close control unclickable"));
          await page.waitForTimeout(250);
          if (await page.locator("[data-vt-timescrub-panel]").count()) {
            checks.failures.push("timescrub: panel still open after its close control");
          }
        }
        if (snapshotGets < 1) {
          checks.failures.push("timescrub: opening the panel fired zero GET /api/data/snapshot or /api/data/aircraft/window — initial fetch regressed");
        }
      } catch (e) {
        if (!e?.skip) checks.failures.push("timescrub: driver error — " + (e?.message || e));
      } finally {
        page.off("request", onSnapshotReq);
      }
      // ── RAIL MENUS (owner report 2026-11-19: "when you click on the
      // mountain, it covers the other icon and moves down. Why does it move?
      // It's where I have my mouse"). The base-map-style menu was laid out in
      // flow beside its button, so opening it pushed the 44px button 23px down
      // onto the time-machine button — and nothing here ever opened it. Every
      // menu on the top-left rail ([data-vt-rail-menu], components/RailMenu)
      // is now driven the way a person drives it — the pointer goes to the
      // button and STAYS there — and must hold, at every width:
      //   • no rail button ([data-vt-rail-btn]) moves when the menu opens, and
      //     the button pressed is still under the pointer;
      //   • the menu covers no rail button and no other control, sits inside
      //     the viewport, and each of its options is reachable (44px on phone);
      //   • a second press on the same spot closes it; so do Escape and a
      //     press outside;
      //   • the same holds with the analyst pane or the time machine open (the
      //     panels that share the space right of the rail): the menu steps
      //     aside instead of opening over or under them.
      // Touch widths tap; the desktop width moves a mouse, which also covers
      // the hover-open path that moved the button in the report.
      try {
        if (!cfg.map) throw { skip: true };
        // start from the page's default layout: a panel an earlier battery
        // failed to close (its own failure, already recorded) must not be read
        // here as "the menu opened over a panel on the default layout"
        const strayPanel = () => page.locator("[data-vt-analyst-panel], [data-vt-timescrub-panel]").count();
        if (await strayPanel()) { await page.keyboard.press("Escape"); await page.waitForTimeout(300); }
        const menuIds = await page.$$eval("[data-vt-rail-menu]", (els) => els.map((e) => e.getAttribute("data-vt-rail-menu")));
        if (!menuIds.length) {
          checks.failures.push("railmenu: no [data-vt-rail-menu] on the map rail (the base-map-style menu is one)");
          throw { skip: true };
        }
        const railSnap = (id) => page.evaluate((menuId) => {
          const box = (el) => { const r = el.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; };
          const rail = {};
          const covered = [];
          for (const el of document.querySelectorAll("[data-vt-rail-btn]")) {
            const k = el.getAttribute("data-vt-rail-btn");
            const b = box(el);
            rail[k] = b;
            const top = document.elementFromPoint(b.x + b.w / 2, b.y + b.h / 2);
            if (!top || !(top === el || el.contains(top))) covered.push(`${k} by <${top ? top.tagName.toLowerCase() : "nothing"} class='${top ? String(top.className).slice(0, 30) : ""}'>`);
          }
          const menu = document.querySelector(`[data-vt-rail-flyout="${menuId}"]`);
          return { rail, covered, menu: menu ? box(menu) : null, mode: menu ? menu.getAttribute("data-vt-rail-placement") : null };
        }, id);
        const moved = (a, b) => Object.keys(a).filter((k) => !b[k] || Math.abs(a[k].x - b[k].x) > 0.5 || Math.abs(a[k].y - b[k].y) > 0.5 || Math.abs(a[k].w - b[k].w) > 0.5 || Math.abs(a[k].h - b[k].h) > 0.5);
        const pressAt = async (x, y) => {
          if (vp.touch) await page.touchscreen.tap(x, y);
          else { await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.up(); }
          await page.waitForTimeout(300);
        };
        // a spot outside the map that is not a control (the top bar's blank
        // stretch): an outside press there closes the menu without touching
        // the map or navigating
        const outside = await page.evaluate(() => {
          for (const fx of [0.5, 0.42, 0.6, 0.35, 0.7]) {
            const x = Math.round(innerWidth * fx), y = 22;
            const el = document.elementFromPoint(x, y);
            if (el && !el.closest("a, button, input, select, [role='button'], .vt-map-page")) return { x, y };
          }
          return null;
        });
        if (!outside) checks.failures.push("railmenu: no blank spot in the top bar to press outside the menu (harness needs one)");
        const panels = [
          { tag: "", open: null },
          { tag: "+analyst", open: "[data-vt-analyst]", panel: "[data-vt-analyst-panel]", ready: "[data-vt-analyst-input]", close: '[data-vt-analyst-panel] [aria-label="Close analyst"]' },
          { tag: "+timescrub", open: "[data-vt-timescrub]", panel: "[data-vt-timescrub-panel]", ready: "[data-vt-timescrub-slider]", close: '[data-vt-timescrub-panel] [aria-label="Close time machine"]' },
        ];
        for (const id of menuIds) {
          for (const sc of panels) {
            const who = `railmenu ${id}${sc.tag}`;
            if (sc.open) {
              await page.click(sc.open, { timeout: 2000 }).catch(() => {});
              await page.waitForSelector(sc.ready, { timeout: 8000 }).catch(() => null);
              await page.waitForTimeout(450); // the panel's 0.18s entry animation
              if (!(await page.locator(sc.panel).count())) { checks.failures.push(`${who}: the panel did not open, so the combination was not checked`); continue; }
            }
            const base = await railSnap(id);
            const btn = base.rail[id];
            if (!btn) { checks.failures.push(`${who}: its button carries no [data-vt-rail-btn="${id}"]`); continue; }
            if (Object.keys(base.rail).length < 5) checks.failures.push(`${who}: only ${Object.keys(base.rail).length} rail buttons carry [data-vt-rail-btn] (fullscreen, globe, analyst, style, time machine expected)`);
            if (base.menu) checks.failures.push(`${who}: menu already open before its button was pressed`);
            const px = btn.x + btn.w / 2, py = btn.y + btn.h / 2;
            if (!vp.touch) {
              // the report's exact motion: the mouse arrives on the button
              await page.mouse.move(px, py);
              await page.waitForTimeout(250);
              const hov = await railSnap(id);
              const hm = moved(base.rail, hov.rail);
              if (hm.length) checks.failures.push(`${who}: hovering its button moved rail button(s) ${hm.join(", ")}`);
            }
            await pressAt(px, py);
            const open = await railSnap(id);
            if (!open.menu) { checks.failures.push(`${who}: one press on its button did not open the menu`); }
            else {
              const mv = moved(base.rail, open.rail);
              if (mv.length) checks.failures.push(`${who}: opening the menu moved rail button(s) ${mv.join(", ")} (e.g. ${mv[0]} y ${Math.round(base.rail[mv[0]].y)} -> ${Math.round(open.rail[mv[0]]?.y ?? -1)})`);
              if (open.covered.length) checks.failures.push(`${who}: rail button covered with the menu open — ${open.covered.join("; ")}`);
              // "overlap" = the placement found no free spot. Never legitimate on
              // the page's default layout; with a panel open on a narrow page
              // there can truly be none (768 with the analyst pane, the layers
              // panel and the legend all up), and the menu then opens on top,
              // beside its button — reported, and still held to everything else.
              const crowded = open.mode === "overlap" && !!sc.open;
              if (open.mode === "overlap" && !sc.open) checks.failures.push(`${who}: the menu found no free spot on the default layout and opened over another control (placement "overlap")`);
              if (crowded) checks.warnings.push(`${who}: no free spot at ${vp.w}px with this panel open — the menu opened on top, beside its button`);
              const geo = await page.evaluate(({ menuId, x, y, phone, crowded }) => {
                const fails = [];
                const menu = document.querySelector(`[data-vt-rail-flyout="${menuId}"]`);
                const slot = document.querySelector(`[data-vt-rail-menu="${menuId}"]`);
                const trigger = document.querySelector(`[data-vt-rail-btn="${menuId}"]`);
                const m = menu.getBoundingClientRect();
                // intersection in px²; boxes closer than half a pixel count as touching
                const area = (r) => { const w = Math.min(m.right, r.right) - Math.max(m.left, r.left), h = Math.min(m.bottom, r.bottom) - Math.max(m.top, r.top); return w > 0.5 && h > 0.5 ? w * h : 0; };
                const under = document.elementFromPoint(x, y);
                if (!under || !(under === trigger || trigger.contains(under))) fails.push("the pressed button is no longer under the pointer");
                if (m.left < -0.5 || m.top < -0.5 || m.right > innerWidth + 0.5 || m.bottom > innerHeight + 0.5) fails.push(`menu outside the viewport (l=${Math.round(m.left)} t=${Math.round(m.top)} r=${Math.round(m.right)} b=${Math.round(m.bottom)} of ${innerWidth}x${innerHeight})`);
                const page_ = document.querySelector(".vt-map-page").getBoundingClientRect();
                if (m.bottom > page_.bottom + 0.5 || m.top < page_.top - 0.5) fails.push(`menu outside the map page (t=${Math.round(m.top)} b=${Math.round(m.bottom)}, page ${Math.round(page_.top)}..${Math.round(page_.bottom)})`);
                for (const el of document.querySelectorAll("[data-vt-rail-btn]")) {
                  if (area(el.getBoundingClientRect()) > 0) fails.push(`menu covers rail button ${el.getAttribute("data-vt-rail-btn")}`);
                }
                // every other control on the map page: its box and the menu's must not intersect
                const hit = new Set();
                for (const el of document.querySelectorAll(".vt-map-page button, .vt-map-page a[href], .vt-map-page input, .vt-map-page select, .vt-map-page textarea, .vt-map-page [role='button'], .vt-layer-panel, .vt-legend-float, .vt-analyst-panel, .vt-timescrub-panel, .vt-site-card, .vt-aero-card, .vt-map-statusbar, .vt-device-notice, .vt-nuke-timebar")) {
                  if (slot.contains(el)) continue;
                  const cs = getComputedStyle(el);
                  const r = el.getBoundingClientRect();
                  if (cs.display === "none" || cs.visibility === "hidden" || r.width < 1 || r.height < 1) continue;
                  if (area(r) > 0) hit.add(`<${el.tagName.toLowerCase()} class='${String(el.className).slice(0, 30)}'>`);
                }
                if (hit.size && !crowded) fails.push(`menu covers ${hit.size} other control(s): ${[...hit].slice(0, 3).join(", ")}`);
                const opts = [...menu.querySelectorAll("button")];
                if (!opts.length) fails.push("menu has no options");
                for (const o of opts) {
                  const r = o.getBoundingClientRect();
                  const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
                  const name = o.textContent.trim();
                  if (!top || !(top === o || o.contains(top))) fails.push(`option "${name}" is not reachable (covered or scrolled out)`);
                  if (phone && (r.width < 43.5 || r.height < 43.5)) fails.push(`option "${name}" is ${Math.round(r.width)}x${Math.round(r.height)} — under the 44px touch target`);
                }
                return { fails, labels: opts.map((o) => o.textContent.trim()) };
              }, { menuId: id, x: px, y: py, phone: vp.w < 640, crowded });
              for (const f of geo.fails) checks.failures.push(`${who}: ${f}`);
              if (id === "style") {
                // the menu's whole offer (2026-11-19): three base styles, five chart views
                const want = ["Natural", "Night", "Minimal", "Satellite", "VFR Sectional", "VFR Terminal", "IFR Low", "IFR High"];
                const gone = want.filter((w) => !geo.labels.includes(w));
                if (gone.length) checks.failures.push(`${who}: option(s) missing from the menu: ${gone.join(", ")}`);
              }
              await page.screenshot({ path: path.join(OUT, `${name}-railmenu-${id}${sc.tag.replace("+", "-")}-${vp.w}.png`), animations: "disabled" });
            }
            // a second press on the same spot closes it, and nothing moved
            await pressAt(px, py);
            const shut = await railSnap(id);
            if (open.menu && shut.menu) checks.failures.push(`${who}: a second press on its button did not close the menu`);
            const back = moved(base.rail, shut.rail);
            if (back.length) checks.failures.push(`${who}: rail button(s) ${back.join(", ")} not back in place after the menu closed`);
            if (!sc.open) {
              // Escape closes (DESIGN.md: Escape closes panels/popovers)
              await pressAt(px, py);
              if (!(await railSnap(id)).menu) checks.failures.push(`${who}: the menu did not reopen on the next press`);
              await page.keyboard.press("Escape");
              await page.waitForTimeout(250);
              if ((await railSnap(id)).menu) checks.failures.push(`${who}: Escape did not close the menu`);
              // a press outside closes
              if (outside) {
                await pressAt(px, py);
                if (!(await railSnap(id)).menu) checks.failures.push(`${who}: the menu did not reopen after Escape`);
                await pressAt(outside.x, outside.y);
                if ((await railSnap(id)).menu) checks.failures.push(`${who}: a press outside did not close the menu`);
              }
            } else {
              // the panel's own close control is the analyst / time-machine
              // batteries' assertion; here it only has to be gone before the
              // next combination (Escape closes panels — DESIGN.md)
              await page.click(sc.close, { timeout: 2000 }).catch(() => {});
              await page.waitForTimeout(250);
              if (await strayPanel()) { await page.keyboard.press("Escape"); await page.waitForTimeout(300); }
              if (await strayPanel()) checks.failures.push(`${who}: the panel would not close afterwards (close control and Escape)`);
            }
            if ((await railSnap(id)).menu) {
              checks.failures.push(`${who}: menu left open at the end of its checks`);
              await page.keyboard.press("Escape");
            }
          }
        }
        // park the mouse off the rail so no later battery starts on a hover
        if (!vp.touch && outside) await page.mouse.move(outside.x, outside.y);
      } catch (e) {
        if (!e?.skip) checks.failures.push("railmenu: driver error — " + (e?.message || e));
      }
      // ── SELF-SEE (DESIGN.md, human-approved 2026-07-04): after any panel/
      // overlay change, ALL registered content must be reachable — visible or
      // behind an on-screen expand control. The 2026-07-04 defect: the panel
      // grew past the viewport with lower rows unreachable while this harness
      // passed. These assertions check what the human actually checks.
      try {
        if (!cfg.map) throw { skip: true };
        // open the panel via its own on-screen control (as a user would)
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(250);
        // expand every collapsed group via its visible header
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        // GROUP_ROW_CAP progressive disclosure: rows past a group's cap sit
        // behind a visible "+N more — show all" button by DESIGN — expand
        // them all so "reachable" means what the UX means. (First crossed
        // 2026-07-15: biomass made environmental 13 > cap 12, hiding the
        // forest row from this check while the app was working correctly.)
        for (let round = 0; round < 8; round++) {
          const more = page.locator(".vt-layer-showmore").first();
          if (!(await more.count())) break;
          await more.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        const layerIds = FIXTURES["/api/data/layers"].layers.map((l) => l.id);
        const selfSee = await page.evaluate((ids) => {
          const fails = [];
          const panel = document.querySelector(".vt-layer-panel");
          if (!panel) { fails.push("self-see: layer panel not rendered after opening"); return fails; }
          const pr = panel.getBoundingClientRect();
          if (pr.bottom > innerHeight + 2) fails.push(`self-see: panel bottom ${Math.round(pr.bottom)} past viewport ${innerHeight} — content unreachable`);
          if (panel.scrollHeight > panel.clientHeight + 4) {
            const oy = getComputedStyle(panel).overflowY;
            if (!/(auto|scroll)/.test(oy)) fails.push("self-see: panel overflows without internal scrolling");
          }
          for (const id of ids) {
            const row = panel.querySelector(`[data-vt-layer="${id}"]`);
            if (!row) { fails.push(`self-see: registered layer '${id}' has no reachable panel row`); continue; }
            row.scrollIntoView({ block: "center" });
            const toggle = row.querySelector('[role="switch"]');
            if (!toggle) { fails.push(`self-see: layer '${id}' has no toggle`); continue; }
            const tr = toggle.getBoundingClientRect();
            if (tr.right > innerWidth + 1 || tr.left < -1) fails.push(`self-see: '${id}' toggle off-screen horizontally at ${Math.round(tr.left)}..${Math.round(tr.right)}`);
            if (tr.bottom > innerHeight + 1 || tr.top < -1) fails.push(`self-see: '${id}' toggle not scrollable into viewport`);
            const hit = document.elementFromPoint(tr.left + tr.width / 2, tr.top + tr.height / 2);
            if (hit && !toggle.contains(hit) && hit !== toggle && !hit.contains(toggle)) {
              fails.push(`self-see: '${id}' toggle covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'>`);
            }
          }
          // v2.4 CONTROL OCCLUSION: with the panel OPEN, no map control may
          // sit under it (the production defect: zoom buttons covered).
          // [data-vt-globe] added W1, [data-vt-analyst] added W6 (console
          // charter): both are registered map controls like the others.
          for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
            const el = document.querySelector(sel);
            if (!el) { fails.push(`self-see: map control ${sel} missing`); continue; }
            const r = el.getBoundingClientRect();
            if (r.width < 4 || r.height < 4) { fails.push(`self-see: map control ${sel} has no size`); continue; }
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
              fails.push(`self-see: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with panel open`);
            }
          }
          // v2.4 ETERNAL-SPINNER rule (armed): any row loading >30s must
          // carry a designed note (retrying / activating / awaiting).
          for (const row of panel.querySelectorAll('[data-vt-rt="loading"]')) {
            const since = Number(row.getAttribute("data-vt-since") || 0);
            if (since && Date.now() - since > 30_000 && !row.querySelector(".vt-layer-covnote")) {
              fails.push(`self-see: '${row.getAttribute("data-vt-layer")}' bare loading spinner >30s — eternal-spinner rule`);
            }
          }
          return fails;
        }, layerIds);
        checks.failures.push(...selfSee);
        // ── SELF-SEE × ANALYST COMBO (2026-07-25 follow-up to the phone-width
        // FAB/panel occlusion fix above, filed as NEXT item (1) in that same
        // session's log): index.css collapses the nav cluster down to the
        // same round `.vt-nav-fab` the phone uses whenever the analyst pane
        // is open, at ANY width <=1279px (`.vt-map-page[data-vt-analyst-
        // open="true"] .vt-nav-fab`) — not just the phone breakpoint the CSS
        // fix above was scoped to. If the layer panel is ALSO open at a mid
        // width (640-1279px, i.e. our 768 canonical width), the FAB COULD sit
        // over the panel's own last row exactly like the already-fixed phone
        // case — no test opened both panels together until now, so this is
        // now a permanent ratchet against that regressing (verified this
        // session with a manual geometry probe that the FAB and panel really
        // do share the same screen rectangle at 768px; the check below found
        // no ACTUAL row occluded — see index.css's updated comment on the
        // 639px rule for the full finding — but it stays wired so a future
        // content/layout change that does cause an overlap gets caught).
        if (vp.w > 639 && vp.w <= 1279) {
          const aBtnCombo = page.locator("[data-vt-analyst]");
          if (await aBtnCombo.count()) {
            await aBtnCombo.click().catch(() => {});
            const aPanelCombo = await page.waitForSelector("[data-vt-analyst-panel]", { timeout: 4000 }).catch(() => null);
            if (aPanelCombo) {
              await page.waitForTimeout(300);
              const comboFails = await page.evaluate((ids) => {
                const fails = [];
                const panel = document.querySelector(".vt-layer-panel");
                if (!panel) return fails;
                for (const id of ids) {
                  const row = panel.querySelector(`[data-vt-layer="${id}"]`);
                  if (!row) continue;
                  row.scrollIntoView({ block: "center" });
                  const toggle = row.querySelector('[role="switch"]');
                  if (!toggle) continue;
                  const tr = toggle.getBoundingClientRect();
                  const hit = document.elementFromPoint(tr.left + tr.width / 2, tr.top + tr.height / 2);
                  if (hit && !toggle.contains(hit) && hit !== toggle && !hit.contains(toggle)) {
                    fails.push(`self-see: '${id}' toggle covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with the analyst pane also open`);
                  }
                }
                return fails;
              }, layerIds);
              checks.failures.push(...comboFails);
              await page.click('[data-vt-analyst-panel] [aria-label="Close analyst"]', { timeout: 2000 }).catch(() => {});
              await page.waitForTimeout(200);
            }
          }
        }
        // ── LEGEND PARITY (DESIGN.md legend rule, human-approved 2026-07-04):
        // every icon the live style draws must have a legend entry, and every
        // legend entry must name an icon registered on the map. Both
        // directions, computed from the LIVE style + source features — not
        // from the legend's own code.
        try {
          await page.click('.vt-legend-head[aria-expanded="false"]', { timeout: 800 }).catch(() => {});
          await page.waitForTimeout(200);
          const parity = await page.evaluate(() => {
            const fails = [];
            const m = window.__vtMap;
            if (!m) return ["legend-parity: __vtMap hook missing"];
            const used = new Set();
            for (const l of (m.getStyle().layers || [])) {
              if (l.type !== "symbol") continue;
              const ii = (l.layout || {})["icon-image"];
              if (!ii) continue;
              if (typeof ii === "string") { used.add(ii); continue; }
              if (Array.isArray(ii) && ii[0] === "get") {
                const prop = ii[1];
                try {
                  for (const f of m.querySourceFeatures(l.source)) {
                    const v = f && f.properties && f.properties[prop];
                    if (v) used.add(String(v));
                  }
                } catch {}
              }
            }
            const legend = new Set(
              [...document.querySelectorAll("[data-vt-icon]")].map((e) => e.getAttribute("data-vt-icon")),
            );
            for (const name of used) {
              if (!legend.has(name)) fails.push(`legend-parity: map draws '${name}' with NO legend entry — failed build per the legend rule`);
            }
            for (const name of legend) {
              if (!m.hasImage || !m.hasImage(name)) fails.push(`legend-parity: legend claims '${name}' but no such icon is registered on the map`);
            }
            for (const img of document.querySelectorAll("[data-vt-icon] img")) {
              if (!img.getAttribute("src")) fails.push("legend-parity: legend entry with empty icon render");
            }
            return fails.length ? fails : [`legend-parity-ok:${used.size} used / ${legend.size} entries`];
          });
          const ok = parity.find((p) => String(p).startsWith("legend-parity-ok:"));
          if (ok) checks.info.legendParity = ok;
          else checks.failures.push(...parity);
          // PR evidence: the legend itself, scrolled into view beside the map
          await page.evaluate(() => document.querySelector("[data-vt-legend]")?.scrollIntoView({ block: "center" }));
          await page.waitForTimeout(200);
          await page.screenshot({ path: path.join(OUT, `${name}-legend-${vp.w}.png`) });
        } catch (e) {
          checks.failures.push("legend-parity: driver error — " + (e?.message || e));
        }
        // restore collapsed-by-default state for the phone screenshot honesty
        if (vp.touch) await page.click('.vt-layer-panel [aria-label="Collapse layers panel"]').catch(() => {});
      } catch (e) {
        if (!e?.skip) checks.failures.push("self-see: driver error — " + (e?.message || e));
      }
      // VERIFIED SWITCH CLICK ([MEASUREMENT-DEBT 2026-07-20] fix): the
      // batteries below restore page state via toggle clicks that were
      // fire-and-forget (`.catch(() => {})`) — a missed click left terrain
      // enabled + the camera auto-tilted, and every later check inherited a
      // pitched viewport (the ~50% data@1440 trail flake; instrumented run
      // pinned rowChecked=true after the "off" sweep = the click never
      // landed). This clicks, VERIFIES aria-checked actually flipped, and
      // retries (re-scrolling) up to 2 more times. Assertions are untouched
      // — this fixes the driver's hand, not the ruler's scale.
      const clickSwitchVerified = async (id, wantChecked) => {
        const sw = page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first();
        for (let attempt = 0; attempt < 3; attempt++) {
          const cur = await sw.getAttribute("aria-checked").catch(() => null);
          if (cur === String(wantChecked)) return true;
          await page.evaluate((lid) => {
            document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
          }, id).catch(() => {});
          await page.waitForTimeout(80);
          await sw.click({ timeout: 4000 }).catch(() => {});
          // poll briefly for the pill to reflect the click
          for (let i = 0; i < 8; i++) {
            await page.waitForTimeout(150);
            const now = await sw.getAttribute("aria-checked").catch(() => null);
            if (now === String(wantChecked)) return true;
          }
        }
        return false;
      };
      // ── TOGGLE CONSISTENCY (state-desync repair 2026-07-04): for EVERY
      // toggleable registry layer, flipping the switch must move pill,
      // label, and actual map state TOGETHER — pill ON with a label still
      // reading "off" (the production defect) is a failed build. Runs at
      // 1440 only: it exercises state wiring, not layout.
      try {
        if (!cfg.map || vp.w !== 1440) throw { skip: true };
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(250);
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        const toggleables = FIXTURES["/api/data/layers"].layers
          .filter((l) => l.status === "live" && l.id !== "imagery") // imagery is the base — stays on
          .map((l) => l.id);
        const desyncs = [];
        for (const id of toggleables) {
          const sw = page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first();
          const before = await sw.getAttribute("aria-checked");
          await page.evaluate((lid) => {
            document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
          }, id);
          await page.waitForTimeout(100);
          await sw.click({ timeout: 4000 }).catch(() => desyncs.push(`toggle-consistency: '${id}' switch UNCLICKABLE`));
          // wait for the layer's effect to settle: label must leave "off"
          // (loading/active/error all acceptable — but never stuck at off)
          let state = null;
          for (let i = 0; i < 30; i++) {
            state = await page.evaluate((lid) => {
              const row = document.querySelector(`[data-vt-layer="${lid}"]`);
              const swEl = row?.querySelector('[role="switch"]');
              return {
                pill: swEl?.getAttribute("aria-checked"),
                rt: row?.getAttribute("data-vt-rt"),
                label: row?.querySelector(".vt-layer-status")?.textContent?.trim() || "",
              };
            }, id);
            if (state && state.rt && state.rt !== "off" && state.rt !== "none") break;
            await page.waitForTimeout(200);
          }
          if (!state || state.pill !== (before === "true" ? "false" : "true")) {
            desyncs.push(`toggle-consistency: '${id}' pill did not flip (before=${before}, after=${state?.pill})`);
          } else if (state.pill === "true" && (state.rt === "off" || state.rt === "none")) {
            desyncs.push(`toggle-consistency: '${id}' pill ON but runtime '${state.rt}' / label '${state.label}' — the production desync`);
          }
          // flip back to leave the page in its default state — VERIFIED
          // (a silent miss here poisoned every later check; see helper)
          if (!(await clickSwitchVerified(id, before === "true"))) {
            desyncs.push(`toggle-consistency: '${id}' flip-back never landed — page left in non-default state`);
          }
          await page.waitForTimeout(150);
        }
        checks.failures.push(...desyncs);
        if (!desyncs.length) checks.info.toggleConsistency = `${toggleables.length} layers toggled clean`;
        if (vp.touch) {} // (1440 only — no panel-state restore needed)
      } catch (e) {
        if (!e?.skip) checks.failures.push("toggle-consistency: driver error — " + (e?.message || e));
      }
      // ── COST-BUDGET BADGE (BUILD ORDER 4 #2, GIP Part 4 cost-budget item):
      // DEFAULT_ON's registry-native costTier sum is 13 ("light" — badge
      // silent, proven by every other capture in this file never showing
      // it). Turning on every remaining real toggleable layer (terrain +
      // both weather fields + rivergauges/alerts/surfacewater/forest/
      // boundaries — genuinely wired ids, unlike the scale battery's
      // synthetic ones) adds 22 more for a 35 total, crossing the >26
      // "heavy" threshold — proves the badge is an actually-wired consumer
      // of the registry field, not decorative dead code.
      try {
        if (!cfg.map || vp.w !== 1440) throw { skip: true };
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(200);
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(100);
        }
        const extraToggle = ["terrain", "weather", "weather_temp", "weather_wind", "rivergauges", "alerts", "surfacewater", "forest", "boundaries"];
        for (const id of extraToggle) {
          await clickSwitchVerified(id, true);
          await page.waitForTimeout(60);
        }
        await page.waitForTimeout(200);
        const costNote = await page.evaluate(() => document.querySelector(".vt-cost-note")?.textContent?.trim() || null);
        checks.info.costBudgetBadge = costNote;
        if (costNote !== "heavy load") {
          checks.failures.push(`cost-budget: expected 'heavy load' badge with ${extraToggle.length} extra layers on (weight 35), got '${costNote}' — cost-budget consumer not wired or threshold drifted`);
        }
        // ── DRAPE-ORDER RATCHET (terrain-lag root cause, probe-proven
        // 2026-07-20): with terrain ON (it is, in this battery), a custom GL
        // layer buried under a draped layer (fill/line/raster/background/
        // hillshade) splits MapLibre's RTT stack and silently defeats the
        // terrain texture cache — 588ms/frame buried vs 180ms floated in the
        // same scene (scratchpad mechanism probe). lib/drapeOrder.ts guards
        // this at runtime; this check makes a regression build-failing.
        const buried = await page.evaluate(() => {
          const DRAPED = new Set(["background", "fill", "line", "raster", "hillshade", "color-relief"]);
          const map = window.__vtMap;
          const order = map?.style?._order || [];
          const layers = map?.style?._layers || {};
          const seq = order.map((id) => ({ id, type: layers[id]?.type }));
          const out = [];
          for (let i = 0; i < seq.length; i++) {
            if (seq[i].type !== "custom") continue;
            if (seq.slice(i + 1).some((l) => DRAPED.has(l.type))) out.push(seq[i].id);
          }
          return out;
        });
        checks.info.buriedCustomLayers = buried.join(",") || "none";
        if (buried.length) {
          checks.failures.push(`drape-order: custom layer(s) buried under draped layers with terrain on: ${buried.join(", ")} — RTT stack split regresses the 2026-07-20 terrain-lag fix (lib/drapeOrder.ts)`);
        }
        // ── TRUE-ALTITUDE DATUM RATCHET (human 2026-07-21 round 16: "the
        // plane shoots way up visually in the sky i dont want that"): the
        // aircraft altScale is pinned to 1 in EVERY mesh state — the
        // exaggeration lifts the terrain only; the displayAlt hook clamps
        // planes/curtain above the exaggerated mesh instead. Any non-1
        // altScale with terrain on is the planes-launched-into-the-sky bug.
        const datum = await page.evaluate(() => ({
          exag: window.__vtMap?.getTerrain?.()?.exaggeration ?? null,
          altScale: window.__vtAir?.getAltScale?.() ?? null,
        }));
        checks.info.terrainDatum = `exag=${datum.exag} altScale=${datum.altScale}`;
        if (datum.altScale != null && Math.abs(datum.altScale - 1) > 1e-6) {
          checks.failures.push(`true-altitude datum: aircraft altScale ${datum.altScale} must stay 1 with terrain on (exaggeration ${datum.exag} lifts terrain, never aircraft — 2026-07-21 round-16 contract)`);
        }
        // flip back off to restore default state for the remaining checks/
        // screenshots — VERIFIED per switch: a missed terrain off-click left
        // the auto-tilted camera pitched for every later check (the trail
        // flake's root cause, instrumented 2026-07-20)
        for (const id of extraToggle) {
          if (!(await clickSwitchVerified(id, false))) {
            checks.failures.push(`cost-budget: '${id}' restore-to-off never landed — page left in non-default state`);
          }
        }
        await page.waitForTimeout(400);
        // one-datum, off side: terrain off must return the planes to 1×
        // (the stale-exaggeration half of the 2026-07-21 report)
        const altOff = await page.evaluate(() =>
          window.__vtMap?.getTerrain?.() ? "terrain-still-on" : (window.__vtAir?.getAltScale?.() ?? null));
        if (typeof altOff === "number" && Math.abs(altOff - 1) > 1e-6) {
          checks.failures.push(`one-datum: aircraft altScale stuck at ${altOff} after terrain off (must return to 1)`);
        }
        await page.waitForTimeout(150);
      } catch (e) {
        if (!e?.skip) checks.failures.push("cost-budget: driver error — " + (e?.message || e));
      }
      // ── U1 FIELDS-ON VISIBILITY (weather-upgrade directive 2026-07-04):
      // with temp + wind toggled ON at the DEFAULT opacity, the fields must
      // visibly render (canvas pixel diff) while the base map and
      // live-tracking layers stay visible: aircraft still rendered, rasters
      // BELOW symbols, opacity at the 60% default, arrows from the sampled
      // grid present. Pixel proof, never HTTP 200s (DESIGN.md tile rule).
      try {
        if (!cfg.map) throw { skip: true };
        // clean OFF capture: collapse the panel so both captures share chrome
        await page.click('.vt-layer-panel [aria-label="Collapse layers panel"]', { timeout: 1200 }).catch(() => {});
        await page.waitForTimeout(400);
        const canvas = page.locator("[data-vt-map] canvas").first();
        const offShot = await canvas.screenshot();
        // toggle the two field layers as a user would (panel -> group -> switch)
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(250);
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        // collapse the legend so the taller v3 block can't push the target
        // rows under the sticky panel head, then center each row before
        // clicking (edge-scrolled switches fail Playwright actionability)
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        await page.waitForTimeout(150);
        for (const id of ["weather_temp", "weather_wind"]) {
          await page.evaluate((lid) => {
            document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
          }, id);
          await page.waitForTimeout(120);
          await page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first().click({ timeout: 4000 });
          await page.waitForTimeout(150);
        }
        // temp VALUE LABELS sub-toggle ON too — the 2026-07-05 production
        // bug lived exactly in this state (labels + arrows together) and
        // the battery never exercised it: labels-off was the default.
        // (.vt-field-controls is a SIBLING of the data-vt-layer row, so the
        // locator anchors on the checkbox's own label text.)
        const valueLabels = page.locator(".vt-field-check", { hasText: "value labels" }).first();
        await valueLabels.evaluate((el) => el.scrollIntoView({ block: "center" }));
        await page.waitForTimeout(120);
        await valueLabels.click({ timeout: 4000 });
        await page.waitForTimeout(150);
        // wait until tiles are loaded AND arrow symbols are actually PLACED
        // (getLayer succeeds before the SDF icon rasterizes — querying placed
        // features is the only honest "it renders" signal)
        let mounted = false;
        for (let i = 0; i < 48; i++) {
          mounted = await page.evaluate(() => {
            const m = window.__vtMap;
            if (!(m && m.getLayer("wx-temp_new") && m.getLayer("wx-wind_new") &&
                  m.getLayer("wx-wind-arrows") && m.areTilesLoaded())) return false;
            // wait for BOTH label sets to actually place — sampling after
            // only the arrows placed raced ahead of temp-label placement at
            // 1440 (flaky false-fail seen 2026-07-05; labels were visibly
            // rendered in the screenshot taken a second later)
            try {
              return m.queryRenderedFeatures({ layers: ["wx-wind-arrows"] }).length > 0 &&
                     m.queryRenderedFeatures({ layers: ["wx-temp-labels"] }).length > 0;
            } catch { return false; }
          });
          if (mounted) break;
          await page.waitForTimeout(250);
        }
        if (!mounted) checks.failures.push("fields-on: wx layers/tiles/arrows never rendered (status probe, tile fixture, or arrow grid broken)");
        await page.waitForTimeout(400); // settle one render pass
        const fieldChecks = await page.evaluate(() => {
          const fails = [];
          const m = window.__vtMap;
          if (!m) return ["fields-on: __vtMap hook missing"];
          for (const l of ["wx-temp_new", "wx-wind_new"]) {
            const op = m.getLayer(l) ? m.getPaintProperty(l, "raster-opacity") : null;
            if (op !== 0.6) fails.push(`fields-on: ${l} raster-opacity ${op} != 0.6 default — registry default broken`);
          }
          const order = (m.getStyle().layers || []).map((l) => l.id);
          if (order.indexOf("wx-temp_new") > order.indexOf("aircraft-sym"))
            fails.push("fields-on: temp raster ABOVE aircraft symbols — live tracking obscured");
          let aircraft = 0, arrows = 0, tempLabels = 0;
          try { aircraft = m.queryRenderedFeatures({ layers: ["aircraft-sym", "aircraft-sym-lo"].filter((l) => m.getLayer(l)) }).length; } catch {}
          try { arrows = m.queryRenderedFeatures({ layers: ["wx-wind-arrows"] }).length; } catch {}
          try { tempLabels = m.queryRenderedFeatures({ layers: ["wx-temp-labels"] }).length; } catch {}
          if (!aircraft) fails.push("fields-on: no aircraft rendered with fields on — live tracking not visible");
          if (!arrows) fails.push("fields-on: no wind arrows rendered from the sampled grid");
          // REPAIR ratchet 2026-07-05: temp value-labels ON must NOT eat the
          // wind arrows (production bug: collision pass split the arrow/kt
          // pair, leaving orphaned kt text). Both label sets coexist, and
          // the by-construction guarantees are pinned: the arrows layer sits
          // fully outside the collision pass (both directions), and temp
          // labels are offset above the shared grid point.
          if (!tempLabels) fails.push("fields-on: no temp value-labels rendered with the sub-toggle on");
          if (arrows && tempLabels) {
            try {
              const lo = (p) => m.getLayoutProperty("wx-wind-arrows", p);
              if (lo("icon-ignore-placement") !== true || lo("text-ignore-placement") !== true ||
                  lo("icon-allow-overlap") !== true || lo("text-allow-overlap") !== true)
                fails.push("fields-on: wx-wind-arrows re-entered the collision pass — arrow/kt pair can be split again");
              const anchor = m.getLayoutProperty("wx-temp-labels", "text-anchor");
              if (anchor !== "bottom")
                fails.push(`fields-on: wx-temp-labels anchor '${anchor}' != 'bottom' — label no longer dodges the arrow at shared grid points`);
            } catch (e) { fails.push("fields-on: collision-pin probe failed — " + (e?.message || e)); }
          }
          // v2.4 occlusion rule re-checked WITH fields on: enabling a layer
          // grows the attribution strip — it may not spread under controls
          // (the 390px defect this caught: 2-line attribution over zoom-out).
          for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
            const el = document.querySelector(sel);
            if (!el) { fails.push(`fields-on: map control ${sel} missing`); continue; }
            const r = el.getBoundingClientRect();
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
              fails.push(`fields-on: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 40)}'> with fields on`);
            }
          }
          return fails;
        });
        checks.failures.push(...fieldChecks);
        // ON capture with identical chrome, then the pixel proof
        await page.click('.vt-layer-panel [aria-label="Collapse layers panel"]', { timeout: 1200 }).catch(() => {});
        await page.waitForTimeout(400);
        const onShot = await canvas.screenshot();
        const meanDiff = pngMeanDiff(offShot, onShot);
        checks.info.fieldsMeanDiff = Math.round(meanDiff * 10) / 10;
        if (meanDiff < 3) checks.failures.push(`fields-on: canvas mean diff ${meanDiff.toFixed(2)} < 3 — fields not visibly rendering at default opacity`);
        // PR evidence: full page with fields on (panel open on desktop,
        // collapsed on touch — same honesty convention as the main shots)
        if (!vp.touch) await page.click(".vt-map-fab", { timeout: 1200 }).catch(() => {});
        await page.waitForTimeout(300);
        await page.screenshot({ path: path.join(OUT, `${name}-fields-${vp.w}.png`) });
      } catch (e) {
        if (!e?.skip) checks.failures.push("fields-on: driver error — " + (e?.message || e));
      }
      // ── LANDING GLOBE SYMBOLS (directive 2026-07-05): the hero globe
      // renders REAL registry silhouettes, not dots — symbol layers fed by
      // the shared icon registry, heading-rotated, every icon actually
      // registered on the map (hasImage), aircraft shapes varied. Placed
      // features are the honest "it renders" signal (same lesson as
      // fields-on). Screenshot with the section in view is the PR evidence.
      try {
        if (name !== "landing") throw { skip: true };
        await page.evaluate(() => document.getElementById("data-intel")?.scrollIntoView({ block: "center" }));
        let up = false;
        for (let i = 0; i < 80; i++) {
          up = await page.evaluate(() => {
            const m = window.__vtGlobe;
            if (!(m && m.getLayer && m.getLayer("di-air-p"))) return false;
            try { return m.queryRenderedFeatures({ layers: ["di-air-p"] }).length > 0; } catch { return false; }
          });
          if (up) break;
          await page.waitForTimeout(250);
        }
        if (!up) {
          checks.failures.push("landing-globe: globe never booted or no aircraft symbols placed");
        } else {
          const g = await page.evaluate(() => {
            const fails = [];
            const m = window.__vtGlobe;
            const style = m.getStyle().layers || [];
            for (const id of ["di-air-p", "di-sites-p"]) {
              const l = style.find((x) => x.id === id);
              if (!l) { fails.push(`landing-globe: layer ${id} missing`); continue; }
              if (l.type !== "symbol") fails.push(`landing-globe: ${id} is ${l.type}, not symbol — dots regressed`);
            }
            const iconsOf = (src) => {
              try { return [...new Set(m.querySourceFeatures(src).map((f) => f.properties.icon))]; } catch { return []; }
            };
            const air = iconsOf("di-air");
            if (air.length < 2) fails.push("landing-globe: aircraft icons not varied: " + air.join(","));
            for (const src of ["di-air", "di-sites"]) {
              const missing = iconsOf(src).filter((i) => !i || !m.hasImage(i));
              if (missing.length) fails.push(`landing-globe: ${src} icons not in shared registry: ` + missing.join(","));
            }
            try {
              const rot = JSON.stringify(m.getLayoutProperty("di-air-p", "icon-rotate"));
              if (!rot.includes("heading")) fails.push("landing-globe: icon-rotate not bound to heading: " + rot);
            } catch { fails.push("landing-globe: icon-rotate unreadable"); }
            return fails;
          });
          checks.failures.push(...g);
          await page.waitForTimeout(400); // settle a render pass for the shot
          await page.screenshot({ path: path.join(OUT, `${name}-globe-${vp.w}.png`) });
        }
      } catch (e) {
        if (!e?.skip) checks.failures.push("landing-globe: driver error — " + (e?.message || e));
      }
      // Perf budget (headless regression guards; on-device budget in DESIGN.md)
      // CALIBRATED GATE ([RULE-REVIEW] 2026-07-05, perf repair 2/3):
      // performance regressions now FAIL the build like visual ones.
      // Thresholds are ~2x the worst numbers observed across recent green
      // runs under SwiftShader (data TTI 893-1306ms; medians 33-117ms by
      // width; p95 50-183ms) — regression guards, not the on-device budget
      // (DESIGN.md owns that; the S24 is the acceptance device). Direction
      // of bias: this change can only make builds FAIL more, never look
      // better (measurement-integrity note).
      if (tti == null) checks.failures.push("TTI: skeleton never cleared (>15s)");
      else if (cfg.map && tti > 3000) checks.failures.push(`TTI ${tti}ms > 3000ms map-page gate (observed ceiling ~1.3s)`);
      else if (tti > 12000) checks.failures.push(`TTI ${tti}ms > 12s headless guard`);
      if (cfg.map && perf.error) checks.failures.push("perf: " + perf.error);
      else if (cfg.map) {
        const MEDIAN_GATE = { 390: 120, 768: 200, 1440: 250 };
        const medGate = MEDIAN_GATE[vp.w] || 300;
        if (perf.median > medGate) checks.failures.push(`perf: median-of-3-passes frame ${perf.median}ms > ${medGate}ms gate @${vp.w} (steady-state jank at 10k features; per-pass: ${JSON.stringify(perf.passes)})`);
        if (perf.p95 > 350) checks.failures.push(`perf: median-of-3-passes p95 frame ${perf.p95}ms > 350ms gate (observed ceiling 183ms; per-pass: ${JSON.stringify(perf.passes)})`);
        if (perf.p95 > 250) checks.warnings.push(`perf: p95 frame ${perf.p95}ms (upload-hitch spikes)`);
        if (!perf.renderedAircraft) checks.warnings.push("perf: no aircraft features rendered in viewport sample");
        // DATA-RICHNESS GUARD (enables low-zoom decimation without data
        // loss, and forbids the cheat): the SOURCE must hold the full
        // fixture regardless of how many icons the renderer draws.
        // Deduped by icao24 — querySourceFeatures returns per-tile copies.
        if (vp.w === 1440) {
          const srcCount = await page.evaluate(() => {
            try {
              const m = window.__vtMap;
              return new Set(m.querySourceFeatures("aircraft").map((f) => f.properties.icao24)).size;
            } catch { return -1; }
          });
          checks.info.aircraftSourceCount = srcCount;
          if (srcCount >= 0 && srcCount < 9500) {
            checks.failures.push(`data-richness: aircraft source holds ${srcCount} < 9500 unique features — decimation must trim RENDERING, never DATA`);
          }

          // TRAIL LIVE-REFRESH RATCHET ([REPAIR 2026-07-05]): the trail was a
          // static snapshot — fetched once at selection, never extended while
          // the popup stayed open. Select an aircraft, require the freshness
          // chip, then hold the popup open across the 30s refresh interval and
          // require (a) a second /api/data/track pull and (b) the trail
          // geometry to GROW (the stateful fixture adds a point per call).
          const spot = await page.evaluate(() => {
            try {
              const m = window.__vtMap;
              const rect = m.getContainer().getBoundingClientRect();
              for (const f of m.querySourceFeatures("aircraft")) {
                const [lo, la] = f.geometry.coordinates;
                const px = m.project([lo, la]);
                if (px.x > 260 && px.x < rect.width - 260 && px.y > 140 && px.y < rect.height - 140) {
                  return { x: rect.left + px.x, y: rect.top + px.y };
                }
              }
            } catch {}
            return null;
          });
          if (!spot) {
            checks.failures.push("trail: no clickable aircraft feature found in viewport");
          } else {
            const callsAtSelect = trackCalls;
            await page.mouse.click(spot.x, spot.y);
            const chip = await page.waitForSelector('[data-testid="trail-freshness"]', { timeout: 8000 }).catch(() => null);
            if (!chip) {
              checks.failures.push("trail: freshness chip absent after selecting an aircraft (data-testid=trail-freshness)");
            } else {
              const chipText = await chip.textContent();
              if (!/last position/.test(chipText || "")) {
                checks.failures.push(`trail: freshness chip lacks 'last position' age text: "${chipText}"`);
              }
              const lenBefore = await page.evaluate(() => {
                try { return typeof window.__vtTrailLen === "number" ? window.__vtTrailLen : -1; } catch { return -1; }
              });
              await page.waitForTimeout(32_000); // one full refresh interval
              const lenAfter = await page.evaluate(() => {
                try { return typeof window.__vtTrailLen === "number" ? window.__vtTrailLen : -1; } catch { return -1; }
              });
              if (trackCalls <= callsAtSelect + 1) {
                checks.failures.push(`trail: no live re-pull while popup open (track calls ${callsAtSelect} -> ${trackCalls}; expected >= ${callsAtSelect + 2})`);
              }
              if (!(lenAfter > lenBefore) || lenBefore < 0) {
                checks.failures.push(`trail: geometry did not extend across the refresh interval (${lenBefore} -> ${lenAfter} points)`);
              }
              checks.info.trail = { callsAtSelect, callsAfter: trackCalls, lenBefore, lenAfter };
              await page.keyboard.press("Escape").catch(() => {});
            }
          }
        }
      }
      results.push({ page: name, width: vp.w, label: vp.label, screenshot: shot, tti, perf, ...checks });
      await ctx.close();
    }
  }
  // ── v2.4 ZERO-COST-WHEN-OFF + interactive budget (all layers off) ────────
  // With every layer toggled off, the page must (a) make NO layer-data API
  // calls (registry + auth are the only allowed /api hits) and (b) go
  // interactive fast — the regression guard for "the site got slower".
  if (!only || only === "data") {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    await page.addInitScript(() => sessionStorage.setItem("vt-layers-all-off", "1"));
    const disallowed = [];
    const ALLOWED = new Set(["/api/data/layers", "/api/auth/me", "/api/health"]);
    page.on("request", (r) => {
      try {
        const u = new URL(r.url());
        if (u.pathname.startsWith("/api/") && !ALLOWED.has(u.pathname)) disallowed.push(u.pathname);
      } catch {}
    });
    await page.route("**/*", (route) => {
      const u = route.request().url();
      if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
      return route.abort();
    });
    const t0 = Date.now();
    await page.goto(`http://127.0.0.1:${port}${PAGES.data.route}`, { waitUntil: "load", timeout: 30000 });
    let ttiOff = null;
    for (let i = 0; i < 60; i++) {
      const gone = await page.evaluate(() => !document.querySelector(".vt-map-skeleton"));
      if (gone) { ttiOff = Date.now() - t0; break; }
      await page.waitForTimeout(100);
    }
    await page.waitForTimeout(4500); // deferred-mount window: violations would fire here
    const failures = [];
    if (ttiOff == null) failures.push("all-off: skeleton never cleared");
    else if (ttiOff > 2500) failures.push(`all-off TTI ${ttiOff}ms > 2500ms budget (headless)`);
    if (disallowed.length) failures.push(`ZERO-COST-WHEN-OFF violated: layer-data calls with all layers off: ${[...new Set(disallowed)].join(", ")}`);
    results.push({ page: "data-all-off", width: 1440, label: "zero-cost", screenshot: "-", tti: ttiOff,
                   failures, warnings: [], info: { disallowed: disallowed.length } });
    await ctx.close();
  }

  // ── DEPLOY-STABLE VENDOR CHUNKS (map speed, 2026-11-19): the built
  // maplibre-gl chunk must not import the entry chunk, or every deploy
  // renames a byte-identical 1 MB library and every returning visitor
  // downloads it again on the way to the first map frame. Static check on
  // the build this run serves (scripts/chunkStability.mjs has the rule).
  if (!only || only === "data") {
    const { vendorChunkViolations } = await import("./chunkStability.mjs");
    const assetsDir = path.join(DIST, "assets");
    const chunks = readdirSync(assetsDir).filter((f) => f.endsWith(".js"))
      .map((name) => ({ name, code: readFileSync(path.join(assetsDir, name), "utf8") }));
    const failures = vendorChunkViolations({ indexHtml: readFileSync(path.join(DIST, "index.html"), "utf8"), chunks });
    results.push({ page: "data-chunks", width: 1440, label: "deploy-stable vendor chunks", screenshot: "-", tti: null,
                   failures, warnings: [], info: { chunks: chunks.length } });
  }

  // ── LAYER-SCALE SYNTHETIC HARNESS (BUILD ORDER 4 #2, GIP Part 4 UI
  // SCALABILITY): datamap.tsx now reads registry-native `group`/`costTier`
  // and caps each open panel group at GROUP_ROW_CAP (12) rows behind a
  // "show all" control, with any group not explicitly opened-by-default
  // starting collapsed. This battery generates 50/100/200-layer synthetic
  // registries (round-robin across a mix of real + brand-new group ids and
  // all three cost tiers) via a per-context route override — the shared
  // FIXTURES/server are untouched, so every other page's determinism is
  // unaffected — and MEASURES whether that architecture actually holds real
  // numbers rather than assuming it does. No silent cap: results are
  // logged for every scale, not just the pass/fail edges.
  if (!only || only === "data" || only === "scale") {
    const SCALE_GROUPS = ["base", "live", "facilities", "environmental", "filings", "signals", "synth_new_a", "synth_new_b"];
    const SCALE_TIERS = ["light", "moderate", "heavy"];
    const syntheticLayers = (n) => {
      const out = [];
      for (let i = 0; i < n; i++) {
        const g = SCALE_GROUPS[i % SCALE_GROUPS.length];
        out.push({
          id: `synth_${i}`, name: `Synthetic layer ${i}`,
          kind: g === "signals" ? "signal" : "raw",
          status: g === "signals" ? "planned" : "live",
          group: g, costTier: SCALE_TIERS[i % SCALE_TIERS.length],
          source: "layer-scale synthetic harness fixture",
          description: `Synthetic scale-test layer #${i} in group ${g}.`,
        });
      }
      return out;
    };
    const scaleResults = [];
    for (const n of [50, 100, 200]) {
      const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      await page.route("**/*", (route) => {
        const u = route.request().url();
        if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
        return route.abort();
      });
      // registered AFTER the general handler above so it wins (Playwright
      // matches most-recently-registered route first) — overrides only the
      // registry response for this context, nothing else in the app.
      await page.route("**/api/data/layers", (route) => route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ layers: syntheticLayers(n), server_version: "scale-harness" }),
      }));
      const t0 = Date.now();
      await page.goto(`http://127.0.0.1:${port}${PAGES.data.route}`, { waitUntil: "load", timeout: 30000 });
      let tti = null;
      for (let i = 0; i < 60; i++) {
        const gone = await page.evaluate(() => !document.querySelector(".vt-map-skeleton"));
        if (gone) { tti = Date.now() - t0; break; }
        await page.waitForTimeout(100);
      }
      await page.waitForTimeout(800);
      const openT0 = Date.now();
      await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
      await page.waitForTimeout(200);
      const openMs = Date.now() - openT0;
      const defaultDom = await page.evaluate(() => {
        const panel = document.querySelector(".vt-layer-panel");
        return { rows: panel ? panel.querySelectorAll("[data-vt-layer]").length : -1,
                 totalNodes: panel ? panel.querySelectorAll("*").length : -1 };
      });
      // expand every group (as the existing self-see battery does) and use
      // "show all" on each capped group — the real worst case a user causes,
      // not a hypothetical one — then re-measure.
      for (let round = 0; round < SCALE_GROUPS.length + 2; round++) {
        const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
        if (!(await btn.count())) break;
        await btn.click().catch(() => {});
        await page.waitForTimeout(80);
      }
      for (let round = 0; round < SCALE_GROUPS.length + 2; round++) {
        const more = page.locator(".vt-layer-showmore").first();
        if (!(await more.count())) break;
        await more.click().catch(() => {});
        await page.waitForTimeout(80);
      }
      const expandedAllDom = await page.evaluate(() => {
        const panel = document.querySelector(".vt-layer-panel");
        return panel ? panel.querySelectorAll("[data-vt-layer]").length : -1;
      });
      // NOTE on toggle state: these synthetic rows correctly render DISABLED
      // ("unwired" guard, datamap.tsx groupOf/unwired) — they have a
      // registry-native `group`/`costTier` but no real map-data fetch/render
      // effect exists for a `synth_N` id (that requires an actual client
      // deploy, not a registry edit alone), so the open-tab-skew guard
      // correctly refuses to render a functional-looking toggle that would
      // flip and paint nothing. This battery measures DOM/reachability at
      // scale, not the cost-badge (that's exercised below with the real
      // fixture's genuinely wired layers, where toggling isn't blocked).
      scaleResults.push({ n, tti, openMs, defaultRows: defaultDom.rows, defaultTotalNodes: defaultDom.totalNodes,
                          expandedAllRows: expandedAllDom });
      if (n === 200) await page.screenshot({ path: path.join(OUT, "data-scale-200.png") });
      await ctx.close();
    }
    console.log("\n[SCALE HARNESS] synthetic layer-count battery (BUILD ORDER 4 #2):");
    for (const r of scaleResults) {
      console.log(`  n=${r.n}: tti=${r.tti}ms panelOpen=${r.openMs}ms defaultRows=${r.defaultRows} defaultDomNodes=${r.defaultTotalNodes} expandedAllRows=${r.expandedAllRows}`);
    }
    // BUDGET ASSERTIONS: the default-open view must stay bounded regardless
    // of total registry size (proves collapse-by-default + GROUP_ROW_CAP
    // actually hold the DOM down, not just today's small registry), "show
    // all" must still reach every synthetic layer (self-see holds at scale,
    // not just for the 21-layer fixture), and TTI must not regress past the
    // existing 3000ms map-page gate used elsewhere in this file.
    const scaleFailures = [];
    for (const r of scaleResults) {
      if (r.tti == null) scaleFailures.push(`scale n=${r.n}: TTI never settled`);
      else if (r.tti > 3000) scaleFailures.push(`scale n=${r.n}: TTI ${r.tti}ms > 3000ms map-page gate at synthetic scale`);
      if (r.defaultRows > 30) scaleFailures.push(`scale n=${r.n}: default-open panel rendered ${r.defaultRows} rows > 30 budget — collapse-by-default/GROUP_ROW_CAP not holding`);
      if (r.expandedAllRows !== r.n) scaleFailures.push(`scale n=${r.n}: "show all" only reached ${r.expandedAllRows}/${r.n} layers — self-see broken at scale`);
    }
    results.push({ page: "data-scale", width: 1440, label: "layer-scale", screenshot: path.join(OUT, "data-scale-200.png"),
                   tti: null, perf: {}, failures: scaleFailures, warnings: [], info: { scaleResults } });
  }

  // ── BY-STATE GRID DROP-DOWN + "ALL STATES" CONTROL (2026-11-19, human:
  // "I want the option to have all power grids for states, but the way you
  // have all the drop-downs"). The shared fixture carries ONE generic
  // `powergrid` layer (see the R15 note on it), so the 51 per-state rows —
  // and the control that switches them together — were never rendered by
  // this harness. This battery serves the REAL registry's grid layers
  // through a per-context route override (shared FIXTURES untouched, as the
  // layer-scale battery above does) and checks, at all three widths, what
  // the human checks: the states sit in ONE drop-down built like its
  // siblings, collapsed by default; the All-states control is its first
  // row; its three positions are derived from the per-state switches; and
  // it is reachable by keyboard and by touch. Tile requests are answered
  // 404 on purpose: this battery is about the panel, and lib/gridMaster.ts
  // owns the state rule (unit-tested).
  if (!only || only === "data" || only === "gridstates") {
    const registryLayers = JSON.parse(readFileSync(path.join(ROOT, "datacore", "layers.json"), "utf8")).layers;
    const gridFixture = [
      ...FIXTURES["/api/data/layers"].layers.filter((l) => l.id === "imagery"),
      // the US master + HIFLD (Facilities), the 51 states, and Canada's
      // provinces as the sibling drop-down the by-state one must match
      ...registryLayers.filter((l) => /^powergrid/.test(l.id) && ["facilities", "grid", "grid_ca"].includes(l.group)),
    ];
    const stateIds = gridFixture.filter((l) => l.group === "grid").map((l) => l.id);
    for (const vp of WIDTHS) {
      const fails = [];
      const shot = (tag) => path.join(OUT, `data-gridstates-${vp.w}-${tag}.png`);
      const ctx = await browser.newContext({ viewport: { width: vp.w, height: vp.h }, deviceScaleFactor: 1, hasTouch: vp.touch });
      const page = await ctx.newPage();
      await page.route("**/*", (route) => {
        const u = route.request().url();
        if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
        return route.abort();
      });
      await page.route("**/api/data/layers", (route) => route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ layers: gridFixture, server_version: "gridstates-harness" }),
      }));
      await page.route("**/tiles-r2/**", (route) => route.fulfill({ status: 404, body: "" }));
      try {
        await page.goto(`http://127.0.0.1:${port}${PAGES.data.route}`, { waitUntil: "load", timeout: 30000 });
        for (let i = 0; i < 60; i++) {
          if (await page.evaluate(() => !document.querySelector(".vt-map-skeleton"))) break;
          await page.waitForTimeout(100);
        }
        await page.waitForTimeout(800);
        if (!(await page.locator(".vt-layer-panel").count())) await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(250);
        // headless renders on software GL, so the page raises its device
        // notice; close it the way a user would, or it sits on the phone shots
        await page.locator('.vt-device-notice button[aria-label="Dismiss"]').click({ timeout: 800 }).catch(() => {});
        const read = () => page.evaluate(() => {
          const groups = [...document.querySelectorAll(".vt-layer-panel .vt-layer-group")];
          const headOf = (g) => g.querySelector(":scope > .vt-layer-group-head");
          const byText = (re) => groups.find((g) => re.test(headOf(g)?.textContent || ""));
          const g = byText(/by state/i), sib = byText(/by province/i);
          if (!g) return null;
          const shape = (x) => x && headOf(x) ? [...headOf(x).children].map((c) => c.tagName + "." + String(c.className).replace(/ closed$/, "")).join(",") : null;
          const ctlRow = g.querySelector('[data-vt-control="powergrid_states_all"]');
          const sw = ctlRow?.querySelector('[role="checkbox"]');
          const r = sw?.getBoundingClientRect();
          const rowR = ctlRow?.getBoundingClientRect();
          const panel = document.querySelector(".vt-layer-panel");
          const hit = r ? document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2) : null;
          return {
            head: headOf(g).textContent.replace(/\s+/g, " ").trim(), expanded: headOf(g).getAttribute("aria-expanded"),
            headTag: headOf(g).tagName, shape: shape(g), sibShape: shape(sib), sibExpanded: sib ? headOf(sib).getAttribute("aria-expanded") : null,
            ctl: !!ctlRow, ctlFirst: !!ctlRow && g.children[1]?.contains(ctlRow), tag: sw?.tagName, checked: sw?.getAttribute("aria-checked"),
            // the control's derived position as the page states it (none | some | all)
            hook: g.querySelector("[data-vt-all-states]")?.getAttribute("data-vt-all-states") ?? null,
            label: sw?.getAttribute("aria-label"), status: ctlRow?.querySelector(".vt-layer-status")?.textContent.trim(),
            note: ctlRow?.querySelector(".vt-layer-covnote")?.textContent || "",
            rowIds: [...g.querySelectorAll("[data-vt-layer]")].map((x) => x.getAttribute("data-vt-layer")),
            rowsOn: g.querySelectorAll('[data-vt-layer] [role="switch"][aria-checked="true"]').length,
            more: !!g.querySelector(".vt-layer-showmore"),
            inView: !!r && r.left >= -1 && r.right <= innerWidth + 1 && r.top >= -1 && r.bottom <= innerHeight + 1,
            hitOk: !!hit && (hit === sw || sw.contains(hit)), rowH: rowR ? Math.round(rowR.height) : 0,
            panelBottomOk: panel.getBoundingClientRect().bottom <= innerHeight + 2,
            hscroll: document.documentElement.scrollWidth > innerWidth + 1,
            load: document.querySelector(".vt-cost-note")?.textContent.trim() || "",
            // (`powergrid_us` is the US master's SOURCE id — same shape as a state id, not a state)
            stateSources: Object.keys(window.__vtMap?.getStyle?.().sources || {}).filter((x) => /^powergrid_[a-z]{2}$/.test(x) && x !== "powergrid_us").length,
            usMasterSource: !!window.__vtMap?.getSource?.("powergrid_us"),
            masterRowInGroup: !!g.querySelector('[data-vt-layer="powergrid"], [data-vt-layer="powergrid_hifld"]'),
          };
        });
        const head = page.locator(".vt-layer-group-head", { hasText: /by state/i }).first();
        const grp = page.locator(".vt-layer-group", { has: page.locator(".vt-layer-group-head", { hasText: /by state/i }) }).first();
        const ctl = page.locator('[data-vt-control="powergrid_states_all"] [role="checkbox"]');
        const top = async () => { await head.evaluate((el) => el.scrollIntoView({ block: "start" })); await page.waitForTimeout(150); };
        let s = await read();
        if (!s) throw new Error("no 'by state' group in the panel");
        // (1) ONE drop-down, built like its siblings, collapsed by default
        if (s.expanded !== "false") fails.push(`gridstates: by-state group not collapsed by default (aria-expanded=${s.expanded})`);
        if (!new RegExp(`\\b0/${stateIds.length} on\\b`, "i").test(s.head)) fails.push(`gridstates: header count reads "${s.head}", expected 0/${stateIds.length} on`);
        if (s.headTag !== "BUTTON" || s.shape !== s.sibShape) fails.push(`gridstates: by-state header (${s.headTag} ${s.shape}) is not built like the by-province one (${s.sibShape})`);
        if (s.ctl) fails.push("gridstates: All-states control rendered while the drop-down is closed");
        await top(); await page.screenshot({ path: shot("collapsed") });
        // (2) keyboard: the header is a button — Enter opens it
        await head.focus(); await page.keyboard.press("Enter"); await page.waitForTimeout(250);
        s = await read();
        if (s.expanded !== "true") fails.push("gridstates: Enter on the focused header did not open the drop-down");
        if (!s.ctl || !s.ctlFirst) fails.push("gridstates: All-states control is not the first row of the open drop-down");
        if (s.tag !== "BUTTON" || s.hook !== "none" || s.checked !== "false" || s.status !== "off") fails.push(`gridstates: none position wrong (tag=${s.tag} state=${s.hook} aria-checked=${s.checked} status="${s.status}")`);
        if (!s.label) fails.push("gridstates: All-states control has no accessible name");
        if (!s.more || s.rowIds.length > 12) fails.push(`gridstates: GROUP_ROW_CAP not holding with the control present (${s.rowIds.length} rows, show-more=${s.more})`);
        if (s.masterRowInGroup) fails.push("gridstates: a master/HIFLD row sits inside the by-state list");
        await top(); s = await read();
        if (!s.inView || !s.hitOk) fails.push(`gridstates: All-states control not reachable (inView=${s.inView} hit=${s.hitOk}) — SELF-SEE`);
        if (vp.w <= 639 && s.rowH < 44) fails.push(`gridstates: All-states row ${s.rowH}px < 44px touch floor`);
        if (!s.panelBottomOk || s.hscroll) fails.push(`gridstates: panel overflows (bottomOk=${s.panelBottomOk} hscroll=${s.hscroll})`);
        await page.screenshot({ path: shot("expanded") });
        // (3) ONE state picked by its own row -> the control reads "some"
        await page.locator(`[data-vt-layer="${stateIds[0]}"] [role="switch"]`).click({ timeout: 4000 });
        await page.waitForTimeout(250);
        s = await read();
        if (s.hook !== "some" || s.checked !== "mixed" || s.status !== `1 of ${stateIds.length} on` || !new RegExp(`\\b1/${stateIds.length} on\\b`, "i").test(s.head))
          fails.push(`gridstates: some position wrong (state=${s.hook} aria-checked=${s.checked} status="${s.status}" head="${s.head}")`);
        // (4) keyboard on the control: Space completes to ALL
        await ctl.focus(); await page.keyboard.press("Space"); await page.waitForTimeout(600);
        await grp.locator(".vt-layer-showmore").first().click().catch(() => {});
        await page.waitForTimeout(250);
        s = await read();
        if (s.hook !== "all" || s.checked !== "true" || s.status !== `all ${stateIds.length} on` || !new RegExp(`\\b${stateIds.length}/${stateIds.length} on\\b`, "i").test(s.head))
          fails.push(`gridstates: all position wrong (state=${s.hook} aria-checked=${s.checked} status="${s.status}" head="${s.head}")`);
        if (s.rowIds.length !== stateIds.length || s.rowsOn !== stateIds.length) fails.push(`gridstates: ${s.rowsOn}/${s.rowIds.length} state rows on after All states (want ${stateIds.length})`);
        if (s.rowIds.some((id) => !/^powergrid_[a-z]{2}$/.test(id))) fails.push("gridstates: a non-state row is listed in the by-state drop-down");
        if (s.stateSources !== stateIds.length) fails.push(`gridstates: ${s.stateSources} state tile sources on the map after All states (want ${stateIds.length})`);
        if (s.usMasterSource) fails.push("gridstates: All states switched the US master on — it must turn the real state layers on, not swap in the overview");
        // COST HONESTY: all-on must say it is the heavy option, in the row and in the panel's load badge
        if (!/heavy/i.test(s.note)) fails.push(`gridstates: all-on row carries no cost note ("${s.note}")`);
        if (!/heavy/i.test(s.load)) fails.push(`gridstates: panel load badge reads "${s.load}" with ${stateIds.length} state layers on`);
        await top(); await page.screenshot({ path: shot("all") });
        // (5) press again -> NONE; nothing left on the map
        await ctl.evaluate((el) => el.scrollIntoView({ block: "center" }));
        await ctl.click({ timeout: 4000 }); await page.waitForTimeout(600);
        s = await read();
        if (s.hook !== "none" || s.checked !== "false" || s.status !== "off" || s.rowsOn !== 0 || s.stateSources !== 0)
          fails.push(`gridstates: off left state on (state=${s.hook} aria-checked=${s.checked} status="${s.status}" rowsOn=${s.rowsOn} sources=${s.stateSources})`);
      } catch (e) {
        fails.push(`gridstates: battery could not run — ${String(e?.message || e).slice(0, 160)}`);
      }
      results.push({ page: "data-gridstates", width: vp.w, label: vp.label, screenshot: shot("all"),
                     tti: null, perf: {}, failures: fails, warnings: [], info: { states: stateIds.length } });
      await ctx.close();
    }
  }

  // ── SELECTED-PLANE GLIDE battery (owner 2026-11-19: "the bluish line runs
  // in front of the plane and ... it jumps when it catches signal") ────────
  // Three fixture aircraft served by a per-context route override (the shared
  // FIXTURES are untouched). Their data is a function of the wall clock:
  //   abc001  flies a slow turn. Its row in the feed snapshot is 40-48 s old
  //           (the snapshot refreshes every 8 s); its one-hex lookup and its
  //           track have fixes up to 5 s old. Selected, the drawn plane and
  //           the head of its track must be ONE point, and the drawn plane
  //           must never jump or reverse when a newer fix or snapshot lands.
  //   abc002  stopped reporting at the start of the battery. Once its last
  //           fix is over 30 s old the card must say the drawn position is an
  //           estimate, and the plane must still be moving.
  //   abc003  stopped reporting 60 s before the battery. Past 120 s it must
  //           hold, and say so.
  // `--page planeglide` runs this battery alone.
  if (!only || only === "data" || only === "planeglide") {
    const pgFailures = [];
    const pgInfo = {};
    const shot = path.join(OUT, "data-planeglide-1440.png");
    const shotPhone = path.join(OUT, "data-planeglide-390.png");
    const T0 = Date.now();
    const V = 230; // m/s ground speed of every fixture plane
    const M_DEG = 111320;
    const SNAP_MS = 8000, ROW_AGE_MS = 40000, LIVE_AGE_MS = 5000;
    const PLANES = {
      abc001: { lon0: -97.0, lat0: 36.0, h0: 60, turn: 0.35, lastFixMs: () => Date.now() - LIVE_AGE_MS },
      abc002: { lon0: -96.2, lat0: 36.6, h0: 90, turn: 0, lastFixMs: () => T0 },
      abc003: { lon0: -97.6, lat0: 35.3, h0: 20, turn: 0, lastFixMs: () => T0 - 60000 },
    };
    // where a plane truly was at wall time `ms` (constant speed, constant turn rate)
    const truth = (hex, ms) => {
      const p = PLANES[hex];
      const t = (ms - T0) / 1000;
      const h0 = (p.h0 * Math.PI) / 180, w = (p.turn * Math.PI) / 180;
      const h = h0 + w * t;
      const e = w === 0 ? V * Math.sin(h0) * t : (-V / w) * (Math.cos(h) - Math.cos(h0));
      const n = w === 0 ? V * Math.cos(h0) * t : (V / w) * (Math.sin(h) - Math.sin(h0));
      return {
        lon: p.lon0 + e / (M_DEG * Math.cos((p.lat0 * Math.PI) / 180)), lat: p.lat0 + n / M_DEG,
        heading: ((((h * 180) / Math.PI) % 360) + 360) % 360,
      };
    };
    // the feed snapshot's row for a plane: its newest fix as of the snapshot, never newer than the plane's last report
    const rowFixMs = (hex, snapMs) => Math.min(hex === "abc001" ? snapMs - ROW_AGE_MS : Infinity, PLANES[hex].lastFixMs());
    const wideRow = (hex, fixMs, atMs) => {
      const s = truth(hex, fixMs);
      return [hex, s.lon, s.lat, 36000, Math.round(V / 0.5144), s.heading, "GLD" + hex.slice(-1), "GLF6", fixMs, "A2", 0, "N" + hex, "adsblol", atMs];
    };
    const FIELDS = ["hex", "lon", "lat", "altFt", "gsKt", "trk", "callsign", "type", "seenAt", "cat", "gnd", "reg", "src"];
    const json = (route, body) => route.fulfill({ contentType: "application/json", body: JSON.stringify(body) });
    let liveCalls = 0;
    const planHistCalls = [];
    const PH_T = Date.now() - 40 * 60_000;
    const phGeo = (changed, nm) => ({ changed, firstDifferingLeg: null, maxLateralNm: nm, lateralEstimated: false, note: null });
    const planHistoryFx = (hex) => ({
      hex, callsign: "GLD" + hex.slice(-1), kind: "raw", source: "FILED_FAA", label: "FILED — 4 versions of the FAA SWIM flight plan KSFO→KSNA",
      honesty: "Flight plan versions from FAA SWIM SFDPS, matched to this aircraft by callsign.",
      plan: { planFirstSeenMs: PH_T - 600_000, originalAsFiled: false, originalIsFirstSeen: true, originalNote: "first seen", versionsReturned: 4, versionsMissing: 0, ended: false },
      original: { version: 1, departure: "KSFO", arrival: "KSNA", cruiseAltFt: 33000 }, current: { version: 4, departure: "KSFO", arrival: "KSNA", cruiseAltFt: 29000 },
      versions: [{ version: 1 }, { version: 2 }, { version: 3 }, { version: 4 }],
      amendments: [
        { version: 2, fromVersion: 1, at: PH_T + 7 * 60_000, atBasis: "SWIM_MESSAGE_TIME", detail: "route text changed", kinds: ["ROUTE_TEXT_ONLY"], change: { kinds: ["ROUTE_TEXT_ONLY"], fixesAdded: [], fixesRemoved: [], destination: null, cruiseAltFt: null, directTo: null, geometry: phGeo(false, 0) }, spansMissingVersions: false, position: null, positionReason: "the nearest ADS-B fix is 1998 s from that time (tolerance 30 s)" },
        { version: 3, fromVersion: 2, at: PH_T + 20 * 60_000, atBasis: "SWIM_MESSAGE_TIME", detail: "cruise 33000→29000 ft", kinds: ["ALTITUDE_CHANGE"], change: { kinds: ["ALTITUDE_CHANGE"], fixesAdded: [], fixesRemoved: [], destination: null, cruiseAltFt: { from: 33000, to: 29000, deltaFt: -4000 }, directTo: null, geometry: phGeo(false, null) }, spansMissingVersions: false, position: { lat: 36.1, lon: -120.4, altFt: 33000, method: "NEAREST", interpolated: false, fixTimeMs: PH_T + 20 * 60_000 - 9000, gapSec: 9, between: null }, positionReason: null },
        { version: 4, fromVersion: 3, at: PH_T + 28 * 60_000, atBasis: "OUR_RECEIVE_TIME", detail: "route changed", kinds: ["REROUTE", "FIXES_REMOVED", "DIRECT_TO"], change: { kinds: ["REROUTE", "FIXES_REMOVED", "DIRECT_TO"], fixesAdded: [], fixesRemoved: ["ZIPRR", "WURLL"], destination: null, cruiseAltFt: null, directTo: { fix: "TCUPS" }, geometry: phGeo(true, 12.4) }, spansMissingVersions: false, position: null, positionReason: "no fix near that time" },
      ],
      restatements: [], offRoute: { assessed: true, reason: null, fixesJudged: 41, episodes: [] }, generatedAt: Date.now(),
    });
    const phZ = (ms) => new Date(ms).toISOString().slice(11, 16) + "Z";
    // The flight card's Plan changes row at the current viewport: the headline quotes the newest change; opened, one
    // line per change (newest first) with its Zulu time, plain words, the route's sideways move through the unit
    // setting (12.4 nm = 14 mi, imperial default) and the position note; the list scrolls inside the card; nothing
    // leaves the window or the card; the control is 44 px on a phone. `expect` false = the row must be absent.
    const checkPlanChanges = async (hex, expect) => {
      const fail = (m) => pgFailures.push(`plan changes ${hex} @${page.viewportSize().width}px: ${m}`);
      const row = page.locator('[data-testid="plan-changes-row"]');
      if (!expect) {
        await page.waitForTimeout(1500);
        if (await row.count()) fail("a row is shown for an aircraft with no FAA-filed plan (it must be absent, never an empty box)");
        return;
      }
      for (let i = 0; i < 50 && !(await row.count()); i++) await page.waitForTimeout(100);
      if (!(await row.count())) { fail("no Plan changes row on the flight card"); return; }
      const read = () => page.evaluate(() => {
        const txt = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
        const r = document.querySelector('[data-testid="plan-changes-row"]');
        const btn = r.querySelector("[data-vt-plan-changes-toggle]");
        const rb = r.getBoundingClientRect(), bb = btn.getBoundingClientRect();
        const list = r.querySelector("[data-vt-plan-changes-list]");
        const card = r.closest(".vt-site-card") || r.parentElement;
        const cb = card.getBoundingClientRect();
        return { headline: txt(r.querySelector("[data-vt-plan-changes-headline]")), open: r.getAttribute("data-open") === "true", btnH: Math.round(bb.height),
          right: Math.round(rb.right - window.innerWidth), outOfCard: Math.round(rb.right - cb.right),
          items: list ? [...list.querySelectorAll("[data-vt-plan-change]")].map(txt) : [], listText: txt(list), listScrolls: list ? getComputedStyle(list).overflowY : null,
          listH: list ? Math.round(list.getBoundingClientRect().height) : 0, pageOver: Math.max(0, document.documentElement.scrollWidth - window.innerWidth) };
      });
      const want3 = `direct to TCUPS · dropped ZIPRR, WURLL`;
      let r = await read();
      if (r.headline.split("No off-route")[0].trim() !== `4 plan versions · 3 changes · latest ${phZ(PH_T + 28 * 60_000)}: ${want3}`) fail(`the headline reads "${r.headline}"`);
      if (!/No off-route stretch in the 41 positions we hold for this flight\./.test(r.headline)) fail(`the closed row lacks its off-route line: "${r.headline}"`);
      if (r.right > 1 || r.outOfCard > 1) fail(`the row leaves the ${r.right > 1 ? "window" : "card"} by ${Math.max(r.right, r.outOfCard)}px`);
      const covered = await page.evaluate(() => { const b = document.querySelector("[data-vt-plan-changes-toggle]"); const q = b.getBoundingClientRect(); const top = document.elementFromPoint(q.left + q.width / 2, q.top + q.height / 2); return top && !b.contains(top) && !top.contains(b) ? (top.className || top.tagName).toString().slice(0, 60) : null; });
      if (covered) fail(`the Plan changes control is covered by another element (${covered})`);
      if (page.viewportSize().width < 640 && r.btnH < 44) fail(`the control is ${r.btnH}px tall on a phone (44)`);
      const tgl = page.locator("[data-vt-plan-changes-toggle]");
      await tgl.scrollIntoViewIfNeeded(); await tgl.click(); await page.waitForTimeout(250);
      r = await read();
      if (!r.open || r.items.length !== 3) fail(`opened, the list has ${r.items.length} changes (3 served)`);
      // each row is two grid cells (time | words): their text joins with no space between
      const want = [
        `${phZ(PH_T + 28 * 60_000)}*${want3} · route moved up to 14 mi · v3→v4 · aircraft position at that moment not known`,
        `${phZ(PH_T + 20 * 60_000)}cruise FL330 → FL290 · v2→v3 · aircraft position: nearest recorded position`,
        `${phZ(PH_T + 7 * 60_000)}route text reworded, same path · v1→v2 · aircraft position at that moment not known`,
      ];
      r.items.forEach((t, i) => { if (t !== want[i]) fail(`change ${i + 1} reads "${t}" (expected "${want[i]}")`); });
      if (!/Version 1 is the plan as we first saw it, not necessarily as filed\./.test(r.listText) || !/\* = the time we received the message/.test(r.listText) || !/derived by us\. Fetched \d+ (s|min) ago\./.test(r.listText)) fail(`the list lacks its origin, time-basis or source line: "${r.listText.slice(-260)}"`);
      if (/NaN|undefined|\bnull\b|Infinity|no data/.test(r.listText + r.headline)) fail("a bare NaN / undefined / null / 'no data' in the row");
      if (r.listScrolls !== "auto" || r.listH > 150) fail(`the open list is ${r.listH}px tall with overflow ${r.listScrolls} (it must scroll inside 148px, the card's chrome may not grow further)`);
      if (r.right > 1 || r.pageOver > 1) fail(`opened, the row leaves the window by ${r.right}px (page overflow ${r.pageOver}px)`);
      await page.locator('[data-testid="plan-changes-row"]').screenshot({ path: path.join(OUT, `data-planchanges-${page.viewportSize().width}.png`) }).catch(() => {});
      await tgl.click(); await page.waitForTimeout(150);
      if ((await read()).open) fail("the list does not close again");
    };

    const { ctx, page, flip } = await bootDemandPage(browser, port, 1);
    const pgErrors = [];
    page.on("pageerror", (e) => pgErrors.push(e.message));
    await page.route("**/api/data/track/aircraft/**", (route) => {
      const hex = new URL(route.request().url()).pathname.split("/").pop();
      if (!PLANES[hex]) return json(route, { kind: "aircraft", id: hex, points: [], count: 0 });
      const last = PLANES[hex].lastFixMs();
      const points = [];
      for (let k = 20; k >= 0; k--) {
        const ms = last - k * 30000;
        const s = truth(hex, ms);
        points.push({ t: Math.floor(ms / 1000), la: s.lat, lo: s.lon, al: 10973 });
      }
      return json(route, { kind: "aircraft", id: hex, points, count: points.length });
    });
    await page.route("**/api/data/aircraft**", (route) => {
      const u = new URL(route.request().url());
      const now = Date.now();
      if (u.pathname.startsWith("/api/data/aircraft/live/")) {
        const hex = u.pathname.split("/").pop();
        if (!PLANES[hex]) return json(route, { at: now, fields: FIELDS, rows: [], source: "snapshot" });
        liveCalls++;
        return json(route, { at: now, fields: FIELDS, rows: [wideRow(hex, PLANES[hex].lastFixMs(), now).slice(0, FIELDS.length)], source: "upstream" });
      }
      const snapMs = Math.floor(now / SNAP_MS) * SNAP_MS;
      if (u.pathname === "/api/data/aircraft/global") {
        return json(route, { at: now, count: 3, full: true, scope: "world", fields: FIELDS,
          rows: Object.keys(PLANES).map((hex) => wideRow(hex, rowFixMs(hex, snapMs), now).slice(0, FIELDS.length)),
          coverage: { opensky: { enabled: false } }, honesty: "plane-glide fixture" });
      }
      // PLAN CHANGES ([PRODUCT] 2026-11-20, components/PlanChanges.tsx): abc001 and abc003 fly a filed plan
      // with four versions and three changes (the shape of a live answer); abc002 has no FAA-filed plan
      if (u.pathname.startsWith("/api/data/aircraft/plan-history/")) {
        const hex = u.pathname.split("/").pop();
        planHistCalls.push(u.pathname + u.search);
        return json(route, hex === "abc002" ? { hex, callsign: null, kind: "raw", source: "NONE", label: "No FAA-filed plan", honesty: "No versions: only FAA-filed plans have them.", plan: null, original: null, current: null, versions: [], amendments: [], restatements: [], offRoute: { assessed: false, reason: "no filed plan", fixesJudged: 0, episodes: [] }, generatedAt: now } : planHistoryFx(hex));
      }
      if (u.pathname !== "/api/data/aircraft") return json(route, {});
      const time = Math.floor(snapMs / 1000);
      if (u.searchParams.get("since") === String(time)) return json(route, { unchanged: true, time, count: 3 });
      const aircraft = Object.keys(PLANES).map((hex) => {
        const fixMs = rowFixMs(hex, snapMs);
        const s = truth(hex, fixMs);
        return {
          icao24: hex, callsign: "GLD" + hex.slice(-1), registration: "N" + hex, lon: s.lon, lat: s.lat,
          altitude_m: 10973, on_ground: false, velocity_ms: V, heading: s.heading, type: "GLF6", category: "A2",
          seen_pos: (snapMs - fixMs) / 1000, provider: "adsblol",
        };
      });
      return json(route, { source: "plane-glide fixture", kind: "raw", time, count: aircraft.length, aircraft });
    });

    // in-page readers: where the selected plane is DRAWN (the track layer's
    // own marker when it draws the plane, else the plane's 3D instance at its
    // shader-glided position) and where the live track's head is
    await page.evaluate(() => {
      const mercToLL = (x, y) => ({ lon: x * 360 - 180, lat: (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI });
      const impl = (id) => { const l = window.__vtMap?.getLayer(id); return l ? (l.implementation || l) : null; };
      window.__pgRead = (index) => {
        const map = window.__vtMap;
        const ft = impl("flight-track-3d");
        const tail = ft && ft.getTailEnd ? ft.getTailEnd() : null;
        const mk = ft && ft.getMarker ? ft.getMarker() : null;
        const air = window.__vtAir;
        const inst = air && air.getInstances ? air.getInstances() : null;
        const stride = 9;
        const count3d = inst ? Math.floor(inst.length / stride) : 0;
        let plane = null, by = null;
        if (mk) { plane = mercToLL(mk.mercX, mk.mercY); by = "marker"; }
        else if (inst && index != null && index < count3d && map.getZoom() >= 8.05) {
          const dt = air.getGlideDtSec();
          const b = index * stride;
          plane = mercToLL(inst[b] + inst[b + 6] * dt, inst[b + 1] + inst[b + 7] * dt); by = "instance";
        }
        const head = tail ? mercToLL(tail.mercX, tail.mercY) : null;
        let gapPx = null, gapM = null;
        if (plane && head) {
          const a = map.project([plane.lon, plane.lat]), b = map.project([head.lon, head.lat]);
          gapPx = Math.hypot(a.x - b.x, a.y - b.y);
          const cos = Math.cos((plane.lat * Math.PI) / 180);
          gapM = Math.hypot((plane.lon - head.lon) * 111320 * cos, (plane.lat - head.lat) * 111320);
        }
        const note = document.querySelector('[data-testid="plane-estimate"]');
        return { t: performance.now(), zoom: map.getZoom(), plane, by, head, gapPx, gapM, count3d,
                 dim: mk ? !!mk.estimated : null, note: note ? note.textContent : null };
      };
      // sample every animation frame for ms: [t, lon, lat, gapPx]
      window.__pgSample = (index, ms) => new Promise((resolve) => {
        const out = [];
        const t0 = performance.now();
        const tick = () => {
          const r = window.__pgRead(index);
          if (r.plane) out.push([r.t, r.plane.lon, r.plane.lat, r.gapPx]);
          if (performance.now() - t0 < ms) requestAnimationFrame(tick); else resolve(out);
        };
        requestAnimationFrame(tick);
      });
    });
    const analyse = (samples, headingDeg) => {
      const hr = (headingDeg * Math.PI) / 180;
      let maxSpeed = 0, maxBack = 0, maxGapPx = 0, moved = 0;
      const metres = (a, b) => [(b[1] - a[1]) * M_DEG * Math.cos((b[2] * Math.PI) / 180), (b[2] - a[2]) * M_DEG];
      for (let i = 1; i < samples.length; i++) {
        const [e, n] = metres(samples[i - 1], samples[i]);
        moved += Math.hypot(e, n);
        maxBack = Math.max(maxBack, -(e * Math.sin(hr) + n * Math.cos(hr)));
        if (samples[i][3] != null) maxGapPx = Math.max(maxGapPx, samples[i][3]);
      }
      // speed over every window of at least 0.6 s: away from close zoom the
      // plane is advanced in 300 ms steps (each under a pixel there), so a
      // frame-to-frame speed would call every step a jump. A real jump (a
      // plane relocated when a new position lands) is kilometres inside one
      // window and reads as several times the ground speed.
      for (let i = 0, j = 0; i < samples.length; i++) {
        while (j < samples.length && samples[j][0] - samples[i][0] < 600) j++;
        if (j >= samples.length) break;
        const [e, n] = metres(samples[i], samples[j]);
        maxSpeed = Math.max(maxSpeed, Math.hypot(e, n) / ((samples[j][0] - samples[i][0]) / 1000));
      }
      return { frames: samples.length, maxSpeed: Math.round(maxSpeed), maxBackM: Math.round(maxBack * 10) / 10,
               maxGapPx: Math.round(maxGapPx * 100) / 100, movedM: Math.round(moved) };
    };
    const selectPlane = async (hex) => {
      await page.keyboard.press("Escape").catch(() => {});
      const s = truth(hex, PLANES[hex].lastFixMs());
      await page.evaluate(([lon, lat]) => window.__vtMap.jumpTo({ center: [lon, lat], zoom: 7, pitch: 0, bearing: 0 }), [s.lon, s.lat]);
      let spot = null;
      for (let i = 0; i < 40 && !spot; i++) {
        await page.waitForTimeout(500);
        spot = await page.evaluate((h) => {
          try {
            const m = window.__vtMap;
            const f = m.querySourceFeatures("aircraft").find((x) => x.properties.icao24 === h);
            if (!f || m.isMoving()) return null;
            const px = m.project(f.geometry.coordinates);
            const rect = m.getContainer().getBoundingClientRect();
            return { x: rect.left + px.x, y: rect.top + px.y };
          } catch { return null; }
        }, hex);
      }
      if (!spot) { pgFailures.push(`planeglide: fixture plane ${hex} never rendered as a clickable symbol`); return false; }
      await page.mouse.click(spot.x, spot.y);
      const chip = await page.waitForSelector('[data-testid="trail-freshness"]', { timeout: 10000 }).catch(() => null);
      if (!chip) { pgFailures.push(`planeglide: selecting ${hex} opened no flight card`); return false; }
      // the click eases the camera in (zoom >= 9.2, pitch 55) and locks the follow
      for (let i = 0; i < 30; i++) {
        await page.waitForTimeout(300);
        if (await page.evaluate(() => !window.__vtMap.isMoving() && window.__vtMap.getZoom() >= 9)) break;
      }
      return true;
    };

    try {
      if (!(await flip("aircraft", true))) pgFailures.push("planeglide: the aircraft layer toggle never landed");
      await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {}); // close the layers panel

      // ── abc001: fresh per-plane fixes over a 40 s old feed row ──
      if (await selectPlane("abc001")) {
        await checkPlanChanges("abc001", true);
        if (!planHistCalls.includes("/api/data/aircraft/plan-history/abc001?callsign=GLD1") && !planHistCalls.includes("/api/data/aircraft/plan-history/abc001")) pgFailures.push(`plan changes: the request was ${JSON.stringify(planHistCalls)}`);
        // the first one-hex fix and the track are in, and the plane has closed
        // on its reckoned position from where its 40 s old feed icon was drawn
        await page.waitForTimeout(8000);
        const s3d = await page.evaluate(() => window.__pgSample(0, 12000)); // spans >= 4 one-hex fixes and a snapshot refresh
        const a3d = analyse(s3d, truth("abc001", Date.now()).heading);
        const r3d = await page.evaluate(() => window.__pgRead(0));
        pgInfo.abc001_3d = { ...a3d, zoom: Math.round(r3d.zoom * 100) / 100, drawnBy: r3d.by, gapM: r3d.gapM == null ? null : Math.round(r3d.gapM), count3d: r3d.count3d, liveCalls };
        if (!s3d.length) pgFailures.push("planeglide 3D: the selected plane is not drawn at all");
        if (a3d.maxGapPx > 1) pgFailures.push(`planeglide 3D: the track's head is ${a3d.maxGapPx}px from the drawn plane (must be the same point; ${pgInfo.abc001_3d.gapM} m at the last sample)`);
        if (a3d.maxSpeed > 1.9 * V) pgFailures.push(`planeglide 3D: the drawn plane covered ground at ${a3d.maxSpeed} m/s (ground speed ${V} m/s; a jump when a new position arrived)`);
        if (a3d.maxBackM > 1) pgFailures.push(`planeglide 3D: the drawn plane moved ${a3d.maxBackM} m backward along its track`);
        if (a3d.movedM < 0.5 * V * 12) pgFailures.push(`planeglide 3D: the drawn plane covered only ${a3d.movedM} m in 12 s at ${V} m/s (it must keep moving between fixes)`);
        if (r3d.by !== "marker") pgFailures.push("planeglide 3D: the selected plane is drawn by the feed's instance buffer, not at the head of its own track");
        if (r3d.count3d !== 2) pgFailures.push(`planeglide 3D: ${r3d.count3d} feed instances drawn with one plane selected (expected 2: the selected plane must not be drawn twice)`);
        if (r3d.note) pgFailures.push(`planeglide 3D: a plane with a 5 s old fix is labelled "${r3d.note}"`);
        await page.screenshot({ path: shot });
        // the same plane in the 2D band (below the 3D hand-off zoom)
        await page.evaluate(() => window.__vtMap.jumpTo({ zoom: 7.2, pitch: 0 }));
        await page.waitForTimeout(1500);
        const s2d = await page.evaluate(() => window.__pgSample(0, 6000));
        const a2d = analyse(s2d, truth("abc001", Date.now()).heading);
        const icons = await page.evaluate(() => {
          const m = window.__vtMap;
          return m.queryRenderedFeatures({ layers: ["aircraft-sym", "aircraft-sym-lo"].filter((l) => m.getLayer(l)) })
            .filter((f) => f.properties.icao24 === "abc001").length;
        });
        pgInfo.abc001_2d = { ...a2d, icons };
        if (!s2d.length) pgFailures.push("planeglide 2D: the selected plane is not drawn at all below the 3D zoom");
        if (a2d.maxGapPx > 1) pgFailures.push(`planeglide 2D: the track's head is ${a2d.maxGapPx}px from the drawn plane`);
        if (a2d.maxSpeed > 1.9 * V) pgFailures.push(`planeglide 2D: the drawn plane covered ground at ${a2d.maxSpeed} m/s (ground speed ${V} m/s)`);
        if (a2d.maxBackM > 1) pgFailures.push(`planeglide 2D: the drawn plane moved ${a2d.maxBackM} m backward along its track`);
        if (icons !== 0) pgFailures.push(`planeglide 2D: the selected plane also has ${icons} 2D symbol(s) drawn (one plane, one representation)`);
      }

      // ── abc002: no new fix; past 30 s the drawn position is an ESTIMATE ──
      if (await selectPlane("abc002")) {
        await checkPlanChanges("abc002", false);
        const age = () => (Date.now() - PLANES.abc002.lastFixMs()) / 1000;
        await page.waitForTimeout(Math.max(7000, (33 - age()) * 1000)); // past 30 s, and past the catch-up from its feed icon
        const r0 = await page.evaluate(() => window.__pgRead(1));
        await page.waitForTimeout(2000);
        const r1 = await page.evaluate(() => window.__pgRead(1));
        const movedM = r0.plane && r1.plane
          ? Math.hypot((r1.plane.lon - r0.plane.lon) * M_DEG * Math.cos((r1.plane.lat * Math.PI) / 180), (r1.plane.lat - r0.plane.lat) * M_DEG) : 0;
        pgInfo.abc002 = { ageSec: Math.round(age()), movedM: Math.round(movedM), note: r1.note, dim: r1.dim, gapPx: r1.gapPx };
        if (movedM < 0.5 * V * 2 || movedM > 1.9 * V * 2) pgFailures.push(`planeglide estimate: ${Math.round(age())} s after its last fix the drawn plane moved ${Math.round(movedM)} m in 2 s at ${V} m/s (speed and track are known: it must keep moving, at its own speed)`);
        if (!/estimate/i.test(r1.note || "")) pgFailures.push(`planeglide estimate: ${Math.round(age())} s after the last real fix the card does not say the drawn position is an estimate (text: ${JSON.stringify(r1.note)})`);
        if (r1.dim !== true) pgFailures.push("planeglide estimate: the estimated plane is not drawn faded");
        if (r1.gapPx == null || r1.gapPx > 1) pgFailures.push(`planeglide estimate: track head ${r1.gapPx}px from the drawn plane`);
        await page.screenshot({ path: path.join(OUT, "data-planeglide-estimated-1440.png") });
      }

      // ── abc003: past 120 s with no fix the estimate HOLDS, and says so ──
      if (await selectPlane("abc003")) {
        const age = () => (Date.now() - PLANES.abc003.lastFixMs()) / 1000;
        await page.waitForTimeout(Math.max(9000, (124 - age()) * 1000)); // past 120 s, and past the catch-up from its feed icon
        const r0 = await page.evaluate(() => window.__pgRead(2));
        await page.waitForTimeout(2000);
        const r1 = await page.evaluate(() => window.__pgRead(2));
        const movedM = r0.plane && r1.plane
          ? Math.hypot((r1.plane.lon - r0.plane.lon) * M_DEG * Math.cos((r1.plane.lat * Math.PI) / 180), (r1.plane.lat - r0.plane.lat) * M_DEG) : -1;
        // straight ahead for no more than 120 s from the last real fix
        const fix = truth("abc003", PLANES.abc003.lastFixMs());
        const reachM = r1.plane
          ? Math.hypot((r1.plane.lon - fix.lon) * M_DEG * Math.cos((fix.lat * Math.PI) / 180), (r1.plane.lat - fix.lat) * M_DEG) : -1;
        pgInfo.abc003 = { ageSec: Math.round(age()), movedM: Math.round(movedM), reachM: Math.round(reachM), note: r1.note };
        if (movedM < 0 || movedM > 5) pgFailures.push(`planeglide hold: ${Math.round(age())} s after its last fix the drawn plane still moved ${Math.round(movedM)} m in 2 s (it must hold past 120 s)`);
        if (reachM > V * 121) pgFailures.push(`planeglide hold: the plane is drawn ${Math.round(reachM)} m past its last real fix (limit ${V * 120} m = 120 s straight ahead)`);
        if (!/held|hold/i.test(r1.note || "")) pgFailures.push(`planeglide hold: the card does not say the position is held (text: ${JSON.stringify(r1.note)})`);
        // phone width: the same plane must still sit at the head of its track
        // (card minimized and layers panel closed so the map is what is shot)
        await page.setViewportSize({ width: 768, height: 1024 });
        await page.waitForTimeout(1200);
        // this session was laid out at 1440 with the legend open; at tablet and phone width the open legend floats
        // over the flight card, so it is folded before the card is read (a known overlap, filed in open_questions)
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(300);
        await checkPlanChanges("abc003", true);
        await page.setViewportSize({ width: 390, height: 844 });
        await page.waitForTimeout(1200);
        await page.click('[aria-label="Collapse layers panel"]', { timeout: 1500 }).catch(() => {});
        await checkPlanChanges("abc003", true);
        await page.click('[aria-label="Minimize details"]', { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(1200);
        const rPhone = await page.evaluate(() => window.__pgRead(2));
        pgInfo.phone = { gapPx: rPhone.gapPx, drawnBy: rPhone.by };
        if (rPhone.gapPx == null || rPhone.gapPx > 1) pgFailures.push(`planeglide 390px: track head ${rPhone.gapPx}px from the drawn plane`);
        await page.screenshot({ path: shotPhone });
      }
    } catch (e) {
      pgFailures.push("planeglide: battery crashed: " + String(e?.message || e).split("\n")[0]);
    }
    for (const e of pgErrors) pgFailures.push("planeglide: pageerror: " + e);
    console.log("\n[PLANE GLIDE] " + JSON.stringify(pgInfo));
    await ctx.close();
    results.push({ page: "data-planeglide", width: 1440, label: "plane-glide", screenshot: shot,
                   tti: null, perf: {}, failures: pgFailures, warnings: [], info: { planeGlide: pgInfo } });
  }

  // ── AIRPORT TRAFFIC battery ([PRODUCT] 2026-11-20; components/AirportTraffic.tsx over GET
  // /api/data/swim/tfms/active) ────────────────────────────────────────────
  // The FAA airport card's Traffic row, driven through the real map: the FAA airport layer is switched on, an
  // airport's symbol is clicked, and the row must say what the FAA's traffic-management feed lists for THAT airport:
  // the delay program with its reason and end time, the count of restrictions (a re-sent restriction counted once,
  // another airport's and the feed's "ALL" rows not counted), and the runways and rates the tower reported. Opened:
  // one line per program and restriction, the advisory's own title, the caveat that a restriction cancelled early
  // can still be listed, and the source line. An airport the feed says nothing about gets a sentence and a control
  // that does nothing. One request per selection; nothing leaves the card or the window at 1440, 768 and 390.
  // `--page airporttraffic` runs this battery alone.
  if (!only || only === "data" || only === "airporttraffic") {
    const atFailures = [];
    const atInfo = {};
    const json = (route, body) => route.fulfill({ contentType: "application/json", body: JSON.stringify(body) });
    const { ctx, page, flip } = await bootDemandPage(browser, port, 1);
    const atErrors = [];
    page.on("pageerror", (e) => atErrors.push(e.message));
    const AT_NOW = Date.now();
    const H = 3600_000;
    const z = (ms) => { const d = new Date(ms); return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}Z`; };
    const sameDay = (ms) => new Date(ms).getUTCDate() === new Date(AT_NOW).getUTCDate();
    const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const zd = (ms) => (sameDay(ms) ? z(ms) : `${MON[new Date(ms).getUTCMonth()]} ${new Date(ms).getUTCDate()} ${z(ms)}`);
    const GDP_END = AT_NOW + 2 * H, R1_END = AT_NOW + 1 * H, R2_END = AT_NOW + 3 * H, CFG_AT = AT_NOW - 40 * 60_000;
    const KEPT_SAVED = AT_NOW - 41 * 60_000, KEPT_READ = AT_NOW - 40 * 60_000;
    const prog = (o) => ({ status: null, airport: null, facility: null, start: AT_NOW - H, end: null, title: null, text: null, reason: null, elements: [], geometry: null, floorFl: null, ceilingFl: null, updatedAt: AT_NOW - 30 * 60_000, ...o });
    const tfmsCalls = [];
    await page.route("**/api/data/airport-status", (route) => json(route, { kind: "raw", source: "FAA National Airspace System Status (fixture)", time: 1, update_time: new Date(AT_NOW).toISOString(), count: 2, events: [
      { airport: "ATL", type: "Ground Delay Program", reason: "Volume", avg: "31 minutes", max: "1 hour 5 minutes", min: "16 minutes", trend: "Increasing", direction: "Arrival", end_time: "18:00 UTC", reopen: null, update_time: new Date(AT_NOW).toISOString() },
      { airport: "BOS", type: "Departure Delay", reason: "Wind", avg: "20 minutes", max: "30 minutes", min: "15 minutes", trend: "Decreasing", direction: "Departure", end_time: null, reopen: null, update_time: new Date(AT_NOW).toISOString() },
    ] }));
    await page.route("**/api/data/swim/tfms/active", (route) => {
      tfmsCalls.push(Date.now());
      // `kept` (2026-11-21): the server saves the list each minute and read 351 rows back at its last start, 40 minutes ago
      return json(route, { at: AT_NOW, source: "FAA SWIM TFMS (SCDS, NDRB pre-approved for public release; not for operational use)", stats: { programs: 9 },
        kept: { saved_every_s: 60, last_saved_at: AT_NOW - 20_000, last_save: "ok", restored: { at: KEPT_READ, saved_at: KEPT_SAVED, cause: "ok", programs: 351, airport_configs: 43 } },
        programs: [
          prog({ id: "ADV:ATCSCC:0059", kind: "ADVISORY", status: "GDP", airport: "ATL", facility: "DCCO", end: GDP_END, title: "ATCSCC ADVZY 059 ATL/ZTL CDM GROUND DELAY PROGRAM" }),
          prog({ id: "GDP:ATL", kind: "GDP", status: "ACTUAL", airport: "ATL", facility: "ZTL", end: GDP_END, title: "gdpAdvisory", reason: "VOLUME" }),
          prog({ id: "RSTR:1", kind: "RESTRICTION", status: "4", airport: "ATL", facility: "ZJX", end: R1_END, title: "MIT ARRIVAL", reason: "WX:Thunderstorms", elements: ["OSPRY", "RDSOX"] }),
          prog({ id: "RSTR:2", kind: "RESTRICTION", status: "5", airport: "ATL", facility: "ZJX", end: R1_END, title: "MIT ARRIVAL", reason: "WX:Thunderstorms", elements: ["OSPRY", "RDSOX"], updatedAt: AT_NOW - 10 * 60_000 }),
          prog({ id: "RSTR:3", kind: "RESTRICTION", status: "4", airport: "ATL/PDK", facility: "A80", end: R2_END, title: "APREQ DEPARTURE", reason: "NONE", elements: ["ATL"] }),
          prog({ id: "RSTR:4", kind: "RESTRICTION", status: "4", airport: "ALL", facility: "ZTL", end: R2_END, title: "MIT DEPARTURE", reason: "VOL:Volume", elements: ["SMKEY"] }),
          prog({ id: "RSTR:5", kind: "RESTRICTION", status: "4", airport: "CLT", facility: "ZTL", end: R2_END, title: "STOP ARRIVAL", reason: "WEATHER", elements: [] }),
          prog({ id: "FCA:1", kind: "FCA", status: "UPDATED", facility: "DCC", title: "FCALEW", geometry: [[45.3, -82.7], [46.1, -81.2]] }),
          prog({ id: "ADV:ATCSCC:0055", kind: "ADVISORY", facility: "DCCO", title: "ATCSCC ADVZY 055 DCC US/MEXICO OUTLOOK" }),
        ],
        airportConfigs: [{ airport: "ATL", arrRunways: "08L/09R/10", depRunways: "08R/09L", aar: 132, adr: 0, weather: "VMC", facility: "A80", at: CFG_AT }] });
    });
    const selectAirport = async (code, lon, lat) => {
      await page.keyboard.press("Escape").catch(() => {});
      await page.evaluate(([lo, la]) => window.__vtMap.jumpTo({ center: [lo, la], zoom: 6, pitch: 0, bearing: 0 }), [lon, lat]);
      let spot = null;
      for (let i = 0; i < 40 && !spot; i++) {
        await page.waitForTimeout(400);
        spot = await page.evaluate((c) => {
          try {
            const m = window.__vtMap;
            const f = m.querySourceFeatures("faa-airports").find((x) => x.properties.airport === c);
            if (!f || m.isMoving()) return null;
            const px = m.project(f.geometry.coordinates);
            const rect = m.getContainer().getBoundingClientRect();
            return { x: rect.left + px.x, y: rect.top + px.y };
          } catch { return null; }
        }, code);
      }
      if (!spot) { atFailures.push(`airport traffic: fixture airport ${code} never rendered as a clickable symbol`); return false; }
      await page.mouse.click(spot.x, spot.y);
      const row = await page.waitForSelector(`[data-testid="airport-traffic-row"][data-airport="${code}"]`, { timeout: 10000 }).catch(() => null);
      if (!row) { atFailures.push(`airport traffic: selecting ${code} showed no Traffic row on its card`); return false; }
      return true;
    };
    const readRow = () => page.evaluate(() => {
      const txt = (el) => (el?.textContent || "").replace(/\s+/g, " ").trim();
      const r = document.querySelector('[data-testid="airport-traffic-row"]');
      if (!r) return null;
      const btn = r.querySelector("[data-vt-airport-traffic-toggle]");
      const rb = r.getBoundingClientRect(), bb = btn.getBoundingClientRect();
      const card = r.closest(".vt-site-card") || r.parentElement;
      const cb = card.getBoundingClientRect();
      const list = r.querySelector("[data-vt-airport-traffic-list]");
      const top = document.elementFromPoint(bb.left + bb.width / 2, bb.top + bb.height / 2);
      const lineEls = [...r.querySelectorAll("[data-vt-airport-traffic-headline] > div")];
      const lines = lineEls.map(txt);
      const headCut = lineEls[0] ? lineEls[0].scrollWidth > lineEls[0].clientWidth + 1 : false;
      const chips = [...card.querySelectorAll(".vt-card-stat")].map((c) => { const l = c.querySelector(".vt-card-stat-l"), v = c.querySelector(".vt-card-stat-v");
        return { l: txt(l), v: txt(v), cut: (v.scrollWidth > v.clientWidth + 1) || (l.scrollWidth > l.clientWidth + 1) }; });
      return { lines, headCut, chips, open: r.getAttribute("data-open") === "true", disabled: btn.disabled, btnH: Math.round(bb.height), covered: !(top && (top === btn || btn.contains(top))),
        right: Math.round(rb.right - window.innerWidth), outOfCard: Math.round(rb.right - cb.right), cardTitle: txt(card.querySelector("h3, .vt-site-card-title, [data-testid='detail-title']")),
        programs: list ? [...list.querySelectorAll("[data-vt-traffic-program]")].map(txt) : [], restrictions: list ? [...list.querySelectorAll("[data-vt-traffic-restriction]")].map(txt) : [],
        config: txt(list?.querySelector("[data-vt-traffic-config]")), source: txt(list?.querySelector("[data-vt-airport-traffic-source]")), listText: txt(list),
        listScrolls: list ? getComputedStyle(list).overflowY : null, listH: list ? Math.round(list.getBoundingClientRect().height) : 0,
        pageOver: Math.max(0, document.documentElement.scrollWidth - window.innerWidth) };
    });
    const HEAD = `Ground delay program · volume · until ${zd(GDP_END)} · 2 restrictions`;
    const CONFIG = `Arriving 08L/09R/10 · departing 08R/09L · 132 arrivals an hour · visual conditions (VMC) · reported ${zd(CFG_AT)}`;
    const checkAtl = async () => {
      const w = page.viewportSize().width;
      const fail = (m) => atFailures.push(`airport traffic ATL @${w}px: ${m}`);
      let r = await readRow();
      if (!r) { fail("no Traffic row"); return; }
      if (r.lines[0] !== HEAD) fail(`the first line reads "${r.lines[0]}" (expected "${HEAD}")`);
      if (r.lines[1] !== CONFIG) fail(`the runway line reads "${r.lines[1]}" (expected "${CONFIG}")`);
      // first run, 1440px: the first line was cut to "Ground delay program · volume · u…": the end time was not readable
      if (r.headCut) fail("the first line is cut: the program's end time must be readable without opening the list");
      if (r.disabled) fail("the control is disabled although there is a list to open");
      // THE CARD'S CHIPS ([PRODUCT] 2026-11-20): seen in this battery's own screenshot, all four were cut ("Ground…",
      // "31 min…", "1 hour…", "Increa…"). Now three, the delays shortened, none cut at any width.
      const wantChips = [["AVG DELAY", "31 min"], ["MAX DELAY", "1 h 5 min"], ["TREND", "Increasing"]];
      if (JSON.stringify(r.chips.map((c) => [c.l.toUpperCase(), c.v])) !== JSON.stringify(wantChips)) fail(`the card's chips read ${JSON.stringify(r.chips.map((c) => [c.l, c.v]))}`);
      if (r.chips.some((c) => c.cut)) fail(`a chip is cut with an ellipsis: ${JSON.stringify(r.chips.filter((c) => c.cut).map((c) => c.l + " " + c.v))}`);
      if (r.covered) fail("another element covers the Traffic control");
      if (r.right > 1 || r.outOfCard > 1) fail(`the row leaves the ${r.right > 1 ? "window" : "card"} by ${Math.max(r.right, r.outOfCard)}px`);
      if (w < 640 && r.btnH < 44) fail(`the control is ${r.btnH}px tall on a phone (44)`);
      const tgl = page.locator("[data-vt-airport-traffic-toggle]");
      await tgl.scrollIntoViewIfNeeded(); await tgl.click(); await page.waitForTimeout(250);
      r = await readRow();
      if (!r.open) { fail("the list does not open"); return; }
      const wantProg = `Ground delay program · volume · until ${zd(GDP_END)} · ATCSCC ADVZY 059 ATL/ZTL CDM GROUND DELAY PROGRAM`;
      if (r.programs.length !== 1 || r.programs[0] !== wantProg) fail(`the program line reads ${JSON.stringify(r.programs)} (expected "${wantProg}")`);
      const wantR = [
        `Call for release, departures · via ATL · set by A80 · until ${zd(R2_END)}`,
        `Miles-in-trail, arrivals · via OSPRY, RDSOX · set by ZJX · weather: thunderstorms · until ${zd(R1_END)}`,
      ];
      // 5 restriction rows served: one is a re-send of another, one names every airport ("ALL"), one names CLT
      if (JSON.stringify(r.restrictions) !== JSON.stringify(wantR)) fail(`the restrictions read ${JSON.stringify(r.restrictions)} (expected ${JSON.stringify(wantR)})`);
      if (r.config !== `Runways and rates · ${CONFIG}`) fail(`opened, the runway line reads "${r.config}"`);
      // the server in this fixture keeps its list across a restart: the line says what was carried over and what a restart can still lose
      const srcHead = `FAA traffic-management feed (SWIM TFMS), as received: 9 programs and restrictions searched for ATL. A restriction is listed until its published end; one cancelled early can still be listed. The list is kept across a server restart: 351 programs and restrictions carried over the last one (saved ${zd(KEPT_SAVED)}, read back ${zd(KEPT_READ)}). What the FAA sent while the server was restarting is missing until it is sent again, and a program cancelled then stays listed until its end. Times are Zulu. Not for operational use. Read `;
      if (!r.source.startsWith(srcHead) || !/^\d+ (s|min) ago\.$/.test(r.source.slice(srcHead.length))) fail(`the source line reads "${r.source}"`);
      if (/starts again when the server restarts|restarts with the server/.test(r.source + r.lines.join(" "))) fail("the row still says the list restarts with the server, on a server that keeps it");
      if (/NaN|undefined|\bnull\b|Infinity/.test(r.listText + r.lines.join(" "))) fail("a bare NaN / undefined / null in the row");
      if (r.listScrolls !== "auto" || r.listH > 150) fail(`the open list is ${r.listH}px tall with overflow ${r.listScrolls} (it must scroll inside 148px)`);
      if (r.right > 1 || r.pageOver > 1) fail(`opened, the row leaves the window by ${r.right}px (page overflow ${r.pageOver}px)`);
      await page.locator('[data-testid="airport-traffic-row"]').screenshot({ path: path.join(OUT, `data-airporttraffic-${w}.png`) }).catch(() => {});
      await tgl.click(); await page.waitForTimeout(150);
      if ((await readRow()).open) fail("the list does not close again");
    };
    try {
      if (!(await flip("faa_airports", true))) atFailures.push("airport traffic: the FAA airport layer toggle never landed");
      if (await selectAirport("ATL", -84.4277, 33.6407)) {
        await checkAtl();
        atInfo.callsAfterAtl = tfmsCalls.length;
        if (tfmsCalls.length !== 1) atFailures.push(`airport traffic: selecting one airport made ${tfmsCalls.length} requests for the feed (exactly 1)`);
        await page.screenshot({ path: path.join(OUT, "data-airporttraffic-card-1440.png") });
        await page.setViewportSize({ width: 768, height: 1024 });
        await page.waitForTimeout(1200);
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        await checkAtl();
        await page.setViewportSize({ width: 390, height: 844 });
        await page.waitForTimeout(1200);
        await page.click('[aria-label="Collapse layers panel"]', { timeout: 1500 }).catch(() => {});
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        await checkAtl();
        await page.setViewportSize({ width: 1440, height: 900 });
        await page.waitForTimeout(1200);
      }
      // an airport the traffic feed says nothing about: a sentence, and a control that does nothing
      const before = tfmsCalls.length;
      if (await selectAirport("BOS", -71.0096, 42.3656)) {
        const r = await readRow();
        const want = "Nothing for BOS among the 9 programs and restrictions held from the FAA feed";
        if (!r || r.lines.length !== 1 || r.lines[0] !== want) atFailures.push(`airport traffic BOS: the row reads ${JSON.stringify(r?.lines)} (expected one line: "${want}")`);
        if (r && !r.disabled) atFailures.push("airport traffic BOS: the control is enabled with nothing to list");
        if (tfmsCalls.length - before !== 1) atFailures.push(`airport traffic BOS: ${tfmsCalls.length - before} requests on selection (exactly 1)`);
      }
      // closing the card removes the row and stops asking (the refresh timer is cleared with the selection)
      await page.keyboard.press("Escape").catch(() => {});
      await page.click('[aria-label="Close details"]', { timeout: 1500 }).catch(() => {});
      await page.waitForTimeout(600);
      if (await page.locator('[data-testid="airport-traffic-row"]').count()) atFailures.push("airport traffic: the row is still on the page after the card is closed");
      atInfo.calls = tfmsCalls.length;
    } catch (e) {
      atFailures.push("airport traffic: battery crashed: " + String(e?.message || e).split("\n")[0]);
    }
    for (const e of atErrors) atFailures.push("airport traffic: pageerror: " + e);
    console.log("\n[AIRPORT TRAFFIC] " + JSON.stringify(atInfo) + (atFailures.length ? "\n  ✗ " + atFailures.join("\n  ✗ ") : ""));
    await ctx.close();
    results.push({ page: "data-airporttraffic", width: 1440, label: "airport-traffic", screenshot: path.join(OUT, "data-airporttraffic-card-1440.png"),
                   tti: null, perf: {}, failures: atFailures, warnings: [], info: { airportTraffic: atInfo } });
  }

  // ── FLIGHT RESTRICTIONS battery ([PRODUCT] 2026-11-21; the /data map's Flight restrictions layer over GET
  // /api/data/swim/notam/tfrs, lib/air/flightRestrictions.ts) ───────────────
  // Driven through the real map: the layer is switched on and must draw one area and one vt-airspace symbol per
  // restriction served (a row with no usable shape is counted in the status line, never drawn), with the notice's
  // index circle on the dashed layer and the restriction's own outline on the solid one. The status line says how
  // many are in force, how many start within 6 hours, how many are held, how many restriction notices arrived with
  // no shape, that the list is not the FAA's complete one, and when the last notice arrived. The legend
  // lists every kind with the same symbol the map draws. A symbol opens the card: four chips, none cut; Details says
  // which shape is drawn and that a circle is larger than the restriction; the card stays inside the window at 1440,
  // 768 and 390. Switched off, both sources and all five layers are gone and no request follows.
  // 2026-11-22: a third fixture row is the real US notice FDC 6/9306, whose outline the server read from the notice's
  // words: it is on the dotted layer (neither solid nor dashed), is cited "FDC 6/9306", and its card says how it was drawn.
  // 2026-11-22 (second): switched on again with an answer that carries rows from the FAA's public list
  // (server/faaTfrList.ts): solid outline, a card that names the FAA's list and its limits, and a status line that
  // says when the list was read in place of "not the FAA's complete list".
  // 2026-11-23: under the layer's row, the server's record of the FAA's list read back (GET
  // /api/data/swim/notam/tfrs/history): what appeared and what left in the last 7 days, checked at 1440, 768 and 390.
  // `--page flightrestrictions` runs this battery alone.
  if (!only || only === "data" || only === "flightrestrictions") {
    const frFailures = [];
    const frInfo = {};
    const fail = (m) => frFailures.push("flight restrictions: " + m);
    const { ctx, page, flip } = await bootDemandPage(browser, port, 1);
    const frErrors = [];
    page.on("pageerror", (e) => frErrors.push(e.message));
    const FR_NOW = Date.now();
    const H = 3600_000;
    const base = { series: "A", number: "1", year: "2026", type: "N", issued: FR_NOW - H, scenario: null };
    const ZAG = { ...base, id: "LDZL:A4134/2026", classification: "INTL", location: "LDZL", qcode: "QRTCA", start: FR_NOW - 20 * 60_000, end: FR_NOW + 9 * H, minFl: 0, maxFl: 999,
      center: [45.8167, 15.95], radiusNm: 20, text: "TEMPORARY RESERVED AREA 'ZAGREB CENTAR' ACTIVATED DUE TO UAS ACTIVITIES.\r\nDURING ACTIVITY PENETRATION NOT ALLOWED.",
      geometry: { kind: "circle", center: [45.8167, 15.95], radiusNm: 20, approximate: true } };
    const VIP = { ...base, id: "FDC:6/1234", classification: "FDC", location: "KZDC", qcode: null, start: FR_NOW + 2 * H, end: FR_NOW + 5 * H, minFl: 0, maxFl: 180, center: null, radiusNm: null,
      text: "TEMPORARY FLIGHT RESTRICTIONS. PURSUANT TO 14 CFR SECTION 91.141 AIRCRAFT FLIGHT OPERATIONS ARE PROHIBITED",
      geometry: { kind: "polygon", rings: [[[38.6, -77.4], [38.6, -76.7], [39.2, -76.7], [39.2, -77.4], [38.6, -77.4]]] } };
    const LEX = { ...base, series: null, year: null, number: "9306", id: "KZDC:9306/", citation: "FDC 6/9306", classification: "FDC", location: "KZDC", qcode: null, start: FR_NOW - H, end: FR_NOW + 190 * H,
      minFl: null, maxFl: null, center: null, radiusNm: null,
      text: "VA..AIRSPACE 20NM W LEXINGTON, VA..TEMPORARY FLIGHT RESTRICTIONS. PURSUANT TO 14 CFR SECTION 91.137(A)(2) TEMPORARY FLIGHT RESTRICTIONS ARE IN EFFECT TO PROVIDE A SAFE ENVIRONMENT FOR FIRE FIGHTING ACFT OPS. WI AN AREA DEFINED AS 375200N0795230W (MOL272036.5) TO 375100N0795000W (MOL270034.6) TO 374830N0795200W (MOL266036.4) TO 374900N0795400W (MOL268037.9) TO POINT OF ORIGIN SFC-4000FT",
      geometry: { kind: "polygon", from: "text", unread: 0, rings: [[[37.866667, -79.875], [37.85, -79.833333], [37.808333, -79.866667], [37.816667, -79.9], [37.866667, -79.875]]] } };
    // 2026-11-22: the same notice as the server sends it once the FAA's public list is read (server/faaTfrList.ts builds
    // this row from the FAA's real detail file for FDC 6/9306), and a four-area notice with its limits as that file writes them
    let withList = false;
    // listed_since: the server's own record of the FAA's list (server/faaTfrArchive.ts, 2026-11-22): 6/9306 was
    // first seen at a read of ours; 6/9276 was already listed when the record began
    const LEX_LIST = { listed_since: { first_seen: FR_NOW - 26 * H, backfill: false }, id: "FAA:6/9306", source: "faa_list", citation: "FDC 6/9306", classification: "FDC", series: null, number: "6/9306", year: null, type: null, qcode: null, scenario: null,
      location: "ZDC", issued: FR_NOW - 14 * H, start: FR_NOW - H, end: FR_NOW + 190 * H, minFl: null, maxFl: null, center: null, radiusNm: null, text: LEX.text,
      geometry: { kind: "polygon", from: "faa_list", rings: LEX.geometry.rings },
      faa: { type: "HAZARDS", rule: "91.137(a)(2)", place: "20NM W LEXINGTON, VIRGINIA", areas_without_outline: 0,
        areas: [{ name: "Hazard Area1", lower: { value: 0, uom: "FT", ref: "ALT" }, upper: { value: 4000, uom: "FT", ref: "ALT" }, scheduled: true }] } };
    const MAYPORT_LIST = { ...LEX_LIST, listed_since: { first_seen: FR_NOW - 3 * H, backfill: true }, id: "FAA:6/9276", citation: "FDC 6/9276", number: "6/9276", location: "ZJX", start: FR_NOW + 2 * H, end: FR_NOW + 70 * H,
      text: "!FDC 6/9276 ZJX FL..AIRSPACE MAYPORT, FL..TEMPORARY FLIGHT RESTRICTIONS. PURSUANT TO 49 USC 40103(B)(3), THE FEDERAL AVIATION ADMINISTRATION (FAA) CLASSIFIES THE AIRSPACE DEFINED IN THIS NOTAM AS 'NATIONAL DEFENSE AIRSPACE'. PURSUANT TO 14 CFR 99.7 SPECIAL SECURITY INSTRUCTIONS",
      geometry: { kind: "polygon", from: "faa_list", rings: [[[30.5, -81.6], [30.5, -81.2], [30.2, -81.2], [30.2, -81.6], [30.5, -81.6]], [[30.9, -81.5], [30.9, -81.3], [30.7, -81.3], [30.7, -81.5], [30.9, -81.5]]] },
      faa: { type: "SECURITY", rule: "99.7", place: "MAYPORT, FLORIDA", areas_without_outline: 0, areas: [
        { name: "Area A", lower: { value: 0, uom: "FT", ref: "HEI" }, upper: { value: 2500, uom: "FT", ref: "ALT" }, scheduled: false },
        { name: "Area B", lower: { value: 1700, uom: "FT", ref: "ALT" }, upper: { value: 14000, uom: "FT", ref: "ALT" }, scheduled: false }] } };
    const BROKEN = { ...base, id: "X:1", classification: "FDC", location: "KZNY", qcode: null, start: FR_NOW, end: FR_NOW + H, text: "TEMPORARY FLIGHT RESTRICTIONS", geometry: { kind: "circle", center: [40, -74], radiusNm: 0 } };
    const calls = [];
    await page.route("**/api/data/swim/notam/tfrs", (route) => {
      calls.push(Date.now());
      return route.fulfill({ contentType: "application/json", body: JSON.stringify({
        at: FR_NOW, tfrs: withList ? [ZAG, VIP, LEX_LIST, MAYPORT_LIST] : [ZAG, VIP, LEX, BROKEN], stats: { notams: 423, tfrs: 18, tfrs_without_shape: withList ? 0 : 7, byClassification: {}, parseErrors: 0 },
        kept: { saved_every_s: 60, last_saved_at: FR_NOW - 30_000, last_save: "ok", restored: null },
        feed: { connected: true, last_message_at: FR_NOW - 12_000 }, source: "FAA SWIM FNS NOTAM (fixture)",
        ...(withList ? { faa_list: { status: "ok", read_at: FR_NOW - 4 * 60_000, last_attempt_at: FR_NOW - 4 * 60_000, last_cause: "ok", listed: 48, with_outline: 36, without_outline: 12, detail_pending: 0, also_in_feed: 1, not_drawn: [], restored: false, served: 2,
          // 2026-11-23: the text reader's running check (server/faaTfrTextCheck.ts), said on the status line
          text_check: { since: FR_NOW - 26 * H, last_judged_at: FR_NOW - H, compared: 8, agree: 7, partial: 1, disagree: 0, not_read: 2, feed_sent_geometry: 0, worst_agree_nm: 0.031, tolerance: "fixture", not_agreeing: [], dropped: 0 } } } : {}) }) });
    });
    // 2026-11-23: the server's record of the FAA's list, read back (GET /api/data/swim/notam/tfrs/history,
    // lib/air/restrictionHistory.ts): the list under the layer's row. One notice left, one appeared, one appeared
    // after a gap between reads; the record began inside the window. `historyFails` makes the route answer 500.
    const historyCalls = [];
    let historyFails = false;
    const hIso = (ms) => new Date(ms).toISOString();
    const hDay = (ms) => hIso(ms).slice(0, 10);
    const H_START = FR_NOW - 24 * H; // the UTC day of hDayRow(1), at any hour the harness runs
    const hDayRow = (n, over) => ({ day: hDay(FR_NOW - n * 24 * H), recorded: true, run: { listed: 48, with_outline: 36, without_outline: 12, detail_pending: 0 }, listed: 0, left: 0, already_listed: 0, unreadable_lines: 0, skipped: null, ...over });
    const HISTORY = {
      at: FR_NOW, days_requested: 7, from_day: hDay(FR_NOW - 6 * 24 * H), to_day: hDay(FR_NOW),
      days: [hDayRow(0, { listed: 2, left: 1 }), hDayRow(1, { already_listed: 48 }), ...[2, 3, 4, 5, 6].map((n) => hDayRow(n, { recorded: false, run: null }))],
      events: [
        { kind: "left", seen: hIso(FR_NOW - 2 * H), prev_read: hIso(FR_NOW - 2.5 * H), notice: "6/9141", citation: "FDC 6/9141", list: { type: "VIP", facility: "ZDC", state: "VA", description: "Middleburg, VA, Saturday, November 21, 2026 through Sunday, November 22, 2026 Local" },
          first_seen: hIso(H_START), first_seen_backfill: true, published_start: hIso(FR_NOW - 40 * H), published_end: hIso(FR_NOW - 3 * H), outline: null },
        { kind: "listed", seen: hIso(FR_NOW - 5 * H), prev_read: hIso(FR_NOW - 5.5 * H), notice: "6/9306", citation: "FDC 6/9306", list: { type: "HAZARDS", facility: "ZDC", state: "VA", description: "20NM W LEXINGTON, VA, Sunday, November 22, 2026 through Sunday, November 29, 2026 UTC" },
          first_seen: null, first_seen_backfill: false, published_start: hIso(FR_NOW - H), published_end: hIso(FR_NOW + 190 * H), outline: true },
        { kind: "listed", seen: hIso(FR_NOW - 9 * H), prev_read: hIso(FR_NOW - 12 * H), notice: "6/9402", citation: "FDC 6/9402", list: { type: "SPACE OPERATIONS", facility: "ZMA", state: "FL", description: "CAPE CANAVERAL, FL, Monday, November 23, 2026 through Tuesday, November 24, 2026 UTC" },
          first_seen: null, first_seen_backfill: false, published_start: null, published_end: null, outline: false },
      ],
      events_cut: 0, limits: { max_days: 31, max_events: 500 },
      record: { status: "recording", started_at: H_START, last_recorded_read: FR_NOW - 4 * 60_000, open: 48, last: { at: FR_NOW - 4 * 60_000, status: "recorded", rows: 0 }, rows_this_process: 3 },
      source: "fixture", note: "fixture",
    };
    await page.route("**/api/data/swim/notam/tfrs/history**", (route) => {
      historyCalls.push(Date.now());
      if (historyFails) return route.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify({ error: "the record could not be read" }) });
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(HISTORY) });
    });
    const readHistory = () => page.evaluate(() => {
      const el = document.querySelector('[data-testid="restriction-history"]');
      if (!el) return null;
      const r = el.getBoundingClientRect();
      const panel = el.closest(".vt-layer-panel")?.getBoundingClientRect() ?? null;
      const t = (n) => (n.textContent || "").replace(/\s+/g, " ").trim();
      const rows = Array.from(el.querySelectorAll("li"));
      return {
        state: el.getAttribute("data-state"), head: t(el.querySelector('[data-testid="restriction-history-head"]') ?? document.createElement("i")),
        rows: rows.map((li) => ({ kind: li.getAttribute("data-kind"), text: t(li), over: Math.round(li.scrollWidth - li.clientWidth) })),
        notes: Array.from(el.querySelectorAll(".vt-legend-note")).map(t), text: t(el),
        // how far the block reaches past its panel, and past the window
        pastPanel: panel ? Math.round(Math.max(0, r.right - panel.right, panel.left - r.left)) : null,
        pastWindow: Math.round(Math.max(0, r.right - window.innerWidth, -r.left)),
        smallest: Math.min(...Array.from(el.querySelectorAll("li, .vt-legend-note")).map((n) => parseFloat(getComputedStyle(n).fontSize))),
      };
    });
    const LAYERS = ["flight-restrictions-fill", "flight-restrictions-line", "flight-restrictions-line-text", "flight-restrictions-line-enclosing", "flight-restrictions-sym"];
    const mapState = () => page.evaluate((ids) => {
      const m = window.__vtMap;
      const has = (id) => { try { return !!m.getLayer(id); } catch { return false; } };
      const src = (id) => { try { return !!m.getSource(id); } catch { return false; } };
      const q = (id, filter) => { try { return m.querySourceFeatures(id, filter ? { filter } : undefined).map((f) => f.properties); } catch { return []; } };
      const uniq = (rows) => Array.from(new Map(rows.map((p) => [p.id, p])).values()).sort((a, b) => String(a.id).localeCompare(String(b.id)));
      return {
        layers: ids.filter(has), sources: ["flight-restrictions-areas", "flight-restrictions-pts"].filter(src),
        symType: has("flight-restrictions-sym") ? m.getLayer("flight-restrictions-sym").type : null,
        icon: has("flight-restrictions-sym") ? m.getLayoutProperty("flight-restrictions-sym", "icon-image") : null,
        dash: has("flight-restrictions-line-enclosing") ? m.getPaintProperty("flight-restrictions-line-enclosing", "line-dasharray") : null,
        textDash: has("flight-restrictions-line-text") ? m.getPaintProperty("flight-restrictions-line-text", "line-dasharray") : null,
        ownDash: has("flight-restrictions-line") ? m.getPaintProperty("flight-restrictions-line", "line-dasharray") ?? null : "missing",
        pts: uniq(q("flight-restrictions-pts")), areas: uniq(q("flight-restrictions-areas")),
      };
    }, LAYERS);
    const rowText = () => page.evaluate(() => (document.querySelector('[data-vt-layer="flight_restrictions"]')?.textContent || "").replace(/\s+/g, " ").trim());
    const openCard = async (id, lon, lat) => {
      await page.keyboard.press("Escape").catch(() => {});
      await page.evaluate(([lo, la]) => window.__vtMap.jumpTo({ center: [lo, la], zoom: 6, pitch: 0, bearing: 0 }), [lon, lat]);
      let spot = null;
      for (let i = 0; i < 40 && !spot; i++) {
        await page.waitForTimeout(400);
        spot = await page.evaluate((rid) => {
          try {
            const m = window.__vtMap;
            const f = m.querySourceFeatures("flight-restrictions-pts").find((x) => x.properties.id === rid);
            if (!f || m.isMoving()) return null;
            const px = m.project(f.geometry.coordinates);
            const rect = m.getContainer().getBoundingClientRect();
            return { x: rect.left + px.x, y: rect.top + px.y };
          } catch { return null; }
        }, id);
      }
      if (!spot) { fail(`fixture restriction ${id} never rendered as a clickable symbol`); return false; }
      await page.mouse.click(spot.x, spot.y);
      const card = await page.waitForSelector(`.vt-site-card[role="dialog"][aria-label^="Flight restriction"]`, { timeout: 10000 }).catch(() => null);
      if (!card) { fail(`clicking ${id} opened no card`); return false; }
      return true;
    };
    const readCard = () => page.evaluate(() => {
      const c = document.querySelector('.vt-site-card[role="dialog"]');
      if (!c) return null;
      const r = c.getBoundingClientRect();
      const chips = Array.from(c.querySelectorAll(".vt-card-stat")).map((el) => {
        const v = el.querySelector(".vt-card-stat-v");
        const l = el.querySelector(".vt-card-stat-l");
        return { l: l?.textContent || "", v: v?.textContent || "", cut: Math.max(v ? v.scrollWidth - v.clientWidth : 0, l ? l.scrollWidth - l.clientWidth : 0) };
      });
      return {
        title: c.getAttribute("aria-label"), cat: (c.querySelector(".vt-site-card-cat")?.textContent || "").trim(), chips,
        facts: Object.fromEntries(Array.from(c.querySelectorAll(".vt-card-fact")).map((el) => [el.querySelector(".vt-card-fact-l")?.textContent || "", el.querySelector(".vt-card-fact-v")?.textContent || ""])),
        body: c.querySelector(".vt-site-card-body")?.textContent || "", open: !!c.querySelector(".vt-card-detbody"),
        left: Math.round(r.left), right: Math.round(r.right - window.innerWidth), top: Math.round(r.top), bottom: Math.round(r.bottom - window.innerHeight),
        pageOver: document.documentElement.scrollWidth - window.innerWidth, text: c.textContent || "",
      };
    });
    const checkCard = async (w) => {
      let c = await readCard();
      if (!c) { fail(`${w}: the card is gone`); return; }
      if (!c.open) {
        await page.locator('.vt-site-card[role="dialog"] button', { hasText: "DETAILS" }).first().click({ timeout: 2000 }).catch(() => {});
        await page.waitForTimeout(250);
        c = await readCard();
      }
      if (c.title !== "Flight restriction LDZL A4134/2026") fail(`${w}: the card is titled "${c.title}"`);
      if (!c.cat.startsWith("Temporary restricted area (ICAO notice) · in force until ")) fail(`${w}: the card's second line reads "${c.cat}"`);
      const want = [["Status", "active"], ["Lower", "surface"], ["Upper", "unlim."]];
      if (c.chips.length !== 4) fail(`${w}: ${c.chips.length} chips (4)`);
      for (const [l, v] of want) if (!c.chips.some((x) => x.l === l && x.v === v)) fail(`${w}: no chip "${l}: ${v}" in ${JSON.stringify(c.chips.map((x) => [x.l, x.v]))}`);
      for (const x of c.chips) if (x.cut > 1) fail(`${w}: the chip "${x.l}: ${x.v}" is cut by ${x.cut}px`);
      if (!c.open) { fail(`${w}: Details did not open`); return; }
      if (!/^a circle of [\d.,]+ (mi|km) around the notice's index point\. It encloses the restriction and is larger than it/.test(c.facts["Drawn shape"] || "")) fail(`${w}: Drawn shape reads "${c.facts["Drawn shape"]}"`);
      if (c.facts["Rule cited"] !== "none cited; ICAO Q-code subject RT, temporary restricted area") fail(`${w}: Rule cited reads "${c.facts["Rule cited"]}"`);
      if (c.facts["Notice"] !== "LDZL A4134/2026" || c.facts["Q-code, as sent"] !== "QRTCA") fail(`${w}: the notice facts read ${JSON.stringify(c.facts)}`);
      if (!c.body.includes("TEMPORARY RESERVED AREA 'ZAGREB CENTAR'") || !c.body.includes("not for flight planning or any operational use.") || !c.body.includes("This is not the FAA's complete list")) fail(`${w}: the card body lacks the notice text, the not-complete sentence or the not-for-flight-planning line`);
      if (/NaN|undefined|\bnull\b|Infinity/.test(c.text)) fail(`${w}: a bare NaN / undefined / null on the card`);
      if (c.left < -1 || c.right > 1 || c.top < -1 || c.bottom > 1 || c.pageOver > 1) fail(`${w}: the card leaves the window (left ${c.left}, right +${c.right}, top ${c.top}, bottom +${c.bottom}, page overflow ${c.pageOver}px)`);
      await page.screenshot({ path: path.join(OUT, `data-flightrestrictions-${w}.png`) }).catch(() => {});
    };
    try {
      const before = await mapState();
      if (before.layers.length || before.sources.length) fail(`drawn before the layer is switched on: ${JSON.stringify(before)}`);
      if (calls.length) fail(`${calls.length} requests before the layer is switched on`);
      if (!(await flip("flight_restrictions", true))) fail("the layer toggle never landed");
      let st = null;
      for (let i = 0; i < 40; i++) {
        await page.waitForTimeout(300);
        st = await mapState();
        if (st.layers.length === 5) break;
      }
      // both fixture rows in view so querySourceFeatures can see them
      await page.evaluate(() => window.__vtMap.jumpTo({ center: [-30, 42], zoom: 1.6, pitch: 0, bearing: 0 }));
      for (let i = 0; i < 30; i++) {
        await page.waitForTimeout(300);
        st = await mapState();
        if (st.pts.length === 3 && st.areas.length >= 2) break;
      }
      frInfo.layers = st.layers; frInfo.points = st.pts.map((p) => p.id);
      if (st.layers.length !== 5 || st.sources.length !== 2) fail(`layers ${JSON.stringify(st.layers)} / sources ${JSON.stringify(st.sources)} (5 layers, 2 sources)`);
      if (st.symType !== "symbol" || st.icon !== "vt-airspace") fail(`the point layer is a ${st.symType} drawing ${st.icon} (a symbol drawing vt-airspace)`);
      if (!Array.isArray(st.dash) || st.ownDash !== null) fail(`the index circle must be dashed and the restriction's own outline solid (dash ${JSON.stringify(st.dash)}, own ${JSON.stringify(st.ownDash)})`);
      if (!Array.isArray(st.textDash) || JSON.stringify(st.textDash) === JSON.stringify(st.dash)) fail(`an outline read from the notice's text must have its own dotted line, unlike both others (text ${JSON.stringify(st.textDash)}, circle ${JSON.stringify(st.dash)})`);
      const wantPts = [
        { id: "FDC:6/1234", kind: "security", color: "#f87171", shape: "own", in_force: false },
        { id: "KZDC:9306/", kind: "hazard", color: "#fb923c", shape: "text", in_force: true },
        { id: "LDZL:A4134/2026", kind: "icao_temporary", color: "#2dd4bf", shape: "enclosing", in_force: true },
      ];
      if (JSON.stringify(st.pts) !== JSON.stringify(wantPts)) fail(`the symbols carry ${JSON.stringify(st.pts)} (expected ${JSON.stringify(wantPts)})`);
      // at this world zoom the 3 NM Lexington area is smaller than a pixel and the map's tiler drops it; it is
      // checked below, after the camera goes to it
      const bigAreas = wantPts.filter((p) => p.id !== "KZDC:9306/");
      if (JSON.stringify(st.areas.filter((p) => p.id !== "KZDC:9306/")) !== JSON.stringify(bigAreas)) fail(`the areas carry ${JSON.stringify(st.areas)}`);
      const row = await rowText();
      frInfo.row = row;
      if (!row.includes("3 restrictions")) fail(`the layer row does not count 3 restrictions: "${row}"`);
      if (!/2 in force, 1 starting within 6 hours of 18 held · 1 could not be read · 7 restriction notices arrived without a shape and are not drawn · not the FAA's complete list · last notice received \d+ (s|min) ago/.test(row)) fail(`the layer row's status reads "${row}"`);
      const legend = await page.evaluate(() => Array.from(document.querySelectorAll('.vt-legend-item[data-vt-icon="vt-airspace"]')).map((el) => (el.textContent || "").trim()));
      frInfo.legend = legend.length;
      const outlineChips = await page.evaluate(() => Array.from(document.querySelectorAll(".vt-legend-chip")).map((el) => (el.textContent || "").trim()).filter((t) => /outline|encloses/i.test(t)));
      for (const t of ["The restriction's own outline", "Outline read from the notice's text", "A circle that encloses it (outline not sent)"]) if (!outlineChips.includes(t)) fail(`the legend lacks the outline style "${t}" (has ${JSON.stringify(outlineChips)})`);
      const wantLegend = ["Security (91.141, 99.7)", "Disaster, hazard or emergency (91.137 to 91.139)", "Space flight operations (91.143)", "Air show or major sporting event (91.145)", "Temporary restricted area (ICAO notice)", "Flight restriction, rule not cited"];
      if (JSON.stringify(legend) !== JSON.stringify(wantLegend)) fail(`the legend lists ${JSON.stringify(legend)}`);
      if (calls.length !== 1) fail(`switching the layer on made ${calls.length} requests (exactly 1)`);
      if (await openCard("LDZL:A4134/2026", 15.95, 45.8167)) {
        await checkCard(1440);
        await page.setViewportSize({ width: 768, height: 1024 });
        await page.waitForTimeout(1200);
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        await checkCard(768);
        await page.setViewportSize({ width: 390, height: 844 });
        await page.waitForTimeout(1200);
        await page.click('[aria-label="Collapse layers panel"]', { timeout: 1500 }).catch(() => {});
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        await checkCard(390);
        await page.setViewportSize({ width: 1440, height: 900 });
        await page.waitForTimeout(1200);
      }
      // a restriction that has not started: its own outline, the cited rule, "Starts"
      if (await openCard("FDC:6/1234", -77.05, 38.9)) {
        const c = await readCard();
        if (c?.title !== "Flight restriction FDC 6/1234" || !c.cat.startsWith("Security (91.141, 99.7) · starts ")) fail(`the second card reads "${c?.title}" / "${c?.cat}"`);
        if (!c?.chips.some((x) => x.l === "Starts") || !c?.chips.some((x) => x.l === "Upper" && x.v === "FL180")) fail(`the second card's chips read ${JSON.stringify(c?.chips.map((x) => [x.l, x.v]))}`);
        await page.screenshot({ path: path.join(OUT, "data-flightrestrictions-own-1440.png") }).catch(() => {});
      }
      // the real US notice: cited as people cite it, and the card says its outline was read from the text
      if (await openCard("KZDC:9306/", -79.87, 37.835)) {
        const near = await mapState();
        const lexArea = near.areas.find((p) => p.id === "KZDC:9306/");
        if (JSON.stringify(lexArea) !== JSON.stringify(wantPts[1])) fail(`zoomed to it, the text-outline area carries ${JSON.stringify(lexArea)} (expected ${JSON.stringify(wantPts[1])})`);
        const drawnOn = await page.evaluate(() => {
          const m = window.__vtMap;
          const n = (id) => { try { return m.queryRenderedFeatures({ layers: [id] }).filter((f) => f.properties.id === "KZDC:9306/").length; } catch { return -1; } };
          return { text: n("flight-restrictions-line-text"), own: n("flight-restrictions-line"), enclosing: n("flight-restrictions-line-enclosing") };
        });
        if (!(drawnOn.text > 0) || drawnOn.own !== 0 || drawnOn.enclosing !== 0) fail(`the text outline must be drawn by the dotted layer only: ${JSON.stringify(drawnOn)}`);
        let c = await readCard();
        if (c && !c.open) {
          await page.locator('.vt-site-card[role="dialog"] button', { hasText: "DETAILS" }).first().click({ timeout: 2000 }).catch(() => {});
          await page.waitForTimeout(250);
          c = await readCard();
        }
        if (c?.title !== "Flight restriction FDC 6/9306" || !c.cat.startsWith("Disaster, hazard or emergency (91.137 to 91.139) · in force until ")) fail(`the text-outline card reads "${c?.title}" / "${c?.cat}"`);
        if (c?.facts["Drawn shape"] !== "an outline read from the notice's text; the notice sent no geometry. Circles and arcs are drawn as short straight sides") fail(`the text-outline card's Drawn shape reads "${c?.facts["Drawn shape"]}"`);
        if (c?.facts["Notice"] !== "FDC 6/9306" || c?.facts["Lower limit"] !== "not given") fail(`the text-outline card's facts read ${JSON.stringify(c?.facts)}`);
        if (!c?.body.includes("This notice has no index limits; its lower and upper limits are written in its text.") || !c?.body.includes("SFC-4000FT")) fail("the text-outline card does not say where its limits are");
        for (const x of c?.chips ?? []) if (x.cut > 1) fail(`text-outline card: the chip "${x.l}: ${x.v}" is cut by ${x.cut}px`);
        if (/NaN|undefined|\bnull\b|Infinity/.test(c?.text ?? "")) fail("text-outline card: a bare NaN / undefined / null");
        await page.screenshot({ path: path.join(OUT, "data-flightrestrictions-text-1440.png") }).catch(() => {});
        // close in, so the dotted outline itself is in the picture a session reviews
        await page.evaluate(() => window.__vtMap.jumpTo({ center: [-79.95, 37.835], zoom: 10.3, pitch: 0, bearing: 0 }));
        await page.waitForTimeout(1500);
        await page.screenshot({ path: path.join(OUT, "data-flightrestrictions-text-outline-1440.png") }).catch(() => {});
        for (const w of [768, 390]) {
          await page.setViewportSize({ width: w, height: w === 768 ? 1024 : 844 });
          await page.waitForTimeout(1200);
          if (w === 390) await page.click('[aria-label="Collapse layers panel"]', { timeout: 1500 }).catch(() => {});
          await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
          const k = await readCard();
          if (!k) fail(`${w}: the text-outline card is gone`);
          else {
            if (k.left < -1 || k.right > 1 || k.top < -1 || k.bottom > 1 || k.pageOver > 1) fail(`${w}: the text-outline card leaves the window (left ${k.left}, right +${k.right}, top ${k.top}, bottom +${k.bottom})`);
            for (const x of k.chips) if (x.cut > 1) fail(`${w}: text-outline card chip "${x.l}: ${x.v}" is cut by ${x.cut}px`);
          }
          await page.screenshot({ path: path.join(OUT, `data-flightrestrictions-text-${w}.png`) }).catch(() => {});
        }
        await page.setViewportSize({ width: 1440, height: 900 });
        await page.waitForTimeout(1200);
      }
      // switched off: everything the layer added is removed, and nothing asks again
      await page.keyboard.press("Escape").catch(() => {});
      await page.click('[aria-label="Close details"]', { timeout: 1500 }).catch(() => {});
      if (!(await page.locator(".vt-layer-panel").count())) await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
      await page.waitForTimeout(300);
      const callsOn = calls.length;
      if (!(await flip("flight_restrictions", false))) fail("the layer would not switch off");
      await page.waitForTimeout(800);
      const off = await mapState();
      if (off.layers.length || off.sources.length) fail(`switched off, the map still holds ${JSON.stringify(off.layers)} ${JSON.stringify(off.sources)}`);
      if (await page.locator('.vt-legend-item[data-vt-icon="vt-airspace"]').count()) fail("switched off, the legend still lists flight restrictions");
      if (calls.length !== callsOn) fail(`switching off made ${calls.length - callsOn} more requests`);
      frInfo.calls = calls.length;
      // ── the same layer once the server has read the FAA's public list (2026-11-22) ──
      // Two rows come from the list. They are drawn with the solid line (the FAA's own outline), their cards
      // name the FAA's public list as the source and carry the limits as the FAA's file writes them, and the
      // status line says when the list was read and how many listed notices have no outline, in place of
      // "not the FAA's complete list".
      withList = true;
      if (!(await flip("flight_restrictions", true))) fail("list: the layer would not switch on again");
      await page.evaluate(() => window.__vtMap.jumpTo({ center: [-30, 42], zoom: 1.6, pitch: 0, bearing: 0 }));
      let ls = null;
      for (let i = 0; i < 40; i++) {
        await page.waitForTimeout(300);
        ls = await mapState();
        if (ls.layers.length === 5 && ls.pts.length === 4) break;
      }
      const wantList = [
        { id: "FAA:6/9276", kind: "security", color: "#f87171", shape: "own", in_force: false, from_list: true },
        { id: "FAA:6/9306", kind: "hazard", color: "#fb923c", shape: "own", in_force: true, from_list: true },
        { id: "FDC:6/1234", kind: "security", color: "#f87171", shape: "own", in_force: false },
        { id: "LDZL:A4134/2026", kind: "icao_temporary", color: "#2dd4bf", shape: "enclosing", in_force: true },
      ];
      if (JSON.stringify(ls.pts) !== JSON.stringify(wantList)) fail(`list: the symbols carry ${JSON.stringify(ls.pts)} (expected ${JSON.stringify(wantList)})`);
      const lrow = await rowText();
      frInfo.listRow = lrow;
      if (!/2 in force, 2 starting within 6 hours \(2 from the FAA's public list, 2 from the notice feed, which holds 18\) · FAA public list read 4 min ago: 48 US restrictions listed, 36 with a published outline, 12 with none \(not drawn\) · other countries: only what the notice feed has sent · our reading of notice text checked against the FAA's outline on 10 notices since [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}Z: 7 of 8 read match, 1 in part, 2 not read · last notice received \d+ (s|min) ago/.test(lrow)) fail(`list: the layer row's status reads "${lrow}"`);
      if (lrow.includes("not the FAA's complete list")) fail("list: the status line still says the list is not complete while the FAA's list is read and fresh");
      if (await openCard("FAA:6/9306", -79.87, 37.835)) {
        const on = await page.evaluate(() => {
          const m = window.__vtMap;
          const n = (id) => { try { return m.queryRenderedFeatures({ layers: [id] }).filter((f) => f.properties.id === "FAA:6/9306").length; } catch { return -1; } };
          return { text: n("flight-restrictions-line-text"), own: n("flight-restrictions-line"), enclosing: n("flight-restrictions-line-enclosing") };
        });
        if (!(on.own > 0) || on.text !== 0 || on.enclosing !== 0) fail(`list: the FAA's outline must be drawn by the solid layer only: ${JSON.stringify(on)}`);
        let c = await readCard();
        if (c && !c.open) {
          await page.locator('.vt-site-card[role="dialog"] button', { hasText: "DETAILS" }).first().click({ timeout: 2000 }).catch(() => {});
          await page.waitForTimeout(250);
          c = await readCard();
        }
        if (c?.title !== "Flight restriction FDC 6/9306" || !c.cat.startsWith("Disaster, hazard or emergency (91.137 to 91.139) · in force until ")) fail(`list: the card reads "${c?.title}" / "${c?.cat}"`);
        if (c?.facts["Drawn shape"] !== "the FAA's published outline for this notice, copied from tfr.faa.gov; it was not sent by the notice feed") fail(`list: Drawn shape reads "${c?.facts["Drawn shape"]}"`);
        if (c?.facts["FAA list type"] !== "HAZARDS" || c?.facts["Place"] !== "20NM W LEXINGTON, VIRGINIA" || c?.facts["Lower limit"] !== "surface") fail(`list: the card's facts read ${JSON.stringify(c?.facts)}`);
        if (!/^[\d,.]+ (ft|m) above sea level$/.test(c?.facts["Upper limit"] || "")) fail(`list: Upper limit reads "${c?.facts["Upper limit"]}" (a height through the unit formatter, above sea level)`);
        if (!(c?.facts["Hours"] || "").startsWith("certain hours only")) fail(`list: the card does not say the restriction is scheduled for certain hours (${JSON.stringify(c?.facts["Hours"])})`);
        if (!c?.chips.some((x) => x.l === "Lower" && x.v === "surface") || !c?.chips.some((x) => x.l === "Upper" && x.v === "4000ft")) fail(`list: the chips read ${JSON.stringify(c?.chips.map((x) => [x.l, x.v]))}`);
        if (!/FAA public list of flight restrictions \(tfr\.faa\.gov\), read \d+ min ago\. The outline and the limits are the FAA's own, copied as published\./.test(c?.body ?? "")) fail("list: the card body does not name the FAA's public list and when it was read");
        if ((c?.body ?? "").includes("This is not the FAA's complete list") || (c?.body ?? "").includes("SWIM FNS NOTAM")) fail("list: a list row's card speaks as a feed row's");
        // since when the server's record of the list has held the notice: our read's time, said as ours
        if (!/^first seen there [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}Z, by this server's read of the list \(every 30 minutes\)$/.test(c?.facts["On the FAA's list"] || "")) fail(`list: "On the FAA's list" reads "${c?.facts["On the FAA's list"]}"`);
        if (!(c?.body ?? "").includes("This server writes down when each restriction appears on the FAA's list and when it leaves")) fail("list: the card body does not say the list's comings and goings are recorded");
        if (/NaN|undefined|\bnull\b|Infinity/.test(c?.text ?? "")) fail("list: a bare NaN / undefined / null on the card");
        for (const x of c?.chips ?? []) if (x.cut > 1) fail(`list: the chip "${x.l}: ${x.v}" is cut by ${x.cut}px`);
        if (c && (c.left < -1 || c.right > 1 || c.top < -1 || c.bottom > 1 || c.pageOver > 1)) fail(`list 1440: the card leaves the window (left ${c.left}, right +${c.right}, top ${c.top}, bottom +${c.bottom})`);
        await page.evaluate(() => window.__vtMap.jumpTo({ center: [-79.95, 37.835], zoom: 10.3, pitch: 0, bearing: 0 }));
        await page.waitForTimeout(1500);
        await page.screenshot({ path: path.join(OUT, "data-flightrestrictions-list-1440.png") }).catch(() => {});
        for (const w of [768, 390]) {
          await page.setViewportSize({ width: w, height: w === 768 ? 1024 : 844 });
          await page.waitForTimeout(1200);
          if (w === 390) await page.click('[aria-label="Collapse layers panel"]', { timeout: 1500 }).catch(() => {});
          await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
          const k = await readCard();
          if (!k) fail(`list ${w}: the card is gone`);
          else {
            if (k.left < -1 || k.right > 1 || k.top < -1 || k.bottom > 1 || k.pageOver > 1) fail(`list ${w}: the card leaves the window (left ${k.left}, right +${k.right}, top ${k.top}, bottom +${k.bottom})`);
            for (const x of k.chips) if (x.cut > 1) fail(`list ${w}: chip "${x.l}: ${x.v}" is cut by ${x.cut}px`);
          }
          await page.screenshot({ path: path.join(OUT, `data-flightrestrictions-list-${w}.png`) }).catch(() => {});
        }
        await page.setViewportSize({ width: 1440, height: 900 });
        await page.waitForTimeout(1200);
      }
      // a notice with several areas: the chips say the limits vary and each area has its own row
      if (await openCard("FAA:6/9276", -81.4, 30.35)) {
        let c = await readCard();
        if (c && !c.open) {
          await page.locator('.vt-site-card[role="dialog"] button', { hasText: "DETAILS" }).first().click({ timeout: 2000 }).catch(() => {});
          await page.waitForTimeout(250);
          c = await readCard();
        }
        if (c?.title !== "Flight restriction FDC 6/9276" || !c.cat.startsWith("Security (91.141, 99.7) · starts ")) fail(`list: the several-area card reads "${c?.title}" / "${c?.cat}"`);
        if (!c?.chips.some((x) => x.l === "Lower" && x.v === "varies") || !c?.chips.some((x) => x.l === "Upper" && x.v === "varies")) fail(`list: the several-area chips read ${JSON.stringify(c?.chips.map((x) => [x.l, x.v]))}`);
        if (!/^surface to [\d,.]+ (ft|m) above sea level$/.test(c?.facts["Area A"] || "") || !/above sea level to [\d,.]+ (ft|m) above sea level$/.test(c?.facts["Area B"] || "")) fail(`list: the per-area limits read ${JSON.stringify([c?.facts["Area A"], c?.facts["Area B"]])}`);
        if (!(c?.facts["Drawn shape"] || "").startsWith("the FAA's published outline for this notice (2 areas)")) fail(`list: the several-area Drawn shape reads "${c?.facts["Drawn shape"]}"`);
        if (!/^already listed when this server's record of the list began, (\d{2}:\d{2}Z|[A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}Z)$/.test(c?.facts["On the FAA's list"] || "")) fail(`list: a notice listed before the record began reads "${c?.facts["On the FAA's list"]}"`);
        for (const x of c?.chips ?? []) if (x.cut > 1) fail(`list: several-area chip "${x.l}: ${x.v}" is cut by ${x.cut}px`);
        await page.screenshot({ path: path.join(OUT, "data-flightrestrictions-list-areas-1440.png") }).catch(() => {});
      }
      await page.keyboard.press("Escape").catch(() => {});
      await page.click('[aria-label="Close details"]', { timeout: 1500 }).catch(() => {});
      if (!(await page.locator(".vt-layer-panel").count())) await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
      await page.waitForTimeout(300);
      // ── the record read back, under the layer's row (2026-11-23) ──
      // One request when the layer goes on. The sentence says the window, the counts and when the record began;
      // each row is the FAA's words and the two reads the change lies between; a gap between reads and a notice
      // with no outline are said. Nothing leaves the panel or the window at 1440, 768 and 390. Switched off the
      // list is gone; switched on with the route failing it says it could not be read.
      const histAt = async (w) => {
        if (!(await page.locator(".vt-layer-panel").count())) await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
        await page.waitForTimeout(400);
        await page.evaluate(() => document.querySelector('[data-testid="restriction-history"]')?.scrollIntoView({ block: "center" }));
        await page.waitForTimeout(500);
        const h = await readHistory();
        if (!h) { fail(`history ${w}: no list under the layer's row`); return; }
        frInfo[`history${w}`] = { rows: h.rows.length, pastPanel: h.pastPanel, pastWindow: h.pastWindow, smallest: h.smallest };
        if (h.state !== "ok") fail(`history ${w}: state "${h.state}"`);
        if (!/^This server's record of the FAA's list, last 7 days: 2 appeared, 1 left\. The record began [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}Z; nothing before that is in it\.$/.test(h.head)) fail(`history ${w}: the sentence reads "${h.head}"`);
        if (JSON.stringify(h.rows.map((r) => r.kind)) !== JSON.stringify(["left", "listed", "listed"])) fail(`history ${w}: rows ${JSON.stringify(h.rows.map((r) => r.kind))}`);
        const [left, listed, gap] = h.rows.map((r) => r.text);
        if (!/^− ?FDC 6\/9141 · VIP · Middleburg, VA left the list between .*Z and .*Z; its published end was .*Z; on the list since before the record began$/.test(left || "")) fail(`history ${w}: the leaving row reads "${left}"`);
        if (!/^\+ ?FDC 6\/9306 · HAZARDS · 20NM W LEXINGTON, VA appeared between .*Z and .*Z; published end [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}Z$/.test(listed || "")) fail(`history ${w}: the appearing row reads "${listed}"`);
        if (!/FDC 6\/9402 · SPACE OPERATIONS · CAPE CANAVERAL, FL appeared between .* \(the two reads were 3\.0 h apart: the list was not read in between\); no outline published, not drawn$/.test(gap || "")) fail(`history ${w}: the row after a gap reads "${gap}"`);
        if (/early|cancel/i.test(h.text)) fail(`history ${w}: the list gives a verdict the record does not hold`);
        if (!h.notes.some((n) => n.startsWith("+ appeared on the FAA's list, − left it."))) fail(`history ${w}: the key to the two marks is missing`);
        if (/NaN|undefined|\bnull\b|Infinity/.test(h.text)) fail(`history ${w}: a bare NaN / undefined / null`);
        if (h.pastPanel > 1 || h.pastWindow > 1) fail(`history ${w}: the list reaches ${h.pastPanel}px past its panel, ${h.pastWindow}px past the window`);
        for (const r of h.rows) if (r.over > 1) fail(`history ${w}: a row is cut by ${r.over}px`);
        if (h.smallest < 10) fail(`history ${w}: text at ${h.smallest}px`);
        await page.screenshot({ path: path.join(OUT, `data-flightrestrictions-history-${w}.png`) }).catch(() => {});
      };
      if (historyCalls.length !== 2) fail(`history: ${historyCalls.length} requests over two switch-ons (2)`);
      await histAt(1440);
      for (const w of [768, 390]) {
        await page.setViewportSize({ width: w, height: w === 768 ? 1024 : 844 });
        await page.waitForTimeout(1200);
        await histAt(w);
      }
      await page.setViewportSize({ width: 1440, height: 900 });
      await page.waitForTimeout(1200);
      if (!(await page.locator(".vt-layer-panel").count())) await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
      await page.waitForTimeout(300);
      const histCallsOn = historyCalls.length;
      if (!(await flip("flight_restrictions", false))) fail("history: the layer would not switch off");
      await page.waitForTimeout(600);
      if (await readHistory()) fail("history: switched off, the list is still under the row");
      historyFails = true;
      if (!(await flip("flight_restrictions", true))) fail("history: the layer would not switch on again");
      let failed = null;
      for (let i = 0; i < 20 && failed?.state !== "error"; i++) { await page.waitForTimeout(300); failed = await readHistory(); }
      if (failed?.state !== "error" || !failed.text.includes("The record of the FAA's list could not be read; trying again in 5 minutes.")) fail(`history: with the route failing the list reads ${JSON.stringify(failed && { state: failed.state, text: failed.text })}`);
      if (historyCalls.length !== histCallsOn + 1) fail(`history: off and on again made ${historyCalls.length - histCallsOn} requests (1)`);
      frInfo.historyCalls = historyCalls.length;
      if (!(await flip("flight_restrictions", false))) fail("list: the layer would not switch off");
      await page.waitForTimeout(800);
      const off2 = await mapState();
      if (off2.layers.length || off2.sources.length) fail(`list: switched off, the map still holds ${JSON.stringify(off2.layers)} ${JSON.stringify(off2.sources)}`);
    } catch (e) {
      fail("battery crashed: " + String(e?.message || e).split("\n")[0]);
    }
    for (const e of frErrors) fail("pageerror: " + e);
    console.log("\n[FLIGHT RESTRICTIONS] " + JSON.stringify(frInfo) + (frFailures.length ? "\n  ✗ " + frFailures.join("\n  ✗ ") : ""));
    await ctx.close();
    results.push({ page: "data-flightrestrictions", width: 1440, label: "flight-restrictions", screenshot: path.join(OUT, "data-flightrestrictions-1440.png"),
                   tti: null, perf: {}, failures: frFailures, warnings: [], info: { flightRestrictions: frInfo } });
  }

  // ── BOOT MARKER battery (2026-11-19) ────────────────────────────────────
  // Measured on production: a visit to pricing, to the AI Engine page or to
  // the landing page, left normally, put the NEXT visit's map in reduced mode
  // ("the last visit ended during startup"). The module that arms the boot
  // marker runs on every page; only the mounted map ever disarmed it. This
  // battery drives the built bundle through what a visitor does:
  //   A  another page, then leave, then the map  -> no reduced mode, no report
  //   B  another page goes to the background     -> the marker is gone
  //   C  open the map from another page of the app -> the marker is armed again
  //   D  the renderer is KILLED during the map's start (chrome://crash: no code
  //      runs) -> the next visit still gets its reduced start. The protection
  //      the marker exists for must survive the fix.
  // `--page bootmarker` runs this battery alone.
  if (!only || only === "data" || only === "bootmarker") {
    const bmFailures = [];
    const bmInfo = {};
    const base = `http://127.0.0.1:${port}`;
    const shotOf = (w) => path.join(OUT, `data-bootmarker-${w}.png`);
    const NOTE = "Reduced mode: the last visit ended during startup";
    const newCtx = async (w) => {
      const ctx = await browser.newContext({ viewport: { width: w, height: w === 390 ? 844 : w === 768 ? 1024 : 900 }, deviceScaleFactor: 1 });
      await ctx.route("**/*", (route) => (route.request().url().startsWith(base) ? route.continue() : route.abort()));
      return ctx;
    };
    const state = (page) => page.evaluate((note) => {
      const j = (k) => { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch { return "unreadable"; } };
      return {
        note: document.body.innerText.includes(note), armed: !!localStorage.getItem("vt-boot-inprogress"),
        streak: localStorage.getItem("vt-boot-deadstreak"), prevKind: j("vt-boot-prev")?.kind ?? null,
        trail: (j("vt-boot-trail") || []).map((c) => c.step),
      };
    }, NOTE);
    const openMap = async (page) => {
      await page.goto(`${base}/app#/data`, { waitUntil: "load", timeout: 30000 });
      for (let i = 0; i < 60; i++) {
        if (await page.evaluate(() => !document.querySelector(".vt-map-skeleton"))) break;
        await page.waitForTimeout(250);
      }
      await page.waitForTimeout(1200);
      if (!(await page.locator(".vt-layer-panel").count())) await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
      await page.waitForTimeout(300);
      await page.locator('.vt-device-notice button[aria-label="Dismiss"]').click({ timeout: 600 }).catch(() => {});
    };
    try {
      // A — another page, left normally, then the map
      for (const [w, firsts] of [[1440, ["/pricing", "/app#/bot", "/"]], [768, ["/pricing"]], [390, ["/pricing"]]]) {
        for (const first of firsts) {
          const ctx = await newCtx(w);
          const page = await ctx.newPage();
          const errs = [];
          page.on("pageerror", (e) => errs.push(e.message));
          await page.goto(base + first, { waitUntil: "load", timeout: 30000 });
          await page.waitForTimeout(1200);
          await page.goto("about:blank");               // a real navigation away: pagehide fires
          await openMap(page);
          const st = await state(page);
          bmInfo[`A ${w} ${first}`] = st;
          if (st.note) bmFailures.push(`bootmarker: after visiting ${first} and leaving, the map at ${w}px says "${NOTE}"`);
          if (st.streak !== "0" || st.prevKind) bmFailures.push(`bootmarker: a visit to ${first} was recorded as a start-up death (streak=${st.streak} report=${st.prevKind}) at ${w}px`);
          if (!st.armed || !st.trail.includes("map-mount")) bmFailures.push(`bootmarker: the map did not arm its own start at ${w}px (armed=${st.armed} trail=${st.trail.join(">")})`);
          for (const e of errs) bmFailures.push(`bootmarker: pageerror at ${w}px after ${first}: ${e}`);
          if (first === "/pricing") await page.screenshot({ path: shotOf(w) });
          await ctx.close();
        }
      }
      // B — another page is sent to the background (the phone case: the tab is
      // later discarded with no pagehide)
      {
        const ctx = await newCtx(1440);
        const page = await ctx.newPage();
        await page.goto(`${base}/pricing`, { waitUntil: "load", timeout: 30000 });
        await page.waitForTimeout(600);
        const before = (await state(page)).armed;
        await page.evaluate(() => {
          Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
          document.dispatchEvent(new Event("visibilitychange"));
        });
        const after = (await state(page)).armed;
        bmInfo["B hidden"] = { before, after };
        if (after) bmFailures.push("bootmarker: a non-map page sent to the background kept its marker armed");
        await ctx.close();
      }
      // C + D — the map opened from another page of the app arms the marker;
      // a renderer killed during that start is still a start-up death
      {
        const ctx = await newCtx(1440);
        const page = await ctx.newPage();
        await page.goto(`${base}/pricing`, { waitUntil: "load", timeout: 30000 });
        await page.waitForTimeout(600);
        await page.evaluate(() => {      // background and back: the marker stands down
          Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
          document.dispatchEvent(new Event("visibilitychange"));
          Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "visible" });
          document.dispatchEvent(new Event("visibilitychange"));
        });
        const stood = !(await state(page)).armed;
        // in-app navigation: no reload, the module is not evaluated again
        await page.evaluate(() => { window.history.pushState({}, "", "/app#/data"); window.dispatchEvent(new PopStateEvent("popstate")); });
        let armed = false;
        for (let i = 0; i < 40 && !armed; i++) { await page.waitForTimeout(150); armed = (await state(page)).armed; }
        const st = await state(page);
        bmInfo["C in-app"] = { stood, armed, trail: st.trail };
        if (!stood) bmFailures.push("bootmarker: C could not set up (the marker did not stand down on the non-map page)");
        if (!armed) bmFailures.push(`bootmarker: opening the map from another page of the app did not arm the marker (trail=${st.trail.join(">")})`);
        // D — kill the renderer: no pagehide, no handler, nothing
        page.on("crash", () => {});
        await page.goto("chrome://crash", { timeout: 5000 }).catch(() => {});
        await page.close().catch(() => {});
        const page2 = await ctx.newPage();
        await openMap(page2);
        const st2 = await state(page2);
        bmInfo["D after a real death"] = st2;
        if (armed && (!st2.note || st2.prevKind !== "boot-crash")) bmFailures.push(`bootmarker: a renderer killed during the map's start was NOT followed by a reduced start (note=${st2.note} report=${st2.prevKind} streak=${st2.streak})`);
        await ctx.close();
      }
    } catch (e) {
      bmFailures.push("bootmarker: battery crashed: " + String(e?.message || e).split("\n")[0]);
    }
    console.log("\n[BOOT MARKER] " + JSON.stringify(bmInfo));
    for (const w of [1440, 768, 390]) {
      results.push({ page: "data-bootmarker", width: w, label: "boot-marker", screenshot: shotOf(w),
                     tti: null, perf: {}, failures: w === 1440 ? bmFailures : [], warnings: [], info: w === 1440 ? { bootMarker: bmInfo } : {} });
    }
  }

  await browser.close();
  srv.close();

  let hard = 0;
  for (const r of results) {
    const status = r.failures.length ? "FAIL" : "PASS";
    if (r.failures.length) hard += r.failures.length;
    console.log(`\n[${status}] ${r.page} @ ${r.width}px (${r.label}) -> ${path.relative(ROOT, r.screenshot)}`);
    r.failures.forEach((f) => console.log(`  ✗ ${f}`));
    r.warnings.forEach((w) => console.log(`  ⚠ ${w}`));
    if (r.info.map) console.log(`  map: ${Math.round(r.info.map.w)}x${Math.round(r.info.map.h)} at y=${Math.round(r.info.map.y)}`);
    if (r.tti != null) console.log(`  tti: ${r.tti}ms | perf: median ${r.perf?.median}ms p95 ${r.perf?.p95}ms over ${r.perf?.frames} frames | rendered ${r.perf?.renderedAircraft ?? "?"} aircraft`);
  }
  writeFileSync(path.join(OUT, "results.json"), JSON.stringify(results, null, 2));
  console.log(`\n${hard} hard failure(s). Results: .visual/results.json`);
  process.exit(hard && !SOFT ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
