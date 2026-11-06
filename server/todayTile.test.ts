// todayTile — the Bot page's "Today" tile (B1, 2026-11-03): equity now vs the
// prior close in $ and %, SPY's move over the same hours, the difference in
// points. Run: npx tsx --test server/todayTile.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { todayTile, spyToday, tileSession, previousTradingDay, todayLineRecord, TODAY_SOURCE, TODAY_SOURCE_BETWEEN, LAST_EQUITY_ROLL_EPS, type TileClock, type TodayLineRecord } from "./todayTile";
import { parseMarketHolidays } from "./optionsChainArchive";
import { zonedInstant } from "./jobHeartbeat";

const here = path.dirname(fileURLToPath(import.meta.url));
/** source with // and /* comments removed, so a comment naming a pattern never counts as the pattern */
const codeOnly = (src: string) => src.replace(/\/\*[\s\S]*?\*\//g, "").split("\n").filter((l) => !/^\s*\/\//.test(l)).join("\n");

// a Tuesday session, 10:02 ET; the bars end with the session in progress
const NOW = Date.parse("2026-11-03T15:02:00Z");
const FETCHED = NOW - 4 * 60_000;
const bars = [{ date: "2026-10-30", close: 680.1 }, { date: "2026-11-02", close: 684.42 }, { date: "2026-11-03", close: 686.13 }];
const noNaN = (t: unknown) => assert.ok(!/NaN|undefined/.test(JSON.stringify(t)), `a NaN or undefined leaked: ${JSON.stringify(t)}`);

test("an up day: dollars and percent from the account read, SPY from today's bar, the excess in points — and the three printed numbers add up", () => {
  const t = todayTile({ equity: "100484.56", lastEquity: "100172.07", spyBars: bars, spyFetchedAt: FETCHED, nowMs: NOW });
  assert.equal(t.day, "2026-11-03");
  assert.equal(t.equity_now, 100484.56);
  assert.equal(t.equity_prev_close, 100172.07);
  assert.equal(t.pnl_usd, 312.49);
  assert.equal(t.pnl_pct, 0.31, "312.49 / 100172.07 = 0.3119 %");
  assert.equal(t.pnl_reason, null);
  assert.equal(t.spy_pct_today, 0.25, "686.13 / 684.42 − 1 = 0.2499 %");
  assert.equal(t.spy_bar_date, "2026-11-03");
  assert.equal(t.spy_asof, new Date(FETCHED).toISOString());
  assert.equal(t.spy_reason, null);
  assert.equal(t.excess_pct, 0.06, "from the rounded sides: 0.31 − 0.25");
  assert.equal(t.asof, new Date(NOW).toISOString());
  assert.equal(t.source, TODAY_SOURCE);
  noNaN(t);
});

test("a down day and a flat day carry their signs; the excess is negative when SPY did better", () => {
  const down = todayTile({ equity: 99655.03, lastEquity: 100423.31, spyBars: bars, spyFetchedAt: FETCHED, nowMs: NOW });
  assert.equal(down.pnl_usd, -768.28);
  assert.equal(down.pnl_pct, -0.77);
  assert.equal(down.excess_pct, -1.02);
  const flat = todayTile({ equity: "100000", lastEquity: "100000", spyBars: bars, spyFetchedAt: FETCHED, nowMs: NOW });
  assert.equal(flat.pnl_usd, 0);
  assert.equal(flat.pnl_pct, 0);
  assert.equal(flat.excess_pct, -0.25);
  noNaN(down); noNaN(flat);
});

test("the percent is evaluateDailyPnl's: an impossible read on either side is null with its reason, never a fabricated zero (the daily-loss halt's own guard)", () => {
  for (const [equity, lastEquity] of [["100000", "0"], ["100000", undefined], ["abc", "100000"], [-5, "100000"], [undefined, undefined]] as const) {
    const t = todayTile({ equity, lastEquity, spyBars: bars, spyFetchedAt: FETCHED, nowMs: NOW });
    assert.equal(t.pnl_usd, null, `equity=${String(equity)} last=${String(lastEquity)}`);
    assert.equal(t.pnl_pct, null);
    assert.match(t.pnl_reason ?? "", /account read invalid/);
    assert.equal(t.excess_pct, null, "no account side, no excess");
    assert.equal(t.spy_pct_today, 0.25, "SPY's side is independent of the account read");
    noNaN(t);
  }
  // a credible side still reads on its own
  const t = todayTile({ equity: "100000", lastEquity: "0", spyBars: bars, spyFetchedAt: FETCHED, nowMs: NOW });
  assert.equal(t.equity_now, 100000);
  assert.equal(t.equity_prev_close, null);
});

test("SPY: nothing cached, an older last bar, or one bar only are null with the reason — an old day's move is never shown as today's", () => {
  const day = "2026-11-03";
  assert.deepEqual(spyToday(null, null, day), { pct: null, asof: null, barDate: null, reason: "no SPY bars fetched yet" });
  assert.deepEqual(spyToday([], FETCHED, day), { pct: null, asof: new Date(FETCHED).toISOString(), barDate: null, reason: "no SPY bars fetched yet" });
  const stale = spyToday(bars.slice(0, 2), FETCHED, day);
  assert.equal(stale.pct, null);
  assert.equal(stale.barDate, "2026-11-02");
  assert.equal(stale.reason, "no SPY bar for today yet (last 2026-11-02)");
  const one = spyToday([bars[2]], FETCHED, day);
  assert.equal(one.pct, null);
  assert.equal(one.reason, "SPY prior close missing");
  // the bars may arrive in any order; a garbage bar is ignored
  const shuffled = [bars[2], { date: "bad", close: NaN } as unknown as { date: string; close: number }, bars[0], bars[1]];
  assert.equal(spyToday(shuffled, FETCHED, day).pct, 0.25);
  // through the tile: SPY null leaves the account side intact and the excess null
  const t = todayTile({ equity: "100484.56", lastEquity: "100172.07", spyBars: bars.slice(0, 2), spyFetchedAt: FETCHED, nowMs: NOW });
  assert.equal(t.pnl_pct, 0.31);
  assert.equal(t.spy_pct_today, null);
  assert.equal(t.excess_pct, null);
  assert.match(t.spy_reason ?? "", /no SPY bar for today yet/);
  noNaN(t);
});

test("the day is the session's: after the close the completed bar is still the day's; past midnight ET it is STILL that session's (2026-11-06: it was the empty new day)", () => {
  const evening = Date.parse("2026-11-04T02:00:00Z"); // 21:00 ET on the 3rd
  const e = todayTile({ equity: "100484.56", lastEquity: "100172.07", spyBars: bars, spyFetchedAt: evening - 60_000, nowMs: evening });
  assert.equal(e.day, "2026-11-03");
  assert.equal(e.phase, "between");
  assert.equal(e.when, "after the close");
  assert.equal(e.spy_pct_today, 0.25);
  assert.equal(e.pnl_pct, 0.31, "the session's own ET day: last_equity is still the close before it");
  const smallHours = Date.parse("2026-11-04T06:00:00Z"); // 01:00 ET on the 4th
  const s = todayTile({ equity: "100484.56", lastEquity: "100172.07", spyBars: bars, spyFetchedAt: smallHours - 60_000, nowMs: smallHours });
  // [REPAIR] 2026-11-06, these three pins changed: before, day "2026-11-04", spy_pct_today null, spy_reason
  // "no SPY bar for today yet (last 2026-11-03)" — a day with no session yet, the bug this fixes
  assert.equal(s.day, "2026-11-03");
  assert.equal(s.spy_pct_today, 0.25);
  assert.equal(s.spy_reason, null);
  assert.equal(s.phase, "between");
  assert.equal(s.when, "overnight");
  // no daily line passed: past the session's own ET day the prior close is not known, so no account number
  assert.equal(s.pnl_pct, null);
  assert.match(s.pnl_reason ?? "", /^the close before the 2026-11-03 session is not recorded \(no daily line for 2026-11-03\)/);
  // a non-finite clock never throws (etDay would on an Invalid Date)
  assert.doesNotThrow(() => todayTile({ equity: "1", lastEquity: "1", spyBars: null, spyFetchedAt: null, nowMs: NaN }));
});

test("source ratchet: the module reuses evaluateDailyPnl and benchmarkVsSpy — it never divides equity by the prior close itself", () => {
  const src = codeOnly(fs.readFileSync(path.join(here, "todayTile.ts"), "utf8"));
  assert.match(src, /import \{ evaluateDailyPnl \} from "\.\/drawdownGuard"/);
  assert.match(src, /import \{ benchmarkVsSpy, type DailyClose \} from "\.\/spyBenchmark"/);
  assert.match(src, /evaluateDailyPnl\(i\.equity, i\.lastEquity\)/);
  assert.match(src, /benchmarkVsSpy\(\[\]/);
  assert.doesNotMatch(src, /\/\s*(prev|last|lastEquity|equity_prev_close)\b/, "the percent is evaluateDailyPnl's, not a second formula");
  assert.doesNotMatch(src, /\.close\s*\//, "SPY's move is benchmarkVsSpy's spyLast, not a second formula");
});

test("wiring ratchet: /api/bot/account serves `today` from todayTile, reading the SPY cache without waiting on it", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.match(bot, /import \{ todayTile, type TodayTile \} from "\.\/todayTile"/);
  const start = bot.indexOf('app.get("/api/bot/account"');
  assert.ok(start > 0, "the account route exists");
  const route = bot.slice(start, bot.indexOf("// Current positions from Alpaca", start));
  assert.match(route, /today: todayFor\(acct\)/, "the account payload carries the tile");
  const helperStart = bot.indexOf("function todayFor(");
  assert.ok(helperStart > 0, "todayFor lives beside the SPY cache");
  const helper = bot.slice(helperStart, bot.indexOf("\n  }\n", helperStart));
  assert.match(helper, /todayTile\(\{ equity: acct\.equity, lastEquity: acct\.last_equity, spyBars: spyBenchCache\?\.bars \?\? null, spyFetchedAt: spyBenchCache\?\.at \?\? null/);
  assert.doesNotMatch(helper, /await/, "the account poll never waits on an Alpaca bars fetch");
  assert.match(helper, /void spyBenchBars\(/, "a cold or expired cache is warmed for the next poll");
});

test("SPY's side names its basis ([RULE-REVIEW] 2026-11-05): price only, no dividends, like the paper account", () => {
  assert.equal(TODAY_SOURCE, "Alpaca paper account: equity now vs last_equity (the prior session's 16:00 ET close) · SPY: today's daily bar vs the prior close, price only, no dividends, like the paper account (Alpaca bars, adjustment=split, feed iex)");
  assert.doesNotMatch(TODAY_SOURCE, /dividend-adjusted|total return/);
});

test("on SPY's ex-dividend day the tile is like for like: an account holding only SPY, credited no dividend, reads 0.00 pp against the price-only bar", () => {
  // 2026-09-18, SPY ex $1.889: 762.60 → 761.69 on price (Yahoo closes); dividend-adjusted, the prior close
  // would be 762.60 × (1 − 1.889 / 762.60) = 760.71 and SPY would read +0.13 % while the holder lost 0.12 %
  const at = Date.parse("2026-09-18T19:00:00Z"); // 15:00 ET
  const priceBars = [{ date: "2026-09-17", close: 762.6 }, { date: "2026-09-18", close: 761.69 }];
  const held = (100_000 * 761.69) / 762.6;
  const t = todayTile({ equity: held, lastEquity: 100_000, spyBars: priceBars, spyFetchedAt: at - 60_000, nowMs: at });
  assert.equal(t.pnl_pct, -0.12);
  assert.equal(t.spy_pct_today, -0.12);
  assert.equal(t.excess_pct, 0);
  const totalBars = [{ date: "2026-09-17", close: 760.71 }, { date: "2026-09-18", close: 761.69 }];
  const old = todayTile({ equity: held, lastEquity: 100_000, spyBars: totalBars, spyFetchedAt: at - 60_000, nowMs: at });
  assert.equal(old.spy_pct_today, 0.13);
  assert.equal(old.excess_pct, -0.25, "the dividend-adjusted series charged the account the dividend it never received");
});

// ── BETWEEN SESSIONS ([REPAIR] 2026-11-06, open_questions.md 2026-11-06 "[UI] Two gaps": "The Today tile labels the
// overnight 'today' with the last session's move, a stale prior close overnight"). The numbers are the live account's
// on 2026-11-05/06: Thursday's daily line read last_equity 93,623.46 (Wednesday's close) at 20:12 ET; by 05:52 ET Friday
// Alpaca had rolled last_equity to 89,888.90 (Thursday's close) and equity read 89,657.32. SPY: −4.34 % on Thursday.

const OPEN: TileClock = { state: "open", source: "live" };
const CLOSED: TileClock = { state: "closed", source: "live" };
const ET_ = "America/New_York";
const cal = (holidays: string[] = [], halfDays: string[] = []) => ({ holidays: new Set(holidays), halfDays: new Set(halfDays) });
const NOV = cal(["2026-11-26"], ["2026-11-27"]);
const spy5 = [{ date: "2026-11-03", close: 686.13 }, { date: "2026-11-04", close: 690 }, { date: "2026-11-05", close: 660.05 }];
const thuLine: TodayLineRecord = { day: "2026-11-05", lastEquity: 93623.46, writtenAt: "2026-11-05T21:21:00.000Z" };

test("tileSession (a): during a session the tile is today's — the broker clock decides, the calendar only without one", () => {
  const at = Date.parse("2026-11-05T15:00:00Z"); // Thu 10:00 ET
  const s = tileSession(at, OPEN, NOV);
  assert.deepEqual({ phase: s.phase, day: s.day, prevDay: s.prevDay, when: s.when, source: s.source },
    { phase: "session", day: "2026-11-05", prevDay: "2026-11-04", when: "during the session", source: "broker clock" });
  assert.equal(s.closeMs, zonedInstant("2026-11-05", 16 * 60, ET_));
  // no reading / an unknown clock: the calendar (a trading day, 09:30 ET to its close)
  for (const c of [null, undefined, { state: "unknown", source: "none" } as TileClock]) {
    const k = tileSession(at, c, NOV);
    assert.equal(k.phase, "session");
    assert.equal(k.source, "market calendar (no usable broker clock reading)");
  }
  assert.equal(tileSession(at, { state: "open", source: "extrapolated" }, NOV).source, "broker clock, extrapolated from its last reading");
  // the broker's word beats the calendar's both ways (an unscheduled closure; a calendar that is wrong)
  assert.equal(tileSession(at, CLOSED, NOV).phase, "between");
  assert.equal(tileSession(at, CLOSED, NOV).day, "2026-11-05", "today's session opened in the calendar: it is the last one");
  assert.equal(tileSession(Date.parse("2026-11-07T17:00:00Z"), OPEN, NOV).phase, "session");
  // 09:29 / 09:30 ET, 15:59 / 16:00 ET by the calendar
  assert.equal(tileSession(Date.parse("2026-11-05T14:29:00Z"), null, NOV).phase, "between");
  assert.equal(tileSession(Date.parse("2026-11-05T14:30:00Z"), null, NOV).phase, "session");
  assert.equal(tileSession(Date.parse("2026-11-05T20:59:00Z"), null, NOV).phase, "session");
  assert.equal(tileSession(Date.parse("2026-11-05T21:00:00Z"), null, NOV).phase, "between");
});

test("tileSession (b)-(d): after the close, overnight, pre-market, weekends and holidays name the last session that ran", () => {
  const cases: Array<[string, TileClock | null, string, string, string]> = [
    // [now, clock, day, prevDay, when]
    ["2026-11-05T23:00:00Z", CLOSED, "2026-11-05", "2026-11-04", "after the close"], // (b) Thu 18:00 ET
    ["2026-11-06T04:59:00Z", CLOSED, "2026-11-05", "2026-11-04", "after the close"], // Thu 23:59 ET
    ["2026-11-06T06:00:00Z", CLOSED, "2026-11-05", "2026-11-04", "overnight"], // (c) Fri 01:00 ET
    ["2026-11-06T10:52:00Z", CLOSED, "2026-11-05", "2026-11-04", "pre-market"], // Fri 05:52 ET (the live reading)
    ["2026-11-07T17:00:00Z", CLOSED, "2026-11-06", "2026-11-05", "weekend"], // (d) Sat 12:00 ET
    ["2026-11-09T01:00:00Z", CLOSED, "2026-11-06", "2026-11-05", "weekend"], // Sun 20:00 ET
    ["2026-11-09T12:00:00Z", CLOSED, "2026-11-06", "2026-11-05", "pre-market"], // Mon 07:00 ET
    ["2026-11-26T17:00:00Z", CLOSED, "2026-11-25", "2026-11-24", "market holiday"], // Thanksgiving, 12:00 ET
    ["2026-11-27T13:00:00Z", CLOSED, "2026-11-25", "2026-11-24", "pre-market"], // the half day, 08:00 ET
    ["2026-11-27T19:00:00Z", null, "2026-11-27", "2026-11-25", "after the close"], // the half day, 14:00 ET, calendar
  ];
  for (const [now, clock, day, prevDay, when] of cases) {
    const s = tileSession(Date.parse(now), clock, NOV);
    assert.deepEqual({ phase: s.phase, day: s.day, prevDay: s.prevDay, when: s.when }, { phase: "between", day, prevDay, when }, now);
  }
  // the half day closes at 13:00 ET (market_calendar.py HALF_DAYS), by the calendar and in closeMs
  assert.equal(tileSession(Date.parse("2026-11-27T17:59:00Z"), null, NOV).phase, "session");
  assert.equal(tileSession(Date.parse("2026-11-27T18:00:00Z"), null, NOV).phase, "between");
  assert.equal(tileSession(Date.parse("2026-11-27T19:00:00Z"), null, NOV).closeMs, zonedInstant("2026-11-27", 13 * 60, ET_));
  // the frozen market_calendar.py itself (parsed, never duplicated) holds Thanksgiving 2026
  const py = fs.readFileSync(path.join(here, "..", "market_calendar.py"), "utf8");
  const real = { holidays: parseMarketHolidays(py), halfDays: parseMarketHolidays(py, "HALF_DAYS") };
  assert.equal(tileSession(Date.parse("2026-11-26T17:00:00Z"), CLOSED, real).day, "2026-11-25");
  assert.equal(previousTradingDay("2026-11-27", real), "2026-11-25");
  assert.equal(previousTradingDay("2026-11-09", null), "2026-11-06", "no calendar: weekends only");
});

test("(a) during a session nothing changed: today, against last_equity, SPY's bar in progress (not final)", () => {
  const at = Date.parse("2026-11-05T17:06:00Z"); // Thu 12:06 ET
  const t = todayTile({ equity: "90500", lastEquity: "93623.46", spyBars: spy5, spyFetchedAt: at - 4 * 60_000, nowMs: at, clock: OPEN, calendar: NOV, lines: [thuLine] });
  assert.equal(t.phase, "session");
  assert.equal(t.day, "2026-11-05");
  assert.equal(t.when, "during the session");
  assert.equal(t.phase_source, "broker clock");
  assert.equal(t.prev_day, "2026-11-04");
  assert.equal(t.equity_prev_close, 93623.46);
  assert.equal(t.prev_close_source, "last_equity");
  assert.equal(t.equity_end, 90500);
  assert.equal(t.end_basis, "now");
  assert.equal(t.pnl_usd, -3123.46);
  assert.equal(t.pnl_pct, -3.34);
  assert.equal(t.spy_pct_today, -4.34);
  assert.equal(t.spy_final, false, "fetched at 12:02 ET: the bar is the session in progress");
  assert.equal(t.excess_pct, 1);
  assert.equal(t.source, TODAY_SOURCE);
  noNaN(t);
});

test("(b) after the close the same evening: the session is named, last_equity is still its prior close, the change runs to equity now; SPY's bar is final once fetched after the close", () => {
  const at = Date.parse("2026-11-05T23:00:00Z"); // Thu 18:00 ET
  const fetched = Date.parse("2026-11-05T22:00:00Z"); // 17:00 ET
  const noLine = todayTile({ equity: "89800.22", lastEquity: "93623.46", spyBars: spy5, spyFetchedAt: fetched, nowMs: at, clock: CLOSED, calendar: NOV, lines: [] });
  assert.equal(noLine.phase, "between");
  assert.equal(noLine.day, "2026-11-05");
  assert.equal(noLine.when, "after the close");
  assert.equal(noLine.prev_close_source, "last_equity", "before the daily line exists (close + 20 min) the session's own ET day trusts last_equity");
  assert.equal(noLine.equity_prev_close, 93623.46);
  assert.equal(noLine.equity_end, 89800.22);
  assert.equal(noLine.end_basis, "now");
  assert.equal(noLine.pnl_usd, -3823.24);
  assert.equal(noLine.pnl_pct, -4.08);
  assert.equal(noLine.spy_pct_today, -4.34);
  assert.equal(noLine.spy_final, true);
  assert.equal(noLine.excess_pct, 0.26);
  assert.equal(noLine.source, TODAY_SOURCE_BETWEEN);
  // with the line written: the same prior close, now named as the line's record
  const withLine = todayTile({ equity: "89800.22", lastEquity: "93623.46", spyBars: spy5, spyFetchedAt: fetched, nowMs: at, clock: CLOSED, calendar: NOV, lines: [thuLine] });
  assert.equal(withLine.prev_close_source, "daily_line");
  assert.equal(withLine.pnl_usd, -3823.24);
  assert.equal(withLine.end_basis, "now");
  // SPY fetched before the close (the 30 min cache): the number is the bar as of 15:50 ET, not the final close
  const early = todayTile({ equity: "89800.22", lastEquity: "93623.46", spyBars: spy5, spyFetchedAt: Date.parse("2026-11-05T20:50:00Z"), nowMs: Date.parse("2026-11-05T21:10:00Z"), clock: CLOSED, calendar: NOV, lines: [] });
  assert.equal(early.spy_pct_today, -4.34);
  assert.equal(early.spy_final, false);
  noNaN(noLine); noNaN(withLine);
});

test("(c) overnight and pre-market: Thursday's session against Wednesday's close, never the empty Friday — to equity now before Alpaca's roll, to Thursday's own close after it", () => {
  const fetched = Date.parse("2026-11-06T10:40:00Z");
  // 01:00 ET, before the roll: last_equity is still Wednesday's close (the line's record)
  const pre = todayTile({ equity: "89814.42", lastEquity: "93623.46", spyBars: spy5, spyFetchedAt: fetched - 6 * 3_600_000, nowMs: Date.parse("2026-11-06T06:00:00Z"), clock: CLOSED, calendar: NOV, lines: [thuLine] });
  assert.equal(pre.day, "2026-11-05");
  assert.equal(pre.when, "overnight");
  assert.equal(pre.equity_prev_close, 93623.46);
  assert.equal(pre.prev_close_source, "daily_line");
  assert.equal(pre.end_basis, "now");
  assert.equal(pre.pnl_usd, -3809.04);
  assert.equal(pre.pnl_pct, -4.07, "the daily line's own −4.07 % (it read equity at 20:12 ET)");
  assert.equal(pre.spy_pct_today, -4.34);
  assert.equal(pre.excess_pct, 0.27);
  // 05:52 ET, after the roll (the live reading): last_equity is Thursday's close; the change is close to close
  const post = todayTile({ equity: "89657.32", lastEquity: "89888.90", spyBars: spy5, spyFetchedAt: fetched, nowMs: Date.parse("2026-11-06T10:52:00Z"), clock: CLOSED, calendar: NOV, lines: [thuLine] });
  assert.equal(post.day, "2026-11-05");
  assert.equal(post.when, "pre-market");
  assert.equal(post.equity_prev_close, 93623.46);
  assert.equal(post.equity_end, 89888.9);
  assert.equal(post.end_basis, "close");
  assert.equal(post.equity_now, 89657.32, "equity now is still served (the hover's since-the-close drift)");
  assert.equal(post.pnl_usd, -3734.56);
  assert.equal(post.pnl_pct, -3.99);
  assert.equal(post.excess_pct, 0.35);
  assert.equal(post.spy_final, true);
  assert.notEqual(post.pnl_usd, -231.58, "the old tile's 'today': the drift since Thursday's close, beside no SPY bar");
  // no line for the session: the prior close is not known, so no number — never last_equity mislabeled
  const none = todayTile({ equity: "89657.32", lastEquity: "89888.90", spyBars: spy5, spyFetchedAt: fetched, nowMs: Date.parse("2026-11-06T10:52:00Z"), clock: CLOSED, calendar: NOV, lines: [{ ...thuLine, day: "2026-11-04" }] });
  assert.equal(none.pnl_usd, null);
  assert.equal(none.pnl_pct, null);
  assert.equal(none.equity_prev_close, null);
  assert.equal(none.equity_end, null);
  assert.equal(none.pnl_reason, "the close before the 2026-11-05 session is not recorded (no daily line for 2026-11-05), and Alpaca's last_equity may already be that session's own close");
  assert.equal(none.spy_pct_today, -4.34, "SPY's side stands alone");
  assert.equal(none.excess_pct, null);
  // a line whose account read failed records no last_equity: the same as no line
  assert.equal(todayTile({ equity: "89657.32", lastEquity: "89888.90", spyBars: spy5, spyFetchedAt: fetched, nowMs: Date.parse("2026-11-06T10:52:00Z"), clock: CLOSED, calendar: NOV, lines: [{ ...thuLine, lastEquity: null }] }).pnl_usd, null);
  // a pre-market bar of the next day in the cache is not the session's
  const withFri = [...spy5, { date: "2026-11-06", close: 662 }];
  const am = todayTile({ equity: "89657.32", lastEquity: "89888.90", spyBars: withFri, spyFetchedAt: Date.parse("2026-11-06T13:30:00Z"), nowMs: Date.parse("2026-11-06T13:31:00Z"), clock: CLOSED, calendar: NOV, lines: [thuLine] });
  assert.equal(am.spy_bar_date, "2026-11-05");
  assert.equal(am.spy_pct_today, -4.34);
  // the session's bar not cached yet: the reason names the session, not "today"
  const stale = todayTile({ equity: "89657.32", lastEquity: "89888.90", spyBars: spy5.slice(0, 2), spyFetchedAt: fetched, nowMs: Date.parse("2026-11-06T10:52:00Z"), clock: CLOSED, calendar: NOV, lines: [thuLine] });
  assert.equal(stale.spy_pct_today, null);
  assert.equal(stale.spy_reason, "no SPY bar for 2026-11-05 yet (last 2026-11-04)");
  assert.equal(stale.spy_final, false);
  noNaN(pre); noNaN(post); noNaN(none); noNaN(am); noNaN(stale);
});

test("(d) weekends and holidays: the last session (Friday; the day before the holiday), close to close once Alpaca has rolled", () => {
  const friLine: TodayLineRecord = { day: "2026-11-06", lastEquity: 89888.9, writtenAt: "2026-11-06T21:22:00Z" };
  const spy6 = [...spy5, { date: "2026-11-06", close: 663.35 }];
  const sat = todayTile({ equity: "90420", lastEquity: "90500", spyBars: spy6, spyFetchedAt: Date.parse("2026-11-07T16:30:00Z"), nowMs: Date.parse("2026-11-07T17:00:00Z"), clock: CLOSED, calendar: NOV, lines: [thuLine, friLine] });
  assert.equal(sat.day, "2026-11-06");
  assert.equal(sat.when, "weekend");
  assert.equal(sat.prev_day, "2026-11-05");
  assert.equal(sat.equity_prev_close, 89888.9);
  assert.equal(sat.equity_end, 90500);
  assert.equal(sat.end_basis, "close");
  assert.equal(sat.pnl_usd, 611.1);
  assert.equal(sat.pnl_pct, 0.68);
  assert.equal(sat.spy_pct_today, 0.5, "663.35 / 660.05 − 1 = 0.4999 %");
  assert.equal(sat.excess_pct, 0.18);
  // Thanksgiving: Wednesday's session, its line, Alpaca rolled to Wednesday's close
  const wedLine: TodayLineRecord = { day: "2026-11-25", lastEquity: 91000, writtenAt: "2026-11-25T21:21:00Z" };
  const spyHol = [{ date: "2026-11-24", close: 670 }, { date: "2026-11-25", close: 673.35 }];
  const hol = todayTile({ equity: "91200", lastEquity: "91300", spyBars: spyHol, spyFetchedAt: Date.parse("2026-11-26T16:00:00Z"), nowMs: Date.parse("2026-11-26T17:00:00Z"), clock: CLOSED, calendar: NOV, lines: [wedLine] });
  assert.equal(hol.day, "2026-11-25");
  assert.equal(hol.when, "market holiday");
  assert.equal(hol.pnl_usd, 300);
  assert.equal(hol.pnl_pct, 0.33);
  assert.equal(hol.spy_pct_today, 0.5);
  noNaN(sat); noNaN(hol);
});

test("between sessions the read itself is still guarded: an impossible equity or last_equity is no number, with the reason", () => {
  const t = todayTile({ equity: "abc", lastEquity: "89888.90", spyBars: spy5, spyFetchedAt: Date.parse("2026-11-06T10:40:00Z"), nowMs: Date.parse("2026-11-06T10:52:00Z"), clock: CLOSED, calendar: NOV, lines: [thuLine] });
  assert.equal(t.pnl_usd, null);
  assert.equal(t.pnl_pct, null);
  assert.match(t.pnl_reason ?? "", /^account read invalid/);
  assert.equal(t.equity_now, null);
  assert.equal(t.excess_pct, null);
  assert.equal(t.spy_pct_today, -4.34);
  noNaN(t);
  // half a cent of float noise is not a roll
  const same = todayTile({ equity: "89814.42", lastEquity: String(93623.46 + LAST_EQUITY_ROLL_EPS / 2), spyBars: spy5, spyFetchedAt: null, nowMs: Date.parse("2026-11-06T06:00:00Z"), clock: CLOSED, calendar: NOV, lines: [thuLine] });
  assert.equal(same.end_basis, "now");
});

test("todayLineRecord: a stored daily line's day, last_equity and write time; anything else is null", () => {
  const line = { version: 1, day: "2026-11-05", written_at: "2026-11-06T01:12:26.339Z", account_vs_spy: { day: { account_pct: -4.07, last_equity: 93623.46, equity: 89814.42 } } };
  assert.deepEqual(todayLineRecord(line), { day: "2026-11-05", lastEquity: 93623.46, writtenAt: "2026-11-06T01:12:26.339Z" });
  assert.deepEqual(todayLineRecord({ ...line, account_vs_spy: { day: { last_equity: null } } }), { day: "2026-11-05", lastEquity: null, writtenAt: "2026-11-06T01:12:26.339Z" });
  assert.equal(todayLineRecord({ ...line, account_vs_spy: { day: { last_equity: 0 } } })?.lastEquity, null);
  assert.equal(todayLineRecord({ ...line, account_vs_spy: undefined })?.lastEquity, null);
  for (const bad of [null, undefined, "x", 3, [], { day: "Nov 5" }, { day: 20261105 }]) assert.equal(todayLineRecord(bad), null, JSON.stringify(bad));
});

test("wiring ratchet: the tile's session reads the shared broker clock cache and the stored daily lines — neither awaited, the lines re-read at most once a minute", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const helperStart = bot.indexOf("function todayFor(");
  const helper = bot.slice(helperStart, bot.indexOf("\n  }\n", helperStart));
  assert.match(helper, /nowMs,\s*clock: todayClock\(nowMs\), calendar: \{ holidays: marketHolidays\(\), halfDays: marketHalfDays\(\) \}, lines: todayLines\(nowMs\) \}\)/);
  const clockStart = bot.indexOf("function todayClock(");
  assert.ok(clockStart > 0 && clockStart < helperStart, "todayClock sits before todayFor");
  const clockFn = bot.slice(clockStart, bot.indexOf("\n  }\n", clockStart));
  assert.match(clockFn, /parseClockReading\(posKillClockCache\.clock, posKillClockCache\.at\)/, "the clock POS-KILL and the order gate share");
  assert.match(clockFn, /clockStateFrom\(live, todayClockLastGood, nowMs\)/);
  assert.doesNotMatch(clockFn, /await|posKillReadMarketOpen\(/, "read, never fetched from the account poll");
  const linesStart = bot.indexOf("function todayLines(");
  const linesFn = bot.slice(linesStart, bot.indexOf("\n  }\n", linesStart));
  assert.match(linesFn, /TODAY_LINES_TTL_MS/);
  assert.match(linesFn, /loadDailyLines\(dailyLineDir\(process\.env\)\)\.slice\(-TODAY_LINES_KEEP\)\.map\(todayLineRecord\)/, "the daily line's own reader, never a second store");
});
