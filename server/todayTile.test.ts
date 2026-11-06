// todayTile — the Bot page's "Today" tile (B1, 2026-11-03): equity now vs the
// prior close in $ and %, SPY's move over the same hours, the difference in
// points. Run: npx tsx --test server/todayTile.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { todayTile, spyToday, TODAY_SOURCE } from "./todayTile";

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

test("the day is the ET session day: after the close the completed bar is still today's; past midnight ET it is yesterday's", () => {
  const evening = Date.parse("2026-11-04T02:00:00Z"); // 21:00 ET on the 3rd
  const e = todayTile({ equity: "100484.56", lastEquity: "100172.07", spyBars: bars, spyFetchedAt: evening - 60_000, nowMs: evening });
  assert.equal(e.day, "2026-11-03");
  assert.equal(e.spy_pct_today, 0.25);
  const smallHours = Date.parse("2026-11-04T06:00:00Z"); // 01:00 ET on the 4th
  const s = todayTile({ equity: "100484.56", lastEquity: "100172.07", spyBars: bars, spyFetchedAt: smallHours - 60_000, nowMs: smallHours });
  assert.equal(s.day, "2026-11-04");
  assert.equal(s.spy_pct_today, null);
  assert.equal(s.spy_reason, "no SPY bar for today yet (last 2026-11-03)");
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
