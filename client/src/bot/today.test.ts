import { test } from "node:test";
import assert from "node:assert/strict";
import { todayView, fmtSignedMoney, shortReason, sessionDayWords, weekdayOf, SPY_STALE_MS } from "./today";
import { MINUS } from "./format";
import type { TodayTile } from "./useBotData";

// 12:06 ET on the session day; the SPY bars were fetched four minutes earlier
const NOW = Date.parse("2026-11-03T17:06:00Z");
const T: TodayTile = {
  day: "2026-11-03", equity_now: 100172.07, equity_prev_close: 100423.31, pnl_usd: -251.24, pnl_pct: -0.25, pnl_reason: null,
  spy_pct_today: 0.25, spy_asof: "2026-11-03T17:02:00Z", spy_bar_date: "2026-11-03", spy_reason: null, excess_pct: -0.5,
  asof: "2026-11-03T17:06:00Z", source: "Alpaca paper account: equity now vs last_equity · SPY: today's daily bar vs the prior close",
};
const noLeak = (v: object) => assert.ok(!/NaN|undefined|null/.test(JSON.stringify(v)), `a NaN / undefined / null leaked into the words: ${JSON.stringify(v)}`);

test("a down day against an up SPY: the change in $ and %, SPY with its own clock, the headline difference in points, the prior close → now path", () => {
  const v = todayView(T, NOW, true);
  assert.equal(v.headline, `${MINUS}0.50 pp`);
  assert.equal(v.headlineCls, "vt-bot-r");
  assert.equal(v.change, `${MINUS}$251 · ${MINUS}0.25%`);
  assert.equal(v.changeCls, "vt-bot-r");
  assert.equal(v.spy, "SPY +0.25%");
  assert.equal(v.spyCls, "vt-bot-g");
  assert.equal(v.spyNote, "12:02 ET");
  assert.equal(v.spyStale, false);
  assert.equal(v.path, "prior $100,423 → $100,172");
  assert.equal(v.pathValue, "$100,423 → $100,172");
  assert.match(v.title, /equity \$100,172\.07 now vs \$100,423\.31 at the prior close/);
  assert.match(v.title, /last_equity/);
  // SPY's basis (2026-11-05): price only, no dividends, like the paper account (was /dividend-adjusted/)
  assert.match(v.title, /SPY \+0\.25% today: its daily bar against the prior close, price only, no dividends \(Alpaca bars, feed iex; fetched 12:02 ET\)/);
  assert.doesNotMatch(v.title, /dividend-adjusted|total return/);
  assert.match(v.title, /read 12:06:00 ET/);
  assert.match(v.title, /headline −0\.50 pp = account − SPY in percentage points/);
  noLeak(v);
});

test("an up day beats SPY: plus signs, green; a flat day is neither", () => {
  const up = todayView({ ...T, equity_now: 100735.8, pnl_usd: 312.49, pnl_pct: 0.31, excess_pct: 0.06 }, NOW, true);
  assert.equal(up.headline, "+0.06 pp");
  assert.equal(up.headlineCls, "vt-bot-g");
  assert.equal(up.change, "+$312 · +0.31%");
  assert.equal(up.changeCls, "vt-bot-g");
  const flat = todayView({ ...T, pnl_usd: 0, pnl_pct: 0, excess_pct: -0.25 }, NOW, true);
  assert.equal(flat.change, "$0 · 0.00%");
  assert.equal(flat.changeCls, "vt-bot-t1");
  assert.equal(flat.headline, `${MINUS}0.25 pp`);
});

test("SPY missing: the headline is a dash and the SPY line carries the server's reason; the account side still reads", () => {
  const v = todayView({ ...T, spy_pct_today: null, spy_asof: "2026-11-03T17:02:00Z", spy_bar_date: "2026-11-02", spy_reason: "no SPY bar for today yet (last 2026-11-02)", excess_pct: null }, NOW, true);
  assert.equal(v.headline, "—");
  assert.equal(v.headlineCls, "vt-bot-t3");
  assert.equal(v.spy, "SPY —");
  assert.equal(v.spyCls, "vt-bot-t3");
  assert.equal(v.spyNote, "no bar today", "the tile has room for a few words; the sentence is in the hover");
  assert.match(v.title, /no SPY bar for today yet \(last 2026-11-02\)/);
  assert.equal(v.change, `${MINUS}$251 · ${MINUS}0.25%`);
  assert.match(v.title, /headline — needs both sides/);
  assert.match(v.title, /never an older day's move/);
  assert.match(v.title, /\(daily bars, price only, no dividends; SPY's side stays empty until today's bar exists/, "the missing side names the same basis, in the same words the harness checks on both pages");
  noLeak(v);
  const cold = todayView({ ...T, spy_pct_today: null, spy_asof: null, spy_bar_date: null, spy_reason: "no SPY bars fetched yet", excess_pct: null }, NOW, true);
  assert.equal(cold.spyNote, "no bars yet");
  assert.match(cold.title, /no SPY bars fetched yet/);
  noLeak(cold);
});

test("the account read invalid: the change line says so, the headline is a dash, SPY still reads", () => {
  const v = todayView({ ...T, equity_prev_close: null, pnl_usd: null, pnl_pct: null, pnl_reason: "account read invalid (equity or prior close not a positive number)", excess_pct: null }, NOW, true);
  assert.equal(v.headline, "—");
  assert.equal(v.change, "acct — · read invalid");
  assert.match(v.title, /account read invalid \(equity or prior close not a positive number\)/);
  assert.equal(v.changeCls, "vt-bot-t3");
  assert.equal(v.spy, "SPY +0.25%");
  assert.equal(v.path, "prior — → $100,172");
  noLeak(v);
});

test("not loaded and not reported are different words; neither prints a number", () => {
  const loading = todayView(undefined, NOW, false);
  assert.equal(loading.headline, "—");
  assert.equal(loading.change, "account not loaded");
  assert.equal(loading.path, "prior close —");
  assert.equal(loading.pathValue, "close —");
  noLeak(loading);
  const old = todayView(undefined, NOW, true);
  assert.equal(old.change, "today not reported by this server");
  assert.equal(todayView(null, NOW, true).change, "today not reported by this server");
});

test("SPY's number carries its age: older than the cache plus its poll reads stale, the hover says so", () => {
  const fresh = todayView(T, Date.parse(T.spy_asof!) + 10 * 60_000, true);
  assert.equal(fresh.spyStale, false);
  const stale = todayView(T, Date.parse(T.spy_asof!) + SPY_STALE_MS + 60_000, true);
  assert.equal(stale.spyStale, true);
  assert.match(stale.title, /stale/);
  assert.equal(stale.spyNote, "12:02 ET", "the clock stays the fetch time, not a relative age that drifts");
});

test("shortReason: the server's sentences become the words the phone tile holds; an unknown reason passes through", () => {
  assert.equal(shortReason("no SPY bars fetched yet"), "no bars yet");
  assert.equal(shortReason("no SPY bar for today yet (last 2026-11-02)"), "no bar today");
  assert.equal(shortReason("SPY prior close missing"), "no prior close");
  assert.equal(shortReason("account read invalid (equity or prior close not a positive number)"), "read invalid");
  assert.equal(shortReason("something new"), "something new");
  assert.equal(shortReason(null), "not read");
});

test("fmtSignedMoney: a plus for gains, a true minus for losses, no sign at zero, a dash when unknown", () => {
  assert.equal(fmtSignedMoney(312.49), "+$312");
  assert.equal(fmtSignedMoney(-251.24), `${MINUS}$251`);
  assert.equal(fmtSignedMoney(0), "$0");
  assert.equal(fmtSignedMoney(1234567.8), "+$1,234,568");
  assert.equal(fmtSignedMoney(null), "—");
  assert.equal(fmtSignedMoney(NaN), "—");
});

// ── BETWEEN SESSIONS ([REPAIR] 2026-11-06): the server names the session the numbers belong to; the tile says it.
// The live numbers of 2026-11-05/06 (server/todayTile.test.ts has the same reads end to end).
const THU: TodayTile = {
  day: "2026-11-05", phase: "between", when: "pre-market", phase_source: "broker clock", prev_day: "2026-11-04",
  equity_now: 89657.32, equity_prev_close: 93623.46, prev_close_source: "daily_line", equity_end: 89888.9, end_basis: "close",
  pnl_usd: -3734.56, pnl_pct: -3.99, pnl_reason: null,
  spy_pct_today: -4.34, spy_asof: "2026-11-06T10:40:00.000Z", spy_bar_date: "2026-11-05", spy_reason: null, spy_final: true,
  excess_pct: 0.35, asof: "2026-11-06T10:52:00.000Z", source: "Between sessions, the last session …",
};
const AT = Date.parse("2026-11-06T10:52:00Z");

test("sessionDayWords / weekdayOf: an ET session date as the tile names it; anything unreadable is a dash, never Invalid Date", () => {
  assert.equal(sessionDayWords("2026-11-05"), "Thu Nov 5");
  assert.equal(sessionDayWords("2026-11-25"), "Wed Nov 25");
  assert.equal(sessionDayWords("2027-01-04"), "Mon Jan 4");
  assert.equal(weekdayOf("2026-11-06"), "Fri");
  for (const bad of [null, undefined, "", "Nov 5", "2026-13-01", "2026-11-5"]) {
    assert.equal(sessionDayWords(bad), "—", String(bad));
    assert.equal(weekdayOf(bad), "", String(bad));
  }
});

test("(a) during a session the eyebrow stays Today; the hover says what decided it", () => {
  const v = todayView({ ...T, phase: "session", when: "during the session", phase_source: "broker clock", prev_day: "2026-11-02", prev_close_source: "last_equity", equity_end: 100172.07, end_basis: "now", spy_final: false }, NOW, true);
  assert.equal(v.eyebrow, "Today");
  assert.equal(v.eyebrowWord, "");
  assert.equal(v.path, "prior $100,423 → $100,172", "unchanged in a session");
  assert.equal(v.pathFrom, "prior");
  assert.equal(v.pathTo, "");
  assert.deepEqual(v.pathSeg, ["$100,423", "$100,172"]);
  assert.equal(v.spyNote, "12:02 ET", "the bar in progress keeps its fetch clock");
  assert.match(v.title, /^Today \(2026-11-03 ET\): equity \$100,172\.07 now vs \$100,423\.31 at the prior close/);
  assert.match(v.title, /session: during the session, by the broker clock$/);
  noLeak(v);
});

test("(c) pre-market after Alpaca's roll: 'Thu Nov 5 session', close to close (Wed → Thu), SPY's final close, the drift since the close in the hover", () => {
  const v = todayView(THU, AT, true);
  assert.equal(v.eyebrow, "Thu Nov 5");
  assert.equal(v.eyebrowWord, " session", "the phone and the folded strip drop the word");
  assert.equal(v.headline, "+0.35 pp");
  assert.equal(v.headlineCls, "vt-bot-g");
  assert.equal(v.change, `${MINUS}$3,735 · ${MINUS}3.99%`);
  assert.equal(v.spy, `SPY ${MINUS}4.34%`);
  assert.equal(v.spyNote, "close");
  assert.equal(v.spyStale, false);
  assert.equal(v.path, "Wed $93,623 → Thu $89,889");
  assert.equal(v.pathValue, "$93,623 → $89,889");
  assert.equal(v.pathFrom, "Wed");
  assert.equal(v.pathTo, "Thu");
  assert.match(v.title, /^Thu Nov 5 session \(2026-11-05 ET, shown pre-market\): equity \$89,888\.90 at the Thu close \(Alpaca's last_equity: its snapshot of that close, rolled to it overnight\) vs \$93,623\.46 at the Wed close \(as the 2026-11-05 daily line recorded Alpaca's last_equity after that session\) — −\$3,735 · −3\.99%; equity now \$89,657\.32, −\$232 since that close \(after-hours and pre-market marks\)/);
  assert.match(v.title, /SPY −4\.34% on 2026-11-05: its daily bar against the prior close, price only, no dividends \(Alpaca bars, feed iex; fetched 05:40 ET, after the close: the final bar\)/);
  assert.match(v.title, /headline \+0\.35 pp = account − SPY in percentage points/);
  assert.match(v.title, /session: pre-market, by the broker clock$/);
  assert.doesNotMatch(v.title, /Today|today/, "nothing between sessions is called today");
  noLeak(v);
  // a final close never reads stale, however long ago it was fetched
  assert.equal(todayView(THU, AT + 20 * 3_600_000, true).spyStale, false);
});

test("(b)/(c) before Alpaca's roll: the change runs to equity now, named as such; an SPY bar fetched before the close is not final", () => {
  const v = todayView({ ...THU, when: "after the close", equity_now: 89800.22, prev_close_source: "last_equity", equity_end: 89800.22, end_basis: "now", pnl_usd: -3823.24, pnl_pct: -4.08, excess_pct: 0.26, asof: "2026-11-05T23:00:00.000Z" }, Date.parse("2026-11-05T23:00:00Z"), true);
  assert.equal(v.eyebrow, "Thu Nov 5");
  assert.equal(v.path, "Wed $93,623 → now $89,800");
  assert.equal(v.pathTo, "now");
  assert.match(v.title, /^Thu Nov 5 session \(2026-11-05 ET, shown after the close\): equity \$89,800\.22 now \(read 18:00:00 ET; after-hours marks included until Alpaca rolls last_equity to the session's close overnight\) vs \$93,623\.46 at the Wed close \(Alpaca last_equity, which it still holds\) — −\$3,823 · −4\.08%/);
  assert.doesNotMatch(v.title, /since that close/, "no since-the-close drift while the change runs to now");
  noLeak(v);
  const early = todayView({ ...THU, spy_final: false, spy_asof: "2026-11-05T20:50:00.000Z" }, Date.parse("2026-11-05T21:10:00Z"), true);
  assert.equal(early.spyNote, "15:50 ET");
  assert.match(early.title, /fetched 15:50 ET, before the close: not the final bar\)/);
  const late = todayView({ ...THU, spy_final: false, spy_asof: "2026-11-05T20:50:00.000Z" }, Date.parse("2026-11-05T20:50:00Z") + SPY_STALE_MS + 60_000, true);
  assert.equal(late.spyStale, true);
  assert.match(late.title, /not the final bar, older than the 30 min cache — stale/);
});

test("(c) no daily line for the session: no account number — the reason in a few words, the sentence in the hover; SPY still reads", () => {
  const v = todayView({ ...THU, equity_prev_close: null, prev_close_source: null, equity_end: null, end_basis: null, pnl_usd: null, pnl_pct: null, excess_pct: null,
    pnl_reason: "the close before the 2026-11-05 session is not recorded (no daily line for 2026-11-05), and Alpaca's last_equity may already be that session's own close" }, AT, true);
  assert.equal(v.change, "acct — · no daily line");
  assert.equal(v.headline, "—");
  assert.equal(v.path, "Wed — → —");
  assert.equal(v.spy, `SPY ${MINUS}4.34%`);
  assert.match(v.title, /the close before the 2026-11-05 session is not recorded \(no daily line for 2026-11-05\)/);
  noLeak(v);
  // and the session's bar not fetched yet
  const noBar = todayView({ ...THU, spy_pct_today: null, spy_final: false, spy_bar_date: "2026-11-04", spy_reason: "no SPY bar for 2026-11-05 yet (last 2026-11-04)", excess_pct: null }, AT, true);
  assert.equal(noBar.spyNote, "no bar yet");
  assert.match(noBar.title, /SPY — no SPY bar for 2026-11-05 yet \(last 2026-11-04\) \(daily bars, price only, no dividends; SPY's side stays empty until that session's bar exists/);
  noLeak(noBar);
});

test("(d) a weekend and a market holiday name the last session; the hover says when the read fell", () => {
  const sat = todayView({ ...THU, day: "2026-11-06", prev_day: "2026-11-05", when: "weekend", equity_now: 90420, equity_prev_close: 89888.9, equity_end: 90500, pnl_usd: 611.1, pnl_pct: 0.68, spy_pct_today: 0.5, excess_pct: 0.18 }, Date.parse("2026-11-07T17:00:00Z"), true);
  assert.equal(sat.eyebrow, "Fri Nov 6");
  assert.equal(sat.path, "Thu $89,889 → Fri $90,500");
  assert.equal(sat.change, "+$611 · +0.68%");
  assert.match(sat.title, /^Fri Nov 6 session \(2026-11-06 ET, shown weekend\)/);
  assert.match(sat.title, /equity now \$90,420\.00, −\$80 since that close/);
  noLeak(sat);
  const hol = todayView({ ...THU, day: "2026-11-25", prev_day: "2026-11-24", when: "market holiday" }, Date.parse("2026-11-26T17:00:00Z"), true);
  assert.equal(hol.eyebrow, "Wed Nov 25");
  assert.equal(hol.pathFrom, "Tue");
  assert.equal(hol.pathTo, "Wed");
  assert.match(hol.title, /shown market holiday/);
});

test("shortReason: the between-sessions reasons in the phone's few words", () => {
  assert.equal(shortReason("no SPY bar for 2026-11-05 yet (last 2026-11-04)"), "no bar yet");
  assert.equal(shortReason("the close before the 2026-11-05 session is not recorded (no daily line for 2026-11-05), and Alpaca's last_equity may already be that session's own close"), "no daily line");
});
