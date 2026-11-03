import { test } from "node:test";
import assert from "node:assert/strict";
import { todayView, fmtSignedMoney, shortReason, SPY_STALE_MS } from "./today";
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
  assert.match(v.title, /dividend-adjusted/);
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
