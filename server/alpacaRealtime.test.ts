// alpacaRealtime.test.ts — real-time DISPLAY prices under the delayed feed
// (2026-11-12, post-open check of the stale-ticker fix: every price on the
// site was 15 minutes old because this account's plan rejects recent SIP data).
// The watchlist route keeps the delayed consolidated snapshot as its base and
// takes the IEX tape's trade only when it is a newer regular-hours print.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { inRegularHoursMs, mergeRealtimeTrades, REALTIME_FEED, tradeMs } from "./alpacaFeed";

// 2026-11-12 (a Thursday) 10:09 ET = 15:09Z; the delayed tape's last trade 09:54 ET
const DELAYED = {
  SPY: {
    latestTrade: { p: 758.21, t: "2026-11-12T14:54:10.929790855Z" },
    dailyBar: { t: "2026-11-12T05:00:00Z", o: 753.98, h: 758.89, l: 753.67, c: 758.38, v: 5728453 },
    prevDailyBar: { t: "2026-11-11T05:00:00Z", c: 759.81 },
  },
};
const IEX = {
  SPY: {
    latestTrade: { p: 759.4, t: "2026-11-12T15:09:02.123456789Z" },
    dailyBar: { t: "2026-11-12T05:00:00Z", o: 754.0, h: 759.5, l: 753.7, c: 759.4, v: 81234 },
    prevDailyBar: { t: "2026-11-11T05:00:00Z", c: 759.8 },
  },
};
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

test("trade times parse with nanoseconds; regular hours are 09:30-16:00 ET on weekdays", () => {
  assert.equal(tradeMs("2026-11-12T15:09:02.123456789Z"), Date.parse("2026-11-12T15:09:02.123Z"));
  assert.ok(Number.isNaN(tradeMs("")) && Number.isNaN(tradeMs(undefined)) && Number.isNaN(tradeMs("not a time")));
  assert.equal(inRegularHoursMs(Date.parse("2026-11-12T14:30:00Z")), true, "09:30 ET");
  assert.equal(inRegularHoursMs(Date.parse("2026-11-12T14:29:59Z")), false, "09:29:59 ET pre-market");
  assert.equal(inRegularHoursMs(Date.parse("2026-11-12T20:59:59Z")), true, "15:59:59 ET");
  assert.equal(inRegularHoursMs(Date.parse("2026-11-12T21:00:00Z")), false, "16:00 ET after hours");
  assert.equal(inRegularHoursMs(Date.parse("2026-11-14T15:00:00Z")), false, "Saturday");
  assert.equal(inRegularHoursMs(NaN), false);
  assert.equal(REALTIME_FEED, "iex");
});

test("same session: the newer IEX trade is the price, the whole tape's volume stays", () => {
  const out = mergeRealtimeTrades(clone(DELAYED), clone(IEX)).SPY;
  assert.equal(out.priceFeed, "iex");
  assert.equal(out.latestTrade.p, 759.4);
  assert.equal(out.dailyBar.c, 759.4, "the watchlist reads dailyBar.c");
  assert.equal(out.dailyBar.h, 759.4, "range widened to include the price");
  assert.equal(out.dailyBar.l, 753.67);
  assert.equal(out.dailyBar.v, 5728453, "volume is the consolidated tape's, never IEX's");
  assert.equal(out.prevDailyBar.c, 759.81, "the change is measured from the consolidated close");
});

test("an older, pre-market, after-hours or empty IEX print never replaces the delayed price", () => {
  type Snaps = { SPY: { latestTrade?: { p: number; t: string } } };
  const cases: Array<[string, (r: Snaps) => void]> = [
    ["older", (r) => { r.SPY.latestTrade!.t = "2026-11-12T14:50:00Z"; }],
    ["pre-market", (r) => { r.SPY.latestTrade!.t = "2026-11-12T14:20:00Z"; }],
    ["after hours", (r) => { r.SPY.latestTrade!.t = "2026-11-12T21:30:00Z"; }],
    ["no price", (r) => { r.SPY.latestTrade!.p = 0; }],
    ["no trade", (r) => { delete r.SPY.latestTrade; }],
  ];
  for (const [name, mutate] of cases) {
    const rt = clone(IEX); mutate(rt);
    const delayedLate = clone(DELAYED);
    if (name === "after hours") delayedLate.SPY.latestTrade.t = "2026-11-12T20:59:00Z";
    const out = mergeRealtimeTrades(delayedLate, rt).SPY;
    assert.equal(out.priceFeed, "delayed_sip", name);
    assert.equal(out.dailyBar.c, 758.38, name);
  }
  assert.deepEqual(Object.keys(mergeRealtimeTrades(clone(DELAYED), null)), ["SPY"], "no IEX body: every symbol kept");
  assert.equal(mergeRealtimeTrades(clone(DELAYED), {}).SPY.priceFeed, "delayed_sip");
});

test("the open's first minutes: IEX on today's session, the delayed tape still on yesterday's", () => {
  const delayed = { SPY: {
    latestTrade: { p: 759.81, t: "2026-11-11T20:59:59Z" },
    dailyBar: { t: "2026-11-11T05:00:00Z", o: 756.8, h: 762.51, l: 754.9, c: 759.81, v: 53905000 },
    prevDailyBar: { t: "2026-11-10T05:00:00Z", c: 757.0 },
  } };
  const rt = { SPY: {
    latestTrade: { p: 754.1, t: "2026-11-12T14:35:00Z" },
    dailyBar: { t: "2026-11-12T05:00:00Z", o: 753.98, h: 754.5, l: 753.67, c: 754.0, v: 12000 },
  } };
  const out = mergeRealtimeTrades(delayed, rt).SPY;
  assert.equal(out.priceFeed, "iex");
  assert.equal(out.dailyBar.t, "2026-11-12T05:00:00Z");
  assert.equal(out.dailyBar.c, 754.1);
  assert.equal(out.prevDailyBar.c, 759.81, "yesterday's consolidated close is the change's base");
});

test("the watchlist route reads the IEX tape alongside the delayed one and merges; the scanner does not", () => {
  const src = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "routes.ts"), "utf8");
  const start = src.indexOf('app.get("/api/market/snapshots"');
  const end = src.indexOf("app.get(", start + 10);
  const route = src.slice(start, end);
  assert.match(route, /ALPACA_DATA_FEED === "delayed_sip" \? fetch\(snapUrl\(REALTIME_FEED\)/);
  assert.match(route, /res\.json\(rtBody \? mergeRealtimeTrades\(body, rtBody\) : body\)/);
  assert.match(route, /\.catch\(\(\) => null\)/, "an IEX failure never fails the route");
  const others = src.slice(0, start) + src.slice(end);
  assert.equal((others.match(/mergeRealtimeTrades\(/g) || []).length, 0, "display route only: the scanner keeps the full tape");
});
