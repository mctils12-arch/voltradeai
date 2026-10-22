import { test } from "node:test";
import assert from "node:assert/strict";
import {
  type Bar, capBars, fitViewport, clampViewport, zoomViewport, panViewport, followViewport, priceRange, niceStep, priceTicks, decimalsFor, priceDecimals,
  xOf, iOf, yOf, priceOfY, timeTicks, indexAtX, etParts, MAX_BARS, MIN_VISIBLE_BARS, RIGHT_PAD_FRAC, timeframeSpec, fmtPrice, fmtBarTimeShort,
} from "./scales";

const T0 = Date.parse("2026-10-19T13:30:00Z") / 1000; // 09:30 ET Monday
const mk = (n: number, stepS: number, start = T0): Bar[] => Array.from({ length: n }, (_, k) => ({ time: start + k * stepS, open: 100 + k, high: 101 + k, low: 99 + k, close: 100.5 + k, volume: 1000 + k }));

test("capBars drops malformed bars, sorts by time and keeps the newest MAX_BARS", () => {
  const bars = mk(MAX_BARS + 50, 60).reverse();
  bars.push({ time: NaN, open: 1, high: 1, low: 1, close: 1, volume: 1 });
  const out = capBars(bars);
  assert.equal(out.length, MAX_BARS);
  assert.ok(out.every((b, k) => k === 0 || b.time > out[k - 1].time));
  assert.equal(out[out.length - 1].time, T0 + (MAX_BARS + 49) * 60);
});

test("the fitted view shows the last `show` bars with right-hand padding; fewer bars than asked shows them all", () => {
  const vp = fitViewport(300, 120);
  assert.equal(vp.from, 180);
  assert.ok(Math.abs(vp.to - (300 + 120 * RIGHT_PAD_FRAC)) < 1e-9);
  const small = fitViewport(5, 120);
  assert.equal(small.to - small.from, MIN_VISIBLE_BARS * (1 + RIGHT_PAD_FRAC));
});

test("zoom keeps the anchor bar under the pointer; pan and zoom never leave fewer than MIN_VISIBLE_BARS in view", () => {
  const n = 300;
  const vp = fitViewport(n, 120);
  const w = 600;
  const anchorX = 450;
  const anchor = iOf(anchorX, vp, w);
  const z = zoomViewport(vp, 0.5, anchor, n);
  assert.ok(Math.abs(xOf(anchor, z, w) - anchorX) < 1e-6);
  assert.ok(Math.abs((z.to - z.from) - (vp.to - vp.from) * 0.5) < 1e-9);
  const tiny = zoomViewport(vp, 1e-6, anchor, n);
  assert.equal(tiny.to - tiny.from, MIN_VISIBLE_BARS);
  const far = panViewport(vp, 10_000, n);
  assert.ok(far.from <= n - MIN_VISIBLE_BARS + 1e-9);
  const back = panViewport(vp, -10_000, n);
  assert.ok(back.to >= MIN_VISIBLE_BARS - 1e-9);
  assert.deepEqual(clampViewport({ from: 10, to: 50 }, n), { from: 10, to: 50 });
});

test("a view on the live edge follows new bars; a view panned into history stays", () => {
  const vp = fitViewport(100, 50);
  const f = followViewport(vp, 100, 103);
  assert.equal(f.from, vp.from + 3);
  const hist = panViewport(vp, -40, 100);
  assert.deepEqual(followViewport(hist, 100, 103), hist);
  assert.deepEqual(followViewport(vp, 100, 100), vp);
});

test("priceRange spans the visible bars plus fitted levels, padded; degenerate inputs still give a usable range", () => {
  const bars = mk(50, 60);
  const vp = { from: 40, to: 52 };
  const r = priceRange(bars, vp);
  assert.ok(r.lo < 139 && r.hi > 150); // bars 40..49: low 139, high 150
  const withLevel = priceRange(bars, vp, [120]);
  assert.ok(withLevel.lo < 120);
  assert.deepEqual(priceRange([], vp), { lo: 0, hi: 1 });
  const flat = priceRange([{ time: 1, open: 10, high: 10, low: 10, close: 10, volume: 1 }], { from: 0, to: 1 });
  assert.ok(flat.lo < 10 && flat.hi > 10);
  assert.ok(Math.abs(priceOfY(yOf(145, r, 300), r, 300) - 145) < 1e-9);
});

test("nice steps and ticks: 1 / 2 / 2.5 / 5 × 10^k, labels with the decimals the step needs", () => {
  assert.equal(niceStep(100, 10), 10);
  assert.equal(niceStep(7, 10), 1);
  assert.equal(niceStep(0.3, 10), 0.05);
  assert.equal(niceStep(12, 4), 5);
  const ticks = priceTicks({ lo: 99.1, hi: 101.9 }, 300);
  assert.ok(ticks.length >= 4 && ticks.every((t) => t >= 99.1 && t <= 101.9));
  assert.equal(decimalsFor(0.25), 2);
  assert.equal(decimalsFor(5), 0);
  assert.equal(decimalsFor(0.1), 1);
  assert.equal(priceDecimals({ lo: 99, hi: 101 }, 300), 2);
  assert.equal(priceDecimals({ lo: 20_000, hi: 21_000 }, 300), 0);
  assert.equal(fmtPrice(1234.5, 2), "1,234.50");
});

test("ET time parts and ticks: intraday ticks land on round ET minutes with the day change as a major tick; daily ticks mark the month", () => {
  const p = etParts(T0);
  assert.equal(p.hm, "09:30");
  assert.equal(p.date, "2026-10-19");
  assert.equal(p.weekdayShort, "Mon");
  const hourly = mk(14, 3600); // 09:30 .. 22:30 ET Monday
  const t = timeTicks(hourly, { from: 0, to: 14 }, 1400, "1Hour");
  assert.ok(t.length > 1);
  // the series' first bar names its day; every later tick is a round ET minute
  assert.deepEqual(t.filter((x) => x.major).map((x) => [x.i, x.label]), [[0, "Mon 19"]]);
  assert.ok(t.filter((x) => !x.major).every((x) => /^\d\d:\d\d$/.test(x.label)));
  // :30-aligned hourly bars still get a tick past every hour boundary
  assert.ok(t.filter((x) => !x.major).length >= 10);
  const twoDays = [...mk(7, 3600), ...mk(7, 3600, T0 + 86_400)];
  const t2 = timeTicks(twoDays, { from: 0, to: 14 }, 1400, "1Hour");
  assert.ok(t2.some((x) => x.major && x.label === "Tue 20"));
  const daily = mk(40, 86_400, Date.parse("2026-09-10T20:00:00Z") / 1000);
  const td = timeTicks(daily, { from: 0, to: 40 }, 1200, "1Day");
  const majors = td.filter((x) => x.major).map((x) => x.label);
  assert.deepEqual(majors, ["Sep 2026", "Oct"]);
  // narrow view: ticks stay at least ~72 px apart
  const dense = timeTicks(hourly, { from: 0, to: 14 }, 200, "1Hour");
  for (let k = 1; k < dense.length; k++) assert.ok(xOf(dense[k].i, { from: 0, to: 14 }, 200) - xOf(dense[k - 1].i, { from: 0, to: 14 }, 200) >= 70 - 1e-9);
});

test("indexAtX maps a pixel to its bar, null past either end; timeframe specs resolve with a default", () => {
  const vp = { from: 0, to: 10 };
  assert.equal(indexAtX(5, vp, 100, 10), 0);
  assert.equal(indexAtX(95, vp, 100, 10), 9);
  assert.equal(indexAtX(95, vp, 100, 5), null);
  assert.equal(timeframeSpec("bogus").tf, "1Hour");
  assert.equal(timeframeSpec("1Day").minutes, 1440);
});

test("the short bar time fits the readout: the clock intraday, the date on the daily chart", () => {
  const t = Date.UTC(2026, 9, 21, 19, 55) / 1000; // 15:55 ET on a Wednesday
  assert.equal(fmtBarTimeShort(t, "1Hour"), "15:55 ET");
  assert.equal(fmtBarTimeShort(t, "1Day"), "Oct 21");
});

