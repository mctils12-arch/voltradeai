import { test } from "node:test";
import assert from "node:assert/strict";
import { pct, usd, bandPct, ageText, callState, skillText, calChart, dotRadius, type ForecastCall, type CalBin } from "./forecast";

const call = (over: Partial<ForecastCall> = {}): ForecastCall => ({
  id: "2026-W44:SPY", week_key: "2026-W44", symbol: "SPY", group: "index", made_at: "2026-10-23T21:00:00Z",
  base_date: "2026-10-23", horizon_end: "2026-10-30", spot: 100, p_up: 0.56, direction: "up", p_call: 0.56,
  range_lo: 100 * Math.exp(-0.03), range_hi: 100 * Math.exp(0.03), band_coverage: 0.6827, sigma_h: 0.03,
  vol_source: "realized_20d", vol_annual: 0.2, model_version: "fc-logit-v1", score: null, ...over,
});

test("formatting: percents, dollars, missing values are a dash", () => {
  assert.equal(pct(0.5612, 1), "56.1%");
  assert.equal(pct(null), "—");
  assert.equal(pct(Number.NaN), "—");
  assert.equal(usd(1234.5), "$1,235");
  assert.equal(usd(12.3), "$12.30");
  assert.equal(usd(undefined), "—");
});

test("bandPct: the log-symmetric band reads as -2.96% / +3.05%", () => {
  const b = bandPct(call());
  assert.ok(Math.abs(b.down - (Math.exp(-0.03) - 1) * 100) < 1e-9);
  assert.ok(Math.abs(b.up - (Math.exp(0.03) - 1) * 100) < 1e-9);
});

test("ageText: seconds, minutes, hours, days; unparseable is a dash", () => {
  const t = Date.parse("2026-10-27T12:00:00Z");
  assert.equal(ageText("2026-10-27T11:59:30Z", t), "30s");
  assert.equal(ageText("2026-10-27T11:00:00Z", t), "60m");
  assert.equal(ageText("2026-10-27T02:00:00Z", t), "10h");
  assert.equal(ageText("2026-10-23T21:00:00Z", t), "4d");
  assert.equal(ageText(null, t), "—");
});

test("callState: open until scored; hit/miss from the score; no-data is its own state", () => {
  assert.equal(callState(call()), "open");
  assert.equal(callState(call({ score: { id: "x", status: "resolved", hit: 1 } })), "hit");
  assert.equal(callState(call({ score: { id: "x", status: "resolved", hit: 0 } })), "miss");
  assert.equal(callState(call({ score: { id: "x", status: "no_price_data" } })), "no-data");
});

test("skillText: sign and the no-skill band", () => {
  assert.equal(skillText(0.031), "+3.1% better");
  assert.equal(skillText(-0.02), "-2.0% worse");
  assert.equal(skillText(0.001), "+0.1% ≈ no skill");
  assert.equal(skillText(null), "—");
});

test("calChart: x zooms to the populated range but never narrower than 30–70%; y is 0–100%", () => {
  const bins = (ps: number[]): CalBin[] => ps.map((p) => ({ lo: p - 0.025, hi: p + 0.025, n: 5, mean_p: p, freq: 0.5 }));
  const g = calChart(560, 300, [bins([0.48, 0.55])]);
  assert.deepEqual(g.xDomain, [0.3, 0.7]);
  assert.deepEqual(g.yDomain, [0, 1]);
  assert.equal(g.x(0.3), g.pad.l);
  assert.equal(g.x(0.7), 560 - g.pad.r);
  assert.equal(g.y(0), 300 - g.pad.b);
  assert.equal(g.y(1), g.pad.t);
  const wide = calChart(560, 300, [bins([0.2, 0.81])]);
  assert.ok(wide.xDomain[0] <= 0.18 && wide.xDomain[1] >= 0.83);
  // the reference diagonal is y = x inside the x domain
  assert.ok(Math.abs(g.diagonal.y1 - g.y(0.3)) < 1e-9 && Math.abs(g.diagonal.y2 - g.y(0.7)) < 1e-9);
  assert.deepEqual(calChart(560, 300, [[]]).xDomain, [0.3, 0.7]);
});

test("dotRadius: area grows with n, 4–10 px, 0 for an empty bin", () => {
  assert.equal(dotRadius(0, 10), 0);
  assert.equal(dotRadius(10, 10), 10);
  assert.ok(dotRadius(1, 100) >= 4 && dotRadius(1, 100) < 5);
});
