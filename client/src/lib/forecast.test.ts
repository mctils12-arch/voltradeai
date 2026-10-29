import { test } from "node:test";
import assert from "node:assert/strict";
import { pct, usd, bandPct, ageText, callState, skillText, calChart, dotRadius, forecastKpis, shortDate, type ForecastCall, type CalBin } from "./forecast";

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

test("forecastKpis: five tiles from the report; dashes with no batch; tones only off the numbers shown", () => {
  const acc = { n: 42, thin: false, hits: 24, hit_rate: 0.5714, brier: 0.2412, brier_coinflip: 0.25, brier_skill_vs_coinflip: 0.0352, in_range: 29, in_range_rate: 0.6905, in_range_target: 0.6827, up_rate: 0.5476 };
  const latest = { week_key: "2026-W44", made_at: "2026-10-23T20:21:00Z", base_date: "2026-10-23", horizon_end: "2026-10-30", on_time: true, model_version: "fc-logit-v1",
    model: { coef: [], features: [], n_samples: 1, first_date: "", last_date: "", validation: { split_date: "", n_train: 1, n_test: 1, train_base_rate: 0.5, test_base_rate: 0.5, brier: 0.25, brier_climatology: 0.25, brier_skill: 0, log_loss: 0.69, hit_rate: 0.5, calibration: [] } },
    train_tickers: [], n_forecasts: 3, skipped: [], candidates_available: true, candidates_scan_at: null, iv_day: null, iv_names: 0,
    forecasts: [call({ group: "index" }), call({ symbol: "XLE", group: "position" }), call({ symbol: "NVDA", group: "candidate" })] };
  const rep = { generated_at: "2026-10-27T12:00:00Z", model_version: "fc-logit-v1", experimental: true, shadow: true, horizon_days: 5, band_coverage: 0.6827, latest,
    accuracy: { ...acc, on_time: acc, by_group: {}, by_vol_source: {} }, calibration: { edges: [], bins: [] }, weeks: [], open: 3, method: "", caveats: [] };
  const k = forecastKpis(rep);
  assert.equal(k.length, 5);
  assert.deepEqual(k.map((t) => t.value), ["3", "57.1%", "+3.5%", "69.0%", "Oct 30"]);
  assert.equal(k[0].line, "1 index · 1 positions · 1 candidates");
  assert.equal(k[1].tone, "g"); // 57.1% hits above the 54.8% base rate
  assert.equal(k[2].tone, "g");
  assert.match(k[4].line, /after the 2026-10-30 close · 3 open/);
  // thin samples carry no tone; no batch -> dashes everywhere
  const thin = forecastKpis({ ...rep, accuracy: { ...rep.accuracy, thin: true } });
  assert.equal(thin[1].tone, undefined); assert.match(thin[1].line, /thin$/);
  const none = forecastKpis({ ...rep, latest: null, open: 0, accuracy: { ...rep.accuracy, n: 0 } });
  assert.deepEqual(none.map((t) => t.value), ["—", "—", "—", "—", "—"]);
  assert.deepEqual(forecastKpis(undefined).map((t) => t.value), ["—", "—", "—", "—", "—"]);
  assert.equal(shortDate("2026-01-05"), "Jan 5"); assert.equal(shortDate("bad"), "—");
});
