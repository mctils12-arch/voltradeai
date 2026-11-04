import { test } from "node:test";
import assert from "node:assert/strict";
import {
  pct, usd, bandPct, ageText, callState, skillText, calChart, dotRadius, forecastKpis, shortDate, type ForecastCall, type CalBin,
  dailyKpis, signedPct, parseHorizon, liveText, dailyRangeNote, dailyLiveNote, timingNote, type DailyCall, type DailyAccuracy,
} from "./forecast";

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

// ── daily horizon (Part C, 2026-11-04) ──────────────────────────────────────
const dcall = (over: Partial<DailyCall> = {}): DailyCall => ({
  id: "D:2026-11-04:SPY", horizon: "daily", session: "2026-11-04", symbol: "SPY", group: "index", made_at: "2026-11-04T01:15:00Z",
  timing: "pre_open", on_time: true, base_date: "2026-11-03", base_close: 754.35, spot: 754.35, horizon_end: "2026-11-04",
  p_up: 0.545, p_base: 0.545, direction: "up", p_call: 0.545, base_rate: 0.545, direction_model: "base-rate-v1",
  remaining_fraction: 1, q: 0.0081, q_1d: 0.0081, q_clim: 0.0072, range_lo: 754.35 * Math.exp(-0.0081), range_hi: 754.35 * Math.exp(0.0081),
  band_coverage: 0.6827, range_source: "har+vix", range_model_version: "fc-range-har-v1", iv: 0.2083, iv_day: "2026-11-03",
  model_version: "fc-daily-v1", score: null, live: null, ...over,
});
const dacc = (over: Partial<DailyAccuracy> = {}): DailyAccuracy => ({
  n: 0, thin: true, hits: 0, hit_rate: null, up_rate: null, brier: null, brier_base_rate: null, brier_coinflip: 0.25,
  bss_vs_base_rate: null, bss_vs_coinflip: null, in_range: 0, in_range_rate: null, in_range_target: 0.6827,
  interval_score_pct: null, interval_score_clim_pct: null, range_skill_vs_constant: null, range_skill_n: 0, width_pct: null, ...over,
});

test("signedPct and parseHorizon", () => {
  assert.equal(signedPct(0.0123), "+1.2%");
  assert.equal(signedPct(-0.004, 2), "-0.40%");
  assert.equal(signedPct(null), "—");
  assert.equal(parseHorizon("weekly"), "weekly");
  assert.equal(parseHorizon("daily"), "daily");
  assert.equal(parseHorizon(null), "daily");
  assert.equal(parseHorizon("garbage"), "daily");
});

test("callState reads a daily call's score the same way", () => {
  assert.equal(callState(dcall()), "open");
  assert.equal(callState(dcall({ score: { id: "x", status: "resolved", hit: 1 } })), "hit");
});

test("dailyKpis: honest defaults before any resolution; skill vs coin flip once resolved; no tone while thin", () => {
  const latest = { session: "2026-11-04", made_at: "2026-11-04T01:15:00Z", base_date: "2026-11-03", timing: "pre_open" as const, on_time: true,
    model_version: "fc-daily-v1", range_model_version: "fc-range-har-v1", direction_model: "base-rate-v1",
    base_rate: { p: 0.545, n: 11340, first: "2023-10-26", last: "2026-11-03", basket: [] }, vix: { value: 20.83, day: "2026-11-03" }, vix_used: true,
    iv_day: "2026-11-03", n_forecasts: 3, skipped: [], candidates_available: true, candidates_scan_at: null,
    range_sources: { "har+vix": 1, "har+chain_iv30": 1, har: 1 },
    forecasts: [dcall(), dcall({ symbol: "XLE", group: "position", range_source: "har" }), dcall({ symbol: "PFE", group: "candidate", timing: "intraday", on_time: false })] };
  const study = { range: { doc: "r", holdout: "2019–2026", spy_coverage: 0.69, spy_skill_vs_constant: 0.21, spy_skill_vs_current: 0.05, iv_names_coverage: 0.68, iv_names_skill_vs_constant: 0.04, names_coverage: 0.66, names_skill_vs_constant: 0.12 },
    direction: { doc: "d", holdout: "2019–2026", tests: 14, passed: 0, spy_base_rate: 0.5508, spy_brier_base_rate: 0.24767, spy_skill_vs_coinflip: 0.0093 } };
  const d = { model_version: "fc-daily-v1", range_model_version: "fc-range-har-v1", direction_model: "base-rate-v1", direction_features: [], horizon: "next session close",
    band_coverage: 0.6827, latest, live: null, accuracy: { ...dacc(), by_timing: {}, by_group: {}, by_range_source: {} }, calibration: { edges: [], bins: [] }, sessions: [], open: 3, study, method: "", caveats: [] };
  const k = dailyKpis(d);
  assert.equal(k.length, 5);
  assert.equal(k[0].label, "Calls for Nov 4");
  assert.equal(k[0].line, "1 index · 1 positions · 1 candidates · 1 intraday");
  assert.deepEqual(k.slice(1, 4).map((t) => t.value), ["—", "—", "—"]);
  assert.match(k[1].line, /P\(up\) = base rate 54\.5% · no feature beat it/);
  assert.match(k[3].line, /research: \+21\.0%/);
  const a = dacc({ n: 40, thin: false, hits: 22, hit_rate: 0.55, up_rate: 0.55, brier: 0.2475, brier_base_rate: 0.2475, bss_vs_base_rate: 0, bss_vs_coinflip: 0.01,
    in_range: 28, in_range_rate: 0.7, interval_score_pct: 2.1, interval_score_clim_pct: 2.6, range_skill_vs_constant: 0.1923, range_skill_n: 40 });
  const k2 = dailyKpis({ ...d, accuracy: { ...a, by_timing: {}, by_group: {}, by_range_source: {} } });
  assert.deepEqual(k2.slice(1, 4).map((t) => t.value), ["+1.0%", "70.0%", "+19.2%"]);
  assert.equal(k2[1].tone, "g");
  assert.equal(k2[3].line, "interval score 2.10% vs 2.60% constant");
  assert.match(k2[1].line, /Brier 0\.2475 · base rate 0\.2475 · coin flip 0\.25$/);
  const thin = dailyKpis({ ...d, accuracy: { ...a, thin: true, by_timing: {}, by_group: {}, by_range_source: {} } });
  assert.equal(thin[1].tone, undefined);
  assert.match(thin[2].line, /thin$/);
  assert.deepEqual(dailyKpis(null).map((t) => t.value), ["—", "—", "—", "—", "—"]);
});

test("liveText / notes: the live band says what it is and how old it is", () => {
  const t = Date.parse("2026-11-04T16:10:00Z");
  const lv = { price: 757.1, price_day: "2026-11-04", at: "2026-11-04T16:05:00Z", remaining_fraction: 0.42, q: 0.0052, lo: 753.17, hi: 761.05, vs_spot_pct: 0.366 };
  assert.equal(liveText(dcall(), lv, t), "live $757.10 (+0.37%) · band for the Nov 4 close $753.17–$761.05 · 42% of the day's variance left · 5m ago");
  assert.match(dailyRangeNote({ range_sources: { "har+vix": 1, "har+chain_iv30": 2, har: 5 }, vix: { value: 20.83, day: "2026-11-03" }, vix_used: true, iv_day: "2026-11-03", base_date: "2026-11-03" }),
    /SPY blends HAR-RV with VIX \(20\.83 at the 2026-11-03 close\); 2 names blend with chain IV30 \(2026-11-03\); 5 use HAR-RV alone\.$/);
  assert.match(dailyRangeNote({ range_sources: { har: 3 }, vix: null, vix_used: false, iv_day: null, base_date: "2026-11-03" }), /SPY has no VIX for its base close \(HAR only\)/);
  assert.match(dailyLiveNote({ live: { session: "2026-11-04", made_at: "2026-11-04T16:05:00Z", et_minutes: 665 }, latest: null }, t), /display only, never scored.*5m ago/);
  assert.match(timingNote({ pre_open: dacc({ n: 3, in_range: 2, in_range_rate: 0.6667, brier: 0.24 }), intraday: dacc() }), /^Pre-open calls: 3 resolved, 67% inside, Brier 0\.24 · intraday calls .*: none resolved\.$/);
});
