"""walkforward.py — the long-term, point-in-time test of the stock score's
price inputs ([RULE-REVIEW] measurement code).

What these pin: every reconstructed input equals the LIVE computation
(analyze.py functions called directly; deep_score's inline ADX/OBV/momentum
blocks through a line-for-line copy whose source lines are pinned in
bot_engine.py); features never read a later bar; the point-in-time universe;
the cost model equals backtest_v2.liquidity_cost_pct; forward returns and the
delisting assumption; the metric math (Spearman, Newey-West, quintiles,
Bonferroni, Deflated Sharpe); regime labels equal backtest_v2.regime_series;
the end-to-end run on synthetic panels (a planted signal is found, pure noise
is not) and its output contract.
"""
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
import pytest

import analyze
import backtest_v2 as bt
import price_panel as pp
import walkforward as wf
import walkforward_synth as ws
from strategies import mean_reversion, momentum

HERE = os.path.dirname(os.path.abspath(__file__))


# ── fixtures ────────────────────────────────────────────────────────────────
def _series(n=520, seed=3, p0=40.0, vol=0.02, gaps=True, v0=3e6, raw_ratio=1.5, symbol="TST"):
    rng = random.Random(seed)
    days = ws.bdays("2016-01-04", "2018-12-31")
    if gaps:   # holidays + a 12-day halt
        days = [d for i, d in enumerate(days) if i % 37 != 5 and not (200 <= i < 208)]
    days = days[:n]
    rows = ws.path_rows(days, p0, 0.0004, vol, v0, rng, raw_ratio=raw_ratio)
    rows[30]["c"] = rows[29]["c"]  # a flat day (zero change) and a zero-volume day
    rows[31]["v"] = 0
    ln = ws.line(symbol, rows)
    return pp.series_from_lines([ln])[0], rows


def _spy(n=None, seed=11):
    rng = random.Random(seed)
    days = ws.bdays("2016-01-04", "2018-12-31")
    rows = ws.path_rows(days, 200.0, 0.0003, 0.01, 8e7, rng)
    s = pp.series_from_lines([ws.line("SPY", rows)])[0]
    return s


def _hist_df(rows, end_i, days_back):
    end = pp.day_num(rows[end_i]["d"])
    sub = [r for r in rows[:end_i + 1] if pp.day_num(r["d"]) >= end - days_back]
    return pd.DataFrame({"Open": [r["o"] for r in sub], "High": [r["h"] for r in sub], "Low": [r["l"] for r in sub],
                         "Close": [r["c"] for r in sub], "Volume": [r["v"] for r in sub]},
                        index=pd.to_datetime([r["d"] for r in sub]))


def _decoded_rows(s):
    return [{"d": s.dates[i], "o": float(s.o[i]), "h": float(s.h[i]), "l": float(s.l[i]), "c": float(s.c[i]), "v": int(s.v[i])}
            for i in range(len(s.c))]


# ── reference copy of bot_engine.deep_score's inline blocks ─────────────────
# Copied line for line from bot_engine.py (deep_score, "ADX (trend strength) +
# OBV", "Strategy module scores" 1 and 2). test_live_source_lines_still_match
# pins the source lines, so a live edit fails this file instead of silently
# staling the mirror.
def _live_deep_block(bars_data, change_pct):
    adx_value = None
    obv_signal = 0
    _deep_closes = []
    if len(bars_data) >= 14:
        highs = [b["h"] for b in bars_data]
        lows = [b["l"] for b in bars_data]
        closes = [b["c"] for b in bars_data]
        _deep_closes = closes
        volumes = [b.get("v", 0) for b in bars_data]
        plus_dm_list, minus_dm_list, tr_list = [], [], []
        for i in range(1, len(bars_data)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0)
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
        if len(tr_list) >= 14:
            atr14 = sum(tr_list[:14]) / 14
            plus_dm14 = sum(plus_dm_list[:14]) / 14
            minus_dm14 = sum(minus_dm_list[:14]) / 14
            for i in range(14, len(tr_list)):
                atr14 = (atr14 * 13 + tr_list[i]) / 14
                plus_dm14 = (plus_dm14 * 13 + plus_dm_list[i]) / 14
                minus_dm14 = (minus_dm14 * 13 + minus_dm_list[i]) / 14
            plus_di = (plus_dm14 / atr14 * 100) if atr14 > 0 else 0
            minus_di = (minus_dm14 / atr14 * 100) if atr14 > 0 else 0
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
            adx_value = round(dx, 1)
        obv = 0
        obv_values = [0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
            obv_values.append(obv)
        if len(obv_values) >= 10:
            recent_obv = sum(obv_values[-5:]) / 5
            prev_obv = sum(obv_values[-10:-5]) / 5
            if recent_obv > prev_obv * 1.05:
                obv_signal = 1
            elif recent_obv < prev_obv * 0.95:
                obv_signal = -1
    mom_12_1 = 0.0
    mom_1m = 0.0
    if _deep_closes and len(_deep_closes) >= 252:
        _ret_12m = (_deep_closes[-1] / _deep_closes[-252] - 1.0) * 100
        _ret_1m = (_deep_closes[-1] / _deep_closes[-21] - 1.0) * 100
        mom_12_1 = _ret_12m - _ret_1m
        mom_1m = _ret_1m
    elif _deep_closes and len(_deep_closes) >= 21:
        mom_1m = (_deep_closes[-1] / _deep_closes[-21] - 1.0) * 100
        mom_12_1 = mom_1m * 0.5
    change_5d = change_pct
    if _deep_closes and len(_deep_closes) >= 6 and _deep_closes[-6] > 0:
        change_5d = (_deep_closes[-1] / _deep_closes[-6] - 1.0) * 100
    return adx_value, obv_signal, mom_12_1, mom_1m, change_5d


LIVE_SOURCE_LINES = {
    "bot_engine.py": [
        "start_d = (datetime.now() - timedelta(days=400)).strftime(\"%Y-%m-%d\")",
        "?timeframe=1Day&start={start_d}&limit=300&adjustment=all&feed={bars_feed()}",
        "if len(bars_data) >= 14:",
        "tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))",
        "atr14 = (atr14 * 13 + tr_list[i]) / 14",
        "dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0",
        "adx_value = round(dx, 1)",
        "if recent_obv > prev_obv * 1.05:",
        "elif recent_obv < prev_obv * 0.95:",
        "_ret_12m = (_deep_closes[-1] / _deep_closes[-252] - 1.0) * 100",
        "_ret_1m = (_deep_closes[-1] / _deep_closes[-21] - 1.0) * 100",
        "mom_12_1 = mom_1m * 0.5  # dampened extrapolation",
        "avg_volume = quick_result.get(\"volume\", 0)",
        "change_5d = (_deep_closes[-1] / _deep_closes[-6] - 1.0) * 100",
        "rsi = vol_metrics.get(\"rsi_14\") or None",
        "volume_ratio = vol_metrics.get(\"volume_ratio_5d\") or 1.0",
        "mr_result = mean_reversion_strategy.score(rsi, change_5d, volume_ratio)",
        "m_result = momentum_strategy.score(mom_12_1, mom_1m, avg_volume)",
        "if volume_ratio > 3:\n        volume_score = 90\n    elif volume_ratio > 2:\n        volume_score = 75\n    elif volume_ratio > 1.5:\n        volume_score = 60\n    elif volume_ratio > 1:\n        volume_score = 50\n    else:\n        volume_score = 35",
        "if ewma_premium > 5:\n            vol_premium_bonus = 8",
        "elif ewma_premium < -5:\n            vol_premium_bonus = 5",
        "momentum_score * 0.25 +\n        mean_reversion_score * 0.20 +\n        vrp_score * 0.25 +\n        squeeze_score_val * 0.15 +\n        volume_score * 0.15\n    ) + vol_premium_bonus",
        "if adx_value > 40:\n            # Strong trend — boost momentum trades, penalize mean reversion\n            adx_adjustment += 6",
        "elif adx_value > 25:\n            adx_adjustment += 3",
        "elif adx_value < 15:",
        "adx_adjustment -= 4",
        "obv_adjustment += 5", "obv_adjustment -= 5",
        "if rs and rs > 5:\n        combined_score += 6",
        "combined_score = max(0, min(100, combined_score))",
        "\"quick_score\": _capped_change * 3 + _vol_score + _direction_bonus + _extreme_penalty,",
        "_capped_change = min(abs(change_pct), 15.0)",
        "_extreme_penalty = -30 if abs(change_pct) > 50 else (-15 if abs(change_pct) > 30 else 0)",
        "_vol_score = min(v / 1_000_000, 5.0) * 2",
        "_direction_bonus = 5 if abs(change_pct) > 10 else (2 if abs(change_pct) > 5 else 0)",
        "if \".\" in sym or len(sym) > 5:",
        "\"change_pct\": round(change_pct, 2),",
    ],
    "analyze.py": [
        "hist = _fetch_alpaca_bars(ticker_symbol, days=365)",
        "rv20 = realized_vol(hist, 20) or 20.0",
        "ewma_rv = ewma_vol(hist, lambd=0.94, window=60) or rv20",
        "vol_metrics = compute_volume_metrics(hist)",
        "rsi = compute_rsi(hist, period=14)",
        "_spy = _fetch_alpaca_bars(\"SPY\", days=30)",
        "rs_score, rs_signal = compute_relative_strength(hist, _spy)",
    ],
}


def test_live_source_lines_still_match():
    for f, lines in LIVE_SOURCE_LINES.items():
        src = open(os.path.join(HERE, f), encoding="utf-8").read()
        for ln in lines:
            assert ln in src, f"{f} no longer contains: {ln[:80]!r} — re-check walkforward.py's mirror"


def test_live_floors_are_the_scanner_constants():
    src = open(os.path.join(HERE, "bot_engine.py"), encoding="utf-8").read()
    assert int(re.search(r"^MIN_PRICE = (\d+)", src, re.M).group(1)) == wf.MIN_PRICE
    assert int(re.search(r"^MIN_VOLUME = (\d+)", src, re.M).group(1)) == wf.MIN_VOLUME
    assert wf.MIN_DOLLAR_VOLUME == wf.MIN_PRICE * wf.MIN_VOLUME


# ── parity with the live computations ───────────────────────────────────────
@pytest.fixture(scope="module")
def tst():
    s, rows = _series()
    spy = _spy()
    f = wf.series_features(s, spy.days, spy.c)
    return s, rows, spy, f


def test_analyze_inputs_match_the_live_functions(tst):
    s, rows, spy, f = tst
    checked = 0
    for t in range(3, len(rows), 3):
        hist = _hist_df(rows, t, wf.HIST_DAYS)
        r = analyze.compute_rsi(hist, period=14)
        assert (np.isnan(f["_rsi"][t]) if r is None else f["_rsi"][t] == r), (t, r, f["_rsi"][t])
        vm = analyze.compute_volume_metrics(hist)
        assert f["_vol_ratio"][t] == (vm.get("volume_ratio_5d") or 1.0), t
        rv = analyze.realized_vol(hist, 20)
        assert (np.isnan(f["_rv20"][t]) if rv is None else f["_rv20"][t] == rv), (t, rv, f["_rv20"][t])
        if len(hist) >= 21:
            ev = analyze.ewma_vol(hist, lambd=0.94, window=60)
            assert f["_ewma"][t] == ev, (t, ev, f["_ewma"][t])
            rv20 = rv or 20.0
            ew = ev or rv20
            prem = ew - rv20
            assert f["vol_premium"][t] == (8 if prem > 5 else 5 if prem < -5 else 0)
        k = spy.index_on(s.days[t])
        if k >= 0 and len(hist) >= 20:
            spy_rows = _decoded_rows(spy)
            sh = _hist_df(spy_rows, k, wf.SPY_RS_DAYS)
            rs, _ = analyze.compute_relative_strength(hist, sh)
            assert f["_rs"][t] == rs, (t, rs, f["_rs"][t])
            assert f["rel_strength"][t] == (6 if rs and rs > 5 else 0)
        checked += 1
    assert checked > 150


def test_deep_score_blocks_match_the_live_code(tst):
    s, rows, spy, f = tst
    dec = _decoded_rows(s)
    for t in range(0, len(rows), 2):
        start = pp.day_num(rows[t]["d"]) - wf.DEEP_DAYS
        bars = [b for b in dec[:t + 1] if pp.day_num(b["d"]) >= start][:wf.DEEP_LIMIT]
        chg = round(float(f["_change_pct"][t]), 2)
        adx, obv, m121, m1, c5 = _live_deep_block(bars, chg)
        assert (np.isnan(f["_dx"][t]) if adx is None else abs(f["_dx"][t] - adx) < 1e-9), (t, adx, f["_dx"][t])
        assert f["obv"][t] == obv * 5, t
        assert f["_mom_12_1"][t] == pytest.approx(m121, rel=1e-12, abs=1e-12)
        assert f["_mom_1m"][t] == pytest.approx(m1, rel=1e-12, abs=1e-12)
        assert f["_chg5"][t] == pytest.approx(c5, rel=1e-12, abs=1e-12)
        exp_adx = 0 if adx is None else (6 if adx > 40 else 3 if adx > 25 else -4 if adx < 15 else 0)
        assert f["adx"][t] == exp_adx, t
        assert f["momentum"][t] == momentum.score(m121, m1, int(f["_v_raw"][t]))["score"]
        r = f["_rsi"][t]
        mr = 50 if not (np.isfinite(r) and r) else mean_reversion.score(r, c5, f["_vol_ratio"][t])["score"]
        assert f["mean_reversion"][t] == mr, t


def test_quick_score_and_composite_arithmetic(tst):
    s, rows, spy, f = tst
    for t in range(1, len(rows), 5):
        c, pc, v = rows[t]["c"], rows[t - 1]["c"], f["_v_raw"][t]
        chg = (c - pc) / pc * 100
        q = min(abs(chg), 15.0) * 3 + min(v / 1_000_000, 5.0) * 2 + (5 if abs(chg) > 10 else (2 if abs(chg) > 5 else 0)) \
            + (-30 if abs(chg) > 50 else (-15 if abs(chg) > 30 else 0))
        assert f["quick_score"][t] == pytest.approx(q, rel=1e-12)
        vr = f["_vol_ratio"][t]
        vs = 90 if vr > 3 else 75 if vr > 2 else 60 if vr > 1.5 else 50 if vr > 1 else 35
        comp = f["momentum"][t] * 0.25 + f["mean_reversion"][t] * 0.20 + 50 * 0.25 + 50 * 0.15 + vs * 0.15 + f["vol_premium"][t]
        comp = max(0, min(100, comp + f["adx"][t] + f["obv"][t] + f["rel_strength"][t]))
        assert f["volume"][t] == vs and f["price_composite"][t] == pytest.approx(comp)


def test_composite_v2_holds_the_preregistered_inputs_neutral(tst):
    """2026-11-06: price_composite_v2 = price_composite with exactly V2_NEUTRALIZED
    (volume -> 50, vol-premium bonus -> 0, relative-strength bonus -> 0); every
    other term identical. Pre-registered before any run includes it."""
    s, rows, spy, f = tst
    assert wf.V2_NEUTRALIZED == ("volume", "vol_premium", "rel_strength")
    assert "price_composite_v2" in wf.INPUTS and "price_composite_v2" in wf.INPUT_INFO
    moved = 0
    for t in range(1, len(rows)):
        v2 = f["momentum"][t] * 0.25 + f["mean_reversion"][t] * 0.20 + 50 * 0.25 + 50 * 0.15 + 50 * 0.15
        v2 = max(0, min(100, v2 + f["adx"][t] + f["obv"][t]))
        assert f["price_composite_v2"][t] == pytest.approx(v2), t
        legacy_terms = (f["volume"][t] - 50) * 0.15 + f["vol_premium"][t] + f["rel_strength"][t]
        if legacy_terms != 0:
            moved += 1
        elif 0 < f["price_composite"][t] < 100:
            assert f["price_composite_v2"][t] == pytest.approx(f["price_composite"][t]), t
    assert moved > 0, "fixture never exercised a neutralized term"


def test_horizons_include_a_descriptive_one_day_and_verdicts_stay_on_five():
    assert wf.HORIZONS == (1, 5, 20) and wf.PRIMARY_HORIZON == 5
    assert wf._nw_lags(1) == 1


def test_raw_share_volume_undoes_the_split_adjustment():
    s, rows = _series(raw_ratio=4.0, gaps=False, n=60)
    v_raw = wf.raw_share_volume(s)
    m = np.arange(len(v_raw)) != 30                             # _series edits bar 30's close but not its raw close
    assert np.allclose(v_raw[m], s.v[m] / 4.0, rtol=1e-5, atol=1.5)   # estimate precision = the raw close's 6 decimals
    assert not np.allclose(v_raw, s.v, rtol=0.5)


# ── no lookahead ────────────────────────────────────────────────────────────
def test_features_never_read_a_later_bar():
    s, rows = _series(n=420)
    spy = _spy()
    full = wf.series_features(s, spy.days, spy.c)
    rng = np.random.default_rng(5)
    for t in list(range(5, 420, 23)) + [419]:
        for mode in ("truncate", "perturb"):
            cut = pp.Series(**{**s.__dict__, "_dates": None})
            sp_days, sp_c = spy.days, spy.c.copy()
            if mode == "truncate":
                for k in ("days", "o", "h", "l", "c", "v", "raw_close"):
                    setattr(cut, k, getattr(s, k)[:t + 1].copy())
                ks = spy.index_at_or_before(s.days[t])
                sp_days, sp_c = spy.days[:ks + 1], spy.c[:ks + 1]
            else:
                for k in ("o", "h", "l", "c", "raw_close"):
                    a = getattr(s, k).copy()
                    a[t + 1:] *= rng.uniform(0.2, 5.0, len(a) - t - 1)
                    setattr(cut, k, a)
                vv = s.v.copy()
                vv[t + 1:] = rng.integers(0, 10**9, len(vv) - t - 1)
                cut.v = vv
                ks = spy.index_at_or_before(s.days[t])
                sp_c[ks + 1:] *= 3.0
            g = wf.series_features(cut, sp_days, sp_c, at=np.array([t]))
            for name in wf.INPUTS:
                assert g[name][0] == full[name][t], (mode, t, name, g[name][0], full[name][t])


def test_regime_labels_are_past_only_and_match_backtest_v2():
    spy = _spy()
    rng = random.Random(9)
    vrows = ws.path_rows(spy.dates[40:], 30.0, -0.0008, 0.04, 3e7, rng)
    vxx = pp.series_from_lines([ws.line("VXX", vrows)])[0]
    ours, missing = wf.regime_labels(spy.days, spy.c, wf.vxx_ratio_by_day([vxx]))
    theirs, quality = bt.regime_series({"date": spy.dates, "close": list(spy.c)}, {"date": vxx.dates, "close": list(vxx.c)})
    assert ours == theirs
    assert missing == pytest.approx(69 / len(spy.c))     # 40 days before VXX lists + its first 29 bars (no 30-bar SMA)
    cut, _ = wf.regime_labels(spy.days[:300], spy.c[:300], wf.vxx_ratio_by_day([vxx]))
    assert cut == ours[:300]


def test_weekly_rebalance_is_the_last_trading_day_of_each_iso_week():
    days = np.array([pp.day_num(d) for d in ws.bdays("2017-01-02", "2017-02-28", holidays=("2017-01-20",))])
    idx = wf.weekly_rebalance_indices(days, pp.day_num("2017-01-01"))
    got = [pp.day_str(days[i]) for i in idx]
    assert "2017-01-19" in got and "2017-01-20" not in got        # holiday Friday -> Thursday
    assert all(date_wd in (3, 4) for date_wd in [pp.date.fromisoformat(d).weekday() for d in got[:-1]])
    assert got[-1] == "2017-02-28"


# ── cost model and outcomes ─────────────────────────────────────────────────
def test_cost_model_equals_backtest_v2_liquidity_cost():
    rng = np.random.default_rng(1)
    v = rng.choice([2e5, 8e5, 3e6, 7e6, 3e7], 300) * rng.uniform(0.5, 1.5, 300)
    px = rng.choice([0.5, 1.2, 3.0, 9.0, 60.0], 300)
    ours = wf.cost_per_side(v, px)
    for i in range(300):
        assert ours[i] == bt.liquidity_cost_pct({"volume": list(v), "close": list(px)}, i), i


def test_forward_returns_costs_and_delisting_assumption(tmp_path):
    days = ws.bdays("2016-01-04", "2017-12-29")
    flat = lambda dd, p: [{"d": d, "o": p, "h": p * 1.01, "l": p * 0.99, "c": p, "v": 4_000_000, "rc": p} for d in dd]
    spy_rows = flat(days, 100.0)
    up = [{**r, "o": 50.0 * 1.01 ** i, "c": 50.0 * 1.01 ** i, "rc": 50.0 * 1.01 ** i, "h": 50.0 * 1.01 ** i * 1.001, "l": 50.0 * 1.01 ** i * 0.999}
          for i, r in enumerate(flat(days, 50.0))]
    gone = flat(days[:400], 20.0)
    merged = flat(days[:400], 20.0)
    lines = [ws.line("SPY", spy_rows), ws.line("UPUP", up),
             ws.line("GONE", gone, status="inactive", flags=["inactive_no_event_date"]),
             ws.line("MERG", merged, status="inactive", event={"kind": "merger", "date": days[399]})]
    ws.write_panel(str(tmp_path), lines, as_of=days[-1])
    b = pp.current_build(str(tmp_path))
    spy = pp.series_from_lines([lines[0]])[0]
    rebal = wf.weekly_rebalance_indices(spy.days, pp.day_num("2016-06-01"))
    keys, meta = [], []
    rows, st = wf.collect_rows(b["dir"], spy.days, spy.c, rebal, days[-1], 1e9, keys, meta)
    sym = np.array([meta[i]["symbol"] for i in rows["sid"]])
    # UPUP: +1%/bar, entry next open, exit close 5 bars after the decision
    i = np.nonzero(sym == "UPUP")[0][0]
    k = int(rows["k"][i])
    t = int(np.searchsorted(pp.series_from_lines([lines[1]])[0].days, spy.days[k]))
    entry, exit_ = 50.0 * 1.01 ** (t + 1), 50.0 * 1.01 ** (t + 5)
    cst = bt._LIQUIDITY_TIERS[-1][1]            # 4M shares/day -> the >1M tier, both sides
    assert rows["gross5"][i] == pytest.approx(exit_ / entry - 1, rel=1e-5)
    assert rows["net5"][i] == pytest.approx(exit_ * (1 - cst) / (entry * (1 + cst)) - 1, rel=1e-5)
    # GONE: undated end before the exit -> last close x (1 - 30%); MERG: dated merger -> last close
    last_dec = {s_: np.nonzero(sym == s_)[0][-1] for s_ in ("GONE", "MERG")}
    for s_, dr in (("GONE", wf.DELIST_RETURN_OTHER), ("MERG", wf.DELIST_RETURN_MERGER)):
        j = last_dec[s_]
        if np.isfinite(rows["gross20"][j]):
            assert rows["dl20"][j]
            assert rows["gross20"][j] == pytest.approx(dr, abs=1e-6)
    assert rows["dl20"].sum() >= 2 and st["delisted_symbols_used"] == {"GONE", "MERG"}
    # point in time: no row after a delisted series' last bar
    gone_last = pp.day_num(days[399])
    assert all(spy.days[rows["k"][j]] <= gone_last for j in np.nonzero(sym == "GONE")[0])


def test_pit_universe_ipo_absent_before_first_bar_and_floors(tmp_path):
    m = ws.synthetic_panel(str(tmp_path), n_names=12, delist=2, ipo=2)
    b = pp.current_build(str(tmp_path))
    spy = next(s for s in pp.iter_series(b["dir"]) if s.symbol == "SPY")
    rebal = wf.weekly_rebalance_indices(spy.days, pp.day_num("2016-03-01"))
    keys, meta = [], []
    rows, _ = wf.collect_rows(b["dir"], spy.days, spy.c, rebal, b["manifest"]["as_of"], 1e9, keys, meta)
    sym = np.array([meta[i]["symbol"] for i in rows["sid"]])
    for s_ in m["ipo"]:
        first = next(x for x in pp.iter_series(b["dir"]) if x.symbol == s_).first_day
        assert (spy.days[rows["k"][sym == s_]] >= first).all()
    for s_ in m["delisted"]:
        last = next(x for x in pp.iter_series(b["dir"]) if x.symbol == s_).last_day
        assert (spy.days[rows["k"][sym == s_]] <= last).all()


def test_dedupe_keeps_one_row_per_identical_security_day():
    rows = {"sid": np.array([0, 1, 2, 3]), "j": np.array([5, 5, 5, 6]), "c": np.array([10.0, 10.0, 11.0, 10.0]),
            "v": np.array([100, 100, 100, 100]), "x": np.array([1, 2, 3, 4])}
    out, removed = wf.dedupe_rows(rows)
    assert removed == 1 and list(out["x"]) == [1, 3, 4]


# ── metric math ─────────────────────────────────────────────────────────────
def test_spearman_quintiles_and_ties():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2)
    x, y = rng.integers(0, 5, 200).astype(float), rng.normal(size=200)
    assert wf.spearman(x, y) == pytest.approx(spearmanr(x, y)[0])
    assert math.isnan(wf.spearman(np.ones(50), y[:50]))
    q = wf.quintile_of(x)
    for v in np.unique(x):
        assert len(set(q[x == v])) == 1, "tied values share a bucket"
    q10 = wf.quintile_of(np.arange(10.0))
    assert list(q10) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_newey_west_reduces_to_iid_t_and_widens_under_autocorrelation():
    rng = np.random.default_rng(4)
    x = rng.normal(0.1, 1, 400)
    m, t = wf.newey_west_t(x, 0)
    assert m == pytest.approx(x.mean())
    assert t == pytest.approx(x.mean() / (x.std(ddof=0) / math.sqrt(len(x))))
    ar = np.convolve(rng.normal(0.05, 1, 420), np.ones(20) / 20, mode="valid")    # overlapping-window series
    assert abs(wf.newey_west_t(ar, 19)[1]) < abs(wf.newey_west_t(ar, 0)[1])
    assert wf.newey_west_t([np.nan, 1.0], 1)[1] is None


def test_bonferroni_and_p_values():
    assert wf.two_sided_p(1.959963984540054) == pytest.approx(0.05, abs=1e-9)
    assert wf.bonferroni(0.01, 18) == pytest.approx(0.18)
    assert wf.bonferroni(0.2, 18) == 1.0 and wf.bonferroni(None, 3) is None


def test_deflated_sharpe_formula():
    from statistics import NormalDist
    nd = NormalDist()
    # N = 1: no deflation -> the PSR against 0 (Bailey & Lopez de Prado 2012)
    sr, n, sk, ku = 0.2, 100, -0.3, 4.0
    psr = nd.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    assert wf.expected_max_sharpe(1, 0.5) == 0.0
    assert wf.deflated_sharpe(sr, n, sk, ku, 0.0) == pytest.approx(psr)
    # E[max] grows with the number of trials, so DSR falls
    v = 0.01
    e = [wf.expected_max_sharpe(k, v) for k in (2, 10, 100, 1000)]
    assert e == sorted(e) and e[0] > 0
    g = 0.5772156649015329
    assert e[1] == pytest.approx(math.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / 10) + g * nd.inv_cdf(1 - 1 / (10 * math.e))))
    d = [wf.deflated_sharpe(sr, n, sk, ku, x) for x in e]
    assert d == sorted(d, reverse=True)
    st = wf.sharpe_stats(list(np.random.default_rng(0).normal(0.1, 1, 300)))
    assert st["kurt"] == pytest.approx(3, abs=0.6) and st["n"] == 300


# ── end to end ──────────────────────────────────────────────────────────────
def test_end_to_end_finds_a_planted_momentum_signal_and_writes_the_contract(tmp_path):
    ws.synthetic_panel(str(tmp_path / "p"), n_names=70, signal=0.003, seed=21)
    out = tmp_path / "o"
    res = wf.run(str(tmp_path / "p"), str(out))
    assert res["status"] == "ok"
    d = json.load(open(out / "latest.json"))
    assert d["schema"] == wf.SCHEMA and d["panel"]["build_id"] == "b20261029"
    assert set(d["inputs"]) == set(wf.INPUTS) and d["composite"] == "price_composite"
    assert set(d["brain_long_term"]) == set(wf.INPUTS)
    for f in wf.INPUTS:
        r = d["inputs"][f]
        # exact pin moved 2026-11-06: {"5", "20"} -> {"1", "5", "20"} (descriptive +1d, human spec section 7)
        assert set(r["horizons"]) == {"1", "5", "20"} and "live_exit" in r and r["verdict"] in ("positive", "negative", "null", "insufficient")
        for hz in r["horizons"].values():
            for k in ("ic", "quintiles", "top_n", "by_regime", "by_spy_200d", "by_year", "by_liquidity"):
                assert k in hz
    for k in ("vrp", "squeeze", "news", "intel", "social", "alt", "fundamentals", "macro_adj"):
        assert k in d["not_measurable"]
    assert d["inputs"]["momentum"]["verdict"] == "positive"
    assert d["inputs"]["momentum"]["horizons"]["20"]["ic"]["mean"] > 0.1
    assert d["inputs"]["momentum"]["live_exit"]["excess_vs_random_pct"] > 0
    assert d["universe"]["delisted_symbol_share"] > 0 and d["universe"]["n_symbols"] >= 70
    assert d["multiple_testing"]["ic_tests"] == len(wf.INPUTS) * len(wf.HORIZONS)
    assert d["prior"].startswith("Stated 2026-10-29, before any real-panel run")
    txt = json.dumps(d)
    assert "S00" not in txt, "no ticker leaves the output (derived statistics only)"
    assert json.load(open(out / "last_run.json"))["status"] == "ok"


def test_end_to_end_pure_noise_finds_nothing(tmp_path):
    ws.synthetic_panel(str(tmp_path / "p"), n_names=60, signal=0.0, seed=5)
    res = wf.run(str(tmp_path / "p"), str(tmp_path / "o"))
    d = json.load(open(tmp_path / "o" / "latest.json"))
    survivors = [f for f in wf.INPUTS if d["inputs"][f]["verdict"] in ("positive", "negative")]
    assert len(survivors) <= 1, survivors      # Bonferroni at 5% family-wise: at most a stray false positive
    for f in wf.INPUTS:
        assert abs(d["inputs"][f]["horizons"]["20"]["ic"]["mean"]) < 0.08


def test_no_panel_is_skipped_and_memory_budget_aborts_cleanly(tmp_path, monkeypatch, capsys):
    out = tmp_path / "o"
    assert wf.run(str(tmp_path / "none"), str(out))["status"] == "skipped"
    assert json.load(open(out / "last_run.json"))["status"] == "skipped"
    ws.synthetic_panel(str(tmp_path / "p"), n_names=8)
    (out / "latest.json").write_text('{"keep": true}')
    monkeypatch.setattr(wf, "RSS_CHECK_EVERY", 1)
    rc = wf.main(["--run", "--panel-root", str(tmp_path / "p"), "--out-dir", str(out), "--max-rss-mb", "1"])
    assert rc == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "error"
    assert json.load(open(out / "latest.json")) == {"keep": True}, "a failed run never overwrites the last good result"
    assert "memory budget" in json.load(open(out / "last_run.json"))["reason"]


# ── pre-rank candidates (PRE-REGISTERED 2026-11-06, research/open_questions.md [SELECTION]) ──
def _closes_series(symbol, days, closes, v=3e6):
    rows = [{"d": d, "o": float(x), "h": float(x) * 1.001, "l": float(x) * 0.999, "c": float(x), "v": int(v), "rc": float(x)}
            for d, x in zip(days, closes)]
    return pp.series_from_lines([ws.line(symbol, rows)])[0]


def _ref_resid_momentum(s, spy, t):
    """Reference: np.linalg.lstsq on the window's own returns (stock bar to bar,
    SPY over the same days), residual sum incl. the intercept / (floored sd x sqrt n)."""
    k_of = [spy.index_on(int(d)) for d in s.days]
    r, m = [], []
    for k in range(t - (wf.MOM_LOOKBACK - 1), t - wf.MOM_SKIP + 1):
        r.append(s.c[k] / s.c[k - 1] - 1.0)
        m.append(spy.c[k_of[k]] / spy.c[k_of[k - 1]] - 1.0)
    r, m = np.array(r), np.array(m)
    (alpha, beta), *_ = np.linalg.lstsq(np.column_stack([np.ones(len(m)), m]), r, rcond=None)
    e = r - alpha - beta * m
    sd = max(math.sqrt(float(e @ e) / (len(r) - 2)), wf.RESID_SIGMA_FLOOR)
    return float((r - beta * m).sum() / (sd * math.sqrt(len(r)))), float(beta)


def test_prerank_candidates_are_preregistered_and_grow_the_bonferroni_family():
    assert wf.PRERANK_CANDIDATES == ("momentum_12_1", "resid_momentum_12_1", "quick_score_no_move")
    assert wf.PRERANK_LIVE == "quick_score" and wf.PRERANK_LIVE in wf.INPUTS
    assert wf.INPUTS[-3:] == wf.PRERANK_CANDIDATES, "appended after the live mirrors: every existing column keeps its place"
    for f in wf.PRERANK_CANDIDATES:
        assert f in wf.INPUT_INFO and wf.INPUT_INFO[f][1].startswith("NOT live"), f
        assert f in wf.PRIOR, f"no prior stated for {f}"
    assert "before any run that includes them" in wf.PRIOR and "Gate-2 reading, fixed now" in wf.PRIOR
    assert (wf.MOM_LOOKBACK, wf.MOM_SKIP, wf.RESID_MIN_OBS, wf.RESID_SIGMA_FLOOR) == (252, 21, 200, 0.005)
    # family pin: 30 (10 inputs x 3 horizons) before the candidates, 39 with them — every
    # existing p_bonferroni is the same raw p x 39 instead of x 30 (more conservative)
    assert len(wf.INPUTS) * len(wf.HORIZONS) == 39
    # nothing in the bot reads them (measurement only)
    for f in ("bot_engine.py", "system_config.py", "score_composite.py", "server/bot.ts", "server/routes.ts"):
        src = open(os.path.join(HERE, f), encoding="utf-8").read()
        for name in wf.PRERANK_CANDIDATES:
            assert name not in src, (f, name)


def test_momentum_12_1_is_the_skip_month_total_return(tst):
    s, rows, spy, f = tst
    for t in range(len(s.c)):
        if t >= wf.MOM_LOOKBACK:
            assert f["momentum_12_1"][t] == pytest.approx(s.c[t - wf.MOM_SKIP] / s.c[t - wf.MOM_LOOKBACK] - 1.0, rel=1e-12), t
        else:
            assert f["momentum_12_1"][t] == wf.NO_VALUE and f["resid_momentum_12_1"][t] == wf.NO_VALUE, t
    # the 231 daily returns of the residual window compound to exactly this return
    t = len(s.c) - 1
    win = s.c[t - wf.MOM_LOOKBACK + 1:t - wf.MOM_SKIP + 1] / s.c[t - wf.MOM_LOOKBACK:t - wf.MOM_SKIP]
    assert len(win) == 231 and np.prod(win) - 1 == pytest.approx(f["momentum_12_1"][t], rel=1e-9)


def test_residual_momentum_matches_a_reference_regression_across_gaps(tst):
    """The fixture has holidays and an 8-bar halt: SPY's return must span the same days."""
    s, rows, spy, f = tst
    checked = 0
    for t in range(wf.MOM_LOOKBACK, len(s.c), 3):
        ref, _ = _ref_resid_momentum(s, spy, t)
        assert f["resid_momentum_12_1"][t] == pytest.approx(ref, rel=1e-8, abs=1e-10), (t, ref, f["resid_momentum_12_1"][t])
        checked += 1
    assert checked > 80


def test_planted_momentum_ranks_first_and_the_skip_month_is_not_read():
    rng = np.random.default_rng(12)
    days = ws.bdays("2016-01-04", "2017-06-30")
    t = 300
    spy = _closes_series("SPY", days, 200 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(days))))
    noise = [_closes_series(f"N{i:02d}", days, 40 * np.cumprod(1 + rng.normal(0, 0.015, len(days)))) for i in range(15)]
    path = np.empty(len(days))                  # +0.4%/day through the window, then -1.5%/day for the skipped month
    for k in range(len(days)):
        path[k] = 0.004 if k <= t - wf.MOM_SKIP else -0.015
    path += rng.normal(0, 0.004, len(days))
    planted = _closes_series("PLNT", days, 30 * np.cumprod(1 + path))
    at = np.array([t])
    vals = {x.symbol: wf.series_features(x, spy.days, spy.c, at=at) for x in noise + [planted]}
    for name in ("momentum_12_1", "resid_momentum_12_1"):
        ranked = sorted(vals, key=lambda k: -vals[k][name][0])
        assert ranked[0] == "PLNT", (name, [(k, round(float(vals[k][name][0]), 3)) for k in ranked[:3]])
    # the live momentum sub-score's 12-1 (deep_score) is scaled by the last month's return; this one is not
    assert vals["PLNT"]["momentum_12_1"][0] > vals["PLNT"]["_mom_12_1"][0] / 100
    # nothing after bar t-21 is read; bar t-21 itself is
    base = {n_: vals["PLNT"][n_][0] for n_ in ("momentum_12_1", "resid_momentum_12_1")}
    for mode in ("after", "at"):
        cut = pp.Series(**{**planted.__dict__, "_dates": None})
        cc = planted.c.copy()
        sc_ = spy.c.copy()
        if mode == "after":
            cc[t - wf.MOM_SKIP + 1:] *= rng.uniform(0.3, 3.0, len(cc) - (t - wf.MOM_SKIP + 1))
            sc_[t - wf.MOM_SKIP + 1:] *= 3.0
        else:
            cc[t - wf.MOM_SKIP] *= 1.5
        cut.c = cc
        g = wf.series_features(cut, spy.days, sc_, at=at)
        for n_, v in base.items():
            assert (g[n_][0] == v) == (mode == "after"), (mode, n_, g[n_][0], v)


def test_residual_and_raw_momentum_disagree_when_beta_explains_the_move():
    """A high-beta name with no alpha rides a strong market to the top of raw
    momentum; a low-beta name with steady alpha is first on residual momentum."""
    rng = np.random.default_rng(31)
    days = ws.bdays("2016-01-04", "2017-03-31")
    m = rng.normal(0.002, 0.01, len(days))
    spy = _closes_series("SPY", days, 200 * np.cumprod(1 + m))
    hi_beta = _closes_series("HIB", days, 50 * np.cumprod(1 + 2.5 * m + rng.normal(0, 0.01, len(days))))
    alpha = _closes_series("ALF", days, 50 * np.cumprod(1 + 0.3 * m + 0.0015 + rng.normal(0, 0.005, len(days))))
    t = len(days) - 1
    fa = wf.series_features(hi_beta, spy.days, spy.c, at=np.array([t]))
    fb = wf.series_features(alpha, spy.days, spy.c, at=np.array([t]))
    assert fa["momentum_12_1"][0] > fb["momentum_12_1"][0] + 0.5
    assert fb["resid_momentum_12_1"][0] > fa["resid_momentum_12_1"][0] + 2.0
    assert abs(fa["resid_momentum_12_1"][0]) < 2.5                      # no alpha: a t-like score near 0
    _, beta_a = _ref_resid_momentum(hi_beta, spy, t)
    _, beta_b = _ref_resid_momentum(alpha, spy, t)
    assert beta_a == pytest.approx(2.5, abs=0.2) and beta_b == pytest.approx(0.3, abs=0.1)


def test_residual_momentum_floor_and_degenerate_markets():
    rng = np.random.default_rng(5)
    days = ws.bdays("2016-01-04", "2017-03-31")
    t = len(days) - 1
    at = np.array([t])
    m = rng.normal(0.0004, 0.01, len(days))
    spy = _closes_series("SPY", days, 200 * np.cumprod(1 + m))
    # a T-bill-like series: 5%/yr, residual sd ~0.002%/day; without the floor its score would be ~150
    bill = _closes_series("BILL", days, 100 * np.cumprod(1 + 0.0002 + rng.normal(0, 0.00002, len(days))))
    got = wf.series_features(bill, spy.days, spy.c, at=at)["resid_momentum_12_1"][0]
    ref, _ = _ref_resid_momentum(bill, spy, t)
    assert got == pytest.approx(ref, rel=1e-6) and 0 < got < 1.0
    # SPY against itself: zero residual, finite score ~0 (the floor keeps the denominator away from 0)
    own = wf.series_features(spy, spy.days, spy.c, at=at)["resid_momentum_12_1"][0]
    assert math.isfinite(own) and abs(own) < 1e-6
    # a flat market has no market model: NO_VALUE, while the raw return still has a value
    flat = np.full(len(days), 100.0)
    fl = wf.series_features(bill, spy.days, flat, at=at)
    assert fl["resid_momentum_12_1"][0] == wf.NO_VALUE and math.isfinite(fl["momentum_12_1"][0])


def test_quick_score_no_move_is_the_quick_score_minus_every_move_term(tst):
    s, rows, spy, f = tst
    for t in range(1, len(s.c), 4):
        chg = float(f["_change_pct"][t])
        move = min(abs(chg), 15.0) * 3 + (5 if abs(chg) > 10 else (2 if abs(chg) > 5 else 0)) \
            + (-30 if abs(chg) > 50 else (-15 if abs(chg) > 30 else 0))
        assert f["quick_score_no_move"][t] == min(f["_v_raw"][t] / 1_000_000, 5.0) * 2, t
        assert f["quick_score"][t] - f["quick_score_no_move"][t] == pytest.approx(move, rel=1e-12, abs=1e-12), t
    # the live quick score is exactly these four terms, and only _vol_score does not read change_pct
    src = open(os.path.join(HERE, "bot_engine.py"), encoding="utf-8").read()
    assert "\"quick_score\": _capped_change * 3 + _vol_score + _direction_bonus + _extreme_penalty," in src
    for ln in ("_capped_change = min(abs(change_pct), 15.0)",
               "_direction_bonus = 5 if abs(change_pct) > 10 else (2 if abs(change_pct) > 5 else 0)",
               "_extreme_penalty = -30 if abs(change_pct) > 50 else (-15 if abs(change_pct) > 30 else 0)"):
        assert ln in src and "change_pct" in ln
    assert "_vol_score = min(v / 1_000_000, 5.0) * 2" in src
    # a different move on the decision bar moves the quick score, not this candidate
    t = 300
    cut = pp.Series(**{**s.__dict__, "_dates": None})
    cut.c, cut.raw_close = s.c.copy(), s.raw_close.copy()
    cut.c[t] *= 1.2
    cut.raw_close[t] *= 1.2                                       # raw share volume (c x v / raw close) unchanged
    g = wf.series_features(cut, spy.days, spy.c, at=np.array([t]))
    assert g["quick_score"][0] != f["quick_score"][t] and g["quick_score_no_move"][0] == f["quick_score_no_move"][t]


def test_no_value_rows_sit_out_their_input_only():
    rng = np.random.default_rng(8)
    n_dates, per = 3, 60
    N = n_dates * per
    rows = {"sid": np.arange(N, dtype=np.int32), "j": np.repeat(np.arange(n_dates), per).astype(np.int32),
            "dv": rng.uniform(3e6, 9e7, N).astype(np.float32)}
    for f in wf.INPUTS:
        rows[f] = rng.normal(size=N).astype(np.float32)
    for h in wf.HORIZONS:
        rows[f"net{h}"] = rng.normal(0, 0.03, N).astype(np.float32)
        rows[f"gross{h}"] = (rows[f"net{h}"] + 0.002).astype(np.float32)
    missing = rng.random(N) < 1 / 3
    rows["momentum_12_1"][missing] = wf.NO_VALUE
    spy = np.full(100, 100.0)
    labels = {"regime": ["NEUTRAL"] * n_dates, "spy_200d": ["above_200d"] * n_dates}
    res, _ = wf.evaluate(rows, np.array([30, 40, 50]), np.arange(100), spy, spy, labels, {}, {f: {} for f in wf.INPUTS}, {})
    for f, mask in (("momentum_12_1", ~missing), ("quick_score", np.ones(N, dtype=bool))):
        ics, names = [], []
        for d in range(n_dates):
            sel = (rows["j"] == d) & mask
            x, y = rows[f][sel].astype(np.float64), rows["net5"][sel].astype(np.float64)
            ics.append(wf.spearman(x, y))
            names.append(sel.sum())
        h5 = res[f]["horizons"]["5"]
        assert h5["ic"]["mean"] == round(float(np.mean(ics)), 4) and h5["ic"]["n_dates"] == n_dates, f
        assert sum(h5["quintiles"]["mean_names"]) == pytest.approx(np.mean(names), abs=0.3), f
        assert res[f]["coverage"]["rows_with_value"] == int(mask.sum())
    assert res["momentum_12_1"]["coverage"]["share"] == round(1 - missing.mean(), 4) < 1
    for d in range(n_dates):
        sel = np.nonzero(rows["j"] == d)[0]
        top = wf.pick_top(rows, "momentum_12_1", sel, wf.TOP_N)
        assert len(top) == wf.TOP_N and np.isfinite(rows["momentum_12_1"][top]).all()
        assert set(top) == set(wf.pick_top(rows, "momentum_12_1", sel[~missing[sel]], wf.TOP_N))
    # fewer valued names than TOP_N: only those are picked, never a NO_VALUE row to fill the ten
    sel = np.nonzero(rows["j"] == 0)[0]
    rows["momentum_12_1"][sel[4:]] = wf.NO_VALUE
    assert list(np.sort(wf.pick_top(rows, "momentum_12_1", sel, wf.TOP_N))) == list(sel[:4][np.isfinite(rows["momentum_12_1"][sel[:4]])])


def test_end_to_end_reports_the_candidates_with_their_coverage(tmp_path):
    m = ws.synthetic_panel(str(tmp_path / "p"), n_names=45, signal=0.003, seed=17, ipo=4)
    assert wf.run(str(tmp_path / "p"), str(tmp_path / "o"))["status"] == "ok"
    d = json.load(open(tmp_path / "o" / "latest.json"))
    assert d["multiple_testing"]["ic_tests"] == 39
    assert d["multiple_testing"]["bonferroni_alpha"] == round(wf.ALPHA / 39, 6)
    for f in wf.INPUTS:
        cov = d["inputs"][f]["coverage"]
        if f in ("momentum_12_1", "resid_momentum_12_1"):
            assert 0.8 < cov["share"] < 1.0, (f, cov)          # the late listings lack a year of history at first
        else:
            assert cov["share"] == 1.0, (f, cov)                # every live mirror always has a value
    for f in ("momentum_12_1", "resid_momentum_12_1"):
        r = d["inputs"][f]
        assert r["horizons"]["20"]["ic"]["mean"] > 0.05, f       # the planted persistent drift
        assert r["live_exit"]["n_picks"] > 0 and r["label"] == wf.INPUT_INFO[f][0]
    assert set(m["ipo"]) and "S00" not in json.dumps(d)
