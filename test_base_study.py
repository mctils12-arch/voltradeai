"""Tests for scripts/base_study.py (the pre-registered base study, 2026-11-19).

Synthetic series only. They pin the study's timing (no lookahead), its tax and
cost arithmetic, its regime split, its comparison metrics and its three decision
rules, and that the committed result file says what the rules say."""
import importlib.util
import json
import math
import os
from datetime import date, timedelta

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("base_study", os.path.join(HERE, "scripts", "base_study.py"))
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

RESULT = os.path.join(HERE, "research", "base_study_2026-11-19.json")


def _days(n, start=date(2015, 1, 2)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _noise(n, seed=7, scale=1e5):
    x, out = seed, []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7fffffff
        out.append((x % 2001 - 1000) / scale)          # -1% .. +1% at the default scale
    return out


def _prices(rets, start=100.0):
    out = [start]
    for r in rets:
        out.append(out[-1] * (1.0 + r))
    return out


# ── the pre-registered menu and constants ────────────────────────────────────
def test_menu_is_the_pre_registered_one():
    assert list(bs.CANDIDATES) == ["OEF", "XLG", "MGK", "MTUM", "SPMO", "QUAL", "SPHQ", "MTUM+QUAL", "RSP", "USMV", "SPLV"]
    assert bs.CANDIDATES["MTUM+QUAL"] == {"MTUM": 0.5, "QUAL": 0.5}
    assert len(bs.FAMILY) == 16 and len(set(bs.FAMILY)) == 16
    assert "B0-ii" not in bs.FAMILY, "SPY buy-and-hold is the benchmark, not a tested series"
    assert (bs.TAX_SHORT, bs.TAX_LONG, bs.LONG_TERM_DAYS) == (0.24, 0.15, 365)
    assert bs.COST_BP == bs.fs.COST_BP == 2.0
    assert bs.END_DAY == "2026-11-18"


def test_live_cfg_is_read_from_system_config_not_restated():
    from system_config import BASE_CONFIG
    cfg = bs.live_cfg()
    assert cfg == {k: BASE_CONFIG[k] for k in bs.VM_KEYS}
    assert (cfg["FLOOR_VM_TARGET_VOL"], cfg["FLOOR_VM_CAP"], cfg["FLOOR_VM_BAND"], cfg["FLOOR_VM_WINDOW"],
            cfg["FLOOR_VM_CAPITAL_PCT"]) == (0.16, 1.5, 0.10, 20, 0.85), "the study describes this floor"


# ── tax ──────────────────────────────────────────────────────────────────────
def test_lots_are_fifo_and_long_term_means_more_than_365_days():
    lots = bs.Lots()
    lots.buy("2020-01-02", 10, 100.0)
    lots.buy("2020-12-01", 10, 150.0)
    st, lt = lots.sell("2021-01-01", 5, 200.0)          # 365 days: NOT more than one year
    assert (st, lt) == (500.0, 0.0)
    st, lt = lots.sell("2021-01-02", 10, 200.0)         # 366 days for the first lot, 32 for the second
    assert lt == 5 * 100.0 and st == 5 * 50.0
    assert math.isclose(lots.units(), 5.0)
    assert lots.unrealized("2021-01-02", 160.0) == (50.0, 0.0)
    # the report's "mean age of lots sold": dollar-weighted days held
    assert math.isclose(lots.sold_dollars, 15 * 200.0)
    assert math.isclose(lots.sold_dollar_days / lots.sold_dollars, (5 * 365 + 5 * 366 + 5 * 32) / 15)


def test_tax_due_nets_and_carries_losses():
    assert bs.tax_due(1000.0, 2000.0, 0.0) == (0.24 * 1000 + 0.15 * 2000, 0.0)
    assert bs.tax_due(-500.0, 2000.0, 0.0) == (0.15 * 1500, 0.0)        # a short-term loss nets against long-term gain
    assert bs.tax_due(1000.0, -400.0, 0.0) == (0.24 * 600, 0.0)
    tax, carry = bs.tax_due(-300.0, -200.0, 0.0)
    assert (tax, carry) == (0.0, 500.0), "a net loss earns no refund; it is carried"
    tax, carry = bs.tax_due(400.0, 1000.0, 500.0)
    assert math.isclose(tax, 0.15 * 900) and carry == 0.0               # the carry offsets short-term first


def test_buy_and_hold_pays_nothing_until_liquidation_then_15_percent_of_the_gain():
    days = _days(600)
    px = [100.0 * (1.0005 ** i) for i in range(600)]
    pre = bs.simulate_weights(days, {"X": px}, {"X": 1.0}, taxed=False)
    post = bs.simulate_weights(days, {"X": px}, {"X": 1.0}, taxed=True)
    assert post["tax_paid_pct_of_start"] == 0.0 and post["rets"] == pre["rets"]
    wealth = px[-1] / px[0]
    assert math.isclose(post["log_growth_holding_pct"], math.log(wealth) / pre["years"] * 100)
    assert math.isclose(post["log_growth_liquidated_pct"], math.log(1 + (wealth - 1) * 0.85) / pre["years"] * 100)
    assert pre["rebalances"] == 0 and pre["turnover_per_year"] == 0.0


def test_a_terminal_loss_earns_no_refund():
    days = _days(300)
    px = [100.0 * (0.999 ** i) for i in range(300)]
    post = bs.simulate_weights(days, {"X": px}, {"X": 1.0}, taxed=True)
    assert math.isclose(post["log_growth_liquidated_pct"], post["log_growth_holding_pct"])


def test_yearly_rebalance_happens_on_the_last_session_of_the_year_and_is_taxed_and_costed():
    days = _days(800)                                    # 2015-01-02 .. early 2018: three year-ends
    a = [100.0 * (1.001 ** i) for i in range(800)]
    b = [100.0] * 800
    pre = bs.simulate_weights(days, {"A": a, "B": b}, {"A": 0.5, "B": 0.5}, taxed=False)
    post = bs.simulate_weights(days, {"A": a, "B": b}, {"A": 0.5, "B": 0.5}, taxed=True)
    assert pre["rebalances"] == 3
    assert pre["cost_paid_pct_of_start"] > 0 and pre["turnover_per_year"] > 0
    assert post["tax_paid_pct_of_start"] > 0, "selling the winner at year-end realizes a gain"
    assert post["log_growth_holding_pct"] < pre["log_growth_holding_pct"]
    # the first year-end is under 365 days from the purchase: that gain is short-term
    i_ye = max(i for i, d in enumerate(days) if d.startswith("2015"))
    half = bs.START_EQUITY / 2
    eq = half * a[i_ye] / a[0] + half
    sold = half * a[i_ye] / a[0] - eq / 2
    gain = sold * (1 - a[0] / a[i_ye])
    one = bs.simulate_weights(days[:i_ye + 3], {"A": a[:i_ye + 3], "B": b[:i_ye + 3]}, {"A": 0.5, "B": 0.5}, taxed=True)
    assert math.isclose(one["tax_paid_pct_of_start"], 0.24 * gain / bs.START_EQUITY * 100, rel_tol=1e-9)


def test_no_rebalance_cost_for_a_single_ticker_and_no_entry_cost_for_anyone():
    days = _days(300)
    px = _prices(_noise(299))
    run = bs.simulate_weights(days, {"X": px}, {"X": 1.0}, taxed=False)
    assert run["cost_paid_pct_of_start"] == 0.0
    assert all(math.isclose(r, px[i + 1] / px[i] - 1.0, abs_tol=1e-12) for i, r in enumerate(run["rets"]))


# ── the floor simulator (the live planning code) ─────────────────────────────
def _floor(n=400, vol_rule=True, cash_mode="cash0", taxed=False, shock=None, fund_rate=0.0, scale=1e5):
    warm = 30
    r = _noise(n + warm - 1, scale=scale)
    if shock is not None:
        r[warm + shock - 1] = -0.20                      # the return INTO sample day `shock`
    base = _prices(r)
    lev = _prices([2 * x for x in r])
    days = _days(n)
    return bs.simulate_floor(days, base[warm:], lev[warm:], base, list(range(warm, warm + n)), [fund_rate] * n,
                             vol_rule, cash_mode, taxed), base[warm:]


def test_vol_rule_off_and_no_reserve_is_exactly_buy_and_hold():
    run, base = _floor(vol_rule=False, cash_mode="invested")
    assert run["rebalances"] == 0 and run["cost_paid_pct_of_start"] == 0.0
    assert all(math.isclose(r, base[i + 1] / base[i] - 1.0, abs_tol=1e-9) for i, r in enumerate(run["rets"]))
    assert math.isclose(run["mean_exposure_of_equity"], 1.0, abs_tol=1e-9)


def test_live_rule_holds_85_percent_capital_and_a_15_percent_reserve():
    run, _ = _floor(vol_rule=False, cash_mode="cash0")
    assert math.isclose(run["exposures"][0], 0.85, abs_tol=1e-6)
    assert math.isclose(run["mean_cash_share"], 0.15, abs_tol=0.01) and run["mean_fund_share"] == 0.0
    tb, _ = _floor(vol_rule=False, cash_mode="tbill")
    assert math.isclose(tb["mean_fund_share"], 0.15, abs_tol=0.01) and tb["mean_cash_share"] < 0.001


def test_quiet_market_takes_the_cap_through_the_2x_fund_never_margin():
    run, _ = _floor(vol_rule=True, cash_mode="cash0", scale=1e6)     # ~0.9% annual vol -> E* = the 1.5x cap
    assert math.isclose(run["exposures"][0], 1.5 * 0.85, abs_tol=1e-6)
    assert max(run["exposures"]) <= 1.5 * 0.85 + 0.10 * 0.85 + 1e-9, "never beyond the cap plus the band"
    inv, _ = _floor(vol_rule=True, cash_mode="invested", scale=1e6)
    assert math.isclose(inv["exposures"][0], 1.5, abs_tol=1e-6)


def test_exposure_never_sees_the_day_it_is_applied_to():
    a, _ = _floor(shock=None)
    b, _ = _floor(shock=200)
    # exposures[i] is decided at close i and held into day i+1
    assert b["exposures"][:200] == a["exposures"][:200], "nothing before the crash can react to it"
    assert b["rets"][:199] == a["rets"][:199]
    assert min(b["exposures"][200:221]) < min(a["exposures"][200:221]), "after the crash the rule cuts exposure"


def test_idle_floor_capital_parks_in_the_fund_and_the_reserve_stays_in_cash():
    run, _ = _floor(shock=100, fund_rate=0.0002)
    assert run["mean_fund_share"] > 0.0, "a cut below 1x parks the idle part"
    assert run["mean_cash_share"] >= 0.145, "the 15% reserve is never parked"
    inv, _ = _floor(shock=100, fund_rate=0.0002, cash_mode="invested")
    assert inv["mean_cash_share"] < 0.03


def test_floor_rebalancing_pays_cost_and_tax():
    pre, _ = _floor(n=700, shock=100)
    post, _ = _floor(n=700, shock=100, taxed=True)
    assert pre["rebalances"] > 0 and pre["cost_paid_pct_of_start"] > 0
    assert pre["tax_paid_pct_of_start"] == 0.0
    assert post["log_growth_liquidated_pct"] <= post["log_growth_holding_pct"] + 1e-12 <= pre["log_growth_holding_pct"] + 1e-9


def test_synthetic_2x_fund_formula():
    tr = {"2020-01-02": 100.0, "2020-01-03": 101.0}
    out = bs.synthetic_2x(tr, {"2020-01-02": 2.52, "2020-01-03": 2.52})
    rf = 2.52 / 100 / 252
    assert math.isclose(out["2020-01-03"], 100.0 * (1 + 2 * 0.01 - (rf + 0.01 / 252) - 0.009 / 252))


# ── comparison metrics ───────────────────────────────────────────────────────
def test_tracking_error_and_information_ratio():
    r = _noise(500)
    te, ir = bs.tracking(r, r)
    assert te == 0.0 and ir is None
    up = [x + 0.0001 * (1 if i % 2 else -1) + 0.0001 for i, x in enumerate(r)]
    te, ir = bs.tracking(up, r)
    assert te > 0 and math.isclose(ir, 0.0001 * 252 / (te / 100), rel_tol=1e-9)


def test_monthly_excess_t_statistic_on_a_known_series():
    days = _days(252)
    bench = [0.0] * 252
    rets = [0.001 if d[5:7] in ("01", "03", "05", "07", "09", "11") else 0.0 for d in days]
    e = bs.excess_t(days, rets, bench)
    months, vals = bs.monthly(days, rets)
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
    assert e["months"] == n and math.isclose(e["t"], mu / (sd / math.sqrt(n)))
    assert 0 < e["p"] < 1
    assert bs.excess_t(days, bench, bench)["t"] == 0.0 and bs.excess_t(days, bench, bench)["p"] == 1.0


def test_longest_underperformance_counts_months_below_an_earlier_relative_high():
    days = _days(21 * 14)
    months = sorted({d[:7] for d in days})
    lag = set(months[2:7])                               # five months of lagging, then catching up
    rets = [(-0.001 if d[:7] in lag else 0.004 if d[:7] in months[7:9] else 0.0) for d in days]
    out = bs.longest_underperformance_months(days, rets, [0.0] * len(days))
    assert 5 <= out <= 7
    assert bs.longest_underperformance_months(days, [0.001] * len(days), [0.0] * len(days)) == 0
    assert bs.longest_underperformance_months(days, [-0.001] * len(days), [0.0] * len(days)) == len(months), "an unfinished stretch counts"


def test_month_regime_uses_the_previous_month_end_and_never_the_month_itself():
    n, warm = 300, 30
    r = _noise(n + warm - 1)
    closes = _prices(r)
    days = _days(n)[1:]                                  # return days
    pos = list(range(warm + 1, warm + n))
    base = bs.month_regimes(days, closes, pos)
    assert set(base.values()) == {"high", "low"}
    target = sorted(base)[5]
    idx = [i for i, d in enumerate(days) if d[:7] == target]
    shocked = list(closes)
    for i in idx[1:]:                                    # wreck the month AFTER its first close
        shocked[pos[i]] = closes[pos[i]] * (0.5 if i % 2 else 1.5)
    reading_month = bs.month_regimes(days, shocked, pos)
    assert reading_month[target] == base[target], "a month's own moves cannot set its own label"
    highs = sum(1 for v in base.values() if v == "high")
    assert abs(highs - len(base) / 2) <= 1, "the split is at the median"


def test_thirds_are_equal_and_cover_the_sample():
    assert bs.thirds(10) == [(0, 3), (3, 6), (6, 10)]
    assert bs.thirds(9) == [(0, 3), (3, 6), (6, 9)]


def test_compare_reports_a_series_against_itself_as_a_tie():
    days = _days(900)
    r = _noise(900)
    c = bs.compare(days, r, r, r, [0.0] * 900, {})
    assert c["tracking_error_pct"] == 0.0 and c["excess"]["t"] == 0.0
    assert c["max_dd_pct"] == c["spy_max_dd_pct"] and c["sharpe"] == c["spy_sharpe"]
    assert not any(s["beats_spy"] for s in c["subperiods"]), "a tie is not a win"
    assert c["rolling3y_beat_spy_pct"] == 0.0 and c["longest_underperformance_months"] == 0


# ── decision rules ───────────────────────────────────────────────────────────
def _side(growth=12.0, spy=10.0, dd=-30.0, spy_dd=-30.0, sharpe=0.8, spy_sharpe=0.7, subs=(True, True, False), hi=True, lo=True):
    return {"liq_log_growth_pct": growth, "spy_liq_log_growth_pct": spy, "max_dd_pct": dd, "spy_max_dd_pct": spy_dd,
            "sharpe": sharpe, "spy_sharpe": spy_sharpe, "subperiods": [{"beats_spy": s} for s in subs],
            "regimes": {"high": {"beats_spy": hi}, "low": {"beats_spy": lo}}}


def test_rule_a_needs_all_six_and_significance_is_the_last_gate():
    assert bs.rule_a(_side(), _side(), True)["outcome"] == "REPLACE"
    r = bs.rule_a(_side(), _side(), False)
    assert r["meets_1_to_5"] and r["outcome"] == "SPY STAYS (not disproved)"
    assert bs.rule_a(_side(), _side(growth=9.0), True)["outcome"] == "SPY STAYS", "must win on the common sample too"
    assert bs.rule_a(_side(subs=(True, False, False)), _side(), True)["outcome"] == "SPY STAYS"
    assert bs.rule_a(_side(lo=False), _side(), True)["outcome"] == "SPY STAYS"
    assert bs.rule_a(_side(dd=-35.1), _side(), True)["outcome"] == "SPY STAYS"
    assert bs.rule_a(_side(dd=-35.0), _side(), True)["outcome"] == "REPLACE", "exactly 5 points worse is allowed"
    assert bs.rule_a(_side(), _side(dd=-36.0), True)["outcome"] == "SPY STAYS"
    assert bs.rule_a(_side(sharpe=0.69), _side(), True)["outcome"] == "SPY STAYS"


def test_rule_a_flags_a_paper_artifact():
    r = bs.rule_a(_side(), _side(), False, _side(growth=9.5), _side())
    assert r["paper_artifact"] is True and r["condition_1_on_total_return"] is False
    assert bs.rule_a(_side(), _side(), False, _side(), _side())["paper_artifact"] is False


def _cell(g, sh, dd, subs, liq=None, after=None):
    return {"log_growth_pct": g, "sharpe": sh, "max_dd_pct": dd, "liq_log_growth_pct": g if liq is None else liq,
            "sub_log_growth_pct": list(subs), "sub_after_tax_log_growth_pct": list(after or subs)}


def test_rule_b_is_the_floor_studys_ship_rule_on_both_samples():
    off = _cell(10.0, 0.50, -50.0, (9, 7, 14))
    good = bs.rule_b_sample(_cell(10.2, 0.56, -45.0, (8, 8, 15)), off)
    risk_only = bs.rule_b_sample(_cell(9.8, 0.56, -45.0, (8, 8, 15)), off)
    bad = bs.rule_b_sample(_cell(9.8, 0.49, -45.0, (8, 8, 15)), off)
    assert good["passes"] and not risk_only["passes"] and risk_only["better_on_risk"] and not bad["better_on_risk"]
    assert bs.rule_b(good, good) == "earns its keep"
    assert bs.rule_b(good, risk_only) == "earns it on risk (Sharpe and drawdown), not on growth"
    assert bs.rule_b(good, bad) == "does not earn it on this evidence"
    assert not bs.rule_b_sample(_cell(10.2, 0.56, -60.1, (8, 8, 15)), off)["passes"], "more than 10 points of extra drawdown fails"


def test_rule_c():
    cells = {"B0-i": _cell(8.0, 0.5, -45.0, (1, 2, 3)), "B0-ii": _cell(9.5, 0.5, -52.0, (1, 2, 3)),
             "B0-iii": _cell(8.2, 0.52, -45.1, (1, 2, 3)),
             "B1-i": _cell(8.0, 0.6, -30.0, (1, 2, 3)), "B1-iii": _cell(8.3, 0.62, -30.2, (1, 2, 3), liq=7.5),
             "B1-ii": _cell(9.6, 0.6, -36.0, (2, 3, 2), liq=8.5)}
    out = bs.rule_c(cells, bench_max_dd=-52.0)
    assert out["tbill_fund_preferred_to_cash_at_0"] and out["fully_invested_preferred_to_tbill_fund"]
    cells["B1-ii"] = _cell(9.6, 0.6, -57.5, (2, 3, 2), liq=8.5)
    assert not bs.rule_c(cells, bench_max_dd=-52.0)["fully_invested_preferred_to_tbill_fund"]
    cells["B0-iii"] = _cell(7.9, 0.5, -45.0, (1, 2, 3))
    assert not bs.rule_c(cells, bench_max_dd=-52.0)["tbill_fund_preferred_to_cash_at_0"]


def test_significance_uses_the_family_of_16_even_when_a_ticker_is_dropped():
    own = {n: {"excess": {"t": 1.0, "p": 0.3173, "months": 120}} for n in bs.FAMILY}
    own["OEF"] = {"dropped": "x"}
    own["MGK"] = {"excess": {"t": 3.5, "p": 0.000465, "months": 120}}
    s = bs.significance(own)
    assert s["family_size"] == 16 and s["tested"] == 15
    assert math.isclose(s["bonferroni_abs_t"], bs.bonferroni_z(16), rel_tol=1e-12) and 2.9 < s["bonferroni_abs_t"] < 3.0
    assert s["series"]["MGK"]["survives_bonferroni"] and not s["series"]["RSP"]["survives_bonferroni"]
    assert s["series"]["MGK"]["bh_q"] < 0.10 < s["series"]["RSP"]["bh_q"]


# ── the committed result says what the rules say ─────────────────────────────
@pytest.mark.skipif(not os.path.exists(RESULT), reason="result file not generated yet")
def test_committed_result_matches_the_rules_and_the_write_up():
    js = json.load(open(RESULT))
    po, tot, sig = js["price_only"], js["total_return"], js["significance"]
    assert js["end_day"] == bs.END_DAY and sig["family_size"] == 16
    for n, v in js["rule_a"].items():
        if "checks" not in v:
            continue
        ok6 = sig["series"][n]["survives_bonferroni"] and sig["series"][n]["t"] > 0
        again = bs.rule_a(po["own"][n], po["common"][n], ok6, tot["own"][n], tot["common"][n])
        assert again["outcome"] == v["outcome"] and again["checks"] == v["checks"], n
    winners = [n for n, v in js["rule_a"].items() if v["outcome"] == "REPLACE"]
    assert js["rule_a_decision"] == (("REPLACE with " + ", ".join(winners)) if winners else "SPY stays")
    bm = bs.rule_b_sample(bs.grid_view(po["modern_grid"]["B1-iii"]), bs.grid_view(po["modern_grid"]["B0-iii"]))
    bl = bs.rule_b_sample(bs.grid_view(po["long_grid"]["B1-iii"]), bs.grid_view(po["long_grid"]["B0-iii"]))
    assert js["rule_b"]["verdict"] == bs.rule_b(bm, bl)
    rc = bs.rule_c({k: bs.grid_view(c) for k, c in po["modern_grid"].items()}, po["modern_grid"]["B0-ii"]["max_dd_pct"])
    assert rc == js["rule_c"]
    # B0-ii is the benchmark: it must tie SPY exactly
    assert po["modern_grid"]["B0-ii"]["tracking_error_pct"] == 0.0
    # every number in the write-up's tables is generated from this file
    md = open(os.path.join(HERE, "research", "base_study_2026-11-19.md")).read()
    for line in bs.markdown(js).splitlines():
        if line.startswith("|") or line.startswith("**Rule"):
            assert line in md, f"the write-up is missing or has altered this generated line: {line[:80]}"
