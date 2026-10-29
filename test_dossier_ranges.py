"""dossier_ranges: vol models, bands, walk-forward coverage (no lookahead),
trade-plan touch probabilities, the calibration ledger and its scorer."""
import math
import random
from datetime import date, datetime, timedelta, timezone

import numpy as np

import dossier_ranges as DR


def _closes(n=900, vol=0.02, seed=3, p0=50.0):
    rnd = random.Random(seed)
    out, p = [], p0
    for _ in range(n):
        p *= math.exp(rnd.gauss(0, vol))
        out.append(p)
    return out


def _bars(start: date, closes, highs=None, lows=None):
    out, d, k = [], start, 0
    while k < len(closes):
        if d.weekday() < 5:
            c = closes[k]
            out.append({"t": f"{d.isoformat()}T04:00:00Z", "c": c,
                        "h": highs[k] if highs else c, "l": lows[k] if lows else c})
            k += 1
        d += timedelta(days=1)
    return out


def test_band_is_zero_drift_lognormal():
    lo, hi = DR.band(100.0, 0.1, 1.0)
    assert math.isclose(lo, 100 * math.exp(-0.1)) and math.isclose(hi, 100 * math.exp(0.1))
    b = DR.bands_for(100.0, 0.1)
    assert b["lo90"] < b["lo68"] < 100 < b["hi68"] < b["hi90"]
    assert math.isclose(b["move68_pct"], (math.exp(0.1) - 1) * 100, rel_tol=1e-3)
    assert DR.bands_for(100.0, None) is None and DR.bands_for(0.0, 0.1) is None


def test_ewma_of_constant_magnitude_returns():
    r = [0.02 if k % 2 else -0.02 for k in range(200)]
    assert math.isclose(DR.ewma_var(r), 0.0004, rel_tol=1e-6)
    assert DR.ewma_var(r[:10]) is None


def test_har_fit_uses_nothing_after_end():
    r = DR.log_returns(_closes())
    end = 600
    a = DR.fit_har(r, 5, end=end)
    r2 = r.copy()
    r2[end + 1:] = r2[end + 1:] * 7.0   # a wildly different future
    b = DR.fit_har(r2, 5, end=end)
    assert a is not None and np.allclose(a, b)
    # and the future does change a fit that is allowed to see it
    assert not np.allclose(DR.fit_har(r, 5), DR.fit_har(r2, 5))


def test_choose_var_falls_back_to_ewma():
    assert DR.choose_var(1e-4, 1e-4)[1] == "har"
    v, m, why = DR.choose_var(1e-2, 1e-4)
    assert m == "ewma" and v == 1e-4 and "sanity" in why
    assert DR.choose_var(None, 2e-4)[1] == "ewma"
    assert DR.choose_var(None, None)[1] == "none"


def test_walk_forward_coverage_near_nominal_on_gaussian_returns():
    c = _closes(n=2200, vol=0.015, seed=11)
    cov = DR.walk_forward_coverage(c, 1, n_eval=1000)
    assert cov["n"] == 1000
    assert 0.63 <= cov["rate68"] <= 0.73 and 0.86 <= cov["rate90"] <= 0.94


def test_walk_forward_forecasts_never_see_the_outcome():
    """Changing ONLY the final close moves only the last origin's outcome."""
    c = _closes(n=700, seed=5)
    a = DR.walk_forward_coverage(c, 1, n_eval=200)
    c2 = list(c)
    c2[-1] = c2[-2] * 1.5      # a +50% day: outside every band of the last origin
    b = DR.walk_forward_coverage(c2, 1, n_eval=200)
    assert b["n"] == a["n"]
    assert a["in90"] - b["in90"] in (0, 1) and a["in68"] - b["in68"] in (0, 1)


def test_sigma_forecast_shapes():
    s = DR.sigma_forecast(_closes())
    assert set(s["horizons"]) == {"1", "5"}
    h1, h5 = s["horizons"]["1"], s["horizons"]["5"]
    assert h1["sigma"] > 0 and h5["sigma"] > h1["sigma"]
    assert math.isclose(h5["sigma"], h5["sigma_daily"] * math.sqrt(5))


def test_touch_probs_match_driftless_gamblers_ruin():
    # long horizon: P(+a before -b) -> b / (a + b) for driftless Brownian motion
    p = DR.touch_probs(0.02, 400, 0.05, 0.10, paths=3000, steps_per_session=20)
    assert abs(p["p_target_first"] - 2 / 3) < 0.04 and p["p_neither"] < 0.01


def test_touch_probs_single_barrier_reflection():
    # stop unreachable: P(max >= a by T) = 2 * (1 - Phi(a / (sigma sqrt T))); discrete steps miss a little
    sig, T, a = 0.02, 5, 0.04
    z = a / (sig * math.sqrt(T))
    exact = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    p = DR.touch_probs(sig, T, a, 5.0, paths=6000)
    assert exact - 0.05 < p["p_target_first"] <= exact + 0.02
    assert p["p_stop_first"] == 0.0


def test_trade_plan_levels_and_rules():
    p = DR.trade_plan(100.0, 0.02)
    s = 0.02 * math.sqrt(5)
    assert math.isclose(p["stop"], 100 * math.exp(-DR.Z90 * s), rel_tol=1e-4)
    assert math.isclose(p["target"], 100 * math.exp(s), rel_tol=1e-4)
    assert p["stop"] < 100 < p["target"] < p["stretch_target"]
    assert math.isclose(p["reward_risk"], (p["target"] - 100) / (100 - p["stop"]), rel_tol=1e-3)
    t = p["touch"]
    assert math.isclose(t["p_target_first"] + t["p_stop_first"] + t["p_neither"], 1.0, abs_tol=1e-3)
    assert "no drift" in p["rules"]["probabilities"]
    assert DR.trade_plan(100.0, None) is None


def test_score_range_split_safe_and_pending():
    bars = _bars(date(2026, 10, 19), [10.0, 10.5, 11.0, 9.0, 10.0, 10.2, 10.1])
    row = DR.range_rows("X", "2026-10-19", 100.0, {"horizons": {"1": {"sigma": 0.03}, "5": {"sigma": 0.07}}}, "now")
    r1, r5 = row
    s1 = DR.score_range(r1, bars)   # 10.0 -> 10.5 = +5%: outside 68% (+3.05%), inside 90% (+5.06%)
    assert s1["realized_close"] == 105.0 and s1["in68"] == 0 and s1["in90"] == 1
    s5 = DR.score_range(r5, bars)   # 5th session after 2026-10-19 -> close 10.2
    assert s5["resolved_date"] == bars[5]["t"][:10] and s5["in68"] == 1
    assert DR.score_range(r5, bars[:5]) is None


def _plan_row(serve="2026-10-20"):
    return {"kind": "plan", "id": "plan|X|" + serve, "ticker": "X", "serve_date": serve, "base_date": "2026-10-19",
            "base_close": 100.0, "entry_ref": 100.0, "stop": 95.0, "target": 105.0, "h": 3}


def test_score_plan_outcomes_and_no_serve_day_lookahead():
    start = date(2026, 10, 19)  # Mon; serve day Tue 10-20
    closes = [100, 100, 100, 100, 100, 100]
    # the serve-day bar itself touches the target — it must be ignored
    hi = [100, 106, 101, 106, 101, 101]
    lo = [100, 99, 99, 99, 99, 99]
    s = DR.score_plan(_plan_row(), _bars(start, closes, hi, lo))
    assert s["outcome"] == "target_first" and s["resolved_date"] == "2026-10-22"
    lo2 = [100, 99, 94, 99, 99, 99]
    assert DR.score_plan(_plan_row(), _bars(start, closes, hi, lo2))["outcome"] == "stop_first"
    lo3 = [100, 99, 99, 94, 99, 99]
    assert DR.score_plan(_plan_row(), _bars(start, closes, hi, lo3))["outcome"] == "ambiguous"
    flat = [101] * 6
    s4 = DR.score_plan(_plan_row(), _bars(start, closes, flat, [99] * 6))
    assert s4["outcome"] == "neither" and s4["resolved_date"] == "2026-10-23"
    assert DR.score_plan(_plan_row(), _bars(start, closes[:3], flat[:3], [99] * 3)) is None


def test_score_plan_rescales_levels_after_a_split():
    start = date(2026, 10, 19)
    closes = [50, 50, 50, 50, 50]      # a 2:1 split: fetched history is halved
    hi = [50, 51, 52.6, 50, 50]        # 52.6 > 105/2
    s = DR.score_plan(_plan_row(), _bars(start, closes, hi, [49.0] * 5))
    assert s["outcome"] == "target_first"


def test_record_dedups_and_ledger_scores(tmp_path):
    lp, sp = str(tmp_path / "l.jsonl"), str(tmp_path / "s.jsonl")
    sig = {"horizons": {"1": {"sigma": 0.03}, "5": {"sigma": 0.07}}}
    rows = DR.range_rows("X", "2026-10-19", 100.0, sig, "a")
    assert DR.record(rows, lp) == 2
    assert DR.record(DR.range_rows("X", "2026-10-19", 120.0, sig, "b"), lp) == 0   # same ticker/day: first wins
    assert DR.record([DR.plan_row("X", "2026-10-20", "2026-10-19", 100.0, DR.trade_plan(100.0, 0.02), "a")], lp) == 1
    bars = _bars(date(2026, 10, 19), [100, 101, 102, 100, 99, 100, 101, 100, 100],
                 highs=[100, 101, 102, 100, 99, 100, 101, 100, 100], lows=[100, 101, 102, 100, 99, 100, 101, 100, 100])

    def fetch(syms, start, end, deadline):
        return {"X": bars}, set()

    now = datetime(2026, 11, 2, 23, 0, tzinfo=timezone.utc)
    st = DR.score_ledger(now=now, fetch=fetch, path=lp, scores_path=sp, completed=lambda b, n: b)
    assert st["resolved"] == 3 and st["pending"] == 0
    assert DR.score_ledger(now=now, fetch=fetch, path=lp, scores_path=sp, completed=lambda b, n: b)["open"] == 0
    rep = DR.ledger_report("X", path=lp, scores_path=sp)
    assert rep["ticker"]["range"]["1"]["n"] == 1 and rep["ticker"]["range"]["1"]["in68"] == 1
    assert rep["all"]["plan"]["n"] == 1 and rep["all"]["plan"]["neither"] == 1


def test_score_ledger_fetch_failure_is_not_a_strike(tmp_path):
    lp, sp = str(tmp_path / "l.jsonl"), str(tmp_path / "s.jsonl")
    DR.record(DR.range_rows("X", "2026-10-19", 100.0, {"horizons": {"1": {"sigma": 0.03}}}, "a"), lp)
    st = DR.score_ledger(now=datetime(2026, 12, 30, tzinfo=timezone.utc), fetch=lambda *a: ({}, {"X"}),
                         path=lp, scores_path=sp, completed=lambda b, n: b)
    assert st["fetch_errors"] == 1 and st["gave_up"] == 0
