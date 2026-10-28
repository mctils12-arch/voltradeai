"""
regime_engine.py — the single owner of regime state (docs/regime/FINDINGS.md
step 2). Pure arithmetic and the state machine on synthetic series; the I/O
layer with injected fetchers and temp paths. Nothing here touches the network.
"""
import json
import math
import os
import random
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, timedelta

import regime_engine as re_


def bdays(start="2015-01-02", n=10):
    d = date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def geometric(n, daily=0.0005, start=100.0, seed=None, noise=0.0):
    rnd = random.Random(seed)
    out = [start]
    for _ in range(n - 1):
        r = daily + (rnd.gauss(0, noise) if noise else 0.0)
        out.append(out[-1] * (1 + r))
    return out


def basket_from(gating, k=8, scale=1.0):
    return {f"S{j}": [c * scale * (1 + 0.01 * j) for c in gating] for j in range(k)}


class TestArithmetic(unittest.TestCase):
    def test_flag_parsing_env_and_config(self):
        self.assertFalse(re_.engine_allowed({}, {}))
        self.assertTrue(re_.engine_allowed({"VOLTRADE_REGIME_ENGINE": "1"}, {}))
        self.assertTrue(re_.engine_allowed({"VOLTRADE_REGIME_ENGINE": "TRUE"}, {}))
        self.assertFalse(re_.engine_allowed({"VOLTRADE_REGIME_ENGINE": "0"}, {"REGIME_ENGINE_ENABLED": False}))
        self.assertTrue(re_.engine_allowed({}, {"REGIME_ENGINE_ENABLED": True}))

    def test_sma_is_none_safe(self):
        self.assertIsNone(re_.sma([1.0, 2.0], 1, 3))
        self.assertIsNone(re_.sma([None, 2.0, 3.0], 2, 3))
        self.assertEqual(re_.sma([1.0, 2.0, 3.0], 2, 3), 2.0)

    def test_ewma_vol_is_annualized_and_none_safe(self):
        closes = [100.0, 101.0, 100.0, 101.0, 100.0]
        v = re_.ewma_vol_series(closes, 20)
        self.assertIsNone(v[0])
        self.assertGreater(v[-1], 0.0)
        self.assertAlmostEqual(v[1], abs(math.log(1.01)) * math.sqrt(252), places=9)
        v2 = re_.ewma_vol_series([100.0, None, 101.0], 20)
        self.assertIsNone(v2[1])

    def test_percentile_rank_and_quantize(self):
        self.assertEqual(re_.percentile_rank([1, 2, 3, 4], 2), 0.5)
        self.assertIsNone(re_.percentile_rank([], 2))
        self.assertEqual(re_.quantize_quarter(0.6), 0.5)
        self.assertEqual(re_.quantize_quarter(0.63), 0.75)
        self.assertEqual(re_.quantize_quarter(1.4), 1.0)

    def test_majority_needs_more_than_half(self):
        self.assertEqual(re_.majority(["A", "A", "A", "B", "B", "B"], "A"), "A")
        self.assertEqual(re_.majority(["B", "B", "B", "B", "A", "A"], "A"), "B")
        self.assertEqual(re_.majority([], "A"), "A")


class TestDecodeTables(unittest.TestCase):
    def test_decode_every_pair(self):
        for vs in ("LOW", "NORMAL", "ELEVATED"):
            self.assertEqual(re_.decode("UP", vs), "VOLATILE_BULL" if vs == "ELEVATED" else "CALM_BULL")
            self.assertEqual(re_.decode("NEUTRAL", vs), "STRESSED" if vs == "ELEVATED" else "CHOPPY")
            self.assertEqual(re_.decode("DOWN", vs), "BEAR")
        for ts in ("UP", "NEUTRAL", "DOWN"):
            self.assertEqual(re_.decode(ts, "CRISIS"), "CRISIS")

    def test_vol_state_thresholds_and_crisis_flag(self):
        self.assertEqual(re_.vol_state_of(None, None), "NORMAL")
        self.assertEqual(re_.vol_state_of(0.10, None), "LOW")
        self.assertEqual(re_.vol_state_of(0.50, 1.2), "NORMAL")
        self.assertEqual(re_.vol_state_of(0.85, 0.9), "ELEVATED")
        self.assertEqual(re_.vol_state_of(0.85, 1.05), "CRISIS")      # inversion
        self.assertEqual(re_.vol_state_of(0.96, None), "CRISIS")      # extreme own-history percentile

    def test_vol_state_hysteresis_holds_a_state_ten_points_past_its_entry(self):
        self.assertEqual(re_.vol_state_of(0.75, None, prev="ELEVATED"), "ELEVATED")   # 80 entered, held down to 70
        self.assertEqual(re_.vol_state_of(0.69, None, prev="ELEVATED"), "NORMAL")
        self.assertEqual(re_.vol_state_of(0.75, None, prev="NORMAL"), "NORMAL")       # not entered from below
        self.assertEqual(re_.vol_state_of(0.25, None, prev="LOW"), "LOW")             # 20 entered, held up to 30
        self.assertEqual(re_.vol_state_of(0.31, None, prev="LOW"), "NORMAL")
        self.assertEqual(re_.vol_state_of(0.90, 1.0, prev="CRISIS"), "CRISIS")        # 95 entered, held to 85 without inversion
        self.assertEqual(re_.vol_state_of(0.84, 0.9, prev="CRISIS"), "ELEVATED")
        self.assertEqual(re_.vol_state_of(0.75, 1.1, prev="CRISIS"), "CRISIS")        # inversion keeps crisis to the elevated floor

    def test_trend_and_breadth_states(self):
        self.assertEqual(re_.trend_state_of(1.0), "UP")
        self.assertEqual(re_.trend_state_of(0.75), "UP")
        self.assertEqual(re_.trend_state_of(0.5), "NEUTRAL")
        self.assertEqual(re_.trend_state_of(0.25), "DOWN")
        self.assertEqual(re_.fast_state_of(0.0), "DOWN")
        self.assertEqual(re_.fast_state_of(0.5), "MIXED")
        self.assertEqual(re_.fast_state_of(1.0), "UP")
        self.assertEqual(re_.breadth_state_of(0.7), "HEALTHY")
        self.assertEqual(re_.breadth_state_of(0.5), "NARROW")
        self.assertEqual(re_.breadth_state_of(0.3), "WEAK")
        self.assertIsNone(re_.breadth_state_of(None))

    def test_legacy_map_and_allowed_strategies(self):
        self.assertEqual(re_.legacy_label("CALM_BULL"), "BULL")
        self.assertEqual(re_.legacy_label("CRISIS"), "PANIC")
        self.assertEqual(re_.legacy_label("STRESSED"), "CAUTION")
        self.assertIn("dip_buy", re_.allowed_strategies("CALM_BULL", "HEALTHY", False))
        self.assertNotIn("dip_buy", re_.allowed_strategies("CALM_BULL", "NARROW", False))
        self.assertNotIn("momentum_long", re_.allowed_strategies("CALM_BULL", "HEALTHY", True))   # panic state
        self.assertEqual(re_.allowed_strategies("CRISIS", "HEALTHY", False), [])
        self.assertEqual(re_.allowed_strategies("BEAR", "HEALTHY", False), ["csp_reduced"])


class TestSignals(unittest.TestCase):
    def test_rising_market_votes_up_and_dial_is_one_in_calm(self):
        n = 700
        dates = bdays(n=n)
        g = geometric(n, 0.0004)
        sig = re_.signal_series(dates, g, basket_from(g), None, None)
        last = sig[-1]
        self.assertEqual(last["f_slow"], 1.0)
        self.assertEqual(last["f_fast"], 1.0)
        self.assertEqual(last["trend_score"], 1.0)
        self.assertEqual(last["breadth"], 1.0)
        self.assertEqual(last["m_vol"], 1.0)        # constant drift: σ̂ ≈ 0 -> dial capped at 1
        self.assertFalse(last["panic"])

    def test_falling_market_votes_down(self):
        n = 400
        dates = bdays(n=n)
        g = geometric(n, -0.002)
        sig = re_.signal_series(dates, g, basket_from(g), None, None)
        self.assertEqual(sig[-1]["f_slow"], 0.0)
        self.assertEqual(sig[-1]["f_fast"], 0.0)
        self.assertEqual(sig[-1]["breadth"], 0.0)

    def test_band_hysteresis_holds_a_vote_inside_one_percent(self):
        n = 300
        dates = bdays(n=n)
        g = [100.0] * n
        # oscillate ±0.5% around the flat average: inside the band -> the vote never flips from its initial 1
        for i in range(250, n):
            g[i] = 100.0 * (1.005 if i % 2 else 0.995)
        sig = re_.signal_series(dates, g, {}, None, None)
        self.assertEqual(sig[-1]["f_slow"], 1.0)
        # a 2% break below flips it, and a 0.5% bounce does not flip it back
        g2 = list(g)
        for i in range(280, n):
            g2[i] = 98.0 if i < 290 else 100.5
        sig2 = re_.signal_series(dates, g2, {}, None, None)
        self.assertEqual(sig2[289]["f_slow"], 0.0)
        self.assertEqual(sig2[-1]["f_slow"], 0.0)

    def test_vol_shock_raises_percentile_and_cuts_the_dial(self):
        n = 900
        dates = bdays(n=n)
        g = geometric(n, 0.0003, seed=1, noise=0.006)
        for i in range(880, n):                          # a shock: ±6% days
            g[i] = g[i - 1] * (1.06 if i % 2 else 0.94)
        sig = re_.signal_series(dates, g, basket_from(g), None, None)
        self.assertGreaterEqual(sig[-1]["vol_pct"], 0.95)
        self.assertLess(sig[-1]["m_vol"], 0.5)
        self.assertIsNotNone(sig[-1]["sigma_target"])
        self.assertTrue(all(s["m_vol"] <= 1.0 for s in sig))

    def test_vix_family_flags_and_vrp(self):
        n = 300
        dates = bdays(n=n)
        g = geometric(n, 0.0, seed=2, noise=0.01)
        vix = [20.0] * n
        vix3m = [22.0] * n
        vix[-1], vix3m[-1] = 30.0, 25.0
        sig = re_.signal_series(dates, g, {}, vix, vix3m)
        self.assertAlmostEqual(sig[-1]["ts"], 1.2, places=9)
        self.assertIsNotNone(sig[-1]["rv"])
        self.assertAlmostEqual(sig[-1]["vrp"], 30.0 - sig[-1]["rv"], places=9)
        self.assertGreaterEqual(sig[-1]["vix_pct_1y"], 0.99)

    def test_panic_state_needs_negative_two_year_return_and_top_quintile_variance(self):
        n = 1400
        dates = bdays(n=n)
        g = geometric(n, 0.0002, seed=3, noise=0.004)
        base = list(g)
        # crash the last 160 sessions: two-year return negative and variance in the top of its own history
        for i in range(n - 160, n):
            g[i] = g[i - 1] * (1 - 0.012 + (0.03 if i % 3 == 0 else 0.0))
        sig = re_.signal_series(dates, g, {}, None, None)
        self.assertTrue(sig[-1]["panic"])
        sig_calm = re_.signal_series(dates, base, {}, None, None)
        self.assertFalse(sig_calm[-1]["panic"])

    def test_breadth_needs_min_members_else_none(self):
        n = 300
        dates = bdays(n=n)
        g = geometric(n, 0.0004)
        sig = re_.signal_series(dates, g, basket_from(g, k=3), None, None)
        self.assertIsNone(sig[-1]["breadth"])
        self.assertEqual(sig[-1]["breadth_bin"], 1.0)


def _sig(trend_score, f_slow=1.0, f_fast=1.0, breadth_bin=1.0, m_vol=1.0, vol_pct=0.5, ts=None):
    return {"f_slow": f_slow, "f_fast": f_fast, "trend_score": trend_score, "breadth": breadth_bin,
            "breadth_bin": breadth_bin, "sigma": 0.15, "sigma_target": 0.18, "vol_pct": vol_pct, "m_vol": m_vol,
            "ts": ts, "rv": 12.0, "vrp": 5.0, "vix": 17.0, "vix_pct_1y": 0.4, "mom24m": 0.2, "var126_pct": 0.3, "panic": False}


class TestTransitions(unittest.TestCase):
    def test_three_of_six_holds_and_four_of_six_flips(self):
        up = _sig(1.0)
        down = _sig(0.0, f_slow=0.0, f_fast=0.0)
        series = [up] * 10 + [down] * 3 + [up] * 3
        states = re_.run_states(series)
        self.assertEqual(states[12]["composite"], "CALM_BULL")       # 3 of 6 -> hold
        self.assertEqual(states[12]["pending_transition"]["candidate"], "BEAR")
        series = [up] * 10 + [down] * 4
        states = re_.run_states(series)
        self.assertEqual(states[-1]["composite"], "BEAR")            # 4 of 6 -> flip
        self.assertEqual(states[-1]["days_in_regime"], 1)
        self.assertEqual(states[-1]["flips_12m"], 1)                 # one primary (slow-trend) flip
        self.assertEqual(states[-1]["composite_changes_12m"], 1)
        self.assertIsNone(states[-1]["pending_transition"])
        # the primary trend label is itself majority-filtered: 3 of 6 slow-leg DOWN days hold UP
        states = re_.run_states([up] * 10 + [down] * 3)
        self.assertEqual(states[-1]["trend_state"], "UP")
        self.assertEqual(states[-1]["trend_state_raw"], "DOWN")
        self.assertEqual(states[-1]["flips_12m"], 0)
        # the fast legs alone never change the label, only the exposure
        fast_down = _sig(0.5, f_slow=1.0, f_fast=0.0)
        states = re_.run_states([up] * 10 + [fast_down] * 10)
        self.assertEqual(states[-1]["composite"], "CALM_BULL")
        self.assertEqual(states[-1]["fast_state"], "DOWN")
        self.assertEqual(states[-1]["flips_12m"], 0)
        self.assertEqual(states[-1]["level"], 0.75)

    def test_downgrade_is_immediate_after_the_filter_and_upgrade_glides(self):
        up = _sig(1.0)
        down = _sig(0.0, f_slow=0.0, f_fast=0.0, breadth_bin=0.0)
        series = [up] * 10 + [down] * 4
        states = re_.run_states(series)
        self.assertEqual(states[-1]["level"], 0.0)                   # fast out once the 6-day filter clears
        series = [up] * 10 + [down] * 10 + [up] * 30
        states = re_.run_states(series)
        levels = [s["level"] for s in states[20:]]
        # upgrades: +0.25 at most every 5 sessions once the filter agrees -> no jump larger than a quarter
        self.assertTrue(all(b - a <= 0.25 + 1e-9 for a, b in zip(levels, levels[1:])))
        self.assertEqual(states[-1]["level"], 1.0)
        first_full = next(i for i, s in enumerate(states) if i > 20 and s["level"] == 1.0)
        # the filter clears on the 4th agreeing day (offset 3), +0.25 then, then +0.25 every 5 sessions: 3 + 15
        self.assertEqual(first_full - 20, 3 + 3 * 5)
        self.assertEqual(states[23]["level"], 0.25)
        self.assertEqual(states[27]["level"], 0.25)
        self.assertEqual(states[28]["level"], 0.5)

    def test_exposure_outputs_are_quantized_and_core_never_below_floor(self):
        s = _sig(1.0, m_vol=0.3)
        states = re_.run_states([s] * 8)
        self.assertEqual(states[-1]["exposure_multiplier"], 0.25)
        self.assertAlmostEqual(states[-1]["exposure_continuous"], 0.3, places=9)
        self.assertEqual(states[-1]["core_multiplier"], 0.5)
        for st_ in states:
            self.assertIn(st_["exposure_multiplier"], (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertGreaterEqual(st_["core_multiplier"], 0.5)

    def test_crisis_from_inversion_and_votes_agree(self):
        crisis = _sig(1.0, vol_pct=0.9, ts=1.1)
        states = re_.run_states([crisis] * 8)
        self.assertEqual(states[-1]["composite"], "CRISIS")
        self.assertEqual(states[-1]["votes_agree"], 1.0)


class TestStateObject(unittest.TestCase):
    def test_compute_state_is_json_serializable_and_complete(self):
        n = 700
        dates = bdays(n=n)
        g = geometric(n, 0.0004, seed=4, noise=0.005)
        vix = [18.0] * n
        vix3m = [20.0] * n
        state = re_.compute_state(dates, g, basket_from(g), vix, vix3m, gating="SPY",
                                  sources={"SPY": "alpaca", "^VIX": "yahoo"}, ebp={"date": "2026-09-01", "ebp": -0.3, "est_prob": 0.09})
        d = asdict(state)
        json.dumps(d, default=str)
        for k in ("trend_state", "fast_state", "vol_state", "breadth_state", "composite_regime", "legacy_label", "exposure_multiplier",
                  "core_multiplier", "allowed_strategies", "confidence", "days_in_regime", "pending_transition",
                  "flips_12m", "alarm", "panic_state", "stress_flags", "ebp", "inputs", "data_quality"):
            self.assertIn(k, d)
        self.assertEqual(d["as_of"], dates[-1])
        self.assertEqual(d["composite_regime"], "CALM_BULL")
        self.assertEqual(d["legacy_label"], "BULL")
        self.assertFalse(d["stress_flags"]["vix_vix3m_inverted"])
        self.assertEqual(d["ebp"]["ebp"], -0.3)
        self.assertEqual(d["inputs"]["sources"]["SPY"], "alpaca")
        self.assertLessEqual(d["confidence"], 1.0)

    def test_quality_factor_lowers_confidence_never_raises_exposure(self):
        n = 400
        dates = bdays(n=n)
        g = geometric(n, 0.0004)
        full = re_.compute_state(dates, g, basket_from(g), [18.0] * n, [20.0] * n, quality_factor=1.0)
        degraded = re_.compute_state(dates, g, basket_from(g), None, None, quality_factor=0.7)
        self.assertLess(degraded.confidence, full.confidence)
        self.assertLessEqual(degraded.exposure_multiplier, full.exposure_multiplier)
        self.assertEqual(re_.quality_factor_of({"^VIX": "missing"}, dates, today=date.fromisoformat(dates[-1])), 0.7)
        self.assertEqual(re_.quality_factor_of({"^VIX": "yahoo"}, dates, today=date.fromisoformat(dates[-1]) + timedelta(days=10)), 0.5)

    def test_exposure_paths_align_to_dates(self):
        n = 300
        dates = bdays(n=n)
        g = geometric(n, 0.0004)
        c, q, core, comps = re_.exposure_paths(dates, g, basket_from(g))
        self.assertEqual(len(c), n)
        self.assertEqual(len(comps), n)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in c))
        self.assertTrue(all(x >= 0.5 for x in core))


class TestIO(unittest.TestCase):
    def test_fetch_closes_alpaca_first_then_yahoo_for_the_rest(self):
        def alp(syms, d):
            return {"SPY": [["2024-01-02", 470.0], ["2024-01-03", 471.0]]}

        def yah(sym, d):
            if sym == "^VIX":
                return {"2024-01-02": 13.0, "2024-01-03": 14.0}
            raise RuntimeError("down")
        closes, sources = re_.fetch_closes(["SPY", "XLK", "^VIX"], 100, fetch_alpaca=alp, fetch_yahoo=yah)
        self.assertEqual(sources, {"SPY": "alpaca", "XLK": "missing", "^VIX": "yahoo"})
        self.assertEqual(closes["SPY"]["2024-01-03"], 471.0)
        self.assertNotIn("XLK", closes)

    def test_align_forward_fills_onto_gating_dates(self):
        dates, out = re_.align({"SPY": {"d1": 1.0, "d2": 2.0, "d3": 3.0}, "X": {"d2": 5.0}}, "SPY")
        self.assertEqual(dates, ["d1", "d2", "d3"])
        self.assertEqual(out["X"], [None, 5.0, 5.0])

    def test_build_inputs_uses_injected_fetchers_and_caches_for_the_day(self):
        n = 30
        dates = bdays(n=n)
        rows = [[d, 100.0 + i] for i, d in enumerate(dates)]

        def alp(syms, d):
            return {s: rows for s in syms}

        def yah(sym, d):
            return {dd: 15.0 for dd in dates}
        with tempfile.TemporaryDirectory() as tmp:
            orig = re_._cache_dir
            re_._cache_dir = lambda: tmp
            try:
                inp = re_.build_inputs("SPY", days=100, fetch_alpaca=alp, fetch_yahoo=yah, ebp=False)
                self.assertFalse(inp["cache_hit"])
                self.assertEqual(inp["dates"], dates)
                self.assertEqual(len(inp["basket"]), len(re_.BREADTH_BASKET))
                self.assertEqual(inp["vix"][-1], 15.0)
                inp2 = re_.build_inputs("SPY", days=100, fetch_alpaca=lambda s, d: (_ for _ in ()).throw(RuntimeError("no")),
                                        fetch_yahoo=yah, ebp=False)
                self.assertTrue(inp2["cache_hit"])
            finally:
                re_._cache_dir = orig

    def test_fetch_ebp_parses_the_fed_csv_and_caches(self):
        csv = "date,gz_spread,ebp,est_prob\n2026-08-01,1.5,-0.2,0.08\n2026-09-01,1.6,-0.39,0.09\n"
        with tempfile.TemporaryDirectory() as tmp:
            orig = re_._cache_dir
            re_._cache_dir = lambda: tmp
            try:
                row = re_.fetch_ebp(fetch=lambda url: csv)
                self.assertEqual(row["date"], "2026-09-01")
                self.assertAlmostEqual(row["ebp"], -0.39)
                row2 = re_.fetch_ebp(fetch=lambda url: (_ for _ in ()).throw(RuntimeError("down")))
                self.assertEqual(row2["ebp"], -0.39)       # served from the cache
            finally:
                re_._cache_dir = orig

    def test_save_load_history_and_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "state.json")
            hp = os.path.join(tmp, "hist.jsonl")
            s1 = {"as_of": "2026-10-27", "composite_regime": "CALM_BULL"}
            r = re_.save_state(s1, sp, hp)
            self.assertTrue(r["appended"])
            r = re_.save_state(dict(s1), sp, hp)
            self.assertFalse(r["appended"])                 # same day, same regime: no new line
            r = re_.save_state({"as_of": "2026-10-28", "composite_regime": "CALM_BULL"}, sp, hp)
            self.assertTrue(r["appended"])
            self.assertEqual(len(re_.read_history(hp)), 2)
            self.assertEqual(re_.load_state(sp)["as_of"], "2026-10-28")
            self.assertEqual(re_.state_age_market_days({"as_of": "2026-10-23"}, today=date(2026, 10, 27)), 2)
            self.assertIsNone(re_.state_age_market_days(None))

    def test_update_state_never_raises_and_marks_stale_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "state.json")
            hp = os.path.join(tmp, "hist.jsonl")
            orig_paths = re_._paths
            re_._paths = lambda: (sp, hp)
            try:
                out = re_.update_state(build=lambda gating: (_ for _ in ()).throw(RuntimeError("network")))
                self.assertIn("error", out)
                self.assertTrue(out["stale"])
                n = 400
                dates = bdays(n=n)
                g = geometric(n, 0.0004)

                def build(gating):
                    return {"dates": dates, "gating": g, "basket": basket_from(g), "vix": [18.0] * n, "vix3m": [20.0] * n,
                            "sources": {"SPY": "alpaca", "^VIX": "yahoo", "^VIX3M": "yahoo"}, "ebp": None, "cache_hit": False}
                out = re_.update_state(build=build)
                self.assertEqual(out["as_of"], dates[-1])
                self.assertTrue(out["persisted"]["appended"])
                again = re_.update_state(build=build)
                self.assertTrue(again.get("cache_hit"))     # bar date unchanged -> served from the persisted state
                self.assertEqual(re_.load_state(sp)["composite_regime"], out["composite_regime"])
            finally:
                re_._paths = orig_paths

    def test_rpc_respects_the_flag_and_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig_paths = re_._paths
            orig_env = os.environ.get("VOLTRADE_REGIME_ENGINE")
            re_._paths = lambda: (os.path.join(tmp, "s.json"), os.path.join(tmp, "h.jsonl"))
            try:
                os.environ["VOLTRADE_REGIME_ENGINE"] = "0"
                orig_allowed = re_.engine_allowed
                re_.engine_allowed = lambda env=None, config=None: False
                try:
                    out = re_.regime_state_rpc()
                finally:
                    re_.engine_allowed = orig_allowed
                self.assertFalse(out["enabled"])
                self.assertIsNone(out["state"])
            finally:
                re_._paths = orig_paths
                if orig_env is None:
                    os.environ.pop("VOLTRADE_REGIME_ENGINE", None)
                else:
                    os.environ["VOLTRADE_REGIME_ENGINE"] = orig_env


if __name__ == "__main__":
    unittest.main()


class TestDaemonRoute(unittest.TestCase):
    def test_daemon_route_resolves_to_the_regime_entry_point(self):
        # READ BEFORE WRITE #4: a Python entry point bot.ts calls is registered in the daemon's table
        import importlib
        import voltrade_daemon
        routes = voltrade_daemon.RPCDispatcher()._routes
        self.assertEqual(routes["regime_state"], ("regime_engine", "regime_state_rpc"))
        mod = importlib.import_module("regime_engine")
        self.assertTrue(callable(getattr(mod, "regime_state_rpc")))

    def test_config_mirror_exists_and_defaults_on(self):
        from system_config import BASE_CONFIG
        self.assertIn("REGIME_ENGINE_ENABLED", BASE_CONFIG)
        self.assertTrue(BASE_CONFIG["REGIME_ENGINE_ENABLED"])   # 2026-10-28 human directive: default flipped ON (was False)
        from storage_config import REGIME_STATE_PATH, REGIME_HISTORY_PATH
        self.assertTrue(REGIME_STATE_PATH.endswith("voltrade_regime_state.json"))
        self.assertTrue(REGIME_HISTORY_PATH.endswith("voltrade_regime_history.jsonl"))
