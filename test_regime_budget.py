"""
Regime risk budget (2026-11-06, human refocus spec section 3 step 1: "market regime:
VIX level + term structure, broad trend, breadth -> overall risk budget / allowed
exposure"). regime_engine.risk_budget_series turns four drivers into one budget in
[0, 1] next to the unchanged labels; tiered_strategy.get_regime_caps applies it to the
tier engine's satellite headroom (never to the passive floor). Pinned here: the driver
scores, the weighted mean and its renormalization, fast-out/slow-in, a causal VIX
anchor, labels unchanged, persistence, the view/flag/staleness rules, the bounded cap,
its counterfactual row, and the tier engine's use of it.
"""
import json
import math
import os
import random
import tempfile
import time
import unittest
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import regime_engine as re_
import tiered_strategy as ts
from system_config import BASE_CONFIG

P = re_.budget_params()
TODAY = datetime.now(timezone.utc).date().isoformat()


def bdays(n, start="2015-01-02"):
    d = date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def sig(trend=1.0, ts_=0.85, vix=15.0, breadth=0.8, f_slow=1.0, f_fast=1.0, up=10, n=13):
    return {"trend_score": trend, "ts": ts_, "vix": vix, "breadth": breadth, "f_slow": f_slow,
            "f_fast": f_fast, "breadth_up": up, "breadth_n": n}


def small_params(**over):
    p = re_.budget_params()
    p.update({"vix_anchor_min_obs": 3, "vix_anchor_window": 5})
    p.update(over)
    return p


def budget_state(budget=0.5, as_of=TODAY, band=None, **over):
    st = {"as_of": as_of, "composite_regime": "STRESSED", "legacy_label": "CAUTION", "exposure_multiplier": 0.5,
          "allowed_strategies": ["csp_reduced"], "params_version": re_.PARAMS_VERSION,
          "risk_budget": budget, "budget_version": re_.BUDGET_VERSION,
          "risk_budget_detail": {"budget": budget, "raw": budget, "band": band or re_.budget_band(budget),
                                 "summary": f"Risk budget {budget:.2f}: test", "drivers": [], "missing": []}}
    st.update(over)
    return st


class TestParamsAndScores(unittest.TestCase):
    def test_params_come_from_base_config(self):
        self.assertIs(BASE_CONFIG["REGIME_BUDGET_DEFAULT"], True)
        self.assertEqual(P["weights"], BASE_CONFIG["REGIME_BUDGET_PARAMS"]["weights"])
        self.assertAlmostEqual(sum(P["weights"].values()), 1.0, places=9)
        self.assertEqual(set(P["weights"]), set(re_.BUDGET_DRIVERS))
        self.assertIsNone(re_.budget_params({}))
        P2 = re_.budget_params()
        P2["weights"]["trend"] = 9.0                        # a copy: never mutates BASE_CONFIG
        self.assertEqual(BASE_CONFIG["REGIME_BUDGET_PARAMS"]["weights"]["trend"], 0.30)

    def test_driver_scores(self):
        _, s = re_.budget_scores(sig(ts_=0.88, breadth=0.60, vix=15.0), 17.0, P)
        self.assertEqual((s["term_structure"], s["breadth"], s["vix_level"], s["trend"]), (1.0, 1.0, 1.0, 1.0))
        _, s = re_.budget_scores(sig(ts_=1.00, breadth=0.40, vix=34.0, trend=0.5), 17.0, P)
        self.assertAlmostEqual(s["term_structure"], 0.5)     # flat curve: half
        self.assertAlmostEqual(s["breadth"], 0.5)            # WEAK boundary: half
        self.assertAlmostEqual(s["vix_level"], 0.5)          # VIX 2x its median: half the risk per dollar
        self.assertAlmostEqual(s["trend"], 0.5)
        _, s = re_.budget_scores(sig(ts_=1.20, breadth=0.10, vix=80.0, trend=0.0), 17.0, P)
        self.assertEqual((s["term_structure"], s["breadth"], s["trend"]), (0.0, 0.0, 0.0))
        self.assertGreater(s["vix_level"], 0.0)              # VIX alone never zeroes the budget
        _, s = re_.budget_scores(sig(ts_=None, breadth=None, vix=None, trend=None), 17.0, P)
        self.assertEqual(set(s.values()), {None})
        _, s = re_.budget_scores(sig(vix=30.0), None, P)     # no anchor yet -> VIX level unavailable
        self.assertIsNone(s["vix_level"])

    def test_bands(self):
        self.assertEqual([re_.budget_band(x) for x in (1.0, 0.8, 0.79, 0.5, 0.49, 0.25, 0.24, 0.0)],
                         ["full", "full", "reduced", "reduced", "defensive", "defensive", "minimal", "minimal"])
        self.assertIsNone(re_.budget_band(None))


class TestSeries(unittest.TestCase):
    def test_weighted_mean_of_the_driver_scores(self):
        p = small_params()
        sigs = [sig(vix=17.0) for _ in range(5)] + [sig(trend=0.5, ts_=1.0, vix=34.0, breadth=0.4)]
        out = re_.risk_budget_series(sigs, p)
        w = p["weights"]
        expect = w["trend"] * 0.5 + w["term_structure"] * 0.5 + w["vix_level"] * 0.5 + w["breadth"] * 0.5
        self.assertAlmostEqual(out["raw"][-1], expect, places=9)
        self.assertAlmostEqual(out["budget"][-1], expect, places=9)     # a fall applies at once
        last = out["last"]
        self.assertAlmostEqual(sum(w2 * (1 - last["scores"][k]) for k, w2 in last["weights"].items()), 1 - expect, places=9)

    def test_missing_drivers_are_left_out_and_the_rest_reweighted(self):
        p = small_params()
        out = re_.risk_budget_series([sig(ts_=None, vix=None, trend=1.0, breadth=0.4)], p)
        w = p["weights"]
        self.assertAlmostEqual(out["raw"][0], (w["trend"] * 1.0 + w["breadth"] * 0.5) / (w["trend"] + w["breadth"]))
        self.assertEqual(out["last"]["missing"], ["term_structure", "vix_level"])
        self.assertEqual(out["last"]["weights"]["vix_level"], 0.0)
        none = re_.risk_budget_series([sig(ts_=None, vix=None, trend=None, breadth=None)], p)
        self.assertIsNone(none["budget"][0])

    def test_fast_out_slow_in(self):
        p = small_params()
        calm, stress = sig(vix=15.0), sig(trend=0.0, ts_=1.2, vix=60.0, breadth=0.1)
        out = re_.risk_budget_series([calm] * 4 + [stress] + [calm] * 12, p)
        b, raw = out["budget"], out["raw"]
        self.assertLess(b[4], 0.2)                          # the cut is immediate
        for i in range(5, len(b)):
            self.assertLessEqual(b[i] - b[i - 1], p["rise_per_session"] + 1e-12)   # the recovery glides
            self.assertLessEqual(b[i], raw[i] + 1e-12)
        self.assertAlmostEqual(b[-1], raw[-1])              # ... and gets back to the reading

    def test_vix_anchor_is_causal_and_a_rolling_median(self):
        p = small_params()
        vix = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0]
        out = re_.risk_budget_series([sig(vix=v) for v in vix], p)
        # day 5 anchors on the 5 sessions BEFORE it (median 30), never on its own 100
        self.assertEqual(out["last"]["vix_anchor"], 30.0)
        self.assertAlmostEqual(out["last"]["scores"]["vix_level"], 0.3)
        out2 = re_.risk_budget_series([sig(vix=v) for v in vix + [100.0]], p)
        self.assertEqual(out2["last"]["vix_anchor"], 40.0)  # window of 5: 20,30,40,50,100 -> 40
        self.assertIsNone(re_.risk_budget_series([sig(vix=v) for v in vix[:3]], p)["last"]["vix_anchor"])

    def test_worse_readings_never_raise_the_budget(self):
        rnd = random.Random(7)
        p = small_params()
        hist = [sig(vix=17.0)] * 5
        for _ in range(300):
            a = sig(trend=rnd.random(), ts_=0.8 + 0.4 * rnd.random(), vix=10 + 50 * rnd.random(), breadth=rnd.random())
            worse = dict(a, trend_score=a["trend_score"] * rnd.random(), ts=a["ts"] + 0.2 * rnd.random(),
                         vix=a["vix"] * (1 + rnd.random()), breadth=a["breadth"] * rnd.random())
            ra = re_.risk_budget_series(hist + [a], p)["raw"][-1]
            rw = re_.risk_budget_series(hist + [worse], p)["raw"][-1]
            self.assertLessEqual(rw, ra + 1e-12)
            self.assertTrue(0.0 <= rw <= 1.0)

    def test_without_a_parameter_table_there_is_no_budget(self):
        out = re_.risk_budget_series([sig()] * 3, {})
        self.assertEqual(out["budget"], [None, None, None])
        self.assertIsNone(out["last"])


def _series_state(n=700, vix=18.0, vix3m=20.0, budget_cfg=None, crash_tail=0):
    dates = bdays(n)
    rnd = random.Random(4)
    g = [100.0]
    for i in range(n - 1):
        drift = -0.01 if i >= n - 1 - crash_tail else 0.0004
        g.append(g[-1] * (1 + drift + rnd.gauss(0, 0.005)))
    basket = {f"S{j}": [c * (1 + 0.01 * j) for c in g] for j in range(8)}
    vx = vix if isinstance(vix, list) else [vix] * n
    v3 = vix3m if isinstance(vix3m, list) else [vix3m] * n
    return re_.compute_state(dates, g, basket, vx, v3, gating="SPY", budget_cfg=budget_cfg)


class TestState(unittest.TestCase):
    def test_state_carries_the_budget_and_plain_english_drivers(self):
        d = asdict(_series_state())
        self.assertEqual(d["budget_version"], re_.BUDGET_VERSION)
        self.assertTrue(0.0 <= d["risk_budget"] <= 1.0)
        det = d["risk_budget_detail"]
        self.assertEqual([x["name"] for x in det["drivers"]], list(re_.BUDGET_DRIVERS))
        self.assertEqual(det["budget"], d["risk_budget"])
        self.assertTrue(det["summary"].startswith(f"Risk budget {d['risk_budget']:.2f} ("))
        for x in det["drivers"]:
            self.assertTrue(x["reason"] and x["short"])
            self.assertAlmostEqual(x["cut"], x["weight"] * (1 - x["score"]), places=3)
        self.assertAlmostEqual(sum(x["cut"] for x in det["drivers"]), 1 - det["raw"], places=3)
        json.dumps(d)

    def test_inverted_term_structure_is_named_in_the_summary(self):
        n = 700
        vix = [18.0] * (n - 1) + [40.0]
        vix3m = [20.0] * (n - 1) + [33.0]                    # VIX/VIX3M 1.21 on the last day
        det = asdict(_series_state(n, vix, vix3m))["risk_budget_detail"]
        ts_d = [x for x in det["drivers"] if x["name"] == "term_structure"][0]
        self.assertEqual(ts_d["score"], 0.0)
        self.assertIn("VIX term structure inverted (1.21)", det["summary"])
        self.assertIn("2.2x its", det["summary"])            # VIX 40 vs its ~18 median

    def test_labels_and_exposure_are_unchanged_by_the_budget(self):
        on, off = asdict(_series_state(crash_tail=30)), asdict(_series_state(crash_tail=30, budget_cfg={}))
        self.assertIsNone(off["risk_budget"])
        self.assertIsNotNone(on["risk_budget"])
        for k in ("composite_regime", "legacy_label", "trend_state", "vol_state", "breadth_state", "exposure_multiplier",
                  "exposure_continuous", "core_multiplier", "allowed_strategies", "confidence", "stress_flags"):
            self.assertEqual(on[k], off[k], k)

    def test_a_pre_budget_persisted_state_recomputes_once(self):
        n = 400
        dates = bdays(n)
        g = [100.0 * (1.0004 ** i) for i in range(n)]

        def build(gating):
            return {"dates": dates, "gating": g, "basket": {f"S{j}": list(g) for j in range(8)}, "vix": [18.0] * n,
                    "vix3m": [20.0] * n, "sources": {"SPY": "alpaca"}, "ebp": None, "cache_hit": False}
        with tempfile.TemporaryDirectory() as tmp:
            sp, hp = os.path.join(tmp, "s.json"), os.path.join(tmp, "h.jsonl")
            with open(sp, "w") as f:      # what the live state looked like before this change
                json.dump({"as_of": dates[-1], "gating_index": "SPY", "params_version": re_.PARAMS_VERSION,
                           "composite_regime": "CALM_BULL"}, f)
            with patch.object(re_, "_paths", lambda: (sp, hp)):
                first = re_.update_state(build=build)
                self.assertFalse(first.get("cache_hit"))
                self.assertIsNotNone(first["risk_budget"])
                again = re_.update_state(build=build)
                self.assertTrue(again.get("cache_hit"))
                self.assertEqual(re_.load_state(sp)["risk_budget"], first["risk_budget"])

    def test_ml_snapshot_carries_the_budget_compactly(self):
        st = asdict(_series_state())
        snap = re_.ml_snapshot(st)
        self.assertEqual(snap["risk_budget"], round(st["risk_budget"], 4))
        self.assertLess(len(json.dumps(snap)), 1000)

    def test_risk_budget_paths_match_the_state(self):
        n = 700
        dates = bdays(n)
        g = [100.0 * (1.0004 ** i) for i in range(n)]
        basket = {f"S{j}": list(g) for j in range(8)}
        b, raw = re_.risk_budget_paths(dates, g, basket, [18.0] * n, [20.0] * n)
        st = re_.compute_state(dates, g, basket, [18.0] * n, [20.0] * n)
        self.assertEqual(len(b), n)
        self.assertAlmostEqual(b[-1], st.risk_budget, places=4)


class TestViewFlagAndConsumer(unittest.TestCase):
    def setUp(self):
        re_._STATE_CACHE.update(at=0.0, state=None)

    def tearDown(self):
        re_._STATE_CACHE.update(at=0.0, state=None)

    def test_flag_default_on_and_explicit_off_wins(self):
        self.assertTrue(re_.budget_enabled({}))
        self.assertFalse(re_.budget_enabled({}, {}))
        for v in ("off", "0", "", "false", " OFF "):
            self.assertFalse(re_.budget_enabled({"VOLTRADE_REGIME_BUDGET": v}), v)
        self.assertEqual(re_.FLAG_DEFAULTS["VOLTRADE_REGIME_BUDGET"], "REGIME_BUDGET_DEFAULT")

    def test_view_applied_only_when_on_and_fresh(self):
        v = re_.risk_budget_view(budget_state(0.6), env={})
        self.assertEqual((v["budget"], v["band"], v["applied"], v["enabled"], v["stale"]), (0.6, "reduced", True, True, False))
        self.assertEqual(v["consumers"], ["tier_exposure_cap"])
        off = re_.risk_budget_view(budget_state(0.6), env={"VOLTRADE_REGIME_BUDGET": "off"})
        self.assertEqual((off["applied"], off["enabled"]), (False, False))
        eng_off = re_.risk_budget_view(budget_state(0.6), env={"VOLTRADE_REGIME_ENGINE": "0"})
        self.assertFalse(eng_off["applied"])
        old = re_.risk_budget_view(budget_state(0.6, as_of="2026-10-01"), env={}, today="2026-11-06")
        self.assertEqual((old["stale"], old["applied"]), (True, False))
        self.assertTrue(re_.risk_budget_view(budget_state(0.6, error="network"), env={})["stale"])
        self.assertIsNone(re_.risk_budget_view({"as_of": TODAY, "composite_regime": "CALM_BULL"}))
        self.assertIsNone(re_.risk_budget_view(None))

    def test_risk_budget_for(self):
        def fresh(env, st):
            re_._STATE_CACHE.update(at=0.0, state=None)   # cached_state keeps a warm 60 s cache
            return re_.risk_budget_for(env=env, loader=lambda: st)
        self.assertEqual(fresh({}, budget_state(0.4))["budget"], 0.4)
        self.assertIsNone(fresh({"VOLTRADE_REGIME_BUDGET": "off"}, budget_state(0.4)))
        self.assertIsNone(fresh({"VOLTRADE_REGIME_ENGINE": "off"}, budget_state(0.4)))
        self.assertIsNone(fresh({}, budget_state(0.4, as_of="2026-01-02")))   # stale: legacy, never a silent cut
        self.assertIsNone(fresh({}, {}))

    def test_passive_floor_tickers_follow_the_floor_mode(self):
        self.assertEqual(re_.passive_floor_tickers({"FLOOR_MODE": "vol_managed_spy", "FLOOR_VM_TICKER": "SPY",
                                                    "FLOOR_VM_LEVERED_TICKER": "SSO"}), {"SPY", "SSO"})
        basket = re_.passive_floor_tickers({"FLOOR_MODE": "basket", "FLOOR_BASKET": BASE_CONFIG["FLOOR_BASKET"],
                                            "FLOOR_TICKER": "QQQ", "DEFENSIVE_FLOOR_TICKER": "GLD"})
        self.assertEqual(basket, {"QQQ", "SMH", "KWEB", "VXUS", "GLD"})
        self.assertEqual(re_.passive_floor_tickers(), {"SPY", "SSO"} if BASE_CONFIG["FLOOR_MODE"] == "vol_managed_spy" else basket)


VM = {"FLOOR_MODE": "vol_managed_spy", "FLOOR_VM_TICKER": "SPY", "FLOOR_VM_LEVERED_TICKER": "SSO"}
POS = [{"symbol": "SPY", "market_value": 60_000}, {"symbol": "SSO", "market_value": 25_000},
       {"symbol": "AAPL", "market_value": 3_000}, {"symbol": "IREN261120P00045000", "market_value": -500}]


class TestBudgetedCap(unittest.TestCase):
    def cap(self, budget, cap=0.95, env=None, positions=POS, state=None, **kw):
        re_._STATE_CACHE.update(at=0.0, state=None)           # cached_state keeps a warm 60 s cache
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cf.jsonl")
            st = state if state is not None else budget_state(budget)
            out = re_.budget_exposure_cap(cap, positions, 100_000, env=env if env is not None else {},
                                          loader=lambda: st, config=VM, path=path, today=TODAY, **kw)
            rows = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
        return out, rows

    def setUp(self):
        re_._CF.update(day=None, keys=set(), rows=0, dropped=0)

    def tearDown(self):
        re_._STATE_CACHE.update(at=0.0, state=None)

    def test_scales_only_the_room_above_the_floor(self):
        out, rows = self.cap(0.5)
        self.assertEqual(out["passive_pct"], 0.85)
        self.assertEqual(out["headroom_legacy"], 0.10)
        self.assertAlmostEqual(out["cap"], 0.90)             # 0.85 + 0.10 x 0.5
        self.assertAlmostEqual(out["headroom"], 0.05)
        self.assertEqual((out["applied"], out["cap_legacy"], out["band"]), (True, 0.95, "reduced"))
        self.assertIn("satellite room 10.0% -> 5.0%", out["summary"])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["kind"], r["sleeve"], r["key"]), ("sizing", "tier_exposure_cap", "tier_exposure_cap"))
        self.assertEqual(r["legacy"], {"multiplier": 1.0, "cap": 0.95})
        self.assertEqual((r["engine"]["cap"], r["engine"]["applied"], r["engine"]["budget"]), (0.9, True, 0.5))
        self.assertAlmostEqual(r["engine"]["multiplier"], round(0.90 / 0.95, 3))
        self.assertTrue(r["differs"])

    def test_bounded_between_the_floor_and_the_legacy_cap(self):
        self.assertAlmostEqual(self.cap(0.0)[0]["cap"], 0.85)    # never below the floor
        self.assertAlmostEqual(self.cap(1.0)[0]["cap"], 0.95)    # never above the legacy cap
        low_cap, rows = self.cap(0.3, cap=0.60)                  # a legacy cap already under the floor: untouched
        self.assertEqual(low_cap["cap"], 0.60)
        self.assertFalse(rows[0]["differs"])
        for b in (0.0, 0.1, 0.37, 0.8, 1.0):
            c = self.cap(b)[0]["cap"]
            self.assertTrue(0.85 - 1e-9 <= c <= 0.95 + 1e-9, (b, c))

    def test_off_stale_or_missing_leaves_the_cap_and_logs_the_would_be(self):
        off, rows = self.cap(0.5, env={"VOLTRADE_REGIME_BUDGET": "off"})
        self.assertEqual((off["cap"], off["applied"]), (0.95, False))
        self.assertAlmostEqual(off["would_cap"], 0.90)
        self.assertEqual((rows[0]["engine"]["applied"], rows[0]["engine"]["cap"]), (False, 0.9))
        stale, _ = self.cap(0.5, state=budget_state(0.5, as_of="2026-01-02"))
        self.assertEqual((stale["cap"], stale["applied"], stale["stale"]), (0.95, False, True))
        missing, rows = self.cap(0.5, state={"as_of": TODAY, "composite_regime": "CALM_BULL"})
        self.assertEqual((missing["cap"], missing["applied"]), (0.95, False))
        self.assertEqual(rows, [])
        re_._STATE_CACHE.update(at=0.0, state=None)
        self.assertEqual(re_.budget_exposure_cap(0.95, POS, 0, env={}, loader=lambda: budget_state(0.5))["cap"], 0.95)

    def test_no_floor_held_means_the_whole_cap_is_satellite_room(self):
        out, _ = self.cap(0.5, positions=[{"symbol": "AAPL", "market_value": 10_000}])
        self.assertEqual(out["passive_pct"], 0.0)
        self.assertAlmostEqual(out["cap"], 0.475)

    def test_never_raises_on_bad_inputs(self):
        re_._STATE_CACHE.update(at=0.0, state=None)
        out = re_.budget_exposure_cap(0.95, [{"symbol": "SPY", "market_value": "x"}], 100_000, env={},
                                      loader=lambda: budget_state(0.5), config=VM, path=os.devnull)
        self.assertEqual((out["cap"], out["applied"]), (0.95, False))
        out = re_.budget_exposure_cap("bad", POS, 100_000, env={}, loader=lambda: budget_state(0.5), config=VM)
        self.assertEqual(out["cap"], "bad")


class TestTierEngineWiring(unittest.TestCase):
    """tiered_strategy reads the budget through regime_engine.cached_state (the persisted state)."""

    def setUp(self):
        re_._STATE_CACHE.update(at=time.time(), state=budget_state(0.5))
        self._env = patch.dict(os.environ, {"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_BUDGET": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        re_._STATE_CACHE.update(at=0.0, state=None)

    def ctx(self, positions):
        return ts.TierContext(equity=100_000, peak_equity=100_000, positions=positions, vxx_ratio=0.9, spy_vs_ma50=1.03)

    def floor_positions(self, satellite_mv):
        f = sorted(re_.passive_floor_tickers())[0]
        return [{"symbol": f, "market_value": 85_000}, {"symbol": "AAPL", "market_value": satellite_mv}]

    def test_get_regime_caps_without_positions_is_unchanged(self):
        caps = ts.get_regime_caps(0.9, 1.03, 100_000)
        self.assertNotIn("RISK_BUDGET_CAP", caps)
        self.assertEqual(caps["MAX_TOTAL_EXPOSURE"], 0.95)

    def test_get_regime_caps_with_positions_is_budgeted(self):
        caps = ts.get_regime_caps(0.9, 1.03, 100_000, positions=self.floor_positions(0))
        self.assertEqual(caps["MAX_TOTAL_EXPOSURE_LEGACY"], 0.95)
        self.assertAlmostEqual(caps["MAX_TOTAL_EXPOSURE"], 0.90)
        self.assertTrue(caps["RISK_BUDGET_CAP"]["applied"])

    def test_master_kill_switch_uses_the_budgeted_cap_and_says_so(self):
        killed, reason = ts.master_kill_switch(self.ctx(self.floor_positions(6_000)))   # 91% >= 90%
        self.assertTrue(killed)
        self.assertTrue(reason.startswith("Portfolio invested 91.0% >= 90%"))
        self.assertIn("[risk budget 0.50 cut the cap from 95% to 90.0%]", reason)
        import shadow_portfolio as sp
        self.assertEqual(sp._masterkill_reason_category(reason), "exposure_ceiling")
        with patch.dict(os.environ, {"VOLTRADE_REGIME_BUDGET": "off"}):
            self.assertEqual(ts.master_kill_switch(self.ctx(self.floor_positions(6_000))), (False, "ok"))

    def test_allocator_scales_proposals_to_the_budgeted_headroom(self):
        acts = [ts.TierAction(tier=1, action="SELL_CSP", ticker=t, strategy="csp", size_pct=0.04, reason="t1")
                for t in ("AMD", "INTC", "F")]
        out = ts._enforce_exposure_cap(self.ctx(self.floor_positions(0)), acts)
        self.assertAlmostEqual(sum(a.size_pct for a in out), 0.05, places=6)   # 0.90 - 0.85, not 0.10
        self.assertIn("risk budget 0.50", out[0].reason)


class TestStudyHelpers(unittest.TestCase):
    """scripts/regime_budget_study.py's statistics (the gate-2 evidence in the experiments entry)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "regime_budget_study.py")
        spec = importlib.util.spec_from_file_location("regime_budget_study", path)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_ranks_and_spearman(self):
        m = self.m
        self.assertEqual(m.ranks([10, 30, 20, 20]), [1.0, 4.0, 2.5, 2.5])
        self.assertAlmostEqual(m.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(m.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(m.spearman([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(m.t_of_r(0.5, 102), 0.5 * math.sqrt(100 / 0.75))

    def test_forward_outcomes(self):
        px = [100.0, 110.0, 99.0, 121.0]
        rv, mdd, ret = self.m.forward_outcomes(px, 0, 3)
        self.assertAlmostEqual(mdd, 99.0 / 110.0 - 1)      # the peak inside the window counts
        self.assertAlmostEqual(ret, 0.21)
        self.assertGreater(rv, 0)
        self.assertIsNone(self.m.forward_outcomes(px, 1, 3))   # window runs past the data

    def test_overlay_lag_and_costs(self):
        px = [100.0, 100.0, 110.0, 121.0]
        daily = self.m.overlay(px, [1.0, 1.0, 1.0, 1.0], lag=2, cost_bp=0.0)
        self.assertEqual([round(x, 6) for x in daily], [0.1, 0.1])           # decided at t, earns t+1 -> t+2
        costly = self.m.overlay(px, [1.0, 1.0, 1.0, 1.0], lag=2, cost_bp=10.0)
        self.assertAlmostEqual(costly[0], 0.1 - 0.001)                     # the entry is charged once
        self.assertAlmostEqual(costly[1], 0.1)
        p = self.m.perf([0.01, -0.02, 0.01], 0.5)
        self.assertAlmostEqual(p["max_dd"], -0.02)
        self.assertEqual(p["avg_exposure"], 0.5)

    def test_series_exposes_the_anchor_each_day_used(self):
        out = re_.risk_budget_series([sig(vix=v) for v in (10.0, 20.0, 30.0, 40.0)], small_params())
        self.assertEqual(out["anchor"], [None, None, None, 20.0])


class TestScanField(unittest.TestCase):
    def test_scan_result_carries_the_view(self):
        import bot_engine as be
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_engine.py")).read()
        self.assertIn('"risk_budget": _scan_risk_budget(', src)
        v = be._scan_risk_budget(budget_state(0.7))
        self.assertEqual((v["budget"], v["band"]), (0.7, "reduced"))
        self.assertIsNone(be._scan_risk_budget(None))
        self.assertIsNone(be._scan_risk_budget({"error": "network", "stale": True}))


if __name__ == "__main__":
    unittest.main()
