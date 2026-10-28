"""
Legacy-vs-engine counterfactual log (2026-10-28 regime live flip, human
directive; [RULE-REVIEW] measurement): regime_engine.log_counterfactual and the
per-kind helpers (pure-ish, temp paths), the bounds (per-day dedupe, row cap,
size rotation), and the call sites — the stock sizer, both options sizers and
Tier-1 pass the LEGACY scalar they computed before the engine replaced it; the
admission loop, the options scanner and deep_score / the tier context are
pinned on the source. Logging only: no decision changes (the sizer outputs are
the same with the log on as test_regime_sizing pins).
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import regime_engine as re_
import position_sizing as ps
import options_execution as oe
import instrument_selector as isel
import tiered_strategy as ts

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SRC = open(os.path.join(ROOT, "bot_engine.py")).read()
TODAY = "2026-10-28"
ON = {"VOLTRADE_REGIME_ENGINE": "1"}
STATE = {"as_of": "2026-10-27", "composite_regime": "STRESSED", "legacy_label": "CAUTION", "exposure_multiplier": 0.5,
         "allowed_strategies": ["csp_reduced"], "params_version": re_.PARAMS_VERSION, "vol_state": "ELEVATED",
         "stress_flags": {"vrp_negative": False, "vix_vix3m_inverted": False}, "ebp": {"ebp": -0.1}}


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


class CFBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "cf.jsonl")
        re_._CF.update(day=None, keys=set(), rows=0, dropped=0)
        re_._STATE_CACHE.update(at=0.0, state=None)
        self._p = patch("regime_engine._cf_path", return_value=self.path)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self.tmp.cleanup()
        re_._CF.update(day=None, keys=set(), rows=0, dropped=0)
        re_._STATE_CACHE.update(at=0.0, state=None)


class TestWriter(CFBase):
    def test_row_schema(self):
        ok = re_.log_counterfactual("sizing", "AMD", {"multiplier": 0.6, "vix_scalar": 0.8, "stress_mult": 0.75},
                                    {"multiplier": 0.5, "source": "engine", "stale": False, "applied": True},
                                    "position_sizing.calculate_position", state=STATE, ticker="amd", sleeve="stock",
                                    env=ON, today=TODAY)
        self.assertTrue(ok)
        (r,) = rows(self.path)
        for k in ("schema", "ts", "date", "kind", "key", "ticker", "sleeve", "source", "legacy", "engine", "differs",
                  "shadow_decision", "regime", "flags"):
            self.assertIn(k, r)
        self.assertEqual((r["schema"], r["date"], r["kind"], r["ticker"], r["sleeve"]), (1, TODAY, "sizing", "AMD", "stock"))
        self.assertEqual(r["legacy"]["multiplier"], 0.6)
        self.assertEqual(r["engine"]["multiplier"], 0.5)
        self.assertTrue(r["differs"])
        self.assertEqual(r["regime"], {"as_of": "2026-10-27", "composite": "STRESSED", "legacy_label": "CAUTION",
                                       "exposure_multiplier": 0.5, "params_version": re_.PARAMS_VERSION, "stale": False})
        self.assertEqual(r["flags"]["VOLTRADE_REGIME_ENGINE"], "1")
        self.assertEqual(set(r["flags"]), set(re_.FLAG_DEFAULTS))

    def test_engine_off_writes_nothing(self):
        self.assertFalse(re_.log_counterfactual("label", "x", {"label": "A"}, {"label": "B"}, "t", state=STATE,
                                                env={"VOLTRADE_REGIME_ENGINE": "off"}, today=TODAY))
        self.assertFalse(re_.log_counterfactual("bogus", "x", {}, {}, "t", state=STATE, env=ON, today=TODAY))
        self.assertEqual(rows(self.path), [])

    def test_one_row_per_content_per_day_and_reset_on_a_new_day(self):
        args = ("selection", "AMD", {"decision": "take"}, {"decision": "block", "blocked_by": ["knife"]}, "src")
        self.assertTrue(re_.log_counterfactual(*args, state=STATE, env=ON, today=TODAY))
        self.assertFalse(re_.log_counterfactual(*args, state=STATE, env=ON, today=TODAY))
        self.assertTrue(re_.log_counterfactual("selection", "AMD", {"decision": "take"}, {"decision": "take"}, "src",
                                               state=STATE, env=ON, today=TODAY))
        self.assertTrue(re_.log_counterfactual(*args, state=STATE, env=ON, today="2026-10-29"))
        self.assertEqual(len(rows(self.path)), 3)

    def test_daily_cap_and_size_rotation(self):
        with patch.dict(re_.COUNTERFACTUAL_PARAMS, {"max_rows_per_day": 2, "max_bytes": 10**9}):
            for i in range(4):
                re_.log_counterfactual("label", f"k{i}", {"label": "A"}, {"label": "B"}, "s", state=STATE, env=ON, today=TODAY)
        self.assertEqual(len(rows(self.path)), 2)
        self.assertEqual(re_._CF["dropped"], 2)
        re_._CF.update(day=None, keys=set(), rows=0, dropped=0)
        with patch.dict(re_.COUNTERFACTUAL_PARAMS, {"max_bytes": 10}):
            re_.log_counterfactual("label", "rot", {"label": "A"}, {"label": "B"}, "s", state=STATE, env=ON, today=TODAY)
        self.assertEqual(len(rows(self.path + ".1")), 2)       # the oversized file moved aside, one generation kept
        self.assertEqual(len(rows(self.path)), 1)

    def test_never_raises(self):
        self._p.stop()
        with patch("regime_engine._cf_path", return_value=os.path.join(self.tmp.name, "no", "such", "dir", "cf.jsonl")):
            self.assertFalse(re_.log_counterfactual("label", "k", {"label": "A"}, {"label": "B"}, "s", state=STATE, env=ON, today=TODAY))
        self._p.start()

    def test_differs(self):
        self.assertFalse(re_.counterfactual_differs("sizing", {"multiplier": 0.5}, {"multiplier": 0.5}))
        self.assertTrue(re_.counterfactual_differs("sizing", {"multiplier": 0.8}, {"multiplier": 0.5}))
        self.assertFalse(re_.counterfactual_differs("label", {"label": "BULL"}, {"label": "BULL"}))
        self.assertTrue(re_.counterfactual_differs("options", {"decision": "take", "size_mult": 1.0}, {"decision": "take", "size_mult": 0.5}))
        self.assertTrue(re_.counterfactual_differs("selection", {"decision": "take"}, {"decision": "take", "dip_neutralized": True}))
        self.assertFalse(re_.counterfactual_differs("selection", {"decision": "take"}, {"decision": "take"}))


class TestHelpers(CFBase):
    def test_sizing_applied_and_would_be(self):
        applied = {"multiplier": 0.5, "source": "engine", "stale": False}
        self.assertTrue(re_.counterfactual_sizing("stock", "AMD", {"multiplier": 0.8}, applied, "s", ticker="AMD",
                                                  env=ON, loader=lambda: STATE, today=TODAY))
        re_._STATE_CACHE.update(at=0.0, state=None)
        self.assertTrue(re_.counterfactual_sizing("tier1", "tier1", {"multiplier": 0.85}, None, "s",
                                                  env=ON, loader=lambda: STATE, today=TODAY))
        a, b = rows(self.path)
        self.assertEqual((a["engine"]["multiplier"], a["engine"]["applied"]), (0.5, True))
        # flag off for the sleeve: the engine's would-be multiplier (options floor 0.25 under csp_reduced), applied False
        self.assertEqual((b["engine"]["multiplier"], b["engine"]["source"], b["engine"]["applied"]), (0.5, "engine", False))
        self.assertEqual(b["sleeve"], "tier1")

    def test_selection_block_pass_and_dip(self):
        v = {"flags": {"knife": 1, "dip": 1, "panic": 0, "rs": 1, "breadth": 0}, "would_block": ["knife", "rs"],
             "neutralize_mr": True, "mr_delta": 3.0, "reasons": {}}
        re_.counterfactual_selection("XYZ", v, {"knife", "dip"}, "s", env=ON, today=TODAY)
        re_.counterfactual_selection("XYZ", v, set(), "s", env=ON, today=TODAY)
        self.assertFalse(re_.counterfactual_selection("XYZ", {}, {"knife"}, "s", env=ON, today=TODAY))
        a, b = rows(self.path)
        self.assertEqual((a["legacy"], a["engine"]["decision"], a["engine"]["blocked_by"]), ({"decision": "take"}, "block", ["knife"]))
        self.assertEqual(a["shadow_decision"], "rejected_regime_knife")
        self.assertTrue(a["engine"]["dip_neutralized"])
        self.assertEqual((b["engine"]["decision"], b["engine"]["dip_neutralized"], b["shadow_decision"]), ("take", False, None))
        self.assertEqual(b["engine"]["would_block"], ["knife", "rs"])

    def test_options_block_and_cut(self):
        crisis = re_.csp_verdict(dict(STATE, vol_state="CRISIS"), {})
        re_.counterfactual_options("KO", crisis, {"crisis", "vrp"}, "s", legacy_delta=0.30,
                                   shadow_decision="rejected_regime_crisis", env=ON, today=TODAY)
        elevated = re_.csp_verdict(STATE, {})
        re_.counterfactual_options("PEP", elevated, set(re_.OPTIONS_GUARDS), "s", legacy_delta=0.30, env=ON, today=TODAY)
        a, b = rows(self.path)
        self.assertEqual((a["engine"]["decision"], a["engine"]["blocked_by"], a["shadow_decision"]), ("block", ["crisis"], "rejected_regime_crisis"))
        self.assertEqual(a["legacy"], {"decision": "take", "size_mult": 1.0, "target_delta": 0.3})
        self.assertEqual((b["engine"]["decision"], b["engine"]["size_mult"], b["engine"]["target_delta"]), ("take", 0.5, 0.15))
        self.assertTrue(b["differs"])

    def test_label_applied_and_would_be(self):
        re_.counterfactual_label("deep_score", "NEUTRAL_BULL", {"label": "CAUTION"}, "s", env=ON, loader=lambda: STATE, today=TODAY)
        re_._STATE_CACHE.update(at=0.0, state=None)
        re_.counterfactual_label("tier_context", "NEUTRAL", None, "s", env=ON, loader=lambda: STATE, today=TODAY)
        a, b = rows(self.path)
        self.assertEqual((a["legacy"]["label"], a["engine"]), ("NEUTRAL_BULL", {"label": "CAUTION", "applied": True}))
        self.assertEqual(b["engine"], {"label": "CAUTION", "applied": False})


STATS = {"overall": {"win_rate": 0.6, "avg_win": 1.0, "avg_loss": 0.8}, "by_strategy": {}}
ENG = {"multiplier": 0.5, "source": "engine", "stale": False, "as_of": "2026-10-27", "composite": "STRESSED", "sleeve": "stock"}


class TestCallSites(CFBase):
    """The sizers pass the legacy scalar they computed BEFORE the engine replaced it."""

    def setUp(self):
        super().setUp()
        self._st = patch("regime_engine.cached_state", return_value=STATE)
        self._env = patch.dict(os.environ, {"VOLTRADE_REGIME_ENGINE": "1"})
        self._st.start()
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._st.stop()
        super().tearDown()

    def test_stock_sizer_logs_vix_scalar_times_stress(self):
        trade = {"ticker": "AAPL", "price": 50.0, "score": 70, "deep_score": 70, "volume": 5_000_000, "ewma_rv": 2.0, "side": "buy"}
        with tempfile.TemporaryDirectory() as d, \
             patch("position_sizing.check_halt_status", return_value={"halted": False, "status": "ok"}), \
             patch("position_sizing._get_historical_stats", return_value=STATS), \
             patch("position_sizing._kelly_fraction", return_value=0.04), \
             patch("position_sizing._volatility_scalar", return_value=1.0), \
             patch("position_sizing._confidence_scalar", return_value=1.0), \
             patch("position_sizing._earnings_scalar", return_value=1.0), \
             patch("position_sizing._time_scalar", return_value=1.0), \
             patch("position_sizing._portfolio_heat_scalar", return_value=1.0), \
             patch("position_sizing._liquidity_scalar", return_value=1.0), \
             patch("position_sizing.DATA_DIR", d), \
             patch("stress_index.get_stress_multiplier", return_value=0.5), \
             patch("stress_index.compute_stress_index", return_value={"level": "stressed", "index": 70}), \
             patch("regime_engine.sizing_for", return_value=ENG):
            r = ps.calculate_position(trade, 100_000.0, [], macro={"vix": 22.0})
        self.assertAlmostEqual(r["position_pct"], 0.04 * 0.5, places=6)      # the decision is the engine's, unchanged by the log
        (row,) = rows(self.path)
        self.assertEqual((row["kind"], row["sleeve"], row["ticker"], row["source"]), ("sizing", "stock", "AAPL", "position_sizing.calculate_position"))
        self.assertEqual(row["legacy"], {"multiplier": 0.4, "vix_scalar": 0.8, "stress_mult": 0.5})
        self.assertEqual((row["engine"]["multiplier"], row["engine"]["applied"]), (0.5, True))

    def test_options_sizers_log_the_vix_scalar(self):
        trade = {"ticker": "AAPL", "price": 50.0, "score": 70, "volume": 5_000_000, "ewma_rv": 2.0}
        common = ("_get_historical_stats", "_kelly_fraction", "_volatility_scalar", "_earnings_scalar", "_time_scalar",
                  "_portfolio_heat_scalar", "_liquidity_scalar")
        values = {"_get_historical_stats": STATS, "_kelly_fraction": 0.04}
        for mod, fn, conf, src in ((oe, oe._dynamic_options_size, "_confidence_scalar", "options_execution._dynamic_options_size"),
                                   (isel, isel._dynamic_options_size, "_confidence_scalar_safe", "instrument_selector._dynamic_options_size")):
            patches = [patch.object(mod, n, return_value=values.get(n, 1.0)) for n in common + (conf,)]
            for p in patches:
                p.start()
            try:
                with patch("regime_engine.sizing_for", return_value=dict(ENG, sleeve="options", multiplier=0.75)):
                    fn(trade, 100_000.0, [], macro={"vix": 22.0})
            finally:
                for p in patches:
                    p.stop()
        a, b = rows(self.path)
        for row, src in ((a, "options_execution._dynamic_options_size"), (b, "instrument_selector._dynamic_options_size")):
            self.assertEqual((row["sleeve"], row["source"], row["legacy"]), ("options", src, {"multiplier": 0.8, "vix_scalar": 0.8}))
            self.assertEqual(row["engine"]["multiplier"], 0.75)

    def test_tier1_logs_the_vxx_rung_and_one_row_per_proposed_or_blocked_name(self):
        ctx = ts.TierContext(equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0, positions=[], vxx_ratio=1.5, spy_vs_ma50=1.0)
        crisis = dict(STATE, vol_state="CRISIS")
        with patch("tiered_strategy._get_t1_universe", return_value=["AMD", "AAPL"]), \
             patch("csp_universe._load_layer2_cache", return_value={}), \
             patch("tiered_strategy._t1_earnings_calendar", return_value=({}, True)), \
             patch("tiered_strategy.log_csp_gate_shadow", return_value={}), \
             patch("regime_engine.cached_state", return_value=crisis), \
             patch.dict(os.environ, {"VOLTRADE_REGIME_OPTIONS": "all", "VOLTRADE_REGIME_SIZING": "off"}):
            actions = ts.tier1_csp_core(ctx)
        self.assertEqual(actions, [])                                         # CRISIS gate blocks every name
        got = rows(self.path)
        sizing = [r for r in got if r["kind"] == "sizing"]
        self.assertEqual(len(sizing), 1)
        self.assertEqual(sizing[0]["legacy"], {"multiplier": 0.3, "vxx_rung": 0.3})   # vxx 1.5 -> the kill-ratio rung
        self.assertFalse(sizing[0]["engine"]["applied"])                     # sizing flag off: the engine's would-be value
        opts = [r for r in got if r["kind"] == "options"]
        self.assertEqual({r["ticker"] for r in opts}, {"AMD", "AAPL"})
        self.assertTrue(all(r["engine"]["decision"] == "block" and r["shadow_decision"] == "rejected_regime_crisis" for r in opts))


class TestSourceWiring(unittest.TestCase):
    def test_admission_loop_logs_every_guarded_candidate_before_the_gate(self):
        i_cf = BOT_SRC.index('_re_cf_sel.counterfactual_selection(ticker, _rg, _re_cf_sel.selection_gates(),')
        i_gate = BOT_SRC.index('_rg_blocked_by = [g for g in _rg["would_block"] if g in _rg_gates_on]')
        self.assertLess(i_cf, i_gate)

    def test_options_scanner_logs_pass_and_block_rows(self):
        self.assertIn('"bot_engine.options_scanner")', BOT_SRC)
        self.assertIn('"bot_engine.options_scanner", shadow_decision=_rg_decision)', BOT_SRC)

    def test_labels_logged_in_deep_score_and_tier_context(self):
        self.assertIn('_re_lbl.counterfactual_label("deep_score", _markov_lbl, _engine_lbl, "bot_engine.deep_score")', BOT_SRC)
        self.assertIn('_scan_engine_lbl_res, "bot_engine.tier_context")', BOT_SRC)

    def test_path_lives_in_storage_config(self):
        from storage_config import REGIME_COUNTERFACTUAL_PATH
        self.assertTrue(REGIME_COUNTERFACTUAL_PATH.endswith("voltrade_regime_counterfactual.jsonl"))


if __name__ == "__main__":
    unittest.main()
