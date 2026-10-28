"""
Options-sleeve guards of the regime engine (docs/regime/FINDINGS.md step 5):
the pure CSP verdict, the gate flag, the gate applier, the once-per-day shadow
bound, the Tier-1 wiring on a fixture context, and the scanner-path pin on
bot_engine's source. SHADOW BY DEFAULT: with VOLTRADE_REGIME_OPTIONS unset no
name is blocked and no size changes; every guard that WOULD act logs a
would_reject_regime_<g> record so it earns a prevention-P&L first.
"""
import os
import unittest
from unittest.mock import patch

import regime_engine as re_
import tiered_strategy as ts

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SRC = open(os.path.join(ROOT, "bot_engine.py")).read()

UNIVERSE = ["AMD", "AAPL", "MSFT", "NVDA", "AMZN"]


def state(**over):
    base = {"as_of": "2026-10-26", "composite_regime": "CALM_BULL", "vol_state": "NORMAL",
            "stress_flags": {"vrp_negative": False, "vix_vix3m_inverted": False, "vrp_points": 3.1, "vix_vix3m": 0.9},
            "ebp": {"ebp": -0.2}}
    base.update(over)
    return base


def stress(**over):
    s = {"vrp_negative": False, "vix_vix3m_inverted": False, "vrp_points": 3.1, "vix_vix3m": 0.9}
    s.update(over)
    return s


def ctx():
    return ts.TierContext(equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0,
                          positions=[], vxx_ratio=1.0, spy_vs_ma50=1.0)


class TestGateFlag(unittest.TestCase):
    def test_default_is_shadow_only_and_any_value_turns_on_crisis_and_vrp(self):
        self.assertEqual(re_.options_gates({}), set())
        self.assertEqual(re_.options_gates({"VOLTRADE_REGIME_OPTIONS": "off"}), set())
        self.assertEqual(re_.options_gates({"VOLTRADE_REGIME_OPTIONS": "all"}), set(re_.OPTIONS_GUARDS))
        self.assertEqual(re_.options_gates({"VOLTRADE_REGIME_OPTIONS": "elevated"}), {"elevated", "crisis", "vrp"})
        self.assertEqual(re_.options_gates({"VOLTRADE_REGIME_OPTIONS": "bogus"}), {"crisis", "vrp"})


class TestVerdict(unittest.TestCase):
    def test_calm_state_has_no_flags(self):
        v = re_.csp_verdict(state(), {"days_to_earnings": None, "dte_max": 45})
        self.assertEqual(sum(v["flags"].values()), 0)
        self.assertEqual(v["would_block"], [])
        self.assertEqual((v["size_mult"], v["target_delta"]), (1.0, None))

    def test_crisis_and_negative_vrp_block_new_short_vol(self):
        v = re_.csp_verdict(state(vol_state="CRISIS"), {})
        self.assertEqual(v["flags"]["crisis"], 1)
        self.assertIn("crisis", v["would_block"])
        v = re_.csp_verdict(state(stress_flags=stress(vrp_negative=True, vrp_points=-2.5)), {})
        self.assertEqual(v["flags"]["vrp"], 1)
        self.assertEqual(v["would_block"], ["vrp"])
        self.assertIn("-2.5", v["reasons"]["vrp"])

    def test_inversion_elevated_and_ebp_scale_size_and_delta(self):
        v = re_.csp_verdict(state(stress_flags=stress(vix_vix3m_inverted=True, vix_vix3m=1.08)), {})
        self.assertEqual((v["flags"]["inversion"], v["size_mult"]), (1, 0.5))
        v = re_.csp_verdict(state(vol_state="ELEVATED"), {})
        self.assertEqual((v["flags"]["elevated"], v["size_mult"], v["target_delta"]), (1, 0.5, 0.15))
        self.assertEqual(v["would_block"], [])                      # a size cut, never a block
        v = re_.csp_verdict(state(vol_state="ELEVATED", stress_flags=stress(vix_vix3m_inverted=True), ebp={"ebp": 0.4}), {})
        self.assertEqual(v["size_mult"], 0.125)                      # the three cuts compound
        self.assertEqual(v["flags"]["ebp"], 1)
        v = re_.csp_verdict(state(ebp={"ebp": None}), {})
        self.assertEqual(v["flags"]["ebp"], 0)

    def test_earnings_inside_the_window_plus_buffer_blocks(self):
        self.assertEqual(re_.csp_verdict(state(), {"days_to_earnings": 10, "dte_max": 45})["would_block"], ["earnings"])
        self.assertEqual(re_.csp_verdict(state(), {"days_to_earnings": 47, "dte_max": 45})["flags"]["earnings"], 1)   # 45 + 2
        self.assertEqual(re_.csp_verdict(state(), {"days_to_earnings": 48, "dte_max": 45})["flags"]["earnings"], 0)
        self.assertEqual(re_.csp_verdict(state(), {"days_to_earnings": 0, "dte_max": 45})["flags"]["earnings"], 1)    # reports today
        self.assertEqual(re_.csp_verdict(state(), {"days_to_earnings": None, "dte_max": 45})["flags"]["earnings"], 0)

    def test_no_state_means_no_claims(self):
        v = re_.csp_verdict(None, {"days_to_earnings": 3})
        self.assertEqual(v["flags"]["earnings"], 1)                  # the calendar is the candidate's own fact
        self.assertEqual(v["flags"]["crisis"], 0)
        self.assertEqual(v["size_mult"], 1.0)


class TestApplyGates(unittest.TestCase):
    def test_only_the_named_gates_act(self):
        v = re_.csp_verdict(state(vol_state="ELEVATED", stress_flags=stress(vix_vix3m_inverted=True)), {"days_to_earnings": 5})
        off = re_.apply_csp_gates(v, set())
        self.assertEqual(off, {"block": False, "blocked_by": [], "size_mult": 1.0, "target_delta": None})
        part = re_.apply_csp_gates(v, {"elevated", "crisis", "vrp"})
        self.assertEqual((part["block"], part["size_mult"], part["target_delta"]), (False, 0.5, 0.15))
        full = re_.apply_csp_gates(v, set(re_.OPTIONS_GUARDS))
        self.assertEqual((full["block"], full["blocked_by"], full["size_mult"]), (True, ["earnings"], 0.25))


class TestShadowOnce(unittest.TestCase):
    def test_one_record_per_key_per_day(self):
        re_._SHADOW_ONCE.update(day=None, keys=set())
        self.assertTrue(re_.shadow_once_per_day("a", today="2026-10-28"))
        self.assertFalse(re_.shadow_once_per_day("a", today="2026-10-28"))
        self.assertTrue(re_.shadow_once_per_day("b", today="2026-10-28"))
        self.assertTrue(re_.shadow_once_per_day("a", today="2026-10-29"))
        re_._SHADOW_ONCE.update(day=None, keys=set())


class TestTier1Wiring(unittest.TestCase):
    def run_t1(self, st, flag, earnings=None, ok=True, reset=True):
        logged = []
        if reset:
            re_._SHADOW_ONCE.update(day=None, keys=set())
        with patch("tiered_strategy._get_t1_universe", return_value=UNIVERSE), \
             patch("csp_universe._load_layer2_cache", return_value={}), \
             patch("regime_engine.engine_allowed", return_value=st is not None), \
             patch("regime_engine.cached_state", return_value=st), \
             patch("tiered_strategy._t1_earnings_calendar", return_value=(earnings or {}, ok)), \
             patch("tiered_strategy.log_csp_gate_shadow", side_effect=lambda rej: logged.extend(rej) or {}), \
             patch.dict(os.environ, {"VOLTRADE_REGIME_OPTIONS": flag or ""}):
            actions = ts.tier1_csp_core(ctx())
        return actions, logged

    def test_engine_off_is_the_baseline_with_inert_metadata(self):
        actions, logged = self.run_t1(None, "all")
        self.assertEqual([a.ticker for a in actions], UNIVERSE[:3])
        self.assertEqual(logged, [])
        for a in actions:
            self.assertIsNone(a.metadata["regime_guards"])
            self.assertEqual(a.metadata["regime_size_mult"], 1.0)
            self.assertEqual(a.metadata["target_delta"], ts.T1_TARGET_DELTA)

    def test_shadow_crisis_proposes_the_same_names_and_logs_would_reject(self):
        base, _ = self.run_t1(None, "")
        actions, logged = self.run_t1(state(vol_state="CRISIS"), "")
        self.assertEqual([a.ticker for a in actions], [a.ticker for a in base])
        self.assertEqual([a.size_pct for a in actions], [a.size_pct for a in base])
        self.assertEqual([r["decision"] for r in logged], ["would_reject_regime_crisis"] * 3)
        self.assertEqual([r["ticker"] for r in logged], UNIVERSE[:3])
        self.assertEqual(logged[0]["metadata"]["regime_guards"]["flags"]["crisis"], 1)
        self.assertEqual(logged[0]["metadata"]["regime_label"], "NEUTRAL")
        for a in actions:
            self.assertEqual(a.metadata["regime_guards"]["would_block"], ["crisis"])

    def test_gated_crisis_proposes_nothing_and_logs_the_three_it_would_have(self):
        actions, logged = self.run_t1(state(vol_state="CRISIS"), "all")
        self.assertEqual(actions, [])
        self.assertEqual([(r["ticker"], r["decision"]) for r in logged], [(t, "rejected_regime_crisis") for t in UNIVERSE[:3]])

    def test_gated_vrp_blocks_with_the_vrp_bucket(self):
        actions, logged = self.run_t1(state(stress_flags=stress(vrp_negative=True, vrp_points=-1.0)), "")
        self.assertEqual(len(actions), 3)
        self.assertEqual({r["decision"] for r in logged}, {"would_reject_regime_vrp"})
        actions, logged = self.run_t1(state(stress_flags=stress(vrp_negative=True, vrp_points=-1.0)), "earnings")   # any value gates vrp
        self.assertEqual(actions, [])
        self.assertEqual({r["decision"] for r in logged}, {"rejected_regime_vrp"})

    def test_gated_elevated_halves_size_and_sets_the_target_delta(self):
        base, _ = self.run_t1(None, "")
        actions, logged = self.run_t1(state(vol_state="ELEVATED"), "elevated")
        self.assertEqual(logged, [])
        self.assertEqual([a.ticker for a in actions], [a.ticker for a in base])
        for a, b in zip(actions, base):
            self.assertAlmostEqual(a.size_pct, b.size_pct * 0.5, places=9)
            self.assertEqual(a.metadata["target_delta"], 0.15)
            self.assertEqual(a.metadata["regime_size_mult"], 0.5)
            self.assertEqual(a.metadata["regime_guards"]["flags"]["elevated"], 1)

    def test_shadow_elevated_changes_nothing_but_shows_what_it_would_do(self):
        base, _ = self.run_t1(None, "")
        actions, logged = self.run_t1(state(vol_state="ELEVATED"), "")
        self.assertEqual(logged, [])
        for a, b in zip(actions, base):
            self.assertEqual(a.size_pct, b.size_pct)
            self.assertEqual(a.metadata["target_delta"], ts.T1_TARGET_DELTA)
            self.assertEqual(a.metadata["regime_size_mult"], 1.0)
            self.assertEqual(a.metadata["regime_guards"]["target_delta"], 0.15)
            self.assertEqual(a.metadata["regime_guards"]["size_mult"], 0.5)

    def test_gated_earnings_removes_the_name_and_keeps_the_slot(self):
        actions, logged = self.run_t1(state(), "earnings", earnings={"AMD": 10, "NVDA": 60})
        self.assertEqual([a.ticker for a in actions], ["AAPL", "MSFT", "NVDA"])
        self.assertEqual([(r["ticker"], r["decision"]) for r in logged], [("AMD", "rejected_regime_earnings")])
        self.assertIn("10 days", logged[0]["reason"])

    def test_shadow_earnings_proposes_the_name_and_logs_would_reject(self):
        actions, logged = self.run_t1(state(), "", earnings={"AMD": 10})
        self.assertEqual([a.ticker for a in actions], ["AMD", "AAPL", "MSFT"])
        self.assertEqual([(r["ticker"], r["decision"]) for r in logged], [("AMD", "would_reject_regime_earnings")])
        self.assertFalse(actions[0].metadata["regime_guards"]["flags"]["crisis"])
        self.assertTrue(actions[0].metadata["regime_guards"]["earnings_lookup_ok"])

    def test_calendar_outage_fails_open_and_is_flagged(self):
        actions, logged = self.run_t1(state(), "all", earnings={}, ok=False)
        self.assertEqual(len(actions), 3)
        self.assertEqual(logged, [])
        self.assertFalse(actions[0].metadata["regime_guards"]["earnings_lookup_ok"])

    def test_shadow_records_are_once_per_name_guard_and_day(self):
        _, first = self.run_t1(state(vol_state="CRISIS"), "")
        _, second = self.run_t1(state(vol_state="CRISIS"), "", reset=False)
        self.assertEqual(len(first), 3)
        self.assertEqual(second, [])
        re_._SHADOW_ONCE.update(day=None, keys=set())

    def test_context_never_raises(self):
        with patch("regime_engine.engine_allowed", side_effect=RuntimeError("boom")):
            c = ts._t1_regime_context()
        self.assertEqual((c["engine"], c["size_mult"], c["gates"]), (None, 1.0, set()))

    def test_earnings_calendar_is_cached_per_window(self):
        calls = []

        def fake(days_ahead=10):
            calls.append(days_ahead)
            return {"AMD": 3}, True
        ts._T1_EARNINGS_CACHE.update(at=0.0, days_ahead=0, cal={}, ok=False)
        with patch("options_scanner._fetch_earnings_calendar_with_status", side_effect=fake):
            self.assertEqual(ts._t1_earnings_calendar(47), ({"AMD": 3}, True))
            self.assertEqual(ts._t1_earnings_calendar(47), ({"AMD": 3}, True))
            self.assertEqual(calls, [47])
        ts._T1_EARNINGS_CACHE.update(at=0.0, days_ahead=0, cal={}, ok=False)


class TestScannerPathWiring(unittest.TestCase):
    def test_scanner_guard_sits_inside_the_options_scan_and_only_touches_sell_side(self):
        i_call = BOT_SRC.index("from options_scanner import get_options_trades")
        i_guard = BOT_SRC.index('if _rg_opt.get("verdict") and str(ot.get("side", "buy")) == "sell" and _rg_opt["verdict"]["would_block"]:')
        i_end = BOT_SRC.index('_timing_log("step6_trade_loop_and_options")')
        self.assertLess(i_call, i_guard)
        self.assertLess(i_guard, i_end)
        self.assertIn('_rg_decision = f"rejected_regime_{_rg_g}" if _rg_g_on else f"would_reject_regime_{_rg_g}"', BOT_SRC)
        self.assertIn('_rg_opt["engine"].shadow_once_per_day(f"scanner:{ot.get(\'ticker\')}:{_rg_decision}")', BOT_SRC)
        # the gate drops the trade only when the flag names the guard
        self.assertIn("                if _rg_g_on:\n                    continue\n            # Convert to bot_engine trade format", BOT_SRC)
        # its own try: the options scan survives a regime failure
        self.assertIn("        _rg_opt = {\"verdict\": None, \"gates\": set(), \"engine\": None}\n        try:\n            import regime_engine as _re_opt", BOT_SRC)


if __name__ == "__main__":
    unittest.main()
