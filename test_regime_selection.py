"""
Stock-selection guards of the regime engine (docs/regime/FINDINGS.md step 4):
the pure verdict, the gate flag, the state cache, the shadow-record keys, and
the wiring in bot_engine (deep_score + the admission loop) pinned on the source.
SHADOW BY DEFAULT: with VOLTRADE_REGIME_SELECTION unset nothing blocks and no
score changes; the flags ride the shadow record so each guard earns a
prevention-P&L first.
"""
import ast
import os
import unittest

import regime_engine as re_
import shadow_portfolio as sp

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(ROOT, "bot_engine.py")).read()


def state(**over):
    base = {"as_of": "2026-10-26", "composite_regime": "CALM_BULL", "breadth_state": "HEALTHY", "panic_state": False,
            "allowed_strategies": ["momentum_long", "csp", "dip_buy"], "inputs": {"above_200d": True, "gating_dd52": -0.02}}
    base.update(over)
    return base


def cand(**over):
    base = {"price": 100.0, "high_52w": 110.0, "closes": [95.0 + i * 0.3 for i in range(30)], "relative_strength_pct": 2.0,
            "mean_reversion_score": 50.0, "dip_bonus": 0.0}
    base.update(over)
    return base


class TestGateFlag(unittest.TestCase):
    def test_default_is_shadow_only(self):
        self.assertEqual(re_.selection_gates({}), set())
        self.assertEqual(re_.selection_gates({"VOLTRADE_REGIME_SELECTION": "off"}), set())
        self.assertEqual(re_.selection_gates({"VOLTRADE_REGIME_SELECTION": "all"}), set(re_.SELECTION_GUARDS))
        self.assertEqual(re_.selection_gates({"VOLTRADE_REGIME_SELECTION": "knife, dip ,bogus"}), {"knife", "dip"})

    def test_cached_state_reads_once_per_window(self):
        calls = []

        def loader():
            calls.append(1)
            return {"as_of": "x"}
        re_._STATE_CACHE.update(at=0.0, state=None)
        self.assertEqual(re_.cached_state(60, loader, now=1000.0)["as_of"], "x")
        self.assertEqual(re_.cached_state(60, loader, now=1030.0)["as_of"], "x")
        self.assertEqual(len(calls), 1)
        re_.cached_state(60, loader, now=1100.0)
        self.assertEqual(len(calls), 2)
        re_._STATE_CACHE.update(at=0.0, state=None)


class TestVerdict(unittest.TestCase):
    def test_calm_bull_healthy_leader_has_no_flags(self):
        v = re_.selection_verdict(state(), cand())
        self.assertEqual(sum(v["flags"].values()), 0)
        self.assertEqual(v["would_block"], [])
        self.assertFalse(v["neutralize_mr"])
        self.assertEqual(v["mr_delta"], 0.0)

    def test_knife_by_52_week_high_and_by_new_20_day_low(self):
        v = re_.selection_verdict(state(), cand(price=70.0, high_52w=100.0))
        self.assertEqual(v["flags"]["knife"], 1)
        self.assertIn("knife", v["would_block"])
        closes = [100.0] * 25
        v = re_.selection_verdict(state(), cand(price=99.0, high_52w=110.0, closes=closes))   # at/below every close of the window
        self.assertEqual(v["flags"]["knife"], 1)
        self.assertIn("20-day low", v["reasons"]["knife"])
        v = re_.selection_verdict(state(), cand(price=101.0, high_52w=110.0, closes=closes))
        self.assertEqual(v["flags"]["knife"], 0)
        v = re_.selection_verdict(state(), cand(price=101.0, high_52w=None, closes=closes[:5]))   # no history -> no claim
        self.assertEqual(v["flags"]["knife"], 0)

    def test_dip_neutralization_only_when_the_state_forbids_dips_and_the_candidate_is_a_dip_bet(self):
        bear = state(composite_regime="BEAR", allowed_strategies=["csp_reduced"], inputs={"above_200d": False, "gating_dd52": -0.2})
        v = re_.selection_verdict(bear, cand(mean_reversion_score=80.0, dip_bonus=8.0))
        self.assertEqual(v["flags"]["dip"], 1)
        self.assertTrue(v["neutralize_mr"])
        self.assertAlmostEqual(v["mr_delta"], 30.0 * 0.20 + 8.0, places=6)
        v = re_.selection_verdict(bear, cand(mean_reversion_score=40.0))          # overbought penalty is not a dip bet
        self.assertEqual(v["flags"]["dip"], 0)
        v = re_.selection_verdict(state(), cand(mean_reversion_score=80.0))        # allowed in CALM_BULL + HEALTHY
        self.assertEqual(v["flags"]["dip"], 0)
        narrow = state(breadth_state="NARROW", allowed_strategies=["momentum_long", "csp"])
        v = re_.selection_verdict(narrow, cand(mean_reversion_score=80.0))
        self.assertEqual(v["flags"]["dip"], 1)
        below = state(inputs={"above_200d": False, "gating_dd52": -0.05})
        v = re_.selection_verdict(below, cand(mean_reversion_score=80.0))          # index below its 200-day: no dip-buys
        self.assertEqual(v["flags"]["dip"], 1)

    def test_panic_rs_and_breadth_guards(self):
        v = re_.selection_verdict(state(panic_state=True), cand())
        self.assertEqual(v["flags"]["panic"], 1)
        self.assertIn("panic", v["would_block"])
        v = re_.selection_verdict(state(), cand(relative_strength_pct=-12.0))
        self.assertEqual(v["flags"]["rs"], 1)
        v = re_.selection_verdict(state(), cand(relative_strength_pct=None))
        self.assertEqual(v["flags"]["rs"], 0)
        v = re_.selection_verdict(state(breadth_state="WEAK", inputs={"above_200d": True, "gating_dd52": -0.01}), cand())
        self.assertEqual(v["flags"]["breadth"], 1)
        v = re_.selection_verdict(state(breadth_state="WEAK", inputs={"above_200d": True, "gating_dd52": -0.08}), cand())
        self.assertEqual(v["flags"]["breadth"], 0)                                  # not near the high: no no-add flag

    def test_no_state_means_no_claims(self):
        v = re_.selection_verdict(None, cand(mean_reversion_score=80.0, dip_bonus=8.0))
        self.assertEqual(sum(v["flags"].values()), 0)
        self.assertEqual(v["would_block"], [])


class TestShadowRecordKeys(unittest.TestCase):
    def test_breakdown_sanitizer_keeps_the_regime_flags_and_stores_absence_as_none(self):
        bd = {"momentum": 70, "regime_knife": 1.0, "regime_dip": 0.0, "regime_mr_delta": 6.0}
        out = sp._sanitize_score_breakdown(bd)
        self.assertEqual(out["regime_knife"], 1.0)
        self.assertEqual(out["regime_dip"], 0.0)
        self.assertEqual(out["regime_mr_delta"], 6.0)
        self.assertIsNone(out["regime_panic"])
        self.assertIsNone(sp._sanitize_score_breakdown({"momentum": 70})["regime_breadth"])


class TestWiring(unittest.TestCase):
    def test_deep_score_computes_the_verdict_before_the_combined_score_and_gates_only_on_the_flag(self):
        i_hook = SRC.index("_regime_sel = _re_sel.selection_verdict(")
        i_combined = SRC.index("    combined_score = (\n        momentum_score * 0.25 +")
        self.assertLess(i_hook, i_combined)
        self.assertIn('_regime_dip_gate = "dip" in _re_sel.selection_gates()', SRC)
        self.assertIn("if _regime_dip_gate and _regime_sel[\"neutralize_mr\"] and mean_reversion_score > 50:", SRC)
        self.assertIn('elif contrarian == "Buy the Dip" and _regime_dip_gate and _regime_sel and _regime_sel.get("neutralize_mr"):', SRC)

    def test_flags_ride_the_shadow_record_and_the_candidate(self):
        self.assertIn('_score_breakdown[f"regime_{_g}"] = float(_v)', SRC)
        self.assertIn('_score_breakdown["regime_mr_delta"] = float(_regime_sel["mr_delta"])', SRC)
        self.assertIn('quick_result["regime_guards"] = _regime_sel if \'_regime_sel\' in dir() else None', SRC)

    def test_admission_loop_blocks_only_the_guards_named_in_the_flag(self):
        i_old_gate = SRC.index('_note_gate(ticker, "rejected_regime_gate", f"no new stock longs in {_scan_regime}")')
        i_new = SRC.index('_rg_blocked_by = [g for g in _rg["would_block"] if g in _rg_gates_on]')
        self.assertLess(i_old_gate, i_new)
        self.assertIn('_note_gate(ticker, f"rejected_regime_{_rg_blocked_by[0]}",', SRC)
        tree = ast.parse(SRC)
        guarded = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and "regime_engine as _re_sel" in (ast.get_source_segment(SRC, node) or ""):
                guarded = True
        self.assertTrue(guarded, "the deep_score guard must be wrapped in try/except so it can never break scoring")


if __name__ == "__main__":
    unittest.main()
