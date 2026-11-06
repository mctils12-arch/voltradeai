"""
[RULE-REVIEW] 2026-11-06, EXPLORATORY (kill date 2027-02-05) — deep_score's
composite v2 (score_composite.py, BASE_CONFIG["SCORE_COMPOSITE_V2"]):
volume / vol-premium / relative-strength held neutral, the ML input of the
blend held at 50, the legacy composite still computed and logged beside it.
Every test here fails on the previous commit (module, config and wiring absent).
"""
import os
import re
import unittest
from unittest.mock import patch

import bot_engine
import score_composite as sc
import signal_attribution as sa
import walkforward as wf
from system_config import BASE_CONFIG
from test_serve_feature_fix_deep_score import _DeepScoreHarness, _detail

HERE = os.path.dirname(os.path.abspath(__file__))


def _hot_detail():
    """A hot name: volume 3.5x its 90-day mean (sub-score 90), EWMA 12 points
    over RV20 (+8), outperforming SPY by 9 points (+6) — 20 points v2 removes."""
    d = _detail()
    d["vol_metrics"] = {"rsi_14": 58.0, "volume_ratio_5d": 3.5, "ewma_rv": 50.0, "rv20": 38.0}
    d["edge_factors"] = {"relative_strength": 9.0}
    return d


class TestPureComposite(unittest.TestCase):
    CFG = sc.v2_config({"enabled": True, "neutralize": ["volume", "vol_premium", "rel_strength"], "ml_neutral": True})

    def test_config_defaults_validation_and_malformed(self):
        self.assertEqual(self.CFG["neutralize"], ("volume", "vol_premium", "rel_strength"))
        c = sc.v2_config({"enabled": True, "neutralize": ["volume", "momentum_typo"]})
        self.assertEqual(c["neutralize"], ("volume",))
        self.assertEqual(c["ignored"], ("momentum_typo",))
        self.assertTrue(c["ml_neutral"], "missing keys take the defaults")
        self.assertFalse(sc.v2_config("on")["enabled"], "a malformed config serves legacy")
        self.assertFalse(sc.v2_config({"enabled": 0})["enabled"])
        self.assertEqual(sc.v2_config({"neutralize": "volume"})["neutralize"], ("volume",))

    def test_removed_points_are_each_terms_weighted_deviation_from_neutral(self):
        comp = {"volume": 90.0, "vol_premium": 8.0, "rel_strength": 6.0}
        self.assertAlmostEqual(sc.removed_points(comp, self.CFG["neutralize"]), 0.15 * 40 + 8 + 6)
        self.assertAlmostEqual(sc.removed_points({"volume": 35.0}, ("volume",)), 0.15 * -15)
        self.assertEqual(sc.removed_points({}, ("volume",)), 0.0)

    def test_not_blended_is_the_rules_total_minus_the_removed_points(self):
        out = sc.composite_v2(84.0, {"volume": 90.0, "vol_premium": 8.0, "rel_strength": 6.0}, {"blended": False}, self.CFG)
        self.assertEqual(out["score"], 64.0)
        self.assertEqual(out["removed"], 20.0)
        self.assertFalse(out["blended"])
        clamped = sc.composite_v2(112.0, {"volume": 50.0}, {"blended": False}, self.CFG)
        self.assertEqual(clamped["score"], 100.0)

    def test_blend_holds_the_ml_input_at_50_when_ml_neutral(self):
        tr = {"blended": True, "gated": False, "rule_weight": 0.72, "ml_weight": 0.28, "ml_only_score": 80.0}
        out = sc.composite_v2(84.0, {"volume": 90.0, "vol_premium": 8.0, "rel_strength": 6.0}, tr, self.CFG)
        self.assertAlmostEqual(out["score"], 64.0 * 0.72 + 50.0 * 0.28, places=4)
        self.assertEqual(out["ml_input"], 50.0)
        keep_ml = sc.v2_config({"enabled": True, "neutralize": [], "ml_neutral": False})
        out2 = sc.composite_v2(84.0, {}, tr, keep_ml)
        self.assertAlmostEqual(out2["score"], 84.0 * 0.72 + 80.0 * 0.28, places=4)
        gated = sc.composite_v2(84.0, {}, dict(tr, gated=True), keep_ml)
        self.assertAlmostEqual(gated["score"], 84.0 * 0.72 + 50.0 * 0.28, places=4)

    def test_missing_inputs_return_none_so_the_caller_serves_legacy(self):
        self.assertIsNone(sc.composite_v2(None, {}, {"blended": False}, self.CFG))
        self.assertIsNone(sc.composite_v2(70.0, {}, {"blended": True, "rule_weight": 0.7}, self.CFG))
        no_ml = sc.v2_config({"enabled": True, "neutralize": [], "ml_neutral": False})
        self.assertIsNone(sc.composite_v2(70.0, {}, {"blended": True, "gated": False, "rule_weight": 0.7, "ml_weight": 0.3}, no_ml))

    def test_reason_line_is_plain_english(self):
        v2 = sc.composite_v2(84.0, {"volume": 90.0, "vol_premium": 8.0, "rel_strength": 6.0},
                             {"blended": True, "gated": False, "rule_weight": 0.72, "ml_weight": 0.28, "ml_only_score": 80.0}, self.CFG)
        line = sc.reason_line(v2, 72.4, self.CFG)
        self.assertTrue(line.startswith("Score v2 60.1 (legacy 72.4): "), line)
        self.assertIn("volume surge, vol-premium bonus, relative-strength bonus held neutral (-20.0 pts", line)
        self.assertIn("ML input held at 50", line)


class TestPins(unittest.TestCase):
    def test_live_config_is_the_preregistered_set_and_exploratory(self):
        cfg = BASE_CONFIG["SCORE_COMPOSITE_V2"]
        self.assertTrue(cfg["enabled"] and cfg["ml_neutral"])
        self.assertEqual(tuple(cfg["neutralize"]), wf.V2_NEUTRALIZED, "live v2 == the walk-forward's pre-registered variant")
        self.assertEqual(cfg["kill_date"], "2027-02-05")
        self.assertEqual(sc.v2_config()["neutralize"], wf.V2_NEUTRALIZED)

    def test_weights_reproduce_the_walkforward_composite_difference(self):
        for vol, vp, rs in ((90.0, 8.0, 6.0), (35.0, 5.0, 0.0), (60.0, 0.0, 6.0), (50.0, 0.0, 0.0)):
            mom, mr, adx, obv = 70.0, 55.0, 3.0, -5.0
            legacy = mom * 0.25 + mr * 0.20 + 50 * 0.25 + 50 * 0.15 + vol * 0.15 + vp + adx + obv + rs
            v2 = mom * 0.25 + mr * 0.20 + 50 * 0.25 + 50 * 0.15 + 50 * 0.15 + adx + obv
            removed = sc.removed_points({"volume": vol, "vol_premium": vp, "rel_strength": rs}, wf.V2_NEUTRALIZED)
            self.assertAlmostEqual(legacy - v2, removed, places=9)

    def test_deep_score_source_keeps_the_terms_the_weights_assume(self):
        src = open(os.path.join(HERE, "bot_engine.py")).read()
        seg = src[src.index("def deep_score("):src.index("def check_sector_correlation(")]
        self.assertIn("volume_score * 0.15", seg)
        self.assertIn(") + vol_premium_bonus", seg)
        self.assertIn("rel_strength_adjustment = combined_score - _pre_rel_strength", seg)
        # v2 is computed after the final clamp of the LEGACY score and served only when enabled
        i_legacy = seg.index("_composite_legacy = combined_score")
        self.assertLess(seg.index("_ml_trace[\"error\"] = f\"{type(_ml_exc).__name__}"), i_legacy)
        self.assertIn('_v2_served = bool(_composite_v2 is not None and _v2_cfg["enabled"])', seg)
        self.assertIn('combined_score = _composite_v2["score"]', seg)
        for k in ("composite_legacy", "composite_v2", "composite_v2_removed", "composite_v2_served"):
            self.assertRegex(seg, rf'"{k}": ')

    def test_attribution_table_reads_the_live_config(self):
        cfg = sa.composite_config()
        self.assertEqual(sa.action_for("volume", cfg)[0], "cut")
        self.assertEqual(sa.action_for("rel_strength", cfg)[0], "cut")
        self.assertEqual(sa.action_for("ml_v2", cfg)[0], "cut")
        self.assertEqual(sa.action_for("momentum", cfg)[0], "kept")


class TestDeepScoreEndToEnd(_DeepScoreHarness):
    def _v2(self, cfg, detail=_hot_detail):
        with patch("bot_engine.get_stock_details", side_effect=lambda t, _diag=None: detail()), \
                patch.dict(BASE_CONFIG, {"SCORE_COMPOSITE_V2": cfg}):
            return self._run("off")

    ON = {"enabled": True, "neutralize": ["volume", "vol_premium", "rel_strength"], "ml_neutral": True}
    OFF = dict(ON, enabled=False)

    def test_disabled_serves_the_legacy_composite_and_still_logs_v2(self):
        res, rec = self._v2(self.OFF)
        bd = rec["score_breakdown"]
        self.assertEqual(res["score_version"], "legacy")
        self.assertAlmostEqual(res["deep_score"], bd["composite_legacy"], delta=0.06)
        self.assertEqual(res["score_legacy"], bd["composite_legacy"])
        self.assertEqual(bd["composite_v2_served"], 0.0)
        self.assertIsNotNone(bd["composite_v2"])
        self.assertAlmostEqual(bd["composite_v2_removed"], 20.0, places=1)
        self.assertFalse(any(r.startswith("Score v2") for r in res["reasons"]))

    def test_enabled_serves_v2_and_the_legacy_composite_is_unchanged(self):
        off, rec_off = self._v2(self.OFF)
        on, rec_on = self._v2(self.ON)
        # the legacy arithmetic is identical with v2 on or off
        self.assertEqual(rec_on["score_breakdown"]["composite_legacy"], rec_off["score_breakdown"]["composite_legacy"])
        self.assertEqual(on["rules_only_score"], off["rules_only_score"])
        self.assertEqual(on["ml_only_score"], off["ml_only_score"])
        self.assertEqual(on["score_legacy"], round(off["deep_score"], 1))
        # the served score is v2, recomputable from the logged terms
        bd, ml = rec_on["score_breakdown"], rec_on["ml"]
        self.assertEqual(on["score_version"], "v2")
        self.assertEqual(bd["composite_v2_served"], 1.0)
        want = sc.composite_v2(bd["rules_raw"], {"volume": bd["volume"], "vol_premium": bd["vol_premium"],
                                                 "rel_strength": bd["rel_strength"]}, ml, sc.v2_config(self.ON))
        self.assertAlmostEqual(on["deep_score"], want["score"], delta=0.11)
        self.assertAlmostEqual(rec_on["score"], on["deep_score"], places=6)
        self.assertEqual(bd["volume"], 90.0)
        self.assertEqual(bd["vol_premium"], 8.0)
        self.assertEqual(bd["rel_strength"], 6.0)
        self.assertLess(on["deep_score"], off["deep_score"], "the hot name loses the anti-predictive points")
        self.assertTrue(any(r.startswith("Score v2 ") and "held neutral (-20.0 pts" in r for r in on["reasons"]), on["reasons"])

    def test_ml_conviction_is_logged_but_not_used(self):
        confident = {"ml_score": 85.0, "ml_confidence": 0.7, "ml_signal": "BUY", "model_type": "lightgbm_global",
                     "win_probability": 0.85, "regime_used": "global"}
        with patch("ml_model_v2.ml_score", return_value=confident):
            off, _ = self._v2(self.OFF)
            on, rec = self._v2(self.ON)
        ml = rec["ml"]
        self.assertTrue(ml["blended"] and not ml["gated"], ml)
        self.assertEqual(ml["ml_only_score"], 85.0, "the model's view is still logged")
        self.assertTrue(any(r.startswith("ML model (logged, not used by score v2):") for r in on["reasons"]), on["reasons"])
        self.assertTrue(any(r.startswith("ML model: BUY") for r in off["reasons"]), "legacy reason text unchanged")
        self.assertIn("ML input held at 50", next(r for r in on["reasons"] if r.startswith("Score v2")))
        bd = rec["score_breakdown"]
        rw, mw = ml["rule_weight"], ml["ml_weight"]
        rules_v2 = max(0.0, min(100.0, bd["rules_raw"] - bd["composite_v2_removed"]))
        self.assertAlmostEqual(on["deep_score"], max(0.0, min(100.0, rules_v2 * rw + 50.0 * mw)), delta=0.11)

    def test_a_failure_in_v2_serves_legacy_and_says_so(self):
        with patch("score_composite.composite_v2", side_effect=RuntimeError("boom")), \
                self.assertLogs("bot_engine", level="WARNING") as logs:
            res, rec = self._v2(self.ON)
        self.assertEqual(res["score_version"], "legacy")
        self.assertIsNone(rec["score_breakdown"]["composite_v2"])
        self.assertAlmostEqual(res["deep_score"], rec["score_breakdown"]["composite_legacy"], delta=0.06)
        self.assertTrue(any("composite v2 failed" in m and "serving the legacy score" in m for m in logs.output))

    def test_malformed_config_serves_legacy(self):
        res, rec = self._v2("yes please")
        self.assertEqual(res["score_version"], "legacy")
        self.assertEqual(rec["score_breakdown"]["composite_v2_served"], 0.0)


if __name__ == "__main__":
    unittest.main()
