"""
PIPELINE 2026-10-26 (counterfactual analysis follow-up): the shadow record
now carries a `score_breakdown` dict — 14 individual sub-scores and
adjustments that compose the rules-only score.  This enables gate-2
decomposition: measuring which sub-scores are predictive vs anti-predictive,
the investigation the anti-predictive composite score finding (#1463) calls
for.
"""
import json
import unittest

import shadow_portfolio as sp


class TestShadowScoreBreakdown(unittest.TestCase):
    def setUp(self):
        import tempfile, os
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = (sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH)
        sp.SHADOW_LOG_PATH = os.path.join(self._tmp.name, "shadow.json")
        sp.SHADOW_LOCK_PATH = sp.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH = self._orig
        self._tmp.cleanup()

    def _last(self):
        with open(sp.SHADOW_LOG_PATH) as f:
            return json.load(f)[-1]

    def test_breakdown_persisted_with_all_14_keys(self):
        bd = {
            "momentum": 65.0, "mean_reversion": 52.3, "vrp": 70.0,
            "squeeze": 50.0, "volume": 60.0, "vol_premium": 8.0,
            "macro": -5.0, "news": 10.0, "adx": 3.0, "obv": 5.0,
            "intel": -2.0, "alt": 7.0, "social": -3.0, "fundamental": 4.0,
        }
        sp.log_candidate("AMD", {}, 71.2, "taken", score_breakdown=bd)
        rec = self._last()
        self.assertIsNotNone(rec["score_breakdown"])
        for k in sp.SCORE_BREAKDOWN_KEYS:
            self.assertIn(k, rec["score_breakdown"])
            self.assertEqual(rec["score_breakdown"][k], bd[k])

    def test_absent_breakdown_is_none(self):
        sp.log_candidate("AMD", {}, 50.0, "rejected_score")
        self.assertIsNone(self._last()["score_breakdown"])
        self.assertIsNone(sp._sanitize_score_breakdown({}))
        self.assertIsNone(sp._sanitize_score_breakdown(None))
        self.assertIsNone(sp._sanitize_score_breakdown("bad"))

    def test_extra_keys_are_dropped(self):
        bd = {"momentum": 65.0, "unknown_key": 99.0}
        result = sp._sanitize_score_breakdown(bd)
        self.assertNotIn("unknown_key", result)
        self.assertEqual(result["momentum"], 65.0)

    def test_nan_and_non_numeric_coerced_to_none(self):
        bd = {"momentum": float("nan"), "vrp": "bad", "volume": True}
        result = sp._sanitize_score_breakdown(bd)
        self.assertIsNone(result["momentum"])
        self.assertIsNone(result["vrp"])
        self.assertIsNone(result["volume"])

    def test_old_records_without_breakdown_still_load(self):
        sp.log_candidate("AMD", {}, 60.0, "taken")
        recs = sp._load_shadow_log()
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0].get("score_breakdown"))


if __name__ == "__main__":
    unittest.main()
