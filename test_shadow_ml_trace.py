"""
PIPELINE 2026-10-16 (bot cockpit audit §4, layer 4 "conviction"): ml_score()
returns model_type / regime_used / win_probability / ml_confidence /
ml_signal per candidate and deep_score dropped all of it. The shadow record
now carries an `ml` trace (None when absent) so the page can show what the
model said and the band reports can split real-model vs fallback scores.
"""
import json
import unittest

import shadow_portfolio as sp


class TestShadowMlTrace(unittest.TestCase):
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

    def test_trace_is_persisted_with_fixed_keys(self):
        sp.log_candidate("AMD", {"momentum_1m": 1.0}, 71.2, "taken", ml={
            "model_type": "lightgbm_neutral", "regime_used": "neutral", "win_probability": 0.6123456,
            "ml_score": 61.2, "ml_confidence": 0.22, "ml_signal": "BUY", "gated": False, "blended": True,
            "rule_weight": 0.72, "ml_weight": 0.28, "rules_only_score": 74.0, "ml_only_score": 61.2,
            "final_score": 71.2, "vxx_source": "macro", "error": None, "extra_key": "dropped",
        })
        ml = self._last()["ml"]
        self.assertEqual(ml["model_type"], "lightgbm_neutral")
        self.assertEqual(ml["regime_used"], "neutral")
        self.assertEqual(ml["win_probability"], 0.6123)
        self.assertIs(ml["gated"], False)
        self.assertIs(ml["blended"], True)
        self.assertEqual(ml["rule_weight"], 0.72)
        self.assertIsNone(ml["error"])
        self.assertNotIn("extra_key", ml)

    def test_absent_trace_is_none_not_zeros(self):
        sp.log_candidate("AMD", {"momentum_1m": 1.0}, 50.0, "rejected_score")
        self.assertIsNone(self._last()["ml"])
        self.assertIsNone(sp.sanitize_ml_trace({}))
        self.assertIsNone(sp.sanitize_ml_trace("lightgbm"))

    def test_fallback_and_error_are_recorded_as_such(self):
        sp.log_candidate("AMD", {}, 55.0, "rejected_score",
                         ml={"model_type": "fallback_rules", "regime_used": "fallback", "ml_score": 58.0, "blended": False})
        ml = self._last()["ml"]
        self.assertEqual(ml["model_type"], "fallback_rules")
        self.assertIs(ml["blended"], False)
        sp.log_candidate("AMD", {}, 55.0, "rejected_score", ml={"error": "ImportError: No module named lightgbm", "blended": False})
        self.assertEqual(self._last()["ml"]["error"], "ImportError: No module named lightgbm")
        self.assertIsNone(self._last()["ml"]["model_type"])

    def test_garbage_values_are_coerced(self):
        t = sp.sanitize_ml_trace({"win_probability": "0.7", "ml_score": float("nan"), "gated": "yes", "model_type": 12, "error": ""})
        self.assertIsNone(t["win_probability"])
        self.assertIsNone(t["ml_score"])
        self.assertIs(t["gated"], True)
        self.assertEqual(t["model_type"], "12")
        self.assertIsNone(t["error"])

    def test_old_records_without_ml_still_read(self):
        sp.log_candidate("AMD", {}, 60.0, "taken")
        recs = sp._load_shadow_log()
        self.assertEqual(len(recs), 1)


if __name__ == "__main__":
    unittest.main()


class TestDeepScoreTraceWiring(unittest.TestCase):
    """Wiring pin: deep_score builds _ml_trace from ml_score()'s result and the
    blend, records the exception text instead of `pass`, and hands the trace
    to log_candidate."""
    def setUp(self):
        with open("bot_engine.py", encoding="utf-8") as f:
            self.src = f.read()

    def test_trace_is_built_and_persisted(self):
        for needle in ('_ml_trace = {"blended": False, "gated": None, "error": None}',
                       '"model_type": ml_result.get("model_type")',
                       '"win_probability": ml_result.get("win_probability")',
                       '_ml_trace.update({"blended": True, "gated": ml_conviction < 8,',
                       'ml={**_ml_trace, "final_score": combined_score}'):
            self.assertIn(needle, self.src, needle)

    def test_the_ml_exception_is_recorded_not_swallowed(self):
        self.assertIn('except Exception as _ml_exc:', self.src)
        self.assertIn('_ml_trace["error"] = f"{type(_ml_exc).__name__}: {_ml_exc}"[:200]', self.src)
        self.assertNotIn('pass  # ML not available, use rule-based only', self.src)
