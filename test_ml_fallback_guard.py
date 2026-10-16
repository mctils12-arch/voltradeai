"""
REPAIR 2026-10-16 (bot cockpit audit, S4 failure signature): deep_score's
ML blend guard compared model_type against the literal "fallback", but
ml_model_v2._rule_based_fallback labels itself "fallback_rules". Result: a
missing or >ML_MAX_AGE_HOURS model silently blended the rules heuristic at
the ML weight and audited it as "ML model: … (NN% confidence)". This pins
the tested helper deep_score now uses.
"""
import re
import unittest

from bot_engine import _ml_result_is_model


class TestMlFallbackGuard(unittest.TestCase):
    def test_real_models_blend(self):
        for mt in ("lightgbm_global", "lightgbm_neutral", "lightgbm_bear", "lightgbm_bull"):
            self.assertTrue(_ml_result_is_model({"model_type": mt, "ml_score": 61.0}), mt)

    def test_the_live_fallback_label_is_excluded(self):
        """The exact string ml_model_v2._rule_based_fallback returns."""
        from ml_model_v2 import _rule_based_fallback
        fb = _rule_based_fallback({"momentum_1m": 3.0, "rsi_14": 30, "volume_ratio": 2.0})
        self.assertEqual(fb["model_type"], "fallback_rules")
        self.assertFalse(_ml_result_is_model(fb))

    def test_any_fallback_flavour_and_bad_shapes_are_excluded(self):
        for bad in ({"model_type": "fallback"}, {"model_type": "fallback_x"}, {"model_type": ""},
                    {}, None, "lightgbm_global", {"ml_score": 70}):
            self.assertFalse(_ml_result_is_model(bad), repr(bad))

    def test_deep_score_uses_the_helper_not_the_literal(self):
        """Wiring pin: the blend branch must call the helper; the old literal
        comparison must be gone so the bug cannot be reintroduced by hand."""
        with open("bot_engine.py", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("if _ml_result_is_model(ml_result):", src)
        self.assertNotRegex(src, re.compile(r'model_type"\)\s*!=\s*"fallback"'))


if __name__ == "__main__":
    unittest.main()
