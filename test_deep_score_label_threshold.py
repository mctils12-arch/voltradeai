"""
[RULE-REVIEW] 2026-10-26: deep_score labelled every shadow record against a
fixed 63 — its `_scan_params.get("MIN_SCORE", 63) if '_scan_params' in
locals() else 63` could never bind, because _scan_params is a local of
_scan_market_inner, not of deep_score — while the scan gates at the
adaptive 63/67/75. The label now comes from _shadow_label_for(score,
min_score), with the scan computing its adaptive threshold once before the
deep-score fan-out and passing it in. Label only; no gate moves.
"""
import inspect
import re
import unittest

import bot_engine
from system_config import BASE_CONFIG


class TestShadowLabelFor(unittest.TestCase):
    def test_default_threshold_is_base_config_min_score(self):
        base = float(BASE_CONFIG["MIN_SCORE"])
        self.assertEqual(bot_engine._shadow_label_for(base, None)[0], "taken")
        self.assertEqual(bot_engine._shadow_label_for(base - 0.1, None)[0], "rejected_score")

    def test_adaptive_threshold_wins_when_passed(self):
        # 70 clears the base 63 but not a BEAR/PANIC 75 — the old code
        # labelled this "taken" every time.
        self.assertEqual(bot_engine._shadow_label_for(70.0, 75)[0], "rejected_score")
        self.assertEqual(bot_engine._shadow_label_for(70.0, 67)[0], "taken")
        self.assertEqual(bot_engine._shadow_label_for(75.0, 75)[0], "taken")

    def test_reason_names_the_threshold_actually_used(self):
        decision, reason = bot_engine._shadow_label_for(70.0, 75)
        self.assertEqual(decision, "rejected_score")
        self.assertIn("MIN_SCORE 75", reason)
        self.assertIn("70.0", reason)
        _, reason_default = bot_engine._shadow_label_for(70.0, None)
        self.assertIn(f"MIN_SCORE {float(BASE_CONFIG['MIN_SCORE']):g}", reason_default)

    def test_numeric_strings_and_ints_are_accepted(self):
        self.assertEqual(bot_engine._shadow_label_for(70.0, "75")[0], "rejected_score")
        self.assertEqual(bot_engine._shadow_label_for(70, 63)[0], "taken")


class TestDeepScoreWiring(unittest.TestCase):
    def test_signature_has_min_score_default_none(self):
        sig = inspect.signature(bot_engine.deep_score)
        self.assertIn("min_score", sig.parameters)
        self.assertIsNone(sig.parameters["min_score"].default)
        # Existing positional callers (lookup_ticker, tests) keep working.
        self.assertEqual(list(sig.parameters)[:2], ["ticker", "quick_result"])

    def test_label_block_uses_the_helper_and_not_the_dead_locals_lookup(self):
        src = inspect.getsource(bot_engine.deep_score)
        self.assertIn("_shadow_label_for(combined_score, min_score)", src)
        self.assertNotIn("'_scan_params' in locals()", src)
        self.assertNotIn('"_scan_params" in locals()', src)

    def test_scan_computes_the_adaptive_threshold_once_and_passes_it(self):
        src = inspect.getsource(bot_engine._scan_market_inner)
        calc = src.find("_adaptive_min_score = _gap_label(")
        fan_out = src.find("def _deep_one(args):")
        self.assertTrue(-1 < calc < fan_out, "the threshold is computed before the fan-out, once")
        self.assertIn("min_score=_adaptive_min_score)", src)
        self.assertEqual(src.count("min_score=_adaptive_min_score"), 1)
        window = src[calc - 400:calc + 600]
        self.assertIn("except Exception as _ams_err:", window)
        self.assertIn(".warning(", window, "an unavailable snapshot is logged, never silent")

    def test_label_threshold_and_gate_threshold_read_the_same_inputs(self):
        src = inspect.getsource(bot_engine._scan_market_inner)

        def kwargs_of(call_marker):
            at = src.find(call_marker)
            self.assertNotEqual(at, -1, call_marker)
            # Walk to the matching close paren — the label call nests
            # float(_m.get(...)) inside its arguments.
            open_at = src.index("(", at + len(call_marker) - 1)
            depth, i = 0, open_at
            while i < len(src):
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            body = src[open_at:i]
            return set(re.findall(r"(?:^|[(,\s])(\w+)=", body))

        label = kwargs_of("_adaptive_min_score = _gap_label(")
        gate = kwargs_of("_scan_params = _gap(")
        self.assertEqual(label, gate, "the label must be judged by the same adaptive inputs the gate uses")
        self.assertEqual(label, {"vxx_ratio", "spy_vs_ma50", "spy_below_200_days"})


if __name__ == "__main__":
    unittest.main()
