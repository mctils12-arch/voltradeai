#!/usr/bin/env python3
"""
VolTradeAI — Critical Path Tests (Code Audit)
Tests: ML features, regime detection, signal integration, config consistency.
Runs without Alpaca/Finnhub API keys (pure unit tests).
"""
import os, sys, json, unittest
import numpy as np

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("VOLTRADE_DATA_DIR", "/tmp/vt_audit_test")
os.makedirs("/tmp/vt_audit_test", exist_ok=True)


class TestMLFeatureConsistency(unittest.TestCase):
    """Verify ML model feature definitions are consistent across modules."""

    def test_feature_count_is_34(self):
        from ml_model_v2 import FEATURE_COLS
        self.assertEqual(len(FEATURE_COLS), 34,
                         f"Expected 34 features, got {len(FEATURE_COLS)}")

    def test_system_config_feature_count_matches(self):
        from system_config import BASE_CONFIG
        from ml_model_v2 import FEATURE_COLS
        self.assertEqual(BASE_CONFIG["ML_FEATURE_COUNT"], len(FEATURE_COLS),
                         "system_config.ML_FEATURE_COUNT must match ml_model_v2.FEATURE_COLS")

    def test_no_duplicate_features(self):
        from ml_model_v2 import FEATURE_COLS
        self.assertEqual(len(FEATURE_COLS), len(set(FEATURE_COLS)),
                         f"Duplicate features: {[f for f in FEATURE_COLS if FEATURE_COLS.count(f) > 1]}")

    def test_feature_names_are_valid_identifiers(self):
        from ml_model_v2 import FEATURE_COLS
        for f in FEATURE_COLS:
            self.assertTrue(f.isidentifier(), f"Feature name '{f}' is not a valid identifier")


class TestRegimeDetection(unittest.TestCase):
    """Test regime classification matches expected thresholds."""

    def test_ml_classify_regime_bear(self):
        from ml_model_v2 import _classify_regime
        # VXX >= 1.15 should be bear
        self.assertEqual(_classify_regime(1.20, 1.0), "bear")
        # SPY < 0.94 should be bear
        self.assertEqual(_classify_regime(1.0, 0.90), "bear")

    def test_ml_classify_regime_bull(self):
        from ml_model_v2 import _classify_regime
        # VXX <= 0.95 AND SPY >= 0.98 should be bull
        self.assertEqual(_classify_regime(0.85, 1.02), "bull")

    def test_ml_classify_regime_neutral(self):
        from ml_model_v2 import _classify_regime
        # Everything else is neutral
        self.assertEqual(_classify_regime(1.0, 1.0), "neutral")
        self.assertEqual(_classify_regime(1.10, 1.0), "neutral")

    def test_system_config_regime_panic(self):
        from system_config import get_market_regime
        self.assertEqual(get_market_regime(1.35, 1.0), "PANIC")
        self.assertEqual(get_market_regime(1.0, 0.93), "PANIC")

    def test_system_config_regime_bear(self):
        from system_config import get_market_regime
        self.assertEqual(get_market_regime(1.20, 1.0), "BEAR")
        self.assertEqual(get_market_regime(1.0, 0.95), "BEAR")

    def test_system_config_regime_caution(self):
        from system_config import get_market_regime
        self.assertEqual(get_market_regime(1.10, 1.0), "CAUTION")

    def test_system_config_regime_bull(self):
        from system_config import get_market_regime
        self.assertEqual(get_market_regime(0.85, 1.02), "BULL")

    def test_system_config_regime_neutral(self):
        from system_config import get_market_regime
        # VXX at 1.03 (slightly above 1.02 threshold), SPY above MA — NEUTRAL not BULL
        self.assertEqual(get_market_regime(1.03, 1.01, markov_state=1), "NEUTRAL")

    def test_system_config_200d_slow_bear(self):
        from system_config import get_market_regime
        # 10+ days below 200d MA forces BEAR
        self.assertEqual(get_market_regime(1.0, 1.0, spy_below_200_days=15), "BEAR")
        # With panic VXX on top of slow bear -> PANIC
        self.assertEqual(get_market_regime(1.35, 1.0, spy_below_200_days=15), "PANIC")

    def test_neutral_regime_blocks_stock_trades(self):
        from system_config import get_adaptive_params
        params = get_adaptive_params(vxx_ratio=1.0, spy_vs_ma50=1.01, markov_state=1)
        if params.get("regime") == "NEUTRAL":
            self.assertEqual(params["MAX_POSITIONS"], 0,
                             "NEUTRAL regime must block new stock trades (MAX_POSITIONS=0)")


class TestMLModelFallback(unittest.TestCase):
    """Test ML model fallback behavior when no trained model exists."""

    def test_ml_score_fallback_returns_valid_dict(self):
        from ml_model_v2 import ml_score
        result = ml_score({"momentum_1m": 5.0, "rsi_14": 45, "volume_ratio": 1.5})
        self.assertIn("ml_score", result)
        self.assertIn("ml_confidence", result)
        self.assertIn("ml_signal", result)
        self.assertIn("model_type", result)
        self.assertGreaterEqual(result["ml_score"], 0)
        self.assertLessEqual(result["ml_score"], 100)

    def test_ml_score_handles_empty_features(self):
        from ml_model_v2 import ml_score
        result = ml_score({})
        self.assertIn("ml_score", result)
        self.assertIn("model_type", result)


class TestFracDiff(unittest.TestCase):
    """Test fractional differentiation helper."""

    def test_frac_diff_short_series(self):
        from ml_model_v2 import _frac_diff
        self.assertEqual(_frac_diff([100, 101, 102]), 0.0)

    def test_frac_diff_normal_series(self):
        from ml_model_v2 import _frac_diff
        closes = [100 + i * 0.5 for i in range(25)]
        result = _frac_diff(closes)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -3.0)
        self.assertLessEqual(result, 3.0)


class TestFeedbackTupleConsistency(unittest.TestCase):
    """Verify _build_feedback_training_data returns consistent tuples."""

    def test_none_return_has_4_elements(self):
        from ml_model_v2 import _build_feedback_training_data
        result = _build_feedback_training_data([])
        self.assertEqual(len(result), 4,
                         "None return must have 4 elements (was 3 before fix)")
        self.assertIsNone(result[0])


class TestAlpacaBaseURL(unittest.TestCase):
    """Verify Alpaca base URL is centralized, not hardcoded."""

    def test_no_hardcoded_paper_urls_in_bot_engine(self):
        with open(os.path.join(os.path.dirname(__file__), "bot_engine.py")) as f:
            content = f.read()
        # Count occurrences — only the env var default should remain
        import re
        hardcoded = re.findall(r'"https://paper-api\.alpaca\.markets[^"]*"', content)
        # Filter out the env var default definitions
        non_default = [h for h in hardcoded
                       if "environ.get" not in content.split(h)[0].split('\n')[-1]]
        self.assertEqual(len(non_default), 0,
                         f"Found hardcoded paper-api URLs: {non_default[:3]}")


class TestBacktestRegimeConsistency(unittest.TestCase):
    """Verify backtest regime parameters match live system."""

    def test_backtest_v2_neutral_blocks_trades(self):
        with open(os.path.join(os.path.dirname(__file__), "backtest_v2.py")) as f:
            content = f.read()
        # Find the NEUTRAL return line in get_adaptive_params
        lines = content.split('\n')
        for line in lines:
            if 'NEUTRAL' in line and 'max_pos' in line:
                self.assertIn('"max_pos": 0', line,
                              f"backtest_v2 NEUTRAL must have max_pos=0, got: {line.strip()}")
                break

    def test_backtest_v1028_neutral_blocks_trades(self):
        with open(os.path.join(os.path.dirname(__file__), "backtest_v1028_full.py")) as f:
            content = f.read()
        # Find the NEUTRAL return line
        lines = content.split('\n')
        for line in lines:
            if 'NEUTRAL' in line and 'max_pos' in line:
                self.assertIn('"max_pos": 0', line,
                              f"backtest_v1028 NEUTRAL must have max_pos=0, got: {line.strip()}")
                break


class TestTimezoneHandling(unittest.TestCase):
    """Verify timezone-aware ET calculations use ZoneInfo, not hardcoded UTC-4."""

    def _check_no_hardcoded_et_offset(self, filepath):
        with open(filepath) as f:
            content = f.read()
        # Check for the broken pattern: (now.hour - 4) % 24 or (now_utc.hour - 4) % 24
        import re
        matches = re.findall(r'\.hour\s*-\s*4\)\s*%\s*24', content)
        return matches

    def test_options_execution_no_hardcoded_offset(self):
        fp = os.path.join(os.path.dirname(__file__), "options_execution.py")
        matches = self._check_no_hardcoded_et_offset(fp)
        self.assertEqual(len(matches), 0,
                         f"options_execution.py still has hardcoded UTC-4 offset: {matches}")

    def test_position_sizing_no_hardcoded_offset(self):
        fp = os.path.join(os.path.dirname(__file__), "position_sizing.py")
        matches = self._check_no_hardcoded_et_offset(fp)
        self.assertEqual(len(matches), 0,
                         f"position_sizing.py still has hardcoded UTC-4 offset: {matches}")

    def test_instrument_selector_no_hardcoded_offset(self):
        fp = os.path.join(os.path.dirname(__file__), "instrument_selector.py")
        matches = self._check_no_hardcoded_et_offset(fp)
        self.assertEqual(len(matches), 0,
                         f"instrument_selector.py still has hardcoded UTC-4 offset: {matches}")


class TestRequirementsTxt(unittest.TestCase):
    """Verify requirements.txt is complete."""

    def test_all_required_packages_listed(self):
        with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
            content = f.read().lower()
        required = ["numpy", "scipy", "scikit-learn", "joblib", "lightgbm",
                     "yfinance", "pandas", "requests", "pytrends"]
        for pkg in required:
            self.assertIn(pkg, content,
                          f"Missing required package in requirements.txt: {pkg}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
