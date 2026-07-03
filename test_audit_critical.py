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
        # PATCH 5 RECONCILIATION 2026-05-03: panic threshold moved to 1.40
        # to match regime_util.PANIC_VXX_THRESHOLD (canonical source).
        # 1.35 is now BEAR; full panic requires VXX >= 1.40.
        self.assertEqual(get_market_regime(1.45, 1.0), "PANIC")
        # SPY >6% below 50d MA alone is BEAR not PANIC under reconciled
        # logic — only VXX panic qualifies for the 30%-cap regime.
        self.assertEqual(get_market_regime(1.0, 0.93), "BEAR")

    def test_system_config_regime_bear(self):
        from system_config import get_market_regime
        # VXX above BEAR_VXX_THRESHOLD (1.15) → BEAR
        self.assertEqual(get_market_regime(1.20, 1.0), "BEAR")
        # SPY below BEAR_SPY_THRESHOLD (0.94) → BEAR
        # NOTE: this assertion was wrong pre-patch too — 0.95 is above the
        # 0.94 threshold so it correctly returns NEUTRAL, not BEAR.
        self.assertEqual(get_market_regime(1.0, 0.93), "BEAR")

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
        # 10+ days below 200d MA forces BEAR (audit Finding #5b: now actually fires)
        self.assertEqual(get_market_regime(1.0, 1.0, spy_below_200_days=15), "BEAR")
        # With panic VXX on top of slow bear -> PANIC (threshold raised to 1.40)
        self.assertEqual(get_market_regime(1.45, 1.0, spy_below_200_days=15), "PANIC")

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

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), "backtest_v2.py")),
        "backtest_v2.py not yet ported — standing top-priority backtest rebuild (CLAUDE.md). "
        "Re-activates automatically once the file exists.")
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

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), "backtest_v1028_full.py")),
        "backtest_v1028_full.py not yet ported — standing top-priority backtest rebuild (CLAUDE.md). "
        "Re-activates automatically once the file exists.")
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


class TestBacktestV2Engine(unittest.TestCase):
    """Offline tests for the rebuilt backtest engine (backtest_v2.py).
    All bars are injected — no network, no keys (CI-safe)."""

    @staticmethod
    def _synth_bars(n, start=100.0, daily=0.001, vol_shape=0.0):
        """Deterministic uptrending OHLCV bars (no randomness — reproducible)."""
        dates, o, h, l, c, v = [], [], [], [], [], []
        px = start
        from datetime import date, timedelta
        d0 = date(2022, 1, 3)
        for i in range(n):
            wiggle = 0.002 * ((i % 7) - 3) / 3.0 + vol_shape
            op = px
            cl = px * (1 + daily + wiggle)
            dates.append((d0 + timedelta(days=i)).isoformat())
            o.append(round(op, 4)); c.append(round(cl, 4))
            h.append(round(max(op, cl) * 1.004, 4))
            l.append(round(min(op, cl) * 0.996, 4))
            v.append(5_000_000)
            px = cl
        return {"date": dates, "open": o, "high": h, "low": l,
                "close": c, "volume": v}

    @staticmethod
    def _bull_vxx(spy_bars):
        """VXX series whose close/30d-avg ratio stays <= 0.90 (BULL input)."""
        n = len(spy_bars["date"])
        closes = [30.0 * (0.995 ** i) for i in range(n)]  # steadily falling
        return {"date": list(spy_bars["date"]), "open": closes,
                "high": closes, "low": closes, "close": closes,
                "volume": [1_000_000] * n}

    def test_neutral_default_blocks_all_entries(self):
        """With no VXX data (vxx_ratio degrades to 1.0) the regime is NEUTRAL
        and max_pos=0 — the engine must take ZERO trades even on a perfect
        momentum tape. This is the regime-consistency contract with live."""
        from backtest_v2 import run_backtest
        bars = self._synth_bars(420, daily=0.004)
        out = run_backtest("TEST", "momentum", 1, bars=bars, spy=bars, vxx=None)
        mom = out["results"][0]
        self.assertEqual(mom["n_trades"], 0,
                         "NEUTRAL regime (max_pos=0) must block all entries")
        self.assertEqual(mom["data_quality"], "degraded")

    def test_bull_regime_allows_entries_and_no_lookahead(self):
        from backtest_v2 import run_backtest, COST_PCT
        bars = self._synth_bars(420, daily=0.004)
        vxx = self._bull_vxx(bars)
        out = run_backtest("TEST", "momentum", 1, bars=bars, spy=bars, vxx=vxx)
        mom = out["results"][0]
        self.assertGreater(mom["n_trades"], 0, "BULL regime should permit entries")
        # No-lookahead: every entry price must equal some bar's OPEN (+cost),
        # never the signal bar's close.
        opens = {round(op * (1 + COST_PCT), 2) for op in bars["open"]}
        for t in out["all_trades"]["momentum"]:
            self.assertIn(round(t["entry_price"], 2), opens,
                          f"entry {t['entry_price']} is not a next-bar open fill")
            self.assertGreater(t["holding_days"], 0)
            # exit strictly after entry (no same-signal-bar exits)
            self.assertLess(t["date_entry"], t["date_exit"])

    def test_scheduler_contract_fields(self):
        """bot.ts overnight scheduler reads strategy/sharpe/totalReturn/
        maxDrawdown from each results[] item — they must exist and be numeric."""
        from backtest_v2 import run_backtest
        bars = self._synth_bars(420, daily=0.004)
        out = run_backtest("TEST", "all", 1, bars=bars, spy=bars,
                           vxx=self._bull_vxx(bars))
        self.assertTrue(out["ok"])
        self.assertIsInstance(out["results"], list)
        for r in out["results"]:
            for field in ("strategy", "sharpe", "totalReturn", "maxDrawdown"):
                self.assertIn(field, r)
            self.assertIsInstance(r["sharpe"], (int, float))
        json.dumps(out)  # bot.ts JSON.parses stdout — must be serializable

    def test_artifact_trade_schema(self):
        """Trade records must match backtest_10yr_results.json's trade keys."""
        from backtest_v2 import run_backtest
        bars = self._synth_bars(420, daily=0.004)
        out = run_backtest("TEST", "momentum", 1, bars=bars, spy=bars,
                           vxx=self._bull_vxx(bars))
        expected = {"date_exit", "date_entry", "ticker", "instrument",
                    "strategy", "entry_price", "exit_price", "holding_days",
                    "pnl", "net_pct", "exit_reason", "score", "vxx_ratio", "vrp"}
        trades = out["all_trades"]["momentum"]
        self.assertTrue(trades, "expected at least one trade under BULL")
        for t in trades:
            self.assertEqual(set(t.keys()), expected)
            self.assertIn(t["exit_reason"], {"stop_loss", "take_profit",
                                             "time_stop", "end_of_backtest"})

    def test_squeeze_refuses_to_fabricate(self):
        """No historical short-interest data -> squeeze must report zero trades
        with a note, never made-up results (CLAUDE.md priority 2)."""
        from backtest_v2 import run_backtest
        bars = self._synth_bars(300)
        out = run_backtest("TEST", "squeeze", 1, bars=bars, spy=bars, vxx=None)
        sq = out["results"][0]
        self.assertEqual(sq["n_trades"], 0)
        self.assertIn("note", sq)

    def test_metrics_math(self):
        from backtest_v2 import compute_metrics, TRADING_DAYS
        # Equity doubling linearly over exactly one trading year
        equity = [100.0 + 100.0 * i / TRADING_DAYS for i in range(TRADING_DAYS + 1)]
        m = compute_metrics(equity, TRADING_DAYS)
        self.assertAlmostEqual(m["total_return_pct"], 100.0, places=1)
        self.assertAlmostEqual(m["cagr_pct"], 100.0, places=1)
        self.assertEqual(m["max_drawdown_pct"], 0.0)
        # A drawdown is measured peak-to-trough
        m2 = compute_metrics([100, 120, 90, 110], 4)
        self.assertAlmostEqual(m2["max_drawdown_pct"], 25.0, places=1)

    def test_regime_blocking_consistent_with_live_config(self):
        """The safety-critical contract: the same regimes that block live
        trading (MAX_POSITIONS=0 in system_config.get_adaptive_params) must
        block backtest entries (max_pos=0), and vice versa."""
        from backtest_v2 import get_adaptive_params as bt_params
        from system_config import get_adaptive_params as live_params
        # Inputs chosen to hit each regime per regime_util thresholds
        cases = {
            "PANIC":   dict(vxx_ratio=1.50, spy_vs_ma50=1.00),
            "BEAR":    dict(vxx_ratio=1.20, spy_vs_ma50=1.00),
            "CAUTION": dict(vxx_ratio=1.08, spy_vs_ma50=1.00),
            "NEUTRAL": dict(vxx_ratio=1.00, spy_vs_ma50=0.99),
            "BULL":    dict(vxx_ratio=0.85, spy_vs_ma50=1.02),
        }
        from system_config import get_market_regime
        for regime, inputs in cases.items():
            self.assertEqual(get_market_regime(**inputs), regime,
                             f"test inputs no longer map to {regime} — update test")
            live_blocked = live_params(**inputs)["MAX_POSITIONS"] == 0
            bt_blocked = bt_params(regime)["max_pos"] == 0
            self.assertEqual(bt_blocked, live_blocked,
                             f"{regime}: backtest blocking (max_pos) diverges "
                             f"from live (MAX_POSITIONS)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
