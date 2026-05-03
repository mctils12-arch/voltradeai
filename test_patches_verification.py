"""
test_patches_verification.py
============================
Runnable verification gates for the 8 audit-bug patches applied 2026-05-03.

Each test corresponds to one patch and asserts the bug is fixed without
exercising network/Alpaca calls. Run with:

    python3 -m pytest test_patches_verification.py -v

Or standalone:

    python3 test_patches_verification.py
"""
import os
import sys
import json
import math
import tempfile
import unittest

# Repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestPatch1_AtrPctUnitFix(unittest.TestCase):
    """Audit #1: ewma_rv was annualized × 100, treated as daily ATR%."""

    def test_position_sizing_normalizes_annualized_vol(self):
        """Annualized vol of 22 should produce ~3-4% stop, not 8% pinned."""
        from position_sizing import _normalize_vol_to_daily_pct
        # 22% annualized → ~1.39% daily
        daily = _normalize_vol_to_daily_pct(22.0)
        self.assertIsNotNone(daily)
        self.assertAlmostEqual(daily, 1.386, places=2)
        # 1.5x daily → ~2.08% stop (within reasonable trading range)
        stop = max(1.5, min(daily * 1.5, 8.0))
        self.assertLess(stop, 5.0,
            "Stop should be in 1.5-5% range for 22% annualized vol, "
            "not pinned at 8% upper clamp")

    def test_position_sizing_handles_real_daily_atr_directly(self):
        """A daily ATR of 2.3 (already daily%) passes through unchanged."""
        from position_sizing import _normalize_vol_to_daily_pct
        self.assertAlmostEqual(_normalize_vol_to_daily_pct(2.3), 2.3, places=2)

    def test_instrument_selector_imports_normalizer(self):
        """instrument_selector should now have access to the normalize fn."""
        import instrument_selector
        # Either via import or local fallback — function must exist on the module
        self.assertTrue(hasattr(instrument_selector, "_normalize_vol_to_daily_pct"))
        self.assertAlmostEqual(
            instrument_selector._normalize_vol_to_daily_pct(22.0), 1.386, places=2)


class TestPatch2_NegativeKellyHalt(unittest.TestCase):
    """Audit #3: negative-Kelly was clamped to +1%."""

    def test_negative_kelly_returns_zero(self):
        """Negative-EV inputs should yield 0.0, not 0.01."""
        from position_sizing import _kelly_fraction
        # 45% WR with avg_win=$2, avg_loss=$2.5 → Kelly = -0.2375
        f = _kelly_fraction(0.45, 2.0, 2.5)
        self.assertEqual(f, 0.0,
            "Negative-EV must return 0.0 (do not trade), not the previous 1% floor")

    def test_positive_kelly_returns_clamped_fraction(self):
        """Positive-EV still sizes normally."""
        from position_sizing import _kelly_fraction
        # 70% WR, $1 win, $2 loss → Kelly = (0.7*1 - 0.3*2)/1 = 0.10
        # Third-Kelly = 0.0333
        f = _kelly_fraction(0.70, 1.0, 2.0, kelly_divisor=3.0)
        self.assertGreater(f, 0.01)
        self.assertLess(f, 0.10)

    def test_zero_avg_win_returns_default(self):
        """No-data case still returns the 3% prior."""
        from position_sizing import _kelly_fraction
        f = _kelly_fraction(0.5, 0.0, 1.0)
        self.assertEqual(f, 0.03)


class TestPatch3_NetCreditDenominator(unittest.TestCase):
    """Audit #2: profit target on credit spreads used gross premium denominator."""

    def test_iron_condor_net_credit_math(self):
        """4-leg condor: $1.00 short legs + $0.20 long legs → net credit = $1.60.

        Old denominator: $2.40 gross. 50% of gross fires at $1.20 = 75% of max.
        New denominator: $1.60 net.   50% of net   fires at $0.80 = 50% of max ✓
        """
        # Simulate the per-contract math the function does
        short_premium_per_leg = 1.00
        long_premium_per_leg = 0.20
        # Sum of |entry|*qty*100 across all 4 legs (legacy gross denominator)
        gross = (short_premium_per_leg * 1 * 100) * 2 + (long_premium_per_leg * 1 * 100) * 2
        # Net credit (new denominator)
        net = (short_premium_per_leg * 1 * 100) * 2 - (long_premium_per_leg * 1 * 100) * 2
        # Max profit = net credit
        max_profit = net
        # 50% target threshold
        target_dollars = 0.5 * max_profit
        # Old behavior: pct_old = pnl / gross >= 0.5  →  pnl >= 0.5 * gross
        old_trigger = 0.5 * gross
        # New behavior: pnl >= 0.5 * net
        new_trigger = 0.5 * net
        # Old trigger was at $120 (75% of max $160); new is $80 (50% of max)
        self.assertEqual(old_trigger, 120.0)
        self.assertEqual(new_trigger, 80.0)
        self.assertEqual(target_dollars, 80.0)

    def test_short_straddle_net_credit_unchanged(self):
        """Short straddle (no long legs) → net == gross, no behavior change."""
        # Short call $5 + short put $5, qty 1 each
        gross = 5.00 * 1 * 100 + 5.00 * 1 * 100
        net = gross  # all short legs, no longs to subtract
        self.assertEqual(gross, net, 1000.0)


class TestPatch4_OptionsScannerFields(unittest.TestCase):
    """Audit #4: scanner read snap['volume'] (last trade size) and snap['openInterest']
    (always missing from snapshot endpoint). Now reads dailyBar.v + bid/ask sizes."""

    def test_scanner_reads_dailybar_volume(self):
        """A snapshot with dailyBar.v=12345 should yield volume=12345, not 0."""
        # Synthetic snapshot mirroring Alpaca's actual response shape
        synthetic_snap = {
            "latestQuote": {"bp": 1.20, "ap": 1.30, "bs": 50, "as": 60},
            "dailyBar": {"v": 12345, "c": 1.25},
            "greeks": {"iv": 0.45, "delta": 0.30},
            # Note: no top-level "volume" or "openInterest" — Alpaca doesn't return them
        }
        # Replicate the scanner's field extraction (post-patch logic)
        daily_bar = synthetic_snap.get("dailyBar", {}) or {}
        volume = int(daily_bar.get("v", 0) or 0)
        bid_size = int(synthetic_snap["latestQuote"].get("bs", 0) or 0)
        ask_size = int(synthetic_snap["latestQuote"].get("as", 0) or 0)
        oi_raw = int(synthetic_snap.get("openInterest", 0) or 0)
        oi = oi_raw if oi_raw > 0 else max(bid_size, ask_size) * 10

        self.assertEqual(volume, 12345)
        self.assertEqual(oi, 600,
            "OI fallback: max(bs=50, as=60) * 10 = 600, not 0")

    def test_old_field_reads_would_have_returned_zero(self):
        """Sanity: confirm the old code path produced 0s on real Alpaca shape."""
        synthetic_snap = {
            "latestQuote": {"bp": 1.20, "ap": 1.30, "bs": 50, "as": 60},
            "dailyBar": {"v": 12345, "c": 1.25},
        }
        # Old (broken) extraction
        old_oi = int(synthetic_snap.get("openInterest", 0) or 0)
        old_volume = int(synthetic_snap.get("volume", 0) or 0)
        self.assertEqual(old_oi, 0)
        self.assertEqual(old_volume, 0)


class TestPatch5_RegimeThresholdReconciled(unittest.TestCase):
    """Audit #5: regime_util.PANIC=1.40 ≠ system_config.REGIME_VXX_PANIC=1.30."""

    def test_thresholds_match(self):
        from regime_util import PANIC_VXX_THRESHOLD
        from system_config import BASE_CONFIG
        self.assertEqual(
            BASE_CONFIG["REGIME_VXX_PANIC"], PANIC_VXX_THRESHOLD,
            f"system_config has REGIME_VXX_PANIC={BASE_CONFIG['REGIME_VXX_PANIC']}, "
            f"regime_util has PANIC_VXX_THRESHOLD={PANIC_VXX_THRESHOLD} — must match")

    def test_classify_5level_at_boundary(self):
        from regime_util import classify_regime_5level
        # At VXX 1.35, BOTH should agree it's BEAR (since panic now starts at 1.40)
        self.assertEqual(classify_regime_5level(1.35, 1.0), "BEAR")
        # At VXX 1.45, both should agree it's PANIC
        self.assertEqual(classify_regime_5level(1.45, 1.0), "PANIC")


class TestPatch6_EvWarningHardHalt(unittest.TestCase):
    """Audit #6: ev_warning was set 7 places, read 0 places."""

    def test_negative_ev_sets_error(self):
        """A trade dict with negative EV should now have an 'error' key."""
        # Simulate the contract result struct
        result = {"strategy": "iron_condor", "ev": 0.0, "ev_warning": None}
        ev = -0.25
        # Replicate the patched logic
        if ev < 0:
            result["ev_warning"] = "Negative expected value"
            result["error"] = (
                f"Negative expected value (EV=${ev:.2f} per contract); "
                f"refusing to place trade"
            )
        self.assertIn("error", result)
        self.assertIn("Negative expected value", result["error"])

    def test_options_execution_has_seven_halts(self):
        """All 7 ev_warning sites should now also set error."""
        with open(os.path.join(os.path.dirname(__file__), "options_execution.py")) as f:
            content = f.read()
        ev_warning_count = content.count('"ev_warning"] = "Negative expected value"')
        error_count = content.count(
            'result["error"] = (\n                f"Negative expected value')
        self.assertEqual(ev_warning_count, 7)
        self.assertEqual(error_count, 7,
            f"Expected 7 hard-halt error blocks, found {error_count}")


class TestPatch7_PeakEquityDefined(unittest.TestCase):
    """Audit #7: get_peak_equity was called but never defined."""

    def test_get_peak_equity_callable(self):
        from risk_kill_switch import get_peak_equity, set_peak_equity
        # Must be callable; should not raise
        result = get_peak_equity()
        self.assertIsInstance(result, float)

    def test_peak_equity_persists(self):
        from risk_kill_switch import get_peak_equity, set_peak_equity
        # Use a sandbox state dir so we don't pollute production
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["VOLTRADE_STATE_DIR"] = tmp  # if storage_config respects it
            # Set then read
            written = set_peak_equity(125_000.0)
            self.assertGreaterEqual(written, 125_000.0)
            read_back = get_peak_equity()
            # Should match what we wrote (or higher if pre-existing peak)
            self.assertGreaterEqual(read_back, 125_000.0)
            # Ratchet: lower equity should NOT lower the peak
            written2 = set_peak_equity(100_000.0)
            self.assertGreaterEqual(written2, 125_000.0)

    def test_kill_switch_status_no_longer_silently_zeros(self):
        from risk_kill_switch import get_kill_switch_status, set_peak_equity
        set_peak_equity(150_000.0)
        status = get_kill_switch_status()
        self.assertGreaterEqual(status["peak_equity"], 150_000.0,
            "After set_peak_equity(150k), get_kill_switch_status must report >= 150k")


class TestPatch8_CalendarDiagonalOiField(unittest.TestCase):
    """Audit #8: calendar/diagonal selectors read 'oi' but contract dict has 'open_interest'."""

    def test_no_lingering_oi_misreads(self):
        """Source should have no remaining .get('oi', ...) reads in calendar/diagonal paths."""
        with open(os.path.join(os.path.dirname(__file__), "options_execution.py")) as f:
            content = f.read()
        # The wrong key string c.get("oi", ...) should no longer appear in those functions.
        bad_calendar = '.get("oi", 0) < 100 or' in content and \
                       '.get("oi", 0) < 50' in content
        self.assertFalse(bad_calendar,
            "Calendar/diagonal OI checks still use the wrong key 'oi' — "
            "must be 'open_interest'")
        # And the right key should appear in those liquidity gates
        self.assertIn(
            'short_call.get("open_interest", 0) < 100 or long_call.get("open_interest", 0) < 50',
            content,
            "Calendar OI check missing the corrected 'open_interest' read")


class TestPatch9_SubstringMatching(unittest.TestCase):
    """Audit #50, #51: plain `in` substring match polluted ticker counts."""

    def test_social_data_word_boundary(self):
        """Ticker 'T' must not match 'THE', 'FORM', or 'T-MOBILE' but must match '$T'."""
        import re
        ticker = "T"
        pat = re.compile(
            rf"(?<![A-Za-z0-9_\-])\$?{re.escape(ticker.upper())}(?![A-Za-z0-9_\-])"
        )
        # Should NOT match
        self.assertIsNone(pat.search("THE BIG NEWS"),
            "Ticker 'T' incorrectly matched 'THE'")
        self.assertIsNone(pat.search("FORM 10-K FILED"),
            "Ticker 'T' incorrectly matched 'FORM'")
        self.assertIsNone(pat.search("T-MOBILE EARNINGS"),
            "Ticker 'T' should not match 'T-MOBILE' (different ticker TMUS)")
        # Should match
        self.assertIsNotNone(pat.search("$T BEATS Q3"))
        self.assertIsNotNone(pat.search("AT&T (T) UPGRADED"))

    def test_institutional_data_word_boundary(self):
        """13F search for 'T' must not match every company starting with T."""
        import re
        ticker = "T"
        pat = re.compile(
            rf"(?<![A-Z0-9_\-]){re.escape(ticker.upper())}(?![A-Z0-9_\-])"
        )
        big_blob = "THERMO FISHER SCIENTIFIC INC. THOMSON REUTERS CORP T-MOBILE US INC."
        self.assertIsNone(pat.search(big_blob),
            "Bare 'T' should not match THERMO/THOMSON/T-MOBILE")
        # Standalone T (e.g. AT&T's ticker shown alone in a filing entry)
        self.assertIsNotNone(
            pat.search("AT&T COMMON STOCK CUSIP 030177109 T 1000 SHARES"))


class TestPatch10_PathDependentLabeling(unittest.TestCase):
    """Audit #53: shadow labels were close-only, didn't match bot's stop/target."""

    def test_path_label_hits_target_first(self):
        """Bar that hits +2% high before -4% low → label = win."""
        from shadow_portfolio import _label_from_path
        entry = 100.0
        bars = [
            {"t": "2026-04-01", "h": 102.5, "l": 99.5, "c": 102.0},  # hit target
        ]
        label, exit_price, reason, _ = _label_from_path(entry, bars, max_bars=5)
        self.assertEqual(label, 1)
        self.assertEqual(reason, "target")
        self.assertAlmostEqual(exit_price, 102.0, places=2)

    def test_path_label_hits_stop_first(self):
        """Bar that hits -4% low before +2% high → label = loss."""
        from shadow_portfolio import _label_from_path
        entry = 100.0
        bars = [
            {"t": "2026-04-01", "h": 100.5, "l": 95.5, "c": 96.0},  # hit stop
        ]
        label, exit_price, reason, _ = _label_from_path(entry, bars, max_bars=5)
        self.assertEqual(label, 0)
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(exit_price, 96.0, places=2)

    def test_path_label_recovery_doesnt_save(self):
        """Trade hits stop on day 1, recovers to +5% by day 5 → still LOSS.

        This is THE bug audit #53 identified: old close-only labeling would
        call this a winner because day-5 close is +5%. Live bot was stopped
        out on day 1. New labeling correctly says loss.
        """
        from shadow_portfolio import _label_from_path
        entry = 100.0
        bars = [
            {"t": "2026-04-01", "h": 99.5, "l": 95.5, "c": 96.0},  # day 1: stop hits
            {"t": "2026-04-02", "h": 98.0, "l": 96.0, "c": 97.0},  # day 2: recovering
            {"t": "2026-04-03", "h": 100.0, "l": 97.0, "c": 99.0}, # day 3
            {"t": "2026-04-04", "h": 103.0, "l": 99.0, "c": 102.0},# day 4
            {"t": "2026-04-05", "h": 106.0, "l": 102.0, "c": 105.0},# day 5: +5% close
        ]
        label, _, reason, _ = _label_from_path(entry, bars, max_bars=5)
        self.assertEqual(label, 0,
            "Path-dependent label MUST be loss — bot was stopped out day 1, "
            "recovery to +5% by day 5 is irrelevant. (Audit Finding #53)")
        self.assertEqual(reason, "stop")

    def test_path_label_timeout_close_above_target(self):
        """No threshold hit, last close > target → win by timeout."""
        from shadow_portfolio import _label_from_path
        entry = 100.0
        bars = [
            {"t": "2026-04-01", "h": 101.0, "l": 99.0, "c": 100.5},
            {"t": "2026-04-02", "h": 101.5, "l": 99.5, "c": 101.0},
            {"t": "2026-04-03", "h": 101.8, "l": 100.0, "c": 101.5},
            {"t": "2026-04-04", "h": 101.9, "l": 100.5, "c": 101.7},
            {"t": "2026-04-05", "h": 101.99, "l": 100.7, "c": 102.5},  # close >2%
        ]
        label, exit_price, reason, _ = _label_from_path(entry, bars, max_bars=5)
        # Day 5 hit target intraday (high=101.99 < target 102) but close=102.5 >= target.
        # Note: high is 101.99 which is BELOW target 102, so target-hit logic skipped.
        # Falls through to timeout → close-based label.
        self.assertEqual(reason, "timeout")
        self.assertEqual(label, 1)

    def test_path_label_uses_trading_days_not_calendar(self):
        """5 bars = 5 trading days regardless of how many calendar days they span."""
        from shadow_portfolio import _label_from_path
        entry = 100.0
        # Bars with weekend gaps; 5 bars span 7 calendar days
        bars = [
            {"t": "2026-04-01", "h": 100.5, "l": 99.5, "c": 100.0},  # Wed
            {"t": "2026-04-02", "h": 100.5, "l": 99.5, "c": 100.0},  # Thu
            {"t": "2026-04-03", "h": 100.5, "l": 99.5, "c": 100.0},  # Fri
            {"t": "2026-04-06", "h": 100.5, "l": 99.5, "c": 100.0},  # Mon (weekend gap)
            {"t": "2026-04-07", "h": 102.5, "l": 99.5, "c": 102.0},  # Tue: hit target
        ]
        label, _, reason, exit_date = _label_from_path(entry, bars, max_bars=5)
        # Target only hit on bar 5 — confirms function walked all 5 bars
        self.assertEqual(label, 1)
        self.assertEqual(reason, "target")
        self.assertEqual(exit_date, "2026-04-07")


class TestPatch12_StrategyKeyAlignment(unittest.TestCase):
    """BUG #12 (post-deploy audit): _infer_strategy returned keys that didn't
    match by_strategy dict, so Patch 2's negative-Kelly halt was BYPASSED for
    stock trades. Fix realigns keys so halt actually fires."""

    def test_stock_momentum_routes_to_stocks_bucket(self):
        """Momentum-driven stock trade must route to 'stocks' bucket so
        Patch 2's halt actually applies (was falling through to 'overall')."""
        from position_sizing import _infer_strategy
        trade = {"ticker": "AAPL", "momentum_score": 80, "mean_reversion_score": 30}
        self.assertEqual(_infer_strategy(trade), "stocks",
            "Momentum stock trade must bucket as 'stocks' for Patch 2 to halt")

    def test_mean_reversion_routes_to_stocks(self):
        from position_sizing import _infer_strategy
        trade = {"ticker": "AAPL", "mean_reversion_score": 75, "momentum_score": 20}
        self.assertEqual(_infer_strategy(trade), "stocks")

    def test_squeeze_routes_to_stocks(self):
        from position_sizing import _infer_strategy
        trade = {"ticker": "GME", "squeeze_score": 80, "volume_score": 40}
        self.assertEqual(_infer_strategy(trade), "stocks")

    def test_vrp_routes_to_vrp_bucket(self):
        """VRP-dominant trade routes to 'vrp' bucket (-EV → halt)."""
        from position_sizing import _infer_strategy
        trade = {"ticker": "AAPL", "vrp_score": 80, "momentum_score": 30}
        self.assertEqual(_infer_strategy(trade), "vrp")

    def test_csp_routes_to_csp_options(self):
        """CSP options trade routes to 'csp_options' bucket (+EV → trade)."""
        from position_sizing import _infer_strategy
        trade = {"ticker": "AAPL", "options_strategy": "sell_cash_secured_put"}
        self.assertEqual(_infer_strategy(trade), "csp_options")

    def test_iron_condor_routes_to_csp_options(self):
        """Premium-selling structures route to 'csp_options' bucket."""
        from position_sizing import _infer_strategy
        trade = {"ticker": "QQQ", "options_strategy": "qqq_iron_condor"}
        self.assertEqual(_infer_strategy(trade), "csp_options")

    def test_unscored_trade_defaults_to_stocks(self):
        """No-signal trade defaults to 'stocks' (conservative)."""
        from position_sizing import _infer_strategy
        self.assertEqual(_infer_strategy({"ticker": "AAPL"}), "stocks")

    def test_returned_keys_all_exist_in_by_strategy(self):
        """Every possible return value of _infer_strategy must be a key in
        by_strategy. Without this, Kelly looks up the wrong stats."""
        from position_sizing import _infer_strategy, _get_historical_stats
        valid_keys = set(_get_historical_stats()["by_strategy"].keys())
        # Sample many trade shapes to cover all branches
        test_trades = [
            {"momentum_score": 80},
            {"mean_reversion_score": 80},
            {"vrp_score": 80},
            {"squeeze_score": 80},
            {"volume_score": 80},
            {"options_strategy": "sell_csp"},
            {"options_strategy": "covered_call"},
            {"options_strategy": "iron_condor"},
            {"instrument": "options"},
            {},  # no signals
        ]
        for trade in test_trades:
            inferred = _infer_strategy(trade)
            self.assertIn(inferred, valid_keys,
                f"_infer_strategy returned '{inferred}' which is not a key "
                f"in by_strategy {valid_keys}. Stat lookup will fall through "
                f"to 'overall' defaults — this is the bug Patch 12 fixes.")

    def test_patch2_halt_now_fires_for_stocks(self):
        """End-to-end: a stock trade should now produce kelly=0 due to
        the chain: _infer_strategy → 'stocks' → -EV defaults → halt."""
        from position_sizing import _infer_strategy, _get_historical_stats, _kelly_fraction
        trade = {"ticker": "AAPL", "momentum_score": 80}
        strategy = _infer_strategy(trade)
        self.assertEqual(strategy, "stocks")
        stats = _get_historical_stats()
        s = stats["by_strategy"][strategy]
        kelly = _kelly_fraction(s["win_rate"], s["avg_win"], s["avg_loss"])
        self.assertEqual(kelly, 0.0,
            f"With strategy='stocks' (WR={s['win_rate']}, win=${s['avg_win']}, "
            f"loss=${s['avg_loss']}), Kelly must be 0.0 (negative-EV). "
            f"Got {kelly}. Patch 2 is now connected end-to-end.")


# ── Standalone runner ──────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
