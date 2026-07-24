"""
Regression tests for backtest_v2.py's liquidity-tiered execution cost.

MEASUREMENT INTEGRITY (2026-07-23, [RULE-REVIEW]): the engine used to apply
a single flat COST_PCT (5 bps) to every fill regardless of trading volume —
realistic for a liquid mega-cap/ETF, fiction for a thin small/micro-cap
(REASONING STANDARD #6). liquidity_cost_pct() tiers the per-side cost by
trailing volume instead. These tests prove: (1) the tiers are ordered the
right way (illiquid costs more); (2) the function is a pure, deterministic
function of the bars/index given — no randomness, unlike the live per-fill
haircut in server/bot.ts — because PROMOTION RULE 3 compares Sharpe/
max-drawdown across code changes, which a non-reproducible backtest would
make meaningless; (3) the fallback path when there's no volume history;
(4) an end-to-end behavioral proof that an illiquid tape now returns less
than an identical liquid tape once costs are the only difference.

All bars are injected — no network, no keys (CI-safe).
"""
import unittest
from datetime import date, timedelta

from backtest_v2 import (
    liquidity_cost_pct,
    run_backtest,
    COST_PCT,
    _LIQUIDITY_TIERS,
    _ILLIQUID_COST_PCT,
)


def _bars_with_volume(n, volume, start=100.0, daily=0.004, vol_shape=0.0):
    """Deterministic uptrending OHLCV bars at a constant volume (no
    randomness — reproducible), mirroring TestBacktestV2Engine's fixture."""
    dates, o, h, l, c, v = [], [], [], [], [], []
    px = start
    d0 = date(2022, 1, 3)
    for i in range(n):
        wiggle = 0.002 * ((i % 7) - 3) / 3.0 + vol_shape
        op = px
        cl = px * (1 + daily + wiggle)
        dates.append((d0 + timedelta(days=i)).isoformat())
        o.append(round(op, 4)); c.append(round(cl, 4))
        h.append(round(max(op, cl) * 1.004, 4))
        l.append(round(min(op, cl) * 0.996, 4))
        v.append(volume)
        px = cl
    return {"date": dates, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _bull_vxx(bars):
    """VXX series whose close/30d-avg ratio stays <= 0.90 (BULL input)."""
    n = len(bars["date"])
    closes = [30.0 * (0.995 ** i) for i in range(n)]
    return {"date": list(bars["date"]), "open": closes, "high": closes,
            "low": closes, "close": closes, "volume": [1_000_000] * n}


class TestLiquidityCostTiers(unittest.TestCase):
    def test_tier_values_match_documented_midpoints(self):
        """Each breakpoint returns the documented midpoint of the matching
        live server/bot.ts tier, and tiers are monotonically increasing as
        liquidity drops."""
        liquid = _bars_with_volume(25, volume=25_000_000)
        mid = _bars_with_volume(25, volume=10_000_000)
        thin = _bars_with_volume(25, volume=2_000_000)
        illiquid = _bars_with_volume(25, volume=500_000)

        self.assertEqual(liquidity_cost_pct(liquid, 24), 0.00035)
        self.assertEqual(liquidity_cost_pct(mid, 24), 0.00075)
        self.assertEqual(liquidity_cost_pct(thin, 24), 0.00115)
        self.assertEqual(liquidity_cost_pct(illiquid, 24), _ILLIQUID_COST_PCT)

        # strictly increasing cost as liquidity drops
        costs = [liquidity_cost_pct(b, 24) for b in (liquid, mid, thin, illiquid)]
        self.assertEqual(costs, sorted(costs))
        self.assertLess(costs[0], costs[-1])

    def test_boundary_is_strictly_greater_than(self):
        """avg_vol exactly ON a breakpoint falls into the NEXT (costlier)
        tier — matches server/bot.ts's own strict '>' semantics."""
        exactly_20m = _bars_with_volume(25, volume=20_000_000)
        self.assertEqual(liquidity_cost_pct(exactly_20m, 24), 0.00075,
                          "exactly 20,000,000 must not qualify for the >20M tier")

    def test_trailing_window_is_20_days_and_no_lookahead(self):
        """A volume spike AFTER bar i must not affect the cost computed at
        bar i (no lookahead); a spike more than 20 days before bar i must
        have rolled out of the trailing window."""
        bars = _bars_with_volume(60, volume=25_000_000)
        # Make bar 30 (and everything after it) illiquid — should not affect
        # the cost computed at bar 29 or earlier.
        for j in range(30, 60):
            bars["volume"][j] = 100_000
        self.assertEqual(liquidity_cost_pct(bars, 29), 0.00035,
                          "future-bar volume leaked into an earlier bar's cost")
        # 20+ days after the spike started, bar 29's liquid volume has
        # rolled out of the trailing-20-day window computed at bar 50.
        self.assertEqual(liquidity_cost_pct(bars, 50), _ILLIQUID_COST_PCT)

    def test_fallback_with_no_volume_history(self):
        empty = {"volume": []}
        self.assertEqual(liquidity_cost_pct(empty, 0), COST_PCT)

    def test_documented_tiers_are_internally_consistent(self):
        """The _LIQUIDITY_TIERS breakpoints must be in descending threshold
        order (the loop assumes this) and every value must sit strictly
        below _ILLIQUID_COST_PCT."""
        thresholds = [t for t, _ in _LIQUIDITY_TIERS]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))
        for _, cost in _LIQUIDITY_TIERS:
            self.assertLess(cost, _ILLIQUID_COST_PCT)


class TestLiquidityCostDeterminism(unittest.TestCase):
    def test_same_inputs_produce_byte_identical_results(self):
        """A backtest must return the EXACT same result every run — unlike
        the live per-fill haircut's random jitter, a non-reproducible
        backtest would make PROMOTION RULE 3's before/after Sharpe and
        max-drawdown comparison across code changes meaningless."""
        bars = _bars_with_volume(420, volume=3_000_000)
        vxx = _bull_vxx(bars)
        out1 = run_backtest("TEST", "momentum", 1, bars=bars, spy=bars, vxx=vxx)
        out2 = run_backtest("TEST", "momentum", 1, bars=bars, spy=bars, vxx=vxx)
        # generated_at is a wall-clock timestamp; strip it before comparing.
        out1.pop("generated_at"); out2.pop("generated_at")
        self.assertEqual(out1, out2)


class TestIlliquidTapeCostsMoreEndToEnd(unittest.TestCase):
    def test_illiquid_ticker_underperforms_an_identical_liquid_tape(self):
        """Same price path, same signals, same regime — the ONLY difference
        is trading volume. The illiquid version must show a lower total
        return than the liquid version, because it pays a bigger execution
        cost on every fill. (Under the pre-fix flat COST_PCT, this assertion
        fails — both would tie exactly, since the flat cost only depended on
        the ticker's, not the tier's cost.)"""
        liquid_bars = _bars_with_volume(420, volume=30_000_000)
        illiquid_bars = _bars_with_volume(420, volume=500_000)
        vxx = _bull_vxx(liquid_bars)

        liquid_out = run_backtest("LIQUID", "momentum", 1, bars=liquid_bars,
                                   spy=liquid_bars, vxx=vxx)
        illiquid_out = run_backtest("ILLIQUID", "momentum", 1, bars=illiquid_bars,
                                     spy=illiquid_bars, vxx=vxx)

        liquid_trades = liquid_out["all_trades"]["momentum"]
        illiquid_trades = illiquid_out["all_trades"]["momentum"]
        self.assertTrue(liquid_trades and illiquid_trades,
                         "expected trades under BULL on both tapes")
        # Same number of trades on the same signal/price path — only the
        # cost per fill differs.
        self.assertEqual(len(liquid_trades), len(illiquid_trades))

        liquid_return = liquid_out["results"][0]["total_return_pct"]
        illiquid_return = illiquid_out["results"][0]["total_return_pct"]
        self.assertLess(illiquid_return, liquid_return,
                         "an illiquid tape must return less than an "
                         "identical liquid tape once execution cost is the "
                         "only difference")


if __name__ == "__main__":
    unittest.main()
