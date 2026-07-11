"""
Regression tests for tiered_strategy.py's Tier 1 (CSP core) position-slot
cap and the master kill-switch visibility contract.

No test file existed for this module before 2026-07-11 despite it being
the sole live code path for CSP/options trading (per system_config.py's
own comment: "CSP/options trades still fire via the tier engine (separate
code path)"). That coverage gap is why the bug below shipped invisibly
for 5+ weeks.

BUG (found + fixed 2026-07-11): tier1_csp_core() computed its options
position-slot cap from caps["MAX_POSITIONS"] — the STOCK position cap,
which system_config.py's regime blocks zero out in PANIC/BEAR/NEUTRAL to
block new stock longs. CSP silently inherited that 0 in exactly the three
regimes system_config's own comments say CSP should keep running in.
Live evidence: 17 options orders in a 200-order Alpaca window, all dated
2026-06-04/05, zero between 2026-06-09 and 2026-07-10 while 185 equity
orders filled in that same window. Fix: a dedicated, regime-constant
MAX_OPTIONS_POSITIONS cap in system_config.py.
"""
from unittest.mock import patch

import pytest

from tiered_strategy import TierContext, TieredStrategy, tier1_csp_core
from system_config import get_adaptive_params

FALLBACK_UNIVERSE = [
    "AMD", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "SPY", "QQQ", "IWM", "AVGO", "CRM", "ORCL", "INTC", "CAT", "GE",
]


def _ctx(vxx_ratio, spy_vs_ma50=1.0, positions=None, equity=100_000.0,
         peak_equity=None):
    return TierContext(
        equity=equity,
        peak_equity=peak_equity if peak_equity is not None else equity,
        buying_power=equity,
        positions=positions or [],
        vxx_ratio=vxx_ratio,
        spy_vs_ma50=spy_vs_ma50,
    )


@pytest.mark.parametrize("vxx_ratio,expected_regime", [
    (1.00, "NEUTRAL"),
    (1.20, "BEAR"),
    (1.50, "PANIC"),
])
def test_tier1_csp_core_produces_candidates_in_stress_regimes(vxx_ratio, expected_regime):
    """The exact bug: these three regimes zero MAX_POSITIONS (stock cap)
    but system_config's own comments say CSP keeps running in all three."""
    caps = get_adaptive_params(vxx_ratio=vxx_ratio, spy_vs_ma50=1.0)
    assert caps["regime"] == expected_regime  # sanity: test hits the intended regime
    assert caps["MAX_POSITIONS"] == 0          # stock cap is correctly zero here

    ctx = _ctx(vxx_ratio=vxx_ratio, positions=[])
    with patch("tiered_strategy._get_t1_universe", return_value=FALLBACK_UNIVERSE):
        actions = tier1_csp_core(ctx)

    assert len(actions) > 0, (
        f"{expected_regime} regime produced zero CSP candidates — "
        "this is the exact silent-blackout bug this test guards against"
    )
    assert all(a.action == "SELL_CSP" and a.tier == 1 for a in actions)


def test_tier1_csp_core_uses_dedicated_options_cap_not_stock_cap():
    """Direct pin: MAX_OPTIONS_POSITIONS must exist and stay nonzero even
    when MAX_POSITIONS (stock) is zeroed by the regime."""
    caps = get_adaptive_params(vxx_ratio=1.0, spy_vs_ma50=1.0)  # NEUTRAL
    assert caps["MAX_POSITIONS"] == 0
    assert caps.get("MAX_OPTIONS_POSITIONS", 0) > 0

    ctx = _ctx(vxx_ratio=1.0, positions=[])
    with patch("tiered_strategy._get_t1_universe", return_value=FALLBACK_UNIVERSE):
        actions = tier1_csp_core(ctx)

    expected = min(3, caps["MAX_OPTIONS_POSITIONS"], len(FALLBACK_UNIVERSE))
    assert len(actions) == expected


def test_tier1_csp_core_still_respects_options_cap_when_full():
    """Fix must not make the slot cap unlimited — six already-held option
    positions (== the default MAX_OPTIONS_POSITIONS) must leave zero slots."""
    held = [
        {"symbol": f"OPT{i}", "asset_class": "us_option", "market_value": 100}
        for i in range(6)
    ]
    ctx = _ctx(vxx_ratio=1.0, positions=held)  # NEUTRAL
    with patch("tiered_strategy._get_t1_universe", return_value=FALLBACK_UNIVERSE):
        actions = tier1_csp_core(ctx)
    assert actions == []


def test_run_tiers_exposes_killed_and_kill_reason_contract():
    """run_tiers()'s own docstring promises {"killed": bool, "kill_reason":
    str}. bot_engine.py never read these two keys before 2026-07-11, so a
    master_kill_switch firing left zero trace in the audit log — pin the
    contract so a future refactor can't silently drop it again."""
    ctx = _ctx(vxx_ratio=1.0, positions=[], equity=80_000.0, peak_equity=100_000.0)
    result = TieredStrategy().run_tiers(ctx)
    assert result["killed"] is True
    assert "DD kill" in result["kill_reason"]
    assert result["actions"] == []
    assert "tier_stats" in result
