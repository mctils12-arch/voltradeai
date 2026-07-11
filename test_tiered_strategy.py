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
from unittest.mock import patch, MagicMock

import pytest

from tiered_strategy import (
    TierContext, TieredStrategy, tier1_csp_core, log_masterkill_csp_shadow,
)
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


# ── log_masterkill_csp_shadow — RULE REVIEW evidence gate (item #20) ────────
#
# master_kill_switch blocks Tier 1 CSP along with T2-4 even though CSP holds
# none of the exposure that typically trips it. Before any threshold/design
# change, CLAUDE.md's RULE REVIEW requires counterfactual evidence: log what
# Tier 1 would have proposed, never act on it. These tests pin that the
# logging (a) only fires for candidates with a real price (backfill_outcomes
# can't label an entry_price<=0 record), (b) never raises even if the shadow
# log itself fails, and (c) is actually wired into run_tiers()'s kill branch
# without changing what run_tiers() returns.

def _layer2_cache_with_prices(prices: dict):
    return {"scores": [{"ticker": t, "price": p} for t, p in prices.items()]}


def test_log_masterkill_csp_shadow_logs_only_priced_candidates():
    ctx = _ctx(vxx_ratio=1.0, positions=[])  # NEUTRAL, T1 fully open
    # FALLBACK_UNIVERSE's first 3 tickers become T1's candidates (cap=3);
    # give only the first two a price in the layer2 cache, and keep both
    # under tier1_csp_core's own affordability gate (~$84 at $100k equity/
    # MAX_OPTIONS_PCT=0.08) so they aren't filtered before reaching us.
    priced = {"AMD": 50.0, "AAPL": 60.0}
    with patch("tiered_strategy._get_t1_universe", return_value=FALLBACK_UNIVERSE), \
         patch("csp_universe._load_layer2_cache", return_value=_layer2_cache_with_prices(priced)), \
         patch("shadow_portfolio.log_candidate") as mock_log:
        logged = log_masterkill_csp_shadow(ctx, "Portfolio invested 92% >= 90%")

    assert logged == 2  # MSFT (3rd candidate) has no price -> skipped
    logged_tickers = {c.kwargs["ticker"] for c in mock_log.call_args_list}
    assert logged_tickers == {"AMD", "AAPL"}
    for call in mock_log.call_args_list:
        assert call.kwargs["decision"] == "rejected_masterkill"
        assert call.kwargs["decision_reason"] == "Portfolio invested 92% >= 90%"
        assert call.kwargs["entry_price"] > 0


def test_log_masterkill_csp_shadow_skips_all_when_no_prices_available():
    ctx = _ctx(vxx_ratio=1.0, positions=[])
    with patch("tiered_strategy._get_t1_universe", return_value=FALLBACK_UNIVERSE), \
         patch("csp_universe._load_layer2_cache", return_value=None), \
         patch("shadow_portfolio.log_candidate") as mock_log:
        logged = log_masterkill_csp_shadow(ctx, "reason")

    assert logged == 0
    mock_log.assert_not_called()


def test_log_masterkill_csp_shadow_returns_zero_when_tier1_has_nothing_to_propose():
    """Options slots already full -> tier1_csp_core() returns [] -> nothing
    to log, and the (mocked, would-fail-if-called) shadow logger is never
    even reached."""
    held = [
        {"symbol": f"OPT{i}", "asset_class": "us_option", "market_value": 100}
        for i in range(6)
    ]
    ctx = _ctx(vxx_ratio=1.0, positions=held)
    with patch("shadow_portfolio.log_candidate") as mock_log:
        logged = log_masterkill_csp_shadow(ctx, "reason")

    assert logged == 0
    mock_log.assert_not_called()


def test_log_masterkill_csp_shadow_never_raises_on_logger_failure():
    """Visibility code must never be able to break the kill-switch path it
    observes — a raising shadow_portfolio.log_candidate must not propagate."""
    ctx = _ctx(vxx_ratio=1.0, positions=[])
    with patch("tiered_strategy._get_t1_universe", return_value=FALLBACK_UNIVERSE), \
         patch("csp_universe._load_layer2_cache",
               return_value=_layer2_cache_with_prices({"AMD": 150.0})), \
         patch("shadow_portfolio.log_candidate", side_effect=RuntimeError("disk full")):
        logged = log_masterkill_csp_shadow(ctx, "reason")  # must not raise

    assert logged == 0


def test_run_tiers_logs_masterkill_shadow_without_changing_return_shape():
    """Wiring pin: run_tiers()'s kill branch must call the shadow logger
    exactly once, and its return value must stay byte-identical to before
    (actions still [], tier_stats still {}) — this is visibility bolted on,
    not a behavior change."""
    ctx = _ctx(vxx_ratio=1.0, positions=[], equity=80_000.0, peak_equity=100_000.0)
    with patch("tiered_strategy.log_masterkill_csp_shadow") as mock_shadow:
        result = TieredStrategy().run_tiers(ctx)

    mock_shadow.assert_called_once()
    assert mock_shadow.call_args.args[0] is ctx
    assert "DD kill" in mock_shadow.call_args.args[1]
    assert result["killed"] is True
    assert result["actions"] == []
    assert result["tier_stats"] == {}
