"""
KNOWN BROKEN #18 continuation (2026-07-29): live TIER2-ERROR occurrences on
2026-07-27/28 (12+ occurrences, essentially every Tier-2 scan during market
hours both days) show surface_score_stats.cumulative_elapsed_sec landing at
312-365s per scan cycle — ALONE exceeding the 300s daemon RPC timeout. Setup
7 (options_scanner.py's _check_ticker, vol_surface.get_surface_score()) was
still being called unconditionally for every high_iv/anchor candidate not
already found by Setup 3 (the 2026-07-28 dead-compute fix only removed the
double-pay case, not the aggregate cost across ~450 distinct candidates).
Since scan_options() runs synchronously inside bot_engine.scan_market()'s
step6, a timed-out run_full_scan() RPC returns NOTHING — not just no options
trades, the entire Tier-2 stock+options scan for that cycle.

This suite pins the new scan-wide budget guard: once
vol_surface.surface_score_scan_budget_exceeded(SETUP7_SURFACE_SCORE_SCAN_
BUDGET_SEC) trips, _check_ticker must stop calling get_surface_score() for
the rest of the scan cycle (recording the skip via
record_surface_score_budget_skip() instead), while candidates evaluated
before the trip must behave exactly as before — this is a scan-completion
fix, not a change to Setup 7's own scoring logic.

Same execute-scan_options()-end-to-end-with-mocks pattern as
test_options_scanner_surface_score_dead_compute.py.
"""
import options_scanner


def _run_scan_options_with(monkeypatch, candidates, r3_side_effect, surface_call_counter,
                            budget_exceeded=False):
    monkeypatch.setattr(options_scanner, "_is_regular_hours", lambda: True)
    monkeypatch.setattr(options_scanner, "_get_vxx_ratio", lambda: 1.0)
    monkeypatch.setattr(options_scanner, "_get_spy_vs_ma50", lambda: 1.0)
    monkeypatch.setattr(options_scanner, "_setup_vxx_panic_put_sale", lambda vxx, spy: None)
    monkeypatch.setattr(options_scanner, "_fetch_earnings_calendar", lambda days_ahead=21: {})
    monkeypatch.setattr(options_scanner, "_get_options_candidates", lambda: candidates)
    monkeypatch.setattr(options_scanner, "_setup_high_iv_premium_sale", r3_side_effect)

    def _fake_get_surface_score(tkr):
        surface_call_counter.append(tkr)
        return {"vrp": {"vrp_20d": 0.10}, "surface_score": 90}

    monkeypatch.setattr("vol_surface.get_surface_score", _fake_get_surface_score)
    monkeypatch.setattr("vol_surface.get_vrp", lambda tkr: {"vrp_20d": 0.0})
    monkeypatch.setattr("vol_surface.surface_score_scan_budget_exceeded", lambda budget: budget_exceeded)
    return options_scanner.scan_options()


def test_setup7_runs_normally_when_budget_not_exceeded(monkeypatch):
    calls = []

    def r3(tkr, price, vxx_ratio):
        return None

    result = _run_scan_options_with(
        monkeypatch,
        candidates=[("MSFT", 300.0, "high_iv")],
        r3_side_effect=r3,
        surface_call_counter=calls,
        budget_exceeded=False,
    )

    assert calls.count("MSFT") == 1, "Setup 7 must still run for every candidate while under budget"


def test_setup7_skipped_for_every_candidate_once_budget_exceeded(monkeypatch):
    """The exact failure mode being fixed: once the scan-wide budget trips,
    get_surface_score() must not run at all — not for the first candidate
    reached after the trip, not for any candidate after it."""
    calls = []

    def r3(tkr, price, vxx_ratio):
        return None

    result = _run_scan_options_with(
        monkeypatch,
        candidates=[("MSFT", 300.0, "high_iv"), ("AAPL", 200.0, "anchor")],
        r3_side_effect=r3,
        surface_call_counter=calls,
        budget_exceeded=True,
    )

    # scan_options() also calls get_surface_score("SPY") once per scan for its
    # own market-wide VRP block, unrelated to Setup 7 / the per-candidate
    # budget gate — assert on the actual candidates, same convention as
    # test_options_scanner_surface_score_dead_compute.py's own tests.
    assert "MSFT" not in calls, "get_surface_score must not run for MSFT once the scan budget is exceeded"
    assert "AAPL" not in calls, "get_surface_score must not run for AAPL once the scan budget is exceeded"
    tickers = [o["ticker"] for o in result["opportunities"]]
    assert tickers == [], "no Setup 3 opportunities were configured, and Setup 7 was fully skipped"


def test_setup7_budget_skip_does_not_block_setup3_output(monkeypatch):
    """Skipping Setup 7 must not affect Setup 3 — the two setups are
    independent; only the expensive VRP-scoring path is budget-gated."""
    calls = []

    def r3(tkr, price, vxx_ratio):
        return {"ticker": tkr, "setup": "high_iv_premium_sale", "score": 75,
                "options_strategy": "iron_condor", "side": "sell"}

    result = _run_scan_options_with(
        monkeypatch,
        candidates=[("AAPL", 200.0, "high_iv")],
        r3_side_effect=r3,
        surface_call_counter=calls,
        budget_exceeded=True,
    )

    assert "AAPL" not in calls, "AAPL was already found by Setup 3, so Setup 7 was never eligible regardless of budget"
    tickers = [o["ticker"] for o in result["opportunities"]]
    assert tickers.count("AAPL") == 1  # Setup 3's opportunity survives untouched


def test_setup7_records_a_budget_skip_via_the_real_vol_surface_function(monkeypatch):
    """End-to-end (not a mocked stand-in for record_surface_score_budget_skip)
    — confirms the real accumulator sees the skip, the way the live
    surface_score_stats audit line does."""
    import vol_surface
    vol_surface.reset_surface_score_scan_stats()

    def r3(tkr, price, vxx_ratio):
        return None

    monkeypatch.setattr(options_scanner, "_is_regular_hours", lambda: True)
    monkeypatch.setattr(options_scanner, "_get_vxx_ratio", lambda: 1.0)
    monkeypatch.setattr(options_scanner, "_get_spy_vs_ma50", lambda: 1.0)
    monkeypatch.setattr(options_scanner, "_setup_vxx_panic_put_sale", lambda vxx, spy: None)
    monkeypatch.setattr(options_scanner, "_fetch_earnings_calendar", lambda days_ahead=21: {})
    monkeypatch.setattr(options_scanner, "_get_options_candidates",
                         lambda: [("MSFT", 300.0, "high_iv")])
    monkeypatch.setattr(options_scanner, "_setup_high_iv_premium_sale", r3)
    monkeypatch.setattr("vol_surface.get_vrp", lambda tkr: {"vrp_20d": 0.0})
    # scan_options() calls get_surface_score("SPY") once per scan for its own
    # market-wide VRP block, unrelated to Setup 7 / the per-candidate budget
    # gate — faked here (not left to hit the real network) so this test's
    # "real accumulator" assertions below aren't polluted by an unrelated
    # live network call, and so we can tell that call apart from any (there
    # should be none) MSFT call.
    real_calls = []
    monkeypatch.setattr("vol_surface.get_surface_score",
                         lambda tkr: (real_calls.append(tkr), {"vrp": {"vrp_20d": 0.0}, "surface_score": 0})[1])
    # Force the budget to already be blown for this scan cycle.
    monkeypatch.setattr("vol_surface.surface_score_scan_budget_exceeded", lambda budget: True)

    options_scanner.scan_options()

    assert "MSFT" not in real_calls, "get_surface_score must not run for MSFT once the scan budget is exceeded"
    stats = vol_surface.get_surface_score_scan_stats()
    assert stats["skipped_budget"] == 1
