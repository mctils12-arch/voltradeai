"""
KNOWN BROKEN #18 continuation (2026-07-28): scan_options()'s Setup 7 block
used to call vol_surface.get_surface_score(tkr) UNCONDITIONALLY for every
"high_iv"/"anchor" candidate, even when Setup 3 (r3, computed just above in
the same _check_ticker closure) already produced an opportunity for that
exact ticker — in which case Setup 7's own result could only ever be
discarded by the pre-existing `not any(o.get("ticker") == tkr for o in
found)` dedup check. get_surface_score() is expensive (an uncached,
un-throttled options-chain + spot + bars fetch via build_surface(), see
vol_surface.py's 2026-07-28 comment) and was being paid in full before that
dedup check ever ran. This suite pins that the dedup check now gates the
WHOLE block (skipping get_surface_score entirely when already found) with
IDENTICAL final output to the pre-fix ordering — a pure dead-compute
removal, not a behavior change, same class as v1.0.503's Setup 4/5 removal.

Full scan_options() with every network call mocked, following the
established pattern for testing this specific closure (test_options_v134_
fixes.py's own test_csp_disabled_in_scan_options/test_low_iv_and_gamma_pin_
disabled_in_scan_options tests inspect scan_options()'s source rather than
exercising the closure directly, since _check_ticker is defined inline and
not independently importable) — this suite goes one step further and
actually EXECUTES scan_options() end-to-end with mocks, so it is a true
regression ratchet on `found`'s contents, not just a source-text pin.
"""
import inspect

import options_scanner


def _run_scan_options_with(monkeypatch, candidates, r3_side_effect, surface_call_counter):
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
    # scan_options()'s own market-wide VRP block (unrelated to Setup 7 /
    # per-candidate logic) always calls get_vrp("SPY") once per scan —
    # mocked here purely to avoid a real, flaky, irrelevant network call in
    # this test; it is not part of what these tests are pinning.
    monkeypatch.setattr("vol_surface.get_vrp", lambda tkr: {"vrp_20d": 0.0})
    return options_scanner.scan_options()


def test_surface_score_skipped_when_setup3_already_found_the_ticker(monkeypatch):
    """AAPL: Setup 3 finds an opportunity -> Setup 7 must NOT call
    get_surface_score for AAPL at all (its result would be discarded anyway)."""
    calls = []

    def r3(tkr, price, vxx_ratio):
        return {"ticker": tkr, "setup": "high_iv_premium_sale", "score": 75,
                "options_strategy": "iron_condor", "side": "sell"}

    result = _run_scan_options_with(
        monkeypatch,
        candidates=[("AAPL", 200.0, "high_iv")],
        r3_side_effect=r3,
        surface_call_counter=calls,
    )

    assert "AAPL" not in calls, "get_surface_score must not run once Setup 3 already found this ticker"
    tickers = [o["ticker"] for o in result["opportunities"]]
    assert tickers.count("AAPL") == 1  # Setup 3's single opportunity, no VRP-boosted duplicate


def test_surface_score_still_runs_when_setup3_found_nothing(monkeypatch):
    """MSFT: Setup 3 finds nothing -> Setup 7 must still evaluate the VRP/
    surface-score path (this is the real, load-bearing case Setup 7 exists
    for) — the fix must not skip it unconditionally."""
    calls = []

    def r3(tkr, price, vxx_ratio):
        return None

    result = _run_scan_options_with(
        monkeypatch,
        candidates=[("MSFT", 300.0, "high_iv")],
        r3_side_effect=r3,
        surface_call_counter=calls,
    )

    # scan_options()'s own market-wide VRP block also calls get_surface_score("SPY")
    # once per scan, unrelated to this candidate loop — assert on MSFT specifically.
    assert calls.count("MSFT") == 1, "get_surface_score must still run when Setup 3 found nothing for this ticker"


def test_surface_score_dedup_ordering_produces_identical_output_to_pre_fix_semantics(monkeypatch):
    """Two tickers: one already found by Setup 3 (skip Setup 7 entirely, new
    behavior), one not found by Setup 3 (Setup 7 runs and, since our fake
    get_surface_score always passes the vrp/surf_score/earnings gate,
    contributes its own opportunity) — final opportunity set for both
    tickers must match what the OLD post-hoc dedup would also have produced,
    since the pre-fix code discarded r7 in exactly the same case."""
    calls = []
    msft_call_count = {"n": 0}

    def r3(tkr, price, vxx_ratio):
        if tkr == "AAPL":
            return {"ticker": "AAPL", "setup": "high_iv_premium_sale", "score": 75,
                    "options_strategy": "iron_condor", "side": "sell"}
        # MSFT: _check_ticker calls this SAME function twice (once for Setup 3's
        # r3, once for Setup 7's r7) — a stateful mock stands in for "Setup 3's
        # pass found nothing, Setup 7's pass (gated on the surface score) does",
        # purely to exercise the append-when-truthy wiring; real-world r3/r7
        # calls share identical args so would agree, but that determinism is
        # incidental to what this test is pinning (does Setup 7 still run and
        # append its own result when Setup 3's didn't find one).
        msft_call_count["n"] += 1
        if msft_call_count["n"] == 1:
            return None  # Setup 3's call
        return {"ticker": "MSFT", "setup": "high_iv_premium_sale", "score": 60,
                "options_strategy": "iron_condor", "side": "sell"}  # Setup 7's r7 call

    result = _run_scan_options_with(
        monkeypatch,
        candidates=[("AAPL", 200.0, "high_iv"), ("MSFT", 300.0, "high_iv")],
        r3_side_effect=r3,
        surface_call_counter=calls,
    )

    assert "AAPL" not in calls  # Setup 7's call skipped for AAPL (Setup 3 already found it)
    assert calls.count("MSFT") == 1  # Setup 7's call still ran for MSFT (Setup 3 found nothing)
    tickers = [o["ticker"] for o in result["opportunities"]]
    assert tickers.count("AAPL") == 1  # from Setup 3 only
    assert tickers.count("MSFT") == 1  # from Setup 7 (Setup 3 found nothing)


def test_scan_options_resets_surface_score_scan_stats_before_the_worker_pool(monkeypatch):
    """Source-text pin (same convention as test_options_v134_fixes.py's
    test_csp_disabled_in_scan_options): reset_surface_score_scan_stats()
    must be called before the ThreadPoolExecutor that fans out
    _check_ticker across candidates, so get_surface_score_scan_stats()
    reflects THIS scan cycle, not a running lifetime total."""
    source = inspect.getsource(options_scanner.scan_options)
    reset_pos = source.find("reset_surface_score_scan_stats()")
    pool_pos = source.find("pool.map(_check_ticker")
    assert reset_pos != -1, "scan_options() must call reset_surface_score_scan_stats()"
    assert pool_pos != -1, "expected the _check_ticker ThreadPoolExecutor block"
    assert reset_pos < pool_pos, "reset must happen before the concurrent _check_ticker fan-out"
