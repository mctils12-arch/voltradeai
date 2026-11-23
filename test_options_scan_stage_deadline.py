"""
[REPAIR 2026-11-23] KNOWN BROKEN #18, the root cause.

The options scanner's per-ticker stage had no wall-clock bound: its length is
(candidates / 3 per second), about 200 s on production, inside a
run_full_scan RPC the daemon abandons at 300 s. On 2026-11-23 16:52-20:16Z
the 19 scans that finished took 239-296 s and 4 returned nothing.

Pins:
  - scan_options(deadline_ts=...) starts no ticker at or after the deadline,
    keeps what it found before it, and counts what it did not reach;
  - with no deadline it reads every candidate, as before;
  - get_options_trades passes its deadline through;
  - bot_engine gives the stage a deadline measured from the scan's own start
    and writes the stage's count to the timing file;
  - the configured deadline leaves room for the steps that follow the stage
    inside the daemon's request timeout.

Same execute-scan_options()-with-mocks pattern as
test_options_scanner_surface_score_scan_budget.py.
"""
import inspect
import re
import time

import options_scanner
import system_config


def _mock_scan(monkeypatch, candidates, r3):
    monkeypatch.setattr(options_scanner, "_is_regular_hours", lambda: True)
    monkeypatch.setattr(options_scanner, "_get_vxx_ratio", lambda: 1.0)
    monkeypatch.setattr(options_scanner, "_get_spy_vs_ma50", lambda: 1.0)
    monkeypatch.setattr(options_scanner, "_setup_vxx_panic_put_sale", lambda vxx, spy: None)
    monkeypatch.setattr(options_scanner, "_fetch_earnings_calendar_with_status",
                        lambda days_ahead=21: ({}, True))
    monkeypatch.setattr(options_scanner, "_get_options_candidates", lambda: candidates)
    monkeypatch.setattr(options_scanner, "_setup_high_iv_premium_sale", r3)
    monkeypatch.setattr(options_scanner, "_log_setup3_earnings_refusals", lambda *a, **k: None)
    monkeypatch.setattr("vol_surface.get_surface_score",
                        lambda tkr: {"vrp": {"vrp_20d": 0.0}, "surface_score": 0})
    monkeypatch.setattr("vol_surface.get_vrp", lambda tkr: {"vrp_20d": 0.0})


def _condor(tkr, price, vxx_ratio):
    return {"ticker": tkr, "setup": "high_iv_premium_sale", "score": 75,
            "options_strategy": "iron_condor", "side": "sell", "price": price}


CANDS = [("AAA", 50.0, "anchor"), ("BBB", 40.0, "high_iv"), ("CCC", 30.0, "high_iv"),
         ("DDD", 20.0, "high_iv")]


def test_stage_deadline_passed_semantics():
    assert options_scanner.stage_deadline_passed(None) is False
    assert options_scanner.stage_deadline_passed(100.0, now=99.9) is False
    assert options_scanner.stage_deadline_passed(100.0, now=100.0) is True
    assert options_scanner.stage_deadline_passed(time.time() - 1) is True
    assert options_scanner.stage_deadline_passed(time.time() + 3600) is False


def test_no_deadline_reads_every_candidate(monkeypatch):
    read = []

    def r3(tkr, price, vxx):
        read.append(tkr)
        return _condor(tkr, price, vxx)

    _mock_scan(monkeypatch, CANDS, r3)
    res = options_scanner.scan_options()
    assert sorted(read) == ["AAA", "BBB", "CCC", "DDD"]
    assert res["deadline_not_reached"] == 0
    assert res["scanned"] == 4
    stats = options_scanner.last_deadline_stats()
    assert stats["deadline_given"] is False and stats["not_reached"] == 0
    assert stats["candidates"] == 4


def test_deadline_already_passed_starts_no_ticker(monkeypatch):
    read = []

    def r3(tkr, price, vxx):
        read.append(tkr)
        return _condor(tkr, price, vxx)

    _mock_scan(monkeypatch, CANDS, r3)
    res = options_scanner.scan_options(deadline_ts=time.time() - 1)
    assert read == [], "no ticker may start once the deadline has passed"
    assert res["opportunities"] == []
    assert res["deadline_not_reached"] == 4
    assert res["scanned"] == 0, "a ticker that was not read is not counted as scanned"
    stats = options_scanner.last_deadline_stats()
    assert stats["deadline_given"] is True
    assert stats["not_reached"] == 4 and stats["candidates"] == 4
    assert sorted(stats["first_not_reached"]) == ["AAA", "BBB", "CCC", "DDD"]


def test_deadline_mid_stage_keeps_what_was_found_and_drops_the_tail(monkeypatch):
    """The stage's clock passes the deadline after the second ticker starts:
    the first two are read and kept, the rest are counted, in candidate order
    (anchors, then largest movers), so the tail is what is lost."""
    read = []
    clock = {"passed": False}

    def r3(tkr, price, vxx):
        read.append(tkr)
        if len(read) >= 2:
            clock["passed"] = True
        return _condor(tkr, price, vxx)

    _mock_scan(monkeypatch, CANDS, r3)
    monkeypatch.setattr(options_scanner, "stage_deadline_passed",
                        lambda deadline_ts, now=None: clock["passed"])
    # one worker so the order of starts is the candidate order
    real_pool = options_scanner.ThreadPoolExecutor
    monkeypatch.setattr(options_scanner, "ThreadPoolExecutor",
                        lambda max_workers=1: real_pool(max_workers=1))
    res = options_scanner.scan_options(deadline_ts=time.time() + 3600)
    assert read == ["AAA", "BBB"]
    assert sorted(o["ticker"] for o in res["opportunities"]) == ["AAA", "BBB"]
    assert res["deadline_not_reached"] == 2
    assert options_scanner.last_deadline_stats()["first_not_reached"] == ["CCC", "DDD"]


def test_stats_reset_between_scans(monkeypatch):
    _mock_scan(monkeypatch, CANDS, _condor)
    options_scanner.scan_options(deadline_ts=time.time() - 1)
    assert options_scanner.last_deadline_stats()["not_reached"] == 4
    options_scanner.scan_options()
    assert options_scanner.last_deadline_stats()["not_reached"] == 0


def test_get_options_trades_passes_the_deadline_through(monkeypatch):
    seen = {}

    def fake_scan(deadline_ts=None):
        seen["deadline_ts"] = deadline_ts
        return {"opportunities": []}

    monkeypatch.setattr(options_scanner, "scan_options", fake_scan)
    monkeypatch.setattr("market_calendar.should_skip_new_options", lambda: (False, ""))
    options_scanner.get_options_trades(100_000, [], deadline_ts=12345.0)
    assert seen["deadline_ts"] == 12345.0
    options_scanner.get_options_trades(100_000, [])
    assert seen["deadline_ts"] is None


def test_config_deadline_leaves_room_inside_the_daemon_timeout():
    """What follows the stage in one scan: the tier engine's CSP prefetch
    (45 s cap) and its gate shadow (15 s cap). The deadline must leave at
    least that much before the daemon abandons the request."""
    import voltrade_daemon
    deadline = system_config.BASE_CONFIG["SCAN_OPTIONS_STAGE_DEADLINE_SEC"]
    assert 0 < deadline <= voltrade_daemon.REQUEST_TIMEOUT_SEC - 60


def test_bot_engine_wires_the_deadline_from_the_scan_start():
    import bot_engine
    src = inspect.getsource(bot_engine._scan_market_inner)
    call = re.search(r"options_trades = get_options_trades\((.*?)\n        \)", src, re.S)
    assert call, "bot_engine must call get_options_trades"
    assert "deadline_ts=" in call.group(1)
    assert "_scan_start + _opt_deadline_sec" in call.group(1), \
        "the deadline is measured from the scan's own start, not from the stage's"
    assert "SCAN_OPTIONS_STAGE_DEADLINE_SEC" in src
    # the stage's count reaches the timing file before the phase line persists it
    assert src.index('"options_stage_deadline"') < src.index('_timing_log("step6_trade_loop_and_options")')
