"""
[REPAIR] 2026-11-04 — Setup 3 condors enforce their documented earnings rule.

options_scanner._setup_high_iv_premium_sale (SETUP 3, "SELL IRON CONDOR")
documents "No earnings within 14 days (IV is high for a regular reason, not
just earnings)", but nothing enforced it: the function has no calendar, its
ML features hard-code days_to_earn=99.0, and scan_options()'s _check_ticker
called it for every high_iv/anchor/any candidate with no gate. Only Setup 7
(VRP+) checked the calendar (> 10 days). Live cost: the GOOGL Nov-27 iron
condor rode through GOOGL's report; the 2026-10-29 -> 11-04 shadow log had
14 of 19 proposed condor names within 0-8 days of earnings.

The condor's expiry is chosen AFTER this scan, by options_execution's
select_contract -> _select_condor (30-DTE entry floor, closest-to-30
preference, chain fetched to 50 days), and options_manager closes the group
at 21 DTE. The preferred expiry's planned hold (floor - 21 = 9 days) sits
inside the 14-day blackout, so the gate is the documented 14 days.

Pins — scan_options() end-to-end with every network call mocked, the
pattern of test_options_scanner_surface_score_dead_compute.py:
  1. a report inside the blackout -> no Setup-3 opportunity, one
     counterfactual shadow record (decision rejected_setup3_earnings);
  2. a report after the blackout and after the planned 21-DTE exit ->
     allowed, nothing logged;
  3. a failed calendar lookup -> Setup 3 FAILS CLOSED (no condor that
     scan), a WARNING line says why, rejected_setup3_earnings_lookup_failed
     records;
  4. Setup 7 unchanged: its own > 10-day check, and it runs on exactly the
     tickers it ran on before (Setup 3's function produced nothing) — so a
     refused Setup-3 condor is not re-emitted through Setup 7's identical
     call;
  5. one calendar per scan, through the status-returning fetch, with a
     window that reaches the blackout;
  6. counterfactual writes: once per ticker/decision/UTC day and bounded per
     scan (each shadow write re-dumps the whole ~45MB log inside the scan's
     300s daemon budget).
"""
import logging

import pytest

import options_scanner


def _condor(tkr, price, score=72.0):
    return {"ticker": tkr, "setup": "high_iv_premium_sale", "score": score, "price": price,
            "action_label": "SELL IRON CONDOR (IV rank 70/100 — premium selling)",
            "options_strategy": "iron_condor", "side": "sell", "reasoning": "fixture condor"}


def _always_condor(tkr, price, vxx_ratio):
    return _condor(tkr, price)


class _Harness:
    """scan_options() with every network edge mocked. Records accumulate across
    run() calls inside one test (the once-per-day bound spans scans)."""

    def __init__(self, monkeypatch, real_shadow=False):
        self.mp = monkeypatch
        self.shadow = []          # shadow_portfolio.log_candidate(**kwargs) calls
        self.surface_calls = []   # vol_surface.get_surface_score(tkr) calls (Setup 7 + the SPY block)
        self.fetches = []         # ("status"|"wrapper", days_ahead) earnings-calendar fetches
        # fresh once-per-day state (raising=False: absent on pre-fix code)
        monkeypatch.setattr(options_scanner, "_SETUP3_SHADOW_ONCE", {"day": None, "keys": set()}, raising=False)
        if not real_shadow:
            monkeypatch.setattr("shadow_portfolio.log_candidate", lambda **kw: self.shadow.append(kw))

    def run(self, candidates, r3, calendar, lookup_ok=True, surface=(0.10, 90)):
        mp = self.mp
        mp.setattr(options_scanner, "_is_regular_hours", lambda: True)
        mp.setattr(options_scanner, "_get_vxx_ratio", lambda: 1.0)
        mp.setattr(options_scanner, "_get_spy_vs_ma50", lambda: 1.0)
        mp.setattr(options_scanner, "_setup_vxx_panic_put_sale", lambda vxx, spy: None)
        # Setup 1 (earnings IV crush) reads the same calendar: no price -> inert, offline.
        mp.setattr(options_scanner, "_fetch_price", lambda tkr: None)

        def _status(days_ahead=10):
            self.fetches.append(("status", days_ahead))
            return dict(calendar), lookup_ok

        def _wrapper(days_ahead=10):  # its documented contract: a failed lookup reads as {}
            self.fetches.append(("wrapper", days_ahead))
            return dict(calendar) if lookup_ok else {}

        mp.setattr(options_scanner, "_fetch_earnings_calendar_with_status", _status)
        mp.setattr(options_scanner, "_fetch_earnings_calendar", _wrapper)
        mp.setattr(options_scanner, "_get_options_candidates", lambda: list(candidates))
        mp.setattr(options_scanner, "_setup_high_iv_premium_sale", r3)

        def _surface(tkr):
            self.surface_calls.append(tkr)
            return {"vrp": {"vrp_20d": surface[0]}, "surface_score": surface[1]}

        mp.setattr("vol_surface.get_surface_score", _surface)
        mp.setattr("vol_surface.get_vrp", lambda tkr: {"vrp_20d": 0.0})
        mp.setattr("vol_surface.surface_score_scan_budget_exceeded", lambda budget: False)
        return options_scanner.scan_options()


@pytest.fixture
def scan(monkeypatch):
    return _Harness(monkeypatch)


def _tickers(result):
    return [o["ticker"] for o in result["opportunities"]]


# ── 1. a report inside the blackout ──────────────────────────────────────────

def test_report_in_5_days_refuses_the_setup3_condor_and_logs_a_counterfactual(scan):
    result = scan.run([("GOOGL", 280.0, "anchor")], _always_condor, {"GOOGL": 5})

    assert _tickers(result) == [], "a Setup-3 condor was proposed 5 days before the report"
    assert len(scan.shadow) == 1
    rec = scan.shadow[0]
    assert rec["ticker"] == "GOOGL"
    assert rec["decision"] == "rejected_setup3_earnings"
    assert rec["score"] == 72.0 and rec["entry_price"] == 280.0
    assert (rec["side"], rec["trade_type"]) == ("sell", "options")
    assert rec["features"] == {}, "an options refusal must never enter load_shadow_data's ML blend"
    assert "5d" in rec["decision_reason"]
    # Setup 7 calls the SAME function; it must not get a second shot at a refused name
    assert "GOOGL" not in scan.surface_calls


def test_counterfactual_written_through_the_real_shadow_logger(monkeypatch, tmp_path):
    """No **kwargs stand-in: the real shadow_portfolio.log_candidate writes the
    record into a temp log, so a signature drift fails here, and the record
    has the shape the +5/+10/+20d backfill labels like any other."""
    import json
    import shadow_portfolio as sp
    monkeypatch.setattr(sp, "SHADOW_LOG_PATH", str(tmp_path / "shadow.json"))
    monkeypatch.setattr(sp, "SHADOW_LOCK_PATH", str(tmp_path / "shadow.json.lock"))
    h = _Harness(monkeypatch, real_shadow=True)

    result = h.run([("GOOGL", 280.0, "anchor")], _always_condor, {"GOOGL": 5})

    assert _tickers(result) == []
    with open(sp.SHADOW_LOG_PATH) as f:
        recs = json.load(f)
    assert len(recs) == 1
    r = recs[0]
    assert (r["ticker"], r["decision"], r["side"], r["trade_type"]) == (
        "GOOGL", "rejected_setup3_earnings", "sell", "options")
    assert (r["score"], r["entry_price"], r["features"]) == (72.0, 280.0, {})
    assert r["outcomes"] == {f"+{d}d": None for d in sp.FORWARD_HORIZONS_DAYS}


def test_blackout_boundary_is_inclusive_at_14_days(scan):
    names = {"D0": 0, "D14": 14, "D15": 15, "NOREPORT": None}
    cal = {k: v for k, v in names.items() if v is not None}
    result = scan.run([(n, 100.0, "high_iv") for n in names], _always_condor, cal)

    assert sorted(_tickers(result)) == ["D15", "NOREPORT"]
    assert sorted((r["ticker"], r["decision"]) for r in scan.shadow) == [
        ("D0", "rejected_setup3_earnings"), ("D14", "rejected_setup3_earnings")]


# ── 2. a report after the blackout and after the planned 21-DTE exit ─────────

def test_report_after_the_blackout_and_the_planned_21dte_exit_is_allowed(scan):
    from options_execution import MULTI_LEG_ENTRY_MIN_DTE
    from options_manager import DTE_EXIT_THRESHOLD
    # the execution's preferred condor expiry is its 30-DTE floor; the manager
    # closes the group at 21 DTE -> planned exit ~9 days out, well before day 30
    assert MULTI_LEG_ENTRY_MIN_DTE - DTE_EXIT_THRESHOLD < 30

    result = scan.run([("GOOGL", 280.0, "anchor")], _always_condor, {"GOOGL": 30})

    assert _tickers(result) == ["GOOGL"]
    assert scan.shadow == []


# ── 3. a failed calendar lookup ───────────────────────────────────────────────

def test_failed_calendar_lookup_fails_closed_and_says_why(scan, caplog):
    caplog.set_level(logging.WARNING, logger="options_scanner")
    result = scan.run([("GOOGL", 280.0, "anchor"), ("MSFT", 500.0, "high_iv")],
                      _always_condor, {}, lookup_ok=False)

    assert _tickers(result) == [], "no Setup-3 condor may be proposed while the report date is unknown"
    assert sorted((r["ticker"], r["decision"]) for r in scan.shadow) == [
        ("GOOGL", "rejected_setup3_earnings_lookup_failed"),
        ("MSFT", "rejected_setup3_earnings_lookup_failed")]
    assert "GOOGL" not in scan.surface_calls and "MSFT" not in scan.surface_calls
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("FAIL-CLOSED" in m and "calendar lookup failed" in m for m in warnings), warnings


# ── 4. Setup 7 unchanged ──────────────────────────────────────────────────────

def test_setup7_keeps_its_own_10_day_check(scan):
    """Setup 3's call finds nothing; Setup 7's identical call (behind its VRP/
    surface gate) returns a condor — the stateful mock of the dead-compute
    suite. Setup 7 still admits 12 days (> 10, its own threshold — the 14-day
    Setup-3 blackout is not applied to it) and still refuses 8 days."""
    calls = {}

    def r3(tkr, price, vxx_ratio):
        calls[tkr] = calls.get(tkr, 0) + 1
        return None if calls[tkr] == 1 else _condor(tkr, price, score=60.0)

    result = scan.run([("AAA", 50.0, "high_iv"), ("BBB", 60.0, "high_iv")], r3, {"AAA": 12, "BBB": 8})

    opps = {o["ticker"]: o for o in result["opportunities"]}
    assert set(opps) == {"AAA"}
    assert opps["AAA"].get("vrp_enhanced") is True
    assert scan.surface_calls.count("AAA") == 1 and scan.surface_calls.count("BBB") == 1
    assert scan.shadow == [], "Setup-3 counterfactuals are written for Setup-3 refusals only"


def test_setup7_does_not_re_emit_a_condor_setup3_refused(scan):
    """Pre-fix, Setup 7 never ran on a ticker whose Setup-3 call produced a
    candidate; it still does not when that candidate is refused for earnings —
    otherwise its identical call would re-emit the same condor under its
    looser 10-day check (12 days here)."""
    result = scan.run([("MSFT", 500.0, "high_iv")], _always_condor, {"MSFT": 12})

    assert _tickers(result) == []
    assert "MSFT" not in scan.surface_calls
    assert [(r["ticker"], r["decision"]) for r in scan.shadow] == [("MSFT", "rejected_setup3_earnings")]


# ── 5. one calendar per scan ──────────────────────────────────────────────────

def test_calendar_fetched_once_per_scan_through_the_status_api(scan):
    scan.run([("AAA", 50.0, "high_iv"), ("BBB", 60.0, "anchor"), ("CCC", 70.0, "high_iv")],
             _always_condor, {})

    assert len(scan.fetches) == 1, f"one calendar per scan, never per ticker: {scan.fetches}"
    kind, days_ahead = scan.fetches[0]
    assert kind == "status", "the scan must see lookup_ok, not the wrapper that reads a failure as {}"
    assert days_ahead >= options_scanner.SETUP3_EARNINGS_BLACKOUT_DAYS


def test_blackout_is_the_documented_14_days_and_covers_the_preferred_planned_hold():
    from options_execution import MULTI_LEG_ENTRY_MIN_DTE
    from options_manager import DTE_EXIT_THRESHOLD
    assert options_scanner.SETUP3_EARNINGS_BLACKOUT_DAYS == 14
    # If the condor entry floor or the 21-DTE exit is retuned so the preferred
    # expiry's hold outruns the blackout, this fails: widen the blackout too.
    assert options_scanner.SETUP3_EARNINGS_BLACKOUT_DAYS >= MULTI_LEG_ENTRY_MIN_DTE - DTE_EXIT_THRESHOLD


def test_refusal_rule_unit():
    f = options_scanner._setup3_earnings_refusal
    cal = {"D0": 0, "D14": 14, "D15": 15}
    assert f("D0", cal, True)[0] == "rejected_setup3_earnings"
    assert f("D14", cal, True)[0] == "rejected_setup3_earnings"
    assert f("D15", cal, True) is None
    assert f("NOREPORT", cal, True) is None
    assert f("D15", cal, False)[0] == "rejected_setup3_earnings_lookup_failed"
    assert f("NOREPORT", {}, False)[0] == "rejected_setup3_earnings_lookup_failed"


# ── 6. counterfactual write bounds ────────────────────────────────────────────

def test_counterfactual_written_once_per_ticker_per_day(scan):
    for _ in range(3):
        result = scan.run([("GOOGL", 280.0, "anchor")], _always_condor, {"GOOGL": 5})
        assert _tickers(result) == [], "the refusal itself holds on every scan"
    assert [r["ticker"] for r in scan.shadow] == ["GOOGL"]


def test_counterfactual_writes_are_bounded_per_scan_and_drain_over_later_scans(scan):
    cap = options_scanner.SETUP3_EARNINGS_SHADOW_MAX_PER_SCAN
    names = ["N%02d" % i for i in range(cap + 2)]
    cands = [(n, 50.0, "high_iv") for n in names]
    cal = {n: 3 for n in names}

    scan.run(cands, _always_condor, cal)
    assert len(scan.shadow) == cap
    scan.run(cands, _always_condor, cal)
    assert len(scan.shadow) == len(names)
    scan.run(cands, _always_condor, cal)
    assert len(scan.shadow) == len(names)
    assert sorted(r["ticker"] for r in scan.shadow) == names


def test_once_per_day_bound_resets_on_a_new_utc_day(monkeypatch):
    logged = []
    monkeypatch.setattr("shadow_portfolio.log_candidate", lambda **kw: logged.append(kw))
    monkeypatch.setattr(options_scanner, "_SETUP3_SHADOW_ONCE", {"day": None, "keys": set()})
    ref = [{"ticker": "GOOGL", "decision": "rejected_setup3_earnings", "reason": "earnings in 5d",
            "score": 72.0, "price": 280.0, "action_label": "SELL IRON CONDOR"}]
    log = options_scanner._log_setup3_earnings_refusals
    assert log(ref, 1.0, "NEUTRAL", today="2026-11-04") == 1
    assert log(ref, 1.0, "NEUTRAL", today="2026-11-04") == 0
    assert log(ref, 1.0, "NEUTRAL", today="2026-11-05") == 1
    assert len(logged) == 2
