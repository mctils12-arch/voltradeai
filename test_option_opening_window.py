#!/usr/bin/env python3
"""
OPENING WINDOW (2026-11-05): no option exit before 09:45 ET.

research/options_bleed_2026-11-04.md: 53 lots closed in the session's first 5
minutes lost $5,083. Those that barely moved filled at a median 1.41x fair
value; the XLF put paid $1.61 against ~$0.49 fair on an up day. The fix holds
every single-leg close and every group close until 09:45 ET, reporting the
rule as deferred so it is judged again after the window. Market-closed
behaviour is unchanged.

Run: python3 -m pytest -q test_option_opening_window.py
"""
import os
import sys
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import options_manager as om

ET = ZoneInfo("America/New_York")


def _et(h, m):
    return datetime(2026, 11, 5, h, m, tzinfo=ET)


def test_window_is_0930_to_0945_et():
    assert om._in_opening_window(_et(9, 29)) is False
    assert om._in_opening_window(_et(9, 30)) is True
    assert om._in_opening_window(_et(9, 44)) is True
    assert om._in_opening_window(_et(9, 45)) is False
    assert om._in_opening_window(_et(15, 0)) is False
    assert om.OPENING_WINDOW_END_ET == (9, 45)


def test_the_window_reads_the_brokers_clock_not_the_machines():
    """No timestamp (every test harness's /v2/clock, or a failed read) is outside the window."""
    saved = dict(om._CLOCK_CACHE)
    try:
        om._CLOCK_CACHE.update({"et": None, "et_at": 0.0})
        assert om._in_opening_window() is False
        stamp = om._parse_clock_timestamp("2026-11-05T09:31:02.123456789-05:00")
        assert stamp is not None and (stamp.hour, stamp.minute) == (9, 31)
        om._CLOCK_CACHE.update({"et": stamp, "et_at": __import__("time").time()})
        assert om._in_opening_window() is True
        late = om._parse_clock_timestamp("2026-11-05T09:46:00-05:00")
        om._CLOCK_CACHE.update({"et": late, "et_at": __import__("time").time()})
        assert om._in_opening_window() is False
        assert om._parse_clock_timestamp("garbage") is None and om._parse_clock_timestamp(None) is None
        utc = om._parse_clock_timestamp("2026-11-05T14:35:00Z")            # 09:35 ET, stamped in UTC
        om._CLOCK_CACHE.update({"et": utc, "et_at": __import__("time").time()})
        assert om._in_opening_window() is True, "a UTC stamp is converted to New York time first"
    finally:
        om._CLOCK_CACHE.clear()
        om._CLOCK_CACHE.update(saved)


def test_a_broken_clock_never_withholds_a_close():
    class Broken:
        @property
        def hour(self):
            raise RuntimeError("no clock")
    assert om._in_opening_window(Broken()) is False


def test_single_leg_close_is_deferred_in_the_window():
    actions = []
    with patch.object(om, "_market_is_open", return_value=True), \
         patch.object(om, "_in_opening_window", return_value=True):
        held = om._defer_single_leg_close(actions, "GOOGL", "GOOGL261127P00300000", "dte_close_bought",
                                          "21 DTE EXIT: 21 DTE", 21)
    assert held is True
    assert actions[0]["order"]["status"] == "deferred"
    assert actions[0]["order"]["detail"] == om.OPENING_WINDOW_DEFERRED_DETAIL
    assert "opening window" in actions[0]["reason"]


def test_single_leg_close_goes_after_the_window_and_market_closed_is_unchanged():
    actions = []
    with patch.object(om, "_market_is_open", return_value=True), \
         patch.object(om, "_in_opening_window", return_value=False):
        assert om._defer_single_leg_close(actions, "X", "X261127P00010000", "profit_target", "PT", 30) is False
    assert actions == [], "after 09:45 the rule submits exactly as before"
    with patch.object(om, "_market_is_open", return_value=False):
        assert om._defer_single_leg_close(actions, "X", "X261127P00010000", "profit_target", "PT", 30) is True
    assert actions[0]["order"]["detail"] == om.SINGLE_LEG_DEFERRED_DETAIL


def test_group_close_is_requested_but_not_sent_in_the_window():
    legs = [{"symbol": "IREN261204C00024000"}, {"symbol": "IREN261204P00020000"}]
    state = {}
    with patch.object(om, "_market_is_open", return_value=True), \
         patch.object(om, "_in_opening_window", return_value=True), \
         patch.object(om, "_close_strategy_mleg", side_effect=AssertionError("must not send in the window")):
        res = om._request_group_close(legs, "IREN", state, "strategy_ic_max_loss_exit", "IC MAX-LOSS EXIT")
    assert res["status"] == "deferred" and "09:45" in res["detail"]
    assert all(state[o["symbol"]].get("close_requested_at") for o in legs), "the request is stamped, so it is pursued after 09:45"
    with patch.object(om, "_market_is_open", return_value=True), \
         patch.object(om, "_in_opening_window", return_value=False), \
         patch.object(om, "_close_strategy_mleg", return_value={"status": "submitted"}) as sent:
        res = om._request_group_close(legs, "IREN", state, "strategy_ic_max_loss_exit", "IC MAX-LOSS EXIT")
    assert sent.called and res["status"] == "submitted"


def test_every_single_leg_exit_rule_goes_through_the_deferral():
    src = open(os.path.join(_ROOT, "options_manager.py")).read()
    body = src[src.index("def manage_options_positions("):]
    for rule in ("dte_critical", "assignment_close", "profit_target", "loss_limit", "bought_loss_limit", "gamma_risk"):
        assert f'"{rule}"' in body
    # every _submit_close_order in the single-leg loop sits after a _defer_single_leg_close for its rule
    assert body.count("_defer_single_leg_close(") >= 7
