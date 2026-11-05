#!/usr/bin/env python3
"""
[REPAIR] 2026-11-05: the QQQ-put convexity overlay honours CONVEXITY_OVERLAY.enabled.

It has been False since 2026-04-17 (the 10-year backtest bled $307K-$508K a year
in premium) but bot_engine._run_convexity_overlay never read it, so it ran every
scan: broker reads plus put orders posted with side "buy_to_open"/"sell_to_close",
which Alpaca's order API does not accept. Off now means off.

Run: python3 -m pytest -q test_convexity_overlay_gate.py
"""
import os
import sys
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bot_engine
import system_config


def _no_io(*_a, **_k):
    raise AssertionError("the overlay touched the broker while switched off")


def test_off_in_config_means_no_reads_and_no_orders():
    assert system_config.BASE_CONFIG["CONVEXITY_OVERLAY"]["enabled"] is False, "the shipped setting"
    with patch("requests.get", side_effect=_no_io), patch("requests.post", side_effect=_no_io), \
         patch.object(bot_engine, "_close_sqqq_position", side_effect=_no_io):
        res = bot_engine._run_convexity_overlay({"vxx_ratio": 1.0, "spy_vs_ma50": 1.0})
    assert res["status"] == "off" and res["actions"] == []
    assert "enabled is False" in res["detail"]


def test_a_missing_switch_is_off():
    with patch.dict(system_config.BASE_CONFIG, {"CONVEXITY_OVERLAY": {}}), \
         patch("requests.get", side_effect=_no_io), patch("requests.post", side_effect=_no_io):
        assert bot_engine._run_convexity_overlay({})["status"] == "off"


def test_on_in_config_runs_as_before():
    calls = []
    with patch.dict(system_config.BASE_CONFIG["CONVEXITY_OVERLAY"], {"enabled": True}), \
         patch.object(bot_engine, "_close_sqqq_position", side_effect=lambda *a, **k: calls.append("sqqq") or []), \
         patch("requests.get", side_effect=RuntimeError("stop here")):
        res = bot_engine._run_convexity_overlay({"vxx_ratio": 1.0, "spy_vs_ma50": 1.0})
    assert calls == ["sqqq"], "switched on, the overlay runs its steps again"
    assert res["status"] != "off"
