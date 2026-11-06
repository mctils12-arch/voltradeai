#!/usr/bin/env python3
"""
[RULE-REVIEW] 2026-11-06: composite v2's legacy-vs-v2 comparison (rollback
trigger 2) reads one definition of the legacy arm only.

The v2 training-honesty deploy replaced the live model's in-sample ML view with
the honest model's near-constant one, so composite_legacy (logged on every
shadow record) changed definition at that deploy. signal_attribution's
composite_comparison now reads SCORE_COMPOSITE_V2["comparison_since"] and pairs
only records from that ET entry day on, reporting how many paired records it
left out.

Run: python3 -m pytest -q test_composite_comparison_since.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import score_composite as sc
import signal_attribution as sa
from system_config import BASE_CONFIG


def _row(day: str, legacy, v2, y: float = 0.01) -> dict:
    bd = {}
    if legacy is not None:
        bd[sa.COMPOSITE_LEGACY_KEY] = legacy
    if v2 is not None:
        bd[sa.COMPOSITE_V2_KEY] = v2
    return {"rec": {"score_breakdown": bd}, "y": y, "day": day, "regime": "NEUTRAL", "ts": f"{day}T15:00:00Z"}


ROWS = [
    _row("2026-11-05", None, None),            # before v2 shipped: no composite keys
    _row("2026-11-06", 72.0, 60.0),            # v2's first day: the old ML term in legacy
    _row("2026-11-06", 55.0, 52.0),
    _row("2026-11-09", 61.0, 58.0),
    _row("2026-11-10", 64.0, 59.0, y=-0.02),
]


def test_records_before_comparison_since_are_counted_not_compared():
    out = sa.composite_comparison(ROWS, "2026-11-09")
    assert out["n"] == 2, out
    assert out["since"] == "2026-11-09"
    assert out["excluded_before_since"] == 2, "the two first-day paired records, not the keyless one"


def test_no_comparison_since_keeps_every_record():
    out = sa.composite_comparison(ROWS)
    assert out["n"] == 4
    assert "since" not in out and "excluded_before_since" not in out
    assert sa.composite_comparison(ROWS, None) == out
    assert sa.composite_comparison(ROWS, "junk") == out, "a malformed date never filters silently"


def test_live_config_scopes_the_comparison_and_the_served_score_ignores_it():
    cfg = BASE_CONFIG["SCORE_COMPOSITE_V2"]
    assert cfg["comparison_since"] == "2026-11-12"
    assert "comparison_since" not in sc.v2_config(), "v2_config copies only its own keys: the served score is unaffected"
    assert sc.v2_config()["enabled"] is True


def test_build_attribution_passes_the_configured_date():
    src = open(os.path.join(_ROOT, "signal_attribution.py"), encoding="utf-8").read()
    assert 'comp = composite_comparison(prim_rows, (cfg or {}).get("comparison_since"))' in src
