"""KNOWN BROKEN #48 step 3 (2026-11-06, [RULE-REVIEW]): one stock exit model.

The live R-ladder (server/bot.ts checkPositionOnTick, mirrored by
live_exit_policy.py) is the stock exit policy of record. The configured exit
block in system_config.py (ATR_STOP_MULTIPLIER / MIN_STOP_PCT / MAX_STOP_PCT /
MIN_TP_PCT / MAX_TP_PCT / ATR_TP_MULTIPLIER / TIME_STOP_DAYS and their
get_adaptive_params regime overrides) had no reader in Python or TypeScript,
never ran live, and was deleted. These tests keep a second, unread exit model
from coming back silently.
"""
import os
import re

import live_exit_policy as lep
from system_config import BASE_CONFIG, get_adaptive_params, get_market_regime

DELETED_EXIT_KEYS = (
    "ATR_STOP_MULTIPLIER", "MIN_STOP_PCT", "MAX_STOP_PCT",
    "MIN_TP_PCT", "MAX_TP_PCT", "ATR_TP_MULTIPLIER", "TIME_STOP_DAYS",
)

# Inputs that hit each regime per regime_util's thresholds (the same cases
# test_audit_critical's regime-consistency test uses).
REGIME_INPUTS = {
    "PANIC":   dict(vxx_ratio=1.50, spy_vs_ma50=1.00),
    "BEAR":    dict(vxx_ratio=1.20, spy_vs_ma50=1.00),
    "CAUTION": dict(vxx_ratio=1.08, spy_vs_ma50=1.00),
    "NEUTRAL": dict(vxx_ratio=1.00, spy_vs_ma50=0.99),
    "BULL":    dict(vxx_ratio=0.85, spy_vs_ma50=1.02),
}

REPO = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"node_modules", "dist", ".git", "research", "docs", "__pycache__", "attached_assets"}


def test_configured_exit_block_is_deleted():
    present = [k for k in DELETED_EXIT_KEYS if k in BASE_CONFIG]
    assert present == [], f"unread configured exit keys are back in BASE_CONFIG: {present}"


def test_no_regime_reintroduces_an_exit_key():
    for regime, inputs in REGIME_INPUTS.items():
        assert get_market_regime(**inputs) == regime, f"inputs no longer map to {regime}"
        for tod in ("open", "regular", "power_hour", "afterhours"):
            params = get_adaptive_params(time_of_day=tod, **inputs)
            assert params["regime"] == regime
            present = [k for k in DELETED_EXIT_KEYS if k in params]
            assert present == [], f"{regime}/{tod}: get_adaptive_params sets {present} again"


def _source_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.startswith("test_") or name.endswith((".test.ts", ".test.tsx")):
                continue
            if name.endswith((".py", ".ts", ".tsx", ".mjs")):
                yield os.path.join(root, name)


def test_no_source_reads_a_deleted_exit_key():
    """A quoted key is a config read (BASE_CONFIG["X"], p.get("X"), cfg['X'],
    a bot.ts inline-Python one-liner). backtest_v2's legacy arm keeps its own
    module constants (unquoted identifiers), which this does not match."""
    pattern = re.compile(r"""["'](%s)["']""" % "|".join(DELETED_EXIT_KEYS))
    hits = []
    for path in _source_files():
        # errors="replace": a stray non-UTF-8 byte cannot hide an ASCII key
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if pattern.search(line):
                    hits.append(f"{os.path.relpath(path, REPO)}:{i}: {line.strip()[:120]}")
    assert hits == [], "a deleted exit key is read again (KB #48 step 3):\n" + "\n".join(hits)


def test_the_policy_of_record_is_the_live_ladder():
    """The numbers of record, as live_exit_policy mirrors server/bot.ts. A
    deliberate RULE-REVIEW change to the live ladder updates this pin."""
    assert (lep.RISK_ATR_MULT, lep.P1_STOP_ATR_MULT, lep.P1_STOP_MIN_PCT, lep.P1_STOP_MAX_PCT) == (2.0, 2.0, 1.5, 8.0)
    assert lep.SCALE1_R == 1.0
    assert lep.SCALE2_R == {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}
    assert lep.TRAIL4_ATR_MULT == {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}
    assert (lep.P2_TRAIL_MIN_PCT, lep.P2_TRAIL_ATR_MULT, lep.P3_TRAIL_MIN_PCT, lep.P3_TRAIL_ATR_MULT) == (1.5, 1.5, 1.0, 1.0)
    assert (lep.TP_MIN_PCT, lep.TP_ATR_MULT, lep.TP_MAX_PCT) == (4.0, 3.0, 15.0)
    assert (lep.TIME_STOP_DAYS, lep.TIME_STOP_BAND_PCT) == (7, 2.0)
    assert abs(lep.TRANCHE - 1.0 / 3.0) < 1e-12
