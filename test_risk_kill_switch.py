"""
Regression tests for risk_kill_switch.py's correlation-cap and free-BP-
buffer percentage math.

No test file existed for this module before 2026-07-30 despite it being
the file backing kill switches #3 (correlation cap) and #4 (margin
buffer) in the module's own docstring.

BUG (found + fixed 2026-07-30): both `_check_correlation` and the
free-BP check in `check_kill_switches` divided exposure by REMAINING
Alpaca buying_power, not TOTAL buying-power capacity (used + free).
Remaining buying_power shrinks toward zero as the account gets used, so
any deployed capital reads as an ever-larger fraction of what's left —
live audit log evidence: "Free BP below 10%: -867.9%" and "Correlation
cap breach: INDEX_ETF at 551.0% (max 60%)", both growing across a single
trading day. Back-calculated from those two numbers: ~$20.7k deployed
against ~$2.1k remaining buying_power (~$110k equity) — a normal ~19%
of-equity exposure level — produced a -868% reading under the broken
formula. Likely triggered for the first time today by the #27 Kelly-
bucket fix (same session) finally letting CSP sizing draw buying_power
down for real after 17+ days floored at 1%.

Fix: denominator is now bp_used + buying_power (a stable, always
non-negative total-capacity approximation) for both checks, matching
the equity-relative convention used everywhere else in the codebase
(bot_engine.py's total_deployed/current_pct/drift_pct). Thresholds
(CORRELATION_CAP, MIN_FREE_BP) are unchanged — this restores their
intended 0-100%-bounded meaning.
"""
import pytest

from risk_kill_switch import (
    _check_correlation, check_kill_switches, reset_kill_state,
    CORRELATION_CAP, MIN_FREE_BP,
)


# ── _check_correlation: pure function, denominator behavior ─────────────────

def test_check_correlation_true_concentration_still_breaches():
    """SPY (INDEX_ETF group) at 75% of TOTAL capacity is a real breach."""
    positions = [{"symbol": "SPY", "market_value": 15_000}]
    msg = _check_correlation(positions, 20_000)  # total capacity, not remaining BP
    assert msg is not None
    assert "INDEX_ETF" in msg
    assert "75.0%" in msg


def test_check_correlation_no_false_breach_against_total_capacity():
    """The exact bug shape: $12k in one group would breach against a small
    REMAINING-buying-power denominator (old code) but is a sane 54.5% of
    TOTAL capacity (used + free) — no real concentration problem."""
    positions = [{"symbol": "SPY", "market_value": 12_000}]

    # Old-broken-style small denominator (remaining BP only) DOES flag —
    # proves the function's basic breach mechanics still work.
    old_style_msg = _check_correlation(positions, 2_000)
    assert old_style_msg is not None
    assert "600.0%" in old_style_msg

    # Fixed usage: total capacity (bp_used + free) = 20_000 + 2_000.
    fixed_msg = _check_correlation(positions, 22_000)
    assert fixed_msg is None, (
        "12k exposure against 22k TOTAL capacity is 54.5%, under the 60% "
        f"cap ({CORRELATION_CAP*100:.0f}%) — must not false-positive"
    )


# ── check_kill_switches: integration, free-BP + correlation together ────────

def _clean_kwargs(**overrides):
    base = dict(
        equity=110_000.0, peak_equity=110_000.0, positions=[],
        daily_pnl_pct=0.0, vxx_ratio=1.0, buying_power=0.0,
    )
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _reset_state():
    reset_kill_state()
    yield
    reset_kill_state()


def test_free_bp_pct_bounded_matches_audit_log_shape():
    """Reproduces the live audit-log shape: ~$20.7k deployed (XLE — not in
    any CORRELATION_GROUPS bucket, isolates this from the corr-cap check),
    ~$2.14k remaining buying_power, ~$110k equity. Old formula: -866.7%.
    Fixed formula: 2140.81 / (20700 + 2140.81) = 9.37%."""
    result = check_kill_switches(**_clean_kwargs(
        positions=[{"symbol": "XLE", "market_value": 20_700.0}],
        buying_power=2_140.81,
    ))
    bp_warnings = [w for w in result["warnings"] if w.startswith("Free BP below")]
    assert len(bp_warnings) == 1, result["warnings"]
    msg = bp_warnings[0]
    assert "9.4%" in msg, msg
    # Ratchet: the pre-fix formula (1 - bp_used/buying_power) produced
    # -866.7% here — nowhere near a real percentage. The fixed value must
    # stay inside a sane bound.
    pct = float(msg.split(":")[-1].strip().rstrip("%"))
    assert -100.0 <= pct <= 100.0, f"free BP pct must be bounded, got {pct}"


def test_correlation_cap_not_falsely_breached_when_bp_used_is_healthy():
    """Same $110k-equity account, but the deployed capital sits partly in
    SPY (INDEX_ETF group, $11.8k) and partly in XLE (no correlation
    group, $9k): SPY is $11.8k of ~$22.9k TOTAL capacity = 51.4%, under
    the 60% cap. Old formula (denominator = remaining BP alone, $2140.81,
    ignoring the other position entirely) put SPY alone at 551% — a false
    breach identical to the live audit log."""
    result = check_kill_switches(**_clean_kwargs(
        positions=[
            {"symbol": "SPY", "market_value": 11_796.0},
            {"symbol": "XLE", "market_value": 9_000.0},
        ],
        buying_power=2_140.81,
    ))
    corr_warnings = [w for w in result["warnings"] if w.startswith("Correlation cap breach")]
    assert corr_warnings == [], (
        f"expected no correlation-cap warning (51.6% < {CORRELATION_CAP*100:.0f}% cap), "
        f"got {corr_warnings}"
    )


def test_correlation_cap_still_fires_on_genuine_overconcentration():
    """Sanity: the fixed denominator doesn't neuter the check — a group at
    70% of total capacity must still breach."""
    result = check_kill_switches(**_clean_kwargs(
        positions=[{"symbol": "SPY", "market_value": 7_000.0}],
        buying_power=3_000.0,  # total capacity = 10_000; 7000/10000 = 70%
    ))
    corr_warnings = [w for w in result["warnings"] if w.startswith("Correlation cap breach")]
    assert len(corr_warnings) == 1
    assert "70.0%" in corr_warnings[0]


def test_free_bp_warning_fires_at_zero_buying_power():
    """Previously-silent gap: the old `if buying_power > 0:` guard meant
    the free-BP warning could never fire once buying_power hit exactly 0
    or went negative — precisely the regime it exists to catch."""
    result = check_kill_switches(**_clean_kwargs(
        positions=[{"symbol": "XLE", "market_value": 5_000.0}],
        buying_power=0.0,
    ))
    bp_warnings = [w for w in result["warnings"] if w.startswith("Free BP below")]
    assert len(bp_warnings) == 1, "buying_power=0 with open positions must still warn"
    assert "0.0%" in bp_warnings[0]


def test_free_bp_pct_no_warning_when_healthy():
    """Control: plenty of free buying power relative to total capacity
    must not warn."""
    result = check_kill_switches(**_clean_kwargs(
        positions=[{"symbol": "XLE", "market_value": 5_000.0}],
        buying_power=50_000.0,  # 50000 / 55000 = 90.9% free
    ))
    bp_warnings = [w for w in result["warnings"] if w.startswith("Free BP below")]
    assert bp_warnings == []
