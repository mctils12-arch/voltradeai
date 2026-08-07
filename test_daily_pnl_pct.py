"""Daily-loss kill-switch INPUT repair (2026-08-07, full-code-review
finding wf_2302cba6-006, adversarially verified before fixing).

The defect: the tier-engine's only call to check_kill_switches() derived
daily_pnl_pct via `acct.get("daily_pnl_pct", 0)` — but Alpaca's /v2/account
response never contains that key (verified against server/bot.ts's own
`acct.equity - acct.last_equity` pattern, the repo's established way to
compute daily P&L from the same endpoint). The .get() default always won,
so daily_pnl_pct was hardcoded 0.0 on every cycle — risk_kill_switch.py's
DAILY_LOSS_LIMIT = -0.03 gate could never fire from a real daily loss.

These tests pin (1) the pure daily-P&L computation and (2) that the tier
engine's kill-switch call site actually uses it instead of the dead key —
a source-level ratchet in the repo's established style (test_kill_switch_
enforcement.py precedent), so a refactor cannot silently reintroduce the
fabricated-zero input.
"""
import re
import unittest

from bot_engine import compute_daily_pnl_pct


class TestComputeDailyPnlPct(unittest.TestCase):
    def test_up_day_is_positive(self):
        acct = {"equity": "103000.00", "last_equity": "100000.00"}
        self.assertAlmostEqual(compute_daily_pnl_pct(acct), 0.03)

    def test_down_day_is_negative(self):
        acct = {"equity": "97000.00", "last_equity": "100000.00"}
        self.assertAlmostEqual(compute_daily_pnl_pct(acct), -0.03)

    def test_flat_day_is_zero(self):
        acct = {"equity": "100000.00", "last_equity": "100000.00"}
        self.assertEqual(compute_daily_pnl_pct(acct), 0.0)

    def test_missing_last_equity_never_crashes_and_returns_zero(self):
        self.assertEqual(compute_daily_pnl_pct({"equity": "100000.00"}), 0.0)

    def test_zero_last_equity_never_divides_by_zero(self):
        acct = {"equity": "100000.00", "last_equity": "0"}
        self.assertEqual(compute_daily_pnl_pct(acct), 0.0)

    def test_empty_account_dict_returns_zero_not_a_crash(self):
        self.assertEqual(compute_daily_pnl_pct({}), 0.0)

    def test_never_reads_the_nonexistent_daily_pnl_pct_key(self):
        # Alpaca's real /v2/account payload has no such key; a stray
        # implementation that still reads it would silently ignore
        # equity/last_equity and this pins the payload never needs it.
        acct = {"equity": "110000.00", "last_equity": "100000.00", "daily_pnl_pct": "999"}
        self.assertAlmostEqual(compute_daily_pnl_pct(acct), 0.10)


class TestTierEngineCallSiteIsWired(unittest.TestCase):
    """Source ratchet: the tier-engine's check_kill_switches() call must
    derive daily_pnl_pct from compute_daily_pnl_pct(acct), not from the
    dead `acct.get("daily_pnl_pct", ...)` read."""

    def setUp(self):
        with open("bot_engine.py", encoding="utf-8") as f:
            self.src = f.read()

    def test_call_site_uses_compute_daily_pnl_pct(self):
        m = re.search(r"daily_pnl\s*=\s*compute_daily_pnl_pct\(acct\)", self.src)
        self.assertIsNotNone(m, "the tier-engine kill-switch input must be computed by compute_daily_pnl_pct(acct)")

    def test_dead_daily_pnl_pct_key_read_is_gone(self):
        self.assertNotIn(
            'acct.get("daily_pnl_pct"', self.src,
            "acct.get(\"daily_pnl_pct\", ...) always hits the default (Alpaca never sends this key) — must not return",
        )


if __name__ == "__main__":
    unittest.main()
