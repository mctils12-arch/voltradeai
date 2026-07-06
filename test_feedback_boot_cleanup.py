"""
REPAIR 2026-07-06 (R12-D4) — regression battery for the boot-time
trade_feedback cleanup, covering all three defects of the old inline filter
(seed wipe-out, lexicographic version compare, fossil retention).

Run: python3 -m pytest test_feedback_boot_cleanup.py -q
"""
import unittest

from feedback_boot_cleanup import clean_feedback, keep_record, parse_version


def _fossil(**over):
    """The exact April 2026 fossil shape (pre-Bug-25b trackClosedTrades):
    code_version 1.0.33, pnl_pct 0, no outcome, entry_features null."""
    r = {
        "ticker": "AAPL", "side": "buy", "pnl_pct": 0, "holding_days": 0,
        "strategy": "auto", "score": 0, "rules_score": None, "ml_score": None,
        "blended_score": 0, "won": 0, "instrument": "stock",
        "entry_features": None, "exit_context": None,
        "timestamp": "2026-04-18T14:00:00Z", "code_version": "1.0.33",
    }
    r.update(over)
    return r


def _seed(**over):
    r = {
        "ticker": "MSFT", "side": "buy", "fill_price": 100.0, "qty": 1,
        "outcome": "win", "pnl_pct": 2.5, "_seed": True,
        "_seed_source": "backtest_10yr_results.json",
    }
    r.update(over)
    return r


def _live_entry(**over):
    r = {
        "ticker": "NVDA", "side": "buy", "qty": 5.0, "expected_price": 200.0,
        "fill_price": 200.3, "slippage_pct": 0.15, "outcome": None,
        "pnl_pct": None, "code_version": "1.0.34",
    }
    r.update(over)
    return r


class TestParseVersion(unittest.TestCase):
    def test_numeric_not_lexicographic(self):
        """THE latent wipe: '1.0.153' >= '1.0.34' must hold numerically
        (it is FALSE as strings — the old filter would delete it)."""
        self.assertGreater(parse_version("1.0.153"), parse_version("1.0.34"))
        self.assertLess("1.0.153", "1.0.34")  # documents the string trap

    def test_garbage_fails_the_floor(self):
        for bad in (None, "", "abc", "1.x.3", 42.5):
            self.assertEqual(parse_version(bad), (0,), bad)


class TestKeepRecord(unittest.TestCase):
    def test_seeds_survive_every_boot(self):
        """D4: seeds carry no code_version by design — the old filter
        deleted every Kelly-prior seed on every deploy."""
        self.assertTrue(keep_record(_seed()))

    def test_april_fossils_finally_removed(self):
        """The 500 records that blocked reseeding and were matchable by
        _find_entry_record on ticker collision."""
        self.assertFalse(keep_record(_fossil()))

    def test_trusted_live_records_kept(self):
        self.assertTrue(keep_record(_live_entry()))
        self.assertTrue(keep_record(_live_entry(code_version="1.0.153")))

    def test_empty_ticker_and_non_dict_dropped(self):
        self.assertFalse(keep_record(_live_entry(ticker="  ")))
        self.assertFalse(keep_record("not-a-dict"))
        self.assertFalse(keep_record(None))


class TestCleanFeedback(unittest.TestCase):
    def test_prod_shaped_mix(self):
        """The exact prod state: 500 fossils + what the file SHOULD hold."""
        records = [_fossil() for _ in range(500)] + [_seed(), _live_entry()]
        valid, removed = clean_feedback(records)
        self.assertEqual(removed, 500)
        self.assertEqual(len(valid), 2)
        self.assertTrue(any(r.get("_seed") for r in valid))
        self.assertTrue(any(r.get("code_version") == "1.0.34" for r in valid))

    def test_post_cleanup_count_unblocks_daemon_reseed(self):
        """The daemon autoseeds only when the file holds <100 records —
        fossil removal must take the count below that threshold."""
        valid, _ = clean_feedback([_fossil() for _ in range(500)])
        self.assertLess(len(valid), 100)

    def test_corrupt_container_degrades_to_empty(self):
        self.assertEqual(clean_feedback("not-a-list"), ([], 0))


if __name__ == "__main__":
    unittest.main()
