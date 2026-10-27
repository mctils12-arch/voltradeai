"""
[REPAIR] 2026-10-27 — audit §6.8 N3: an RSI-overbought "SELL" was executed
as a BUY.

deep_score labels score > 65 with RSI > 70 (and VRP <= 5) as side="sell",
action_label="SELL", trade_type="stock". _scan_market_inner then rewrote
every stock sell/short to side="buy" ("stock shorting disabled"), so the
overbought exit/avoid signal opened a LONG. There is no live, gated short
path (stock shorting is disabled, backtest -$419K), so the fix drops the
candidate from entries and relabels its shadow record
"rejected_bearish_stock_signal" so the rule's prevention-P&L is measurable.

The scan loop needs Alpaca, so the wiring is pinned on the source (the
pattern test_shadow_scan_gate_relabel established); the decision itself is
a pure helper tested directly.
"""
import inspect
import re
import unittest

import bot_engine

SRC = inspect.getsource(bot_engine._scan_market_inner)


class TestStockShortEntryRejection(unittest.TestCase):
    def test_overbought_stock_sell_is_rejected_not_flipped(self):
        reason = bot_engine._stock_short_entry_rejection("sell", "stock", "SELL")
        self.assertIsNotNone(reason)
        self.assertIn("SELL", reason)
        self.assertIn("not traded", reason)

    def test_stock_short_and_missing_trade_type_are_rejected(self):
        self.assertIsNotNone(bot_engine._stock_short_entry_rejection("short", "stock"))
        # scan dicts default trade_type to "stock" when absent
        self.assertIsNotNone(bot_engine._stock_short_entry_rejection("sell", None))

    def test_options_sell_keeps_its_path(self):
        # SELL OPTIONS (VRP > 5) is handed to the instrument selector with
        # side="sell" — that path is unchanged by this fix.
        self.assertIsNone(bot_engine._stock_short_entry_rejection("sell", "options", "SELL OPTIONS"))

    def test_buy_passes(self):
        self.assertIsNone(bot_engine._stock_short_entry_rejection("buy", "stock", "BUY"))


class TestScanWiring(unittest.TestCase):
    def test_no_sell_to_buy_rewrite_remains(self):
        # The old body: `if side in ("sell", "short") and ...: side = "buy"`.
        self.assertIsNone(
            re.search(r'if side in \("sell", "short"\)[^\n]*:\s*\n\s*side = "buy"', SRC),
            "a stock sell/short must never be rewritten into a BUY entry",
        )
        self.assertNotRegex(SRC, r'\n\s+side = "buy"\n',
                            "no code path in the scan loop may flip a candidate's side to buy")

    def test_helper_gates_then_drops_before_sizing_and_instrument(self):
        call = SRC.find("_stock_short_entry_rejection(")
        self.assertNotEqual(call, -1, "the scan must consult the bearish-stock gate")
        window = SRC[call:call + 400]
        self.assertIn('_note_gate(ticker, "rejected_bearish_stock_signal"', window)
        self.assertRegex(window, r"\n\s+continue\b")
        self.assertLess(call, SRC.find("select_instrument("),
                        "the gate must run before the instrument decision")
        self.assertLess(call, SRC.find("calculate_position("),
                        "the gate must run before sizing")
        self.assertLess(call, SRC.find("trades.append({"))

    def test_side_read_from_candidate_feeds_the_gate(self):
        side_read = SRC.find('side = stock.get("side", "buy")')
        self.assertNotEqual(side_read, -1)
        self.assertLess(side_read, SRC.find("_stock_short_entry_rejection("))


class TestDeepScoreStillLabelsOverbought(unittest.TestCase):
    """The label itself is unchanged — only its execution is fixed."""

    def test_overbought_branch_still_emits_sell(self):
        ds = inspect.getsource(bot_engine.deep_score)
        idx = ds.find("elif rsi_overbought:")
        self.assertNotEqual(idx, -1)
        self.assertIn('side = "sell"', ds[idx:idx + 120])
        self.assertIn('action_label = "SELL"', ds[idx:idx + 120])


if __name__ == "__main__":
    unittest.main()
