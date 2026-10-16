"""
REPAIR 2026-10-16 (bot cockpit audit S5 gate B13 / §8 V18): the sizing
engine's refusal (Kelly <= 0 for the bucket, halted asset, qty 0) was
discarded by the stock loop. The scan result now carries `sizing_blocks`
and the shadow record moves to `rejected_sizing`.
"""
import json
import os
import tempfile
import unittest

import shadow_portfolio as sp
from bot_engine import _record_sizing_block, _infer_bucket_for_block


class TestRecordSizingBlock(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = (sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH)
        sp.SHADOW_LOG_PATH = os.path.join(self._tmp.name, "shadow.json")
        sp.SHADOW_LOCK_PATH = sp.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH = self._orig
        self._tmp.cleanup()

    def test_block_is_recorded_and_shadow_bucket_moves(self):
        sp.log_candidate("AMD", {"momentum_1m": 1.0}, 74.3, "taken", decision_reason="score 74.3 >= MIN_SCORE 63")
        blocks = []
        _record_sizing_block(blocks, "amd", 74.3,
                             {"blocked": True, "block_reason": "Negative expectancy for strategy 'stocks' (WR=28%); refusing to size"},
                             strategy="stocks")
        self.assertEqual(blocks, [{"ticker": "AMD", "strategy": "stocks", "score": 74.3,
                                   "reason": "Negative expectancy for strategy 'stocks' (WR=28%); refusing to size"}])
        with open(sp.SHADOW_LOG_PATH) as f:
            rec = json.load(f)[-1]
        self.assertEqual(rec["decision"], "rejected_sizing")
        self.assertTrue(rec["decision_reason"].startswith("sizing: Negative expectancy"))

    def test_never_raises(self):
        blocks = []
        _record_sizing_block(blocks, None, "x", None)
        self.assertEqual(blocks[0]["reason"], "blocked")
        self.assertIsNone(blocks[0]["score"])

    def test_bucket_inference(self):
        self.assertEqual(_infer_bucket_for_block({"trade_type": "options"}), "csp_options")
        self.assertEqual(_infer_bucket_for_block({"trade_type": "etf"}), "etf")
        self.assertEqual(_infer_bucket_for_block({"trade_type": "stock"}), "stocks")
        self.assertEqual(_infer_bucket_for_block(None), "stocks")


class TestScanWiring(unittest.TestCase):
    def test_scan_returns_sizing_blocks_and_the_loop_records_them(self):
        with open("bot_engine.py", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("_sizing_blocks = []", src)
        self.assertIn("_record_sizing_block(_sizing_blocks, ticker, final_score, sizing,", src)
        self.assertEqual(src.count('"sizing_blocks":'), 2, "both scan return dicts carry sizing_blocks")


if __name__ == "__main__":
    unittest.main()
