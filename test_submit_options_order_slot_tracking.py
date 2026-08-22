"""
STALE-ORDER-SWEEP FIX 2026-08-22 (open_questions.md KNOWN BROKEN, filed this
session): server/bot.ts's `openOrders` tracked-order array — the input that
feeds `sweepStaleOrders()` (tier1Reflex, every ~45s, cancels DAY limit orders
unfilled for STALE_ORDER_MINUTES=12+ minutes) — was never populated anywhere
in bot.ts for ANY order type, so the sweeper was fully dead code. Live
evidence: two CSP DAY limit orders (NU, XLP) sat "accepted" for 2+ hours,
counted by countOpenOptionsOpeningOrders() into a real 6/6 OPTIONS-SLOT-FULL
block on TLT/IBIT/CSX all afternoon, with no mechanism able to free the
squatted slots early.

This test pins the options-order half of the fix: submit_options_order()'s
single-leg success path must return qty/limit_price/side alongside order_id
so the tier dispatcher (server/bot.ts) can build a TrackedOrder and register
it with the sweeper. Mirrors the existing "detail" field's exact numbers
(qty, limit_price) already embedded in the human-readable string — this
just exposes them as machine-readable fields too, a pure additive return-value
change (never touching the raw HTTP POST body/headers/retry behavior that
FROZEN PATHS reserves for submit_options_order).
"""
import unittest
from unittest.mock import patch, MagicMock

import options_execution


class TestSubmitOptionsOrderReturnsTrackingFields(unittest.TestCase):
    def _mock_response(self, status_code=200, order_id="test-order-id-123"):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"id": order_id}
        resp.headers = {"content-type": "application/json"}
        resp.text = ""
        return resp

    def test_successful_sell_put_returns_qty_limit_price_and_side(self):
        contract = {
            "strategy": "sell_cash_secured_put",
            "occ_symbol": "NU260904P00014000",
            "side": "sell",
            "qty": 1,
            "limit_price": 0.18,
        }
        with patch.object(options_execution, "requests") as mock_requests:
            mock_requests.post.return_value = self._mock_response()
            result = options_execution.submit_options_order(contract)

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["order_id"], "test-order-id-123")
        # The fields a caller needs to register this resting DAY limit order
        # with a stale-order sweeper — absent before this fix.
        self.assertEqual(result["qty"], 1)
        self.assertEqual(result["limit_price"], 0.18)
        self.assertEqual(result["side"], "sell")

    def test_successful_buy_put_returns_qty_limit_price_and_side(self):
        contract = {
            "strategy": "buy_put",
            "occ_symbol": "SPY260918P00500000",
            "side": "buy",
            "qty": 2,
            "limit_price": 3.45,
        }
        with patch.object(options_execution, "requests") as mock_requests:
            mock_requests.post.return_value = self._mock_response(order_id="hedge-order-456")
            result = options_execution.submit_options_order(contract)

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["qty"], 2)
        self.assertEqual(result["limit_price"], 3.45)
        self.assertEqual(result["side"], "buy")

    def test_error_response_does_not_claim_tracking_fields(self):
        contract = {
            "strategy": "sell_cash_secured_put",
            "occ_symbol": "NU260904P00014000",
            "side": "sell",
            "qty": 1,
            "limit_price": 0.18,
        }
        with patch.object(options_execution, "requests") as mock_requests:
            mock_requests.post.return_value = self._mock_response(status_code=422)
            result = options_execution.submit_options_order(contract)

        self.assertEqual(result["status"], "error")
        self.assertNotIn("order_id", result)
        self.assertNotIn("qty", result)


if __name__ == "__main__":
    unittest.main()
