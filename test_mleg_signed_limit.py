"""MLEG LIMIT SIGN (2026-11-06, human-approved frozen-path change, the sign only).

Alpaca's API reference, Create an Order (docs.alpaca.markets/reference/postorder),
limit_price: "In case of `mleg`, the limit_price parameter is expressed with the
following notation: - A positive value indicates a debit, representing a cost or
payment to be made. - A negative value signifies a credit, reflecting an amount to
be received." The Replace an Order reference says the same.

Before: options_execution._submit_mleg_order sent abs(limit_price), so a credit
entry went out as a debit cap, which every credit satisfies. Live
(/api/diag/orders 2026-10-19..11-04): all five limit mleg orders were credit
condor entries sent positive (0.71, 4.59, 3.56, 9.83, 4.48) and filled within
0.2 s at -0.70, -4.55, -3.53, -9.85, -4.59, three of them below the intended credit.

Pinned here:
  - the rule: a credit is negative and a debit positive, taken from the
    structure, never from the number's own sign; it is mleg_ladder.signed_limit,
    the rule the mid-first closes already use;
  - per caller, through submit_options_order (the real dispatch) and directly:
    iron_condor, qqq_iron_condor, short_straddle, bull_put_credit_spread go out
    negative; bull_call_spread, bear_put_spread, buy_straddle, calendar_spread,
    diagonal_spread positive; a mid-first debit vertical at its positive net mid;
  - a credit entry priced 0.71 is sent as -0.71 (the 2026-11-04 condor's price);
  - nothing else changes: the whole payload, the URL, the headers, the timeout;
    a debit's limit string is exactly what the old code sent, a credit's is
    the old string with a minus sign;
  - _submit_mleg_order sends the number as signed by its caller (no abs);
  - every caller of _submit_mleg_order names its side through
    _mleg_credit_limit / _mleg_debit_limit: a bare number fails this file.
"""
import ast
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mleg_ladder  # noqa: E402
import options_execution as oe  # noqa: E402

SIGN_HELPERS = ("_mleg_credit_limit", "_mleg_debit_limit")


def _occ(right, strike, root="XYZ", ymd="261218"):
    return f"{root}{ymd}{right}{int(round(strike * 1000)):08d}"


def _ok(order_id="mleg-1", status="pending_new"):
    r = MagicMock(status_code=200, headers={"content-type": "application/json"}, text="")
    r.json.return_value = {"id": order_id, "status": status}
    return r


def _old_limit_string(v):
    """What the pre-2026-11-06 code put in the payload for any caller's number."""
    return str(round(abs(v), 2))


def _leg(symbol, side):
    return {"symbol": symbol, "side": side, "ratio_qty": "1",
            "position_intent": "buy_to_open" if side == "buy" else "sell_to_open"}


# Contract dicts shaped like each selector's output (the keys the wrappers read).
def condor(net_credit=0.71, qty=3, strategy="iron_condor"):
    c = {"strategy": strategy, "qty": qty, "net_credit": net_credit, "error": None,
         "short_call": _occ("C", 16), "long_call": _occ("C", 17),
         "short_put": _occ("P", 12), "long_put": _occ("P", 11)}
    if strategy == "qqq_iron_condor":
        c["limit_price"] = net_credit    # _select_qqq_iron_condor records both
    return c


def short_straddle(limit=3.56, call_credit=1.80, put_credit=1.76):
    c = {"strategy": "short_straddle", "qty": 1, "call_leg": _occ("C", 14), "put_leg": _occ("P", 14),
         "call_credit": call_credit, "put_credit": put_credit, "error": None}
    if limit is not None:
        c["limit_price"] = limit
    return c


def buy_straddle(limit=5.0):
    return {"strategy": "buy_straddle", "qty": 1, "call_leg": _occ("C", 14), "put_leg": _occ("P", 14),
            "call_debit": 2.6, "put_debit": 2.4, "limit_price": limit, "error": None}


def bull_put_credit_spread(net_credit=0.71):
    return {"strategy": "bull_put_credit_spread", "qty": 2, "short_leg": _occ("P", 12), "long_leg": _occ("P", 11),
            "net_credit": net_credit, "limit_price": net_credit, "error": None}


def debit_vertical(strategy="bull_call_spread", net_debit=2.5):
    right = "C" if strategy == "bull_call_spread" else "P"
    return {"strategy": strategy, "qty": 2, "long_leg": _occ(right, 14), "short_leg": _occ(right, 15),
            "net_debit": net_debit, "limit_price": net_debit, "error": None}


def time_spread(strategy="calendar_spread", net_debit=1.4):
    return {"strategy": strategy, "qty": 1, "long_leg": _occ("C", 14, ymd="270115"),
            "short_leg": _occ("C", 15 if strategy == "diagonal_spread" else 14),
            "net_debit": net_debit, "limit_price": net_debit, "error": None}


class _SendCase(unittest.TestCase):
    """The order path with the broker clock open, the fill ledger stubbed and the POST mocked."""

    def setUp(self):
        self.post = MagicMock(return_value=_ok())
        for p in (patch.object(oe.requests, "post", self.post),
                  patch.object(oe.broker_clock, "check_order_allowed", return_value=None),
                  patch.object(oe._fill_ledger, "record_mleg_order", return_value=0)):
            p.start()
            self.addCleanup(p.stop)

    def sent(self, fn, *args, **kwargs):
        self.post.reset_mock()
        out = fn(*args, **kwargs)
        self.assertEqual(self.post.call_count, 1, f"exactly one POST ({out})")
        return self.post.call_args.kwargs["json"]


class TestTheRule(unittest.TestCase):
    def test_a_credit_is_negative_and_a_debit_positive(self):
        self.assertEqual(oe._mleg_credit_limit(0.71), -0.71)
        self.assertEqual(oe._mleg_debit_limit(2.5), 2.5)

    def test_the_sign_comes_from_the_structure_not_the_number(self):
        self.assertEqual(oe._mleg_credit_limit(-0.71), -0.71, "an already-signed credit stays a credit")
        self.assertEqual(oe._mleg_debit_limit(-2.5), 2.5, "a debit is positive whatever the number's sign")

    def test_it_is_the_rule_the_mid_first_closes_use(self):
        for v in (0.01, 0.7, 0.71, 3.05, 3.56, 4.48, 4.59, 9.83):
            with self.subTest(v=v):
                self.assertEqual(oe._mleg_credit_limit(v), mleg_ladder.signed_limit("sell", v))
                self.assertEqual(oe._mleg_debit_limit(v), mleg_ladder.signed_limit("buy", v))

    def test_a_zero_credit_is_zero_not_minus_zero(self):
        z = oe._mleg_credit_limit(0)
        self.assertEqual((z, math.copysign(1.0, z), str(round(z, 2))), (0.0, 1.0, "0.0"))


class TestEveryCallerSendsItsSign(_SendCase):
    def test_a_credit_entry_priced_0_71_goes_out_as_minus_0_71(self):
        # the 2026-11-04 condor's price (live: sent 0.71, filled at -0.70 within 0.14 s)
        c = condor(net_credit=0.71, qty=3)
        payload = self.sent(oe.submit_options_order, c)
        self.assertEqual(payload, {
            "order_class": "mleg", "qty": "3", "type": "limit", "limit_price": "-0.71", "time_in_force": "day",
            "legs": [_leg(c["short_call"], "sell"), _leg(c["long_call"], "buy"),
                     _leg(c["short_put"], "sell"), _leg(c["long_put"], "buy")],
        })
        bp = bull_put_credit_spread(net_credit=0.71)
        payload = self.sent(oe.submit_options_order, bp)
        self.assertEqual(payload, {
            "order_class": "mleg", "qty": "2", "type": "limit", "limit_price": "-0.71", "time_in_force": "day",
            "legs": [_leg(bp["long_leg"], "buy"), _leg(bp["short_leg"], "sell")],
        })

    def test_credit_structures_are_sent_negative(self):
        cases = [
            ("iron_condor", condor(4.48, qty=1), "-4.48"),
            ("qqq_iron_condor", condor(4.59, qty=2, strategy="qqq_iron_condor"), "-4.59"),
            ("short_straddle", short_straddle(limit=3.56), "-3.56"),
            ("short_straddle without a limit (bids summed)", short_straddle(limit=None), "-3.56"),
            ("bull_put_credit_spread", bull_put_credit_spread(9.83), "-9.83"),
        ]
        for name, contract, want in cases:
            with self.subTest(name):
                self.assertEqual(self.sent(oe.submit_options_order, contract)["limit_price"], want)

    def test_debit_structures_are_sent_positive(self):
        cases = [
            ("bull_call_spread", debit_vertical("bull_call_spread", 2.5), "2.5"),
            ("bear_put_spread", debit_vertical("bear_put_spread", 1.25), "1.25"),
            ("buy_straddle", buy_straddle(5.0), "5.0"),
            ("calendar_spread (dispatch: the 'spread' match)", time_spread("calendar_spread", 1.4), "1.4"),
            ("diagonal_spread (dispatch: the 'spread' match)", time_spread("diagonal_spread", 6.35), "6.35"),
        ]
        for name, contract, want in cases:
            with self.subTest(name):
                self.assertEqual(self.sent(oe.submit_options_order, contract)["limit_price"], want)

    def test_each_wrapper_called_directly(self):
        cases = [
            (oe._submit_condor_order, condor(0.71), "-0.71"),
            (oe._submit_straddle_order, short_straddle(3.56), "-3.56"),
            (oe._submit_spread_order, bull_put_credit_spread(0.71), "-0.71"),
            (oe._submit_spread_order, debit_vertical("bull_call_spread", 2.5), "2.5"),
            (oe._submit_buy_straddle_order, buy_straddle(5.0), "5.0"),
            (oe._submit_calendar_order, time_spread("calendar_spread", 1.4), "1.4"),
            (oe._submit_diagonal_order, time_spread("diagonal_spread", 6.35), "6.35"),
        ]
        for fn, contract, want in cases:
            with self.subTest(fn=fn.__name__, strategy=contract["strategy"]):
                self.assertEqual(self.sent(fn, contract)["limit_price"], want)

    def test_a_credit_vertical_named_by_strategy_alone_is_still_a_credit(self):
        c = {"strategy": "bull_put_credit_spread", "qty": 1, "short_leg": _occ("P", 12), "long_leg": _occ("P", 11),
             "limit_price": 0.71, "error": None}       # no net_credit key: the strategy name decides
        self.assertEqual(self.sent(oe.submit_options_order, c)["limit_price"], "-0.71")

    def test_a_debit_is_byte_identical_and_a_credit_is_the_old_string_negated(self):
        for v in (0.07, 1.25, 2.5, 5.0, 12.345, 6.355):
            with self.subTest(debit=v):
                got = self.sent(oe.submit_options_order, debit_vertical("bull_call_spread", v))["limit_price"]
                self.assertEqual(got, _old_limit_string(v))
        for v in (0.71, 4.59, 3.56, 9.83, 4.48, 12.345):   # the five live credit limits + a rounding case
            with self.subTest(credit=v):
                got = self.sent(oe.submit_options_order, condor(v))["limit_price"]
                self.assertEqual(got, "-" + _old_limit_string(v))

    def test_a_mid_first_debit_vertical_goes_out_at_its_positive_net_mid(self):
        c = debit_vertical("bull_call_spread", 3.2)
        c["leg_quotes"] = [{"symbol": c["long_leg"], "side": "buy", "bid": 5.00, "ask": 5.20},
                           {"symbol": c["short_leg"], "side": "sell", "bid": 2.00, "ask": 2.10}]
        with patch.object(mleg_ladder, "settings",
                          return_value={"on": True, "steps": 3, "step_seconds": 60.0, "touch_seconds": 120.0}):
            c = oe._mid_first_entry(c)
        self.assertEqual(c["limit_price"], 3.05)
        self.assertEqual(self.sent(oe.submit_options_order, c)["limit_price"], "3.05")

    def test_the_contract_is_not_mutated_by_the_sign(self):
        c = condor(0.71)
        before = dict(c)
        self.sent(oe.submit_options_order, c)
        self.assertEqual(c, before, "the decision journal reads the contract's own (unsigned) prices")


class TestTheFrozenSend(_SendCase):
    LEGS = [{"symbol": _occ("C", 16), "side": "sell", "ratio_qty": 1},
            {"symbol": _occ("C", 17), "side": "buy", "ratio_qty": 1}]

    def test_the_number_goes_out_as_its_caller_signed_it(self):
        self.assertEqual(self.sent(oe._submit_mleg_order, self.LEGS, 2, -1.2, "bear call")["limit_price"], "-1.2")
        self.assertEqual(self.sent(oe._submit_mleg_order, self.LEGS, 2, 1.2, "debit")["limit_price"], "1.2")

    def test_transmission_is_unchanged(self):
        with patch.object(oe, "ALPACA_KEY", "k-id"), patch.object(oe, "ALPACA_SECRET", "k-secret"):
            out = oe._submit_mleg_order(self.LEGS, 2, -1.2, "bear call")
        self.assertEqual(self.post.call_count, 1)
        args, kwargs = self.post.call_args
        self.assertEqual(args, (f"{oe.ALPACA_BASE}/v2/orders",))
        self.assertEqual(set(kwargs), {"headers", "json", "timeout"})
        self.assertEqual(kwargs["timeout"], 15)
        self.assertEqual(kwargs["headers"], {"APCA-API-KEY-ID": "k-id", "APCA-API-SECRET-KEY": "k-secret",
                                             "Content-Type": "application/json"})
        self.assertEqual(set(kwargs["json"]), {"order_class", "qty", "type", "limit_price", "time_in_force", "legs"})
        self.assertEqual((out["status"], out["order_id"]), ("pending_new", "mleg-1"))

    def test_a_refusal_is_reported_as_before(self):
        self.post.return_value = MagicMock(status_code=422, headers={"content-type": "application/json"},
                                           text="", json=lambda: {"message": "invalid limit_price"})
        out = oe._submit_condor_order(condor(0.71))
        self.assertEqual(out, {"status": "error", "detail": "Iron condor x3: Alpaca rejected mleg order — invalid limit_price"})

    def test_nothing_is_sent_while_the_broker_clock_says_closed(self):
        with patch.object(oe.broker_clock, "check_order_allowed",
                          side_effect=oe.broker_clock.OrderBlocked("ORDER-BLOCKED: market closed")):
            out = oe._submit_condor_order(condor(0.71))
        self.post.assert_not_called()
        self.assertEqual(out["status"], "error")


def _python_sources():
    skip = {".git", "node_modules", "dist", "client", "__pycache__", ".venv", "venv"}
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for f in files:
            if f.endswith(".py") and not f.startswith("test_"):
                yield os.path.join(root, f)


class TestNoCallerSendsABareNumber(unittest.TestCase):
    """Static: every _submit_mleg_order call names its side, so a credit can never again
    go out as a positive number."""

    @staticmethod
    def _is_helper_call(node):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SIGN_HELPERS

    def test_every_caller_in_options_execution_names_its_side(self):
        path = os.path.join(HERE, "options_execution.py")
        with open(path) as f:
            tree = ast.parse(f.read(), path)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_submit_mleg_order"]
        self.assertGreaterEqual(len(calls), 6, "the six multi-leg wrappers")
        for call in calls:
            with self.subTest(line=call.lineno):
                limit = next((k.value for k in call.keywords if k.arg == "limit_price"),
                             call.args[2] if len(call.args) > 2 else None)
                self.assertIsNotNone(limit)
                ok = self._is_helper_call(limit) or (
                    isinstance(limit, ast.IfExp) and self._is_helper_call(limit.body) and self._is_helper_call(limit.orelse))
                self.assertTrue(ok, f"options_execution.py:{call.lineno} passes a bare limit to _submit_mleg_order; "
                                    f"name the side with _mleg_credit_limit / _mleg_debit_limit")

    def test_no_other_module_calls_it(self):
        callers = []
        for path in _python_sources():
            if os.path.basename(path) == "options_execution.py":
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                if "_submit_mleg_order(" in f.read():
                    callers.append(os.path.relpath(path, HERE))
        server = os.path.join(HERE, "server")
        for f in sorted(os.listdir(server)) if os.path.isdir(server) else []:
            if f.endswith(".ts") and not f.endswith(".test.ts"):
                with open(os.path.join(server, f), encoding="utf-8", errors="replace") as fh:
                    if "_submit_mleg_order(" in fh.read():
                        callers.append(f"server/{f}")
        self.assertEqual(callers, [], "a new caller must go through the signed wrappers in options_execution.py")


if __name__ == "__main__":
    unittest.main()
