"""REPAIR 2026-11-04 — a multi-leg strategy's LOSS LIMIT is measured against
its NET credit (net debit), as the rule's own comments always said ("Lost 2x
credit", "Lost 50% of debit"), and the iron-condor 50%-of-max-loss exit
derives the max loss from the legs when no leg's state recorded one.

The bug: options_manager._manage_strategy_group divided the group's
unrealized loss by the GROSS premium of every leg (shorts and longs added
together). On a spread that is several times the net, which typically puts the
trigger beyond the structure's own max loss: the GOOGL Nov-27 condor (short
310P / long 300P / short 355C / long 365C, x2) had 2x gross = $9,700 against
a $1,090 max loss, so the rule could never fire. A debit vertical's max loss is
its net debit, below 50 % of its gross, so the debit rule could not fire
either. And the IC max-loss exit read max_loss only from the legs' state, so
with GOOGL's state re-initialised it did nothing as well.

Pinned here ("old:" claims fail on the pre-fix code):
  (a) a condor whose loss passes 2x its NET credit closes as one group, through
      the pursued group close (old: nothing).
  (b) all-short structures keep today's decisions at every mark: the standalone
      CSP rule (a separate, untouched path) on both sides of its threshold, and a
      lone short put / a short straddle judged as a group, swept across the
      threshold against the pre-fix formula.
  (c) a debit vertical losing 50 % of its NET debit closes (old: nothing); a
      long straddle (all legs long) keeps today's decisions at every mark.
  (d) a condor with no max_loss in state: the IC exit uses the max loss derived
      from the legs (old: silent); a recorded max_loss still wins; shapes the
      formula does not fit are never priced.
  (e) GOOGL-like numbers. The old trigger lay beyond the condor's max loss. So
      does the corrected 2x-credit trigger ($1,820 vs $1,090) — a condor that
      collected more than a third of its wing width can never lose 2x its credit
      — so on this condor the stop that acts is the IC exit at $545 on the
      leg-derived max loss (old: no exit at any reachable loss). GOOGL's put
      side alone, a credit vertical, closes at the documented 2x net credit
      ($900), which the old $4,660 trigger never could.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402

DAYS = 30  # beyond the 21-DTE and 5-DTE rules, so only the P&L rules can act
SNAP = {"bid": 1.00, "ask": 1.20, "mid": 1.10, "delta": -0.3, "gamma": 0.01, "theta": -0.02, "vega": 0.05, "iv": 0.4}
CREDIT = ("short_straddle", "iron_condor", "csp_normal_market", "bull_put_credit_spread",
          "sell_cash_secured_put", "covered_call", "qqq_iron_condor")
DEBIT = ("buy_straddle", "bull_call_spread", "bear_put_spread")
LOSS_LIMIT_KEYS = {"action", "ticker", "strategy", "legs", "reason", "type", "combined_pnl", "order"}
IC_EXIT_KEYS = LOSS_LIMIT_KEYS | {"max_loss", "loss_pct_of_max"}


def occ(root, cp, strike, days=DAYS):
    exp = (datetime.now() + timedelta(days=days)).strftime("%y%m%d")
    return f"{root}{exp}{cp}{int(round(strike * 1000)):08d}"


def leg(symbol, qty, side, entry, cur):
    """An Alpaca /v2/positions row; unrealized_pl worked from the mark, as Alpaca reports it."""
    sign = 1 if side == "long" else -1
    return {"symbol": symbol, "qty": str(qty * sign), "side": side, "avg_entry_price": str(entry),
            "current_price": str(cur), "market_value": str(cur * qty * 100 * sign),
            "unrealized_pl": str((cur - entry) * qty * 100 * sign), "asset_class": "us_option"}


def pnl_of(legs):
    return sum(float(p["unrealized_pl"]) for p in legs)


def gross_of(legs):
    """The pre-fix denominator: every leg's premium added, shorts and longs alike."""
    return sum(abs(float(p["avg_entry_price"])) * abs(int(float(p["qty"]))) * 100 for p in legs)


def net_credit_of(legs):
    return sum((1 if p["side"] == "short" else -1) * abs(float(p["avg_entry_price"])) * abs(int(float(p["qty"]))) * 100
               for p in legs)


def pre_fix_loss_limit(legs, strategy):
    """Rule 3 exactly as it stood before 2026-11-04: P&L over the GROSS premium."""
    gross = gross_of(legs)
    if gross <= 0:
        return False
    if strategy in CREDIT:
        return pnl_of(legs) / gross <= -om.LOSS_LIMIT_MULTIPLIER
    if strategy in DEBIT:
        return pnl_of(legs) / gross <= -0.50
    return False


def state_for(legs, strategy, root, max_loss=None):
    """The legs' options state as register_options_entry writes it, held 10 days."""
    ts = (datetime.now() - timedelta(days=10)).isoformat()
    state = {}
    for p in legs:
        entry, short = float(p["avg_entry_price"]), p["side"] == "short"
        st = {"entry_price": entry, "entry_delta": 0.2, "side": p["side"], "entry_date": ts[:10], "entry_timestamp": ts,
              "strategy": strategy, "ticker": root, "setup": "test", "qty": abs(int(float(p["qty"]))),
              "initial_credit": entry if short else 0, "highest_value": entry}
        if max_loss is not None:
            st["max_loss"] = max_loss
        state[p["symbol"]] = st
    return state


class Broker:
    """requests.get / requests.post stand-in: positions, no open orders, market open, mleg accepted."""

    def __init__(self, positions):
        self.positions = positions
        self.posts = []

    def get(self, url, **kw):
        if "/v2/positions" in url:
            return MagicMock(status_code=200, json=lambda: self.positions)
        if "/v2/orders" in url:
            return MagicMock(status_code=200, json=lambda: [])
        if "/v2/clock" in url:
            return MagicMock(status_code=200, json=lambda: {"is_open": True})
        return MagicMock(status_code=404, json=lambda: {}, text="")

    def post(self, url, **kw):
        self.posts.append((url, kw.get("json")))
        return MagicMock(status_code=200, text="", json=lambda: {"id": "mleg-1"})


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = os.path.join(self.tmp.name, "voltrade_options_state.json")
        self.submits = []

        def fake_submit(occ_symbol, qty, side, limit_price):
            self.submits.append((occ_symbol, qty, side, limit_price))
            return {"status": "submitted", "order_id": f"leg-{len(self.submits)}"}
        self.patches = [
            patch.object(om, "DATA_DIR", self.tmp.name),
            patch.object(om, "OPTIONS_STATE_PATH", self.state_path),
            patch.object(om, "_get_option_snapshot", return_value=dict(SNAP)),
            patch.object(om, "_attempt_roll", return_value={"rolled": False, "detail": "no roll in tests"}),
            patch.object(om, "_submit_close_order", side_effect=fake_submit),
            patch.dict(sys.modules, {"vol_surface": MagicMock(get_surface_score=lambda t: {"vrp": {"vrp_20d": 0.0}}),
                                     "ml_model_v2": MagicMock(track_fill=lambda *a, **k: None)}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()
        # the clock read is cached module-wide for 60 s: never leave this file's "open" behind
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})

    def run_manager(self, positions, state):
        """One manage_options_positions tick against these broker positions and this state."""
        with open(self.state_path, "w") as f:
            json.dump(state, f)
        broker = Broker(positions)
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})
        with patch.object(om.requests, "get", side_effect=broker.get), patch.object(om.requests, "post", side_effect=broker.post):
            result = om.manage_options_positions(100000, allow_rolls=False)
        with open(self.state_path) as f:
            saved = json.load(f)
        return result, saved, broker

    def judge(self, legs, strategy):
        """_manage_strategy_group on one group with no prior state, as Phase 1 calls it."""
        broker = Broker(legs)
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})
        group = {"legs": legs, "strategy": strategy, "ticker": "X", "setup": ""}
        with patch.object(om.requests, "get", side_effect=broker.get), patch.object(om.requests, "post", side_effect=broker.post):
            return om._manage_strategy_group(group, 100000, {})

    @staticmethod
    def group_closes(actions):
        return [a for a in actions if a.get("action") == "CLOSE_STRATEGY"]


# ── (a) condor: loss past 2x its NET credit closes the group ─────────────────
def low_credit_condor(short_put_mark, long_put_mark=0.75):
    """5-wide condor x5 that collected $0.90 (18 % of its width): net credit $450,
    gross $1,550, max loss 5 x 100 x 5 - 450 = $2,050. Call side has decayed
    (+$150); the put side is tested."""
    root = "LOWC"
    return root, [
        leg(occ(root, "P", 95), 5, "short", 1.00, short_put_mark),
        leg(occ(root, "P", 90), 5, "long", 0.55, long_put_mark),
        leg(occ(root, "C", 105), 5, "short", 1.00, 0.20),
        leg(occ(root, "C", 110), 5, "long", 0.55, 0.05),
    ]


class TestCondorLossLimitOnNetCredit(Case):
    def test_condor_loss_past_twice_its_net_credit_closes_as_one_group(self):
        root, legs = low_credit_condor(short_put_mark=3.40)  # loss $950
        self.assertAlmostEqual(pnl_of(legs), -950.0, places=6)
        self.assertAlmostEqual(net_credit_of(legs), 450.0, places=6)
        self.assertAlmostEqual(gross_of(legs), 1550.0, places=6)
        # the old trigger (2x gross = $3,100) is beyond what this condor can lose ($2,050): it could never fire
        self.assertGreater(om.LOSS_LIMIT_MULTIPLIER * gross_of(legs), 2050.0)
        self.assertFalse(pre_fix_loss_limit(legs, "iron_condor"))
        result, saved, broker = self.run_manager(legs, state_for(legs, "iron_condor", root, max_loss=2050.0))
        acts = self.group_closes(result["actions"])
        self.assertEqual(len(acts), 1, result["actions"])
        act = acts[0]
        self.assertEqual(act["type"], "strategy_loss_limit", "the IC exit is at $1,025 (50 % of $2,050) — not reached; Rule 3 at $900 is")
        self.assertEqual(set(act), LOSS_LIMIT_KEYS, "the action keeps its shape")
        self.assertEqual(sorted(act["legs"]), sorted(p["symbol"] for p in legs))
        self.assertAlmostEqual(act["combined_pnl"], -950.0, places=2)
        self.assertEqual(act["reason"], "LOSS LIMIT: loss $950 is 211% of the net credit $450 (limit 200%) on iron_condor — cutting")
        # the close is the pursued group close (one mleg for all four legs; every leg stamped, none popped)
        self.assertEqual(len(broker.posts), 1)
        self.assertEqual(broker.posts[0][1]["order_class"], "mleg")
        self.assertEqual(len(broker.posts[0][1]["legs"]), 4)
        self.assertFalse(self.submits, "no per-leg orders after an accepted mleg")
        for p in legs:
            self.assertEqual(saved[p["symbol"]]["close_rule"], "strategy_loss_limit")
            self.assertEqual(saved[p["symbol"]]["strategy"], "iron_condor")

    def test_just_short_of_twice_the_net_credit_the_condor_is_held(self):
        root, legs = low_credit_condor(short_put_mark=3.20)  # loss $850 < $900
        self.assertAlmostEqual(pnl_of(legs), -850.0, places=6)
        result, saved, broker = self.run_manager(legs, state_for(legs, "iron_condor", root, max_loss=2050.0))
        self.assertEqual(self.group_closes(result["actions"]), [], result["actions"])
        self.assertFalse(broker.posts)
        self.assertFalse(any(saved[p["symbol"]].get("close_requested_at") for p in legs))


# ── (b) all-short structures: today's decisions, at every mark ────────────────
class TestAllShortStructuresUnchanged(Case):
    def test_standalone_csp_loss_limit_both_sides_of_the_threshold(self):
        """A CSP is not a multi-leg strategy: the single-leg rule judges it (cost to
        close >= 2x the credit). This change does not touch that path — pinned."""
        root = "CSPX"
        sym = occ(root, "P", 100)
        st = state_for([leg(sym, 1, "short", 2.00, 2.00)], "sell_cash_secured_put", root)
        held, _, _ = self.run_manager([leg(sym, 1, "short", 2.00, 3.99)], st)
        self.assertEqual([a for a in held["actions"] if a.get("action") in ("CLOSE", "CLOSE_STRATEGY")], [])
        self.assertFalse(self.submits)
        st = state_for([leg(sym, 1, "short", 2.00, 2.00)], "sell_cash_secured_put", root)
        cut, _, _ = self.run_manager([leg(sym, 1, "short", 2.00, 4.00)], st)
        closes = [a for a in cut["actions"] if a.get("action") == "CLOSE"]
        self.assertEqual([(a["type"], a["occ_symbol"], a["loss_ratio"]) for a in closes], [("loss_limit", sym, 2.0)])
        self.assertEqual([(s, side) for s, _q, side, _px in self.submits], [(sym, "buy")])

    def _sweep(self, build, marks, strategy, boundary):
        decisions = []
        for m in marks:
            legs = build(m)
            fired = any(a.get("type") == "strategy_loss_limit" for a in self.judge(legs, strategy))
            self.assertEqual(fired, pre_fix_loss_limit(legs, strategy),
                             f"{strategy} at mark {m}: pnl {pnl_of(legs):.2f} on gross {gross_of(legs):.2f}")
            decisions.append((m, fired))
        self.assertIn((boundary, True), decisions, "the exact threshold fires, as before")
        self.assertTrue(any(not f for _m, f in decisions) and any(f for _m, f in decisions), decisions)

    def test_lone_short_put_judged_as_a_group_decides_exactly_as_before(self):
        # credit $2.00 x1: net credit == gross == $200, the limit is a $400 loss (mark 6.00)
        sym = occ("CSPG", "P", 100)
        marks = [round(5.90 + i * 0.01, 2) for i in range(21)] + [2.10, 9.50]
        self._sweep(lambda m: [leg(sym, 1, "short", 2.00, m)], marks, "sell_cash_secured_put", boundary=6.00)

    def test_short_straddle_decides_exactly_as_before(self):
        # $3.00 put + $2.50 call x1: net credit == gross == $550, the limit is a $1,100 loss
        put, call = occ("STRD", "P", 100), occ("STRD", "C", 100)
        marks = [round(15.90 + i * 0.01, 2) for i in range(21)] + [4.00, 25.00]
        self._sweep(lambda m: [leg(put, 1, "short", 3.00, m), leg(call, 1, "short", 2.50, 0.50)],
                    marks, "short_straddle", boundary=16.00)


# ── (c) debit structures: 50 % of the NET debit ───────────────────────────────
def bull_call_spread(long_mark):
    """long 100C at $5.00 / short 105C at $3.00, x2: net debit $400 (its max loss), gross $1,600."""
    root = "DEBV"
    return root, [leg(occ(root, "C", 100), 2, "long", 5.00, long_mark), leg(occ(root, "C", 105), 2, "short", 3.00, 2.65)]


class TestDebitLossLimitOnNetDebit(Case):
    def test_debit_vertical_losing_half_its_net_debit_closes(self):
        root, legs = bull_call_spread(long_mark=3.55)  # loss $220 of a $400 debit
        self.assertAlmostEqual(pnl_of(legs), -220.0, places=6)
        self.assertAlmostEqual(-net_credit_of(legs), 400.0, places=6)
        # the old trigger (50 % of the $1,600 gross) is beyond the spread's whole $400 debit: it could never fire
        self.assertGreater(0.50 * gross_of(legs), -net_credit_of(legs))
        self.assertFalse(pre_fix_loss_limit(legs, "bull_call_spread"))
        result, saved, broker = self.run_manager(legs, state_for(legs, "bull_call_spread", root))
        acts = self.group_closes(result["actions"])
        self.assertEqual(len(acts), 1, result["actions"])
        self.assertEqual(acts[0]["type"], "strategy_loss_limit")
        self.assertEqual(set(acts[0]), LOSS_LIMIT_KEYS)
        self.assertEqual(acts[0]["reason"], "LOSS LIMIT: loss $220 is 55% of the net debit $400 on bull_call_spread — cutting debit position")
        self.assertEqual(len(broker.posts), 1)
        self.assertTrue(all(saved[p["symbol"]]["close_rule"] == "strategy_loss_limit" for p in legs))

    def test_debit_vertical_short_of_half_its_net_debit_is_held(self):
        root, legs = bull_call_spread(long_mark=3.70)  # loss $190 < $200
        self.assertAlmostEqual(pnl_of(legs), -190.0, places=6)
        result, _, broker = self.run_manager(legs, state_for(legs, "bull_call_spread", root))
        self.assertEqual(self.group_closes(result["actions"]), [], result["actions"])
        self.assertFalse(broker.posts)

    def test_long_straddle_decides_exactly_as_before(self):
        # $3.00 call + $2.00 put x1, all long: net debit == gross == $500, the limit is a $250 loss
        call, put = occ("LSTR", "C", 100), occ("LSTR", "P", 100)
        decisions = []
        for m in [round(1.40 + i * 0.01, 2) for i in range(21)] + [0.05, 6.00]:
            legs = [leg(call, 1, "long", 3.00, 1.00), leg(put, 1, "long", 2.00, m)]
            fired = any(a.get("type") == "strategy_loss_limit" for a in self.judge(legs, "buy_straddle"))
            self.assertEqual(fired, pre_fix_loss_limit(legs, "buy_straddle"), f"mark {m}")
            decisions.append((m, fired))
        self.assertIn((1.50, True), decisions, "the exact threshold fires, as before")
        self.assertIn((1.51, False), decisions)


# ── (d) the IC max-loss exit without a recorded max_loss ──────────────────────
def condor_w10(short_put_mark=8.60, long_put_mark=1.50):
    """10-wide condor x1, credit $4.00 (net $400): max loss 10 x 100 - 400 = $600;
    at the default marks the loss is $350 (58 % of max loss)."""
    root = "WIDE"
    return root, [
        leg(occ(root, "P", 90), 1, "short", 3.00, short_put_mark),
        leg(occ(root, "P", 80), 1, "long", 1.00, long_put_mark),
        leg(occ(root, "C", 110), 1, "short", 3.00, 0.50),
        leg(occ(root, "C", 120), 1, "long", 1.00, 0.10),
    ]


class TestIcMaxLossFromTheLegs(Case):
    def test_condor_with_no_recorded_max_loss_exits_on_the_leg_derived_one(self):
        root, legs = condor_w10()
        self.assertAlmostEqual(pnl_of(legs), -350.0, places=6)
        result, saved, broker = self.run_manager(legs, state_for(legs, "iron_condor", root))  # no max_loss key
        acts = self.group_closes(result["actions"])
        self.assertEqual(len(acts), 1, result["actions"])
        act = acts[0]
        self.assertEqual(act["type"], "strategy_ic_max_loss_exit")
        self.assertEqual(set(act), IC_EXIT_KEYS)
        self.assertEqual(act["max_loss"], 600.0)
        self.assertEqual(act["loss_pct_of_max"], 58.3)
        self.assertEqual(act["reason"], "IC MAX-LOSS EXIT: Unrealized loss $350.00 is 58% of max loss $600.00 (from the legs; none recorded) — "
                                        "closing before full max loss (threshold: 50%)")
        self.assertTrue(all(saved[p["symbol"]]["close_rule"] == "strategy_ic_max_loss_exit" for p in legs))

    def test_a_recorded_max_loss_still_wins(self):
        root, legs = condor_w10()
        result, _, broker = self.run_manager(legs, state_for(legs, "iron_condor", root, max_loss=1000.0))
        # $350 is 35 % of the recorded $1,000 (no IC exit) and under 2x the $400 credit (no loss limit)
        self.assertEqual(self.group_closes(result["actions"]), [], result["actions"])
        self.assertFalse(broker.posts)

    def test_the_recorded_max_loss_reason_is_unchanged(self):
        root, legs = condor_w10()
        result, _, _ = self.run_manager(legs, state_for(legs, "iron_condor", root, max_loss=600.0))
        act = self.group_closes(result["actions"])[0]
        self.assertEqual(act["reason"], "IC MAX-LOSS EXIT: Unrealized loss $350.00 is 58% of max loss $600.00 — "
                                        "closing before full max loss (threshold: 50%)")

    def test_max_loss_from_legs_prices_only_condors_and_credit_verticals(self):
        f = om._max_loss_from_legs
        _, condor = condor_w10()
        self.assertEqual(f(condor, net_credit_of(condor)), 600.0)
        put_credit = [leg(occ("V", "P", 100), 3, "short", 2.00, 2.00), leg(occ("V", "P", 95), 3, "long", 1.00, 1.00)]
        self.assertEqual(f(put_credit, net_credit_of(put_credit)), 1200.0)  # 5 x 100 x 3 - 300
        call_credit = [leg(occ("V", "C", 105), 1, "short", 2.00, 2.00), leg(occ("V", "C", 110), 1, "long", 1.00, 1.00)]
        self.assertEqual(f(call_credit, net_credit_of(call_credit)), 400.0)
        uneven = [leg(occ("U", "P", 90), 1, "short", 3.00, 3.00), leg(occ("U", "P", 80), 1, "long", 1.00, 1.00),
                  leg(occ("U", "C", 110), 1, "short", 2.00, 2.00), leg(occ("U", "C", 115), 1, "long", 1.20, 1.20)]
        self.assertEqual(f(uneven, net_credit_of(uneven)), 720.0)  # the widest wing: 10 x 100 - 280
        never = {
            "no legs": [],
            "long straddle": [leg(occ("N", "C", 100), 1, "long", 3.00, 3.00), leg(occ("N", "P", 100), 1, "long", 2.00, 2.00)],
            "short straddle": [leg(occ("N", "C", 100), 1, "short", 3.00, 3.00), leg(occ("N", "P", 100), 1, "short", 2.00, 2.00)],
            "condor missing a wing": condor[:3],
            "unequal quantities": [leg(occ("N", "P", 100), 2, "short", 2.00, 2.00), leg(occ("N", "P", 95), 1, "long", 1.00, 1.00)],
            "debit vertical": [leg(occ("N", "C", 100), 1, "long", 5.00, 5.00), leg(occ("N", "C", 105), 1, "short", 3.00, 3.00)],
            "mixed expiries": [leg(occ("N", "P", 100), 1, "short", 2.00, 2.00), leg(occ("N", "P", 95, days=60), 1, "long", 1.00, 1.00)],
            "crossed short strikes": [leg(occ("N", "P", 110), 1, "short", 9.00, 9.00), leg(occ("N", "P", 100), 1, "long", 3.00, 3.00),
                                      leg(occ("N", "C", 90), 1, "short", 9.00, 9.00), leg(occ("N", "C", 100), 1, "long", 3.00, 3.00)],
        }
        for name, legs in never.items():
            self.assertEqual(f(legs, net_credit_of(legs)), 0.0, name)


# ── (e) GOOGL-like numbers ─────────────────────────────────────────────────────
def googl_condor(sp, lp, sc, lc):
    """short 310P $6.95 / long 300P $4.70 / short 355C $7.45 / long 365C $5.15, x2:
    gross $4,850, net credit $910, max loss 10 x 100 x 2 - 910 = $1,090."""
    root = "GOOGL"
    return root, [
        leg(occ(root, "P", 310), 2, "short", 6.95, sp),
        leg(occ(root, "P", 300), 2, "long", 4.70, lp),
        leg(occ(root, "C", 355), 2, "short", 7.45, sc),
        leg(occ(root, "C", 365), 2, "long", 5.15, lc),
    ]


class TestGooglLikeCondor(Case):
    def test_the_arithmetic(self):
        _, legs = googl_condor(6.95, 4.70, 7.45, 5.15)
        gross, net = gross_of(legs), net_credit_of(legs)
        max_loss = om._max_loss_from_legs(legs, net)
        self.assertAlmostEqual(gross, 4850.0, places=6)
        self.assertAlmostEqual(net, 910.0, places=6)
        self.assertEqual(max_loss, 1090.0)
        # old trigger: 2x gross = $9,700 — nine times what the condor can lose: never
        self.assertAlmostEqual(om.LOSS_LIMIT_MULTIPLIER * gross, 9700.0, places=6)
        self.assertGreater(om.LOSS_LIMIT_MULTIPLIER * gross, max_loss)
        # corrected trigger: 2x net credit = $1,820 — still beyond $1,090 (it collected 45.5 % of its wing)
        self.assertAlmostEqual(om.LOSS_LIMIT_MULTIPLIER * net, 1820.0, places=6)
        self.assertGreater(om.LOSS_LIMIT_MULTIPLIER * net, max_loss)
        # so the stop on this condor is the IC exit, at half the leg-derived max loss
        self.assertAlmostEqual(om.IC_MAX_LOSS_EXIT_PCT * max_loss, 545.0, places=6)

    def test_with_no_recorded_max_loss_the_ic_exit_acts_at_545(self):
        # call side decayed (+$400), puts gapped in; the legs' strategy is known (as after the
        # 11-02 adoption or a relabel) but no leg's state recorded max_loss
        root, below = googl_condor(15.90, 9.00, 0.40, 0.10)
        self.assertAlmostEqual(pnl_of(below), -530.0, places=6)
        result, _, broker = self.run_manager(below, state_for(below, "iron_condor", root))
        self.assertEqual(self.group_closes(result["actions"]), [], result["actions"])
        self.assertFalse(broker.posts)
        root, past = googl_condor(16.05, 9.00, 0.40, 0.10)
        self.assertAlmostEqual(pnl_of(past), -560.0, places=6)
        result, saved, broker = self.run_manager(past, state_for(past, "iron_condor", root))
        acts = self.group_closes(result["actions"])
        self.assertEqual([(a["type"], a["max_loss"], a["loss_pct_of_max"]) for a in acts], [("strategy_ic_max_loss_exit", 1090.0, 51.4)])
        self.assertIn("max loss $1090.00 (from the legs; none recorded)", acts[0]["reason"])
        self.assertEqual(len(broker.posts), 1)

    def test_at_the_full_max_loss_the_old_code_had_no_exit_at_all(self):
        # put spread worth its whole width, calls worthless: the condor's worst case, -$1,090
        root, worst = googl_condor(20.00, 10.00, 0.00, 0.00)
        self.assertAlmostEqual(pnl_of(worst), -1090.0, places=6)
        self.assertFalse(pre_fix_loss_limit(worst, "iron_condor"), "old Rule 3: -$1,090 on a $4,850 gross is -22 %")
        result, _, _ = self.run_manager(worst, state_for(worst, "iron_condor", root))
        acts = self.group_closes(result["actions"])
        self.assertEqual([(a["type"], a["max_loss"], a["loss_pct_of_max"]) for a in acts], [("strategy_ic_max_loss_exit", 1090.0, 100.0)])

    def test_googl_put_side_as_a_credit_vertical_closes_at_twice_its_net_credit(self):
        """short 310P $6.95 / long 300P $4.70, x2: net credit $450, gross $2,330, max loss $1,550.
        Old trigger 2x gross = $4,660 (never); the documented one is 2x net credit = $900."""
        def spread(sp):
            return [leg(occ("GOOGL", "P", 310), 2, "short", 6.95, sp), leg(occ("GOOGL", "P", 300), 2, "long", 4.70, 9.00)]
        legs = spread(15.80)  # loss $910
        self.assertAlmostEqual(net_credit_of(legs), 450.0, places=6)
        self.assertAlmostEqual(gross_of(legs), 2330.0, places=6)
        self.assertEqual(om._max_loss_from_legs(legs, net_credit_of(legs)), 1550.0)
        self.assertGreater(om.LOSS_LIMIT_MULTIPLIER * gross_of(legs), 1550.0)
        self.assertAlmostEqual(pnl_of(legs), -910.0, places=6)
        self.assertFalse(pre_fix_loss_limit(legs, "bull_put_credit_spread"))
        result, _, broker = self.run_manager(legs, state_for(legs, "bull_put_credit_spread", "GOOGL"))
        acts = self.group_closes(result["actions"])
        self.assertEqual([a["type"] for a in acts], ["strategy_loss_limit"], result["actions"])
        self.assertEqual(acts[0]["reason"], "LOSS LIMIT: loss $910 is 202% of the net credit $450 (limit 200%) on bull_put_credit_spread — cutting")
        held = spread(15.70)  # loss $890
        self.assertAlmostEqual(pnl_of(held), -890.0, places=6)
        result, _, broker = self.run_manager(held, state_for(held, "bull_put_credit_spread", "GOOGL"))
        self.assertEqual(self.group_closes(result["actions"]), [], result["actions"])
        self.assertFalse(broker.posts)


if __name__ == "__main__":
    unittest.main()
