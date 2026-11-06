"""
[RULE-REVIEW] 2026-11-06 — tax-loss harvesting (tax_filter.harvest_step;
refocus spec §6 "harvest losses to offset gains").

A losing stock position whose loss offsets gains already realized this year
is sold and its proceeds put into the sector SPDR ETF (correlated, not
substantially identical); the sold symbol and options on it are blocked from
re-entry for 30 days. Pins: the realized-gains snapshot's freshness, every
candidate rule, every gate of the step (switch, kill switch, shadow mode,
clock, per-scan limit), the order sequence (the replacement only after the
sale is accepted), the wash-sale re-entry blocks at all three entry gates,
and the scan wiring.
"""
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import tax_filter as tf

TODAY = date(2026, 11, 6)
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)
SECTORS = {"AMD": "Technology", "JPM": "Financials", "ZZZ": "Unknown", "XLK": "etf", "GLD": "etf"}


def _pos(sym, qty, price, upl, asset_class="us_equity", side="long"):
    return {"symbol": sym, "qty": str(qty), "current_price": str(price), "unrealized_pl": str(upl),
            "asset_class": asset_class, "side": side}


def _realized(st=6194.0, lt=0.0, loss_sales=None, as_of=NOW):
    return {"schema": "tax_realized/v1", "asOf": as_of.isoformat(), "year": as_of.year, "shortAfterNetting": st,
            "longAfterNetting": lt, "recentLossSales": loss_sales or {}, "recentBuys": {}}


class TempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self._p = patch.object(tf, "_data_dir", lambda: self.dir)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def write(self, name, body):
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(body, f)

    def read(self, name):
        with open(os.path.join(self.dir, name)) as f:
            return json.load(f)

    def log_rows(self):
        p = os.path.join(self.dir, tf.DECISIONS_FILE)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(x) for x in f if x.strip()]


class TestRealizedSnapshot(TempDir):
    def test_missing_stale_and_other_year_are_unknown(self):
        snap, why = tf.realized_snapshot(now=NOW)
        self.assertIsNone(snap)
        self.assertIn("no readable sleeve-ledger snapshot", why)
        self.write(tf.REALIZED_FILE, _realized(as_of=NOW - timedelta(hours=60)))
        snap, why = tf.realized_snapshot(now=NOW)
        self.assertIsNone(snap)
        self.assertIn("60 h old", why)
        self.write(tf.REALIZED_FILE, _realized(as_of=datetime(2025, 12, 31, 23, tzinfo=timezone.utc)))
        self.assertIsNone(tf.realized_snapshot(now=datetime(2026, 1, 1, 1, tzinfo=timezone.utc))[0], "last year's gains offset nothing now")

    def test_a_fresh_snapshot_is_used(self):
        self.write(tf.REALIZED_FILE, _realized(as_of=NOW - timedelta(hours=5)))
        snap, why = tf.realized_snapshot(now=NOW)
        self.assertEqual((snap["shortAfterNetting"], why), (6194.0, ""))


class TestCandidates(unittest.TestCase):
    def cands(self, positions, realized=None, **kw):
        return tf.harvest_candidates(positions, realized if realized is not None else _realized(), SECTORS.get,
                                     today=TODAY, cfg=tf.config({"HARVEST_ENABLED": True}), **kw)

    def test_a_loser_against_this_years_gains_is_swapped_into_its_sector_etf(self):
        (row,) = self.cands([_pos("AMD", 50, 100.0, -800.0)])
        self.assertEqual((row["action"], row["replacement"]), ("harvest", "XLK"))
        self.assertEqual(row["taxSaving"], round(800.0 * 0.24, 2))
        self.assertEqual(row["washSaleUntil"], (TODAY + timedelta(days=30)).isoformat())
        self.assertTrue(row["reason"].startswith("Harvest: sell 50 AMD at a $800 loss"), row["reason"])

    def test_long_term_gains_absorb_what_short_term_gains_cannot(self):
        (row,) = self.cands([_pos("AMD", 50, 100.0, -800.0)], realized=_realized(st=300.0, lt=1000.0))
        self.assertAlmostEqual(row["taxSaving"], 300 * 0.24 + 500 * 0.15, places=2)

    def test_every_reason_not_to_harvest(self):
        cases = [
            ([_pos("AMD", 50, 100.0, -800.0)], _realized(st=0.0), {}, "no gains realized this year"),
            ([_pos("GLD", 10, 370.0, -900.0)], _realized(), {"floor_tickers": {"GLD"}}, "floor-managed"),
            ([_pos("ZZZ", 50, 100.0, -800.0)], _realized(), {}, "no replacement"),
            ([_pos("XLK", 50, 100.0, -800.0)], _realized(), {}, "no replacement"),
            ([_pos("AMD", 50, 100.0, -800.0), _pos("AMD261218P00090000", -1, 2.0, -10.0, "us_option")], _realized(), {},
             "an open option on AMD"),
            ([_pos("AMD", 50, 100.0, -800.0)], _realized(loss_sales={"XLK": "2026-10-20T15:00:00Z"}), {}, "XLK was sold at a loss"),
            ([_pos("AMD", 50, 100.0, -800.0)], _realized(), {"open_orders": {"AMD"}}, "an order is already working"),
            ([_pos("AMD", 50, 100.0, -60.0)], _realized(), {}, "under the $20 net bar"),
        ]
        for positions, realized, kw, expect in cases:
            with self.subTest(expect=expect):
                rows = [r for r in self.cands(positions, realized=realized, **kw) if r["ticker"] in ("AMD", "GLD", "ZZZ", "XLK")]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["action"], "none")
                self.assertIn(expect, rows[0]["reason"])

    def test_unknown_realized_gains_are_never_guessed(self):
        (row,) = tf.harvest_candidates([_pos("AMD", 50, 100.0, -800.0)], None, SECTORS.get, today=TODAY,
                                       realized_why="this year's realized gains are unknown (x)")
        self.assertEqual(row["action"], "none")
        self.assertIn("realized gains are unknown", row["reason"])

    def test_winners_shorts_and_options_are_not_candidates(self):
        rows = self.cands([_pos("AMD", 50, 100.0, 800.0), _pos("JPM", 10, 100.0, -50.0, side="short"),
                           _pos("AMD261218P00090000", -1, 2.0, -10.0, "us_option")])
        self.assertEqual(rows, [])


class FakeBroker:
    def __init__(self, sell_ok=True, buy_ok=True):
        self.orders, self.sell_ok, self.buy_ok = [], sell_ok, buy_ok

    def order(self, symbol, qty, side, px):
        self.orders.append((symbol, qty, side, px))
        ok = self.sell_ok if side == "sell" else self.buy_ok
        return {"ok": ok, "order_id": f"o{len(self.orders)}" if ok else None, "error": None if ok else "rejected"}


class TestHarvestStep(TempDir):
    def deps(self, broker, positions=None, killed=False, clock=(True, "open")):
        return {"positions": lambda: positions if positions is not None else [_pos("AMD", 50, 100.0, -800.0)],
                "sector_of": SECTORS.get, "quote": lambda s: (99.9, 100.1) if s == "AMD" else (199.0, 200.0),
                "order": broker.order, "clock": lambda: clock, "open_orders": lambda: set(),
                "floor_tickers": {"GLD"}, "killed": killed}

    def run_step(self, broker, cfg=None, shadowed=False, **kw):
        self.write(tf.REALIZED_FILE, _realized(as_of=datetime.now(timezone.utc)))
        with patch("entry_shadow.is_shadowed", return_value=shadowed):
            return tf.harvest_step(self.deps(broker, **kw), cfg=cfg or tf.config({"HARVEST_ENABLED": True}), today=TODAY)

    def test_a_clean_harvest_sells_then_buys_and_blocks_re_entry(self):
        b = FakeBroker()
        out = self.run_step(b)
        self.assertIsNone(out["error"])
        self.assertEqual(b.orders, [("AMD", 50, "sell", 99.9), ("XLK", int(50 * 99.9 // 200.0), "buy", 200.0)])
        self.assertEqual(out["candidates"][0]["action"], "harvested")
        block = tf.wash_block("AMD", today=TODAY)
        self.assertEqual(block, {"soldOn": TODAY.isoformat(), "washSaleUntil": (TODAY + timedelta(days=30)).isoformat(),
                                 "replacement": "XLK"})
        self.assertIsNone(tf.wash_block("AMD", today=TODAY + timedelta(days=31)), "day 31: the buy-back no longer washes")
        review = self.read(tf.HARVEST_REVIEW_FILE)
        self.assertEqual(review["harvest"][0]["ticker"], "AMD")
        self.assertEqual(self.log_rows()[-1]["kind"], "harvest")

    def test_every_gate_downgrades_the_swap_to_a_recommendation(self):
        for label, kw in (("switch", {"cfg": tf.config({"HARVEST_ENABLED": False})}), ("kill", {"killed": True}),
                          ("shadow", {"shadowed": True}), ("clock", {"clock": (False, "broker clock: market closed")})):
            with self.subTest(gate=label):
                b = FakeBroker()
                out = self.run_step(b, **kw)
                self.assertEqual(b.orders, [])
                self.assertEqual(out["candidates"][0]["action"], "recommend")
                self.assertTrue(out["candidates"][0]["reason"].startswith("Recommended (not acted on: "))
                self.assertIsNone(tf.wash_block("AMD", today=TODAY))

    def test_a_refused_sale_buys_nothing_and_blocks_nothing(self):
        b = FakeBroker(sell_ok=False)
        out = self.run_step(b)
        self.assertEqual([o[2] for o in b.orders], ["sell"])
        self.assertEqual(out["candidates"][0]["action"], "failed")
        self.assertIsNone(tf.wash_block("AMD", today=TODAY))

    def test_a_refused_replacement_still_records_the_harvest(self):
        b = FakeBroker(buy_ok=False)
        out = self.run_step(b)
        self.assertEqual(out["candidates"][0]["action"], "harvested")
        self.assertIn("proceeds stay in cash", out["candidates"][0]["reason"])
        self.assertIsNotNone(tf.wash_block("AMD", today=TODAY))

    def test_the_per_scan_limit(self):
        b = FakeBroker()
        positions = [_pos("AMD", 50, 100.0, -800.0), _pos("JPM", 50, 100.0, -900.0)]
        out = self.run_step(b, cfg=tf.config({"HARVEST_ENABLED": True, "HARVEST_MAX_PER_SCAN": 1}), positions=positions)
        self.assertEqual(sorted(r["action"] for r in out["candidates"]), ["harvested", "recommend"])

    def test_a_broken_dependency_is_reported_never_raised(self):
        def boom():
            raise RuntimeError("positions down")
        out = tf.harvest_step({"positions": boom}, today=TODAY)
        self.assertIn("RuntimeError", out["error"])


class TestWashSaleEntryBlocks(TempDir):
    def setUp(self):
        super().setUp()
        self.write(tf.HARVEST_STATE_FILE, {"AMD": {"soldOn": TODAY.isoformat(), "replacement": "XLK",
                                                   "washSaleUntil": (date.today() + timedelta(days=10)).isoformat()}})

    def test_the_scan_gate_refuses_the_harvested_symbol_even_with_the_filter_off(self):
        row = tf.gate_scan_trade({"ticker": "AMD", "side": "buy", "instrument": "stock"}, None,
                                 sizing_input={"ticker": "AMD"}, cfg=tf.config({"ENABLED": False}))
        self.assertFalse(row["allowed"])
        self.assertTrue(row["washSale"])
        self.assertIn("wash sale", row["reason"])

    def test_option_opens_and_tier_buys_on_it_are_refused(self):
        csp = {"strike": 90.0, "qty": 1, "limit_price": 1.0, "cash_required": 9000.0, "days_to_expiry": 30, "side": "sell"}
        self.assertFalse(tf.gate_option_contract(csp, "AMD", "sell_cash_secured_put", shadow=False)["allowed"])
        self.assertFalse(tf.gate_tier_buy("AMD", "trend_long")["allowed"])
        self.assertTrue(tf.gate_tier_buy("SSO", "trend_long", stats={"by_strategy": {"etf": {"win_rate": 0.6, "avg_win": 4, "avg_loss": 2, "trades": 9}}})["allowed"])


class TestWiring(unittest.TestCase):
    def test_scan_market_runs_the_review_after_the_tier_block_and_returns_it(self):
        import inspect
        import bot_engine
        src = inspect.getsource(bot_engine._scan_market_inner)
        at = src.index("_tax_harvest = _tf_hv.harvest_step({")
        self.assertLess(src.index("if _HAS_TIERED:"), at)
        self.assertLess(at, src.index('_timing_log("tier_engine_complete")'))
        window = src[at:at + 600]
        for dep in ('"order": _floor_vm_order', '"clock": _floor_vm_broker_clock', '"open_orders": _open_order_symbols',
                    '"floor_tickers": FLOOR_AND_LEG_TICKERS', '"killed": kill_status is None or bool(kill_status.get("killed"))'):
            self.assertIn(dep, window)
        self.assertIn('"tax_harvest": _tax_harvest,', src)
        self.assertNotIn("/v2/orders", open(tf.__file__, encoding="utf-8").read(), "orders go through bot_engine's helper")


if __name__ == "__main__":
    unittest.main()
