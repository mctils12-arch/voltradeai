"""
ENTRY SHADOW MODE (human directive 2026-10-28, paper account): "move the
options sleeve and stock entries to shadow mode. Keep the floor basket and all
exits live." — entry_shadow.py + server/entryShadow.ts.

Pinned here:
  1. flag: BASE_CONFIG["SHADOW_ENTRY_SLEEVES"] = "stocks,options"; the env
     var VOLTRADE_SHADOW_ENTRIES wins whenever it is PRESENT ("off"/"0"/""/
     "none" = nothing shadowed); the Node mirror's default and decision
     constants equal the Python ones.
  2. every Python entry site, flag ON: no order call + a shadow record;
     flag OFF (env "off"): the order call as before —
     evaluate_and_execute, the third leg's SVXY buy, the convexity put open,
     the bot.ts inline options scripts (scanner-direct run for real), rolls
     (close-only, re-open logged).
  3. exits and the floor stay live with the flag ON: SVXY spike exit, the
     options manager's close, the floor basket buy.
  4. shadow_portfolio.log_shadow_entries: relabel the fresh "taken" record
     instead of duplicating it, append when there is none, once per
     sleeve+ticker+day ("shadow_mode_repeat" after that).
No network, no orders: every requests call is mocked.
"""
import inspect
import json
import os
import re
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import entry_shadow  # noqa: E402
import shadow_portfolio  # noqa: E402
from system_config import BASE_CONFIG  # noqa: E402

ON = {"VOLTRADE_SHADOW_ENTRIES": "stocks,options"}
OFF = {"VOLTRADE_SHADOW_ENTRIES": "off"}
OFF_VALUES = ("off", "0", "", "none", " OFF ", "false")


class _TempShadowLog(unittest.TestCase):
    """Every test writes to its own shadow log and starts with an empty
    in-process once-per-day memory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_entry_shadow_")
        self.log_path = os.path.join(self.tmp, "voltrade_shadow_candidates.json")
        for p in (patch.object(shadow_portfolio, "SHADOW_LOG_PATH", self.log_path),
                  patch.object(shadow_portfolio, "SHADOW_LOCK_PATH", self.log_path + ".lock")):
            p.start()
            self.addCleanup(p.stop)
        entry_shadow._SEEN.update(day=None, keys=set())
        self.addCleanup(lambda: entry_shadow._SEEN.update(day=None, keys=set()))

    def records(self):
        try:
            with open(self.log_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def env(self, values):
        p = patch.dict(os.environ, values)
        p.start()
        self.addCleanup(p.stop)


# ── 1. flag ──────────────────────────────────────────────────────────────────
class TestFlag(unittest.TestCase):
    def test_base_config_default_shadows_stocks_and_options(self):
        self.assertEqual(BASE_CONFIG["SHADOW_ENTRY_SLEEVES"], "stocks,options")
        self.assertEqual(entry_shadow.shadowed_sleeves({}), {"stocks", "options"})
        self.assertTrue(entry_shadow.is_shadowed("stocks", {}))
        self.assertTrue(entry_shadow.is_shadowed("options", {}))
        self.assertEqual(entry_shadow.status({}),
                         {"stocks": True, "options": True, "source": "config", "value": "stocks,options"})

    def test_present_env_wins_including_off_values(self):
        for v in OFF_VALUES:
            with self.subTest(value=v):
                env = {"VOLTRADE_SHADOW_ENTRIES": v}
                self.assertEqual(entry_shadow.shadowed_sleeves(env), set())
                self.assertFalse(entry_shadow.is_shadowed("stocks", env))
                self.assertFalse(entry_shadow.is_shadowed("options", env))
                self.assertEqual(entry_shadow.status(env)["source"], "env")

    def test_partial_values_narrow(self):
        self.assertEqual(entry_shadow.shadowed_sleeves({"VOLTRADE_SHADOW_ENTRIES": "stocks"}), {"stocks"})
        self.assertEqual(entry_shadow.shadowed_sleeves({"VOLTRADE_SHADOW_ENTRIES": "options"}), {"options"})
        self.assertEqual(entry_shadow.shadowed_sleeves({"VOLTRADE_SHADOW_ENTRIES": "Options, Stocks"}),
                         {"stocks", "options"})
        self.assertEqual(entry_shadow.shadowed_sleeves({"VOLTRADE_SHADOW_ENTRIES": "all"}), {"stocks", "options"})
        self.assertEqual(entry_shadow.shadowed_sleeves({"VOLTRADE_SHADOW_ENTRIES": "floor"}), set(),
                         "the floor is never a shadowable sleeve")

    def test_absent_env_and_no_config_is_live(self):
        self.assertEqual(entry_shadow.shadowed_sleeves({}, {}), set())


# ── Node mirror parity ───────────────────────────────────────────────────────
class TestNodeMirror(unittest.TestCase):
    src = open(os.path.join(ROOT, "server", "entryShadow.ts")).read()

    def test_node_default_equals_base_config(self):
        block = re.search(r"ENTRY_SHADOW_DEFAULTS[^=]*=\s*Object\.freeze\(\{(.*?)\}\)", self.src, re.S)
        self.assertIsNotNone(block, "ENTRY_SHADOW_DEFAULTS table not found in server/entryShadow.ts")
        ts = dict(re.findall(r"(VOLTRADE_[A-Z_]+):\s*\"([^\"]*)\"", block.group(1)))
        self.assertEqual(set(ts), {entry_shadow.ENV_VAR})
        self.assertEqual(ts[entry_shadow.ENV_VAR], entry_shadow.flag_raw({})[0])

    def test_node_reader_treats_a_present_env_var_as_authoritative(self):
        self.assertIn('if (v !== undefined && v !== null) return { value: String(v), source: "env" };', self.src)
        self.assertIn('const OFF_VALUES = new Set(["", "0", "off", "none", "false"]);', self.src)
        self.assertEqual(set(entry_shadow.OFF_VALUES), {"", "0", "off", "none", "false"})

    def test_decision_and_status_constants_match(self):
        for name, py in (("SHADOW_ENTRY_DECISION", shadow_portfolio.SHADOW_ENTRY_DECISION),
                         ("SHADOW_REPEAT_DECISION", shadow_portfolio.SHADOW_REPEAT_DECISION),
                         ("SHADOW_ORDER_STATUS", entry_shadow.SHADOW_ORDER_STATUS)):
            m = re.search(rf'export const {name} = "([^"]+)";', self.src)
            self.assertIsNotNone(m, name)
            self.assertEqual(m.group(1), py, name)

    def test_ledger_floor_set_matches_bot_ts(self):
        bot = open(os.path.join(ROOT, "server", "bot.ts")).read()
        m = re.search(r"const FLOOR_AND_LEG_TICKERS = new Set\(\[([^\]]*)\]\)", bot)
        self.assertIsNotNone(m)
        self.assertEqual(set(re.findall(r'"([A-Z]+)"', m.group(1))), entry_shadow.floor_ledger_tickers())
        self.assertEqual(entry_shadow.ledger_sleeve("SMH"), "floor")
        self.assertEqual(entry_shadow.ledger_sleeve("NVDA"), "stocks")
        self.assertEqual(entry_shadow.ledger_sleeve("NVDA261120P00120000"), "options")

    def test_daemon_route_registered(self):
        daemon = open(os.path.join(ROOT, "voltrade_daemon.py")).read()
        self.assertIn('"entry_shadow_record": ("entry_shadow", "record_node_entries")', daemon)


# ── 4. the shadow log ────────────────────────────────────────────────────────
class TestShadowLog(_TempShadowLog):
    def _taken(self, ticker="NVDA", age_s=60):
        ts = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
        rec = shadow_portfolio._build_record(ticker, {"rsi": 55.0}, 72.0, "taken", "score ok", 120.0)
        rec["timestamp"] = ts
        with open(self.log_path, "w") as f:
            json.dump([rec], f)

    def test_relabels_the_fresh_taken_record_instead_of_duplicating(self):
        self._taken()
        res = entry_shadow.record_entries([entry_shadow.build_entry(
            sleeve="stocks", ticker="NVDA", source="executeTrades.stock", qty=10, price=121.0, score=72,
            relabel=True)])
        self.assertEqual(res["recorded"], 1)
        recs = self.records()
        self.assertEqual(len(recs), 1, "one decision = one record")
        self.assertEqual(recs[0]["decision"], "shadow_mode_entry")
        self.assertEqual(recs[0]["features"], {"rsi": 55.0}, "features kept for the ML blend it was already in")
        se = recs[0]["shadow_entry"]
        self.assertEqual((se["sleeve"], se["source"], se["qty"], se["ledger_sleeve"]),
                         ("stocks", "executeTrades.stock", 10.0, "stocks"))
        self.assertEqual(se["day"], entry_shadow.et_day())

    def test_appends_a_measurable_record_when_none_exists(self):
        row = entry_shadow.record_entry(sleeve="stocks", ticker="SVXY", source="bot_engine._run_third_leg",
                                        qty=100, price=50.0)
        self.assertTrue(row["new"])
        recs = self.records()
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r["ticker"], r["decision"], r["entry_price"]), ("SVXY", "shadow_mode_entry", 50.0))
        self.assertEqual(r["features"], {}, "never enters load_shadow_data's ML blend")
        self.assertEqual(r["outcomes"], {"+5d": None, "+10d": None, "+20d": None},
                         "the +5/+10/+20 backfill picks it up (entry_price > 0)")
        self.assertEqual(r["shadow_entry"]["ledger_sleeve"], "floor",
                         "the P&L ledger books SVXY as floor — the record says so")

    def test_once_per_sleeve_ticker_day_then_repeat(self):
        self._taken()
        e = dict(sleeve="stocks", ticker="NVDA", source="executeTrades.stock", qty=10, price=121.0, relabel=True)
        entry_shadow.record_entry(**e)
        entry_shadow._SEEN.update(day=None, keys=set())  # a restarted process: the log is the dedupe
        recs = self.records()
        nxt = shadow_portfolio._build_record("NVDA", {}, 73.0, "taken", "score ok", 122.0)
        with open(self.log_path, "w") as f:
            json.dump(recs + [nxt], f)
        row = entry_shadow.record_entry(**e)
        self.assertFalse(row["new"])
        decisions = [r["decision"] for r in self.records()]
        self.assertEqual(decisions, ["shadow_mode_entry", "shadow_mode_repeat"])
        # a different sleeve on the same ticker is its own entry
        self.assertTrue(entry_shadow.record_entry(sleeve="options", ticker="NVDA", source="x", price=1.0)["new"])

    def test_record_failure_never_raises(self):
        with patch.object(shadow_portfolio, "_load_shadow_log", side_effect=RuntimeError("disk")):
            res = entry_shadow.record_entries([entry_shadow.build_entry(sleeve="stocks", ticker="A", source="s")])
        self.assertIn("error", res)


# ── 2. Python entry sites ────────────────────────────────────────────────────
CSP = {"occ_symbol": "AMD261204P00140000", "strike": 140.0, "expiry": "2026-12-04", "option_type": "put",
       "side": "sell", "qty": 1, "limit_price": 3.2, "cash_required": 14000.0, "max_loss": 13680.0}


class TestEvaluateAndExecute(_TempShadowLog):
    def _run(self, env):
        import options_execution as oe
        self.env(env)
        submit = MagicMock(return_value={"status": "submitted", "order_id": "o1"})
        with patch.object(oe, "should_use_options", return_value={
                "use_options": True, "strategy": "sell_cash_secured_put", "edge_pct": 6.0, "reason": "VRP 6"}), \
             patch.object(oe, "select_contract", return_value=dict(CSP)), \
             patch.object(oe, "submit_options_order", submit), \
             patch("options_manager.register_options_entry") as reg:
            out = oe.evaluate_and_execute({"ticker": "AMD", "price": 150.0, "deep_score": 71}, 100000, [])
        return out, submit, reg

    def test_flag_on_no_order_and_a_shadow_record(self):
        out, submit, reg = self._run(ON)
        submit.assert_not_called()
        reg.assert_not_called()
        self.assertEqual(out["instrument"], "options", "bot.ts must not fall back to a stock buy")
        self.assertEqual(out["order"]["status"], "shadowed")
        recs = self.records()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["ticker"], "AMD")
        self.assertEqual(recs[0]["entry_price"], 150.0)
        c = recs[0]["shadow_entry"]["contract"]
        self.assertEqual((c["occ_symbol"], c["strike"], c["expiry"], c["limit_price"]),
                         ("AMD261204P00140000", 140.0, "2026-12-04", 3.2))
        self.assertEqual(recs[0]["shadow_entry"]["source"], "options_execution.evaluate_and_execute")

    def test_env_off_submits_as_before(self):
        out, submit, reg = self._run(OFF)
        submit.assert_called_once()
        self.assertEqual(out["order"]["status"], "submitted")
        self.assertEqual(self.records(), [])

    def test_stocks_only_leaves_options_live(self):
        _out, submit, _reg = self._run({"VOLTRADE_SHADOW_ENTRIES": "stocks"})
        submit.assert_called_once()


def _resp(payload, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = payload
    return m


class TestThirdLeg(_TempShadowLog):
    """The SVXY buy is a stocks-sleeve entry; its protective exits stay live."""

    def setUp(self):
        super().setUp()
        import bot_engine
        self.be = bot_engine
        p = patch.object(bot_engine, "DATA_DIR", self.tmp)
        p.start()
        self.addCleanup(p.stop)

    def _run(self, env, held_plpc=None, zone="BAND", vxx=1.10):
        self.env(env)
        positions = [] if held_plpc is None else [
            {"symbol": "SVXY", "qty": "300", "market_value": "15000", "unrealized_plpc": str(held_plpc)}]

        def fake_get(url, *a, **kw):
            if url.endswith("/v2/account"):
                return _resp({"equity": "100000"})
            if url.endswith("/v2/positions"):
                return _resp(positions)
            if url.endswith("/v2/orders"):
                return _resp([])
            if url.endswith("/v2/stocks/bars"):
                return _resp({"bars": {"VXX": [{"c": c} for c in (30, 29, 28, 27, 26, 25)]}})
            if url.endswith("/v2/stocks/snapshots"):
                return _resp({"SVXY": {"latestTrade": {"p": 50.0}}})
            return _resp({})

        dwell = {"zone": zone, "buy_ok": zone == "BAND", "spike": vxx > 1.25, "recovery_exit_ok": False}
        post = MagicMock(return_value=_resp({"id": "o"}))
        with patch("requests.get", side_effect=fake_get), patch("requests.post", post), \
             patch.object(self.be, "_resolve_third_leg_dwell", return_value=dwell):
            res = self.be._run_third_leg({"vxx_ratio": vxx, "spy_vs_ma50": 1.0, "spy_below_200_days": 0,
                                          "spy_above_200d": True})
        return res, [c.kwargs.get("json") or {} for c in post.call_args_list]

    def test_flag_on_no_svxy_buy_and_a_shadow_record(self):
        res, posts = self._run(ON)
        self.assertEqual([p for p in posts if p.get("side") == "buy"], [])
        shadow = [a for a in res["actions"] if a.get("type") == "shadow_entry"]
        self.assertEqual(len(shadow), 1)
        self.assertTrue(shadow[0]["new"])
        recs = self.records()
        self.assertEqual((recs[0]["ticker"], recs[0]["entry_price"]), ("SVXY", 50.0))
        self.assertEqual(recs[0]["shadow_entry"]["sleeve"], "stocks")

    def test_env_off_buys_as_before(self):
        res, posts = self._run(OFF)
        buys = [p for p in posts if p.get("side") == "buy" and p.get("symbol") == "SVXY"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(self.records(), [])

    def test_flag_on_spike_exit_still_sells(self):
        res, posts = self._run(ON, held_plpc=-0.03, zone="SPIKE", vxx=1.30)
        sells = [p for p in posts if p.get("side") == "sell" and p.get("symbol") == "SVXY"]
        self.assertEqual(len(sells), 1, "exits are never shadowed")


class TestConvexityOverlay(_TempShadowLog):
    """The QQQ-put buy-to-open is an options entry; its roll close stays live."""

    def _run(self, env, existing_put_dte=None):
        import bot_engine
        self.env(env)
        today = datetime.now().date()
        positions = []
        if existing_put_dte is not None:
            exp = (today + timedelta(days=existing_put_dte)).strftime("%y%m%d")
            positions.append({"symbol": f"QQQ{exp}P00400000", "qty": "2"})
        exp_new = (today + timedelta(days=60)).strftime("%Y-%m-%d")
        occ_new = f"QQQ{(today + timedelta(days=60)).strftime('%y%m%d')}P00400000"

        def fake_get(url, *a, **kw):
            if url.endswith("/v2/positions/SQQQ"):
                return _resp({}, 404)
            if url.endswith("/v2/account"):
                return _resp({"equity": "100000"})
            if url.endswith("/v2/positions"):
                return _resp(positions)
            if url.endswith("/v2/orders"):
                return _resp([])
            if url.endswith("/v2/stocks/snapshots"):
                return _resp({"QQQ": {"latestTrade": {"p": 500.0}}})
            if url.endswith("/v2/options/contracts"):
                return _resp({"option_contracts": [{"symbol": occ_new, "strike_price": "400",
                                                    "expiration_date": exp_new}]})
            if "options/quotes/latest" in url:
                return _resp({"quotes": {occ_new: {"bp": 1.0, "ap": 1.2}}})
            return _resp({})

        post = MagicMock(return_value=_resp({"id": "o"}))
        import storage_config
        with patch("requests.get", side_effect=fake_get), patch("requests.post", post), \
             patch.object(storage_config, "DATA_DIR", self.tmp), \
             patch.object(bot_engine, "should_escalate_hedge", return_value=False), \
             patch.object(bot_engine, "data_feed", return_value="iex"):
            res = bot_engine._run_convexity_overlay({"vxx_ratio": 1.0, "spy_vs_ma50": 1.0})
        return res, [c.kwargs.get("json") or {} for c in post.call_args_list], occ_new

    def test_flag_on_no_put_open_and_a_shadow_record(self):
        res, posts, occ = self._run(ON)
        self.assertEqual([p for p in posts if p.get("side") == "buy_to_open"], [])
        recs = self.records()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["ticker"], "QQQ")
        self.assertEqual(recs[0]["shadow_entry"]["contract"]["occ_symbol"], occ)
        self.assertEqual(recs[0]["shadow_entry"]["sleeve"], "options")

    def test_env_off_opens_as_before(self):
        _res, posts, occ = self._run(OFF)
        self.assertEqual([p.get("symbol") for p in posts if p.get("side") == "buy_to_open"], [occ])

    def test_flag_on_roll_close_still_sells(self):
        _res, posts, _occ = self._run(ON, existing_put_dte=10)
        self.assertEqual(len([p for p in posts if p.get("side") == "sell_to_close"]), 1, "the close is an exit")
        self.assertEqual([p for p in posts if p.get("side") == "buy_to_open"], [])


class TestRollsCloseOnly(_TempShadowLog):
    """A roll's open half is an entry: under shadow the manager closes (live)
    and the would-be re-open is logged; scan_market wires it."""

    def test_skip_detail_reaches_the_close_reason_and_becomes_a_shadow_entry(self):
        import options_manager
        with patch("options_manager._attempt_roll", side_effect=AssertionError("must not roll")):
            r = options_manager._roll_or_skip(False, "X", 1, "sell", {}, 1e5, options_manager.ROLL_SKIPPED_SHADOW_DETAIL)
        self.assertEqual(r["detail"], options_manager.ROLL_SKIPPED_SHADOW_DETAIL)
        actions = [{"action": "CLOSE", "ticker": "AMD", "occ_symbol": "AMD261120P00140000", "type": "dte_close",
                    "reason": f"21 DTE EXIT: 15 DTE — roll failed, closing ({r['detail']})"},
                   {"action": "CLOSE", "ticker": "TSLA", "type": "profit_target", "reason": "50% profit"}]
        with patch.object(entry_shadow, "_underlying_price", return_value=150.0):
            entries = entry_shadow.roll_skip_entries(actions)
        self.assertEqual([e["ticker"] for e in entries], ["AMD"], "only the skipped roll is an entry")
        self.assertEqual(entries[0]["details"]["contract"]["rolled_from"], "AMD261120P00140000")
        self.assertEqual(entries[0]["price"], 150.0)

    def test_scan_market_disables_rolls_only_under_options_shadow(self):
        import bot_engine
        src = inspect.getsource(bot_engine._scan_market_inner)
        at = src.index('_es_options = _es.is_shadowed("options")')
        win = src[at:at + 1600]
        self.assertIn("manage_options_positions(equity=portfolio_value, allow_rolls=False,", win)
        self.assertIn("roll_skip_detail=ROLL_SKIPPED_SHADOW_DETAIL", win)
        self.assertIn("_es.roll_skip_entries(", win)
        self.assertIn("options_mgmt_result = manage_options_positions(equity=portfolio_value)", win,
                      "flag off: the manager runs exactly as before")
        self.assertIn('"entry_shadow": _scan_entry_shadow(', inspect.getsource(bot_engine._scan_market_inner))


class TestBotTsInlineScripts(_TempShadowLog):
    """The bot.ts SELL_CSP / BUY_PUT / scanner-direct scripts gate before
    submit_options_order; the scanner-direct one is run for real."""
    bot = open(os.path.join(ROOT, "server", "bot.ts"), encoding="utf-8").read()

    def _script(self, anchor):
        at = self.bot.index(anchor)
        start = self.bot.index('python3 -c "', at) + len('python3 -c "')
        return self.bot[start:self.bot.index('"`', start)]

    def test_every_options_script_checks_the_flag_before_submitting(self):
        for anchor, source in (("const cspTmpPath", "tier_dispatcher.SELL_CSP"),
                               ("const hedgeTmpPath", "tier_dispatcher.BUY_PUT"),
                               ("SCANNER PATH: Direct to select_contract + submit", "executeTrades.options_scanner")):
            with self.subTest(anchor=anchor):
                code = self._script(anchor)
                gate = code.index("if entry_shadow.is_shadowed('options'):")
                self.assertIn(f"source='{source}'", code[gate:gate + 400])
                self.assertLess(gate, code.index("submit_options_order(contract)", gate))
                self.assertLess(code.index("contract = select_contract("), gate, "contract chosen first (logged)")

    def _run_scanner(self, env, submit):
        self.env(env)
        payload = os.path.join(self.tmp, "payload.json")
        with open(payload, "w") as f:
            json.dump({"ticker": "AMD", "strategy": "sell_cash_secured_put", "price": 150.0,
                       "equity": 100000, "setup": "vxx_panic_put_sale"}, f)
        code = self._script("SCANNER PATH: Direct to select_contract + submit").replace("${scanTmpPath}", payload)
        printed = []
        with patch.dict(sys.modules, {"options_execution": types.SimpleNamespace(
                select_contract=lambda *a, **k: dict(CSP), submit_options_order=submit)}), \
             patch("options_manager.register_options_entry") as reg, \
             patch("options_manager.merge_options_meta"), \
             patch("builtins.print", side_effect=lambda s: printed.append(s)):
            exec(compile(code, "bot.ts:scanner-direct", "exec"), {"__name__": "__bot_ts__"})
        return json.loads(printed[-1]), reg

    def test_scanner_direct_flag_on_logs_and_does_not_submit(self):
        submit = MagicMock(return_value={"status": "pending_new"})
        out, reg = self._run_scanner(ON, submit)
        submit.assert_not_called()
        reg.assert_not_called()
        self.assertEqual(out["order"]["status"], "shadowed")
        self.assertEqual(self.records()[0]["shadow_entry"]["source"], "executeTrades.options_scanner")

    def test_scanner_direct_env_off_submits(self):
        submit = MagicMock(return_value={"status": "pending_new"})
        out, reg = self._run_scanner(OFF, submit)
        submit.assert_called_once()
        self.assertTrue(reg.called)
        self.assertEqual(self.records(), [])


# ── 3. the floor stays live ──────────────────────────────────────────────────
class TestFloorStaysLive(unittest.TestCase):
    def test_floor_functions_never_consult_the_switch(self):
        import bot_engine
        for fn in (bot_engine._manage_spy_floor, bot_engine._manage_defensive_floor):
            self.assertNotIn("entry_shadow", inspect.getsource(fn), fn.__name__)

    def test_floor_basket_buy_posts_with_the_flag_on(self):
        import bot_engine
        cfg = {"FLOOR_TICKER": "QQQ", "FLOOR_BASKET_ENABLED": True, "FLOOR_BASKET": {"SMH": 1.0},
               "FLOOR_BASKET_DRIFT_THRESHOLD": 0.03, "SPY_FLOOR_BULL": 0.5,
               "TREND_EXIT_DEATH_CROSS_CAP": 0.20, "TREND_EXIT_EARLY_WARNING_CAP": 0.50}
        snap = MagicMock()
        snap.json.return_value = {"latestTrade": {"p": 100.0}}
        tmp = tempfile.mkdtemp(prefix="vt_floor_shadow_")
        macro = {"qqq_price": 500.0, "qqq_ma50": 480.0, "qqq_ma200": 460.0, "vxx_ratio": 1.0,
                 "spy_vs_ma50": 1.0, "spy_below_200_days": 0, "spy_above_200d": True}
        with patch.dict(os.environ, ON), \
             patch.object(bot_engine, "DATA_DIR", tmp), \
             patch.object(bot_engine, "data_feed", return_value="iex"), \
             patch("system_config.get_market_regime", return_value="BULL"), \
             patch.object(bot_engine, "_is_regular_hours", return_value=True), \
             patch("system_config.BASE_CONFIG", cfg), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=[]), \
             patch.object(bot_engine.requests, "get", return_value=snap), \
             patch.object(bot_engine.requests, "post") as post:
            self.assertTrue(entry_shadow.is_shadowed("stocks"))
            bot_engine._manage_spy_floor(macro)
        buys = [c.kwargs.get("json", {}) for c in post.call_args_list
                if (c.kwargs.get("json") or {}).get("side") == "buy"]
        self.assertEqual([b.get("symbol") for b in buys], ["SMH"], "the floor basket buy is never shadowed")


if __name__ == "__main__":
    unittest.main()
