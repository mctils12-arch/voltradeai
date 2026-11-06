#!/usr/bin/env python3
"""
INSTRUMENT COUNTERFACTUAL (2026-11-06; human refocus spec §3 step 7 "log why the
rejected instrument lost"): instrument_selector.option_alternative +
instrument_counterfactual.py (the nightly outcome job) + the decision journal's
rejected[].would_be.

Pins:
- the option alternative the selector records for every decision: why options
  lost in plain English (not scored / ruled out / no strategy / routed to stock
  / outscored / chosen), the score BEFORE a rule-out zeroed it, the ATM contract
  of a chain it was handed (the lower strike on a tie) else null + why, and the
  ATM proxy spec (a bought call, a put for a bearish idea, 30-45 DTE);
- select_instrument carries it, without a network call of its own;
- the leg math on fixtures: shares, a bought call on premium, a cash-secured put
  on capital, a leg expiring inside the horizon (intrinsic), an entry day with
  no option trade (unpriced, never a later price), a stale exit mark;
- the plain-English line, exactly;
- backfill() end to end with injected fetchers: the proxy resolves once (entry-
  day facts only) and is reused, horizons settle as trading days pass, an
  options entry vs its stock alternative, give-up after GIVE_UP_DAYS, a fetch
  error never raises, the off switch;
- the journal keeps would_be (normalize and the row cut);
- parity with server/instrumentCounterfactual.ts and system_config; the wiring.

Run: bash pytest_locked.sh -q test_instrument_counterfactual.py
"""
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import decision_journal as dj  # noqa: E402
import instrument_counterfactual as icf  # noqa: E402
import instrument_selector as isel  # noqa: E402

TS_SRC = open(os.path.join(_ROOT, "server", "instrumentCounterfactual.ts"), encoding="utf-8").read()


def trading_days(start: str, n: int) -> list:
    d, out = date.fromisoformat(start), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def bars(days, closes) -> list:
    return [{"t": f"{d}T05:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": 1000} for d, c in zip(days, closes)]


DAYS = trading_days("2026-11-02", 26)                     # Mon 2026-11-02 = the entry day
AMD = bars(DAYS, [150.0, 151.5, 152.0, 151.0, 152.5, 153.15] + [154.0] * 20)
C145 = "AMD261211C00145000"
C150 = "AMD261211C00150000"
C155 = "AMD261211C00155000"
OPTION_BARS = {
    C145: bars(DAYS, [9.0, 9.9, 10.5, 10.0, 11.0, 12.42] + [13.0] * 20),
    C150: bars(DAYS[1:], [6.0] * 25),                     # no trade on the entry day
    C155: bars(DAYS[1:], [3.0] * 25),                     # nor this one
}
CONTRACTS = [{"symbol": s} for s in (C145, C150, C155, "AMD261204C00150000", "AMD261211P00150000")]

DECISION = {
    "v": 1, "kind": "decision", "id": "d1", "ts": "2026-11-02T15:30:00.000Z", "entry_date": "2026-11-02",
    "ticker": "AMD", "mode": "live", "source": "bot.ts executeTrades.stock", "choice": "stock_over_option",
    "chosen": {"instrument": "stock", "symbol": "AMD", "side": "buy", "strategy": "buy_stock", "price": 150.2,
               "qty": 12, "notional": 1802.4},
    "rejected": {"instrument": "option", "reason": "scored 41 vs 63 for the stock", "strategy": "sell_cash_secured_put",
                 "selector_score": 41, "scored": True, "contract": None, "contract_reason": "no chain",
                 "proxy": {"kind": "atm_call", "right": "C", "side": "buy", "dte_min": 30, "dte_max": 45,
                           "underlying": "AMD", "underlying_price": 150.2}},
    "summary": "Chose stock over options for AMD.",
}
TLT_DAYS = DAYS
TLT = bars(TLT_DAYS, [88.0, 88.5, 89.0, 88.0, 87.0, 86.0] + [86.0] * 20)
P85 = "TLT261219P00085000"
OPT_DECISION = {
    "v": 1, "kind": "decision", "id": "d3", "ts": "2026-11-02T16:00:00.000Z", "entry_date": "2026-11-02",
    "ticker": "TLT", "mode": "live", "source": "bot.ts executeTrades.options", "choice": "option_over_stock",
    "chosen": {"instrument": "option", "symbol": P85, "side": "sell", "strategy": "sell_cash_secured_put", "price": 1.2,
               "qty": 1, "notional": 8500, "legs": [{"symbol": P85, "side": "sell"}], "cash_required": 8500,
               "max_loss": 8380, "credit": None},
    "rejected": {"instrument": "stock", "reason": "scored 50 vs 68 for the option", "symbol": "TLT", "side": "buy",
                 "entry_price": 88.1, "qty": 96, "notional": 8457.6, "stop": 83.7, "target": 94.0},
    "summary": "Chose options over the stock for TLT.",
}


class _LedgerCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        p = patch.object(icf, "DATA_DIR", tmp.name)
        p.start()
        self.addCleanup(p.stop)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(icf.ENV_VAR, None)
        self.calls = {"bars": 0, "options": 0, "contracts": 0}

    def write(self, *rows):
        with open(icf.ledger_path(), "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def fetchers(self, series=None, option_series=None, contracts=None, fail=False):
        series = series if series is not None else {"AMD": AMD, "TLT": TLT}
        option_series = option_series if option_series is not None else {**OPTION_BARS, P85: bars(DAYS, [1.2, 1.0, 0.9, 0.8, 0.7, 0.6] + [0.5] * 20)}

        def fb(sym, start, end):
            self.calls["bars"] += 1
            if fail:
                raise RuntimeError("Alpaca bars HTTP 500")
            return [b for b in series.get(sym, []) if start <= b["t"][:10] <= end]

        def fo(symbols, start, end):
            self.calls["options"] += 1
            return {s: [b for b in option_series.get(s, []) if start <= b["t"][:10] <= end] for s in symbols if s in option_series}

        def fc(under, kind, gte, lte, k_lo, k_hi):
            self.calls["contracts"] += 1
            self.assertEqual((under, kind, gte, lte), ("AMD", "call", "2026-12-02", "2026-12-17"),
                             "30-45 days after the entry, the right type")
            self.assertAlmostEqual(k_lo, 150.2 * 0.85)
            self.assertAlmostEqual(k_hi, 150.2 * 1.15)
            return list(CONTRACTS if contracts is None else contracts)

        return {"fetch_bars": fb, "fetch_option_bars_fn": fo, "fetch_contracts_fn": fc}

    def outcomes(self):
        return [r for r in icf.read_rows() if r.get("kind") == "outcome"]


# ── the option alternative (instrument_selector) ─────────────────────────────
class TestOptionAlternative(unittest.TestCase):
    TRADE = {"ticker": "AMD", "price": 150.2, "side": "buy"}

    def test_not_scored_ruled_out_routed_none_outscored_and_chosen(self):
        a = isel.option_alternative(self.TRADE, None, None, "outside regular hours (09:30-16:00 ET)", "stock", 63)
        self.assertEqual(a["selector_reason"], "options were not scored: outside regular hours (09:30-16:00 ET)")
        self.assertEqual((a["scored"], a["strategy"], a["selector_score"], a["chosen_by_selector"]), (False, None, None, False))
        csp = {"instrument": "options", "strategy": "sell_cash_secured_put", "score": 0.0,
               "reasoning": "Options (sell_cash_secured_put) [RULED OUT: options exposure 17.0% at max 16%]"}
        a = isel.option_alternative(self.TRADE, csp, "options exposure 17.0% at max 16%", None, "stock", 63, raw_score=66.4)
        self.assertEqual(a["selector_reason"], "options were ruled out: options exposure 17.0% at max 16%")
        self.assertEqual((a["strategy"], a["selector_score"]), ("sell_cash_secured_put", 66.4), "the score before the rule-out")
        routed = {"instrument": "stock", "strategy": "stock", "score": 55.0,
                  "reasoning": "buy_call disabled by backtest (WR=22% historically) — routing to stock"}
        a = isel.option_alternative(self.TRADE, routed, None, None, "stock", 63, raw_score=55.0)
        self.assertEqual(a["selector_reason"],
                         "the options scorer routed it to stock: buy_call disabled by backtest (WR=22% historically) — routing to stock")
        self.assertIsNone(a["strategy"], "'stock' is not an options strategy")
        none = {"instrument": "options", "strategy": "none", "score": 20.0, "reasoning": "Options not compelling: VRP 1.0%, IVR 50"}
        self.assertEqual(isel.option_alternative(self.TRADE, none, None, None, "stock", 63)["selector_reason"],
                         "options had no viable strategy: Options not compelling: VRP 1.0%, IVR 50")
        low = {"instrument": "options", "strategy": "sell_cash_secured_put", "score": 41.0, "reasoning": "x"}
        self.assertEqual(isel.option_alternative(self.TRADE, low, None, None, "etf", 72, raw_score=41.0)["selector_reason"],
                         "options (sell_cash_secured_put) scored 41 vs 72 for the etf")
        chosen = isel.option_alternative(self.TRADE, {**low, "score": 68.0}, None, None, "options", 68, raw_score=68.0)
        self.assertEqual(chosen["selector_reason"], "the selector chose options (sell_cash_secured_put, scored 68)")
        self.assertTrue(chosen["chosen_by_selector"], "an entry that fell back to the stock records this option")

    def test_proxy_spec_and_the_null_contract(self):
        a = isel.option_alternative(self.TRADE, None, None, "x", "stock", 63)
        self.assertEqual(a["proxy"], {"kind": "atm_call", "right": "C", "side": "buy", "dte_min": 30, "dte_max": 45,
                                      "underlying": "AMD", "underlying_price": 150.2})
        self.assertIsNone(a["contract"])
        self.assertIn("no option chain was fetched for this candidate in the scan", a["contract_reason"])
        bear = isel.option_alternative({**self.TRADE, "side": "sell"}, None, None, "x", "stock", 63)
        self.assertEqual((bear["proxy"]["kind"], bear["proxy"]["right"], bear["proxy"]["side"]), ("atm_put", "P", "buy"))
        with patch.dict("system_config.BASE_CONFIG", {"INSTRUMENT_COUNTERFACTUAL": {"proxy_dte_min": 21, "proxy_dte_max": 28}}):
            p = isel.option_alternative(self.TRADE, None, None, "x", "stock", 63)["proxy"]
        self.assertEqual((p["dte_min"], p["dte_max"]), (21, 28), "the window is system_config's")

    def test_a_chain_the_selector_was_given_names_its_atm_contract(self):
        import pandas as pd
        calls = pd.DataFrame([
            {"contractSymbol": C145, "strike": 145.0, "bid": 7.9, "ask": 8.1, "lastPrice": 8.0, "impliedVolatility": 0.41},
            {"contractSymbol": "AMD261211C00152500", "strike": 152.5, "bid": 4.0, "ask": 4.2, "lastPrice": 4.1, "impliedVolatility": 0.4},
            {"contractSymbol": C150, "strike": 147.5, "bid": 6.0, "ask": 6.4, "lastPrice": 6.2, "impliedVolatility": 0.4},
        ])
        a = isel.option_alternative({**self.TRADE, "price": 150.2, "calls_df": calls, "T": 39 / 365}, None, None, "x", "stock", 63)
        self.assertEqual(a["contract"]["symbol"], "AMD261211C00152500", "the strike nearest 150.2")
        self.assertEqual((a["contract"]["mid"], a["contract"]["right"], a["contract"]["dte"]), (4.1, "C", 39))
        self.assertIsNone(a["contract_reason"])
        tie = isel._atm_from_chain([{"strike": 149.0}, {"strike": 151.0}, {"strike": "x"}, "junk"], 150.0)
        self.assertEqual(tie["strike"], 149.0, "a tie goes to the lower strike")
        self.assertIsNone(isel._atm_from_chain([{"bid": 1}], 150.0))
        empty = isel.option_alternative({**self.TRADE, "calls_df": []}, None, None, "x", "stock", 63)
        self.assertEqual(empty["contract_reason"], "the selector was given an option chain without a usable strike")


class TestSelectInstrumentCarriesIt(unittest.TestCase):
    def _run(self, hours: bool, score: float):
        isel._intel_cache.clear()
        intel = {"ticker": "AMD", "iv_rank": None, "fns_ran": [], "fns_skipped": []}
        with patch.object(isel, "_is_regular_hours", return_value=hours), \
                patch.object(isel, "get_instrument_intelligence", side_effect=lambda *a, **k: dict(intel)), \
                patch.object(isel, "_build_sizing", return_value={"type": "stock", "fraction": 0.02}), \
                patch.object(isel, "_check_etf_volume", return_value=True), \
                patch("macro_data.get_macro_snapshot", return_value={"vxx_ratio": 1.0, "vxx_latest": 25.0}), \
                patch("requests.get", side_effect=AssertionError("no network call in the scan path")):
            return isel.select_instrument({"ticker": "ZZZT", "price": 50.0, "deep_score": score, "volume": 5_000_000,
                                           "side": "buy", "vrp": 0.5, "ewma_rv": 2.0, "rsi": 50, "action_label": "BUY"},
                                          100_000, [], {"vix": 18})

    def test_the_result_carries_the_option_alternative(self):
        r = self._run(hours=False, score=80)
        self.assertEqual(r["chosen"], "stock")
        a = r["option_alternative"]
        self.assertEqual(a["selector_reason"], "options were not scored: outside regular hours (09:30-16:00 ET)")
        self.assertEqual(a["proxy"]["underlying_price"], 50.0)
        r = self._run(hours=True, score=58)
        self.assertEqual(r["option_alternative"]["selector_reason"], "options were not scored: score 58 is below the 65 options minimum")
        json.dumps(r["option_alternative"])     # JSON-safe: it rides the scan result to bot.ts


# ── the leg math ──────────────────────────────────────────────────────────────
class TestLegMath(unittest.TestCase):
    def test_shares_close_to_close_from_the_entry_day(self):
        r = icf.share_leg(AMD, "2026-11-02", DAYS[5])
        self.assertEqual((r["priced"], r["v0"], r["vN"], r["pct"], r["basis"]), (True, 150.0, 153.15, 2.1, "price"))
        self.assertEqual(icf.share_leg(AMD, "2026-11-02", DAYS[5], side="sell")["pct"], -2.1)
        self.assertFalse(icf.share_leg([], "2026-11-02", DAYS[5])["priced"])
        self.assertFalse(icf.share_leg(AMD[:1], "2026-11-02", DAYS[5])["priced"], "no close after the entry day")

    def test_a_bought_call_on_its_premium_and_a_sold_put_on_its_capital(self):
        call = icf.option_position([{"symbol": C145, "side": "buy"}], OPTION_BARS, AMD, "2026-11-02", DAYS[5])
        self.assertEqual((call["v0"], call["vN"], call["pct"], call["basis"], call["stale_days"]), (9.0, 12.42, 38.0, "premium", 0))
        put_bars = {P85: bars(DAYS, [1.2, 1.0])}
        csp = icf.option_position([{"symbol": P85, "side": "sell"}], put_bars, TLT, "2026-11-02", DAYS[1], capital_per_share=85.0)
        self.assertEqual((csp["v0"], csp["vN"], csp["basis"], csp["basis_per_share"]), (-1.2, -1.0, "capital", 85.0))
        self.assertAlmostEqual(csp["pct"], round(0.2 / 85 * 100, 3))
        no_cap = icf.option_position([{"symbol": P85, "side": "sell"}], put_bars, TLT, "2026-11-02", DAYS[1])
        self.assertEqual(no_cap["basis_per_share"], 85.0, "a lone sold put without a recorded capital: its strike")

    def test_expiry_inside_the_horizon_is_intrinsic_and_an_entry_day_without_a_trade_is_unpriced(self):
        weekly = "TLT261106P00085000"                    # expires Fri 2026-11-06
        tlt = bars(DAYS, [88.0, 87.0, 86.0, 84.0, 80.0, 81.0])
        r = icf.option_position([{"symbol": weekly, "side": "sell"}], {weekly: bars(DAYS[:5], [1.2, 1.5, 2.0, 2.5, 5.1])},
                                tlt, "2026-11-02", DAYS[5], capital_per_share=85.0)
        self.assertTrue(r["expired"])
        self.assertEqual(r["vN"], -5.0, "85 strike - 80 close on the expiry day")
        self.assertAlmostEqual(r["pct"], round((-5.0 + 1.2) / 85 * 100, 3))
        miss = icf.option_position([{"symbol": C150, "side": "buy"}], OPTION_BARS, AMD, "2026-11-02", DAYS[5])
        self.assertEqual(miss, {"priced": False, "reason": "the AMD $150 call expiring 2026-12-11 had no trade on the entry day (2026-11-02)"})
        self.assertFalse(icf.option_position([], OPTION_BARS, AMD, "2026-11-02", DAYS[5])["priced"])
        self.assertFalse(icf.option_position([{"symbol": "AMD"}], OPTION_BARS, AMD, "2026-11-02", DAYS[5])["priced"])

    def test_a_stale_exit_mark_and_a_vertical_on_its_recorded_capital(self):
        thin = {C145: bars([DAYS[0], DAYS[2]], [9.0, 10.0])}
        r = icf.option_position([{"symbol": C145, "side": "buy"}], thin, AMD, "2026-11-02", DAYS[5])
        self.assertEqual((r["vN"], r["stale_days"]), (10.0, 3), "the last trade was 3 trading days before day 5")
        short_p, long_p = "TLT261219P00085000", "TLT261219P00080000"
        legs = [{"symbol": short_p, "side": "sell"}, {"symbol": long_p, "side": "buy"}]
        ob = {short_p: bars(DAYS[:2], [1.2, 0.9]), long_p: bars(DAYS[:2], [0.4, 0.3])}
        v = icf.option_position(legs, ob, TLT, "2026-11-02", DAYS[1], capital_per_share=4.2)
        self.assertEqual((v["v0"], v["vN"]), (-0.8, -0.6))
        self.assertAlmostEqual(v["pct"], round(0.2 / 4.2 * 100, 3))
        self.assertEqual(icf._capital_per_share({"cash_required": 8500, "qty": 1}), 85.0)
        self.assertEqual(icf._capital_per_share({"max_loss": 840, "qty": 2}), 4.2)
        self.assertIsNone(icf._capital_per_share({}))


class TestLine(unittest.TestCase):
    def test_the_plain_english_lines(self):
        ch = icf.share_leg(AMD, "2026-11-02", DAYS[5])
        rj = icf.option_position([{"symbol": C145, "side": "buy"}], OPTION_BARS, AMD, "2026-11-02", DAYS[5])
        self.assertEqual(icf.outcome_line(DECISION, "+5d", ch, rj, proxy=True),
                         "chose stock: +2.1%; the ATM call proxy it passed over would have made +38.0% on the premium paid"
                         " — the choice cost 35.9 points (5 trading days, both legs close-to-close from the entry day)")
        lost = {**rj, "pct": -12.0}
        self.assertEqual(icf.outcome_line(DECISION, "+1d", ch, lost, proxy=True),
                         "chose stock: +2.1%; the ATM call proxy it passed over would have lost 12.0% on the premium paid"
                         " — the choice was right by 14.1 points (1 trading day, both legs close-to-close from the entry day)")
        self.assertEqual(icf.outcome_line(DECISION, "+5d", ch, icf._unpriced("no listed call"), proxy=True),
                         "chose stock: +2.1%; the ATM call proxy it passed over could not be priced (no listed call)"
                         " (5 trading days, both legs close-to-close from the entry day)")
        csp = {"priced": True, "pct": 0.235, "basis": "capital"}
        stock = {"priced": True, "pct": 0.568, "basis": "price"}
        self.assertEqual(icf.outcome_line(OPT_DECISION, "+1d", csp, stock, proxy=False),
                         "chose options (sell cash secured put): +0.2% on the capital at risk; the stock it passed over would"
                         " have made +0.6% — the choice cost 0.3 points (1 trading day, both legs close-to-close from the entry day)")


# ── the nightly job ───────────────────────────────────────────────────────────
class TestBackfill(_LedgerCase):
    def test_the_proxy_resolves_once_and_horizons_settle_as_trading_days_pass(self):
        self.write(DECISION)
        f = self.fetchers()
        s1 = icf.backfill(fetch_bars=f["fetch_bars"], fetch_option_bars_fn=f["fetch_option_bars_fn"],
                          fetch_contracts_fn=f["fetch_contracts_fn"], now=datetime(2026, 11, 4, 22, tzinfo=timezone.utc))
        self.assertEqual({k: s1[k] for k in ("decisions", "outcomes_written", "proxies_written", "pending", "written")},
                         {"decisions": 1, "outcomes_written": 1, "proxies_written": 1, "pending": 1, "written": 2})
        proxy = [r for r in icf.read_rows() if r.get("kind") == "proxy"][0]
        self.assertEqual(proxy["contract"]["symbol"], C145,
                         "the 2026-12-11 expiry (nearest the window's middle); 150 and 155 did not trade on the entry day")
        self.assertEqual((proxy["contract"]["dte"], proxy["cf_id"], proxy["reason"]), (39, "d1", None))
        o1 = self.outcomes()[0]
        self.assertEqual((o1["horizon"], o1["end_date"], o1["chosen"]["pct"], o1["rejected"]["pct"], o1["diff_pp"], o1["winner"]),
                         ("+1d", "2026-11-03", 1.0, 10.0, -9.0, "rejected"))
        self.assertTrue(o1["proxy"])
        s2 = icf.backfill(fetch_bars=f["fetch_bars"], fetch_option_bars_fn=f["fetch_option_bars_fn"],
                          fetch_contracts_fn=f["fetch_contracts_fn"], now=datetime(2026, 11, 10, 22, tzinfo=timezone.utc))
        self.assertEqual((s2["outcomes_written"], s2["proxies_written"]), (1, 0))
        self.assertEqual(self.calls["contracts"], 1, "the proxy is resolved once and reused")
        o5 = [o for o in self.outcomes() if o["horizon"] == "+5d"][0]
        self.assertEqual((o5["chosen"]["pct"], o5["rejected"]["pct"], o5["diff_pp"]), (2.1, 38.0, -35.9))
        self.assertEqual(o5["line"], "chose stock: +2.1%; the ATM call proxy it passed over would have made +38.0% on the premium "
                                     "paid — the choice cost 35.9 points (5 trading days, both legs close-to-close from the entry day)")
        s3 = icf.backfill(fetch_bars=f["fetch_bars"], fetch_option_bars_fn=f["fetch_option_bars_fn"],
                          fetch_contracts_fn=f["fetch_contracts_fn"], now=datetime(2026, 12, 4, 22, tzinfo=timezone.utc))
        self.assertEqual(s3["outcomes_written"], 1)
        s4 = icf.backfill(fetch_bars=f["fetch_bars"], fetch_option_bars_fn=f["fetch_option_bars_fn"],
                          fetch_contracts_fn=f["fetch_contracts_fn"], now=datetime(2026, 12, 5, 22, tzinfo=timezone.utc))
        self.assertEqual((s4["outcomes_written"], s4["written"], s4["pending"]), (0, 0, 0), "all three horizons settled")
        self.assertEqual(sorted(o["horizon"] for o in self.outcomes()), ["+1d", "+20d", "+5d"])

    def test_an_options_entry_against_its_stock_alternative(self):
        self.write(OPT_DECISION)
        f = self.fetchers()
        icf.backfill(fetch_bars=f["fetch_bars"], fetch_option_bars_fn=f["fetch_option_bars_fn"],
                     fetch_contracts_fn=f["fetch_contracts_fn"], now=datetime(2026, 11, 3, 22, tzinfo=timezone.utc))
        o = self.outcomes()[0]
        self.assertEqual(o["chosen"]["basis"], "capital")
        self.assertEqual(o["chosen"]["basis_per_share"], 85.0, "cash_required 8500 / (100 x 1)")
        self.assertAlmostEqual(o["chosen"]["pct"], round(0.2 / 85 * 100, 3))
        self.assertAlmostEqual(o["rejected"]["pct"], round((88.5 / 88.0 - 1) * 100, 3))
        self.assertEqual(self.calls["contracts"], 0, "no proxy: the stock leg needs none")
        self.assertFalse(o["proxy"])

    def test_an_unresolvable_proxy_is_recorded_once_and_its_leg_stays_unpriced(self):
        self.write(DECISION)
        f = self.fetchers(contracts=[])
        icf.backfill(**f, now=datetime(2026, 11, 3, 22, tzinfo=timezone.utc))
        proxy = [r for r in icf.read_rows() if r.get("kind") == "proxy"][0]
        self.assertIsNone(proxy["contract"])
        self.assertEqual(proxy["reason"], "no listed call expiring 30-45 days after 2026-11-02 within 15% of $150.20")
        o = self.outcomes()[0]
        self.assertTrue(o["chosen"]["priced"])
        self.assertEqual(o["rejected"], {"priced": False, "reason": proxy["reason"]})
        self.assertIsNone(o["diff_pp"])

    def test_give_up_fetch_errors_and_the_off_switch(self):
        old = {**DECISION, "id": "d2", "ts": "2026-08-03T15:00:00.000Z", "entry_date": "2026-08-03"}
        self.write(old)
        f = self.fetchers(series={})
        s = icf.backfill(**f, now=datetime(2026, 11, 4, 22, tzinfo=timezone.utc))
        self.assertEqual(s["gave_up"], 3)
        rows = self.outcomes()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["unpriced"] and r["diff_pp"] is None for r in rows))
        self.assertIn("no price data within 70 days of the entry", rows[0]["line"])
        self.assertEqual(icf.backfill(**f, now=datetime(2026, 11, 5, 22, tzinfo=timezone.utc))["written"], 0, "terminal")
        self.write(DECISION)
        bad = self.fetchers(fail=True)
        s = icf.backfill(**bad, now=datetime(2026, 11, 4, 22, tzinfo=timezone.utc))
        self.assertEqual((s["fetch_errors"], s["written"], s["pending"]), (1, 0, 1), "never raises; retried tomorrow")
        os.environ[icf.ENV_VAR] = "off"
        self.assertEqual(icf.backfill(**self.fetchers(), now=datetime(2026, 11, 4, 22, tzinfo=timezone.utc)),
                         {"enabled": False, "decisions": 0, "outcomes_written": 0, "proxies_written": 0, "pending": 0,
                          "gave_up": 0, "fetch_errors": 0, "written": 0})

    def test_too_young_and_the_per_run_cap(self):
        self.write(DECISION, {**DECISION, "id": "d9", "ts": "2026-11-02T15:31:00.000Z"})
        f = self.fetchers()
        self.assertEqual(icf.backfill(**f, now=datetime(2026, 11, 2, 22, tzinfo=timezone.utc))["pending"], 2, "under a day old")
        s = icf.backfill(max_decisions=1, **f, now=datetime(2026, 11, 3, 22, tzinfo=timezone.utc))
        self.assertEqual((s["outcomes_written"], s["pending"]), (1, 2), "one decision per run here; the other waits")

    def test_append_rotates_above_the_cap_and_both_generations_fold(self):
        self.write(DECISION)
        with patch.object(icf, "max_bytes", return_value=10):
            self.assertEqual(icf.append_rows([{"kind": "outcome", "cf_id": "d1", "horizon": "+1d", "line": "x"}]), 1)
        self.assertTrue(os.path.exists(icf.ledger_path() + ".1"))
        decisions, outcomes, proxies = icf.fold(icf.read_rows())
        self.assertEqual((list(decisions), list(outcomes["d1"]), proxies), (["d1"], ["+1d"], {}))
        with open(icf.ledger_path(), "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        self.assertEqual(len(icf.read_rows()), 2, "a malformed line is skipped")
        with patch.object(icf.os, "open", side_effect=OSError("disk full")):
            self.assertEqual(icf.append_rows([{"kind": "outcome"}]), 0, "a write failure never raises")


# ── the journal keeps would_be ────────────────────────────────────────────────
class TestJournalWouldBe(unittest.TestCase):
    def test_normalize_and_the_row_cut_keep_would_be(self):
        wb = {"instrument": "option", "proxy": {"kind": "atm_call", "underlying_price": float("nan")}}
        r = dj.normalize({"action": "buy", "ticker": "AMD", "rejected": [
            {"instrument": "option", "reason": "outscored", "would_be": wb}, {"instrument": "etf", "reason": "no ETF", "would_be": {}}]})
        self.assertEqual(r["rejected"][0]["would_be"], {"instrument": "option", "proxy": {"kind": "atm_call", "underlying_price": None}})
        self.assertNotIn("would_be", r["rejected"][1], "an empty would_be is dropped")
        big = dj.normalize({"action": "buy", "ticker": "AMD", "summary": "s" * 900,
                            "stages": {"selection": {"reasons": ["r" * 2000] * 6, "subscores": {f"k{i}": i for i in range(60)}}},
                            "rejected": [{"instrument": "option", "reason": "r" * 300, "would_be": {"instrument": "option"}}]
                            + [{"instrument": f"i{i}", "reason": "r" * 300} for i in range(5)]})
        cut = json.loads(dj._fit(big))
        self.assertTrue(cut.get("truncated"))
        self.assertEqual(cut["rejected"][0]["would_be"], {"instrument": "option"})


# ── parity + wiring ───────────────────────────────────────────────────────────
class TestParityAndWiring(unittest.TestCase):
    def ts_const(self, name):
        m = re.search(rf"export const {name} = ([^;]+);", TS_SRC)
        self.assertIsNotNone(m, name)
        return m.group(1).strip()

    def test_constants_match_the_ts_twin_and_system_config(self):
        import system_config
        cfg = system_config.BASE_CONFIG["INSTRUMENT_COUNTERFACTUAL"]
        self.assertEqual(cfg, {"enabled": True, "max_bytes": icf.DEFAULT_MAX_BYTES,
                               "proxy_dte_min": icf.DEFAULT_PROXY_DTE[0], "proxy_dte_max": icf.DEFAULT_PROXY_DTE[1]},
                         "default ON, with its one-setting off switch")
        self.assertEqual(self.ts_const("INSTRUMENT_CF_FILENAME"), f'"{icf.FILENAME}"')
        self.assertEqual(self.ts_const("INSTRUMENT_CF_ENV"), f'"{icf.ENV_VAR}"')
        self.assertEqual(int(self.ts_const("INSTRUMENT_CF_MAX_BYTES").replace("_", "")), icf.DEFAULT_MAX_BYTES)
        self.assertEqual(int(self.ts_const("INSTRUMENT_CF_ROW_MAX_BYTES").replace("_", "")), icf.ROW_MAX_BYTES)
        self.assertEqual((int(self.ts_const("PROXY_DTE_MIN")), int(self.ts_const("PROXY_DTE_MAX"))), icf.DEFAULT_PROXY_DTE)
        self.assertEqual(re.findall(r'"([^"]+)"', self.ts_const("INSTRUMENT_CF_HORIZONS")), list(icf.HORIZONS))
        self.assertEqual(isel._CF_PROXY_DTE_DEFAULT, icf.DEFAULT_PROXY_DTE)
        self.assertEqual(icf.OFF_VALUES, dj.OFF_VALUES, "the journal's off-switch vocabulary")

    def test_the_scan_carries_it_the_daemon_resolves_it_nothing_prints(self):
        be = open(os.path.join(_ROOT, "bot_engine.py"), encoding="utf-8").read()
        scan = be[be.index("def _scan_market_inner("):]
        self.assertIn('"option_alternative": instrument_decision.get("option_alternative"),', scan)
        daemon = open(os.path.join(_ROOT, "voltrade_daemon.py"), encoding="utf-8").read()
        self.assertIn('"instrument_cf_backfill": ("instrument_counterfactual", "backfill"),', daemon)
        src = open(icf.__file__, encoding="utf-8").read()
        self.assertNotRegex(src, r"(?m)^\s*print\(", "bot.ts parses the subprocess stdout")
        sel = open(isel.__file__, encoding="utf-8").read()
        block = sel[sel.index("def option_alternative("):sel.index("#  SECTION G: BACKWARD-COMPAT SHIM")]
        self.assertNotIn("requests.", block, "no network call in the scan path")


if __name__ == "__main__":
    unittest.main()
