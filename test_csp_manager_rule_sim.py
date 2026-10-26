"""
Research test 2 gate 3 — the manager's rules on the CSPs the bot would have
sold (scripts/csp_manager_rule_sim.py), [RULE-REVIEW] 2026-10-26. Synthetic
entries and bar paths; the rule constants are pinned to options_manager.py.
The fixture's bid/ask are derived from the model at the entry's own IV — an
archived row is quoted consistently with its IV, and a fixture that is not
(a $2 bid on a put the model values at $3.40) is underwater from day one.
"""
import math
import os
import re
import sys
import unittest
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import csp_manager_rule_sim as sim  # noqa: E402

S0 = 100.0


def bdays(start: str, n: int):
    d = date.fromisoformat(start); out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def entry(day="2026-07-06", dte=39, strike=95.0, bid=None, ask=None, iv=0.45, u="AAA", **extra):
    exp = (date.fromisoformat(day) + timedelta(days=dte)).isoformat()
    if bid is None or ask is None:
        mid = sim.bs_put(S0, strike, dte / 365.0, iv or 0.45)
        bid = round(mid - 0.10, 4) if bid is None else bid
        ask = round(mid + 0.10, 4) if ask is None else ask
    row = {"u": u, "day": day, "occ": f"{u}X", "exp": exp, "dte": dte, "strike": strike, "bid": bid, "ask": ask,
           "mid": round((bid + ask) / 2, 4), "spreadPct": round((ask - bid) / ((bid + ask) / 2), 4) if bid + ask else None,
           "iv": iv, "delta": -0.3, "oi": 500, "reason": "ok", "feed": "indicative"}
    row.update(extra)
    return row


def path(closes, start="2026-07-06"):
    return {"date": bdays(start, len(closes)), "close": list(closes)}


RISING = [S0 + 1.0 * i for i in range(60)]
CRASH = [S0, 99.0] + [68.0] * 58                  # a 32% gap on day 2
DRIFT = [S0 - 0.25 * i for i in range(60)]         # ends just in the money: no target, no stop, intrinsic at expiry


class TestPricer(unittest.TestCase):
    def test_bs_put_limits_and_monotonicity(self):
        self.assertAlmostEqual(sim.bs_put(100, 100, 0, 0.3), 0.0)
        self.assertAlmostEqual(sim.bs_put(50, 100, 0, 0.3), 50.0)
        self.assertAlmostEqual(sim.bs_put(50, 100, 0.1, 0.3), 100 * math.exp(-sim.R_RISK_FREE * 0.1) - 50, places=3)
        self.assertLess(sim.bs_put(200, 100, 0.1, 0.3), 1e-6)
        self.assertLess(sim.bs_put(100, 95, 0.1, 0.2), sim.bs_put(100, 95, 0.1, 0.6))

    def test_rules_are_the_live_manager_constants(self):
        with open(os.path.join(ROOT, "options_manager.py")) as f:
            src = f.read()
        val = lambda name: float(re.search(rf"^{name}\s*=\s*([\d.]+)", src, re.M).group(1))
        self.assertEqual(sim.RULES["profit_target_pct"], val("PROFIT_TARGET_PCT"))
        self.assertEqual(sim.RULES["loss_limit_mult"], val("LOSS_LIMIT_MULTIPLIER"))
        self.assertEqual(sim.RULES["dte_exit"], val("DTE_EXIT_THRESHOLD"))
        self.assertEqual(sim.RULES["dte_critical"], val("DTE_CRITICAL"))


class TestRules(unittest.TestCase):
    def test_rising_path_takes_profit_before_the_dte_exit(self):
        e = entry()
        r = sim.simulate_entry(e, path(RISING), "live")
        self.assertEqual(r["exit_reason"], "profit_target")
        self.assertGreater(r["pnl"], 0)
        self.assertLess(r["hold_days"], 39 - sim.RULES["dte_exit"])
        self.assertAlmostEqual(r["credit"], round(e["bid"] * 100, 2))
        self.assertFalse(r["censored"])
        self.assertLess(r["pnl"], r["pnl_gross"], "the net pays the entry and exit spread")

    def test_crash_hits_the_loss_limit_and_no_stop_holds(self):
        live = sim.simulate_entry(entry(), path(CRASH), "live")
        self.assertEqual(live["exit_reason"], "loss_limit")
        self.assertLessEqual(live["pnl"], -sim.RULES["loss_limit_mult"] * live["credit"])
        held = sim.simulate_entry(entry(), path(CRASH), "no_stop")
        self.assertIn(held["exit_reason"], ("dte_exit", "dte_critical"))
        self.assertGreater(held["hold_days"], live["hold_days"])

    def test_near_the_money_drift_closes_at_21_dte_live_and_runs_to_expiry_without_dte_exits(self):
        live = sim.simulate_entry(entry(iv=0.35), path(DRIFT), "live")
        self.assertEqual(live["exit_reason"], "dte_exit")
        self.assertLessEqual((date.fromisoformat(live["exp"]) - date.fromisoformat(live["exit_day"])).days, sim.RULES["dte_exit"])
        held = sim.simulate_entry(entry(iv=0.35), path(DRIFT), "no_dte_exit")
        self.assertEqual(held["exit_reason"], "expiry")
        self.assertEqual(held["exit_mark"], round(max(95.0 - held["S_exit"], 0.0), 4), "expiry settles at intrinsic, no spread")

    def test_censored_when_bars_end_before_an_exit(self):
        r = sim.simulate_entry(entry(iv=0.35), path(DRIFT[:6]), "live")
        self.assertTrue(r["censored"])
        self.assertEqual(r["exit_reason"], "censored")

    def test_unusable_entries_return_none(self):
        self.assertIsNone(sim.simulate_entry(entry(bid=0, ask=0.2), path(RISING[:10]), "live"))
        self.assertIsNone(sim.simulate_entry(entry(iv=None), path(RISING[:10]), "live"))
        self.assertIsNone(sim.simulate_entry(entry(day="2026-07-04"), path(RISING[:10]), "live"), "no bar on the entry day")
        with self.assertRaises(ValueError):
            sim.simulate_entry(entry(), path(RISING[:10]), "bogus")

    def test_earnings_gate_skips_only_entries_whose_window_holds_an_earnings_date(self):
        e = entry()
        gated = sim.simulate_entry(e, path(RISING), "earnings_gate", earnings=["2026-07-20"])
        self.assertEqual(gated.get("skipped"), "earnings")
        kept = sim.simulate_entry(e, path(RISING), "earnings_gate", earnings=["2026-09-20", "bad-date"])
        self.assertEqual(kept["exit_reason"], "profit_target")


class TestSets(unittest.TestCase):
    def test_dedupe_keeps_the_first_entry_per_name_per_window_and_drops_unselectable_rows(self):
        rows = [entry(day="2026-07-06"), entry(day="2026-07-08"), entry(day="2026-07-13"),
                entry(day="2026-07-06", u="BBB"), {**entry(day="2026-07-07", u="CCC"), "reason": "no_liquid", "occ": None}]
        kept = sim.dedupe(rows, days=5)
        self.assertEqual([(r["u"], r["day"]) for r in kept], [("AAA", "2026-07-06"), ("BBB", "2026-07-06"), ("AAA", "2026-07-13")])

    def test_simulate_all_and_aggregate(self):
        entries = [entry(day="2026-07-06"), entry(day="2026-07-13", u="BBB"), entry(day="2026-07-13", u="ZZZ")]
        bars = {"AAA": path(RISING), "BBB": path(CRASH)}
        s = sim.simulate_all(entries, bars, "live", regime_by_day={"2026-07-06": "BULL", "2026-07-13": "NEUTRAL"})
        self.assertEqual(s["unusable"], 1, "ZZZ has no bars")
        agg = sim.aggregate(s)
        o = agg["overall"]
        self.assertEqual((o["n"], o["n_closed"], o["n_censored"]), (2, 2, 0))
        self.assertEqual(agg["by_reason"], {"profit_target": 1, "loss_limit": 1})
        self.assertEqual(set(agg["by_regime"]), {"BULL", "NEUTRAL"})
        self.assertEqual(agg["by_regime"]["BULL"]["win_rate"], 1.0)
        self.assertEqual(agg["worst"][0]["u"], "BBB")
        self.assertIn("36-42", agg["by_dte_entry"])
        self.assertAlmostEqual(o["pnl_sum"], agg["per_underlying"]["AAA"]["pnl_sum"] + agg["per_underlying"]["BBB"]["pnl_sum"], places=2)
        self.assertIsNotNone(o["expectancy_pct_collateral"])

    def test_live_comparison_joins_on_the_occ_root(self):
        aggs = {"live": {"per_underlying": {"AAA": {"n": 2, "pnl_sum": 50.0}, "BBB": {"n": 1, "pnl_sum": -300.0}}}}
        sleeves = {"optionsByUnderlying": [{"root": "AAA", "realized": -120, "fills": 7}, {"root": "QQQ", "realized": 5, "fills": 2}]}
        self.assertEqual(sim.live_comparison(aggs, sleeves), [{"u": "AAA", "live_realized": -120, "live_fills": 7, "sim_n": 2, "sim_pnl_sum": 50.0}])
        self.assertEqual(sim.live_comparison(aggs, None), [])

    def test_render_states_the_prior_the_verdict_and_the_honesty_section(self):
        entries = [entry(day="2026-07-06"), entry(day="2026-07-13", u="BBB")]
        bars = {"AAA": path(RISING), "BBB": path(CRASH)}
        aggs = {v: sim.aggregate(sim.simulate_all(entries, bars, v)) for v in sim.VARIANTS}
        meta = {"date": "2026-10-26", "generated_at": "2026-10-26T21:00:00Z", "entry_rows": 2, "entry_ok": 2, "entry_days": 2,
                "entry_underlyings": 2, "deduped": 2, "dedupe_days": 5, "bars_names": 2, "source": "test"}
        md = sim.render(aggs, meta)
        for s in ("## PRIOR", sim.PRIOR, "## VERDICT", "no_stop", "no_dte_exit", "earnings_gate", "## Honesty", "MODEL MARK", "LOWER bound", "discount the best by four"):
            self.assertIn(s, md)
        self.assertIn("live rules: expectancy", sim.verdict(aggs))


if __name__ == "__main__":
    unittest.main()
