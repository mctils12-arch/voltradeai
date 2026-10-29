"""test_brain_evidence.py — brain_evidence.py (Brain page, 2026-10-29).

Pins: the Node config mirror (server/brain.ts BRAIN_CONFIG_MIRROR) equals
system_config BASE_CONFIG; the daemon route; bucket statistics on synthetic
shadow records (first-seen dedupe, SPY same-window excess, day-clustered CI,
thin buckets carry no CI); the trimmed source_validation shape; the entry
point degrades block by block instead of raising.
"""
import json
import os
import re
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import brain_evidence as be

ROOT = os.path.dirname(os.path.abspath(__file__))


def _rec(ticker, day, ret, decision="rejected_score", hour=15, exit_day=None, sleeve=None):
    ts = datetime.strptime(day, "%Y-%m-%d").replace(hour=hour, tzinfo=timezone.utc).isoformat()
    ex = exit_day or (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
    r = {"ticker": ticker, "timestamp": ts, "decision": decision, "score": 60.0, "features": {"x": 1.0},
         "outcomes": {"+5d": {"label": 1 if ret > 0 else 0, "return_pct": ret, "exit_date": ex}, "+20d": None},
         "outcome_exit_aligned": {"label": 1 if ret > 0 else 0, "return_pct": ret, "exit_date": ex}}
    if sleeve:
        r["shadow_entry"] = {"sleeve": sleeve}
    return r


def _spy(days, start=100.0, step=0.1):
    d0 = datetime(2026, 6, 1)
    return {(d0 + timedelta(days=i)).strftime("%Y-%m-%d"): start + i * step for i in range(days)}


class TestConfigMirror(unittest.TestCase):
    def test_node_mirror_matches_base_config(self):
        src = open(os.path.join(ROOT, "server", "brain.ts")).read()
        block = src[src.index("BRAIN_CONFIG_MIRROR"):]
        block = block[:block.index("});")]
        snap = be.config_snapshot()
        pairs = re.findall(r"^\s*(\w+):\s*(.+?),\s*$", block, re.M)
        self.assertGreaterEqual(len(pairs), 10)
        for key, lit in pairs:
            val = json.loads(lit)
            self.assertIn(key, snap, key)
            self.assertEqual(snap[key], val, f"server/brain.ts BRAIN_CONFIG_MIRROR.{key}={lit} != BASE_CONFIG {snap[key]!r}")

    def test_snapshot_reads_tiers_and_convexity(self):
        snap = be.config_snapshot()
        self.assertIn("CONVEXITY_OVERLAY_enabled", snap)
        self.assertIn("TIERS_ENABLED", snap)
        for k in be.CONFIG_KEYS:
            self.assertIn(k, snap)

    def test_daemon_route(self):
        src = open(os.path.join(ROOT, "voltrade_daemon.py")).read()
        self.assertIn('"brain_evidence": ("brain_evidence", "brain_evidence")', src)


class TestBuckets(unittest.TestCase):
    def test_spy_window_return(self):
        closes = {"2026-06-01": 100.0, "2026-06-02": 101.0, "2026-06-05": 110.0}
        self.assertAlmostEqual(be.spy_window_return(closes, "2026-06-01", "2026-06-05"), 10.0)
        # weekend entry day -> last close on/before
        self.assertAlmostEqual(be.spy_window_return(closes, "2026-06-03", "2026-06-05"), (110 / 101 - 1) * 100)
        self.assertIsNone(be.spy_window_return(closes, "2026-06-05", "2026-06-05"))
        self.assertIsNone(be.spy_window_return({}, "2026-06-01", "2026-06-05"))
        self.assertIsNone(be.spy_window_return(closes, "2026-06-01", None))

    def test_dedupe_first_seen_and_excess(self):
        recs = []
        for i in range(40):
            day = (datetime(2026, 6, 1) + timedelta(days=i % 10)).strftime("%Y-%m-%d")
            recs.append(_rec(f"T{i}", day, -1.0))
        # a later copy of the same (ticker, day) with a different outcome must be ignored
        recs.append(_rec("T0", "2026-06-01", 50.0, hour=18))
        out = be.bucket_stats(recs, _spy(60))
        cell = out["rejected_score"]
        self.assertEqual(cell["records"], 41)
        xa = cell["xa"]
        self.assertEqual(xa["n"], 40)
        self.assertEqual(xa["mean_pct"], -1.0)
        self.assertEqual(xa["hit_rate"], 0.0)
        self.assertEqual(xa["first_day"], "2026-06-01")
        self.assertIn("ci95", xa)
        # SPY rose 0.1/day over 7 days from ~100 -> excess below -1
        self.assertLess(xa["vs_spy"]["mean_excess_pct"], -1.0)
        self.assertEqual(xa["vs_spy"]["n"], 40)
        self.assertFalse(xa["thin"])

    def test_thin_bucket_has_no_ci_and_sleeves_split(self):
        recs = [_rec(f"S{i}", "2026-06-02", 1.0, decision="shadow_mode_entry", sleeve="stocks") for i in range(5)]
        recs += [_rec("O1", "2026-06-02", 1.0, decision="shadow_mode_entry", sleeve="options")]
        out = be.bucket_stats(recs, None)
        st = out["shadow_mode_entry:stocks"]["xa"]
        self.assertEqual(st["n"], 5)
        self.assertTrue(st["thin"])
        self.assertNotIn("ci95", st)
        self.assertNotIn("vs_spy", st)   # no SPY -> no benchmark claimed
        self.assertIn("shadow_mode_entry:options", out)

    def test_unlabeled_records_count_but_measure_nothing(self):
        r = _rec("U", "2026-06-02", 1.0)
        r["outcome_exit_aligned"] = None
        r["outcomes"] = {"+5d": None}
        out = be.bucket_stats([r], None)
        self.assertEqual(out["rejected_score"]["records"], 1)
        self.assertEqual(out["rejected_score"]["xa"]["n"], 0)
        self.assertNotIn("mean_pct", out["rejected_score"]["xa"])


class TestSourceValidationTrim(unittest.TestCase):
    def test_trim_keeps_verdicts_discount_and_ablation(self):
        rep = {
            "method": {"primary_outcome": "+5d", "multiple_testing": "Benjamini-Hochberg across 2 sources at FDR 0.1"},
            "population": {"+5d": {"n_unique": 300, "n_days": 40, "first_day": "2026-06-01", "last_day": "2026-09-01", "mean_outcome_pct": -0.5, "regimes": {}}},
            "summary": {"PASS": [], "FAIL_INVERTED": ["adj.news"]},
            "sources": {"adj.news": {"n": 300, "ic": -0.1, "ci95": [-0.2, -0.01], "p": 0.01, "q_bh": 0.02, "verdict": "FAIL_INVERTED",
                                      "verdict_reason": "x", "terciles": {"spread": -1}, "by_regime": {"BULL": {}}, "kind": "adjustment", "group": "news"}},
            "reference": {"score": {"n": 300, "ic": -0.05, "ci95": [-0.1, 0.0], "terciles": {}}},
            "selection": {"method": {}, "features": {"status": "NOTHING_SELECTED", "steps": [{"candidate": "a", "tried": 20}, {"candidate": "b", "tried": 19}],
                                                     "holdout_live_score": {"n": 50, "ic": -0.1, "ci95": [-0.2, 0.0], "se": 0.05, "p": 0.1}},
                          "breakdown": {"status": "INSUFFICIENT", "reason": "too few"}},
            "ablation": {"status": "OK", "un_inverts": True, "variants": {"drop_unvalidated": {"n": 300, "ic": 0.01, "quintiles": {"Q1": {}, "Q5_minus_Q1": 0.2}, "removed": ["news"]}}},
        }
        t = be.trim_source_validation(rep)
        self.assertEqual(t["sources"]["adj.news"]["verdict"], "FAIL_INVERTED")
        self.assertEqual(t["sources"]["adj.news"]["q_bh"], 0.02)
        self.assertNotIn("by_regime", t["sources"]["adj.news"])
        self.assertEqual(t["population"]["n_unique"], 300)
        self.assertEqual(t["selection"]["features"]["variants_tried"], 39)
        self.assertNotIn("se", t["selection"]["features"]["holdout_live_score"])
        self.assertEqual(t["selection"]["breakdown"]["variants_tried"], 0)
        self.assertEqual(t["ablation"]["variants"]["drop_unvalidated"]["q5_minus_q1_pct"], 0.2)
        self.assertTrue(t["ablation"]["un_inverts"])


class TestEntryPoint(unittest.TestCase):
    def test_empty_log_degrades_without_raising(self):
        with patch("shadow_portfolio._load_shadow_log", return_value=[]):
            out = be.brain_evidence(fetch_spy=False)
        self.assertEqual(out["shadow_log"]["records"], 0)
        self.assertEqual(out["buckets"], {})
        self.assertIn("error", out["source_validation"])
        self.assertFalse(out["spy"]["ok"])
        self.assertIn("config", out)
        json.dumps(out, default=str)

    def test_a_failing_block_is_an_error_not_an_exception(self):
        recs = [_rec(f"T{i}", "2026-06-02", 1.0) for i in range(3)]
        with patch("shadow_portfolio._load_shadow_log", return_value=recs), \
                patch.object(be, "_full_sv", side_effect=RuntimeError("boom")):
            out = be.brain_evidence(fetch_spy=False)
        self.assertIn("boom", out["source_validation"]["error"])
        self.assertEqual(out["buckets"]["rejected_score"]["xa"]["n"], 3)

    def test_spy_fetch_failure_is_reported(self):
        recs = [_rec("A", "2026-06-02", 1.0)]
        be._SPY_CACHE.update({"at": 0.0, "closes": None, "key": None})

        def fake_fetch(tickers, start, end, deadline=None, unfetched=None):
            unfetched.add("SPY")
            return {}
        with patch("shadow_portfolio._fetch_bars_paginated", side_effect=fake_fetch):
            res = be.spy_closes_for(recs)
        self.assertIn("error", res)


class TestWalkforward(unittest.TestCase):
    """The long-term column reads walkforward.py's latest.json (research/walkforward.md)."""

    def test_reads_the_harness_output_into_brain_items(self):
        import tempfile
        import walkforward as wf
        import walkforward_synth as ws
        with tempfile.TemporaryDirectory() as tmp:
            ws.synthetic_panel(os.path.join(tmp, "p"), n_names=14, signal=0.003, seed=3)
            self.assertEqual(wf.run(os.path.join(tmp, "p"), os.path.join(tmp, "o"), code_version="9.9.9")["status"], "ok")
            out = be._walkforward_summary(os.path.join(tmp, "o", "latest.json"))
        self.assertEqual(out["schema"], "walkforward/v1")
        self.assertEqual(out["code_version"], "9.9.9")
        self.assertEqual(set(out["items"]), set(wf.INPUTS))
        it = out["items"]["momentum"]
        for k in ("id", "title", "source_file", "date", "horizon", "finding", "compared_to", "sign", "variants_tried", "quotes"):
            self.assertIn(k, it)
        self.assertIn(it["sign"], ("positive", "negative", "null", "insufficient"))
        self.assertEqual(it["sign"], out["verdicts"]["momentum"])
        self.assertIn("vrp", out["not_measurable"])
        self.assertGreater(out["universe"]["n_symbols"], 0)

    def test_missing_or_foreign_file_is_an_error_block(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("no walk-forward result yet", be._walkforward_summary(os.path.join(tmp, "latest.json"))["error"])
            p = os.path.join(tmp, "x.json")
            with open(p, "w") as fh:
                json.dump({"schema": "other/v9"}, fh)
            self.assertIn("unexpected walk-forward schema", be._walkforward_summary(p)["error"])

    def test_entry_point_carries_the_block(self):
        with patch("shadow_portfolio._load_shadow_log", return_value=[]), \
                patch.object(be, "_walkforward_summary", return_value={"error": "no walk-forward result yet"}):
            out = be.brain_evidence(fetch_spy=False)
        self.assertIn("walkforward", out)
        self.assertIn("error", out["walkforward"])


if __name__ == "__main__":
    unittest.main()
