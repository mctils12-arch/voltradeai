"""
[RESEARCH] 2026-10-27 — source_validation.py: every data source tested ALONE
against forward outcomes on the deduped shadow population, BH-corrected.
Synthetic shadow records with PLANTED signals: a positive adjustment must
PASS, an inverted one must read FAIL_INVERTED, noise must not pass, a dead
input reads CONSTANT, and per-scan copies must not inflate n.
"""
import ast
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np

import source_validation as sv

HERE = os.path.dirname(os.path.abspath(__file__))


def _feature_cols_from_source():
    tree = ast.parse(open(os.path.join(HERE, "ml_model_v2.py")).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "FEATURE_COLS" for t in n.targets):
            return [e.value for e in n.value.elts]
    raise AssertionError("FEATURE_COLS not found")


def make_records(n_days=40, per_day=8, copies=3, seed=7, with_breakdown=True):
    """Shadow-shaped records. Outcome +5d = 2*good + noise; adj.news is the
    planted GOOD signal, adj.social the planted INVERTED one (= -good),
    adj.alt pure noise, feat.intel_score constant 0 (dead, like production)."""
    rng = np.random.default_rng(seed)
    recs = []
    t0 = datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc)
    day = 0
    d = t0
    while day < n_days:
        if d.weekday() < 5:
            for k in range(per_day):
                good = float(rng.normal())
                y = 2.0 * good + float(rng.normal())
                for c in range(copies):
                    ts = d + timedelta(minutes=5 * c)
                    feats = {}
                    for f, _g in sv.FEATURE_SOURCES:
                        feats[f] = float(rng.normal())
                    feats["intel_score"] = 0.0
                    feats["spy_vs_ma50"] = 1.03
                    rec = {
                        "ticker": f"T{day:02d}{k}",
                        "timestamp": ts.isoformat(),
                        "score": 60 + good,
                        "decision": "taken",
                        "vxx_ratio": 0.9 if day % 2 else 1.08,
                        "features": feats,
                        "outcomes": {"+5d": {"label": int(y > 2), "return_pct": y},
                                     "+10d": {"label": int(y > 2), "return_pct": y * 0.8},
                                     "+20d": None},
                        "ml": {"rules_only_score": 60 + good, "ml_only_score": 50.0, "win_probability": 0.5},
                    }
                    if with_breakdown:
                        rec["score_breakdown"] = {
                            "momentum": 50.0, "mean_reversion": float(rng.normal()), "vrp": float(rng.normal()),
                            "squeeze": float(rng.normal()), "volume": float(rng.normal()),
                            "vol_premium": 0.0, "macro": float(rng.normal()),
                            # copies after the first carry a different value: dedup must keep the FIRST
                            "news": round(good if c == 0 else -good, 3),
                            "adx": float(rng.normal()), "obv": float(rng.normal()),
                            "intel": float(rng.normal()), "alt": float(rng.normal()),
                            "social": round(-good, 3), "fundamental": float(rng.normal()),
                        }
                    recs.append(rec)
            day += 1
        d += timedelta(days=1)
    # tier-1 masterkill CSP rows: no features, score 0 — must be excluded
    for i in range(30):
        recs.append({"ticker": f"CSP{i}", "timestamp": t0.isoformat(), "score": 0.0,
                     "decision": "rejected_masterkill", "features": {},
                     "outcomes": {"+5d": {"label": 0, "return_pct": -9.0}}})
    return recs


class TestRegistryPins(unittest.TestCase):
    def test_feature_sources_match_model_feature_cols(self):
        self.assertEqual([f for f, _ in sv.FEATURE_SOURCES], _feature_cols_from_source())

    def test_adjustment_sources_match_persisted_breakdown(self):
        from shadow_portfolio import SCORE_BREAKDOWN_KEYS
        self.assertEqual(tuple(k for k, _ in sv.ADJUSTMENT_SOURCES), tuple(SCORE_BREAKDOWN_KEYS))

    def test_prereg_constants(self):
        # Pre-registered before any production run; a change is a new PREREG_VERSION.
        self.assertEqual(sv.PRIMARY_OUTCOME, "+5d")
        self.assertEqual((sv.MIN_N, sv.FDR_Q, sv.IC_MIN), (100, 0.10, 0.05))
        self.assertIn("v1", sv.PREREG_VERSION)


class TestStatsPrimitives(unittest.TestCase):
    def test_benjamini_hochberg_known_values(self):
        q = sv.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
        self.assertAlmostEqual(q[0], 0.04)
        self.assertAlmostEqual(q[1], 0.04 * 4 / 3)
        self.assertAlmostEqual(q[2], 0.04 * 4 / 3)  # monotone step-up
        self.assertAlmostEqual(q[3], 0.20)
        self.assertEqual(sv.benjamini_hochberg([]), [])

    def test_ic_stats_ci_brackets_ic_and_is_deterministic(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        y = 0.5 * x + rng.normal(size=400)
        days = [f"2026-09-{(i % 20) + 1:02d}" for i in range(400)]
        a = sv.ic_stats(x, y, days, sv.DayBootstrap(days))
        b = sv.ic_stats(x, y, days, sv.DayBootstrap(days))
        self.assertEqual(a, b)
        self.assertLess(a["ci95"][0], a["ic"])
        self.assertGreater(a["ci95"][1], a["ic"])
        self.assertLess(a["p"], 0.001)

    def test_constant_source_flagged(self):
        days = ["2026-09-01"] * 10 + ["2026-09-02"] * 10
        st = sv.ic_stats(np.zeros(20), np.arange(20.0), days, sv.DayBootstrap(days))
        self.assertTrue(st.get("constant"))

    def test_clustered_se_wider_than_naive_when_days_share_moves(self):
        # 10 names per day all move with a day shock; the source is the day
        # shock + noise. Row-level resampling would understate the SE.
        rng = np.random.default_rng(3)
        days, x, y = [], [], []
        for d in range(30):
            shock = rng.normal()
            for _ in range(10):
                days.append(f"d{d:02d}")
                x.append(shock + 0.1 * rng.normal())
                y.append(shock + rng.normal())
        x, y = np.array(x), np.array(y)
        st = sv.ic_stats(x, y, days, sv.DayBootstrap(days))
        naive_se = (1 - st["ic"] ** 2) / np.sqrt(len(x))
        self.assertGreater(st["se"], naive_se)


class TestValidateSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = make_records()
        cls.rep = sv.validate_sources(cls.recs)

    def test_population_is_deduped_and_deep_scored_only(self):
        pop = self.rep["population"]["+5d"]
        self.assertEqual(pop["n_unique"], 40 * 8)          # copies collapse; masterkill rows excluded
        self.assertEqual(pop["n_days"], 40)
        self.assertEqual(pop["with_score_breakdown"], 40 * 8)

    def test_planted_positive_adjustment_passes(self):
        r = self.rep["sources"]["adj.news"]
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertGreater(r["ic"], 0.5)                    # first-seen copy (= +good) was used
        self.assertLessEqual(r["q_bh"], sv.FDR_Q)

    def test_planted_inverted_adjustment_flagged(self):
        r = self.rep["sources"]["adj.social"]
        self.assertEqual(r["verdict"], "FAIL_INVERTED", r)
        self.assertLess(r["ci95"][1], 0)
        self.assertIn("adj.social", self.rep["summary"]["FAIL_INVERTED"])

    def test_noise_does_not_pass(self):
        noise = [k for k in ("adj.alt", "adj.adx", "adj.obv", "adj.intel", "adj.fundamental")
                 if self.rep["sources"][k]["verdict"] == "PASS"]
        self.assertEqual(noise, [])

    def test_dead_inputs_constant(self):
        self.assertEqual(self.rep["sources"]["feat.intel_score"]["verdict"], "CONSTANT")
        self.assertEqual(self.rep["sources"]["adj.momentum"]["verdict"], "CONSTANT")
        self.assertIn("feat.intel_score", self.rep["summary"]["CONSTANT"])

    def test_market_wide_flag_and_regime_split(self):
        self.assertTrue(self.rep["sources"]["feat.vxx_ratio"]["market_wide"])
        by_reg = self.rep["sources"]["adj.news"]["by_regime"]
        self.assertEqual(set(by_reg), {"BULL", "CAUTION"})
        self.assertGreater(by_reg["BULL"]["ic"], 0.5)

    def test_secondary_outcomes_descriptive(self):
        sec = self.rep["sources"]["adj.news"]["secondary"]
        self.assertGreater(sec["+10d"]["ic"], 0.5)
        self.assertEqual(sec["+20d"]["n"], 0)               # unlabeled horizon -> empty, not an error
        self.assertNotIn("verdict", sec["+10d"])

    def test_reference_rows_outside_family(self):
        self.assertIn("score", self.rep["reference"])
        self.assertNotIn("score", self.rep["sources"])
        self.assertGreater(self.rep["reference"]["score"]["ic"], 0.5)

    def test_insufficient_below_min_n(self):
        rep = sv.validate_sources(make_records(n_days=6, per_day=5, copies=1))
        self.assertEqual(rep["sources"]["adj.news"]["verdict"], "INSUFFICIENT")

    def test_records_without_breakdown(self):
        rep = sv.validate_sources(make_records(n_days=20, with_breakdown=False))
        self.assertEqual(rep["sources"]["adj.news"]["n"], 0)
        self.assertEqual(rep["sources"]["adj.news"]["verdict"], "INSUFFICIENT")

    def test_aggregate_only_and_probe_safe(self):
        blob = json.dumps(self.rep)
        self.assertNotIn("T00", blob)
        self.assertNotIn("CSP", blob)
        self.assertNotIn('"ticker"', blob)

        def depth(o, d=0):
            if isinstance(o, dict):
                self.assertLessEqual(len(o), 100, "sanitizeDiag keeps 100 keys per object")
                return max([depth(v, d + 1) for v in o.values()] or [d])
            if isinstance(o, list):
                return max([depth(v, d + 1) for v in o] or [d])
            return d
        self.assertLessEqual(depth(self.rep), 8, "sanitizeDiag truncates below depth 8")

    def test_empty_log(self):
        rep = sv.validate_sources([])
        self.assertEqual(rep["population"]["+5d"]["n_unique"], 0)
        self.assertEqual(len(rep["summary"]["INSUFFICIENT"]) + len(rep["summary"]["CONSTANT"]),
                         len(sv.source_registry()))

    def test_production_entry_point_reads_the_shadow_log(self):
        with mock.patch("shadow_portfolio._load_shadow_log", return_value=make_records(n_days=8)) as m:
            rep = sv.get_source_validation_report()
        m.assert_called_once()
        self.assertEqual(rep["population"]["+5d"]["n_days"], 8)


class TestCli(unittest.TestCase):
    def test_cli_runs_on_a_copied_log(self):
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "log.json")
            out = os.path.join(d, "rep.json")
            with open(p, "w") as f:
                json.dump(make_records(n_days=15), f)
            cp = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "source_validation.py"),
                                 "--log", p, "--out", out], capture_output=True, text=True, timeout=120)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertIn("primary +5d", cp.stdout)
            self.assertEqual(json.load(open(out))["population"]["+5d"]["n_days"], 15)


if __name__ == "__main__":
    unittest.main()
