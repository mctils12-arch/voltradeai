"""
[RESEARCH] 2026-10-27 — source_validation.py step 2: forward selection over
sources (time-ordered fit/select/holdout by entry day, Bonferroni per step)
and the deep_score ablation (unvalidated intel/alt/social/news/fundamental
adjustments removed, recomputed from the logged components), plus the
shadow-only logging of the components the breakdown used to miss.
"""
import ast
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np

import shadow_portfolio as sp
import source_validation as sv

HERE = os.path.dirname(os.path.abspath(__file__))
UNVAL = sv.UNVALIDATED_ADJUSTMENTS


def _blend(rules, rw=0.7, mw=0.3, ml=60.0, gated=False):
    return max(0.0, min(100.0, rules * rw + (50.0 if gated else ml) * mw))


def ablation_records(n_days=40, per_day=8, seed=11, inverted=True, log_raw=True):
    """Each candidate's TRUE quality g drives the outcome. The technical
    sub-scores carry +g; the 'unvalidated' adjustments carry -3g (when
    inverted), so the logged score ranks backwards until they are removed."""
    rng = np.random.default_rng(seed)
    recs, d, day = [], datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc), 0
    while day < n_days:
        if d.weekday() < 5:
            for k in range(per_day):
                g = float(rng.normal())
                y = g + 0.5 * float(rng.normal())
                bd = {key: 0.0 for key, _ in sv.ADJUSTMENT_SOURCES}
                bd.update({"momentum": 50 + 10 * g, "mean_reversion": 50.0, "vrp": 50.0,
                           "squeeze": 50.0, "volume": 50.0})
                adj = (-3.0 if inverted else 0.0) * g
                for key in UNVAL:
                    bd[key] = round(adj + 0.3 * float(rng.normal()), 1)
                bd["momentum"] = round(bd["momentum"], 1)
                raw = (bd["momentum"] * 0.25 + 50 * 0.20 + 50 * 0.25 + 50 * 0.15 + 50 * 0.15
                       + sum(bd[key] for key in UNVAL))
                if log_raw:
                    bd["rules_raw"] = round(raw, 1)
                rules_only = round(max(0.0, min(100.0, raw)), 1)
                gated = bool(k % 3 == 0)
                score = _blend(rules_only, gated=gated)
                recs.append({
                    "ticker": f"A{day:02d}{k}", "timestamp": d.isoformat(), "score": score,
                    "decision": "taken", "vxx_ratio": 1.0,
                    "features": {"spy_vs_ma50": 1.0, "change_pct_today": 1.0},
                    "outcomes": {"+5d": {"label": int(y > 0), "return_pct": y}},
                    "score_breakdown": bd,
                    "ml": {"blended": True, "gated": gated, "rule_weight": 0.7, "ml_weight": 0.3,
                           "rules_only_score": rules_only, "ml_only_score": 60.0},
                })
            day += 1
        d += timedelta(days=1)
    return recs


def selection_records(n_days=80, per_day=10, seed=5, signal=True):
    rng = np.random.default_rng(seed)
    recs, d, day = [], datetime(2026, 5, 4, 14, 0, tzinfo=timezone.utc), 0
    while day < n_days:
        if d.weekday() < 5:
            for k in range(per_day):
                feats = {f: float(rng.normal()) for f, _ in sv.FEATURE_SOURCES}
                y = (0.8 * feats["momentum_3m"] if signal else 0.0) + float(rng.normal())
                recs.append({"ticker": f"S{day:02d}{k}", "timestamp": d.isoformat(),
                             "score": 60 - feats["momentum_3m"], "vxx_ratio": 1.0, "features": feats,
                             "outcomes": {"+5d": {"label": int(y > 0), "return_pct": y}}})
            day += 1
        d += timedelta(days=1)
    return recs


class TestAblatedScore(unittest.TestCase):
    def setUp(self):
        self.rec = ablation_records(n_days=1, per_day=2)[1]   # k=1: not gated

    def test_baseline_reconstructs_logged_score(self):
        for r in ablation_records(n_days=3):
            s, approx = sv.ablated_score(r, ())
            self.assertAlmostEqual(s, r["score"], delta=0.2)
            self.assertFalse(approx)

    def test_removal_uses_rules_raw_then_blend(self):
        bd = self.rec["score_breakdown"]
        want = _blend(max(0.0, min(100.0, bd["rules_raw"] - bd["news"] - bd["social"])))
        self.assertAlmostEqual(sv.ablated_score(self.rec, ("news", "social"))[0], want, places=6)

    def test_gated_ml_blends_fifty(self):
        r = dict(self.rec, ml=dict(self.rec["ml"], gated=True))
        raw = r["score_breakdown"]["rules_raw"]
        self.assertAlmostEqual(sv.ablated_score(r, ())[0], _blend(raw, gated=True), places=6)

    def test_not_blended_is_rules_only(self):
        r = dict(self.rec, ml={"blended": False})
        raw = r["score_breakdown"]["rules_raw"]
        self.assertAlmostEqual(sv.ablated_score(r, ("alt",))[0],
                               max(0.0, min(100.0, raw - r["score_breakdown"]["alt"])), places=6)

    def test_without_rules_raw_falls_back_and_flags_clamp(self):
        bd = {k: v for k, v in self.rec["score_breakdown"].items() if k != "rules_raw"}
        r = dict(self.rec, score_breakdown=bd, ml=dict(self.rec["ml"], rules_only_score=100.0))
        s, approx = sv.ablated_score(r, ())
        self.assertTrue(approx)
        self.assertAlmostEqual(s, _blend(100.0), places=6)

    def test_missing_component_or_breakdown_is_none(self):
        bd = dict(self.rec["score_breakdown"], social=None)
        self.assertIsNone(sv.ablated_score(dict(self.rec, score_breakdown=bd), ("social",))[0])
        self.assertIsNone(sv.ablated_score({"score": 70}, ())[0])


class TestAblationReport(unittest.TestCase):
    def test_removing_inverted_adjustments_uninverts(self):
        rows = sv.build_population(ablation_records(), sv.PRIMARY_OUTCOME)
        rep = sv.ablation_report(rows)
        self.assertEqual(rep["status"], "OK")
        self.assertEqual(rep["dropped_reconstruction_mismatch"], 0)
        v = rep["variants"]
        self.assertLess(v["baseline"]["ic"], 0)
        self.assertGreater(v["drop_unvalidated"]["ic"], 0.5)
        self.assertTrue(rep["baseline_inverted"])
        self.assertTrue(rep["un_inverts"])
        self.assertTrue(rep["improvement_significant"])
        self.assertLess(v["baseline"]["quintiles"]["Q5_minus_Q1"], 0)
        self.assertGreater(v["drop_unvalidated"]["quintiles"]["Q5_minus_Q1"], 0)
        for k in UNVAL:
            self.assertIn(f"drop_{k}", v)
        self.assertIn("drop_unvalidated_and_boosts", v)   # extras logged on these records

    def test_honest_null_when_adjustments_are_not_inverted(self):
        rep = sv.ablation_report(sv.build_population(ablation_records(inverted=False), "+5d"))
        self.assertGreater(rep["variants"]["baseline"]["ic"], 0)
        self.assertFalse(rep["un_inverts"])

    def test_reconstruction_mismatch_dropped_and_counted(self):
        recs = ablation_records()
        for r in recs[:25]:
            r["score"] = r["score"] + 5.0
        rep = sv.ablation_report(sv.build_population(recs, "+5d"))
        self.assertEqual(rep["dropped_reconstruction_mismatch"], 25)

    def test_old_records_without_extras_skip_boost_variant(self):
        recs = ablation_records(log_raw=False)
        for r in recs:
            for k in ("recommendation", "contrarian", "rel_strength", "insider_mspr"):
                r["score_breakdown"].pop(k, None)
        rep = sv.ablation_report(sv.build_population(recs, "+5d"))
        self.assertEqual(rep["status"], "OK")
        self.assertNotIn("drop_unvalidated_and_boosts", rep["variants"])
        self.assertTrue(rep["un_inverts"])

    def test_insufficient_below_min_n(self):
        rep = sv.ablation_report(sv.build_population(ablation_records(n_days=5), "+5d"))
        self.assertEqual(rep["status"], "INSUFFICIENT")


class TestForwardSelection(unittest.TestCase):
    def test_day_split_is_time_ordered_and_disjoint(self):
        rows = sv.build_population(selection_records(n_days=20, per_day=3), "+5d")
        fit, sel, hold = sv._day_split(rows)
        fd, sd, hd = ({r["day"] for r in p} for p in (fit, sel, hold))
        self.assertFalse(fd & sd or sd & hd or fd & hd)
        self.assertLess(max(fd), min(sd))
        self.assertLess(max(sd), min(hd))
        self.assertEqual(len(fit) + len(sel) + len(hold), len(rows))

    def test_selects_the_planted_source_and_scores_holdout_once(self):
        recs = selection_records()
        out = sv.selection_and_ablation(recs, sv.validate_sources(recs))
        fs = out["selection"]["features"]
        self.assertEqual(fs["status"], "SELECTED", fs)
        self.assertEqual(fs["selected"][0], "feat.momentum_3m")
        self.assertLessEqual(len(fs["selected"]), 2)        # noise rarely survives Bonferroni
        self.assertGreater(fs["holdout_selected"]["ic"], 0.3)
        self.assertLess(fs["holdout_live_score"]["ic"], 0)  # the planted live score is inverted
        self.assertGreater(fs["holdout_vs_live_score"]["ic_gain"], 0.5)
        self.assertGreater(fs["weights"]["feat.momentum_3m"], 0)
        self.assertEqual(out["selection"]["breakdown"]["status"], "INSUFFICIENT")

    def test_pure_noise_selects_nothing(self):
        recs = selection_records(signal=False)
        fs = sv.forward_selection(sv.build_population(recs, "+5d"),
                                  [s for s in sv.source_registry() if s["kind"] == "feature"])
        self.assertEqual(fs["status"], "NOTHING_SELECTED", fs.get("steps"))
        self.assertFalse(fs["steps"][0]["accepted"])
        self.assertEqual(fs["steps"][0]["tried"], len(sv.FEATURE_SOURCES))

    def test_too_few_rows(self):
        fs = sv.forward_selection(sv.build_population(selection_records(n_days=10), "+5d"),
                                  sv.source_registry())
        self.assertEqual(fs["status"], "INSUFFICIENT")


class TestProductionReport(unittest.TestCase):
    def test_probe_report_carries_selection_and_ablation_within_sanitizer_limits(self):
        recs = ablation_records() + selection_records(n_days=30)
        with mock.patch("shadow_portfolio._load_shadow_log", return_value=recs):
            rep = sv.get_source_validation_report()
        for k in ("sources", "summary", "selection", "ablation", "population"):
            self.assertIn(k, rep)
        blob = json.dumps(rep)
        self.assertNotIn('"ticker"', blob)
        self.assertNotIn("A00", blob)

        def depth(o, d=0):
            if isinstance(o, dict):
                self.assertLessEqual(len(o), 100)
                return max([depth(v, d + 1) for v in o.values()] or [d])
            if isinstance(o, list):
                return max([depth(v, d + 1) for v in o] or [d])
            return d
        self.assertLessEqual(depth(rep), 8)


class TestShadowOnlyComponentLogging(unittest.TestCase):
    def test_extras_persisted_by_sanitizer(self):
        bd = {"momentum": 60.0, "recommendation": 8.0, "contrarian": 4.0, "rel_strength": 6.0,
              "insider_mspr": -5.0, "rules_raw": 104.3}
        out = sp._sanitize_score_breakdown(bd)
        for k in sp.SCORE_BREAKDOWN_EXTRA_KEYS:
            self.assertEqual(out[k], bd[k])
        self.assertIsNone(sp._sanitize_score_breakdown({"momentum": 1.0})["rules_raw"])

    def test_deep_score_logs_the_extras_without_touching_the_score(self):
        src = open(os.path.join(HERE, "bot_engine.py")).read()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "deep_score")
        seg = ast.get_source_segment(src, fn)
        for k, var in (("recommendation", "recommendation_adjustment"), ("contrarian", "contrarian_adjustment"),
                       ("rel_strength", "rel_strength_adjustment"), ("insider_mspr", "insider_mspr_adjustment"),
                       ("rules_raw", "rules_raw_score")):
            self.assertIn(f'"{k}": round({var}, 1)', seg)
        new_vars = {"recommendation_adjustment", "contrarian_adjustment", "rel_strength_adjustment",
                    "insider_mspr_adjustment", "rules_raw_score"}
        # shadow-only: none of them ever flows into combined_score
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "combined_score" for t in targets):
                    used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                    self.assertFalse(used & new_vars, ast.get_source_segment(src, node))


if __name__ == "__main__":
    unittest.main()
