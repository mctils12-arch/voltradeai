"""PRODUCT 2026-10-16 (bot page v2): ml_insight summarizes what the model is
doing from persisted fields only; absent values are None, never zeros."""
import unittest

from ml_insight import recent_scoring, CONSTANT_FRACTION


def _rec(i, ml=None, vxx=1.0, f=None):
    feats = {"momentum_1m": float(i) / 10, "vxx_ratio": vxx, "range_pct": 0.0}
    if f:
        feats.update(f)
    return {"timestamp": f"2026-10-16T00:00:{i:02d}Z", "score": 60 + i, "features": feats,
            "vxx_ratio": vxx, "regime_label": "NEUTRAL_BULL", "ml": ml}


class TestRecentScoring(unittest.TestCase):
    def test_empty(self):
        r = recent_scoring([])
        self.assertEqual(r["window"], 0)
        self.assertEqual(r["constant_features"], [])
        self.assertIsNone(r["win_probability"]["p50"])
        self.assertIsNone(r["vxx_ratio"]["last"])

    def test_trace_histogram_and_constancy(self):
        recs = [_rec(i, ml={"model_type": "lightgbm_neutral", "regime_used": "neutral", "win_probability": 0.5 + i / 100,
                             "gated": i % 2 == 0, "blended": True}) for i in range(10)]
        recs.append(_rec(10, ml={"model_type": "fallback_rules", "regime_used": "fallback", "blended": False}))
        recs.append(_rec(11, ml={"error": "ImportError: x", "blended": False}))
        recs.append(_rec(12))  # older caller: no trace
        r = recent_scoring(recs)
        self.assertEqual(r["window"], 13)
        self.assertEqual(r["traced"], 12)
        self.assertEqual(r["model_types"]["lightgbm_neutral"], 10)
        self.assertEqual(r["model_types"]["fallback_rules"], 1)
        self.assertEqual(r["model_types"]["error"], 1)
        self.assertEqual(r["model_types"]["not recorded"], 1)
        self.assertEqual(r["regime_used"], {"neutral": 10, "fallback": 1})
        self.assertEqual(r["gated"], 5)
        self.assertEqual(r["blended"], 10)
        self.assertEqual(r["errors"], 1)
        self.assertEqual(r["win_probability"]["n"], 10)
        self.assertIn(r["win_probability"]["p50"], (0.54, 0.55))
        # vxx_ratio and range_pct constant across the window; momentum is not
        self.assertEqual(sorted(r["constant_features"]), ["range_pct", "vxx_ratio"])
        self.assertEqual(r["feature_count"], 3)
        self.assertEqual(r["vxx_ratio"]["last"], 1.0)
        self.assertEqual(r["last_score"], 72)

    def test_window_is_the_newest_records(self):
        recs = [_rec(i % 60, vxx=1.0 if i < 500 else 1.2) for i in range(600)]
        r = recent_scoring(recs, window=100)
        self.assertEqual(r["window"], 100)
        self.assertEqual(r["vxx_ratio"]["min"], 1.2)

    def test_constant_threshold(self):
        recs = [_rec(i, f={"x": 1.0 if i < 99 else 2.0}) for i in range(100)]
        r = recent_scoring(recs)
        row = next(f for f in r["features"] if f["name"] == "x")
        self.assertGreaterEqual(row["mode_fraction"], CONSTANT_FRACTION)
        self.assertTrue(row["constant"])


if __name__ == "__main__":
    unittest.main()


# ── [REPAIR] 2026-11-03 (approved item B3, serve_features.py) ────────────────
# The constancy MEASURE, split by what it can honestly judge. The fixture
# mirrors the live Issues-panel reading ("12 / 34 never move"): six per-ticker
# features pinned at deep_score's constants, credit_spread pinned at 0.0 across
# days, the five training-constant features at their training constants, and
# the other market-wide features moving a little within the day (the macro
# snapshot refreshes) — so the RAW count is 12 and the honest count is 7.
import random as _random

from ml_model_v2 import FEATURE_COLS as _COLS
from ml_insight import recent_scoring as _rs, fixed_vector as _fixed_vector
import serve_features as _sf

_LIVE_CONSTANTS = {"price_vs_52w_high": 0.0, "ewma_vol": 2.0, "range_pct": 0.0, "cross_sec_rank": 0.5,
                   "sector_momentum": 0.0, "trend_strength": 5.0, "credit_spread": 0.0,
                   "intel_score": 0.0, "news_sentiment": 0.0, "insider_signal": 0.0,
                   "days_to_earnings": 0.5, "earnings_surprise": 0.0}


def _live_like_records(n_recent=300, n_wide=2500, days_wide=5, with_fix=False, fix_mode="shadow", seed=3,
                       fix_from=None):
    """Older `n_wide` records across `days_wide` days, then `n_recent` on the newest day.
    with_fix attaches a serve_fix block from record index `fix_from` (default:
    the newest day only, i.e. a fix deployed today)."""
    rng = _random.Random(seed)
    recs = []
    total = n_wide + n_recent
    for i in range(total):
        day_i = (i * days_wide) // n_wide if i < n_wide else days_wide   # the newest day is day index days_wide
        feats = {}
        for c in _COLS:
            if c in _LIVE_CONSTANTS:
                feats[c] = _LIVE_CONSTANTS[c]
            elif c in _sf.MARKET_WIDE_FEATURES:
                feats[c] = round(1.0 + 0.1 * day_i + 0.001 * ((i // 5) % 7), 6)   # one value per ~5 records, moves by day
            else:
                feats[c] = round(rng.uniform(-10, 10), 4)
        rec = {"timestamp": f"2026-10-{20 + day_i:02d}T{10 + (i % 6):02d}:{(i % 60):02d}:00+00:00", "score": 60.0,
               "features": feats, "vxx_ratio": feats["vxx_ratio"], "regime_label": "NEUTRAL",
               "ml": {"model_type": "lightgbm_global", "win_probability": 0.5, "blended": True}}
        if with_fix and i >= (n_wide if fix_from is None else fix_from):
            diff = {k: round(rng.uniform(0.1, 9.0), 4) for k in _sf.FIXED_FEATURES}
            if fix_mode == "on":
                rec["features"] = {**feats, **diff}
                rec["serve_fix"] = {"mode": "on", "served": "fixed", "features_fixed_diff": diff,
                                    "features_current_diff": {k: _LIVE_CONSTANTS[k] for k in diff},
                                    "ml_prob_current": 0.5, "ml_prob_fixed": 0.5 + 0.01 * (i % 5)}
            else:
                rec["serve_fix"] = {"mode": "shadow", "served": "current", "features_fixed_diff": diff,
                                    "features_current_diff": {k: _LIVE_CONSTANTS[k] for k in diff},
                                    "ml_prob_current": 0.5, "ml_prob_fixed": 0.5 + 0.01 * (i % 5)}
        recs.append(rec)
    return recs


class TestHonestConstancyMeasure(unittest.TestCase):
    def test_raw_count_is_unchanged_and_the_honest_split_adds_up(self):
        r = _rs(_live_like_records())
        self.assertEqual(r["window"], 300)
        self.assertEqual(r["feature_count"], 34)
        # BEFORE (the ruler the panel reads today): 12 / 34
        self.assertEqual(sorted(r["constant_features"]), sorted(_LIVE_CONSTANTS))
        # AFTER: 7 serve-path defects — 6 per-ticker + credit_spread, constant ACROSS DAYS
        self.assertEqual(sorted(r["constant_features_serve_bug"]),
                         sorted(["price_vs_52w_high", "ewma_vol", "range_pct", "cross_sec_rank",
                                 "sector_momentum", "trend_strength", "credit_spread"]))
        self.assertEqual(sorted(r["constant_features_training_constant"]), sorted(_sf.TRAINING_CONSTANT_FEATURES))
        self.assertEqual(r["constant_features_market_wide_window"], ["credit_spread"])
        self.assertEqual(r["judgeable_feature_count"], 34 - 5)
        self.assertEqual(r["window_days"], 1)
        self.assertEqual(r["market_wide_days"], 6)
        self.assertEqual(r["market_wide_window"], 2800)
        # no serve_fix records -> the fixed count is unknown, never a fabricated 0
        self.assertIsNone(r["constant_features_fixed"])
        self.assertIsNone(r["constant_features_fixed_unknown"])
        self.assertEqual(r["serve_fix"]["records"], 0)
        rows = {f["name"]: f for f in r["features"]}
        self.assertEqual(rows["credit_spread"]["kind"], "market_wide")
        self.assertTrue(rows["credit_spread"]["judgeable"])
        self.assertTrue(rows["credit_spread"]["constant_bug"])
        self.assertEqual(rows["credit_spread"]["distinct_values_across_days"], 1)
        self.assertEqual(rows["vxx_ratio"]["kind"], "market_wide")
        self.assertFalse(rows["vxx_ratio"]["constant_bug"])
        self.assertGreater(rows["vxx_ratio"]["distinct_values_across_days"], 5)
        self.assertEqual(rows["intel_score"]["kind"], "training_constant")
        self.assertFalse(rows["intel_score"]["judgeable"])
        self.assertTrue(rows["intel_score"]["constant"])
        self.assertFalse(rows["intel_score"]["constant_bug"])
        self.assertEqual(rows["range_pct"]["kind"], "per_ticker")
        self.assertTrue(rows["range_pct"]["judgeable"])
        self.assertTrue(rows["range_pct"]["constant_bug"])

    def test_market_wide_constant_within_one_day_is_not_a_bug_without_a_second_day(self):
        recs = _live_like_records(n_wide=0, days_wide=1)   # one day only, nothing to judge across
        for rec in recs:
            rec["features"]["vxx_ratio"] = 1.0        # pinned within the window, as every scan would show
        r = _rs(recs)
        self.assertIn("vxx_ratio", r["constant_features"])                 # raw: constant in the window
        self.assertIn("vxx_ratio", r["constant_features_market_wide_window"])
        self.assertNotIn("vxx_ratio", r["constant_features_serve_bug"])    # honest: unjudgeable
        row = next(f for f in r["features"] if f["name"] == "vxx_ratio")
        self.assertFalse(row["judgeable"])
        self.assertEqual(row["days_across"], 1)
        self.assertEqual(r["judgeable_feature_count"], 34 - 5 - 8)

    def test_market_wide_pinned_across_days_is_a_bug(self):
        recs = _live_like_records()
        for rec in recs:
            rec["features"]["vxx_ratio"] = 1.0        # the pre-2026-10-16 pin, across six days
        r = _rs(recs)
        self.assertIn("vxx_ratio", r["constant_features_serve_bug"])

    def test_fixed_count_comes_from_the_recorded_corrected_vectors(self):
        r = _rs(_live_like_records(with_fix=True))
        self.assertEqual(r["serve_fix"]["records"], 300)
        self.assertEqual(r["serve_fix"]["modes"], {"shadow": 300})
        self.assertEqual(r["serve_fix"]["served"], {"current": 300})
        self.assertEqual(r["serve_fix"]["errors"], 0)
        self.assertEqual(r["serve_fix"]["prob_delta"]["n"], 300)
        self.assertAlmostEqual(r["serve_fix"]["prob_delta"]["mean"], 0.02, places=3)
        # the served vectors still carry the constants (shadow mode) ...
        self.assertEqual(len(r["constant_features_serve_bug"]), 7)
        # ... the corrected vectors move all six per-ticker ones ...
        self.assertEqual(r["constant_features_fixed"], [])
        rows = {f["name"]: f for f in r["features"]}
        for name in _sf.FIXED_FEATURES:
            if name != "credit_spread":
                self.assertIs(rows[name]["fixed_constant"], False, name)
        # ... and credit_spread (market-wide) is honestly UNKNOWN until the fix
        # has recorded corrected values on a second day — never assumed fixed
        self.assertEqual(r["constant_features_fixed_unknown"], ["credit_spread"])
        self.assertIsNone(rows["credit_spread"]["fixed_constant"])

    def test_market_wide_fixed_verdict_needs_two_days_of_corrected_values(self):
        r = _rs(_live_like_records(with_fix=True, fix_from=0))   # the fix has run across all six days
        self.assertEqual(r["constant_features_fixed"], [])
        self.assertEqual(r["constant_features_fixed_unknown"], [])
        row = next(f for f in r["features"] if f["name"] == "credit_spread")
        self.assertIs(row["fixed_constant"], False)

    def test_on_mode_records_hold_the_corrected_vector_directly(self):
        recs = _live_like_records(with_fix=True, fix_mode="on")
        r = _rs(recs)
        # served = fixed: the raw measure now sees the corrected per-ticker values move
        for name in ("range_pct", "ewma_vol", "cross_sec_rank", "price_vs_52w_high", "sector_momentum", "trend_strength"):
            self.assertNotIn(name, r["constant_features"], name)
        self.assertEqual(r["serve_fix"]["served"], {"fixed": 300})
        self.assertEqual(_fixed_vector(recs[-1]), recs[-1]["features"])

    def test_fixed_vector_helper(self):
        rec = {"features": {"a": 1.0, "b": 2.0}, "serve_fix": {"served": "current", "features_fixed_diff": {"b": 3.0}}}
        self.assertEqual(_fixed_vector(rec), {"a": 1.0, "b": 3.0})
        self.assertIsNone(_fixed_vector({"features": {"a": 1.0}}))
        self.assertIsNone(_fixed_vector({"features": {"a": 1.0}, "serve_fix": {"served": "current", "error": "x"}}))

    def test_a_fix_that_still_leaves_a_feature_constant_is_reported_not_hidden(self):
        recs = _live_like_records(with_fix=True)
        for rec in recs:
            if "serve_fix" in rec:
                rec["serve_fix"]["features_fixed_diff"]["range_pct"] = 0.7   # corrected, but to one value
        r = _rs(recs)
        self.assertEqual(r["constant_features_fixed"], ["range_pct"])

    def test_errored_fix_records_do_not_count_as_corrected_vectors(self):
        recs = _live_like_records(with_fix=True)
        for rec in recs:
            if "serve_fix" in rec:
                rec["serve_fix"]["error"] = "RuntimeError: boom"
        r = _rs(recs)
        self.assertEqual(r["serve_fix"]["errors"], 300)
        self.assertEqual(r["constant_features_fixed"], [])
        self.assertEqual(len(r["constant_features_fixed_unknown"]), 7)
