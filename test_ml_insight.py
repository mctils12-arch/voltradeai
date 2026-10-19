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
