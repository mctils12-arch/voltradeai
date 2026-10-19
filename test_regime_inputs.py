"""
REPAIR 2026-10-16 (bot cockpit audit S3): serve-time regime features must
be computed with the training formulas from the macro snapshot, never from
intelligence.py (which has no vxx_ratio) or an empty return list.
"""
import unittest

from regime_inputs import regime_inputs, MARKOV_RETURN_THRESHOLD


def _macro(closes, vxx=1.12, spy_ma=1.03, with_vxx_series=True):
    return {
        "vxx_ratio": vxx, "vxx_latest": 20.0 * vxx, "vxx_avg30": 20.0,
        "vxx_closes": [20.0] * 30 if with_vxx_series else [],
        "spy_vs_ma50": spy_ma, "spy_closes_60d": closes,
    }


class TestRegimeInputs(unittest.TestCase):
    def test_vxx_comes_from_macro_not_pinned(self):
        r = regime_inputs(_macro([100, 101, 102, 103, 104, 105], vxx=1.27))
        self.assertAlmostEqual(r["vxx_ratio"], 1.27)
        self.assertEqual(r["vxx_source"], "macro")

    def test_default_when_macro_has_no_vxx_series(self):
        r = regime_inputs({"vxx_ratio": 1.0, "vxx_closes": [], "spy_closes_60d": []})
        self.assertEqual(r["vxx_ratio"], 1.0)
        self.assertEqual(r["vxx_source"], "default")
        self.assertEqual(r["markov_state"], 1.0)
        self.assertEqual(r["spy_closes_n"], 0)

    def test_markov_state_uses_training_formula_on_last_five_returns(self):
        up = [100 * (1 + 0.003) ** i for i in range(8)]       # mean +0.3% > +0.2%
        down = [100 * (1 - 0.003) ** i for i in range(8)]     # mean −0.3% < −0.2%
        flat = [100, 100.1, 99.9, 100.05, 99.95, 100.0]       # within ±0.2%
        self.assertEqual(regime_inputs(_macro(up))["markov_state"], 2.0)
        self.assertEqual(regime_inputs(_macro(down))["markov_state"], 0.0)
        self.assertEqual(regime_inputs(_macro(flat))["markov_state"], 1.0)
        r = regime_inputs(_macro(up))
        self.assertEqual(len(r["spy_returns_5"]), 5)
        self.assertEqual(len(r["spy_returns"]), 7)
        self.assertAlmostEqual(r["spy_returns_5"][-1], 0.003, places=6)

    def test_fewer_than_three_returns_is_neutral(self):
        self.assertEqual(regime_inputs(_macro([100, 101]))["markov_state"], 1.0)

    def test_regime_score_is_the_training_closed_form(self):
        # (spy_ma−0.94)/0.12×50 + (1.3−vxx)/0.6×30 + 20, clipped to [0, 100]
        r = regime_inputs(_macro([100] * 6, vxx=1.0, spy_ma=1.0))
        self.assertAlmostEqual(r["regime_score"], (1.0 - 0.94) / 0.12 * 50 + (1.3 - 1.0) / 0.6 * 30 + 20, places=2)
        self.assertEqual(regime_inputs(_macro([100] * 6, vxx=2.5, spy_ma=0.80))["regime_score"], 0.0)
        self.assertEqual(regime_inputs(_macro([100] * 6, vxx=0.6, spy_ma=1.20))["regime_score"], 100.0)

    def test_bull_is_reachable_once_vxx_is_real(self):
        """The audit's headline: with vxx pinned at 1.0 classify_regime could
        never return bull. With the real ratio it can."""
        from ml_model_v2 import _classify_regime
        pinned = regime_inputs({"vxx_ratio": 1.0, "spy_vs_ma50": 1.05, "spy_closes_60d": []})
        real = regime_inputs(_macro([100] * 6, vxx=0.90, spy_ma=1.05))
        self.assertNotEqual(_classify_regime(pinned["vxx_ratio"], pinned["spy_vs_ma50"]), "bull")
        self.assertEqual(_classify_regime(real["vxx_ratio"], real["spy_vs_ma50"]), "bull")

    def test_garbage_is_tolerated(self):
        for bad in (None, "x", 3, {"vxx_ratio": "n/a", "spy_vs_ma50": None, "spy_closes_60d": [None, "9", -1, 0, 101.0]}):
            r = regime_inputs(bad)
            self.assertIn(r["markov_state"], (0.0, 1.0, 2.0))
            self.assertGreaterEqual(r["regime_score"], 0.0)
        self.assertEqual(regime_inputs({"spy_closes_60d": [None, "9", -1, 0, 101.0]})["spy_closes_n"], 2)


if __name__ == "__main__":
    unittest.main()


class TestDeepScoreWiring(unittest.TestCase):
    """Wiring pin: deep_score must take its regime block from regime_inputs(macro)
    and never again from intelligence.py or an empty return list."""
    def setUp(self):
        with open("bot_engine.py", encoding="utf-8") as f:
            self.src = f.read()

    def test_regime_inputs_is_used_for_the_four_regime_features(self):
        self.assertIn("_ri = regime_inputs(macro)", self.src)
        for k in ("vxx_ratio", "spy_vs_ma50", "markov_state", "regime_score"):
            self.assertRegex(self.src, r'"%s":\s+_ri\["%s"\]' % (k, k))

    def test_the_old_sources_are_gone(self):
        self.assertNotIn('intel.get("vxx_ratio"', self.src)
        self.assertNotIn('_spy_rets = []', self.src)
        self.assertNotIn('"spy_returns_5d"', self.src)

    def test_markov_detector_gets_the_real_returns_and_vxx(self):
        self.assertIn('_get_markov_regime(_ri["spy_returns"], float(_ri["vxx_ratio"]), float(_ri["spy_vs_ma50"]))', self.src)
