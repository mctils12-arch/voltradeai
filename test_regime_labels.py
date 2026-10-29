"""
Legacy-label subordination (docs/regime/FINDINGS.md step 7): under
VOLTRADE_REGIME_LABELS the engine's legacy label (BULL / NEUTRAL / CAUTION / BEAR /
PANIC) replaces the Markov label wherever a LABEL is persisted or acted on in the
Python engine (the shadow record, regime_at_entry, the tier context); the ML's
markov_state / regime_score are untouched. Flag off, engine off or a stale state:
the Markov label stands. Pinned on the engine's pure function and on the source.
"""
import os
import unittest

import regime_engine as re_

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SRC = open(os.path.join(ROOT, "bot_engine.py")).read()
TODAY = "2026-10-28"
ENV_ON = {"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_LABELS": "1"}


def state(**over):
    base = {"as_of": "2026-10-27", "composite_regime": "STRESSED", "legacy_label": "CAUTION"}
    base.update(over)
    return base


class TestFlag(unittest.TestCase):
    def test_flag_parsing(self):
        self.assertTrue(re_.labels_enabled({}))   # 2026-10-28 human directive: env absent -> BASE_CONFIG default ON (was off)
        self.assertFalse(re_.labels_enabled({"VOLTRADE_REGIME_LABELS": "0"}))
        for v in ("1", "true", "ON", "yes"):
            self.assertTrue(re_.labels_enabled({"VOLTRADE_REGIME_LABELS": v}))


class TestLabelFor(unittest.TestCase):
    def setUp(self):
        re_._STATE_CACHE.update(at=0.0, state=None)

    tearDown = setUp

    def test_none_unless_engine_and_flag_are_on(self):
        self.assertIsNone(re_.label_for({"VOLTRADE_REGIME_ENGINE": "0", "VOLTRADE_REGIME_LABELS": "1"}, loader=state, today=TODAY))
        self.assertIsNone(re_.label_for({"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_LABELS": ""}, loader=state, today=TODAY))

    def test_fresh_state_gives_the_legacy_label(self):
        r = re_.label_for(ENV_ON, loader=state, today=TODAY)
        self.assertEqual(r, {"label": "CAUTION", "composite": "STRESSED", "as_of": "2026-10-27"})

    def test_label_falls_back_to_the_decode_of_the_composite(self):
        re_._STATE_CACHE.update(at=0.0, state=None)
        r = re_.label_for(ENV_ON, loader=lambda: state(legacy_label=None, composite_regime="CRISIS"), today=TODAY)
        self.assertEqual(r["label"], "PANIC")
        for composite, label in (("CALM_BULL", "BULL"), ("CHOPPY", "NEUTRAL"), ("VOLATILE_BULL", "CAUTION"), ("BEAR", "BEAR")):
            self.assertEqual(re_.legacy_label(composite), label)
        self.assertIn(re_.legacy_label("CHOPPY"), re_.LEGACY_LABELS)

    def test_stale_failed_or_missing_state_means_none(self):
        for st in (state(as_of="2026-10-20"), state(error="boom"), state(stale=True), None, {}):
            re_._STATE_CACHE.update(at=0.0, state=None)
            self.assertIsNone(re_.label_for(ENV_ON, loader=lambda st=st: st, today=TODAY), st)

    def test_unknown_label_is_never_passed_through(self):
        re_._STATE_CACHE.update(at=0.0, state=None)
        self.assertIsNone(re_.label_for(ENV_ON, loader=lambda: state(legacy_label="NEUTRAL_BULL", composite_regime="WHATEVER"), today=TODAY))


class TestWiring(unittest.TestCase):
    def test_deep_score_overrides_the_markov_label_only_after_the_markov_call(self):
        i_markov = BOT_SRC.index("_regime_ctx = _get_markov_regime(_ri[\"spy_returns\"]")
        i_override = BOT_SRC.index('_regime_ctx = {**_regime_ctx, "regime_label": _engine_lbl["label"], "regime_label_source": "engine",')
        i_features = BOT_SRC.index("# 34 features — ALL computed from real data")
        self.assertLess(i_markov, i_override)
        self.assertLess(i_override, i_features)
        self.assertIn("_engine_lbl = _re_lbl.label_for()", BOT_SRC)
        # the Markov label is kept beside the override so a read can compare the two
        self.assertIn('"regime_label_markov": _regime_ctx.get("regime_label")', BOT_SRC)

    def test_the_label_consumers_read_regime_ctx_unchanged(self):
        # shadow record, candidate -> regime_at_entry, tier context: all still read regime_ctx["regime_label"]
        self.assertIn('quick_result["regime_label"] = _regime_ctx.get("regime_label", "NEUTRAL")', BOT_SRC)
        self.assertIn('regime_label=_regime_ctx.get("regime_label", "NEUTRAL") if \'_regime_ctx\' in locals() else "NEUTRAL",', BOT_SRC)
        self.assertIn('"regime_at_entry": stock.get("regime_label", "UNKNOWN"),', BOT_SRC)
        self.assertIn('regime=_scan_engine_label or (_regime_ctx.get("regime_label", "NEUTRAL") if \'_regime_ctx\' in locals() else "NEUTRAL"),', BOT_SRC)

    def test_ml_features_keep_the_markov_definitions(self):
        self.assertIn('"markov_state":         _ri["markov_state"],', BOT_SRC)
        self.assertIn('"regime_score":         _ri["regime_score"],', BOT_SRC)


if __name__ == "__main__":
    unittest.main()
