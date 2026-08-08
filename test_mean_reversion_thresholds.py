"""
test_mean_reversion_thresholds.py — pins strategies/mean_reversion.score()'s
default-path behavior (no `thresholds` kwarg) byte-identical to the
pre-parametrization hardcoded version, and pins the override mechanism
itself. Written alongside the DEFAULT_THRESHOLDS refactor (research/
open_questions.md, 2026-07-24 entry, LADDER PATH step 4 prerequisite) —
the refactor's entire point is "zero behavior change for any existing
caller"; this is the evidence for that claim, not an assumption of it.
"""
import unittest

from strategies import mean_reversion


def _pre_refactor_score(rsi, change_pct_5d, volume_ratio):
    """Verbatim copy of the original hardcoded score() body (before the
    DEFAULT_THRESHOLDS parametrization), kept ONLY as an independent
    oracle for the byte-identical-output test below — never imported
    elsewhere, never the "real" implementation."""
    if rsi is None:
        return {"score": 0, "signal": "NO DATA", "reason": "No data"}
    s = 0
    if rsi < 20: s += 40
    elif rsi < 30: s += 30
    elif rsi < 40: s += 15
    elif rsi > 70: s -= 15
    if change_pct_5d and change_pct_5d < -10: s += 30
    elif change_pct_5d and change_pct_5d < -5: s += 20
    elif change_pct_5d and change_pct_5d < -3: s += 10
    if volume_ratio and volume_ratio > 2: s += 20
    elif volume_ratio and volume_ratio > 1.5: s += 10
    s = max(0, min(100, s))
    sig = "STRONG BUY" if s >= 65 else "BUY" if s >= 45 else "WATCH" if s >= 25 else "NO EDGE"
    return {"score": s, "signal": sig,
            "reason": f"RSI: {rsi:.0f}, 5d drop: {(change_pct_5d or 0):.1f}%, vol: {(volume_ratio or 1):.1f}x"}


# Representative grid spanning every branch of every band (RSI extreme/
# oversold/mild/neutral/overbought x big/med/small/no drop x
# high/med/no volume spike, plus the None-rsi short-circuit).
CASES = [
    (None, -5, 1.0),
    (15, -12, 2.5), (15, -12, 1.6), (15, -12, 0.8),
    (25, -7, 2.5), (25, -3.5, 1.6), (25, 1, 0.8),
    (35, -4, 2.5), (35, None, None),
    (55, -2, 0.5),
    (75, -10, 3.0), (75, 0, 0.5),
    (20, -10, 2.0),  # boundary values (strict `<`/`>` per original code)
    (30, -5, 1.5),
    (40, -3, 1.0),
    (70, 0, 1.0),
]


class TestDefaultPathUnchanged(unittest.TestCase):
    def test_matches_pre_refactor_oracle_on_representative_grid(self):
        for rsi, chg, vr in CASES:
            with self.subTest(rsi=rsi, chg=chg, vr=vr):
                self.assertEqual(mean_reversion.score(rsi, chg, vr),
                                  _pre_refactor_score(rsi, chg, vr))

    def test_thresholds_none_identical_to_omitted(self):
        for rsi, chg, vr in CASES:
            with self.subTest(rsi=rsi, chg=chg, vr=vr):
                self.assertEqual(mean_reversion.score(rsi, chg, vr),
                                  mean_reversion.score(rsi, chg, vr, thresholds=None))

    def test_empty_override_dict_identical_to_omitted(self):
        for rsi, chg, vr in CASES:
            with self.subTest(rsi=rsi, chg=chg, vr=vr):
                self.assertEqual(mean_reversion.score(rsi, chg, vr),
                                  mean_reversion.score(rsi, chg, vr, thresholds={}))


class TestOverrideMechanism(unittest.TestCase):
    def test_partial_override_only_changes_named_keys(self):
        # Only tighten rsi_extreme (20 -> 15); rsi=18 no longer qualifies
        # for the extreme band (40pts) and should fall through to the
        # oversold band (30pts) instead — chg/vr bands untouched.
        base = mean_reversion.score(18, 0, 0)
        tightened = mean_reversion.score(18, 0, 0, thresholds={"rsi_extreme": 15})
        self.assertEqual(base["score"], 40)
        self.assertEqual(tightened["score"], 30)

    def test_unknown_extra_key_ignored_not_erroring(self):
        # merge semantics: {**DEFAULT_THRESHOLDS, **thresholds} silently
        # accepts extra keys (never read) rather than raising — documents
        # the actual (permissive) contract so a future caller isn't
        # surprised by a silent typo swallowing an intended override.
        r = mean_reversion.score(18, 0, 0, thresholds={"not_a_real_key": 999})
        self.assertEqual(r["score"], mean_reversion.score(18, 0, 0)["score"])

    def test_full_variant_dict_produces_different_signal_class(self):
        deep_drop_variant = {"chg_big": -18, "chg_med": -10, "chg_small": -6}
        # A -8% 5d drop clears the default "med" band (-5, 20pts) but only
        # the deepened variant's "small" band (-6, 10pts) -> fewer points
        # under the variant, same drop magnitude.
        default_r = mean_reversion.score(50, -8, 1.0)
        variant_r = mean_reversion.score(50, -8, 1.0, thresholds=deep_drop_variant)
        self.assertEqual(default_r["score"], 20)
        self.assertEqual(variant_r["score"], 10)


if __name__ == "__main__":
    unittest.main()
