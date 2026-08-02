"""
[REPAIR 2026-08-02] Regression test for _inline_train_allowed (bot_engine.py),
the gate that stops bot_engine.py's __main__ block from importing lightgbm +
sklearn (~150MB RSS) and calling train_model() on every scan/full/manage
invocation.

BACKGROUND: this gate was originally shipped inline (MEM FIX 2026-04-21,
env var VOLTRADE_INLINE_ML_TRAIN) fixing a real Tier2-every-minute OOM
SIGKILL, but had zero regression coverage — nothing in CI would fail if a
future edit silently reintroduced an unconditional train_model() call in
__main__. Found this session while investigating stranded PR #77
("fix/tier2-full-scan-oom", open since 2026-04-20 with zero CI runs ever
triggered — the recurring zero-CI-aging bug tracked in wishlist.md): PR #77
proposed the same fix under a different env var name, but the underlying
bug had already been independently fixed a day after PR #77 was opened, so
PR #77 itself is stale and superseded, not re-applied. This test is the one
concrete gap that investigation surfaced: pin the gate that already ships,
so a regression fails loudly instead of silently reintroducing the OOM.

The gate itself was extracted from __main__ into a standalone pure function
(_inline_train_allowed) in the same PR as this test, with no behavior change
(mode == "train" or env VOLTRADE_INLINE_ML_TRAIN == "1", identical to the
prior inline logic) — purely to make it importable and testable here.

Run: python3 -m pytest test_inline_train_gate.py -q
"""
import unittest

from bot_engine import _inline_train_allowed


class TestInlineTrainGateDefaultOff(unittest.TestCase):
    def test_scan_mode_with_no_env_is_gated_off(self):
        """The Tier2-every-minute case that caused the original OOM: mode
        'scan', no env override. Must stay off by default."""
        self.assertFalse(_inline_train_allowed("scan", {}))

    def test_full_mode_with_no_env_is_gated_off(self):
        self.assertFalse(_inline_train_allowed("full", {}))

    def test_manage_mode_with_no_env_is_gated_off(self):
        self.assertFalse(_inline_train_allowed("manage", {}))

    def test_env_flag_set_to_zero_stays_gated_off(self):
        self.assertFalse(_inline_train_allowed("scan", {"VOLTRADE_INLINE_ML_TRAIN": "0"}))

    def test_unset_and_garbage_env_values_stay_gated_off(self):
        self.assertFalse(_inline_train_allowed("full", {"VOLTRADE_INLINE_ML_TRAIN": ""}))
        self.assertFalse(_inline_train_allowed("full", {"VOLTRADE_INLINE_ML_TRAIN": "true"}))


class TestInlineTrainGateOptIn(unittest.TestCase):
    def test_train_mode_is_always_allowed_regardless_of_env(self):
        """Explicit manual `python3 bot_engine.py train` invocation must
        always be able to train, independent of the env override."""
        self.assertTrue(_inline_train_allowed("train", {}))
        self.assertTrue(_inline_train_allowed("train", {"VOLTRADE_INLINE_ML_TRAIN": "0"}))

    def test_env_flag_set_to_one_allows_any_mode(self):
        self.assertTrue(_inline_train_allowed("scan", {"VOLTRADE_INLINE_ML_TRAIN": "1"}))
        self.assertTrue(_inline_train_allowed("full", {"VOLTRADE_INLINE_ML_TRAIN": "1"}))
        self.assertTrue(_inline_train_allowed("manage", {"VOLTRADE_INLINE_ML_TRAIN": "1"}))


class TestInlineTrainGateOSEnvironCompatible(unittest.TestCase):
    def test_accepts_real_os_environ_object(self):
        """bot_engine.py's __main__ passes os.environ (not a plain dict) —
        confirm the .get() call works against the real environ type too."""
        import os
        self.assertFalse(_inline_train_allowed("scan", os.environ))


if __name__ == "__main__":
    unittest.main()
