#!/usr/bin/env python3
"""
Regression tests for the TIER2 full-scan SIGKILL fix.

Context: production 8:13 AM ET OOM kills traced to inline train_model()
fired on every `bot_engine.py full` invocation when the model was >24h
old. Fix moves inline training behind BOT_ENGINE_INLINE_TRAIN and forces
fast_mode=True when it does run; adds phase breadcrumbs to stderr so
SIGKILL'd scans pinpoint where Python died.
"""
import ast
import os
import re
import sys
import unittest


ENGINE_PATH = os.path.join(os.path.dirname(__file__), "bot_engine.py")
BOT_TS_PATH = os.path.join(os.path.dirname(__file__), "server", "bot.ts")


def _engine_main_source() -> str:
    with open(ENGINE_PATH) as f:
        src = f.read()
    idx = src.find('if __name__ == "__main__":')
    assert idx != -1, "bot_engine.py missing __main__ block"
    return src[idx:]


class TestInlineTrainGate(unittest.TestCase):
    """Inline training must be opt-in, not default-on."""

    def test_inline_training_gated_by_env_var(self):
        main_src = _engine_main_source()
        self.assertIn("BOT_ENGINE_INLINE_TRAIN", main_src,
            "Inline train_model() call must be gated on BOT_ENGINE_INLINE_TRAIN "
            "— otherwise every `full` scan re-trains ML inline, risking OOM")

    def test_inline_train_uses_fast_mode(self):
        """Even with the escape hatch, training must use fast_mode=True."""
        main_src = _engine_main_source()
        # Strip comments before matching to avoid false positives from docs.
        code_only = "\n".join(
            line.split("#", 1)[0] for line in main_src.splitlines()
        )
        train_calls = re.findall(r"train_model\([^)]*\)", code_only)
        self.assertTrue(train_calls, "Expected at least one train_model(...) call in __main__")
        for call in train_calls:
            self.assertIn("fast_mode=True", call,
                f"train_model call '{call}' must pass fast_mode=True to cap RSS")

    def test_no_unconditional_train_model(self):
        """
        Regression guard: don't reintroduce an unconditional `train_model()`
        at the top of bot_engine.py __main__.
        """
        main_src = _engine_main_source()
        # Grab only the top-level statements of __main__ via AST
        with open(ENGINE_PATH) as f:
            tree = ast.parse(f.read())
        main_block = None
        for node in tree.body:
            if isinstance(node, ast.If) and ast.dump(node.test).startswith("Compare"):
                # `if __name__ == "__main__":`
                if any(isinstance(n, ast.Str) or (isinstance(n, ast.Constant) and n.value == "__main__")
                       for n in ast.walk(node.test)):
                    main_block = node
                    break
        self.assertIsNotNone(main_block, "Could not locate __main__ AST block")

        # Top-level (direct children) must not contain `train_model(...)` calls.
        for stmt in main_block.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "train_model":
                    # It's fine if it's nested inside an `if _inline_train:` branch;
                    # top-level means a direct child of the __main__ If node.
                    if stmt in main_block.body and isinstance(stmt, ast.Expr) and stmt.value is sub:
                        self.fail("Top-level train_model() call in __main__ — "
                                  "must be nested behind the BOT_ENGINE_INLINE_TRAIN gate")


class TestPhaseBreadcrumbs(unittest.TestCase):
    """Stderr must carry `[mem]` phase markers so SIGKILL leaves a trace."""

    def test_phase_helper_defined(self):
        main_src = _engine_main_source()
        self.assertIn("def _phase(", main_src,
            "bot_engine __main__ must define a _phase() helper for breadcrumbs")

    def test_scan_entry_and_dispatch_phases_logged(self):
        main_src = _engine_main_source()
        self.assertIn("scan entry rss~", main_src,
            "Scan entry RSS must still be logged (pre-existing breadcrumb)")
        self.assertIn("dispatch_", main_src,
            "Dispatch phase breadcrumb missing — needed to pinpoint where "
            "SIGKILL landed (import vs train vs scan)")


class TestBotTsClassification(unittest.TestCase):
    """server/bot.ts must surface `[mem]` last-phase in TIER2-ERROR on SIGKILL."""

    def test_sigkill_error_extracts_last_phase(self):
        with open(BOT_TS_PATH) as f:
            src = f.read()
        self.assertIn("last_phase", src,
            "TIER2-ERROR classification must include last_phase on SIGKILL "
            "— without it the activity log can't tell where Python died")
        self.assertIn("[mem]", src,
            "Classification must scan stderr for `[mem]` breadcrumbs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
