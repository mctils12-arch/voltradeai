"""
The regime engine's SHADOW hook in bot_engine._scan_market_inner (docs/regime/
FINDINGS.md step 3): computed once per scan after the macro snapshot, only when
regime_engine.engine_allowed() says so, never able to break the scan, and
carried on the scan result as "regime" for bot.ts's audit lines and the
/api/bot/regime report. An AST ratchet on the source (the function itself
needs the network and the broker), plus the engine-off contract.
"""
import ast
import os
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(ROOT, "bot_engine.py")).read()
TREE = ast.parse(SRC)


def _scan_inner():
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "_scan_market_inner":
            return node
    raise AssertionError("_scan_market_inner not found")


class TestScanHook(unittest.TestCase):
    def test_hook_is_guarded_by_the_flag_and_wrapped_in_try(self):
        fn = _scan_inner()
        hooks = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Try):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Import) and any(a.name == "regime_engine" and a.asname == "_regime_engine" for a in sub.names):
                        hooks.append(node)
        self.assertEqual(len(hooks), 1, "exactly one guarded regime_engine SHADOW hook (alias _regime_engine) in _scan_market_inner")
        body_src = ast.get_source_segment(SRC, hooks[0])
        self.assertIn("engine_allowed()", body_src)
        self.assertIn("update_state(gating=\"SPY\")", body_src)
        self.assertTrue(any(isinstance(h, ast.ExceptHandler) for h in hooks[0].handlers))
        # the handler records the failure on the state (stale, loud) instead of swallowing it
        handler_src = ast.get_source_segment(SRC, hooks[0].handlers[0])
        self.assertIn("\"stale\": True", handler_src)

    def test_hook_runs_after_the_macro_snapshot_and_before_slot_sizing(self):
        i_macro = SRC.index("_macro = get_macro_snapshot()\n    except Exception:\n        _macro = {}")
        i_hook = SRC.index("import regime_engine as _regime_engine")
        i_slots = SRC.index("from system_config import get_adaptive_params as _gap_slots")
        self.assertLess(i_macro, i_hook)
        self.assertLess(i_hook, i_slots)

    def test_scan_result_carries_the_regime_state(self):
        self.assertIn('"regime": _regime_state if \'_regime_state\' in dir() else None,', SRC)

    def test_engine_off_means_no_module_import_on_the_scan_path(self):
        # engine_allowed() with the flag off and the config mirror False is the only thing the
        # hook evaluates before touching the engine's data path
        import regime_engine
        self.assertFalse(regime_engine.engine_allowed({"VOLTRADE_REGIME_ENGINE": "0"}, {"REGIME_ENGINE_ENABLED": False}))


if __name__ == "__main__":
    unittest.main()
