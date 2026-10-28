"""
bot_engine.py defines no module-level `logger` (it logs through
logging.getLogger("bot_engine") / local loggers), yet two regime-engine except
handlers (step 5 scanner path, step 7 label override) called `logger.debug` —
a NameError raised INSIDE the handler. In deep_score that NameError escapes
into the ML block's outer `except`, silently skipping the ML blend for the
candidate; in the options scan it escapes into `except: pass`, dropping the
rest of the options scan. Found while wiring the 2026-10-28 regime live flip.
Ratchet: every bare `logger` read in bot_engine must resolve to a binding.
"""
import ast
import os
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(ROOT, "bot_engine.py")).read()


class TestNoUnboundLogger(unittest.TestCase):
    def test_every_logger_read_is_bound(self):
        tree = ast.parse(SRC)
        bound_module = any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "logger" for t in n.targets)
            for n in tree.body
        ) or any(
            isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "logger" for a in n.names)
            for n in tree.body
        )
        unbound = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    local.add(n.id)
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and n.id == "logger" and isinstance(n.ctx, ast.Load):
                    if "logger" not in local and not bound_module:
                        unbound.append(n.lineno)
        self.assertEqual(sorted(set(unbound)), [], "bare `logger` reads with no binding in bot_engine.py (NameError at runtime)")


if __name__ == "__main__":
    unittest.main()
