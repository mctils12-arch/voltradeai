#!/usr/bin/env python3
"""
[REPAIR] 2026-11-06: every bare `logging.` call in bot_engine resolves.

deep_score's regime-label override and regime-ML-snapshot error handlers called
logging.getLogger(...) while bot_engine had no module-level `import logging`
(only function-local `import logging as _x` aliases), so when regime_engine
raised, the handler raised NameError: the ML block was silently skipped, or the
whole shadow record was lost. Found by the 2026-11-06 ML audit.

Run: python3 -m pytest -q test_bot_engine_logging_import.py
"""
import ast
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SRC = os.path.join(_ROOT, "bot_engine.py")


def _module_binds_logging(tree) -> bool:
    return any(isinstance(n, ast.Import) and any(a.name == "logging" and a.asname in (None, "logging") for a in n.names)
               for n in tree.body)


def _unbound_logging_uses(tree):
    """(function, line) for every `logging.<attr>` inside a function that neither
    imports `logging` itself nor can rely on a module-level import."""
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {a.asname or a.name.split(".")[0] for n in ast.walk(fn) if isinstance(n, ast.Import) for a in n.names}
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "logging" \
                    and "logging" not in local:
                out.append((fn.name, n.lineno))
    return out


def test_bot_engine_imports_logging_at_module_level():
    tree = ast.parse(open(SRC).read())
    assert _module_binds_logging(tree), "bot_engine.py must `import logging` at module level"


def test_no_bare_logging_call_depends_on_a_missing_import():
    tree = ast.parse(open(SRC).read())
    if _module_binds_logging(tree):
        return
    assert _unbound_logging_uses(tree) == []


def test_the_two_regime_handlers_resolve_at_runtime():
    import bot_engine
    assert hasattr(bot_engine, "logging"), "the module namespace the deep_score handlers read"
    bot_engine.logging.getLogger("bot_engine").debug("regime label override unavailable: test")
