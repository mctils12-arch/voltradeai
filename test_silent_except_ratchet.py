"""
SILENT-EXCEPT CLASS RATCHET — [REPAIR 2026-07-06] loop-health follow-through.

The 8/10-REPAIR loop-health trigger (experiments.md 2026-07-06, v1.0.151)
diagnosed the break-generator as OBSERVABILITY DEBT: broad exception
handlers that swallow errors with zero trace. Two of the last ten
incidents were exactly this shape — _fetch_snap hid the SIP-403
entitlement loss behind empty scans (fixed v1.0.148), and deep_score's
5 enrichment fetchers degraded silently the same way (fixed v1.0.151).
Those fixes closed INSTANCES; this test freezes the CLASS.

WHAT COUNTS AS A SILENT HANDLER (AST-based, not regex):
  an `except Exception:` / `except BaseException:` / bare `except:`
  whose body consists ONLY of pass/continue/break and/or returns of
  nothing-ish values (None, constants, empty dict/list/tuple) — i.e.
  the error is unconditionally discarded with no logging, no capture,
  no re-raise, no fallback computation.

THE RATCHET: every runtime module is pinned at its audited 2026-07-06
count. The pin is EXACT in both directions:
  - count > pin  → FAIL. Do not add a silent handler. Either capture
    the error (see bot_engine._run_diag_fetch for the house pattern —
    "ExcType: message" into a diag dict surfaced on the token-gated
    /api/diag/scanner probe) or log it, or narrow the except to the
    specific exception type the code genuinely intends to tolerate
    (e.g. `except OSError:` around best-effort cleanup) — a narrowed
    handler exits this scan by definition.
  - count < pin  → also FAIL, with instructions to LOWER the pin in
    the same commit. This keeps every pin exactly true, so the debt
    surface shrinks monotonically and a later regression can't hide
    inside a stale allowance.
Raising any pin is weakening a test — forbidden by CLAUDE.md.

Scope: repo-root runtime *.py plus strategies/ and alphadesk/ (code the
loop actually runs). scripts/ is session-run tooling and test_* files
are excluded.

Run: python3 -m pytest test_silent_except_ratchet.py -q
"""
import ast
import os
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))

# Audited 2026-07-06 (v1.0.151 era). Every entry is the EXACT count of
# silent broad-except handlers in that file. Shrink one -> lower its pin
# in the same commit (delete the entry at zero). Never raise a pin.
BASELINE = {
    "alpaca_feed.py": 1,
    "alt_data.py": 8,
    "analyze.py": 32,
    "backtest_v2.py": 1,
    "backup_to_github.py": 2,
    "bot_engine.py": 72,
    "csp_universe.py": 7,
    "diagnostics.py": 7,
    "etf_analyzer.py": 13,
    "etf_data_sources.py": 10,
    "finnhub_data.py": 4,
    "insights.py": 7,
    "institutional_data.py": 5,
    "instrument_selector.py": 3,
    "intelligence.py": 7,
    "intraday_shorts.py": 3,
    "macro_data.py": 4,
    "ml_diagnostics.py": 3,
    "ml_model.py": 12,
    "ml_model_v2.py": 24,
    "ml_retrain_safe.py": 1,
    "ml_status.py": 2,
    "ml_toggle.py": 1,
    "options_execution.py": 6,
    "options_manager.py": 10,
    "options_scanner.py": 14,
    "position_sizing.py": 3,
    "probability_engine.py": 1,
    "risk_kill_switch.py": 12,
    "shadow_portfolio.py": 8,
    "social_data.py": 7,
    "stress_index.py": 7,
    "system_config.py": 1,
    "tiered_strategy.py": 3,
    "voltrade_daemon.py": 9,
    "alphadesk/alphadesk/analysis.py": 1,
    "alphadesk/alphadesk/edgar.py": 1,
    "alphadesk/alphadesk/llm_reader.py": 1,
    "alphadesk/alphadesk/market.py": 1,
    "alphadesk/alphadesk/providers.py": 6,
}

SCAN_DIRS = ("strategies", "alphadesk")


def _is_broad(handler):
    t = handler.type
    return t is None or (isinstance(t, ast.Name) and t.id in ("Exception", "BaseException"))


def _is_trivial_stmt(stmt):
    """A statement that discards the error without any trace."""
    if isinstance(stmt, (ast.Pass, ast.Continue, ast.Break)):
        return True
    if isinstance(stmt, ast.Return):
        v = stmt.value
        if v is None or isinstance(v, ast.Constant):
            return True
        if isinstance(v, ast.Dict):
            return not v.keys
        if isinstance(v, (ast.List, ast.Tuple)):
            return not v.elts
    return False


def silent_handler_lines(path):
    """Line numbers of silent broad-except handlers in a Python file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Unparseable runtime code is its own (compileall-covered) problem;
        # this ratchet only measures what it can parse.
        return []
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and _is_broad(node)
        and all(_is_trivial_stmt(s) for s in node.body)
    ]


def scan_repo():
    """{repo-relative path: [line, ...]} for every in-scope runtime module."""
    results = {}

    def take(rel, abspath):
        lines = silent_handler_lines(abspath)
        if lines:
            results[rel] = lines

    for f in sorted(os.listdir(REPO)):
        if f.endswith(".py") and not f.startswith("test_"):
            take(f, os.path.join(REPO, f))
    for d in SCAN_DIRS:
        base = os.path.join(REPO, d)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", "node_modules")]
            for f in sorted(files):
                if f.endswith(".py") and not f.startswith("test_"):
                    abspath = os.path.join(root, f)
                    take(os.path.relpath(abspath, REPO).replace(os.sep, "/"), abspath)
    return results


class TestSilentExceptRatchet(unittest.TestCase):
    def test_no_new_silent_handlers_and_pins_exact(self):
        found = scan_repo()
        grew, shrank = [], []
        for rel in sorted(set(found) | set(BASELINE)):
            n, pin = len(found.get(rel, [])), BASELINE.get(rel, 0)
            if n > pin:
                grew.append(f"{rel}: {n} silent broad-except handlers (pin {pin}); "
                            f"lines {found[rel]}")
            elif n < pin:
                shrank.append(f"{rel}: now {n} (pin {pin}) — lower the pin to {n}"
                              + ("" if n else " (delete the entry)"))
        msgs = []
        if grew:
            msgs.append(
                "NEW silent broad-except handler(s) — the exact defect class behind the "
                "v1.0.148 (SIP-403 scan-blind) and v1.0.151 (deep_score enrichment) "
                "incidents. Capture the error (bot_engine._run_diag_fetch pattern), log "
                "it, or narrow the except to the specific type you mean to tolerate:\n  "
                + "\n  ".join(grew)
            )
        if shrank:
            msgs.append(
                "Silent-handler count DROPPED below its pin (good!) — tighten the "
                "ratchet in this same commit so the improvement can't regress:\n  "
                + "\n  ".join(shrank)
            )
        self.assertEqual(msgs, [], "\n\n".join(msgs))

    def test_scanner_classifies_the_known_shapes(self):
        """Pin the scanner itself so a refactor can't silently gut the ratchet."""
        import tempfile
        cases = [
            # (source, expected number of silent handlers)
            ("try:\n    x()\nexcept Exception:\n    pass\n", 1),
            # module-level return only errors at compile(), not ast.parse() — still scanned
            ("try:\n    x()\nexcept Exception:\n    return {}\n", 1),
            ("def f():\n    try:\n        return x()\n    except Exception:\n        return {}\n", 1),
            ("def f():\n    try:\n        return x()\n    except Exception:\n        return None\n", 1),
            ("def f():\n    try:\n        return x()\n    except Exception:\n        return []\n", 1),
            ("def f():\n    for i in y:\n        try:\n            x(i)\n        except Exception:\n            continue\n", 1),
            ("def f():\n    try:\n        return x()\n    except:\n        pass\n", 1),
            # NOT silent: narrowed type
            ("def f():\n    try:\n        return x()\n    except OSError:\n        return {}\n", 0),
            # NOT silent: error is captured/logged
            ("def f(diag):\n    try:\n        return x()\n    except Exception as e:\n        diag['src'] = str(e)\n        return {}\n", 0),
            ("def f():\n    try:\n        return x()\n    except Exception as e:\n        print('warn', e)\n        return {}\n", 0),
            # NOT silent: re-raise
            ("def f():\n    try:\n        return x()\n    except Exception:\n        raise\n", 0),
            # NOT silent: non-empty fallback value (a real default, not a discard)
            ("def f():\n    try:\n        return x()\n    except Exception:\n        return {'fallback': True}\n", 0),
        ]
        for src, expected in cases:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
                tf.write(src)
                path = tf.name
            try:
                self.assertEqual(
                    len(silent_handler_lines(path)), expected,
                    f"scanner misclassified:\n{src}",
                )
            finally:
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
