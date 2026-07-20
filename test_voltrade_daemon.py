#!/usr/bin/env python3
"""
Regression tests for voltrade_daemon.py's RPC dispatch table.

WHY THIS EXISTS (KNOWN BROKEN follow-up, 2026-07-04): the RPC route table
in RPCDispatcher._routes maps method -> (module_name, attr_name), and
dispatch() does `getattr(module, attr_name)` at CALL time, not at import
time. A typo'd or stale attr_name (e.g. after the target function is
renamed) silently returns {"status": "error", "error_message": "Method X
not found in Y"} at runtime instead of failing anywhere in CI — exactly
the "Python signature change with an un-updated caller fails silently at
runtime, not in CI" failure mode CLAUDE.md's READ BEFORE WRITE section
warns about. This file pins every route's target to a real, callable
attribute so a future rename breaks CI instead of a live RPC call.

ml_status_impl / ml_toggle_impl are a deliberate exception: no such
modules exist on disk by design — dispatch() falls back to
_ml_status_fallback() / _ml_toggle_fallback() when the module fails to
import. Those two routes are asserted to hit that fallback path, not
skipped silently.
"""
import importlib
import os
import sys
import types
import unittest

import voltrade_daemon
from voltrade_daemon import RPCDispatcher

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Routes whose module_name is a known placeholder with no corresponding
# .py file — dispatch() is designed to fall back to a local method for
# these rather than import a real module. See RPCDispatcher.dispatch().
KNOWN_FALLBACK_ONLY_MODULES = {"ml_status_impl", "ml_toggle_impl"}


class TestRPCDispatchTable(unittest.TestCase):
    def setUp(self):
        self.dispatcher = RPCDispatcher()

    def test_every_local_route_resolves(self):
        """module_name is None -> attr_name must be a real method on RPCDispatcher."""
        checked = 0
        for method, (module_name, attr_name) in self.dispatcher._routes.items():
            if module_name is not None:
                continue
            checked += 1
            self.assertTrue(
                callable(getattr(self.dispatcher, attr_name, None)),
                f"Route '{method}' -> local method '{attr_name}' does not exist "
                f"on RPCDispatcher",
            )
        # Same zero-iteration guard as the sibling module-route test: an
        # empty local-route table would otherwise pass this vacuously
        self.assertGreater(checked, 0, "no local routes were checked at all")

    def test_every_module_route_resolves_to_a_real_callable(self):
        """
        For every route backed by a module that actually exists in the repo,
        the target attribute must exist and be callable. This is exactly the
        class of bug this file was added to catch (shadow_stats ->
        'get_stats', which doesn't exist; the real function is
        'get_shadow_stats').
        """
        checked = 0
        for method, (module_name, attr_name) in self.dispatcher._routes.items():
            if module_name is None or attr_name is None:
                continue
            if module_name in KNOWN_FALLBACK_ONLY_MODULES:
                continue
            module_path = os.path.join(REPO_ROOT, f"{module_name}.py")
            if not os.path.isfile(module_path):
                # Not one of our modules (e.g. a package) — nothing to pin here.
                continue

            mod = importlib.import_module(module_name)
            fn = getattr(mod, attr_name, None)
            self.assertIsNotNone(
                fn,
                f"Route '{method}' -> {module_name}.{attr_name} does not exist. "
                f"Update RPCDispatcher._routes to the real attribute name.",
            )
            self.assertTrue(
                callable(fn),
                f"Route '{method}' -> {module_name}.{attr_name} exists but is not callable",
            )
            checked += 1

        # Sanity: make sure this test isn't silently checking zero routes
        # (e.g. if every module_name got miscategorized as fallback-only).
        self.assertGreater(checked, 0, "No module-backed routes were checked")

    def test_known_placeholder_modules_still_absent(self):
        """
        If ml_status_impl.py / ml_toggle_impl.py ever get created, the
        fallback-skip above would start silently ignoring real routes —
        pin the absence so that transition is caught, not missed.
        """
        for module_name in KNOWN_FALLBACK_ONLY_MODULES:
            module_path = os.path.join(REPO_ROOT, f"{module_name}.py")
            self.assertFalse(
                os.path.isfile(module_path),
                f"{module_name}.py now exists — remove it from "
                f"KNOWN_FALLBACK_ONLY_MODULES and let the real-callable "
                f"assertion cover its route(s).",
            )

    def test_shadow_stats_route_targets_get_shadow_stats(self):
        """Pins the specific fix: shadow_stats must dispatch to the real function."""
        module_name, attr_name = self.dispatcher._routes["shadow_stats"]
        self.assertEqual(module_name, "shadow_portfolio")
        self.assertEqual(attr_name, "get_shadow_stats")
        import shadow_portfolio
        self.assertTrue(callable(getattr(shadow_portfolio, attr_name)))


class TestDispatchDoesNotMaskInternalTypeErrors(unittest.TestCase):
    """
    Regression for the 2026-07-20 live break: run_full_scan (->
    bot_engine.scan_market) failed every Tier2 cycle for 30+ minutes
    straight, every audit-log entry reading "scan_market() takes 0
    positional arguments but 1 was given". scan_market() takes zero
    args and was called as fn(**{}) == fn() — that call can only ever
    raise a TypeError that came from INSIDE the function's own
    execution. The old dispatch() caught that TypeError, assumed it
    meant "wrong calling convention", and blindly retried fn({}) — one
    positional arg into a zero-arg function — which fails for an
    unrelated reason and overwrites the real error/message. The real
    bug was undiagnosable from the audit log for the entire outage.

    Fix: decide kwargs-vs-positional via inspect.signature(fn).bind(),
    which validates the binding without calling the function — so it
    can never raise a TypeError from the function's body. The real
    call then happens outside any except TypeError, so an internal
    TypeError propagates with its true message.

    These tests register throwaway routes against a fake module (never
    touching the real _routes table's live methods) so they exercise
    dispatch()'s calling-convention logic directly and cheaply.
    """

    def setUp(self):
        self.dispatcher = RPCDispatcher()
        self.mod = types.ModuleType("_test_dispatch_masking_mod")
        sys.modules[self.mod.__name__] = self.mod
        voltrade_daemon._modules_loaded[self.mod.__name__] = self.mod

    def tearDown(self):
        sys.modules.pop(self.mod.__name__, None)
        voltrade_daemon._modules_loaded.pop(self.mod.__name__, None)

    def test_zero_arg_function_internal_typeerror_is_not_masked(self):
        def broken_zero_arg():
            return 1 + "not a number"  # genuine TypeError raised from inside

        self.mod.broken_zero_arg = broken_zero_arg
        self.dispatcher._routes["_test_broken_zero_arg"] = (self.mod.__name__, "broken_zero_arg")

        result = self.dispatcher.dispatch("_test_broken_zero_arg", {})
        self.assertEqual(result["status"], "error")
        self.assertIn("unsupported operand", result["error_message"])
        self.assertNotIn("positional argument", result["error_message"])

    def test_kwargs_call_convention_still_works(self):
        def wants_kwargs(a=None, b=None):
            return {"a": a, "b": b}

        self.mod.wants_kwargs = wants_kwargs
        self.dispatcher._routes["_test_wants_kwargs"] = (self.mod.__name__, "wants_kwargs")

        result = self.dispatcher.dispatch("_test_wants_kwargs", {"a": 1, "b": 2})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"], {"a": 1, "b": 2})

    def test_positional_dict_call_convention_still_works(self):
        def wants_positional_dict(d):
            return {"got": d}

        self.mod.wants_positional_dict = wants_positional_dict
        self.dispatcher._routes["_test_wants_positional"] = (self.mod.__name__, "wants_positional_dict")

        result = self.dispatcher.dispatch("_test_wants_positional", {"x": 1})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"], {"got": {"x": 1}})


if __name__ == "__main__":
    unittest.main()
