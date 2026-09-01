#!/usr/bin/env python3
"""
test_options_outcome_breakdown.py

Covers ml_model_v2.options_outcome_breakdown() and its companion
OPTIONS_EXIT_REASONS constant, added 2026-09-01 (scheduled-routine session,
[PIPELINE]) to answer KNOWN BROKEN #12(c)'s own NEXT step (1) in
research/open_questions.md: whether the live win/loss/open records
/api/diag/ml already shows are attributable to the standalone single-leg
options exit path or to equity/ETF trading. The function classifies purely
by exit_reason (never ticker, matching server/diag.ts's hard whitelist).

Two things must stay true for this classification to be honest:
  1. options_outcome_breakdown() buckets correctly (unit tests below).
  2. OPTIONS_EXIT_REASONS never silently drifts out of sync with the actual
     literal strings options_manager.py's 8 close sites pass to
     _record_options_exit_feedback() — a 9th close site with a new reason
     string that isn't added here would make this diagnostic quietly
     undercount. test_exit_reason_constant_matches_options_manager_call_sites
     is a static source scan that catches that drift.

Run: python3 -m pytest test_options_outcome_breakdown.py -v
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_model_v2 import OPTIONS_EXIT_REASONS, options_outcome_breakdown


class TestOptionsOutcomeBreakdown(unittest.TestCase):
    def test_empty_feedback(self):
        self.assertEqual(options_outcome_breakdown([]), {})

    def test_none_feedback(self):
        self.assertEqual(options_outcome_breakdown(None), {})

    def test_non_dict_and_seeded_records_ignored(self):
        feedback = [
            "not a dict",
            {"_seed": True, "exit_reason": "profit_target", "outcome": "win"},
        ]
        self.assertEqual(options_outcome_breakdown(feedback), {})

    def test_equity_exit_reasons_excluded(self):
        feedback = [
            {"exit_reason": "stop_loss", "outcome": "loss"},
            {"exit_reason": "trailing_stop", "outcome": "win"},
            {"exit_reason": "take_profit", "outcome": "win"},
            {"exit_reason": "time_stop", "outcome": "flat"},
            {"exit_reason": "position_kill", "outcome": "loss"},
        ]
        self.assertEqual(options_outcome_breakdown(feedback), {})

    def test_options_exit_reasons_bucketed_by_outcome(self):
        feedback = [
            {"exit_reason": "profit_target", "outcome": "win"},
            {"exit_reason": "profit_target", "outcome": "win"},
            {"exit_reason": "loss_limit", "outcome": "loss"},
            {"exit_reason": "bought_loss_limit", "outcome": "loss"},
            {"exit_reason": "gamma_risk", "outcome": "loss"},
            {"exit_reason": "dte_critical"},  # no 'outcome' key
        ]
        self.assertEqual(
            options_outcome_breakdown(feedback),
            {"win": 2, "loss": 3, "open": 1},
        )

    def test_missing_outcome_defaults_to_open_matching_diag_probe(self):
        # Mirrors server/bot.ts's /api/diag/ml `_outcomes` bucketing exactly
        # (outcome-is-None -> 'open') so the two numbers stay comparable.
        feedback = [{"exit_reason": "assignment_close", "outcome": None}]
        self.assertEqual(options_outcome_breakdown(feedback), {"open": 1})

    def test_mixed_options_and_equity_records_only_counts_options(self):
        feedback = [
            {"exit_reason": "dte_close", "outcome": "win"},
            {"exit_reason": "stop_loss", "outcome": "win"},
            {"exit_reason": "dte_close_bought", "outcome": "open"},
            {"exit_reason": None, "outcome": "loss"},  # orphan_exit-style record
        ]
        self.assertEqual(
            options_outcome_breakdown(feedback),
            {"win": 1, "open": 1},
        )

    def test_options_exit_reasons_constant_has_exactly_the_documented_8(self):
        self.assertEqual(
            OPTIONS_EXIT_REASONS,
            frozenset({
                "dte_critical", "assignment_close", "dte_close",
                "dte_close_bought", "profit_target", "loss_limit",
                "bought_loss_limit", "gamma_risk",
            }),
        )

    def test_exit_reason_constant_matches_options_manager_call_sites(self):
        """Static source scan: every literal reason string passed as the 7th
        positional argument to _record_options_exit_feedback(...) in
        options_manager.py must be a member of OPTIONS_EXIT_REASONS, and
        every member of OPTIONS_EXIT_REASONS must actually be used by at
        least one call site. Either direction of drift fails this test
        instead of silently rotting the diagnostic."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "options_manager.py")
        with open(path) as f:
            src = f.read()
        calls = re.findall(
            r"_record_options_exit_feedback\([^)]*?,\s*\"([a-z_]+)\"\s*,\s*pos_state\.get",
            src,
        )
        self.assertGreaterEqual(len(calls), 8, "expected to find all 8 known call sites")
        found = set(calls)
        self.assertEqual(
            found, set(OPTIONS_EXIT_REASONS),
            f"options_manager.py call sites {found} and ml_model_v2.OPTIONS_EXIT_REASONS "
            f"{set(OPTIONS_EXIT_REASONS)} have drifted apart",
        )


if __name__ == "__main__":
    unittest.main()
