#!/usr/bin/env python3
"""
[RULE-REVIEW] 2026-11-06 — source_validation prereg v2.1: one LOGGED-ONLY forward
source, earn.beats_last_3 (the earnings_history block deep_score logs on every
deep-scored shadow record since 2026-11-06; earnings_history.py).

Pins the measurement change on its own:
- registered with kind "logged" (two-sided feature verdict rule), group
  "earnings"; the attribution row's forward source points at it;
- on records WITHOUT the block (every existing record), every existing
  source's result, the BH family, the reference rows and both forward-selection
  arms are byte-identical with and without the new source;
- the new source never enters either selection arm, even when it carries data;
- INSULATION: once it has data, every existing source's result (q-value and
  verdict included) is byte-identical with and without it, for a strongly
  predictive (p ~ 0) and for a noise logged source; its own q is its BH q in
  the full (m+1) family, so it still pays the whole multiple-testing penalty;
- its pre-registered population: one record per (ticker, ET day), the FIRST
  that carries the value (a name's first deep score of a day precedes the
  hourly fill, so the first-seen rule would measure nothing);
- the attribution row reads untested until labels accrue, then gets a verdict.

Run: bash pytest_locked.sh -q test_source_validation_logged.py
"""
import random
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import earnings_history as eh

NOW = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)


class TestLoggedOnlySource(unittest.TestCase):
    def setUp(self):
        import signal_attribution as sa
        import source_validation as sv
        from test_source_validation import make_records
        self.sa, self.sv, self.make = sa, sv, make_records

    def plant(self, recs, beats):
        out = []
        for r in recs:
            r = dict(r)
            if r.get("features"):
                r["earnings_history"] = {**eh.empty_block(None), "beats_last_3": beats(r)}
            out.append(r)
        return out

    def with_beats(self, recs):
        # beats rise with the planted good signal (score = 60 + good)
        return self.plant(recs, lambda r: max(0, min(3, round(r["score"] - 60 + 1.5))))

    def noisy(self):
        rng, noise = random.Random(11), {}

        def beats(r):
            k = (r["ticker"], r["timestamp"][:10])
            if k not in noise:
                noise[k] = rng.randint(0, 3)
            return noise[k]
        return beats

    def test_existing_q_values_never_move_even_with_a_strong_logged_source(self):
        """INSULATION (v2.1): with the logged source carrying data — strongly
        predictive (p ~ 0) or noise — every existing source's result, its q-value
        and verdict included, is byte-identical to the run without it. (Had it
        joined the family's own correction, the strong case would lower 15
        existing q-values in this fixture and turn feat.earnings_surprise, pure
        noise, from FAIL into PASS.)"""
        recs = self.make()
        for planted, case in ((self.with_beats(recs), "strong"), (self.plant(recs, self.noisy()), "noise")):
            with_src = self.sv.validate_sources(planted)
            self.assertIn("q_bh", with_src["sources"]["earn.beats_last_3"], case)      # it is tested and corrected
            with patch.object(self.sv, "LOGGED_SOURCES", ()):
                without = self.sv.validate_sources(planted)
            self.assertEqual(set(with_src["sources"]) - set(without["sources"]), {"earn.beats_last_3"}, case)
            for nm, r in without["sources"].items():
                self.assertEqual(with_src["sources"][nm], r, (case, nm))
            self.assertTrue(with_src["method"]["multiple_testing"].startswith(
                without["method"]["multiple_testing"] + "; 1 logged-only source(s) corrected against all"), case)
            if case == "strong":
                self.assertLess(with_src["sources"]["earn.beats_last_3"]["p"], 1e-4)
                self.assertEqual(with_src["sources"]["feat.earnings_surprise"]["verdict"], "FAIL")

    def test_its_own_q_is_its_bh_q_in_the_full_family(self):
        """The logged source still pays the whole multiple-testing penalty: its q is
        its Benjamini-Hochberg q in the (m+1) family (the m existing tested sources
        plus itself), on the exact p-values (spied), never a family of its own."""
        calls = []
        real = self.sv.benjamini_hochberg

        def spy(ps):
            calls.append(list(ps))
            return real(ps)

        for planted, case in ((self.with_beats(self.make()), "strong"), (self.plant(self.make(), self.noisy()), "noise")):
            calls.clear()
            with patch.object(self.sv, "benjamini_hochberg", side_effect=spy):
                rep = self.sv.validate_sources(planted)
            m = sum(1 for r in rep["sources"].values() if "q_bh" in r and r["kind"] != "logged")
            self.assertEqual([len(c) for c in calls], [m, m + 1], case)      # the family alone, then the full family
            self.assertEqual(calls[1][:m], calls[0], case)                    # the same m p-values, plus the logged one
            p_logged = calls[1][m]
            q_full = real(calls[1])[m]
            self.assertEqual(rep["sources"]["earn.beats_last_3"]["q_bh"], round(q_full, 4), case)
            self.assertEqual(rep["sources"]["earn.beats_last_3"]["p"], round(p_logged, 5), case)
            self.assertGreaterEqual(q_full, p_logged, case)                  # never below its own p: the penalty is paid

    def test_registered_as_a_logged_only_source(self):
        reg = {s["name"]: s for s in self.sv.source_registry()}
        self.assertEqual((reg["earn.beats_last_3"]["kind"], reg["earn.beats_last_3"]["group"]), ("logged", "earnings"))
        row = next(s for s in self.sa.SIGNAL_REGISTRY if s["id"] == "earnings_beats")
        self.assertEqual(row["fw"], "earn.beats_last_3")
        self.assertEqual(self.sa.action_for("earnings_beats", None), ("not_used", "Not used by the score (logged only, since 2026-11-06)."))
        self.assertIn("v2.1", self.sv.PREREG_VERSION)

    def test_untested_until_labels_accrue_then_measured(self):
        recs = self.make()
        table = self.sa.build_attribution(recs, walkforward=None, cfg=False, now=NOW)
        row = next(r for r in table["signals"] if r["id"] == "earnings_beats")
        self.assertEqual((row["verdict"], row["forward"]["verdict"], row["forward"]["n"]), ("untested", "INSUFFICIENT", 0))
        planted = self.with_beats(recs)
        v = self.sv.validate_sources(planted)["sources"]["earn.beats_last_3"]
        self.assertEqual(v["verdict"], "PASS", v.get("verdict_reason"))
        self.assertGreaterEqual(v["n"], self.sv.MIN_N)
        row = next(r for r in self.sa.build_attribution(planted, walkforward=None, cfg=False, now=NOW)["signals"]
                   if r["id"] == "earnings_beats")
        self.assertEqual((row["verdict"], row["action"]), ("predictive", "not_used"))

    def test_its_population_is_the_first_record_per_ticker_day_that_carries_it(self):
        # live timing: a name's FIRST deep score of the day is a cache miss (the hourly fill runs after it)
        seen, planted = set(), []
        for r in self.with_beats(self.make()):
            key = (r["ticker"], r["timestamp"][:10])
            if r.get("features") and key not in seen:
                seen.add(key)
                r = {k: v for k, v in r.items() if k != "earnings_history"}
            planted.append(r)
        rep = self.sv.validate_sources(planted)
        v = rep["sources"]["earn.beats_last_3"]
        self.assertEqual(v["population_rule"], self.sv.LOGGED_POPULATION_RULE)
        self.assertEqual(v["n"], rep["population"]["+5d"]["n_unique"], "one record per ticker-day, none lost to the miss")
        self.assertEqual(v["verdict"], "PASS")
        first_seen = self.sv.build_population(planted, "+5d")
        self.assertEqual(sum(1 for r in first_seen if r["rec"].get("earnings_history")), 0,
                         "the first-seen rule the other sources use would have measured nothing")
        mine = self.sv.build_population(planted, "+5d", require=self.sv.source_registry()[-1]["get"])
        self.assertTrue(all(r["rec"]["timestamp"] > f["rec"]["timestamp"] for r, f in zip(mine, first_seen)))

    def test_existing_results_identical_on_records_without_it_and_never_a_selection_candidate(self):
        recs = self.make()
        with_src = self.sv.validate_sources(recs)
        with patch.object(self.sv, "LOGGED_SOURCES", ()):
            without = self.sv.validate_sources(recs)
            sel_without = self.sv.selection_and_ablation(recs, without)
        self.assertEqual(set(with_src["sources"]) - set(without["sources"]), {"earn.beats_last_3"})
        for nm, r in without["sources"].items():
            self.assertEqual(with_src["sources"][nm], r, nm)
        self.assertEqual(with_src["method"]["multiple_testing"], without["method"]["multiple_testing"])
        self.assertEqual(with_src["reference"], without["reference"])
        self.assertEqual(self.sv.selection_and_ablation(recs, with_src), sel_without)
        planted = self.with_beats(recs)
        sel = self.sv.selection_and_ablation(planted, self.sv.validate_sources(planted))
        with patch.object(self.sv, "LOGGED_SOURCES", ()):
            self.assertEqual(sel, self.sv.selection_and_ablation(planted, self.sv.validate_sources(planted)))


if __name__ == "__main__":
    unittest.main()
