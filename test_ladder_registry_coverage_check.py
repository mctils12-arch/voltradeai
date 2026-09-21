"""
Tests for scripts/ladder_registry_coverage_check.py -- the check that
cross-references scripts/data_stream_registry_check.py's "built"
candidates against datacore/signal_ladder.json's roots (see that script's
module docstring for the full rationale: a built pipeline can otherwise
carry zero ladder-bookkeeping entry with nothing to notice).

Five things are asserted:
  1. The ALIASES table covers every currently-built candidate (a future
     session adding a new "built" candidate without an ALIASES entry
     fails this loudly instead of the coverage check silently ignoring it).
  2. No ALIASES entry points at a ladder id that has since been renamed
     or removed (stale-alias drift).
  3. The exact set of built-but-ladder-untracked pipelines is pinned. This
     is a ratchet in the same spirit as PROGRAM_STATE.md's Q11
     (layersRegistry renderKind/lod pin) -- it can move in either
     direction, but only on a conscious edit to this test, never silently.
  4. epa_camd_cems specifically is regression-pinned as COVERED -- the
     fix this check's first run motivated in the same PR.
  5. usgs_volcano_alerts specifically is regression-pinned as COVERED --
     the first of the 7 originally-queued gaps to get a real ladder entry
     (raw_only, same session as PR #1133's own epa_camd_cems fix,
     different PR).
  6. cboe_vix_term_structure specifically is regression-pinned as
     COVERED -- the second of the 7 originally-queued gaps, added as
     gate1_pass (not raw_only) since its own module header and manifest
     both document a real GATE 1 cross-check against FRED's VIXCLS series.

Run: python3 -m pytest test_ladder_registry_coverage_check.py -v
"""
import importlib.util
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

_spec = importlib.util.spec_from_file_location(
    "ladder_registry_coverage_check",
    os.path.join(REPO_ROOT, "scripts", "ladder_registry_coverage_check.py"),
)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

# The exact set of registry-"built" candidate ids this session confirmed
# have no matching root in datacore/signal_ladder.json, by hand, after
# fixing epa_camd_cems in the same PR. Each needs its own module read to
# assign an honest ladder status (raw_only vs a real gate number) before
# it can be added -- see research/open_questions.md for the filed NEXT.
EXPECTED_UNCOVERED_IDS = [
    "entsoe_eu_power",
    "fda_calendar",
    "global_energy_monitor",
    "sec_ftd",
    "so2_column_gibs",
]


class TestLadderRegistryCoverage(unittest.TestCase):
    def test_every_built_candidate_has_an_alias_entry(self):
        result = check.audit()
        self.assertEqual(
            result["unaliased_built_candidates"], [],
            "a 'built' candidate in scripts/data_stream_registry_check.py has no "
            "ALIASES entry in scripts/ladder_registry_coverage_check.py -- add one "
            "(a real ladder id, or an empty list if it genuinely has none) so this "
            "check can classify it instead of silently skipping it",
        )

    def test_no_stale_alias_targets(self):
        result = check.audit()
        self.assertEqual(
            result["stale_alias_targets"], [],
            "an ALIASES entry points at a datacore/signal_ladder.json root id that "
            "no longer exists -- the root was renamed/removed and ALIASES was not "
            "updated to match",
        )

    def test_uncovered_set_is_pinned(self):
        result = check.audit()
        uncovered_ids = sorted(u["id"] for u in result["uncovered"])
        self.assertEqual(
            uncovered_ids, sorted(EXPECTED_UNCOVERED_IDS),
            "the set of built-but-ladder-untracked pipelines changed. If you added "
            "a datacore/signal_ladder.json root for one of these, give it a real "
            "ALIASES entry above AND remove it from EXPECTED_UNCOVERED_IDS here "
            "(don't just delete it from the expected list). If this list grew, a "
            "pipeline shipped 'built' without ladder bookkeeping -- add a root for "
            "it or state why it is exempt.",
        )

    def test_epa_camd_cems_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "epa_camd_cems", uncovered_ids,
            "epa_camd_cems regressed back to uncovered -- it was added to "
            "datacore/signal_ladder.json (status raw_only) in the same PR that "
            "added this check; if that root was removed, this test should be "
            "updated deliberately, not left to fail silently",
        )

    def test_usgs_volcano_alerts_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "usgs_volcano_alerts", uncovered_ids,
            "usgs_volcano_alerts regressed back to uncovered -- it was added to "
            "datacore/signal_ladder.json (status raw_only) in the PR that removed "
            "it from EXPECTED_UNCOVERED_IDS; if that root was removed, this test "
            "should be updated deliberately, not left to fail silently",
        )

    def test_cboe_vix_term_structure_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "cboe_vix_term_structure", uncovered_ids,
            "cboe_vix_term_structure regressed back to uncovered -- it was added "
            "to datacore/signal_ladder.json (status gate1_pass, on a verified "
            "GATE 1 cross-check vs. FRED VIXCLS) in the PR that removed it from "
            "EXPECTED_UNCOVERED_IDS; if that root was removed, this test should "
            "be updated deliberately, not left to fail silently",
        )


class TestCoverageDetectorCatchesRealGaps(unittest.TestCase):
    """Proves the checker isn't vacuously passing -- feed it a registry
    module with a deliberately unaliased/uncovered candidate and confirm
    it actually flags it, same discipline as
    test_data_stream_registry_check.py's drift-detector tests."""

    class _FakeRegistry:
        pass

    def test_detects_unaliased_built_candidate(self):
        fake = self._FakeRegistry()
        fake.CANDIDATES = [{
            "id": "totally_new_unaliased_candidate", "name": "fixture",
            "status": "built", "manifest_keys": [], "layer_ids": [], "note": "",
        }]
        result = check.audit(registry_module=fake)
        self.assertEqual(result["unaliased_built_candidates"], ["totally_new_unaliased_candidate"])

    def test_detects_genuinely_uncovered_candidate(self):
        fake = self._FakeRegistry()
        fake.CANDIDATES = [{
            "id": "fda_calendar", "name": "fixture", "status": "built",
            "manifest_keys": [], "layer_ids": [], "note": "",
        }]
        result = check.audit(registry_module=fake)
        self.assertEqual(len(result["uncovered"]), 1)
        self.assertEqual(result["uncovered"][0]["id"], "fda_calendar")

    def test_recognizes_a_covered_candidate(self):
        fake = self._FakeRegistry()
        fake.CANDIDATES = [{
            "id": "epa_camd_cems", "name": "fixture", "status": "built",
            "manifest_keys": [], "layer_ids": [], "note": "",
        }]
        result = check.audit(registry_module=fake)
        self.assertEqual(result["uncovered"], [])
        self.assertEqual(result["unaliased_built_candidates"], [])


if __name__ == "__main__":
    unittest.main()
