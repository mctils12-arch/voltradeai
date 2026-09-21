"""
Tests for scripts/ladder_registry_coverage_check.py -- the check that
cross-references scripts/data_stream_registry_check.py's "built"
candidates against datacore/signal_ladder.json's roots (see that script's
module docstring for the full rationale: a built pipeline can otherwise
carry zero ladder-bookkeeping entry with nothing to notice).

Ten things are asserted:
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
  7. so2_column_gibs specifically is regression-pinned as COVERED -- the
     third of the 7 originally-queued gaps, added as raw_only since
     client/src/pages/datamap.tsx's own inline comment for the layer
     states "As-is display only -- no predictive claim".
  8. fda_calendar specifically is regression-pinned as COVERED -- the
     fifth of the 7 originally-queued gaps, added as raw_only since
     server/fdaEvents.ts's own module header states its hypothesis as
     "gate 2, not attempted" and the live endpoint self-labels
     kind:"raw".
  9. entsoe_eu_power specifically is regression-pinned as COVERED -- the
     sixth of the 7 originally-queued gaps, added as ONE raw_only root
     covering all three ENTSO-E modules (euLoad/euGenerationMix/
     euDayAheadPrices), each still "HYPOTHESIS (gate-locked)" per its own
     module header. This session also corrected an unsupported
     "euLoad gate1_pass 2026-07-07" label found in datamap.tsx's
     euPowerOpen comment (a ship date mislabeled as a gate result) --
     the ladder entry's status is raw_only, not gate1_pass.
  10. global_energy_monitor specifically is regression-pinned as
     COVERED -- the seventh and last of the 7 originally-queued gaps,
     added as raw_only (scripts/gem_ingest.py's own catalogued-registry
     framing, no gate-1/gate-2 attempt filed anywhere on this asset
     registry itself, distinct from the already-tracked derived
     gem_methane_plume_proximity root built on top of it). Closes the
     original 7-gap queue: EXPECTED_UNCOVERED_IDS is now empty.

Run: python3 -m pytest test_ladder_registry_coverage_check.py -v
"""
import importlib.util
import os
import sys
import unittest
import unittest.mock

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

_spec = importlib.util.spec_from_file_location(
    "ladder_registry_coverage_check",
    os.path.join(REPO_ROOT, "scripts", "ladder_registry_coverage_check.py"),
)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

# The exact set of registry-"built" candidate ids confirmed to have no
# matching root in datacore/signal_ladder.json. Empty since the
# global_energy_monitor PR closed the last of the original 7 gaps queued
# by the epa_camd_cems cross-check PR (#1133) -- a future session adding
# a new "built" candidate without a ladder root will grow this list again,
# deliberately, not silently.
EXPECTED_UNCOVERED_IDS = []


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

    def test_sec_ftd_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "sec_ftd", uncovered_ids,
            "sec_ftd regressed back to uncovered -- it was added to "
            "datacore/signal_ladder.json (status raw_only; the settlement-"
            "stress composite hypothesis in secFtd.ts's own header stays "
            "gate-locked/untested, per the 2026-09-21 backfill session) in "
            "the PR that removed it from EXPECTED_UNCOVERED_IDS; if that "
            "root was removed, this test should be updated deliberately, "
            "not left to fail silently",
        )

    def test_so2_column_gibs_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "so2_column_gibs", uncovered_ids,
            "so2_column_gibs regressed back to uncovered -- it was added to "
            "datacore/signal_ladder.json (status raw_only, per datamap.tsx's "
            "own 'as-is display only -- no predictive claim' comment) in the "
            "PR that removed it from EXPECTED_UNCOVERED_IDS; if that root was "
            "removed, this test should be updated deliberately, not left to "
            "fail silently",
        )

    def test_fda_calendar_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "fda_calendar", uncovered_ids,
            "fda_calendar regressed back to uncovered -- it was added to "
            "datacore/signal_ladder.json (status raw_only; fdaEvents.ts's "
            "own header calls its IV-ramp-into-catalysts idea 'gate 2, not "
            "attempted', and the live endpoint self-labels kind:'raw') in "
            "the PR that removed it from EXPECTED_UNCOVERED_IDS; if that "
            "root was removed, this test should be updated deliberately, "
            "not left to fail silently",
        )

    def test_entsoe_eu_power_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "entsoe_eu_power", uncovered_ids,
            "entsoe_eu_power regressed back to uncovered -- it was added to "
            "datacore/signal_ladder.json as ONE raw_only root covering all "
            "three ENTSO-E modules (euLoad/euGenerationMix/euDayAheadPrices), "
            "each still 'HYPOTHESIS (gate-locked)' per its own module header, "
            "in the PR that removed it from EXPECTED_UNCOVERED_IDS; if that "
            "root was removed, this test should be updated deliberately, "
            "not left to fail silently",
        )

    def test_global_energy_monitor_is_covered(self):
        result = check.audit()
        uncovered_ids = {u["id"] for u in result["uncovered"]}
        self.assertNotIn(
            "global_energy_monitor", uncovered_ids,
            "global_energy_monitor regressed back to uncovered -- it was added to "
            "datacore/signal_ladder.json (status raw_only; scripts/gem_ingest.py's "
            "own catalogued-registry framing, no gate-1/gate-2 attempt filed on the "
            "raw asset registry itself) in the PR that emptied EXPECTED_UNCOVERED_IDS; "
            "if that root was removed, this test should be updated deliberately, "
            "not left to fail silently",
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
        # Every real ALIASES entry now maps to a live ladder root (the
        # global_energy_monitor PR closed the last empty-mapping gap), so
        # "genuinely uncovered" is exercised via a temporary fake ALIASES
        # entry rather than a real, currently-empty-mapped candidate id --
        # there no longer is one. patch.dict restores the real table after
        # the test regardless of pass/fail.
        fake = self._FakeRegistry()
        fake.CANDIDATES = [{
            "id": "fixture_candidate_with_no_ladder_root", "name": "fixture",
            "status": "built", "manifest_keys": [], "layer_ids": [], "note": "",
        }]
        with unittest.mock.patch.dict(
            check.ALIASES, {"fixture_candidate_with_no_ladder_root": []}
        ):
            result = check.audit(registry_module=fake)
        self.assertEqual(len(result["uncovered"]), 1)
        self.assertEqual(result["uncovered"][0]["id"], "fixture_candidate_with_no_ladder_root")

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
