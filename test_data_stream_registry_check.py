"""
Tests for scripts/data_stream_registry_check.py — the EDGE DOCTRINE
compiled-knowledge registry (see that file's module docstring for why it
exists: replacing repeated grep-and-read-experiments.md archaeology about
"is free-data candidate X already built" with a runtime-verified fact).

Two things are asserted:
  1. The hand-curated CANDIDATES table matches actual repo state RIGHT NOW
     (zero drift) — this is the regression guard: if a future session
     deletes/renames a file a "built" candidate depends on, or the table
     goes stale relative to reality, this test fails loudly instead of the
     table silently lying to the next session that trusts it.
  2. The drift *detector itself* actually detects drift when given a
     deliberately wrong table (proves the check isn't vacuously passing).

Run: python3 -m pytest test_data_stream_registry_check.py -v
"""
import importlib.util
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

_spec = importlib.util.spec_from_file_location(
    "data_stream_registry_check",
    os.path.join(REPO_ROOT, "scripts", "data_stream_registry_check.py"),
)
registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(registry)


class TestRegistryMatchesRepoState(unittest.TestCase):
    """Guards against the table drifting from what's actually on disk."""

    def test_no_drift_against_live_repo(self):
        result = registry.audit()
        self.assertEqual(
            result["drift"], [],
            f"data_stream_registry_check.CANDIDATES has drifted from repo state: {result['drift']}\n"
            "Either the source file for a 'built' candidate was moved/deleted (fix the code or the "
            "table), or a formerly-unbuilt candidate now has a manifest/layer on disk (flip its status "
            "to 'built' in scripts/data_stream_registry_check.py)."
        )

    def test_every_built_candidate_has_some_manifest_or_layer_key(self):
        # A 'built' candidate with BOTH lists empty can never be verified —
        # that's a table-authoring bug, not a real pass.
        for c in registry.CANDIDATES:
            if c["status"] == "built":
                self.assertTrue(
                    c["manifest_keys"] or c["layer_ids"],
                    f"{c['id']} is marked built but has no manifest_keys or layer_ids to verify against",
                )

    def test_candidate_ids_are_unique(self):
        ids = [c["id"] for c in registry.CANDIDATES]
        self.assertEqual(len(ids), len(set(ids)), "duplicate candidate id in CANDIDATES")

    def test_status_values_are_from_the_known_set(self):
        known = {"built", "declined_gate1_fail", "declined_dead_source",
                 "blocked_free_key", "blocked_registration", "candidate_unbuilt"}
        for c in registry.CANDIDATES:
            self.assertIn(c["status"], known, f"{c['id']} has unknown status {c['status']!r}")

    def test_edge_doctrine_named_list_is_fully_resolved(self):
        """The exact 6 candidates CLAUDE.md's EDGE DOCTRINE #1 names by example
        must each be either built or explicitly declined — never left as an
        open question, since that's the literal axis (a) build list."""
        named = [c for c in registry.CANDIDATES if c["edge_doctrine_named"]]
        self.assertEqual(len(named), 6, "expected exactly 6 EDGE-DOCTRINE-named candidates")
        for c in named:
            self.assertIn(
                c["status"], {"built", "declined_gate1_fail", "declined_dead_source"},
                f"EDGE-DOCTRINE-named candidate {c['id']} is neither built nor declined: {c['status']}",
            )


class TestDriftDetectorCatchesRealDrift(unittest.TestCase):
    """Proves the checker isn't vacuously passing — feed it a table that
    disagrees with reality and confirm it actually flags it."""

    def test_detects_built_candidate_with_missing_manifest(self):
        fake_candidates = [{
            "id": "fake_stream", "name": "Fake stream that does not exist",
            "edge_doctrine_named": False, "status": "built",
            "manifest_keys": ["this_manifest_key_should_never_exist_xyz"], "layer_ids": [],
            "note": "test fixture",
        }]
        original = registry.CANDIDATES
        registry.CANDIDATES = fake_candidates
        try:
            result = registry.audit()
        finally:
            registry.CANDIDATES = original
        self.assertEqual(len(result["drift"]), 1)
        self.assertEqual(result["drift"][0]["id"], "fake_stream")

    def test_detects_unbuilt_candidate_with_existing_manifest(self):
        # fdicfailures.json genuinely exists on disk (server/fdicBanks.ts) —
        # claim it's unbuilt and confirm the detector flags the stale claim.
        real_manifest_keys = registry._load_manifest_keys_on_disk()
        existing_key = next(iter(real_manifest_keys), None)
        if existing_key is None:
            self.skipTest("no manifests on disk to test against")
        fake_candidates = [{
            "id": "stale_unbuilt_claim", "name": "Something actually already built",
            "edge_doctrine_named": False, "status": "candidate_unbuilt",
            "manifest_keys": [existing_key], "layer_ids": [],
            "note": "test fixture",
        }]
        original = registry.CANDIDATES
        registry.CANDIDATES = fake_candidates
        try:
            result = registry.audit()
        finally:
            registry.CANDIDATES = original
        self.assertEqual(len(result["drift"]), 1)
        self.assertIn("stale", result["drift"][0]["drift"])

    def test_clean_table_produces_no_drift(self):
        fake_candidates = [{
            "id": "clean_fixture", "name": "A correctly-tracked absent stream",
            "edge_doctrine_named": False, "status": "candidate_unbuilt",
            "manifest_keys": [], "layer_ids": [], "note": "test fixture",
        }]
        original = registry.CANDIDATES
        registry.CANDIDATES = fake_candidates
        try:
            result = registry.audit()
        finally:
            registry.CANDIDATES = original
        self.assertEqual(result["drift"], [])


if __name__ == "__main__":
    unittest.main()
