"""
Regression tests for scripts/gem_methane_gate2c.py — GEM methane-plume gate-2(c)
ticker resolution. Pure-function tests only: no network calls, no dependency on
the live /api/data/methane-plumes endpoint or backtest_v2's Alpaca/Yahoo fetch.

Covers the 2026-07-23 widening of build_ownership_index()'s name index to also
match GEM's "Abbreviation" field (previously only Full Name/Name), added because
1,833/26,250 real ownership.json.gz entities carry a populated Abbreviation that
a free-text oil/gas Operator/Owner/Parent string could match even when it never
matches Full Name or Name verbatim.
"""
import gzip
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "gem_methane_gate2c",
    os.path.join(os.path.dirname(__file__), "scripts", "gem_methane_gate2c.py"))
gate2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate2c)

build_ownership_index = gate2c.build_ownership_index
normalize_company_name = gate2c.normalize_company_name
resolve_cik_from_entity_id = gate2c.resolve_cik_from_entity_id
resolve_ticker_for_plume = gate2c.resolve_ticker_for_plume


def _write_ownership(gem_dir: str, entities: list[dict]) -> None:
    with gzip.open(os.path.join(gem_dir, "ownership.json.gz"), "wt") as f:
        json.dump({"entities": entities}, f)


class TestBuildOwnershipIndexAbbreviation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resolves_by_abbreviation_when_full_name_and_name_dont_match(self):
        # 5+ chars: normalize_company_name's own short-fragment guard (len(tokens)<2
        # and len(out)<5 -> "") only lets a single-token key through at >=5 chars —
        # real GEM abbreviations split roughly 1138/1833 under that floor (verified
        # against the live datacore/gem/ownership.json.gz), so this test uses one of
        # the 695 that clear it. The under-5 case is its own test below.
        _write_ownership(self.tmp, [
            {"Entity ID": "E1", "Full Name": "Aalborg Portland (Anqing) Co Ltd",
             "Name": "Aalborg Portland (Anqing)", "Abbreviation": "AAAID"},
        ])
        idx = build_ownership_index(self.tmp)
        self.assertEqual(idx["name_to_id"].get(normalize_company_name("AAAID")), "E1")

    def test_full_name_and_name_still_index_alongside_abbreviation(self):
        _write_ownership(self.tmp, [
            {"Entity ID": "E1", "Full Name": "Example Energy Corp", "Name": "Example Energy",
             "Abbreviation": "EXENG"},
        ])
        idx = build_ownership_index(self.tmp)
        self.assertEqual(idx["name_to_id"].get(normalize_company_name("Example Energy Corp")), "E1")
        self.assertEqual(idx["name_to_id"].get(normalize_company_name("Example Energy")), "E1")
        self.assertEqual(idx["name_to_id"].get(normalize_company_name("EXENG")), "E1")

    def test_short_abbreviation_under_five_chars_is_not_indexed(self):
        # Documents a real interaction found while widening this index: the
        # pre-existing normalize_company_name short-fragment guard (added to stop
        # spurious FULL NAME substring matches) also applies to Abbreviation, and
        # DELIBERATELY so — a 2-4 char code matching arbitrary free-text
        # Operator/Owner/Parent strings would be a far more likely false positive
        # than a genuine match. 1138 of 1833 real Abbreviation values fall under
        # this floor and are correctly never indexed, not a bug to work around.
        _write_ownership(self.tmp, [
            {"Entity ID": "E1", "Full Name": "Some Mining Company", "Abbreviation": "SMC"},
        ])
        idx = build_ownership_index(self.tmp)
        self.assertNotIn(normalize_company_name("SMC"), idx["name_to_id"])
        self.assertEqual(normalize_company_name("SMC"), "")

    def test_colliding_abbreviation_across_two_entities_is_dropped_not_guessed(self):
        _write_ownership(self.tmp, [
            {"Entity ID": "E1", "Full Name": "First Energy Holdings", "Abbreviation": "DUP"},
            {"Entity ID": "E2", "Full Name": "Second Mining Group", "Abbreviation": "DUP"},
        ])
        idx = build_ownership_index(self.tmp)
        self.assertNotIn(normalize_company_name("DUP"), idx["name_to_id"])
        # the entities' own Full Name matches are unaffected by the dropped collision
        self.assertEqual(idx["name_to_id"].get(normalize_company_name("First Energy Holdings")), "E1")
        self.assertEqual(idx["name_to_id"].get(normalize_company_name("Second Mining Group")), "E2")

    def test_missing_abbreviation_field_is_skipped_not_an_error(self):
        _write_ownership(self.tmp, [
            {"Entity ID": "E1", "Full Name": "No Abbreviation Corp"},
        ])
        idx = build_ownership_index(self.tmp)
        self.assertEqual(idx["name_to_id"].get(normalize_company_name("No Abbreviation Corp")), "E1")
        self.assertEqual(len(idx["name_to_id"]), 1)  # no phantom entry from a missing field


class TestResolveTickerForPlumeViaAbbreviation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_oil_gas_operator_matches_via_abbreviation_end_to_end(self):
        _write_ownership(self.tmp, [
            {"Entity ID": "E1", "Full Name": "Example Energy Corporation",
             "Abbreviation": "EXENG", "US SEC Central Index Key": "0001234567"},
        ])
        idx = build_ownership_index(self.tmp)
        cik_ticker = {"1234567": "EXNG"}
        plume = {"nearestAsset": {"kind": "oil_gas_extraction", "operator": "EXENG"}}
        ticker, method = resolve_ticker_for_plume(plume, {}, idx, cik_ticker)
        self.assertEqual(ticker, "EXNG")
        self.assertEqual(method, "name_operator")

    def test_no_match_when_operator_name_matches_nothing(self):
        _write_ownership(self.tmp, [
            {"Entity ID": "E1", "Full Name": "Example Energy Corporation", "Abbreviation": "EXENG"},
        ])
        idx = build_ownership_index(self.tmp)
        plume = {"nearestAsset": {"kind": "oil_gas_extraction", "operator": "Totally Unrelated Co"}}
        ticker, method = resolve_ticker_for_plume(plume, {}, idx, {})
        self.assertIsNone(ticker)
        self.assertEqual(method, "none")


class TestResolveCikFromEntityIdUnaffected(unittest.TestCase):
    """Sanity check the Abbreviation widening didn't disturb the separate
    entity-id/CIK-chain resolution path (coal mines resolve by ID, not name)."""

    def test_direct_cik_still_resolves(self):
        idx = {"entities_by_id": {"E1": {}}, "cik_by_entity": {"E1": "123"}}
        self.assertEqual(resolve_cik_from_entity_id("E1", idx), "123")

    def test_unknown_entity_returns_none(self):
        idx = {"entities_by_id": {}, "cik_by_entity": {}}
        self.assertIsNone(resolve_cik_from_entity_id("E999", idx))


if __name__ == "__main__":
    unittest.main()
