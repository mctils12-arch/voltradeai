"""
Regression tests for scripts/ats_resolver_gate0.py. Pure-function tests
only: no network calls, no dependency on live NASDAQ symbol directory or
live ATS APIs (matches the existing test_illiquid_universe_probe.py /
test_gtrends_probe.py convention for research probe scripts).
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "ats_resolver_gate0",
    os.path.join(os.path.dirname(__file__), "scripts", "ats_resolver_gate0.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestNormalizeCompanyName(unittest.TestCase):
    def test_strips_share_class_suffix_after_dash(self):
        self.assertEqual(
            probe.normalize_company_name("Acme Robotics Inc. - Class A Common Stock"),
            "acme robotics")

    def test_strips_corporate_suffix_words(self):
        self.assertEqual(probe.normalize_company_name("Widget Holdings Corp"), "widget")

    def test_strips_parentheticals(self):
        self.assertEqual(probe.normalize_company_name("Beta Systems (The)"), "beta systems")

    def test_single_word_name_unchanged(self):
        self.assertEqual(probe.normalize_company_name("Sweetgreen Inc"), "sweetgreen")

    def test_all_suffix_words_yields_empty(self):
        self.assertEqual(probe.normalize_company_name("The Company Inc"), "")


class TestDeriveSlugCandidates(unittest.TestCase):
    def test_multi_word_name_produces_nospace_and_firstword_variants(self):
        cands = probe.derive_slug_candidates("Warby Parker Inc", "WRBY")
        self.assertIn("warbyparker", cands)
        self.assertIn("warby", cands)
        self.assertIn("wrby", cands)

    def test_firstword_not_crowded_out_by_full_name(self):
        # regression: a real sanity check this session (Palantir
        # Technologies Inc -> real slug "palantir") showed a naive
        # 2-slot cap could drop the first-word guess in favor of a
        # redundant hyphenated encoding of the full name.
        cands = probe.derive_slug_candidates("Palantir Technologies Inc", "PLTR")
        self.assertIn("palantir", cands)

    def test_single_word_name_has_no_duplicate_variant(self):
        cands = probe.derive_slug_candidates("Sweetgreen Inc", "SG")
        # no_space == first_word here ("sweetgreen"); must not appear twice
        self.assertEqual(cands.count("sweetgreen"), 1)

    def test_ticker_always_included(self):
        cands = probe.derive_slug_candidates("Anything Corp", "ZZZZ")
        self.assertIn("zzzz", cands)

    def test_capped_at_three_candidates(self):
        cands = probe.derive_slug_candidates("Four Word Company Name Inc", "FOUR")
        self.assertLessEqual(len(cands), 3)

    def test_empty_name_falls_back_to_ticker_only(self):
        cands = probe.derive_slug_candidates("Inc Corp Ltd", "ABCD")
        self.assertEqual(cands, ["abcd"])


class TestClassifyResponse(unittest.TestCase):
    def test_non_200_is_a_miss(self):
        self.assertEqual(probe.classify_response("greenhouse", 404, ""),
                          {"hit": False, "job_count": None})

    def test_greenhouse_hit(self):
        body = '{"jobs": [{"id": 1}, {"id": 2}]}'
        self.assertEqual(probe.classify_response("greenhouse", 200, body),
                          {"hit": True, "job_count": 2})

    def test_ashby_hit(self):
        body = '{"jobs": [{"id": 1}]}'
        self.assertEqual(probe.classify_response("ashby", 200, body),
                          {"hit": True, "job_count": 1})

    def test_lever_hit_is_a_bare_json_array(self):
        body = '[{"id": "a"}, {"id": "b"}, {"id": "c"}]'
        self.assertEqual(probe.classify_response("lever", 200, body),
                          {"hit": True, "job_count": 3})

    def test_smartrecruiters_hit(self):
        body = '{"content": [{"id": "x"}]}'
        self.assertEqual(probe.classify_response("smartrecruiters", 200, body),
                          {"hit": True, "job_count": 1})

    def test_smartrecruiters_empty_content_is_a_miss(self):
        # SmartRecruiters 200s {"content": []} for ANY identifier, real
        # or fabricated (verified live) -- schema match alone is not a
        # valid hit signal for this platform, unlike the other three.
        body = '{"offset": 0, "limit": 100, "totalFound": 0, "content": []}'
        self.assertEqual(probe.classify_response("smartrecruiters", 200, body),
                          {"hit": False, "job_count": None})

    def test_200_with_wrong_shape_is_a_miss(self):
        # e.g. a generic 200 error page instead of a real board
        self.assertEqual(probe.classify_response("greenhouse", 200, '{"error": "not found"}'),
                          {"hit": False, "job_count": None})

    def test_200_with_unparseable_body_is_a_miss(self):
        self.assertEqual(probe.classify_response("lever", 200, "not json"),
                          {"hit": False, "job_count": None})

    def test_lever_dict_instead_of_list_is_a_miss(self):
        self.assertEqual(probe.classify_response("lever", 200, '{"jobs": []}'),
                          {"hit": False, "job_count": None})


class TestSummarize(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(probe.summarize([]),
                          {"n_tickers": 0, "n_resolved": 0, "coverage_pct": 0.0,
                           "hits_by_platform": {}, "resolved_tickers": []})

    def test_mixed_hits_and_misses(self):
        rows = [
            {"ticker": "AAA", "resolved": True, "platform": "greenhouse"},
            {"ticker": "BBB", "resolved": True, "platform": "greenhouse"},
            {"ticker": "CCC", "resolved": True, "platform": "lever"},
            {"ticker": "DDD", "resolved": False, "platform": None},
        ]
        out = probe.summarize(rows)
        self.assertEqual(out["n_tickers"], 4)
        self.assertEqual(out["n_resolved"], 3)
        self.assertEqual(out["coverage_pct"], 75.0)
        self.assertEqual(out["hits_by_platform"], {"greenhouse": 2, "lever": 1})
        self.assertEqual(out["resolved_tickers"], ["AAA", "BBB", "CCC"])

    def test_zero_coverage(self):
        rows = [{"ticker": "X", "resolved": False, "platform": None}]
        self.assertEqual(probe.summarize(rows)["coverage_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
