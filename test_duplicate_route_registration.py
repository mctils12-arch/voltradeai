"""
Regression tests for scripts/duplicate_route_registration.py — the
duplicate_route_registration counter (D15, MASTER PROGRAM §0.7 DETECT duty).

Two things are pinned: (1) the detector's synthetic semantics — two
`app.<method>("path", ...)` registrations for the same method+path (raw or
after `:param` normalization), anywhere across the scanned files, count as
redundant; a different method, a different path, a commented-out
registration, and a multi-line call each behave correctly; (2) the real live
tree's current count matches `ci/counter_baseline.txt`'s pin (0), so a
future session cannot silently introduce a route collision without this
test (and the CI ratchet) catching it.
"""
import unittest

from scripts.duplicate_route_registration import (
    compute,
    extract_route_registrations,
    find_duplicate_routes,
    normalize_path,
)


class TestNormalizePath(unittest.TestCase):
    def test_param_segment_collapsed_to_canonical_name(self):
        self.assertEqual(normalize_path("/api/foo/:id"), "/api/foo/:param")
        self.assertEqual(normalize_path("/api/foo/:ticker"), "/api/foo/:param")

    def test_path_with_no_param_unchanged(self):
        self.assertEqual(normalize_path("/api/health"), "/api/health")

    def test_multiple_param_segments_all_collapsed(self):
        self.assertEqual(
            normalize_path("/api/:a/bar/:b"), "/api/:param/bar/:param"
        )


class TestExtractRouteRegistrations(unittest.TestCase):
    def test_finds_single_line_registration(self):
        src = 'app.get("/api/health", (req, res) => res.send("ok"));\n'
        hits = extract_route_registrations(src)
        self.assertEqual(hits, [("get", "/api/health", 1)])

    def test_finds_multiline_registration(self):
        # server/billing.ts's real shape: method call opens on one line,
        # the path string is stated on the next.
        src = (
            "app.post(\n"
            '  "/api/billing/webhook",\n'
            "  express.raw({ type: \"application/json\" }),\n"
            "  async (req, res) => {}\n"
            ");\n"
        )
        hits = extract_route_registrations(src)
        # Line of the `app.post(` call itself (line 1), not the path
        # string's own line (2) — the call site is what a human fixing a
        # collision needs to find first.
        self.assertEqual(hits, [("post", "/api/billing/webhook", 1)])

    def test_commented_out_registration_does_not_count(self):
        src = '// app.get("/api/old", handler);\n'
        self.assertEqual(extract_route_registrations(src), [])

    def test_block_commented_registration_does_not_count(self):
        src = (
            "/*\n"
            ' * app.get("/api/old", handler);\n'
            " */\n"
        )
        self.assertEqual(extract_route_registrations(src), [])

    def test_router_dot_calls_are_not_matched(self):
        # This detector only scans `app.<method>`, not `router.<method>` —
        # documented scope limit (see module docstring).
        src = 'router.get("/nested", handler);\n'
        self.assertEqual(extract_route_registrations(src), [])

    def test_single_quotes_also_match(self):
        src = "app.get('/api/health', handler);\n"
        hits = extract_route_registrations(src)
        self.assertEqual(hits, [("get", "/api/health", 1)])


class TestFindDuplicateRoutes(unittest.TestCase):
    def test_same_method_and_path_across_two_files_is_a_duplicate(self):
        routes = [
            ("get", "/api/health", "server/routes.ts", 10),
            ("get", "/api/health", "server/adminStats.ts", 5),
        ]
        dups = find_duplicate_routes(routes)
        self.assertEqual(
            dups,
            {
                ("get", "/api/health"): [
                    ("server/routes.ts", 10),
                    ("server/adminStats.ts", 5),
                ]
            },
        )

    def test_same_path_different_method_is_not_a_duplicate(self):
        routes = [
            ("get", "/api/plan", "server/routes.ts", 1),
            ("post", "/api/plan", "server/routes.ts", 2),
        ]
        self.assertEqual(find_duplicate_routes(routes), {})

    def test_different_param_names_same_shape_is_a_duplicate(self):
        # Already-normalized by the caller in practice (compute() normalizes
        # before this is called) — this test exercises the grouping alone.
        routes = [
            ("get", "/api/foo/:param", "server/routes.ts", 1),
            ("get", "/api/foo/:param", "server/routes.ts", 40),
        ]
        dups = find_duplicate_routes(routes)
        self.assertEqual(len(dups), 1)

    def test_unique_routes_produce_no_duplicates(self):
        routes = [
            ("get", "/api/a", "server/routes.ts", 1),
            ("get", "/api/b", "server/routes.ts", 2),
            ("post", "/api/a", "server/routes.ts", 3),
        ]
        self.assertEqual(find_duplicate_routes(routes), {})

    def test_three_registrations_of_the_same_route_count_two_redundant(self):
        routes = [
            ("get", "/api/a", "server/routes.ts", 1),
            ("get", "/api/a", "server/adminStats.ts", 2),
            ("get", "/api/a", "server/bot.ts", 3),
        ]
        dups = find_duplicate_routes(routes)
        self.assertEqual(len(dups[("get", "/api/a")]), 3)


class TestLiveRepoCount(unittest.TestCase):
    def test_live_tree_matches_pinned_baseline(self):
        result = compute()
        self.assertEqual(
            result["count"], 0,
            "duplicate_route_registration regressed — two app.<method>() "
            "calls now register the same method+path (or the same shape "
            "after :param normalization), which makes the second one "
            "silently unreachable dead code in Express's registration-order "
            "routing. Rename/remove the duplicate (the usual fix) rather "
            "than raising ci/counter_baseline.txt's pin.",
        )


if __name__ == "__main__":
    unittest.main()
