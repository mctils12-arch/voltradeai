"""
Regression tests for scripts/param_shadows_outer_binding.py — the
param_shadows_outer_binding counter (D14, MASTER PROGRAM §0.7 DETECT duty).

Two things are pinned: (1) the detector's synthetic semantics — a top-level
function parameter that shadows another top-level binding of the same name
AND is referenced in the body counts; an unrelated name, an unused shadow, a
too-short name, and a nested (non-top-level) declaration each do not; (2) the
real live tree's current count matches ci/counter_baseline.txt's pin (0), so
a future session cannot silently introduce a new shadowing parameter without
this test (and the CI ratchet) catching it.
"""
import unittest

from scripts.param_shadows_outer_binding import (
    compute,
    find_shadowing_params,
    top_level_binding_names,
    top_level_functions,
)


class TestTopLevelBindingNames(unittest.TestCase):
    def test_const_let_and_function_all_counted(self):
        src = (
            "const OUTER_A = 1;\n"
            "let outerB = 2;\n"
            "function outerC() { return 3; }\n"
        )
        names = top_level_binding_names(src)
        self.assertEqual(names, {"OUTER_A", "outerB", "outerC"})

    def test_indented_nested_declaration_not_counted_as_top_level(self):
        src = (
            "function container() {\n"
            "  const nested = 1;\n"
            "  return nested;\n"
            "}\n"
        )
        names = top_level_binding_names(src)
        self.assertEqual(names, {"container"})
        self.assertNotIn("nested", names)


class TestTopLevelFunctions(unittest.TestCase):
    def test_finds_named_function_and_arrow_with_block_body(self):
        src = (
            "function helperOne(x: number) {\n"
            "  return x + 1;\n"
            "}\n"
            "\n"
            "const helperTwo = (y: number) => {\n"
            "  return y * 2;\n"
            "};\n"
        )
        fns = {fn["name"] for fn in top_level_functions(src)}
        self.assertEqual(fns, {"helperOne", "helperTwo"})

    def test_single_expression_arrow_is_skipped(self):
        src = "const double = (n: number) => n * 2;\n"
        fns = top_level_functions(src)
        self.assertEqual(fns, [])


class TestFindShadowingParams(unittest.TestCase):
    def test_used_shadowing_param_counts(self):
        src = (
            "const focusSat = \"outer-value\";\n"
            "\n"
            "function helper(focusSat: string) {\n"
            "  console.log(focusSat);\n"
            "}\n"
        )
        hits = find_shadowing_params(src)
        self.assertEqual(hits, [{"function": "helper", "param": "focusSat"}])

    def test_used_shadowing_param_counts_for_arrow_functions_too(self):
        src = (
            "const outerValue = 1;\n"
            "\n"
            "const helperArrow = (outerValue: number) => {\n"
            "  return outerValue * 2;\n"
            "};\n"
        )
        hits = find_shadowing_params(src)
        self.assertEqual(hits, [{"function": "helperArrow", "param": "outerValue"}])

    def test_unrelated_param_name_does_not_count(self):
        src = (
            "const OUTER_NAME = 1;\n"
            "\n"
            "function helper(other: number) {\n"
            "  return other;\n"
            "}\n"
        )
        self.assertEqual(find_shadowing_params(src), [])

    def test_unused_shadowing_param_does_not_count(self):
        src = (
            "const OUTER_NAME = 1;\n"
            "\n"
            "function helper(OUTER_NAME: number) {\n"
            "  return 42;\n"
            "}\n"
        )
        self.assertEqual(find_shadowing_params(src), [])

    def test_no_matching_top_level_binding_does_not_count(self):
        src = (
            "function helper(soloParam: number) {\n"
            "  return soloParam;\n"
            "}\n"
        )
        self.assertEqual(find_shadowing_params(src), [])

    def test_short_name_below_min_len_is_excluded(self):
        # PI/MU/J2-style conventional short constants: excluded by design
        # (see MIN_NAME_LEN in the module docstring) to avoid flagging every
        # unrelated same-named short parameter as noise.
        src = (
            "const PI = 3.14159;\n"
            "\n"
            "function helper(PI: number) {\n"
            "  return PI * 2;\n"
            "}\n"
        )
        self.assertEqual(find_shadowing_params(src), [])

    def test_shadow_of_a_nested_binding_by_a_nested_function_is_out_of_scope(self):
        # Both the "outer" binding and the shadowing function are indented
        # (nested inside `container`) — neither is module-top-level, so this
        # detector, by design, does not attempt it (would need real
        # closure-chain scope resolution, not a grep — see module docstring).
        src = (
            "function container() {\n"
            "  const outerName = 1;\n"
            "  function nested(outerName: number) {\n"
            "    return outerName;\n"
            "  }\n"
            "  return nested;\n"
            "}\n"
        )
        self.assertEqual(find_shadowing_params(src), [])

    def test_multiple_hits_in_one_file_all_count(self):
        src = (
            "const alphaValue = 1;\n"
            "const betaValue = 2;\n"
            "\n"
            "function useAlpha(alphaValue: number) {\n"
            "  return alphaValue + 1;\n"
            "}\n"
            "\n"
            "function useBeta(betaValue: number) {\n"
            "  return betaValue + 1;\n"
            "}\n"
        )
        hits = find_shadowing_params(src)
        self.assertEqual(len(hits), 2)


class TestLiveRepoCount(unittest.TestCase):
    def test_live_tree_matches_pinned_baseline(self):
        result = compute()
        self.assertEqual(
            result["count"], 0,
            "param_shadows_outer_binding regressed — a new top-level function "
            "parameter now shadows another top-level binding of the same "
            "name and is referenced in the body. Rename the parameter (the "
            "usual fix) rather than raising ci/counter_baseline.txt's pin.",
        )


if __name__ == "__main__":
    unittest.main()
