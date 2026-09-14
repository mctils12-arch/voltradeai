"""test_grid_generation_gate1_national_sweep.py — pure-function battery for
scripts/grid_generation_gate1_national_sweep.py. No network — build_report's
own EIA-930/EIA-860 I/O is reused unchanged from grid_generation_gate1_ba.py
and is already covered by that module's own test file; this file only tests
the new filtering/summarizing logic, same convention as every sibling
gate1_*.py test file.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "grid_generation_gate1_national_sweep",
    os.path.join(os.path.dirname(__file__), "scripts", "grid_generation_gate1_national_sweep.py"))
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def test_is_real_finding_true_for_fail():
    assert sweep.is_real_finding({"verdict": "FAIL", "ratio_of_capacity": 1.286}) is True


def test_is_real_finding_true_for_substantive_inconclusive():
    assert sweep.is_real_finding({"verdict": "INCONCLUSIVE", "reason": "some other reason"}) is True


def test_is_real_finding_false_for_trivial_missing_on_one_side():
    assert sweep.is_real_finding({"verdict": "INCONCLUSIVE", "reason": "missing on one side"}) is False


def test_is_real_finding_false_for_pass():
    assert sweep.is_real_finding({"verdict": "PASS", "ratio_of_capacity": 0.615}) is False


def test_summarize_report_filters_and_flattens_across_regions():
    report = {
        "regions": {
            "SWPP": {
                "verdicts": {
                    "solar": {"verdict": "FAIL", "ratio_of_capacity": 1.286},
                    "wind": {"verdict": "PASS", "ratio_of_capacity": 0.632},
                }
            },
            "ISNE": {
                "verdicts": {
                    "nuclear": {"verdict": "FAIL", "ratio_of_capacity": 1.105},
                    "wind": {"verdict": "PASS", "ratio_of_capacity": 0.834},
                }
            },
            "FPL": {
                "verdicts": {
                    "coal": {"verdict": "INCONCLUSIVE", "reason": "missing on one side"},
                }
            },
        }
    }
    findings = sweep.summarize_report(report)
    assert findings == [
        {"ba": "ISNE", "fuel": "nuclear", "verdict": "FAIL", "ratio_of_capacity": 1.105},
        {"ba": "SWPP", "fuel": "solar", "verdict": "FAIL", "ratio_of_capacity": 1.286},
    ]


def test_summarize_report_sorted_by_ba_then_fuel():
    report = {
        "regions": {
            "ZZZZ": {"verdicts": {"gas": {"verdict": "FAIL"}}},
            "AAAA": {"verdicts": {"wind": {"verdict": "FAIL"}, "coal": {"verdict": "FAIL"}}},
        }
    }
    findings = sweep.summarize_report(report)
    assert [(f["ba"], f["fuel"]) for f in findings] == [
        ("AAAA", "coal"), ("AAAA", "wind"), ("ZZZZ", "gas"),
    ]


def test_summarize_report_empty_when_all_pass_or_trivial():
    report = {
        "regions": {
            "PJM": {"verdicts": {"gas": {"verdict": "PASS"}}},
            "NYIS": {"verdicts": {"oil": {"verdict": "INCONCLUSIVE", "reason": "missing on one side"}}},
        }
    }
    assert sweep.summarize_report(report) == []
