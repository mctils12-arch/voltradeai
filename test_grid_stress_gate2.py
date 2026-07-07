"""test_grid_stress_gate2.py — gate-2 computation battery
(scripts/grid_stress_gate2.py). Pure-function checks plus
committed-artifact coherence: verdicts must be recomputable from the
recorded spot-validation counts so the conclusion cannot drift from
the data (gate-1 battery precedent)."""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "grid_stress_gate2",
    os.path.join(os.path.dirname(__file__), "scripts", "grid_stress_gate2.py"))
g2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g2)


def test_pct_rank():
    assert g2.pct_rank([1, 2, 3, 4], 2.5) == 0.5
    assert g2.pct_rank([1, 2, 3, 4], 0) == 0.0
    assert g2.pct_rank([1, 2, 3, 4], 9) == 1.0
    assert g2.pct_rank([], 1) is None, "empty distribution -> None, never guessed"


def test_split_halves_and_dates():
    assert g2.in_half("2022-12-31", g2.TRAIN_YEARS) and not g2.in_half("2023-01-01", g2.TRAIN_YEARS)
    assert g2.in_half("2023-01-01", g2.VALID_YEARS) and not g2.in_half("2026-01-01", g2.VALID_YEARS)
    assert g2.month_of("2021-02-15") == 2 and g2.year_of("2021-02-15") == 2021


def test_committed_gate2_artifact_coherent():
    a = json.load(open(os.path.join(os.path.dirname(__file__), "datacore",
                                    "gridvision", "gate2_result.json")))
    v1, v2 = a["v1_forecast_exceedance"], a["v2_extreme_peak"]
    # verdicts recomputable from recorded spot counts — the void rule
    # fired because the outcomes missed the documented events, and that
    # fact is pinned here (2026-07-07 run)
    assert len(v1["spot_validation"]["detected"]) < 6 and v1["verdict"].startswith("VOID")
    assert len(v2["spot_validation"]["detected"]) < 5 and v2["verdict"].startswith("VOID")
    assert a["verdict"].startswith("VOID"), "gate-2 NOT passed — no predictive claim may ship"
    # the honest record must carry both diagnoses and the discount note
    assert "well-forecast" in v1["void_diagnosis"]
    assert "grows" in v2["void_diagnosis"] and "anti-fishing" in v2["void_diagnosis"]
    assert "multiple_hypothesis_note" in a
    assert "no-single-summer-carry" in a["stability_note"]
    # lift numbers recorded for the record, not as claims
    assert v1["validation"]["overall_lift"] is not None
    assert v2["validation"]["overall_lift"] is not None
