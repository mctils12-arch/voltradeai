"""test_grid_capacity.py — GRID VISION A1 gate-1 battery
(scripts/grid_capacity_tx.py). Pure-function tests on the OSM tag
conventions verified in the live Texas extract (multi-value
"138000;69000" voltages, circuits=N, kV suffixes), plus
committed-artifact coherence with the PRE-STATED gate-1 criteria."""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "grid_capacity_tx",
    os.path.join(os.path.dirname(__file__), "scripts", "grid_capacity_tx.py"))
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def test_parse_voltages_conventions():
    assert gc.parse_voltages("345000") == ([345000], 0)
    assert gc.parse_voltages("138000;69000") == ([138000, 69000], 0)
    assert gc.parse_voltages("138 kV") == ([138000], 0)
    assert gc.parse_voltages("138") == ([138000], 0), "bare small number = kV"
    assert gc.parse_voltages(None) == ([], 0)
    assert gc.parse_voltages("medium") == ([], 1), "junk counted, never guessed"
    v, bad = gc.parse_voltages("345000;?")
    assert v == [345000] and bad == 1


def test_voltage_class_buckets():
    assert gc.voltage_class(500000) == "345kV+"
    assert gc.voltage_class(345000) == "345kV+"
    assert gc.voltage_class(230000) == "230kV"
    assert gc.voltage_class(138000) == "100-229kV"
    assert gc.voltage_class(115000) == "100-229kV"
    assert gc.voltage_class(69000) == "69-99kV"
    assert gc.voltage_class(34500) == "<69kV"
    assert gc.voltage_class(None) == "unknown"


def test_circuit_km_accounting():
    # multi-voltage way: one circuit per voltage — never multiplied by circuits
    out = gc.circuit_km_for(10.0, [138000, 69000], "2")
    assert out == {"100-229kV": 10.0, "69-99kV": 10.0}
    # single voltage x circuits=2
    out = gc.circuit_km_for(10.0, [345000], "2")
    assert out == {"345kV+": 20.0}
    # no circuits tag -> 1 circuit
    assert gc.circuit_km_for(10.0, [345000], None) == {"345kV+": 10.0}
    # unparseable circuits tag falls back to 1, never crashes
    assert gc.circuit_km_for(10.0, [345000], "two") == {"345kV+": 10.0}
    # no voltage -> honest unknown, 1 circuit
    assert gc.circuit_km_for(10.0, [], None) == {"unknown": 10.0}


def test_linestring_km_known_distance():
    # one degree of latitude ~111.2 km
    km = gc.linestring_km([[0.0, 0.0], [0.0, 1.0]])
    assert 110.5 < km < 111.9
    # two segments accumulate
    km2 = gc.linestring_km([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    assert abs(km2 - 2 * km) < 0.01


def test_committed_artifact_gate1_coherent():
    path = os.path.join(os.path.dirname(__file__), "datacore", "gridvision",
                        "tx_grid_registry.json")
    a = json.load(open(path))
    assert "OpenStreetMap contributors" in a["provenance"]["license"], \
        "ODbL attribution required — this artifact derives from OSM"
    g = a["gate1"]
    assert "PRE-STATED" in g["criteria_prestated"].upper() or "before comparison" in g["criteria_prestated"], \
        "criteria must be recorded as pre-stated"
    c = g["computed"]
    # checks recomputable from computed values — verdict can't drift from data
    assert g["checks"]["a_hv_circuit_km_in_band"] == (44_250 <= c["hv_circuit_km_gte69kV"] <= 115_000)
    assert g["checks"]["b_unknown_share_lt_35pct"] == (c["unknown_voltage_share"] < 0.35)
    assert g["checks"]["c_345kV_gte_8000km"] == (c["kv345_circuit_km"] >= 8_000)
    assert g["verdict"] == ("PASS" if all(g["checks"].values()) else "FAIL")
    # 2026-07-07 run facts (pin the ingested state; re-runs update deliberately)
    lines = a["lines"]
    assert lines["ways"] == 31220 and 89_000 < lines["km"] < 91_000
    census = lines["voltage_census_top20_km"]
    assert census["138000"] > census["345000"] > census["115000"], \
        "ERCOT structure: 138kV subtransmission dominates, 345kV backbone second"
    assert lines["unknown_voltage"]["share_of_km"] < 0.05, \
        "TX voltage tagging measured excellent (3.8%) — regression = data loss"
    assert a["substations"]["total"] == 12330
    assert a["provenance"]["circuit_km_convention"].startswith("multi-voltage"), \
        "circuit-counting convention must be stated in the artifact"
