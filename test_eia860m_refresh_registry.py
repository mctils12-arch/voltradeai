"""
test_eia860m_refresh_registry.py — pure-function battery for
scripts/eia860m_refresh_registry.py (eia860m_capacity_by_plant_fuel/
capacity_delta_report/build_missing_plants_supplement) plus the extended
build_powerplants.build_plants() capacity_override behavior it relies on.
No network, no xlsx, no csv touched on disk except tiny synthetic CSV
fixtures for build_plants() itself (mirroring build_powerplants.py's own
existing CSV-based tests, since that function's only I/O is a CSV read);
build_missing_plants_supplement's own file-reading dependencies
(gppd/plant-directory/generator-xlsx loaders) are monkeypatched with
synthetic in-memory data rather than real files, same spirit as
test_eia860_add_missing_plants.py's synthetic-tuple fixtures.
load_eia860m_operating_plant_rows (the one true xlsx-reading function) is
exercised only by running the script live against a real manually
downloaded EIA-860M file, same convention as every sibling eia860_*.py
test file in this repo.
"""
import csv
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "eia860m_refresh_registry",
    os.path.join(os.path.dirname(__file__), "scripts", "eia860m_refresh_registry.py"))
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)

_bpp_spec = importlib.util.spec_from_file_location(
    "build_powerplants", os.path.join(os.path.dirname(__file__), "scripts", "build_powerplants.py"))
bpp = importlib.util.module_from_spec(_bpp_spec)
_bpp_spec.loader.exec_module(bpp)


# ---- eia860m_capacity_by_plant_fuel ----

def test_capacity_by_plant_fuel_sums_operating_rows_by_plant_and_fuel():
    rows = [
        (1, "SUN", "(OP) Operating", 100.0),
        (1, "SUN", "(OP) Operating", 50.0),
        (2, "SUN", "(OP) Operating", 25.0),
        (1, "WND", "(OP) Operating", 10.0),
    ]
    cap, _skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert cap == {(1, "solar"): 150.0, (2, "solar"): 25.0, (1, "wind"): 10.0}


def test_capacity_by_plant_fuel_excludes_non_operating_status():
    rows = [
        (1, "SUN", "(OP) Operating", 100.0),
        (1, "SUN", "(OA) Out of service but expected to return to service in next calendar year", 999.0),
        (1, "SUN", "(OS) Out of service and NOT expected to return", 999.0),
        (1, "SUN", "(RE) Retired", 999.0),
        (1, "SUN", "(SB) Standby/Backup", 999.0),
    ]
    cap, _skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert cap == {(1, "solar"): 100.0}


def test_capacity_by_plant_fuel_excludes_unmapped_energy_source():
    rows = [(1, "SUN", "(OP) Operating", 100.0), (1, "NG", "(OP) Operating", 500.0)]
    cap, _skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert cap == {(1, "solar"): 100.0}


def test_capacity_by_plant_fuel_excludes_missing_or_blank_plant_id():
    rows = [
        (1, "SUN", "(OP) Operating", 100.0),
        (None, "SUN", "(OP) Operating", 50.0),
        ("", "SUN", "(OP) Operating", 25.0),
        ("  ", "SUN", "(OP) Operating", 25.0),
    ]
    cap, _skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert cap == {(1, "solar"): 100.0}


def test_capacity_by_plant_fuel_excludes_non_numeric_plant_id():
    rows = [(1, "SUN", "(OP) Operating", 100.0), ("N/A", "SUN", "(OP) Operating", 50.0)]
    cap, skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert cap == {(1, "solar"): 100.0}
    # counted, not silently dropped (this is what fixes the silent_py_handlers
    # ratchet: the int(plant_id) except-block now does real work, not a bare continue)
    assert skipped == 1


def test_capacity_by_plant_fuel_skip_count_zero_when_all_plant_ids_valid():
    rows = [(1, "SUN", "(OP) Operating", 100.0), (2, "WND", "(OP) Operating", 10.0)]
    _cap, skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert skipped == 0


def test_capacity_by_plant_fuel_accepts_string_digit_plant_id():
    rows = [("7", "SUN", "(OP) Operating", 12.0)]
    cap, _skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert cap == {(7, "solar"): 12.0}


def test_capacity_by_plant_fuel_treats_none_capacity_as_zero_not_a_crash():
    cap, _skipped = refresh.eia860m_capacity_by_plant_fuel([(1, "SUN", "(OP) Operating", None)])
    assert cap == {(1, "solar"): 0.0}


def test_capacity_by_plant_fuel_empty_input():
    assert refresh.eia860m_capacity_by_plant_fuel([]) == ({}, 0)


def test_capacity_by_plant_fuel_two_fuels_same_plant_kept_separate():
    rows = [(1, "SUN", "(OP) Operating", 10.0), (1, "WND", "(OP) Operating", 20.0)]
    cap, _skipped = refresh.eia860m_capacity_by_plant_fuel(rows)
    assert cap == {(1, "solar"): 10.0, (1, "wind"): 20.0}


# ---- build_powerplants.build_plants() capacity_override extension ----

_GPPD_HEADER = ["country", "name", "gppd_idnr", "capacity_mw", "latitude",
                "longitude", "primary_fuel", "owner"]


def _write_gppd_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_GPPD_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(idnr, fuel, mw, name="Plant"):
    return {"country": "USA", "name": name, "gppd_idnr": idnr, "capacity_mw": mw,
            "latitude": "30.0", "longitude": "-97.0", "primary_fuel": fuel, "owner": "Utility"}


def test_build_plants_capacity_override_applies_to_matching_code_and_fuel(tmp_path):
    csv_path = tmp_path / "gppd.csv"
    _write_gppd_csv(csv_path, [_row("USA0000001", "Solar", "10.0")])
    plants, _, _ = bpp.build_plants(str(csv_path), {}, set(), {},
                                     capacity_override={(1, "solar"): 25.5})
    assert plants[0][1] == 25.5


def test_build_plants_capacity_override_ignores_unmatched_code(tmp_path):
    csv_path = tmp_path / "gppd.csv"
    _write_gppd_csv(csv_path, [_row("USA0000001", "Solar", "10.0")])
    plants, _, _ = bpp.build_plants(str(csv_path), {}, set(), {},
                                     capacity_override={(2, "solar"): 999.0})
    assert plants[0][1] == 10.0


def test_build_plants_capacity_override_requires_matching_fuel_not_just_code(tmp_path):
    # same plant code, but override keyed to a different fuel -> must not apply
    # (a plant with generators of two different fuel-coded types at the same
    # Plant ID cannot cross-contaminate the wrong GPPD row)
    csv_path = tmp_path / "gppd.csv"
    _write_gppd_csv(csv_path, [_row("USA0000001", "Solar", "10.0")])
    plants, _, _ = bpp.build_plants(str(csv_path), {}, set(), {},
                                     capacity_override={(1, "wind"): 999.0})
    assert plants[0][1] == 10.0
    assert plants[0][2] == "solar"


def test_build_plants_capacity_override_none_matches_prior_behavior_exactly(tmp_path):
    csv_path = tmp_path / "gppd.csv"
    _write_gppd_csv(csv_path, [
        _row("USA0000001", "Solar", "10.0"),
        _row("USA0000002", "Wind", "20.0"),
        _row("WRI1000005", "Gas", "5.0"),  # non-USA-code idnr -> code=None path
    ])
    with_default_arg = bpp.build_plants(str(csv_path), {}, set(), {})
    with_explicit_none = bpp.build_plants(str(csv_path), {}, set(), {}, capacity_override=None)
    assert with_default_arg == with_explicit_none


def test_build_plants_capacity_override_empty_dict_changes_nothing(tmp_path):
    csv_path = tmp_path / "gppd.csv"
    _write_gppd_csv(csv_path, [_row("USA0000001", "Solar", "10.0")])
    baseline = bpp.build_plants(str(csv_path), {}, set(), {})
    overridden = bpp.build_plants(str(csv_path), {}, set(), {}, capacity_override={})
    assert baseline == overridden


def test_build_plants_capacity_override_rounds_to_one_decimal(tmp_path):
    csv_path = tmp_path / "gppd.csv"
    _write_gppd_csv(csv_path, [_row("USA0000001", "Solar", "10.0")])
    plants, _, _ = bpp.build_plants(str(csv_path), {}, set(), {},
                                     capacity_override={(1, "solar"): 25.449})
    assert plants[0][1] == 25.4


def test_build_plants_capacity_override_leaves_other_row_fields_untouched(tmp_path):
    csv_path = tmp_path / "gppd.csv"
    _write_gppd_csv(csv_path, [_row("USA0000001", "Solar", "10.0", name="Foo Solar Farm")])
    plants, _, _ = bpp.build_plants(str(csv_path), {}, set(), {},
                                     capacity_override={(1, "solar"): 42.0})
    name, mw, fuel, owner, lat, lon, verified = plants[0]
    assert (name, fuel, owner, lat, lon, verified) == ("Foo Solar Farm", "solar", "Utility", 30.0, -97.0, 0)
    assert mw == 42.0


# ---- capacity_delta_report ----

def _p(name, mw, fuel):
    return [name, mw, fuel, "", 0.0, 0.0, 0]


def test_capacity_delta_report_computes_totals_and_delta_by_fuel():
    before = [_p("A", 10.0, "solar"), _p("B", 5.0, "wind")]
    after = [_p("A", 15.0, "solar"), _p("B", 5.0, "wind"), _p("C", 3.0, "wind")]
    report = refresh.capacity_delta_report(before, after)
    solar = next(r for r in report if r["fuel"] == "solar")
    wind = next(r for r in report if r["fuel"] == "wind")
    assert solar == {"fuel": "solar", "rows_before": 1, "rows_after": 1,
                      "mw_before": 10.0, "mw_after": 15.0, "mw_delta": 5.0}
    assert wind == {"fuel": "wind", "rows_before": 1, "rows_after": 2,
                     "mw_before": 5.0, "mw_after": 8.0, "mw_delta": 3.0}


def test_capacity_delta_report_ignores_other_fuels_not_requested():
    before = [_p("A", 10.0, "coal")]
    after = [_p("A", 999.0, "coal")]
    report = refresh.capacity_delta_report(before, after)
    assert {r["fuel"] for r in report} == {"solar", "wind"}


def test_capacity_delta_report_empty_registries():
    report = refresh.capacity_delta_report([], [])
    assert all(r["mw_before"] == 0.0 and r["mw_after"] == 0.0 for r in report)


# ---- build_missing_plants_supplement (I/O boundary monkeypatched) ----

def test_build_missing_plants_supplement_adds_only_codes_absent_from_gppd(monkeypatch):
    monkeypatch.setattr(refresh._amp, "load_gppd_country_idnr_rows",
                         lambda path: [("USA", "USA0000001")])  # GPPD already has code 1
    monkeypatch.setattr(refresh._amp, "load_eia860_plant_directory",
                         lambda path: {1: ("Existing", "TX", 30.0, -97.0, "U"),
                                        2: ("New Plant", "TX", 31.0, -98.0, "U")})

    def fake_generator_rows(path):
        if path == "solar.xlsx":
            return [("OP", 1, 10.0), ("OP", 2, 5.0)]
        return []
    monkeypatch.setattr(refresh._amp, "load_eia860_generator_rows", fake_generator_rows)

    rows, report = refresh.build_missing_plants_supplement(
        "gppd.csv", "plants.xlsx", "solar.xlsx", "wind.xlsx")

    assert [r[0] for r in rows] == ["New Plant"]
    assert rows[0][1] == 5.0
    assert rows[0][2] == "solar"
    solar_report = next(r for r in report if r["fuel"] == "solar")
    assert solar_report["rows_added"] == 1


def test_build_missing_plants_supplement_no_missing_codes_yields_no_rows(monkeypatch):
    monkeypatch.setattr(refresh._amp, "load_gppd_country_idnr_rows",
                         lambda path: [("USA", "USA0000001")])
    monkeypatch.setattr(refresh._amp, "load_eia860_plant_directory",
                         lambda path: {1: ("Existing", "TX", 30.0, -97.0, "U")})
    monkeypatch.setattr(refresh._amp, "load_eia860_generator_rows",
                         lambda path: [("OP", 1, 10.0)])

    rows, report = refresh.build_missing_plants_supplement(
        "gppd.csv", "plants.xlsx", "solar.xlsx", "wind.xlsx")

    assert rows == []
    assert all(r["rows_added"] == 0 for r in report)


def test_build_missing_plants_supplement_reports_both_fuels_even_if_empty(monkeypatch):
    monkeypatch.setattr(refresh._amp, "load_gppd_country_idnr_rows", lambda path: [])
    monkeypatch.setattr(refresh._amp, "load_eia860_plant_directory", lambda path: {})
    monkeypatch.setattr(refresh._amp, "load_eia860_generator_rows", lambda path: [])

    rows, report = refresh.build_missing_plants_supplement(
        "gppd.csv", "plants.xlsx", "solar.xlsx", "wind.xlsx")

    assert rows == []
    assert {r["fuel"] for r in report} == {"solar", "wind"}


# ---- constants ----

def test_energy_source_to_fuel_covers_solar_and_wind():
    assert refresh.ENERGY_SOURCE_TO_FUEL == {"SUN": "solar", "WND": "wind"}


def test_operating_status_prefix_matches_recent_capacity_check_convention():
    assert refresh.OPERATING_STATUS_PREFIX == "(OP)"
