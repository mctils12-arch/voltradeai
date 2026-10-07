import datetime, importlib.util, os

_spec = importlib.util.spec_from_file_location(
    "refresh_nasr", os.path.join(os.path.dirname(__file__), "scripts", "refresh_nasr.py"))
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)
D = datetime.date


def test_cycle_math_matches_known_faa_dates():
    assert rn.current_cycle(D(2026, 10, 4)) == D(2026, 10, 1)
    assert rn.current_cycle(D(2026, 10, 28)) == D(2026, 10, 1)
    assert rn.current_cycle(D(2026, 10, 29)) == D(2026, 10, 29)
    assert rn.current_cycle(D(2026, 9, 3)) == D(2026, 9, 3)  # prior committed cycle
    assert rn.current_cycle(D(2026, 9, 30)) == D(2026, 9, 3)


def test_bundle_url_matches_faa_naming():
    assert rn.bundle_url(D(2026, 10, 1)).endswith("/01_Oct_2026_CSV.zip")
    assert rn.bundle_url(D(2026, 10, 29)).endswith("/29_Oct_2026_CSV.zip")


def test_diff_fixes():
    old = {"A": [1, 1], "B": [2, 2], "C": [3, 3]}
    new = {"A": [1, 1], "C": [3.01, 3], "D": [4, 4]}
    assert rn.diff_fixes(old, new) == (["D"], ["B"], ["C"])


def test_committed_cycle_parses():
    assert isinstance(rn.committed_cycle(), D)


def _build(tmp_path, nav_rows):
    import csv, json
    spec = importlib.util.spec_from_file_location(
        "build_nasr_fixes", os.path.join(os.path.dirname(__file__), "scripts", "build_nasr_fixes.py"))
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    with open(tmp_path / "FIX_BASE.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EFF_DATE", "FIX_ID", "LAT_DECIMAL", "LONG_DECIMAL"])
        w.writerow(["2026/10/01", "SHADOW", "40.0", "-100.0"])
    with open(tmp_path / "NAV_BASE.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["NAV_ID", "LAT_DECIMAL", "LONG_DECIMAL", "MAG_VARN", "MAG_VARN_HEMIS", "MAG_VARN_YEAR"])
        w.writerows(nav_rows)
    out = tmp_path / "out.json"
    b.main(str(tmp_path), str(out))
    return json.loads(out.read_text())


def test_navaid_magvar_emitted_east_positive_west_negative(tmp_path):
    d = _build(tmp_path, [["SNS", "36.6", "-121.9", "14", "E", "2020"],
                          ["PVD", "41.7", "-71.4", "15", "W", "2015"]])
    assert d["magVar"]["SNS"] == [14.0, 2020]
    assert d["magVar"]["PVD"] == [-15.0, 2015]
    assert d["fixes"]["SNS"] == [36.6, -121.9]


def test_magvar_never_guessed_or_shadowed(tmp_path):
    d = _build(tmp_path, [["NOVAR", "30.0", "-90.0", "", "", ""],
                          ["SHADOW", "35.0", "-95.0", "10", "E", "2020"],
                          ["DUP", "31.0", "-91.0", "5", "E", "2020"],
                          ["DUP", "32.0", "-92.0", "9", "W", "2020"]])
    assert "NOVAR" in d["fixes"] and "NOVAR" not in d["magVar"]
    assert d["fixes"]["SHADOW"] == [40.0, -100.0] and "SHADOW" not in d["magVar"]
    assert d["magVar"]["DUP"] == [5.0, 2020]  # first navaid row wins, as for position
