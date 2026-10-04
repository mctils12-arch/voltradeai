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
