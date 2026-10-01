import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("g1", pathlib.Path(__file__).parent / "scripts" / "flightplan_gate1.py")
g1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g1)

ROW = ["a", -100.0, 40.0, 35000, 450, 90, "UAL12", "B738", 0, "A3", 0, "N1", "x"]

def plan(**kw):
    p = {"source": "FILED_FAA", "pathEstimated": False, "points": [{"lat": 40, "lon": -100}, {"lat": 41, "lon": -90}],
         "deviation": {"crossTrackNm": 3.2}}
    p.update(kw); return p

def test_measure_reads_server_crosstrack_for_placed_plans():
    assert g1.measure(plan(), ROW) == 3.2

def test_measure_rejects_estimated_unmeasurable_and_terminal():
    assert g1.measure(plan(pathEstimated=True), ROW) is None
    assert g1.measure(plan(deviation={"crossTrackNm": None}), ROW) is None
    assert g1.measure(plan(points=[{"lat": 40, "lon": -100}, {"lat": 40.1, "lon": -100}]), ROW) is None
    assert g1.measure(None, ROW) is None

def test_eligible_and_summarize():
    assert g1.eligible(ROW) and not g1.eligible(ROW[:3] + [5000] + ROW[4:])
    s = g1.summarize([1, 2, 3, 20])
    assert s["n"] == 4 and s["le10"] == 0.75

def test_kind_of_defaults_to_unknown_for_older_servers():
    assert g1.kind_of(plan(routeKind="direct")) == "direct"
    assert g1.kind_of(plan()) == "unknown" and g1.kind_of(None) == "unknown"
