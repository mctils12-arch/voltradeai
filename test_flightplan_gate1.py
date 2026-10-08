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

def test_dist_band_splits_terminal_enroute_and_long_haul():
    far = lambda lon: plan(points=[{"lat": 40, "lon": -100}, {"lat": 40, "lon": lon}])
    assert g1.dist_band(far(-97), ROW) == "60-150"   # ~138 nm
    assert g1.dist_band(far(-95), ROW) == "150-400"  # ~230 nm
    assert g1.dist_band(far(-90), ROW) == ">400"     # ~460 nm

def test_reject_reason_mirrors_measure_and_is_none_when_measurable():
    assert g1.reject_reason(plan(), ROW) is None
    assert g1.reject_reason(None, ROW) == "no_plan"
    assert g1.reject_reason(plan(source="X"), ROW) == "not_filed"
    assert g1.reject_reason(plan(pathEstimated=True), ROW) == "unplaced"
    assert g1.reject_reason(plan(deviation={"crossTrackNm": None}), ROW) == "no_crosstrack"
    near = plan(points=[{"lat": 40, "lon": -100}, {"lat": 40.1, "lon": -100}])
    assert g1.reject_reason(near, ROW) == "near_dest"
    for p in (plan(), plan(pathEstimated=True), plan(deviation={"crossTrackNm": None}), near, None):
        assert (g1.reject_reason(p, ROW) is None) == (g1.measure(p, ROW) is not None)

def test_measure_radial_shadow_reads_report_only_field_with_same_filters():
    sh = {"radialShadow": {"crossTrackNm": 4.5, "radialTokens": 1, "pointCount": 5}}
    assert g1.measure_radial_shadow(plan(pathEstimated=True, **sh), ROW) == 4.5
    assert g1.measure_radial_shadow(plan(**{}), ROW) is None                       # no shadow field
    assert g1.measure_radial_shadow(plan(source="X", **sh), ROW) is None           # not filed
    assert g1.measure_radial_shadow(plan(radialShadow={"crossTrackNm": None}), ROW) is None
    near = [{"lat": 40, "lon": -100}, {"lat": 40.1, "lon": -100}]
    assert g1.measure_radial_shadow(plan(points=near, **sh), ROW) is None          # terminal area
    assert g1.measure_radial_shadow(None, ROW) is None

def test_pool_unique_dedupes_by_hex_first_observation_wins():
    r1 = {"radial_samples": [["a", 1.0], ["b", 2.0]]}
    r2 = {"radial_samples": [["b", 9.0], ["c", 3.0]]}   # b re-sampled: 9.0 must be ignored
    s, n_raw = g1.pool_unique([r1, r2])
    assert n_raw == 4 and s["n"] == 3 and s["p90"] == 3.0 and s["median"] == 2.0
    assert g1.pool_unique([{}]) == ({"n": 0}, 0)
