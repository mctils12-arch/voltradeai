"""scripts/build_nasr_atc.py (2026-10-14): FAA NASR frequencies + ARTCC rings ->
datacore/aircraft/nasr_atc.json. Synthetic CSVs with the real NASR column names."""
import csv, importlib.util, json, os

_spec = importlib.util.spec_from_file_location(
    "build_nasr_atc", os.path.join(os.path.dirname(__file__), "scripts", "build_nasr_atc.py"))
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)

FRQ_COLS = ["EFF_DATE", "FACILITY", "FAC_NAME", "FACILITY_TYPE", "ARTCC_OR_FSS_ID", "CPDLC", "TOWER_HRS",
            "SERVICED_FACILITY", "SERVICED_FAC_NAME", "SERVICED_SITE_TYPE", "LAT_DECIMAL", "LONG_DECIMAL",
            "SERVICED_CITY", "SERVICED_STATE", "SERVICED_COUNTRY", "TOWER_OR_COMM_CALL",
            "PRIMARY_APPROACH_RADIO_CALL", "FREQ", "SECTORIZATION", "FREQ_USE", "REMARK"]


def _frq(ftype, served, site, freq, sector, use, tc="CHARLOTTE", ac="CHARLOTTE", artcc="", hrs="24", lat="35.2", lon="-80.9"):
    r = dict.fromkeys(FRQ_COLS, "")
    r.update({"EFF_DATE": "2026/10/01", "FACILITY": served, "FACILITY_TYPE": ftype, "ARTCC_OR_FSS_ID": artcc,
              "TOWER_HRS": hrs, "SERVICED_FACILITY": served, "SERVICED_SITE_TYPE": site, "LAT_DECIMAL": lat,
              "LONG_DECIMAL": lon, "TOWER_OR_COMM_CALL": tc, "PRIMARY_APPROACH_RADIO_CALL": ac, "FREQ": freq,
              "SECTORIZATION": sector, "FREQ_USE": use})
    return r


def _write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _build(tmp_path):
    _write(tmp_path / "APT_BASE.csv",
           ["ARPT_ID", "ICAO_ID", "LAT_DECIMAL", "LONG_DECIMAL", "MAG_VARN", "MAG_HEMIS", "TWR_TYPE_CODE", "RESP_ARTCC_ID"],
           [{"ARPT_ID": "CLT", "ICAO_ID": "KCLT", "LAT_DECIMAL": "35.21318694", "LONG_DECIMAL": "-80.95137916",
             "MAG_VARN": "7", "MAG_HEMIS": "W", "TWR_TYPE_CODE": "ATCT-TRACON", "RESP_ARTCC_ID": "ZTL"}])
    _write(tmp_path / "FRQ.csv", FRQ_COLS, [
        _frq("ATCT-TRACON", "CLT", "AIRPORT", "118.1", "RWY 18/36", "LCL/P"),
        _frq("ATCT-TRACON", "CLT", "AIRPORT", "257.8", "", "LCL/P"),            # UHF: dropped
        _frq("ATCT-TRACON", "CLT", "AIRPORT", "121.5", "", "EMERG"),            # guard: dropped
        _frq("ATCT-TRACON", "CLT", "AIRPORT", "120.05", "120-295 8000 FT & BLW", "APCH/P DEP/P IC"),
        _frq("ATCT-TRACON", "CLT", "AIRPORT", "120.05", "WEST", "CHARLOTTE DP"),  # DP-specific: dropped
        _frq("ATCT-TRACON", "CLT", "AIRPORT", "122.95", "", "UNICOM", tc=""),
        _frq("ATCT-TRACON", "CLT", "AIRPORT", "112.9/76X", "", "LCL/P"),        # navaid voice form: dropped
        _frq("RCAG", "ALBEMARLE", "RCAG", "133.15", "LOW", "ALBEMARLE RCAG", artcc="ZTL", lat="35.3704", lon="-80.5067"),
        _frq("RCAG", "ALBEMARLE", "RCAG", "133.15", "LOW", "ALBEMARLE RCAG", artcc="ZTL", lat="35.3704", lon="-80.5067"),
        _frq("RCAG", "ALBEMARLE", "AIRPORT", "127.55", "LOW", "ALBEMARLE RCAG", artcc="ZTL"),  # serviced-airport row: not a site
    ])
    seg_cols = ["LOCATION_ID", "LOCATION_NAME", "ALTITUDE", "TYPE", "POINT_SEQ", "LAT_DECIMAL", "LONG_DECIMAL", "BNDRY_PT_DESCRIP"]
    ring = [(36, -82), (36, -79), (34, -79), (34, -82)]
    segs = []
    for alt in ("LOW", "HIGH"):
        for i, (la, lo) in enumerate(ring):
            segs.append({"LOCATION_ID": "ZTL", "LOCATION_NAME": "ATLANTA", "ALTITUDE": alt, "TYPE": "ARTCC",
                         "POINT_SEQ": str((i + 1) * 10), "LAT_DECIMAL": str(la), "LONG_DECIMAL": str(lo),
                         "BNDRY_PT_DESCRIP": "TO POINT OF BEGINNING" if i == 3 else "TO"})
    for k, rr in enumerate(([(10, 170), (10, 179), (5, 179)], [(20, -170), (20, -160), (15, -160)])):
        for i, (la, lo) in enumerate(rr):
            segs.append({"LOCATION_ID": "ZAK", "LOCATION_NAME": "OAKLAND OCEANIC ARTCC", "ALTITUDE": "UNLIMITED",
                         "TYPE": "CTA/FIR", "POINT_SEQ": str(k * 100 + (i + 1) * 10), "LAT_DECIMAL": str(la),
                         "LONG_DECIMAL": str(lo), "BNDRY_PT_DESCRIP": "TO POINT OF BEGINNING" if i == 2 else "TO"})
    _write(tmp_path / "ARB_SEG.csv", seg_cols, segs)
    out = tmp_path / "out.json"
    b.main(str(tmp_path), str(out), check=False)
    return json.loads(out.read_text())


def test_kinds_parse_the_published_freq_use_forms():
    assert b.kinds("LCL/P") == ["T"]
    assert b.kinds("APCH/P DEP/P IC") == ["AD"]
    assert b.kinds("APCH/S") == ["AP"]
    assert b.kinds("DEP/P") == ["DP"]
    assert b.kinds("GND/P;CD/P") == ["G", "C"]
    assert b.kinds("D-ATIS") == ["A"]
    assert b.kinds("CHARLOTTE DP") == [] and b.kinds("BTSEY STAR") == [] and b.kinds("EMERG") == []


def test_airport_rows_vhf_only_icao_keyed_magvar_signed(tmp_path):
    d = _build(tmp_path)
    assert d["cycle"] == "2026-10-01"
    (clt,) = d["airports"]
    assert clt["id"] == "KCLT" and clt["lid"] == "CLT" and clt["mv"] == -7.0 and clt["art"] == "ZTL"
    assert clt["tc"] == "CHARLOTTE" and clt["h"] == "24"
    assert clt["f"] == [[118.1, "T", "RWY 18/36"], [120.05, "AD", "120-295 8000 FT & BLW"], [122.95, "UN", ""]]


def test_center_sites_deduped_and_rings_split_on_point_of_beginning(tmp_path):
    d = _build(tmp_path)
    assert d["rcag"] == [[35.3704, -80.5067, "ZTL", 133.15, "L", "ALBEMARLE"]]
    assert d["artcc"]["ZTL"]["n"] == "ATLANTA"
    assert len(d["artcc"]["ZTL"]["L"]) == 4 and len(d["artcc"]["ZTL"]["H"]) == 4
    assert len(d["artcc"]["ZAK"]["U"]) == 2, "oceanic: one list per ring"
    assert b.point_in_ring(35.2, -80.9, d["artcc"]["ZTL"]["L"])
    assert not b.point_in_ring(37, -80.9, d["artcc"]["ZTL"]["L"])


def test_committed_table_matches_the_fix_cycle():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "datacore", "aircraft", "nasr_atc.json")) as f:
        atc = json.load(f)
    with open(os.path.join(here, "datacore", "aircraft", "nasr_fixes.json")) as f:
        fixes = json.load(f)
    assert atc["cycle"] == fixes["cycle"], "refresh_nasr rebuilds both from one bundle"
