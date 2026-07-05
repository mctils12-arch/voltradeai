"""
test_entity_spine.py — aircraft entity spine build script battery
(scripts/build_entity_spine.py). Fixture rows are copied VERBATIM from the
real FAA files downloaded 2026-07-05 (public domain), including the header
BOM, space-padded fields, and trailing comma — the parser is judged against
the actual format, not a cleaned-up one.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "build_entity_spine", os.path.join(os.path.dirname(__file__), "scripts", "build_entity_spine.py"))
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

MASTER_HEADER = ("﻿N-NUMBER,SERIAL NUMBER,MFR MDL CODE,ENG MFR MDL,YEAR MFR,TYPE REGISTRANT,NAME,"
                 "STREET,STREET2,CITY,STATE,ZIP CODE,REGION,COUNTY,COUNTRY,LAST ACTION DATE,"
                 "CERT ISSUE DATE,CERTIFICATION,TYPE AIRCRAFT,TYPE ENGINE,STATUS CODE,MODE S CODE,"
                 "FRACT OWNER,AIR WORTH DATE,OTHER NAMES(1),OTHER NAMES(2),OTHER NAMES(3),"
                 "OTHER NAMES(4),OTHER NAMES(5),EXPIRATION DATE,UNIQUE ID,KIT MFR, KIT MODEL,"
                 "MODE S CODE HEX,")

SWA_ROW = ("1801U,60223                         ,13844FZ,13124,2022,3,"
           "SOUTHWEST AIRLINES CO                             ,2702 LOVE FIELD DR # HDQ-4GC     ,"
           "                                 ,DALLAS            ,TX,752351908 ,2,113,US,20230818,"
           "20220712,1T        ,5,5 ,V ,50241472, ,20220405,"
           "                                                  ,"
           "                                                  ,"
           "                                                  ,"
           "                                                  ,"
           "                                                  ,"
           "20290731,1355056,          ,                    ,A1433A    ,")

ACFTREF_TEXT = ("﻿CODE,MFR,MODEL,TYPE-ACFT,TYPE-ENG,AC-CAT,BUILD-CERT-IND,NO-ENG,NO-SEATS,AC-WEIGHT,SPEED,TC-DATA-SHEET,TC-DATA-HOLDER,\n"
                "13844FZ,BOEING                        ,737-8               ,5,5 ,1,0,02,175,CLASS 3,0000,               ,                                                  ,\n")


def test_parse_acftref_real_row():
    ref = sp.parse_acftref(ACFTREF_TEXT)
    assert ref["13844FZ"] == {"mfr": "BOEING", "model": "737-8"}


def test_parse_master_real_row_matched():
    ref = sp.parse_acftref(ACFTREF_TEXT)
    ents = sp.parse_master(MASTER_HEADER + "\n" + SWA_ROW + "\n", {"a1433a"}, ref)
    e = ents["a1433a"]
    assert e["n_number"] == "N1801U"
    assert e["owner"] == "SOUTHWEST AIRLINES CO"
    assert e["registrant_type"] == "corporation"
    assert e["mfr"] == "BOEING" and e["model"] == "737-8"
    assert e["year_mfr"] == 2022
    assert e["cert_issue"] == "2022-07-12"
    assert e["expiration"] == "2029-07-31"
    assert e["evidence"]["match"] == "exact Mode S hex"
    assert "public domain" in e["evidence"]["source"]


def test_parse_master_filters_to_wanted_hexes():
    ref = sp.parse_acftref(ACFTREF_TEXT)
    ents = sp.parse_master(MASTER_HEADER + "\n" + SWA_ROW + "\n", {"deadbe"}, ref)
    assert ents == {}


def test_parse_master_refuses_changed_format():
    with pytest.raises(SystemExit):
        sp.parse_master("N-NUMBER,NAME\n123,X\n", {"a1433a"}, {})


def test_date8_honesty():
    assert sp._date8("20220712") == "2022-07-12"
    assert sp._date8("") is None
    assert sp._date8("2022071") is None
    assert sp._date8("NONE    ") is None
