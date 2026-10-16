"""Offline tests for sec_filings_classify.py (Lazy Prices PR 2b)."""
import json

import sec_filings_classify as c
import sec_filings_text as sft


def doc(sic, etype="operating", tickers=("AAA",)):
    return {"cik": 5, "name": "X", "entityType": etype, "sic": sic, "sicDescription": "d",
            "tickers": list(tickers), "exchanges": [], "formerNames": [{"name": "OLD"}, {}]}


def test_slim_profile_normalizes():
    p = c.slim_profile(doc("3571"), "2026-10-16")
    assert p["cik"] == "0000000005" and p["sic"] == "3571" and p["formerNames"] == ["OLD"]
    assert c.slim_profile(doc(" "), "d")["sic"] is None


def test_classify_vehicles_operating_unknown():
    assert c.classify({"sic": "6189"}) == c.KIND_VEHICLE      # ABS trust
    assert c.classify({"sic": "6770"}) == c.KIND_VEHICLE      # blank check
    assert c.classify({"sic": "2834"}) == c.KIND_OPERATING
    assert c.classify({"sic": None}) == c.KIND_UNKNOWN        # never trusted as operating
    assert c.classify({"sic": "0000"}) == c.KIND_UNKNOWN
    # delisted company: no tickers, still operating by SIC
    assert c.classify(c.slim_profile(doc("1311", tickers=()), "d")) == c.KIND_OPERATING


def test_survivorship_by_kind_separates_vehicles():
    profs = {"a": {"sic": "3571"}, "b": {"sic": "3571"}, "c": {"sic": "6189"}, "d": {"sic": None}}
    r = c.survivorship_by_kind(["a", "b", "c", "d"], profs, current_ciks={"a"})
    assert r["operating"] == {"n": 2, "not_current": 1, "not_current_share": 0.5}
    assert r["vehicle"]["not_current_share"] == 1.0
    assert r["unknown"]["n"] == 1
    assert c.survivorship_by_kind([], {}, set())["operating"]["not_current_share"] is None


def test_get_profile_caches_write_once(tmp_path):
    calls = []

    class FakeClient:
        def get_json(self, url):
            calls.append(url)
            return doc("3571")

    p1 = c.get_profile("5", FakeClient(), str(tmp_path))
    p2 = c.get_profile("0000000005", FakeClient(), str(tmp_path))
    assert p1 == p2 and len(calls) == 1 and calls[0].endswith("CIK0000000005.json")
    assert json.loads(sft.read_gz(c.profile_path(str(tmp_path), "5")))["sic"] == "3571"
