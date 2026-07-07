"""test_gridvision_naip_stac.py — GRID VISION Phase B MPC STAC helper battery
(scripts/gridvision_naip_stac.py). No network: the STAC feature fixture is a
VERBATIM (trimmed) item from the real MPC /search response over an Austin-area
bbox on 2026-07-07; the sign/search fetchers are injected.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "gridvision_naip_stac", os.path.join(os.path.dirname(__file__), "scripts", "gridvision_naip_stac.py"))
naip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naip)

# VERBATIM real MPC naip item (2026-07-07 probe) + a synthetic older one so
# best_item has to actually pick the most recent.
REAL_ITEM_2022 = {
    "id": "tx_m_3009751_nw_14_060_20220608",
    "bbox": [-97.753936, 30.184639, -97.683024, 30.253346],
    "properties": {"datetime": "2022-06-08T16:00:00Z", "naip:year": "2022",
                   "naip:state": "tx", "gsd": 0.6},
    "assets": {"image": {
        "href": "https://naipeuwest.blob.core.windows.net/naip/v002/tx/2022/tx_060cm_2022/30097/m_3009751_nw_14_060_20220608.tif",
        "type": "image/tiff; application=geotiff; profile=cloud-optimized"}},
}
OLDER_ITEM_2018 = {
    "id": "tx_m_3009751_nw_14_060_20180101",
    "bbox": [-97.75, 30.18, -97.68, 30.25],
    "properties": {"datetime": "2018-01-01T16:00:00Z", "naip:year": "2018",
                   "naip:state": "tx", "gsd": 1.0},
    "assets": {"image": {"href": "https://naipeuwest.blob.core.windows.net/naip/x.tif",
                         "type": "image/tiff; application=geotiff; profile=cloud-optimized"}},
}
FC = {"type": "FeatureCollection", "features": [OLDER_ITEM_2018, REAL_ITEM_2022]}


def test_build_search_body_shape():
    b = naip.build_search_body([-97.8, 30.2, -97.7, 30.3], "2018-01-01", "2023-12-31", limit=25)
    assert b["collections"] == ["naip"]
    assert b["bbox"] == [-97.8, 30.2, -97.7, 30.3]
    assert b["datetime"] == "2018-01-01T00:00:00Z/2023-12-31T23:59:59Z"
    assert b["limit"] == 25
    with pytest.raises(ValueError):
        naip.build_search_body([1, 2, 3], "2018-01-01", "2023-12-31")


def test_parse_items_carries_capture_date():
    items = naip.parse_items(FC)
    assert len(items) == 2
    it = items[1]
    assert it["id"] == "tx_m_3009751_nw_14_060_20220608"
    assert it["capture_date"] == "2022-06-08"      # freshness honesty on every record
    assert it["naip_year"] == "2022" and it["state"] == "tx" and it["gsd"] == 0.6
    assert it["image_href"].endswith("_20220608.tif")
    assert "cloud-optimized" in it["image_type"]
    assert it["signed_href"] is None
    # missing datetime -> capture_date None, never guessed
    bare = naip.parse_items({"features": [{"id": "x", "properties": {}, "assets": {}}]})
    assert bare[0]["capture_date"] is None and bare[0]["image_href"] is None


def test_best_item_picks_most_recent():
    best = naip.best_item(naip.parse_items(FC))
    assert best["capture_date"] == "2022-06-08"
    # no usable items -> None
    assert naip.best_item([]) is None
    assert naip.best_item([{"capture_date": None, "image_href": "z", "id": "a"}]) is None
    assert naip.best_item([{"capture_date": "2020-01-01", "image_href": None, "id": "a"}]) is None


def test_warn_if_stale_request():
    assert naip.warn_if_stale_request("2023-12-31") is None
    assert naip.warn_if_stale_request("2025-06-01") is not None
    assert "2023-12-31" in naip.warn_if_stale_request("2025-06-01")


def test_search_with_injected_fetch():
    calls = {}

    def fake_post(url, body, timeout=60):
        calls["url"] = url
        calls["body"] = body
        return FC

    items, warning = naip.search([-97.8, 30.2, -97.7, 30.3], "2018-01-01", "2024-06-01",
                                 limit=5, post=fake_post)
    assert calls["url"] == naip.SEARCH_URL
    assert calls["body"]["limit"] == 5
    assert len(items) == 2
    assert warning is not None  # asked past 2023 mirror end


def test_sign_href_with_injected_get():
    def fake_get(url, timeout=60):
        assert url.startswith(naip.SIGN_URL + "?href=")
        return {"href": "https://blob/x.tif?st=2026&sig=abc", "msft:expiry": "2026-07-08T00:07:22Z"}

    signed, expiry = naip.sign_href("https://naipeuwest.blob.core.windows.net/x.tif", get=fake_get)
    assert "sig=abc" in signed and expiry.startswith("2026-07-08")
    assert naip.sign_href(None) == (None, None)
