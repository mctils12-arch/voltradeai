"""FAA chart editions: the 56-day cycle, where each family's GeoTIFFs live on
aeronav.faa.gov, and discovery of the newest published edition.

The FAA posts a cycle's files ~20 days before the effective date. Discovery
probes the current and next cycle dates and returns every family whose
directory listing exists — no hand-maintained dates or file lists: chart
names come from the FAA's own directory listing each run.

Pure functions are unit-tested in tests/test_faa_charts.py.
"""
from __future__ import annotations

import datetime as dt
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Mirrors server/aeroCharts.ts FAA_CYCLE_ANCHOR / FAA_CYCLE_DAYS (cross-checked
# there against the FAA service's own editions 2026-05-14 and 2026-07-09;
# 2026-09-03 = anchor + 4 cycles is the aeronav directory that exists).
FAA_CYCLE_ANCHOR = dt.date(2026, 1, 22)
FAA_CYCLE_DAYS = 56
AERONAV = "https://aeronav.faa.gov"


def cycle_start(day: dt.date) -> dt.date:
    """Start of the 56-day cycle containing `day`."""
    k = (day - FAA_CYCLE_ANCHOR).days // FAA_CYCLE_DAYS
    return FAA_CYCLE_ANCHOR + dt.timedelta(days=k * FAA_CYCLE_DAYS)


def next_cycle(day: dt.date) -> dt.date:
    return cycle_start(day) + dt.timedelta(days=FAA_CYCLE_DAYS)


def aeronav_date(d: dt.date) -> str:
    """aeronav directory segment: MM-DD-YYYY."""
    return d.strftime("%m-%d-%Y")


@dataclass(frozen=True)
class Family:
    """One of our four chart layers (server/aeroCharts.ts AeroChartId), the
    FAA tile service it replaces, where its GeoTIFFs live, and the zoom band
    our bake covers. `max_bake_zoom` is the level at which the chart's native
    resolution is reached; deeper levels are enlarged by the server/client,
    exactly as they are today past the FAA's own top level."""
    id: str
    service: str
    directory: str  # relative to an edition date: "visual/{d}/sectional-files"
    file_re: str
    min_zoom: int  # the FAA service's lowest level (overviews are built below)
    max_bake_zoom: int
    measure_zoom: int  # coarse ownership measurement level
    refine_zoom: int  # edge refinement level (within the FAA band)


FAMILIES: Dict[str, Family] = {
    # Sectional 1:500,000 scanned at ~42 m/px: z11 (~52 m/px at the equator,
    # ~37 m/px at 45N) already carries the full native detail.
    "sectional": Family("sectional", "VFR_Sectional", "visual/{d}/sectional-files", r"^[A-Za-z_.\-]+\.zip$", 8, 11, 10, 11),
    # TAC 1:250,000 (~21 m/px): z12.
    "tac": Family("tac", "VFR_Terminal", "visual/{d}/tac-files", r"^[A-Za-z_.\-]+_TAC\.zip$", 10, 12, 10, 12),
    # IFR enroute low (ENR_L*) + area charts (ENR_A*); Alaska low (ENR_AKL*).
    "ifrlow": Family("ifrlow", "IFR_AreaLow", "enroute/{d}", r"^ENR_(L\d+|A\d+|AKL\d+)\.zip$", 8, 11, 8, 11),
    # IFR enroute high (ENR_H*, ENR_AKH*); the FAA band tops out at z9.
    "ifrhigh": Family("ifrhigh", "IFR_High", "enroute/{d}", r"^ENR_(H\d+|AKH\d+)\.zip$", 5, 9, 7, 9),
}


_HREF = re.compile(r'href="([^"]+)"', re.I)


def parse_listing(html: str, file_re: str) -> List[str]:
    """File names (sorted, de-duplicated) from an aeronav directory listing
    that match the family's pattern."""
    pat = re.compile(file_re)
    names = set()
    for h in _HREF.findall(html):
        name = h.rstrip("/").rsplit("/", 1)[-1]
        if pat.match(name):
            names.add(name)
    return sorted(names)


def family_url(fam: Family, edition: dt.date, name: str = "") -> str:
    base = f"{AERONAV}/{fam.directory.format(d=aeronav_date(edition))}/"
    return base + name


def _http_get(url: str, timeout: float = 30.0) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"user-agent": "VolTradeAI-chartbake/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def list_family(fam: Family, edition: dt.date, get: Callable[[str], Optional[str]] = _http_get) -> List[str]:
    html = get(family_url(fam, edition))
    return parse_listing(html, fam.file_re) if html else []


def newest_published(fam: Family, today: dt.date,
                     get: Callable[[str], Optional[str]] = _http_get) -> Optional[dt.date]:
    """The newest edition whose files are posted: the next cycle if the FAA
    has already published it (they post ~20 days early), else the current
    one. None when neither directory lists any files."""
    for ed in (next_cycle(today), cycle_start(today)):
        if list_family(fam, ed, get):
            return ed
    return None
