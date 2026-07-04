# SENTINEL-2 CHANGE DETECTION — pipeline spec (Tier 3, geospatial directive 2026-07-04)

Status: SPEC — no build yet. This generalizes the Cushing tank-shadow
idea (EDGE DOCTRINE standing example) and the facility watch-list into
ONE system: weekly, facility-scale change detection at every strategic
site. The honest version of "object counting" at 10m resolution:
**activity indices, never object counts.**

## What it is (and honestly is not)

- IS: "activity index up/down at facility X this week" — changes in
  surface reflectance/area over a facility's yards, laydown areas,
  stockpiles, and tank farms, at Sentinel-2's 10m/pixel resolution,
  labeled facility-scale change.
- IS NOT: car/coil/railcar counting (needs sub-meter, paid — see the
  wishlist gate at the bottom), daily cadence (revisit is ~5 days per
  satellite pair, effectively weekly after clouds), or ground truth
  (it is an ESTIMATE and is labeled as one everywhere it surfaces).

## Data source

Copernicus Sentinel-2 L2A (atmospherically corrected surface
reflectance), free and open under the Copernicus data policy
(attribution "Contains modified Copernicus Sentinel data"). Access:
Copernicus Data Space Ecosystem (CDSE) — free account, OData/STAC
catalog + S3-compatible download; per-band windowed reads via COG
range requests keep transfer tiny (a facility AOI is ~100x100 px = KB
per band, not full 100km scenes). LICENSING CHECK required at build
time per the standing rule (CDSE quota terms for API accounts) — a
spec states this; the build PR verifies it from the primary source.

## Per-facility measurements (v1)

For each AOI (polygon around the verified facility footprint — derived
from the imagery-verification work, which already established WHERE
each facility is):

1. **Yard-occupancy index**: fraction of AOI pixels whose visible-band
   reflectance departs from that pixel's own rolling seasonal baseline
   by >2σ. Steel yards: finished coil/slab stacks change local
   brightness/texture; laydown areas fill and empty.
2. **Tank-shadow index (tank farms only)**: NDVI-masked dark-pixel
   fraction on the sun-facing side of each floating-roof tank ring
   (shadow length ∝ roof depth ∝ inventory). Cushing is the flagship
   AOI; EIA weekly Cushing stocks are the published ground truth.
3. **Water/berth occupancy (ports)**: NDWI water mask changes at
   berths — large hulls read as non-water pixels inside the berth
   polygon; verifies AIS-derived in-port counts (port-dwell pipeline).

Each reading: {facility_id, scene_id, capture_ts, cloud_pct, index
values, baseline window} — archived per the doctrine (readings JSONL +
scene IDs; we never re-host imagery).

## Processing budget (Railway constraint)

All heavy processing runs OFF the serving path: a scheduled job (Tier-3
cadence or session-run script) pulls AOI windows, computes indices,
appends readings. The site serves only the stored readings. No
server-side geometry per visitor (DESIGN.md performance budget).

## Cloud/quality handling (the honest failure mode)

Scenes with cloud_pct > 40% over the AOI are recorded but flagged
unusable; indices carry a quality flag; missing weeks stay missing
(never interpolated silently). Winter sun angles change shadow
geometry — the tank-shadow index normalizes by solar elevation from
scene metadata or it is fiction.

## Ladder path (per facility class, separately)

- GATE 1 (DATA): tank farms — weekly tank-shadow index vs EIA Cushing
  stocks (published weekly; the exact precedent in CLAUDE.md); steel
  yards — quarterly yard index vs STLD disclosed shipments (fusion (a)
  ground truth); ports — berth index vs our own AIS in-port counts.
- GATE 2 (SIGNAL): validated indices vs forward returns of the tied
  exposure (USO/XLE, STLD, XRT/IYT) against random-entry base rates,
  regime-split.
- SURFACE: nothing appears on /data before gate 2 EXCEPT the raw
  imagery-date-honest scene metadata ("last usable scene: date") —
  which is RAW. Index values are SIGNAL-class.

## What sub-meter paid imagery would add (wishlist gate)

Actual object counting (cars in employee lots, coils on yards, railcars
at sidings): needs 0.3–0.5m imagery — Planet SkySat/Pelican tasking or
Maxar archive, both quote-priced; archive singles historically ~$10–25/km²
class with minimum orders; monitoring subscriptions $1000s/mo class.
BUILD-FIRST verdict: the free 10m version must first prove the
facility-scale concept (pass gate 2 on ANY facility class) before a
sub-meter purchase may enter wishlist.md with exact quotes — buying
counting resolution for an unvalidated signal class would be paying to
decorate a hypothesis.

## Build order (when a [PIPELINE] session picks this up)

1. AOI polygons for the 16 verified sites (extend site_verify.py output).
2. CDSE access + windowed COG reader + readings archive (licensing
   check first).
3. Cushing tank-shadow index + EIA reconciliation (gate 1 attempt).
4. Yard indices for the 4 STLD mills; port berth masks for the 9 ports.
5. /data surface for scene metadata (RAW); indices stay gated.
