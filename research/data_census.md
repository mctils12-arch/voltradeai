# DATA CENSUS — free/keyless/free-key sources NOT yet archived

Permanent document (DATACORE MAXIMUS Phase 1, started 2026-07-06).
Every entry was PROBED LIVE before filing (HTTP results recorded by the
probing agent; no fabrication). Update protocol: append new sections,
mark built items [BUILT vX], mark declined items [DECLINED: reason].
Excludes everything already in datacore/manifests/ (~38 streams as of
2026-07-06).

## SECTION 1 — INTERNATIONAL + MACRO INSTITUTIONS (probed 2026-07-06)

RANKED TOP-5 of this section (signal × uncrowdedness × build ease):
1. JODI oil/gas, 2. ECB Data Portal, 3. Eurostat, 4. Bundesbank,
5. UN Comtrade preview.

1. **JODI oil/gas [BUILT v1.0.169]** — world_Primary_CSV.zip (23MB→283MB CSV), keyless
   static file, monthly (~19th, 2-mo lag), 2002+. Flat CSV:
   REF_AREA,TIME_PERIOD,ENERGY_PRODUCT,FLOW_BREAKDOWN(CLOSTLV=closing
   stocks),OBS_VALUE. Free w/ acknowledgment. SIGNAL: non-OECD
   crude/product stock builds (Saudi/UAE/India) invisible in EIA →
   Brent structure; moderately uncrowded; DIRECT gate-1 partner for
   the tank-shadow root. Easiest build in the census.
2. **ECB Data Portal** — data-api.ecb.europa.eu keyless SDMX-JSON,
   probed 200 (daily EUR/USD refs). Daily FX/€STR, weekly Eurosystem
   balance sheet, monthly money/lending; 1999+. Free w/ attribution.
   SIGNAL: FX/rates crowded; weekly balance-sheet COMPOSITION changes
   less watched. Regime-feature input.
3. **Eurostat** — ec.europa.eu/eurostat/api/dissemination JSON-stat,
   probed 200 (May-2026 EU industrial production, 2d after release).
   Monthly/quarterly, fixed release calendar, decades of history,
   CC BY 4.0. SIGNAL: headline crowded; NACE-code sector splits are
   the uncrowded layer; release-calendar event studies possible.
4. **Bundesbank** — api.statistiken.bundesbank.de keyless SDMX-CSV,
   probed 200 (daily 10y Bund yield). Daily, 1970s+, free w/
   attribution. SIGNAL: crowded but a cheap daily curve input.
5. **UN Comtrade preview** — comtradeapi.un.org/public keyless
   (~500 rec/req, daily cap; free key raises to 250 calls/day),
   probed 200 (US→China Jan-2026). Monthly, 1-6mo reporter lag,
   2010+. Free w/ citation, bulk redistribution needs permission.
   SIGNAL: HS-6 bilateral flows for slow structural theses; most
   uncrowded of the section; too lagged for direct alpha.
6. **OECD SDMX** — sdmx.oecd.org keyless, probed 200 (US CLI).
   CC BY 4.0. SIGNAL: CLI dispersion as regime feature; crowded/slow.
7. **World Bank + Pink Sheet** — keyless, probed 200 (xlsx 765KB
   monthly commodities 1960+). CC BY 4.0. Validation ground truth,
   not alpha; niche commodities (rubber, DAP) uncrowded but slow.
8. **IMF** — legacy API DEAD (dataservices.imf.org DNS gone); new
   api.imf.org/external/sdmx/2.1 probed 200. Monthly PCPS. ~10/5s
   throttle. Reference/validation only.
9. **UK ONS** — legacy API RETIRED 2024-11-25; website JSON + beta
   API probed 200. OGL v3. SIGNAL: the weekly "real-time indicators"
   (card spend, shipping) are the only uncrowded corner.
10. **Bundesbank/BoJ/PBOC** — BoJ: no API, bulk zips probed 200
    (monthly CGPI). PBOC: scrape-hostile (WAF, HTML/xlsx only) —
    build cost high; only if a China thesis demands it.
11. **WTO** — 401 without free key (apiportal.wto.org self-serve).
    Annual/quarterly; near-zero alpha. SKIP unless tariff thesis.

FREE-KEY ITEMS → BLOCKED-FOR-MIKE: none required from this section
for the top-5 (all keyless); WTO key only if ever wanted (skip).

## SECTION 2 — EXCHANGE + MARKET STRUCTURE (probed 2026-07-06)

RANKED TOP-5: 1. OCC volume-query, 2. DTCC SBSDR equities,
3. FINRA Query API cluster, 4. SEC FTD, 5. SEC MIDAS.

1. **OCC daily options volume [BUILT v1.0.165, /api/data/occ-volume]** ⚠️ ARCHIVE-NOW (2-year purge) —
   marketdata.theocc.com/volume-query probed 200 (5MB CSV/day, all
   symbols; per-symbol variant works). Keyless. Fields: qty,
   underlying, actype C=customer/F=firm/M=market-maker, put/call, per
   exchange, daily. HARD 2-YEAR HISTORY CAP (probed) — every day not
   archived is permanently lost. License: informational use;
   redistribution needs OCC permission (bot-internal + gated signals
   OK; raw resale needs review). SIGNAL: customer-vs-MM put/call BY
   TICKER = the retired ISEE's superior successor, genuinely
   uncrowded on small caps. (ISE/ISEE itself probed DEAD — TLS reset,
   retired post-Nasdaq; do not wishlist.)
2. **DTCC SBSDR equity swaps** — pddata.dtcc.com cumulative
   SEC_CUMULATIVE_EQUITIES probed 200 (147MB zip → 1.24GB CSV/day):
   every disseminated equity total-return swap (the Archegos
   instrument) — notional legs, timestamps, underlier codes. Keyless,
   daily. License: SEC-mandated public dissemination, attribution.
   SIGNAL: fresh large TRS notional clustered on small/mid caps =
   near-zero-retail-competition positioning signal. COST: 147MB/day
   needs a dedicated pipeline (volume budget!), underlier parsing is
   real work. www.dtcc.com itself 403s (Akamai); CFTC-side slices
   503 from this environment.
3. **FINRA Query API cluster [BUILT — part 1 short-interest/threshold, part 2 weekly/monthly/blocks summaries]** — api.finra.org keyless, probed 200:
   consolidatedShortInterest (bi-monthly, 2017-12→present, 204
   settlement dates, days-to-cover precomputed),
   weeklySummary/monthlySummary/blocksSummary (ATS/dark-pool venue
   volumes, 2021-12→present), thresholdList (daily, 2016→present).
   SIGNAL: aggregate dark-pool % is crowded (DIX clones); venue-level
   COMPOSITION shift pre-announcement × Form 4 clusters is not.
   Threshold persistence × FTD × daily short volume = the
   settlement-stress composite nobody computes.
4. **SEC fails-to-deliver [BUILT v1.0.171, server/secFtd.ts]** — sec.gov cnsfails zips probed 200
   (bi-monthly halves, 2004→present, public domain, resale-safe).
   SIGNAL: raw FTD spikes maximally crowded; edge only in the
   composite above.
5. **SEC MIDAS [BUILT v1.0.265, server/secMidas.ts]** — quarterly per-security per-day lit/hidden/odd-lot/
   cancel metrics, probed 200 (2013→2025Q4, 68MB CSV/quarter, public
   domain). SIGNAL: cross-sectional HFT-colonization score
   (cancel/trade ratio) = a FILTER protecting EDGE DOCTRINE #2 (which
   small caps are already colonized by fast money).
6. **CBOE daily stats** — cdn.cboe.com probed 200 (daily P/C ratios
   JSON to 2020; VIX_History.csv 1990→present; per-expiry VX futures
   settle CSVs = free VIX term structure). License: informational
   use; resale needs Cboe license. SIGNAL: total P/C zero edge;
   equity-vs-index P/C spread × small-cap short volume is the
   residual. VX curve replaces any need for CME.
7. **Nasdaq threshold list** — nasdaqtrader.com daily pipe-delimited
   probed 200, date-substituted URLs archive. NYSE's equivalent API
   is DEAD headless (500s + bot check) — FINRA thresholdList covers
   both.
8. **FINRA margin statistics** — monthly xlsx probed 200 (1997→,
   needs identifying research UA — generic UA 403s). Regime feature
   only.
9. **CME** — DEAD from server infra (Akamai 403 on everything, FTP
   TLS reset). Covered by: CFTC COT (positioning, live), CBOE VX
   CSVs (vol curve). Only unlock path: free CME DataMine registration
   (human) + likely still browser-fingerprinted — NOT recommended.

## SECTION 3 — OPEN INFRASTRUCTURE + SCIENTIFIC (probed 2026-07-06)

RANKED: 1. EPA CAMD CEMS, 2. Global Energy Monitor, 3. ENTSO-E.

1. **EPA CAMD CEMS ★ THE STANDOUT [BUILT v1.0.385, server/epaCamd.ts,
   /api/data/plant-operations — TX pilot]** — api.epa.gov/easey probed
   200 with DEMO_KEY: UNIT-LEVEL power-plant operations (grossLoad MW,
   opTime, SO2/CO2/NOx mass, heat input, fuel, unit type) for every US
   Part-75 plant; bulk files to 1995. Public domain. SIGNAL:
   grossLoad×opTime = DIRECT plant-utilization ground truth —
   ladder-gate-1 truth source for the whole power vertical
   (validates GPPD/satellite-thermal inferences) + merchant-generator
   earnings estimates. CORRECTED cadence honesty (live-probed
   2026-07-18, supersedes the original "hourly"/"partial earlier
   arrival" framing above): this is QUARTERLY data — the in-progress
   quarter is rejected outright by the live API (the completed prior
   quarter is the upper bound), not partially available early. "FREE
   KEY → BLOCKED-FOR-MIKE" was ALSO stale: the shared DEMO_KEY works
   today (live-verified, 445 facility rows + 9,009 daily unit-rows for
   one TX quarter in one call each) — a dedicated api.data.gov key
   (wishlist 9a, still worth getting) only raises the ceiling past the
   TX pilot, it was never a hard blocker on building this. /data map
   layer SHIPPED 2026-07-20 (`plant_operations`, facilities group) —
   TX facility markers tinted by ground-truth operating-hours
   utilization, not fuel type.
2. **Global Energy Monitor [BUILT v1.0.176, scripts/gem_ingest.py —
   9b resolved, Mike enabled Drive access]** — March 2026 release:
   182,400 facilities, 22,296 GW, 200 countries, unit-level with status
   (announced/construction/operating). CC BY 4.0 (no share-alike!) —
   joinable into proprietary products with attribution.
   SIGNAL: status transitions per owner ticker = capex/commissioning
   timelines.
3. **ENTSO-E Transparency [BUILT v1.0.186, server/euLoad.ts,
   /api/data/eu-load — token resolved 2026-07-07]** — web-api.tp.entsoe.eu
   alive (probed: token-enforced XML). Hourly EU load/generation/prices
   by bidding zone, 2015→. 400 req/min. SIGNAL: EU zonal spreads →
   utilities, gas, carbon. Shipped: actual total load (A65/A16) for 8
   zones; generation mix + day-ahead prices filed as follow-ups (same
   token, separate builds, not yet built).
4. **OSM power features** — Overpass main instance probed 200 (1.2s);
   CRITICAL OPS FINDINGS: no-User-Agent = 406; both community
   mirrors UNREACHABLE from our proxy (mirror failover is worthless
   for us); 2 concurrent slots/IP, <10k queries/day fair use.
   Geofabrik us-latest.osm.pbf = 12.0GB daily rebuilds, probed 200.
   ODbL PRECISE READ: signals/analyses (produced works) sellable
   with attribution; a redistributed geometry DATABASE must be ODbL
   (share-alike). Taginfo verified: power=line 1.16M ways,
   minor_line 1.56M, towers 18.4M nodes.
5. **OpenInfraMap** — no API/tiles; architecture reference only
   (imposm3→PostGIS→Tegola). Our lighter equivalent: osmium
   tags-filter → tippecanoe/planetiler → static PMTiles (~150-400MB
   global HV estimated, no runtime DB, HTTP range requests).
6. **OpenAQ v3** — 401 without free key (60/min, 2k/hr free tier;
   S3 bulk archive keyless). SIGNAL: SO2/NO2 downwind of smelters/
   refineries = operating-rate proxy outside CEMS coverage. FREE
   KEY → BLOCKED-FOR-MIKE (low priority; S3 bulk path exists).
7. **USGS earthquakes** — GeoJSON feeds probed 200, minute
   regeneration, public domain. SIGNAL: M5+ near mapped facilities
   (our graph) → supply-disruption events within minutes.
8. **NOAA NDBC buoys** — realtime2 text probed 200 (10-min rows,
   45-day rolling; decades in historical/). Public domain. SIGNAL:
   wave/wind at port-approach buoys → dwell spikes 1-3 days out
   (joins port-dwell archive).
9. **Esri World Imagery metadata** ✓ EXACT ENDPOINT FOUND (Phase 3a
   unblocked): World_Imagery/MapServer/identify with point geometry
   returns {DATE (YYYYMMDD), RESOLUTION, ACCURACY, SOURCE, DESCRIPTION
   sensor, MinMapLevel/MaxMapLevel} per point — scale-dependent via
   imageDisplay/mapExtent params. Also
   metadata.maptiles.arcgis.com: 197 versioned services, newest
   World_Imagery_Metadata_2026_r06 (queryable footprints — pairs
   with Wayback for change detection). Keyless reads; Esri terms:
   internal recency checks fine, no redistribution.
10. **CDSE quotas** ✓ EXACT NUMBERS (Phase 3b constraint): free tier
    = 10,000 requests AND 10,000 PU/month, 300/min. One 1280×720 S2
    render ≈ 3.5 PU → naive per-viewport fetching dies at ~90
    renders/day. PATTERN MANDATE: pre-render fixed facility chips on
    schedule (200 sites × weekly × 1 PU ≈ 8% of quota), cache on
    volume, serve statically — never render per-user-viewport.

## PHASE 2-3 FAILURE-MODE REGISTER (researched before building)

- OVERPASS: never bulk-extract (fair-use + slots); pattern =
  Geofabrik PBF → osmium tags-filter → offline tiles; Overpass only
  for small-bbox freshness diffs with UA header + backoff. Mirrors
  unreachable from our proxy — single-instance dependency is REAL.
- GRID GEOMETRY: ~1-2GB raw global power=line GeoJSON (estimated);
  as PMTiles ~150-400MB → fits the volume, served static via range
  requests, zero runtime DB. Decimation gates: ≥230kV at z<6, ≥100kV
  at z<9, all at z≥11; towers only z≥12.
- MOBILE 390px: vertex density + multi-layer line styling are the
  jank sources — one styled layer with voltage→color ramp, minzoom
  gating, tiles <75KB gz, harness gate mandatory (estimated, to be
  measured in the build).
- CDSE: PU pool is the binding constraint (see #10) — scheduled
  chips, never viewport renders.
- VOLUME GROWTH: DTCC 147MB/day and OCC 5MB/day are the two heavy
  candidates — DTCC needs its own budget decision before build
  (filed in build order, not started by default).

## CONSOLIDATED BLOCKED-FOR-MIKE ADDITIONS (from this census)

(a) EPA CAMD free key — api.data.gov signup, instant, unlocks the
    highest-value root in the census. (b) GEM download form-fill
    (name/email, 2 min). (c) ENTSO-E token (register + email, ≤3
    days). (d) OpenAQ key (low priority — S3 bulk exists).

## CENSUS MASTER RANKING (build order feed, all sections)

1. OCC options volume (archive-now, 2yr purge, keyless)
2. EPA CAMD CEMS (free key → Mike; power-vertical ground truth)
3. JODI oil/gas (keyless static CSV; tank-shadow gate-1 partner)
4. FINRA Query API cluster (keyless; settlement-stress composite)
5. GEM asset registry (form-fill → Mike; CC BY 4.0 join spine)
6. SEC FTD (keyless; composite ingredient)
7. ECB + Eurostat + Bundesbank (keyless; regime features)
8. USGS quakes + NDBC buoys (keyless; event joins to our graph)
9. ENTSO-E (token → Mike; EU power vertical)
10. SEC MIDAS (keyless quarterly; small-cap colonization filter)
11. DTCC SBSDR (keyless but 147MB/day — volume budget decision first)
DEAD/SKIP: CME (bot-blocked), ISE (retired), NYSE threshold API
(bot-blocked; FINRA covers), WTO (no alpha), PBOC (scrape-hostile).
