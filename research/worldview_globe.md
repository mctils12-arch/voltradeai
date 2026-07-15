# WORLDVIEW-CLASS DATA GLOBE — build charter & order

Human master directive, 2026-07-08. This is the multi-session build order for the
worldview-class data globe. It governs WHAT and in WHAT ORDER; CLAUDE.md governs
HOW everything ships (tests, promotion ladder, honesty, one-change-per-PR).

## THE THESIS — use, don't compete

The unoccupied territory is the intersection of four things nobody has together:
real integrated proprietary data + Google-Earth-quality visualization + premium
UI + backend inference. Google Earth (beautiful basemap, no live proprietary
intelligence), NASA Worldview (incredible free data, dated UI), the reference
demo (great UI, fake data), Bloomberg (data, ugly) are INPUTS, not rivals. We
stand on their free/licensed data and win on integration, UI, and inference.

CORE PRINCIPLE: our data stays REAL (live SGP4 satellites, live vessels/aircraft,
honest counts + freshness). We copy the reference's INTERFACE craft and rendering
performance only — never its fake data, never its fake-military/porthole look.
The globe is a real full sphere; no CRT bezel, no FLIR/NVG/targeting cosplay.

HONESTY (non-negotiable, applies to every item below): real data or an honest
"unavailable"; provenance + freshness + confidence on every number; no fabricated
models or completeness; licensing respected (NASA public domain, Google via our
key with mandatory attribution, OSM/NE attribution). Free substitutes are labeled
estimates, never passed as ground truth.

## PRIORITY GATE — RELIABILITY FIRST

The layer-reliability audit (feeds that stall / don't follow viewport / satellite
retry) ranks AHEAD of new cosmetic work. Reliable real data in a clean skin beats
a pretty fake. Reliability status (2026-07-08):
- BUG 1 single-shot resilience — SHIPPED (#372).
- BUG 4 listener-leak (6 layers) — SHIPPED (#374).
- Satellites toggle → cache GP elements — SHIPPED (#375).
- Aircraft/vessel caps + honest coverage — SHIPPED (#376).
- Fires viewport-bound + honest cap (BUG 2/3) — SHIPPED (#377, concurrent session).
Remaining reliability is opportunistic; the visual build below is now UNGATED.

## THE SIX PILLARS (from the directive)

1. BASEMAP — Google Photorealistic 3D Tiles (our key) for real 3D cities +
   terrain; 2D map / 3D globe / deep-zoom-to-street; real elevation relief;
   Google/NE/OSM admin boundaries; compass + clean nav; NO porthole frame.
2. NASA GIBS/Worldview integration — the free proprietary-grade library, each a
   toggleable dated layer with source + freshness + time-slider + a trading
   hypothesis.
3. UNIFIED OBJECT INTERACTION — same UX for sat/plane/ship/train: click → detail
   panel (type, id, altitude/height, speed, heading, operator, status); toggles:
   isolate/zoom, show path/orbit, show HISTORICAL track from OUR archive.
4. SATELLITE FIDELITY (honest tiers) — orbital path + isolate; 4D Gaussian
   splats ONLY where real ground imagery exists (constellations once, instanced;
   named sats); unphotographed objects stay honest markers; coverage/footprint
   tools.
5. UI — Google-Earth-warm + reference control craft, no military cosplay: clean
   left DATA LAYERS panel (icon, name, live count, source, freshness, ON/OFF
   pill — also the reliability surface); geographic style presets
   (Natural/Night/Terrain/Minimal); clickable callouts; compass; premium/warm;
   mobile-first 390px; our design system + premium bar.
6. BACKEND INFERENCE — the actual product. Every stream feeds the platform's
   analysis/signals, cross-referenced via the entity graph; each correlation
   filed as a hypothesis through the validation ladder. The globe SHOWS it; the
   backend TRADES/REASONS on it.

## BUILD ORDER (each = its own verified PR; serial merges; T-CLIENT unless noted)

### Phase G0 — foundations (started)
- [SHIPPED #379] G0a Real 3D terrain relief — `map.setTerrain({source:"terrain-dem",
  exaggeration:1.3})` on the existing Mapterhorn terrarium DEM (confirmed valid by
  research; globe+terrain compatible in MapLibre v5). Degrade-safe.
- [SHIPPED #382] G0b Compass + nav craft — `NavigationControl({showCompass:true,
  showZoom:true, visualizePitch:true})`, themed `.maplibregl-ctrl-group`/
  `.maplibregl-ctrl-compass` chrome in index.css. Zero cost. datamap.tsx:~502.
  (COLLISION NOTE: a concurrent session shipped this same item as #383, unaware
  #382 had already merged it — closed as a pure duplicate per the OPS GOTCHAS
  double-build rule; no code survived from #383, only its docs-correction
  landed via this entry. Restating the OPS GOTCHAS lesson: CLAIM a roadmap item
  in your first commit before building it, not after.)
- G0c Deep-zoom policy — raise/rationalize maxZoom; keep Esri imagery to its
  native z, plan the hand-off zoom (~z16) to Google 3D Tiles (Phase G3).

### Phase G1 — style presets (Natural / Night / Terrain / Minimal)
[SHIPPED #382] Switcher in the top-left control column. Preset = imagery/DEM/
label layer set toggled on the ONE MapLibre globe (`mapPreset` state +
`vt-preset-switch` bottom-center segmented control); Night = VIIRS Black Marble
base (static, `blackmarble` source/layer — a base-look swap, distinct from G2a's
dated/toggleable Night lights DATA layer below, which uses its own source and
carries a real date). Real-first identity; no tactical filters.

### Phase G2 — NASA GIBS layers (biggest free-data unlock)
Access pattern (verified vs live GetCapabilities 2026-07-08, EPSG:3857, no key,
public domain, attribution "imagery via NASA GIBS/ESDIS"):
`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{Layer}/default/{Time}/{TileMatrixSet}/{z}/{y}/{x}.{ext}`
`{Time}`=YYYY-MM-DD (daily) / ISO (sub-daily) / `default`; `{TileMatrixSet}`=
`GoogleMapsCompatible_LevelN`; ext jpg (imagery) / png (overlay). Daily layers
default to T-1/T-2 and a missing date returns a SILENT transparent tile → build a
"step back one day" fallback and default the time-slider to yesterday. Ship a
shared GIBS raster-layer factory + a time-scrubber, then add layers by value:
- [SHIPPED v1.0.224] G2a Night lights `VIIRS_SNPP_DayNightBand_At_Sensor_Radiance`.
  Factory: `client/src/lib/gibs.ts` (`gibsTileUrl`/`gibsDefaultDate`/`gibsStepDate`/
  `gibsIsLatestAvailable`, 5 tests) — reusable by G2b-h. Access re-verified LIVE
  2026-07-08 (not just trusted from the charter's earlier research): Level8 is
  the correct TileMatrixSet for this specific layer, Level9 explicitly rejected
  by GIBS with a WMTS `InvalidParameterValue` exception (not a network fluke);
  fetched tile pixel-checked (70% non-transparent, real data, not a blank tile).
  Registry entry `nightlights` (RAW, field:true, group "environmental");
  per-row date scrubber (prev/next day, "next" disabled once at the honest
  "yesterday" ceiling — GIBS never has same-day data). Hypothesis (Pillar 6)
  filed in open_questions.md: metro/industrial radiance MoM/YoY delta =
  regional economic activity → regional-bank/retail/utility by CBSA — gate 2
  blocked on (a) a daily-radiance ARCHIVE (this PR is display-only, no
  pipeline yet — the actual gate-2 prerequisite) and (b) a metro→ticker join;
  NOT yet a signal. NEXT (G2b): fires (GOES-East_ABI_FireTemp).
- [SHIPPED v1.0.264] G2b Fires — raster `GOES-East_ABI_FireTemp` (PNG,
  `GoogleMapsCompatible_Level7`; access + non-blank field verified live
  2026-07-10: z=0 tile 99% non-transparent, a continuous full-disk brightness
  field, not just discrete hotspots). Genuinely different cadence class from
  every G2 layer before it: ~10-min, IRREGULAR scan gaps (the GetCapabilities
  Time dimension lists non-uniform intervals, not a fixed schedule) — no
  day-granularity scrubber applies. Uses the WMTS REST spec's literal
  "default" time token (gibsTileUrl already accepted this — no factory
  change needed) to always request GIBS's own freshest scan; what "default"
  resolved to is read back honestly via the `layer-time-actual` response
  header (CORS-exposed, confirmed live) through a new `gibsLatestScanTime`
  helper — a HEAD request against the always-in-domain z=0/y=0/x=0 tile,
  refreshed every 5 min alongside the tile source itself (GIBS marks every
  response `no-store` — verified never safe to cache). Registry `firetemp`
  (RAW, field:true, environmental); ThermometerSun icon; own legend chip;
  freshness note ("scan: {time} UTC" or an honest "scan time unknown"
  instead of a scrubber). Deliberately positioned as a COMPLEMENT to the
  existing NASA FIRMS point-detection `fires` layer, not a replacement or a
  new hypothesis: FIRMS gives discrete confirmed detections (~3h latency),
  this gives a continuous heat-intensity field (~10-min) — the fires ×
  facilities cross-tie already filed against FIRMS (#388) is unchanged.
  `*_Thermal_Anomalies_*_All` layers are MVT VECTOR (not used here — this is
  the GOES raster branch of the original charter note). NEXT (still open,
  same shape as TEMPO NO2 in G2g): a proper sub-daily/ISO-timestamp scrubber
  for browsing PAST scans, not just "latest" — bigger lift, deferred.
- [SHIPPED v1.0.228] G2c Aerosol optical depth `MODIS_Combined_Value_Added_AOD`
  (PNG, `GoogleMapsCompatible_Level6`, daily — access + TileMatrixSet re-verified
  LIVE against GetCapabilities 2026-07-08, a real yesterday tile pixel-checked
  non-blank at 21% coverage, NOT trusted from the charter). Registry `aerosol`
  (RAW, field:true, group "environmental"); reuses the `gibs.ts` factory + the
  night-lights dated-daily pattern (own date scrubber, opacity slider, legend,
  CloudFog icon). Hypothesis (Pillar 6): AOD anomaly over a NAMED industrial
  basin/shipping lane = output/throughput proxy → single-basin operators +
  commodity; filed in open_questions.md with prior (weak/heavily meteorology-and-
  dust-confounded → residual is anomaly-vs-baseline, not level), ladder path
  (gate-2 blocked on a daily-AOD archive over facility/lane polygons + a
  de-confounding baseline), and the AOD × facility-archive cross-tie to #388.
  Still-open dust/aerosol-index variants (`GOES-East_ABI_Dust`,
  `OMPS_Aerosol_Index`) remain future adds.
- [SHIPPED v1.0.230] G2d Root-zone soil moisture `SMAP_L4_Analyzed_Root_Zone_Soil_Moisture`
  (PNG, `GoogleMapsCompatible_Level6`; ~6-day processing lag — access + non-blank
  land coverage re-verified live 2026-07-08: 07-01/07-02 carry data, 07-07 does
  NOT, so the scrubber defaults 7 days back via the new `gibsDefaultDate(now,
  latencyDays)` factory param rather than to a guaranteed-blank "yesterday").
  Registry `soilmoisture` (RAW, field:true, environmental); Droplets icon.
  COMPLETES the ag-supply-chain cross-tie triad: NDVI (crop health) × soil
  moisture (water in the root zone) × river gauges (barge draft on the corridor
  that ships the grain) — three RAW observations over one geography; a future
  gate-2 ag signal would be built on that stack. Hypothesis filed
  open_questions.md. Still-open: `GRACE_..._Mascon_CRI` groundwater (NOT in the
  EPSG:3857 GetCapabilities as of 2026-07-08 — needs a different endpoint/CRS).
- [SHIPPED v1.0.229] G2e Vegetation `VIIRS_SNPP_NDVI_8Day` (PNG,
  `GoogleMapsCompatible_Level8`, 8-day composite — access + non-blank LAND
  coverage re-verified live 2026-07-08: a yesterday request returns the current
  composite, US/N.America tile 41% coverage, ocean legitimately transparent).
  Registry `vegetation` (RAW, field:true, environmental); same gibs.ts factory +
  dated-scrubber mirror (Leaf icon). Hypothesis (Pillar 6): NDVI anomaly over a
  named crop belt = yield proxy → CBOT corn/soy/wheat; filed open_questions.md
  with prior (seasonal/weather-confounded → residual is anomaly-vs-crop-calendar),
  ladder path (gate-2 blocked on an NDVI archive over crop-belt polygons + a
  planting-calendar baseline). Still-open: `MODIS_Terra_L3_NDVI_16Day` variant
  and disturbance `OPERA_L3_DIST-ALERT-HLS_Color_Index` (palm/pulp/miner
  concessions) as future adds.
- G2f Floods `OPERA_L3_Dynamic_Surface_Water_Extent-HLS`, `MODIS_Combined_Flood_3-Day`.
  Hypothesis: flood over industrial parks/ports/farmland → insurers, auto/semi
  plants, acreage loss.
- [SHIPPED v1.0.231] G2g NO2 throughput `TROPOMI_L2_Nitrogen_Dioxide_Tropospheric_Column`
  (PNG, `GoogleMapsCompatible_Level6`, DAILY — access + continuous non-blank field
  verified live 2026-07-08: yesterday tile 100% over N.America; ocean legitimately
  LOW/dark, industrial zones HIGH). Registry `no2` (RAW, field:true, environmental);
  Factory icon; same daily-scrubber mirror. Hypothesis (Pillar 6): NO2 over an
  industrial zone/port = real-time throughput nowcast leading PMI/IP; filed
  open_questions.md. STILL OPEN (needs the sub-daily/ISO-timestamp factory work
  the daily scrubber can't do yet): US hourly `TEMPO_L3_NO2_Vertical_Column_Troposphere`
  (geostationary, irregular per-scan ISO timestamps — genuinely differentiated but
  needs a scan-time picker, not a day scrubber) and `OMPS_SO2_Planetary_Boundary_Layer`
  (point-source smelter/refinery on/off — daily, a clean future add).
- G2h Sea ice `AMSRU2_Sea_Ice_Concentration_12km`; snowpack `MODIS_Terra_NDSI_Snow_Cover`
  + SWE; chlorophyll `MODIS_Aqua_L2_Chlorophyll_A`; biomass `GEDI_..._Biomass..._Mean`
  (STATIC, no slider). Hypotheses per inventory (routing/hydro/seafood/carbon).
  [SHIPPED v1.0.318] biomass slice: `GEDI_ISS_L4B_Aboveground_Biomass_Density_
  Mean_201904-202303`, GoogleMapsCompatible_Level7, requested via GIBS's
  "default" token (no hardcoded date — survives a future mission-life
  reprocessing rollover). Genuinely static (single 2019-04–2023-03 composite,
  confirmed via live GetCapabilities: one `Value` entry, not a daily
  sequence) — no scrubber, unlike every other G2 layer. Access + real field
  verified live: Amazon basin 95% non-transparent, US Pacific-NW 99%,
  open ocean ~0.04% (legitimately blank, GEDI ±51.6° land-only). Hypothesis
  filed in open_questions.md: standing-carbon CONTEXT/baseline, not a
  standalone signal (a static composite can't show change) — gate 2 blocked
  on a future repeat/reprocessed GEDI product for change detection. STILL
  OPEN: sea ice, snowpack, chlorophyll (all three are actually daily
  products with real dates in GIBS's own capabilities, unlike biomass —
  a future session adding them should decide fresh whether they get the
  full G2a-style scrubber or a simpler "always latest" pattern like
  firetemp, not assume "static" applies to them the way it genuinely does
  to biomass).
Every GIBS SIGNAL claim above is a HYPOTHESIS → filed in open_questions.md with a
ladder path; the raster overlay ships as RAW (labeled, dated), the interpreted
signal stays gate-2-locked until validated. The backend consumes the same layers
via the datacore boundary for cross-stream inference (Pillar 6).

### Phase G3 — Google Photorealistic 3D Tiles (deep zoom, COST-GATED)
Renderer: deck.gl `Tile3DLayer` via `MapboxOverlay({interleaved:true})` — shares
MapLibre's WebGL context + camera, officially supports the v5 globe, lazy-loaded
(~300-500KB) only in 3D mode. Auth: API KEY only (no session token; Google embeds
`session=` in child-tile URIs). BILLING: each `root.json` request = one billable
session (1,000/mo free, then $6/1,000) buying ≥3h of child fetches → runaway =
page loads/remounts issuing fresh root requests. COST GUARDS (mirror
scripts/runpod_budget.py, MANDATORY before wiring): (a) feature flag + SERVER-PROXY
the key (never client-embed GOOGLE_MAPS_API_KEY); (b) issue root only below ~z16 on
user interaction, never on load; (c) append-only daily root-request ledger + hard
cap in the proxy (~33/day ≈ free tier) + debounce React remounts; (d) GCP $20/mo
budget tripwire. Attribution: Google logo + live-aggregated glTF `asset.copyright`
credits, de-duped, freq-sorted, bottom of map — compliance requirement. Terrain
(G0a) and Google mesh are mutually exclusive per view: terrain mid/far, Google mesh
deep-zoom.

### Phase G4 — unified object interaction (data-driven, all four types)
One interaction layer over sat/plane/ship/train: click → unified detail panel
(type, identifier, altitude/height, speed, heading, operator, status — whatever the
object HAS, honest nulls otherwise). Per-object toggles: ISOLATE (hide others,
focus+zoom this one), SHOW PATH/ORBIT, SHOW HISTORICAL TRACK from our archive
(/api/data/track/:kind/:id already serves it — render as a path on the globe; if no
history, say so). Consistent across all four. Satellites add: orbital path +
isolate + coverage/footprint (geometry.ts already computes footprint/next-pass).

### Phase G5 — satellite fidelity (honest tiers, per orbital_models.md)
4D Gaussian splats ONLY where real ground imagery exists (Starlink/constellation
modeled once per design from Earth photos, instanced; ISS/named). Unphotographed
objects/debris stay honest markers — never fabricate. GPU-cost-capped per RunPod
gate; splatting was CANCELLED $0 in orbital_models.md (0 candidates) — revisit only
if real imagery appears. glTF for named sats.

### Phase G6 — admin boundaries + polish
Boundaries: KEEP Natural Earth 110m (level-0, in repo) + add NE 10m admin-1
(states/provinces, public domain) + OSM `admin_level` (ODbL) for deep-zoom detail.
NOT Google boundaries (Data-Driven Styling needs Google's own renderer + Map ID,
billed per map load, incompatible with MapLibre + against build-your-own doctrine).
Plus periodic premium-experience polish passes (DESIGN.md), mobile-390 audits.

## PILLAR 6 — BACKEND INFERENCE (runs alongside G2+)
Every integrated stream is not just a globe layer — it feeds analysis via the
entity graph. Cross-refs to file as hypotheses (open_questions.md, ladder path):
fires × facility locations; night-lights delta × regional-economic tickers; drought
× ag commodities; flood × insurer/industrial exposure; NO2/SO2 × industrial
throughput/operator; vessel/aircraft dwell × company. The globe is the showcase;
the value is the validated signal feeding the trading platform and the /data
product surface. NEVER surface a signal the ladder hasn't validated.

## COORDINATION
Concurrent sessions run under the WORKSTREAM PARTITION. This program is primarily
T-CLIENT (globe/datamap.tsx/index.css/style presets) + shared datacore/layers.json
(GIBS registry entries) + a server proxy for Google 3D Tiles + backend inference in
datacore/T-BOT. Serial-merge shared files (layers.json, package.json, routes.ts);
version = read-and-increment at commit time; rebase on collision.

## STATUS LOG
- 2026-07-08: charter filed. G0a (3D terrain) SHIPPED #379. Reliability gate cleared
  (#372/#374/#375/#376/#377). Research verified: GIBS access + 15-layer inventory
  (live GetCapabilities), Google 3D Tiles path (deck.gl, key-only auth, cost model).
  NEXT: G0b compass, then G1 style presets, then G2a night-lights (first GIBS layer).
- 2026-07-08: G0b + G1 SHIPPED together via #382 (a concurrent session, unclaimed —
  see the G0b collision note above; #383's identical G0b attempt closed as a
  duplicate, no code delta survived).
- 2026-07-08: G2a (GIBS factory + time-scrubber + night-lights layer) SHIPPED
  v1.0.224. NEXT: G2b fires (GOES-East_ABI_FireTemp), or G0c deep-zoom policy —
  either fits; pick per the next session's own judgment.
- 2026-07-10 [PRODUCT session]: G2c/G2d/G2e/G2g shipped by concurrent sessions
  between 07-08 and 07-10 (see their own [SHIPPED] markers above — not this
  session's work, noted here only so this log stays a true timeline). This
  session claimed the still-open G2b (fires) — SHIPPED v1.0.264, full detail
  in the G2b bullet above; new client-side helper `gibsLatestScanTime`
  (client/src/lib/gibs.ts) is reusable by any future sub-daily/irregular GIBS
  layer (e.g. TEMPO NO2's still-open scrubber problem in G2g). NEXT: G0c
  deep-zoom policy, G2f floods, G2h (sea ice/snowpack/chlorophyll/biomass —
  static, no slider), or Phase G4 unified object interaction — pick per the
  next session's own judgment; none block on each other.
