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
- [SHIPPED v1.0.222] G0b Compass + nav craft — `NavigationControl({showCompass:false})`
  → `{showCompass:true, showZoom:true, visualizePitch:true}` (datamap.tsx:452);
  dragRotate/pitchWithRotate were already on by default (never disabled), so the
  control now surfaces real bearing/tilt state that was already interactive but
  invisible. `.maplibregl-ctrl-compass` hairline added in index.css — see
  experiments.md 2026-07-08 [PRODUCT] for the full trace.
- G0c Deep-zoom policy — raise/rationalize maxZoom; keep Esri imagery to its
  native z, plan the hand-off zoom (~z16) to Google 3D Tiles (Phase G3).

### Phase G1 — style presets (Natural / Night / Terrain / Minimal)
Switcher in the top-left control column (next to globe/fullscreen). Preset =
imagery/DEM/label layer set toggled on the ONE MapLibre globe (mutate layer
visibility / `setStyle` while preserving data layers — insertion point is the
inline style object). Night = VIIRS Black Marble base (ties Phase G2). Real-first
identity; NO tactical filters as defaults (at most one optional tasteful extra).

### Phase G2 — NASA GIBS layers (biggest free-data unlock)
Access pattern (verified vs live GetCapabilities 2026-07-08, EPSG:3857, no key,
public domain, attribution "imagery via NASA GIBS/ESDIS"):
`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{Layer}/default/{Time}/{TileMatrixSet}/{z}/{y}/{x}.{ext}`
`{Time}`=YYYY-MM-DD (daily) / ISO (sub-daily) / `default`; `{TileMatrixSet}`=
`GoogleMapsCompatible_LevelN`; ext jpg (imagery) / png (overlay). Daily layers
default to T-1/T-2 and a missing date returns a SILENT transparent tile → build a
"step back one day" fallback and default the time-slider to yesterday. Ship a
shared GIBS raster-layer factory + a time-scrubber, then add layers by value:
- G2a Night lights `VIIRS_SNPP_DayNightBand_At_Sensor_Radiance` (+NOAA20; clean
  `VIIRS_Black_Marble`). Hypothesis: metro/industrial radiance MoM/YoY delta =
  regional economic activity → regional-bank/retail/utility by CBSA.
- G2b Fires — raster `GOES-East_ABI_FireTemp` (10-min) for the map; the
  `*_Thermal_Anomalies_*_All` layers are MVT VECTOR (use a vector source or the
  GOES raster). Hypothesis: fires within N km of insured/industrial/utility
  assets → P&C insurers, wildfire-liability utilities, timber REITs.
- G2c Aerosols/dust `MODIS_Combined_Value_Added_AOD`, `GOES-East_ABI_Dust`,
  `OMPS_Aerosol_Index`. Hypothesis: AOD over basin/shipping lane = output/PMI
  proxy → dry-bulk, air-freight, China-export.
- G2d Drought/soil-moisture `SMAP_L4_Analyzed_Root_Zone_Soil_Moisture`,
  `GRACE_..._Mascon_CRI`. Hypothesis: basin drought = ag-yield-down + hydro-down +
  barge-draft → ag futures, hydro utilities, inland shipping.
- G2e Vegetation `MODIS_Terra_L3_NDVI_16Day` / `VIIRS_SNPP_NDVI_8Day`; disturbance
  `OPERA_L3_DIST-ALERT-HLS_Color_Index`. Hypothesis: NDVI anomaly over crop belt =
  yield → CBOT corn/soy/wheat; disturbance clusters → palm/pulp/miner concessions.
- G2f Floods `OPERA_L3_Dynamic_Surface_Water_Extent-HLS`, `MODIS_Combined_Flood_3-Day`.
  Hypothesis: flood over industrial parks/ports/farmland → insurers, auto/semi
  plants, acreage loss.
- G2g NO2 throughput `TROPOMI_L2_Nitrogen_Dioxide_Tropospheric_Column`; US hourly
  `TEMPO_L3_NO2_Vertical_Column_Troposphere` (genuinely differentiated). SO2
  `OMPS_SO2_Planetary_Boundary_Layer` (point-source smelter/refinery on/off).
  Hypothesis: NO2 over industrial zones/ports = real-time throughput nowcast (leads
  PMI/IP); SO2 over a named smelter = operating rate → operator + its commodity.
- G2h Sea ice `AMSRU2_Sea_Ice_Concentration_12km`; snowpack `MODIS_Terra_NDSI_Snow_Cover`
  + SWE; chlorophyll `MODIS_Aqua_L2_Chlorophyll_A`; biomass `GEDI_..._Biomass..._Mean`
  (STATIC, no slider). Hypotheses per inventory (routing/hydro/seafood/carbon).
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
- 2026-07-08: G0b (compass + pitch indicator) SHIPPED v1.0.222. NEXT: G1 style
  presets (Natural/Night/Terrain/Minimal switcher), then G2a night-lights.
