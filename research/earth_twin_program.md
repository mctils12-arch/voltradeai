# EARTH TWIN PROGRAM — the "Infinite 4D Earth Digital Twin" build charter

INSTALLED 2026-07-15 by human directive (verbatim intent preserved below).
Multi-session program like GRID VISION / ORBITAL / SCALE / WORLDVIEW;
RESUME STATE at the bottom is authoritative. CLAUDE.md governs HOW
everything ships (tests, one-change-per-PR, honesty, promotion ladder);
this charter names WHAT the twin builds toward and the honest engineering
path for every requested capability — including the ones that are
impossible as literally stated, each of which gets its best-possible
solution instead of removal (the directive's own rule).

## THE HUMAN DIRECTIVE (compressed, no intent lost)

Build a continuously-zoomable, real-time, multi-layer digital twin of
Earth: seamless space→planet→city→street transitions with no loading
screens; TIME as the fourth dimension (scrub history, watch live);
terrain that is true 3D (cliffs, canyons, bathymetry); an ocean whose
WATER layer can be toggled off to reveal the mapped seafloor, with
uncertainty displayed where mapping is partial; underground
infrastructure; volumetric atmosphere; aircraft at their TRUE altitude
(never intersecting mountains) that resolve from icon → silhouette →
accurate model as you approach; satellites that appear at the right
zoom, resolve to identifiable spacecraft ("this is a CubeSat, this is
X"), and naturally disappear as you zoom into the surface; celestial
context (sun/moon/planets, launches, debris); buildings/roads/vegetation
that gain detail by distance; everything modular and toggleable; every
layer a plug-in so new APIs need configuration, not new rendering code;
every dataset carrying source/timestamp/confidence/resolution/license;
inferred data ALWAYS visually distinct from verified data; graceful
degradation — the engine must never fail because data is unavailable.
Where a feature is physically impossible with current public data, do
not remove it: design an approximation hierarchy (1 verified measured →
2 public datasets → 3 procedural reconstruction → 4 AI inference with
confidence labels → 5 placeholder until better data exists).

## RECONCILIATION WITH THE CONSTITUTION (read this before building)

The directive's approximation hierarchy maps onto our standing honesty
rules almost one-to-one, with ONE deliberate tightening: tiers 3–4
(procedural/AI reconstruction) are permitted ONLY where the
reconstruction is (a) visually unmistakable as inferred, (b) carries a
stated confidence and method, and (c) does not manufacture specific
factual claims (a procedurally-textured generic building mass is
acceptable and labeled; an invented room interior presented as a scan
is fraud with good typography and is banned). Every layer this program
ships is classified RAW or SIGNAL per the standing /data rule, and
every inferred/estimated product is labeled per BUILD-FIRST honesty.
"Never fail because data is unavailable" is already codified in
client/src/lib/resilientLoad.ts — this program extends that standard,
never weakens it.

## WHAT ALREADY EXISTS (do NOT rebuild — extend; verified by survey 2026-07-15)

The twin is ~40% built. Sessions MUST read this inventory before
claiming a slice; the biggest failure mode of this program is a session
rebuilding something that ships already.

- GLOBE: MapLibre GL v5 native globe projection, DEFAULT-ON
  (datamap.tsx readGlobePref), runtime globe↔flat toggle, style presets
  (Natural/Night/Terrain/Minimal), compass/pitch nav. [WORLDVIEW G0/G1]
- 3D TERRAIN: real mesh terrain — raster-dem `terrain-dem` (Mapterhorn
  terrarium: Copernicus GLO-30 + national DEMs), setTerrain exaggeration
  1.3 + hillshade, opt-in via the `terrain` registry layer. LAND ONLY —
  oceans are flat zero. [WORLDVIEW G0a #379]
- TIME-DIMENSIONED IMAGERY: NASA GIBS factory (client/src/lib/gibs.ts)
  with per-layer date scrubbers, honest latency defaults, blank-tile
  guards, sub-daily scan-time readback: nightlights, aerosol, NDVI,
  soil moisture, NO2, GOES fire temp. [WORLDVIEW G2a–g]
- TIME SCRUBBER (W3): /api/data/snapshot archive replay (hour/day
  buckets) painted as amber "historical replay — not live" points.
  Per-GIBS-layer day scrubbers are separate. NOT yet one unified axis.
- SATELLITES: the full CelesTrak active population (~16k) client-fetched
  (sidesteps the R17 Railway firewall), SGP4 in a Web Worker, GPU
  instanced points with TRUE 3D altitude (LEO/MEO/GEO shells visually
  distinct via projectTileFor3D), far-side occlusion cull, click-to-
  identify (name/NORAD/orbit class/altitude/period/epoch age), SATCAT
  operator→ticker entity join, Starlink coverage query. Deep-space
  (MEO/GEO, ~800 objects) honestly SKIPPED by the near-earth-only inline
  SGP4 kernel — flagged, never faked. [ORBITAL O1–O3, O7, far-side cull]
- LIVE MOVERS: aircraft (adsb.lol, WebGL symbols, heading-rotated,
  altitude→COLOR tint only — they render ON the surface today), vessels
  (key-gated), trains; permanent JSONL position archives + historical
  track API (/api/data/track/:kind/:id) + W4 cross-layer query engine.
- REGISTRY: datacore/layers.json, 113 layers, 8-field schema (id, name,
  kind RAW|SIGNAL, status, group, costTier, source, description). Each
  layer today is HAND-WIRED as its own useEffect in datamap.tsx — there
  is no generic data-driven layer engine and no per-layer LOD/time
  metadata. Layer panel has groups, opacity sliders, cost badge.
- VIEWPORT SERVING (SCALE S1): server/viewport.ts applyViewport (bbox,
  antimeridian-aware, states count_dropped_offscreen) applied to
  aircraft + vessels only. S2 server tiling and S3 spatial index NOT
  built.
- GOOGLE PHOTOREALISTIC 3D TILES: server proxy + billing-cap ledger
  built (tiles3dProxy.ts/tiles3dBudget.ts, /api/data/3dtiles/*,
  key-gated); ZERO client wiring. [WORLDVIEW G3, cost-gated]
- CONTEXT SYSTEMS: entity graph + location dossier (click → everything
  known near a point), analyst console (W6), units system
  (client/src/lib/units.ts — MANDATORY for every new display), SYMBOLS
  NOT DOTS icon registry, perf/visual harness at 390/768/1440,
  resilientLoad degrade-safety.
- DEPS REALITY: maplibre-gl ^5.24, pmtiles, topojson. NO three.js, NO
  deck.gl, NO glTF loader, NO satellite.js. The only 3D renderer today
  is MapLibre itself (globe + terrain + custom WebGL layers) — the
  orbital stack proved that path carries instanced 3D fine.

## THE ARCHITECTURE SPINE — four cross-cutting systems (build these first)

Everything the directive asks for hangs off four generic systems. Build
the spine once; every vertical then becomes configuration + data.

### A1 — LOD DIRECTOR (the "4D camera" — zoom-gated existence + fidelity)

One central module (client/src/lib/lod.ts, pure + tested) that converts
the camera state (zoom, pitch, center lat → derived CAMERA ALTITUDE in
km) into a per-layer visibility/fidelity decision, replacing today's
scattered per-style-layer minzoom values.

- Registry v2 (A2) gives every layer an optional `lod` block:
  `{camMin, camMax, fadeBand, ramp: ["icon","silhouette","model"]}` —
  camera-altitude envelope in km, soft fade at the edges (opacity ramp,
  never a pop), and a named fidelity ramp.
- THE SATELLITE CASE (the directive's own example, verbatim honored):
  orbital_sats gets an envelope like camMin≈50km — from space the whole
  population is visible; as the camera descends toward the street the
  layer fades out entirely and costs zero (worker pausable). Zooming
  back out, it returns. Exact numbers are TUNED IN THE PR with
  screenshots, not guessed here.
- THE AIRCRAFT CASE: far = class icon (today's), mid = oriented 3D
  silhouette, near = design-class model (E4). Ramp thresholds by camera
  altitude, not raw zoom, so pitch/tilt behaves.
- Fidelity is REVEALED not dropped (SCALE reframe): the ramp only ever
  changes representation, never which objects exist; counts stay honest.
- PERF: the director is also the budget governor — it knows every
  active layer's declared cost and the current envelope, so the "heavy
  load" badge becomes viewport-real (SCALE supporting move, delivered
  here).

### A2 — REGISTRY v2 (the plug-in framework: "configuration, not new rendering code")

Extend datacore/layers.json's schema (additive, backward-compatible —
all 113 existing entries stay valid) with the directive's dataset
contract:

- `altitudeRef`: surface | agl | msl | orbit | depth | underground —
  what z means for this layer.
- `time`: none | live | dated-daily | dated-subdaily | archive —
  what the global time axis (A3) can do with it, plus latencyDays.
- `lod`: the A1 block.
- `provenance`: source (exists), license, updateFreq, resolution,
  coverage (e.g. "US-only", "15% of seafloor direct-measured"),
  confidence: verified | derived | estimated | inferred | placeholder.
- `renderKind`: point-symbol | raster-field | vector | track | grid |
  custom — which GENERIC ENGINE draws it.
- GENERIC ENGINES: implement point-symbol, raster-field, vector-tile,
  and track engines ONCE (most of the 113 layers are one of these
  four; the code already exists in the per-layer useEffects — this is
  extraction, not invention). A new API source whose shape matches an
  engine then ships as a registry entry + a server adapter, ZERO new
  client rendering code — the directive's plug-in requirement, made
  real. Custom layers (orbital, wind particles) keep bespoke code.
- MIGRATION IS OPPORTUNISTIC: never a big-bang rewrite of 113 layers
  (churn risk, attribution death). The engines ship with 2–3 exemplar
  migrations; every layer TOUCHED afterward migrates in that PR; new
  layers must use engines.
- INFERRED-VS-VERIFIED VISUAL LANGUAGE (directive requirement,
  system-wide): one shared style vocabulary — solid = verified/measured,
  dashed/hatched + a labeled chip = estimated/inferred, dimmed +
  "no public data" = placeholder. Rendered from the registry
  `confidence` field so it can never drift per-layer. DESIGN.md gets
  the vocabulary; the legend renders it from the same registry (single
  source of truth, like iconDataURL).

### A3 — GLOBAL TIME AXIS (the fourth dimension, unified)

One time controller replacing today's three disjoint time UIs (W3
snapshot scrubber, per-GIBS day scrubbers, implicit "live").

- One scrubber: LIVE ←→ history. Each layer declares its `time`
  capability (A2); the controller drives every time-capable layer to
  the selected instant — GIBS layers to the nearest available date
  (honest snap, latency-aware), archive layers (aircraft/vessels/
  trains/orbital elements) to the archived bucket, live-only layers
  fade out with a "no history for this layer" chip (never silently
  freeze at stale data).
- PLAYBACK: step/play controls over a chosen window (the W3 machinery
  already reads buckets; this generalizes it).
- FUTURE is allowed ONLY where a real forecast model exists, labeled
  as forecast: NOAA GFS/NWS forecast hours (their model, attributed),
  NHC hurricane cones, tide predictions (harmonic, NOAA), satellite
  positions (SGP4 propagation IS a forecast — epoch-age caveats
  already shipped). The scrubber's future zone is visually distinct.
  NEVER an AI-invented future. Examples the directive lists (weather
  evolution, wildfires, hurricanes, night lights, construction/
  deforestation change) all resolve to real dated archives we already
  render or list below.
- HISTORY DEPTH IS HONEST: our position archives start 2026-06/07 —
  the scrubber states its floor per layer instead of pretending depth
  we don't have. GIBS goes back years (their archive; we pass dates
  through). Time turns our free feeds into the paid product
  (BUILD-FIRST precedent) — the axis makes the compounding visible.

### A4 — STREAMING & PERFORMANCE (already chartered — this program CONSUMES it)

The directive's streaming section (quadtrees, chunk streaming, culling,
predictive caching, no loading screens) is the SCALE program. This
charter adds no second streaming architecture — it adds REQUIREMENTS
into SCALE's queue: S1 bbox serving swept across all point layers, S2
server-side tiling/clustering, S3 spatial index (SQLite R*Tree on the
archives) now also serve the time axis (A3 queries are bbox+time).
Perf bar per the existing harness: layer-heavy /data stays smooth at
390/768/1440; every twin slice ships a harness perf assertion. 60fps
on capable desktops, no-jank mobile; hardware ray tracing is NOT a
target (browser reality; WebGPU is a future note, not a dependency).
GPU instancing, workers, transferable buffers: the orbital stack is
the in-repo pattern to copy.

## THE VERTICAL WORLDS — domain by domain, with named free data

Each vertical lists: what ships, the data (all free unless flagged),
and the honest ceiling. Every layer: RAW/SIGNAL classification,
registry v2 contract, units.ts formatting, SYMBOLS NOT DOTS.

### V1 — ORBIT (mostly built; finish identity + LOD)

- ZOOM ENVELOPE (A1): satellites appear from space, fade below ~street
  zoom — the directive's exact ask; mostly configuration once A1 lands.
- IDENTITY BEFORE MODELS ("this is a CubeSat, this is X"): SATCAT
  already gives object type (payload/rocket body/debris), RCS size
  class, operator, country, launch date, orbit class. The click card
  says "CubeSat-class payload, operated by X" TODAY with zero new data
  — ship the identity language first; models are polish on top.
- O5 DESIGN-CLASS glTF MODELS (ORBITAL charter, unchanged): one real
  model per design/generation from published GROUND imagery
  (Starlink/ISS/GPS-block), GPU-instanced, resolved on focus/zoom;
  unphotographed objects stay honest symbolic markers. Requires the
  program's ONE new dependency decision: a minimal glTF loader (spike:
  tiny custom parser vs lazy-loaded three.js used ONLY in focus view —
  same lazy pattern as TimeScrubber). Splats stay cancelled per
  orbital_models.md ($0, no candidates).
- SDP4 UPGRADE (unblocks the ~800 deep-space skips + GPS/GEO honesty +
  GPS-DOP tool): worker-only satellite.js exactly as propagate.ts's own
  header prescribes. One slice, big honesty win.
- SPACE CONTEXT (directive's Moon/Mars/planets/asteroids): see V7.
- Debris IS already in the population (SATCAT type). Launch events +
  orbit-history archive remain RELAY-GATED (R17) per ORBITAL O4 —
  restated, not re-decided.

### V2 — AIR (true altitude — the flagship gap)

- TRUE-3D AIRCRAFT: today altitude only tints color. Build
  `airLayer.ts` on the PROVEN orbital pattern (CustomLayerInterface +
  projectTileFor3D + worker interpolation): every aircraft at its real
  baro/geometric altitude (adsb.lol carries it), velocity-extrapolated
  between polls exactly like satWorker interpolates. LEO-shell
  precedent means this is porting, not research.
- TERRAIN CONSISTENCY ("never intersect mountains"): when 3D terrain is
  on, plotted altitude must respect the SAME vertical scale/exaggeration
  as the DEM (1.3), and a clamp guard keeps rendered z ≥ terrain
  surface + margin (bad baro data exists; clamping is honest if the
  clamp state is exposed on the click card). AGL vs MSL handled via
  registry altitudeRef + the DEM sample.
- LOD RAMP (A1): icon (today) → oriented 3D silhouette by class
  (heading/pitch-from-vertical-rate — DERIVED, labeled) → design-class
  glTF (share the V1 loader; airliner families from open models,
  license-checked per asset). HONEST CEILING: airline paint, moving
  control surfaces, gear animation, engine rotation are tier-3/4
  showcase — gear-by-flight-phase is INFERENCE (plausible, labeled),
  control-surface animation is decoration; both rank below every data
  slice and may only ship labeled as representative animation, never
  implied telemetry. Contrails: only as clearly-stylized trail
  rendering (we already draw velocity vectors), never fake physics.
- WAKE/ROLL/PITCH: not in ADS-B. Derived bank from turn rate is
  tier-4 labeled inference; fine as polish, never as data.

### V3 — SURFACE (terrain built; buildings/roads/vegetation tiers)

- TERRAIN: shipped (land). Directive's cliffs/canyons ceiling: GLO-30
  is 30m — real LiDAR sharpness only where USGS 3DEP covers (US, free,
  up to 1m). Slice: a 3DEP high-res DEM upgrade path for US deep zoom
  (self-tiled terrarium pmtiles — we already run pmtiles pipelines).
  Photogrammetry-grade global 3D is exactly the Google 3D Tiles path
  (G3) — cost-gated, already chartered; this program's deep-zoom
  hand-off lands there. Terrain synthesis (tier 3) is NOT built —
  30m truth beats invented detail.
- BUILDINGS: (far→near) OSM footprint fill-extrusions with real
  render_height where mapped [PUBLIC, ODbL] — spike picks the vector
  source (self-hosted pmtiles extracts vs OpenFreeMap hosted tiles;
  full-planet buildings pmtiles is tens of GB, extracts or hosted
  tiles are the realistic v1) → Google photorealistic mesh at deep
  zoom where the key + budget allow (G3 client wiring is a slice
  here) → INTERIORS: TIER 5 PLACEHOLDER, PERMANENTLY HONEST — public
  interior data does not exist at scale, and fabricating rooms is
  banned above. The card says "no public interior data." This is the
  one directive item we deliberately cap at placeholder.
- ROADS: OSM lines → lanes/markings where OSM tags them (lanes=,
  turn:lanes) → surface/potholes: no public dataset; construction
  zones via 511/DOT feeds are US-state patchwork [PUBLIC, per-state
  slice, coverage-labeled]. Traffic lights/signs exist in OSM tags
  where mapped — coverage-honest.
- VEGETATION: NDVI field is live (canopy health, dated). Per-tree:
  city street-tree inventories exist for major cities [PUBLIC CSV,
  species/height/health — a genuinely great honest layer]; global
  canopy HEIGHT raster (Meta/WRI 1m canopy height, CC-BY) as a field
  layer. Individual procedural trees with species/branches (tier 3)
  are showcase-only and rank last; seasonal state derives from NDVI
  date (derived, labeled).

### V4 — OCEAN (the directive's showcase toggle — all real, all free)

- BATHYMETRY TERRAIN: GEBCO 2025 global grid (15 arc-sec, free,
  attribution) as a SECOND raster-dem. Two-step path: v1 = swap-in
  AWS Terrain Tiles (terrarium encoding, ETOPO bathymetry baked in at
  low zooms — zero pipeline, honest coarse label) to prove the toggle;
  v2 = self-tiled GEBCO pmtiles via a scripts/ pipeline (we own it,
  BUILD-DON'T-BUY) for real 15-arcsec fidelity.
- THE WATER TOGGLE: `ocean_water` registry layer — ON: styled water
  surface (flat tint v1; animated shader v2 polish). OFF: bathymetric
  terrain + depth-palette hypsometric tint + hillshade = "drain the
  ocean." This is genuinely cheap once the DEM source exists — the
  terrain machinery is already built.
- UNCERTAINTY, EXACTLY AS DEMANDED: GEBCO publishes the TID grid (per-
  cell source type: direct sounding vs satellite-gravity
  interpolation). Ship it as the `seafloor_confidence` overlay —
  measured vs interpolated seafloor, visually distinct. The directive
  says "if only partial mapping exists, display uncertainty" — GEBCO
  literally hands us that layer; ~25% of the seafloor is
  direct-measured and we SAY SO. This is the honesty machinery as a
  hero feature.
- SEAFLOOR FEATURES: GEBCO Undersea Feature Names gazetteer (trenches,
  seamounts, ridges — point/label layer, free); NOAA AWOIS wrecks +
  obstructions [PUBLIC, US waters, coverage-labeled]; hydrothermal
  vents (InterRidge, free).
- SUBMARINE CABLES: TeleGeography's public map data is CC BY-NC-SA —
  NON-COMMERCIAL, so it trips the MONETIZATION TRIPWIRE on a billed
  product. LICENSE-GATED: do not ship it on /data without the
  compliance check passing; the build-first alternative is FCC cable
  landing license filings [US-touching cables, public domain] — file
  the fuller-coverage decision to wishlist.md if NC licensing blocks.
- MARINE PROTECTED AREAS: WDPA has NC redistribution terms →
  LICENSE-GATED same as cables; NOAA MPA inventory [US, public
  domain] is the clean v1.
- WATER DYNAMICS: tides (NOAA CO-OPS stations + harmonic predictions —
  real forecast, labeled); currents (NOAA RTOFS/OSCAR fields);
  sea state (already have buoys live); storm surge (NOAA SLOSH/NHC —
  forecast-labeled). Waves/reflections/caustics: shader polish tier,
  after data slices.
- OCEAN CONDITIONS OVER TIME: GIBS sea-ice + chlorophyll (G2h queue,
  already chartered) ride the A3 axis.

### V5 — UNDERGROUND (honest sparse — coverage labels carry this vertical)

Public subsurface data is genuinely thin; the vertical ships as real
layers with loud coverage honesty, not fabricated geology volumes.

- FAULT LINES: USGS Quaternary faults [US, public]; GEM Global Active
  Faults [global, open — verify license at build].
- AQUIFERS: USGS principal aquifers [US, public].
- MINES: MSHA mine registry [US, public]; USGS MRDS global mineral
  sites [public].
- PIPELINES: EIA/HIFLD oil/gas [US, public — HIFLD precedent already
  in-repo]; no CEII detail, same caveat as the grid layers.
- TUNNELS/SUBWAYS: OSM tunnel=yes / railway=subway [ODbL] — renders as
  a distinct sub-surface style class.
- CAVES/KARST: USGS karst map [US, public]; OSM cave entrances.
- GEOLOGIC LAYERS: USGS/state geologic map services (2D formation maps,
  public) as a raster/vector field — honest 2D "what's under here"
  rather than a fake 3D volume.
- FIBER ROUTES: NOT PUBLIC (carrier-proprietary) → tier-5 placeholder
  with "no public data"; cable LANDING points are public via V4.
- GPR: no public global layer → tier-5 placeholder, named in the panel.
- UTILITIES (water/sewer/gas distribution): municipal-patchwork; only
  where a city publishes open data — coverage-labeled per-city slices,
  never a fabricated national layer.
- ARCHAEOLOGY: known-structure gazetteers only (open heritage
  registers); never inferred structures.
- RENDER: underground layers draw as styled sub-surface classes
  (dashed/sectioned per A2's inferred-vs-verified vocabulary +
  altitudeRef=underground/depth). A camera that literally descends
  below the DEM is a showcase experiment AFTER data slices ship.

### V6 — ATMOSPHERE (volumetric feel from real fields)

- WIND FIELD: NOAA GFS (NOMADS filtered GRIB → server pipeline decodes
  to compact grid JSON, T-DATACORE) driving a GPU particle-advection
  custom layer (nullschool-style; the custom-layer muscle is proven by
  orbital). Multiple pressure levels = the directive's jet streams
  (250 hPa). Forecast hours ride A3's future zone, labeled.
- Already live: radar (US nowCOAST), OWM temp/wind rasters, wind-arrow
  grid, AOD, NO2, GOES fire temp; alerts. These join the axis, not get
  rebuilt.
- CLOUDS: GIBS corrected-reflectance + GOES full-disk animation frames
  (dated/sub-daily on A3). TRUE volumetric 3D clouds: tier-3 shader
  showcase, ranked behind every data slice; v1 volumetric FEEL comes
  from MapLibre's globe atmosphere + real sun position (V7).
- AURORA: NOAA SWPC OVATION oval [public, 30-min forecast, labeled].
- STORMS: NHC/JTWC tracks + forecast cones [public, forecast-labeled];
  smoke (NOAA HMS [public]); dust (GIBS, G2 queue).
- HUMIDITY/PRESSURE/VISIBILITY: OWM/NWS fields where keys/coverage
  allow — each a registry raster-field entry, nothing bespoke.

### V7 — CELESTIAL & LIGHTING (real ephemeris, no cosplay)

- SUN: real solar position (standard ephemeris math, a small pure lib
  or in-repo formulas — spike picks) drives the globe's day/night
  terminator, atmosphere tint, and light direction; night side blends
  the Black Marble base we already carry. Seasonal daylight and
  eclipse geometry fall out of the same math (eclipse shadow track as
  a dated event overlay). Real shadows from terrain: MapLibre
  hillshade responds to light azimuth — true dynamic shadow mapping is
  a renderer-level showcase, ranked last.
- MOON: real position/phase (same ephemeris); ties to tides (V4)
  contextually.
- PLANETS/ISS-CLASS CONTEXT: "sky now" positions of planets/Mars/etc.
  as an oriented celestial overlay from the camera point — real
  ephemeris, showcase-labeled. Full navigable solar system (standing
  on Mars) is OUT OF SCOPE for the /data twin (different engine
  class); the honest v1 is real positions + real orbits rendered as
  context, filed as a possible future standalone mode.
- ASTEROIDS/COMETS: JPL SBDB/CNEOS close-approach feed [public] as a
  dated event layer, not a full population render.
- LAUNCHES: upcoming launches via public schedules (e.g. Launch
  Library 2 — verify license/rate at build) + new-object appearances
  in the GP set (ORBITAL cross-tie (c), relay caveats apply).

## THE "IMPOSSIBLE" LIST — every hard ask, its named solution tier

Per the directive: nothing is removed; everything gets its best honest
tier. (Tier legend: 1 verified / 2 public / 3 procedural-labeled /
4 inference-labeled / 5 placeholder.)

| Ask | Reality | Shipped solution (tier) |
|---|---|---|
| Room-level interiors | No public data at scale; fabrication banned | 5 — "no public interior data" card; footprint+height stay tier 2 |
| Street-level photoreal everywhere | Only Google mesh has it; costs money | 2 — G3 cost-gated hand-off at deep zoom; OSM extrusions elsewhere |
| Potholes / road surface | No public dataset | 5 — placeholder; construction zones tier 2 (511 feeds, patchwork) |
| Moving control surfaces / gear / engine fans | Not in ADS-B | 4 — representative animation, labeled, polish-ranked |
| Aircraft pitch/roll/wake | Not broadcast | 4 — derived bank from turn rate, labeled derived |
| Per-tree species/branches globally | No global per-tree data | 2 city street-tree inventories + canopy-height raster; 3 for procedural showcase only |
| Full seafloor detail | ~25% direct-mapped | 2 — GEBCO + TID uncertainty overlay AS A FEATURE |
| Fiber routes | Carrier-proprietary | 5 — placeholder; landing stations tier 2 |
| Underground 3D geology volumes | Only 2D public maps | 2 — 2D formation/fault/aquifer layers, honest "2D" framing |
| Population / economic activity live | Modeled products only | 2/4 — WorldPop/GPW rasters + night-lights delta, labeled modeled/proxy |
| Future prediction | Only real models exist | 2 — GFS/NHC/tides/SGP4 forecasts, labeled; AI-invented futures banned |
| Interiors of satellites (reaction wheels etc.) | Public models only for marquee craft | 2 — O5 design-class glTF where published; symbolic markers otherwise |
| Military assets | Public info only | 2 — SATCAT class flags, public airspace/zone boundaries; nothing inferred |
| 60fps + infinite world + ray tracing | Browser reality | SCALE + LOD director + GPU instancing; ray tracing out of scope |
| Never fail on missing data | — | resilientLoad standard + per-layer degrade chips (already policy) |

## BUILD ORDER (each slice = own PR, own tag, perf-gated; E0 gates all)

Phases interleave with the standing programs — WORLDVIEW G-items,
ORBITAL O-items and SCALE S-items keep their charters; this program
sequences the NEW work and names where it consumes theirs. Sessions
pick the next unblocked slice top-down unless a higher CLAUDE.md
priority preempts.

- E0 — SPINE: A2 registry v2 schema + validation test (additive) →
  A1 LOD director (pure lib + tests) wired to 2 exemplars (orbital_sats
  envelope = the directive's satellite behavior, shipped here; aircraft
  ramp stub) → generic point-symbol + raster-field engines extracted
  with 2–3 exemplar migrations. 3–5 PRs. GATES THE REST.
- E1 — TIME: A3 global axis v1 (unify W3 + GIBS scrubbers; live/hist
  modes; per-layer capability chips) → playback → forecast zone.
- E2 — OCEAN v1: bathymetry DEM source (terrarium v1) + water toggle +
  depth palette → GEBCO self-tiled pipeline v2 + TID confidence overlay
  → gazetteer names. The showcase moment; ship it early.
- E3 — AIR TRUE-3D: airLayer altitude port (worker + custom layer) +
  terrain clamp → LOD silhouette ramp → (later, shared loader) glTF
  class models.
- E4 — ORBIT IDENTITY: SATCAT identity language on click cards (zero
  new data) → SDP4 worker upgrade (deep-space honesty + GPS DOP) →
  O5 glTF models + focus resolve (shared loader decision spike first).
- E5 — SURFACE: OSM buildings extrusion spike (source decision) +
  layer → G3 Google 3D Tiles CLIENT wiring (cost-gated, budget ledger
  already built) → 3DEP US high-res DEM upgrade → roads detail tiers.
- E6 — OCEAN v2 + UNDERGROUND: wrecks, vents, tides, currents;
  cables/MPAs license gates resolved; faults/aquifers/mines/pipelines/
  tunnels sweep (each its own thin PR off the generic engines).
- E7 — ATMOSPHERE: GFS pipeline + wind particle layer → jet-stream
  levels → aurora, storm tracks, smoke → cloud animation on A3.
- E8 — CELESTIAL: sun ephemeris + terminator/lighting → moon/planets
  overlay → eclipse/asteroid/launch event layers.
- E9 — POLISH PASSES: water shader, derived-animation tiers, camera
  descent-below-surface experiment, volumetric cloud showcase — each
  ranked behind any available data slice (Amendment 5 order).

TERRITORIES: client engines/LOD/time/layers = T-CLIENT; server
adapters/pipelines/archives = T-DATACORE; registry layers.json +
routes.ts = SHARED (serialize, last commit, minimal). The GFS/GEBCO
pipelines are scripts/ + datacore server modules (T-DATACORE).

## STANDING GATES (every slice, no exceptions)

- HONESTY: RAW vs SIGNAL classification; confidence field + the shared
  inferred-vs-verified visual vocabulary; coverage stated ("US-only",
  "25% direct-measured"); licensing verified BEFORE shipping (NC
  sources are license-gated — monetization tripwire); free substitutes
  labeled estimates.
- PERF: harness assertion at 390/768/1440 per slice; LOD envelopes
  keep the everything-on cost viewport-real; no silent drops — counts
  stated.
- UNITS: every displayed distance/length/speed/temperature through
  units.ts formatters (standing directive — transfers to every new
  layer here).
- SYMBOLS NOT DOTS: every point layer a registry SDF symbol with a
  legend entry from the same icon registry.
- PROMOTION LADDER: tests, version bump, one logical change per PR,
  backtest N/A for display surfaces but stated; client PRs run the
  visual harness.
- SIGNALS from any new data root (e.g. an ocean/underground root that
  looks tradeable) enter the ROOT VALIDATION LADDER before any
  predictive claim — the twin displays RAW freely, SELLS/claims only
  gate-2+.

## CROSS-SYSTEM TIES (filed honestly, per the standing principle)

- REAL/OPERATIONAL: bathymetry+port approaches × vessel draft/dwell
  (grounding-risk & congestion context for the port-dwell root); GFS
  wind × vessel routing deviation; aurora/space weather × grid-stress
  layers (GIC risk is a real documented mechanism — hypothesis to
  open_questions.md before any claim); faults/aquifers × facility
  siting in the location dossier.
- SHOWCASE (stated plainly, no alpha claimed): water toggle, celestial
  lighting, aircraft models, underground descent. Brand value is a
  legitimate justification on its own (ORBITAL precedent) — these make
  the data's accuracy visible and believable (Amendment 5: EXPERIENCE
  IS THE DOOR).
- The ENTITY GRAPH remains the connective tissue: every new layer
  assesses its dossier/graph join at build time (cables→carriers,
  mines→operators, wrecks→none: honest no-tie).

## RESUME STATE (update every session that touches this program)

- 2026-07-15: charter installed (this file). Survey of existing stack
  completed and recorded above (globe/terrain/GIBS/orbital/archives/
  registry/viewport/3d-tiles-proxy inventory). NOTHING NEW BUILT YET.
  NEXT: E0 slice 1 — registry v2 additive schema + validation test,
  then the LOD director with the orbital_sats camera-altitude envelope
  as its first proof (the directive's satellite zoom behavior).
  License gates open at install: TeleGeography cables (CC BY-NC-SA),
  WDPA (NC terms), GEM faults (verify), Launch Library 2 (verify).
