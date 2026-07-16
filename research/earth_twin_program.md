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

## HUMAN CLARIFICATION (2026-07-15, second directive — governs execution)

The human clarified the same day the charter was installed:
1. This program is an UPDATE TO THE SYSTEM ARCHITECTURE — not a rewrite
   of the system's data overview. VISION.md / GIP.md / CLAUDE.md and the
   existing data platform (archives, registry, ladder, honesty rules)
   remain in full force; EARTH TWIN builds ON them.
2. DATA IS NEVER MADE UP. Every layer is built from data that already
   exists (our archives, our feeds, named public datasets) or from NEW
   data we CREATE by real means (recording, accumulation, pipelines,
   derivation from measured inputs — the BUILD-FIRST precedents). The
   approximation tiers stand, but tier-3/4 output is derived-and-labeled,
   never invented.
3. CONTINUOUS BUILD MANDATE: this program is built CONTINUOUSLY UNTIL
   FINISHED. Every session that reads this charter and finds unshipped
   slices treats the next unblocked slice as queued work under the
   SESSION BUDGET fall-through — the program does not wait for
   per-slice human prompting. Sessions update RESUME STATE every time.

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
- SUBMARINE CABLES (license resolution, human-discussed 2026-07-15):
  TeleGeography's map data is CC BY-NC-SA (non-commercial) — NEVER
  wire it in, even while the platform is pre-revenue: billing exists
  and is intended (P5), so NC data would be a guaranteed future
  rip-out, violating the compounding-asset principle. The v1 is
  commercially-clean data: OSM submarine cables
  (communication=line, submarine=yes) under ODbL [commercial use OK
  with attribution + DB share-alike] + FCC cable-landing-license
  filings [US-touching, public domain], coverage-labeled. The moat
  play: routes in every public map are schematic anyway — accumulate
  OUR OWN cable dataset from public filings/landing records/operator
  announcements, labeled schematic/estimated (BUILD-FIRST tier 2). If
  TeleGeography-grade coverage ever matters commercially, that is a
  paid-license wishlist entry (tier 4, human decides).
- MARINE PROTECTED AREAS (same resolution): WDPA is NC — never wire
  it in. v1 = NOAA MPA inventory [US, public domain] + OSM
  boundary=protected_area [ODbL, global, commercial-clean] +
  national open portals (e.g. EU Natura 2000). Coverage-labeled.
- LICENSE ENFORCEMENT IS ARCHITECTURE: registry v2's
  provenance.license field makes the MONETIZATION TRIPWIRE
  machine-checkable per layer (providerCompliance.ts precedent) — a
  billing activation with an NC-licensed layer live must surface as a
  compliance warning automatically, not rely on a manual checklist.
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

## O6 — FOCUS, FIND & FOLLOW (human directive 2026-07-15, screenshot-driven; compressed, no intent lost)

The human, looking at the live followed-satellite view on their phone:
1. AUTO-3D AT ZOOM: zooming toward satellites shows them as 3D
   renderings WITHOUT clicking, at a certain zoom band; zooming past
   them (toward the surface) they go away "back to the earth"; with
   the layer off, zoom behaves normally.
2. CLICK = DEEPER FOCUS + ORBIT ARC: clicking zooms in closer than
   the browse view; you watch the satellite fly; its full orbit
   track pops up around the globe FOR THAT ONE; you can pan anywhere
   — the other side of the globe still shows the arc while the sat
   keeps moving elsewhere — until an explicit X closes the focus.
3. FIND & GROUP: with ~12k objects you cannot find the ISS — need
   search and grouping (e.g. all Starlink), plus an orbit toggle for
   a viewed group with orbits distinguishable per object.
4. SAME LOGIC FOR OTHER LIVE MOVERS: e.g. see all American flights
   at a moment (operator grouping for aircraft).
Honesty constraints carried over: 3D forms only for catalogued
classes (unknown stays a dot); arcs are SGP4-propagated real tracks;
group filters are name/callsign-prefix decodes of broadcast fields,
never inferred operators. Slices: O6-1 arc+follow-v2, O6-2 auto-3D
minis, O6-3 search/groups/group-orbits, O6-4 aircraft operator
filter.

LIVE-FEEDBACK ROUND 2 (2026-07-15/16, screenshots): fixed in the
v1.0.347 fix pack (screen-space picking, altitude-anchored ring,
RAAN-spread orbit cap 150, toggle-reload robustness, always-center,
camera-driven model attitude, zoom-to-inspect, follow tools cluster,
card minimize). REMAINING DIRECTIVE ITEMS, chartered as next slices:
- O6-5 SDP4 DEEP SPACE (the "i don't see the geo sats" ask): port the
  deep-space corrections (dscom/dpper/dsinit/dspace, Vallado
  reference) into propagate.ts so the ~800 skipped MEO/GEO objects
  (GPS, GLONASS, Galileo, GEO comms) get REAL positions; validate
  against published SDP4 test vectors before rendering; then unlock
  GPS/GLONASS/Galileo/GEO group chips and O7 GPS-DOP queries. BIG
  slice — own session recommended, worker + arc + minis inherit it
  for free once propagate() handles deep space.
- O6-6 MORE REAL MODELS (the "very accurate for the far out ones"
  ask): NASA 3D resources verified public-domain candidates through
  the existing earthtwin_real_mesh.mjs pipeline — GPS (Block IIR/IIF
  models exist), TDRS, Aqua/Terra/Landsat EO birds; pair with SDP4 so
  the far-out craft are visible first. Representative-form upgrade
  for Starlink (documented flat-bus + single-wing design, labeled
  "documented design, not imagery") as the most-clicked constellation.
  [O6-6a high-res ISS/Hubble + O6-6b Starlink form SHIPPED v1.0.348-9;
  remaining: GPS/TDRS real assets, textured-mesh rendering tier if
  vertex colors ever stop being enough at 480px.]

## O6-7 — CELESTIAL, TWO TIERS (human directive 2026-07-16, refined
## same day: "when you zoom out far enough … they appear at literally
## accurate scale")

Reference bar: the Google-Earth startup frame (atmosphere limb glow,
star field, sun-lit globe). Two honest tiers, both from real
ephemeris math (Meeus/VSOP-class algorithms — pure computation, no
data feed, no license):
1. EARTH-VIEW TIER (all current map zooms): the sun's real position
   drives globe lighting + the day/night terminator; the moon drawn
   at its real direction + true angular size + real phase; visual
   quality pass — atmosphere halo + star backdrop to the reference
   bar. All labeled "computed ephemeris".
2. SOLAR-SYSTEM TIER (zoom out past the map's minimum): hand off to
   a celestial scene at LITERALLY ACCURATE SCALE — Earth shrinks to
   truth, the Moon a real 30-earth-diameter distance away at true
   size, Sun/planets at real ephemeris positions; the emptiness IS
   the honesty (the human explicitly wants true scale on zoom-out).
   Zooming back toward Earth returns to the map seamlessly. New
   renderer surface (the map camera can't leave the globe) — spike
   the handoff first (camera continuity is the hard part), then
   bodies. Satellites' GEO shell visible at tier boundary once SDP4
   lands (the shells + moon at scale together is the showcase frame).
Every body: real position or absent — never decorative placement.

## RESUME STATE (update every session that touches this program)

- 2026-07-16 (scheduled-routine session, T-CLIENT): E2 v2 DATAMAP WIRING
  SHIPPED (v1.0.365, PR #497 — re-based/re-versioned after #496 claimed
  1.0.364 first) — the "seafloor_confidence" TID layer named
  in the E2 v2 pipeline session's own WIRING RECIPE below is now live:
  own raster-dem source + color-relief layer per SEAFLOOR_V2_REGIONS
  entry, legend + measured shares fetched live from the provenance
  sidecar, registry status planned -> live, layersWiring/layersRegistry
  ratchets updated. Drive-verified (ad hoc Playwright, not committed):
  toggling the switch at the Mariana bbox loads real pmtiles data
  (isSourceLoaded === true) and paints; toggle-off cleans up fully. Full
  detail in experiments.md. NOT done this session (deliberately, one
  logical change per PR): the recipe's OTHER half — blending the v2
  regional DEM into the existing depth "seafloor" layer — is now the
  next unclaimed E2 item. NEXT (unchanged otherwise): O6-7 tier 2 spike,
  GPS/TDRS real models, React memo boundaries, SCALE S2, keepFraction
  (HUMAN INPUT).

- 2026-07-16 (session #4 wave): O6-5 SDP4 SHIPPED+CROSS-CHECKED
  (#492); O6-6a/b high-res ISS/Hubble + Starlink documented form;
  three live-feedback fix rounds (#490 #493 + v1.0.353 rider), each
  validated by the interactive Playwright drive (drive_follow
  pattern — real server + fixture sats; 15/15). O6-7 TIER 1 SHIPPED
  (v1.0.353): day/night terminator + sun/moon overhead markers,
  computed ephemeris (lib/celestial/ephemeris.ts), anchor-validated.
  NEXT: O6-7 tier 2 spike (true-scale solar-system zoom-out — camera
  handoff is the hard part), GPS/TDRS real models, GEBCO v2 + TID,
  React memo boundaries (fresh session), SCALE S2, keepFraction
  (HUMAN INPUT). Deep-space perf watch: worker tick with ~800 SDP4
  objects — profile if lag reports return.

- 2026-07-16 (E2 v2 session, T-DATACORE): GEBCO v2 + TID CONFIDENCE
  SHIPPED as the data/pipeline half — datamap wiring deliberately
  excluded (session constraint; it is its own next slice). Built:
  scripts/gebco_seafloor_tiles.py (GEBCO download-app subsetting API
  client — contract read from the app's own JS, VERIFIED LIVE
  2026-07-16 through the proxy — plus pure-python terrarium-PNG and
  PMTiles v3 writers, proven against the real pmtiles JS reader in
  server/seafloorTiles.test.ts); datacore/gebco/tid_decode.json
  (VERBATIM GEBCO_2026 TID table — note codes 47 grounded-Argo and
  48 animal-borne are NEW vs older grids; attribution + terms
  in-file); REAL committed demo assets: Mariana Trench 138–146E/
  7–15N at native 15 arc-sec, tiled z0–9 → client/public/tiles/
  seafloor_gebco_mariana.pmtiles (22.4MB) + seafloor_tid_mariana
  .pmtiles (362KB) + provenance JSON carrying MEASURED shares
  (65.85% direct / 34.07% indirect / 0.08% land; min cell −10,931 m
  at the Challenger Deep, decoded back out of the committed archive
  by test); client/src/lib/seafloorV2.ts (decode + confidence
  classes from GEBCO's own grouping + tidConfidenceColorRelief step
  expression with transparent gaps + legend from the same table);
  layers.json seafloor_confidence entry (PLANNED). Tests: 17 pytest
  + 8 client + 5 server; all suites green, tsc baseline unchanged,
  build clean. WIRING RECIPE (next session, T-CLIENT): v2 dem as a
  second raster-dem source (pmtiles protocol, tileSize 256,
  encoding terrarium; v1 ETOPO1 stays the global fallback beneath —
  v2 covers only its bbox and renders transparent elsewhere);
  depth ramp = existing bathymetryColorRelief() unchanged; TID
  overlay = raster-dem source + color-relief using
  tidConfidenceColorRelief(); legend rows from
  tidConfidenceLegend(); surface GEBCO_ATTRIBUTION +
  GEBCO_NOT_FOR_NAVIGATION + the provenance's measured shares on
  the panel (the honesty-as-hero moment); prefer raster-resampling
  "nearest" if the layer type supports it (cross-group fringe
  caveat documented in seafloorV2.ts). SCALE PATH (documented, not
  built): more regions = more pipeline runs (14,400 deg² per basket
  item cap); full-planet z0–9 is tens of GB → boot-fetched volume
  asset per the power_us precedent, never a repo commit; global
  netCDF (7.0GB + 3.5GB TID) path documented in the script header.

- 2026-07-15 (session #3, O6 wave — same session as v1.0.340-342):
  O6 SHIPPED COMPLETE (v1.0.343-345): O6-1 orbit arc (ArcLayer +
  sampleOrbitArc) + follow v2 (drag = camera only, card X ends
  focus, per-class zoom-in); O6-2 auto-3D minis (<2500km cam,
  catalogued-only, cap 12); O6-3 SatFinder (search → same focus path
  as click, constellation chips w/ sentinel-mask filtering, group
  orbits RAAN-colored, cap disclosed); O6-4 airline callsign-prefix
  filter via wireLivePoints transformData (feeds every renderer).
  NEXT (unchanged queue): GEBCO v2 + TID, React memo boundaries
  (fresh session), keepFraction density (HUMAN INPUT), hover
  altitude labels, more real models, SCALE S2, 1Hz orbital repaint
  reconsideration. O6 follow-ups worth a look after live feedback:
  arc re-sample cadence (arc is epoch-static per focus — fine for
  one period), group-orbit live refresh (one-shot per toggle),
  MINI_MAX_CAM_KM tuning.

- 2026-07-15 (session #3 continued) — O5-3b SHIPPED (v1.0.339): the
  REAL ISS on the globe. Loader-spike decision: NO three.js — NASA's
  official public-domain ISS_stationary.glb (42MB, 247k tris) is
  preprocessed OFFLINE by scripts/earthtwin_iss_mesh.mjs into a
  committed 439KB client/public/models/iss-25544.vtm (vertex-cluster
  decimation keyed by material + normal octant so thin solar panels
  keep both faces; per-vertex colors SAMPLED from the real textures
  at each vertex UV — gold wings/white modules are the asset's own
  colors, nothing invented; provenance JSON ships beside the asset).
  Client: lib/orbital/realMesh.ts (registry {25544}, .vtm decoder →
  model3d Mesh soup, cached lazy fetch on follow only, failure falls
  back to the representative form), modelLayer.setRealMesh precedence
  (real > form > nothing, gentler rotation for real models), card
  caption names NASA + admits simplification. Integrity ratchet: the
  test decodes the COMMITTED asset against the COMMITTED meta (tris,
  ±1.2 extent, palette spread ≥8 buckets). Extending the registry =
  drop a new .vtm + one REAL_MODELS entry (same honest pipeline —
  documented public assets only).
  MERGE NOTE: concurrent session shipped #485 (E1 LIVE/HISTORICAL
  badge — REMOVED from our queue, supersession precedent) and took
  v1.0.335 on main; this branch's numbers 335-338 collide in the log
  (attribution unaffected: different files), branch continued at
  .339. origin/main merged into the branch (datamap auto-merged,
  experiments.md keep-both, version kept highest).
  NEXT: GEBCO v2 + TID confidence overlay, vessels delta
  (time/since + Cache-Control), React memo boundaries
  (LayersPanel/Legend/DetailCard), keepFraction density decision
  (human input), aircraft 3D polish (per-class silhouettes, ground
  line, altitude label on hover), more real models where public
  assets verify (Hubble/JWST candidates — NASA 3D resources).
- 2026-07-15 (session #3, after the #486 merge + branch restart):
  #486 DEPLOYED and verified live (health ok, ISS asset serving).
  SHIPPED since: v1.0.340 vessels delta (SCALE S1(b) — liveDelta.ts,
  time/since + bbox snapshot cache + Cache-Control w/ no-store error
  pins) and v1.0.341 E3 polish (per-class silhouettes from the
  broadcast emitter category only + altitude drop-lines; stable-sort
  shape groups, rows pick-aligned). Queue after this: GEBCO v2 + TID,
  React memo boundaries (fresh-session recommended: 5.9k-line
  component refactor), keepFraction density (HUMAN INPUT), altitude
  label on hover (deferred from the polish slice — hover DOM
  machinery is its own change), more real models (Hubble/JWST
  candidates), SCALE S2 server aggregation, 1Hz orbital repaint.
- 2026-07-15 (session #3, post-merge — #484 MERGED + DEPLOYED, site
  live at v1.0.334; note: the auto-merge SQUASHED the branch, so the
  40 per-commit versions collapsed into one deploy unit — acceptable
  here, nothing touched trading logic; branch restarted from main per
  the merged-branch rule): OUTAGE-CLASS SWEEP COMPLETED —
  v1.0.335 secMidas READ side streamed (the every-boot ~190MB
  cache-warm; summarizeMidasStreamed test-pinned identical) +
  v1.0.336 querySnapshot hour-files streamed (the last unbounded
  sync-zlib on any hot path). Audit verdict: the 11 remaining
  gunzipSync sites are KB-scale daily pulls, deliberately left sync.
  SESSION #3 CONTINUED — E3 SHIPPED (v1.0.338): true-altitude
  aircraft live on the branch — 2D class icons hand off to 3D
  heading-oriented silhouettes at real baro altitude at z8+
  (lib/air/airLayer, WebGL2 instanced, terrain-exaggeration-matched
  altitude, orbital far-side cull verbatim, one card handler for both
  renderers, 5 tests). Also v1.0.337 [RULE-REVIEW] harness fix:
  self-see now expands GROUP_ROW_CAP show-more buttons — main's #482
  had crossed the environmental cap and turned the harness red on
  PURE MAIN (verified); measurement change shipped alone with the
  bias statement. Harness after: 0 hard failures at all widths.
  NEXT: O5-3b real ISS model (NASA public-domain asset + loader
  spike), E1 LIVE/HISTORICAL mode chip, GEBCO v2 + TID confidence,
  vessels delta, React memo boundaries, keepFraction density decision
  (human input), aircraft 3D polish (per-class silhouettes, ground
  line, altitude-label on hover).
  PRIOR NEXT (superseded above): E3 TRUE-ALTITUDE AIRCRAFT — the original
  directive's 'see the planes at altitude'. Recipe: airLayer.ts on
  the modelLayer/satLayer template (CustomLayerInterface,
  projectTileFor3D, far-side cull) rendering heading-oriented plane
  silhouette geometry per aircraft at real baro altitude, fed from
  wireLivePoints' lastPayload each tick; LOD split — existing 2D
  symbol layer below ~z8, 3D silhouettes above (altitude
  imperceptible at continent zoom, symbols-not-dots preserved);
  terrain-clamp guard (rendered z >= DEM sample + margin, clamp
  state on the click card); velocity-vector layer folds in. Then:
  O5-3b real ISS model (NASA public-domain asset spike), E1
  LIVE/HISTORICAL mode chip, GEBCO v2 pipeline, vessels delta,
  React memo boundaries, keepFraction density decision (human).

- 2026-07-15: charter installed (this file). Survey of existing stack
  completed and recorded above (globe/terrain/GIBS/orbital/archives/
  registry/viewport/3d-tiles-proxy inventory). NOTHING NEW BUILT YET.
  NEXT: E0 slice 1 — registry v2 additive schema + validation test,
  then the LOD director with the orbital_sats camera-altitude envelope
  as its first proof (the directive's satellite zoom behavior).
  License gates open at install: TeleGeography cables (CC BY-NC-SA),
  WDPA (NC terms), GEM faults (verify), Launch Library 2 (verify).
- 2026-07-15 (same day, CONTINUOUS BUILD session #1) — E0 SPINE + first
  two vertical slices SHIPPED, four PR-sized commits on
  claude/4d-earth-digital-twin-5e7nks:
  - E0-1 (v1.0.317) REGISTRY v2: additive schema in layers.json _doc
    (altitudeRef/time/lod/provenance/renderKind) + 5 exemplar entries
    annotated + 7 validation test blocks in server/layersRegistry.test.ts
    INCLUDING the LICENSE RATCHET (provenance.commercialOk=false may
    never ship — the monetization tripwire machine-checked; resolution
    of the cables/WDPA license gates: NC sources NEVER wired, v1 = OSM
    ODbL + US-gov public domain, own-accumulation is the moat play,
    paid license = wishlist).
  - E0-2 (v1.0.318) LOD DIRECTOR: client/src/lib/lod.ts (pure, 6 tests —
    512px-world camera-altitude math verified against installed
    maplibre 5.24.0's transform.getCameraAltitude; lodOpacity FAILS
    OPEN); SatLayer.setGlobalOpacity (u_opacity uniform, zero-draw at
    0, far-side cull byte-identical); datamap move-handler drives the
    registry envelope {camMinKm:100, fadeBandKm:150} — satellites fade
    out near the ground, worker pauses ('stop' keeps gp; 'start' resumes
    instantly), panel note states the hidden state, click handler goes
    dormant while hidden (stale buffer honesty). THE DIRECTIVE'S
    SATELLITE ZOOM BEHAVIOR IS LIVE.
  - E2-1 (v1.0.319) SEAFLOOR — DRAIN THE OCEAN v1: NOAA ETOPO1 via AWS
    Terrain Tiles terrarium (verified live; z0 tile header names
    ETOPO1), MapLibre color-relief (verified present + globe-capable in
    5.24.0) with lib/bathymetry.ts depth ramp transparent above sea
    level — one source of truth for map ramp AND legend chips; own
    raster-dem source, never touches terrain-dem/setTerrain;
    FIELD_OPACITY_PROP override (color-relief-opacity); honesty pins
    (interpolation + not-for-navigation) tested. GEBCO 15-arcsec + TID
    confidence overlay remains the chartered v2.
  - E4-1 (v1.0.320) SATELLITE IDENTITY: lib/orbital/identity.ts (pure,
    5 tests) — SATCAT objectType/RCS buckets ('smallsat/CubeSat-class
    size' ONLY for SMALL-RCS payloads, labeled derived), owner code,
    launch date, documented opStatus decode, operator→ticker via
    entityJoin with join provenance on-card; SATCAT fetched in the
    background on layer enable (module cache, 24h TTL, never blocks,
    never replaces the index-aligned GP ref).
  GATES: all suites green per commit (registry 24/24, client libs
  155/155, server 684/684 at E0-1, tsc 66-line baseline unchanged,
  builds clean). VISUAL HARNESS CAVEAT: two container restarts occurred
  in this sandbox while the harness ran alongside heavy parallel work —
  run recorded in experiments.md with whatever outcome the final
  attempt produced; confidence otherwise rests on zero-DOM/CSS-diff
  reasoning + full test gates (the 2026-07-15 PFAS session's precedent).
  NEXT (in charter order): E0 remainder — generic point-symbol +
  raster-field ENGINES extracted with 2-3 exemplar migrations (the last
  E0 piece); then E1 global time axis v1 (unify W3 + GIBS scrubbers);
  then E2 v2 (GEBCO self-tiled pmtiles pipeline + TID confidence
  overlay + gazetteer names); E3 true-altitude aircraft (port the
  orbital 3D path); aircraft LOD ramp consumes the E0-2 director.
  Hypothesis to judge (stated at E0-1): if E2/E3-class slices still
  need large hand-wired useEffects after the engines land, the spine
  design is wrong — revisit before sweeping.
- 2026-07-15 (session #2, human clarification of the 4D intent —
  verbatim intent: "you could zoom in on a satellite and see, if we
  had a pic, a 3D rendering of it — so if it was a CubeSat or a sat we
  know of we would have that and it would be moving around, but if you
  clicked on it, it would remain in focus"): this SHAPES O5's build
  order. O5-1 SHIPPED same session (v1.0.330): click → live camera
  FOLLOW (1Hz re-center on the fresh SGP4 position, 800ms ease, amber
  focus ring; user drag/empty-ground click/LOD-hide all release —
  the user always wins; sentinel ticks never move the camera). O5-2
  NEXT: 3D rendering at focus — honest tiers per orbital_program.md:
  (a) class-REPRESENTATIVE parametric forms (CubeSat box+panels,
  rocket-body cylinder, generic bus) labeled "representative form,
  derived from catalog class — not a photo of this unit" (these are
  the charter's symbolic markers, in 3D); (b) real models ONLY for
  documented craft with verifiable public assets (ISS via NASA 3D
  resources public domain; Starlink per published imagery) — loader
  decision spike first (lazy three.js vs minimal glTF parser).
  O5-2(a) SHIPPED same session (v1.0.331): lib/orbital/model3d (pure
  forms — cubesat/smallsat/bus/rocket-body/fragment, honest-label +
  unknown-stays-fragment + determinism all test-pinned) + SatModelView
  (zero-dep raw-WebGL1 card viewer, lit + tumbling, GL cleanup,
  no-WebGL degrade); renders only when the catalog knows the class.
  O5-2b SHIPPED same session (v1.0.332, human corrected: ON THE MAP,
  not a side viewer — SatModelView deleted): lib/orbital/modelLayer,
  a CustomLayerInterface on the satLayer template drawing the form at
  the followed satellite's LIVE position/altitude at constant ~72px
  (anchor.w-scaled clip offsets, test-pinned), identical far-side
  cull, repaint self-requested only while visible, sentinel ticks
  hide the model, unknown class = ring-only. The full directive arc
  is now LIVE: whole sky → click → follow → the point becomes a
  spacecraft riding its orbit on the globe.
  O5-3a SHIPPED same session (v1.0.333, human: "don't display as a dot
  if we have the sat info"): SYMBOLS NOT DOTS applied to the sky —
  identified objects render as SDF type glyphs in the point field
  (payload = bus+wings, rocket body = capsule, debris = shard, 2.6x
  size), color keeps orbit class; a dot now MEANS unidentified;
  misaligned shape buffers fall back to dots (never a mislabel,
  test-pinned); legend decode line added.
  O5-3b = the real-model upgrade (ISS/Starlink, loader spike) — next.
  SITE OUTAGE, same session (2026-07-15 ~09:30+): production 502 crash
  loop, root-caused + fixed — secMidas boot archive built a ~189MB
  string into gzipSync under the 512MB cap the first time a new SEC
  quarter published (dedup guard hid it for weeks; the innocent #482
  deploy restarted into it). Fix = streamed archive write, verified
  end-to-end in-sandbox (fixed build boots, archives real 2025q4,
  serves health at 130MB heap). Hotfix PR #483 (cherry-picked, own
  branch claude/hotfix-midas-oom, v1.0.319-on-main's-line); merge on
  green CI restores the site. The same commit is on this branch.
  HARNESS after E1-1+O5-1+O5-2: ALL PASS 0 hard failures; perf
  medians improved vs session start: 33/83/117ms at 390/768/1440
  (were 50/133/167), TTI down ~30% at every width.
- 2026-07-15 (session #2) — E1 SLICE 1 SHIPPED (v1.0.329, after the
  perf pass v1.0.324-328 recorded in scale_program.md): GLOBAL TIME
  AXIS v1 — lib/timeAxis (subscribe-store + gibsDateForAxis honest
  latency clamping, 4 tests); the Time Machine panel PUBLISHES the
  axis on every committed scrub and returns the world to LIVE on
  close; all five dated GIBS layers FOLLOW (one scrubber = archives
  replay + dated imagery at the same instant); per-layer scrubbers
  remain manual overrides; firetemp honestly excluded (sub-daily
  latest-scan-only). E1 REMAINDER: registry-driven follower discovery
  (use the v2 time blocks instead of the hand-list of five setters),
  a visible LIVE/HISTORICAL mode chip outside the panel, forecast
  zone, sub-daily scan picker.
- 2026-07-15 (same session, verify pass) — 3-lens adversarial review
  (correctness/constitution/UX-perf) over the four slices: NO
  high-severity findings; constitution clean (frozen paths untouched,
  test files additions-only, honesty/units/attribution pass). Fixes
  shipped: v1.0.321 (SATCAT empty-response cache poisoning — the one
  MEDIUM; + retry-on-click + not-in-catalog wording), v1.0.322 (LOD
  legend copy said "street zoom", fade actually completes at z≈9.5
  city scale — copy fixed, envelope numbers untouched; + resize
  listener), v1.0.323 (deterministic seafloor-below-hillshade z-order
  + coastline-bleed honesty note). Full suites after fixes: server
  685/685, client libs 155/155, tsc baseline unchanged, build clean.
  VISUAL GATE MET: solo harness run = 0 hard failures at 390/768/1440
  (the earlier restarts were concurrency-induced — never run the
  harness alongside other heavy work in this sandbox); screenshots
  reviewed, Seafloor row design-consistent.
  NEW QUEUED SLICE from review evidence — E4-2 SATCAT PARSE OFF-THREAD:
  parseSatcat measured ~280-340 ms synchronous main-thread block on a
  desktop-class CPU (~63k rows; expect ~1 s+ on mobile) exactly at
  layer-enable; move fetch+parse into a worker (satWorker pattern) and
  consider HTTP cache headers (the module cache dies per reload → 6 MB
  re-download per page load with the layer on). Slot it with the E0
  engines work. MERGE NOTE for the human: the branch carries the four
  logical changes as four version-gated commits — merge preserving
  per-commit attribution (or as separate PRs); PROMOTION #6 visual
  harness could not complete in this sandbox (container-level restarts
  under headless-Chromium WebGL load, twice) — run `npm run visual --
  --page data` on a normal machine at review time.
- 2026-07-15 (continuous-build session, T-CLIENT): cross-reference —
  scale_program.md's queue item (c), 1Hz orbital repaint, shipped this
  session (v1.0.343, SatLayer.updatePositions accumulation-based
  repaint skip). Full detail lives in scale_program.md's RESUME STATE;
  noted here only because it touches the same satWorker/SatLayer O5
  code this charter's follow/model-layer work depends on — followed-
  satellite camera tracking is unaffected (modelLayer.setAnchor still
  repaints every tick unconditionally). ALSO BACKFILLED THIS SESSION
  (pure record-keeping, no code changed): PR #487 shipped v1.0.342 (the
  real Hubble model + the generalized earthtwin_real_mesh.mjs tool, per
  the commit's own message) but never got its own experiments.md
  session-log entry — only v1.0.340-341 was logged. JWST is the
  natural next real-model candidate now that the mesh tool is
  generalized and reusable (queue carried over from the v1.0.339/341
  NEXT notes above).
