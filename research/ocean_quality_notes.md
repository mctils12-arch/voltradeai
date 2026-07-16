# Ocean quality pass — drained-ocean basemap + globe atmosphere (research half)

2026-07-16 · EARTH TWIN E2 quality pass, subagent research deliverable.
Companion code: `client/src/lib/oceanBasemap.ts`, `client/src/lib/globeAtmosphere.ts`
(+ tests). NEW FILES ONLY — datamap.tsx wiring belongs to the parent session.

Reference target: Google Earth's Atlantic view — detailed shaded seafloor
ridges + a thin blue atmosphere rim at the limb.

## 1. Source comparison (all terms read + all endpoints live-fetched this session)

| Candidate | Resolution / coverage | License (verdict, citation) | Sample fetch (via HTTPS proxy, 2026-07-16) | Visual vs. reference |
|---|---|---|---|---|
| **GEBCO WMS** `wms.gebco.net/mapserv`, layer `GEBCO_LATEST` | GEBCO_2024 grid, 15 arc-sec (~465 m/px eq.), global incl. land + sub-ice variants | **Commercial OK.** "The GEBCO Grid is placed in the public domain and may be used free of charge." Users may "Commercially exploit The GEBCO Grid … by including it in their own product or application." Attribution form requested in the WMS Abstract; "should NOT be used for navigation." (gebco.net/data-products/gridded-bathymetry/terms-of-use) | GetCapabilities: CRS **EPSG:3857**/4326/3395, PNG+JPEG. GetMap 512×512 EPSG:3857 N-Atlantic → **HTTP 200, image/jpeg, 61,590 B, decoded valid 512×512 JPEG, ~1.1 s**. z7 Mid-Atlantic-Ridge tile → 200, 31,334 B, abyssal-hill fabric + fracture zones clearly readable | **Best match** — same shaded-relief-over-depth-tint language as the reference; strongest ridge texture of all candidates |
| NOAA NCEI `DEM_mosaics/DEM_global_mosaic_hillshade` ImageServer | ETOPO 2022 15 arc-sec + coastal DEMs, global | **Commercial OK.** US Government work → public domain (17 U.S.C. §105); NCEI product page states citation form only (doi:10.25921/fd45-gt74), no use restriction | exportImage bboxSR=3857, same bbox → **200, image/jpeg, 31,176 B, valid 512×512, ~1.1 s** | Good; slightly darker/flatter shading than GEBCO |
| EMODnet Bathymetry `tiles.emodnet-bathymetry.eu` | DTM 2022 ~1/16 arc-min in **European seas only** (GEBCO fill elsewhere); web_mercator tileset = `baselayer`/`baselayer_land` only | **Commercial OK.** "data products created by EMODnet are owned by the EU and therefore licensed under Creative Commons CC-BY 4.0" (emodnet.ec.europa.eu/en/terms-use-emodnet-online-services-data-and-data-products) | `/2020/baselayer/web_mercator/3/2/3.png` → **200, PNG 256×256 RGBA, 142,617 B, ~2.5 s** | Softer atlas style, less ridge drama; right answer later for a Europe zoom-in tier, wrong for the global hero globe |
| GMRT WMS `gmrt.org/services/mapserver/wms_merc`, layer `GMRT` | Multibeam synthesis ~100 m where surveyed, GEBCO-class base elsewhere, global | **Commercial OK.** "licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) License"; credit to GMRT required; "Data should not be used for navigation purposes." (gmrt.org/about/terms_of_use.php) | GetMap EPSG:3857 same bbox → **200, image/jpeg, 38,549 B, ~1.0 s** | Pastel palette mismatched to reference; academic (LDEO) service with unstated capacity — not a production primary. Candidate for a future surveyed-swath detail tier |
| Esri World Ocean Base | z0–16, global, prettiest cartography | **FORBIDDEN for us — EXCLUDED.** Esri Web Site and Service Terms of Use: "You may download, view, copy, and print Services … solely for your own internal purposes, or for your own noncommercial external purposes unless otherwise specified." and "Services may not be reproduced or transmitted for commercial purposes". The site sells data access (MONETIZATION TRIPWIRE) → cannot ship. Consistent with the repo's standing Esri World Imagery reading (data_census.md #9: no redistribution) | not fetched — excluded on license before wiring | n/a |

**Recommendation: GEBCO WMS `GEBCO_LATEST` primary, NOAA NCEI hillshade fallback.**
Both are public-domain-grade with global coverage; GEBCO is visually the
closest to the Google Earth reference and is served from a cached MapServer
run by BODC explicitly "for use in your applications" (WMS Abstract).

Honesty notes (must survive into the UI): GEBCO/ETOPO are soundings blended
with satellite-gravity interpolation — indicative relief, never navigational
(both sources say so verbatim). GEBCO's own Abstract: "GEBCO's grids are
mainly deeper water data sets and do not contain detailed bathymetry in
shallower water areas." The existing E2 charter TID-confidence overlay
(measured vs interpolated) remains the honest companion; `GEBCO_LATEST_3`
(measured-only pixels, interpolation transparent) is available on the same
endpoint for it.

## 2. The land-pixel tradeoff (decided honestly, alternatives recorded)

`GEBCO_LATEST` paints land with hypsometric green/tan. MapLibre cannot mask
a raster by another source's elevation, so while the ocean basemap renders
above the imagery base, land shows GEBCO's tint instead of satellite imagery.
Options considered:

- (a) Accept GEBCO land during drained-ocean mode — what the configs assume.
  One layer, zero extra requests; the whole globe becomes a coherent relief
  rendering (closest to the reference screenshot, which is also relief-tinted
  in ocean and photographic only on land).
- (b) Keep land imagery by adding a `hillshade` layer on the EXISTING
  `seafloor-dem` (terrarium/ETOPO1) source under the color-relief tint —
  no new source, land imagery untouched (hillshade shades land too, matching
  the existing optional terrain-hillshade look), but only ~1 arc-min detail:
  visibly blurrier ridges than the 15 arc-sec GEBCO render.
- (c) Vector land mask over the raster — rejected: needs a new land-polygon
  source and still cannot restore the imagery beneath the raster.

Parent's call between (a) — recommended — and (b); both are honest.

## 3. Precise wiring instructions (parent session, datamap.tsx)

All ids/configs come from `client/src/lib/oceanBasemap.ts`:

1. Import `{ OCEAN_BASEMAP_SOURCE_ID, OCEAN_BASEMAP_LAYER_ID,
   oceanBasemapSource, oceanBasemapLayer, oceanBasemapFallbackSource }`.
2. In the existing seafloor effect (datamap.tsx ~line 1335, `enabled.seafloor`
   branch), after `seafloor-dem`/`seafloor-relief` setup:
   - `map.addSource(OCEAN_BASEMAP_SOURCE_ID, oceanBasemapSource())`
   - `map.addLayer(oceanBasemapLayer(opacityOf("seafloor")), "seafloor-relief")`
     — i.e. insert **directly below the existing `seafloor-relief`
     color-relief layer**, which stays on top as the legend-bearing depth
     tint (its stops/legend chips remain the one source of truth,
     lib/bathymetry.ts). Suggested rebalance so the imagery reads through:
     drop the color-relief layer's `color-relief-opacity` toward ~0.25 while
     the ocean basemap is active (parent judgment; ramp/legend unchanged).
   - Teardown in the `!enabled.seafloor` branch mirrors the existing pattern
     (removeLayer then removeSource, both guarded).
   - Failure path: on source/tile `error` for the GEBCO source, swap tiles to
     `oceanBasemapFallbackSource()` (NOAA) — degrade, never break; keep the
     status-note honest about which source is live.
3. Status note should carry provenance + the non-navigational caveat, e.g.
   "ocean drained — GEBCO 2024 shaded relief (15 arc-sec; soundings +
   gravity interpolation, not navigational)".
4. Attribution: source configs already carry the exact requested strings —
   MapLibre's AttributionControl surfaces them automatically once the source
   is added; nothing extra to wire, do not truncate them.
5. RAW overlay classification (RAW OVERLAYS vs SIGNALS rule): this is a raw
   third-party rendering with source attribution, no predictive claim — no
   ladder gate needed; label stays "RAW".

### setSky wiring (from `client/src/lib/globeAtmosphere.ts`)

Verified against the installed maplibre-gl@5.24.0 (`dist/maplibre-gl.d.ts`
line 12956 `setSky`; style-spec `index.d.ts` line 473 `SkySpecification`;
shipped shader source — details in the module header):

- v5.24 sky/atmosphere API = `map.setSky(SkySpecification)` with EXACTLY
  these fields: `sky-color`, `horizon-color`, `fog-color`,
  `fog-ground-blend`, `horizon-fog-blend`, `sky-horizon-blend`,
  `atmosphere-blend` (+ `-transition` variants). There are NO
  projection-level atmosphere options in v5.24 (`ProjectionSpecification`
  is `{ type }` only); there is no `setAtmosphere` — Mapbox-style `fog`/
  `star-intensity`/`space-color` fields DO NOT EXIST here.
- The globe limb glow is a built-in physical Rayleigh/Mie shader: only
  `atmosphere-blend` scales it (× the library's own globe→mercator
  transition); its blue is hardcoded physics, and its sun direction follows
  `map.setLight().position` — a future tie to the O6-7 real-sun terminator
  is one `setLight` call (cross-system note, real but optional).
- `sky-color`/`horizon-color` paint a screen-space gradient that is fully
  transparent at full globe and fades IN during the projection transition;
  fog needs pitch ≥ ~60° (`calculateFogBlendOpacity`).

Wiring: ONE call after style load —
`map.setSky(DEFAULT_SKY)` (the union preset). No zoom listener needed: the
limb preset's `atmosphere-blend` is already the zoom-interpolated expression
(`1` at z0–5 → `0` at z7, `LIMB_VISIBLE_ZOOM_MAX`), and the horizon fields
only become visible where they belong. If the map starts in flat/mercator
preference, the same call is still correct (limb pass is inert there).
Re-apply after any `setStyle` (style reloads reset sky). Guard with
try/catch like every other premium visual (degrade, never break) — under
SwiftShader/CI the atmosphere pass simply renders cheaply, no special-case.

## 4. Evidence file inventory (scratchpad only, NOT committed)

gebco_caps.xml (24,513 B), gebco_atlantic.jpg (61,590 B, 512², decoded OK),
gebco_z7.jpg (31,334 B), noaa_hillshade.png (31,176 B jpeg), emodnet_tile.png
(142,617 B), gmrt_caps.xml (10,533 B), gmrt_atlantic.jpg (38,549 B) — all
HTTP 200 through the configured proxy, 2026-07-16.

## 5. Gates run (this branch, 2026-07-16)

- Client unit tests: 239/239 pass (`npx tsx --test client/src/lib/**/*.test.ts`),
  including the 10 new ocean/atmosphere tests.
- `npx tsc --noEmit`: byte-identical error list vs. pre-change baseline
  (66 pre-existing lines, none new).
- `npm run build`: succeeds (vite client + esbuild server).
