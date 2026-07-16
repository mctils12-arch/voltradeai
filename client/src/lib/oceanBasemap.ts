// Ocean basemap — EARTH TWIN E2 drained-ocean QUALITY pass (research half,
// 2026-07-16; findings + wiring recipe in research/ocean_quality_notes.md).
//
// Goal: Google-Earth-grade shaded seafloor for the /data globe. The existing
// E2-1 layer (seafloor-dem terrarium + color-relief tint, lib/bathymetry.ts)
// carries the depth LEGEND; this module adds the pre-rendered shaded-relief
// IMAGERY that makes ridges, trenches and abyssal-hill fabric actually
// readable. Configs only — datamap.tsx wiring belongs to the parent session.
//
// ── LICENSING RECORD (every candidate read this session, 2026-07-16) ──
// The repo sells data access (VISION.md), so only commercial-OK sources may
// be recommended. Verdicts, with the operative sentence quoted:
//
// 1. GEBCO WMS (wms.gebco.net) — RECOMMENDED, commercial OK.
//    Terms: https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use
//    "The GEBCO Grid is placed in the public domain and may be used free of
//    charge." Users may "Commercially exploit The GEBCO Grid, by, for
//    example, combining it with other information, or by including it in
//    their own product or application." Attribution requested (form below);
//    "should NOT be used for navigation or for any other purpose involving
//    safety at sea." WMS GetCapabilities AccessConstraints (fetched live
//    2026-07-16): no rate limit stated; no-navigation + no-warranty only.
//    Live-verified from this environment (through the HTTPS proxy):
//    GetCapabilities lists CRS EPSG:3857/4326/3395, layers GEBCO_LATEST
//    (shaded relief), GEBCO_LATEST_2 (flat), GEBCO_LATEST_3 (measured-only),
//    _SUB_ICE_TOPO, _TID; sample 512×512 EPSG:3857 GetMap (N-Atlantic,
//    bbox -10018754,0,-5009377,5009377) → HTTP 200, image/jpeg, 61,590 B,
//    decoded as valid 512×512 JPEG, ~1.1 s.
//
// 2. NOAA NCEI DEM_global_mosaic_hillshade (gis.ngdc.noaa.gov ImageServer)
//    — FALLBACK, commercial OK. US Government work (17 U.S.C. §105 — public
//    domain; NCEI ETOPO page states citation form only, no use restriction).
//    ETOPO 2022 (15 arc-sec) + coastal DEMs, hypsometric tint + hillshade.
//    Live-verified: exportImage bboxSR=3857 same bbox → HTTP 200, image/jpeg,
//    31,176 B, valid 512×512, ~1.1 s. Dynamic (uncached) rendering — treat as
//    fallback, not primary, to be polite to a shared federal service.
//
// 3. EMODnet Bathymetry (tiles.emodnet-bathymetry.eu) — commercial OK but
//    NOT selected. Terms: https://emodnet.ec.europa.eu/en/terms-use-emodnet-
//    online-services-data-and-data-products — "data products created by
//    EMODnet are owned by the EU and therefore licensed under Creative
//    Commons CC-BY 4.0." Live-verified z3 web-mercator tile → 200, PNG 256px,
//    142,617 B. High-res DTM is EUROPEAN SEAS only (GEBCO fill elsewhere);
//    softer atlas style, web_mercator tileset carries only baselayer/
//    baselayer_land. Right answer for a future Europe-zoom upgrade, not for
//    the global hero globe.
//
// 4. GMRT (gmrt.org WMS) — commercial OK (CC BY 4.0: "licensed under a
//    Creative Commons Attribution 4.0 International (CC BY 4.0) License";
//    "Data should not be used for navigation purposes.") — NOT selected:
//    academic service (Lamont-Doherty) with unstated capacity, pastel style
//    mismatched to the reference look; its ~100 m multibeam swaths make it a
//    candidate for a zoom-in detail tier later. Live-verified EPSG:3857
//    GetMap → 200, image/jpeg, 38,549 B.
//
// 5. Esri World Ocean Base — EXCLUDED, commercial use forbidden for us.
//    Esri Web Site and Service Terms of Use
//    (https://www.esri.com/en-us/legal/terms/web-site-service): "You may
//    download, view, copy, and print Services available on Esri Web sites
//    solely for your own internal purposes, or for your own noncommercial
//    external purposes unless otherwise specified." and "Services may not
//    be reproduced or transmitted for commercial purposes". This site sells
//    data subscriptions (MONETIZATION TRIPWIRE, CLAUDE.md) → forbidden.
//    Consistent with the repo's standing Esri World Imagery reading
//    (research/data_census.md #9: keyless reads for internal recency checks
//    only, no redistribution).
//
// HONESTY: all of these render the GEBCO/ETOPO class of grids — soundings
// blended with satellite-gravity interpolation. Indicative relief, never
// navigational. GEBCO_LATEST includes hypsometric LAND coloring: MapLibre
// cannot mask a raster by another source's elevation, so while the drained-
// ocean basemap is on, land shows GEBCO's tint instead of satellite imagery
// (the notes file records the alternatives considered and rejected).

import type { RasterSourceSpecification, LayerSpecification } from "maplibre-gl";

/** Source + layer ids (parent wires them into datamap.tsx). */
export const OCEAN_BASEMAP_SOURCE_ID = "ocean-basemap";
export const OCEAN_BASEMAP_LAYER_ID = "ocean-basemap";

/** GEBCO WMS endpoint (BODC-operated). Live-verified 2026-07-16. */
export const GEBCO_WMS_ENDPOINT = "https://wms.gebco.net/mapserv";

/** Attribution in the exact form the WMS GetCapabilities Abstract requests. */
export const GEBCO_ATTRIBUTION =
  "Imagery reproduced from the GEBCO_2024 Grid, GEBCO Compilation Group (2024) " +
  "GEBCO 2024 Grid (doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f)";

export const NOAA_HILLSHADE_ATTRIBUTION =
  "NOAA NCEI ETOPO 2022 global relief (doi:10.25921/fd45-gt74) — public domain";

/** WMS tile pixel size. 512 halves request count vs 256 and was the size
 *  live-verified against the endpoint. */
export const OCEAN_BASEMAP_TILE_SIZE = 512;

/** Native grid is 15 arc-sec (~465 m/px at the equator). A 512px tile hits
 *  ~305 m/px at z8 — beyond that the WMS only upsamples, so the source stops
 *  requesting there (MapLibre overzooms the z8 tiles for free) instead of
 *  hammering the server for no added detail. */
export const OCEAN_BASEMAP_MAX_ZOOM = 8;

/** Shaded-relief layer verified in GetCapabilities. GEBCO_LATEST_SUB_ICE_TOPO
 *  swaps in under-ice topography around Greenland/Antarctica. */
export type GebcoWmsLayer =
  | "GEBCO_LATEST"
  | "GEBCO_LATEST_2"
  | "GEBCO_LATEST_3"
  | "GEBCO_LATEST_SUB_ICE_TOPO";

/**
 * MapLibre raster-source tiles template for the GEBCO WMS: MapLibre
 * substitutes {bbox-epsg-3857} per tile. width/height must equal the
 * source's tileSize or the imagery is silently rescaled.
 */
export function gebcoWmsTileUrl(
  layer: GebcoWmsLayer = "GEBCO_LATEST",
  size: number = OCEAN_BASEMAP_TILE_SIZE,
): string {
  return (
    `${GEBCO_WMS_ENDPOINT}?service=WMS&version=1.3.0&request=GetMap` +
    `&layers=${layer}&styles=&crs=EPSG:3857&bbox={bbox-epsg-3857}` +
    `&width=${size}&height=${size}&format=image/jpeg`
  );
}

/** RECOMMENDED: GEBCO shaded-relief raster source (global, commercial-OK). */
export function oceanBasemapSource(): RasterSourceSpecification {
  return {
    type: "raster",
    tiles: [gebcoWmsTileUrl()],
    tileSize: OCEAN_BASEMAP_TILE_SIZE,
    maxzoom: OCEAN_BASEMAP_MAX_ZOOM,
    attribution: GEBCO_ATTRIBUTION,
  };
}

/**
 * FALLBACK: NOAA NCEI global DEM hillshade (public domain). Dynamic
 * ImageServer export — same {bbox-epsg-3857} substitution works on ArcGIS
 * exportImage. Use only if the GEBCO WMS errors (degrade, never break).
 */
export const NOAA_HILLSHADE_EXPORT =
  "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic_hillshade/ImageServer/exportImage" +
  `?bbox={bbox-epsg-3857}&bboxSR=3857&imageSR=3857&size=${OCEAN_BASEMAP_TILE_SIZE},${OCEAN_BASEMAP_TILE_SIZE}&format=jpgpng&f=image`;

export function oceanBasemapFallbackSource(): RasterSourceSpecification {
  return {
    type: "raster",
    tiles: [NOAA_HILLSHADE_EXPORT],
    tileSize: OCEAN_BASEMAP_TILE_SIZE,
    maxzoom: OCEAN_BASEMAP_MAX_ZOOM,
    attribution: NOAA_HILLSHADE_ATTRIBUTION,
  };
}

/**
 * Recommended paint: linear resampling keeps the relief smooth on the globe;
 * the slight contrast/saturation trim seats the (bright, atlas-toned) GEBCO
 * imagery into the dark design system without lying about the data —
 * geometry is untouched, only tone. Fade keeps tile loads jank-free
 * (PREMIUM EXPERIENCE STANDARD: perceived performance is a feature).
 */
export function oceanBasemapPaint(opacityPct = 100): Record<string, unknown> {
  return {
    "raster-opacity": Math.max(0, Math.min(100, opacityPct)) / 100,
    "raster-resampling": "linear",
    "raster-fade-duration": 300,
    "raster-contrast": 0.05,
    "raster-saturation": -0.05,
  };
}

/** Full layer config; parent inserts with the repo's usual beforeId anchor
 *  (below seafloor-relief — see research/ocean_quality_notes.md). */
export function oceanBasemapLayer(opacityPct = 100): LayerSpecification {
  return {
    id: OCEAN_BASEMAP_LAYER_ID,
    type: "raster",
    source: OCEAN_BASEMAP_SOURCE_ID,
    paint: oceanBasemapPaint(opacityPct),
  } as LayerSpecification;
}

/** Machine-readable license register — the honesty machinery, exportable to
 *  provenance UI. Every entry was read this session (quotes in header). */
export interface OceanSourceVerdict {
  name: string;
  license: string;
  commercialUse: "allowed" | "forbidden";
  termsUrl: string;
  role: "recommended" | "fallback" | "not-selected" | "excluded";
}

export const OCEAN_SOURCE_VERDICTS: Record<string, OceanSourceVerdict> = {
  gebco: {
    name: "GEBCO WMS shaded relief (GEBCO_2024 grid, 15 arc-sec)",
    license: "Public domain; attribution requested; not for navigation",
    commercialUse: "allowed",
    termsUrl: "https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use",
    role: "recommended",
  },
  noaa: {
    name: "NOAA NCEI DEM global mosaic hillshade (ETOPO 2022 + coastal DEMs)",
    license: "US Government work — public domain (17 U.S.C. §105); cite NCEI",
    commercialUse: "allowed",
    termsUrl: "https://www.ncei.noaa.gov/products/etopo-global-relief-model",
    role: "fallback",
  },
  emodnet: {
    name: "EMODnet Bathymetry tiles (DTM 2022, European seas hi-res)",
    license: "CC BY 4.0 (EU-owned data product)",
    commercialUse: "allowed",
    termsUrl:
      "https://emodnet.ec.europa.eu/en/terms-use-emodnet-online-services-data-and-data-products",
    role: "not-selected",
  },
  gmrt: {
    name: "GMRT synthesis WMS (multibeam ~100 m where surveyed)",
    license: "CC BY 4.0; not for navigation",
    commercialUse: "allowed",
    termsUrl: "https://www.gmrt.org/about/terms_of_use.php",
    role: "not-selected",
  },
  esriWorldOceanBase: {
    name: "Esri World Ocean Base",
    license:
      "Esri Web Site and Service Terms of Use — noncommercial external use only",
    commercialUse: "forbidden",
    termsUrl: "https://www.esri.com/en-us/legal/terms/web-site-service",
    role: "excluded",
  },
};
