// Ocean basemap configs — EARTH TWIN E2 quality pass. Pins: (1) the WMS
// URL template is exactly what MapLibre's {bbox-epsg-3857} substitution
// needs, (2) attribution is always carried, (3) every paint key exists in
// the INSTALLED style spec (imported, not memorized), (4) the licensing
// register never lets a commercial-forbidden source into a config.
import { test } from "node:test";
import assert from "node:assert/strict";
import { v8 } from "@maplibre/maplibre-gl-style-spec";
import {
  OCEAN_BASEMAP_SOURCE_ID,
  OCEAN_BASEMAP_LAYER_ID,
  OCEAN_BASEMAP_TILE_SIZE,
  OCEAN_BASEMAP_MAX_ZOOM,
  GEBCO_ATTRIBUTION,
  NOAA_HILLSHADE_ATTRIBUTION,
  gebcoWmsTileUrl,
  oceanBasemapSource,
  oceanBasemapFallbackSource,
  oceanBasemapPaint,
  oceanBasemapLayer,
  OCEAN_SOURCE_VERDICTS,
} from "./oceanBasemap.js";

test("GEBCO WMS template: valid URL, MapLibre bbox token, EPSG:3857, size matches tileSize", () => {
  const tpl = gebcoWmsTileUrl();
  assert.ok(tpl.startsWith("https://wms.gebco.net/mapserv?"), "verified GEBCO endpoint");
  assert.ok(tpl.includes("bbox={bbox-epsg-3857}"),
    "MapLibre raster sources speak WMS via the {bbox-epsg-3857} substitution token");
  assert.ok(tpl.includes("crs=EPSG:3857"), "WMS 1.3.0 uses crs= (not srs=)");
  assert.ok(tpl.includes("service=WMS") && tpl.includes("version=1.3.0") && tpl.includes("request=GetMap"));
  assert.ok(tpl.includes("layers=GEBCO_LATEST&"), "shaded-relief layer verified in GetCapabilities");
  assert.ok(tpl.includes(`width=${OCEAN_BASEMAP_TILE_SIZE}`) && tpl.includes(`height=${OCEAN_BASEMAP_TILE_SIZE}`),
    "WMS pixel size must equal the source tileSize or tiles get silently rescaled");
  // URL must be well-formed once the token is substituted with a real bbox
  const substituted = tpl.replace("{bbox-epsg-3857}", "-10018754.171,0,-5009377.086,5009377.086");
  assert.doesNotThrow(() => new URL(substituted));
  // and the template itself must carry no whitespace or unencoded braces
  assert.ok(!/\s/.test(tpl), "no whitespace in the tiles template");
});

test("recommended source: raster, single template, attribution in GEBCO's requested form", () => {
  const src = oceanBasemapSource();
  assert.equal(src.type, "raster");
  assert.equal(src.tiles?.length, 1);
  assert.equal(src.tileSize, OCEAN_BASEMAP_TILE_SIZE);
  assert.equal(src.maxzoom, OCEAN_BASEMAP_MAX_ZOOM);
  assert.ok(OCEAN_BASEMAP_MAX_ZOOM <= 9,
    "15 arc-sec native (~465 m/px): requesting past ~z8 only upsamples — overzoom instead");
  assert.equal(src.attribution, GEBCO_ATTRIBUTION);
  assert.ok(GEBCO_ATTRIBUTION.includes("GEBCO") && GEBCO_ATTRIBUTION.includes("doi:10.5285/"),
    "attribution must keep the exact form the WMS Abstract requests, incl. the DOI");
});

test("fallback source: NOAA exportImage speaks the same bbox token and carries its own attribution", () => {
  const src = oceanBasemapFallbackSource();
  assert.equal(src.type, "raster");
  const tpl = (src.tiles ?? [])[0] ?? "";
  assert.ok(tpl.includes("gis.ngdc.noaa.gov"), "NOAA NCEI ImageServer");
  assert.ok(tpl.includes("bbox={bbox-epsg-3857}") && tpl.includes("bboxSR=3857") && tpl.includes("imageSR=3857"),
    "ArcGIS exportImage takes the same {bbox-epsg-3857} substitution");
  assert.ok(tpl.includes("f=image"), "must return raw image bytes, not JSON");
  assert.equal(src.attribution, NOAA_HILLSHADE_ATTRIBUTION);
  assert.ok(NOAA_HILLSHADE_ATTRIBUTION.includes("NOAA"), "public-domain source still gets credit");
});

test("paint: every key exists in the installed spec's raster paint; opacity clamped", () => {
  const specKeys = new Set(Object.keys((v8 as unknown as { paint_raster: object }).paint_raster));
  for (const preset of [oceanBasemapPaint(), oceanBasemapPaint(35)]) {
    for (const k of Object.keys(preset)) {
      assert.ok(specKeys.has(k), `paint key ${k} must exist in maplibre's v8 paint_raster spec`);
    }
  }
  assert.equal((oceanBasemapPaint(35) as Record<string, unknown>)["raster-opacity"], 0.35);
  assert.equal((oceanBasemapPaint(250) as Record<string, unknown>)["raster-opacity"], 1, "clamped high");
  assert.equal((oceanBasemapPaint(-5) as Record<string, unknown>)["raster-opacity"], 0, "clamped low");
});

test("layer: raster layer bound to the ocean-basemap source id", () => {
  const layer = oceanBasemapLayer(80) as { id: string; type: string; source?: string; paint?: Record<string, unknown> };
  assert.equal(layer.id, OCEAN_BASEMAP_LAYER_ID);
  assert.equal(layer.type, "raster");
  assert.equal(layer.source, OCEAN_BASEMAP_SOURCE_ID);
  assert.equal(layer.paint?.["raster-opacity"], 0.8);
});

test("licensing register: recommended/fallback are commercial-OK; Esri stays excluded and unused", () => {
  const verdicts = Object.values(OCEAN_SOURCE_VERDICTS);
  assert.ok(verdicts.length >= 5, "all evaluated candidates stay on the record");
  for (const v of verdicts) {
    assert.ok(v.termsUrl.startsWith("https://"), `${v.name}: terms citation required`);
    if (v.role === "recommended" || v.role === "fallback") {
      assert.equal(v.commercialUse, "allowed",
        `${v.name}: the repo sells data access — only commercial-OK sources may ship`);
    }
  }
  const esri = OCEAN_SOURCE_VERDICTS.esriWorldOceanBase;
  assert.equal(esri.commercialUse, "forbidden");
  assert.equal(esri.role, "excluded");
  // and no exported config may reference an Esri endpoint
  const allTiles = [...(oceanBasemapSource().tiles ?? []), ...(oceanBasemapFallbackSource().tiles ?? [])];
  for (const t of allTiles) assert.ok(!/arcgisonline|arcgis\.com/i.test(t),
    "no Esri basemap endpoint may appear in a shipped config (noncommercial-only terms)");
});
