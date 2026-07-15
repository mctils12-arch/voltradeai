/**
 * GEBCO WMS raster-tile factory (earth_twin_program.md Phase E2 v2 —
 * "seafloor_confidence" overlay).
 *
 * The charter originally scoped E2 v2 as a self-tiled GEBCO pmtiles
 * pipeline (BUILD-DON'T-BUY: download the full 15-arc-second global grid,
 * tile it ourselves). Live probing this session found a lighter, equally
 * honest path: GEBCO/BODC already serve the exact confidence layer the
 * charter wants — GEBCO_2024_3, "those pixels that are based on measured
 * data ... areas based on interpolation are set to black" (verified live
 * via GetCapabilities, 2026-07-15) — over their own public WMS. Proxying
 * their live tiles (same pattern this file's sibling gibs.ts already uses
 * for NASA GIBS) ships the honesty overlay today with zero storage burden
 * and zero large-download risk, instead of deferring it behind a
 * multi-GB self-tiling pipeline. The self-tiled full-fidelity relief (E2
 * v2's other half, real 15-arc-second bathymetry) remains a distinct,
 * larger future slice — this only covers the confidence grid.
 *
 * MapLibre GL's raster source spec supports a `{bbox-epsg-3857}` template
 * token specifically for wiring WMS GetMap servers as XYZ-style tile
 * sources (each tile request substitutes the tile's own EPSG:3857 bbox).
 */

export const GEBCO_WMS_BASE = "https://wms.gebco.net/2024/mapserv";

/** GEBCO_2024_3: per-pixel Source Data Type Identifier rendered as a binary
 *  mask — measured (ship/sonar soundings) vs interpolated (satellite-gravity
 *  derived). This is GEBCO's own layer, not a derived product we computed. */
export const GEBCO_TID_LAYER = "GEBCO_2024_3";

const TILE_SIZE = 256;

/** Build the WMS GetMap tile URL template for a GEBCO layer. Returns a
 * template string containing the literal `{bbox-epsg-3857}` token — MapLibre
 * substitutes it per-tile; this function performs no per-tile work itself. */
export function gebcoWmsTileUrl(layer: string): string {
  const params = new URLSearchParams({
    service: "WMS",
    version: "1.3.0",
    request: "GetMap",
    layers: layer,
    format: "image/png",
    transparent: "true",
    width: String(TILE_SIZE),
    height: String(TILE_SIZE),
    crs: "EPSG:3857",
  });
  return `${GEBCO_WMS_BASE}?${params.toString()}&bbox={bbox-epsg-3857}`;
}
