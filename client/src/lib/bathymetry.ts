// Bathymetry depth palette — EARTH TWIN E2-1 ("drain the ocean" v1,
// research/earth_twin_program.md V4).
//
// ONE SOURCE OF TRUTH: the map's color-relief ramp AND the panel legend both
// render from BATHYMETRY_STOPS (the iconDataURL legend principle applied to a
// color ramp) — the palette can never drift between the map and its legend.
//
// The ramp is designed for an OVERLAY over satellite imagery: ocean depths get
// a hypsometric tint (deep = dark navy → shallow = pale teal) and everything
// at or above sea level is FULLY TRANSPARENT, so land imagery shows through
// untouched. "Draining the ocean" = toggling this layer on.
//
// HONESTY: the v1 DEM is NOAA ETOPO1 (via the open Terrain Tiles bucket) at
// ~1 arc-minute — a real published dataset that blends ship soundings with
// satellite-gravity interpolation. Depths are indicative, never navigational.
// The charter's E2 v2 upgrade (GEBCO 15-arcsec + its per-cell TID source
// grid) adds the measured-vs-interpolated confidence overlay.

export interface DepthStop {
  /** stop elevation in meters (negative = below sea level). */
  elevM: number;
  color: string;
  /** legend label for the zone this stop anchors. */
  label: string;
  /** representative depth (m, positive) for unit-formatted legend text. */
  depthM: number;
}

/** Deep → shallow. Elevations must ascend for the interpolate expression. */
export const BATHYMETRY_STOPS: DepthStop[] = [
  { elevM: -11000, color: "#04102f", label: "Hadal", depthM: 11000 },
  { elevM: -6000, color: "#0a2a5e", label: "Abyssal", depthM: 6000 },
  { elevM: -3000, color: "#155e9e", label: "Bathyal", depthM: 3000 },
  { elevM: -1000, color: "#2b8fc9", label: "Slope", depthM: 1000 },
  { elevM: -200, color: "#52c5d9", label: "Shelf", depthM: 200 },
  { elevM: -1, color: "#7fe0d3", label: "Coastal", depthM: 0 },
];

/** Land (>= 0 m) is fully transparent so imagery shows through. */
export const BATHYMETRY_LAND_TRANSPARENT = "rgba(127,224,211,0)";

/**
 * MapLibre `color-relief-color` expression: interpolate over ["elevation"]
 * through the depth stops, snapping to transparent at sea level (the last
 * stop extends to all land elevations).
 */
export function bathymetryColorRelief(): unknown[] {
  const expr: unknown[] = ["interpolate", ["linear"], ["elevation"]];
  for (const s of BATHYMETRY_STOPS) expr.push(s.elevM, s.color);
  expr.push(0, BATHYMETRY_LAND_TRANSPARENT);
  return expr;
}
