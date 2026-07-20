// ELEVATION UNITS FOR CUSTOM-LAYER PROJECTION (probe-proven 2026-07-20):
// MapLibre's custom-layer projection matrices consume DIFFERENT elevation
// units per projection phase —
//   · full GLOBE (u_projection_transition > 0.999): METERS (the globe
//     prelude scales the sphere by 1 + ele/GLOBE_RADIUS);
//   · MERCATOR (the flat-map projection, and the globe's mercator phase):
//     MERCATOR-UNIT z — the matrix multiplies vec4(x, y, z, 1) raw, and
//     meters explode w (measured live: z = 10,668 m → w ≈ −1.84e9 while
//     z = 0 matched map.project to the pixel). This is why the 3D flight
//     trail rendered as a giant smeared sheet on flat maps while the globe
//     was perfect. The CPU pick paths already split exactly this way
//     (pickNearestAircraftScreen: meters+sphere / …ScreenMercator:
//     mercatorZFromAltitude) — this file gives the SHADERS the same split.
//
// TS ground truth stays single-sourced in orbital/occlusion.ts
// (mercatorZFromAltitude, empirically pinned); the GLSL below is its exact
// shader mirror via the identity cos(gd(π(1−2y))) = sech(π(1−2y)).

export { mercatorZFromAltitude } from "./orbital/occlusion";

/** 2π × MapLibre's GLOBE_RADIUS (6371008.8 m) — the mercator world
 *  circumference the z conversion divides by. */
export const EARTH_CIRCUMFERENCE_M = 2 * Math.PI * 6371008.8;

/**
 * GLSL: `float vtProjElev(float eleMeters, float mercY)` — the elevation
 * value projectTileFor3D actually needs in the CURRENT projection phase.
 * Paste AFTER the MapLibre prelude (PI comes from it; the transition
 * uniform exists only in the globe prelude, hence the #ifdef).
 */
export const VT_PROJ_ELEV_GLSL = `
float vtProjElev(float eleMeters, float mercY) {
#ifdef GLOBE
  if (u_projection_transition > 0.999) return eleMeters; // globe wants meters
#endif
  // mercator matrices consume mercator-unit z: meters / (2πR·cos(lat)),
  // with cos(lat) = sech(π(1 − 2y)) — the exact CPU mercatorZFromAltitude
  return eleMeters * cosh(PI * (1.0 - 2.0 * mercY)) * ${(1 / EARTH_CIRCUMFERENCE_M).toExponential(10)};
}
`;
