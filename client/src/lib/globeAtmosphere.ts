// Globe atmosphere / sky presets — EARTH TWIN E2 quality pass (research
// half, 2026-07-16; wiring recipe in research/ocean_quality_notes.md).
//
// Everything here was verified against the INSTALLED maplibre-gl@5.24.0,
// not docs-from-memory:
//
// 1. node_modules/@maplibre/maplibre-gl-style-spec/dist/index.d.ts:473
//    `type SkySpecification` — the ONLY sky/atmosphere fields v5.24 accepts:
//    "sky-color", "horizon-color", "fog-color", "fog-ground-blend",
//    "horizon-fog-blend", "sky-horizon-blend", "atmosphere-blend"
//    (each also has a "-transition" variant). All are
//    PropertyValueSpecification — zoom-interpolated expressions are legal,
//    and the spec note on atmosphere-blend says "It is best to interpolate
//    this expression when using globe projection."
// 2. node_modules/maplibre-gl/dist/maplibre-gl.d.ts:12956 `setSky(sky:
//    SkySpecification, options?: StyleSetterOptions)`; :5152-5199 the Sky
//    class (atmosphereMesh + calculateFogBlendOpacity — fog only fades IN
//    above ~pitch 60, invisible below).
// 3. dist/maplibre-gl.js (the shipped shaders — read this session):
//    - The GLOBE LIMB is a physical Rayleigh/Mie scattering shader whose
//      uniforms are ONLY u_sun_pos, u_globe_position, u_globe_radius,
//      u_atmosphere_blend, u_inv_proj_matrix. The scattering coefficients
//      (vec3(5.5e-6, 13.0e-6, 22.4e-6) — Earth-blue) are HARDCODED:
//      sky-color/horizon-color/fog-color have ZERO effect on the limb glow.
//      Its intensity = atmosphere-blend × projectionTransition (auto-fades
//      as the globe morphs to mercator; skipped entirely at 0).
//    - u_sun_pos is derived from the style LIGHT (light.position, honoring
//      anchor "map") — map.setLight() aims the limb scattering. With the
//      default viewport-anchored light the glow tracks the camera (always
//      lit); the O6-7 terminator's real sun azimuth could drive it instead.
//    - The sky-color/horizon-color gradient is a separate screen-space pass
//      whose opacity is (1 − projectionTransition): it paints the MERCATOR
//      horizon/sky and is fully transparent on the finished globe. So one
//      combined setSky is safe: limb fields rule zoomed-out, horizon fields
//      take over as the projection transitions and the camera pitches.
//
// Honesty note: the limb glow is a hardcoded-physics render of Earth's
// atmosphere, not data; presets only choose intensity and the mercator-side
// sky colors. No invented fields — every key below exists in the installed
// SkySpecification.

import type { SkySpecification } from "maplibre-gl";

/**
 * Zoomed-out globe: Google-Earth-style thin blue atmosphere rim against
 * space. Full physical scattering at hero-globe altitudes, fading out by
 * z7 where the library is already morphing toward mercator (the shader
 * additionally multiplies by projectionTransition, so the fade is doubly
 * smooth and the pass is skipped once it hits 0 — free perf at street
 * zooms). Exactly the interpolation pattern the v5.24 spec recommends.
 */
export const GLOBE_LIMB_SKY: SkySpecification = {
  "atmosphere-blend": [
    "interpolate",
    ["linear"],
    ["zoom"],
    0, 1,
    5, 1,
    7, 0,
  ],
};

/**
 * Tilted-terrain horizon (3D relief + pitch ≥ ~60°, where v5.24 actually
 * draws fog — calculateFogBlendOpacity): deep-space blue overhead falling
 * to a pale atmospheric band at the horizon, fog seated onto the terrain so
 * distant relief recedes instead of ending in a hard edge. Colors are
 * presentation (dark design system), not data.
 */
export const TERRAIN_HORIZON_SKY: SkySpecification = {
  "sky-color": "#0a1a33",
  "horizon-color": "#b9d4ee",
  "fog-color": "#8fb3d6",
  "sky-horizon-blend": 0.9,
  "horizon-fog-blend": 0.85,
  "fog-ground-blend": 0.6,
};

/**
 * Recommended single call for datamap: the union of both presets. Safe
 * because the two shader passes are disjoint (limb = globe-side, sky/fog =
 * mercator/pitch-side; see header §3) — no field of one affects the other.
 * `map.setSky(DEFAULT_SKY)` once after style load covers hero globe,
 * zoom-in transition, and tilted-terrain work.
 */
export const DEFAULT_SKY: SkySpecification = {
  ...GLOBE_LIMB_SKY,
  ...TERRAIN_HORIZON_SKY,
};

/** Zoom band where the limb glow is (fully or partially) visible — for the
 *  parent's wiring/tests; mirrors GLOBE_LIMB_SKY's interpolation stops. */
export const LIMB_VISIBLE_ZOOM_MAX = 7;
