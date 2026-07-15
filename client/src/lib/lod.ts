// LOD director — EARTH TWIN A1 (research/earth_twin_program.md).
//
// Pure camera-altitude → layer-visibility math. The registry (layers.json,
// REGISTRY v2) gives a layer an optional `lod` camera-altitude envelope in
// km; this module turns the map's camera state into a 0..1 opacity for that
// envelope. The map page owns the wiring (listen on move, apply opacity,
// pause/resume workers); this module owns ONLY the math, so it is testable
// without MapLibre, WebGL, or a DOM.
//
// HONESTY RAILS (charter A1): LOD is a render choice, always reversible by
// zoom, never a storage choice. An envelope that hides a layer must surface
// that state on-panel ("hidden at this zoom (LOD)") — the CALLER's duty;
// this module makes the state unambiguous (opacity === 0) so the caller
// can't miss it. Fades are soft bands, never pops.
//
// CAMERA-ALTITUDE MATH (verified against installed maplibre-gl 5.24.0 —
// dist getCameraAltitude(): cos(pitch)·cameraToCenterDistance/pixelPerMeter
// + centerElevation, where cameraToCenterDistance = 0.5·height/tan(fov/2)
// px and meters/px at the center latitude is 78271.51696·cos(lat)/2^zoom —
// MapLibre worlds are 512px-tile based, so the constant is EARTH
// CIRCUMFERENCE / 512, NOT the 256px-web-mercator 156543).

/** Camera-altitude visibility envelope, in km (registry v2 `lod` block). */
export interface LodEnvelope {
  /** Below this camera altitude the layer is fully hidden. */
  camMinKm?: number;
  /** Above this camera altitude the layer is fully hidden. */
  camMaxKm?: number;
  /** Soft-fade width applied inside each bound (default 0 = hard step). */
  fadeBandKm?: number;
}

export const EARTH_CIRCUMFERENCE_M = 40075016.686;
/** meters/px at zoom 0, equator, for MapLibre's 512px world tiles. */
export const METERS_PER_PIXEL_Z0 = EARTH_CIRCUMFERENCE_M / 512; // 78271.51696...
export const DEFAULT_FOV_DEG = 36.87; // MapLibre default vertical field of view

/** Ground meters per screen pixel at a latitude/zoom (512px-tile worlds). */
export function metersPerPixel(latDeg: number, zoom: number): number {
  const lat = Math.max(-89.999, Math.min(89.999, latDeg));
  return (METERS_PER_PIXEL_Z0 * Math.cos((lat * Math.PI) / 180)) / Math.pow(2, zoom);
}

export interface CameraAltitudeInput {
  zoom: number;
  latDeg: number;
  canvasHeightPx: number;
  /** vertical field of view in degrees (MapLibre default 36.87). */
  fovDeg?: number;
  pitchDeg?: number;
  /** terrain elevation at the map center in meters (0 without terrain). */
  centerElevationM?: number;
}

/**
 * Camera altitude above sea level in meters — the same quantity MapLibre's
 * transform.getCameraAltitude() reports, computed from public map state so
 * it works as a fallback when that (formally internal) API is unavailable.
 */
export function cameraAltitudeMeters(input: CameraAltitudeInput): number {
  const fovDeg = input.fovDeg ?? DEFAULT_FOV_DEG;
  const pitchRad = ((input.pitchDeg ?? 0) * Math.PI) / 180;
  const halfFovRad = (fovDeg * Math.PI) / 360;
  const cameraToCenterPx = (0.5 * input.canvasHeightPx) / Math.tan(halfFovRad);
  const altitude =
    Math.cos(pitchRad) * cameraToCenterPx * metersPerPixel(input.latDeg, input.zoom);
  return altitude + (input.centerElevationM ?? 0);
}

/** Duck-typed subset of a MapLibre map this module reads (keeps it testable). */
export interface MapLike {
  transform?: { getCameraAltitude?: () => number };
  getZoom(): number;
  getCenter(): { lat: number };
  getCanvas(): { height: number };
  getPitch?: () => number;
  getVerticalFieldOfView?: () => number;
}

/**
 * Camera altitude in km from a live map. Prefers MapLibre's own
 * transform.getCameraAltitude() (exact, terrain-aware, globe-aware —
 * present and typed in the installed v5.24.0); falls back to the pure
 * formula from public getters if that ever disappears. Returns null only
 * if both paths throw (caller should then NOT gate anything — fail open,
 * never hide a layer on broken math).
 */
export function cameraAltitudeKmFromMap(map: MapLike): number | null {
  try {
    const alt = map.transform?.getCameraAltitude?.();
    if (typeof alt === "number" && Number.isFinite(alt)) return alt / 1000;
  } catch {
    /* fall through to the pure formula */
  }
  try {
    const m = cameraAltitudeMeters({
      zoom: map.getZoom(),
      latDeg: map.getCenter().lat,
      canvasHeightPx: map.getCanvas().height,
      fovDeg: map.getVerticalFieldOfView?.(),
      pitchDeg: map.getPitch?.(),
    });
    return Number.isFinite(m) ? m / 1000 : null;
  } catch {
    return null;
  }
}

/**
 * Envelope opacity at a camera altitude: 1 fully visible, 0 fully hidden,
 * linear fade across fadeBandKm INSIDE each bound. No envelope (or no
 * bounds) → always 1. A null camera altitude (broken math) → 1: fail OPEN —
 * LOD may never hide data because a measurement failed.
 */
export function lodOpacity(
  env: LodEnvelope | null | undefined,
  camAltKm: number | null,
): number {
  if (!env || camAltKm == null || !Number.isFinite(camAltKm)) return 1;
  const fade = Math.max(0, env.fadeBandKm ?? 0);
  let opacity = 1;
  if (typeof env.camMinKm === "number") {
    if (camAltKm <= env.camMinKm) opacity = 0;
    else if (fade > 0 && camAltKm < env.camMinKm + fade) {
      opacity = Math.min(opacity, (camAltKm - env.camMinKm) / fade);
    }
  }
  if (typeof env.camMaxKm === "number") {
    if (camAltKm >= env.camMaxKm) opacity = 0;
    else if (fade > 0 && camAltKm > env.camMaxKm - fade) {
      opacity = Math.min(opacity, (env.camMaxKm - camAltKm) / fade);
    }
  }
  return Math.max(0, Math.min(1, opacity));
}
