// Satellite FOLLOW — EARTH TWIN / ORBITAL O5 slice 1 (human directive
// 2026-07-15: "if you clicked on it, it would remain in focus"). Clicking a
// satellite locks the camera onto it: each worker tick re-centers the map on
// the object's fresh SGP4 position and moves a focus ring with it. Pure
// position math lives here (testable without a map); datamap owns the
// camera/ring wiring and the stop conditions (user drag takes control back,
// LOD hiding the layer stops the follow — an invisible object is never
// silently "followed").

import { SAT_STRIDE, readSatAt, mercatorToLonLat } from './satBuffer.js';

export interface FollowTarget {
  lonDeg: number;
  latDeg: number;
  altKm: number;
}

/**
 * The followed object's current ground position from the layer's packed
 * position buffer. null = no honest position this tick (sentinel slot —
 * deep-space/invalid — or a bad index): the caller must NOT move the
 * camera on a null, never guess.
 */
export function followTarget(
  buffer: ArrayLike<number> | null,
  index: number,
): FollowTarget | null {
  if (!buffer || index < 0 || (index + 1) * SAT_STRIDE > buffer.length) return null;
  const s = readSatAt(buffer, index);
  if (!s.valid) return null;
  const ll = mercatorToLonLat(s.mercX, s.mercY);
  if (!Number.isFinite(ll.lonDeg) || !Number.isFinite(ll.latDeg)) return null;
  return { lonDeg: ll.lonDeg, latDeg: ll.latDeg, altKm: s.altMeters / 1000 };
}

/** GeoJSON for the focus ring at a target (single source of truth for the
 *  selection visual — the ring is interaction chrome, not a data layer). */
export function focusRingFeatureCollection(t: FollowTarget | null): {
  type: 'FeatureCollection';
  features: Array<{ type: 'Feature'; geometry: { type: 'Point'; coordinates: [number, number] }; properties: Record<string, never> }>;
} {
  return {
    type: 'FeatureCollection',
    features: t
      ? [{ type: 'Feature', geometry: { type: 'Point', coordinates: [t.lonDeg, t.latDeg] }, properties: {} }]
      : [],
  };
}
