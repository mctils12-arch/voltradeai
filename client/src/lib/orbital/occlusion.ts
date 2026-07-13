// Earth-occlusion (far-side) culling math for the orbital satellite layer.
//
// PROBLEM: MapLibre's globe prelude clips far-side geometry for SURFACE
// features (projectTile → interpolateProjection applies globeComputeClippingZ),
// but the 3D path the satellite layer uses (projectTileFor3D →
// interpolateProjectionFor3D) applies NO far-side clipping at all — verified
// against maplibre-gl v5's generated projection shaders. Result: satellites
// physically behind the earth render on top of the globe.
//
// A plain hemisphere test is WRONG for satellites: a GEO object at ~6.6 earth
// radii whose sub-satellite point is past the limb can still be genuinely
// visible around the side of the disk. The correct test is exact sphere
// occlusion: the satellite is hidden iff the segment from the camera to the
// satellite intersects the earth sphere.
//
// CAMERA RECONSTRUCTION: MapLibre's u_projection_clipping_plane is (verified
// against GlobeTransform._computeClippingPlane, maplibre-gl v5):
//   plane.xyz = unit vector from globe center TOWARD the camera
//               (built from a normalized 2-vector, then pure rotations —
//                length preserved; re-normalized with scale ≈ 1)
//   plane.w   = -(radius² / distanceCameraToCenter) = -1/d  (unit sphere)
// so the camera position in unit-sphere space is C = plane.xyz · (-1/plane.w).
//
// This module is the SINGLE SOURCE OF TRUTH for that math on the CPU side:
// the vertex shader in ./satLayer inlines the identical formulas (GLSL can't
// import TS), and ./pick uses these functions so click-picking never selects
// a satellite the cull has hidden. Pure + hermetic: unit-tested directly.

/** MapLibre's globe shader earth radius (meters) — must match `#define GLOBE_RADIUS`. */
export const GLOBE_RADIUS_M = 6371008.8;

/**
 * Occlusion sphere radius (unit-sphere space), slightly under 1 so satellites
 * grazing the exact limb stay visible instead of flickering on float noise.
 * The GPU cull in satLayer.ts uses the same constant (inlined as its square).
 */
export const OCCLUSION_RADIUS = 0.999;

export type Vec3 = readonly [number, number, number];

/**
 * Normalized web-mercator [0..1] x/y + altitude (meters) → position in
 * MapLibre's unit-sphere globe frame. EXACT replica of the globe prelude's
 * projectToSphere() (y = north pole axis, longitude offset +π) scaled by
 * (1 + alt/GLOBE_RADIUS), so CPU tests mirror what the GPU renders.
 */
export function mercatorToSphere(mercX: number, mercY: number, altMeters: number): Vec3 {
  const sphericalX = mercX * Math.PI * 2 + Math.PI;
  const sphericalY = 2 * Math.atan(Math.exp(Math.PI - mercY * Math.PI * 2)) - Math.PI * 0.5;
  const len = Math.cos(sphericalY);
  const r = 1 + altMeters / GLOBE_RADIUS_M;
  return [
    Math.sin(sphericalX) * len * r,
    Math.sin(sphericalY) * r,
    Math.cos(sphericalX) * len * r,
  ];
}

/**
 * Camera position in unit-sphere space from MapLibre's globe clipping plane.
 * Returns null when the plane is degenerate (w must be strictly negative:
 * w = -1/d for a camera at finite positive distance d) — callers treat null
 * as "no occlusion information; don't cull".
 */
export function cameraFromClippingPlane(
  plane: ArrayLike<number>,
): Vec3 | null {
  const w = plane[3];
  if (!(w < 0) || !isFinite(w)) return null;
  const d = -1 / w;
  return [plane[0] * d, plane[1] * d, plane[2] * d];
}

/**
 * True when the earth (sphere of OCCLUSION_RADIUS at the origin) blocks the
 * straight segment from camera `cam` to point `p` — i.e. the satellite at `p`
 * is hidden behind the planet from this viewpoint. Exact segment–sphere test:
 * closest approach of the segment to the origin, checked to lie strictly
 * between the endpoints.
 */
export function earthOccludes(cam: Vec3, p: Vec3): boolean {
  const vx = p[0] - cam[0];
  const vy = p[1] - cam[1];
  const vz = p[2] - cam[2];
  const vv = vx * vx + vy * vy + vz * vz;
  if (vv === 0) return false;
  const t = -(cam[0] * vx + cam[1] * vy + cam[2] * vz) / vv;
  if (t <= 0 || t >= 1) return false; // closest approach outside the segment
  const cx = cam[0] + t * vx;
  const cy = cam[1] + t * vy;
  const cz = cam[2] + t * vz;
  return cx * cx + cy * cy + cz * cz < OCCLUSION_RADIUS * OCCLUSION_RADIUS;
}
