// moonSurface — PURE PERSPECTIVE SURFACE RENDER for the focused-close Moon
// (CELESTIAL v2 B4 "MOON FOR REAL", directive §4). When the camera skims the
// Moon (MIN_ZOOM_RADII = 1.05 radii, ~90 km altitude), the disc subtends
// thousands of px and only a PATCH of the surface fills the screen — the
// billboard disc-sprite (capped at ~448 px of texels across the whole
// hemisphere) cannot show that patch sharply. This module ray-casts the
// visible sphere patch in true perspective and samples the streamed LROC
// tile mosaic (base 8k fallback) per pixel, so real 100 m/px imagery lands
// where the eye is looking. `npx tsx --test`-able (no DOM, no network).
//
// ORIENTATION IS INHERITED, NOT REDEFINED: a surface normal's body longitude
// / latitude is computed in the SAME (node frame, W) factorisation the
// sprite path uses (textureSphere.buildSphereLUT + composeTexturedSprite,
// rotation.ts equatorNodeDirEclOfDate/primeMeridianDeg): lonNode =
// atan2(n·Y, n·X), lat = asin(n·Z), body lon = lonNode − W, with X = the
// equator node (world ecliptic), Z = the spin axis, Y = Z × X. The base 8k
// is sampled through textureSphere.equirectUV at exactly this body lon/lat,
// so the patch registers pixel-for-pixel with the far-tier disc — a feature
// (Tycho, Mare Imbrium) lands in the identical selenographic place. Pinned
// in moonSurface.test.ts.

import type { TexLike } from "./textureSphere.ts";
import { lambertWeight } from "./textureSphere.ts";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

const RAD = 180 / Math.PI;

/**
 * Nearest positive ray-sphere intersection parameter along `d` (need not be
 * unit), or −1 for a miss / behind. o = ray origin, c = sphere centre, R
 * radius. Returns t such that the hit point is o + t·d.
 */
export function raySphereNearT(o: Vec3, d: Vec3, c: Vec3, R: number): number {
  const ox = o.x - c.x;
  const oy = o.y - c.y;
  const oz = o.z - c.z;
  const A = d.x * d.x + d.y * d.y + d.z * d.z;
  const B = 2 * (ox * d.x + oy * d.y + oz * d.z);
  const C = ox * ox + oy * oy + oz * oz - R * R;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return -1;
  const sq = Math.sqrt(disc);
  const t0 = (-B - sq) / (2 * A);
  if (t0 > 0) return t0;
  const t1 = (-B + sq) / (2 * A);
  return t1 > 0 ? t1 : -1;
}

/**
 * Body longitude/latitude (degrees) of a surface normal, in the node-frame/W
 * factorisation shared with the sprite path. `n` unit (or near-unit); X, Y,
 * Z the orthonormal node frame (X = equator node, Z = spin axis, Y = Z × X),
 * wDeg the IAU prime-meridian angle. lonDeg = atan2(n·Y, n·X) − W.
 */
export function surfaceLonLat(
  n: Vec3,
  X: Vec3,
  Y: Vec3,
  Z: Vec3,
  wDeg: number,
): { lonDeg: number; latDeg: number } {
  const bx = n.x * X.x + n.y * X.y + n.z * X.z;
  const by = n.x * Y.x + n.y * Y.y + n.z * Y.z;
  let bz = n.x * Z.x + n.y * Z.y + n.z * Z.z;
  if (bz > 1) bz = 1;
  else if (bz < -1) bz = -1;
  return { lonDeg: Math.atan2(by, bx) * RAD - wDeg, latDeg: Math.asin(bz) * RAD };
}

/** A streamed high-res detail window over part of the sphere (from the tile
 *  mosaic). lon measured CONTINUOUSLY from lonMin so a seam-crossing window
 *  stays monotonic; latMax at the north edge. */
export interface DetailOverlay {
  tex: TexLike;
  lonMin: number;
  lonSpan: number;
  latMax: number;
  latSpan: number;
}

/** In-window nearest texel [r,g,b], or null when the point is outside the
 *  detail window (caller falls back to the base texture). */
export function sampleDetail(
  ov: DetailOverlay,
  lonDeg: number,
  latDeg: number,
): [number, number, number] | null {
  // unwrap lon into [0,360) measured from the window's west edge
  let d = (lonDeg - ov.lonMin) % 360;
  if (d < 0) d += 360;
  if (d > ov.lonSpan) return null;
  const dv = ov.latMax - latDeg;
  if (dv < 0 || dv > ov.latSpan) return null;
  const tex = ov.tex;
  const u = d / ov.lonSpan;
  const v = dv / ov.latSpan;
  let tx = (u * tex.width) | 0;
  let ty = (v * tex.height) | 0;
  if (tx >= tex.width) tx = tex.width - 1;
  if (ty >= tex.height) ty = tex.height - 1;
  const i = (ty * tex.width + tx) * 4;
  return [tex.data[i], tex.data[i + 1], tex.data[i + 2]];
}

/** Base equirect nearest texel [r,g,b] — the textureSphere.equirectUV
 *  convention (lon 0 at u = 0.5, east-increasing; v = 0 at the north pole),
 *  inlined for per-pixel speed. */
function sampleBase(tex: TexLike, lonDeg: number, latDeg: number): [number, number, number] {
  let u = lonDeg / 360 + 0.5;
  u -= Math.floor(u);
  let v = 0.5 - latDeg / 180;
  if (v < 0) v = 0;
  else if (v > 1) v = 1;
  let tx = (u * tex.width) | 0;
  let ty = (v * tex.height) | 0;
  if (tx >= tex.width) tx = tex.width - 1;
  if (ty >= tex.height) ty = tex.height - 1;
  const i = (ty * tex.width + tx) * 4;
  return [tex.data[i], tex.data[i + 1], tex.data[i + 2]];
}

/**
 * The view descriptor for one patch render. Buffer pixel (bx,by) maps to the
 * canvas point (originX + bx·stepX, originY + by·stepY); the camera pinhole
 * (cx,cy,k) + basis (r,u,f) turn that into a world ray from `cam`. Geometry
 * is the Moon sphere (center, radius); orientation is (X,Y,Z,wDeg); lighting
 * is Lambert against unit `sun` (toward the Sun, world frame).
 */
export interface MoonSurfaceView {
  bw: number;
  bh: number;
  originX: number;
  originY: number;
  stepX: number;
  stepY: number;
  cx: number;
  cy: number;
  k: number;
  r: Vec3;
  u: Vec3;
  f: Vec3;
  cam: Vec3;
  center: Vec3;
  radius: number;
  X: Vec3;
  Y: Vec3;
  Z: Vec3;
  wDeg: number;
  sun: Vec3;
}

/**
 * Fill RGBA rows [rowStart, rowEnd) of `out` (bw·bh·4) for the patch view.
 * Misses write alpha 0 (sky/other bodies show through at the limb). Chunkable
 * across macrotasks (the caller bounds rows per task to stay under the frame
 * budget). Returns the hit count in the rendered band (drive/eviction signal).
 */
export function renderMoonSurfaceRows(
  view: MoonSurfaceView,
  base: TexLike,
  detail: DetailOverlay | null,
  out: Uint8ClampedArray,
  rowStart: number,
  rowEnd: number,
): { hits: number } {
  const { bw, bh, originX, originY, stepX, stepY, cx, cy, k, r, u, f, cam, center, radius, X, Y, Z, wDeg, sun } = view;
  const y0 = Math.max(0, rowStart);
  const y1 = Math.min(bh, rowEnd);
  let hits = 0;
  for (let by = y0; by < y1; by++) {
    const py = originY + by * stepY;
    const b = (cy - py) / k;
    for (let bx = 0; bx < bw; bx++) {
      const o = (by * bw + bx) * 4;
      const px = originX + bx * stepX;
      const a = (px - cx) / k;
      // world ray = a·r + b·u + f (unnormalised is fine for raySphere)
      const dx = a * r.x + b * u.x + f.x;
      const dy = a * r.y + b * u.y + f.y;
      const dz = a * r.z + b * u.z + f.z;
      const t = raySphereNearT(cam, { x: dx, y: dy, z: dz }, center, radius);
      if (t < 0) {
        out[o + 3] = 0;
        continue;
      }
      // surface normal
      let nx = cam.x + t * dx - center.x;
      let ny = cam.y + t * dy - center.y;
      let nz = cam.z + t * dz - center.z;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl;
      ny /= nl;
      nz /= nl;
      const n = { x: nx, y: ny, z: nz };
      const { lonDeg, latDeg } = surfaceLonLat(n, X, Y, Z, wDeg);
      const rgb = (detail && sampleDetail(detail, lonDeg, latDeg)) || sampleBase(base, lonDeg, latDeg);
      const lit = nx * sun.x + ny * sun.y + nz * sun.z;
      const w = lambertWeight(lit);
      out[o] = rgb[0] * w;
      out[o + 1] = rgb[1] * w;
      out[o + 2] = rgb[2] * w;
      out[o + 3] = 255;
      hits++;
    }
  }
  return { hits };
}

/** Total buffer pixels — chunk-planning helper (rows per UPGRADE_CHUNK_PX). */
export function surfacePixelCount(view: MoonSurfaceView): number {
  return view.bw * view.bh;
}
