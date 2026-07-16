// CPU nearest-point picking for the orbital satellite layer (ORBITAL program
// O3: research/orbital_program.md, "O3 picking" recipe).
//
// SatLayer is a custom MapLibre CustomLayerInterface (raw WebGL points) with
// no queryRenderedFeatures hit-testing (see satLayer.ts's PICKING note). A
// click resolves to a satellite by nearest-distance search over the layer's
// own position buffer, which is index-aligned to the GP array the worker was
// initialized with (satWorker.ts's INDEX ALIGNMENT contract: buffer slot i
// always corresponds to gp[i], including skipped/sentinel slots).
//
// Pure + hermetic: no DOM/WebGL/map dependency, so it is directly unit
// testable (`npx tsx --test`).

import { SENTINEL_SKIP } from './satBuffer.js';
import { earthOccludes, mercatorToSphere, type Vec3 } from './occlusion.js';
import type { GpRecord } from './tle.js';

export interface PickResult {
  index: number;
  gp: GpRecord;
  /** altitude in meters, as packed by the worker (see satBuffer.ts stride layout). */
  altMeters: number;
  /** 0=LEO, 1=MEO, 2=GEO (packed classCode; sentinel slots are never returned). */
  classCode: number;
}

/**
 * Find the nearest RENDERED satellite (classCode !== SENTINEL_SKIP) to
 * (clickX, clickY) in normalized web-mercator [0..1] space, within
 * `toleranceUnits`. Returns null for an honest "no hit" — a click far from
 * every point never snaps to the globally nearest object, which would make
 * every click somewhere on the globe register a hit regardless of distance.
 *
 * `gp` must be the SAME array (same order/length) that was sent to the
 * worker's `init` message — the index alignment is the caller's contract to
 * keep, not something this function can verify.
 *
 * `cameraSphere` (SatLayer.getGlobeCamera()): when non-null, satellites the
 * globe's far-side cull has hidden (behind the earth from this camera; see
 * ./occlusion) are excluded — a click near the limb must never select an
 * invisible object. Null (mercator view / mid-transition) skips the filter,
 * matching the shader, which only culls in full globe mode.
 */
export function pickNearestSatellite(
  positions: ArrayLike<number>,
  stride: number,
  gp: GpRecord[],
  clickX: number,
  clickY: number,
  toleranceUnits: number,
  cameraSphere?: Vec3 | null,
): PickResult | null {
  const total = Math.min(gp.length, Math.floor(positions.length / stride));
  const tol2 = toleranceUnits * toleranceUnits;
  let bestI = -1;
  let bestD2 = tol2;
  for (let i = 0; i < total; i++) {
    const base = i * stride;
    const classCode = positions[base + 3];
    if (classCode === SENTINEL_SKIP) continue; // not rendered — never pickable
    const dx = positions[base] - clickX;
    const dy = positions[base + 1] - clickY;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) {
      if (
        cameraSphere &&
        earthOccludes(
          cameraSphere,
          mercatorToSphere(positions[base], positions[base + 1], positions[base + 2]),
        )
      ) {
        continue; // hidden behind the earth — not rendered, so not pickable
      }
      bestD2 = d2;
      bestI = i;
    }
  }
  if (bestI < 0) return null;
  const base = bestI * stride;
  return {
    index: bestI,
    gp: gp[bestI],
    altMeters: positions[base + 2],
    classCode: positions[base + 3],
  };
}

/**
 * Convert a screen-pixel click tolerance to normalized web-mercator units at
 * a given map zoom (web-mercator tile world is 256px at zoom 0, doubling per
 * zoom level — the standard slippy-map relation).
 */
export function pixelToleranceToMercUnits(pixels: number, zoom: number): number {
  return pixels / (256 * Math.pow(2, zoom));
}

/**
 * O6 pick fix — SCREEN-SPACE picking for full globe mode. A satellite
 * renders displaced from its ground point along the sphere normal by its
 * altitude (projectTileFor3D); at MEO that displacement is enormous, so
 * ground-mercator picking selected whatever LEO object's nadir happened to
 * sit under the cursor. Here every candidate is projected with the SAME
 * matrix the shader used this frame (SatLayer.getGlobeProjection) via the
 * same CPU sphere math the far-side cull uses (occlusion.mercatorToSphere)
 * — the pick agrees with the pixels.
 *
 * clickX/clickY and width/height are CSS pixels (map event .point + canvas
 * client size — the projection is resolution-independent in NDC).
 */
export function pickNearestSatelliteScreen(
  positions: ArrayLike<number>,
  stride: number,
  gp: GpRecord[],
  matrix: ArrayLike<number>, // column-major mat4, the frame's mainMatrix
  clickX: number,
  clickY: number,
  width: number,
  height: number,
  tolerancePx: number,
  cameraSphere?: Vec3 | null,
): PickResult | null {
  const total = Math.min(gp.length, Math.floor(positions.length / stride));
  const tol2 = tolerancePx * tolerancePx;
  let bestI = -1;
  let bestD2 = tol2;
  for (let i = 0; i < total; i++) {
    const base = i * stride;
    if (positions[base + 3] === SENTINEL_SKIP) continue; // not rendered
    const p = mercatorToSphere(positions[base], positions[base + 1], positions[base + 2]);
    // clip = M * (p, 1)  (column-major)
    const w = matrix[3] * p[0] + matrix[7] * p[1] + matrix[11] * p[2] + matrix[15];
    if (!(w > 0)) continue; // behind the camera
    const cx = (matrix[0] * p[0] + matrix[4] * p[1] + matrix[8] * p[2] + matrix[12]) / w;
    const cy = (matrix[1] * p[0] + matrix[5] * p[1] + matrix[9] * p[2] + matrix[13]) / w;
    const sx = ((cx + 1) / 2) * width;
    const sy = ((1 - cy) / 2) * height;
    const dx = sx - clickX;
    const dy = sy - clickY;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) {
      if (cameraSphere && earthOccludes(cameraSphere, p)) continue; // hidden — not pickable
      bestD2 = d2;
      bestI = i;
    }
  }
  if (bestI < 0) return null;
  const base = bestI * stride;
  return {
    index: bestI,
    gp: gp[bestI],
    altMeters: positions[base + 2],
    classCode: positions[base + 3],
  };
}
