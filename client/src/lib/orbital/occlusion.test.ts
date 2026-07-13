// Tests for the earth-occlusion (far-side) culling math. These are ALSO the
// tests for the GPU cull in satLayer.ts: the vertex shader inlines the exact
// same formulas (GLSL can't import TS), and satLayer.test.ts pins the shader
// source to keep the two in sync.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  GLOBE_RADIUS_M,
  OCCLUSION_RADIUS,
  cameraFromClippingPlane,
  earthOccludes,
  mercatorToSphere,
} from './occlusion.js';
import { lonLatToMercator } from './satBuffer.js';

const approx = (a: number, b: number, eps = 1e-9) =>
  assert.ok(Math.abs(a - b) < eps, `expected ${a} ≈ ${b}`);

// ── mercatorToSphere: mirrors the globe prelude's projectToSphere frame ──

test('mercatorToSphere: lon 0, lat 0, sea level → (0,0,1) unit sphere', () => {
  const m = lonLatToMercator(0, 0);
  const p = mercatorToSphere(m.x, m.y, 0);
  approx(p[0], 0);
  approx(p[1], 0);
  approx(p[2], 1);
});

test('mercatorToSphere: 90°E equator → +x axis; 90°W → -x axis', () => {
  const e = mercatorToSphere(lonLatToMercator(90, 0).x, 0.5, 0);
  approx(e[0], 1);
  approx(e[2], 0, 1e-9);
  const w = mercatorToSphere(lonLatToMercator(-90, 0).x, 0.5, 0);
  approx(w[0], -1);
});

test('mercatorToSphere: northern latitude → positive y (north = +y axis)', () => {
  const m = lonLatToMercator(0, 45);
  const p = mercatorToSphere(m.x, m.y, 0);
  assert.ok(p[1] > 0.5, `expected northern point to have y>0.5, got ${p[1]}`);
  approx(Math.hypot(p[0], p[1], p[2]), 1, 1e-9);
});

test('mercatorToSphere: altitude scales the radius by 1 + alt/GLOBE_RADIUS', () => {
  const m = lonLatToMercator(0, 0);
  const p = mercatorToSphere(m.x, m.y, GLOBE_RADIUS_M); // one earth radius up
  approx(Math.hypot(p[0], p[1], p[2]), 2, 1e-9);
});

// ── cameraFromClippingPlane: C = plane.xyz · (-1/plane.w) ──

test('cameraFromClippingPlane: plane (0,0,1,-1/3) → camera at (0,0,3)', () => {
  const c = cameraFromClippingPlane([0, 0, 1, -1 / 3]);
  assert.ok(c);
  approx(c[0], 0);
  approx(c[1], 0);
  approx(c[2], 3);
});

test('cameraFromClippingPlane: degenerate planes (w=0, w>0, NaN) → null', () => {
  assert.equal(cameraFromClippingPlane([0, 0, 1, 0]), null);
  assert.equal(cameraFromClippingPlane([0, 0, 1, 0.5]), null);
  assert.equal(cameraFromClippingPlane([0, 0, 1, NaN]), null);
});

// ── earthOccludes: exact segment–sphere test ──

const CAM: readonly [number, number, number] = [0, 0, 3];

test('near-side LEO satellite (between camera and earth) is visible', () => {
  assert.equal(earthOccludes(CAM, [0, 0, 1.1]), false);
});

test('far-side LEO satellite (directly behind the disk) is occluded', () => {
  assert.equal(earthOccludes(CAM, [0, 0, -1.1]), true);
});

test('GEO satellite directly behind the earth is occluded', () => {
  assert.equal(earthOccludes(CAM, [0, 0, -6.6]), true);
});

test('GEO satellite past the limb but off to the side is VISIBLE (hemisphere test would wrongly hide it)', () => {
  // sub-satellite point is on the far hemisphere (z<0), yet the sightline
  // clears the disk — the case that makes exact occlusion, not a hemisphere
  // dot-product, the required test.
  assert.equal(earthOccludes(CAM, [6.6, 0, -1]), false);
});

test('satellite beside the earth (sightline misses the sphere) is visible', () => {
  assert.equal(earthOccludes(CAM, [2, 0, 0]), false);
});

test('limb-grazing sightline inside the bias radius stays visible', () => {
  // Construct a true tangent ray from CAM=(0,0,3): tangent point T on the
  // unit sphere has z=1/d=1/3, x=sqrt(1-1/9). A point past T along that ray
  // has closest-approach EXACTLY 1 > OCCLUSION_RADIUS → not culled (the
  // 0.999 bias exists so limb points don't flicker on float noise).
  assert.ok(OCCLUSION_RADIUS < 1);
  const tz = 1 / 3;
  const tx = Math.sqrt(1 - tz * tz);
  // P = CAM + 2·(T − CAM) — beyond the tangent point, on the far hemisphere
  const p: readonly [number, number, number] = [2 * tx, 0, 2 * tz - 3];
  assert.ok(p[2] < 0, 'test point is on the far hemisphere');
  assert.equal(earthOccludes(CAM, p), false);
});

test('degenerate zero-length segment is visible (no self-occlusion NaN)', () => {
  assert.equal(earthOccludes(CAM, [0, 0, 3]), false);
});

test('camera pulled far out still occludes only the true shadow cone', () => {
  const far: readonly [number, number, number] = [0, 0, 50];
  assert.equal(earthOccludes(far, [0, 0, -1.05]), true); // behind the disk
  assert.equal(earthOccludes(far, [1.2, 0, -1.05]), false); // beside it
});
