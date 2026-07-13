// Pins the satellite vertex shader's far-side cull to the CPU mirror in
// ./occlusion. GLSL can't import TS, so satLayer.ts INLINES earthOccludes /
// cameraFromClippingPlane — these tests are the sync mechanism: if someone
// edits the shader math or the occlusion constants, one side fails here.
//
// The shader itself can't execute under node (no WebGL); occlusion.test.ts
// exercises the identical math numerically. Here we assert the structural
// contract of the generated source: the cull exists, uses the same formulas
// and constants, is guarded to the globe variant, and never runs mid
// projection transition.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { VERT_SRC } from './satLayer.js';
import { OCCLUSION_RADIUS } from './occlusion.js';

const PRELUDE = '/* prelude stub */';
const src = VERT_SRC(PRELUDE, '#define GLOBE');

test('far-side cull block exists and is guarded to the GLOBE shader variant', () => {
  const ifdef = src.indexOf('#ifdef GLOBE');
  const endif = src.indexOf('#endif');
  assert.ok(ifdef >= 0, 'has #ifdef GLOBE');
  assert.ok(endif > ifdef, 'has matching #endif');
  const block = src.slice(ifdef, endif);
  // globe-only symbols must ONLY appear inside the guard, or the mercator
  // variant (whose prelude lacks them) would fail to compile.
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(block.includes(sym), `cull block uses ${sym}`);
    const outside = src.slice(0, ifdef) + src.slice(endif);
    assert.ok(!outside.includes(sym), `${sym} must not leak outside #ifdef GLOBE`);
  }
});

test('cull mirrors occlusion.ts: camera reconstruction, segment test, radius²', () => {
  // camera = plane.xyz · (-1/plane.w)  (cameraFromClippingPlane)
  assert.ok(
    src.includes('u_projection_clipping_plane.xyz * (-1.0 / u_projection_clipping_plane.w)'),
    'reconstructs the camera exactly as cameraFromClippingPlane does',
  );
  // segment–sphere closest-approach test (earthOccludes)
  assert.ok(src.includes('float t = -dot(cam, v) / dot(v, v);'), 'closest-approach parameter');
  assert.ok(src.includes('t > 0.0 && t < 1.0'), 'strictly-between-endpoints check');
  // the inlined constant must be OCCLUSION_RADIUS² (limb anti-flicker bias)
  const r2 = (OCCLUSION_RADIUS * OCCLUSION_RADIUS).toFixed(6);
  assert.ok(src.includes(r2), `inlines OCCLUSION_RADIUS² (${r2})`);
});

test('cull is disabled mid globe↔mercator transition and on a degenerate plane', () => {
  assert.ok(
    src.includes('u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0'),
    'gated to full globe mode with a valid clipping plane',
  );
});

test('altitude enters the sphere position (GEO must cull differently from LEO)', () => {
  assert.ok(
    src.includes('projectToSphere(a_data.xy) * (1.0 + a_data.z / GLOBE_RADIUS)'),
    'satellite sphere position includes altitude, matching mercatorToSphere',
  );
});

test('sentinel slots are still culled before the far-side test', () => {
  const sentinel = src.indexOf('cls < 0.0');
  const cull = src.indexOf('#ifdef GLOBE');
  assert.ok(sentinel >= 0 && sentinel < cull, 'sentinel check precedes the far-side cull');
});
