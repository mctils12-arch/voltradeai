// On-map 3D spacecraft layer — ORBITAL O5-2b. Pins the shader contract the
// same way satLayer.test.ts does (GLSL can't import TS): the anchor projects
// through projectTileFor3D, the far-side cull matches ./occlusion exactly and
// stays inside #ifdef GLOBE, and the constant-pixel-size trick multiplies by
// anchor w. Plus the no-GL API smoke: a layer with no anchor draws nothing.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SatModelLayer, MODEL_VERT_SRC, MODEL_PIXELS } from './modelLayer.js';
import { OCCLUSION_RADIUS } from './occlusion.js';

const src = MODEL_VERT_SRC('/* prelude stub */', '#define GLOBE');

test('far-side cull: identical formulas to satLayer/occlusion, guarded to GLOBE', () => {
  const ifdef = src.indexOf('#ifdef GLOBE');
  const endif = src.indexOf('#endif');
  assert.ok(ifdef >= 0 && endif > ifdef);
  const block = src.slice(ifdef, endif);
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(block.includes(sym), `cull block uses ${sym}`);
    const outside = src.slice(0, ifdef) + src.slice(endif);
    assert.ok(!outside.includes(sym), `${sym} must not leak outside #ifdef GLOBE (mercator prelude lacks it)`);
  }
  assert.ok(src.includes('u_projection_clipping_plane.xyz * (-1.0 / u_projection_clipping_plane.w)'),
    'camera reconstruction matches cameraFromClippingPlane');
  const r2 = (OCCLUSION_RADIUS * OCCLUSION_RADIUS).toFixed(6);
  assert.ok(src.includes(r2), `inlines OCCLUSION_RADIUS² (${r2})`);
  assert.ok(src.includes('u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0'),
    'cull disabled mid-transition / degenerate plane, same gate as satLayer');
});

test('anchor projects through projectTileFor3D and offsets scale by anchor.w (constant pixel size)', () => {
  assert.ok(src.includes('projectTileFor3D(u_anchor.xy, u_anchor.z)'),
    'the model rides the SAME altitude-aware projection as the point field');
  assert.ok(/\* anchor\.w/.test(src),
    'clip-space offsets multiply by anchor.w — the constant-screen-size contract');
  assert.ok(src.includes('u_rot * a_normal'), 'normals rotate with the tumble for honest lighting');
});

test('API smoke (no GL): form/anchor setters, unknown class = no mesh, no-anchor render is a no-op', () => {
  const layer = new SatModelLayer();
  assert.equal(layer.getForm(), null);
  assert.equal(layer.getAnchor(), null);
  layer.setForm('cubesat');
  assert.equal(layer.getForm(), 'cubesat');
  layer.setForm(null); // unknown class: honest ring-only follow, never a guessed spacecraft
  assert.equal(layer.getForm(), null);
  layer.setAnchor({ mercX: 0.3, mercY: 0.4, altMeters: 550_000 });
  assert.deepEqual(layer.getAnchor(), { mercX: 0.3, mercY: 0.4, altMeters: 550_000 });
  // with an anchor but NO mesh (form null), render must return before touching GL
  const explodingGl = new Proxy({}, { get() { throw new Error('gl touched with no mesh'); } });
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false);
  // and with neither anchor nor mesh, same no-op
  layer.setAnchor(null);
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false);
  assert.equal(MODEL_PIXELS >= 48 && MODEL_PIXELS <= 160, true, 'focused-object size stays sane');
});
