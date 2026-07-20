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
  // vtProjElev (2026-07-20) adds a second GLOBE guard — collect ALL guarded
  // blocks; the invariant (globe symbols never leak outside a guard) stands
  const __guards = Array.from(src.matchAll(/#ifdef GLOBE[\s\S]*?#endif/g)).map((m) => m[0]);
  const __guarded = __guards.join('\n');
  const __outside = src.replace(/#ifdef GLOBE[\s\S]*?#endif/g, '');
  assert.ok(__guards.length > 0);
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(__guarded.includes(sym), `cull block uses ${sym}`);
    const outside = __outside;
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
  assert.ok(src.includes('projectTileFor3D(u_anchor.xy, vtProjElev(u_anchor.z, u_anchor.y))'),
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
  // O6 ring fix: an anchor with NO mesh now DRAWS (the focus ring rides the
  // anchor at true altitude) — with a broken GL the failure latch must trip
  // gracefully instead of crashing the map.
  const explodingGl = new Proxy({}, { get() { throw new Error('gl touched'); } });
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), true, 'ring draw attempted → latch trips on a broken GL');
  // with neither anchor nor mesh nor minis, render is a pure no-op (fresh layer)
  const idle = new SatModelLayer();
  idle.render(explodingGl as any, {} as any);
  assert.equal(idle.getRenderFailed(), false, 'nothing to draw → GL never touched');
  assert.equal(MODEL_PIXELS >= 48 && MODEL_PIXELS <= 160, true, 'focused-object size stays sane');
});

test('O5-3b real mesh: precedence over the form, cleared = fall back, renderable without a form', () => {
  const layer = new SatModelLayer();
  const real = {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    colors: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    vertexCount: 3,
  };
  layer.setForm('bus');
  assert.notEqual(layer.getActiveMesh(), null);
  const formMesh = layer.getActiveMesh();
  layer.setRealMesh(real);
  assert.equal(layer.getActiveMesh(), real, 'real model outranks the representative form');
  layer.setRealMesh(null);
  assert.equal(layer.getActiveMesh(), formMesh, 'clearing falls back to the form, not to nothing');
  // ISS-before-SATCAT case: real mesh with NO form still renders
  layer.setForm(null);
  layer.setRealMesh(real);
  assert.equal(layer.getActiveMesh(), real);
  layer.setRealMesh(null);
  assert.equal(layer.getActiveMesh(), null, 'nothing left → render is a no-op again');
});
