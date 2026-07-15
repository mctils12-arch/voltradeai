// True-altitude aircraft — EARTH TWIN E3. Pins the silhouette geometry, the
// instance-building honesty, CPU picking, and the shader contract (same
// pinning discipline as satLayer.test.ts/modelLayer.test.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  AIR_3D_MIN_ZOOM,
  AIR_INST_STRIDE,
  AIR_BAND,
  PLANE_SILHOUETTE,
  buildAircraftInstances,
  pickNearestAircraft,
  AIR_VERT_SRC,
  AirLayer,
} from './airLayer.js';
import { lonLatToMercator } from '../orbital/satBuffer.js';
import { OCCLUSION_RADIUS } from '../orbital/occlusion.js';

test('silhouette: whole triangles, bounded local extent, deterministic', () => {
  assert.equal(PLANE_SILHOUETTE.length % 6, 0, 'x,y pairs in whole triangles');
  assert.ok(PLANE_SILHOUETTE.length / 6 >= 8, 'enough triangles to read as a plane');
  for (let i = 0; i < PLANE_SILHOUETTE.length; i++) {
    assert.ok(Math.abs(PLANE_SILHOUETTE[i]) <= 0.62 + 1e-6, `vertex ${i} within the declared extent (float32 storage)`);
  }
});

test('buildAircraftInstances: fields map through, bands match the 2D ALT_COLOR semantics, junk rows skipped', () => {
  const { inst, rows } = buildAircraftInstances([
    { lon: -100, lat: 40, altitude_m: 11000, heading: 90, on_ground: false },
    { lon: 10, lat: 50, altitude_m: 1200, heading: 270, on_ground: false },
    { lon: 0, lat: 0, altitude_m: 5000, heading: 10, on_ground: true },
    { lon: null, lat: 40 },                       // no position → skipped
    { lon: 20, lat: 89 },                          // beyond mercator → skipped
    { lon: 30, lat: 30, altitude_m: -200, heading: null }, // negative baro clamps to 0
  ]);
  assert.equal(rows.length, 4);
  assert.equal(inst.length, 4 * AIR_INST_STRIDE);
  const m0 = lonLatToMercator(-100, 40);
  assert.ok(Math.abs(inst[0] - m0.x) < 1e-7 && Math.abs(inst[1] - m0.y) < 1e-7);
  assert.equal(inst[2], 11000);
  assert.equal(inst[3], 90);
  assert.equal(inst[4], AIR_BAND.CRUISE);
  assert.equal(inst[1 * AIR_INST_STRIDE + 4], AIR_BAND.LOW, '<3000m = low band (amber)');
  assert.equal(inst[2 * AIR_INST_STRIDE + 2], 0, 'on-ground renders at 0 regardless of reported alt');
  assert.equal(inst[2 * AIR_INST_STRIDE + 4], AIR_BAND.GROUND);
  assert.equal(inst[3 * AIR_INST_STRIDE + 2], 0, 'negative baro clamps to 0, never below the surface');
  assert.equal(inst[3 * AIR_INST_STRIDE + 3], 0, 'null heading renders north-up, not NaN');
});

test('pickNearestAircraft: nearest within tolerance, honest miss outside it', () => {
  const { inst } = buildAircraftInstances([
    { lon: -100, lat: 40, altitude_m: 10000, heading: 0 },
    { lon: -100.001, lat: 40.001, altitude_m: 9000, heading: 0 },
    { lon: 50, lat: -20, altitude_m: 8000, heading: 0 },
  ]);
  const m = lonLatToMercator(-100.0008, 40.0008);
  const hit = pickNearestAircraft(inst, m.x, m.y, 0.0001);
  assert.equal(hit, 1, 'nearest of the pair wins');
  assert.equal(pickNearestAircraft(inst, 0.9, 0.9, 1e-7), -1, 'nothing close = honest miss');
  assert.equal(pickNearestAircraft(null, 0, 0, 1), -1);
});

test('shader contract: altitude-aware projection, GLOBE-guarded cull identical to the orbital layers, bearing-relative rotation, w-scaled offsets', () => {
  const src = AIR_VERT_SRC('/* prelude stub */', '#define GLOBE');
  assert.ok(src.includes('projectTileFor3D(a_inst.xy, a_inst.z * u_altScale)'),
    'anchor = position + REAL altitude (terrain-exaggeration matched)');
  const ifdef = src.indexOf('#ifdef GLOBE');
  const endif = src.indexOf('#endif');
  const outside = src.slice(0, ifdef) + src.slice(endif);
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(src.slice(ifdef, endif).includes(sym) && !outside.includes(sym),
      `${sym} contained in the GLOBE guard`);
  }
  const r2 = (OCCLUSION_RADIUS * OCCLUSION_RADIUS).toFixed(6);
  assert.ok(src.includes(r2), 'cull radius² matches ./occlusion');
  assert.ok(src.includes('radians(a_inst.w - u_bearing)'),
    'heading rotates relative to map bearing — parity with icon-rotation-alignment: map');
  assert.ok(/\* anchor\.w/.test(src), 'constant-pixel offsets scale by anchor w');
});

test('API smoke (no GL): counts, zoom gate semantics, no-instance render is a no-op', () => {
  const layer = new AirLayer();
  assert.equal(layer.getCounts().total, 0);
  const { inst } = buildAircraftInstances([{ lon: 0, lat: 0, altitude_m: 10000, heading: 0 }]);
  layer.setInstances(inst);
  assert.equal(layer.getCounts().total, 1);
  assert.equal(layer.getCounts().drawn, false, 'no map bound → zoom 0 → below the split, not drawn');
  layer.setInstances(null);
  assert.equal(layer.getCounts().total, 0);
  const explodingGl = new Proxy({}, { get() { throw new Error('gl touched with no instances'); } });
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false);
  assert.ok(AIR_3D_MIN_ZOOM >= 6 && AIR_3D_MIN_ZOOM <= 12, 'hand-off in the regional-zoom band');
});
