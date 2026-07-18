// True-altitude aircraft — EARTH TWIN E3. Pins the silhouette geometry, the
// instance-building honesty, CPU picking, and the shader contract (same
// pinning discipline as satLayer.test.ts/modelLayer.test.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  AIR_3D_MIN_ZOOM,
  AIR_INST_STRIDE,
  AIR_BAND,
  AIR_SHAPE,
  AIR_SILHOUETTES,
  PLANE_SILHOUETTE,
  shapeForCategory,
  buildAircraftInstances,
  pickNearestAircraft,
  pickNearestAircraftScreen,
  AIR_VERT_SRC,
  AirLayer,
} from './airLayer.js';
import { lonLatToMercator } from '../orbital/satBuffer.js';
import { mercatorToSphere } from '../orbital/occlusion.js';
import { OCCLUSION_RADIUS } from '../orbital/occlusion.js';

test('silhouettes: every class shape is whole triangles within the declared extent', () => {
  assert.equal(AIR_SILHOUETTES.length, 5, 'one shape per AIR_SHAPE code');
  assert.equal(AIR_SILHOUETTES[AIR_SHAPE.DEFAULT], PLANE_SILHOUETTE, 'DEFAULT is the generic airliner');
  for (const [si, sil] of AIR_SILHOUETTES.entries()) {
    assert.equal(sil.length % 6, 0, `shape ${si}: x,y pairs in whole triangles`);
    assert.ok(sil.length / 6 >= 5, `shape ${si}: enough triangles to read as a form`);
    for (let i = 0; i < sil.length; i++) {
      assert.ok(Math.abs(sil[i]) <= 0.62 + 1e-6, `shape ${si} vertex ${i} within the declared extent (float32 storage)`);
    }
  }
});

test('shapeForCategory: decodes only the documented ADS-B emitter categories, never guesses', () => {
  assert.equal(shapeForCategory('A1'), AIR_SHAPE.LIGHT);
  assert.equal(shapeForCategory('A5'), AIR_SHAPE.HEAVY);
  assert.equal(shapeForCategory('A6'), AIR_SHAPE.PERF);
  assert.equal(shapeForCategory('A7'), AIR_SHAPE.ROTOR);
  for (const c of ['A2', 'A3', 'A4', 'B1', 'C2', '', null, undefined, 'a5-junk']) {
    assert.equal(shapeForCategory(c as any), AIR_SHAPE.DEFAULT, `${c} stays the generic form`);
  }
  assert.equal(shapeForCategory('a7'), AIR_SHAPE.ROTOR, 'case-insensitive on the raw feed value');
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

test('buildAircraftInstances: stable shape sort → contiguous draw groups, rows stay pick-aligned', () => {
  const { inst, rows, groups } = buildAircraftInstances([
    { lon: 1, lat: 1, altitude_m: 100, category: 'A7' },   // rotor
    { lon: 2, lat: 2, altitude_m: 200, category: null },   // default
    { lon: 3, lat: 3, altitude_m: 300, category: 'A5' },   // heavy
    { lon: 4, lat: 4, altitude_m: 400 },                   // default
    { lon: 5, lat: 5, altitude_m: 500, category: 'A5' },   // heavy
  ]);
  assert.deepEqual(groups.map((g) => g.shape), [AIR_SHAPE.DEFAULT, AIR_SHAPE.HEAVY, AIR_SHAPE.ROTOR], 'ascending shape codes');
  assert.deepEqual(groups.map((g) => g.count), [2, 2, 1]);
  assert.deepEqual(groups.map((g) => g.start), [0, 2, 4], 'contiguous ranges covering every instance');
  // stable within a shape: the two defaults keep input order (lon 2 then 4)
  assert.deepEqual(rows.map((r) => r.lon), [2, 4, 3, 5, 1]);
  // rows[i] still describes instance i (CPU picking depends on it)
  for (let i = 0; i < rows.length; i++) {
    assert.equal(inst[i * AIR_INST_STRIDE + 2], (rows[i].altitude_m as number), `row ${i} aligned with its instance`);
    assert.equal(inst[i * AIR_INST_STRIDE + 5], shapeForCategory(rows[i].category), 'shape stored in the record');
  }
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
  assert.ok(src.includes('projectTileFor3D(g_xy, a_inst.z * u_altScale * a_altFrac)'),
    'anchor = GLIDED position + REAL altitude (terrain-exaggeration matched, drop-line fraction)');
  // GLIDE contract (the satLayer shader-glide pattern): dead-reckoned
  // anchor = poll position + broadcast velocity × capped elapsed seconds,
  // antimeridian-wrapped and pole-clamped; the far-side cull tests the
  // GLIDED position (cull where drawn, not where polled).
  assert.ok(src.includes('fract(a_inst.x + a_vel.x * u_dtSec)'), 'X glides and wraps the antimeridian');
  assert.ok(src.includes('clamp(a_inst.y + a_vel.y * u_dtSec, 0.0, 1.0)'), 'Y glides and clamps at the poles');
  assert.ok(src.includes('projectToSphere(g_xy)'), 'far-side cull uses the glided position');
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
  // drop-line contract: altitude scales by the per-vertex fraction and the
  // ground end fades — silhouettes (altFrac 1) keep full altitude + alpha
  assert.ok(src.includes('a_inst.z * u_altScale * a_altFrac'), 'altitude scaled by the drop-line fraction');
  assert.ok(src.includes('mix(0.25, 1.0, a_altFrac)'), 'line fades toward the surface end');
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

test('glide packing: broadcast velocity → merc-per-second fields; frozen rows pack 0', () => {
  const { inst } = buildAircraftInstances([
    { lon: 0, lat: 0, altitude_m: 10000, heading: 90, velocity_ms: 250 },   // due east at the equator
    { lon: 0, lat: 0, altitude_m: 0, heading: 90, velocity_ms: 250, on_ground: true }, // ground → frozen
    { lon: 0, lat: 0, altitude_m: 8000, heading: null, velocity_ms: 250 },  // no track → frozen
    { lon: 0, lat: 0, altitude_m: 8000, heading: 90, velocity_ms: null },   // no speed → frozen
  ]);
  const vx = inst[6], vy = inst[7];
  assert.ok(vx > 0, 'eastbound velX positive');
  assert.ok(Math.abs(vy) < 1e-9, 'due-east at the equator: no Y drift');
  // 250 m/s eastward = 250/40075017 of the world circumference per second
  assert.ok(Math.abs(vx - 250 / 40_075_017) / (250 / 40_075_017) < 0.01, 'velX magnitude ≈ ground speed');
  for (const i of [1, 2, 3]) {
    assert.equal(inst[i * AIR_INST_STRIDE + 6], 0, `frozen row ${i}: velX 0 (never a guessed vector)`);
    assert.equal(inst[i * AIR_INST_STRIDE + 7], 0, `frozen row ${i}: velY 0`);
  }
});

test('pick agrees with the glided pixels: dtSec moves the pick point exactly like the shader', () => {
  const { inst } = buildAircraftInstances([
    { lon: -100, lat: 40, altitude_m: 10000, heading: 90, velocity_ms: 250 },
  ]);
  const base = lonLatToMercator(-100, 40);
  const glided = { x: base.x + inst[6] * 20, y: base.y + inst[7] * 20 };
  assert.ok(Math.abs(glided.x - base.x) > 1e-6, '20s at 250 m/s is a real displacement in mercator');
  assert.equal(pickNearestAircraft(inst, glided.x, glided.y, 1e-7, 20), 0, 'glided pick hits at the drawn spot');
  assert.equal(pickNearestAircraft(inst, glided.x, glided.y, 1e-7, 0), -1, 'unglided pick misses the drawn spot');
  assert.equal(pickNearestAircraft(inst, base.x, base.y, 1e-7, 0), 0, 'dt 0 = the pre-glide behavior');
});

test('glide anchor + repaint tick: honest cap, unwired = pre-glide behavior', () => {
  let t = 1000;
  const layer = new AirLayer({ now: () => t });
  assert.equal(layer.getGlideDtSec(), 0, 'unwired layer glides zero');
  layer.setTickTime();
  t += 5000;
  assert.equal(layer.getGlideDtSec(), 5, 'elapsed seconds since the anchor');
  t += 60_000;
  assert.equal(layer.getGlideDtSec(), 25, 'clamps at MAX_AIR_GLIDE_SEC — past a missed poll the display freezes');
  // glideRepaintTick without a map/instances is a safe no-op (parent may
  // tick before the first payload)
  layer.glideRepaintTick();
});

test('pickNearestAircraftScreen: altitude displacement respected (the 45°-tilt click fix)', () => {
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const { inst } = buildAircraftInstances([
    { lon: -90, lat: 30, altitude_m: 0, heading: 0, on_ground: true },
    { lon: -90, lat: 30, altitude_m: 12_000, heading: 0 }, // same ground point, at cruise
  ]);
  const W = 1000, H = 1000;
  const project = (i: number) => {
    const p = mercatorToSphere(inst[i * AIR_INST_STRIDE], inst[i * AIR_INST_STRIDE + 1], inst[i * AIR_INST_STRIDE + 2]);
    return [((p[0] + 1) / 2) * W, ((1 - p[1]) / 2) * H];
  };
  const sHigh = project(1), sGround = project(0);
  assert.ok(Math.hypot(sHigh[0] - sGround[0], sHigh[1] - sGround[1]) > 0.5, 'they render apart');
  assert.equal(pickNearestAircraftScreen(inst, I, sHigh[0], sHigh[1], W, H, 0.4), 1, 'the airborne plane under the cursor wins');
  assert.equal(pickNearestAircraftScreen(inst, I, sGround[0], sGround[1], W, H, 0.4), 0);
  assert.equal(pickNearestAircraftScreen(inst, I, 5, 5, W, H, 2), -1, 'far from both = honest miss');
  assert.equal(pickNearestAircraftScreen(null, I, 0, 0, W, H, 10), -1);
});
