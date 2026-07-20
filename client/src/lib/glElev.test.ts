// Elevation-unit conversion — the GLSL mirror must equal the pinned CPU
// mercatorZFromAltitude, which must equal MapLibre's own public API.
// Run: npx tsx --test client/src/lib/glElev.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mercatorZFromAltitude, EARTH_CIRCUMFERENCE_M, VT_PROJ_ELEV_GLSL } from './glElev.js';

const near = (a: number, b: number, rel: number, msg: string) =>
  assert.ok(Math.abs(a - b) <= Math.abs(b) * rel + 1e-15, `${msg} (${a} vs ${b})`);

/** the GLSL formula, executed in TS: ele × cosh(π(1−2y)) / (2πR) */
const glslMirror = (ele: number, mercY: number) =>
  (ele * Math.cosh(Math.PI * (1 - 2 * mercY))) / EARTH_CIRCUMFERENCE_M;

test('GLSL mirror ≡ the pinned CPU mercatorZFromAltitude (sech identity)', () => {
  for (const y of [0.5, 0.3, 0.38, 0.62, 0.9, 0.05]) {
    for (const ele of [0, 500, 10668, 550_000]) {
      near(glslMirror(ele, y), mercatorZFromAltitude(ele, y), 1e-9,
        `y=${y} ele=${ele}`);
    }
  }
});

test('matches MapLibre MercatorCoordinate.fromLngLat (the public ground truth)', async () => {
  const ml: any = await import('maplibre-gl');
  const MercatorCoordinate = ml.MercatorCoordinate ?? ml.default?.MercatorCoordinate;
  assert.ok(MercatorCoordinate, 'MercatorCoordinate export resolved');
  for (const [lng, lat] of [[-92.93, 34.49], [0, 0], [10, 60], [-150, -35]] as Array<[number, number]>) {
    const alt = 10_668;
    const mc = MercatorCoordinate.fromLngLat({ lng, lat }, alt);
    near(mercatorZFromAltitude(alt, mc.y), mc.z, 1e-6,
      `mercator z at ${lng},${lat}`);
  }
});

test('the shader snippet carries both projection branches', () => {
  assert.ok(VT_PROJ_ELEV_GLSL.includes('#ifdef GLOBE'), 'globe branch present');
  assert.ok(VT_PROJ_ELEV_GLSL.includes('u_projection_transition > 0.999'), 'meters only in FULL globe');
  assert.ok(VT_PROJ_ELEV_GLSL.includes('cosh(PI * (1.0 - 2.0 * mercY))'), 'sech(π(1−2y)) latitude factor');
});
