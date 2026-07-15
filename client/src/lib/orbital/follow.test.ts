// Satellite follow math — ORBITAL O5 slice 1. Hermetic buffer fixtures.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { followTarget, focusRingFeatureCollection } from './follow.js';
import { SAT_STRIDE, SENTINEL_SKIP, lonLatToMercator } from './satBuffer.js';

function bufferWith(slots: Array<{ lon: number; lat: number; altM: number; cls: number }>): Float32Array {
  const buf = new Float32Array(slots.length * SAT_STRIDE);
  slots.forEach((s, i) => {
    const m = lonLatToMercator(s.lon, s.lat);
    buf[i * SAT_STRIDE] = m.x;
    buf[i * SAT_STRIDE + 1] = m.y;
    buf[i * SAT_STRIDE + 2] = s.altM;
    buf[i * SAT_STRIDE + 3] = s.cls;
  });
  return buf;
}

test("followTarget: round-trips a real slot's position, altitude in km", () => {
  const buf = bufferWith([
    { lon: -100, lat: 40, altM: 550_000, cls: 0 },
    { lon: 10, lat: -30, altM: 20_000_000, cls: 1 },
  ]);
  const t0 = followTarget(buf, 0)!;
  assert.ok(Math.abs(t0.lonDeg - -100) < 1e-4 && Math.abs(t0.latDeg - 40) < 1e-4);
  assert.ok(Math.abs(t0.altKm - 550) < 1e-6);
  const m = lonLatToMercator(-100, 40);
  assert.ok(Math.abs(t0.mercX - m.x) < 1e-7 && Math.abs(t0.mercY - m.y) < 1e-7,
    "raw mercator coords pass through for the on-map model anchor");
  const t1 = followTarget(buf, 1)!;
  assert.ok(Math.abs(t1.lonDeg - 10) < 1e-4 && Math.abs(t1.latDeg - -30) < 1e-4);
});

test('followTarget: sentinel slots and bad indices return null — the camera never moves on a guess', () => {
  const buf = bufferWith([{ lon: 0, lat: 0, altM: 0, cls: SENTINEL_SKIP }]);
  assert.equal(followTarget(buf, 0), null, 'sentinel (deep-space/invalid) = no honest position');
  assert.equal(followTarget(buf, 1), null, 'out-of-range index');
  assert.equal(followTarget(buf, -1), null);
  assert.equal(followTarget(null, 0), null);
});

test('focusRingFeatureCollection: one ring at the target, empty when nothing is followed', () => {
  const fc = focusRingFeatureCollection({ lonDeg: 5, latDeg: 6, altKm: 550 });
  assert.equal(fc.features.length, 1);
  assert.deepEqual(fc.features[0].geometry.coordinates, [5, 6]);
  assert.equal(focusRingFeatureCollection(null).features.length, 0, 'no follow = no ring, never a stale ring');
});
