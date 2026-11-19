// Aircraft dead-reckoning glide — pure-math pins (the satellite SMOOTH SKY
// honesty model applied to the 15s aircraft poll; see airGlide.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_AIR_GLIDE_SEC,
  AIR_GLIDE_2D_MIN_ZOOM,
  AIR_GLIDE_STEP_MS,
  airGlideDtSec,
  glideDegPerSec,
  mercVelPerSec,
  shouldGlidePerFrame,
} from './airGlide.js';
import { AIR_3D_MIN_ZOOM } from './airLayer.js';
import { lonLatToMercator, wrapMercDelta } from '../orbital/satBuffer.js';

test('constants: cap sized for the 15s poll, 2D floor below the 3D hand-off, 3-4Hz step', () => {
  assert.ok(MAX_AIR_GLIDE_SEC >= 15 && MAX_AIR_GLIDE_SEC <= 30, 'cap ≈ one missed poll, then freeze');
  assert.ok(AIR_GLIDE_2D_MIN_ZOOM < AIR_3D_MIN_ZOOM, '2D glide band exists below the hand-off');
  assert.ok(AIR_GLIDE_STEP_MS >= 250 && AIR_GLIDE_STEP_MS <= 350, 'step in the measured 3-4Hz window');
});

test('airGlideDtSec: zero before the anchor / on broken clocks, linear, clamps at the cap', () => {
  assert.equal(airGlideDtSec(1000, 2000), 0, 'anchor in the future glides zero');
  assert.equal(airGlideDtSec(NaN, 0), 0, 'broken clock glides zero — pre-glide behavior');
  assert.equal(airGlideDtSec(11_000, 1000), 10);
  assert.equal(airGlideDtSec(1000 + MAX_AIR_GLIDE_SEC * 1000 + 60_000, 1000), MAX_AIR_GLIDE_SEC,
    'holds at the honesty cap rather than flying fiction');
});

test('glideDegPerSec: broadcast track decomposes correctly; latitude scales longitude', () => {
  const north = glideDegPerSec(40, 0, 111.32, false)!;
  assert.ok(Math.abs(north.dLat - 0.001) < 1e-6, 'due north: ~0.001 deg lat per second at 111.32 m/s');
  assert.ok(Math.abs(north.dLon) < 1e-9, 'due north: no longitude drift');
  const east0 = glideDegPerSec(0, 90, 200, false)!;
  const east60 = glideDegPerSec(60, 90, 200, false)!;
  assert.ok(Math.abs(east60.dLon / east0.dLon - 2) < 0.01, 'at lat 60 a degree of lon is half as wide → dLon doubles');
  assert.ok(Math.abs(east0.dLat) < 1e-9 && Math.abs(east60.dLat) < 1e-9);
  const south = glideDegPerSec(40, 180, 100, false)!;
  assert.ok(south.dLat < 0, 'southbound decreases latitude');
});

test('glideDegPerSec: frozen (null) whenever the broadcast inputs are missing — never a guessed vector', () => {
  assert.equal(glideDegPerSec(40, 90, 200, true), null, 'on ground: frozen');
  assert.equal(glideDegPerSec(40, null, 200, false), null, 'no track: frozen');
  assert.equal(glideDegPerSec(40, 90, null, false), null, 'no ground speed: frozen');
  assert.equal(glideDegPerSec(40, 90, 0, false), null, 'zero speed: frozen');
  assert.equal(glideDegPerSec(null, 90, 200, false), null, 'no latitude: frozen');
  assert.equal(glideDegPerSec(88, 90, 200, false), null, 'outside the mercator band: frozen');
  assert.equal(glideDegPerSec(40, NaN, 200, false), null, 'junk track: frozen');
});

test('mercVelPerSec: identical path to the 2D degrees glide (one dead-reckoning, two renderers)', () => {
  const lon = -100, lat = 40, hdg = 47, v = 230;
  const deg = glideDegPerSec(lat, hdg, v, false)!;
  const merc = mercVelPerSec(lon, lat, hdg, v, false);
  for (const dt of [5, 15, MAX_AIR_GLIDE_SEC]) {
    const from2d = lonLatToMercator(lon + deg.dLon * dt, lat + deg.dLat * dt);
    const gx = lonLatToMercator(lon, lat).x + merc.vx * dt;
    const gy = lonLatToMercator(lon, lat).y + merc.vy * dt;
    // agreement within a meter of world-mercator (1m ≈ 2.5e-8 units)
    assert.ok(Math.abs(gx - from2d.x) < 1e-6, `x agrees at dt=${dt}`);
    assert.ok(Math.abs(gy - from2d.y) < 1e-6, `y agrees at dt=${dt}`);
  }
});

test('mercVelPerSec: antimeridian wrap and frozen rows', () => {
  const near180 = mercVelPerSec(179.999, 30, 90, 250, false);
  assert.ok(near180.vx > 0 && near180.vx < 1e-4, 'eastbound across the antimeridian: tiny positive step, not a world traverse');
  assert.ok(Math.abs(wrapMercDelta(0.999) - -0.001) < 1e-12, 'wrap helper sanity');
  const frozen = mercVelPerSec(0, 0, 90, 250, true);
  assert.deepEqual(frozen, { vx: 0, vy: 0 }, 'ground: packs zero velocity');
  assert.deepEqual(mercVelPerSec(null, 0, 90, 250, false), { vx: 0, vy: 0 }, 'no lon: zero');
});

test('shouldGlidePerFrame: per-frame repaint only where the step tick would visibly jump', () => {
  assert.equal(shouldGlidePerFrame(40, 4), false, 'continent zoom: steps are sub-pixel — cheap tick suffices');
  assert.equal(shouldGlidePerFrame(40, AIR_3D_MIN_ZOOM), false, 'at the hand-off a 300ms step is still <1px');
  assert.equal(shouldGlidePerFrame(40, 11), true, 'close zoom: stepped motion would read as jank → 60fps');
  assert.equal(shouldGlidePerFrame(NaN, 11), false, 'broken input fails closed (step tick still runs)');
});
