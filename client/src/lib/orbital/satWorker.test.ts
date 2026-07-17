// Hermetic tests for the satellite propagation worker's pure core + runtime.
// NO real Worker, NO WebGL, NO network, NO wall-clock: propagate/isDeepSpace,
// time, timers, and postMessage are all injected. Run:
//   npx tsx --test client/src/lib/orbital/satWorker.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { GpRecord } from './tle.ts';
import { lonLatToMercator, mercatorToLonLat, readSatAt } from './satBuffer.ts';
import {
  SAT_STRIDE,
  SENTINEL_SKIP,
  VEL_SAMPLE_MS,
  orbitClassCode,
  packPositions,
  analyzePopulation,
  positionsMessage,
  readyMessage,
  createSatWorker,
  type PropagationDeps,
  type SatWorkerOutbound,
  type SatPositionsMessage,
} from './satWorker.ts';
import { propagate } from './propagate.ts';

// --------------------------------------------------------------------------
// fixtures
// --------------------------------------------------------------------------

function gp(noradId: number, over: Partial<GpRecord> = {}): GpRecord {
  return {
    noradId,
    name: 'SAT-' + noradId,
    epoch: '2026-07-07T00:00:00',
    inclination: 51,
    raan: 100,
    ecc: 0.001,
    argp: 90,
    meanAnomaly: 10,
    meanMotion: 15,
    bstar: 0,
    ...over,
  };
}

// Real ISS element set (matches propagate.test.ts regression fixture).
const ISS: GpRecord = {
  noradId: 25544,
  name: 'ISS (ZARYA)',
  epoch: '2026-07-07T12:12:01.987776',
  inclination: 51.6304,
  raan: 199.5144,
  ecc: 0.00066874,
  argp: 267.6545,
  meanAnomaly: 92.3678,
  meanMotion: 15.48933372,
  bstar: 0.00011369272,
};
const ISS_EPOCH_MS = Date.parse('2026-07-07T12:12:01.987Z');

// --------------------------------------------------------------------------
// orbitClassCode
// --------------------------------------------------------------------------

test('orbitClassCode: altitude -> packed class code', () => {
  assert.equal(orbitClassCode(550), 0); // LEO
  assert.equal(orbitClassCode(20000), 1); // MEO
  assert.equal(orbitClassCode(35786), 2); // GEO
});

// --------------------------------------------------------------------------
// packPositions: valid packing, sentinels, index alignment, honest counts
// --------------------------------------------------------------------------

test('packPositions: packs valid, sentinels nulls, counts deep-space vs invalid', () => {
  const pop = [gp(1), gp(2), gp(3), gp(4)];
  // Stub propagation: #1 valid LEO, #4 valid MEO, #2/#3 null.
  const stub: PropagationDeps = {
    propagate: (g) => {
      if (g.noradId === 1) return { lonDeg: -40, latDeg: 12, altKm: 550 };
      if (g.noradId === 4) return { lonDeg: 100, latDeg: -33, altKm: 20000 };
      return null;
    },
    // #2 is deep-space (needs SDP4); #3 is a plain invalid.
    isDeepSpace: (g) => g.noradId === 2,
  };

  const r = packPositions(pop, 0, stub);
  assert.equal(r.total, 4);
  assert.equal(r.shown, 2);
  assert.equal(r.deepSpaceSkipped, 1);
  assert.equal(r.invalidSkipped, 1);
  assert.equal(r.buffer.length, 4 * SAT_STRIDE);

  // index 0 (SAT-1): valid, LEO, mercator matches, index-aligned to input.
  const s0 = readSatAt(r.buffer, 0);
  assert.equal(s0.valid, true);
  assert.equal(s0.classCode, 0);
  const m0 = lonLatToMercator(-40, 12);
  assert.ok(Math.abs(s0.mercX - m0.x) < 1e-6 && Math.abs(s0.mercY - m0.y) < 1e-6);
  assert.ok(Math.abs(s0.altMeters - 550_000) < 1e-3);

  // index 1 (SAT-2 deep-space) and 2 (SAT-3 invalid): sentinel, no fake position.
  assert.equal(readSatAt(r.buffer, 1).classCode, SENTINEL_SKIP);
  assert.equal(readSatAt(r.buffer, 1).valid, false);
  assert.equal(readSatAt(r.buffer, 2).classCode, SENTINEL_SKIP);

  // index 3 (SAT-4): valid MEO.
  assert.equal(readSatAt(r.buffer, 3).classCode, 1);
});

test('packPositions: empty population -> empty buffer, zero counts', () => {
  const r = packPositions([], 12345);
  assert.equal(r.buffer.length, 0);
  assert.deepEqual(
    [r.total, r.shown, r.deepSpaceSkipped, r.invalidSkipped],
    [0, 0, 0, 0],
  );
});

test('packPositions: real ISS propagates to a plausible LEO slot (round-trip)', () => {
  const t = ISS_EPOCH_MS + 90 * 60000;
  const r = packPositions([ISS], t, { propagate });
  assert.equal(r.shown, 1);
  assert.equal(r.deepSpaceSkipped, 0);
  const s = readSatAt(r.buffer, 0);
  assert.equal(s.valid, true);
  assert.equal(s.classCode, 0); // LEO
  // Unpack mercator back to lon/lat and compare to the validated reference.
  const geo = mercatorToLonLat(s.mercX, s.mercY);
  assert.ok(Math.abs(geo.lonDeg - 61.1089) < 0.02, `lon ${geo.lonDeg}`);
  assert.ok(Math.abs(geo.latDeg - -8.8584) < 0.02, `lat ${geo.latDeg}`);
  assert.ok(Math.abs(s.altMeters / 1000 - 421.437) < 0.5, `alt ${s.altMeters}`);
});

// --------------------------------------------------------------------------
// GLIDE velocity (fields [4..6]) — finite difference of two REAL propagations
// --------------------------------------------------------------------------

test('packPositions: writes per-second finite-difference velocity from a second propagation', () => {
  // Time-dependent stub: drifts +0.36 deg lon, -0.18 deg lat, +1 km alt per second.
  const stub: PropagationDeps = {
    propagate: (_g, t) => ({ lonDeg: 10 + 0.36 * (t / 1000), latDeg: -0.18 * (t / 1000), altKm: 500 + t / 1000 }),
  };
  const r = packPositions([gp(1)], 0, stub);
  const s = readSatAt(r.buffer, 0);
  // 0.36 deg/s lon == 0.001 mercator-x units/s exactly (0.36/360).
  assert.ok(Math.abs(s.velX - 0.001) < 1e-7, `velX ${s.velX}`);
  // lat -> mercY is nonlinear; verify against the exact projected delta.
  const m0 = lonLatToMercator(10, 0);
  const m1 = lonLatToMercator(10 + 0.36 * (VEL_SAMPLE_MS / 1000), -0.18 * (VEL_SAMPLE_MS / 1000));
  const expVy = ((m1.y - m0.y) * 1000) / VEL_SAMPLE_MS;
  assert.ok(Math.abs(s.velY - expVy) < 1e-9, `velY ${s.velY} vs ${expVy}`);
  assert.ok(Math.abs(s.velAlt - 1000) < 1e-3, `velAlt ${s.velAlt} (1 km/s = 1000 m/s)`);
  // tick position itself is untouched by the velocity sampling
  assert.ok(Math.abs(s.mercX - m0.x) < 1e-7 && Math.abs(s.mercY - m0.y) < 1e-7);
});

test('packPositions: antimeridian crossing yields a small wrapped velocity, never a world-traverse', () => {
  // 0.36 deg/s eastward across lon 180: x goes 0.9999... -> 0.0004...
  const stub: PropagationDeps = {
    propagate: (_g, t) => ({ lonDeg: 179.9 + 0.36 * (t / 1000), latDeg: 0, altKm: 500 }),
  };
  const s = readSatAt(packPositions([gp(1)], 0, stub).buffer, 0);
  assert.ok(Math.abs(s.velX - 0.001) < 1e-7, `wrapped velX ${s.velX} (unwrapped would be ~ -2/s)`);
});

test('packPositions: failed second sample (decay boundary) -> zero velocity, position still shown', () => {
  const stub: PropagationDeps = {
    propagate: (_g, t) => (t === 0 ? { lonDeg: 5, latDeg: 5, altKm: 700 } : null),
  };
  const r = packPositions([gp(1)], 0, stub);
  assert.equal(r.shown, 1, 'the real tick position still renders');
  const s = readSatAt(r.buffer, 0);
  assert.equal(s.valid, true);
  assert.deepEqual([s.velX, s.velY, s.velAlt], [0, 0, 0], 'hold at the real position — no fabricated rate');
});

test('packPositions: sentinel slots carry zero velocity (nothing to glide)', () => {
  const r = packPositions([gp(1)], 0, { propagate: () => null });
  const s = readSatAt(r.buffer, 0);
  assert.equal(s.valid, false);
  assert.deepEqual([s.velX, s.velY, s.velAlt], [0, 0, 0]);
});

test('GLIDE HONESTY: tick position + velocity·1s matches a REAL propagation 1s later to <100 m (ISS)', () => {
  // The glide the shader draws must be physics, not decoration: extrapolating
  // the packed velocity a full second (2× the 500 ms sampling interval, and
  // well past a 60 fps frame) must land within 100 m of where SGP4 actually
  // puts the ISS. Measured at this fixture: ~0.2-0.3 m total error (float32
  // buffer quantization + first-order truncation) — the 100 m bound leaves
  // two orders of magnitude of headroom, not a tuned pass.
  const t0 = ISS_EPOCH_MS + 90 * 60000;
  const s = readSatAt(packPositions([ISS], t0).buffer, 0);
  assert.equal(s.valid, true);
  const predX = ((s.mercX + s.velX * 1) % 1 + 1) % 1;
  const predY = s.mercY + s.velY * 1;
  const predAltM = s.altMeters + s.velAlt * 1;
  const truth = propagate(ISS, t0 + 1000)!;
  assert.ok(truth, 'reference propagation must succeed');
  const pll = mercatorToLonLat(predX, predY);
  // great-circle horizontal error at the satellite's orbital radius + radial error
  const R = 6_371_000 + truth.altKm * 1000;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(pll.latDeg - truth.latDeg);
  const dLon = toRad(pll.lonDeg - truth.lonDeg);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(truth.latDeg)) * Math.cos(toRad(pll.latDeg)) * Math.sin(dLon / 2) ** 2;
  const horizM = 2 * R * Math.asin(Math.sqrt(a));
  const altErrM = predAltM - truth.altKm * 1000;
  const totalM = Math.hypot(horizM, altErrM);
  assert.ok(
    totalM < 100,
    `1s glide error ${totalM.toFixed(2)} m must be < 100 m (measured ~0.23 m at this fixture; ` +
      `horiz ${horizM.toFixed(2)} m, alt ${altErrM.toFixed(2)} m) — the glide must be honest physics`,
  );
});

// --------------------------------------------------------------------------
// analyzePopulation
// --------------------------------------------------------------------------

test('analyzePopulation: near-earth / deep-space / missing-elements breakdown', () => {
  const pop = [
    gp(1), // complete, near-earth
    gp(2), // complete, deep-space (via stub)
    gp(3, { meanMotion: null }), // missing an element
  ];
  const c = analyzePopulation(pop, { isDeepSpace: (g) => g.noradId === 2 });
  assert.deepEqual(c, { total: 3, nearEarth: 1, deepSpace: 1, noElements: 1 });
});

// --------------------------------------------------------------------------
// message shapes
// --------------------------------------------------------------------------

test('positionsMessage / readyMessage: wire shapes carry stride + honest counts', () => {
  const pop = [gp(1), gp(2)];
  const stub: PropagationDeps = {
    propagate: (g) => (g.noradId === 1 ? { lonDeg: 0, latDeg: 0, altKm: 800 } : null),
    isDeepSpace: (g) => g.noradId === 2,
  };
  const pack = packPositions(pop, 7, stub);
  const pm = positionsMessage(pack, 7);
  assert.equal(pm.type, 'positions');
  assert.equal(pm.stride, SAT_STRIDE);
  assert.equal(pm.timeMs, 7);
  assert.equal(pm.total, 2);
  assert.equal(pm.shown, 1);
  assert.equal(pm.deepSpaceSkipped, 1);
  assert.ok(pm.buf instanceof ArrayBuffer);
  assert.equal(pm.buf.byteLength, 2 * SAT_STRIDE * 4);

  const rm = readyMessage(pop, stub);
  assert.deepEqual(rm, { type: 'ready', total: 2, nearEarth: 1, deepSpace: 1, noElements: 0 });
});

// --------------------------------------------------------------------------
// runtime: createSatWorker with injected post + timers + clock
// --------------------------------------------------------------------------

interface Posted {
  msg: SatWorkerOutbound;
  transfer?: Transferable[];
}

function harness() {
  const posts: Posted[] = [];
  let intervalCb: (() => void) | null = null;
  let intervalMs = 0;
  let cleared = 0;
  const runtime = createSatWorker(
    (msg, transfer) => posts.push({ msg, transfer }),
    {
      now: () => 1000,
      setIntervalFn: (cb, ms) => {
        intervalCb = cb;
        intervalMs = ms;
        return 'HANDLE';
      },
      clearIntervalFn: () => {
        cleared++;
      },
      propagate: (g) => (g.noradId === 1 ? { lonDeg: 5, latDeg: 5, altKm: 700 } : null),
      isDeepSpace: (g) => g.noradId === 2,
    },
  );
  return {
    runtime,
    posts,
    fireInterval: () => intervalCb?.(),
    intervalMs: () => intervalMs,
    cleared: () => cleared,
  };
}

test('runtime: init posts ready; tick posts transferable positions', () => {
  const h = harness();
  h.runtime.handle({ type: 'init', gp: [gp(1), gp(2)] });
  assert.equal(h.posts.length, 1);
  assert.equal(h.posts[0].msg.type, 'ready');
  assert.equal(h.runtime.size(), 2);

  h.runtime.handle({ type: 'tick', timeMs: 42 });
  assert.equal(h.posts.length, 2);
  const pm = h.posts[1].msg as SatPositionsMessage;
  assert.equal(pm.type, 'positions');
  assert.equal(pm.timeMs, 42);
  assert.equal(pm.shown, 1);
  assert.equal(pm.deepSpaceSkipped, 1);
  // buffer is transferred (zero-copy handoff to the main thread).
  assert.deepEqual(h.posts[1].transfer, [pm.buf]);
});

test('runtime: start drives an immediate tick + interval; stop clears it', () => {
  const h = harness();
  h.runtime.handle({ type: 'init', gp: [gp(1)] });
  h.posts.length = 0; // drop the ready

  h.runtime.handle({ type: 'start', hz: 2 });
  // immediate first frame
  assert.equal(h.posts.length, 1);
  assert.equal(h.posts[0].msg.type, 'positions');
  assert.equal(h.intervalMs(), 500); // 1000/2 Hz

  h.fireInterval();
  assert.equal(h.posts.length, 2);
  assert.equal((h.posts[1].msg as SatPositionsMessage).timeMs, 1000); // injected now()

  h.runtime.handle({ type: 'stop' });
  assert.equal(h.cleared(), 1);
  h.runtime.dispose();
  assert.equal(h.cleared(), 1); // idempotent (already stopped)
});

test('runtime: start twice clears the prior interval (no leak)', () => {
  const h = harness();
  h.runtime.handle({ type: 'init', gp: [gp(1)] });
  h.runtime.handle({ type: 'start' });
  h.runtime.handle({ type: 'start' }); // should clear the first
  assert.equal(h.cleared(), 1);
});
