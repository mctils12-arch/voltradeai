// Flight track model — pins the handoff contract (design_handoff_flight_
// track_3d, 2026-07-20): densification spacing, honest altitude gaps,
// derived gs/vs, the exact teal→blue→violet ramp stops, playback sampling.
// Run: npx tsx --test client/src/lib/air/trackModel.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildTrackSamples,
  altRampColor,
  sampleAt,
  headingAt,
  distMeters,
  TRACK_DENSIFY_M,
  TRACK_MAX_SAMPLES,
  RAMP_LO,
  RAMP_MID,
  RAMP_HI,
  CURTAIN_ALPHA,
  CURTAIN_BOTTOM_MUL,
  CURTAIN_BELOW_TERRAIN_M,
  TRACE_ABOVE_TERRAIN_M,
  type RawTrackPoint,
} from './trackModel.js';

const near = (a: number, b: number, tol: number, msg: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} (${a} vs ${b})`);

test('densification: consecutive samples sit ~TRACK_DENSIFY_M apart, linear interp only', () => {
  // two fixes ~12km apart, 60s apart → ~100 densified steps
  const raw: RawTrackPoint[] = [
    { t: 1000, la: 39.0, lo: -105.0, al: 3000 },
    { t: 1060, la: 39.0, lo: -104.861, al: 3600 }, // ~12 km east
  ];
  const { samples } = buildTrackSamples(raw);
  assert.ok(samples.length > 50, `densified (${samples.length} samples)`);
  for (let i = 1; i < samples.length; i++) {
    const d = distMeters(samples[i - 1].lat, samples[i - 1].lon, samples[i].lat, samples[i].lon);
    assert.ok(d < TRACK_DENSIFY_M * 2.2, `spacing ${d}m at ${i} under ~2x target`);
  }
  // linear altitude: midpoint sample ≈ 3300m (no smoothing/no curves)
  const mid = samples[Math.floor(samples.length / 2)];
  near(mid.altM, 3300, 40, 'midpoint altitude is the linear interpolant');
  // endpoints are the REAL fixes, exactly
  assert.equal(samples[0].altM, 3000);
  assert.equal(samples[samples.length - 1].altM, 3600);
});

test('sample cap: very long tracks degrade spacing instead of exploding geometry', () => {
  const raw: RawTrackPoint[] = [];
  for (let i = 0; i < 100; i++) raw.push({ t: i * 60, la: 30 + i * 0.15, lo: -100, al: 9000 }); // ~1650 km
  const { samples } = buildTrackSamples(raw);
  assert.ok(samples.length <= TRACK_MAX_SAMPLES * 1.1, `capped (${samples.length})`);
});

test('altitude gaps are honest: no-broadcast fixes break the curtain, never invent altitude', () => {
  const raw: RawTrackPoint[] = [
    { t: 0, la: 40, lo: -100, al: 2000 },
    { t: 60, la: 40.05, lo: -100, al: null },  // altitude not broadcast
    { t: 120, la: 40.1, lo: -100, al: 2400 },
  ];
  const { samples, altMin, altMax } = buildTrackSamples(raw);
  const gapSamples = samples.filter((s) => s.gap);
  assert.ok(gapSamples.length > 0, 'gap samples exist');
  for (const s of gapSamples) {
    assert.ok(Number.isNaN(s.altM), 'gap sample carries NaN altitude');
    assert.ok(Number.isFinite(s.lat) && Number.isFinite(s.lon), 'position history still real');
  }
  // ramp domain ignores gaps
  assert.equal(altMin, 2000);
  assert.equal(altMax, 2400);
});

test('derived gs/vs match the prototype derivation on a constant-rate track', () => {
  // due-north at ~120 m/s, climbing 5 m/s
  const raw: RawTrackPoint[] = [];
  for (let i = 0; i < 8; i++) {
    raw.push({ t: i * 30, la: 40 + (i * 30 * 120) / 111319.49, lo: -100, al: 1000 + i * 150 });
  }
  const { samples } = buildTrackSamples(raw);
  const mid = samples[Math.floor(samples.length / 2)];
  near(mid.gsKt, 120 * 1.94384, 6, 'ground speed ≈ 233 kt derived from fixes');
  near(mid.vsFpm, 5 * 3.28084 * 60, 40, 'vertical speed ≈ +984 fpm derived from fixes');
});

test('ramp stops are the exact handoff colors: #38d1c1 → #4da3ff → #a06bff', () => {
  assert.deepEqual(RAMP_LO.map((v) => Math.round(v * 255)), [0x38, 0xd1, 0xc1]);
  assert.deepEqual(RAMP_MID.map((v) => Math.round(v * 255)), [0x4d, 0xa3, 0xff]);
  assert.deepEqual(RAMP_HI.map((v) => Math.round(v * 255)), [0xa0, 0x6b, 0xff]);
  // min → lo, midpoint → mid, max → hi; two linear halves
  assert.deepEqual(altRampColor(0, 0, 1000), RAMP_LO);
  assert.deepEqual(altRampColor(500, 0, 1000), RAMP_MID);
  assert.deepEqual(altRampColor(1000, 0, 1000), RAMP_HI);
  const q = altRampColor(250, 0, 1000); // halfway lo→mid
  near(q[0], (RAMP_LO[0] + RAMP_MID[0]) / 2, 1e-9, 'first half is linear');
  // degenerate domain (flat track) pins to the mid color, never NaN
  assert.deepEqual(altRampColor(500, 500, 500), RAMP_MID);
});

test('curtain constants are the handoff values (34% alpha, ×[.25,.28,.45] bottom, 40m drape, 16m trace)', () => {
  assert.equal(CURTAIN_ALPHA, 0.34);
  assert.deepEqual(CURTAIN_BOTTOM_MUL, [0.25, 0.28, 0.45]);
  assert.equal(CURTAIN_BELOW_TERRAIN_M, 40);
  assert.equal(TRACE_ABOVE_TERRAIN_M, 16);
});

test('sampleAt: time interpolation, clamped ends, gap-aware readouts', () => {
  const raw: RawTrackPoint[] = [
    { t: 0, la: 40, lo: -100, al: 1000 },
    { t: 100, la: 41, lo: -100, al: 2000 },
  ];
  const { samples } = buildTrackSamples(raw);
  const s = sampleAt(samples, 50)!;
  near(s.lat, 40.5, 0.01, 'position interpolates in time');
  near(s.altM, 1500, 20, 'altitude interpolates in time');
  assert.equal(sampleAt(samples, -5)!.t, samples[0].t, 'clamps to start');
  assert.equal(sampleAt(samples, 1e9)!.t, samples[samples.length - 1].t, 'clamps to end');
  assert.equal(sampleAt([], 0), null);
});

test('headingAt follows the track tangent (due-east track → ~90°)', () => {
  const raw: RawTrackPoint[] = [
    { t: 0, la: 40, lo: -100, al: 1000 },
    { t: 100, la: 40, lo: -99, al: 1000 },
  ];
  const { samples } = buildTrackSamples(raw);
  const h = headingAt(samples, 50)!;
  near(h, 90, 1.5, 'east tangent');
});
