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
  trimToCurrentFlight,
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

test('non-held fixes carry held:false — the default is "real data" (round-22 held-flag follow-up)', () => {
  const raw: RawTrackPoint[] = [
    { t: 0, la: 40, lo: -100, al: 2000 },
    { t: 60, la: 40.05, lo: -100, al: 2400 },
  ];
  const { samples } = buildTrackSamples(raw);
  assert.ok(samples.every((s) => s.held === false), 'no fix flagged held → no sample is held');
});

test('held propagates from mergeTrackWithCrumbs-style fixes through buildTrackSamples and sampleAt (round-22 held-flag follow-up)', () => {
  const raw: RawTrackPoint[] = [
    { t: 0, la: 40, lo: -100, al: 2000 },                         // real broadcast
    { t: 60, la: 40.05, lo: -100, al: 2000, held: true },         // ALTITUDE HOLD carried forward
    { t: 120, la: 40.1, lo: -100, al: 2400 },                     // real broadcast resumes
  ];
  const { samples } = buildTrackSamples(raw);
  // the raw held fix itself is held
  const rawHeld = samples.find((s) => s.t === 60);
  assert.equal(rawHeld?.held, true, 'the raw held fix is flagged held');
  // interpolated points bracketing a held fix inherit held — a mid-segment
  // reading is not fresh data just because one endpoint is
  const beforeMid = samples.find((s) => s.t > 0 && s.t < 60);
  const afterMid = samples.find((s) => s.t > 60 && s.t < 120);
  assert.equal(beforeMid?.held, true, 'segment leading into the held fix is flagged held');
  assert.equal(afterMid?.held, true, 'segment leading out of the held fix is flagged held');
  // fully real segments (both endpoints real) are never flagged
  const realOnly = buildTrackSamples([
    { t: 0, la: 40, lo: -100, al: 2000 },
    { t: 60, la: 40.05, lo: -100, al: 2400 },
  ]).samples;
  assert.ok(realOnly.every((s) => !s.held), 'no held endpoint anywhere → nothing flagged');
  // sampleAt (playback/scrub) also reports held when either bracketing
  // sample is held, so the playhead readout never overstates freshness
  const scrub = sampleAt(samples, 45)!;
  assert.equal(scrub.held, true, 'sampleAt inherits held from its bracketing samples');
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

test('trimToCurrentFlight: a time hole splits flights — the newest block wins', () => {
  const mk = (t: number, al: number | null): { t: number; la: number; lo: number; al: number | null } =>
    ({ t, la: 39 + t / 1e6, lo: -107, al });
  const old = [mk(0, 3000), mk(300, 3200), mk(600, 3400)];
  const fresh = [mk(600 + 46 * 60, 2000), mk(600 + 46 * 60 + 300, 2500)];
  const out = trimToCurrentFlight([...old, ...fresh]);
  assert.equal(out.length, 2, 'only the newest contiguous block remains');
  assert.equal(out[0].t, fresh[0].t);
});

test('trimToCurrentFlight: a parked dwell (no broadcast altitude) splits flights, keeping one ground lead-in fix', () => {
  const mk = (t: number, al: number | null) => ({ t, la: 39, lo: -107, al });
  const flight1 = [mk(0, 5000), mk(300, 5200)];
  // parked 20 min: fixes every 5 min, no altitude
  const parked = [mk(600, null), mk(900, null), mk(1200, null), mk(1500, null), mk(1800, null)];
  const flight2 = [mk(2100, 800), mk(2400, 1500)];
  const out = trimToCurrentFlight([...flight1, ...parked, ...flight2]);
  assert.equal(out[0].t, 1800, 'starts at the dwell\'s last fix (ground lead-in)');
  assert.equal(out.length, 3, 'lead-in + the new flight');
});

test('trimToCurrentFlight: never-airborne track keeps only the trailing hour; short/clean tracks pass through', () => {
  const mk = (t: number, al: number | null) => ({ t, la: 39, lo: -107, al });
  const parkedAllDay: any[] = [];
  for (let t = 0; t <= 6 * 3600; t += 300) parkedAllDay.push(mk(t, null));
  const out = trimToCurrentFlight(parkedAllDay);
  assert.ok((out[out.length - 1].t as number) - (out[0].t as number) <= 3600, 'trailing hour only');
  // clean continuous flight untouched (same array back)
  const clean = [mk(0, 3000), mk(300, 3100), mk(600, 3200)];
  assert.equal(trimToCurrentFlight(clean), clean, 'no-trim returns the input array');
});
