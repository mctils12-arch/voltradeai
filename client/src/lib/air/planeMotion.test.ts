// SELECTED-PLANE MOTION pins (owner 2026-11-19: "the bluish line runs in front
// of the plane and ... it jumps when it catches signal on the plane ... we
// know what speed and heading it going"). One drawn position: reckoned from
// the newest real fix at that fix's own time, blended (never stepped) onto
// each new fix, never backward, marked an estimate past 30 s, held past 120 s.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  MOTION_MAX_CATCHUP_SEC, MOTION_MAX_CORR_FRAC, MOTION_MIN_FWD_FRAC,
  PLANE_ESTIMATED_AFTER_SEC, PLANE_HOLD_AFTER_SEC, PLANE_SLOW_MPS,
  createPlaneMotion, feedRowAgeMs, feedRowFixTimeSec, motionFixFromRow, motionPhase,
  planeMotionNote, reckonLimitSec, retargetPlaneMotion, selectedPlaneCadenceMs, stepPlaneMotion,
  type MotionFix, type PlaneMotion,
} from './planeMotion.js';
import { MAX_AIR_GLIDE_SEC, AIR_GLIDE_STEP_MS } from './airGlide.js';
import { TRACK_ESTIMATED_ALPHA, FT_VERT_STRIDE, buildTailVertices } from './flightTrackLayer.js';
import {
  DEAD_RECKONING_MAX_SEC, DEAD_RECKONING_SHORT_SEC, ESTIMATE_MIN_GS_KT,
} from '../../../../shared/aircraftEstimate.js';

const M = 111_320;
const V = 230; // m/s
const LAT = 36, LON = -97;
const cosLat = Math.cos((LAT * Math.PI) / 180);
/** a fix `eastM` / `northM` metres from the origin */
const fixAt = (eastM: number, northM: number, over: Partial<MotionFix> = {}): MotionFix => ({
  lon: LON + eastM / (M * cosLat), lat: LAT + northM / M, altM: 10_000, headingDeg: 90, speedMs: V, onGround: false, ...over,
});
/** metres east / north of the origin */
const en = (p: { lon: number; lat: number }): [number, number] => [(p.lon - LON) * M * cosLat, (p.lat - LAT) * M];
const FRAME_MS = 1000 / 60;
/** run frames from t0 to t1 (ms), returning each frame's [t, east, north] */
function run(m: PlaneMotion, t0: number, t1: number): Array<[number, number, number]> {
  const out: Array<[number, number, number]> = [];
  for (let t = t0; t <= t1 + 1e-6; t += FRAME_MS) {
    const o = stepPlaneMotion(m, t);
    const [e, n] = en(o);
    out.push([t, e, n]);
  }
  return out;
}
/** largest speed (m/s) between consecutive frames, and the smallest eastward step speed */
function speeds(fr: Array<[number, number, number]>): { max: number; minEast: number } {
  let max = 0, minEast = Infinity;
  for (let i = 1; i < fr.length; i++) {
    const dt = (fr[i][0] - fr[i - 1][0]) / 1000;
    const de = fr[i][1] - fr[i - 1][1], dn = fr[i][2] - fr[i - 1][2];
    max = Math.max(max, Math.hypot(de, dn) / dt);
    minEast = Math.min(minEast, de / dt);
  }
  return { max, minEast };
}

test('the limits are the server estimate\'s own: 30 s live, 120 s reckoned, nothing under 40 kt past 30 s', () => {
  assert.equal(PLANE_ESTIMATED_AFTER_SEC, DEAD_RECKONING_SHORT_SEC);
  assert.equal(PLANE_HOLD_AFTER_SEC, DEAD_RECKONING_MAX_SEC);
  assert.equal(PLANE_ESTIMATED_AFTER_SEC, 30);
  assert.equal(PLANE_HOLD_AFTER_SEC, 120);
  assert.ok(Math.abs(PLANE_SLOW_MPS - ESTIMATE_MIN_GS_KT * 0.514444) < 1e-9);
  assert.ok(PLANE_ESTIMATED_AFTER_SEC > MAX_AIR_GLIDE_SEC, 'the selected plane no longer freezes at the 25 s cap of the other planes');
  assert.equal(reckonLimitSec(V, true), 120);
  assert.equal(reckonLimitSec(PLANE_SLOW_MPS - 0.1, true), 30, 'slow: the short window only');
  assert.equal(reckonLimitSec(V, false), 0, 'no speed or track: never extrapolated');
  assert.equal(reckonLimitSec(null, true), 0);
});

test('with no new fix the plane keeps moving, straight ahead at its last speed and track, for 60 s and more', () => {
  const m = createPlaneMotion(fixAt(0, 0), 0, 0);
  for (const sec of [1, 10, 25, 26, 40, 60, 119]) {
    const o = stepPlaneMotion(m, sec * 1000);
    const [e, n] = en(o);
    assert.ok(Math.abs(e - V * sec) < 0.5, `${sec} s: ${e.toFixed(1)} m east, expected ${V * sec}`);
    assert.ok(Math.abs(n) < 0.5, `${sec} s: no turn is invented (north ${n.toFixed(2)} m)`);
  }
});

test('marked an estimate once the last real fix is over 30 s old; live before', () => {
  const m = createPlaneMotion(fixAt(0, 0), 0, 0);
  assert.equal(stepPlaneMotion(m, 5_000).phase, 'live');
  assert.equal(stepPlaneMotion(m, 30_000).phase, 'live');
  const o = stepPlaneMotion(m, 30_100);
  assert.equal(o.phase, 'estimated');
  assert.ok(Math.abs(o.ageSec - 30.1) < 1e-6, 'the age of the last real fix is reported');
  assert.equal(stepPlaneMotion(m, 119_000).phase, 'estimated');
  assert.equal(motionPhase(31, true, 120), 'estimated');
  assert.equal(motionPhase(31, false, 0), 'held', 'no speed or track: nothing to estimate with');
});

test('past 120 s it HOLDS at the 120 s point: never further, never back to the fix', () => {
  const m = createPlaneMotion(fixAt(0, 0), 0, 0);
  const at120 = en(stepPlaneMotion(m, 120_000));
  const o = stepPlaneMotion(m, 121_000);
  assert.equal(o.phase, 'held');
  for (const sec of [121, 180, 600, 3600]) {
    const [e, n] = en(stepPlaneMotion(m, sec * 1000));
    assert.ok(Math.abs(e - at120[0]) < 1e-6 && Math.abs(n - at120[1]) < 1e-6, `${sec} s: held`);
    assert.ok(Math.abs(e - V * 120) < 0.5, 'exactly 120 s of travel past the last real fix');
  }
});

test('a slow aircraft (under 40 kt) is not extrapolated past 30 s', () => {
  const m = createPlaneMotion(fixAt(0, 0, { speedMs: 10 }), 0, 0);
  const [e30] = en(stepPlaneMotion(m, 30_000));
  const o = stepPlaneMotion(m, 60_000);
  assert.ok(Math.abs(en(o)[0] - e30) < 1e-6, 'held at the 30 s point');
  assert.equal(o.phase, 'held');
});

test('no speed, no track or on the ground: drawn at the real fix, never moved', () => {
  for (const over of [{ speedMs: null }, { headingDeg: null }, { onGround: true }] as Array<Partial<MotionFix>>) {
    const m = createPlaneMotion(fixAt(100, 50, over), 0, 0);
    const o = stepPlaneMotion(m, 20_000);
    const [e, n] = en(o);
    assert.ok(Math.abs(e - 100) < 1e-6 && Math.abs(n - 50) < 1e-6, JSON.stringify(over));
    assert.equal(o.reckoned, false);
    assert.equal(o.phase, 'live');
    assert.equal(stepPlaneMotion(m, 31_000).phase, 'held');
  }
});

test('a plane selected when its fix is already past the limit is drawn where it was seen, not 2 min ahead', () => {
  const m = createPlaneMotion(fixAt(0, 0), -150_000, 0); // the fix is 150 s old
  const o = stepPlaneMotion(m, 1_000);
  const [e] = en(o);
  assert.ok(Math.abs(e) < 1e-6, 'not extrapolated');
  assert.equal(o.phase, 'held');
  assert.equal(o.reckoned, false);
  assert.match(planeMotionNote(o) ?? '', /last real position/);
  // a fresh fix later is reckoned normally
  retargetPlaneMotion(m, fixAt(40_000, 0), 2_000, 2_000);
  assert.equal(stepPlaneMotion(m, 2_000).reckoned, true);
});

test('a row older than the 25 s cap of the other planes is reckoned from its OWN time (the line no longer leads the plane)', () => {
  // the feed row is 40 s old when the plane is selected; the one-plane lookup's fix is 5 s old.
  // Both describe the same straight flight, so both reckon to the same point.
  const now = 100_000;
  const fromRow = createPlaneMotion(fixAt(0, 0), now - 40_000, now);
  const fromLookup = createPlaneMotion(fixAt(V * 35, 0), now - 5_000, now);
  const a = en(stepPlaneMotion(fromRow, now)), b = en(stepPlaneMotion(fromLookup, now));
  assert.ok(Math.abs(a[0] - b[0]) < 0.5, `row ${a[0].toFixed(1)} m vs lookup ${b[0].toFixed(1)} m`);
  assert.ok(Math.abs(a[0] - V * 40) < 0.5);
});

test('a new fix is a target, not an assignment: no frame moves faster than 1.75x ground speed', () => {
  const m = createPlaneMotion(fixAt(0, 0), 0, 0);
  const before = run(m, 0, 10_000);
  // truth is 150 m further along and 80 m to the left of the reckoning
  retargetPlaneMotion(m, fixAt(V * 10 + 150, 80), 10_000, 10_000);
  const after = run(m, 10_000 + FRAME_MS, 14_000);
  const all = before.slice(-2).concat(after);
  const s = speeds(all);
  assert.ok(s.max <= (1 + MOTION_MAX_CORR_FRAC) * V + 1, `max ${s.max.toFixed(1)} m/s`);
  assert.ok(s.minEast > 0, 'always forward');
  // the first frame after the fix moves like the frame before it: no step
  const stepBefore = before[before.length - 1][1] - before[before.length - 2][1];
  const stepAfter = after[0][1] - before[before.length - 1][1];
  assert.ok(Math.abs(stepAfter - stepBefore) < 0.5, `step before ${stepBefore.toFixed(2)} m, after ${stepAfter.toFixed(2)} m`);
  // and it converges on the new reckoning
  const end = after[after.length - 1];
  assert.ok(Math.abs(end[1] - (V * 14 + 150)) < 1 && Math.abs(end[2] - 80) < 1, `ends ${end[1].toFixed(1)}, ${end[2].toFixed(1)}`);
  assert.equal(stepPlaneMotion(m, 14_000).correcting, false, 'settled');
});

test('over-run: a fix BEHIND the drawn plane never moves it backward; it slows and the real position catches up', () => {
  const m = createPlaneMotion(fixAt(0, 0), 0, 0);
  run(m, 0, 20_000);
  // the plane had slowed: truth is 900 m behind where it is drawn
  retargetPlaneMotion(m, fixAt(V * 20 - 900, 0), 20_000, 20_000);
  const fr = run(m, 20_000 + FRAME_MS, 32_000);
  const s = speeds(fr);
  assert.ok(s.minEast >= MOTION_MIN_FWD_FRAC * V - 1, `slowest forward speed ${s.minEast.toFixed(1)} m/s (must stay >= ${MOTION_MIN_FWD_FRAC * V})`);
  assert.ok(s.max <= V + 1, 'never faster than its own speed while truth catches up');
  const end = fr[fr.length - 1];
  assert.ok(Math.abs(end[1] - (V * 32 - 900)) < 1, 'caught up');
});

test('re-acquired after a long silence: a large gap is closed quickly but continuously, never in one step', () => {
  const m = createPlaneMotion(fixAt(0, 0), 0, 0);
  run(m, 0, 2_000);
  const gap = 12_000; // metres ahead
  retargetPlaneMotion(m, fixAt(V * 2 + gap, 0), 2_000, 2_000);
  const fr = run(m, 2_000 + FRAME_MS, 2_000 + (MOTION_MAX_CATCHUP_SEC + 3) * 1000);
  const s = speeds(fr);
  assert.ok(s.max <= V + gap / MOTION_MAX_CATCHUP_SEC + 1, `max ${s.max.toFixed(0)} m/s`);
  const biggestStep = Math.max(...fr.slice(1).map((f, i) => f[1] - fr[i][1]));
  assert.ok(biggestStep < gap / 20, `largest single-frame move ${biggestStep.toFixed(0)} m of a ${gap} m gap`);
  const end = fr[fr.length - 1];
  assert.ok(Math.abs(end[1] - (V * (2 + MOTION_MAX_CATCHUP_SEC + 3) + gap)) < 2, 'arrived');
});

test('selecting a plane starts where its icon is drawn and closes on the reckoned position', () => {
  const now = 50_000;
  // the feed drew it 10 s past its row; the row is really 40 s old
  const m = createPlaneMotion(fixAt(0, 0), now - 40_000, now, { lon: LON + (V * 10) / (M * cosLat), lat: LAT });
  const first = en(stepPlaneMotion(m, now));
  assert.ok(Math.abs(first[0] - V * 10) < 1e-6, 'the first frame is exactly where the icon was');
  const fr = run(m, now + FRAME_MS, now + 8_000);
  assert.ok(speeds(fr).minEast > 0);
  assert.ok(Math.abs(fr[fr.length - 1][1] - V * 48) < 2, 'on the reckoned position within 8 s');
});

test('heading and altitude ease to a new fix; heading takes the shortest arc', () => {
  const m = createPlaneMotion(fixAt(0, 0, { headingDeg: 350, altM: 3000 }), 0, 0);
  retargetPlaneMotion(m, fixAt(10, 0, { headingDeg: 10, altM: 3100 }), 100, 100);
  const o1 = stepPlaneMotion(m, 150);
  assert.ok(o1.headingDeg! > 350 || o1.headingDeg! < 10, `through north, not the long way round (${o1.headingDeg})`);
  assert.ok(o1.altM! > 3000 && o1.altM! < 3100, 'altitude eases, never steps');
  const o2 = stepPlaneMotion(m, 3_000);
  assert.ok(Math.abs(o2.headingDeg! - 10) < 0.01);
  assert.ok(Math.abs(o2.altM! - 3100) < 0.01);
});

test('stepping twice at the same instant integrates nothing; a clock running backward is ignored; a long stall settles', () => {
  const m = createPlaneMotion(fixAt(0, 0), 0, 0);
  retargetPlaneMotion(m, fixAt(V + 200, 0), 1_000, 1_000);
  const a = stepPlaneMotion(m, 1_100);
  const b = stepPlaneMotion(m, 1_100);
  assert.deepEqual(a, b);
  const c = stepPlaneMotion(m, 1_050); // the frame timestamp can trail performance.now()
  assert.deepEqual(c, a);
  const d = stepPlaneMotion(m, 60_000);
  assert.equal(d.correcting, false);
});

test('the card sentence: nothing while live, "estimate" when estimated, "held" when held', () => {
  assert.equal(planeMotionNote({ phase: 'live', reckoned: true, limitSec: 120 }), null);
  const est = planeMotionNote({ phase: 'estimated', reckoned: true, limitSec: 120 })!;
  assert.match(est, /estimate/);
  assert.match(est, /No turn is assumed/);
  assert.match(est, /faded/);
  const held = planeMotionNote({ phase: 'held', reckoned: true, limitSec: 120 })!;
  assert.match(held, /held/i);
  assert.match(held, /2 min/);
  assert.match(planeMotionNote({ phase: 'held', reckoned: true, limitSec: 30 })!, /30 s/);
  assert.match(planeMotionNote({ phase: 'held', reckoned: false, limitSec: 0 })!, /last real position/);
});

test('refresh cadence: every frame only where a step would be seen; the terrain-saturation pace is kept', () => {
  const S = AIR_GLIDE_STEP_MS;
  assert.equal(selectedPlaneCadenceMs({ stepVisible: true, correcting: false, overloaded: false, terrain: false }, S), 0);
  assert.equal(selectedPlaneCadenceMs({ stepVisible: false, correcting: true, overloaded: false, terrain: true }, S), 0);
  assert.equal(selectedPlaneCadenceMs({ stepVisible: false, correcting: false, overloaded: false, terrain: false }, S), S);
  assert.equal(selectedPlaneCadenceMs({ stepVisible: true, correcting: true, overloaded: true, terrain: false }, S), S, 'overloaded: never per frame');
  assert.equal(selectedPlaneCadenceMs({ stepVisible: true, correcting: true, overloaded: true, terrain: true }, S), S * 3, 'the ~1.1 Hz paced exemption');
});

test('feed row age and fix time: the row is as old as the feed says, not "true now"', () => {
  const recv = 1_800_000_000_000;
  // worldwide: the row's age at the server clock + time since arrival (no device-vs-server comparison)
  assert.equal(feedRowAgeMs({ global: true, recvWallMs: recv }, { seen_pos: 40 }, recv + 3_000), 43_000);
  assert.equal(feedRowAgeMs({ global: true, recvWallMs: recv }, { seen_pos: null }, recv), null);
  assert.equal(feedRowFixTimeSec({ global: true, at: recv }, { seen_pos: 40 }), recv / 1000 - 40);
  assert.equal(feedRowFixTimeSec({ global: true, at: recv }, {}), null);
  // area: snapshot age + the row's age in the snapshot + time since arrival
  const snap = recv / 1000 - 6;
  assert.equal(feedRowAgeMs({ time: snap, recvWallMs: recv }, { seen_pos: 2 }, recv + 1_000), 9_000);
  assert.equal(feedRowAgeMs({ time: snap, recvWallMs: recv }, {}, recv), 6_000);
  assert.equal(feedRowFixTimeSec({ time: snap }, { seen_pos: 2 }), snap - 2);
  // clocks that do not plausibly agree (a device clock off by an hour, a fixture's `time: 1`): the snapshot age is not used
  assert.equal(feedRowAgeMs({ time: 1, recvWallMs: recv }, { seen_pos: 2 }, recv), 2_000);
  assert.equal(feedRowAgeMs({ time: 1, recvWallMs: recv }, {}, recv), null);
  assert.equal(feedRowAgeMs({ time: recv / 1000 + 3600, recvWallMs: recv }, { seen_pos: 1 }, recv), 1_000);
  assert.deepEqual(motionFixFromRow({ lon: 1, lat: 2, altitude_m: 900, heading: 10, velocity_ms: 50, on_ground: true }),
    { lon: 1, lat: 2, altM: 0, headingDeg: 10, speedMs: 50, onGround: true });
});

test('an estimated tail is drawn faded: every vertex alpha is scaled, nothing else changes', () => {
  const tail = { fromMercX: 0.2, fromMercY: 0.4, fromAltM: 9000, fromGroundZ: 0, toMercX: 0.2001, toMercY: 0.4, toAltM: 9000, toGroundZ: 0, altMin: 0, altMax: 10000 };
  const real = buildTailVertices(tail, 1);
  const est = buildTailVertices({ ...tail, estimated: true }, 1);
  assert.equal(est.length, real.length);
  assert.ok(real.length > 0);
  for (let i = 0; i < real.length; i++) {
    const isAlpha = i % FT_VERT_STRIDE === FT_VERT_STRIDE - 1;
    if (isAlpha) assert.ok(Math.abs(est[i] - real[i] * TRACK_ESTIMATED_ALPHA) < 1e-6);
    else assert.equal(est[i], real[i]);
  }
  assert.ok(TRACK_ESTIMATED_ALPHA > 0.3 && TRACK_ESTIMATED_ALPHA < 0.7, 'visibly fainter, still readable');
});

// ── wiring (source text): the page draws the selected plane from this model only ──
const page = readFileSync(new URL('../../pages/datamap.tsx', import.meta.url), 'utf-8');

test('wiring: the tail, the plane, and the follow camera all read the one motion model', () => {
  const tail = page.slice(page.indexOf('const updateFlightTail = '), page.indexOf('const updateFlightMarker = '));
  assert.match(tail, /stepPlaneMotion\(lv\.motion/);
  assert.match(tail, /liveTailEndRef\.current = \{ lon: lo, lat: la/);
  assert.match(tail, /toMercX: m\.x, toMercY: m\.y/);
  const marker = page.slice(page.indexOf('const updateFlightMarker = '), page.indexOf('const paintFollowedTrail = '));
  assert.match(marker, /lon = tailEnd \? tailEnd\.lon : lv\.motion\.lon/, 'the plane is drawn at the tail\'s own end');
  assert.doesNotMatch(marker, /below3d/, 'at every zoom, not only below the 3D hand-off');
  assert.match(marker, /syncHiddenAircraftIcon\(map, fid\)/);
  assert.match(page, /followTarget=\{\(\) => \{[\s\S]{0,2500}stepPlaneMotion\(lv\.motion/);
  assert.doesNotMatch(page, /airGlideDtSec|tailGlideDtSec/, 'no second glide clock for the selected plane');
});

test('wiring: the selected plane is hidden from BOTH feed renderers, and advances on the shared frame loop', () => {
  assert.match(page, /const hide = hiddenIconHexRef\.current;\s*\n\s*const shown = hide \? rowsNow\.filter/);
  const sync = page.slice(page.indexOf('const syncHiddenAircraftIcon = '), page.indexOf('const updateFlightTail = '));
  assert.match(sync, /map\.setFilter\("aircraft-sym", not\)/);
  assert.match(sync, /airRebuildRef\.current\?\.\(\)/);
  assert.match(page, /frameCore\(\)\.register\(\(_dtMs, nowMs\) => \{[\s\S]{0,1200}updateFlightTail\(nowMs\);\s*\n\s*\}, PRIORITY\.SIM, \{ label: "selected-plane" \}\)/);
  assert.match(page, /offPlaneFrame\(\);/, 'unregistered on teardown (Law IV)');
});

test('wiring: every source of a fix is stamped with its own age, and a re-delivered fix keeps its first anchor', () => {
  const ingest = page.slice(page.indexOf('const ingestSelectedLiveFix = '), page.indexOf('const tailInnerOnGround = '));
  assert.match(ingest, /retargetPlaneMotion\(cur\.motion, mfix, anchorMs, nowMs\)/);
  assert.match(ingest, /const sameFix = cur\.motion\.vel != null && fix\.lo === cur\.fix\.lo && fix\.la === cur\.fix\.la/);
  assert.doesNotMatch(page, /ageMs: 0\b/, 'no feed row is stamped "true now"');
  assert.match(page, /ingestSelectedLiveFix\(fid, fix, live, \{ ageMs: feedRowAgeMs\(meta, live, Date\.now\(\)\) \?\? 0 \}\)/);
  assert.match(page, /ingestSelectedLiveFix\(fid, fix, r, \{ ageMs: Math\.max\(0, Date\.now\(\) - f\.fixMs\)/);
  assert.match(page, /data-testid="plane-estimate"/);
});
