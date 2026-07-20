// Damped orbit camera rig — pins the handoff's tuned control rates and the
// MapLibre mapping (design_handoff_flight_track_3d, 2026-07-20). Sign
// conventions were verified EMPIRICALLY against the approved prototype
// (headless-Chromium probe, session 2026-07-20): dial clockwise-drag →
// bearing decreases; canvas drag-right → bearing decreases; drag-down →
// toward top-down; tilt-up → toward grazing; zoom-in shrinks distance.
// The prototype's N/S pan axis moved opposite its own "Pan forward" label
// and README ("up = away from camera") while E/W matched — a prototype
// sign slip; this rig implements the DOCUMENTED intent (↑ = away).
// Run: npx tsx --test client/src/lib/cameraRig.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  stepRig,
  clampPitch,
  camDistanceMeters,
  panDelta,
  bearingDeltaToNorth,
  dialBearing,
  zoomGoalStep,
  makeHoldRegistry,
  RIG_DAMPING_PER_S,
  RIG_ROTATE_DEG_S,
  RIG_TILT_DEG_S,
  RIG_ZOOM_LEVELS_S,
  RIG_WHEEL_ZOOM_PER_DY,
  RIG_PITCH_MIN,
  RIG_PITCH_MAX,
  RIG_BUTTON_MAX_ZOOM,
  RIG_ZOOM_MAX_AHEAD,
  type Rig,
} from './cameraRig.js';

const near = (a: number, b: number, tol: number, msg: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} (${a} vs ${b})`);

const mkRig = (): Rig => ({
  cur: { bearing: 0, pitch: 30, zoom: 10, lng: -100, lat: 40 },
  goal: { bearing: 0, pitch: 30, zoom: 10, lng: -100, lat: 40 },
});

test('damping law: value moves toward goal by 1 − e^(−7·dt)', () => {
  const rig = mkRig();
  rig.goal.bearing = 90;
  stepRig(rig, 0.1);
  const k = 1 - Math.exp(-RIG_DAMPING_PER_S * 0.1);
  near(rig.cur.bearing, 90 * k, 1e-9, 'exact prototype damping step');
  assert.equal(RIG_DAMPING_PER_S, 7, 'damping factor 7/s is the tuned value');
});

test('stepRig reports settling (rAF keep-alive contract)', () => {
  const rig = mkRig();
  rig.goal.zoom = 11;
  assert.equal(stepRig(rig, 0.016), true, 'still moving');
  for (let i = 0; i < 400; i++) stepRig(rig, 0.032);
  assert.equal(stepRig(rig, 0.032), false, 'settled');
  near(rig.cur.zoom, 11, 0.01, 'converged on the goal');
});

test('tuned rates translate the prototype exactly', () => {
  near(RIG_ROTATE_DEG_S, 1.5 * (180 / Math.PI), 1e-9, 'rotate 1.5 rad/s');
  near(RIG_TILT_DEG_S, 0.9 * (180 / Math.PI), 1e-9, 'tilt 0.9 rad/s');
  near(RIG_ZOOM_LEVELS_S, 1.6 / Math.LN2, 1e-9, 'dist ×e^(1.6·dt) ⇔ 2.308 zoom levels/s');
  near(RIG_WHEEL_ZOOM_PER_DY, 0.0011 / Math.LN2, 1e-12, 'wheel ×e^(0.0011·ΔY)');
});

test('pitch clamp is 2°–84° (grazing tilt inside MapLibre\'s supported envelope — the prototype\'s 88° froze production: tile-cover explosion past ~85°)', () => {
  assert.equal(RIG_PITCH_MIN, 2);
  assert.equal(RIG_PITCH_MAX, 84);
  assert.equal(clampPitch(120), 84);
  assert.equal(clampPitch(-5), 2);
});

test('camera distance halves per zoom level and scales with viewport height', () => {
  const d10 = camDistanceMeters(10, 40, 800);
  const d11 = camDistanceMeters(11, 40, 800);
  near(d10 / d11, 2, 1e-9, 'zoom +1 ⇔ distance ×0.5 (exponential zoom is lossless)');
  near(camDistanceMeters(10, 40, 1600) / d10, 2, 1e-9, 'distance tracks viewport height');
});

test('panDelta: ↑ pans away from the camera (bearing-relative), 0.9×dist/s', () => {
  // bearing 0 (facing north): up = north, right = east
  const up = panDelta(0, 0, -1, 10, 40, 800, 1);
  assert.ok(up.dLat > 0 && Math.abs(up.dLng) < 1e-12, '↑ at bearing 0 → north');
  const right = panDelta(0, 1, 0, 10, 40, 800, 1);
  assert.ok(right.dLng > 0 && Math.abs(right.dLat) < 1e-12, '→ at bearing 0 → east');
  // bearing 90 (facing east): up = east
  const upE = panDelta(90, 0, -1, 10, 40, 800, 1);
  assert.ok(upE.dLng > 0 && Math.abs(upE.dLat) < 1e-9, '↑ at bearing 90 → east');
  // magnitude: 0.9 × camDistance per second along the ground
  const dist = camDistanceMeters(10, 40, 800);
  near(up.dLat * 111319.49, dist * 0.9, dist * 0.01, 'rate = 0.9×distance/s');
});

test('bearingDeltaToNorth takes the shortest way home', () => {
  near(bearingDeltaToNorth(10), -10, 1e-9, 'small east → rotate back west');
  near(bearingDeltaToNorth(350), 10, 1e-9, 'wraps the short way');
  near(bearingDeltaToNorth(-190), -170, 1e-9, 'normalizes first');
  near(bearingDeltaToNorth(720), 0, 1e-9, 'already north (mod 360)');
});

test('zoomGoalStep: runaway-zoom regression (live report 2026-07-20)', () => {
  // a held button cannot pile the goal more than the ahead-cap past the camera
  const g1 = zoomGoalStep(10, 12, 0.5, 0, 22);
  near(g1, 10 + RIG_ZOOM_MAX_AHEAD, 1e-9, 'goal capped just ahead of the camera');
  // and never past the imagery ceiling, whatever the map allows
  const g2 = zoomGoalStep(17.4, 17.4, 5, 0, 22);
  near(g2, RIG_BUTTON_MAX_ZOOM, 1e-9, 'button zoom stops at the imagery ceiling');
  assert.ok(RIG_BUTTON_MAX_ZOOM <= 18, 'ceiling sits inside real imagery coverage');
  // zoom-out respects the map floor and the ahead-cap symmetrically
  const g3 = zoomGoalStep(3.7, 3.7, -9, 3.6, 22);
  near(g3, 3.6, 1e-9, 'floor respected');
  const g4 = zoomGoalStep(10, 8, -0.5, 0, 22);
  near(g4, 10 - RIG_ZOOM_MAX_AHEAD, 1e-9, 'downward coast equally bounded');
});

test('hold registry: release by KEY survives re-renders (latch regression — "rotate would not stop", live 2026-07-20)', () => {
  const reg = makeHoldRegistry();
  let spins = 0;
  // render 1 creates closure A; the press stores it under the button key
  const closureA = () => { spins += 1; };
  reg.press('rotl', closureA);
  reg.tick(0.016);
  assert.equal(spins, 1, 'held button ticks');
  // a data tick re-renders the component → handlers become closure B;
  // the release still lands because it targets the KEY, not the closure
  reg.release('rotl');
  reg.tick(0.016);
  assert.equal(spins, 1, 'released button NEVER ticks again');
  assert.equal(reg.size(), 0, 'registry empty after release');
  // global failsafe: clear() stops everything (window pointerup/blur/hidden)
  reg.press('zoomin', () => { spins += 10; });
  reg.press('pann', () => { spins += 100; });
  reg.clear();
  reg.tick(0.016);
  assert.equal(spins, 1, 'failsafe clear stops all holds');
});

test('dialBearing: clockwise screen drag decreases bearing (prototype-verified)', () => {
  // pointer sweeps +0.5 rad clockwise (y-down screen angle increases)
  const b = dialBearing(45, 0.2, 0.7);
  near(b, 45 - 0.5 * (180 / Math.PI), 1e-9, 'gheading = grab − delta, exactly');
});
