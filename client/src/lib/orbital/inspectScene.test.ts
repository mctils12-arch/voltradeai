// INSPECT MODE scene — pure-math tests (node:test, no GL, no DOM). The
// solarView precedent: mount()'s visuals are exercised by the integration
// PR's visual harness; everything mathematical here is hermetic and pinned.
// Frame convention under test: craft local ENU (spherical Earth, geocentric
// lat/lon), scene axes x=E, y=U(up), z=-N; nadir = scene (0,-1,0).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ecefUnit,
  ecefToEnu,
  enuToScene,
  sunDirLocal,
  moonDirLocal,
  earthAngularRadiusRad,
  phaseFractionFromDirs,
  deriveVelocityDirScene,
  meshExtent,
  createOrbitCam,
  camApplyDrag,
  camApplyDolly,
  camStep,
  camEye,
  camDollyBounds,
  wrapAngle,
  mount,
  PITCH_LIMIT_RAD,
  DEFAULT_EXTENT,
  EARTH_RADIUS_M,
  SUN_ANGULAR_RADIUS_RAD,
  INSPECT_PROVENANCE,
  type EnuDir,
} from './inspectScene.js';
import { subsolarPoint, moonState } from '../celestial/ephemeris.js';
import { buildFormMesh } from './model3d.js';

const DEG = Math.PI / 180;
const near = (a: number, b: number, tol: number, msg?: string): void =>
  assert.ok(Math.abs(a - b) <= tol, msg ?? `expected ${a} ≈ ${b} (±${tol})`);
const nearDir = (d: EnuDir, e: number, n: number, u: number, tol: number, msg: string): void => {
  near(d.e, e, tol, `${msg}: e`);
  near(d.n, n, tol, `${msg}: n`);
  near(d.u, u, tol, `${msg}: u`);
};

// ── 1. sun-direction frame math: hand-computed anchor geometries ──────────

test('frame anchor A: sub-point directly below the craft → direction on the zenith axis', () => {
  // Sun through sub-point (0,0), craft at (0,0): Earth-center → sub-point →
  // craft → sun are co-linear, so from the craft the sun sits at the ZENITH
  // (+u). ("Directly below" = the sub-point is the craft's nadir point.)
  const d = ecefToEnu(ecefUnit(0, 0), 0, 0);
  nearDir(d, 0, 0, 1, 1e-12, 'co-located sub-point');
});

test('frame anchor B: subsolar 90° east on the equator → sun due EAST on the horizon', () => {
  // Sub-point (0, 90E): ECEF (0,1,0). Craft at (0,0): E=(0,1,0), N=(0,0,1),
  // U=(1,0,0) → {e:1, n:0, u:0}.
  const d = ecefToEnu(ecefUnit(0, 90), 0, 0);
  nearDir(d, 1, 0, 0, 1e-12, 'sun due east');
  // and 90° WEST mirrors to {e:-1}
  const w = ecefToEnu(ecefUnit(0, -90), 0, 0);
  nearDir(w, -1, 0, 0, 1e-12, 'sun due west');
});

test('frame anchor C: craft at the north pole, sub-point on the equator → due south', () => {
  // Craft (90, 0): E=(0,1,0), N=(-1,0,0), U=(0,0,1). Sub-point (0,0):
  // ECEF (1,0,0) → {e:0, n:-1, u:0} — from the pole every direction is south.
  const d = ecefToEnu(ecefUnit(0, 0), 90, 0);
  nearDir(d, 0, -1, 0, 1e-12, 'south from the pole');
});

test('frame anchor D: solstice-style offset — sub-point 23.44° north of the craft', () => {
  // Craft (0,0), sub-point (23.44, 0): ECEF (cos δ, 0, sin δ) →
  // {e:0, n:sin δ, u:cos δ} (sun tipped toward local north).
  const d = ecefToEnu(ecefUnit(23.44, 0), 0, 0);
  nearDir(d, 0, Math.sin(23.44 * DEG), Math.cos(23.44 * DEG), 1e-12, 'solstice tilt');
});

test('sunDirLocal is the ephemeris sub-point through the frame math (unit; zenith/nadir anchors)', () => {
  const T = Date.UTC(2026, 6, 15, 12, 0, 0);
  const sp = subsolarPoint(T);
  // craft AT the subsolar point → sun at zenith
  const atSub = sunDirLocal(T, sp.latDeg, sp.lonDeg);
  nearDir(atSub, 0, 0, 1, 1e-9, 'craft over subsolar point');
  // craft at the antipode → sun at nadir (midnight directly below)
  const antiLon = sp.lonDeg > 0 ? sp.lonDeg - 180 : sp.lonDeg + 180;
  const atAnti = sunDirLocal(T, -sp.latDeg, antiLon);
  nearDir(atAnti, 0, 0, -1, 1e-9, 'craft at antipode');
  // always a unit vector
  const d = sunDirLocal(T, 33.7, -84.4);
  near(Math.hypot(d.e, d.n, d.u), 1, 1e-9, 'unit length');
});

test('enuToScene: x=E, y=U, z=-N (right-handed GL mapping; nadir = (0,-1,0))', () => {
  const eq = (d: EnuDir, exp: [number, number, number], msg: string): void => {
    const s = enuToScene(d);
    for (let i = 0; i < 3; i++) near(s[i], exp[i], 1e-12, `${msg}[${i}]`);
  };
  eq({ e: 1, n: 0, u: 0 }, [1, 0, 0], 'east → +x');
  eq({ e: 0, n: 0, u: 1 }, [0, 1, 0], 'zenith → +y');
  eq({ e: 0, n: 1, u: 0 }, [0, 0, -1], 'north → -z');
  eq({ e: 0, n: 0, u: -1 }, [0, -1, 0], 'nadir → scene -y');
});

// ── 2. Earth angular size: 2·asin(R/(R+h)) ────────────────────────────────

test('earth angular size: ~140° at LEO (400 km), ~17.4° at GEO — the real values', () => {
  const leo = (2 * earthAngularRadiusRad(400_000)) / DEG;
  near(leo, 140.45, 0.2, `LEO full disc ${leo}°`);
  const geo = (2 * earthAngularRadiusRad(35_786_000)) / DEG;
  near(geo, 17.39, 0.05, `GEO full disc ${geo}°`);
  // monotonic: higher → smaller Earth
  assert.ok(earthAngularRadiusRad(800_000) < earthAngularRadiusRad(400_000));
  // sanity: the formula is asin(R/(R+h)) exactly
  near(earthAngularRadiusRad(400_000), Math.asin(EARTH_RADIUS_M / (EARTH_RADIUS_M + 400_000)), 1e-12);
});

test('sun disc constant: true mean angular size ~0.53° diameter', () => {
  near((2 * SUN_ANGULAR_RADIUS_RAD) / DEG, 0.533, 1e-9);
});

// ── 3. moon direction / size passthrough from the ephemeris ───────────────

test('moon: real ephemeris direction (zenith over the sublunar point) + true angular size', () => {
  const T = Date.UTC(2026, 6, 15, 3, 30, 0);
  const m = moonState(T);
  // craft directly over the sublunar point → moon on the zenith axis
  const local = moonDirLocal(T, m.latDeg, m.lonDeg);
  nearDir(local.dir, 0, 0, 1, 1e-9, 'craft over sublunar point');
  // angular RADIUS is exactly arcmin/2 converted — a passthrough, no invention
  near(local.angularRadiusRad, (m.angularSizeArcmin / 2 / 60) * DEG, 1e-12);
  // plausible physical range for the Moon: 0.24°–0.29° radius
  const radDeg = local.angularRadiusRad / DEG;
  assert.ok(radDeg > 0.24 && radDeg < 0.29, `moon radius ${radDeg}°`);
  // illumination passthrough
  near(local.illuminatedFraction, m.illuminatedFraction, 1e-12);
});

test('phase-from-geometry matches the ephemeris illuminated fraction (shader consistency)', () => {
  // The disc shader lights the moon sphere with the real sun direction; the
  // resulting fraction is (1-cos(separation))/2 — pin that against the
  // ephemeris' own elongation-based fraction at several epochs.
  for (const T of [Date.UTC(2026, 0, 3), Date.UTC(2026, 3, 17, 9), Date.UTC(2026, 9, 28, 18)]) {
    const m = moonState(T);
    const sun = sunDirLocal(T, 0, 0);
    const moon = moonDirLocal(T, 0, 0);
    const geo = phaseFractionFromDirs(moon.dir, sun);
    near(geo, m.illuminatedFraction, 0.08, `epoch ${T}: geometric ${geo} vs ephemeris ${m.illuminatedFraction}`);
  }
});

// ── 4. free-orbit camera math ─────────────────────────────────────────────

test('pitch spans ±89° INCLUDING the nadir side — camera below the craft, Earth at your back', () => {
  const cam = createOrbitCam(1.2);
  camApplyDrag(cam, 0, 1e6); // huge downward-pitch drag
  near(cam.pitch, PITCH_LIMIT_RAD, 1e-12, 'clamped at +89°');
  camApplyDrag(cam, 0, -1e6);
  near(cam.pitch, -PITCH_LIMIT_RAD, 1e-12, 'clamped at -89°');
  // at pitch -89° the EYE is on the nadir side (scene y < 0): looking at the
  // craft means looking AWAY from Earth — the "moon side" pose.
  const eye = camEye(cam);
  assert.ok(eye[1] < -0.9 * cam.dist, 'eye below the craft');
  near(PITCH_LIMIT_RAD / DEG, 89, 1e-9);
});

test('yaw is a full 360°, kept wrapped to [-π, π)', () => {
  const cam = createOrbitCam(1.2);
  for (let i = 0; i < 50; i++) camApplyDrag(cam, 500, 0);
  assert.ok(cam.yaw >= -Math.PI - 1e-9 && cam.yaw < Math.PI + 1e-9, `yaw wrapped: ${cam.yaw}`);
  // ±π are the same direction; the wrap lands both on -π (half-open range)
  near(wrapAngle(3 * Math.PI), -Math.PI, 1e-9);
  near(wrapAngle(-3 * Math.PI), -Math.PI, 1e-9);
  near(wrapAngle(0.5), 0.5, 1e-12);
  near(wrapAngle(2 * Math.PI + 0.25), 0.25, 1e-9);
});

test('dolly bounds: ~2× to ~50× the craft extent, target clamped, no ground constraint needed', () => {
  const b = camDollyBounds(1.2);
  near(b.min, 2.4, 1e-12);
  near(b.max, 60, 1e-12);
  const cam = createOrbitCam(1.2);
  camApplyDolly(cam, 1e9);
  near(cam.distTarget, b.max, 1e-12, 'zoom out stops at 50×');
  camApplyDolly(cam, 1e-9);
  near(cam.distTarget, b.min, 1e-12, 'zoom in stops at 2× — can never dive to the terrain');
  camApplyDolly(cam, NaN); // hostile input ignored
  near(cam.distTarget, b.min, 1e-12);
  camApplyDolly(cam, -1);
  near(cam.distTarget, b.min, 1e-12);
});

test('inertial damping converges: velocities decay to zero, dolly settles exactly on target', () => {
  const cam = createOrbitCam(1.2);
  cam.velYaw = 3; // rad/s flick
  cam.velPitch = -1.5;
  camApplyDolly(cam, 8);
  let moving = true;
  let steps = 0;
  while (moving && steps < 2000) {
    moving = camStep(cam, 16);
    steps++;
  }
  assert.ok(steps < 2000, 'settles (no infinite ringing)');
  assert.equal(cam.velYaw, 0);
  assert.equal(cam.velPitch, 0);
  assert.equal(cam.dist, cam.distTarget, 'dolly lands exactly on target');
  assert.ok(Math.abs(cam.pitch) <= PITCH_LIMIT_RAD + 1e-12, 'pitch stayed in bounds throughout');
  // NO snap-back: wherever it settled is where it stays
  const settled = { yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist };
  camStep(cam, 16);
  assert.deepEqual({ yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist }, settled);
});

test('drag does not integrate twice: while dragging, camStep leaves yaw/pitch to the pointer', () => {
  const cam = createOrbitCam(1.2);
  cam.dragging = true;
  camApplyDrag(cam, 100, 0);
  const yawAfterDrag = cam.yaw;
  camStep(cam, 16); // dragging → inertia not applied on top
  assert.equal(cam.yaw, yawAfterDrag);
  cam.dragging = false;
  camStep(cam, 16); // released → inertia now carries the motion
  assert.notEqual(cam.yaw, yawAfterDrag);
});

test('camEye: spherical orbit around the origin (craft), y-up', () => {
  const cam = createOrbitCam(1.2);
  cam.yaw = 0;
  cam.pitch = 0;
  cam.dist = 10;
  assert.deepEqual(camEye(cam).map((v) => Math.round(v * 1e9) / 1e9), [0, 0, 10]);
  cam.pitch = Math.PI / 4;
  const e = camEye(cam);
  near(e[1], 10 * Math.SQRT1_2, 1e-9, 'pitch raises the eye');
  near(Math.hypot(e[0], e[1], e[2]), 10, 1e-9, 'orbit radius = dist');
});

test('meshExtent: max |coordinate| of the mesh; DEFAULT_EXTENT when absent', () => {
  assert.equal(meshExtent(null), DEFAULT_EXTENT);
  const form = buildFormMesh('bus');
  const ext = meshExtent(form);
  // bus form spans ~2.2 with its solar wings — the dolly bounds scale with it
  assert.ok(ext > 0.5 && ext <= 3, `form extent sane: ${ext}`);
  const tiny = {
    positions: new Float32Array([0.2, -0.7, 0.1, 0, 0, 0, 0.1, 0.1, 0.1]),
    normals: new Float32Array(9),
    colors: new Float32Array(9),
    vertexCount: 3,
  };
  near(meshExtent(tiny), 0.7, 1e-6);
});

// ── 5. velocity indicator: derived, never fabricated ──────────────────────

test('velocity direction derives from position deltas; identical samples yield null', () => {
  // eastward motion on the equator → scene +x (east)
  const east = deriveVelocityDirScene(
    { latDeg: 0, lonDeg: 0, altMeters: 400_000 },
    { latDeg: 0, lonDeg: 0.1, altMeters: 400_000 },
  );
  assert.ok(east, 'direction exists');
  assert.ok(east[0] > 0.99, `east → scene +x: ${east[0]}`);
  near(east[1], 0, 0.05);
  // pure climb → scene +y (zenith)
  const up = deriveVelocityDirScene(
    { latDeg: 0, lonDeg: 0, altMeters: 400_000 },
    { latDeg: 0, lonDeg: 0, altMeters: 401_000 },
  );
  assert.ok(up && up[1] > 0.999, 'climb → scene +y');
  // northward motion → scene -z
  const north = deriveVelocityDirScene(
    { latDeg: 0, lonDeg: 0, altMeters: 400_000 },
    { latDeg: 0.1, lonDeg: 0, altMeters: 400_000 },
  );
  assert.ok(north && north[2] < -0.99, 'north → scene -z');
  // no motion → null, NEVER a fabricated heading
  assert.equal(
    deriveVelocityDirScene(
      { latDeg: 10, lonDeg: 20, altMeters: 500_000 },
      { latDeg: 10, lonDeg: 20, altMeters: 500_000 },
    ),
    null,
  );
});

// ── 6. provenance honesty ─────────────────────────────────────────────────

test('INSPECT_PROVENANCE labels every tier and never claims imagery', () => {
  assert.ok(
    INSPECT_PROVENANCE.includes(
      'Earth shown as a simplified globe with the real day/night terminator — not live imagery',
    ),
    'the required Earth label, verbatim',
  );
  assert.ok(INSPECT_PROVENANCE.includes('real ephemeris direction'), 'moon tier labeled');
  assert.ok(INSPECT_PROVENANCE.includes('no decorative starfield'), 'black-sky honesty stated');
  assert.ok(INSPECT_PROVENANCE.includes('true angular size'), 'angular-size claim stated');
  assert.ok(!/satellite imagery|photo|photograph/i.test(INSPECT_PROVENANCE), 'no imagery claim anywhere');
});

// ── 7. API smoke: no GL, no crash — the modelLayer failure latch ──────────

const explodingGl = new Proxy({}, { get() { throw new Error('gl touched'); } });

function fakeCanvas(): unknown {
  return {
    className: '',
    style: {} as Record<string, string>,
    width: 0,
    height: 0,
    clientWidth: 800,
    clientHeight: 600,
    getContext: () => explodingGl,
    addEventListener() { /* noop */ },
    removeEventListener() { /* noop */ },
    setPointerCapture() { /* noop */ },
    remove() { /* noop */ },
  };
}

const state = () => ({ latDeg: 0, lonDeg: 0, altMeters: 400_000, timeMs: Date.UTC(2026, 6, 15) });

test('mount with a broken GL: latch trips, handle stays safe (setMesh/render/dispose), page continues', () => {
  const container = {
    clientWidth: 800,
    clientHeight: 600,
    appendChild() { /* noop */ },
    ownerDocument: { createElement: () => fakeCanvas() },
  } as unknown as HTMLElement;
  const handle = mount(container, { mesh: null, meshLabel: 'test craft', getState: state });
  assert.equal(handle.getRenderFailed(), true, 'GL init exploded → latch tripped, no throw');
  // every handle call stays inert-safe after the latch
  handle.render();
  handle.setMesh(buildFormMesh('cubesat'), 'representative cubesat');
  assert.equal(handle.getMeshLabel(), 'representative cubesat');
  const cam0 = handle.getCamera();
  assert.ok(Number.isFinite(cam0.yaw) && Number.isFinite(cam0.dist));
  handle.dispose();
  handle.dispose(); // idempotent
});

test('mount with no DOM at all: inert handle, latch tripped, never throws', () => {
  const bare = { clientWidth: 0, clientHeight: 0, appendChild() { /* noop */ } } as unknown as HTMLElement;
  const handle = mount(bare, { mesh: null, meshLabel: 'x', getState: state });
  assert.equal(handle.getRenderFailed(), true);
  handle.render();
  handle.dispose();
});
