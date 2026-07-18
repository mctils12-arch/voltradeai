// followCamera — frame math, live glide, Earth-shadow (B6 first slice),
// orbit/free-look cameras, neighbor directions, inert-mount latch.
// Hermetic: no DOM, no GL — the mount path is exercised via its inert branch.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ecefUnit, ecefToEnu, enuToScene, sceneToEcefMat3,
  sunDirLocal, moonDirLocal, earthAngularRadiusRad,
  earthShadow, glidedCraftState, neighborDirsScene,
  createOrbitCam, camApplyDrag, camApplyDolly, camStep, camEye, camDollyBounds, wrapAngle,
  createLookCam, lookApplyDrag, lookApplyZoom, lookStep, lookDirScene,
  meshExtent, mountFollowCamera,
  EARTH_RADIUS_M, PITCH_LIMIT_RAD, LOOK_EL_LIMIT_DEG, LOOK_FOV_MIN_DEG, LOOK_FOV_MAX_DEG,
  MAX_GLIDE_SEC, NEIGHBOR_CAP,
  ORBIT_VIEW_PROVENANCE, ONBOARD_VIEW_PROVENANCE, GENERIC_MESH_LABEL,
} from './followCamera.js';
import { subsolarPoint } from '../celestial/ephemeris.js';
import { SAT_STRIDE, SENTINEL_SKIP, lonLatToMercator } from './satBuffer.js';

const close = (a: number, b: number, eps = 1e-9): boolean => Math.abs(a - b) < eps;

// ── frame math (the retired scene's anchors, re-pinned) ─────────────────────

test('ecefUnit / ecefToEnu anchors', () => {
  const u = ecefUnit(0, 0);
  assert.ok(close(u[0], 1) && close(u[1], 0) && close(u[2], 0));
  // the local zenith maps to pure Up at any site
  const z = ecefToEnu(ecefUnit(37.42, -122.08), 37.42, -122.08);
  assert.ok(close(z.e, 0, 1e-12) && close(z.n, 0, 1e-12) && close(z.u, 1, 1e-12));
  // a point 90° east on the equator is due east on the horizon
  const e = ecefToEnu(ecefUnit(0, 90), 0, 0);
  assert.ok(close(e.e, 1, 1e-12) && close(e.n, 0, 1e-12) && close(e.u, 0, 1e-12));
});

test('enuToScene: x=E, y=U, z=-N (right-handed)', () => {
  assert.deepEqual(enuToScene({ e: 1, n: 0, u: 0 }), [1, 0, -0]);
  assert.deepEqual(enuToScene({ e: 0, n: 1, u: 0 }), [0, 0, -1]);
  assert.deepEqual(enuToScene({ e: 0, n: 0, u: 1 }), [0, 1, -0]);
});

test('sceneToEcefMat3: columns are E, U, -N in ECEF', () => {
  const m = sceneToEcefMat3(0, 0); // at lat 0 lon 0: U=(1,0,0), E=(0,1,0), N=(0,0,1)
  // scene.x (E) → (0,1,0)
  assert.ok(close(m[0], 0) && close(m[1], 1) && close(m[2], 0));
  // scene.y (U) → (1,0,0)
  assert.ok(close(m[3], 1) && close(m[4], 0) && close(m[5], 0));
  // scene.z (-N) → (0,0,-1)
  assert.ok(close(m[6], 0) && close(m[7], 0) && close(m[8], -1));
});

test('sunDirLocal anchors: sun on zenith at the subsolar point, nadir at the antipode', () => {
  const t = Date.UTC(2026, 6, 18, 12, 0, 0);
  const sp = subsolarPoint(t);
  const atSub = sunDirLocal(t, sp.latDeg, sp.lonDeg);
  assert.ok(atSub.u > 0.999999, `zenith at subsolar point, got u=${atSub.u}`);
  const anti = sunDirLocal(t, -sp.latDeg, sp.lonDeg > 0 ? sp.lonDeg - 180 : sp.lonDeg + 180);
  assert.ok(anti.u < -0.999999, `nadir at the antipode, got u=${anti.u}`);
});

test('moonDirLocal: unit-ish direction, real angular size ~0.25°, fraction in [0,1]', () => {
  const t = Date.UTC(2026, 6, 18, 3, 0, 0);
  const m = moonDirLocal(t, 10, 20);
  const len = Math.hypot(m.dir.e, m.dir.n, m.dir.u);
  assert.ok(Math.abs(len - 1) < 1e-9);
  assert.ok(m.angularRadiusRad > 0.0038 && m.angularRadiusRad < 0.0052);
  assert.ok(m.illuminatedFraction >= 0 && m.illuminatedFraction <= 1);
});

test('earthAngularRadiusRad: ~70° at ISS altitude, ~8.7° at GEO', () => {
  const iss = earthAngularRadiusRad(420_000) / (Math.PI / 180);
  assert.ok(iss > 69 && iss < 71, `ISS ${iss}`);
  const geo = earthAngularRadiusRad(35_786_000) / (Math.PI / 180);
  assert.ok(geo > 8 && geo < 9.5, `GEO ${geo}`);
});

// ── B6 first slice: umbra/penumbra from real cone geometry ──────────────────

test('earthShadow: sun overhead → sunlit; sun at nadir (anti-solar) → umbra', () => {
  assert.deepEqual(earthShadow(420_000, { e: 0, n: 0, u: 1 }), { factor: 1, region: 'sunlit' });
  const um = earthShadow(420_000, { e: 0, n: 0, u: -1 });
  assert.equal(um.region, 'umbra');
  assert.equal(um.factor, 0);
  // GEO anti-solar is still inside the umbra cone (real eclipse seasons)
  const geo = earthShadow(35_786_000, { e: 0, n: 0, u: -1 });
  assert.equal(geo.region, 'umbra');
});

test('earthShadow: sun on the horizon stays sunlit at orbital altitude', () => {
  // at 420 km with the sun on the local horizon the craft still sees the sun
  // over the limb (horizon dip ~20°) — geometrically outside the shadow.
  const s = earthShadow(420_000, { e: 1, n: 0, u: 0 });
  assert.equal(s.region, 'sunlit');
});

test('earthShadow: monotonic penumbra ramp between sunlit and umbra', () => {
  // sweep the sun from horizon (u=0) to nadir (u=-1) at ISS altitude: the
  // factor must be non-increasing and pass through a penumbra band.
  let prev = 1;
  let sawPenumbra = false;
  for (let i = 0; i <= 200; i++) {
    const u = -i / 200;
    const e = Math.sqrt(Math.max(0, 1 - u * u));
    const s = earthShadow(420_000, { e, n: 0, u });
    assert.ok(s.factor <= prev + 1e-12, `factor rose at u=${u}`);
    if (s.region === 'penumbra') {
      sawPenumbra = true;
      assert.ok(s.factor > 0 && s.factor < 1);
    }
    prev = s.factor;
  }
  assert.ok(sawPenumbra, 'sweep must cross the penumbra band');
});

// ── live glide from the shared buffer ───────────────────────────────────────

function bufferWith(slots: Array<{ lon: number; lat: number; altM: number; cls: number; velX?: number; velY?: number; velAlt?: number }>): Float32Array {
  const buf = new Float32Array(slots.length * SAT_STRIDE);
  slots.forEach((s, i) => {
    const m = lonLatToMercator(s.lon, s.lat);
    buf[i * SAT_STRIDE] = m.x;
    buf[i * SAT_STRIDE + 1] = m.y;
    buf[i * SAT_STRIDE + 2] = s.altM;
    buf[i * SAT_STRIDE + 3] = s.cls;
    buf[i * SAT_STRIDE + 4] = s.velX ?? 0;
    buf[i * SAT_STRIDE + 5] = s.velY ?? 0;
    buf[i * SAT_STRIDE + 6] = s.velAlt ?? 0;
  });
  return buf;
}

test('glidedCraftState: exact tick at dt=0, advances along the real velocity, caps at MAX_GLIDE_SEC', () => {
  const velX = 0.0001; // normalized-mercator units / s eastward
  const buf = bufferWith([{ lon: 0, lat: 0, altM: 420_000, cls: 0, velX, velAlt: 10 }]);
  const s0 = glidedCraftState(buf, 0, 1000, 1000, 42)!;
  assert.ok(close(s0.lonDeg, 0, 1e-9) && close(s0.latDeg, 0, 1e-9));
  assert.equal(s0.timeMs, 42);
  const s1 = glidedCraftState(buf, 0, 1000, 2000, 43)!; // +1s
  // 1e-4° tolerance: buffer floats are f32 (same storage the GPU layer reads)
  assert.ok(close(s1.lonDeg, velX * 360, 1e-4), `glided lon ${s1.lonDeg}`);
  assert.ok(close(s1.altMeters, 420_010, 1e-2));
  const sCap = glidedCraftState(buf, 0, 1000, 100_000, 44)!; // way past the cap
  assert.ok(close(sCap.lonDeg, velX * MAX_GLIDE_SEC * 360, 1e-4), 'holds at the honesty cap');
});

test('glidedCraftState: sentinel slot and bad index → null; antimeridian wraps', () => {
  const buf = bufferWith([
    { lon: 179.99, lat: 10, altM: 500_000, cls: 0, velX: 0.0002 },
    { lon: 0, lat: 0, altM: 0, cls: SENTINEL_SKIP },
  ]);
  assert.equal(glidedCraftState(buf, 1, 0, 0, 0), null);
  assert.equal(glidedCraftState(buf, 5, 0, 0, 0), null);
  assert.equal(glidedCraftState(null, 0, 0, 0, 0), null);
  const wrapped = glidedCraftState(buf, 0, 0, 1000, 0)!; // crosses lon 180
  assert.ok(wrapped.lonDeg < -179.9 || wrapped.lonDeg > 179.99, `wrapped lon ${wrapped.lonDeg}`);
});

// ── neighbor directions (onboard markers) ───────────────────────────────────

test('neighborDirsScene: directly-above neighbor is +Up; east neighbor is +x; sentinel + self skipped', () => {
  const buf = bufferWith([
    { lon: 0, lat: 0, altM: 400_000, cls: 0 },                 // self
    { lon: 0, lat: 0, altM: 800_000, cls: 0 },                 // straight up
    { lon: 30, lat: 0, altM: 400_000, cls: 0 },                // east along the orbit shell
    { lon: 50, lat: 50, altM: 0, cls: SENTINEL_SKIP },         // invalid
  ]);
  const set = neighborDirsScene(buf, 0, { latDeg: 0, lonDeg: 0, altMeters: 400_000 });
  assert.equal(set.total, 2);
  assert.equal(set.count, 2);
  // nearest-first: the 400 km overhead gap is shorter than the 30° arc
  assert.ok(close(set.dirs[0], 0, 1e-6) && close(set.dirs[1], 1, 1e-6) && close(set.dirs[2], 0, 1e-6),
    'first neighbor is straight up in scene frame');
  assert.ok(close(set.rangesM[0], 400_000, 1));
  // second: eastward (+x), dipping below the horizon (y<0) across the chord
  assert.ok(set.dirs[3] > 0.9, `east component ${set.dirs[3]}`);
  assert.ok(set.dirs[4] < 0, 'chord dips below the local horizon');
});

test('neighborDirsScene: cap keeps the NEAREST objects and reports the true total', () => {
  const slots = [{ lon: 0, lat: 0, altM: 400_000, cls: 0 }];
  for (let i = 1; i <= 20; i++) slots.push({ lon: i * 3, lat: 0, altM: 400_000, cls: 0 });
  const buf = bufferWith(slots);
  const set = neighborDirsScene(buf, 0, { latDeg: 0, lonDeg: 0, altMeters: 400_000 }, 5);
  assert.equal(set.total, 20);
  assert.equal(set.count, 5);
  for (let i = 1; i < set.count; i++) assert.ok(set.rangesM[i] >= set.rangesM[i - 1], 'sorted nearest-first');
  assert.ok(NEIGHBOR_CAP >= 1024, 'default cap covers a real constellation view');
});

// ── orbit camera (retired-scene mechanics re-pinned) ────────────────────────

test('orbit cam: full-sphere drag — yaw wraps 360°, pitch clamps ±89°', () => {
  const cam = createOrbitCam(1.2);
  camApplyDrag(cam, 4000, 0);
  assert.ok(cam.yaw >= -Math.PI && cam.yaw <= Math.PI, 'yaw stays wrapped');
  camApplyDrag(cam, 0, 1e6);
  assert.ok(close(cam.pitch, PITCH_LIMIT_RAD));
  camApplyDrag(cam, 0, -1e7);
  assert.ok(close(cam.pitch, -PITCH_LIMIT_RAD), 'reaches the Earth-side pole');
  assert.ok(close(wrapAngle(3 * Math.PI), Math.PI) || close(wrapAngle(3 * Math.PI), -Math.PI));
});

test('orbit cam: dolly bounds 2×–50× extent; camStep converges without ringing', () => {
  const cam = createOrbitCam(1.0);
  const b = camDollyBounds(1.0);
  assert.ok(close(b.min, 2) && close(b.max, 50));
  camApplyDolly(cam, 1e9);
  assert.ok(close(cam.distTarget, b.max));
  camApplyDolly(cam, 1e-9);
  assert.ok(close(cam.distTarget, b.min));
  camApplyDolly(cam, NaN); // ignored
  assert.ok(close(cam.distTarget, b.min));
  for (let i = 0; i < 300; i++) camStep(cam, 16);
  assert.ok(close(cam.dist, cam.distTarget, 1e-6));
  assert.equal(camStep(cam, 16), false, 'settled loop reports idle');
});

test('orbit cam: camEye — pitch<0 puts the camera below the craft (Earth side)', () => {
  const cam = createOrbitCam(1.2);
  cam.pitch = -0.8;
  cam.yaw = 0;
  const e = camEye(cam);
  assert.ok(e[1] < 0, 'below the craft');
  assert.ok(close(Math.hypot(e[0], e[1], e[2]), cam.dist, 1e-9));
});

// ── onboard free-look camera ────────────────────────────────────────────────

test('look cam: free-look clamps elevation, wraps azimuth, fov zoom bounded', () => {
  const cam = createLookCam();
  lookApplyDrag(cam, 0, -1e6);
  assert.ok(close(cam.elDeg, LOOK_EL_LIMIT_DEG));
  lookApplyDrag(cam, 0, 1e7);
  assert.ok(close(cam.elDeg, -LOOK_EL_LIMIT_DEG), 'looks straight down at the Earth');
  lookApplyDrag(cam, 100000, 0);
  assert.ok(cam.azDeg >= 0 && cam.azDeg < 360);
  lookApplyZoom(cam, 1e-9);
  assert.ok(close(cam.fovTargetDeg, LOOK_FOV_MIN_DEG));
  lookApplyZoom(cam, 1e9);
  assert.ok(close(cam.fovTargetDeg, LOOK_FOV_MAX_DEG));
  for (let i = 0; i < 300; i++) lookStep(cam, 16);
  assert.ok(close(cam.fovDeg, cam.fovTargetDeg, 1e-6));
});

test('look cam: direction anchors — el -90 is nadir, az 0 el 0 is north, az 90 east', () => {
  const cam = createLookCam();
  cam.elDeg = -90; cam.azDeg = 0;
  let d = lookDirScene(cam);
  assert.ok(close(d[1], -1, 1e-9), 'nadir look = scene -y (Earth below)');
  cam.elDeg = 0; cam.azDeg = 0;
  d = lookDirScene(cam);
  assert.ok(close(d[2], -1, 1e-9), 'north = scene -z');
  cam.azDeg = 90;
  d = lookDirScene(cam);
  assert.ok(close(d[0], 1, 1e-9), 'east = scene +x');
});

// ── misc ────────────────────────────────────────────────────────────────────

test('meshExtent: max |coordinate|; empty mesh falls back to the default', () => {
  assert.equal(meshExtent(null), 1.2);
  const m = { positions: new Float32Array([0.2, -1.6, 0.4]), normals: new Float32Array(3), colors: new Float32Array(3), vertexCount: 1 };
  assert.ok(close(meshExtent(m as any), 1.6, 1e-6)); // f32 storage tolerance
});

test('provenance: every simplification labeled — the caption discipline', () => {
  for (const p of [ORBIT_VIEW_PROVENANCE, ONBOARD_VIEW_PROVENANCE]) {
    assert.match(p, /NOT live imagery/);
    assert.match(p, /no decorative starfield/);
    assert.match(p, /time never stops/);
  }
  assert.match(ORBIT_VIEW_PROVENANCE, /umbra/);
  assert.match(ORBIT_VIEW_PROVENANCE, /self-shadow approximation/);
  assert.match(ONBOARD_VIEW_PROVENANCE, /markers/);
  assert.match(GENERIC_MESH_LABEL, /not imagery of this unit/);
});

test('mount without a DOM: inert handle, latched failed, view/mesh setters still safe', () => {
  const h = mountFollowCamera(null as unknown as HTMLElement, {
    mesh: null,
    meshLabel: 'x',
    getState: () => null,
  });
  assert.equal(h.getRenderFailed(), true, 'no-DOM mount latches failed (parent falls back to the map ease)');
  h.setView('onboard');
  assert.equal(h.getView(), 'onboard');
  assert.equal(h.getProvenance(), ONBOARD_VIEW_PROVENANCE);
  h.setMesh(null, 'y');
  assert.equal(h.getMeshLabel(), 'y');
  h.setLook(410, -200);
  assert.ok(h.getLook().elDeg >= -LOOK_EL_LIMIT_DEG);
  h.render(); // must not throw
  h.dispose();
});
