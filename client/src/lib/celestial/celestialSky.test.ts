// Realistic celestial sky — EARTH TWIN W1. Two kinds of check:
//   1. CROSS-CHECK vs astronomy-engine — celestialState / the path builders are
//      our COMPOSITION of astronomy-engine (topocentric az/el, apparent size,
//      magnitude). The test recomputes the same quantities directly from
//      astronomy-engine at three epochs and asserts our composition reproduces
//      them (catches frame/unit/convention bugs in OUR glue; the ephemeris
//      accuracy itself is astronomy-engine's, arcsecond-class vs JPL).
//   2. PINNED ANCHORS — literal az/el/size/phase snapshots (astronomy-engine
//      2.1.x, generated 2026-07-18) documenting the real-world values, so a
//      future refactor that silently changes the answer is caught even if it
//      still "agrees with itself".
// Plus the spec's contract tests: phase↔illumination consistency, angular-size
// formula, magnitude ordering, ecliptic great-circle closure, every fragment
// shader declares highp, and the exploding-GL latch. No WebGL context is used.
import { test } from "node:test";
import assert from "node:assert/strict";
import Astronomy from "astronomy-engine";
import {
  celestialState,
  angularDiameterDeg,
  directionVector,
  interpolateDirection,
  magnitudePointScale,
  glareIntensity,
  discRadiusOnSphere,
  eclipticPath,
  bodySkyPath,
  skyPosition,
  celestialState as _cs,
  BODY_RADIUS_M,
  SKY_BODIES,
  AU_M,
  MIN_POINT_ANGULAR_DIAMETER_DEG,
  SKY_SPHERE_RADIUS,
  CELESTIAL_PROVENANCE,
  CELESTIAL_FRAGMENT_SHADERS,
  CELESTIAL_VERTEX_SHADERS,
  mountCelestialSky,
  type SkyBodyId,
  type CelestialBody,
} from "./celestialSky.js";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

// ── epochs / observers ───────────────────────────────────────────────────────
interface Epoch { iso: string; lat: number; lon: number; }
const EPOCHS: Epoch[] = [
  { iso: "2000-01-01T12:00:00Z", lat: 0, lon: 0 },              // theory center, equator
  { iso: "2026-01-01T00:00:00Z", lat: 40.7128, lon: -74.006 }, // NYC, present era
  { iso: "2026-06-21T12:00:00Z", lat: -33.8688, lon: 151.2093 }, // Sydney, S. solstice
];

const get = (bodies: CelestialBody[], id: SkyBodyId): CelestialBody =>
  bodies.find((b) => b.id === id)!;

/** Smallest angular distance between two azimuths, degrees (handles wrap). */
const circDiff = (a: number, b: number): number => {
  let d = (a - b) % 360;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return Math.abs(d);
};

// ── 1. CROSS-CHECK vs astronomy-engine (composition correctness) ─────────────
test("position/size cross-check vs astronomy-engine at three epochs", () => {
  for (const ep of EPOCHS) {
    const dateMs = Date.parse(ep.iso);
    const bodies = celestialState(dateMs, ep.lat, ep.lon);
    const time = Astronomy.MakeTime(new Date(dateMs));
    const obs = new Astronomy.Observer(ep.lat, ep.lon, 0);
    for (const def of SKY_BODIES) {
      const b = get(bodies, def.id);
      const eq = Astronomy.Equator(def.body, time, obs, true, true);
      const hor = Astronomy.Horizon(time, obs, eq.ra, eq.dec);
      const il = Astronomy.Illumination(def.body, time);
      const diam = 2 * Math.asin(BODY_RADIUS_M[def.id] / (eq.dist * AU_M)) * RAD;
      assert.ok(circDiff(b.azDeg, hor.azimuth) < 1e-6, `${ep.iso} ${def.id} az ${b.azDeg} vs ${hor.azimuth}`);
      assert.ok(Math.abs(b.elDeg - hor.altitude) < 1e-6, `${ep.iso} ${def.id} el`);
      assert.ok(Math.abs(b.angularDiameterDeg - diam) < 1e-9, `${ep.iso} ${def.id} diam`);
      assert.ok(Math.abs(b.apparentMagnitude - il.mag) < 1e-9, `${ep.iso} ${def.id} mag`);
      assert.ok(Math.abs(b.distanceAu - eq.dist) < 1e-12, `${ep.iso} ${def.id} dist`);
    }
  }
});

// ── 2. PINNED ANCHORS (real-world snapshot) ──────────────────────────────────
// az, el, [diamDeg], generated from astronomy-engine 2.1.x on 2026-07-18.
test("pinned real-world anchors (Sun/Moon/Venus/Jupiter)", () => {
  // [epochIndex][id] = { az, el, diam?, phase?, frac? }
  const A = [
    { // 2000-01-01T12:00Z equator
      sun: { az: 178.069, el: 66.953, diam: 0.541971 },
      moon: { az: 257.206, el: 30.570, diam: 0.498779, phase: 122.67, frac: 0.2301 },
      venus: { az: 242.842, el: 46.105 },
      jupiter: { az: 81.166, el: -13.258 },
    },
    { // 2026-01-01T00:00Z NYC
      sun: { az: 261.084, el: -25.947 },
      moon: { az: 103.261, el: 55.566, diam: 0.559657, phase: 34.11, frac: 0.9140 },
      venus: { az: 261.386, el: -27.373 },
      jupiter: { az: 75.092, el: 16.704 },
    },
    { // 2026-06-21T12:00Z Sydney
      sun: { az: 255.505, el: -62.422 },
      moon: { az: 283.911, el: 18.733, phase: 94.78, frac: 0.4583 },
      venus: { az: 277.322, el: -25.978 },
      jupiter: { az: 272.295, el: -36.412 },
    },
  ] as const;
  EPOCHS.forEach((ep, i) => {
    const bodies = celestialState(Date.parse(ep.iso), ep.lat, ep.lon);
    for (const id of ["sun", "moon", "venus", "jupiter"] as SkyBodyId[]) {
      const a = (A[i] as Record<string, { az: number; el: number; diam?: number; phase?: number; frac?: number }>)[id];
      const b = get(bodies, id);
      assert.ok(circDiff(b.azDeg, a.az) < 0.05, `epoch ${i} ${id} az ${b.azDeg} vs ${a.az}`);
      assert.ok(Math.abs(b.elDeg - a.el) < 0.05, `epoch ${i} ${id} el ${b.elDeg} vs ${a.el}`);
      if (a.diam !== undefined) assert.ok(Math.abs(b.angularDiameterDeg - a.diam) < 0.001, `epoch ${i} ${id} diam`);
      if (a.phase !== undefined) assert.ok(Math.abs(b.phaseAngle - a.phase) < 0.1, `epoch ${i} ${id} phase`);
      if (a.frac !== undefined) assert.ok(Math.abs(b.illuminatedFraction - a.frac) < 0.005, `epoch ${i} ${id} frac`);
    }
  });
});

// ── 3. angular-size formula + real magnitudes ────────────────────────────────
test("angular-diameter formula: exact spherical form + known values", () => {
  // exact 2·asin(R/d)
  const d = 384_400_000; // ~mean lunar distance, m
  assert.equal(
    angularDiameterDeg(BODY_RADIUS_M.moon, d),
    2 * Math.asin(BODY_RADIUS_M.moon / d) * RAD,
  );
  // Sun and Moon are both ~0.5°; planets are sub-arcminute (<1/60°)
  for (const ep of EPOCHS) {
    const bodies = celestialState(Date.parse(ep.iso), ep.lat, ep.lon);
    assert.ok(Math.abs(get(bodies, "sun").angularDiameterDeg - 0.53) < 0.03, "Sun ~0.53°");
    assert.ok(Math.abs(get(bodies, "moon").angularDiameterDeg - 0.52) < 0.05, "Moon ~0.5°");
    for (const id of ["mercury", "venus", "mars", "jupiter", "saturn"] as SkyBodyId[]) {
      assert.ok(get(bodies, id).angularDiameterDeg < 1 / 60, `${id} sub-arcminute`);
    }
  }
});

test("angularDiameterDeg internal-consistency with reported distance", () => {
  const bodies = celestialState(Date.parse(EPOCHS[1].iso), EPOCHS[1].lat, EPOCHS[1].lon);
  for (const b of bodies) {
    const expected = angularDiameterDeg(BODY_RADIUS_M[b.id], b.distanceAu * AU_M);
    assert.ok(Math.abs(b.angularDiameterDeg - expected) < 1e-9, `${b.id} size↔distance`);
  }
});

// ── 4. moon phase ↔ illumination consistency ─────────────────────────────────
test("moon illuminated fraction = (1 + cos(phaseAngle)) / 2", () => {
  for (const ep of EPOCHS) {
    const moon = get(celestialState(Date.parse(ep.iso), ep.lat, ep.lon), "moon");
    const expected = (1 + Math.cos(moon.phaseAngle * DEG)) / 2;
    assert.ok(
      Math.abs(moon.illuminatedFraction - expected) < 0.01,
      `${ep.iso} moon frac ${moon.illuminatedFraction} vs ${expected}`,
    );
    assert.ok(moon.illuminatedFraction >= 0 && moon.illuminatedFraction <= 1);
    assert.ok(moon.phaseAngle >= 0 && moon.phaseAngle <= 180);
  }
});

// ── 5. apparent-magnitude ordering sanity ────────────────────────────────────
test("magnitude ordering: Sun < Moon < Venus < Jupiter < Saturn; planets < Moon", () => {
  for (const ep of EPOCHS) {
    const b = celestialState(Date.parse(ep.iso), ep.lat, ep.lon);
    const m = (id: SkyBodyId) => get(b, id).apparentMagnitude;
    assert.ok(m("sun") < m("moon"), "Sun brighter than Moon");
    assert.ok(m("moon") < m("venus"), "Moon brighter than Venus");
    assert.ok(m("venus") < m("jupiter"), "Venus brighter than Jupiter");
    assert.ok(m("jupiter") < m("saturn"), "Jupiter brighter than Saturn");
    // Sun is the brightest, Moon second, of every body
    for (const id of ["moon", "mercury", "venus", "mars", "jupiter", "saturn"] as SkyBodyId[]) {
      assert.ok(m("sun") < m(id), `Sun brightest vs ${id}`);
    }
    for (const id of ["mercury", "venus", "mars", "jupiter", "saturn"] as SkyBodyId[]) {
      assert.ok(m("moon") < m(id), `Moon brighter than ${id}`);
    }
  }
});

test("magnitudePointScale is monotonically decreasing in magnitude and clamped", () => {
  assert.ok(magnitudePointScale(-26) === 1, "Sun saturates at 1");
  assert.ok(magnitudePointScale(10) === 0, "beyond naked eye ⇒ 0");
  assert.ok(magnitudePointScale(-4) > magnitudePointScale(2), "brighter ⇒ bigger scale");
  assert.ok(magnitudePointScale(0) > magnitudePointScale(3));
});

// ── 6. paths: ecliptic is a great circle (closure + planarity) ───────────────
test("eclipticPath is a closed great circle", () => {
  const pts = eclipticPath(Date.parse("2026-06-21T12:00:00Z"), 40.7, -74, 180);
  assert.equal(pts.length, 181);
  // closure: last point equals the first
  assert.ok(Math.abs(pts[0].azDeg - pts[180].azDeg) < 1e-6, "az closes");
  assert.ok(Math.abs(pts[0].elDeg - pts[180].elDeg) < 1e-6, "el closes");
  // planarity: every direction vector lies in one plane through the origin ⇒
  // a great circle. Take the plane normal from two well-separated points and
  // assert every point is perpendicular to it (dot ≈ 0).
  const dirs = pts.map((p) => directionVector(p.azDeg, p.elDeg));
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
  ];
  let n = cross(dirs[0], dirs[45]);
  const nl = Math.hypot(n[0], n[1], n[2]) || 1;
  n = [n[0] / nl, n[1] / nl, n[2] / nl];
  let maxDot = 0;
  for (const dvec of dirs) {
    maxDot = Math.max(maxDot, Math.abs(dvec[0] * n[0] + dvec[1] * n[1] + dvec[2] * n[2]));
    assert.ok(Math.abs(Math.hypot(dvec[0], dvec[1], dvec[2]) - 1) < 1e-9, "unit vectors");
  }
  assert.ok(maxDot < 1e-6, `all points coplanar (great circle); maxDot=${maxDot}`);
});

test("bodySkyPath returns finite az/el for Moon and planets", () => {
  const dateMs = Date.parse("2026-01-01T00:00:00Z");
  for (const id of ["moon", "venus", "jupiter"] as SkyBodyId[]) {
    const pts = bodySkyPath(dateMs, 40.7, -74, id, 60);
    assert.equal(pts.length, 61);
    for (const p of pts) {
      assert.ok(Number.isFinite(p.azDeg) && p.azDeg >= 0 && p.azDeg <= 360, `${id} az finite`);
      assert.ok(Number.isFinite(p.elDeg) && p.elDeg >= -90 && p.elDeg <= 90, `${id} el finite`);
    }
  }
  // requesting fewer than the 8-sample smoothness floor is bumped to 8 ⇒ 9 pts
  assert.equal(bodySkyPath(dateMs, 0, 0, "sun" as SkyBodyId, 4).length, 9);
});

// ── geometry helpers ─────────────────────────────────────────────────────────
test("directionVector matches the ENU/scene convention", () => {
  const zenith = directionVector(0, 90);
  assert.ok(Math.abs(zenith[1] - 1) < 1e-12, "el=90 ⇒ +Y (up)");
  const east = directionVector(90, 0);
  assert.ok(Math.abs(east[0] - 1) < 1e-12, "az=90 ⇒ +X (east)");
  const north = directionVector(0, 0);
  assert.ok(Math.abs(north[2] + 1) < 1e-12, "az=0 ⇒ -Z (north)");
  // always unit length
  for (const [az, el] of [[37, 12], [200, -45], [359, 89]]) {
    const d = directionVector(az, el);
    assert.ok(Math.abs(Math.hypot(d[0], d[1], d[2]) - 1) < 1e-12);
  }
});

test("interpolateDirection stays on the unit sphere and clamps t", () => {
  const a = directionVector(10, 20), b = directionVector(40, 25);
  for (const t of [-1, 0, 0.5, 1, 2]) {
    const d = interpolateDirection(a, b, t);
    assert.ok(Math.abs(Math.hypot(d[0], d[1], d[2]) - 1) < 1e-12, `unit at t=${t}`);
  }
  assert.deepEqual(interpolateDirection(a, b, -5), interpolateDirection(a, b, 0));
  assert.deepEqual(interpolateDirection(a, b, 9), interpolateDirection(a, b, 1));
});

test("discRadiusOnSphere applies the sub-arcminute point floor", () => {
  // a true 0.01° planet is floored up to MIN_POINT_ANGULAR_DIAMETER_DEG
  const floored = discRadiusOnSphere(0.01);
  const atFloor = SKY_SPHERE_RADIUS * Math.tan((MIN_POINT_ANGULAR_DIAMETER_DEG * 0.5) * DEG);
  assert.ok(Math.abs(floored - atFloor) < 1e-9, "tiny planet floored to visible");
  // the Sun's true ~0.53° is ABOVE the floor ⇒ not inflated
  const sun = discRadiusOnSphere(0.53);
  assert.ok(sun > atFloor, "Sun drawn at its true (larger) size");
});

test("skyPosition scales the unit direction by the sphere radius", () => {
  const d = directionVector(123, 45);
  const p = skyPosition(d);
  assert.ok(Math.abs(Math.hypot(p[0], p[1], p[2]) - SKY_SPHERE_RADIUS) < 1e-6);
});

test("glareIntensity: zero off-screen, grows toward center and with zoom", () => {
  const look = directionVector(180, 30);
  const sunCentered = directionVector(180, 30);
  const sunOff = directionVector(0, 30); // opposite side of the sky
  assert.equal(glareIntensity(look, sunOff, 60), 0, "Sun off-screen ⇒ no glare");
  const wide = glareIntensity(look, sunCentered, 60);
  const zoomed = glareIntensity(look, sunCentered, 12);
  assert.ok(wide > 0 && zoomed > 0);
  assert.ok(zoomed >= wide, "narrower fov (zoom in) ⇒ at least as much glare");
  // partially-off-center Sun glares less than a centered one
  const offCenter = glareIntensity(look, directionVector(180 + 25, 30), 60);
  assert.ok(offCenter < wide, "sun off-center ⇒ less glare");
});

// ── determinism + horizon flag ───────────────────────────────────────────────
test("celestialState is a pure deterministic function of its inputs", () => {
  const a = _cs(Date.parse("2026-01-01T00:00:00Z"), 40.7, -74);
  const b = celestialState(Date.parse("2026-01-01T00:00:00Z"), 40.7, -74);
  assert.deepEqual(a, b);
});

test("belowHorizon matches the sign of elevation", () => {
  for (const ep of EPOCHS) {
    for (const b of celestialState(Date.parse(ep.iso), ep.lat, ep.lon)) {
      assert.equal(b.belowHorizon, b.elDeg < 0, `${ep.iso} ${b.id}`);
    }
  }
});

// ── 7. shader contract: every fragment shader declares highp, none mediump ──
test("every fragment shader declares `precision highp float` and no mediump", () => {
  assert.ok(CELESTIAL_FRAGMENT_SHADERS.length >= 3, "sun/moon/planet fragment shaders present");
  for (const src of CELESTIAL_FRAGMENT_SHADERS) {
    assert.match(src, /precision\s+highp\s+float\s*;/, "fragment shader declares highp float");
    assert.doesNotMatch(src, /mediump/, "no mediump anywhere (link-black bug guard)");
    assert.match(src, /#version 300 es/, "webgl2 shader");
  }
  // vertex shaders must not introduce mediump on the shared uniforms either
  for (const src of CELESTIAL_VERTEX_SHADERS) {
    assert.doesNotMatch(src, /mediump/, "vertex shader has no mediump");
  }
});

test("provenance string is the exact spec text", () => {
  assert.equal(
    CELESTIAL_PROVENANCE,
    "computed ephemeris (astronomy-engine) — apparent size and position accurate; not true scene distance",
  );
});

// ── 7. exploding-GL latch + no-DOM smoke test ────────────────────────────────
test("mountCelestialSky latches renderFailed with no DOM and never throws", () => {
  const view = () => ({
    timeMs: Date.parse("2026-01-01T00:00:00Z"),
    observerLatDeg: 40.7, observerLonDeg: -74,
    lookAzDeg: 180, lookElDeg: 30, fovDeg: 45,
  });
  // container whose ownerDocument is null and no global document ⇒ inert handle
  const container = { ownerDocument: null } as unknown as HTMLElement;
  const handle = mountCelestialSky(container, { getView: view });
  assert.equal(handle.getRenderFailed(), true, "no DOM ⇒ renderFailed latched");
  assert.doesNotThrow(() => handle.render(), "render is a no-op, never throws");
  assert.doesNotThrow(() => handle.setTime(Date.now()), "setTime never throws");
  // paths API (sprint W1 toggle) exists and is safe on the inert handle
  assert.equal(handle.getPathsVisible(), false, "paths default OFF");
  assert.doesNotThrow(() => handle.setPathsVisible(true), "setPathsVisible never throws");
  assert.doesNotThrow(() => handle.dispose(), "dispose never throws");
  assert.equal(handle.getState().bodies, SKY_BODIES.length);
});

test("mountCelestialSky latches when WebGL2 is unavailable (getContext ⇒ null)", () => {
  // minimal fake DOM: a canvas whose getContext returns null (headless GL fail)
  const canvasStub = {
    style: {} as Record<string, string>,
    className: "",
    width: 0, height: 0, clientWidth: 800, clientHeight: 600,
    getContext: () => null,
    remove() { /* noop */ },
  };
  const fakeDoc = {
    createElement: () => canvasStub,
    addEventListener() { /* noop — visibility gate wiring is independent of GL success */ },
    removeEventListener() { /* noop */ },
  } as unknown as Document;
  const container = {
    ownerDocument: fakeDoc, clientWidth: 800, clientHeight: 600,
    appendChild() { /* noop */ },
  } as unknown as HTMLElement;
  const handle = mountCelestialSky(container, {
    getView: () => ({
      timeMs: Date.now(), observerLatDeg: 0, observerLonDeg: 0,
      lookAzDeg: 0, lookElDeg: 45, fovDeg: 60,
    }),
  });
  assert.equal(handle.getRenderFailed(), true, "WebGL2 unavailable ⇒ renderFailed latched");
  assert.doesNotThrow(() => handle.render());
  assert.doesNotThrow(() => handle.dispose());
});

// ── 8. page-visibility gate: skip the per-frame WebGL draw while hidden ─────
// (round-10 terrain audit follow-up, open_questions.md: "celestialSky runs an
// unconditional rAF WebGL draw on the normal map — a fixed per-frame tax;
// gate by visibility?"). A full fake WebGL2 context lets this exercise the
// REAL rAF loop end-to-end (gl.clear call count as the "did it draw" probe)
// rather than only pinning the source text.
function makeFakeGl() {
  let clearCalls = 0;
  const noop = () => {};
  const gl: Record<string, unknown> = {
    // constants — arbitrary distinct values, only identity matters to this fake
    VERTEX_SHADER: 1, FRAGMENT_SHADER: 2, COMPILE_STATUS: 3, LINK_STATUS: 4,
    ARRAY_BUFFER: 5, STATIC_DRAW: 6, DYNAMIC_DRAW: 7, FLOAT: 8,
    COLOR_BUFFER_BIT: 9, DEPTH_TEST: 10, BLEND: 11,
    SRC_ALPHA: 12, ONE: 13, ONE_MINUS_SRC_ALPHA: 14, TRIANGLES: 15, LINE_STRIP: 16,
    createShader: () => ({}), shaderSource: noop, compileShader: noop,
    getShaderParameter: () => true, getShaderInfoLog: () => "", deleteShader: noop,
    createProgram: () => ({}), attachShader: noop, linkProgram: noop,
    getProgramParameter: () => true, getProgramInfoLog: () => "", deleteProgram: noop,
    getAttribLocation: () => 0, getUniformLocation: () => ({}),
    createBuffer: () => ({}), bindBuffer: noop, bufferData: noop, deleteBuffer: noop,
    viewport: noop, clearColor: noop,
    clear: () => { clearCalls++; },
    disable: noop, enable: noop, blendFunc: noop, useProgram: noop,
    uniformMatrix4fv: noop, uniform3f: noop, uniform1f: noop,
    vertexAttribPointer: noop, enableVertexAttribArray: noop, drawArrays: noop,
  };
  return { gl, clearCalls: () => clearCalls };
}

test("celestialSky rAF loop skips the WebGL draw while document.hidden, catches up on return", () => {
  const fake = makeFakeGl();
  const canvasStub = {
    style: {} as Record<string, string>, className: "",
    width: 0, height: 0, clientWidth: 800, clientHeight: 600,
    getContext: () => fake.gl,
    remove() { /* noop */ },
  };
  const visListeners = new Map<string, () => void>();
  const fakeDoc = {
    hidden: false,
    createElement: () => canvasStub,
    addEventListener: (type: string, cb: () => void) => { visListeners.set(type, cb); },
    removeEventListener: (type: string) => { visListeners.delete(type); },
  } as unknown as Document;
  const container = {
    ownerDocument: fakeDoc, clientWidth: 800, clientHeight: 600,
    appendChild() { /* noop */ },
  } as unknown as HTMLElement;

  // capture the rAF loop callback without auto-firing it (this test drives frames manually)
  let rafCb: (() => void) | null = null;
  let rafIdSeq = 0;
  const origRaf = (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame;
  const origCancel = (globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame;
  (globalThis as { requestAnimationFrame: (cb: () => void) => number }).requestAnimationFrame =
    (cb: () => void) => { rafCb = cb; return ++rafIdSeq; };
  (globalThis as { cancelAnimationFrame: (id: number) => void }).cancelAnimationFrame = () => {};

  try {
    const handle = mountCelestialSky(container, {
      getView: () => ({
        timeMs: Date.parse("2026-01-01T00:00:00Z"),
        observerLatDeg: 40.7, observerLonDeg: -74,
        lookAzDeg: 180, lookElDeg: 30, fovDeg: 45,
      }),
    });
    assert.equal(handle.getRenderFailed(), false, "fake GL2 context accepted");
    assert.ok(visListeners.has("visibilitychange"), "registers a visibilitychange listener");
    const afterMount = fake.clearCalls();
    assert.ok(afterMount >= 1, "the initial mount-time drawFrame() ran");
    assert.ok(rafCb, "rAF loop scheduled");

    // visible frame ⇒ draws
    rafCb!();
    assert.equal(fake.clearCalls(), afterMount + 1, "visible frame drew");

    // hidden ⇒ the rAF tick must NOT draw (this is the fix under test)
    fakeDoc.hidden = true;
    const beforeHidden = fake.clearCalls();
    assert.ok(rafCb, "loop still reschedules itself while hidden");
    rafCb!();
    assert.equal(fake.clearCalls(), beforeHidden, "hidden rAF tick skipped the draw");

    // a visibilitychange fired WHILE still hidden must not draw either
    visListeners.get("visibilitychange")!();
    assert.equal(fake.clearCalls(), beforeHidden, "visibilitychange while still hidden ⇒ no draw");

    // becoming visible again ⇒ the listener catches up with one immediate draw
    fakeDoc.hidden = false;
    visListeners.get("visibilitychange")!();
    assert.equal(fake.clearCalls(), beforeHidden + 1, "return-to-visible draws once immediately");

    // and the rAF loop resumes drawing every tick again
    const beforeResume = fake.clearCalls();
    rafCb!();
    assert.equal(fake.clearCalls(), beforeResume + 1, "rAF resumes drawing once visible again");

    handle.dispose();
    assert.ok(!visListeners.has("visibilitychange"), "dispose removes the visibilitychange listener");
  } finally {
    if (origRaf === undefined) delete (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame;
    else (globalThis as { requestAnimationFrame: unknown }).requestAnimationFrame = origRaf;
    if (origCancel === undefined) delete (globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame;
    else (globalThis as { cancelAnimationFrame: unknown }).cancelAnimationFrame = origCancel;
  }
});
