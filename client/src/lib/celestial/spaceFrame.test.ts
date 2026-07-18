// Continuous space frame — pure view-math tests. mountSpaceFrame needs a DOM
// and is exercised by the integration drives (real dist + SwiftShader);
// everything mathematical here is hermetic and pinned. Same node:test style
// as ephemeris.test.ts / solarSystem.test.ts. The label-stacking and
// distance-label suites descend from the retired solarView.test.ts — the
// guarantees travel with the code (stacking upgraded to box-aware after
// drive round 1 caught text overprint the point rule could not see).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  perspectiveScalePx,
  bodyDiscPx,
  distanceForDiscPx,
  zoomStepFactor,
  ZOOM_STEP_PER_NOTCH,
  mapGlobeDiscPx,
  mapAnchorOpacity,
  MAP_FADE_HI_PX,
  MAP_FADE_LO_PX,
  markerNeeded,
  MARKER_MAX_DISC_PX,
  apparentLitFraction,
  obliquityDeg,
  eclFromEq,
  eqFromEcl,
  earthAxisEcl,
  entryCameraDir,
  subCameraLatLon,
  slerpUnit,
  rotateAbout,
  easeInOutCubic,
  expLerp,
  flyDurationMs,
  camBasis,
  northRollDeg,
  projectPoint,
  layoutLabelStacks,
  LABEL_COLLIDE_PX,
  fmtSpaceDistance,
  FRAME_DISC_FRACTION,
  MIN_DISTANCE_RADII,
  DEFAULT_FOV_DEG,
  defaultBodyRegistry,
  seamExitArmed,
  type Vec3,
} from "./spaceFrame.js";
import { solarSystemState, BODY_RADIUS_M, BODY_ORDER, AU_M } from "./solarSystem.js";
import { subsolarPoint } from "./ephemeris.js";

const close = (a: number, b: number, tol: number, msg: string): void =>
  assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

// ── projection ──────────────────────────────────────────────────────────────

test("perspective disc: exact round trip and the true angular sizes", () => {
  const k = perspectiveScalePx(DEFAULT_FOV_DEG, 900);
  // round trip disc ↔ distance at several scales
  for (const disc of [1, 51.36, 300, 612]) {
    const d = distanceForDiscPx(BODY_RADIUS_M.earth, disc, k);
    close(bodyDiscPx(BODY_RADIUS_M.earth, d, k), disc, 1e-9, `round trip at ${disc}px`);
  }
  // THE PHYSICAL PAYOFF, pinned: from mean lunar distance (384,400 km) Earth
  // subtends ~1.9° — a real disc, no compression needed.
  const lunar = 384_400_000;
  const angularDeg = 2 * Math.asin(BODY_RADIUS_M.earth / lunar) * (180 / Math.PI);
  close(angularDeg, 1.9, 0.05, "Earth from the Moon ≈ 1.9°");
  const px = bodyDiscPx(BODY_RADIUS_M.earth, lunar, k);
  close(px, 2 * k * Math.tan((angularDeg / 2) * Math.PI / 180), 1e-9, "px = 2k·tan(θ/2)... exact form");
  assert.ok(px > 40 && px < 50, `≈45px at 900px viewport (${px.toFixed(1)})`);
  // inside-the-body guard
  assert.equal(bodyDiscPx(BODY_RADIUS_M.earth, BODY_RADIUS_M.earth / 2, k), Number.POSITIVE_INFINITY);
});

test("projectPoint: center/off-axis orientation and behind-camera contract", () => {
  // dir is target→camera: camera at +y looking back at the origin, z-up ref
  const basis = camBasis({ x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 });
  const cam: Vec3 = { x: 0, y: 1000, z: 0 };
  const k = 500;
  const center = projectPoint({ x: 0, y: 0, z: 0 }, cam, basis, k, 400, 300);
  close(center.x, 400, 1e-9, "target dead-center x");
  close(center.y, 300, 1e-9, "target dead-center y");
  close(center.depth, 1000, 1e-9, "depth = distance along view axis");
  // +z world (up-reference) → screen up (smaller y)
  const up = projectPoint({ x: 0, y: 0, z: 100 }, cam, basis, k, 400, 300);
  assert.ok(up.y < 300, "up-reference projects screen-up");
  // behind the camera → depth ≤ 0, NaN screen coords
  const behind = projectPoint({ x: 0, y: 2000, z: 0 }, cam, basis, k, 400, 300);
  assert.ok(behind.depth <= 0 && Number.isNaN(behind.x), "behind flagged, never drawn");
});

test("camBasis: orthonormal, up from the reference axis, roll 0 by construction", () => {
  const axis = earthAxisEcl(Date.UTC(2026, 6, 18));
  for (const dir of [
    { x: 1, y: 0, z: 0 },
    { x: 0.3, y: -0.8, z: 0.5 },
    { x: -0.6, y: 0.6, z: 0.2 },
  ] as Vec3[]) {
    const n = Math.hypot(dir.x, dir.y, dir.z);
    const d = { x: dir.x / n, y: dir.y / n, z: dir.z / n };
    const b = camBasis(d, axis);
    const dp = (a: Vec3, c: Vec3): number => a.x * c.x + a.y * c.y + a.z * c.z;
    close(dp(b.f, b.r), 0, 1e-12, "f⊥r");
    close(dp(b.f, b.u), 0, 1e-12, "f⊥u");
    close(dp(b.r, b.u), 0, 1e-12, "r⊥u");
    close(Math.hypot(b.u.x, b.u.y, b.u.z), 1, 1e-12, "unit up");
    // the axis-referenced up puts the axis in the (f,u) plane → roll is 0:
    // this is why the live map needs NO CSS rotation (map draws north-up)
    close(northRollDeg(b, axis), 0, 1e-9, "north roll 0");
  }
  // degenerate: looking straight down the axis still yields a valid basis
  const b = camBasis(earthAxisEcl(0), earthAxisEcl(0));
  assert.ok(Number.isFinite(b.r.x) && Math.hypot(b.r.x, b.r.y, b.r.z) > 0.99, "degenerate fallback");
});

// ── exponential zoom ────────────────────────────────────────────────────────

test("zoomStepFactor: one notch multiplies by 1.18; multiplicative in deltaY", () => {
  close(zoomStepFactor(100), ZOOM_STEP_PER_NOTCH, 1e-12, "one notch out");
  close(zoomStepFactor(-100), 1 / ZOOM_STEP_PER_NOTCH, 1e-12, "one notch in");
  close(zoomStepFactor(0), 1, 1e-12, "no delta");
  // trackpad glide: many small deltas compose to the same curve
  close(zoomStepFactor(40) * zoomStepFactor(60), zoomStepFactor(100), 1e-12, "composition");
});

test("expLerp: geometric interpolation (uniform feel across magnitudes)", () => {
  close(expLerp(1e6, 1e12, 0), 1e6, 1, "start");
  close(expLerp(1e6, 1e12, 1), 1e12, 1, "end");
  close(expLerp(1e6, 1e12, 0.5), 1e9, 1e3, "midpoint = geometric mean");
});

test("easeInOutCubic: clamped, symmetric, monotone", () => {
  assert.equal(easeInOutCubic(-1), 0);
  assert.equal(easeInOutCubic(0), 0);
  assert.equal(easeInOutCubic(1), 1);
  assert.equal(easeInOutCubic(2), 1);
  close(easeInOutCubic(0.5), 0.5, 1e-12, "midpoint");
  let prev = -1;
  for (let t = 0; t <= 1.001; t += 0.05) {
    const v = easeInOutCubic(t);
    assert.ok(v >= prev, "monotone");
    prev = v;
  }
});

test("flyDurationMs: clamped to [1100, 3200], grows with ratio and swing", () => {
  assert.ok(flyDurationMs(1, 0) >= 1100);
  assert.ok(flyDurationMs(1e9, 180) <= 3200);
  assert.ok(flyDurationMs(1000, 0) > flyDurationMs(10, 0), "longer hop, longer fly");
  assert.ok(flyDurationMs(10, 120) > flyDurationMs(10, 0), "bigger swing, longer fly");
});

// ── the seam ────────────────────────────────────────────────────────────────

test("map globe disc: pins the measured maplibre values", () => {
  // measured on the real map (probe 2026-07-18): zoom −2, lat 37.5 → ≈51.4px
  close(mapGlobeDiscPx(-2, 37.5), 51.36, 0.05, "zoom floor at lat 37.5");
  // equator at zoom 0: 512/π
  close(mapGlobeDiscPx(0, 0), 512 / Math.PI, 1e-9, "equator zoom 0");
  // the seam is scale-continuous by construction: entry camera distance
  // reproduces the map's disc exactly
  const k = perspectiveScalePx(DEFAULT_FOV_DEG, 900);
  const d0 = distanceForDiscPx(BODY_RADIUS_M.earth, mapGlobeDiscPx(-2, 37.5), k);
  close(bodyDiscPx(BODY_RADIUS_M.earth, d0, k) / mapGlobeDiscPx(-2, 37.5), 1, 1e-12, "entry scale = 1");
  // and that camera sits inside lunar orbit — the Moon is immediately near
  assert.ok(d0 < 3.85e8, `entry camera ${(d0 / 1e6).toFixed(0)} Mm — inside lunar distance`);
});

test("crossfade: live map above HI, impostor below LO — and the lunar payoff stays live", () => {
  assert.equal(mapAnchorOpacity(MAP_FADE_HI_PX), 1);
  assert.equal(mapAnchorOpacity(MAP_FADE_HI_PX + 100), 1);
  assert.equal(mapAnchorOpacity(MAP_FADE_LO_PX), 0);
  assert.equal(mapAnchorOpacity(2), 0);
  const mid = mapAnchorOpacity((MAP_FADE_HI_PX + MAP_FADE_LO_PX) / 2);
  assert.ok(mid > 0.4 && mid < 0.6, "smoothstep midpoint");
  // THE CONTRACT: from lunar distance (even apogee) at every canonical
  // harness viewport height, Earth's disc stays ABOVE the fade band — the
  // Earth you see from the Moon is the live map, not the impostor.
  const apogee = 406_700_000;
  for (const h of [844, 1024, 900]) {
    const k = perspectiveScalePx(DEFAULT_FOV_DEG, h);
    const disc = bodyDiscPx(BODY_RADIUS_M.earth, apogee, k);
    assert.ok(disc > MAP_FADE_HI_PX, `live from the Moon at h=${h} (${disc.toFixed(1)}px)`);
  }
});

test("marker threshold: sub-2px bodies are flagged, never inflated", () => {
  assert.equal(markerNeeded(MARKER_MAX_DISC_PX - 0.01), true);
  assert.equal(markerNeeded(MARKER_MAX_DISC_PX), false);
  assert.equal(markerNeeded(0.001), true);
  // at a whole-system distance (30 AU) every planet is honestly sub-pixel
  const k = perspectiveScalePx(DEFAULT_FOV_DEG, 900);
  for (const id of ["mercury", "venus", "earth", "moon", "mars"] as const) {
    assert.ok(markerNeeded(bodyDiscPx(BODY_RADIUS_M[id], 30 * AU_M, k)), `${id} marked at 30 AU`);
  }
});

test("fly-to framing: aspect-safe fraction of the SHORT side, floored at closest approach", () => {
  // at 1440×900 the framing sits ≈4.5 radii out (the number in the docs)
  const k = perspectiveScalePx(DEFAULT_FOV_DEG, 900);
  const d = distanceForDiscPx(BODY_RADIUS_M.moon, FRAME_DISC_FRACTION * 900, k);
  close(d / BODY_RADIUS_M.moon, 4.5, 0.15, "≈4.5 body radii at 1440×900");
  assert.ok(d / BODY_RADIUS_M.moon > MIN_DISTANCE_RADII, "framing outside closest approach");
});

// ── phase / lighting ────────────────────────────────────────────────────────

test("apparentLitFraction: full toward the sun, new opposite, half at quadrature", () => {
  const sun: Vec3 = { x: 1, y: 0, z: 0 };
  close(apparentLitFraction(sun, { x: 1, y: 0, z: 0 }), 1, 1e-12, "camera sunward: full");
  close(apparentLitFraction(sun, { x: -1, y: 0, z: 0 }), 0, 1e-12, "camera anti-sun: new");
  close(apparentLitFraction(sun, { x: 0, y: 1, z: 0 }), 0.5, 1e-12, "quadrature: half");
});

test("camera-relative Moon phase reproduces the Earth-observed illumination", () => {
  // from Earth's position, apparentLitFraction(sun→moon, earth→moon) must
  // equal the classic elongation-based illuminated fraction to a few %
  for (const t of [Date.UTC(2026, 0, 3), Date.UTC(2026, 6, 18), Date.UTC(2026, 9, 26, 12)]) {
    const state = solarSystemState(t);
    const moon = state.find((b) => b.id === "moon")!;
    const earth = state.find((b) => b.id === "earth")!;
    const sun = state.find((b) => b.id === "sun")!;
    const unit = (a: Vec3): Vec3 => {
      const l = Math.hypot(a.x, a.y, a.z) || 1;
      return { x: a.x / l, y: a.y / l, z: a.z / l };
    };
    const toSun = unit({ x: sun.helioAu.x - moon.helioAu.x, y: sun.helioAu.y - moon.helioAu.y, z: sun.helioAu.z - moon.helioAu.z });
    const toEarth = unit({ x: earth.helioAu.x - moon.helioAu.x, y: earth.helioAu.y - moon.helioAu.y, z: earth.helioAu.z - moon.helioAu.z });
    const geometric = apparentLitFraction(toSun, toEarth);
    // independent classic form: (1 − cos elongation)/2 from ecliptic longitudes
    const lonM = Math.atan2(moon.geoAu.y, moon.geoAu.x);
    const lonS = Math.atan2(sun.geoAu.y, sun.geoAu.x);
    const classic = (1 - Math.cos(lonM - lonS)) / 2;
    close(geometric, classic, 0.03, `phase agreement at ${new Date(t).toISOString()}`);
  }
});

// ── frames: ecliptic ↔ equatorial ↔ Earth lat/lon ───────────────────────────

test("obliquity and the Earth axis: (0, sin ε, cos ε) in the ecliptic frame", () => {
  const t = Date.UTC(2026, 6, 18);
  close(obliquityDeg(t), 23.4358, 0.01, "mean obliquity 2026");
  const axis = earthAxisEcl(t);
  const e = obliquityDeg(t) * Math.PI / 180;
  close(axis.x, 0, 1e-12, "axis x");
  close(axis.y, Math.sin(e), 1e-12, "axis y = sin ε");
  close(axis.z, Math.cos(e), 1e-12, "axis z = cos ε");
});

test("eclFromEq/eqFromEcl: exact inverses, length-preserving", () => {
  const t = Date.UTC(2026, 3, 1, 6);
  for (const veq of [
    { x: 1, y: 0, z: 0 },
    { x: 0.2, y: -0.7, z: 0.68 },
    { x: -0.5, y: 0.5, z: -0.7 },
  ] as Vec3[]) {
    const back = eqFromEcl(eclFromEq(veq, t), t);
    close(back.x, veq.x, 1e-12, "x round trip");
    close(back.y, veq.y, 1e-12, "y round trip");
    close(back.z, veq.z, 1e-12, "z round trip");
  }
});

test("entryCameraDir ↔ subCameraLatLon: round trip over the globe", () => {
  const t = Date.UTC(2026, 6, 18, 15, 30);
  for (const [lat, lon] of [[37.5, -96.77], [0, 0], [-33.9, 151.2], [64.1, -21.9], [85, 179]] as const) {
    const dir = entryCameraDir(lat, lon, t);
    close(Math.hypot(dir.x, dir.y, dir.z), 1, 1e-12, "unit dir");
    const back = subCameraLatLon(dir, t);
    close(back.latDeg, lat, 1e-9, `lat round trip @${lat},${lon}`);
    close(back.lonDeg, lon, 1e-9, `lon round trip @${lat},${lon}`);
  }
});

test("TERMINATOR CONSISTENCY: the space-frame Sun agrees with the map's subsolar point", () => {
  // Two independent truncations (Schlyter system state vs the Meeus-derived
  // ephemeris.ts the map terminator uses) must put the Sun over the same
  // spot to ~1° — this is the honesty seam that makes the terminator painted
  // on the shrinking globe agree with where the Sun hangs in the frame.
  for (const t of [Date.UTC(2026, 0, 15, 3), Date.UTC(2026, 6, 18, 15, 30), Date.UTC(2026, 10, 2, 21)]) {
    const state = solarSystemState(t);
    const sun = state.find((b) => b.id === "sun")!;
    const earth = state.find((b) => b.id === "earth")!;
    const toSun: Vec3 = {
      x: sun.helioAu.x - earth.helioAu.x,
      y: sun.helioAu.y - earth.helioAu.y,
      z: sun.helioAu.z - earth.helioAu.z,
    };
    const fromFrame = subCameraLatLon(toSun, t);
    const fromMap = subsolarPoint(t);
    close(fromFrame.latDeg, fromMap.latDeg, 1, `subsolar lat @${new Date(t).toISOString()}`);
    const dLon = Math.abs(((fromFrame.lonDeg - fromMap.lonDeg + 540) % 360) - 180);
    assert.ok(dLon < 1, `subsolar lon @${new Date(t).toISOString()} (Δ${dLon.toFixed(3)}°)`);
  }
});

// ── rotation / interpolation primitives ─────────────────────────────────────

test("rotateAbout: Rodrigues rotation, right-handed", () => {
  const z: Vec3 = { x: 0, y: 0, z: 1 };
  const r = rotateAbout({ x: 1, y: 0, z: 0 }, z, 90);
  close(r.x, 0, 1e-12, "x→y (90° about z) x");
  close(r.y, 1, 1e-12, "x→y (90° about z) y");
  const back = rotateAbout(r, z, -90);
  close(back.x, 1, 1e-12, "inverse rotation");
});

test("slerpUnit: endpoints, constant angular rate, antiparallel fallback", () => {
  const a: Vec3 = { x: 1, y: 0, z: 0 };
  const b: Vec3 = { x: 0, y: 1, z: 0 };
  close(slerpUnit(a, b, 0).x, 1, 1e-12, "t=0 → a");
  close(slerpUnit(a, b, 1).y, 1, 1e-12, "t=1 → b");
  const mid = slerpUnit(a, b, 0.5);
  close(Math.atan2(mid.y, mid.x) * 180 / Math.PI, 45, 1e-9, "midpoint at 45°");
  close(Math.hypot(mid.x, mid.y, mid.z), 1, 1e-12, "stays unit");
  // antiparallel: deterministic, unit-length, passes through a waypoint
  const anti = slerpUnit(a, { x: -1, y: 0, z: 0 }, 0.5);
  close(Math.hypot(anti.x, anti.y, anti.z), 1, 1e-9, "antiparallel midpoint unit");
  assert.ok(Math.abs(anti.x) < 1e-6, "antiparallel midpoint ⊥ endpoints");
});

test("seamExitArmed: only idle-at-anchor or the fly-home flight can exit (phone-drive regression)", () => {
  // idle at Earth → armed (zooming in hands back to the map)
  assert.equal(seamExitArmed(false, false, true), true);
  // idle focused elsewhere → never
  assert.equal(seamExitArmed(false, false, false), false);
  // THE REGRESSION: flying to the Moon (not exit-armed) swings the camera
  // close past Earth — its true disc exceeds the map disc mid-arc, and the
  // un-gated seam ejected the flight to the map. Must stay unarmed.
  assert.equal(seamExitArmed(true, false, true), false);
  assert.equal(seamExitArmed(true, false, false), false);
  // the fly-home flight exits the moment it crosses the seam
  assert.equal(seamExitArmed(true, true, true), true);
  assert.equal(seamExitArmed(true, true, false), true);
});

// ── the body registry (future-proofing contract) ────────────────────────────

test("defaultBodyRegistry: all ten bodies declared, Earth is the only map anchor, the Sun the only light", () => {
  const reg = defaultBodyRegistry();
  assert.deepEqual(reg.map((b) => b.id), [...BODY_ORDER], "same canonical order");
  const anchors = reg.filter((b) => b.mapAnchor === "maplibre");
  assert.equal(anchors.length, 1, "exactly one live-map anchor");
  assert.equal(anchors[0].id, "earth", "the anchor is Earth (today)");
  assert.ok(reg.filter((b) => b.mapAnchor === null).length === reg.length - 1, "all others null");
  const emissive = reg.filter((b) => b.emissive);
  assert.equal(emissive.length, 1, "exactly one light source");
  assert.equal(emissive[0].id, "sun");
  // true radii, never invented: km = the ephemeris lib's meters / 1000
  for (const b of reg) {
    close(b.radiusKm, BODY_RADIUS_M[b.id as keyof typeof BODY_RADIUS_M] / 1000, 1e-9, `${b.id} radius`);
    assert.ok(/^#[0-9a-f]{6}$/i.test(b.color), `${b.id} presentation color`);
  }
});

test("registry ephemeris: positions are exactly the solarSystem.ts state (meters), memo-consistent", () => {
  const reg = defaultBodyRegistry();
  const t = Date.UTC(2026, 6, 18, 12);
  const state = solarSystemState(t);
  for (const b of reg) {
    const p = b.ephemeris(t);
    const ref = state.find((s) => s.id === b.id)!;
    close(p.x, ref.helioAu.x * AU_M, 1, `${b.id} x`);
    close(p.y, ref.helioAu.y * AU_M, 1, `${b.id} y`);
    close(p.z, ref.helioAu.z * AU_M, 1, `${b.id} z`);
  }
  // the shared single-instant memo returns identical objects on repeat calls
  assert.equal(reg[3].ephemeris(t), reg[3].ephemeris(t), "memoized instant");
  // and moves when time moves
  const later = reg[4].ephemeris(t + 86_400_000);
  const now = reg[4].ephemeris(t);
  assert.ok(Math.hypot(later.x - now.x, later.y - now.y, later.z - now.z) > 1e8, "bodies move");
});

// ── salvaged from the retired solarView (same tested semantics) ─────────────

test("layoutLabelStacks: bodies never move — colliding label ROWS stack into free slots", () => {
  // far apart horizontally (beyond a text row's width) → no offset
  assert.deepEqual(layoutLabelStacks([{ x: 100, y: 100 }, { x: 300, y: 100 }]), [0, 0]);
  // co-located anchors → the later label steps one row down
  assert.deepEqual(layoutLabelStacks([{ x: 100, y: 100 }, { x: 101, y: 100 }]), [0, LABEL_COLLIDE_PX]);
  // BOX-aware (the drive-round-1 cluster): anchors 15-25px apart never
  // collided under the old point rule while their ~130px text overprinted —
  // they must stack now
  const cluster = layoutLabelStacks([{ x: 712, y: 467 }, { x: 720, y: 477 }, { x: 725, y: 491 }]);
  const rows = cluster.map((o, i) => [467, 477, 491][i] + o);
  rows.sort((a, b) => a - b);
  for (let i = 1; i < rows.length; i++) {
    assert.ok(rows[i] - rows[i - 1] >= LABEL_COLLIDE_PX, `rows ${i - 1}/${i} separated (${rows.join(",")})`);
  }
  // three mutually co-located labels → a clean descending stack
  assert.deepEqual(
    layoutLabelStacks([{ x: 50, y: 50 }, { x: 51, y: 50 }, { x: 49, y: 51 }]),
    [0, LABEL_COLLIDE_PX, 2 * LABEL_COLLIDE_PX - 1],
  );
  // slot-filling: a label DROPS INTO the free gap between occupied rows
  // (rows at 0 and 28: the third label steps off row 0 and fits at 14)
  assert.deepEqual(
    layoutLabelStacks([{ x: 0, y: 0 }, { x: 0, y: 2 * LABEL_COLLIDE_PX }, { x: 0, y: 0 }]),
    [0, 0, LABEL_COLLIDE_PX],
  );
  // beyond the box width horizontally at identical y → independent
  assert.deepEqual(layoutLabelStacks([{ x: 0, y: 0 }, { x: 200, y: 0 }], 130), [0, 0]);
  // custom box width is honored
  assert.deepEqual(layoutLabelStacks([{ x: 0, y: 0 }, { x: 50, y: 0 }], 40), [0, 0]);
  // degenerate inputs never throw
  assert.deepEqual(layoutLabelStacks([]), []);
  assert.deepEqual(layoutLabelStacks([{ x: 5, y: 5 }]), [0]);
});

test("layoutLabelStacks: float fixed point terminates (the v1.0.396 freeze)", () => {
  // Root-caused 2026-07-18 with CDP Debugger.pause on the hung production
  // build: yj + LABEL_COLLIDE_PX rounds DOWN in doubles so the stepped
  // label still "collides" with the row it just cleared — the un-guarded
  // while(moved) loop reassigned the identical y forever, freezing the
  // main thread inside draw(). Minimal repro: two co-located labels AT the
  // fixed-point row. Pin the arithmetic first so the repro can never rot:
  const yj = 507.7892615010763; // captured from the live hang (row j=7)
  assert.ok(yj + LABEL_COLLIDE_PX - yj < LABEL_COLLIDE_PX,
    "precondition: yj+14 rounds below a full 14px step (the fixed point)");
  const off = layoutLabelStacks([{ x: 734, y: yj }, { x: 734, y: yj }]);
  assert.equal(off[0], 0);
  close(off[1], LABEL_COLLIDE_PX, 1e-9, "collider steps one row despite the rounding");

  // the FULL 9-anchor cluster captured from the hung frame (fly-home from
  // the 46.79 AU ceiling, all planet labels bunched at screen center) —
  // must terminate and keep every colliding pair ≥ one row apart (float ulp)
  const captured = [
    { x: 296.79522785883955, y: 582.83788911509 },
    { x: 824.9470585240113, y: 465.7892615010763 },
    { x: 712.6186448368059, y: 434.31606905960007 },
    { x: 743.7449151024564, y: 428.8568344688953 },
    { x: 737.340879383839, y: 427.50977870705674 },
    { x: 479.85371824806964, y: 431.78774920442225 },
    { x: 756.2369562475345, y: 425.17322912657295 },
    { x: 734.066901292434, y: 422.0088203633746 },
    { x: 734, y: 421.99999999999994 },
  ];
  const offs = layoutLabelStacks(captured); // hung forever before the guard
  for (let i = 0; i < captured.length; i++) {
    for (let j = 0; j < i; j++) {
      if (Math.abs(captured[i].x - captured[j].x) >= 130) continue;
      const sep = Math.abs(captured[i].y + offs[i] - (captured[j].y + offs[j]));
      assert.ok(sep >= LABEL_COLLIDE_PX - 1e-9,
        `rows ${j}/${i} separated to double precision (${sep})`);
    }
  }
});

test("distance labels: units-preference formatter below 0.01 AU, AU above", () => {
  const moon = fmtSpaceDistance(384_400_000);
  assert.ok(/mi$|km$/.test(moon), `moon label carries a unit (${moon})`);
  assert.ok(!moon.includes("AU"));
  assert.equal(fmtSpaceDistance(1.5 * AU_M), "1.500 AU");
  assert.equal(fmtSpaceDistance(30 * AU_M), "30.00 AU");
});
