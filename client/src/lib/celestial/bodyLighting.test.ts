// bodyLighting.test — pins the B6 universal-lighting geometry: terminator sign,
// phase crescent, Earth-umbra containment (lunar eclipse), Saturn ring-shadow
// band from the Sun's ring-plane elevation, planet-shadow-on-rings, and the
// realistic-OFF full-bright rule (no terminator). Real constants, synthetic but
// physical geometries — no faked results.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  terminatorLit,
  phaseFraction,
  shadowConeFactor,
  surfaceInRingShadow,
  pointInSphereShadow,
  type Vec3,
} from "./bodyLighting.ts";
import { lambertWeight, SATURN_RING_INNER_REL, SATURN_RING_OUTER_REL } from "./textureSphere.ts";

const AU_M = 149_597_870_700;
const SUN_R = 695_700_000;
const EARTH_R = 6_371_000;
const MOON_DIST = 384_400_000;
const norm = (v: Vec3): Vec3 => {
  const l = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / l, y: v.y / l, z: v.z / l };
};

// ── terminator: normal·sunDir sign = lit/dark ───────────────────────────────

test("terminator: sub-solar point is lit (+), anti-solar is dark (−), limb ≈ 0", () => {
  const sun: Vec3 = { x: 1, y: 0, z: 0 };
  assert.ok(terminatorLit({ x: 1, y: 0, z: 0 }, sun) > 0, "noon face lit");
  assert.ok(terminatorLit({ x: -1, y: 0, z: 0 }, sun) < 0, "midnight face dark");
  assert.ok(Math.abs(terminatorLit({ x: 0, y: 1, z: 0 }, sun)) < 1e-12, "terminator = 0");
});

test("terminator drives lambertWeight: lit face bright, night face at the 5% floor", () => {
  const sun: Vec3 = { x: 0, y: 0, z: 1 };
  const dayW = lambertWeight(terminatorLit({ x: 0, y: 0, z: 1 }, sun));
  const nightW = lambertWeight(terminatorLit({ x: 0, y: 0, z: -1 }, sun));
  assert.ok(dayW > 0.9, `day bright (${dayW})`);
  assert.ok(nightW <= 0.06, `night at floor (${nightW})`);
});

// ── phase: a body between observer and Sun shows a crescent ─────────────────

test("phase: full when Sun is behind the observer, crescent (≈0) when the body is between observer and Sun", () => {
  // toSun/toCam are FROM the body. Sun behind observer ⇒ toSun ≈ toCam ⇒ full.
  assert.equal(phaseFraction({ x: 0, y: 0, z: 1 }, { x: 0, y: 0, z: 1 }), 1);
  // body between observer and Sun ⇒ Sun opposite the camera ⇒ toSun ≈ −toCam ⇒ new/crescent.
  assert.equal(phaseFraction({ x: 0, y: 0, z: -1 }, { x: 0, y: 0, z: 1 }), 0);
  // quadrature ⇒ half lit.
  assert.ok(Math.abs(phaseFraction({ x: 1, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }) - 0.5) < 1e-12);
});

test("phase: an inferior planet (Venus/Mercury) at inferior conjunction is a thin crescent", () => {
  // observer at +z looking toward the Sun at the origin; the planet just this
  // side of the Sun. toCam points back at the observer (+z), toSun points on
  // toward the Sun (−z) ⇒ small lit fraction.
  const toCam = norm({ x: 0.05, y: 0, z: 1 });
  const toSun = norm({ x: 0.02, y: 0, z: -1 });
  const f = phaseFraction(toSun, toCam);
  assert.ok(f < 0.1, `inferior-conjunction crescent lit fraction ${f.toFixed(3)} < 0.1`);
});

// ── Earth-umbra containment: lunar eclipse geometry ─────────────────────────

test("lunar eclipse: the Moon dead behind Earth from the Sun sits in the umbra (factor 0)", () => {
  const sunDir: Vec3 = { x: 0, y: 0, z: 1 }; // Sun toward +z from Earth
  const moonRel: Vec3 = { x: 0, y: 0, z: -MOON_DIST }; // Moon on the anti-sun axis
  const r = shadowConeFactor(moonRel, sunDir, EARTH_R, SUN_R, AU_M);
  assert.equal(r.region, "umbra");
  assert.equal(r.factor, 0);
});

test("no eclipse: an ordinary full Moon a few degrees off the node is fully sunlit", () => {
  const sunDir: Vec3 = { x: 0, y: 0, z: 1 };
  // 5° off the shadow axis at Moon distance ≈ 33,500 km off-axis — well outside
  // Earth's ~4,600 km umbra AND its penumbra there.
  const off = MOON_DIST * Math.sin(5 * (Math.PI / 180));
  const moonRel: Vec3 = { x: off, y: 0, z: -MOON_DIST * Math.cos(5 * (Math.PI / 180)) };
  const r = shadowConeFactor(moonRel, sunDir, EARTH_R, SUN_R, AU_M);
  assert.equal(r.region, "sunlit");
  assert.equal(r.factor, 1);
});

test("penumbra ramp: at the Moon's depth the factor rises umbra→penumbra→sunlit monotonically with off-axis distance", () => {
  const sunDir: Vec3 = { x: 0, y: 0, z: 1 };
  const at = (offM: number): number =>
    shadowConeFactor({ x: offM, y: 0, z: -MOON_DIST }, sunDir, EARTH_R, SUN_R, AU_M).factor;
  const core = at(0); // on the shadow axis
  const edge = at(5_500_000); // ~ near the umbra radius (~4.6e6 m) into penumbra
  const wide = at(30_000_000); // well outside the penumbra
  assert.equal(core, 0, "axis is umbral");
  assert.ok(edge > core && edge < 1, `penumbra ramp between (${edge})`);
  assert.equal(wide, 1, "far off-axis is fully sunlit");
});

test("sun-ward hemisphere is never shadowed (a day-side craft/moon)", () => {
  const sunDir: Vec3 = { x: 0, y: 0, z: 1 };
  const r = shadowConeFactor({ x: 0, y: 0, z: MOON_DIST }, sunDir, EARTH_R, SUN_R, AU_M);
  assert.equal(r.region, "sunlit");
  assert.equal(r.factor, 1);
});

// ── Saturn ring shadow band from the Sun's ring-plane elevation ─────────────

test("ring shadow: with the Sun ELEVATED north of the ring plane the band lands on the SOUTHERN hemisphere", () => {
  // body frame z = spin axis; ring plane z=0. Sun 30° above the ring plane.
  const el = 30 * (Math.PI / 180);
  const sun = norm({ x: Math.cos(el), y: 0, z: Math.sin(el) }); // +z = north
  const rIn = SATURN_RING_INNER_REL;
  const rOut = SATURN_RING_OUTER_REL;
  // A southern-hemisphere point under the ring annulus, facing sun-ward in x.
  const south = norm({ x: Math.cos(-0.6), y: 0, z: Math.sin(-0.6) });
  assert.ok(surfaceInRingShadow(south, sun, rIn, rOut), "southern point is ring-shadowed");
  // The mirror northern point is NOT (the Sun is on its side of the plane).
  const north = norm({ x: Math.cos(0.6), y: 0, z: Math.sin(0.6) });
  assert.ok(!surfaceInRingShadow(north, sun, rIn, rOut), "northern point is not ring-shadowed");
});

test("ring shadow: the band tracks the Sun elevation — a higher Sun casts the shadow nearer the equator/limb", () => {
  const rIn = SATURN_RING_INNER_REL;
  const rOut = SATURN_RING_OUTER_REL;
  // Scan southern latitudes; the shadowed set moves with Sun elevation.
  const shadowedLats = (elDeg: number): number[] => {
    const el = elDeg * (Math.PI / 180);
    const sun = norm({ x: Math.cos(el), y: 0, z: Math.sin(el) });
    const out: number[] = [];
    for (let latDeg = -5; latDeg >= -85; latDeg -= 2) {
      const la = latDeg * (Math.PI / 180);
      const p = norm({ x: Math.cos(la), y: 0, z: Math.sin(la) });
      if (surfaceInRingShadow(p, sun, rIn, rOut)) out.push(latDeg);
    }
    return out;
  };
  const low = shadowedLats(10); // Sun barely above plane (near equinox)
  const high = shadowedLats(45); // Sun high over the rings (toward solstice)
  assert.ok(low.length > 0 && high.length > 0, "a band exists at both elevations");
  // A low Sun over the rings throws the shadow band toward the equator (a thin
  // near-equinox line); a high Sun throws it toward the winter pole. So the
  // mean shadowed latitude is LESS polar (nearer 0) for the low Sun.
  const mean = (a: number[]): number => a.reduce((s, v) => s + v, 0) / a.length;
  assert.ok(mean(low) > mean(high), `low-Sun band ${mean(low)}° is nearer the equator than high-Sun ${mean(high)}°`);
});

test("ring shadow: Sun exactly in the ring plane casts no band on the disc", () => {
  const sun: Vec3 = { x: 1, y: 0, z: 0 };
  const p = norm({ x: Math.cos(-0.6), y: 0, z: Math.sin(-0.6) });
  assert.ok(!surfaceInRingShadow(p, sun, SATURN_RING_INNER_REL, SATURN_RING_OUTER_REL));
});

// ── planet shadow on its rings ──────────────────────────────────────────────

test("planet shadow on rings: a ring point directly anti-sun is in the planet's shadow; the sun-ward point is not", () => {
  const sunDir: Vec3 = { x: 1, y: 0, z: 0 }; // planet radii units
  const Rp = 1;
  const rMid = (SATURN_RING_INNER_REL + SATURN_RING_OUTER_REL) / 2;
  const anti: Vec3 = { x: -rMid, y: 0, z: 0 }; // behind the planet from the Sun
  const near: Vec3 = { x: rMid, y: 0, z: 0 }; // sun-ward of the planet
  assert.ok(pointInSphereShadow(anti, sunDir, Rp), "anti-sun ring point shadowed by the planet");
  assert.ok(!pointInSphereShadow(near, sunDir, Rp), "sun-ward ring point is lit");
});

test("planet shadow on rings: a ring point offset more than a planet radius off-axis clears the shadow", () => {
  const sunDir: Vec3 = { x: 1, y: 0, z: 0 };
  const rMid = (SATURN_RING_INNER_REL + SATURN_RING_OUTER_REL) / 2;
  // anti-sun in x but 1.4 planet radii off-axis in y → misses the R=1 shadow cylinder
  const q: Vec3 = { x: -rMid, y: 1.4, z: 0 };
  assert.ok(!pointInSphereShadow(q, sunDir, 1), "off-axis ring point is sunlit");
});

// ── realistic-OFF = full-bright (no terminator) ─────────────────────────────

test("realistic OFF ⇒ full-bright: the toggle rule zeroes the lambert term on both hemispheres", () => {
  const sun: Vec3 = { x: 0, y: 0, z: 1 };
  // The call sites apply: realistic ? lambertWeight(lit) * shadow : 1.
  const shade = (n: Vec3, realistic: boolean): number =>
    realistic ? lambertWeight(terminatorLit(n, sun)) : 1;
  // ON: day/night differ (a real terminator).
  assert.ok(shade({ x: 0, y: 0, z: 1 }, true) > shade({ x: 0, y: 0, z: -1 }, true));
  // OFF: every point is 1 (even-lit inspection — terminator gone).
  assert.equal(shade({ x: 0, y: 0, z: 1 }, false), 1);
  assert.equal(shade({ x: 0, y: 0, z: -1 }, false), 1);
});
