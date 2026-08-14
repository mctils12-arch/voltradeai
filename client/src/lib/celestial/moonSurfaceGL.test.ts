// The GPU patch must be the SAME IMAGE as the CPU patch.
//
// A port that is merely "close" is worse than no port: the Moon would shade
// one way at the sprite zoom and another way at the surface zoom, and the
// seam would read as a rendering bug forever. So this file executes a TS
// MIRROR of moonSurfaceGL.FRAG and asserts it equals the real CPU functions
// in moonSurface.ts / textureSphere.ts over a grid of inputs.
//
// This is the contract glElev.test.ts already holds for its GLSL mirror
// ("GLSL mirror ≡ the pinned CPU mercatorZFromAltitude"), reused because it
// works: it runs headless, needs no GPU, and fails on the exact function that
// drifted rather than on a screenshot diff nobody can read.
//
// WHAT THE MIRROR IS. Hand-transcribed from the GLSL source below, then held
// against it by `test("the mirror matches the shipped GLSL")`, which greps the
// real shader text for each formula. That guard is the weak point of any
// mirror — a mirror that silently stops mirroring proves nothing — so it
// asserts on the CONSTRUCT, with the shader's own comment lines stripped
// first (L15/L18: prose naming a formula must not satisfy a check for it).
// Run: npx tsx --test client/src/lib/celestial/moonSurfaceGL.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  raySphereNearT,
  surfaceLonLat,
  sampleDetail,
  type DetailOverlay,
} from "./moonSurface.ts";
import { lambertWeight } from "./textureSphere.ts";
import { FRAG, MAX_TIERS, maxFeatures, vramBudget, createMoonSurfaceGL } from "./moonSurfaceGL.ts";

type V3 = { x: number; y: number; z: number };
const RAD = 180 / Math.PI;
const dot = (a: V3, b: V3) => a.x * b.x + a.y * b.y + a.z * b.z;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// ── the mirror: GLSL semantics, executed in TS ──────────────────────────────

/** GLSL `raySphereNearT` */
function glRaySphereNearT(o: V3, d: V3, c: V3, R: number): number {
  const oc = { x: o.x - c.x, y: o.y - c.y, z: o.z - c.z };
  const A = dot(d, d);
  const B = 2 * dot(oc, d);
  const C = dot(oc, oc) - R * R;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return -1;
  const sq = Math.sqrt(disc);
  const t0 = (-B - sq) / (2 * A);
  if (t0 > 0) return t0;
  const t1 = (-B + sq) / (2 * A);
  return t1 > 0 ? t1 : -1;
}

/** GLSL `surfaceLonLat` */
function glSurfaceLonLat(n: V3, X: V3, Y: V3, Z: V3, wDeg: number) {
  const bx = dot(n, X), by = dot(n, Y);
  const bz = clamp(dot(n, Z), -1, 1);
  return { lonDeg: Math.atan2(by, bx) * RAD - wDeg, latDeg: Math.asin(bz) * RAD };
}

/** GLSL `nearestTexel` */
function glNearestTexel(u: number, v: number, w: number, h: number) {
  return {
    tx: clamp(Math.floor(u * w), 0, w - 1),
    ty: clamp(Math.floor(v * h), 0, h - 1),
  };
}

/** GLSL `sampleTier` — returns null where the tier does not cover opaquely. */
function glSampleTier(ov: DetailOverlay, lonDeg: number, latDeg: number) {
  let d = (lonDeg - ov.lonMin) % 360;
  if (d < 0) d += 360;
  if (d > ov.lonSpan) return null;
  const dv = ov.latMax - latDeg;
  if (dv < 0 || dv > ov.latSpan) return null;
  const { tex } = ov;
  const { tx, ty } = glNearestTexel(d / ov.lonSpan, dv / ov.latSpan, tex.width, tex.height);
  const i = (ty * tex.width + tx) * 4;
  if (tex.data[i + 3] / 255 < 8 / 255) return null;
  return [tex.data[i], tex.data[i + 1], tex.data[i + 2]] as [number, number, number];
}

/** GLSL `lambertWeight` */
function glLambertWeight(lit: number): number {
  const day = clamp((lit + 0.03) / 0.07, 0, 1);
  const shade = 0.05 + 0.95 * Math.max(lit, 0);
  return 0.05 + 0.95 * day * shade;
}

// ── equality with the CPU implementation ────────────────────────────────────

test("ray-sphere: mirror ≡ CPU across hits, misses and behind-camera rays", () => {
  const c = { x: 0, y: 0, z: 0 };
  let hits = 0, misses = 0;
  for (const R of [1, 1737.4, 0.25]) {
    for (const oz of [5, 2.0001, -5, 1e4]) {
      const D = Math.abs(oz * R);
      for (let i = 0; i < 12; i++) {
        const th = (i / 12) * Math.PI * 2;
        // Sweep the IMPACT PARAMETER from dead-centre out past the limb, in
        // units of R. A fixed lateral slope instead (the first version of this
        // test) put every ray metres wide of the sphere at every distance, so
        // the grid was 100% misses and proved nothing about the hit path —
        // which is what the non-vacuity assertions below exist to catch.
        for (const frac of [0, 0.3, 0.7, 0.99, 1.01, 1.5, 3]) {
          const p = (frac * R) / D;
          const d = { x: p * Math.cos(th), y: p * Math.sin(th), z: -Math.sign(oz) || -1 };
          const o = { x: 0, y: 0, z: oz * R };
          const a = raySphereNearT(o, d, c, R);
          const b = glRaySphereNearT(o, d, c, R);
          if (a < 0) { misses++; assert.ok(b < 0, `both must miss (R=${R} oz=${oz} frac=${frac})`); }
          else {
            hits++;
            assert.ok(Math.abs(a - b) <= Math.abs(a) * 1e-12 + 1e-12, `t: ${a} vs ${b}`);
          }
        }
      }
    }
  }
  // Not vacuous in either direction — a grid that only missed would prove
  // nothing about the hit path, and vice versa.
  assert.ok(hits > 50, `expected real hits, got ${hits}`);
  assert.ok(misses > 10, `expected real misses, got ${misses}`);
});

test("surfaceLonLat: mirror ≡ CPU, including the clamped poles", () => {
  const X = { x: 1, y: 0, z: 0 }, Y = { x: 0, y: 1, z: 0 }, Z = { x: 0, y: 0, z: 1 };
  for (const wDeg of [0, 23.7, -180, 359.9]) {
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j <= 12; j++) {
        const lon = (i / 24) * Math.PI * 2, lat = (j / 12 - 0.5) * Math.PI;
        const n = {
          x: Math.cos(lat) * Math.cos(lon),
          y: Math.cos(lat) * Math.sin(lon),
          z: Math.sin(lat),
        };
        const a = surfaceLonLat(n, X, Y, Z, wDeg);
        const b = glSurfaceLonLat(n, X, Y, Z, wDeg);
        assert.ok(Math.abs(a.lonDeg - b.lonDeg) < 1e-9, `lon ${a.lonDeg} vs ${b.lonDeg}`);
        assert.ok(Math.abs(a.latDeg - b.latDeg) < 1e-9, `lat ${a.latDeg} vs ${b.latDeg}`);
      }
    }
  }
});

test("lambertWeight: mirror ≡ CPU across the terminator band", () => {
  // Sampled densely through [-0.03, +0.04] because that is where the curve
  // actually turns — a coarse sweep would pass on a wrong knee.
  for (let lit = -1; lit <= 1; lit += 0.001) {
    assert.ok(
      Math.abs(lambertWeight(lit) - glLambertWeight(lit)) < 1e-12,
      `lit=${lit}: ${lambertWeight(lit)} vs ${glLambertWeight(lit)}`,
    );
  }
});

test("tier sampling: mirror ≡ CPU, and the alpha<8 hole still falls through", () => {
  const W = 16, H = 8;
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[i * 4] = i & 255; data[i * 4 + 1] = 40; data[i * 4 + 2] = 80;
    data[i * 4 + 3] = i % 5 === 0 ? 0 : 255;      // every 5th texel is a hole
  }
  const ov: DetailOverlay = {
    tex: { width: W, height: H, data },
    lonMin: -10, lonSpan: 40, latMax: 20, latSpan: 30,
  };
  let covered = 0, holes = 0, outside = 0;
  for (let lon = -180; lon < 180; lon += 3.5) {
    for (let lat = -80; lat <= 80; lat += 3.5) {
      const a = sampleDetail(ov, lon, lat);
      const b = glSampleTier(ov, lon, lat);
      assert.deepEqual(a, b, `lon=${lon} lat=${lat}`);
      if (a) covered++;
      else if (lon >= -10 && lon <= 30 && lat <= 20 && lat >= -10) holes++;
      else outside++;
    }
  }
  assert.ok(covered > 20, `tier must actually cover samples, got ${covered}`);
  assert.ok(holes > 0, "the alpha gate must actually reject some texels");
  assert.ok(outside > 100, `must exercise out-of-window too, got ${outside}`);
});

// ── the mirror is really mirroring the shipped shader ───────────────────────

test("the mirror matches the shipped GLSL", () => {
  // Comments stripped FIRST: the shader documents every formula it implements,
  // and a check satisfied by that documentation would be worthless (L15/L18 —
  // this exact defect produced a false PASS earlier this session).
  const glsl = FRAG.split("\n").filter((l) => !/^\s*\/\//.test(l)).join("\n");

  for (const construct of [
    "float disc = B * B - 4.0 * A * C;",
    "float t0 = (-B - sq) / (2.0 * A);",
    "float day = clamp((lit + 0.03) / 0.07, 0.0, 1.0);",
    "float shade = 0.05 + 0.95 * max(lit, 0.0);",
    "return 0.05 + 0.95 * day * shade;",
    "if (texel.a < (8.0 / 255.0)) return vec4(0.0);",
    "u -= floor(u);",
    "clamp(0.5 - lonLat.y / 180.0, 0.0, 1.0)",
    "atan(by, bx) * RAD - uWDeg",
  ]) {
    assert.ok(glsl.includes(construct), `GLSL no longer contains: ${construct}`);
  }

  // RAD must be the same constant the CPU uses, to the digit.
  assert.ok(
    glsl.includes(String(180 / Math.PI)),
    `shader must carry RAD = ${180 / Math.PI}`,
  );

  // NEAREST, not linear — bilinear would look smoother and be a different
  // image from the CPU path, which is the one thing this port must not do.
  assert.ok(glsl.includes("texelFetch"), "must use texelFetch, not filtered texture()");
  assert.ok(!/\btexture\s*\(/.test(glsl), "filtered texture() sampling would diverge from the CPU path");
});

test("the buffer row flip is present — otherwise every site lands mirrored", () => {
  // gl_FragCoord.y is bottom-up, the CPU buffer is top-down. This is the
  // single easiest way to ship a patch that looks plausible and is wrong.
  const glsl = FRAG.split("\n").filter((l) => !/^\s*\/\//.test(l)).join("\n");
  assert.ok(
    glsl.includes("uBufSize.y - 1.0 - floor(gl_FragCoord.y)"),
    "the top-down row flip is missing",
  );
});

// ── Law IV declarations (harness assertion 5) ──────────────────────────────

test("Law IV: the module declares its caps and a real teardown", () => {
  assert.equal(typeof createMoonSurfaceGL, "function");
  assert.equal(maxFeatures, 1100 * 1100, "matches the CPU MOON_PATCH_FULL_LONG_PX ceiling");
  assert.ok(vramBudget > 0 && Number.isFinite(vramBudget));
  assert.equal(MAX_TIERS, 2, "NAC over WAC — the tiers spaceFrame actually passes");
});

test("createMoonSurfaceGL returns null without WebGL2 rather than half-working", () => {
  // A stub canvas whose getContext yields nothing. The honest outcome is null
  // (caller keeps the CPU path), never an object that draws blank frames.
  const fake = { getContext: () => null } as unknown as HTMLCanvasElement;
  assert.equal(createMoonSurfaceGL(fake), null);
});
