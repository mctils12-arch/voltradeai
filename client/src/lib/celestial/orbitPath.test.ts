// orbitPath tests — sampling from the real ephemeris (closure, point count,
// local offsets), the scaleModel layout pass-through (identity at c=0,
// per-point compressDistance otherwise), staleness, and the pref store.
// Celestial v2 B3. `npx tsx --test`.

import test from "node:test";
import assert from "node:assert/strict";

import {
  ORBIT_SAMPLES_PLANET,
  ORBIT_SAMPLES_MOON,
  ORBIT_RESAMPLE_MS,
  sampleOrbitPolyline,
  layoutOrbitPolyline,
  orbitPolylineStale,
  getOrbitPathsPref,
  setOrbitPathsPref,
  subscribeOrbitPathsPref,
  type Vec3,
} from "./orbitPath.js";
import { compressDistance, AU_M } from "./scaleModel.js";
import { moonLocalOffsetEclM, moonOrbitPeriodDays } from "./moons.js";
import { solarSystemState } from "./solarSystem.js";

const len = (x: number, y: number, z: number): number => Math.hypot(x, y, z);

test("sampling: point count, exact circle closure, and local-offset mode", () => {
  // synthetic circular orbit — closure must be exact to float precision
  const P = 100; // days
  const eph = (t: number): Vec3 => {
    const a = (2 * Math.PI * (t / 86_400_000)) / P;
    return { x: Math.cos(a) * 1e9, y: Math.sin(a) * 1e9, z: 0 };
  };
  const line = sampleOrbitPolyline("x", null, eph, null, 0, P, 64);
  assert.equal(line.pts.length, 64 * 3, "3 floats per vertex");
  assert.ok(Math.abs(line.maxRadiusM - 1e9) < 1, "max radius recorded");
  // the loop start is the ephemeris at t0; the segment last→first (renderer
  // join) spans exactly one sample step of a closed curve
  const gap = len(line.pts[0] - line.pts[63 * 3], line.pts[1] - line.pts[63 * 3 + 1], line.pts[2] - line.pts[63 * 3 + 2]);
  const step = (2 * Math.PI * 1e9) / 64;
  assert.ok(Math.abs(gap - step) < step * 0.01, "uniform closing segment");
  // parent mode: offsets are eph − parent
  const parent = (_t: number): Vec3 => ({ x: 5e8, y: -5e8, z: 1e8 });
  const local = sampleOrbitPolyline("x", "p", eph, parent, 0, P, 8);
  assert.equal(local.parentId, "p");
  const p0 = eph(0);
  assert.equal(local.pts[0], p0.x - 5e8);
  assert.equal(local.pts[1], p0.y + 5e8);
  assert.equal(local.pts[2], p0.z - 1e8);
});

test("real ephemeris: Mars' heliocentric loop closes to the perturbation scale", () => {
  const t0 = Date.UTC(2026, 6, 18);
  const mars = (t: number): Vec3 => {
    const s = solarSystemState(t).find((b) => b.id === "mars")!;
    return { x: s.helioAu.x * AU_M, y: s.helioAu.y * AU_M, z: s.helioAu.z * AU_M };
  };
  const line = sampleOrbitPolyline("mars", null, mars, null, t0, 686.98, ORBIT_SAMPLES_PLANET);
  assert.equal(line.pts.length, ORBIT_SAMPLES_PLANET * 3);
  const n = ORBIT_SAMPLES_PLANET - 1;
  const gap = len(line.pts[0] - line.pts[n * 3], line.pts[1] - line.pts[n * 3 + 1], line.pts[2] - line.pts[n * 3 + 2]);
  // one sample step ≈ 2π·1.52 AU/192; element drift over one Mars year is
  // far smaller — the closing segment must look like any other segment
  const step = (2 * Math.PI * 1.5237 * AU_M) / ORBIT_SAMPLES_PLANET;
  assert.ok(gap < step * 1.5, `closing segment ${(gap / 1e9).toFixed(2)} Gm ≈ step ${(step / 1e9).toFixed(2)} Gm`);
  assert.ok(line.maxRadiusM > 1.3 * AU_M && line.maxRadiusM < 1.7 * AU_M, "aphelion-ish max radius");
});

test("real ephemeris: Io's local loop stays on its ellipse", () => {
  const t0 = Date.UTC(2026, 6, 18);
  const line = sampleOrbitPolyline(
    "io", "jupiter",
    (t) => moonLocalOffsetEclM("io", t),
    // moonLocalOffsetEclM is already parent-centric — identity parent
    null,
    t0, moonOrbitPeriodDays("io"), ORBIT_SAMPLES_MOON,
  );
  for (let i = 0; i < ORBIT_SAMPLES_MOON; i++) {
    const r = len(line.pts[i * 3], line.pts[i * 3 + 1], line.pts[i * 3 + 2]);
    assert.ok(Math.abs(r - 421_800_000) / 421_800_000 < 0.01, `Io vertex ${i} on the a=421,800 km ellipse`);
  }
});

test("layout pass-through: c=0 is the bit-exact identity; c>0 compresses each point exactly like a body position", () => {
  const pts = new Float64Array([
    0.3 * AU_M, 0.1 * AU_M, -0.05 * AU_M,
    -2 * AU_M, 1.5 * AU_M, 0.2 * AU_M,
    25 * AU_M, -12 * AU_M, 3 * AU_M,
  ]);
  const id = layoutOrbitPolyline(pts, 0);
  for (let i = 0; i < pts.length; i++) assert.equal(id[i], pts[i], `identity at ${i}`);
  for (const c of [0.25, 1]) {
    const out = layoutOrbitPolyline(pts, c);
    for (let v = 0; v < 3; v++) {
      const r = len(pts[v * 3], pts[v * 3 + 1], pts[v * 3 + 2]);
      const rOut = len(out[v * 3], out[v * 3 + 1], out[v * 3 + 2]);
      assert.ok(Math.abs(rOut - compressDistance(r, c)) < 1e-3, `vertex ${v} radius = compressDistance at c=${c}`);
      // direction preserved: cross product ≈ 0
      const cx = pts[v * 3 + 1] * out[v * 3 + 2] - pts[v * 3 + 2] * out[v * 3 + 1];
      const cy = pts[v * 3 + 2] * out[v * 3] - pts[v * 3] * out[v * 3 + 2];
      const cz = pts[v * 3] * out[v * 3 + 1] - pts[v * 3 + 1] * out[v * 3];
      assert.ok(len(cx, cy, cz) / (r * rOut) < 1e-12, `vertex ${v} direction preserved`);
    }
  }
  // in-place reuse: the memo buffer is filled, not reallocated
  const buf = new Float64Array(pts.length);
  const out2 = layoutOrbitPolyline(pts, 0.5, buf);
  assert.equal(out2, buf, "fills the caller's buffer");
});

test("staleness: resample beyond ORBIT_RESAMPLE_MS of clock drift, either direction", () => {
  const t0 = Date.UTC(2026, 6, 18);
  assert.equal(orbitPolylineStale(t0, t0), false);
  assert.equal(orbitPolylineStale(t0, t0 + ORBIT_RESAMPLE_MS - 1), false);
  assert.equal(orbitPolylineStale(t0, t0 + ORBIT_RESAMPLE_MS + 1), true);
  assert.equal(orbitPolylineStale(t0, t0 - ORBIT_RESAMPLE_MS - 1), true, "backward warp counts");
});

test("pref store: default on, set/subscribe round trip", () => {
  assert.equal(getOrbitPathsPref(), true, "default ON");
  let fired = 0;
  const off = subscribeOrbitPathsPref(() => { fired++; });
  setOrbitPathsPref(false);
  assert.equal(getOrbitPathsPref(), false);
  assert.equal(fired, 1);
  setOrbitPathsPref(false);
  assert.equal(fired, 1, "no-op set does not notify");
  setOrbitPathsPref(true);
  assert.equal(fired, 2);
  off();
  setOrbitPathsPref(false);
  assert.equal(fired, 2, "unsubscribed");
  setOrbitPathsPref(true); // leave default for other tests
});
