// moonSurface.test — ray-cast geometry, the (node frame, W) orientation
// (alignment to the base-8k equirect convention), the detail-window sampler,
// and a full patch render on synthetic textures.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  raySphereNearT,
  surfaceLonLat,
  sampleDetail,
  renderMoonSurfaceRows,
  type DetailOverlay,
  type MoonSurfaceView,
  type Vec3,
} from "./moonSurface.ts";
import { equirectUV, type TexLike } from "./textureSphere.ts";

const solid = (w: number, h: number, r: number, g: number, b: number): TexLike => {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return { data, width: w, height: h };
};

// ── ray-sphere ──────────────────────────────────────────────────────────────

test("ray straight at a unit sphere hits at the near surface", () => {
  const o = { x: 0, y: 0, z: 10 };
  const d = { x: 0, y: 0, z: -1 };
  const t = raySphereNearT(o, d, { x: 0, y: 0, z: 0 }, 1);
  assert.ok(Math.abs(t - 9) < 1e-9, `t=${t}`);
});

test("a ray missing the sphere returns -1", () => {
  const t = raySphereNearT({ x: 0, y: 0, z: 10 }, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 0 }, 1);
  assert.equal(t, -1);
});

test("non-unit ray direction still yields the correct hit point", () => {
  const o = { x: 0, y: 0, z: 10 };
  const d = { x: 0, y: 0, z: -2 }; // magnitude 2
  const t = raySphereNearT(o, d, { x: 0, y: 0, z: 0 }, 1);
  // hit point z = 10 + t*(-2) should be +1
  assert.ok(Math.abs(10 + t * -2 - 1) < 1e-9);
});

// ── orientation: alignment to the base equirect (equirectUV) convention ─────

const Xax = { x: 1, y: 0, z: 0 };
const Yax = { x: 0, y: 1, z: 0 };
const Zax = { x: 0, y: 0, z: 1 };
const normalFromLonLat = (lonDeg: number, latDeg: number): Vec3 => {
  const lon = (lonDeg * Math.PI) / 180;
  const lat = (latDeg * Math.PI) / 180;
  return { x: Math.cos(lat) * Math.cos(lon), y: Math.cos(lat) * Math.sin(lon), z: Math.sin(lat) };
};

test("surfaceLonLat inverts the node-frame lon/lat with W=0", () => {
  for (const [lon, lat] of [[0, 0], [30, 20], [-45, -60], [120, 10]] as const) {
    const n = normalFromLonLat(lon, lat);
    const s = surfaceLonLat(n, Xax, Yax, Zax, 0);
    assert.ok(Math.abs(s.lonDeg - lon) < 1e-6, `lon ${s.lonDeg} vs ${lon}`);
    assert.ok(Math.abs(s.latDeg - lat) < 1e-6, `lat ${s.latDeg} vs ${lat}`);
  }
});

test("W offset shifts body longitude (prime-meridian rotation)", () => {
  const n = normalFromLonLat(50, 0);
  const s = surfaceLonLat(n, Xax, Yax, Zax, 30);
  assert.ok(Math.abs(s.lonDeg - 20) < 1e-6, `got ${s.lonDeg}`); // 50 - 30
});

test("the node-frame lon/lat feeds equirectUV exactly like the base texture", () => {
  // a normal at (lon,lat) must sample the base at equirectUV(lon,lat) —
  // this is the pixel-for-pixel alignment contract with the disc sprite.
  const n = normalFromLonLat(-11.36, -43.31); // Tycho
  const { lonDeg, latDeg } = surfaceLonLat(n, Xax, Yax, Zax, 0);
  const uv = equirectUV(lonDeg, latDeg);
  const ref = equirectUV(-11.36, -43.31);
  assert.ok(Math.abs(uv.u - ref.u) < 1e-9 && Math.abs(uv.v - ref.v) < 1e-9);
});

// ── detail-window sampler ───────────────────────────────────────────────────

test("sampleDetail returns in-window texels and null outside", () => {
  const ov: DetailOverlay = {
    tex: solid(4, 4, 10, 20, 30),
    lonMin: -20,
    lonSpan: 40,
    latMax: 20,
    latSpan: 40,
  };
  assert.deepEqual(sampleDetail(ov, 0, 0), [10, 20, 30]); // centre
  assert.equal(sampleDetail(ov, 90, 0), null); // east of window
  assert.equal(sampleDetail(ov, 0, 80), null); // north of window
  assert.equal(sampleDetail(ov, 0, -80), null); // south of window
});

test("sampleDetail unwraps longitude across the seam", () => {
  const ov: DetailOverlay = {
    tex: solid(2, 2, 1, 2, 3),
    lonMin: 170, // window 170..190 (crosses +180)
    lonSpan: 20,
    latMax: 5,
    latSpan: 10,
  };
  assert.deepEqual(sampleDetail(ov, 175, 0), [1, 2, 3]);
  assert.deepEqual(sampleDetail(ov, -175, 0), [1, 2, 3]); // == 185, inside
  assert.equal(sampleDetail(ov, 160, 0), null);
});

// ── full patch render ───────────────────────────────────────────────────────

function frontalView(bw: number, bh: number, base: TexLike, detail: DetailOverlay | null): {
  view: MoonSurfaceView;
  out: Uint8ClampedArray;
} {
  // camera on +X looking toward -X (f = -X); pole Z = +Z is screen-up (u),
  // node X = +X faces the camera ⇒ the sub-camera point is lon 0, lat 0 (the
  // prime-meridian/equator crossing). Screen-right r = +Y. Full-lit (sun
  // behind the camera, +X).
  const k = 400;
  const view: MoonSurfaceView = {
    bw,
    bh,
    originX: 0,
    originY: 0,
    stepX: 1,
    stepY: 1,
    cx: bw / 2,
    cy: bh / 2,
    k,
    r: { x: 0, y: 1, z: 0 },
    u: { x: 0, y: 0, z: 1 },
    f: { x: -1, y: 0, z: 0 },
    cam: { x: 10, y: 0, z: 0 },
    center: { x: 0, y: 0, z: 0 },
    radius: 1,
    X: Xax,
    Y: Yax,
    Z: Zax,
    wDeg: 0,
    sun: { x: 1, y: 0, z: 0 }, // toward the camera ⇒ near side fully lit
  };
  return { view, out: new Uint8ClampedArray(bw * bh * 4) };
}

test("centre pixel hits the sub-camera point; corners miss (transparent)", () => {
  const base = solid(8, 8, 200, 200, 200);
  const { view, out } = frontalView(64, 64, base, null);
  renderMoonSurfaceRows(view, base, null, out, 0, 64);
  const centre = (32 * 64 + 32) * 4;
  assert.equal(out[centre + 3], 255, "centre is opaque (hit)");
  assert.ok(out[centre] > 150, "centre lit near-white");
  const corner = (0 * 64 + 0) * 4;
  assert.equal(out[corner + 3], 0, "corner misses the disc");
});

test("detail window paints over the base where it covers, base elsewhere", () => {
  const base = solid(8, 8, 200, 50, 50); // reddish base
  const detail: DetailOverlay = {
    tex: solid(4, 4, 50, 50, 200), // blue detail
    lonMin: -10,
    lonSpan: 20,
    latMax: 10,
    latSpan: 20, // small window around the sub-camera point
  };
  const { view, out } = frontalView(64, 64, base, detail);
  renderMoonSurfaceRows(view, base, detail, out, 0, 64);
  const c = (32 * 64 + 32) * 4;
  // sub-camera point is inside the detail window ⇒ blue-dominant
  assert.ok(out[c + 2] > out[c], "centre uses blue detail");
  // a hit near the limb (far from centre) falls outside the ±10° window ⇒
  // red-dominant base. Scan along the row for the outermost opaque pixel.
  let limbIdx = -1;
  for (let bx = 63; bx >= 0; bx--) {
    if (out[(32 * 64 + bx) * 4 + 3] === 255) { limbIdx = bx; break; }
  }
  assert.ok(limbIdx >= 0);
  const l = (32 * 64 + limbIdx) * 4;
  assert.ok(out[l] > out[l + 2], "limb uses red base (outside detail window)");
});

test("terminator: a pixel on the far-from-sun limb is darker than the sub-solar point", () => {
  const base = solid(8, 8, 200, 200, 200);
  const { view, out } = frontalView(80, 80, base, null);
  // light from +Y (screen-right) so the +Y limb is bright, the -Y limb dark
  view.sun = { x: 0, y: 1, z: 0 };
  renderMoonSurfaceRows(view, base, null, out, 0, 80);
  // find opaque pixels near the left (-X) and right (+X) limb on the centre row
  const row = 40;
  let left = -1;
  let right = -1;
  for (let bx = 0; bx < 80; bx++) {
    if (out[(row * 80 + bx) * 4 + 3] === 255) { left = bx; break; }
  }
  for (let bx = 79; bx >= 0; bx--) {
    if (out[(row * 80 + bx) * 4 + 3] === 255) { right = bx; break; }
  }
  assert.ok(left >= 0 && right >= 0 && right > left);
  const lVal = out[(row * 80 + left) * 4];
  const rVal = out[(row * 80 + right) * 4];
  assert.ok(rVal > lVal, `+X limb (${rVal}) brighter than -X limb (${lVal})`);
});

test("chunked rows compose to the same buffer as a single pass", () => {
  const base = solid(8, 8, 180, 180, 180);
  const a = frontalView(48, 48, base, null);
  const b = frontalView(48, 48, base, null);
  renderMoonSurfaceRows(a.view, base, null, a.out, 0, 48);
  for (let r = 0; r < 48; r += 7) renderMoonSurfaceRows(b.view, base, null, b.out, r, r + 7);
  assert.deepEqual(Array.from(a.out), Array.from(b.out));
});

// ── B5: per-body base-texture longitude offset (planet map family) ──────────

/** base with the west edge (u≈0) RED and the centre (u≈0.5) BLUE, so the
 *  sampled meridian at the sub-camera point (lon 0) is decidable by colour. */
function meridianBase(): TexLike {
  const w = 4, h = 2;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { data[i * 4] = 120; data[i * 4 + 1] = 120; data[i * 4 + 2] = 120; data[i * 4 + 3] = 255; }
  const setCol = (col: number, r: number, g: number, b: number): void => {
    for (let y = 0; y < h; y++) { const i = (y * w + col) * 4; data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255; }
  };
  setCol(0, 255, 0, 0); // u≈0 (west edge) red
  setCol(2, 0, 0, 255); // u≈0.5 (centre) blue
  return { data, width: w, height: h };
}

test("texLonOffsetDeg=0 samples the u=0.5 meridian at the sub-camera point (Moon)", () => {
  const base = meridianBase();
  const { view, out } = frontalView(32, 32, base, null);
  view.texLonOffsetDeg = 0;
  renderMoonSurfaceRows(view, base, null, out, 0, 32);
  const c = (16 * 32 + 16) * 4;
  assert.ok(out[c + 2] > out[c], `centre blue (u=0.5): b=${out[c + 2]} r=${out[c]}`);
});

test("texLonOffsetDeg=180 shifts the sub-camera meridian by 180° (planet maps)", () => {
  const base = meridianBase();
  const { view, out } = frontalView(32, 32, base, null);
  view.texLonOffsetDeg = 180; // planet map family: lon 0 at the texture's west edge
  renderMoonSurfaceRows(view, base, null, out, 0, 32);
  const c = (16 * 32 + 16) * 4;
  assert.ok(out[c] > out[c + 2], `centre red (u wraps to 0): r=${out[c]} b=${out[c + 2]}`);
});

test("texLonOffsetDeg omitted ≡ offset 0 (Moon path byte-identical)", () => {
  const base = solid(8, 8, 200, 200, 200);
  const a = frontalView(40, 40, base, null);      // texLonOffsetDeg undefined
  const b = frontalView(40, 40, base, null);
  b.view.texLonOffsetDeg = 0;                      // explicit 0
  renderMoonSurfaceRows(a.view, base, null, a.out, 0, 40);
  renderMoonSurfaceRows(b.view, base, null, b.out, 0, 40);
  assert.deepEqual(Array.from(a.out), Array.from(b.out));
});

// ── B6 universal lighting on the close surface patch ─────────────────────────

test("B6 fullBright (realistic OFF): the night side of the patch is full-bright", () => {
  const base = solid(8, 8, 200, 200, 200);
  const { view, out } = frontalView(48, 48, base, null);
  view.sun = { x: -1, y: 0, z: 0 }; // sun BEHIND the body ⇒ near side is night
  view.fullBright = true;
  renderMoonSurfaceRows(view, base, null, out, 0, 48);
  const c = (24 * 48 + 24) * 4;
  assert.equal(out[c + 3], 255, "centre is opaque");
  assert.equal(out[c], 200, "night-lit centre is full-bright (no terminator)");
});

test("B6 shadowFactor (eclipse): the lit patch darkens by the umbra factor, matching the far sprite", () => {
  const base = solid(8, 8, 200, 200, 200);
  const lit = frontalView(48, 48, base, null); // sun toward the camera (+x): full lit
  const ecl = frontalView(48, 48, base, null);
  ecl.view.shadowFactor = 0.2;
  renderMoonSurfaceRows(lit.view, base, null, lit.out, 0, 48);
  renderMoonSurfaceRows(ecl.view, base, null, ecl.out, 0, 48);
  const c = (24 * 48 + 24) * 4;
  assert.ok(lit.out[c] > 150, "un-eclipsed patch is bright");
  assert.ok(Math.abs(ecl.out[c] - lit.out[c] * 0.2) < 3, "patch darkens by the eclipse scalar");
});

test("B6 shadowFactor omitted ≡ 1 (fully sunlit patch, byte-identical)", () => {
  const base = solid(8, 8, 180, 160, 140);
  const a = frontalView(40, 40, base, null);
  const b = frontalView(40, 40, base, null);
  b.view.shadowFactor = 1;
  renderMoonSurfaceRows(a.view, base, null, a.out, 0, 40);
  renderMoonSurfaceRows(b.view, base, null, b.out, 0, 40);
  assert.deepEqual(Array.from(a.out), Array.from(b.out));
});
