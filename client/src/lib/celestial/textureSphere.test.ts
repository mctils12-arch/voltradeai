// textureSphere tests — sampling conventions, the LUT/compose split, the
// tidal-lock ⇒ near-side-faces-Earth pin against the REAL ephemeris,
// retrograde spin direction, ring geometry off the IAU pole, and the
// galactic sky frame. Hermetic node:test via tsx (no DOM).
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  equirectUV,
  sampleEquirectRGB,
  buildSphereLUT,
  composeTexturedSprite,
  lambertWeight,
  ringBandsFromTextures,
  ringCircle3D,
  SATURN_RING_INNER_REL,
  SATURN_RING_OUTER_REL,
  galacticBasisEcl,
  dirToGalacticLonLat,
  buildSkyRays,
  renderSkyToBuffer,
  axialTiltDeg,
  spinObliquityDeg,
  type TexLike,
  type Vec3,
} from "./textureSphere.js";
import {
  axisEclOfDate,
  equatorNodeDirEclOfDate,
  primeMeridianDeg,
  primeMeridianDirEclOfDate,
  rotationRateDegPerDay,
} from "./rotation.js";
import { rotateAbout } from "./spaceFrame.js";
import { solarSystemState, AU_M } from "./solarSystem.js";

const close = (a: number, b: number, tol: number, msg: string): void => {
  assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);
};

function tex(w: number, h: number, fill: (x: number, y: number) => [number, number, number]): TexLike {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = fill(x, y);
      const i = (y * w + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return { data, width: w, height: h };
}

test("equirect conventions: prime meridian at center, east → +u, north pole at v=0", () => {
  assert.deepEqual(equirectUV(0, 0), { u: 0.5, v: 0.5 });
  close(equirectUV(90, 0).u, 0.75, 1e-12, "90°E");
  close(equirectUV(-90, 0).u, 0.25, 1e-12, "90°W");
  close(equirectUV(180, 0).u, 0, 1e-12, "antimeridian wraps to 0");
  close(equirectUV(540, 0).u, 0, 1e-12, "lon wraps mod 360");
  assert.equal(equirectUV(0, 90).v, 0, "north pole at top");
  assert.equal(equirectUV(0, -90).v, 1, "south pole at bottom");
  const t = tex(4, 2, (x, y) => [x * 10, y * 10, 0]);
  assert.deepEqual(sampleEquirectRGB(t, 0, 0), [20, 10, 0], "center texel (x=2,y=1)");
  assert.deepEqual(sampleEquirectRGB(t, -180, 89), [0, 0, 0], "west edge top");
});

test("sphere LUT: center pixel is the sub-camera point; node frame lon/lat are exact", () => {
  // axis up on screen, node toward the viewer ⇒ the camera looks at
  // (lonNode 0, lat 0)
  const lut = buildSphereLUT(8, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 });
  const R = lut.shadeR;
  const c = (R + 1) * lut.size + (R + 1);
  assert.equal(lut.mask[c], 1);
  close(lut.lonNode[c], 0, 1e-4, "center lonNode");
  close(lut.lat[c], 0, 1e-4, "center lat");
  // straight above center on screen → toward the north pole
  const top = (R + 1 - Math.round(R * 0.9)) * lut.size + (R + 1);
  assert.equal(lut.mask[top], 1);
  assert.ok(lut.lat[top] > 60, `top of disc is high latitude (${lut.lat[top].toFixed(1)}°)`);
  // right of center on screen → east of the sub-camera meridian
  const right = (R + 1) * lut.size + (R + 1 + Math.round(R * 0.5));
  assert.ok(lut.lonNode[right] > 5, `right of disc is east (${lut.lonNode[right].toFixed(1)}°)`);
  // corners outside the disc are masked out
  assert.equal(lut.mask[0], 0);
});

test("compose: front-lit center shows the exact texel; night side falls to the 5% floor; W shifts columns", () => {
  const lut = buildSphereLUT(8, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 });
  const t = tex(36, 18, (x) => [x * 7, 100, 200]); // longitude-striped
  const out = new Uint8ClampedArray(lut.size * lut.size * 4);
  composeTexturedSprite(lut, t, 0, { x: 0, y: 0, z: 1 }, out);
  const R = lut.shadeR;
  const c = ((R + 1) * lut.size + (R + 1)) * 4;
  // center texel at u=0.5 → x=18 → r=126, full-lit weight = 1
  assert.equal(out[c], 126, "center red channel = texel · 1.0");
  assert.equal(out[c + 1], 100);
  assert.equal(out[c + 2], 200);
  assert.equal(out[c + 3], 255);
  assert.equal(out[3], 0, "outside the disc is transparent");
  // light from behind: everything at the 5% floor
  const night = new Uint8ClampedArray(out.length);
  composeTexturedSprite(lut, t, 0, { x: 0, y: 0, z: -1 }, night);
  close(night[c], 126 * lambertWeight(-1), 1.5, "night side at the floor");
  assert.ok(night[c] < 10, "night side dark");
  // spinning by W = 10° samples a column 10° west of the sub-camera point
  const spun = new Uint8ClampedArray(out.length);
  composeTexturedSprite(lut, t, 10, { x: 0, y: 0, z: 1 }, spun);
  const expectU = ((-10 / 360 + 0.5) % 1 + 1) % 1;
  const expectR = Math.min(35, Math.floor(expectU * 36)) * 7;
  assert.equal(spun[c], expectR, "W offsets the texture column (the spin path)");
  // emissive: no lighting — full-bright texel
  const em = new Uint8ClampedArray(out.length);
  composeTexturedSprite(lut, t, 0, null, em);
  assert.equal(em[c], 126, "emissive full bright");
});

test("compose: texLonOffsetDeg shifts the source-map prime meridian (B5 planet maps)", () => {
  const lut = buildSphereLUT(8, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 });
  const t = tex(36, 18, (x) => [x * 7, 100, 200]); // longitude-striped: r ∝ column
  const R = lut.shadeR;
  const c = ((R + 1) * lut.size + (R + 1)) * 4;
  // offset 0 (Moon/Sun): centre (lon 0) → u=0.5 → x=18 → r=126 (unchanged)
  const a = new Uint8ClampedArray(lut.size * lut.size * 4);
  composeTexturedSprite(lut, t, 0, { x: 0, y: 0, z: 1 }, a, { texLonOffsetDeg: 0 });
  assert.equal(a[c], 126, "offset 0 keeps the u=0.5 centre meridian");
  // offset 180 (marsmap/mercurymap/venusmap — lon 0 at the texture's west edge):
  // u = 0.5 + 0.5 → wraps to 0 → x=0 → r=0
  const b = new Uint8ClampedArray(lut.size * lut.size * 4);
  composeTexturedSprite(lut, t, 0, { x: 0, y: 0, z: 1 }, b, { texLonOffsetDeg: 180 });
  assert.equal(b[c], 0, "offset 180 samples the west-edge column (x=0)");
  // omitting the field ≡ offset 0 (default byte-identical — the Moon regression)
  const d = new Uint8ClampedArray(lut.size * lut.size * 4);
  composeTexturedSprite(lut, t, 0, { x: 0, y: 0, z: 1 }, d);
  assert.deepEqual(Array.from(d), Array.from(a));
});

test("bump: emboss shading perturbs the lit weight without breaking bounds", () => {
  const lut = buildSphereLUT(8, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 }, true);
  const t = tex(16, 8, () => [128, 128, 128]);
  const bump = tex(16, 8, (x) => [x * 15, 0, 0]); // west→east height ramp
  const flat = new Uint8ClampedArray(lut.size * lut.size * 4);
  const bumped = new Uint8ClampedArray(lut.size * lut.size * 4);
  const light = { x: 0.7, y: 0, z: 0.714 };
  composeTexturedSprite(lut, t, 0, light, flat);
  composeTexturedSprite(lut, t, 0, light, bumped, { bump, bumpStrength: 1.2 });
  let diff = 0;
  for (let i = 0; i < flat.length; i += 4) {
    if (flat[i + 3]) diff += Math.abs(flat[i] - bumped[i]);
  }
  assert.ok(diff > 0, "bump changes shading");
  for (let i = 0; i < bumped.length; i += 4) {
    assert.ok(bumped[i] <= 255 && bumped[i] >= 0, "bounded");
  }
});

test("node frame ⊥ pole, and rotate(node, axis, W) reproduces the IAU prime-meridian direction", () => {
  const t = Date.UTC(2026, 6, 18, 12);
  for (const id of ["earth", "moon", "mars", "jupiter", "venus", "uranus"] as const) {
    const axis = axisEclOfDate(id, t);
    const node = equatorNodeDirEclOfDate(id, t);
    close(axis.x * node.x + axis.y * node.y + axis.z * node.z, 0, 1e-9, `${id} node ⊥ pole`);
    const w = primeMeridianDeg(id, t);
    const pm = primeMeridianDirEclOfDate(id, t);
    const rot = rotateAbout(node, axis, w);
    close(rot.x, pm.x, 1e-9, `${id} pm.x`);
    close(rot.y, pm.y, 1e-9, `${id} pm.y`);
    close(rot.z, pm.z, 1e-9, `${id} pm.z`);
  }
});

test("TIDAL LOCK ⇒ the LRO near side (texture center) faces Earth within real libration", () => {
  // Body-fixed longitude of the Moon→Earth direction must stay within the
  // Moon's optical libration (~8°) — this is exactly what makes the
  // texture's centered near side render toward Earth.
  for (let m = 0; m < 12; m++) {
    const t = Date.UTC(2026, m, 7, 3);
    const state = new Map(solarSystemState(t).map((b) => [b.id, b.helioAu]));
    const e = state.get("earth")!;
    const mo = state.get("moon")!;
    const d: Vec3 = { x: (e.x - mo.x) * AU_M, y: (e.y - mo.y) * AU_M, z: (e.z - mo.z) * AU_M };
    const axis = axisEclOfDate("moon", t);
    const pm = primeMeridianDirEclOfDate("moon", t);
    const yb = {
      x: axis.y * pm.z - axis.z * pm.y,
      y: axis.z * pm.x - axis.x * pm.z,
      z: axis.x * pm.y - axis.y * pm.x,
    };
    const lon = Math.atan2(
      d.x * yb.x + d.y * yb.y + d.z * yb.z,
      d.x * pm.x + d.y * pm.y + d.z * pm.z,
    ) * 180 / Math.PI;
    assert.ok(Math.abs(lon) < 9, `near side faces Earth (month ${m}: lon ${lon.toFixed(2)}°)`);
  }
});

test("retrograde spin: Venus/Uranus W runs backwards vs Jupiter (texture drift reverses for free)", () => {
  const t = Date.UTC(2026, 6, 18);
  const dt = 3_600_000;
  const drift = (id: "venus" | "jupiter" | "uranus"): number => {
    let d = primeMeridianDeg(id, t + dt) - primeMeridianDeg(id, t);
    if (d > 180) d -= 360;
    if (d < -180) d += 360;
    return d;
  };
  assert.ok(rotationRateDegPerDay("venus") < 0 && drift("venus") < 0, "Venus retrograde");
  assert.ok(rotationRateDegPerDay("uranus") < 0 && drift("uranus") < 0, "Uranus retrograde");
  assert.ok(rotationRateDegPerDay("jupiter") > 0 && drift("jupiter") > 0, "Jupiter prograde");
});

test("ring bands: real C→A extent, strip-sampled colors/alphas, monotone radii", () => {
  close(SATURN_RING_INNER_REL, 1.2387, 5e-4, "C-ring inner edge");
  close(SATURN_RING_OUTER_REL, 2.2695, 5e-4, "A-ring outer edge");
  const color = tex(64, 4, (x) => [200 - x * 2, 180, 150]);
  const alpha = tex(64, 4, (x) => [x < 32 ? 255 : 64, 0, 0]);
  const bands = ringBandsFromTextures(color, alpha, 16);
  assert.equal(bands.length, 16);
  assert.ok(Math.abs(bands[0].r0 - SATURN_RING_INNER_REL) < 1e-12, "starts at the inner edge");
  assert.ok(Math.abs(bands[15].r1 - SATURN_RING_OUTER_REL) < 1e-12, "ends at the outer edge");
  for (let i = 1; i < bands.length; i++) {
    assert.ok(bands[i].r0 >= bands[i - 1].r1 - 1e-12, "bands tile outward");
  }
  assert.ok(bands[0].r > bands[15].r, "color sampled along the strip");
  close(bands[0].alpha, 0.85, 0.02, "opaque half × the 0.85 reference opacity");
  close(bands[15].alpha, (64 / 255) * 0.85, 0.02, "translucent half");
});

test("ring circle: perpendicular to Saturn's IAU pole, correct radius; Saturn tilt ≈ 26.7°", () => {
  const t = Date.UTC(2026, 6, 18);
  const axis = axisEclOfDate("saturn", t);
  const center = { x: 1e12, y: -4e11, z: 2e10 };
  const pts = ringCircle3D(center, axis, 8e7, 48);
  for (let i = 0; i < 48; i++) {
    const dx = pts[i * 3] - center.x;
    const dy = pts[i * 3 + 1] - center.y;
    const dz = pts[i * 3 + 2] - center.z;
    close(Math.hypot(dx, dy, dz), 8e7, 1, "radius");
    close((dx * axis.x + dy * axis.y + dz * axis.z) / 8e7, 0, 1e-9, "in the ring plane (⊥ pole)");
  }
  const tilt = axialTiltDeg(axis);
  assert.ok(tilt > 25 && tilt < 29, `Saturn axial tilt ${tilt.toFixed(1)}° (≈26.7° to the ecliptic family)`);
});

test("card-tilt derivation (classic obliquity = spin-vector tilt): Earth 23.4°, Venus ~178°, Uranus ~98°", () => {
  const t = Date.UTC(2026, 6, 18);
  const tilt = (id: "earth" | "venus" | "uranus"): number =>
    spinObliquityDeg(axisEclOfDate(id, t), rotationRateDegPerDay(id));
  close(tilt("earth"), 23.44, 0.1, "Earth");
  const venus = tilt("venus");
  assert.ok(venus > 176 && venus < 180, `Venus near-flipped spin (${venus.toFixed(1)}°)`);
  const uranus = tilt("uranus");
  assert.ok(uranus > 96 && uranus < 100, `Uranus sideways (${uranus.toFixed(1)}°)`);
});

test("galactic sky frame: orthonormal, NGP at galactic lat 90, center at (0,0), real ecliptic placement", () => {
  const t = Date.UTC(2026, 6, 18);
  const g = galacticBasisEcl(t);
  close(g.X.x ** 2 + g.X.y ** 2 + g.X.z ** 2, 1, 1e-9, "X unit");
  close(g.Z.x ** 2 + g.Z.y ** 2 + g.Z.z ** 2, 1, 1e-9, "Z unit");
  close(g.X.x * g.Z.x + g.X.y * g.Z.y + g.X.z * g.Z.z, 0, 1e-9, "X ⊥ Z");
  const pole = dirToGalacticLonLat(g.Z, g);
  close(pole.latDeg, 90, 1e-6, "NGP at b=90");
  const cen = dirToGalacticLonLat(g.X, g);
  close(cen.latDeg, 0, 1e-6, "center at b=0");
  close(cen.lonDeg, 0, 1e-6, "center at l=0");
  // the NGP sits at ecliptic latitude ≈ +29.8° — pins the eq→ecl leg
  close(Math.asin(g.Z.z) * 180 / Math.PI, 29.81, 0.35, "NGP ecliptic latitude");
});

test("sky rays + render: center ray is the forward axis; the band paints where the galactic plane crosses", () => {
  const rays = buildSkyRays(9, 7, 40);
  const ci = ((3 * 9) + 4) * 3;
  close(rays[ci], 0, 1e-6, "center ray x");
  close(rays[ci + 1], 0, 1e-6, "center ray y");
  close(rays[ci + 2], 1, 1e-6, "center ray z (forward)");
  for (let i = 0; i < rays.length; i += 3) {
    close(Math.hypot(rays[i], rays[i + 1], rays[i + 2]), 1, 1e-5, "unit rays");
  }
  // panorama: bright equator band, dark poles; aim the camera at the
  // galactic center → center pixel bright, aim at the pole → dark
  const pano = tex(64, 32, (_x, y) => (y >= 14 && y < 18 ? [220, 220, 220] : [5, 5, 5]));
  const g = galacticBasisEcl(Date.UTC(2026, 6, 18));
  const w = 9;
  const h = 7;
  const out = new Uint8ClampedArray(w * h * 4);
  renderSkyToBuffer(rays, w, h, g.X, g.Y, g.Z, g, pano, out);
  const c = ((3 * w) + 4) * 4;
  assert.equal(out[c], 220, "looking at the galactic center: band bright");
  const out2 = new Uint8ClampedArray(w * h * 4);
  renderSkyToBuffer(rays, w, h, g.Z, g.X, g.Y, g, pano, out2);
  assert.equal(out2[c], 5, "looking at the pole: off-band dark");
});

test("sky render: row-band chunking reproduces the whole-buffer result", () => {
  const rays = buildSkyRays(12, 10, 40);
  const g = galacticBasisEcl(Date.UTC(2026, 6, 19));
  const pano = tex(64, 32, (x, y) => [x * 3, y * 6, 255 - x]);
  const w = 12;
  const h = 10;
  const whole = new Uint8ClampedArray(w * h * 4);
  renderSkyToBuffer(rays, w, h, g.X, g.Y, g.Z, g, pano, whole, true, 1.1);
  const banded = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y += 3) {
    renderSkyToBuffer(rays, w, h, g.X, g.Y, g.Z, g, pano, banded, true, 1.1, y, Math.min(h, y + 3));
  }
  for (let i = 0; i < whole.length; i++) assert.equal(banded[i], whole[i], `pixel byte ${i} matches`);
});

test("sky render: bilinear interpolates between texels (nearest snaps)", () => {
  const rays = buildSkyRays(9, 7, 40);
  const g = galacticBasisEcl(Date.UTC(2026, 6, 19));
  // a horizontal gradient panorama — bilinear must produce in-between values
  const pano = tex(64, 32, (x) => [Math.round((x / 63) * 255), 128, 0]);
  const w = 9;
  const h = 7;
  const near = new Uint8ClampedArray(w * h * 4);
  const bil = new Uint8ClampedArray(w * h * 4);
  renderSkyToBuffer(rays, w, h, g.X, g.Y, g.Z, g, pano, near, false, 1);
  renderSkyToBuffer(rays, w, h, g.X, g.Y, g.Z, g, pano, bil, true, 1);
  let differ = 0;
  for (let i = 0; i < w * h; i++) if (near[i * 4] !== bil[i * 4]) differ++;
  assert.ok(differ > 0, "bilinear differs from nearest on a gradient");
  // exposure lift brightens the band
  const dim = new Uint8ClampedArray(w * h * 4);
  const lit = new Uint8ClampedArray(w * h * 4);
  renderSkyToBuffer(rays, w, h, g.X, g.Y, g.Z, g, pano, dim, false, 1);
  renderSkyToBuffer(rays, w, h, g.X, g.Y, g.Z, g, pano, lit, false, 1.5);
  let brighter = 0;
  for (let i = 0; i < w * h; i++) if (lit[i * 4] >= dim[i * 4]) brighter++;
  assert.equal(brighter, w * h, "exposure ≥1 never darkens");
});
