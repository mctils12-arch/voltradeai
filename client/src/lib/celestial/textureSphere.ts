// textureSphere — PURE MATH for texture-mapped body rendering in the
// hand-rolled Canvas2D space frame (SPACE VIEW VISUAL UPGRADE, human
// directive 2026-07-18: research/directives/space_view_handoff_2026-07-18.md
// + space_view_reference_2026-07-18.html). NO three.js, NO dependency —
// the reference used three.js for prototyping convenience; this module
// reproduces the RESULT inside the existing renderer.
//
// WHAT LIVES HERE (all pure, typed-array in/out, `npx tsx --test`-able):
//  · equirectangular sampling conventions (prime meridian at texture
//    center — the planetary-map standard; the Moon's LRO mosaics center
//    the near side, so the tidal lock makes the near side face Earth,
//    which the tests pin against the real ephemeris);
//  · the SPHERE LUT / COMPOSE split. Orientation is factored as
//    (node frame, W) per rotation.ts equatorNodeDirEclOfDate: a pixel's
//    body longitude = its node-frame longitude − W. The LUT (per-pixel
//    normals + node-frame lon/lat — the trig-heavy part) depends only on
//    the body's axis orientation IN CAMERA SPACE, which changes only when
//    the camera orbits; W enters at COMPOSE time as a pure texture-column
//    offset. A body spinning under time warp therefore re-composes
//    (~texel fetch + Lambert multiply per pixel, no trig) without ever
//    rebuilding the LUT — this is what keeps every main-thread task
//    under the 16 ms budget. Retrograde Venus/Uranus come free from the
//    IAU negative Ẇ (W decreases ⇒ the sampling offset reverses).
//  · Lambert lighting IDENTICAL to the untextured sprite shader's curve
//    (soft ±0.03/0.07 terminator, 5% floor) so textured and color-only
//    bodies read as one family; optional emboss bump (Moon relief from
//    moonbump1k gradients — presentation shading, not data);
//  · Saturn ring band extraction from the threex ring strips + 3D ring
//    circle geometry perpendicular to Saturn's IAU pole (rotation.ts).
//    Band radii span the REAL main-ring extent (C inner → A outer,
//    NSSDC 74,658–136,775 km) — a stated deviation from the reference's
//    decorative 1.35–2.4 span;
//  · the Milky Way sky mapping: camera rays → REAL galactic frame
//    (J2000 NGP/galactic-center constants) → equirect panorama. The
//    reference oriented its sky sphere arbitrarily; we align the band to
//    the true galactic plane. In-band longitude handedness of the
//    panorama is presentation (stated) — the panorama itself is
//    © Solar System Scope CC-BY 4.0 (spaceAssets.MILKY_WAY_CREDIT).

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface TexLike {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

const v3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 =>
  v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const norm3 = (a: Vec3): Vec3 => {
  const l = Math.hypot(a.x, a.y, a.z) || 1;
  return v3(a.x / l, a.y / l, a.z / l);
};

// ── equirect sampling ───────────────────────────────────────────────────────

/** Planetary-map convention: lon 0 (the prime meridian) at texture center
 *  (u = 0.5), longitude increasing eastward with u; v = 0 at the north
 *  pole. (The registration of each source map's prime meridian is the map
 *  author's — approximate for the gas giants, exact for the LRO Moon.) */
export function equirectUV(lonDeg: number, latDeg: number): { u: number; v: number } {
  const u = (((lonDeg / 360 + 0.5) % 1) + 1) % 1;
  const v = Math.min(1, Math.max(0, 0.5 - latDeg / 180));
  return { u, v };
}

/** Nearest-texel fetch (RGB). */
export function sampleEquirectRGB(tex: TexLike, lonDeg: number, latDeg: number): [number, number, number] {
  const { u, v } = equirectUV(lonDeg, latDeg);
  const x = Math.min(tex.width - 1, Math.floor(u * tex.width));
  const y = Math.min(tex.height - 1, Math.floor(v * tex.height));
  const i = (y * tex.width + x) * 4;
  return [tex.data[i], tex.data[i + 1], tex.data[i + 2]];
}

// ── the sphere LUT / compose split ──────────────────────────────────────────

export interface SphereLUT {
  /** shaded disc radius, px (sprite size = 2·shadeR + 2, the sprite-cache
   *  convention of the frame's untextured shader). */
  shadeR: number;
  size: number;
  /** camera-space unit normal per pixel (x right, y up, z toward viewer). */
  nx: Float32Array;
  ny: Float32Array;
  nz: Float32Array;
  /** body longitude in the W = 0 node frame, degrees. */
  lonNode: Float32Array;
  /** body latitude, degrees. */
  lat: Float32Array;
  /** east tangent per pixel (bump shading only; null unless requested). */
  ex: Float32Array | null;
  ey: Float32Array | null;
  ez: Float32Array | null;
  /** 1 = inside the disc. */
  mask: Uint8Array;
}

/** Allocate an empty LUT (fill it with fillSphereLUTRows — chunkable). */
export function createEmptySphereLUT(shadeR: number, withTangents = false): SphereLUT {
  const R = Math.max(2, Math.round(shadeR));
  const size = R * 2 + 2;
  const n = size * size;
  return {
    shadeR: R,
    size,
    nx: new Float32Array(n),
    ny: new Float32Array(n),
    nz: new Float32Array(n),
    lonNode: new Float32Array(n),
    lat: new Float32Array(n),
    ex: withTangents ? new Float32Array(n) : null,
    ey: withTangents ? new Float32Array(n) : null,
    ez: withTangents ? new Float32Array(n) : null,
    mask: new Uint8Array(n),
  };
}

/**
 * Fill LUT rows [rowStart, rowEnd) — the trig-heavy step, chunkable across
 * macrotasks so no single main-thread task exceeds the frame budget.
 */
export function fillSphereLUTRows(
  lut: SphereLUT,
  axisCam: Vec3,
  nodeCam: Vec3,
  rowStart: number,
  rowEnd: number,
): void {
  const { shadeR: R, size, nx, ny, nz, lonNode, lat, mask, ex, ey, ez } = lut;
  const withTangents = !!ex;
  const Z = norm3(axisCam);
  const X = norm3(nodeCam);
  const Y = cross(Z, X); // node frame: lonNode = atan2(n·Y, n·X)
  const y0 = Math.max(0, rowStart);
  const y1 = Math.min(size, rowEnd);
  for (let py = y0; py < y1; py++) {
    for (let px = 0; px < size; px++) {
      const x = (px - R - 1) / R;
      const y = -(py - R - 1) / R; // screen y down → up
      const z2 = 1 - x * x - y * y;
      if (z2 < 0) continue;
      const z = Math.sqrt(z2);
      const i = py * size + px;
      mask[i] = 1;
      nx[i] = x;
      ny[i] = y;
      nz[i] = z;
      const bx = x * X.x + y * X.y + z * X.z;
      const by = x * Y.x + y * Y.y + z * Y.z;
      const bz = x * Z.x + y * Z.y + z * Z.z;
      lonNode[i] = Math.atan2(by, bx) * RAD;
      lat[i] = Math.asin(Math.max(-1, Math.min(1, bz))) * RAD;
      if (withTangents && ex && ey && ez) {
        // east = d(surface)/d(lon) direction = axis × n (normalized)
        let tx = Z.y * z - Z.z * y;
        let ty = Z.z * x - Z.x * z;
        let tz = Z.x * y - Z.y * x;
        const tl = Math.hypot(tx, ty, tz) || 1;
        tx /= tl; ty /= tl; tz /= tl;
        ex[i] = tx; ey[i] = ty; ez[i] = tz;
      }
    }
  }
}

/**
 * Build the per-pixel geometry LUT for a sphere of shaded radius `shadeR`
 * whose spin axis and W=0 node direction are `axisCam`/`nodeCam` (unit,
 * CAMERA space — orthogonal by construction upstream: the IAU node lies in
 * the body's equator plane). The trig-heavy step: one atan2 + asin per
 * disc pixel, rebuilt only when the axis moves in camera space (camera
 * orbits), never when the body spins. One-shot convenience over the
 * chunkable create/fill pair.
 */
export function buildSphereLUT(
  shadeR: number,
  axisCam: Vec3,
  nodeCam: Vec3,
  withTangents = false,
): SphereLUT {
  const lut = createEmptySphereLUT(shadeR, withTangents);
  fillSphereLUTRows(lut, axisCam, nodeCam, 0, lut.size);
  return lut;
}

/** The untextured sprite shader's exact lighting curve (soft terminator
 *  ±0.03/0.07, 5% floor) — shared so textured/untextured bodies match. */
export function lambertWeight(lit: number): number {
  const day = Math.max(0, Math.min(1, (lit + 0.03) / 0.07));
  const shade = 0.05 + 0.95 * Math.max(lit, 0);
  return 0.05 + 0.95 * day * shade;
}

/**
 * Compose a textured sprite into `out` (RGBA, size²·4): texel fetch at
 * (lonNode − W, lat) + Lambert light. lightCam null ⇒ emissive (full-bright
 * texel — the Sun; its glow is drawn by the frame). No trig — the warp
 * spin path. Optional emboss bump from a height map's u/v gradients
 * (presentation relief; strength ~1.2).
 */
export function composeTexturedSprite(
  lut: SphereLUT,
  tex: TexLike,
  wDeg: number,
  lightCam: Vec3 | null,
  out: Uint8ClampedArray,
  opts?: { bump?: TexLike | null; bumpStrength?: number; rowStart?: number; rowEnd?: number; texLonOffsetDeg?: number },
): void {
  const { size, nx, ny, nz, lonNode, lat, mask, ex, ey, ez } = lut;
  const tw = tex.width;
  const th = tex.height;
  const td = tex.data;
  const bump = opts?.bump ?? null;
  const bk = opts?.bumpStrength ?? 1.2;
  const s = lightCam ? norm3(lightCam) : null;
  // source-map prime-meridian offset (0 for the Moon/Sun ⇒ byte-identical;
  // 180 for the planet map family whose lon 0 sits at the texture's left edge)
  const texOff = (opts?.texLonOffsetDeg ?? 0) / 360;
  const i0 = Math.max(0, opts?.rowStart ?? 0) * size;
  const n = Math.min(size, opts?.rowEnd ?? size) * size;
  for (let i = i0; i < n; i++) {
    const o = i * 4;
    if (!mask[i]) {
      out[o + 3] = 0;
      continue;
    }
    const lon = lonNode[i] - wDeg;
    let u = lon / 360 + 0.5 + texOff;
    u -= Math.floor(u); // wrap to [0,1)
    const v = Math.min(1, Math.max(0, 0.5 - lat[i] / 180));
    const tx = Math.min(tw - 1, (u * tw) | 0);
    const ty = Math.min(th - 1, (v * th) | 0);
    const ti = (ty * tw + tx) * 4;
    let w = 1;
    if (s) {
      let lit = nx[i] * s.x + ny[i] * s.y + nz[i] * s.z;
      if (bump && ex && ey && ez) {
        const bw = bump.width;
        const bh = bump.height;
        const bx = Math.min(bw - 1, (u * bw) | 0);
        const by = Math.min(bh - 1, (v * bh) | 0);
        const xm = (by * bw + (bx > 0 ? bx - 1 : bw - 1)) * 4;
        const xp = (by * bw + (bx < bw - 1 ? bx + 1 : 0)) * 4;
        const ym = ((by > 0 ? by - 1 : 0) * bw + bx) * 4;
        const yp = ((by < bh - 1 ? by + 1 : bh - 1) * bw + bx) * 4;
        const gu = (bump.data[xp] - bump.data[xm]) / 255;
        const gv = (bump.data[yp] - bump.data[ym]) / 255;
        // north tangent = n × east
        const nxx = ny[i] * ez[i] - nz[i] * ey[i];
        const nyy = nz[i] * ex[i] - nx[i] * ez[i];
        const nzz = nx[i] * ey[i] - ny[i] * ex[i];
        const se = s.x * ex[i] + s.y * ey[i] + s.z * ez[i];
        const sn = s.x * nxx + s.y * nyy + s.z * nzz;
        lit -= bk * (gu * se - gv * sn); // v grows southward ⇒ gv flips
        if (lit > 1) lit = 1;
        else if (lit < -1) lit = -1;
      }
      w = lambertWeight(lit);
    }
    out[o] = td[ti] * w;
    out[o + 1] = td[ti + 1] * w;
    out[o + 2] = td[ti + 2] * w;
    out[o + 3] = 255;
  }
}

// ── Saturn rings ────────────────────────────────────────────────────────────

/** REAL main-ring extent in Saturn equatorial radii (NSSDC: C-ring inner
 *  edge 74,658 km, A-ring outer edge 136,775 km, R♄ 60,268 km). Stated
 *  deviation from the reference's decorative 1.35–2.4 span. */
export const SATURN_RING_INNER_REL = 74_658 / 60_268;
export const SATURN_RING_OUTER_REL = 136_775 / 60_268;

export interface RingBand {
  /** band edges, planet radii. */
  r0: number;
  r1: number;
  r: number;
  g: number;
  b: number;
  /** 0..1 (pattern strip × the reference's 0.85 ring opacity). */
  alpha: number;
}

/** Average RGB across the strip's short axis at long-axis fraction t. */
function stripSample(tex: TexLike, t: number): [number, number, number] {
  const alongX = tex.width >= tex.height;
  const len = alongX ? tex.width : tex.height;
  const cw = alongX ? tex.height : tex.width;
  const p = Math.min(len - 1, Math.max(0, Math.floor(t * len)));
  let r = 0;
  let g = 0;
  let b = 0;
  for (let c = 0; c < cw; c++) {
    const i = alongX ? (c * tex.width + p) * 4 : (p * tex.width + c) * 4;
    r += tex.data[i];
    g += tex.data[i + 1];
    b += tex.data[i + 2];
  }
  return [r / cw, g / cw, b / cw];
}

/**
 * Slice the ring strips into `nBands` radial bands between innerRel and
 * outerRel planet radii. Color from the color strip; alpha from the
 * pattern strip's red channel × 0.85 (the reference's ring opacity).
 */
export function ringBandsFromTextures(
  colorTex: TexLike,
  alphaTex: TexLike,
  nBands: number,
  innerRel: number = SATURN_RING_INNER_REL,
  outerRel: number = SATURN_RING_OUTER_REL,
): RingBand[] {
  const bands: RingBand[] = [];
  const n = Math.max(4, nBands);
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n;
    const [r, g, b] = stripSample(colorTex, t);
    const [a] = stripSample(alphaTex, t);
    bands.push({
      r0: innerRel + ((outerRel - innerRel) * i) / n,
      r1: innerRel + ((outerRel - innerRel) * (i + 1)) / n,
      r,
      g,
      b,
      alpha: (a / 255) * 0.85,
    });
  }
  return bands;
}

/**
 * Sampled 3D circle of radius `radius` around `center`, in the plane
 * perpendicular to `axis` (Saturn's IAU pole → the ring plane), xyz-
 * interleaved. Deterministic basis; nSeg segments (closed by the caller).
 */
export function ringCircle3D(center: Vec3, axis: Vec3, radius: number, nSeg: number): Float64Array {
  const Z = norm3(axis);
  const helper = Math.abs(Z.z) < 0.9 ? v3(0, 0, 1) : v3(1, 0, 0);
  const A = norm3(cross(Z, helper));
  const B = cross(Z, A);
  const out = new Float64Array(nSeg * 3);
  for (let i = 0; i < nSeg; i++) {
    const t = (i / nSeg) * Math.PI * 2;
    const c = Math.cos(t) * radius;
    const sn = Math.sin(t) * radius;
    out[i * 3] = center.x + A.x * c + B.x * sn;
    out[i * 3 + 1] = center.y + A.y * c + B.y * sn;
    out[i * 3 + 2] = center.z + A.z * c + B.z * sn;
  }
  return out;
}

// ── the Milky Way sky ───────────────────────────────────────────────────────

/** J2000 galactic frame anchors (equatorial): the galactic center and the
 *  north galactic pole. IAU-standard values. */
export const GALACTIC_CENTER_EQ = { raDeg: 266.405, decDeg: -28.936 };
export const GALACTIC_NORTH_POLE_EQ = { raDeg: 192.85948, decDeg: 27.12825 };

function obliquityRad(dateMs: number): number {
  const d = (dateMs - Date.UTC(1999, 11, 31)) / 86_400_000;
  return (23.4393 - 3.563e-7 * d) * DEG;
}

function eqToEcl(v: Vec3, dateMs: number): Vec3 {
  const e = obliquityRad(dateMs);
  return v3(
    v.x,
    Math.cos(e) * v.y + Math.sin(e) * v.z,
    -Math.sin(e) * v.y + Math.cos(e) * v.z,
  );
}

function radecUnit(raDeg: number, decDeg: number): Vec3 {
  const ra = raDeg * DEG;
  const dec = decDeg * DEG;
  return v3(Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec));
}

export interface GalacticBasis {
  /** toward the galactic center (l=0, b=0), ecliptic of date. */
  X: Vec3;
  Y: Vec3;
  /** the north galactic pole, ecliptic of date. */
  Z: Vec3;
}

/** The real galactic frame in ecliptic-of-date coordinates (same J2000≈
 *  frame-of-date conflation the rest of the codebase states). */
export function galacticBasisEcl(dateMs: number): GalacticBasis {
  const Z = norm3(eqToEcl(radecUnit(GALACTIC_NORTH_POLE_EQ.raDeg, GALACTIC_NORTH_POLE_EQ.decDeg), dateMs));
  let X = eqToEcl(radecUnit(GALACTIC_CENTER_EQ.raDeg, GALACTIC_CENTER_EQ.decDeg), dateMs);
  // orthogonalize (the catalog anchors are ⊥ to ~0.1°)
  const d = dot(X, Z);
  X = norm3(v3(X.x - Z.x * d, X.y - Z.y * d, X.z - Z.z * d));
  const Y = cross(Z, X);
  return { X, Y, Z };
}

export function dirToGalacticLonLat(d: Vec3, g: GalacticBasis): { lonDeg: number; latDeg: number } {
  const u = norm3(d);
  const x = dot(u, g.X);
  const y = dot(u, g.Y);
  const z = dot(u, g.Z);
  return {
    lonDeg: Math.atan2(y, x) * RAD,
    latDeg: Math.asin(Math.max(-1, Math.min(1, z))) * RAD,
  };
}

/**
 * Camera-space unit rays for a w×h sky render target (x right, y up,
 * z forward), one per pixel, xyz-interleaved. Fixed per (w, h, fov) —
 * built once and reused for every sky repaint.
 */
export function buildSkyRays(w: number, h: number, fovDeg: number): Float32Array {
  const out = new Float32Array(w * h * 3);
  const k = h / 2 / Math.tan((fovDeg / 2) * DEG);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const x = (px - cx) / k;
      const y = -(py - cy) / k;
      const l = Math.hypot(x, y, 1);
      const i = (py * w + px) * 3;
      out[i] = x / l;
      out[i + 1] = y / l;
      out[i + 2] = 1 / l;
    }
  }
  return out;
}

/**
 * Paint the panorama for the current camera basis into `out` (RGBA w×h).
 * world dir = r·ray.x + u·ray.y + f·ray.z → galactic lon/lat → equirect.
 * u = 0.5 − lon/360 (galactic center at panorama center; in-band
 * handedness is presentation — see header). Per pixel: 9 mults + atan2 +
 * asin + texel fetch; callers size the target so one repaint stays a few
 * ms and cache it against the camera basis.
 *
 * `bilinear` (default false = nearest, the exact-equality test's path)
 * interpolates the four surrounding texels — the galaxy detail upgrade
 * (2026-07-19) renders the panorama at a higher working resolution and
 * with bilinear sampling so the band no longer reads as soft wallpaper
 * when it fills the screen far out. An optional `exposure` (default 1)
 * lifts contrast/brightness of the band multiplicatively.
 */
export function renderSkyToBuffer(
  rays: Float32Array,
  w: number,
  h: number,
  f: Vec3,
  r: Vec3,
  u: Vec3,
  g: GalacticBasis,
  tex: TexLike,
  out: Uint8ClampedArray,
  bilinear = false,
  exposure = 1,
  rowStart = 0,
  rowEnd = -1,
): void {
  const tw = tex.width;
  const th = tex.height;
  const td = tex.data;
  // render only rows [rowStart, rowEnd) — the frame chunks the panorama across
  // macrotasks so no single sky repaint exceeds the 16 ms budget at high
  // working resolution (default renders the whole buffer).
  const y1 = rowEnd < 0 ? h : Math.min(h, rowEnd);
  const iStart = Math.max(0, rowStart) * w;
  const n = y1 * w;
  const ex = exposure;
  for (let i = iStart; i < n; i++) {
    const rx = rays[i * 3];
    const ry = rays[i * 3 + 1];
    const rz = rays[i * 3 + 2];
    const dx = r.x * rx + u.x * ry + f.x * rz;
    const dy = r.y * rx + u.y * ry + f.y * rz;
    const dz = r.z * rx + u.z * ry + f.z * rz;
    const gx = dx * g.X.x + dy * g.X.y + dz * g.X.z;
    const gy = dx * g.Y.x + dy * g.Y.y + dz * g.Y.z;
    const gz = dx * g.Z.x + dy * g.Z.y + dz * g.Z.z;
    const lon = Math.atan2(gy, gx) * RAD;
    const lat = Math.asin(gz > 1 ? 1 : gz < -1 ? -1 : gz) * RAD;
    let tu = 0.5 - lon / 360;
    tu -= Math.floor(tu);
    const tv = Math.min(1, Math.max(0, 0.5 - lat / 180));
    const o = i * 4;
    let cr: number;
    let cg: number;
    let cb: number;
    if (bilinear) {
      // bilinear over four texels, u wrapping horizontally, v clamped
      const fx = tu * tw - 0.5;
      const fy = tv * th - 0.5;
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const wx = fx - x0;
      const wy = fy - y0;
      const xa = ((x0 % tw) + tw) % tw;
      const xb = (xa + 1) % tw;
      const ya = y0 < 0 ? 0 : y0 >= th ? th - 1 : y0;
      const yb = y0 + 1 < 0 ? 0 : y0 + 1 >= th ? th - 1 : y0 + 1;
      const iAA = (ya * tw + xa) * 4;
      const iBA = (ya * tw + xb) * 4;
      const iAB = (yb * tw + xa) * 4;
      const iBB = (yb * tw + xb) * 4;
      const w00 = (1 - wx) * (1 - wy);
      const w10 = wx * (1 - wy);
      const w01 = (1 - wx) * wy;
      const w11 = wx * wy;
      cr = td[iAA] * w00 + td[iBA] * w10 + td[iAB] * w01 + td[iBB] * w11;
      cg = td[iAA + 1] * w00 + td[iBA + 1] * w10 + td[iAB + 1] * w01 + td[iBB + 1] * w11;
      cb = td[iAA + 2] * w00 + td[iBA + 2] * w10 + td[iAB + 2] * w01 + td[iBB + 2] * w11;
    } else {
      const tx = Math.min(tw - 1, (tu * tw) | 0);
      const ty = Math.min(th - 1, (tv * th) | 0);
      const ti = (ty * tw + tx) * 4;
      cr = td[ti];
      cg = td[ti + 1];
      cb = td[ti + 2];
    }
    if (ex !== 1) {
      cr *= ex;
      cg *= ex;
      cb *= ex;
    }
    out[o] = cr;
    out[o + 1] = cg;
    out[o + 2] = cb;
    out[o + 3] = 255;
  }
}

// ── misc derivations ────────────────────────────────────────────────────────

/** Tilt of the IAU-defined north pole to the ecliptic, degrees. */
export function axialTiltDeg(axisEcl: Vec3): number {
  const a = norm3(axisEcl);
  return Math.acos(Math.max(-1, Math.min(1, a.z))) * RAD;
}

/**
 * The CLASSIC obliquity (what "axial tilt" means on a fact sheet): the
 * tilt of the RIGHT-HAND-RULE spin vector to the ecliptic. The IAU model
 * defines "north" as the pole north of the invariable plane and encodes
 * retrograde spin as negative Ẇ — so the spin vector is sign(Ẇ)·pole.
 * Earth ≈ 23.4°, Venus ≈ 178° (near-flipped), Uranus ≈ 98° (sideways).
 */
export function spinObliquityDeg(axisEcl: Vec3, rotationRateDegPerDay: number): number {
  const a = norm3(axisEcl);
  const s = rotationRateDegPerDay < 0 ? -1 : 1;
  return Math.acos(Math.max(-1, Math.min(1, s * a.z))) * RAD;
}
