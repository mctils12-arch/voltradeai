// starCatalog — PURE MATH + ASSET CODEC for the REAL bright-star sky in the
// Canvas2D space frame (GALAXY / STARFIELD DETAIL upgrade, human directive
// 2026-07-19: "the galaxy in the background need to be way more detailed" —
// answered with REAL star data, never a fabricated random sky).
//
// DATA: the Yale Bright Star Catalog (BSC5), 9096 naked-eye stars, PUBLIC
// DOMAIN. Source JSON: brettonw/YaleBrightStarCatalog (bsc5-short.json) —
// per star RA ("06h 45m 08.9s"), Dec ("-16° 42′ 58″"), V (visual magnitude,
// -1.46 Sirius … 7.96), K (color temperature, Kelvin), and a name/constellation
// on the bright ones. We do NOT ship the 932 KB JSON: a build step (see
// scripts note below) parses it with THESE pure functions and writes a compact
// fixed-stride binary (client/public/space/bsc5.bin, ~107 KB) holding, per
// star, the precomputed J2000 equatorial unit direction (3×int16), the V
// magnitude (int16 ×100), and the blackbody RGB (3×uint8). This module both
// ENCODES that binary (shared by the build step + the roundtrip test) and
// DECODES it at load. NO NETWORK here — the frame fetches the bundled asset
// and hands the ArrayBuffer to decodeStarAsset.
//
// REGISTRATION WITH THE PANORAMA: the Milky Way panorama is mapped through the
// REAL galactic frame in ECLIPTIC-of-date coordinates (textureSphere
// galacticBasisEcl, from the J2000 IAU galactic anchors). A star's stored
// direction is J2000 EQUATORIAL; eqToEclDir converts it to the SAME
// ecliptic-of-date frame with the SAME obliquity series the ephemeris and the
// panorama use — so a real bright star lands exactly where the panorama's own
// star field shows it. Both are the one real sky.
//
// HONESTY: positions and colors are REAL (Yale BSC). Magnitude→size and
// K→RGB are PRESENTATION mappings (a monotonic real-magnitude scale; a
// standard blackbody-locus approximation) — approximate naked-eye appearance,
// stated, never claimed as photometry.

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

const DEG = Math.PI / 180;

// ── RA / Dec string parsing ─────────────────────────────────────────────────

/** "06h 45m 08.9s" → degrees (hours ×15). Tolerant of spacing/decimals. */
export function parseRA(ra: string): number {
  const m = ra.match(/(-?\d+(?:\.\d+)?)\s*h\s*(\d+(?:\.\d+)?)\s*m\s*(\d+(?:\.\d+)?)\s*s/i);
  if (!m) throw new Error(`bad RA: ${ra}`);
  const h = parseFloat(m[1]);
  const min = parseFloat(m[2]);
  const s = parseFloat(m[3]);
  return (h + min / 60 + s / 3600) * 15;
}

/** "-16° 42′ 58″" → degrees. The SIGN is read from the leading token (so
 *  "-00° 30′ 11″" is correctly negative even though the degree field is 0). */
export function parseDec(dec: string): number {
  const m = dec.match(/([+-]?)\s*(\d+(?:\.\d+)?)\s*[°d]\s*(\d+(?:\.\d+)?)\s*['′m]\s*(\d+(?:\.\d+)?)\s*["″s]/i);
  if (!m) throw new Error(`bad Dec: ${dec}`);
  const sign = m[1] === "-" ? -1 : 1;
  const d = parseFloat(m[2]);
  const min = parseFloat(m[3]);
  const s = parseFloat(m[4]);
  return sign * (d + min / 60 + s / 3600);
}

/** J2000 equatorial unit vector from RA/Dec in degrees. */
export function raDecToUnitEq(raDeg: number, decDeg: number): Vec3 {
  const ra = raDeg * DEG;
  const dec = decDeg * DEG;
  return { x: Math.cos(dec) * Math.cos(ra), y: Math.cos(dec) * Math.sin(ra), z: Math.sin(dec) };
}

// ── equatorial → ecliptic-of-date (IDENTICAL to spaceFrame.eclFromEq /
// textureSphere's eqToEcl — the same Schlyter §5 obliquity series, so stars
// register with the ephemeris bodies AND the panorama) ──────────────────────

export function obliquityDeg(dateMs: number): number {
  const d = (dateMs - Date.UTC(1999, 11, 31)) / 86_400_000;
  return 23.4393 - 3.563e-7 * d;
}

/** Rotate an equatorial vector into the ecliptic-of-date frame. */
export function eqToEclDir(veq: Vec3, dateMs: number): Vec3 {
  const e = obliquityDeg(dateMs) * DEG;
  return {
    x: veq.x,
    y: Math.cos(e) * veq.y + Math.sin(e) * veq.z,
    z: -Math.sin(e) * veq.y + Math.cos(e) * veq.z,
  };
}

// ── V magnitude → point size + brightness ───────────────────────────────────
// Magnitude is a LOG scale (a 5-mag step = ×100 flux); a linear-in-magnitude
// ramp is therefore already a log-compression of flux — the right perceptual
// footing. Brighter (lower V) ⇒ bigger AND brighter, monotonically.

export const MAG_LIMIT = 8; // catalog faintest ≈ 7.96
const SIZE_MIN = 0.5;
const SIZE_MAX = 2.8;
const SIZE_SLOPE = 0.26;
const INTENSITY_FLOOR = 0.12;
const MAG_FULL = 1.5; // V ≤ 1.5 renders at full intensity

/** Rendered point radius (px) for a star of visual magnitude V. */
export function starSizePx(mag: number): number {
  const s = SIZE_MIN + (MAG_LIMIT - mag) * SIZE_SLOPE;
  return s < SIZE_MIN ? SIZE_MIN : s > SIZE_MAX ? SIZE_MAX : s;
}

/** Brightness multiplier (0..1) for a star of visual magnitude V — a floor so
 *  the faintest catalog stars stay just-visible, saturating at V ≤ 1.5. */
export function starIntensity(mag: number): number {
  const t = (MAG_LIMIT - mag) / (MAG_LIMIT - MAG_FULL);
  return t < INTENSITY_FLOOR ? INTENSITY_FLOOR : t > 1 ? 1 : t;
}

// ── color temperature (Kelvin) → RGB ────────────────────────────────────────
// A blackbody-locus approximation (Tanner Helland's widely-used fit). Hot
// stars trend blue-white, cool stars orange-red — the real spectral color,
// presented (not photometry). Given a plausible temperature range 2000–48000 K.

export function kelvinToRGB(kelvin: number): [number, number, number] {
  const t = Math.max(1000, Math.min(40000, kelvin)) / 100;
  let r: number;
  let g: number;
  let b: number;
  // red
  if (t <= 66) r = 255;
  else r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
  // green
  if (t <= 66) g = 99.4708025861 * Math.log(t) - 161.1195681661;
  else g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  // blue
  if (t >= 66) b = 255;
  else if (t <= 19) b = 0;
  else b = 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  const clamp = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  return [clamp(r), clamp(g), clamp(b)];
}

// ── gnomonic projection onto the sky sphere (stars are at infinity) ──────────
// Scale-invariant: identical to spaceFrame.projectPoint for a point at
// camPos + dir·R for any R>0 (R cancels). front=false ⇒ behind the camera.

export interface CamBasisLike {
  f: Vec3;
  r: Vec3;
  u: Vec3;
}

export interface StarScreenPos {
  x: number;
  y: number;
  front: boolean;
}

export function projectStarScreen(
  dirEcl: Vec3,
  basis: CamBasisLike,
  k: number,
  cxPx: number,
  cyPx: number,
): StarScreenPos {
  const fwd = dirEcl.x * basis.f.x + dirEcl.y * basis.f.y + dirEcl.z * basis.f.z;
  if (fwd <= 1e-6) return { x: NaN, y: NaN, front: false };
  const rr = dirEcl.x * basis.r.x + dirEcl.y * basis.r.y + dirEcl.z * basis.r.z;
  const uu = dirEcl.x * basis.u.x + dirEcl.y * basis.u.y + dirEcl.z * basis.u.z;
  return { x: cxPx + (k * rr) / fwd, y: cyPx - (k * uu) / fwd, front: true };
}

// ── the compact binary asset codec ──────────────────────────────────────────
// Layout (little-endian):
//   [0..7]   magic "VTSTAR01" (ASCII)
//   [8..11]  uint32 count
//   [12..15] uint32 stride (=12)
//   then count records, stride 12:
//     +0  int16  ex  (J2000 EQUATORIAL unit ×32767)
//     +2  int16  ey
//     +4  int16  ez
//     +6  int16  magCenti (V ×100)
//     +8  uint8  r
//     +9  uint8  g
//     +10 uint8  b
//     +11 uint8  flags (reserved, 0)

export const STAR_MAGIC = "VTSTAR01";
export const STAR_STRIDE = 12;

export interface StarRecord {
  /** J2000 EQUATORIAL unit vector. */
  ex: number;
  ey: number;
  ez: number;
  /** visual magnitude V. */
  mag: number;
  r: number;
  g: number;
  b: number;
}

export function encodeStarAsset(records: StarRecord[]): ArrayBuffer {
  const n = records.length;
  const buf = new ArrayBuffer(16 + n * STAR_STRIDE);
  const dv = new DataView(buf);
  for (let i = 0; i < 8; i++) dv.setUint8(i, STAR_MAGIC.charCodeAt(i));
  dv.setUint32(8, n, true);
  dv.setUint32(12, STAR_STRIDE, true);
  const q = (v: number): number => Math.max(-32767, Math.min(32767, Math.round(v * 32767)));
  const cb = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  for (let i = 0; i < n; i++) {
    const rec = records[i];
    const o = 16 + i * STAR_STRIDE;
    dv.setInt16(o, q(rec.ex), true);
    dv.setInt16(o + 2, q(rec.ey), true);
    dv.setInt16(o + 4, q(rec.ez), true);
    dv.setInt16(o + 6, Math.max(-32768, Math.min(32767, Math.round(rec.mag * 100))), true);
    dv.setUint8(o + 8, cb(rec.r));
    dv.setUint8(o + 9, cb(rec.g));
    dv.setUint8(o + 10, cb(rec.b));
    dv.setUint8(o + 11, 0);
  }
  return buf;
}

export interface StarCatalog {
  count: number;
  /** 3×count J2000 EQUATORIAL unit directions (re-normalized after int16 quant). */
  dirEq: Float32Array;
  /** count visual magnitudes. */
  mag: Float32Array;
  /** 3×count blackbody RGB (0..255). */
  rgb: Uint8Array;
  /** sparse index→name for the brightest labeled stars (attached by the loader). */
  names: Map<number, string>;
}

export function decodeStarAsset(buf: ArrayBuffer): StarCatalog {
  const dv = new DataView(buf);
  for (let i = 0; i < 8; i++) {
    if (dv.getUint8(i) !== STAR_MAGIC.charCodeAt(i)) throw new Error("bad star asset magic");
  }
  const count = dv.getUint32(8, true);
  const stride = dv.getUint32(12, true);
  if (stride !== STAR_STRIDE) throw new Error(`unexpected stride ${stride}`);
  if (buf.byteLength < 16 + count * stride) throw new Error("star asset truncated");
  const dirEq = new Float32Array(count * 3);
  const mag = new Float32Array(count);
  const rgb = new Uint8Array(count * 3);
  for (let i = 0; i < count; i++) {
    const o = 16 + i * stride;
    let ex = dv.getInt16(o, true) / 32767;
    let ey = dv.getInt16(o + 2, true) / 32767;
    let ez = dv.getInt16(o + 4, true) / 32767;
    const l = Math.hypot(ex, ey, ez) || 1;
    ex /= l;
    ey /= l;
    ez /= l;
    dirEq[i * 3] = ex;
    dirEq[i * 3 + 1] = ey;
    dirEq[i * 3 + 2] = ez;
    mag[i] = dv.getInt16(o + 6, true) / 100;
    rgb[i * 3] = dv.getUint8(o + 8);
    rgb[i * 3 + 1] = dv.getUint8(o + 9);
    rgb[i * 3 + 2] = dv.getUint8(o + 10);
  }
  return { count, dirEq, mag, rgb, names: new Map() };
}

// ── credit (surfaced on-screen — public domain courtesy line) ────────────────
export const STAR_CATALOG_CREDIT = "Stars: Yale Bright Star Catalog (public domain)";
