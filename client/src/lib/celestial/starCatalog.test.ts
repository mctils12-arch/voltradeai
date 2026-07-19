import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseRA, parseDec, raDecToUnitEq, eqToEclDir, obliquityDeg,
  kelvinToRGB, starSizePx, starIntensity,
  encodeStarAsset, decodeStarAsset, projectStarScreen,
  MAG_LIMIT, STAR_STRIDE,
  type StarRecord,
} from "./starCatalog.ts";

const RAD = 180 / Math.PI;
const close = (a: number, b: number, eps: number, msg: string): void =>
  assert.ok(Math.abs(a - b) <= eps, `${msg}: ${a} vs ${b} (eps ${eps})`);

// ── RA/Dec string parsing ───────────────────────────────────────────────────

test("parseRA: Sirius 06h 45m 08.9s → 101.287°", () => {
  close(parseRA("06h 45m 08.9s"), 101.2871, 1e-3, "Sirius RA deg");
  close(parseRA("00h 00m 00.0s"), 0, 1e-9, "zero RA");
  close(parseRA("12h 00m 00s"), 180, 1e-9, "12h → 180°");
});

test("parseDec: sign read from the leading token, not the degree field", () => {
  close(parseDec("-16° 42′ 58″"), -16.7161, 1e-3, "Sirius Dec");
  close(parseDec("+45° 13′ 45″"), 45.2292, 1e-3, "positive Dec");
  // the trap: degree field 00 but the whole value is negative
  close(parseDec("-00° 30′ 11″"), -0.503, 1e-3, "negative sub-degree Dec");
  close(parseDec("+00° 30′ 00″"), 0.5, 1e-6, "positive sub-degree Dec");
});

// ── RA/Dec → scene ecliptic frame: pins a KNOWN bright star (Sirius) ─────────

test("Sirius maps to the correct sky direction (equatorial + ecliptic frame)", () => {
  const raDeg = parseRA("06h 45m 08.9s");
  const decDeg = parseDec("-16° 42′ 58″");
  const eq = raDecToUnitEq(raDeg, decDeg);
  close(Math.hypot(eq.x, eq.y, eq.z), 1, 1e-9, "unit equatorial");
  // equatorial declination recovered from z
  close(Math.asin(eq.z) * RAD, decDeg, 1e-6, "dec from z");
  // ecliptic-of-date: Sirius sits at ecliptic latitude ≈ -39.6°, longitude ≈ 104°
  const ecl = eqToEclDir(eq, Date.UTC(2026, 6, 19));
  const betaDeg = Math.asin(ecl.z) * RAD;
  const lambdaDeg = ((Math.atan2(ecl.y, ecl.x) * RAD) + 360) % 360;
  close(betaDeg, -39.6, 0.3, "Sirius ecliptic latitude");
  close(lambdaDeg, 104.1, 0.5, "Sirius ecliptic longitude");
});

test("eqToEclDir matches the ephemeris obliquity series and preserves length", () => {
  close(obliquityDeg(Date.UTC(2000, 0, 1)), 23.4393, 1e-3, "obliquity ~2000");
  const eq = raDecToUnitEq(83.0, 22.0);
  const ecl = eqToEclDir(eq, Date.UTC(2026, 6, 19));
  close(Math.hypot(ecl.x, ecl.y, ecl.z), 1, 1e-9, "rotation preserves unit length");
  // the ecliptic pole check: equatorial +z (celestial north pole) → ecliptic
  // (0, sin ε, cos ε)
  const np = eqToEclDir({ x: 0, y: 0, z: 1 }, Date.UTC(2026, 6, 19));
  const eRad = obliquityDeg(Date.UTC(2026, 6, 19)) / RAD;
  close(np.y, Math.sin(eRad), 1e-9, "NCP ecliptic y = sin ε");
  close(np.z, Math.cos(eRad), 1e-9, "NCP ecliptic z = cos ε");
});

// ── K color temperature → RGB blackbody ─────────────────────────────────────

test("kelvinToRGB: hot star blue-white, cool star orange-red", () => {
  const hot = kelvinToRGB(30000); // O/B blue
  const cool = kelvinToRGB(3000); // M red
  assert.ok(hot[2] > hot[0], `hot star blue>red: ${hot}`);
  assert.ok(cool[0] > cool[2], `cool star red>blue: ${cool}`);
  // a hotter star has a higher blue/red ratio than a cooler one (monotone hue)
  const ratio = (c: number[]): number => c[2] / (c[0] + 1);
  assert.ok(ratio(kelvinToRGB(20000)) > ratio(kelvinToRGB(9750)), "20000K bluer than 9750K");
  assert.ok(ratio(kelvinToRGB(9750)) > ratio(kelvinToRGB(4500)), "9750K bluer than 4500K");
  // ~6500K is near-white (all channels high, roughly balanced)
  const white = kelvinToRGB(6500);
  assert.ok(white[0] > 200 && white[1] > 200 && white[2] > 180, `~6500K near white: ${white}`);
});

// ── magnitude → size + brightness (brighter ⇒ bigger + brighter) ─────────────

test("starSizePx / starIntensity: monotonic in magnitude, Sirius the biggest", () => {
  const mags = [-1.46, 0, 2, 4, 6, 7.9];
  const sizes = mags.map(starSizePx);
  const ints = mags.map(starIntensity);
  for (let i = 1; i < mags.length; i++) {
    assert.ok(sizes[i] < sizes[i - 1], `size decreases with magnitude at ${mags[i]}`);
    assert.ok(ints[i] <= ints[i - 1], `intensity non-increasing with magnitude at ${mags[i]}`);
  }
  assert.ok(starSizePx(-1.46) >= starSizePx(0), "Sirius at least as large as a mag-0 star");
  assert.equal(starIntensity(-1.46), 1, "Sirius full intensity");
  assert.ok(starIntensity(MAG_LIMIT) >= 0.1, "faint stars keep a visibility floor");
  assert.ok(starSizePx(7.9) >= 0.5, "faint stars keep a minimum size");
});

// ── projection onto the sky sphere ──────────────────────────────────────────

test("projectStarScreen: forward star → center; behind → front=false", () => {
  const basis = { f: { x: 0, y: 0, z: 1 }, r: { x: 1, y: 0, z: 0 }, u: { x: 0, y: 1, z: 0 } };
  const c = projectStarScreen({ x: 0, y: 0, z: 1 }, basis, 500, 640, 360);
  assert.ok(c.front, "star straight ahead is in front");
  close(c.x, 640, 1e-6, "center x");
  close(c.y, 360, 1e-6, "center y");
  const behind = projectStarScreen({ x: 0, y: 0, z: -1 }, basis, 500, 640, 360);
  assert.equal(behind.front, false, "star behind the camera not drawn");
  // a star to the right of forward projects right of center
  const right = projectStarScreen({ x: Math.sin(0.1), y: 0, z: Math.cos(0.1) }, basis, 500, 640, 360);
  assert.ok(right.x > 640, "rightward star projects right of center");
});

// ── the compact binary codec ────────────────────────────────────────────────

test("encode/decode roundtrip preserves direction, magnitude, and color", () => {
  const recs: StarRecord[] = [
    { ex: 1, ey: 0, ez: 0, mag: -1.46, r: 170, g: 190, b: 255 },
    { ex: 0, ey: 0.6, ez: -0.8, mag: 4.61, r: 255, g: 200, b: 120 },
    { ex: -0.5773, ey: 0.5773, ez: 0.5773, mag: 7.96, r: 255, g: 255, b: 255 },
  ];
  const buf = encodeStarAsset(recs);
  assert.equal(buf.byteLength, 16 + recs.length * STAR_STRIDE, "stride layout");
  const cat = decodeStarAsset(buf);
  assert.equal(cat.count, 3, "count");
  for (let i = 0; i < 3; i++) {
    // directions re-normalized after int16 quantization — compare as unit dirs
    const dx = cat.dirEq[i * 3];
    const dy = cat.dirEq[i * 3 + 1];
    const dz = cat.dirEq[i * 3 + 2];
    close(Math.hypot(dx, dy, dz), 1, 1e-4, `unit dir ${i}`);
    const l = Math.hypot(recs[i].ex, recs[i].ey, recs[i].ez);
    close(dx, recs[i].ex / l, 5e-4, `dir x ${i}`);
    close(dy, recs[i].ey / l, 5e-4, `dir y ${i}`);
    close(dz, recs[i].ez / l, 5e-4, `dir z ${i}`);
    close(cat.mag[i], recs[i].mag, 0.005, `mag ${i}`);
    assert.equal(cat.rgb[i * 3], recs[i].r, `r ${i}`);
    assert.equal(cat.rgb[i * 3 + 1], recs[i].g, `g ${i}`);
    assert.equal(cat.rgb[i * 3 + 2], recs[i].b, `b ${i}`);
  }
});

test("decodeStarAsset rejects a bad magic", () => {
  const buf = new ArrayBuffer(16);
  assert.throws(() => decodeStarAsset(buf), /magic/);
});
