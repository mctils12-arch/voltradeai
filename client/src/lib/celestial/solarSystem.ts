// Solar-system ephemeris — EARTH TWIN O6-7 tier 2 (charter: "hand off to a
// celestial scene at LITERALLY ACCURATE SCALE — Earth shrinks to truth, the
// Moon a real 30-earth-diameter distance away at true size, Sun/planets at
// real ephemeris positions").
//
// SOURCE OF THE THEORY (every constant below is copied from it, verified
// against the live page 2026-07-16, section numbers cited inline):
//   Paul Schlyter, "How to compute planetary positions",
//   https://stjarnhimlen.se/comp/ppcomp.html — a low-precision planetary
//   theory derived from T. van Flandern & K. Pulkkinen, "Low-precision
//   formulae for planetary positions", Astrophysical Journal Supplement
//   Series 41, 391-411 (1979). Osculating elements + the major perturbation
//   terms for the Moon, Jupiter, Saturn and Uranus (§4, §9, §10).
//
// HONEST ACCURACY TIER — arcmin-class, visualization-grade. The source (§2)
// designs for 1-2 arcmin near the present epoch; measured against JPL
// Horizons anchors (solarSystem.test.ts, epochs 2000-01-01 and 2026-01-01):
// planets ≤ 1.3 arcmin angular error, Moon ≤ 3.1 arcmin; heliocentric
// distances within 0.013% (inner planets) / 0.24% (outer, Saturn worst).
// Error grows away from the present (source simplification #5: ~2 arcmin per
// 1000 yr for planets, ~7 arcmin for the Moon, growing quadratically); the
// Uranus/Neptune elements embed their 4200-yr mutual perturbation and are
// only valid within a few centuries of now. TT/UT difference (~69 s in 2026)
// is IGNORED per source simplification #3 — that alone is ~0.6 arcmin on the
// Moon, negligible for the rest. Sufficient for drawing the solar system at
// true scale; NOT for navigation, occultation timing, or telescope pointing.
//
// Frame: heliocentric ECLIPTIC OF DATE (the theory's native frame — §5.1).
// All bodies come out in the same frame, so RELATIVE geometry (what the
// true-scale view draws) is frame-consistent. eclipticOfDateToJ2000LonDeg
// (§8) converts longitudes to the J2000 frame — used by the anchor tests to
// compare against JPL Horizons J2000-ecliptic vectors.
//
// Hermetic: pure math, no network, no DOM, no clock reads — a pure function
// of the supplied timestamp, unit-testable with `npx tsx --test`.

const DEG = Math.PI / 180;

// §3: "Day 0.0 occurs at 2000 Jan 0.0 UT (or 1999 Dec 31, 0:00 UT)".
// Epoch ms for 1999-12-31 00:00 UT; d = days (fractional) since then.
const DAY0_MS = Date.UTC(1999, 11, 31, 0, 0, 0);
const MS_PER_DAY = 86_400_000;

/** Schlyter day number d (§3) for a JS timestamp. Exported for tests. */
export function dayNumber(dateMs: number): number {
  return (dateMs - DAY0_MS) / MS_PER_DAY;
}

// ── physical constants ───────────────────────────────────────────────────────

/** 1 astronomical unit in meters — exact by IAU 2012 Resolution B2. */
export const AU_M = 149_597_870_700;

/**
 * Earth EQUATORIAL radius, meters (WGS84/IAU 6378.137 km). Schlyter gives
 * the Moon's distance in "Earth radii" with a = 60.2666 — 60.2666 × 6378.14
 * km = the canonical 384,400 km mean lunar distance, i.e. the theory's
 * Earth radius is the equatorial one. Used ONLY for that conversion.
 */
export const EARTH_EQ_RADIUS_M = 6_378_137;

/**
 * Volumetric mean radii, meters — NASA Space Science Data Coordinated
 * Archive planetary fact sheets (nssdc.gsfc.nasa.gov/planetary/factsheet/),
 * Sun = IAU 2015 Resolution B3 nominal solar radius. These set the TRUE
 * drawn sizes in the true-scale view and the angular diameters below.
 */
export const BODY_RADIUS_M: Record<BodyId, number> = {
  sun: 695_700_000,
  mercury: 2_439_700,
  venus: 6_051_800,
  earth: 6_371_000,
  moon: 1_737_400,
  mars: 3_389_500,
  jupiter: 69_911_000,
  saturn: 58_232_000,
  uranus: 25_362_000,
  neptune: 24_622_000,
};

export type BodyId =
  | "sun" | "mercury" | "venus" | "earth" | "moon"
  | "mars" | "jupiter" | "saturn" | "uranus" | "neptune";

/** Draw order / canonical listing: Sun outward, Moon after Earth. */
export const BODY_ORDER: BodyId[] = [
  "sun", "mercury", "venus", "earth", "moon",
  "mars", "jupiter", "saturn", "uranus", "neptune",
];

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface BodyState {
  id: BodyId;
  name: string;
  /** true physical radius, meters (source above). */
  radiusM: number;
  /** heliocentric ecliptic-of-date position, AU (Sun = origin). */
  helioAu: Vec3;
  /** geocentric ecliptic-of-date position, AU (Earth = origin). */
  geoAu: Vec3;
  /** heliocentric distance, AU. */
  rSunAu: number;
  /** geocentric distance. */
  rEarthAu: number;
  rEarthM: number;
  /** true angular diameter seen from Earth's center, degrees. */
  angularDiameterFromEarthDeg: number;
}

// ── orbital elements (§4 of the source, copied verbatim) ────────────────────
// N = longitude of ascending node, i = inclination, w = argument of
// perihelion (degrees, with linear rates per day d); a = semi-major axis
// (AU; Earth radii for the Moon); e = eccentricity; M = mean anomaly (deg).

interface Elements {
  N: number; i: number; w: number; a: number; e: number; M: number;
}

type ElementFn = (d: number) => Elements;

const ELEMENTS: Record<Exclude<BodyId, "sun" | "earth">, ElementFn> & {
  /** §4 "Sun": the SUN'S GEOCENTRIC orbit (i.e. Earth's orbit reflected). */
  sunGeocentric: ElementFn;
} = {
  sunGeocentric: (d) => ({
    N: 0.0,
    i: 0.0,
    w: 282.9404 + 4.70935e-5 * d,
    a: 1.0,
    e: 0.016709 - 1.151e-9 * d,
    M: 356.047 + 0.9856002585 * d,
  }),
  moon: (d) => ({
    N: 125.1228 - 0.0529538083 * d,
    i: 5.1454,
    w: 318.0634 + 0.1643573223 * d,
    a: 60.2666, // Earth radii (§4 note)
    e: 0.0549,
    M: 115.3654 + 13.0649929509 * d,
  }),
  mercury: (d) => ({
    N: 48.3313 + 3.24587e-5 * d,
    i: 7.0047 + 5.0e-8 * d,
    w: 29.1241 + 1.01444e-5 * d,
    a: 0.387098,
    e: 0.205635 + 5.59e-10 * d,
    M: 168.6562 + 4.0923344368 * d,
  }),
  venus: (d) => ({
    N: 76.6799 + 2.4659e-5 * d,
    i: 3.3946 + 2.75e-8 * d,
    w: 54.891 + 1.38374e-5 * d,
    a: 0.72333,
    e: 0.006773 - 1.302e-9 * d,
    M: 48.0052 + 1.6021302244 * d,
  }),
  mars: (d) => ({
    N: 49.5574 + 2.11081e-5 * d,
    i: 1.8497 - 1.78e-8 * d,
    w: 286.5016 + 2.92961e-5 * d,
    a: 1.523688,
    e: 0.093405 + 2.516e-9 * d,
    M: 18.6021 + 0.5240207766 * d,
  }),
  jupiter: (d) => ({
    N: 100.4542 + 2.76854e-5 * d,
    i: 1.303 - 1.557e-7 * d,
    w: 273.8777 + 1.64505e-5 * d,
    a: 5.20256,
    e: 0.048498 + 4.469e-9 * d,
    M: 19.895 + 0.0830853001 * d,
  }),
  saturn: (d) => ({
    N: 113.6634 + 2.3898e-5 * d,
    i: 2.4886 - 1.081e-7 * d,
    w: 339.3939 + 2.97661e-5 * d,
    a: 9.55475,
    e: 0.055546 - 9.499e-9 * d,
    M: 316.967 + 0.0334442282 * d,
  }),
  uranus: (d) => ({
    N: 74.0005 + 1.3978e-5 * d,
    i: 0.7733 + 1.9e-8 * d,
    w: 96.6612 + 3.0565e-5 * d,
    a: 19.18171 - 1.55e-8 * d,
    e: 0.047318 + 7.45e-9 * d,
    M: 142.5905 + 0.011725806 * d,
  }),
  neptune: (d) => ({
    N: 131.7806 + 3.0173e-5 * d,
    i: 1.77 - 2.55e-7 * d,
    w: 272.8461 - 6.027e-6 * d,
    a: 30.05826 + 3.313e-8 * d,
    e: 0.008606 + 2.15e-9 * d,
    M: 260.2471 + 0.005995147 * d,
  }),
};

// ── core orbit math (§5-§7 of the source) ────────────────────────────────────

const norm360 = (deg: number): number => ((deg % 360) + 360) % 360;

/**
 * Kepler's equation E - e·sin E = M, solved by the source's §5.2 iteration
 * (Newton's method in radians). e ≤ 0.21 here (Mercury) — converges in a
 * handful of steps; 25 is a generous hard cap.
 */
export function eccentricAnomalyRad(Mdeg: number, e: number): number {
  const M = norm360(Mdeg) * DEG;
  let E = M + e * Math.sin(M) * (1 + e * Math.cos(M)); // §5.2 first approx
  for (let iter = 0; iter < 25; iter++) {
    const dE = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E -= dE;
    if (Math.abs(dE) < 1e-9) break;
  }
  return E;
}

interface Spherical {
  lonDeg: number; // ecliptic longitude, of date
  latDeg: number; // ecliptic latitude
  r: number;      // in the element set's unit (AU, or Earth radii for Moon)
}

/** §6+§7: elements → position in the orbital plane → ecliptic spherical. */
function elementsToSpherical(el: Elements): Spherical {
  const E = eccentricAnomalyRad(el.M, el.e);
  const xv = el.a * (Math.cos(E) - el.e);
  const yv = el.a * Math.sqrt(1 - el.e * el.e) * Math.sin(E);
  const v = Math.atan2(yv, xv); // true anomaly
  const r = Math.hypot(xv, yv);
  const N = el.N * DEG;
  const i = el.i * DEG;
  const vw = v + el.w * DEG;
  const xh = r * (Math.cos(N) * Math.cos(vw) - Math.sin(N) * Math.sin(vw) * Math.cos(i));
  const yh = r * (Math.sin(N) * Math.cos(vw) + Math.cos(N) * Math.sin(vw) * Math.cos(i));
  const zh = r * Math.sin(vw) * Math.sin(i);
  return {
    lonDeg: norm360(Math.atan2(yh, xh) / DEG),
    latDeg: Math.atan2(zh, Math.hypot(xh, yh)) / DEG,
    r,
  };
}

function sphericalToVec(s: Spherical): Vec3 {
  const lon = s.lonDeg * DEG;
  const lat = s.latDeg * DEG;
  return {
    x: s.r * Math.cos(lat) * Math.cos(lon),
    y: s.r * Math.cos(lat) * Math.sin(lon),
    z: s.r * Math.sin(lat),
  };
}

// ── perturbations (§9 Moon, §10 Jupiter/Saturn/Uranus — copied verbatim) ────

function moonPerturbed(d: number): Spherical {
  const mo = ELEMENTS.moon(d);
  const su = ELEMENTS.sunGeocentric(d);
  const base = elementsToSpherical(mo);
  // §9 fundamental arguments (degrees)
  const Ms = su.M;                    // Sun mean anomaly
  const Mm = mo.M;                    // Moon mean anomaly
  const Nm = mo.N;                    // Moon node longitude
  const Ls = Ms + su.w;               // Sun mean longitude (Ns = 0)
  const Lm = Mm + mo.w + Nm;          // Moon mean longitude
  const D = Lm - Ls;                  // mean elongation
  const F = Lm - Nm;                  // argument of latitude
  const s = (x: number) => Math.sin(x * DEG);
  const c = (x: number) => Math.cos(x * DEG);
  const dLon =
    -1.274 * s(Mm - 2 * D)            // Evection
    + 0.658 * s(2 * D)                // Variation
    - 0.186 * s(Ms)                   // Yearly Equation
    - 0.059 * s(2 * Mm - 2 * D)
    - 0.057 * s(Mm - 2 * D + Ms)
    + 0.053 * s(Mm + 2 * D)
    + 0.046 * s(2 * D - Ms)
    + 0.041 * s(Mm - Ms)
    - 0.035 * s(D)                    // Parallactic Equation
    - 0.031 * s(Mm + Ms)
    - 0.015 * s(2 * F - 2 * D)
    + 0.011 * s(Mm - 4 * D);
  const dLat =
    -0.173 * s(F - 2 * D)
    - 0.055 * s(Mm - F - 2 * D)
    - 0.046 * s(Mm + F - 2 * D)
    + 0.033 * s(F + 2 * D)
    + 0.017 * s(2 * Mm + F);
  const dR = -0.58 * c(Mm - 2 * D) - 0.46 * c(2 * D); // Earth radii
  return {
    lonDeg: norm360(base.lonDeg + dLon),
    latDeg: base.latDeg + dLat,
    r: base.r + dR,
  };
}

/** §10 longitude (and Saturn latitude) perturbations, degrees. */
function outerPerturbations(
  d: number,
): { jupiterLon: number; saturnLon: number; saturnLat: number; uranusLon: number } {
  const Mj = ELEMENTS.jupiter(d).M;
  const Ms = ELEMENTS.saturn(d).M;
  const Mu = ELEMENTS.uranus(d).M;
  const s = (x: number) => Math.sin(x * DEG);
  const c = (x: number) => Math.cos(x * DEG);
  return {
    jupiterLon:
      -0.332 * s(2 * Mj - 5 * Ms - 67.6)
      - 0.056 * s(2 * Mj - 2 * Ms + 21)
      + 0.042 * s(3 * Mj - 5 * Ms + 21)
      - 0.036 * s(Mj - 2 * Ms)
      + 0.022 * c(Mj - Ms)
      + 0.023 * s(2 * Mj - 3 * Ms + 52)
      - 0.016 * s(Mj - 5 * Ms - 69),
    saturnLon:
      0.812 * s(2 * Mj - 5 * Ms - 67.6)
      - 0.229 * c(2 * Mj - 4 * Ms - 2)
      + 0.119 * s(Mj - 2 * Ms - 3)
      + 0.046 * s(2 * Mj - 6 * Ms - 69)
      + 0.014 * s(Mj - 3 * Ms + 32),
    saturnLat:
      -0.02 * c(2 * Mj - 4 * Ms - 2)
      + 0.018 * s(2 * Mj - 6 * Ms - 49),
    uranusLon:
      0.04 * s(Ms - 2 * Mu + 6)
      + 0.035 * s(Ms - 3 * Mu + 33)
      - 0.015 * s(Mj - Mu + 20),
  };
}

// ── precession to J2000 (§8) ────────────────────────────────────────────────

/**
 * §8: add `3.82394E-5 * (365.2422 * (Epoch − 2000.0) − d)` degrees to an
 * ecliptic-of-date longitude to refer it to a standard epoch. For J2000
 * that is −3.82394e-5·d. Used by the anchor tests to compare against JPL
 * Horizons J2000-frame vectors.
 */
export function eclipticOfDateToJ2000LonDeg(lonDeg: number, dateMs: number): number {
  return norm360(lonDeg + 3.82394e-5 * (-dayNumber(dateMs)));
}

// ── public API ───────────────────────────────────────────────────────────────

const NAMES: Record<BodyId, string> = {
  sun: "Sun", mercury: "Mercury", venus: "Venus", earth: "Earth",
  moon: "Moon", mars: "Mars", jupiter: "Jupiter", saturn: "Saturn",
  uranus: "Uranus", neptune: "Neptune",
};

const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const len = (v: Vec3): number => Math.hypot(v.x, v.y, v.z);

/** True angular diameter, degrees: 2·asin(R/dist) — exact spherical form. */
export function angularDiameterDeg(radiusM: number, distanceM: number): number {
  if (!(distanceM > radiusM)) return 180; // inside the body — cap honestly
  return (2 * Math.asin(radiusM / distanceM)) / DEG;
}

/**
 * The whole system at one instant: heliocentric + geocentric ecliptic-of-
 * date positions for Sun, Moon, Earth and the 8 planets (BODY_ORDER order).
 * Pure function of the timestamp.
 */
export function solarSystemState(dateMs: number): BodyState[] {
  const d = dayNumber(dateMs);

  // Sun's geocentric position (§4 "Sun" elements) → Earth's heliocentric
  // position is its exact negation (same frame, opposite origin).
  const sunGeoSph = elementsToSpherical(ELEMENTS.sunGeocentric(d));
  const sunGeo = sphericalToVec(sunGeoSph);
  const earthHelio: Vec3 = { x: -sunGeo.x, y: -sunGeo.y, z: -sunGeo.z };

  // Moon: §9 gives GEOCENTRIC spherical in Earth radii → AU.
  const moonSph = moonPerturbed(d);
  const moonGeo = sphericalToVec({
    lonDeg: moonSph.lonDeg,
    latDeg: moonSph.latDeg,
    r: (moonSph.r * EARTH_EQ_RADIUS_M) / AU_M,
  });

  // Planets: heliocentric spherical + §10 perturbations where they apply.
  const pert = outerPerturbations(d);
  const helioSph = (id: Exclude<BodyId, "sun" | "earth" | "moon">): Spherical => {
    const s = elementsToSpherical(ELEMENTS[id](d));
    if (id === "jupiter") s.lonDeg = norm360(s.lonDeg + pert.jupiterLon);
    if (id === "saturn") {
      s.lonDeg = norm360(s.lonDeg + pert.saturnLon);
      s.latDeg += pert.saturnLat;
    }
    if (id === "uranus") s.lonDeg = norm360(s.lonDeg + pert.uranusLon);
    return s;
  };

  const helioOf: Record<BodyId, Vec3> = {
    sun: { x: 0, y: 0, z: 0 },
    earth: earthHelio,
    moon: { x: earthHelio.x + moonGeo.x, y: earthHelio.y + moonGeo.y, z: earthHelio.z + moonGeo.z },
    mercury: sphericalToVec(helioSph("mercury")),
    venus: sphericalToVec(helioSph("venus")),
    mars: sphericalToVec(helioSph("mars")),
    jupiter: sphericalToVec(helioSph("jupiter")),
    saturn: sphericalToVec(helioSph("saturn")),
    uranus: sphericalToVec(helioSph("uranus")),
    neptune: sphericalToVec(helioSph("neptune")),
  };

  return BODY_ORDER.map((id) => {
    const helio = helioOf[id];
    const geo = id === "moon" ? moonGeo : sub(helio, earthHelio);
    const rEarthAu = id === "earth" ? 0 : len(geo);
    const rEarthM = rEarthAu * AU_M;
    return {
      id,
      name: NAMES[id],
      radiusM: BODY_RADIUS_M[id],
      helioAu: helio,
      geoAu: geo,
      rSunAu: len(helio),
      rEarthAu,
      rEarthM,
      angularDiameterFromEarthDeg:
        id === "earth" ? 0 : angularDiameterDeg(BODY_RADIUS_M[id], rEarthM),
    };
  });
}
