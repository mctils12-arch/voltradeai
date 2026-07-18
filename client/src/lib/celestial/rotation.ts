// rotation — AXIAL TILT + TRUE ROTATION for every rendered body (celestial
// v2 B3, directive §3, 2026-07-18: "every planet + the Moon rendered with
// correct axial tilt and rotating at its true rate (Moon tidally locked —
// always same face to Earth)").
//
// SOURCE (all constants copied verbatim, retrieved 2026-07-18):
//   NAIF generic kernel pck00011.tpc (JPL), which implements the IAU WGCCRE
//   2015 report — B. A. Archinal et al. (2018), "Report of the IAU Working
//   Group on Cartographic Coordinates and Rotational Elements: 2015",
//   Celest. Mech. Dyn. Astron. 130:22. Model:
//     pole  α = α0 + α1·T + Σ ai·sin(θi)     (ICRF R.A., degrees)
//           δ = δ0 + δ1·T + Σ di·cos(θi)     (ICRF Dec, degrees)
//     prime meridian W = W0 + Ẇ·d + Ẅ·d² + Σ wi·sin(θi)   (degrees)
//   with T = Julian centuries, d = days, from J2000 (2000-01-01 12:00 TDB;
//   ΔT ~69 s ignored — the codebase's stated convention), and θi the
//   per-system nutation/precession angles θ0 + θ1·T (+ θ2·T²).
//
// TRUNCATION (stated): trigonometric series terms with amplitude < 0.1° are
// dropped (display tier — sub-tenth-degree pole/meridian wobble is far
// below a pixel on any rendered disc). Kept series: the Moon's full 13-term
// libration set (up to 3.56° in W), Triton's 9-term pole precession (up to
// 32° — dropping it would be materially wrong), Neptune's 0.7° pole cone,
// Mars' large slow terms, Phobos/Deimos' multi-degree terms, and the
// Galilean >0.01° terms (Io's 0.094° pair kept since it is only 2 terms).
// Mercury's ≤0.011° longitude librations and Jupiter's ≤0.002° pole terms
// are dropped under the cutoff.
//
// W IS MEASURED from the node of the body's equator on the ICRF equator
// (the point at R.A. = α+90°), along the body's equator. Positive Ẇ =
// prograde rotation; Venus and Uranus carry negative Ẇ (retrograde) —
// exactly as the IAU model states them.
//
// TIDAL LOCKING is not asserted here — it FALLS OUT of the real constants:
// each curated moon's |Ẇ| equals its orbital mean motion (tested against
// the moons.ts elements), and the Moon's prime meridian faces Earth to
// within its real optical libration (cross-checked against the independent
// Schlyter ephemeris in rotation.test.ts).
//
// Hermetic: pure math, no DOM, no clock reads. `npx tsx --test`-able.

const DEG = Math.PI / 180;
/** J2000 epoch (2000-01-01 12:00; TDB≈TT≈UTC at this tier — stated). */
export const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export type RotationBodyId =
  | "sun" | "mercury" | "venus" | "earth" | "moon" | "mars"
  | "jupiter" | "saturn" | "uranus" | "neptune"
  | "phobos" | "deimos" | "io" | "europa" | "ganymede" | "callisto"
  | "titan" | "triton";

/** One trig term: angle θ = th0 + th1·T (+ th2·T²) degrees; contributes
 *  ra·sin θ to α, dec·cos θ to δ, pm·sin θ to W. */
interface NutTerm {
  th0: number;
  th1: number;
  th2?: number;
  ra: number;
  dec: number;
  pm: number;
}

interface RotModel {
  /** α0, α1 (deg, deg/century) */
  ra: [number, number];
  /** δ0, δ1 (deg, deg/century) */
  dec: [number, number];
  /** W0, Ẇ (deg, deg/DAY), optional Ẅ (deg/day²) */
  pm: [number, number, number?];
  nut?: NutTerm[];
}

// Per-system nutation-precession angles (pck00011 BODYn_NUT_PREC_ANGLES),
// inlined per body as self-contained term lists — verbatim numbers.

// Moon angles E1..E13 (BODY3_NUT_PREC_ANGLES pairs, deg + deg/century):
const MOON_E: ReadonlyArray<readonly [number, number]> = [
  [125.045, -1935.5364525], [250.089, -3871.072905], [260.008, 475263.3328725],
  [176.625, 487269.629985], [357.529, 35999.0509575], [311.589, 964468.49931],
  [134.963, 477198.869325], [276.617, 12006.300765], [34.226, 63863.5132425],
  [15.134, -5806.6093575], [119.743, 131.84064], [239.961, 6003.1503825],
  [25.053, 473327.79642],
];
const MOON_RA_A = [-3.8787, -0.1204, 0.07, -0.0172, 0, 0.0072, 0, 0, 0, -0.0052, 0, 0, 0.0043];
const MOON_DEC_A = [1.5419, 0.0239, -0.0278, 0.0068, 0, -0.0029, 0.0009, 0, 0, 0.0008, 0, 0, -0.0009];
const MOON_PM_A = [3.561, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047, -0.0046, 0.0028, 0.0052, 0.004, 0.0019, -0.0044];

const MOON_NUT: NutTerm[] = MOON_E.map(([th0, th1], i) => ({
  th0, th1, ra: MOON_RA_A[i], dec: MOON_DEC_A[i], pm: MOON_PM_A[i],
}));

// Jupiter-system angles used by the Galileans (BODY5 pairs J3..J8):
const J3: readonly [number, number] = [283.90, 4850.7];
const J4: readonly [number, number] = [355.80, 1191.3];
const J5: readonly [number, number] = [119.90, 262.1];
const J6: readonly [number, number] = [229.80, 64.3];
const J8: readonly [number, number] = [113.35, 6070.0];

// Neptune-system angle N (BODY8 pair 1) and Triton's N7 (pair 8):
const NEP_N: readonly [number, number] = [357.85, 52.316];
const N7: readonly [number, number] = [177.85, 52.316];

// Mars-system angles (BODY4 triplets) used by Phobos (M1..M5), Deimos
// (M6..M10) and the >0.1° Mars planet terms:
const M1: readonly [number, number, number] = [190.72646643, 15917.10818695, 0];
const M2: readonly [number, number, number] = [21.4689247, 31834.27934054, 0];
const M3: readonly [number, number, number] = [332.86082793, 19139.89694742, 0];
const M4: readonly [number, number, number] = [394.93256437, 38280.79631835, 0];
const M5: readonly [number, number, number] = [189.6327156, 41215158.1842005, 12.711923222];
const M6: readonly [number, number, number] = [121.46893664, 660.22803474, 0];
const M7: readonly [number, number, number] = [231.05028581, 660.9912354, 0];
const M8: readonly [number, number, number] = [251.37314025, 1320.50145245, 0];
const M9: readonly [number, number, number] = [217.98635955, 38279.9612555, 0];
const M10: readonly [number, number, number] = [196.19729402, 19139.83628608, 0];

const term = (
  th: readonly [number, number] | readonly [number, number, number],
  ra: number, dec: number, pm: number,
): NutTerm => ({ th0: th[0], th1: th[1], th2: th.length > 2 ? th[2] : undefined, ra, dec, pm });

/** pck00011 constants, verbatim (see header). */
const MODELS: Record<RotationBodyId, RotModel> = {
  sun: { ra: [286.13, 0], dec: [63.87, 0], pm: [84.176, 14.1844] },
  mercury: { ra: [281.0103, -0.0328], dec: [61.4155, -0.0049], pm: [329.5988, 6.1385108] },
  venus: { ra: [272.76, 0], dec: [67.16, 0], pm: [160.20, -1.4813688] },
  earth: { ra: [0, -0.641], dec: [90, -0.557], pm: [190.147, 360.9856235] },
  moon: { ra: [269.9949, 0.0031], dec: [66.5392, 0.0130], pm: [38.3213, 13.17635815, -1.4e-12], nut: MOON_NUT },
  mars: {
    ra: [317.269202, -0.10927547], dec: [54.432516, -0.05827105],
    pm: [176.049863, 350.891982443297],
    // only the >0.1° slow terms (see truncation note)
    nut: [
      term([79.398797, 0.5042615, 0], 0.419057, 0, 0),
      term([166.325722, 0.5042615, 0], 0, 1.591274, 0),
      term([95.391654, 0.5042615, 0], 0, 0, 0.584542),
    ],
  },
  jupiter: { ra: [268.056595, -0.006499], dec: [64.495303, 0.002413], pm: [284.95, 870.536] },
  saturn: { ra: [40.589, -0.036], dec: [83.537, -0.004], pm: [38.90, 810.7939024] },
  uranus: { ra: [257.311, 0], dec: [-15.175, 0], pm: [203.81, -501.1600928] },
  neptune: {
    ra: [299.36, 0], dec: [43.46, 0], pm: [249.978, 541.1397757],
    nut: [term(NEP_N, 0.70, -0.51, -0.48)],
  },
  phobos: {
    ra: [317.67071657, -0.10844326], dec: [52.88627266, -0.06134706],
    pm: [35.1877444, 1128.84475928, 9.536137031212154e-9],
    nut: [
      term(M1, -1.78428399, -1.07516537, 1.42421769),
      term(M2, 0.02212824, 0.00668626, -0.02273783),
      term(M3, -0.01028251, -0.0064874, 0.00410711),
      term(M4, -0.00475595, 0.00281576, 0.00631964),
      term(M5, 0, 0, -1.143),
    ],
  },
  deimos: {
    ra: [316.65705808, -0.10518014], dec: [53.50992033, -0.05979094],
    pm: [79.39932954, 285.16188899],
    nut: [
      term(M6, 3.09217726, 1.83936004, -2.73954829),
      term(M7, 0.22980637, 0.1432532, -0.39968606),
      term(M8, 0.06418655, 0.01911409, -0.06563259),
      term(M9, 0.02533537, -0.0148259, -0.0291294),
      term(M10, 0.00778695, 0.0019243, 0.0169916),
    ],
  },
  io: {
    ra: [268.05, -0.009], dec: [64.50, 0.003], pm: [200.39, 203.4889538],
    nut: [term(J3, 0.094, 0.040, -0.085), term(J4, 0.024, 0.011, -0.022)],
  },
  europa: {
    ra: [268.08, -0.009], dec: [64.51, 0.003], pm: [36.022, 101.3747235],
    nut: [
      term(J4, 1.086, 0.468, -0.980),
      term(J5, 0.060, 0.026, -0.054),
      term(J6, 0.015, 0.007, -0.014),
      term([352.25, 2382.6], 0.009, 0.002, -0.008), // J7
    ],
  },
  ganymede: {
    ra: [268.20, -0.009], dec: [64.57, 0.003], pm: [44.064, 50.3176081],
    nut: [
      term(J4, -0.037, -0.016, 0.033),
      term(J5, 0.431, 0.186, -0.389),
      term(J6, 0.091, 0.039, -0.082),
    ],
  },
  callisto: {
    ra: [268.72, -0.009], dec: [64.83, 0.003], pm: [259.51, 21.5710715],
    nut: [
      term(J5, -0.068, -0.029, 0.061),
      term(J6, 0.590, 0.254, -0.533),
      term(J8, 0.010, -0.004, -0.009),
    ],
  },
  titan: { ra: [39.4827, 0], dec: [83.4279, 0], pm: [186.5855, 22.5769768] },
  triton: {
    ra: [299.36, 0], dec: [41.17, 0], pm: [296.53, -61.2572637],
    // pole/meridian cone about Neptune's pole: harmonics k·N7, k = 1..9
    nut: [
      term(N7, -32.35, 22.55, 22.25),
      term([2 * N7[0], 2 * N7[1]], -6.28, 2.10, 6.73),
      term([3 * N7[0], 3 * N7[1]], -2.08, 0.55, 2.05),
      term([4 * N7[0], 4 * N7[1]], -0.74, 0.16, 0.74),
      term([5 * N7[0], 5 * N7[1]], -0.28, 0.05, 0.28),
      term([6 * N7[0], 6 * N7[1]], -0.11, 0.02, 0.11),
      term([7 * N7[0], 7 * N7[1]], -0.07, 0.01, 0.05),
      term([8 * N7[0], 8 * N7[1]], -0.02, 0.00, 0.02),
      term([9 * N7[0], 9 * N7[1]], -0.01, 0.00, 0.01),
    ],
  },
};

export const ROTATION_IDS = Object.freeze(Object.keys(MODELS)) as readonly RotationBodyId[];

export function hasRotationModel(id: string): id is RotationBodyId {
  return Object.prototype.hasOwnProperty.call(MODELS, id);
}

/** Pole R.A./Dec (ICRF, degrees) at `dateMs`, series included. */
export function poleRaDecDeg(id: RotationBodyId, dateMs: number): { raDeg: number; decDeg: number } {
  const m = MODELS[id];
  const T = (dateMs - J2000_MS) / 86_400_000 / 36525;
  let ra = m.ra[0] + m.ra[1] * T;
  let dec = m.dec[0] + m.dec[1] * T;
  if (m.nut) {
    for (const n of m.nut) {
      const th = (n.th0 + n.th1 * T + (n.th2 ?? 0) * T * T) * DEG;
      if (n.ra) ra += n.ra * Math.sin(th);
      if (n.dec) dec += n.dec * Math.cos(th);
    }
  }
  return { raDeg: ra, decDeg: dec };
}

/** Prime-meridian angle W (degrees, [0, 360)), series included. Measured
 *  from the body-equator/ICRF-equator node (R.A. = α+90°). */
export function primeMeridianDeg(id: RotationBodyId, dateMs: number): number {
  const m = MODELS[id];
  const d = (dateMs - J2000_MS) / 86_400_000;
  const T = d / 36525;
  let w = m.pm[0] + m.pm[1] * d + (m.pm[2] ?? 0) * d * d;
  if (m.nut) {
    for (const n of m.nut) {
      const th = (n.th0 + n.th1 * T + (n.th2 ?? 0) * T * T) * DEG;
      if (n.pm) w += n.pm * Math.sin(th);
    }
  }
  return ((w % 360) + 360) % 360;
}

/** Signed rotation rate, deg/day (negative = retrograde: Venus, Uranus,
 *  Triton's W). */
export function rotationRateDegPerDay(id: RotationBodyId): number {
  return MODELS[id].pm[1];
}

/** Sidereal rotation period, days (360/|Ẇ|). */
export function rotationPeriodDays(id: RotationBodyId): number {
  return 360 / Math.abs(MODELS[id].pm[1]);
}

// ── frames (same conventions as moons.ts; duplicated to keep this module
// import-light — spaceFrame imports it) ─────────────────────────────────────

function obliquityRad(dateMs: number): number {
  const d = (dateMs - Date.UTC(1999, 11, 31)) / 86_400_000;
  return (23.4393 - 3.563e-7 * d) * DEG;
}

function eqToEcl(v: Vec3, dateMs: number): Vec3 {
  const e = obliquityRad(dateMs);
  return {
    x: v.x,
    y: Math.cos(e) * v.y + Math.sin(e) * v.z,
    z: -Math.sin(e) * v.y + Math.cos(e) * v.z,
  };
}

/** Spin-axis (positive pole) unit vector, ECLIPTIC OF DATE — the frame
 *  every spaceFrame position lives in. */
export function axisEclOfDate(id: RotationBodyId, dateMs: number): Vec3 {
  const { raDeg, decDeg } = poleRaDecDeg(id, dateMs);
  const ra = raDeg * DEG;
  const dec = decDeg * DEG;
  return eqToEcl(
    { x: Math.cos(dec) * Math.cos(ra), y: Math.cos(dec) * Math.sin(ra), z: Math.sin(dec) },
    dateMs,
  );
}

/**
 * The W = 0 reference direction: the ascending node of the body's equator
 * on the ICRF equator (R.A. = α+90°), ecliptic of date. Perpendicular to
 * the pole by construction. Texture renderers factor the orientation as
 * (node frame, W): a pixel's body longitude is its node-frame longitude
 * minus W, so a spinning body re-samples the texture without rebuilding
 * the geometry LUT (SPACE VIEW upgrade 2026-07-18 — textureSphere.ts).
 */
export function equatorNodeDirEclOfDate(id: RotationBodyId, dateMs: number): Vec3 {
  const { raDeg } = poleRaDecDeg(id, dateMs);
  const ra = raDeg * DEG;
  return eqToEcl({ x: Math.cos(ra + Math.PI / 2), y: Math.sin(ra + Math.PI / 2), z: 0 }, dateMs);
}

/**
 * Unit vector from the body's center through the surface point where its
 * PRIME MERIDIAN crosses its equator, ecliptic of date — pole + this vector
 * fully orient the body (B4/B5 surface textures consume exactly this pair).
 * Construction per the IAU convention: rotate the equator/ICRF-equator node
 * direction about the pole by W.
 */
export function primeMeridianDirEclOfDate(id: RotationBodyId, dateMs: number): Vec3 {
  const { raDeg, decDeg } = poleRaDecDeg(id, dateMs);
  const w = primeMeridianDeg(id, dateMs) * DEG;
  const ra = raDeg * DEG;
  const dec = decDeg * DEG;
  // node of the body equator on the ICRF equator (X), pole (Z), Y = Z × X
  const nx = Math.cos(ra + Math.PI / 2);
  const ny = Math.sin(ra + Math.PI / 2);
  const px = Math.cos(dec) * Math.cos(ra);
  const py = Math.cos(dec) * Math.sin(ra);
  const pz = Math.sin(dec);
  const yx = py * 0 - pz * ny;
  const yy = pz * nx - px * 0;
  const yz = px * ny - py * nx;
  const eq = {
    x: nx * Math.cos(w) + yx * Math.sin(w),
    y: ny * Math.cos(w) + yy * Math.sin(w),
    z: 0 * Math.cos(w) + yz * Math.sin(w),
  };
  return eqToEcl(eq, dateMs);
}
