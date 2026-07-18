// moons — REAL ephemeris for the curated major moons (celestial v2 B3,
// directive §3, 2026-07-18: Io, Europa, Ganymede, Callisto (Jupiter),
// Titan (Saturn), Triton (Neptune), Phobos + Deimos (Mars) "rendered as
// bodies: each … a real object at its ephemeris position orbiting its
// planet").
//
// SOURCE OF THE ELEMENTS (copied verbatim, retrieved 2026-07-18):
//   JPL Solar System Dynamics, "Planetary Satellite Mean Orbital Elements",
//   https://ssd.jpl.nasa.gov/sats/elem/ (static table sats/elem/sep.html).
//   Epoch 2000-01-01.5 TDB; elements referred to the LOCAL LAPLACE PLANE,
//   whose pole (R.A./Dec, ICRF) the table lists per satellite; the node is
//   measured from the ascending node of the Laplace plane on the ICRF
//   equator; P is the table's period column; P_apsis / P_node are the
//   argument-of-periapsis and node precession periods (years).
// MEAN RADII (km): JPL SSD "Planetary Satellite Physical Parameters",
//   https://ssd.jpl.nasa.gov/sats/phys_par/ (values from Archinal et al.
//   2018, the IAU WGCCRE 2015 report). Retrieved 2026-07-18.
//
// PROPAGATION MODEL — a precessing Kepler ellipse:
//   node(t) = node0 − sign(cos i)·360·t/P_node   (J2 secular sign: the node
//             REGRESSES for prograde orbits, ADVANCES for retrograde ones —
//             Triton; P_node = 0 in the table ⇒ no node motion)
//   ω(t)    = ω0 + 360·t/P_apsis                 (apsidal precession of the
//             argument of periapsis, prograde for this low-inclination
//             family; P_apsis = 0 ⇒ none)
//   L rate  = 360/P                              (the table's period is the
//             period of the MEAN LONGITUDE L = node + ω + M)
//   M(t)    = M0 + (360/P − dω/dt − dnode/dt)·t
// VALIDATION of these conventions: applying the same recipe to the table's
// own MOON row (mean ecliptic elements: a 384400 km, e 0.0554, ω 318.15,
// M 135.27, i 5.16, node 125.08, P 27.322 d, P_apsis 5.997 yr, P_node
// 18.600 yr) reproduces Schlyter's independently-published lunar rates to
// 5 decimals (node −0.052953°/d, ω +0.164357°/d, M +13.064993°/d) and the
// Moon's position against solarSystem.ts within the mean-vs-perturbed gap —
// pinned in moons.test.ts. Same test pins each curated moon's derived
// anomalistic period within 1% of its familiar sidereal period.
//
// FRAMES: Kepler in the Laplace plane → ICRF equatorial via the listed
// Laplace pole (basis: X = ascending node of the Laplace plane on the ICRF
// equator at R.A. pole+90°, Z = pole) → ecliptic via the same obliquity
// series solarSystem.ts uses (Schlyter §5: ε = 23.4393° − 3.563e-7°·d).
// STATED APPROXIMATIONS (visualization tier, matching the codebase's
// arcmin-class Schlyter ephemeris): ICRF/J2000 axes are treated as the
// equatorial frame of date when rotating into the ecliptic (≤0.4° frame
// conflation per quarter-century of precession); TT/TDB−UTC (~69 s) is
// ignored (source simplification #3 in solarSystem.ts). Mean elements
// carry no short-period perturbations — these are display-grade positions,
// NOT for occultation timing or navigation.
//
// Hermetic: pure math, no DOM, no clock reads. `npx tsx --test`-able.

import { eccentricAnomalyRad } from "./solarSystem.js";

const DEG = Math.PI / 180;

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export type MoonId =
  | "phobos" | "deimos"
  | "io" | "europa" | "ganymede" | "callisto"
  | "titan" | "triton";

export const MOON_IDS: readonly MoonId[] = Object.freeze([
  "phobos", "deimos", "io", "europa", "ganymede", "callisto", "titan", "triton",
]);

/** Table epoch 2000-01-01.5 TDB ≈ 2000-01-01 12:00 UTC (ΔT ignored — stated). */
export const MOON_EPOCH_MS = Date.UTC(2000, 0, 1, 12, 0, 0);

/**
 * One row of the JPL mean-elements table (+ the phys_par radius).
 * `poleRaDeg`/`poleDecDeg` = the LOCAL LAPLACE PLANE pole, ICRF; null ⇒ the
 * reference plane is the ECLIPTIC itself (the table's Earth-Moon row — used
 * only by the validation test, not a curated body here).
 */
export interface MoonElements {
  parent: "mars" | "jupiter" | "saturn" | "neptune" | "earth";
  aKm: number;
  e: number;
  wDeg: number;      // argument of periapsis at epoch
  MDeg: number;      // mean anomaly at epoch
  iDeg: number;      // inclination to the reference plane
  nodeDeg: number;   // node, from the reference plane's ICRF-equator node
  PDays: number;     // table period column (mean-longitude period)
  PapsisYr: number;  // apsidal precession period, years (0 = none listed)
  PnodeYr: number;   // node precession period, years (0 = none listed)
  poleRaDeg: number | null;
  poleDecDeg: number | null;
  radiusKm: number;  // phys_par mean radius
}

/** Verbatim JPL rows (see header for source + retrieval date). */
export const MOONS: Record<MoonId, MoonElements> = {
  phobos: {
    parent: "mars", aKm: 9375, e: 0.015, wDeg: 216.3, MDeg: 189.7, iDeg: 1.1,
    nodeDeg: 169.2, PDays: 0.3187, PapsisYr: 1.1, PnodeYr: 2.3,
    poleRaDeg: 317.7, poleDecDeg: 52.9, radiusKm: 11.08,
  },
  deimos: {
    parent: "mars", aKm: 23457, e: 0.000, wDeg: 0.0, MDeg: 205.0, iDeg: 1.8,
    nodeDeg: 54.3, PDays: 1.2625, PapsisYr: 0, PnodeYr: 56.2,
    poleRaDeg: 316.6, poleDecDeg: 53.5, radiusKm: 6.2,
  },
  io: {
    parent: "jupiter", aKm: 421800, e: 0.004, wDeg: 49.1, MDeg: 330.9, iDeg: 0.0,
    nodeDeg: 0.0, PDays: 1.762732, PapsisYr: 1.333, PnodeYr: 0,
    poleRaDeg: 268.1, poleDecDeg: 64.5, radiusKm: 1821.49,
  },
  europa: {
    parent: "jupiter", aKm: 671100, e: 0.009, wDeg: 45.0, MDeg: 345.4, iDeg: 0.5,
    nodeDeg: 184.0, PDays: 3.525463, PapsisYr: 1.394, PnodeYr: 30.202,
    poleRaDeg: 268.1, poleDecDeg: 64.5, radiusKm: 1560.80,
  },
  ganymede: {
    parent: "jupiter", aKm: 1070400, e: 0.001, wDeg: 198.3, MDeg: 324.8, iDeg: 0.2,
    nodeDeg: 58.5, PDays: 7.155588, PapsisYr: 68.301, PnodeYr: 137.812,
    poleRaDeg: 268.2, poleDecDeg: 64.6, radiusKm: 2631.20,
  },
  callisto: {
    parent: "jupiter", aKm: 1882700, e: 0.007, wDeg: 43.8, MDeg: 87.4, iDeg: 0.3,
    nodeDeg: 309.1, PDays: 16.690440, PapsisYr: 277.921, PnodeYr: 577.264,
    poleRaDeg: 268.7, poleDecDeg: 64.8, radiusKm: 2410.30,
  },
  titan: {
    parent: "saturn", aKm: 1221900, e: 0.029, wDeg: 78.3, MDeg: 11.7, iDeg: 0.3,
    nodeDeg: 78.6, PDays: 15.945448, PapsisYr: 346.680, PnodeYr: 687.370,
    poleRaDeg: 36.4, poleDecDeg: 84.0, radiusKm: 2574.76,
  },
  triton: {
    parent: "neptune", aKm: 354800, e: 0.000, wDeg: 0.0, MDeg: 63.0, iDeg: 157.3,
    nodeDeg: 178.1, PDays: 5.876994, PapsisYr: 0, PnodeYr: 340.379,
    poleRaDeg: 299.8, poleDecDeg: 43.1, radiusKm: 1352.60,
  },
};

// ── the precessing-ellipse rates (deg/day; see header for signs) ────────────

const YR_DAYS = 365.25;

export interface MoonRates {
  dNode: number; // deg/day, signed
  dW: number;    // deg/day, signed
  dM: number;    // deg/day — the anomalistic mean motion
  dL: number;    // deg/day — 360/P, the mean-longitude rate
}

export function moonRates(el: Pick<MoonElements, "PDays" | "PapsisYr" | "PnodeYr" | "iDeg">): MoonRates {
  const dL = 360 / el.PDays;
  const dW = el.PapsisYr > 0 ? 360 / (el.PapsisYr * YR_DAYS) : 0;
  const retro = Math.cos(el.iDeg * DEG) < 0 ? -1 : 1; // sign(cos i)
  const dNode = el.PnodeYr > 0 ? (-retro * 360) / (el.PnodeYr * YR_DAYS) : 0;
  return { dNode, dW, dM: dL - dW - dNode, dL };
}

/** The closed-ellipse period (mean anomaly through 360°) — what one full
 *  drawn orbit spans, days. Within 1% of the familiar sidereal period for
 *  every curated moon (tested). */
export function moonOrbitPeriodDays(id: MoonId): number {
  return 360 / moonRates(MOONS[id]).dM;
}

// ── frames ──────────────────────────────────────────────────────────────────

/** Schlyter §5 mean obliquity (same series solarSystem.ts uses);
 *  d = days since 1999-12-31 00:00 UT. */
function obliquityRad(dateMs: number): number {
  const d = (dateMs - Date.UTC(1999, 11, 31)) / 86_400_000;
  return (23.4393 - 3.563e-7 * d) * DEG;
}

/** Equatorial → ecliptic-of-date (the spaceFrame.eclFromEq rotation,
 *  duplicated here ON PURPOSE: importing spaceFrame would be circular —
 *  spaceFrame's registry imports this module). */
function eqToEcl(v: Vec3, dateMs: number): Vec3 {
  const e = obliquityRad(dateMs);
  return {
    x: v.x,
    y: Math.cos(e) * v.y + Math.sin(e) * v.z,
    z: -Math.sin(e) * v.y + Math.cos(e) * v.z,
  };
}

/**
 * Parent-centric position from a JPL mean-elements row at `dateMs`, METERS,
 * in the ECLIPTIC-OF-DATE frame (the frame every spaceFrame position uses).
 * Pure. Exported generically so the validation test can run the table's
 * Moon row through the exact same code path.
 */
export function keplerLaplaceOffsetEclM(el: MoonElements, dateMs: number): Vec3 {
  const t = (dateMs - MOON_EPOCH_MS) / 86_400_000; // days since epoch
  const r8 = (x: number): number => ((x % 360) + 360) % 360;
  const rates = moonRates(el);
  const node = (el.nodeDeg + rates.dNode * t) * DEG;
  const w = (el.wDeg + rates.dW * t) * DEG;
  const M = r8(el.MDeg + rates.dM * t);
  // Kepler solve (shared with the planet ephemeris — one solver, one truth)
  const E = eccentricAnomalyRad(M, el.e);
  const aM = el.aKm * 1000;
  const xv = aM * (Math.cos(E) - el.e);
  const yv = aM * Math.sqrt(1 - el.e * el.e) * Math.sin(E);
  // perifocal → reference plane (Rz(node)·Rx(i)·Rz(w) applied to (xv,yv,0))
  const i = el.iDeg * DEG;
  const cosw = Math.cos(w); const sinw = Math.sin(w);
  const cosn = Math.cos(node); const sinn = Math.sin(node);
  const cosi = Math.cos(i); const sini = Math.sin(i);
  const xw = xv * cosw - yv * sinw;
  const yw = xv * sinw + yv * cosw;
  const xr = xw * cosn - yw * cosi * sinn;
  const yr = xw * sinn + yw * cosi * cosn;
  const zr = yw * sini;
  if (el.poleRaDeg == null || el.poleDecDeg == null) {
    // reference plane IS the ecliptic (the table's Earth-Moon row): the
    // ecliptic's ICRF-equator node is the equinox = the ecliptic X axis
    return { x: xr, y: yr, z: zr };
  }
  // reference plane → ICRF equatorial via the Laplace pole basis:
  // X = plane's ascending node on the ICRF equator (R.A. = pole R.A.+90°),
  // Z = pole, Y = Z × X
  const ra = el.poleRaDeg * DEG;
  const dec = el.poleDecDeg * DEG;
  const nx = Math.cos(ra + Math.PI / 2);
  const ny = Math.sin(ra + Math.PI / 2);
  const nz = 0;
  const px = Math.cos(dec) * Math.cos(ra);
  const py = Math.cos(dec) * Math.sin(ra);
  const pz = Math.sin(dec);
  const yx = py * nz - pz * ny;
  const yy = pz * nx - px * nz;
  const yz = px * ny - py * nx;
  const eq = {
    x: nx * xr + yx * yr + px * zr,
    y: ny * xr + yy * yr + py * zr,
    z: nz * xr + yz * yr + pz * zr,
  };
  return eqToEcl(eq, dateMs);
}

/** Curated-moon parent-centric offset, meters, ecliptic of date. */
export function moonLocalOffsetEclM(id: MoonId, dateMs: number): Vec3 {
  return keplerLaplaceOffsetEclM(MOONS[id], dateMs);
}

/** Presentation colors (approximate naked-eye/probe-image tint — NOT data;
 *  the spaceFrame honesty caption covers all body colors). */
export const MOON_COLOR: Record<MoonId, string> = {
  phobos: "#8a8078",
  deimos: "#968c82",
  io: "#d9c05b",
  europa: "#d8cfc2",
  ganymede: "#a89b8a",
  callisto: "#7d7468",
  titan: "#d9a75b",
  triton: "#d8c8c4",
};

export const MOON_NAME: Record<MoonId, string> = {
  phobos: "Phobos", deimos: "Deimos", io: "Io", europa: "Europa",
  ganymede: "Ganymede", callisto: "Callisto", titan: "Titan", triton: "Triton",
};
