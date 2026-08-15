// Orbital coverage / footprint geometry — pure spherical trigonometry.
//
// WORKSTREAM: ORBITAL program (d)+(e-imagery), the COVERAGE / FOOTPRINT
// GEOMETRY engine (research/orbital_program.md → "COVERAGE / FOOTPRINT
// ANALYSIS" + "O7 COVERAGE/FOOTPRINT TOOLS").
//
// This module takes satellite POSITIONS as input ({latDeg,lonDeg,altKm})
// and computes ground coverage caps, topocentric look angles, a GPS DOP
// geometry proxy, and a generic next-pass finder. It does NOT do SGP4 /
// orbit propagation itself — positions are injected by the caller (the
// parent wires them to the SGP4 propagator). Consequently this file has
// ZERO dependencies, does ZERO network I/O, and reads NO clock: every
// function is a pure function of its arguments, fully hermetic and
// deterministic.
//
// EARTH MODEL: SPHERICAL TRIGONOMETRY (footprint law-of-sines, ENU
// look-angles) over a locally-radius-corrected sphere. The angular
// derivations assume a sphere; the RADIUS fed into them is
// wgs84GeocentricRadiusKm(latDeg) — the true WGS-84 distance from Earth's
// centre to the ellipsoid surface at that point's own geodetic latitude —
// not a single constant. This is a standard "local osculating sphere"
// approximation, not full ellipsoidal geodesy: it removes the dominant
// (radius) term of the ellipsoid/sphere mismatch while keeping every
// existing angular formula unchanged. It is NOT survey-grade geodesy:
// altKm is measured along the ellipsoid NORMAL (propagate.ts's
// eciToGeodetic) but is added here along the LOCAL RADIAL direction, a
// residual sub-0.01% effect at LEO/GEO altitudes (Q24). Callers needing
// centimetre geodesy must use a full WGS84 model.
//
// HISTORY (Q24, research/PROGRAM_STATE.md): before this fix, every function
// below added altKm to the single constant EARTH_MEAN_RADIUS_KM (6371, mean
// radius) regardless of latitude — measured at 0.02-0.2% range/geometry
// error (LEO 550km/GEO, equator vs pole) because propagate.ts's altKm is
// height above the WGS-84 ellipsoid (6378.137 equatorial / 6356.752 polar),
// not above a 6371 sphere. Fixed by computing the radius AT the point's own
// latitude instead of assuming one radius for the whole planet — see
// wgs84GeocentricRadiusKm() below. EARTH_MEAN_RADIUS_KM is kept, unchanged,
// as its own correct constant (a spherical MEAN, still useful elsewhere,
// e.g. earthConstants.test.ts's cross-file conflicting_const guard) — it is
// simply no longer what this module adds altKm to.
//
// HONESTY: min-elevation "masks" are CONVENTIONS, not physics. Where a
// value is published (Starlink user-terminal ~25deg, GPS horizon 0deg)
// it is a real published figure; where inferred it must be labelled
// ESTIMATED by the caller. Every function that uses a mask takes it as
// an explicit argument and the coverage cap it produces is only as exact
// as that mask — never present an assumed cone as ground truth.

import { EARTH_EQUATORIAL_RADIUS_KM, EARTH_POLAR_RADIUS_KM } from './satDerived.js';

/**
 * Mean Earth radius (km) — kept for API/back-compat and cross-file
 * consistency checks (earthConstants.test.ts's conflicting_const guard
 * pins this exact value against `satDerived.ts`'s EARTH_EQUATORIAL_RADIUS_KM
 * being different on purpose).
 *
 * RENAMED from `EARTH_RADIUS_KM` (Q14). `orbital/satDerived.ts` exported that
 * same name as 6378.137, so two files in this directory disagreed by 7.1km
 * about what "the" Earth radius is, and an importer got whichever module they
 * happened to reach for. Both values are RIGHT — mean radius for a spherical
 * cap, WGS-84 equatorial for SGP4 apsides — which is exactly why the shared
 * name was the defect and unifying the numbers would have been the wrong fix.
 *
 * NO LONGER what altKm is added to below (Q24, fixed — see the module header
 * and wgs84GeocentricRadiusKm()): groundFootprintRadiusKm() and
 * elevationAzimuthFromSite() now use the point's own latitude-corrected WGS-84
 * radius instead of this single mean value. This constant itself is
 * unchanged and still correct as a mean radius; it just isn't the number the
 * geometry below needs.
 */
export const EARTH_MEAN_RADIUS_KM = 6371;

/**
 * WGS-84 geocentric radius (km): the distance from Earth's centre to the
 * ellipsoid SURFACE at geodetic latitude latDeg. Standard closed form
 * (e.g. Vallado "Fundamentals of Astrodynamics and Applications" eq. 3-27,
 * or any WGS-84 reference): reduces exactly to EARTH_EQUATORIAL_RADIUS_KM at
 * the equator and EARTH_POLAR_RADIUS_KM at the poles (pinned in
 * geometry.test.ts).
 *
 * THE Q24 FIX: propagate.ts's eciToGeodetic() emits altKm as height above
 * the WGS-84 ellipsoid AT THE POINT'S OWN LATITUDE — not above a fixed
 * sphere. Every function below that used to add altKm to the constant
 * EARTH_MEAN_RADIUS_KM now adds it to wgs84GeocentricRadiusKm(latDeg)
 * instead, which removes the dominant (radius) term of the 0.02-0.2% error
 * the module header used to log, without touching any of the angular
 * (spherical-trig) derivations, which stay exact for a sphere of THAT
 * radius at THAT point — the "local osculating sphere" approximation
 * described in the module header above.
 */
export function wgs84GeocentricRadiusKm(latDeg: number): number {
  const phi = latDeg * DEG2RAD;
  const aCos = EARTH_EQUATORIAL_RADIUS_KM * Math.cos(phi);
  const bSin = EARTH_POLAR_RADIUS_KM * Math.sin(phi);
  const num = Math.sqrt(
    (EARTH_EQUATORIAL_RADIUS_KM * aCos) ** 2 + (EARTH_POLAR_RADIUS_KM * bSin) ** 2,
  );
  const den = Math.sqrt(aCos * aCos + bSin * bSin);
  return num / den;
}

/**
 * Published Starlink user-terminal minimum-elevation mask (degrees).
 * CONVENTION: ~25deg is the commonly-cited public figure; treat coverage
 * built on it as reflecting that assumption, not a guaranteed link.
 */
export const STARLINK_MIN_ELEV_DEG = 25;

/**
 * A common satellite-tracking / acquisition minimum-elevation mask
 * (degrees). CONVENTION: 10deg is a widely-used default that trims the
 * noisy near-horizon geometry; not a physical constant.
 */
export const DEFAULT_MIN_ELEV_DEG = 10;

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** Normalize a longitude to the half-open range [-180, 180) degrees. */
export function normalizeLonDeg(lonDeg: number): number {
  let x = ((lonDeg + 180) % 360 + 360) % 360 - 180;
  // Guard the +180 boundary landing exactly on 180 due to fp.
  if (x === 180) x = -180;
  return x;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A satellite position: geodetic lat/lon (deg) and altitude above the
 *  WGS-84 ellipsoid (km, per propagate.ts's eciToGeodetic). Altitude is
 *  added to wgs84GeocentricRadiusKm(latDeg), not to a fixed constant. */
export interface GeoPosition {
  latDeg: number;
  lonDeg: number;
  altKm: number;
}

/** Result of the ground-footprint cap computation. */
export interface FootprintRadius {
  /** Great-circle surface radius of the coverage cap (km) = R * lambda. */
  radiusKm: number;
  /** Central (Earth-centre) half-angle of the cap, radians. */
  centralAngleRad: number;
  /** Same central half-angle in degrees. */
  centralAngleDeg: number;
}

/** Topocentric look angles from a ground site to a satellite. */
export interface LookAngles {
  /** Elevation above the local horizon, degrees ([-90, 90]). */
  elevationDeg: number;
  /** Azimuth, degrees clockwise from true North ([0, 360)). */
  azimuthDeg: number;
  /** Straight-line slant range site->satellite, km. */
  rangeKm: number;
}

/** Dilution-of-precision proxy values (dimensionless). See gpsDop(). */
export interface Dop {
  /** Geometric DOP: sqrt(trace of the full 4x4 cofactor). */
  gdop: number;
  /** Position DOP (E,N,U): sqrt(Q_EE+Q_NN+Q_UU). */
  pdop: number;
  /** Horizontal DOP (E,N): sqrt(Q_EE+Q_NN). */
  hdop: number;
  /** Vertical DOP (U): sqrt(Q_UU). */
  vdop: number;
  /** Time DOP (clock): sqrt(Q_tt). */
  tdop: number;
}

/** A ground-track position sampler: maps epoch-ms -> satellite position.
 *  The parent injects one backed by the SGP4 propagator; tests inject a
 *  deterministic synthetic track. */
export type PositionSampler = (dateMs: number) => GeoPosition;

/** Options for nextPassOverSite(). */
export interface NextPassOptions {
  /** Start of the search window (epoch ms). */
  startMs: number;
  /** End of the search window (epoch ms). Defaults to startMs +
   *  windowSec*1000. */
  endMs?: number;
  /** Alternative to endMs: window length in seconds (default 24h). */
  windowSec?: number;
  /** Coarse scan step (seconds, default 30). MUST be shorter than the
   *  minimum expected pass duration or brief high-mask passes can be
   *  stepped over entirely — an honest limitation of coarse sampling. */
  stepSec?: number;
  /** Minimum-elevation mask defining "visible" (degrees, default 10). */
  minElevDeg?: number;
  /** Bisection resolution for the refined rise/set times (seconds,
   *  default 1). */
  refineSec?: number;
  /** If true and the satellite is already above the mask at startMs,
   *  report that in-progress pass with nextRiseMs = startMs. Default
   *  false: only a genuine below->above crossing counts as a "pass",
   *  so an already-overhead satellite is skipped to the next rise. */
  includeInProgress?: boolean;
}

/** Result of nextPassOverSite(). */
export interface PassResult {
  /** Refined rise time (epoch ms) — when elevation crosses up through
   *  the mask. Equals startMs for an in-progress pass when
   *  includeInProgress is set. */
  nextRiseMs: number;
  /** Refined set time (epoch ms) — when elevation crosses back down
   *  through the mask. Equals endMs if the pass had not set by the end
   *  of the search window (see setWithinWindow). */
  setMs: number;
  /** Maximum elevation observed during the pass (deg), at coarse
   *  stepSec resolution. */
  maxElevationDeg: number;
  /** Pass duration (setMs - nextRiseMs) in seconds. */
  durationSec: number;
  /** False if the pass had not ended by endMs (set time is a window
   *  clamp, not a true set). */
  setWithinWindow: boolean;
}

// ---------------------------------------------------------------------------
// 1. Ground-footprint cap radius
// ---------------------------------------------------------------------------

/**
 * Spherical-cap ground-coverage radius for a satellite.
 *
 * GEOMETRY (law of sines on the Earth-centre / satellite / edge-observer
 * triangle, the standard coverage-geometry derivation, e.g. Wertz "Space
 * Mission Analysis and Design", Roddy "Satellite Communications"):
 *
 *   Let R = Earth radius AT THE SUB-SATELLITE LATITUDE (Q24 —
 *   wgs84GeocentricRadiusKm(latDeg), not a fixed constant), r = R + altKm,
 *   eps = min elevation. An observer at the edge of coverage sees the
 *   satellite at elevation eps; the interior angle of the triangle at that
 *   observer is (90deg + eps). By the law of sines the angle subtended at
 *   the satellite is  asin( R * cos(eps) / r ), and the Earth-centre
 *   central half-angle is
 *
 *       lambda = 90deg - eps - asin( R * cos(eps) / r ).
 *
 *   The surface (great-circle) radius of the cap is  R * lambda  (lambda
 *   in radians).
 *
 * Monotonic increasing in altitude at fixed mask. At eps = 90deg
 * (satellite must be at zenith) lambda -> 0. At eps = 0 (true horizon)
 * lambda = 90deg - asin(R/r), the maximum geometric cap.
 *
 * The cap is only as meaningful as the mask: a Starlink 25deg cap is the
 * area within which a terminal sees the satellite >=25deg up, NOT a
 * guaranteed-service area.
 */
export function groundFootprintRadiusKm(
  altKm: number,
  minElevDeg: number,
  latDeg: number,
): FootprintRadius {
  if (!(altKm > 0)) {
    return { radiusKm: 0, centralAngleRad: 0, centralAngleDeg: 0 };
  }
  const R = wgs84GeocentricRadiusKm(latDeg);
  const r = R + altKm;
  const eps = clamp(minElevDeg, -90, 90) * DEG2RAD;
  const sinAtSat = clamp((R * Math.cos(eps)) / r, -1, 1);
  const angAtSat = Math.asin(sinAtSat);
  let lambda = Math.PI / 2 - eps - angAtSat;
  if (lambda < 0) lambda = 0;
  return {
    radiusKm: R * lambda,
    centralAngleRad: lambda,
    centralAngleDeg: lambda * RAD2DEG,
  };
}

// ---------------------------------------------------------------------------
// 2. Footprint circle (small-circle ring for map rendering)
// ---------------------------------------------------------------------------

/**
 * Central angle (great-circle angular separation) between two geographic
 * points, degrees. Wrap-invariant, so it is the correct way to verify a
 * point lies on a footprint cap regardless of antimeridian wrapping.
 */
export function centralAngleDeg(
  lat1Deg: number,
  lon1Deg: number,
  lat2Deg: number,
  lon2Deg: number,
): number {
  const p1 = lat1Deg * DEG2RAD;
  const p2 = lat2Deg * DEG2RAD;
  const dl = (lon2Deg - lon1Deg) * DEG2RAD;
  const cosd = clamp(
    Math.sin(p1) * Math.sin(p2) + Math.cos(p1) * Math.cos(p2) * Math.cos(dl),
    -1,
    1,
  );
  return Math.acos(cosd) * RAD2DEG;
}

/**
 * The coverage footprint as a ring of [lon, lat] points (GeoJSON axis
 * order) — the small-circle on the sphere at the cap's central half-angle
 * around the sub-satellite point.
 *
 * For each bearing theta (evenly spaced over 360deg) the point at angular
 * distance lambda along that bearing is found with the standard spherical
 * destination-point formula (Bowring / "aviation formulary"):
 *
 *   lat2 = asin( sin(lat1)cos(d) + cos(lat1)sin(d)cos(theta) )
 *   lon2 = lon1 + atan2( sin(theta)sin(d)cos(lat1),
 *                        cos(d) - sin(lat1)sin(lat2) )
 *
 * where d = lambda (the cap central half-angle) and (lat1,lon1) is the
 * sub-satellite point.
 *
 * RETURNS exactly nPoints points, an OPEN ring (first point is not
 * repeated at the end). A caller needing a closed GeoJSON LinearRing
 * appends a copy of the first point.
 *
 * The cap's central angle (d, below) is computed at the SUB-SATELLITE
 * point's own latitude (Q24 — groundFootprintRadiusKm(altKm, minElevDeg,
 * subLatDeg)), matching the same latitude-corrected radius every other
 * function in this module now uses.
 *
 * ANTIMERIDIAN: every longitude is normalized to [-180, 180). A cap that
 * straddles the antimeridian therefore returns points on both sides with
 * a jump of ~360deg between two consecutive entries. A naive renderer
 * will draw a spurious horizontal streak across the map; such renderers
 * must split the ring wherever |lon[i+1]-lon[i]| > 180. (MapLibre/GeoJSON
 * antimeridian-cutting handles this if the polygon is split correctly.)
 *
 * POLES: the destination formula is exact point-by-point even when the
 * cap contains a pole, so each returned [lon,lat] is correct. But a cap
 * that ENCLOSES a pole is not a simple lon-monotonic polygon — the
 * longitudes sweep the full range and the ring encircles the pole. A
 * renderer must treat such a ring as pole-enclosing (stitch across the
 * pole) rather than as a plain polygon. This function does not attempt
 * that stitching; it honestly returns the raw small-circle samples.
 */
export function footprintCircle(
  subLatDeg: number,
  subLonDeg: number,
  altKm: number,
  minElevDeg: number,
  nPoints = 64,
): number[][] {
  const { centralAngleRad: d } = groundFootprintRadiusKm(altKm, minElevDeg, subLatDeg);
  const lat1 = subLatDeg * DEG2RAD;
  const lon1 = subLonDeg * DEG2RAD;
  const sinLat1 = Math.sin(lat1);
  const cosLat1 = Math.cos(lat1);
  const sinD = Math.sin(d);
  const cosD = Math.cos(d);
  const ring: number[][] = [];
  const n = Math.max(3, Math.floor(nPoints));
  for (let i = 0; i < n; i++) {
    const theta = (2 * Math.PI * i) / n;
    const sinLat2 = clamp(sinLat1 * cosD + cosLat1 * sinD * Math.cos(theta), -1, 1);
    const lat2 = Math.asin(sinLat2);
    const lon2 =
      lon1 +
      Math.atan2(
        Math.sin(theta) * sinD * cosLat1,
        cosD - sinLat1 * sinLat2,
      );
    ring.push([normalizeLonDeg(lon2 * RAD2DEG), lat2 * RAD2DEG]);
  }
  return ring;
}

// ---------------------------------------------------------------------------
// 3. Topocentric look angles (elevation / azimuth / range)
// ---------------------------------------------------------------------------

/**
 * ECEF (Earth-Centred Earth-Fixed) position of a lat/lon at a given
 * radius-from-centre. Spherical model — the radius the caller supplies is
 * what makes it locally correct (Q24: wgs84GeocentricRadiusKm(latDeg) at
 * that point, not a fixed constant).
 */
function ecef(latRad: number, lonRad: number, radiusKm: number): [number, number, number] {
  const cosLat = Math.cos(latRad);
  return [
    radiusKm * cosLat * Math.cos(lonRad),
    radiusKm * cosLat * Math.sin(lonRad),
    radiusKm * Math.sin(latRad),
  ];
}

/**
 * Look angles from a ground site (assumed at surface, altitude 0) to a
 * satellite. Topocentric ENU (East-North-Up) transform on the spherical
 * Earth model.
 *
 * METHOD (standard topocentric conversion, e.g. Vallado "Fundamentals of
 * Astrodynamics and Applications", the SEZ/ENU look-angle algorithm):
 *   1. Both points -> ECEF (spherical). d = satECEF - siteECEF.
 *   2. Project d onto the site's local East/North/Up basis:
 *        East  = (-sin lon,  cos lon, 0)
 *        North = (-sin lat cos lon, -sin lat sin lon, cos lat)
 *        Up    = ( cos lat cos lon,  cos lat sin lon, sin lat)
 *   3. elevation = atan2(u, hypot(e,n));  range = |d|;
 *      azimuth   = atan2(e, n) measured clockwise from North, in [0,360).
 *
 * Directly overhead -> elevation ~90deg; on the geometric horizon ->
 * elevation ~0deg (and slightly negative just below). The radius at each
 * point is now wgs84GeocentricRadiusKm(latDeg) (Q24, was a fixed mean
 * radius) — the residual bias vs a true WGS84 geodetic horizon is the
 * sub-0.01% "altitude measured along the ellipsoid normal, applied here
 * along the local radial" term described in the module header; stated,
 * not hidden.
 */
export function elevationAzimuthFromSite(
  satLatDeg: number,
  satLonDeg: number,
  satAltKm: number,
  siteLatDeg: number,
  siteLonDeg: number,
): LookAngles {
  const satLat = satLatDeg * DEG2RAD;
  const satLon = satLonDeg * DEG2RAD;
  const siteLat = siteLatDeg * DEG2RAD;
  const siteLon = siteLonDeg * DEG2RAD;

  const sat = ecef(satLat, satLon, wgs84GeocentricRadiusKm(satLatDeg) + satAltKm);
  const site = ecef(siteLat, siteLon, wgs84GeocentricRadiusKm(siteLatDeg));
  const dx = sat[0] - site[0];
  const dy = sat[1] - site[1];
  const dz = sat[2] - site[2];

  const sinLat = Math.sin(siteLat);
  const cosLat = Math.cos(siteLat);
  const sinLon = Math.sin(siteLon);
  const cosLon = Math.cos(siteLon);

  const east = -sinLon * dx + cosLon * dy;
  const north = -sinLat * cosLon * dx - sinLat * sinLon * dy + cosLat * dz;
  const up = cosLat * cosLon * dx + cosLat * sinLon * dy + sinLat * dz;

  const range = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const horiz = Math.sqrt(east * east + north * north);
  const elevationDeg = Math.atan2(up, horiz) * RAD2DEG;
  let azimuthDeg = Math.atan2(east, north) * RAD2DEG;
  if (azimuthDeg < 0) azimuthDeg += 360;

  return { elevationDeg, azimuthDeg, rangeKm: range };
}

// ---------------------------------------------------------------------------
// 4. Visibility convenience
// ---------------------------------------------------------------------------

/**
 * True when the satellite is at or above the site's minimum-elevation
 * mask. The mask is the caller's convention (see module header) — the
 * boolean is only as meaningful as the mask supplied.
 */
export function isVisible(
  satLatDeg: number,
  satLonDeg: number,
  satAltKm: number,
  siteLatDeg: number,
  siteLonDeg: number,
  minElevDeg: number = DEFAULT_MIN_ELEV_DEG,
): boolean {
  const { elevationDeg } = elevationAzimuthFromSite(
    satLatDeg,
    satLonDeg,
    satAltKm,
    siteLatDeg,
    siteLonDeg,
  );
  return elevationDeg >= minElevDeg;
}

// ---------------------------------------------------------------------------
// 5. GPS DOP geometry proxy
// ---------------------------------------------------------------------------

/**
 * Invert a small NxN matrix by Gauss-Jordan elimination with partial
 * pivoting. Returns null if the matrix is singular / too ill-conditioned
 * to invert (pivot below tolerance).
 */
function invertMatrix(a: number[][]): number[][] | null {
  const n = a.length;
  // Augmented [a | I].
  const m: number[][] = a.map((row, i) => {
    const r = row.slice();
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });
  for (let col = 0; col < n; col++) {
    // Partial pivot.
    let pivotRow = col;
    let pivotAbs = Math.abs(m[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(m[r][col]);
      if (v > pivotAbs) {
        pivotAbs = v;
        pivotRow = r;
      }
    }
    if (pivotAbs < 1e-12) return null; // singular
    if (pivotRow !== col) {
      const tmp = m[col];
      m[col] = m[pivotRow];
      m[pivotRow] = tmp;
    }
    const pivot = m[col][col];
    for (let j = 0; j < 2 * n; j++) m[col][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[col][j];
    }
  }
  return m.map((row) => row.slice(n));
}

/**
 * Geometric Dilution-of-Precision proxy from the sky geometry of the
 * satellites above the horizon.
 *
 * METHOD (textbook GNSS DOP, e.g. Kaplan & Hegarty "Understanding GPS"):
 * each satellite contributes a row to the design matrix H built from its
 * line-of-sight unit vector in local ENU plus a receiver-clock term:
 *
 *     row_i = [ cos(el) sin(az),  cos(el) cos(az),  sin(el),  1 ]
 *
 * The cofactor matrix is  Q = (H^T H)^-1 . Then:
 *     GDOP = sqrt(trace Q)            (position + clock)
 *     PDOP = sqrt(Q_EE + Q_NN + Q_UU)
 *     HDOP = sqrt(Q_EE + Q_NN)
 *     VDOP = sqrt(Q_UU)
 *     TDOP = sqrt(Q_tt)
 * Well-spread satellites -> near-orthogonal rows -> small DOP; clustered
 * satellites -> near-collinear rows -> large DOP.
 *
 * RETURNS null when fewer than 4 satellites are supplied (an
 * underdetermined 4-unknown solution has no DOP) or when the geometry is
 * degenerate enough that H^T H is not invertible.
 *
 * HONEST LIMITATION: this is a pure-GEOMETRY proxy, NOT a survey-grade
 * DOP. It assumes ideal, equally-weighted, error-free pseudoranges;
 * spherical-Earth ENU line-of-sight; and no elevation-dependent or
 * atmospheric weighting, no measurement covariance, no multipath/SNR
 * model. It captures how the sky geometry dilutes precision — the shape
 * of the problem — and nothing about the actual ranging error budget.
 * Use it to compare geometries, not to claim a positioning accuracy.
 */
export function gpsDop(
  satsAboveHorizon: Array<{ elevationDeg: number; azimuthDeg: number }>,
): Dop | null {
  if (!Array.isArray(satsAboveHorizon) || satsAboveHorizon.length < 4) {
    return null;
  }
  // H^T H accumulated directly (4x4), avoids materializing the tall H.
  const HtH: number[][] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  for (const s of satsAboveHorizon) {
    const el = s.elevationDeg * DEG2RAD;
    const az = s.azimuthDeg * DEG2RAD;
    const cosEl = Math.cos(el);
    const row = [cosEl * Math.sin(az), cosEl * Math.cos(az), Math.sin(el), 1];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) HtH[i][j] += row[i] * row[j];
    }
  }
  const Q = invertMatrix(HtH);
  if (!Q) return null;
  const qEE = Q[0][0];
  const qNN = Q[1][1];
  const qUU = Q[2][2];
  const qTT = Q[3][3];
  // Guard tiny negative diagonals from round-off in near-singular cases.
  const safe = (x: number) => Math.sqrt(Math.max(0, x));
  const dop: Dop = {
    gdop: safe(qEE + qNN + qUU + qTT),
    pdop: safe(qEE + qNN + qUU),
    hdop: safe(qEE + qNN),
    vdop: safe(qUU),
    tdop: safe(qTT),
  };
  if (!Number.isFinite(dop.gdop)) return null;
  return dop;
}

// ---------------------------------------------------------------------------
// 6. Generic next-pass-over-a-site finder (cross-tie (a): EO next pass)
// ---------------------------------------------------------------------------

/**
 * Find the next satellite pass over a ground site.
 *
 * GENERIC: the caller injects `sampler(dateMs) -> {latDeg,lonDeg,altKm}`
 * (the parent wires it to the SGP4 propagator); this function performs no
 * propagation and reads no clock, so it is fully deterministic given the
 * sampler and options.
 *
 * SEARCH STRATEGY:
 *   1. Coarse forward scan from startMs to endMs in stepSec increments,
 *      evaluating elevation at each step.
 *   2. Detect a RISE as a below-mask -> at/above-mask transition between
 *      two consecutive coarse samples. (If includeInProgress and the sat
 *      is already above the mask at startMs, that in-progress pass is
 *      reported with rise = startMs.)
 *   3. Refine the rise instant by BISECTION between the bracketing coarse
 *      samples down to `refineSec` resolution.
 *   4. Continue the coarse scan through the pass, tracking the maximum
 *      elevation, until elevation drops back below the mask (the set),
 *      then bisect the set instant to `refineSec` as well.
 *
 * RESOLUTION: rise and set times are accurate to ~refineSec (default 1s).
 * maxElevationDeg and durationSec are reported at coarse stepSec sampling
 * of the pass interior — the true peak between two coarse samples is not
 * separately optimized (an honestly-stated coarse-resolution figure).
 *
 * If the pass has not set by endMs, setMs is clamped to endMs and
 * setWithinWindow is false.
 *
 * Returns null when no rise occurs within the search window.
 */
export function nextPassOverSite(
  sampler: PositionSampler,
  siteLatDeg: number,
  siteLonDeg: number,
  opts: NextPassOptions,
): PassResult | null {
  const startMs = opts.startMs;
  const stepSec = opts.stepSec ?? 30;
  const refineSec = opts.refineSec ?? 1;
  const mask = opts.minElevDeg ?? DEFAULT_MIN_ELEV_DEG;
  const windowSec = opts.windowSec ?? 24 * 3600;
  const endMs = opts.endMs ?? startMs + windowSec * 1000;
  const stepMs = stepSec * 1000;
  const refineMs = Math.max(1, refineSec * 1000);

  const elevAt = (ms: number): number => {
    const p = sampler(ms);
    return elevationAzimuthFromSite(
      p.latDeg,
      p.lonDeg,
      p.altKm,
      siteLatDeg,
      siteLonDeg,
    ).elevationDeg;
  };

  // Bisect a monotone-crossing bracket [belowMs (elev<mask), aboveMs
  // (elev>=mask)] to find the mask-crossing time to refineMs precision.
  const bisectCrossing = (belowMs: number, aboveMs: number): number => {
    let lo = belowMs;
    let hi = aboveMs;
    while (Math.abs(hi - lo) > refineMs) {
      const mid = (lo + hi) / 2;
      if (elevAt(mid) >= mask) hi = mid;
      else lo = mid;
    }
    return hi;
  };

  let prevMs = startMs;
  let prevElev = elevAt(startMs);

  // In-progress pass handling.
  let riseMs: number | null = null;
  if (prevElev >= mask) {
    if (opts.includeInProgress) {
      riseMs = startMs;
    } else {
      // Skip forward until it sets, so we only report a fresh rise.
      let t = startMs;
      while (t < endMs && elevAt(t) >= mask) {
        prevMs = t;
        prevElev = elevAt(t);
        t += stepMs;
      }
      // Loop continues below from prevMs (now below mask, or at end).
    }
  }

  if (riseMs === null) {
    // Scan for a below->above crossing.
    let t = prevMs + stepMs;
    for (; t <= endMs; t += stepMs) {
      const e = elevAt(t);
      if (prevElev < mask && e >= mask) {
        riseMs = bisectCrossing(prevMs, t);
        break;
      }
      prevMs = t;
      prevElev = e;
    }
    if (riseMs === null) return null; // no pass in window
  }

  // Walk the pass: track max elevation, find the set.
  let maxElev = elevAt(riseMs);
  let setMs = endMs;
  let setWithinWindow = false;
  let belowMs = riseMs;
  let t = riseMs + stepMs;
  for (; t <= endMs; t += stepMs) {
    const e = elevAt(t);
    if (e > maxElev) maxElev = e;
    if (e < mask) {
      // Crossed down between belowMs (still above the mask) and t (now
      // below): bisect the descending crossing for the set instant.
      setMs = bisectSetCrossing(belowMs, t, mask, refineMs, elevAt);
      setWithinWindow = true;
      break;
    }
    belowMs = t;
  }

  return {
    nextRiseMs: riseMs,
    setMs,
    maxElevationDeg: maxElev,
    durationSec: (setMs - riseMs) / 1000,
    setWithinWindow,
  };
}

/**
 * Bisect a descending crossing: [aboveMs (elev>=mask), belowMs
 * (elev<mask)] -> the set instant where elevation drops through the mask,
 * to refineMs precision.
 */
function bisectSetCrossing(
  aboveMs: number,
  belowMs: number,
  mask: number,
  refineMs: number,
  elevAt: (ms: number) => number,
): number {
  let lo = aboveMs; // elev >= mask
  let hi = belowMs; // elev < mask
  while (Math.abs(hi - lo) > refineMs) {
    const mid = (lo + hi) / 2;
    if (elevAt(mid) >= mask) lo = mid;
    else hi = mid;
  }
  return hi;
}
