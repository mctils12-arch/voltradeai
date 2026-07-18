// Orbit-derived display values for the satellite detail card (satellite-UX
// directive 2026-07-18 §1: compact chip row ALT · SPEED · INCL · PERIOD plus
// APOGEE/PERIGEE in the Details expander).
//
// HONESTY: these are standard two-body derivations over the CATALOGUED GP
// elements (CelesTrak OMM) — semi-major axis from mean motion, apogee/perigee
// from a·(1±e), speed from vis-viva at the current radius. Derived from real
// catalogued elements, never measured downlink values; the card's provenance
// line states the derivation. Display conversion happens in the caller via
// lib/units.ts formatters (units directive 2026-07-13).

/** GM of Earth, km^3/s^2 (WGS-84). */
export const EARTH_MU_KM3_S2 = 398600.4418;
/** Earth equatorial radius, km (WGS-84) — the SGP4 reference radius. */
export const EARTH_RADIUS_KM = 6378.137;

/** Semi-major axis (km) from mean motion in rev/day; null when unusable. */
export function semiMajorAxisKm(meanMotionRevDay: number | null | undefined): number | null {
  if (meanMotionRevDay == null || !isFinite(meanMotionRevDay) || meanMotionRevDay <= 0) return null;
  const nRadS = (meanMotionRevDay * 2 * Math.PI) / 86400;
  return Math.cbrt(EARTH_MU_KM3_S2 / (nRadS * nRadS));
}

/** Apogee/perigee ALTITUDES (km above the equatorial radius) from mean
 *  motion + eccentricity; null when either element is missing/degenerate. */
export function apsidesKm(
  meanMotionRevDay: number | null | undefined,
  ecc: number | null | undefined,
): { apogeeKm: number; perigeeKm: number } | null {
  const a = semiMajorAxisKm(meanMotionRevDay);
  if (a == null || ecc == null || !isFinite(ecc) || ecc < 0 || ecc >= 1) return null;
  return {
    apogeeKm: a * (1 + ecc) - EARTH_RADIUS_KM,
    perigeeKm: a * (1 - ecc) - EARTH_RADIUS_KM,
  };
}

/** Orbital speed in km/h — vis-viva at the live altitude when one exists
 *  this tick, else the circular-orbit speed at the semi-major axis. */
export function orbitalSpeedKmh(
  meanMotionRevDay: number | null | undefined,
  altKm: number | null | undefined,
): number | null {
  const a = semiMajorAxisKm(meanMotionRevDay);
  if (a == null) return null;
  const r = altKm != null && isFinite(altKm) && altKm > -EARTH_RADIUS_KM ? EARTH_RADIUS_KM + altKm : a;
  const v2 = EARTH_MU_KM3_S2 * (2 / r - 1 / a);
  if (!(v2 > 0)) return null;
  return Math.sqrt(v2) * 3600;
}

/** Orbital period in minutes from mean motion in rev/day. */
export function periodMinutes(meanMotionRevDay: number | null | undefined): number | null {
  if (meanMotionRevDay == null || !isFinite(meanMotionRevDay) || meanMotionRevDay <= 0) return null;
  return 1440 / meanMotionRevDay;
}
