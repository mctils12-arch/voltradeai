// Celestial ephemeris — EARTH TWIN O6-7 tier 1 (charter: "the sun's real
// position drives … the day/night terminator; the moon drawn at its real
// direction + real phase — all labeled 'computed ephemeris'").
//
// Pure computation: standard low-precision solar/lunar formulas (Meeus,
// "Astronomical Algorithms" — truncated series). No feed, no license, no
// network. HONEST ACCURACY: subsolar point good to ~0.01°, sublunar point
// to ~1° (truncated lunar series), phase fraction to a few percent —
// display-grade, labeled as such; never presented as survey ephemeris.

const DEG = Math.PI / 180;
const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0); // 2000-01-01 12:00 TT≈UTC (display grade)

const norm360 = (d: number): number => ((d % 360) + 360) % 360;
const normLon = (d: number): number => {
  const n = norm360(d);
  return n > 180 ? n - 360 : n;
};

export interface SubPoint {
  latDeg: number;
  lonDeg: number;
}

interface Ecliptic {
  lambdaDeg: number; // ecliptic longitude
  betaDeg: number;   // ecliptic latitude
}

/** Greenwich mean sidereal time, degrees. */
export function gmstDeg(dateMs: number): number {
  const d = (dateMs - J2000_MS) / 86_400_000;
  return norm360(280.46061837 + 360.98564736629 * d);
}

function eclipticToSubpoint(e: Ecliptic, dateMs: number): SubPoint {
  const d = (dateMs - J2000_MS) / 86_400_000;
  const eps = (23.439 - 0.0000004 * d) * DEG;
  const lam = e.lambdaDeg * DEG;
  const bet = e.betaDeg * DEG;
  const sinDec = Math.sin(bet) * Math.cos(eps) + Math.cos(bet) * Math.sin(eps) * Math.sin(lam);
  const dec = Math.asin(Math.max(-1, Math.min(1, sinDec)));
  const y = Math.sin(lam) * Math.cos(eps) - Math.tan(bet) * Math.sin(eps);
  const ra = Math.atan2(y, Math.cos(lam));
  return { latDeg: dec / DEG, lonDeg: normLon(ra / DEG - gmstDeg(dateMs)) };
}

function sunEclipticLongitudeDeg(dateMs: number): number {
  const d = (dateMs - J2000_MS) / 86_400_000;
  const L = 280.460 + 0.9856474 * d;             // mean longitude
  const g = (357.528 + 0.9856003 * d) * DEG;     // mean anomaly
  return norm360(L + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g));
}

/** The point on Earth where the sun is directly overhead right now. */
export function subsolarPoint(dateMs: number): SubPoint {
  return eclipticToSubpoint({ lambdaDeg: sunEclipticLongitudeDeg(dateMs), betaDeg: 0 }, dateMs);
}

export interface MoonState extends SubPoint {
  /** illuminated fraction 0..1 (elongation-based — display grade). */
  illuminatedFraction: number;
  /** true angular diameter, arcminutes (from the instantaneous distance). */
  angularSizeArcmin: number;
  /** waxing (true) / waning (false) — picks the phase glyph's side. */
  waxing: boolean;
  distanceKm: number;
}

/** The point on Earth where the moon is directly overhead + real phase. */
export function moonState(dateMs: number): MoonState {
  const d = (dateMs - J2000_MS) / 86_400_000;
  const Lp = 218.316 + 13.176396 * d;            // mean longitude
  const Mp = (134.963 + 13.064993 * d) * DEG;    // mean anomaly
  const F = (93.272 + 13.229350 * d) * DEG;      // argument of latitude
  const lambda = norm360(Lp + 6.289 * Math.sin(Mp));
  const beta = 5.128 * Math.sin(F);
  const distanceKm = 385001 - 20905 * Math.cos(Mp);
  const sp = eclipticToSubpoint({ lambdaDeg: lambda, betaDeg: beta }, dateMs);
  const elongation = norm360(lambda - sunEclipticLongitudeDeg(dateMs));
  const illuminatedFraction = (1 - Math.cos(elongation * DEG)) / 2;
  const angularSizeArcmin = (2 * Math.asin(1737.4 / distanceKm)) / DEG * 60;
  return { ...sp, illuminatedFraction, angularSizeArcmin, waxing: elongation < 180, distanceKm };
}

/** Phase glyph from illumination + waxing side (the 8 canonical steps). */
export function moonPhaseGlyph(illum: number, waxing: boolean): string {
  if (illum < 0.04) return '🌑';
  if (illum > 0.96) return '🌕';
  if (illum < 0.35) return waxing ? '🌒' : '🌘';
  if (illum < 0.65) return waxing ? '🌓' : '🌗';
  return waxing ? '🌔' : '🌖';
}

/**
 * The night hemisphere as a GeoJSON polygon (the classic terminator layer):
 * for each longitude, the terminator latitude satisfies
 * tan(latT) = -cos(lon - subsolarLon) / tan(subsolarLat); the polygon closes
 * over whichever pole is in darkness. Clamped to ±85° (mercator limit) —
 * an honest display constraint, stated in the layer description.
 */
export function nightPolygon(dateMs: number, steps = 180): GeoJSON.Feature {
  const sun = subsolarPoint(dateMs);
  const decRad = sun.latDeg * DEG;
  const ring: [number, number][] = [];
  const darkPoleLat = sun.latDeg >= 0 ? -85 : 85;
  for (let i = 0; i <= steps; i++) {
    const lon = -180 + (360 * i) / steps;
    const H = (lon - sun.lonDeg) * DEG;
    let latT: number;
    if (Math.abs(decRad) < 1e-6) {
      latT = 0; // equinox: terminator through the poles — flat at 0 is the limit form
    } else {
      latT = Math.atan(-Math.cos(H) / Math.tan(decRad)) / DEG;
    }
    ring.push([lon, Math.max(-85, Math.min(85, latT))]);
  }
  // close over the dark pole
  ring.push([180, darkPoleLat], [-180, darkPoleLat], ring[0]);
  return {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [ring] },
  };
}
