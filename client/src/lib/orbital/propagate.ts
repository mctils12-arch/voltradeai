// SGP4 propagation — inline, zero-dependency implementation.
//
// DECISION (workstream (a), 2026-07-07): INLINE a near-earth SGP4 rather than
// depend on satellite.js. Rationale in the findings note; summary:
//   - Zero new dependency => this module + its tests are fully hermetic
//     (`npx tsx --test`, no install, no network) and add 0 KB to the bundle.
//     satellite.js@7.0.1 is 37 KB min / 14.2 KB gzip / 642 KB unpacked.
//   - The near-earth SGP4 kernel below is a faithful port of the standard
//     Vallado / "Revisiting Spacetrack Report #3" algorithm (WGS-72
//     constants), the same math satellite.js implements. It is validated in
//     propagate.test.ts against the canonical SGP4 verification object
//     (NORAD 88888) to sub-km ECI agreement.
//
// SCOPE / HONESTY: this covers NEAR-EARTH orbits (orbital period < 225 min:
// LEO — the overwhelming majority of the active catalog: ISS, Starlink,
// OneWeb, Iridium, imaging sats). DEEP-SPACE objects (period >= 225 min:
// GEO comms, GPS/GLONASS/Galileo MEO, Molniya) require the SDP4 deep-space
// corrections, which are NOT implemented here. For those, `propagate()`
// returns null (never a faked position) and `isDeepSpace()` returns true —
// the caller surfaces the count. If full-catalog deep-space is needed, the
// integrating session can `npm i satellite.js` and route deep-space objects
// to it; see the findings note.
//
// Frames: SGP4 produces TEME position/velocity; positions are converted to
// geodetic lat/lon/alt via GMST (IAU-82) and a WGS-84 iterative reduction.

import type { GpRecord } from './tle.js';

// --- WGS-72 gravity model constants (SGP4 standard) ---
const PI = Math.PI;
const TWO_PI = 2 * PI;
const DEG2RAD = PI / 180;
const RAD2DEG = 180 / PI;
const MINUTES_PER_DAY = 1440;
// rev/day -> rad/min : n[rad/min] = n[rev/day] * 2π / 1440
const XPDOTP = MINUTES_PER_DAY / TWO_PI;

const MU = 398600.8; // km^3 / s^2
const RADIUS_EARTH_KM = 6378.135;
const XKE = 60.0 / Math.sqrt((RADIUS_EARTH_KM * RADIUS_EARTH_KM * RADIUS_EARTH_KM) / MU);
const TUMIN = 1.0 / XKE;
const J2 = 0.001082616;
const J3 = -0.00000253881;
const J4 = -0.00000165597;
const J3OJ2 = J3 / J2;
const X2O3 = 2.0 / 3.0;

// Deep-space threshold: SGP4 uses SDP4 when the orbital period >= 225 min.
const DEEPSPACE_PERIOD_MIN = 225.0;

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------

/** Julian date from a UTC calendar instant (Vallado jday). */
function jday(
  year: number,
  mon: number,
  day: number,
  hr: number,
  minute: number,
  sec: number,
): number {
  return (
    367.0 * year -
    Math.floor(7 * (year + Math.floor((mon + 9) / 12.0)) * 0.25) +
    Math.floor((275 * mon) / 9.0) +
    day +
    1721013.5 +
    ((sec / 60.0 + minute) / 60.0 + hr) / 24.0
  );
}

/** Julian date from an epoch-ms value (treated as UTC). */
function jdayFromMs(ms: number): number {
  const d = new Date(ms);
  return jday(
    d.getUTCFullYear(),
    d.getUTCMonth() + 1,
    d.getUTCDate(),
    d.getUTCHours(),
    d.getUTCMinutes(),
    d.getUTCSeconds() + d.getUTCMilliseconds() / 1000,
  );
}

/**
 * Parse an OMM EPOCH string to epoch-ms. CelesTrak epochs are UTC but carry no
 * timezone designator (e.g. "2026-07-07T12:12:01.987776"); JS would otherwise
 * read them as LOCAL time, so we force UTC.
 */
function epochToMs(epoch: string | null): number | null {
  if (!epoch) return null;
  let s = epoch.trim();
  if (s === '') return null;
  const hasTz = /([zZ]|[+\-]\d{2}:?\d{2})$/.test(s);
  if (!hasTz) s += 'Z';
  const ms = Date.parse(s);
  return Number.isFinite(ms) ? ms : null;
}

/** Greenwich Mean Sidereal Time (radians), IAU-82, from a UT1≈UTC Julian date. */
function gstime(jdut1: number): number {
  const tut1 = (jdut1 - 2451545.0) / 36525.0;
  let temp =
    -6.2e-6 * tut1 * tut1 * tut1 +
    0.093104 * tut1 * tut1 +
    (876600.0 * 3600 + 8640184.812866) * tut1 +
    67310.54841;
  temp = ((temp * DEG2RAD) / 240.0) % TWO_PI;
  if (temp < 0) temp += TWO_PI;
  return temp;
}

// ---------------------------------------------------------------------------
// SGP4 kernel (near-earth)
// ---------------------------------------------------------------------------

interface Satrec {
  error: number;
  deepSpace: boolean;
  // orbital elements at epoch (radians / rad-per-min)
  inclo: number;
  nodeo: number;
  argpo: number;
  mo: number;
  ecco: number;
  bstar: number;
  no: number; // mean motion (rad/min), kozai-corrected
  // derived secular / periodic coefficients
  con41: number;
  isimp: number;
  aycof: number;
  xlcof: number;
  cc1: number;
  cc4: number;
  cc5: number;
  eta: number;
  x1mth2: number;
  x7thm1: number;
  omgcof: number;
  xmcof: number;
  nodecf: number;
  mdot: number;
  argpdot: number;
  nodedot: number;
  t2cof: number;
  d2: number;
  d3: number;
  d4: number;
  t3cof: number;
  t4cof: number;
  t5cof: number;
  delmo: number;
  sinmao: number;
}

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** initl — common initialization (Vallado). `epochDays` = days since 1949-12-31. */
function initl(ecco: number, inclo: number, noKozai: number) {
  const eccsq = ecco * ecco;
  const omeosq = 1.0 - eccsq;
  const rteosq = Math.sqrt(omeosq);
  const cosio = Math.cos(inclo);
  const cosio2 = cosio * cosio;

  const ak = Math.pow(XKE / noKozai, X2O3);
  const d1 = (0.75 * J2 * (3.0 * cosio2 - 1.0)) / (rteosq * omeosq);
  let del = d1 / (ak * ak);
  const adel = ak * (1.0 - del * del - del * (1.0 / 3.0 + (134.0 * del * del) / 81.0));
  del = d1 / (adel * adel);
  const no = noKozai / (1.0 + del);

  const ao = Math.pow(XKE / no, X2O3);
  const sinio = Math.sin(inclo);
  const po = ao * omeosq;
  const con42 = 1.0 - 5.0 * cosio2;
  const con41 = -con42 - cosio2 - cosio2;
  const posq = po * po;
  const rp = ao * (1.0 - ecco);

  return { no, ao, con41, con42, cosio, cosio2, omeosq, posq, rp, rteosq, sinio };
}

/** sgp4init — build a Satrec from normalized elements. */
function sgp4init(
  epochMs: number,
  gp: {
    inclo: number; // rad
    nodeo: number; // rad
    argpo: number; // rad
    mo: number; // rad
    ecco: number;
    noKozai: number; // rad/min
    bstar: number;
  },
): Satrec {
  const s: Satrec = {
    error: 0,
    deepSpace: false,
    inclo: gp.inclo,
    nodeo: gp.nodeo,
    argpo: gp.argpo,
    mo: gp.mo,
    ecco: gp.ecco,
    bstar: gp.bstar,
    no: gp.noKozai,
    con41: 0,
    isimp: 0,
    aycof: 0,
    xlcof: 0,
    cc1: 0,
    cc4: 0,
    cc5: 0,
    eta: 0,
    x1mth2: 0,
    x7thm1: 0,
    omgcof: 0,
    xmcof: 0,
    nodecf: 0,
    mdot: 0,
    argpdot: 0,
    nodedot: 0,
    t2cof: 0,
    d2: 0,
    d3: 0,
    d4: 0,
    t3cof: 0,
    t4cof: 0,
    t5cof: 0,
    delmo: 0,
    sinmao: 0,
  };

  const ss = 78.0 / RADIUS_EARTH_KM + 1.0;
  const qzms2ttemp = (120.0 - 78.0) / RADIUS_EARTH_KM;
  const qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;

  const init = initl(gp.ecco, gp.inclo, gp.noKozai);
  s.no = init.no;
  s.con41 = init.con41;

  const { no, ao, con41, con42, cosio, cosio2, omeosq, posq, rp, rteosq, sinio } = init;

  // deep-space objects are out of scope for this near-earth kernel
  if (TWO_PI / no >= DEEPSPACE_PERIOD_MIN) {
    s.deepSpace = true;
    return s;
  }

  if (omeosq >= 0.0 || no >= 0.0) {
    s.isimp = 0;
    if (rp < 220.0 / RADIUS_EARTH_KM + 1.0) s.isimp = 1;

    let sfour = ss;
    let qzms24 = qzms2t;
    const perige = (rp - 1.0) * RADIUS_EARTH_KM;

    if (perige < 156.0) {
      sfour = perige - 78.0;
      if (perige < 98.0) sfour = 20.0;
      const qzms24temp = (120.0 - sfour) / RADIUS_EARTH_KM;
      qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
      sfour = sfour / RADIUS_EARTH_KM + 1.0;
    }

    const pinvsq = 1.0 / posq;
    const tsi = 1.0 / (ao - sfour);
    s.eta = ao * s.ecco * tsi;
    const etasq = s.eta * s.eta;
    const eeta = s.ecco * s.eta;
    const psisq = Math.abs(1.0 - etasq);
    const coef = qzms24 * Math.pow(tsi, 4.0);
    const coef1 = coef / Math.pow(psisq, 3.5);
    const cc2 =
      coef1 *
      s.no *
      (ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq)) +
        ((0.375 * J2 * tsi) / psisq) * con41 * (8.0 + 3.0 * etasq * (8.0 + etasq)));
    s.cc1 = s.bstar * cc2;

    let cc3 = 0.0;
    if (s.ecco > 1e-4) {
      cc3 = (-2.0 * coef * tsi * J3OJ2 * s.no * sinio) / s.ecco;
    }
    s.x1mth2 = 1.0 - cosio2;
    s.cc4 =
      2.0 *
      s.no *
      coef1 *
      ao *
      omeosq *
      (s.eta * (2.0 + 0.5 * etasq) +
        s.ecco * (0.5 + 2.0 * etasq) -
        ((J2 * tsi) / (ao * psisq)) *
          (-3.0 * con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta)) +
            0.75 * s.x1mth2 * (2.0 * etasq - eeta * (1.0 + etasq)) * Math.cos(2.0 * s.argpo)));
    s.cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq);

    const cosio4 = cosio2 * cosio2;
    const temp1 = 1.5 * J2 * pinvsq * s.no;
    const temp2 = 0.5 * temp1 * J2 * pinvsq;
    const temp3 = -0.46875 * J4 * pinvsq * pinvsq * s.no;
    s.mdot =
      s.no + 0.5 * temp1 * rteosq * con41 + 0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
    s.argpdot =
      -0.5 * temp1 * con42 +
      0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
      temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4);
    const xhdot1 = -temp1 * cosio;
    s.nodedot =
      xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) + 2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio;
    s.omgcof = s.bstar * cc3 * Math.cos(s.argpo);
    s.xmcof = 0.0;
    if (s.ecco > 1e-4) s.xmcof = (-X2O3 * coef * s.bstar) / eeta;
    s.nodecf = 3.5 * omeosq * xhdot1 * s.cc1;
    s.t2cof = 1.5 * s.cc1;

    if (Math.abs(cosio + 1.0) > 1.5e-12) {
      s.xlcof = (-0.25 * J3OJ2 * sinio * (3.0 + 5.0 * cosio)) / (1.0 + cosio);
    } else {
      s.xlcof = (-0.25 * J3OJ2 * sinio * (3.0 + 5.0 * cosio)) / 1.5e-12;
    }
    s.aycof = -0.5 * J3OJ2 * sinio;

    const delmotemp = 1.0 + s.eta * Math.cos(s.mo);
    s.delmo = delmotemp * delmotemp * delmotemp;
    s.sinmao = Math.sin(s.mo);
    s.x7thm1 = 7.0 * cosio2 - 1.0;

    if (s.isimp !== 1) {
      const cc1sq = s.cc1 * s.cc1;
      s.d2 = 4.0 * ao * tsi * cc1sq;
      const temp = (s.d2 * tsi * s.cc1) / 3.0;
      s.d3 = (17.0 * ao + sfour) * temp;
      s.d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * s.cc1;
      s.t3cof = s.d2 + 2.0 * cc1sq;
      s.t4cof = 0.25 * (3.0 * s.d3 + s.cc1 * (12.0 * s.d2 + 10.0 * cc1sq));
      s.t5cof =
        0.2 *
        (3.0 * s.d4 + 12.0 * s.cc1 * s.d3 + 6.0 * s.d2 * s.d2 + 15.0 * cc1sq * (2.0 * s.d2 + cc1sq));
    }
  }

  return s;
}

/** sgp4 — propagate a near-earth Satrec to tsince minutes past epoch (TEME, km, km/s). */
function sgp4(s: Satrec, tsince: number): { position: Vec3; velocity: Vec3 } | null {
  if (s.deepSpace) return null;

  const xmdf = s.mo + s.mdot * tsince;
  const argpdf = s.argpo + s.argpdot * tsince;
  const nodedf = s.nodeo + s.nodedot * tsince;
  let argpm = argpdf;
  let mm = xmdf;
  const t2 = tsince * tsince;
  const nodem = nodedf + s.nodecf * t2;
  let tempa = 1.0 - s.cc1 * tsince;
  let tempe = s.bstar * s.cc4 * tsince;
  let templ = s.t2cof * t2;

  if (s.isimp !== 1) {
    const delomg = s.omgcof * tsince;
    const delmtemp = 1.0 + s.eta * Math.cos(xmdf);
    const delm = s.xmcof * (delmtemp * delmtemp * delmtemp - s.delmo);
    const temp = delomg + delm;
    mm = xmdf + temp;
    argpm = argpdf - temp;
    const t3 = t2 * tsince;
    const t4 = t3 * tsince;
    tempa = tempa - s.d2 * t2 - s.d3 * t3 - s.d4 * t4;
    tempe = tempe + s.bstar * s.cc5 * (Math.sin(mm) - s.sinmao);
    templ = templ + s.t3cof * t3 + t4 * (s.t4cof + tsince * s.t5cof);
  }

  let nm = s.no;
  let em = s.ecco;
  const inclm = s.inclo;

  if (nm <= 0.0) {
    s.error = 2;
    return null;
  }

  const am = Math.pow(XKE / nm, X2O3) * tempa * tempa;
  nm = XKE / Math.pow(am, 1.5);
  em -= tempe;
  if (em >= 1.0 || em < -0.001) {
    s.error = 1;
    return null;
  }
  if (em < 1e-6) em = 1e-6;
  mm += s.no * templ;
  let xlm = mm + argpm + nodem;

  const nodemMod = nodem % TWO_PI;
  argpm %= TWO_PI;
  xlm %= TWO_PI;
  mm = (xlm - argpm - nodemMod) % TWO_PI;

  const sinim = Math.sin(inclm);
  const cosim = Math.cos(inclm);

  const ep = em;
  const xincp = inclm;
  const argpp = argpm;
  const nodep = nodemMod;
  const mp = mm;
  const sinip = sinim;
  const cosip = cosim;

  // long-period periodics
  const axnl = ep * Math.cos(argpp);
  let temp = 1.0 / (am * (1.0 - ep * ep));
  const aynl = ep * Math.sin(argpp) + temp * s.aycof;
  const xl = mp + argpp + nodep + temp * s.xlcof * axnl;

  // Kepler's equation
  const u = (xl - nodep) % TWO_PI;
  let eo1 = u;
  let tem5 = 9999.9;
  let ktr = 1;
  let sineo1 = 0;
  let coseo1 = 0;
  while (Math.abs(tem5) >= 1e-12 && ktr <= 10) {
    sineo1 = Math.sin(eo1);
    coseo1 = Math.cos(eo1);
    tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
    tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
    if (Math.abs(tem5) >= 0.95) tem5 = tem5 > 0.0 ? 0.95 : -0.95;
    eo1 += tem5;
    ktr += 1;
  }

  // short-period preliminary quantities
  const ecose = axnl * coseo1 + aynl * sineo1;
  const esine = axnl * sineo1 - aynl * coseo1;
  const el2 = axnl * axnl + aynl * aynl;
  const pl = am * (1.0 - el2);
  if (pl < 0.0) {
    s.error = 4;
    return null;
  }

  const rl = am * (1.0 - ecose);
  const rdotl = (Math.sqrt(am) * esine) / rl;
  const rvdotl = Math.sqrt(pl) / rl;
  const betal = Math.sqrt(1.0 - el2);
  temp = esine / (1.0 + betal);
  const sinu = (am / rl) * (sineo1 - aynl - axnl * temp);
  const cosu = (am / rl) * (coseo1 - axnl + aynl * temp);
  let su = Math.atan2(sinu, cosu);
  const sin2u = (cosu + cosu) * sinu;
  const cos2u = 1.0 - 2.0 * sinu * sinu;
  temp = 1.0 / pl;
  const temp1 = 0.5 * J2 * temp;
  const temp2 = temp1 * temp;

  const con41 = s.con41; // near-earth: inclination unchanged, so this holds
  const x1mth2 = s.x1mth2;
  const x7thm1 = s.x7thm1;

  const mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + 0.5 * temp1 * x1mth2 * cos2u;
  su -= 0.25 * temp2 * x7thm1 * sin2u;
  const xnode = nodep + 1.5 * temp2 * cosip * sin2u;
  const xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
  const mvt = rdotl - (nm * temp1 * x1mth2 * sin2u) / XKE;
  const rvdot = rvdotl + (nm * temp1 * (x1mth2 * cos2u + 1.5 * con41)) / XKE;

  // orientation vectors
  const sinsu = Math.sin(su);
  const cossu = Math.cos(su);
  const snod = Math.sin(xnode);
  const cnod = Math.cos(xnode);
  const sini = Math.sin(xinc);
  const cosi = Math.cos(xinc);
  const xmx = -snod * cosi;
  const xmy = cnod * cosi;
  const ux = xmx * sinsu + cnod * cossu;
  const uy = xmy * sinsu + snod * cossu;
  const uz = sini * sinsu;
  const vx = xmx * cossu - cnod * sinsu;
  const vy = xmy * cossu - snod * sinsu;
  const vz = sini * cossu;

  const position: Vec3 = {
    x: mrt * ux * RADIUS_EARTH_KM,
    y: mrt * uy * RADIUS_EARTH_KM,
    z: mrt * uz * RADIUS_EARTH_KM,
  };
  const vkmpersec = (RADIUS_EARTH_KM * XKE) / 60.0;
  const velocity: Vec3 = {
    x: (mvt * ux + rvdot * vx) * vkmpersec,
    y: (mvt * uy + rvdot * vy) * vkmpersec,
    z: (mvt * uz + rvdot * vz) * vkmpersec,
  };

  return { position, velocity };
}

/** TEME ECI position (km) -> geodetic lat/lon/alt, given GMST (rad). */
function eciToGeodetic(pos: Vec3, gmst: number): { latDeg: number; lonDeg: number; altKm: number } {
  const a = 6378.137; // WGS-84 semi-major (km)
  const b = 6356.7523142; // WGS-84 semi-minor (km)
  const R = Math.sqrt(pos.x * pos.x + pos.y * pos.y);
  const f = (a - b) / a;
  const e2 = 2.0 * f - f * f;

  let lon = Math.atan2(pos.y, pos.x) - gmst;
  while (lon < -PI) lon += TWO_PI;
  while (lon > PI) lon -= TWO_PI;

  let lat = Math.atan2(pos.z, R);
  let C = 1.0;
  for (let i = 0; i < 20; i++) {
    const sinLat = Math.sin(lat);
    C = 1.0 / Math.sqrt(1.0 - e2 * sinLat * sinLat);
    lat = Math.atan2(pos.z + a * C * e2 * sinLat, R);
  }
  const altKm = R / Math.cos(lat) - a * C;

  return { latDeg: lat * RAD2DEG, lonDeg: lon * RAD2DEG, altKm };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** True if the element set is deep-space (period >= 225 min) — out of scope here. */
export function isDeepSpace(gp: GpRecord): boolean {
  if (gp.meanMotion == null || gp.meanMotion <= 0) return false;
  const noKozai = gp.meanMotion / XPDOTP; // rad/min
  return TWO_PI / noKozai >= DEEPSPACE_PERIOD_MIN;
}

/**
 * Propagate a GP element record to `dateMs` (Unix ms, UTC). Returns geodetic
 * position, or null when: elements are incomplete, the object is deep-space
 * (needs SDP4 — see module header), the epoch is unparseable, or SGP4 flags a
 * decayed/degenerate orbit. NEVER returns a faked/extrapolated-beyond-validity
 * position.
 */
export function propagate(
  gp: GpRecord,
  dateMs: number,
): { latDeg: number; lonDeg: number; altKm: number } | null {
  if (
    gp.meanMotion == null ||
    gp.ecc == null ||
    gp.inclination == null ||
    gp.raan == null ||
    gp.argp == null ||
    gp.meanAnomaly == null
  ) {
    return null;
  }
  const epochMs = epochToMs(gp.epoch);
  if (epochMs == null) return null;

  const noKozai = gp.meanMotion / XPDOTP; // rad/min
  if (!(noKozai > 0)) return null;

  const s = sgp4init(epochMs, {
    inclo: gp.inclination * DEG2RAD,
    nodeo: gp.raan * DEG2RAD,
    argpo: gp.argp * DEG2RAD,
    mo: gp.meanAnomaly * DEG2RAD,
    ecco: gp.ecc,
    noKozai,
    bstar: gp.bstar ?? 0,
  });
  if (s.deepSpace) return null;

  const tsince = (dateMs - epochMs) / 60000.0; // minutes
  const rv = sgp4(s, tsince);
  if (rv == null) return null;

  const gmst = gstime(jdayFromMs(dateMs));
  const geo = eciToGeodetic(rv.position, gmst);
  if (!Number.isFinite(geo.latDeg) || !Number.isFinite(geo.lonDeg) || !Number.isFinite(geo.altKm)) {
    return null;
  }
  return geo;
}

/**
 * Instantaneous orbit-class label from a single geodetic altitude.
 * Boundaries per the ORBITAL charter: LEO < 2000 km, MEO < 35000 km, else GEO
 * (geostationary ~35786 km). This is the altitude-based classifier used by the
 * live-position layer; SATCAT's apogee/perigee classifier (classifyOrbit in
 * tle.ts) is the metadata-based one — they answer different questions.
 */
export function orbitClassFromAltKm(altKm: number): 'LEO' | 'MEO' | 'GEO' {
  if (altKm < 2000) return 'LEO';
  if (altKm < 35000) return 'MEO';
  return 'GEO';
}

/**
 * Age of a TLE/OMM epoch in days at `nowMs`. Orbit uncertainty grows with
 * epoch age; the caller flags "stale orbit" past a threshold. Returns null for
 * an absent/unparseable epoch. `nowMs` is passed in (never wall-clock) so
 * callers and tests stay deterministic.
 */
export function epochAgeDays(epoch: string | null, nowMs: number): number | null {
  const epochMs = epochToMs(epoch);
  if (epochMs == null) return null;
  return (nowMs - epochMs) / 86400000.0;
}

// Internal kernel exposed for hermetic ECI-level verification (see tests).
export const _sgp4Internal = {
  jday,
  jdayFromMs,
  epochToMs,
  gstime,
  eciToGeodetic,
  sgp4init,
  sgp4,
  XPDOTP,
  DEG2RAD,
};
