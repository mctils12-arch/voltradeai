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
// SCOPE / HONESTY: NEAR-EARTH orbits (orbital period < 225 min: LEO — ISS,
// Starlink, OneWeb, Iridium, imaging sats) use the SGP4 kernel. DEEP-SPACE
// objects (period >= 225 min: GEO comms, GPS/GLONASS/Galileo/BeiDou MEO,
// Molniya) use the SDP4 deep-space corrections — dscom / dpper / dsinit /
// dspace below, ported from the same Vallado reference and wired in exactly
// where the reference does (method 'd': lunar-solar secular + periodic
// perturbations, 12h/24h resonance integration). `isDeepSpace()` still
// labels them for callers. Error paths (decayed orbit, perturbed e out of
// range, pl < 0) return null — never a faked position.
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
  method: 'n' | 'd';
  gsto: number; // GMST at epoch (rad) — deep-space resonance time base
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
  // --- deep-space (SDP4) state — populated when method === 'd' ---
  // dscom lunar-solar periodic coefficients (dpper inputs)
  e3: number; ee2: number;
  peo: number; pgho: number; pho: number; pinco: number; plo: number;
  se2: number; se3: number;
  sgh2: number; sgh3: number; sgh4: number;
  sh2: number; sh3: number;
  si2: number; si3: number;
  sl2: number; sl3: number; sl4: number;
  xgh2: number; xgh3: number; xgh4: number;
  xh2: number; xh3: number;
  xi2: number; xi3: number;
  xl2: number; xl3: number; xl4: number;
  zmol: number; zmos: number;
  // dsinit secular rates + resonance terms (dspace inputs)
  irez: number; // 0 none, 1 synchronous (24h), 2 half-day (12h)
  d2201: number; d2211: number; d3210: number; d3222: number;
  d4410: number; d4422: number;
  d5220: number; d5232: number; d5421: number; d5433: number;
  dedt: number; didt: number; dmdt: number; dnodt: number; domdt: number;
  del1: number; del2: number; del3: number;
  xfact: number; xlamo: number; xli: number; xni: number; atime: number;
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

// ---------------------------------------------------------------------------
// SDP4 deep-space routines (Vallado dscom / dpper / dsinit / dspace)
// ---------------------------------------------------------------------------

/**
 * dscom — deep-space common variables: lunar-solar geometry at epoch and the
 * periodic coefficient seeds. `epoch` = days since 1949 Dec 31 00:00 UT.
 */
function dscom(
  epoch: number,
  ep: number,
  argpp: number,
  tc: number,
  inclp: number,
  nodep: number,
  np: number,
) {
  const zes = 0.01675;
  const zel = 0.0549;
  const c1ss = 2.9864797e-6;
  const c1l = 4.7968065e-7;
  const zsinis = 0.39785416;
  const zcosis = 0.91744867;
  const zcosgs = 0.1945905;
  const zsings = -0.98088458;

  const nm = np;
  const em = ep;
  const snodm = Math.sin(nodep);
  const cnodm = Math.cos(nodep);
  const sinomm = Math.sin(argpp);
  const cosomm = Math.cos(argpp);
  const sinim = Math.sin(inclp);
  const cosim = Math.cos(inclp);
  const emsq = em * em;
  const betasq = 1.0 - emsq;
  const rtemsq = Math.sqrt(betasq);

  // lunar-solar orientation at epoch
  const day = epoch + 18261.5 + tc / 1440.0;
  const xnodce = (4.523602 - 9.2422029e-4 * day) % TWO_PI;
  const stem = Math.sin(xnodce);
  const ctem = Math.cos(xnodce);
  const zcosil = 0.91375164 - 0.03568096 * ctem;
  const zsinil = Math.sqrt(1.0 - zcosil * zcosil);
  const zsinhl = (0.089683511 * stem) / zsinil;
  const zcoshl = Math.sqrt(1.0 - zsinhl * zsinhl);
  const gam = 5.8351514 + 0.001944368 * day;
  let zx = (0.39785416 * stem) / zsinil;
  const zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
  zx = Math.atan2(zx, zy);
  zx = gam + zx - xnodce;
  const zcosgl = Math.cos(zx);
  const zsingl = Math.sin(zx);

  // first pass: solar terms; second pass: lunar terms
  let zcosg = zcosgs;
  let zsing = zsings;
  let zcosi = zcosis;
  let zsini = zsinis;
  let zcosh = cnodm;
  let zsinh = snodm;
  let cc = c1ss;
  const xnoi = 1.0 / nm;

  let s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  let ss1 = 0, ss2 = 0, ss3 = 0, ss4 = 0, ss5 = 0, ss6 = 0, ss7 = 0;
  let sz1 = 0, sz2 = 0, sz3 = 0;
  let sz11 = 0, sz12 = 0, sz13 = 0, sz21 = 0, sz22 = 0, sz23 = 0, sz31 = 0, sz32 = 0, sz33 = 0;
  let z1 = 0, z2 = 0, z3 = 0;
  let z11 = 0, z12 = 0, z13 = 0, z21 = 0, z22 = 0, z23 = 0, z31 = 0, z32 = 0, z33 = 0;

  for (let lsflg = 1; lsflg <= 2; lsflg++) {
    const a1 = zcosg * zcosh + zsing * zcosi * zsinh;
    const a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
    const a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
    const a8 = zsing * zsini;
    const a9 = zsing * zsinh + zcosg * zcosi * zcosh;
    const a10 = zcosg * zsini;
    const a2 = cosim * a7 + sinim * a8;
    const a4 = cosim * a9 + sinim * a10;
    const a5 = -sinim * a7 + cosim * a8;
    const a6 = -sinim * a9 + cosim * a10;

    const x1 = a1 * cosomm + a2 * sinomm;
    const x2 = a3 * cosomm + a4 * sinomm;
    const x3 = -a1 * sinomm + a2 * cosomm;
    const x4 = -a3 * sinomm + a4 * cosomm;
    const x5 = a5 * sinomm;
    const x6 = a6 * sinomm;
    const x7 = a5 * cosomm;
    const x8 = a6 * cosomm;

    z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
    z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
    z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
    z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq;
    z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq;
    z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq;
    z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5);
    z12 =
      -6.0 * (a1 * a6 + a3 * a5) +
      emsq * (-24.0 * (x2 * x7 + x1 * x8) + -6.0 * (x3 * x6 + x4 * x5));
    z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
    z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
    z22 =
      6.0 * (a4 * a5 + a2 * a6) +
      emsq * (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
    z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
    z1 = z1 + z1 + betasq * z31;
    z2 = z2 + z2 + betasq * z32;
    z3 = z3 + z3 + betasq * z33;
    s3 = cc * xnoi;
    s2 = (-0.5 * s3) / rtemsq;
    s4 = s3 * rtemsq;
    s1 = -15.0 * em * s4;
    s5 = x1 * x3 + x2 * x4;
    s6 = x2 * x3 + x1 * x4;
    s7 = x2 * x4 - x1 * x3;

    if (lsflg === 1) {
      ss1 = s1; ss2 = s2; ss3 = s3; ss4 = s4; ss5 = s5; ss6 = s6; ss7 = s7;
      sz1 = z1; sz2 = z2; sz3 = z3;
      sz11 = z11; sz12 = z12; sz13 = z13;
      sz21 = z21; sz22 = z22; sz23 = z23;
      sz31 = z31; sz32 = z32; sz33 = z33;
      zcosg = zcosgl;
      zsing = zsingl;
      zcosi = zcosil;
      zsini = zsinil;
      zcosh = zcoshl * cnodm + zsinhl * snodm;
      zsinh = snodm * zcoshl - cnodm * zsinhl;
      cc = c1l;
    }
  }

  const zmol = (4.7199672 + (0.2299715 * day - gam)) % TWO_PI;
  const zmos = (6.2565837 + 0.017201977 * day) % TWO_PI;

  // solar periodic coefficients
  const se2 = 2.0 * ss1 * ss6;
  const se3 = 2.0 * ss1 * ss7;
  const si2 = 2.0 * ss2 * sz12;
  const si3 = 2.0 * ss2 * (sz13 - sz11);
  const sl2 = -2.0 * ss3 * sz2;
  const sl3 = -2.0 * ss3 * (sz3 - sz1);
  const sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes;
  const sgh2 = 2.0 * ss4 * sz32;
  const sgh3 = 2.0 * ss4 * (sz33 - sz31);
  const sgh4 = -18.0 * ss4 * zes;
  const sh2 = -2.0 * ss2 * sz22;
  const sh3 = -2.0 * ss2 * (sz23 - sz21);

  // lunar periodic coefficients
  const ee2 = 2.0 * s1 * s6;
  const e3 = 2.0 * s1 * s7;
  const xi2 = 2.0 * s2 * z12;
  const xi3 = 2.0 * s2 * (z13 - z11);
  const xl2 = -2.0 * s3 * z2;
  const xl3 = -2.0 * s3 * (z3 - z1);
  const xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel;
  const xgh2 = 2.0 * s4 * z32;
  const xgh3 = 2.0 * s4 * (z33 - z31);
  const xgh4 = -18.0 * s4 * zel;
  const xh2 = -2.0 * s2 * z22;
  const xh3 = -2.0 * s2 * (z23 - z21);

  return {
    sinim, cosim, em, emsq, nm,
    e3, ee2, peo: 0.0, pgho: 0.0, pho: 0.0, pinco: 0.0, plo: 0.0,
    se2, se3, sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4,
    s1, s2, s3, s4, s5,
    ss1, ss2, ss3, ss4, ss5,
    sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33,
    xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
    z1, z3, z11, z13, z21, z23, z31, z33,
    zmol, zmos,
  };
}

/**
 * dpper — deep-space lunar-solar periodic perturbations. With `init` true
 * (sgp4init) the periodics are computed but NOT applied (Vallado gsfc
 * convention: peo..pho stay 0, elements pass through). Per-call (`init`
 * false) they perturb e/i/node/argp/M; inclinations under 0.2 rad take the
 * Lyddane branch. opsmode 'i' (improved) — the afspc-only node fixes are
 * no-ops there and omitted.
 */
function dpper(
  s: Satrec,
  t: number,
  init: boolean,
  ep: number,
  inclp: number,
  nodep: number,
  argpp: number,
  mp: number,
) {
  const zns = 1.19459e-5;
  const zes = 0.01675;
  const znl = 1.5835218e-4;
  const zel = 0.0549;

  // solar periodics
  let zm = init ? s.zmos : s.zmos + zns * t;
  let zf = zm + 2.0 * zes * Math.sin(zm);
  let sinzf = Math.sin(zf);
  let f2 = 0.5 * sinzf * sinzf - 0.25;
  let f3 = -0.5 * sinzf * Math.cos(zf);
  const ses = s.se2 * f2 + s.se3 * f3;
  const sis = s.si2 * f2 + s.si3 * f3;
  const sls = s.sl2 * f2 + s.sl3 * f3 + s.sl4 * sinzf;
  const sghs = s.sgh2 * f2 + s.sgh3 * f3 + s.sgh4 * sinzf;
  const shs = s.sh2 * f2 + s.sh3 * f3;

  // lunar periodics
  zm = init ? s.zmol : s.zmol + znl * t;
  zf = zm + 2.0 * zel * Math.sin(zm);
  sinzf = Math.sin(zf);
  f2 = 0.5 * sinzf * sinzf - 0.25;
  f3 = -0.5 * sinzf * Math.cos(zf);
  const sel = s.ee2 * f2 + s.e3 * f3;
  const sil = s.xi2 * f2 + s.xi3 * f3;
  const sll = s.xl2 * f2 + s.xl3 * f3 + s.xl4 * sinzf;
  const sghl = s.xgh2 * f2 + s.xgh3 * f3 + s.xgh4 * sinzf;
  const shll = s.xh2 * f2 + s.xh3 * f3;

  let pe = ses + sel;
  let pinc = sis + sil;
  let pl = sls + sll;
  let pgh = sghs + sghl;
  let ph = shs + shll;

  if (!init) {
    pe -= s.peo;
    pinc -= s.pinco;
    pl -= s.plo;
    pgh -= s.pgho;
    ph -= s.pho;
    inclp += pinc;
    ep += pe;
    const sinip = Math.sin(inclp);
    const cosip = Math.cos(inclp);

    if (inclp >= 0.2) {
      // apply periodics directly
      ph /= sinip;
      pgh -= cosip * ph;
      argpp += pgh;
      nodep += ph;
      mp += pl;
    } else {
      // apply periodics with the Lyddane modification (low inclination)
      const sinop = Math.sin(nodep);
      const cosop = Math.cos(nodep);
      let alfdp = sinip * sinop;
      let betdp = sinip * cosop;
      const dalf = ph * cosop + pinc * cosip * sinop;
      const dbet = -ph * sinop + pinc * cosip * cosop;
      alfdp += dalf;
      betdp += dbet;
      nodep %= TWO_PI;
      let xls = mp + argpp + cosip * nodep;
      const dls = pl + pgh - pinc * nodep * sinip;
      xls += dls;
      const xnoh = nodep;
      nodep = Math.atan2(alfdp, betdp);
      if (Math.abs(xnoh - nodep) > PI) {
        nodep = nodep < xnoh ? nodep + TWO_PI : nodep - TWO_PI;
      }
      mp += pl;
      argpp = xls - mp - cosip * nodep;
    }
  }

  return { ep, inclp, nodep, argpp, mp };
}

/**
 * dsinit — deep-space initialization: lunar-solar secular rates, resonance
 * classification (irez: 1 = synchronous/24h, 2 = half-day/12h with e >= 0.5)
 * and the resonance d/del coefficients + integrator seed (xlamo/xli/xni).
 */
function dsinit(o: {
  cosim: number; sinim: number; emsq: number; argpo: number;
  s1: number; s2: number; s3: number; s4: number; s5: number;
  ss1: number; ss2: number; ss3: number; ss4: number; ss5: number;
  sz1: number; sz3: number; sz11: number; sz13: number;
  sz21: number; sz23: number; sz31: number; sz33: number;
  t: number; tc: number; gsto: number; mo: number; mdot: number; no: number;
  nodeo: number; nodedot: number; xpidot: number;
  z1: number; z3: number; z11: number; z13: number;
  z21: number; z23: number; z31: number; z33: number;
  ecco: number; eccsq: number;
  em: number; argpm: number; inclm: number; mm: number; nm: number; nodem: number;
}) {
  const q22 = 1.7891679e-6;
  const q31 = 2.1460748e-6;
  const q33 = 2.2123015e-7;
  const root22 = 1.7891679e-6;
  const root44 = 7.3636953e-9;
  const root54 = 2.1765803e-9;
  const rptim = 4.37526908801129966e-3; // earth rotation, rad/min
  const root32 = 3.7393792e-7;
  const root52 = 1.1428639e-7;
  const znl = 1.5835218e-4;
  const zns = 1.19459e-5;

  const {
    cosim, sinim, argpo, s1, s2, s3, s4, s5, ss1, ss2, ss3, ss4, ss5,
    sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33, t, tc, gsto, mo, mdot,
    no, nodeo, nodedot, xpidot, z1, z3, z11, z13, z21, z23, z31, z33,
    ecco, eccsq,
  } = o;
  let { em, argpm, inclm, mm, nm, nodem, emsq } = o;

  let irez = 0;
  if (nm < 0.0052359877 && nm > 0.0034906585) irez = 1;
  if (nm >= 8.26e-3 && nm <= 9.24e-3 && em >= 0.5) irez = 2;

  // solar secular rates
  const ses = ss1 * zns * ss5;
  const sis = ss2 * zns * (sz11 + sz13);
  const sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq);
  const sghs = ss4 * zns * (sz31 + sz33 - 6.0);
  let shs = -zns * ss2 * (sz21 + sz23);
  if (inclm < 5.2359877e-2 || inclm > PI - 5.2359877e-2) shs = 0.0;
  if (sinim !== 0.0) shs /= sinim;
  const sgs = sghs - cosim * shs;

  // lunar secular rates
  const dedt = ses + s1 * znl * s5;
  const didt = sis + s2 * znl * (z11 + z13);
  const dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq);
  const sghl = s4 * znl * (z31 + z33 - 6.0);
  let shll = -znl * s2 * (z21 + z23);
  if (inclm < 5.2359877e-2 || inclm > PI - 5.2359877e-2) shll = 0.0;
  let domdt = sgs + sghl;
  let dnodt = shs;
  if (sinim !== 0.0) {
    domdt -= (cosim / sinim) * shll;
    dnodt += shll / sinim;
  }

  // deep-space resonance effects
  const dndt = 0.0;
  const theta = (gsto + tc * rptim) % TWO_PI;
  em += dedt * t;
  inclm += didt * t;
  argpm += domdt * t;
  nodem += dnodt * t;
  mm += dmdt * t;

  let d2201 = 0, d2211 = 0, d3210 = 0, d3222 = 0, d4410 = 0, d4422 = 0;
  let d5220 = 0, d5232 = 0, d5421 = 0, d5433 = 0;
  let del1 = 0, del2 = 0, del3 = 0;
  let xfact = 0, xlamo = 0, xli = 0, xni = 0, atime = 0;

  if (irez !== 0) {
    const aonv = Math.pow(nm / XKE, X2O3);

    // geopotential resonance for 12-hour orbits
    if (irez === 2) {
      const cosisq = cosim * cosim;
      const emo = em;
      em = ecco;
      const emsqo = emsq;
      emsq = eccsq;
      const eoc = em * emsq;
      const g201 = -0.306 - (em - 0.64) * 0.44;

      let g211: number, g310: number, g322: number, g410: number, g422: number, g520: number;
      let g533: number, g521: number, g532: number;
      if (em <= 0.65) {
        g211 = 3.616 - 13.247 * em + 16.29 * emsq;
        g310 = -19.302 + 117.39 * em - 228.419 * emsq + 156.591 * eoc;
        g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc;
        g410 = -41.122 + 242.694 * em - 471.094 * emsq + 313.953 * eoc;
        g422 = -146.407 + 841.88 * em - 1629.014 * emsq + 1083.435 * eoc;
        g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.276 * eoc;
      } else {
        g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc;
        g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc;
        g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc;
        g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc;
        g422 = -3581.69 + 16178.11 * em - 24462.77 * emsq + 12422.52 * eoc;
        if (em > 0.715) {
          g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc;
        } else {
          g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq;
        }
      }
      if (em < 0.7) {
        g533 = -919.2277 + 4988.61 * em - 9064.77 * emsq + 5542.21 * eoc;
        g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc;
        g532 = -853.666 + 4690.25 * em - 8624.77 * emsq + 5341.4 * eoc;
      } else {
        g533 = -37995.78 + 161616.52 * em - 229838.2 * emsq + 109377.94 * eoc;
        g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc;
        g532 = -40023.88 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc;
      }

      const sini2 = sinim * sinim;
      const f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq);
      const f221 = 1.5 * sini2;
      const f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq);
      const f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq);
      const f441 = 35.0 * sini2 * f220;
      const f442 = 39.375 * sini2 * sini2;
      const f522 =
        9.84375 * sinim *
        (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq) +
          0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq));
      const f523 =
        sinim *
        (4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq) +
          6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq));
      const f542 =
        29.53125 * sinim *
        (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq));
      const f543 =
        29.53125 * sinim *
        (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq));

      const xno2 = nm * nm;
      const ainv2 = aonv * aonv;
      let temp1 = 3.0 * xno2 * ainv2;
      let temp = temp1 * root22;
      d2201 = temp * f220 * g201;
      d2211 = temp * f221 * g211;
      temp1 *= aonv;
      temp = temp1 * root32;
      d3210 = temp * f321 * g310;
      d3222 = temp * f322 * g322;
      temp1 *= aonv;
      temp = 2.0 * temp1 * root44;
      d4410 = temp * f441 * g410;
      d4422 = temp * f442 * g422;
      temp1 *= aonv;
      temp = temp1 * root52;
      d5220 = temp * f522 * g520;
      d5232 = temp * f523 * g532;
      temp = 2.0 * temp1 * root54;
      d5421 = temp * f542 * g521;
      d5433 = temp * f543 * g533;
      xlamo = (mo + nodeo + nodeo - (theta + theta)) % TWO_PI;
      xfact = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no;
      em = emo;
      emsq = emsqo;
    }

    // synchronous resonance terms
    if (irez === 1) {
      const g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq);
      const g310 = 1.0 + 2.0 * emsq;
      const g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq);
      const f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim);
      const f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim);
      let f330 = 1.0 + cosim;
      f330 = 1.875 * f330 * f330 * f330;
      del1 = 3.0 * nm * nm * aonv * aonv;
      del2 = 2.0 * del1 * f220 * g200 * q22;
      del3 = 3.0 * del1 * f330 * g300 * q33 * aonv;
      del1 = del1 * f311 * g310 * q31 * aonv;
      xlamo = (mo + nodeo + argpo - theta) % TWO_PI;
      xfact = mdot + xpidot - rptim + dmdt + domdt - no;
    }

    // initialize the resonance integrator
    xli = xlamo;
    xni = no;
    atime = 0.0;
    nm = no + dndt;
  }

  return {
    em, argpm, inclm, mm, nm, nodem,
    irez, atime,
    d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433,
    dedt, didt, dmdt, dnodt, domdt,
    del1, del2, del3,
    xfact, xlamo, xli, xni,
  };
}

/**
 * dspace — deep-space secular effects and resonance integration (720-min
 * Euler-Maclaurin steps) at call time. Integrator state is deliberately NOT
 * persisted across calls (matches satellite.js observed behavior): every call
 * restarts from the epoch seed, so the output is a pure function of
 * (initialized Satrec, t) — deterministic for any call order.
 */
function dspace(
  s: Satrec,
  t: number,
  tc: number,
  em: number,
  argpm: number,
  inclm: number,
  mm: number,
  nodem: number,
  nm: number,
) {
  const fasx2 = 0.13130908;
  const fasx4 = 2.8843198;
  const fasx6 = 0.37448087;
  const g22 = 5.7686396;
  const g32 = 0.95240898;
  const g44 = 1.8014998;
  const g52 = 1.050833;
  const g54 = 4.4108898;
  const rptim = 4.37526908801129966e-3; // earth rotation, rad/min
  const stepp = 720.0;
  const stepn = -720.0;
  const step2 = 259200.0; // stepp^2 / 2

  const theta = (s.gsto + tc * rptim) % TWO_PI;
  em += s.dedt * t;
  inclm += s.didt * t;
  argpm += s.domdt * t;
  nodem += s.dnodt * t;
  mm += s.dmdt * t;

  if (s.irez !== 0) {
    let atime = s.atime;
    let xni = s.xni;
    let xli = s.xli;
    let xndt = 0.0;
    let xnddt = 0.0;
    let xldot = 0.0;
    let ft = 0.0;

    // epoch restart (with un-persisted state this always resets to epoch)
    if (atime === 0.0 || t * atime <= 0.0 || Math.abs(t) < Math.abs(atime)) {
      atime = 0.0;
      xni = s.no;
      xli = s.xlamo;
    }
    const delt = t > 0.0 ? stepp : stepn;

    let iretn = 381;
    while (iretn === 381) {
      if (s.irez !== 2) {
        // near-synchronous resonance dot terms
        xndt =
          s.del1 * Math.sin(xli - fasx2) +
          s.del2 * Math.sin(2.0 * (xli - fasx4)) +
          s.del3 * Math.sin(3.0 * (xli - fasx6));
        xldot = xni + s.xfact;
        xnddt =
          s.del1 * Math.cos(xli - fasx2) +
          2.0 * s.del2 * Math.cos(2.0 * (xli - fasx4)) +
          3.0 * s.del3 * Math.cos(3.0 * (xli - fasx6));
        xnddt *= xldot;
      } else {
        // near-half-day resonance dot terms
        const xomi = s.argpo + s.argpdot * atime;
        const x2omi = xomi + xomi;
        const x2li = xli + xli;
        xndt =
          s.d2201 * Math.sin(x2omi + xli - g22) +
          s.d2211 * Math.sin(xli - g22) +
          s.d3210 * Math.sin(xomi + xli - g32) +
          s.d3222 * Math.sin(-xomi + xli - g32) +
          s.d4410 * Math.sin(x2omi + x2li - g44) +
          s.d4422 * Math.sin(x2li - g44) +
          s.d5220 * Math.sin(xomi + xli - g52) +
          s.d5232 * Math.sin(-xomi + xli - g52) +
          s.d5421 * Math.sin(xomi + x2li - g54) +
          s.d5433 * Math.sin(-xomi + x2li - g54);
        xldot = xni + s.xfact;
        xnddt =
          s.d2201 * Math.cos(x2omi + xli - g22) +
          s.d2211 * Math.cos(xli - g22) +
          s.d3210 * Math.cos(xomi + xli - g32) +
          s.d3222 * Math.cos(-xomi + xli - g32) +
          s.d5220 * Math.cos(xomi + xli - g52) +
          s.d5232 * Math.cos(-xomi + xli - g52) +
          2.0 *
            (s.d4410 * Math.cos(x2omi + x2li - g44) +
              s.d4422 * Math.cos(x2li - g44) +
              s.d5421 * Math.cos(xomi + x2li - g54) +
              s.d5433 * Math.cos(-xomi + x2li - g54));
        xnddt *= xldot;
      }

      // integrator: step until within one step of t, then evaluate
      if (Math.abs(t - atime) >= stepp) {
        iretn = 381;
      } else {
        ft = t - atime;
        iretn = 0;
      }
      if (iretn === 381) {
        xli += xldot * delt + xndt * step2;
        xni += xndt * delt + xnddt * step2;
        atime += delt;
      }
    }

    nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
    const xl = xli + xldot * ft + xndt * ft * ft * 0.5;
    let dndt: number;
    if (s.irez !== 1) {
      mm = xl - 2.0 * nodem + 2.0 * theta;
      dndt = nm - s.no;
    } else {
      mm = xl - nodem - argpm + theta;
      dndt = nm - s.no;
    }
    nm = s.no + dndt;
  }

  return { em, argpm, inclm, mm, nodem, nm };
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
    method: 'n',
    gsto: 0,
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
    e3: 0, ee2: 0,
    peo: 0, pgho: 0, pho: 0, pinco: 0, plo: 0,
    se2: 0, se3: 0,
    sgh2: 0, sgh3: 0, sgh4: 0,
    sh2: 0, sh3: 0,
    si2: 0, si3: 0,
    sl2: 0, sl3: 0, sl4: 0,
    xgh2: 0, xgh3: 0, xgh4: 0,
    xh2: 0, xh3: 0,
    xi2: 0, xi3: 0,
    xl2: 0, xl3: 0, xl4: 0,
    zmol: 0, zmos: 0,
    irez: 0,
    d2201: 0, d2211: 0, d3210: 0, d3222: 0,
    d4410: 0, d4422: 0,
    d5220: 0, d5232: 0, d5421: 0, d5433: 0,
    dedt: 0, didt: 0, dmdt: 0, dnodt: 0, domdt: 0,
    del1: 0, del2: 0, del3: 0,
    xfact: 0, xlamo: 0, xli: 0, xni: 0, atime: 0,
  };

  const ss = 78.0 / RADIUS_EARTH_KM + 1.0;
  const qzms2ttemp = (120.0 - 78.0) / RADIUS_EARTH_KM;
  const qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;

  const init = initl(gp.ecco, gp.inclo, gp.noKozai);
  s.no = init.no;
  s.con41 = init.con41;

  const { no, ao, con41, con42, cosio, cosio2, omeosq, posq, rp, rteosq, sinio } = init;

  // GMST at epoch + days since 1949 Dec 31 — the deep-space time base.
  const jdEpoch = jdayFromMs(epochMs);
  s.gsto = gstime(jdEpoch);
  const epoch1950 = jdEpoch - 2433281.5;

  if (TWO_PI / no >= DEEPSPACE_PERIOD_MIN) {
    s.deepSpace = true;
    s.method = 'd';
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
    const xpidot = s.argpdot + s.nodedot;
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

    // --- deep-space initialization (SDP4, Vallado method 'd' path) ---
    if (s.method === 'd') {
      s.isimp = 1;
      const tc = 0.0;
      const inclm = s.inclo;
      const eccsq = s.ecco * s.ecco;

      const dc = dscom(epoch1950, s.ecco, s.argpo, tc, s.inclo, s.nodeo, s.no);
      s.e3 = dc.e3; s.ee2 = dc.ee2;
      s.peo = dc.peo; s.pgho = dc.pgho; s.pho = dc.pho; s.pinco = dc.pinco; s.plo = dc.plo;
      s.se2 = dc.se2; s.se3 = dc.se3;
      s.sgh2 = dc.sgh2; s.sgh3 = dc.sgh3; s.sgh4 = dc.sgh4;
      s.sh2 = dc.sh2; s.sh3 = dc.sh3;
      s.si2 = dc.si2; s.si3 = dc.si3;
      s.sl2 = dc.sl2; s.sl3 = dc.sl3; s.sl4 = dc.sl4;
      s.xgh2 = dc.xgh2; s.xgh3 = dc.xgh3; s.xgh4 = dc.xgh4;
      s.xh2 = dc.xh2; s.xh3 = dc.xh3;
      s.xi2 = dc.xi2; s.xi3 = dc.xi3;
      s.xl2 = dc.xl2; s.xl3 = dc.xl3; s.xl4 = dc.xl4;
      s.zmol = dc.zmol; s.zmos = dc.zmos;

      // dpper at init computes but does not apply periodics (see dpper doc).
      const dp = dpper(s, 0.0, true, s.ecco, s.inclo, s.nodeo, s.argpo, s.mo);
      s.ecco = dp.ep;
      s.inclo = dp.inclp;
      s.nodeo = dp.nodep;
      s.argpo = dp.argpp;
      s.mo = dp.mp;

      const ds = dsinit({
        cosim: dc.cosim, sinim: dc.sinim, emsq: dc.emsq, argpo: s.argpo,
        s1: dc.s1, s2: dc.s2, s3: dc.s3, s4: dc.s4, s5: dc.s5,
        ss1: dc.ss1, ss2: dc.ss2, ss3: dc.ss3, ss4: dc.ss4, ss5: dc.ss5,
        sz1: dc.sz1, sz3: dc.sz3, sz11: dc.sz11, sz13: dc.sz13,
        sz21: dc.sz21, sz23: dc.sz23, sz31: dc.sz31, sz33: dc.sz33,
        t: 0.0, tc, gsto: s.gsto, mo: s.mo, mdot: s.mdot, no: s.no,
        nodeo: s.nodeo, nodedot: s.nodedot, xpidot,
        z1: dc.z1, z3: dc.z3, z11: dc.z11, z13: dc.z13,
        z21: dc.z21, z23: dc.z23, z31: dc.z31, z33: dc.z33,
        ecco: s.ecco, eccsq,
        em: dc.em, argpm: 0.0, inclm, mm: 0.0, nm: dc.nm, nodem: 0.0,
      });
      s.irez = ds.irez; s.atime = ds.atime;
      s.d2201 = ds.d2201; s.d2211 = ds.d2211; s.d3210 = ds.d3210; s.d3222 = ds.d3222;
      s.d4410 = ds.d4410; s.d4422 = ds.d4422;
      s.d5220 = ds.d5220; s.d5232 = ds.d5232; s.d5421 = ds.d5421; s.d5433 = ds.d5433;
      s.dedt = ds.dedt; s.didt = ds.didt; s.dmdt = ds.dmdt; s.dnodt = ds.dnodt; s.domdt = ds.domdt;
      s.del1 = ds.del1; s.del2 = ds.del2; s.del3 = ds.del3;
      s.xfact = ds.xfact; s.xlamo = ds.xlamo; s.xli = ds.xli; s.xni = ds.xni;
    }

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
  const xmdf = s.mo + s.mdot * tsince;
  const argpdf = s.argpo + s.argpdot * tsince;
  const nodedf = s.nodeo + s.nodedot * tsince;
  let argpm = argpdf;
  let mm = xmdf;
  const t2 = tsince * tsince;
  let nodem = nodedf + s.nodecf * t2;
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
  let inclm = s.inclo;

  // deep-space secular effects + resonance integration
  if (s.method === 'd') {
    const dsp = dspace(s, tsince, tsince, em, argpm, inclm, mm, nodem, nm);
    em = dsp.em;
    argpm = dsp.argpm;
    inclm = dsp.inclm;
    mm = dsp.mm;
    nodem = dsp.nodem;
    nm = dsp.nm;
  }

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

  nodem %= TWO_PI;
  argpm %= TWO_PI;
  xlm %= TWO_PI;
  mm = (xlm - argpm - nodem) % TWO_PI;

  const sinim = Math.sin(inclm);
  const cosim = Math.cos(inclm);

  let ep = em;
  let xincp = inclm;
  let argpp = argpm;
  let nodep = nodem;
  let mp = mm;
  let sinip = sinim;
  let cosip = cosim;

  // deep-space lunar-solar periodics
  if (s.method === 'd') {
    const dp = dpper(s, tsince, false, ep, xincp, nodep, argpp, mp);
    ep = dp.ep;
    xincp = dp.inclp;
    nodep = dp.nodep;
    argpp = dp.argpp;
    mp = dp.mp;
    if (xincp < 0.0) {
      xincp = -xincp;
      nodep += PI;
      argpp -= PI;
    }
    if (ep < 0.0 || ep > 1.0) {
      s.error = 3;
      return null;
    }
  }

  // long-period periodics (deep space recomputes the inclination-dependent
  // coefficients from the perturbed inclination — Vallado)
  let aycof = s.aycof;
  let xlcof = s.xlcof;
  if (s.method === 'd') {
    sinip = Math.sin(xincp);
    cosip = Math.cos(xincp);
    aycof = -0.5 * J3OJ2 * sinip;
    if (Math.abs(cosip + 1.0) > 1.5e-12) {
      xlcof = (-0.25 * J3OJ2 * sinip * (3.0 + 5.0 * cosip)) / (1.0 + cosip);
    } else {
      xlcof = (-0.25 * J3OJ2 * sinip * (3.0 + 5.0 * cosip)) / 1.5e-12;
    }
  }

  const axnl = ep * Math.cos(argpp);
  let temp = 1.0 / (am * (1.0 - ep * ep));
  const aynl = ep * Math.sin(argpp) + temp * aycof;
  const xl = mp + argpp + nodep + temp * xlcof * axnl;

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

  let con41 = s.con41; // near-earth: inclination unchanged, so this holds
  let x1mth2 = s.x1mth2;
  let x7thm1 = s.x7thm1;
  if (s.method === 'd') {
    // deep space: recompute from the perturbed inclination
    const cosisq = cosip * cosip;
    con41 = 3.0 * cosisq - 1.0;
    x1mth2 = 1.0 - cosisq;
    x7thm1 = 7.0 * cosisq - 1.0;
  }

  const mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + 0.5 * temp1 * x1mth2 * cos2u;
  // decayed satellite (mrt < 1 earth radius) — honest null, never a
  // subterranean position. Gated to 'd' to keep the near-earth path
  // bit-for-bit identical to its validated pre-SDP4 behavior.
  if (s.method === 'd' && mrt < 1.0) {
    s.error = 6;
    return null;
  }
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

/**
 * True if the element set is deep-space (period >= 225 min). These objects
 * propagate through the SDP4 deep-space path; exported so callers can label
 * them (GEO/MEO/HEO vs LEO kernels have different accuracy expectations).
 */
export function isDeepSpace(gp: GpRecord): boolean {
  if (gp.meanMotion == null || gp.meanMotion <= 0) return false;
  const noKozai = gp.meanMotion / XPDOTP; // rad/min
  return TWO_PI / noKozai >= DEEPSPACE_PERIOD_MIN;
}

/**
 * Propagate a GP element record to `dateMs` (Unix ms, UTC). Near-earth sets
 * use SGP4; deep-space sets (period >= 225 min) use the SDP4 corrections.
 * Returns geodetic position, or null when: elements are incomplete, the
 * epoch is unparseable, or SGP4/SDP4 flags a decayed/degenerate orbit.
 * NEVER returns a faked/extrapolated-beyond-validity position.
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
