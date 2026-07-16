// Solar-system ephemeris — EARTH TWIN O6-7 tier 2. Pinned against JPL
// Horizons vector anchors at two epochs (J2000.0, where the low-precision
// theory is centered, and 2026-01-01, the product's operating era) plus
// physical-scale invariants. Follows the anchor-validation discipline of
// ephemeris.test.ts (tier 1).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  solarSystemState,
  dayNumber,
  eclipticOfDateToJ2000LonDeg,
  eccentricAnomalyRad,
  angularDiameterDeg,
  AU_M,
  BODY_ORDER,
  BODY_RADIUS_M,
  type BodyId,
  type Vec3,
} from "./solarSystem.js";

// ── anchors ──────────────────────────────────────────────────────────────────
// Source: NASA/JPL Horizons API (https://ssd.jpl.nasa.gov/api/horizons.api),
// queried 2026-07-16. Query per body:
//   EPHEM_TYPE='VECTORS', VEC_TABLE='1', OUT_UNITS='AU-D',
//   REF_PLANE='ECLIPTIC', REF_SYSTEM='J2000',
//   CENTER='500@10' (heliocentric; Moon uses '500@399' = geocentric),
//   START_TIME = the epoch below (Horizons vector times are TDB).
// Values are the X,Y,Z position rows copied verbatim from the API output
// (planetary source DE441). Epochs: 2451545.0 TDB = 2000-01-01 12:00 and
// 2461041.5 TDB = 2026-01-01 00:00.
//
// TIME-SCALE HONESTY: our ephemeris evaluates at the same NOMINAL clock time
// in UTC, ignoring TDB−UTC (64.2 s in 2000, 69.2 s in 2026 — the source's
// stated simplification #3). That contributes ≤0.6' on the Moon and ≤0.02'
// on planets; it is folded into the tolerances below.
type Anchor = [number, number, number];
const HORIZONS: Record<"E2000" | "E2026", Partial<Record<BodyId, Anchor>>> = {
  E2000: {
    mercury: [-1.300936053754522e-1, -4.472876181353563e-1, -2.459830695805179e-2],
    venus: [-7.18302296345389e-1, -3.265430819980606e-2, 4.101418202684621e-2],
    earth: [-1.771350992727098e-1, 9.672416867665306e-1, -4.085281582511366e-6],
    mars: [1.390715921746351, -1.341631815101244e-2, -3.446766277581799e-2],
    jupiter: [4.001177435589426, 2.938575782470499, -1.01785283451815e-1],
    saturn: [6.406410428378656, 6.569988452110556, -3.690759730763678e-1],
    uranus: [1.443185527592405e1, -1.373432340215935e1, -2.381417673271042e-1],
    neptune: [1.681204760586415e1, -2.499176325272639e1, 1.272225117089998e-1],
    moon: [-1.949281649686695e-3, -1.838126040073046e-3, 2.424579738820632e-4],
  },
  E2026: {
    mercury: [-2.152013043776677e-1, -4.092076157002721e-1, -1.370326959192366e-2],
    venus: [8.887724961635655e-2, -7.217623823170807e-1, -1.504405657506193e-2],
    earth: [-1.742814809205833e-1, 9.677589202546031e-1, -5.944511017526243e-5],
    mars: [3.405796768622151e-1, -1.387002015945254, -3.741722678770108e-2],
    jupiter: [-1.694003692586729, 4.928882358715955, 1.742622062607817e-2],
    saturn: [9.507341652288904, 2.577390400089007e-1, -3.829356818352184e-1],
    uranus: [9.880310929544422, 1.680001212340429e1, -6.571963922668925e-2],
    neptune: [2.987212235896263e1, 5.189383438441971e-1, -6.990674873090486e-1],
    moon: [9.647578994045626e-4, 2.201875129333777e-3, 2.122555367574619e-4],
  },
};
const EPOCH_MS = {
  E2000: Date.UTC(2000, 0, 1, 12, 0, 0),
  E2026: Date.UTC(2026, 0, 1, 0, 0, 0),
} as const;

const DEG = Math.PI / 180;

/** Rotate our ecliptic-of-date vector into the anchors' J2000 frame (§8). */
function toJ2000(v: Vec3, dateMs: number): Vec3 {
  const lonDeg = Math.atan2(v.y, v.x) / DEG;
  const lat = Math.atan2(v.z, Math.hypot(v.x, v.y));
  const r = Math.hypot(v.x, v.y, v.z);
  const lon = eclipticOfDateToJ2000LonDeg(lonDeg, dateMs) * DEG;
  return {
    x: r * Math.cos(lat) * Math.cos(lon),
    y: r * Math.cos(lat) * Math.sin(lon),
    z: r * Math.sin(lat),
  };
}

/** Great-circle separation between two position vectors, arcminutes. */
function sepArcmin(a: Anchor, b: Vec3): number {
  const la = Math.hypot(a[0], a[1], a[2]);
  const lb = Math.hypot(b.x, b.y, b.z);
  const dot = (a[0] * b.x + a[1] * b.y + a[2] * b.z) / (la * lb);
  return (Math.acos(Math.max(-1, Math.min(1, dot))) / DEG) * 60;
}

// Tolerances: measured maxima at these anchors are 1.25' angular / 0.013%
// distance (inner) / 0.24% distance (outer) for planets, and 3.01' / 0.064%
// for the Moon. Asserted with ~2x headroom so the pin catches real
// regressions (a dropped perturbation term, a sign flip) without flaking on
// the theory's own noise floor.
const PLANET_SEP_ARCMIN = 3.0;
const INNER_DIST_REL = 0.0005; // mercury..mars: 0.05%
const OUTER_DIST_REL = 0.005;  // jupiter..neptune: 0.5%
const MOON_SEP_ARCMIN = 6.0;
const MOON_DIST_REL = 0.0025;

test("planets vs JPL Horizons heliocentric anchors (J2000.0 + 2026-01-01)", () => {
  for (const tag of ["E2000", "E2026"] as const) {
    const state = solarSystemState(EPOCH_MS[tag]);
    for (const body of state) {
      const anchor = HORIZONS[tag][body.id];
      if (!anchor || body.id === "moon") continue;
      const mine = toJ2000(body.helioAu, EPOCH_MS[tag]);
      const sep = sepArcmin(anchor, mine);
      assert.ok(
        sep < PLANET_SEP_ARCMIN,
        `${tag} ${body.id}: ${sep.toFixed(2)}' from Horizons (limit ${PLANET_SEP_ARCMIN}')`,
      );
      const rAnchor = Math.hypot(anchor[0], anchor[1], anchor[2]);
      const rel = Math.abs(body.rSunAu - rAnchor) / rAnchor;
      const limit = ["jupiter", "saturn", "uranus", "neptune"].includes(body.id)
        ? OUTER_DIST_REL
        : INNER_DIST_REL;
      assert.ok(
        rel < limit,
        `${tag} ${body.id} distance: rel err ${(rel * 100).toFixed(4)}% (limit ${limit * 100}%)`,
      );
    }
  }
});

test("moon vs JPL Horizons geocentric anchors", () => {
  for (const tag of ["E2000", "E2026"] as const) {
    const anchor = HORIZONS[tag].moon!;
    const moon = solarSystemState(EPOCH_MS[tag]).find((b) => b.id === "moon")!;
    const mine = toJ2000(moon.geoAu, EPOCH_MS[tag]);
    const sep = sepArcmin(anchor, mine);
    assert.ok(sep < MOON_SEP_ARCMIN, `${tag} moon: ${sep.toFixed(2)}' (limit ${MOON_SEP_ARCMIN}')`);
    const rAnchor = Math.hypot(anchor[0], anchor[1], anchor[2]);
    const rel = Math.abs(moon.rEarthAu - rAnchor) / rAnchor;
    assert.ok(rel < MOON_DIST_REL, `${tag} moon distance: rel err ${(rel * 100).toFixed(4)}%`);
  }
});

test("day number: the source's own worked example (1990-04-19 00:00 UT = -3543)", () => {
  // ppcomp.html §3 states d = -3543 for 19 April 1990 — an exact integer pin.
  assert.equal(dayNumber(Date.UTC(1990, 3, 19, 0, 0, 0)), -3543);
  assert.equal(dayNumber(Date.UTC(1999, 11, 31, 0, 0, 0)), 0); // day 0.0 = 2000 Jan 0.0 UT
});

test("Kepler solver: degenerate + max-eccentricity cases converge", () => {
  // e = 0: E = M exactly.
  for (const M of [0, 45, 123.4, 359]) {
    assert.ok(Math.abs(eccentricAnomalyRad(M, 0) - ((M % 360) * DEG)) < 1e-12, `e=0, M=${M}`);
  }
  // Mercury's e = 0.2056 — the residual of Kepler's equation must vanish.
  for (const M of [0, 10, 90, 180, 270, 350.5]) {
    const E = eccentricAnomalyRad(M, 0.2056);
    const residual = E - 0.2056 * Math.sin(E) - (M % 360) * DEG;
    assert.ok(Math.abs(residual) < 1e-8, `Kepler residual at M=${M}: ${residual}`);
  }
});

test("scale math: 1 AU exact; Moon ~0.5 deg; Sun ~0.53 deg; distances in real bands", () => {
  assert.equal(AU_M, 149_597_870_700); // IAU 2012 Resolution B2, exact
  // Canonical mean-distance check: Moon radius 1737.4 km over 384,400 km.
  const moonMean = angularDiameterDeg(BODY_RADIUS_M.moon, 384_400_000);
  assert.ok(Math.abs(moonMean - 0.518) < 0.005, `mean-distance moon diameter ~0.518 deg (got ${moonMean.toFixed(4)})`);
  // Across 2026, instantaneous values stay inside the published bands:
  // Moon 29.3'..34.1' (perigee..apogee), Sun 31.46'..32.53'.
  for (let month = 0; month < 12; month++) {
    const state = solarSystemState(Date.UTC(2026, month, 7, 6, 0, 0));
    const by = Object.fromEntries(state.map((b) => [b.id, b]));
    const moonDeg = by.moon.angularDiameterFromEarthDeg;
    assert.ok(moonDeg > 29.3 / 60 && moonDeg < 34.1 / 60, `month ${month}: moon ${(moonDeg * 60).toFixed(2)}'`);
    const sunDeg = by.sun.angularDiameterFromEarthDeg;
    assert.ok(sunDeg > 31.4 / 60 && sunDeg < 32.6 / 60, `month ${month}: sun ${(sunDeg * 60).toFixed(2)}'`);
    // Earth-Sun distance inside the real perihelion..aphelion band.
    assert.ok(by.earth.rSunAu > 0.9829 && by.earth.rSunAu < 1.0171, `month ${month}: r=${by.earth.rSunAu}`);
    // Moon distance inside the real perigee..apogee band (km).
    const moonKm = by.moon.rEarthM / 1000;
    assert.ok(moonKm > 356_000 && moonKm < 407_000, `month ${month}: moon ${moonKm.toFixed(0)} km`);
  }
});

test("frame consistency: geo = helio - earthHelio; Sun/Earth reciprocal; all bodies present", () => {
  const t = Date.UTC(2026, 6, 16, 12, 0, 0);
  const state = solarSystemState(t);
  assert.deepEqual(state.map((b) => b.id), BODY_ORDER);
  const by = Object.fromEntries(state.map((b) => [b.id, b]));
  assert.equal(by.earth.rEarthAu, 0);
  // Sun's geocentric distance is Earth's heliocentric distance, exactly.
  assert.ok(Math.abs(by.sun.rEarthAu - by.earth.rSunAu) < 1e-12);
  for (const b of state) {
    // helio − earthHelio must reproduce geo for every body (Moon included:
    // its helio position is constructed as earthHelio + geo).
    const dx = b.helioAu.x - by.earth.helioAu.x - b.geoAu.x;
    const dy = b.helioAu.y - by.earth.helioAu.y - b.geoAu.y;
    const dz = b.helioAu.z - by.earth.helioAu.z - b.geoAu.z;
    assert.ok(Math.hypot(dx, dy, dz) < 1e-12, `${b.id} helio/geo mismatch`);
    // angular diameters are sane: nothing from Earth looks bigger than the
    // Moon does, except nothing (Moon is the largest apparent disc here).
    if (b.id !== "earth" && b.id !== "moon" && b.id !== "sun") {
      assert.ok(b.angularDiameterFromEarthDeg < 0.02, `${b.id} apparent size ${b.angularDiameterFromEarthDeg}`);
    }
  }
  // The Moon sits ~30 Earth diameters away — the charter's own showcase line.
  const earthDiameters = by.moon.rEarthM / (2 * BODY_RADIUS_M.earth);
  assert.ok(earthDiameters > 27 && earthDiameters < 33, `moon at ${earthDiameters.toFixed(1)} Earth diameters`);
});

test("determinism: same instant, same state (pure function of time)", () => {
  const t = Date.UTC(2026, 3, 1, 0, 0, 0);
  assert.deepEqual(solarSystemState(t), solarSystemState(t));
});
