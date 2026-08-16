// Hermetic tests for the orbital coverage/footprint geometry engine.
// Pure math: no network, no clock, no deps beyond node:test/node:assert.
// Run: npx tsx --test client/src/lib/orbital/geometry.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  STARLINK_MIN_ELEV_DEG,
  groundFootprintRadiusKm,
  footprintCircle,
  centralAngleDeg,
  normalizeLonDeg,
  elevationAzimuthFromSite,
  isVisible,
  gpsDop,
  nextPassOverSite,
  wgs84GeocentricRadiusKm,
  type GeoPosition,
  type PositionSampler,
} from "./geometry.ts";
import {
  EARTH_EQUATORIAL_RADIUS_KM,
  EARTH_POLAR_RADIUS_KM,
} from "./satDerived.ts";

// --------------------------------------------------------------------------
// 0. wgs84GeocentricRadiusKm — the Q24 fix itself
// (research/PROGRAM_STATE.md: propagate.ts's altKm is height above the
// WGS-84 ellipsoid AT THE POINT'S OWN LATITUDE, not above a fixed sphere;
// every function below now adds altKm to this instead of EARTH_MEAN_RADIUS_KM)
// --------------------------------------------------------------------------

test("wgs84GeocentricRadiusKm: reduces to the equatorial/polar constants at 0/90deg (to fp precision)", () => {
  // Not bit-exact equality: the square/sqrt round trip through the closed
  // form introduces ~1e-12 relative fp noise even at these exact angles.
  assert.ok(Math.abs(wgs84GeocentricRadiusKm(0) - EARTH_EQUATORIAL_RADIUS_KM) < 1e-9);
  assert.ok(Math.abs(wgs84GeocentricRadiusKm(90) - EARTH_POLAR_RADIUS_KM) < 1e-9);
  assert.ok(Math.abs(wgs84GeocentricRadiusKm(-90) - EARTH_POLAR_RADIUS_KM) < 1e-9);
});

test("wgs84GeocentricRadiusKm: matches the standard WGS-84 table at 45deg (~6367.4895km)", () => {
  // Independently computed (python, WGS-84 a/b) — see the PR description.
  const r45 = wgs84GeocentricRadiusKm(45);
  assert.ok(Math.abs(r45 - 6367.4895) < 1e-3, `r45=${r45}`);
});

test("wgs84GeocentricRadiusKm: strictly decreasing from equator to pole (oblate, a > b)", () => {
  const r0 = wgs84GeocentricRadiusKm(0);
  const r30 = wgs84GeocentricRadiusKm(30);
  const r60 = wgs84GeocentricRadiusKm(60);
  const r90 = wgs84GeocentricRadiusKm(90);
  assert.ok(r0 > r30 && r30 > r60 && r60 > r90, `${r0} ${r30} ${r60} ${r90}`);
  assert.ok(r0 <= EARTH_EQUATORIAL_RADIUS_KM + 1e-9);
  assert.ok(r90 >= EARTH_POLAR_RADIUS_KM - 1e-9);
});

test("wgs84GeocentricRadiusKm: symmetric north/south", () => {
  for (const lat of [10, 37, 65]) {
    assert.equal(wgs84GeocentricRadiusKm(lat), wgs84GeocentricRadiusKm(-lat));
  }
});

// --------------------------------------------------------------------------
// 1. Footprint radius sanity
// --------------------------------------------------------------------------

test("footprint radius: LEO (~550km) at 25deg mask is a plausible few-hundred-km cap", () => {
  const r = groundFootprintRadiusKm(550, STARLINK_MIN_ELEV_DEG, 0);
  // ~940 km by the coverage-geometry law-of-sines derivation.
  assert.ok(r.radiusKm > 300 && r.radiusKm < 1500, `LEO radius ${r.radiusKm}`);
  assert.ok(r.centralAngleDeg > 0 && r.centralAngleDeg < 20);
});

test("footprint radius: GEO (~35786km) at 5deg covers a continent-scale cap", () => {
  const r = groundFootprintRadiusKm(35786, 5, 0);
  // ~8500 km surface radius; central half-angle ~76deg (near-hemispheric).
  assert.ok(r.radiusKm > 6000 && r.radiusKm < 10000, `GEO radius ${r.radiusKm}`);
  assert.ok(r.centralAngleDeg > 60 && r.centralAngleDeg < 85);
});

test("footprint radius: monotonic increasing in altitude at a fixed mask", () => {
  const a = groundFootprintRadiusKm(300, 25, 0).radiusKm;
  const b = groundFootprintRadiusKm(550, 25, 0).radiusKm;
  const c = groundFootprintRadiusKm(1000, 25, 0).radiusKm;
  const d = groundFootprintRadiusKm(35786, 25, 0).radiusKm;
  assert.ok(a < b && b < c && c < d, `${a} ${b} ${c} ${d}`);
});

test("footprint radius: zenith-only mask (90deg) -> zero cap; horizon (0deg) -> max cap", () => {
  const zenith = groundFootprintRadiusKm(550, 90, 0);
  assert.ok(zenith.radiusKm < 1e-6 && zenith.centralAngleRad < 1e-6);
  const horizon = groundFootprintRadiusKm(550, 0, 0);
  const masked = groundFootprintRadiusKm(550, 25, 0);
  assert.ok(horizon.radiusKm > masked.radiusKm, "horizon cap must exceed a masked cap");
});

test("footprint radius: non-positive altitude returns a zero cap", () => {
  assert.deepEqual(groundFootprintRadiusKm(0, 10, 0), {
    radiusKm: 0,
    centralAngleRad: 0,
    centralAngleDeg: 0,
  });
});

test("footprint radius: at a fixed altitude/mask, the cap is SMALLER near the pole than the equator (Q24 — smaller local Earth radius)", () => {
  const eq = groundFootprintRadiusKm(550, 25, 0).radiusKm;
  const pole = groundFootprintRadiusKm(550, 25, 89).radiusKm;
  assert.ok(pole < eq, `expected pole cap ${pole} < equator cap ${eq}`);
  // The whole point of Q24: the gap is real but small (sub-1%), not the
  // 7km-of-altitude-sized error the original queue entry assumed (L21).
  assert.ok((eq - pole) / eq < 0.01, `unexpectedly large gap: ${(eq - pole) / eq}`);
});

// --------------------------------------------------------------------------
// 2. Footprint circle
// --------------------------------------------------------------------------

test("footprintCircle: returns exactly nPoints [lon,lat] pairs", () => {
  const ring = footprintCircle(30, -100, 550, 25, 64);
  assert.equal(ring.length, 64);
  for (const p of ring) {
    assert.equal(p.length, 2);
    assert.ok(p[0] >= -180 && p[0] < 180, `lon in range: ${p[0]}`);
    assert.ok(p[1] >= -90 && p[1] <= 90, `lat in range: ${p[1]}`);
  }
});

test("footprintCircle: every point lies at ~the cap central angle from the sub-point", () => {
  const subLat = 30;
  const subLon = -100;
  const cap = groundFootprintRadiusKm(550, 25, subLat).centralAngleDeg;
  const ring = footprintCircle(subLat, subLon, 550, 25, 64);
  for (const [lon, lat] of ring) {
    const d = centralAngleDeg(subLat, subLon, lat, lon);
    assert.ok(Math.abs(d - cap) < 1e-6, `point angular dist ${d} vs cap ${cap}`);
  }
});

test("footprintCircle: honours a custom nPoints and a floor of 3", () => {
  assert.equal(footprintCircle(0, 0, 550, 25, 16).length, 16);
  assert.equal(footprintCircle(0, 0, 550, 25, 2).length, 3); // clamped up
});

test("footprintCircle: a polar cap (contains the North Pole) still returns valid samples on the cap", () => {
  // Sub-point near the pole with a wide GEO-ish cap so the pole is inside.
  const subLat = 88;
  const subLon = 10;
  const cap = groundFootprintRadiusKm(35786, 5, subLat).centralAngleDeg;
  const ring = footprintCircle(subLat, subLon, 35786, 5, 64);
  assert.equal(ring.length, 64);
  // Longitudes must sweep widely (ring encircles the pole), and every
  // sample is still exactly on the cap.
  let minLon = 180;
  let maxLon = -180;
  for (const [lon, lat] of ring) {
    const d = centralAngleDeg(subLat, subLon, lat, lon);
    assert.ok(Math.abs(d - cap) < 1e-6, `polar cap point off cap: ${d}`);
    minLon = Math.min(minLon, lon);
    maxLon = Math.max(maxLon, lon);
  }
  assert.ok(maxLon - minLon > 180, "polar cap longitudes should sweep widely");
});

test("normalizeLonDeg: wraps into [-180,180)", () => {
  assert.equal(normalizeLonDeg(190), -170);
  assert.equal(normalizeLonDeg(-190), 170);
  assert.equal(normalizeLonDeg(180), -180);
  assert.equal(normalizeLonDeg(0), 0);
  assert.equal(normalizeLonDeg(540), -180);
});

// --------------------------------------------------------------------------
// 3. Elevation / azimuth / range
// --------------------------------------------------------------------------

test("look angles: a satellite directly overhead -> elevation ~90deg, range ~altitude", () => {
  const la = elevationAzimuthFromSite(10, 20, 550, 10, 20);
  assert.ok(Math.abs(la.elevationDeg - 90) < 1e-3, `elev ${la.elevationDeg}`);
  assert.ok(Math.abs(la.rangeKm - 550) < 1e-3, `range ${la.rangeKm}`);
});

test("look angles: satellite at the geometric horizon -> elevation ~0deg (hand-checked geometry)", () => {
  // For a site at (0,0) and an equatorial satellite, elevation is exactly
  // 0 when the sub-point is at central angle arccos(R/r) away.
  // Site and sub-point are both at latDeg=0, where wgs84GeocentricRadiusKm
  // reduces exactly to EARTH_EQUATORIAL_RADIUS_KM (pinned above) — so the
  // hand-checked formula below matches elevationAzimuthFromSite's post-Q24
  // radius exactly, not just approximately.
  const h = 550;
  const r = EARTH_EQUATORIAL_RADIUS_KM + h;
  const horizonLonDeg = (Math.acos(EARTH_EQUATORIAL_RADIUS_KM / r) * 180) / Math.PI;
  const la = elevationAzimuthFromSite(0, horizonLonDeg, h, 0, 0);
  assert.ok(Math.abs(la.elevationDeg) < 1e-6, `horizon elev ${la.elevationDeg}`);
  // The satellite is due east, so azimuth is 90.
  assert.ok(Math.abs(la.azimuthDeg - 90) < 1e-6, `azimuth ${la.azimuthDeg}`);
});

test("look angles: azimuth points to the correct cardinal direction", () => {
  // Site at (0,0); place the sub-point due N/E/S/W.
  assert.ok(Math.abs(elevationAzimuthFromSite(5, 0, 550, 0, 0).azimuthDeg - 0) < 1e-6);
  assert.ok(Math.abs(elevationAzimuthFromSite(0, 5, 550, 0, 0).azimuthDeg - 90) < 1e-6);
  assert.ok(Math.abs(elevationAzimuthFromSite(-5, 0, 550, 0, 0).azimuthDeg - 180) < 1e-6);
  assert.ok(Math.abs(elevationAzimuthFromSite(0, -5, 550, 0, 0).azimuthDeg - 270) < 1e-6);
});

test("look angles: a satellite below the horizon reports negative elevation", () => {
  // Antipodal-ish: sub-point far from the site -> below the horizon.
  const la = elevationAzimuthFromSite(0, 90, 550, 0, 0);
  assert.ok(la.elevationDeg < 0, `expected below horizon, got ${la.elevationDeg}`);
});

test("isVisible: respects the supplied mask", () => {
  // Overhead is visible at any mask; horizon fails a 10deg mask.
  assert.equal(isVisible(0, 0, 550, 0, 0, 25), true);
  const h = 550;
  const r = EARTH_EQUATORIAL_RADIUS_KM + h;
  const horizonLonDeg = (Math.acos(EARTH_EQUATORIAL_RADIUS_KM / r) * 180) / Math.PI;
  assert.equal(isVisible(0, horizonLonDeg, h, 0, 0, 10), false);
  assert.equal(isVisible(0, horizonLonDeg, h, 0, 0, 0), true);
});

// --------------------------------------------------------------------------
// 4. GPS DOP geometry proxy
// --------------------------------------------------------------------------

test("gpsDop: fewer than 4 satellites returns null", () => {
  assert.equal(gpsDop([]), null);
  assert.equal(
    gpsDop([
      { elevationDeg: 90, azimuthDeg: 0 },
      { elevationDeg: 10, azimuthDeg: 0 },
      { elevationDeg: 10, azimuthDeg: 120 },
    ]),
    null,
  );
});

test("gpsDop: 4 well-spread satellites -> a small finite DOP", () => {
  const dop = gpsDop([
    { elevationDeg: 90, azimuthDeg: 0 }, // zenith
    { elevationDeg: 10, azimuthDeg: 0 },
    { elevationDeg: 10, azimuthDeg: 120 },
    { elevationDeg: 10, azimuthDeg: 240 },
  ]);
  assert.ok(dop !== null);
  assert.ok(Number.isFinite(dop!.gdop) && dop!.gdop > 0, `gdop ${dop!.gdop}`);
  assert.ok(dop!.gdop < 10, `well-spread gdop should be small, got ${dop!.gdop}`);
  // Sanity ordering of the sub-DOPs.
  assert.ok(dop!.pdop <= dop!.gdop + 1e-9);
  assert.ok(dop!.hdop <= dop!.pdop + 1e-9);
});

test("gpsDop: 4 clustered satellites -> a much larger DOP than a well-spread set", () => {
  const spread = gpsDop([
    { elevationDeg: 90, azimuthDeg: 0 },
    { elevationDeg: 10, azimuthDeg: 0 },
    { elevationDeg: 10, azimuthDeg: 120 },
    { elevationDeg: 10, azimuthDeg: 240 },
  ]);
  // All four bunched in one sky region (similar el + az) -> near-collinear
  // LOS rows -> poor geometry -> large finite DOP.
  const clustered = gpsDop([
    { elevationDeg: 50, azimuthDeg: 10 },
    { elevationDeg: 55, azimuthDeg: 20 },
    { elevationDeg: 60, azimuthDeg: 30 },
    { elevationDeg: 65, azimuthDeg: 40 },
  ]);
  assert.ok(spread !== null && clustered !== null);
  assert.ok(Number.isFinite(clustered!.gdop), `clustered gdop finite: ${clustered!.gdop}`);
  assert.ok(clustered!.gdop > 100, `clustered gdop should be large, got ${clustered!.gdop}`);
  assert.ok(
    clustered!.gdop > spread!.gdop * 20,
    `clustered ${clustered!.gdop} should dwarf spread ${spread!.gdop}`,
  );
});

test("gpsDop: degenerate (identical) geometry returns null", () => {
  const identical = Array.from({ length: 4 }, () => ({
    elevationDeg: 45,
    azimuthDeg: 30,
  }));
  assert.equal(gpsDop(identical), null);
});

// --------------------------------------------------------------------------
// 5. nextPassOverSite (synthetic deterministic sampler)
// --------------------------------------------------------------------------

// A fake equatorial ground track: the sub-point moves along the equator at a
// constant rate. Fully deterministic; no Date.now(), no propagation.
function makeEquatorialSampler(
  subLon0Deg: number,
  rateDegPerSec: number,
  altKm: number,
): PositionSampler {
  return (ms: number): GeoPosition => ({
    latDeg: 0,
    lonDeg: normalizeLonDeg(subLon0Deg + rateDegPerSec * (ms / 1000)),
    altKm,
  });
}

// Independent brute-force rise finder (fine linear scan) for cross-checking.
function bruteRiseMs(
  sampler: PositionSampler,
  siteLat: number,
  siteLon: number,
  startMs: number,
  endMs: number,
  mask: number,
  fineMs = 100,
): number | null {
  let prev: number | null = null;
  for (let t = startMs; t <= endMs; t += fineMs) {
    const p = sampler(t);
    const e = elevationAzimuthFromSite(
      p.latDeg,
      p.lonDeg,
      p.altKm,
      siteLat,
      siteLon,
    ).elevationDeg;
    if (prev !== null && prev < mask && e >= mask) return t;
    prev = e;
  }
  return null;
}

test("nextPassOverSite: finds the pass; refined rise agrees with a brute-force scan", () => {
  const sampler = makeEquatorialSampler(-30, 0.05, 550);
  const opts = {
    startMs: 0,
    windowSec: 3600,
    stepSec: 30,
    minElevDeg: 10,
    refineSec: 1,
  };
  const pass = nextPassOverSite(sampler, 0, 0, opts);
  assert.ok(pass !== null, "expected a pass");
  const brute = bruteRiseMs(sampler, 0, 0, 0, 3600 * 1000, 10);
  assert.ok(brute !== null);
  assert.ok(
    Math.abs(pass!.nextRiseMs - brute!) < 1500,
    `rise ${pass!.nextRiseMs} vs brute ${brute}`,
  );
  // Passes directly overhead -> max elevation near 90.
  assert.ok(pass!.maxElevationDeg > 85, `maxElev ${pass!.maxElevationDeg}`);
  assert.ok(pass!.durationSec > 0 && pass!.setWithinWindow);
  assert.ok(pass!.setMs > pass!.nextRiseMs);
});

test("nextPassOverSite: consecutive passes are ordered and ~one period apart", () => {
  const rate = 0.05; // deg/s -> period = 360/rate = 7200 s
  const sampler = makeEquatorialSampler(-30, rate, 550);
  const p1 = nextPassOverSite(sampler, 0, 0, {
    startMs: 0,
    windowSec: 3600,
    stepSec: 30,
    minElevDeg: 10,
    refineSec: 1,
  });
  assert.ok(p1 !== null);
  const p2 = nextPassOverSite(sampler, 0, 0, {
    startMs: p1!.setMs + 60_000,
    windowSec: 8000,
    stepSec: 30,
    minElevDeg: 10,
    refineSec: 1,
  });
  assert.ok(p2 !== null);
  assert.ok(p2!.nextRiseMs > p1!.nextRiseMs, "second rise must be later");
  const periodMs = (360 / rate) * 1000;
  assert.ok(
    Math.abs(p2!.nextRiseMs - p1!.nextRiseMs - periodMs) < 2000,
    `pass spacing ${p2!.nextRiseMs - p1!.nextRiseMs} vs period ${periodMs}`,
  );
});

test("nextPassOverSite: a window that ends before the rise returns null", () => {
  const sampler = makeEquatorialSampler(-30, 0.05, 550);
  // Rise is ~300s in; a 100s window sees nothing.
  const pass = nextPassOverSite(sampler, 0, 0, {
    startMs: 0,
    windowSec: 100,
    stepSec: 30,
    minElevDeg: 10,
  });
  assert.equal(pass, null);
});

test("nextPassOverSite: a site the ground track never reaches returns null over a full day", () => {
  // Equatorial track, site at 80N: central angle >= 80deg always, far
  // beyond the ~15deg footprint -> never visible.
  const sampler = makeEquatorialSampler(-30, 0.05, 550);
  const pass = nextPassOverSite(sampler, 80, 0, {
    startMs: 0,
    windowSec: 24 * 3600,
    stepSec: 30,
    minElevDeg: 10,
  });
  assert.equal(pass, null);
});

test("nextPassOverSite: includeInProgress reports an already-overhead pass at startMs; default skips to the next rise", () => {
  const rate = 0.05;
  const sampler = makeEquatorialSampler(-30, rate, 550);
  // At t=600s the sub-point is at lon 0 -> directly overhead the (0,0) site.
  const startOverheadMs = 600_000;
  assert.ok(
    isVisible(
      sampler(startOverheadMs).latDeg,
      sampler(startOverheadMs).lonDeg,
      sampler(startOverheadMs).altKm,
      0,
      0,
      10,
    ),
    "precondition: overhead at start",
  );
  const inProgress = nextPassOverSite(sampler, 0, 0, {
    startMs: startOverheadMs,
    windowSec: 8000,
    stepSec: 30,
    minElevDeg: 10,
    includeInProgress: true,
  });
  assert.ok(inProgress !== null);
  assert.equal(inProgress!.nextRiseMs, startOverheadMs);

  const skipped = nextPassOverSite(sampler, 0, 0, {
    startMs: startOverheadMs,
    windowSec: 10_000,
    stepSec: 30,
    minElevDeg: 10,
  });
  assert.ok(skipped !== null);
  // Default: the in-progress pass is skipped; next rise is ~one period later.
  const periodMs = (360 / rate) * 1000;
  assert.ok(
    skipped!.nextRiseMs > startOverheadMs + periodMs / 2,
    `default should skip to next period, got ${skipped!.nextRiseMs}`,
  );
});
