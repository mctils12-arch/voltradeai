// Hermetic tests for Starlink ground-site coverage queries (pure geometry,
// no DOM/WebGL/map). Run: npx tsx --test client/src/lib/orbital/siteQuery.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { siteCoverageReport, isStarlinkName } from './siteQuery.ts';
import { lonLatToMercator, CLASS_CODE, SENTINEL_SKIP } from './satBuffer.ts';
import type { GpRecord } from './tle.ts';

function gp(noradId: number, name: string): GpRecord {
  return {
    noradId, name, epoch: null, inclination: null, raan: null, ecc: null,
    argp: null, meanAnomaly: null, meanMotion: null, bstar: null,
  };
}

/** Pack one satellite slot at a given geodetic position + altitude (km). */
function slot(latDeg: number, lonDeg: number, altKm: number, classCode: number): number[] {
  const { x, y } = lonLatToMercator(lonDeg, latDeg);
  return [x, y, altKm * 1000, classCode];
}

test('isStarlinkName: matches the CelesTrak naming convention, not substrings elsewhere in a name', () => {
  assert.equal(isStarlinkName('STARLINK-1007'), true);
  assert.equal(isStarlinkName('starlink-30512'), true);
  assert.equal(isStarlinkName('ISS (ZARYA)'), false);
  assert.equal(isStarlinkName('NOT STARLINK-1'), false); // must anchor at the start
  assert.equal(isStarlinkName(null), false);
  assert.equal(isStarlinkName(undefined), false);
});

test('siteCoverageReport: a Starlink directly overhead the site is visible; a non-Starlink overhead is ignored', () => {
  const siteLat = 30, siteLon = -95; // Houston-ish
  const positions = new Float32Array([
    ...slot(30, -95, 550, CLASS_CODE.LEO), // i=0 STARLINK, overhead
    ...slot(30, -95, 550, CLASS_CODE.LEO), // i=1 non-Starlink, overhead — must be excluded by name filter
  ]);
  const gps = [gp(1, 'STARLINK-4201'), gp(2, 'ONEWEB-0123')];
  const report = siteCoverageReport(positions, 4, gps, siteLat, siteLon);
  assert.equal(report.totalModeled, 1);
  assert.equal(report.visible.length, 1);
  assert.equal(report.visible[0].noradId, 1);
  assert.ok(report.visible[0].elevationDeg > 80); // overhead => near 90deg
});

test('siteCoverageReport: a Starlink far below the horizon does not count as coverage (honest blackout)', () => {
  const siteLat = 30, siteLon = -95;
  const positions = new Float32Array([
    ...slot(-30, 85, 550, CLASS_CODE.LEO), // antipodal-ish — well below horizon
  ]);
  const gps = [gp(1, 'STARLINK-1000')];
  const report = siteCoverageReport(positions, 4, gps, siteLat, siteLon);
  assert.equal(report.totalModeled, 1); // modeled (valid position) but not visible
  assert.equal(report.visible.length, 0); // a genuine blackout, not "nothing matched"
});

test('siteCoverageReport: deep-space/invalid (SENTINEL_SKIP) Starlink slots are excluded from totalModeled, not counted as absent coverage', () => {
  const siteLat = 30, siteLon = -95;
  const positions = new Float32Array([
    ...slot(30, -95, 550, SENTINEL_SKIP), // this tick's propagation failed for this member
  ]);
  const gps = [gp(1, 'STARLINK-9999')];
  const report = siteCoverageReport(positions, 4, gps, siteLat, siteLon);
  assert.equal(report.totalModeled, 0);
  assert.equal(report.visible.length, 0);
});

test('siteCoverageReport: respects a custom elevation mask and name predicate', () => {
  const siteLat = 0, siteLon = 0;
  // 5deg lateral offset at 550km altitude works out to ~41deg elevation
  // (verified directly against elevationAzimuthFromSite) — clears a lenient
  // 5deg mask but not a strict 70deg one, proving the mask is honored rather
  // than just "visible at all".
  const positions = new Float32Array([
    ...slot(5, 0, 550, CLASS_CODE.MEO),
  ]);
  const gps = [gp(7, 'GPS BIIF-3')];
  const namePredicate = (n: string | null) => typeof n === 'string' && n.startsWith('GPS');

  const strict = siteCoverageReport(positions, 4, gps, siteLat, siteLon, {
    namePredicate, minElevDeg: 70,
  });
  assert.equal(strict.totalModeled, 1);
  assert.equal(strict.visible.length, 0); // present, but doesn't clear the strict mask

  const lenient = siteCoverageReport(positions, 4, gps, siteLat, siteLon, {
    namePredicate, minElevDeg: 5,
  });
  assert.equal(lenient.visible.length, 1);
});

test('siteCoverageReport: sorts visible members by elevation descending (most-overhead first)', () => {
  const siteLat = 40, siteLon = -75;
  const positions = new Float32Array([
    ...slot(40, -70, 550, CLASS_CODE.LEO), // farther east — lower elevation
    ...slot(40, -75, 550, CLASS_CODE.LEO), // directly overhead — highest elevation
  ]);
  const gps = [gp(1, 'STARLINK-1'), gp(2, 'STARLINK-2')];
  const report = siteCoverageReport(positions, 4, gps, siteLat, siteLon);
  assert.equal(report.visible.length, 2);
  assert.equal(report.visible[0].noradId, 2); // overhead one ranks first
  assert.ok(report.visible[0].elevationDeg > report.visible[1].elevationDeg);
});

test('siteCoverageReport: caps the search at min(gp.length, buffer slots) and skips a missing/null gp entry', () => {
  const positions = new Float32Array([
    ...slot(0, 0, 550, CLASS_CODE.LEO),
    ...slot(0, 0, 550, CLASS_CODE.LEO),
  ]);
  const gps = [gp(1, 'STARLINK-1')]; // buffer has 2 slots, gp only has 1 — must not throw
  const report = siteCoverageReport(positions, 4, gps, 0, 0);
  assert.equal(report.totalModeled, 1);
  assert.equal(report.visible.length, 1);
});

// ── coverageQueryAllowed: the O7 click-routing guard ─────────────────────────
// Regression for the live bug where clicking the LA port marker surfaced the
// "Starlink coverage" card instead of the port: the coverage fallback may only
// claim clicks on genuinely empty ground (raster-only basemap ⇒ any rendered
// vector feature at the point belongs to a data layer that owns the click).
import { coverageQueryAllowed } from './siteQuery.ts';

test('coverageQueryAllowed: empty ground (no rendered features) → coverage card allowed', () => {
  assert.equal(coverageQueryAllowed([]), true);
});

test('coverageQueryAllowed: any rendered feature under the click → coverage card suppressed', () => {
  assert.equal(coverageQueryAllowed([{ layer: { id: 'portdwell-labels' } }]), false);
  assert.equal(coverageQueryAllowed([
    { layer: { id: 'powergrid_hifld_plants-pt' } },
    { layer: { id: 'powergrid_hifld_sub-pt' } },
  ]), false);
});
