// Celestial ephemeris — EARTH TWIN O6-7 tier 1. Pinned against known
// astronomical anchors (solstices/equinox, documented new/full moons near
// J2000 where the truncated series is most exact) + physical invariants.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { subsolarPoint, moonState, moonPhaseGlyph, nightPolygon, gmstDeg } from './ephemeris.js';

test('subsolar point: solstice/equinox declinations + noon-UTC longitude', () => {
  const jun = subsolarPoint(Date.UTC(2026, 5, 21, 12, 0, 0)); // June solstice 2026
  assert.ok(Math.abs(jun.latDeg - 23.43) < 0.2, `June solstice declination (got ${jun.latDeg.toFixed(2)})`);
  const dec = subsolarPoint(Date.UTC(2026, 11, 21, 12, 0, 0));
  assert.ok(Math.abs(dec.latDeg + 23.43) < 0.2, `December solstice (got ${dec.latDeg.toFixed(2)})`);
  const equinox = subsolarPoint(Date.UTC(2026, 2, 20, 12, 0, 0)); // 2026-03-20
  assert.ok(Math.abs(equinox.latDeg) < 0.6, `equinox declination ~0 (got ${equinox.latDeg.toFixed(2)})`);
  // at 12:00 UTC the sun is near the Greenwich meridian (± equation of time ~4°)
  assert.ok(Math.abs(jun.lonDeg) < 6, `noon UTC subsolar lon near 0 (got ${jun.lonDeg.toFixed(2)})`);
  const midnight = subsolarPoint(Date.UTC(2026, 5, 21, 0, 0, 0));
  assert.ok(Math.abs(Math.abs(midnight.lonDeg) - 180) < 6, 'midnight UTC subsolar lon near the antimeridian');
});

test('moon: documented J2000-era anchors (new 2000-01-06, full 2000-01-21) + physical bounds', () => {
  const newMoon = moonState(Date.UTC(2000, 0, 6, 18, 14, 0));
  assert.ok(newMoon.illuminatedFraction < 0.05, `documented new moon dark (got ${newMoon.illuminatedFraction.toFixed(3)})`);
  const fullMoon = moonState(Date.UTC(2000, 0, 21, 4, 40, 0)); // total lunar eclipse night — full by definition
  assert.ok(fullMoon.illuminatedFraction > 0.95, `documented full moon lit (got ${fullMoon.illuminatedFraction.toFixed(3)})`);
  for (const t of [Date.UTC(2026, 6, 16), Date.UTC(2026, 0, 1), Date.UTC(2026, 9, 10)]) {
    const m = moonState(t);
    assert.ok(m.angularSizeArcmin > 28 && m.angularSizeArcmin < 34.5, `angular size in the real band (got ${m.angularSizeArcmin.toFixed(1)}')`);
    assert.ok(m.distanceKm > 356_000 && m.distanceKm < 407_000, 'distance within perigee..apogee');
    assert.ok(m.illuminatedFraction >= 0 && m.illuminatedFraction <= 1);
    assert.ok(Math.abs(m.latDeg) < 29, 'sublunar latitude bounded by inclination+obliquity');
  }
});

test('phase glyph: 8 canonical steps, waxing/waning sides distinct', () => {
  assert.equal(moonPhaseGlyph(0.0, true), '🌑');
  assert.equal(moonPhaseGlyph(1.0, true), '🌕');
  assert.equal(moonPhaseGlyph(0.25, true), '🌒');
  assert.equal(moonPhaseGlyph(0.25, false), '🌘');
  assert.equal(moonPhaseGlyph(0.5, true), '🌓');
  assert.equal(moonPhaseGlyph(0.5, false), '🌗');
  assert.equal(moonPhaseGlyph(0.8, true), '🌔');
  assert.equal(moonPhaseGlyph(0.8, false), '🌖');
});

test('night polygon: every terminator vertex ~90° from the subsolar point; closes over the dark pole', () => {
  const t = Date.UTC(2026, 5, 21, 12, 0, 0); // June solstice noon
  const sun = subsolarPoint(t);
  const feat = nightPolygon(t) as any;
  const ring: [number, number][] = feat.geometry.coordinates[0];
  const gc = (aLat: number, aLon: number, bLat: number, bLon: number) => {
    const A = Math.PI / 180;
    return Math.acos(Math.max(-1, Math.min(1,
      Math.sin(aLat * A) * Math.sin(bLat * A) +
      Math.cos(aLat * A) * Math.cos(bLat * A) * Math.cos((aLon - bLon) * A)))) / A;
  };
  let checked = 0;
  for (const [lon, lat] of ring.slice(0, 181)) {
    if (Math.abs(lat) >= 84.9) continue; // clamped vertices are the stated display constraint
    const d = gc(lat, lon, sun.latDeg, sun.lonDeg);
    assert.ok(Math.abs(d - 90) < 1.0, `terminator vertex ${lon.toFixed(0)}°: ${d.toFixed(2)}° from subsolar`);
    checked++;
  }
  assert.ok(checked > 100, 'most vertices actually checked');
  // June: the SOUTH pole is dark — the closure runs along -85
  assert.ok(ring.some(([, lat]) => lat === -85), 'closes over the dark (southern) pole in June');
  assert.ok(gmstDeg(t) >= 0 && gmstDeg(t) < 360);
});
