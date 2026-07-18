// rotation tests — IAU pole/prime-meridian model sanity: cited rotation
// periods, retrograde signs, axial tilts, Earth-vs-GMST consistency, and
// the tidal locks FALLING OUT of the real constants (Moon's prime meridian
// faces the Schlyter Earth within real libration). Celestial v2 B3.
// `npx tsx --test`.

import test from "node:test";
import assert from "node:assert/strict";

import {
  ROTATION_IDS,
  hasRotationModel,
  poleRaDecDeg,
  primeMeridianDeg,
  rotationRateDegPerDay,
  rotationPeriodDays,
  axisEclOfDate,
  primeMeridianDirEclOfDate,
  type Vec3,
} from "./rotation.js";
import { moonOrbitPeriodDays, MOON_IDS } from "./moons.js";
import { solarSystemState } from "./solarSystem.js";
import { gmstDeg } from "./ephemeris.js";

const T = Date.UTC(2026, 6, 18, 12);
const angleDeg = (a: Vec3, b: Vec3): number => {
  const d = (a.x * b.x + a.y * b.y + a.z * b.z) /
    (Math.hypot(a.x, a.y, a.z) * Math.hypot(b.x, b.y, b.z));
  return (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI;
};

test("rotation periods match the cited sidereal values (NSSDC/IAU) within 0.1%", () => {
  // {id: period days} — sidereal rotation, NSSDC planetary fact sheets
  const cited: Record<string, number> = {
    sun: 25.38,       // Carrington sidereal
    mercury: 58.6462,
    venus: 243.018,   // retrograde — sign tested separately
    earth: 0.99727,
    moon: 27.3217,
    mars: 1.02596,
    jupiter: 0.41354, // System III
    saturn: 0.44401,
    uranus: 0.71833,  // retrograde
    neptune: 0.66526,
  };
  for (const [id, p] of Object.entries(cited)) {
    assert.ok(hasRotationModel(id), `${id} modeled`);
    const got = rotationPeriodDays(id as never);
    assert.ok(Math.abs(got - p) / p < 0.001, `${id}: ${got.toFixed(5)} vs ${p} d`);
  }
});

test("retrograde rotators carry negative W rates (Venus, Uranus, Triton's W)", () => {
  assert.ok(rotationRateDegPerDay("venus") < 0, "Venus retrograde");
  assert.ok(rotationRateDegPerDay("uranus") < 0, "Uranus retrograde");
  assert.ok(rotationRateDegPerDay("triton") < 0, "Triton W negative (IAU model)");
  assert.ok(rotationRateDegPerDay("earth") > 0);
  assert.ok(rotationRateDegPerDay("jupiter") > 0);
});

test("Earth: IAU W rate is the GMST rate, and the axis is the obliquity axis", () => {
  const w1 = primeMeridianDeg("earth", T);
  const w2 = primeMeridianDeg("earth", T + 86_400_000);
  const wRate = (((w2 - w1) % 360) + 360) % 360;
  const gRate = (((gmstDeg(T + 86_400_000) - gmstDeg(T)) % 360) + 360) % 360;
  assert.ok(Math.abs(wRate - gRate) < 1e-3, `W/day ${wRate} vs GMST/day ${gRate}`);
  // spin axis vs ecliptic north = the obliquity (23.44°)
  const tilt = angleDeg(axisEclOfDate("earth", T), { x: 0, y: 0, z: 1 });
  assert.ok(Math.abs(tilt - 23.44) < 0.5, `Earth tilt ${tilt.toFixed(2)}°`);
});

test("axial tilts to the ecliptic land on the documented values", () => {
  // pole-to-ecliptic-north angles under the IAU north convention (north =
  // north of the invariable plane — Uranus reads 82.2°, the familiar 97.8°
  // obliquity of the opposite-handed convention; retrogradeness is carried
  // by the negative W rate, tested above).
  const northAngle: Record<string, [number, number]> = {
    sun: [7.25, 0.3], moon: [1.54, 0.3], mars: [26.7, 1.0],
    jupiter: [2.2, 0.5], saturn: [28.1, 1.0], uranus: [82.3, 1.0],
    neptune: [28.5, 1.0], venus: [1.2, 1.0],
  };
  for (const [id, [want, tol]] of Object.entries(northAngle)) {
    const got = angleDeg(axisEclOfDate(id as never, T), { x: 0, y: 0, z: 1 });
    assert.ok(Math.abs(got - want) < tol, `${id}: ${got.toFixed(2)}° vs ${want}°`);
  }
});

test("tidal locking falls out of the constants: every curated moon's |W rate| equals its orbital mean motion", () => {
  for (const id of MOON_IDS) {
    assert.ok(hasRotationModel(id), `${id} modeled`);
    const w = Math.abs(rotationRateDegPerDay(id as never));
    const n = 360 / moonOrbitPeriodDays(id);
    assert.ok(Math.abs(w - n) / n < 0.01, `${id}: |W| ${w.toFixed(4)} vs n ${n.toFixed(4)} °/day`);
  }
});

test("the Moon's prime meridian faces the (independent Schlyter) Earth within real libration", () => {
  // optical libration reaches ±8°; measured 5.5–7.4° at these epochs.
  // 10° catches any W-convention or frame error (those give 90–180°).
  for (const t of [Date.UTC(2000, 5, 15), Date.UTC(2013, 2, 7), Date.UTC(2026, 6, 18)]) {
    const s = solarSystemState(t).find((b) => b.id === "moon")!;
    const toEarth = { x: -s.geoAu.x, y: -s.geoAu.y, z: -s.geoAu.z };
    const a = angleDeg(primeMeridianDirEclOfDate("moon", t), toEarth);
    assert.ok(a < 10, `${new Date(t).toISOString().slice(0, 10)}: ${a.toFixed(2)}°`);
  }
});

test("model surface: every id resolves, W wraps to [0,360), axes are unit, pole series move the pole", () => {
  for (const id of ROTATION_IDS) {
    const w = primeMeridianDeg(id, T);
    assert.ok(w >= 0 && w < 360, `${id} W in range (${w})`);
    const ax = axisEclOfDate(id, T);
    assert.ok(Math.abs(Math.hypot(ax.x, ax.y, ax.z) - 1) < 1e-9, `${id} unit axis`);
    const pm = primeMeridianDirEclOfDate(id, T);
    assert.ok(Math.abs(Math.hypot(pm.x, pm.y, pm.z) - 1) < 1e-9, `${id} unit PM dir`);
    // PM direction ⊥ axis (it lies in the equator plane)
    assert.ok(Math.abs(ax.x * pm.x + ax.y * pm.y + ax.z * pm.z) < 1e-9, `${id} PM ⊥ axis`);
    const p = poleRaDecDeg(id, T);
    assert.ok(Number.isFinite(p.raDeg) && Number.isFinite(p.decDeg), `${id} finite pole`);
  }
  // Triton's pole cone is real: the pole moves degrees across decades
  const a = axisEclOfDate("triton", Date.UTC(2000, 0, 1));
  const b = axisEclOfDate("triton", Date.UTC(2026, 0, 1));
  assert.ok(angleDeg(a, b) > 1, "Triton pole precesses visibly across 26 years");
});

test("rotation advances at the stated rate between nearby instants", () => {
  // one hour of Jupiter: 870.536/24 = 36.27° — pinned through the wrap
  const w1 = primeMeridianDeg("jupiter", T);
  const w2 = primeMeridianDeg("jupiter", T + 3_600_000);
  const step = (((w2 - w1) % 360) + 360) % 360;
  assert.ok(Math.abs(step - 870.536 / 24) < 0.01, `Jupiter/hour ${step.toFixed(3)}°`);
});
