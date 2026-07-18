// moons tests — element sanity against cited values, sign/frame validation
// via the Moon-row cross-check against the independent Schlyter ephemeris,
// orbit geometry (range, closure, retrograde Triton), and the honesty of
// the declared table (celestial v2 B3). `npx tsx --test`.

import test from "node:test";
import assert from "node:assert/strict";

import {
  MOON_IDS,
  MOONS,
  MOON_EPOCH_MS,
  MOON_COLOR,
  MOON_NAME,
  moonRates,
  moonOrbitPeriodDays,
  moonLocalOffsetEclM,
  keplerLaplaceOffsetEclM,
  type MoonElements,
  type Vec3,
} from "./moons.js";
import { solarSystemState, AU_M } from "./solarSystem.js";

const len = (v: Vec3): number => Math.hypot(v.x, v.y, v.z);
const angleDeg = (a: Vec3, b: Vec3): number => {
  const d = (a.x * b.x + a.y * b.y + a.z * b.z) / (len(a) * len(b));
  return (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI;
};

/** The JPL table's own Earth-Moon row (mean ECLIPTIC elements — reference
 *  plane is the ecliptic, so poleRa/Dec are null). Used ONLY to validate
 *  the propagation recipe against the independent Schlyter ephemeris —
 *  the Moon's production ephemeris stays solarSystem.ts (one ephemeris,
 *  never two). */
const JPL_MOON_ROW: MoonElements = {
  parent: "earth", aKm: 384400, e: 0.0554, wDeg: 318.15, MDeg: 135.27,
  iDeg: 5.16, nodeDeg: 125.08, PDays: 27.322, PapsisYr: 5.997, PnodeYr: 18.600,
  poleRaDeg: null, poleDecDeg: null, radiusKm: 1737.4,
};

/** Familiar sidereal orbital periods, days (NSSDC planetary satellite fact
 *  sheets) — the citable values the task's 1% sanity bar refers to. */
const FAMILIAR_PERIOD_DAYS: Record<string, number> = {
  phobos: 0.31891, deimos: 1.26244,
  io: 1.769138, europa: 3.551181, ganymede: 7.154553, callisto: 16.689017,
  titan: 15.945421, triton: 5.876854,
};

test("sign conventions: the recipe reproduces Schlyter's independent lunar rates", () => {
  // Schlyter §4 Moon rates (deg/day): N −0.0529538083, w +0.1643573223,
  // M +13.0649929509. The JPL row's derived rates must land on them —
  // this pins BOTH precession signs and the L = node+w+M decomposition.
  const r = moonRates(JPL_MOON_ROW);
  assert.ok(Math.abs(r.dNode - -0.0529538083) < 2e-4, `node regression ${r.dNode}`);
  assert.ok(Math.abs(r.dW - 0.1643573223) < 2e-4, `apsidal advance ${r.dW}`);
  assert.ok(Math.abs(r.dM - 13.0649929509) < 5e-4, `anomalistic rate ${r.dM}`);
});

test("frame validation: the JPL Moon row lands on the Schlyter Moon within the mean-vs-perturbed gap", () => {
  // Perturbations the mean elements omit reach ~1.3° (evection) + rate
  // rounding accumulates ~0.03°/yr — measured 0.46°/1.5°/2.6° at these
  // epochs. 3° would catch any sign, frame, or node-convention error
  // (those produce tens of degrees).
  for (const t of [Date.UTC(2000, 5, 15), Date.UTC(2010, 0, 1), Date.UTC(2026, 6, 18)]) {
    const jpl = keplerLaplaceOffsetEclM(JPL_MOON_ROW, t);
    const s = solarSystemState(t).find((b) => b.id === "moon")!;
    const geo = { x: s.geoAu.x * AU_M, y: s.geoAu.y * AU_M, z: s.geoAu.z * AU_M };
    assert.ok(angleDeg(jpl, geo) < 3, `${new Date(t).toISOString().slice(0, 10)}: ${angleDeg(jpl, geo).toFixed(2)}°`);
    assert.ok(Math.abs(len(jpl) - len(geo)) / len(geo) < 0.02, "radial distance within 2%");
  }
});

test("periods: each curated moon's closed-orbit period is within 1% of the cited sidereal period", () => {
  for (const id of MOON_IDS) {
    const p = moonOrbitPeriodDays(id);
    const fam = FAMILIAR_PERIOD_DAYS[id];
    assert.ok(Math.abs(p - fam) / fam < 0.01, `${id}: ${p.toFixed(6)} vs ${fam} d`);
  }
});

test("orbit geometry: offsets stay inside the ellipse's radial range at all phases", () => {
  const t0 = Date.UTC(2026, 6, 18);
  for (const id of MOON_IDS) {
    const el = MOONS[id];
    const P = moonOrbitPeriodDays(id) * 86_400_000;
    for (let k = 0; k < 12; k++) {
      const r = len(moonLocalOffsetEclM(id, t0 + (k / 12) * P));
      const lo = el.aKm * 1000 * (1 - el.e) * 0.995;
      const hi = el.aKm * 1000 * (1 + el.e) * 1.005;
      assert.ok(r >= lo && r <= hi, `${id} phase ${k}: r=${(r / 1000).toFixed(0)}km outside [${lo / 1000}, ${hi / 1000}]`);
    }
  }
});

test("closure: one anomalistic period returns each moon to its start, up to the cited precession", () => {
  const t0 = Date.UTC(2026, 6, 18);
  for (const id of MOON_IDS) {
    const Pd = moonOrbitPeriodDays(id);
    const a = moonLocalOffsetEclM(id, t0);
    const b = moonLocalOffsetEclM(id, t0 + Pd * 86_400_000);
    // The only physical non-closure is the ellipse's own precession over
    // one period (Io's apsis moves fastest: 1.308°/orbit at P_apsis
    // 1.333 yr) — bound the gap by the chord of that rotation plus a
    // small numeric slack. A sign/frame bug would blow through this by
    // orders of magnitude.
    const r = moonRates(MOONS[id]);
    const precessDeg = (Math.abs(r.dW) + Math.abs(r.dNode)) * Pd;
    const bound = MOONS[id].aKm * 1000 * ((precessDeg * Math.PI) / 180 + 0.005);
    const gap = len({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
    assert.ok(gap < bound, `${id}: gap ${(gap / 1000).toFixed(0)} km vs bound ${(bound / 1000).toFixed(0)} km`);
  }
});

test("Triton is retrograde, everything else prograde (ecliptic angular momentum)", () => {
  const t0 = Date.UTC(2026, 6, 18);
  for (const id of MOON_IDS) {
    const a = moonLocalOffsetEclM(id, t0);
    const b = moonLocalOffsetEclM(id, t0 + 60_000);
    const hz = a.x * b.y - a.y * b.x; // z of r × Δr
    if (id === "triton") assert.ok(hz < 0, "Triton orbits retrograde");
    else assert.ok(hz > 0, `${id} orbits prograde`);
  }
});

test("Galilean ladder: semi-major axes order Io < Europa < Ganymede < Callisto, and the table is complete", () => {
  assert.ok(MOONS.io.aKm < MOONS.europa.aKm);
  assert.ok(MOONS.europa.aKm < MOONS.ganymede.aKm);
  assert.ok(MOONS.ganymede.aKm < MOONS.callisto.aKm);
  assert.equal(MOON_IDS.length, 8, "the directive's curated eight");
  for (const id of MOON_IDS) {
    const el = MOONS[id];
    assert.ok(el.aKm > 0 && el.PDays > 0 && el.radiusKm > 0, `${id} physical`);
    assert.ok(el.e >= 0 && el.e < 0.05, `${id} near-circular family (e=${el.e})`);
    assert.ok(/^#[0-9a-f]{6}$/i.test(MOON_COLOR[id]), `${id} presentation color`);
    assert.ok(MOON_NAME[id].length > 1, `${id} display name`);
  }
  assert.equal(MOON_EPOCH_MS, Date.UTC(2000, 0, 1, 12), "table epoch 2000-01-01.5");
});

test("motion is continuous and epoch-anchored: position at the epoch matches the epoch elements", () => {
  // at t = epoch, M = M0 exactly — radial distance follows from (a, e, M0)
  const el = MOONS.titan;
  const r0 = len(moonLocalOffsetEclM("titan", MOON_EPOCH_MS));
  // E from M0=11.7°, e=0.029: r = a(1 − e·cos E) — a coarse closed-form check
  const E = (11.7 * Math.PI) / 180 + el.e * Math.sin((11.7 * Math.PI) / 180);
  const expected = el.aKm * 1000 * (1 - el.e * Math.cos(E));
  assert.ok(Math.abs(r0 - expected) / expected < 0.001, `Titan epoch radius ${(r0 / 1000).toFixed(0)} km`);
  // and a minute of motion is a sane fraction of the orbit
  const b = moonLocalOffsetEclM("titan", MOON_EPOCH_MS + 60_000);
  const chord = len({ x: r0 * 0 + b.x - moonLocalOffsetEclM("titan", MOON_EPOCH_MS).x, y: b.y - moonLocalOffsetEclM("titan", MOON_EPOCH_MS).y, z: b.z - moonLocalOffsetEclM("titan", MOON_EPOCH_MS).z });
  const expectedChord = (2 * Math.PI * el.aKm * 1000 * (60 / 86400)) / el.PDays;
  assert.ok(chord > 0.5 * expectedChord && chord < 2 * expectedChord, "per-minute motion at the orbital rate");
});
