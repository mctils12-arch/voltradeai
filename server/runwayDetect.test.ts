// Runway detection (2026-10-14, human request: which runway a plane takes off
// from / lands on, and "activate the plane for a takeoff when it's on the
// runway"). Fixture = real KCLT runway ends from FAA CIFP cycle 2610.
// Run: npx tsx --test server/runwayDetect.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  runwayEnds, classifyRunwayFix, departureRunway, runwayStateLabel, RUNWAY_HONESTY,
  ROLL_KT, FINAL_MAX_NM, type RunwayFix,
} from "./runwayDetect";

const RWY = (id: string, lat: number, lon: number, bearingMag: number, lengthFt: number) =>
  [id, { id, ll: { lat, lon }, bearingMag, lengthFt }] as const;
const KCLT = {
  magVar: -7,
  runways: new Map([
    RWY("RW01L", 35.20061944444445, -80.96529722222222, 3, 9000),
    RWY("RW01R", 35.19999166666666, -80.95081111111111, 3, 10000),
    RWY("RW18", 35.22473333333333, -80.93615833333334, 183, 8676),
    RWY("RW19L", 35.22739722222222, -80.95316944444444, 183, 10000),
    RWY("RW19R", 35.22528333333334, -80.96741944444445, 183, 9000),
    RWY("RW36", 35.20095555555556, -80.93411666666667, 3, 8676),
  ]),
};
const ELEV = 748;
const ends = runwayEnds(KCLT);
const end = (id: string) => ends.find((e) => e.id === id)!;

/** a point `alongM` down runway `id` from its threshold, `crossM` to the right */
function at(id: string, alongM: number, crossM = 0) {
  const e = end(id);
  const h = (e.headingTrue * Math.PI) / 180;
  const east = alongM * Math.sin(h) + crossM * Math.cos(h);
  const north = alongM * Math.cos(h) - crossM * Math.sin(h);
  return { lat: e.lat + north / 110_540, lon: e.lon + east / (111_320 * Math.cos((e.lat * Math.PI) / 180)) };
}
const fix = (p: { lat: number; lon: number }, o: Partial<RunwayFix>): RunwayFix =>
  ({ ...p, altFt: null, onGround: true, gsKt: 0, trk: 0, ...o });

test("runwayEnds: every coded KCLT end, heading from the opposite threshold, real lengths", () => {
  assert.deepEqual(ends.map((e) => e.id).sort(), ["01L", "01R", "18", "19L", "19R", "36"]);
  for (const e of ends) {
    const opp = ends.find((o) => o.id !== e.id && Math.abs(((e.headingTrue - o.headingTrue + 540) % 360) - 180) > 179 && Math.abs(o.lengthM - e.lengthM) < 1);
    assert.ok(opp, `${e.id} has a reciprocal end`);
  }
  assert.ok(Math.abs(end("36").headingTrue - 356) < 2, "36 runs ~356 true (3 mag, -7 var)");
  assert.ok(Math.abs(end("36").lengthM - 8676 / 3.28084) < 60, "threshold-to-threshold length ~ coded length");
});

test("on the runway: lined up below ROLL_KT, takeoff roll at speed, landing roll when just airborne", () => {
  const p = at("36", 300);
  assert.deepEqual(
    (({ state, runway }) => ({ state, runway }))(classifyRunwayFix(fix(p, { trk: 356, gsKt: 0 }), ends, { elevFt: ELEV })!),
    { state: "LINED_UP", runway: "36" });
  assert.equal(classifyRunwayFix(fix(p, { trk: 356, gsKt: ROLL_KT + 60 }), ends, { elevFt: ELEV })!.state, "TAKEOFF_ROLL");
  assert.equal(classifyRunwayFix(fix(p, { trk: 356, gsKt: 110 }), ends, { elevFt: ELEV, recentlyAirborne: true })!.state, "LANDING_ROLL");
  // same pavement, opposite direction = the reciprocal end
  assert.equal(classifyRunwayFix(fix(at("18", 300), { trk: 176, gsKt: 0 }), ends, { elevFt: ELEV })!.runway, "18");
});

test("parallel runways never collide: the closest centreline wins (01L / 01R / 36)", () => {
  for (const id of ["01L", "01R", "36"]) {
    const m = classifyRunwayFix(fix(at(id, 500, 20), { trk: 356, gsKt: 120 }), ends, { elevFt: ELEV });
    assert.equal(m?.runway, id);
  }
});

test("not on a runway: a taxiway beside it, a cross-heading, or no broadcast track", () => {
  assert.equal(classifyRunwayFix(fix(at("36", 500, 200), { trk: 356, gsKt: 15 }), ends, { elevFt: ELEV }), null, "200 m off = taxiway");
  assert.equal(classifyRunwayFix(fix(at("36", 500), { trk: 90, gsKt: 15 }), ends, { elevFt: ELEV }), null, "crossing the runway");
  assert.equal(classifyRunwayFix(fix(at("36", 500), { trk: null }), ends, { elevFt: ELEV }), null, "no track, no claim");
});

test("airborne: on final before the threshold; departed past the far end; neither without AGL", () => {
  const final = at("19L", -6 * 1852, 120);
  const m = classifyRunwayFix(fix(final, { onGround: false, altFt: ELEV + 1900, trk: 176, gsKt: 140, vsFpm: -700 }), ends, { elevFt: ELEV });
  assert.deepEqual(m && { state: m.state, runway: m.runway }, { state: "ON_FINAL", runway: "19L" });
  assert.equal(classifyRunwayFix(fix(at("19L", -(FINAL_MAX_NM + 1) * 1852), { onGround: false, altFt: ELEV + 4000, trk: 176, gsKt: 160 }), ends, { elevFt: ELEV }), null,
    "beyond the final window");
  const out = at("36", end("36").lengthM + 2 * 1852, -100);
  const d = classifyRunwayFix(fix(out, { onGround: false, altFt: ELEV + 1500, trk: 356, gsKt: 170, vsFpm: 2500 }), ends, { elevFt: ELEV });
  assert.deepEqual(d && { state: d.state, runway: d.runway }, { state: "DEPARTED", runway: "36" });
  assert.equal(classifyRunwayFix(fix(out, { onGround: false, altFt: null, trk: 356 }), ends, { elevFt: ELEV }), null, "no altitude, no airborne claim");
});

test("departureRunway: N843S, KCLT 2026-10-14 — our archive's first fixes say runway 36", () => {
  // the real archived fixes (lat, lon, alt m) from /api/data/track/aircraft/ab8c8e
  const pts: Array<[number, number, number | null]> = [
    [35.2011, -80.9329, null], [35.2144, -80.9353, 198], [35.2152, -80.9354, 206], [35.216, -80.9354, 213],
  ];
  const fixes: RunwayFix[] = pts.map(([la, lo, al], i) => {
    const [la2, lo2] = pts[Math.min(i + 1, pts.length - 1)];
    const [la1, lo1] = i + 1 < pts.length ? [la, lo] : pts[i - 1];
    const trk = (Math.atan2((lo2 - lo1) * Math.cos((la1 * Math.PI) / 180), la2 - la1) * 180 / Math.PI + 360) % 360;
    return { lat: la, lon: lo, altFt: al == null ? null : al * 3.28084, onGround: false, gsKt: 160, trk };
  });
  const dep = departureRunway(fixes, ends, ELEV);
  assert.equal(dep?.runway, "36");
  assert.equal(dep?.state, "TAKEOFF_ROLL");
  assert.equal(runwayStateLabel(dep!), "Taking off — runway 36");
  assert.match(RUNWAY_HONESTY, /not an ATC runway assignment/);
});
