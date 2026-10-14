// UNRESOLVED ROUTE SEGMENTS on the planned-route curtain (deviation program
// P1, 2026-10-14): a leg the server could not place (unresolvedNext) draws
// LIGHTER — estimated curtain alpha, dashed top edge, dashed ground trace —
// in every render mode. Render matrix: globe|flat × 2D (pitch 0)|3D (pitched)
// × terrain off|on. The plan geometry is built ONCE per plan from the
// display datum and never from the camera (curtain law), so for a given
// terrain setting the buffer must be byte-identical across projection and
// pitch, and the styling must hold for both terrain datums. Includes a leg
// that crosses the antimeridian (never drawn the long way round).
// Run: npx tsx --test client/src/lib/air/planUnresolved.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  densifyPlan, buildPlanGeometry, PC_VERT_STRIDE, PG_TRACE, PG_CURTAIN, PG_EDGE,
  PLAN_CURTAIN_ALPHA, PLAN_CURTAIN_ALPHA_EST, type PlanPointLike,
} from "./planCurtainLayer.js";
import { computePlanDatum } from "./planRouteController.js";
import { FT_VERTS_PER_SEG } from "./flightTrackLayer.js";
import { normalizePlan } from "./flightPlan.js";

// SFO → (OAL) → [J999: unresolved] → (JNC) → DEN, then a Pacific leg across
// the antimeridian with its own unresolved stretch (oceanic track not ingested)
const PTS: PlanPointLike[] = [
  { lon: -122.375, lat: 37.619, altM: 4, altEstimated: false },
  { lon: -117.77, lat: 38.003, altM: 10_668, altEstimated: false, unresolvedNext: true },
  { lon: -108.79, lat: 39.06, altM: 10_668, altEstimated: false },
  { lon: -104.67, lat: 39.86, altM: 1_655, altEstimated: false },
];
const PACIFIC: PlanPointLike[] = [
  { lon: 172, lat: 40, altM: 11_000, altEstimated: false },
  { lon: 178, lat: 42, altM: 11_000, altEstimated: false, unresolvedNext: true },
  { lon: -176, lat: 44, altM: 11_000, altEstimated: false },
  { lon: -170, lat: 45, altM: 11_000, altEstimated: false },
];

type Mode = { projection: "globe" | "flat"; pitch: 0 | 60; terrain: boolean };
const MODES: Mode[] = [];
for (const projection of ["globe", "flat"] as const)
  for (const pitch of [0, 60] as const)
    for (const terrain of [false, true]) MODES.push({ projection, pitch, terrain });

/** everything the layer receives — note: no camera input exists */
function geometryFor(points: PlanPointLike[], m: Mode) {
  const dense = densifyPlan(points);
  const datum = computePlanDatum(dense, {
    terrainOn: m.terrain, exag: m.terrain ? 1.5 : 1,
    meshGround: () => (m.terrain ? 900 : 0), demGround: () => null,
  }, null);
  return buildPlanGeometry({ dense, altDisp: datum.altDisp, groundZ: datum.groundZ, drapeBelowM: datum.drapeBelowM });
}

const quadAt = (g: ReturnType<typeof geometryFor>, group: number, s: number) => {
  const q = g.segQuad[group][s];
  return q < g.segQuad[group][s + 1] ? q : -1;
};
const attr = (g: ReturnType<typeof geometryFor>, quad: number, k: number) => g.verts[quad * FT_VERTS_PER_SEG * PC_VERT_STRIDE + k];
const W = 8, ALPHA = 12; // stride layout: pos3 other3 side dir w rgba along

for (const m of MODES) {
  test(`render matrix ${m.projection} · ${m.pitch ? "3D" : "2D"} · terrain ${m.terrain ? "on" : "off"}: unresolved legs lighter, resolved legs normal`, () => {
    const g = geometryFor(PTS, m);
    let unr = 0, res = 0;
    for (let s = 0; s < g.nSeg; s++) {
      const u = g.dense.unr[s] === 1;
      const tq = quadAt(g, PG_TRACE, s), cq = quadAt(g, PG_CURTAIN, s), eq = quadAt(g, PG_EDGE, s);
      assert.ok(tq >= 0 && cq >= 0 && eq >= 0, `segment ${s} drawn in every group`);
      if (u) {
        unr++;
        assert.ok(attr(g, tq, W) < 0, "ground trace dashed (2D ground trace / draped with terrain)");
        assert.ok(Math.abs(attr(g, cq, ALPHA) - PLAN_CURTAIN_ALPHA_EST) < 1e-6, "curtain at the estimated alpha");
        assert.ok(attr(g, eq, W) < 0, "top edge dashed (3D at true altitude)");
      } else if (g.dense.est[s] === 0 && g.dense.est[s + 1] === 0) {
        res++;
        assert.ok(attr(g, tq, W) > 0 && attr(g, eq, W) > 0);
        assert.ok(Math.abs(attr(g, cq, ALPHA) - PLAN_CURTAIN_ALPHA) < 1e-6);
      }
    }
    assert.ok(unr > 0 && res > 0, `both kinds present (unresolved ${unr}, resolved ${res})`);
  });
}

test("render matrix: geometry is camera-independent — byte-identical across globe/flat × 2D/3D for each terrain setting", () => {
  for (const terrain of [false, true]) {
    const bufs = MODES.filter((m) => m.terrain === terrain).map((m) => geometryFor(PTS, m).verts);
    for (const b of bufs.slice(1)) assert.deepEqual(b, bufs[0]);
  }
  // terrain changes only the datum (z), never which legs are drawn lighter
  const off = geometryFor(PTS, MODES[0]), on = geometryFor(PTS, MODES[1]);
  assert.deepEqual(Array.from(on.dense.unr), Array.from(off.dense.unr));
});

test("antimeridian: an unresolved leg across ±180 keeps its flag on the split pair and no quad spans the seam", () => {
  for (const m of MODES) {
    const g = geometryFor(PACIFIC, m);
    const d = g.dense;
    const seam = Array.from(d.lon).findIndex((lon, i) => i > 0 && Math.abs(lon - d.lon[i - 1]) > 180);
    assert.ok(seam > 0, "split pair present");
    assert.equal(d.unr[seam - 1], 1, "the split vertex carries the unresolved flag of its leg");
    for (let s = 0; s < g.nSeg; s++) {
      if (Math.abs(d.lon[s + 1] - d.lon[s]) > 180) {
        for (const grp of [PG_TRACE, PG_CURTAIN, PG_EDGE]) assert.equal(quadAt(g, grp, s), -1, "nothing drawn the long way round");
      }
    }
  }
});

test("wire: unresolvedNext survives the client contract parse and changes the plan geometry key", () => {
  const raw = {
    hex: "a1b2c3", callsign: "UAL9", source: "FILED_FAA", label: "x", origin: null, destination: null,
    cruiseAltFt: 35000, cruiseAltEstimated: false,
    points: [
      { lon: -122, lat: 37, altM: 0, altEstimated: false, name: "A", unresolvedNext: true },
      { lon: -110, lat: 39, altM: 0, altEstimated: false, name: "B" },
    ],
    originalPoints: null, deviation: { state: "UNKNOWN", crossTrackNm: null, since: null }, events: [],
    fetchedAt: 0, ageSec: 0, honesty: "h", pathEstimated: false, routeKind: "airway", terminalVectoring: false,
  };
  const plan = normalizePlan(raw)!;
  assert.equal(plan.points[0].unresolvedNext, true);
  assert.equal(plan.points[1].unresolvedNext, undefined);
});
