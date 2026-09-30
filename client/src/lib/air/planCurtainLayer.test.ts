// Planned-route (gray) curtain — pins the FLIGHT PROGRAM contract:
// geometry built from the shared plan contract, the seam at the live
// curtain's drawn end, estimated-altitude flagging (fainter + dashed edge),
// antimeridian discipline, Law IV (dispose frees everything, declared caps
// hold), NONE draws nothing, the destination label text per source, the
// crossfade, and the controller's fetch/abort/teardown lifecycle.
// Run: npx tsx --test client/src/lib/air/planCurtainLayer.test.ts
import { test } from 'node:test';
import type { CustomRenderMethodInput } from 'maplibre-gl';
import assert from 'node:assert/strict';
import {
  PlanCurtainLayer,
  densifyPlan,
  buildPlanGeometry,
  buildSeamVertices,
  buildOriginalLineVertices,
  locateOnPlan,
  crossfadeEase,
  projectMercToScreen,
  planWorstCaseBytes,
  PC_VERT_SRC,
  PC_FRAG_SRC,
  PC_VERT_STRIDE,
  PLAN_MAX_POINTS,
  PLAN_CROSSFADE_MS,
  PLAN_CURTAIN_ALPHA,
  PLAN_CURTAIN_ALPHA_EST,
  PLAN_EDGE_WIDTH_PX,
  PLAN_GRAY,
  PLAN_BOTTOM_MUL,
  PG_TRACE,
  PG_CURTAIN,
  PG_EDGE,
  maxFeatures,
  vramBudget,
  type PlanGeometry,
  type PlanSeam,
} from './planCurtainLayer.js';
import { FT_VERTS_PER_SEG, FlightTrackLayer } from './flightTrackLayer.js';
import { TRACE_ABOVE_TERRAIN_M, CURTAIN_BELOW_TERRAIN_M } from './trackModel.js';
import {
  normalizePlan,
  planDestLabel,
  planQueryString,
  deviationText,
  fmtAgeShort,
  planGeometryKey,
  shouldDeviationRefetch,
  shouldCallsignRefetch,
  PLAN_CALLSIGN_REFETCH_MIN_GAP_MS,
  PLAN_DEVIATION_REFETCH_M,
  PLAN_REFRESH_MS,
  type FlightPlan,
} from './flightPlan.js';
import {
  startPlanRoute,
  createPlanRouteStore,
  computePlanDatum,
  fillAlongRoute,
  PLAN_LAYER_ID,
  PLAN_DEM_RADIUS_M,
} from './planRouteController.js';
import { lonLatToMercator } from '../orbital/satBuffer.js';
import { fmtKm } from '../units.js';
import { FrameLoop, type FrameHost } from '../../render/frameCore.js';
import { fitsBudget, budgetBytes, verifyLayerContract } from '../../render/layerContract.js';

// ── fixtures (the shared API contract, verbatim field names) ────────────────

const SFO = { icao: 'KSFO', iata: 'SFO', name: 'San Francisco Intl', lat: 37.619, lon: -122.375, elevM: 4 };
const LAX = { icao: 'KLAX', iata: 'LAX', name: 'Los Angeles Intl', lat: 33.942, lon: -118.408, elevM: 38 };
const SIN = { icao: 'WSSS', iata: 'SIN', name: 'Singapore Changi', lat: 1.364, lon: 103.991, elevM: 7 };

const filedWire = {
  hex: 'a1b2c3', callsign: 'UAL1',
  source: 'FILED_FAA', label: 'FILED — FAA SWIM flight plan',
  origin: SFO, destination: LAX,
  cruiseAltFt: 37000, cruiseAltEstimated: false,
  points: [
    { lon: -122.375, lat: 37.619, altM: 4, altEstimated: true, name: 'SFO' },
    { lon: -121.2, lat: 36.8, altM: 11278, altEstimated: false, name: 'WP1' },
    { lon: -119.9, lat: 35.4, altM: 11278, altEstimated: false, name: 'WP2' },
    { lon: -118.408, lat: 33.942, altM: 38, altEstimated: true, name: 'LAX' },
  ],
  originalPoints: null,
  deviation: { state: 'ON_PLAN', crossTrackNm: 0.4, since: null },
  events: [],
  fetchedAt: 1759000000000, ageSec: 12,
  honesty: 'Route and cruise altitude are FILED (FAA); climb/descent altitudes are estimated.',
};

const predictedWire = {
  ...filedWire,
  hex: 'b2c3d4', callsign: 'SIA31',
  source: 'ROUTE_DB_PREDICTED',
  label: 'PREDICTED — usual route for SIA31 (adsb.lol route DB), great-circle path',
  origin: SFO, destination: SIN,
  cruiseAltFt: null, cruiseAltEstimated: true,
  points: [
    { lon: SFO.lon, lat: SFO.lat, altM: 4, altEstimated: true, name: 'SFO' },
    { lon: SIN.lon, lat: SIN.lat, altM: 7, altEstimated: true, name: 'SIN' },
  ],
  deviation: { state: 'UNKNOWN', crossTrackNm: null, since: null },
};

const noneWire = {
  hex: 'c3d4e5', callsign: 'N123AB', source: 'NONE', label: 'No flight plan available',
  origin: null, destination: null, cruiseAltFt: null, cruiseAltEstimated: true,
  points: [], originalPoints: null,
  deviation: { state: 'UNKNOWN', crossTrackNm: null, since: null },
  events: [], fetchedAt: null, ageSec: null,
  honesty: 'No filed plan and no route history for this callsign — nothing is drawn.',
};

const plan = (w: unknown): FlightPlan => normalizePlan(w) as FlightPlan;

/** flat datum (terrain off, sea-level ground): display alt = MSL. */
const flatGeom = (p: FlightPlan): PlanGeometry => {
  const dense = densifyPlan(p.points);
  return buildPlanGeometry({
    dense,
    altDisp: Float32Array.from(dense.altM),
    groundZ: new Float32Array(dense.n),
    drapeBelowM: 0,
  });
};

const vert = (v: Float32Array, i: number) => Array.from(v.subarray(i * PC_VERT_STRIDE, (i + 1) * PC_VERT_STRIDE));
const near = (a: number, b: number, tol: number, msg: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} (${a} vs ${b})`);

// ── manual frame loop + fake GL ─────────────────────────────────────────────

function manualLoop(): FrameLoop {
  const host: FrameHost = {
    now: () => 0,
    scheduler: { request: () => 1, cancel: () => {} },
    visibility: { isHidden: () => false, subscribe: () => () => {} },
  };
  return new FrameLoop(() => host);
}

function fakeGl() {
  const live = new Set<object>();
  const calls: { draws: [number, number][]; subData: number; bufferData: number } = { draws: [], subData: 0, bufferData: 0 };
  const flags = { depthMask: null as boolean | null, cullDisabled: false, depthRange: null as number[] | null };
  let programs = 0;
  const GL = {
    BLEND: 1, SRC_ALPHA: 2, ONE_MINUS_SRC_ALPHA: 3, DEPTH_TEST: 4, LEQUAL: 5, CULL_FACE: 6,
    ARRAY_BUFFER: 7, ELEMENT_ARRAY_BUFFER: 8, STATIC_DRAW: 9, TRIANGLES: 10, UNSIGNED_INT: 11,
    LINES: 12, FLOAT: 13, VERTEX_SHADER: 14, FRAGMENT_SHADER: 15, COMPILE_STATUS: 16, LINK_STATUS: 17,
    DYNAMIC_DRAW: 18,
  };
  const gl = {
    ...GL,
    drawingBufferWidth: 800, drawingBufferHeight: 600,
    enable: () => {}, disable: (c: number) => { if (c === GL.CULL_FACE) flags.cullDisabled = true; },
    depthMask: (v: boolean) => { flags.depthMask = v; }, depthFunc: () => {},
    depthRange: (a: number, b: number) => { flags.depthRange = [a, b]; }, blendFunc: () => {},
    createShader: () => ({}), shaderSource: () => {}, compileShader: () => {},
    getShaderParameter: () => true, deleteShader: () => {},
    createProgram: () => { programs++; return { p: programs }; }, attachShader: () => {}, linkProgram: () => {},
    getProgramParameter: () => true, useProgram: () => {},
    getAttribLocation: () => 0, getUniformLocation: () => ({}),
    uniformMatrix4fv: () => {}, uniform4f: () => {}, uniform1f: () => {}, uniform2f: () => {},
    createBuffer: () => { const b = {}; live.add(b); return b; },
    deleteBuffer: (b: object) => { live.delete(b); },
    deleteProgram: () => { programs--; },
    bindBuffer: () => {},
    bufferData: () => { calls.bufferData++; },
    bufferSubData: () => { calls.subData++; },
    enableVertexAttribArray: () => {}, vertexAttribPointer: () => {}, disableVertexAttribArray: () => {},
    drawElements: (_m: number, count: number, _t: number, off: number) => { calls.draws.push([count, off]); },
  };
  return { gl, live, calls, flags, programs: () => programs };
}

const renderArgs = {
  shaderData: { variantName: 'mercator', vertexShaderPrelude: 'uniform mat4 u_projection_matrix;', define: '' },
  defaultProjectionData: {
    mainMatrix: new Float32Array(16), tileMercatorCoords: [0, 0, 1, 1],
    clippingPlane: [0, 0, 0, 0], projectionTransition: 0, fallbackMatrix: new Float32Array(16),
  },
} as unknown as CustomRenderMethodInput;

/** a seam ON the plan: the dense vertex k nudged 30% toward k+1. */
function seamOnPlan(g: PlanGeometry, k: number, altM?: number): PlanSeam {
  const x = g.merc[k * 2] + (g.merc[k * 2 + 2] - g.merc[k * 2]) * 0.3;
  const y = g.merc[k * 2 + 1] + (g.merc[k * 2 + 3] - g.merc[k * 2 + 1]) * 0.3;
  return { mercX: x, mercY: y, altM: altM ?? g.altDisp[k], groundZ: 0 };
}

// ── contract + geometry ─────────────────────────────────────────────────────

test('geometry builds from the contract fixture: trace | curtain | edge groups, gray, drape rule', () => {
  const p = plan(filedWire);
  const dense = densifyPlan(p.points);
  assert.ok(dense.n > 4, 'legs densified along great circles');
  assert.ok(dense.n <= PLAN_MAX_POINTS, 'densified under the Law IV cap');
  const groundZ = new Float32Array(dense.n).fill(120);
  const g = buildPlanGeometry({ dense, altDisp: Float32Array.from(dense.altM), groundZ, drapeBelowM: CURTAIN_BELOW_TERRAIN_M });
  assert.equal(g.quads, g.groupEnd[PG_EDGE]);
  assert.equal(g.groupEnd[PG_TRACE], g.nSeg, 'one trace quad per segment');
  assert.equal(g.groupEnd[PG_CURTAIN] - g.groupEnd[PG_TRACE], g.nSeg, 'one curtain quad per segment (all altitudes known)');
  assert.equal(g.groupEnd[PG_EDGE] - g.groupEnd[PG_CURTAIN], g.nSeg, 'one edge quad per segment');
  // trace vertex at ground + 16 m
  near(vert(g.verts, 0)[2], 120 + TRACE_ABOVE_TERRAIN_M, 1e-3, 'trace rides ground + 16 m');
  // first curtain quad: [top_a, bottom_a, top_b, bottom_b]
  const q = g.groupEnd[PG_TRACE] * FT_VERTS_PER_SEG;
  const top = vert(g.verts, q), bot = vert(g.verts, q + 1);
  near(top[2], dense.altM[0], 1e-3, 'curtain top at the planned altitude');
  near(bot[2], 120 - CURTAIN_BELOW_TERRAIN_M, 1e-3, 'curtain bottom = ground − drape (the live curtain rule)');
  assert.deepEqual([top[6], top[7], top[8]], [0, 0, 0], 'world-space wall vertex (no extrusion, no rim)');
  near(top[9], PLAN_GRAY[0], 1e-6, 'neutral gray (DESIGN.md --text-secondary)');
  near(bot[9], PLAN_GRAY[0] * PLAN_BOTTOM_MUL, 1e-6, 'bottom darkens toward the ground');
  // segQuad sub-ranges are monotone and end at groupEnd
  for (const grp of [PG_TRACE, PG_CURTAIN, PG_EDGE]) {
    for (let s = 1; s <= g.nSeg; s++) assert.ok(g.segQuad[grp][s] >= g.segQuad[grp][s - 1]);
    assert.equal(g.segQuad[grp][g.nSeg], g.groupEnd[grp]);
  }
});

test('densify: great-circle interior points, linear altitude between KNOWN ends, honest gaps', () => {
  const d = densifyPlan([
    { lon: 0, lat: 0, altM: 1000 },
    { lon: 10, lat: 0, altM: 3000 },
    { lon: 20, lat: 0, altM: null },
  ], 64, 1500);
  assert.ok(d.n <= 64);
  // equator great circle stays on the equator
  for (let i = 0; i < d.n; i++) near(d.lat[i], 0, 1e-9, 'equatorial GC');
  const mid = Array.from(d.lon).findIndex((l) => l > 4.9 && l < 5.1);
  if (mid >= 0) near(d.altM[mid], 2000, 60, 'altitude interpolated linearly');
  // second leg has an unknown end → NaN (no curtain), position still real
  const lastLeg = Array.from(d.lon).map((l, i) => [l, d.altM[i]] as const).filter(([l]) => l > 10 && l < 20);
  assert.ok(lastLeg.length > 0 && lastLeg.every(([, a]) => Number.isNaN(a)), 'unknown altitude stays a gap');
  assert.ok(d.alongM[d.n - 1] > 2.2e6 && d.alongM[d.n - 1] < 2.25e6, 'along-route meters ≈ 20° of equator');
});

test('Law IV: densify never exceeds PLAN_MAX_POINTS, even for a huge waypoint list', () => {
  const pts = Array.from({ length: 5000 }, (_, i) => ({ lon: -170 + (i / 4999) * 340, lat: Math.sin(i / 50) * 40, altM: 10000 }));
  const d = densifyPlan(pts);
  assert.ok(d.n <= PLAN_MAX_POINTS, `n=${d.n}`);
  assert.ok(d.decimated > 0, 'decimation reported, never silent');
  assert.equal(maxFeatures, PLAN_MAX_POINTS);
});

test('estimated segments: fainter curtain, DASHED (negative-width) top edge; filed segments solid', () => {
  const g = flatGeom(plan(filedWire));
  const d = g.dense;
  const edgeBase = g.groupEnd[PG_CURTAIN];
  let sawEst = false, sawFiled = false;
  for (let s = 0; s < g.nSeg; s++) {
    const est = d.est[s] === 1 || d.est[s + 1] === 1;
    const cq = g.segQuad[PG_CURTAIN][s], eq = g.segQuad[PG_EDGE][s];
    const curtainTop = vert(g.verts, cq * FT_VERTS_PER_SEG);
    const edge = vert(g.verts, eq * FT_VERTS_PER_SEG);
    assert.ok(eq >= edgeBase);
    if (est) {
      sawEst = true;
      near(curtainTop[12], PLAN_CURTAIN_ALPHA_EST, 1e-6, 'estimated curtain is fainter');
      assert.equal(edge[8], -PLAN_EDGE_WIDTH_PX, 'estimated top edge is dashed (negative width)');
    } else {
      sawFiled = true;
      near(curtainTop[12], PLAN_CURTAIN_ALPHA, 1e-6, 'filed curtain alpha');
      assert.equal(edge[8], PLAN_EDGE_WIDTH_PX, 'filed top edge is solid');
    }
  }
  assert.ok(sawEst && sawFiled, 'fixture exercises both (climb/descent estimated, cruise filed)');
});

test('antimeridian: SFO→SIN great circle splits EXACTLY at ±180, no quad spans the wrap', () => {
  const p = plan(predictedWire);
  const g = flatGeom(p);
  const d = g.dense;
  let pairs = 0;
  for (let i = 0; i + 1 < d.n; i++) {
    if (Math.abs(d.lon[i + 1] - d.lon[i]) > 180) {
      pairs++;
      assert.equal(Math.abs(d.lon[i]), 180, 'split vertex sits on the meridian');
      assert.equal(d.lat[i], d.lat[i + 1], 'split pair shares latitude (zero-width seam)');
      assert.equal(d.alongM[i + 1], d.alongM[i], 'zero along-route distance across the split');
    }
  }
  assert.equal(pairs, 1, 'the trans-Pacific route crosses the dateline once');
  for (let q = 0; q < g.quads; q++) {
    const xs = [0, 1, 2, 3].map((k) => g.verts[(q * 4 + k) * PC_VERT_STRIDE]);
    assert.ok(Math.max(...xs) - Math.min(...xs) <= 0.5, `quad ${q} spans the antimeridian`);
  }
});

// ── the seam ────────────────────────────────────────────────────────────────

test('SEAM at the live position: gray starts exactly at the live end and draws only what lies ahead', () => {
  const g = flatGeom(plan(filedWire));
  const k = Math.floor(g.dense.n / 2);
  const seam = seamOnPlan(g, k, 10500);
  const ll = { lon: 0, lat: 0 };
  // locate in lon/lat (the layer converts mercator → lon/lat the same way)
  const mid = g.dense;
  ll.lon = mid.lon[k] + (mid.lon[k + 1] - mid.lon[k]) * 0.3;
  ll.lat = mid.lat[k] + (mid.lat[k + 1] - mid.lat[k]) * 0.3;
  const loc = locateOnPlan(g.dense, ll.lon, ll.lat)!;
  assert.equal(loc.ahead, k + 1, 'the first plan vertex AHEAD of the plane');
  assert.ok(loc.crossTrackM < 500, `on the plan (${loc.crossTrackM} m off)`);
  const v = buildSeamVertices(seam, g, loc);
  assert.equal(v.length / (PC_VERT_STRIDE * FT_VERTS_PER_SEG), 3, 'trace + curtain + edge');
  const t0 = vert(v, 0);
  assert.equal(t0[0], Math.fround(seam.mercX), 'seam trace starts at the live end x');
  assert.equal(t0[1], Math.fround(seam.mercY), 'seam trace starts at the live end y');
  const wallTop = vert(v, 4);
  near(wallTop[2], 10500, 1e-3, 'seam curtain top = the LIVE curtain end altitude (no step)');
  const wallTopB = vert(v, 6);
  assert.equal(wallTopB[0], g.verts.length ? Math.fround(g.merc[(k + 1) * 2]) : NaN, 'seam ends on the first vertex ahead');
});

test('layer frame loop: seam re-located per frame from the seam source; fixed geometry drawn from the plane forward', () => {
  const loop = manualLoop();
  const layer = new PlanCurtainLayer({ loop });
  const g = flatGeom(plan(filedWire));
  const k = 3;
  let seam: PlanSeam | null = seamOnPlan(g, k);
  layer.setSeamSource(() => seam);
  layer.setPlan(g);
  assert.equal(layer.isFrameRegistered(), true, 'registered with frameCore (Law I: the one rAF loop)');
  loop.tick(16);
  assert.equal(layer.getLocation()?.ahead, k + 1);
  const { gl, calls } = fakeGl();
  (layer as any).renderInner(gl, renderArgs, true);
  // first three draws = the fixed slot's three groups from segment k+1
  const first = calls.draws.slice(0, 3).map(([, off]) => off / 24);
  assert.deepEqual(first, [g.segQuad[PG_TRACE][k + 1], g.segQuad[PG_CURTAIN][k + 1], g.segQuad[PG_EDGE][k + 1]],
    'the part the plane passed is not drawn (sub-range from the plane forward)');
  const seamDraw = calls.draws[3];
  assert.equal(seamDraw[0], 3 * 6, 'the seam piece: 3 quads');
  // the plane advances: only the seam sub-updates, the plan never re-uploads
  const before = calls.bufferData;
  seam = seamOnPlan(g, k + 5);
  loop.tick(32);
  assert.equal(layer.getLocation()?.ahead, k + 6);
  calls.draws.length = 0;
  (layer as any).renderInner(gl, renderArgs, true);
  assert.equal(calls.bufferData, before, 'fixed plan geometry is NOT re-uploaded as the plane moves');
  assert.ok(calls.subData >= 1, 'seam piece is a bufferSubData update');
  assert.equal(calls.draws[0][1] / 24, g.segQuad[PG_TRACE][k + 6]);
  // no live end → nothing drawn (the gray waits for the live curtain)
  seam = null;
  loop.tick(48);
  assert.equal(layer.getLocation(), null);
  layer.dispose();
});

test('getTailEnd: FlightTrackLayer exposes exactly the drawn live-tail end (the seam source)', () => {
  const ft = new FlightTrackLayer();
  assert.equal(ft.getTailEnd(), null, 'no tail → null (datamap falls back to the last sample)');
  ft.setTail({
    fromMercX: 0.2, fromMercY: 0.4, fromAltM: 1000, fromGroundZ: 10,
    toMercX: 0.2005, toMercY: 0.4002, toAltM: 1100, toGroundZ: 12,
    altMin: 1000, altMax: 1100,
  });
  assert.deepEqual(ft.getTailEnd(), { mercX: 0.2005, mercY: 0.4002, altZ: 1100, groundZ: 12 });
  ft.setTail(null);
  assert.equal(ft.getTailEnd(), null);
});

test('locate is progress-monotonic on a route that doubles back', () => {
  // out along lat 10 then back along lat 10.05 (a parallel return leg ~5.6 km away)
  const d = densifyPlan([
    { lon: 0, lat: 10, altM: 10000 }, { lon: 5, lat: 10, altM: 10000 },
    { lon: 5, lat: 10.05, altM: 10000 }, { lon: 0, lat: 10.05, altM: 10000 },
  ], 400, 1500);
  // the plane has been progressing along the OUTBOUND leg…
  const prior = locateOnPlan(d, 2.45, 10.0)!;
  assert.ok(d.lat[prior.seg] < 10.01, 'on the outbound leg');
  // …and is now a touch nearer the parallel RETURN leg (both legs are great
  // circles bulging north: ~2.6 km to the return leg vs ~2.95 km outbound)
  const cold = locateOnPlan(d, 2.5, 10.036)!;
  assert.ok(d.lat[cold.seg] > 10.04, 'without history the globally-nearest (return) leg wins');
  const hinted = locateOnPlan(d, 2.5, 10.036, prior.ahead)!;
  assert.ok(d.lat[hinted.seg] < 10.01, 'with the progress hint the outbound leg keeps the seam');
  assert.ok(hinted.ahead >= prior.ahead, 'progress never runs backwards');
  // past the destination → nothing ahead
  const past = locateOnPlan(d, -1, 10.05)!;
  assert.equal(past.ahead, d.n, 'past the end: nothing left to draw');
});

// ── crossfade ───────────────────────────────────────────────────────────────

test('a new plan crossfades over PLAN_CROSSFADE_MS (frame-loop lerp, not an assignment)', () => {
  const loop = manualLoop();
  const layer = new PlanCurtainLayer({ loop });
  const g1 = flatGeom(plan(filedWire));
  layer.setSeamSource(() => seamOnPlan(g1, 2));
  layer.setPlan(g1);
  loop.tick(1000); loop.tick(1100); loop.tick(1200); loop.tick(1300); // dt 100 ×3 (clamped)
  assert.equal(layer.getFade(), 1, 'first appearance faded in');
  const g2 = flatGeom(plan({ ...filedWire, points: filedWire.points.map((p) => ({ ...p, lat: p.lat + 0.001 })) }));
  layer.setPlan(g2);
  assert.equal(layer.getFade(), 0, 'replacement starts transparent');
  assert.equal((layer as any).prev != null, true, 'outgoing plan kept for the fade');
  loop.tick(1316);
  assert.ok(layer.getFade() > 0 && layer.getFade() < 1, 'mid-fade');
  for (let t = 1332; t < 1700; t += 16) loop.tick(t);
  assert.equal(layer.getFade(), 1);
  assert.equal((layer as any).prev, null, 'outgoing plan retired after the fade');
  assert.equal(crossfadeEase(0), 0);
  assert.equal(crossfadeEase(1), 1);
  near(crossfadeEase(0.5), 0.5, 1e-9, 'smoothstep midpoint');
  assert.equal(PLAN_CROSSFADE_MS, 250);
  layer.dispose();
});

test('in-place swap (same plan, refined datum): no re-fade, no outgoing slot, seam placed synchronously', () => {
  const loop = manualLoop();
  const layer = new PlanCurtainLayer({ loop });
  const g = flatGeom(plan(filedWire));
  layer.setSeamSource(() => seamOnPlan(g, 4));
  layer.setPlan(g);
  assert.equal(layer.getLocation()?.ahead, 5, 'located inside setPlan — no unplaced frame');
  for (let t = 0; t <= 400; t += 16) loop.tick(t);
  assert.equal(layer.getFade(), 1);
  layer.setPlan(flatGeom(plan(filedWire)), { crossfade: false });
  assert.equal(layer.getFade(), 1, 'a datum refinement does not re-fade');
  assert.equal((layer as any).prev, null, 'and keeps no outgoing slot');
  assert.equal(layer.getLocation()?.ahead, 5);
  layer.dispose();
});

// ── Law IV ──────────────────────────────────────────────────────────────────

test('dispose frees EVERYTHING: GL buffers, program, frame registration, label, CPU geometry', () => {
  const loop = manualLoop();
  const layer = new PlanCurtainLayer({ loop });
  const { gl, live, programs } = fakeGl();
  layer.onAdd({ triggerRepaint: () => {}, getTerrain: () => null } as any, gl);
  const p = plan({ ...filedWire, originalPoints: filedWire.points.map((q) => ({ ...q, lon: q.lon - 0.2 })) });
  const g = flatGeom(p);
  const od = densifyPlan(p.originalPoints!);
  layer.setOriginal(buildOriginalLineVertices(od, Float32Array.from(od.altM), new Float32Array(od.n)));
  const el = { style: { transform: '', display: '' } };
  layer.setLabel({ el, mercX: 0.2, mercY: 0.4, z: 0 });
  layer.setSeamSource(() => seamOnPlan(g, 1));
  layer.setPlan(g);
  loop.tick(16);
  (layer as any).renderInner(gl, renderArgs, true);
  // replace once so a retired slot's buffers go through the garbage path too
  layer.setPlan(flatGeom(plan(filedWire)));
  loop.tick(32);
  (layer as any).renderInner(gl, renderArgs, true);
  assert.ok(live.size > 0, 'buffers were created');
  assert.equal(loop.registrationCount, 1);
  layer.dispose();
  assert.equal(live.size, 0, 'every GL buffer deleted');
  assert.equal(programs(), 0, 'program deleted');
  assert.equal(loop.registrationCount, 0, 'frame loop registration removed');
  assert.equal(layer.getVertexCount(), 0, 'CPU geometry released');
  assert.equal(el.style.display, 'none', 'label hidden on teardown');
  layer.dispose(); // idempotent
});

test('Law IV declarations: contract shape + budget derived from the layer\'s own arithmetic', () => {
  const mod = { maxFeatures, vramBudget, dispose: PlanCurtainLayer.prototype.dispose };
  assert.deepEqual(verifyLayerContract('planCurtainLayer', mod), []);
  const bytes = planWorstCaseBytes();
  assert.ok(fitsBudget(bytes, vramBudget), `worst case ${Math.round(bytes / 1024)} KB exceeds ${vramBudget} MB`);
  assert.ok(bytes * 1000 > budgetBytes(vramBudget), 'budget is not a decoration');
});

test('GL state matches the live curtain: depth write OFF, cull OFF, full depth range', () => {
  const layer = new PlanCurtainLayer({ loop: manualLoop() });
  const g = flatGeom(plan(filedWire));
  layer.setSeamSource(() => seamOnPlan(g, 1));
  layer.setPlan(g);
  layer.frame(16);
  const { gl, flags } = fakeGl();
  (layer as any).renderInner(gl, renderArgs, true);
  assert.equal(flags.depthMask, false);
  assert.equal(flags.cullDisabled, true);
  assert.deepEqual(flags.depthRange, [0, 1]);
  layer.dispose();
});

test('shader pins: live-curtain projection + far-side cull, crossfade alpha, dashing', () => {
  const vs = PC_VERT_SRC('/*prelude*/', '#define GLOBE');
  assert.match(vs, /projectTileFor3D\(a_pos\.xy, vtProjElev\(a_pos\.z, a_pos\.y\)\)/);
  assert.match(vs, /0\.998001/, 'same far-side occlusion radius² as every custom layer');
  assert.match(vs, /u_projection_transition > 0\.0 && u_projection_clipping_plane\.w < 0\.0/, 'whole-transition cull');
  assert.match(vs, /a_color\.a \* u_alpha/, 'crossfade multiplies alpha');
  assert.match(vs, /abs\(a_ext\.z\)/, 'negative width = dashed, magnitude = width');
  assert.match(PC_FRAG_SRC, /if \(v_cull > 0\.01\) discard;/);
  assert.match(PC_FRAG_SRC, /fract\(v_along \/ u_dashM\)/);
});

test('render with nothing located is a no-op; GL failures self-heal (streak + re-arm)', () => {
  const layer = new PlanCurtainLayer({ loop: manualLoop() });
  const exploding = new Proxy({}, { get: () => { throw new Error('boom'); } }) as unknown as WebGL2RenderingContext;
  layer.render(exploding, renderArgs);
  assert.equal(layer.getRenderFailed(), false, 'nothing set → never touches gl');
  const g = flatGeom(plan(filedWire));
  layer.setSeamSource(() => seamOnPlan(g, 1));
  layer.setPlan(g);
  layer.frame(16);
  for (let i = 0; i < 5; i++) layer.render(exploding, renderArgs);
  assert.equal(layer.getRenderFailed(), true);
  layer.setPlan(flatGeom(plan(filedWire)));
  assert.equal(layer.getRenderFailed(), false, 'a new plan re-arms rendering');
  layer.dispose();
});

test('label projection: CPU mirror of the frame matrix; behind camera / off screen → null', () => {
  const m = new Float32Array(16); m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  const p = projectMercToScreen(m, 0, 0, 0, 0, 800, 600)!;
  assert.deepEqual([p.x, p.y], [400, 300]);
  assert.equal(projectMercToScreen(m, 0, 2, 0, 0, 800, 600), null, 'off screen');
  const back = new Float32Array(16); back[15] = -1;
  assert.equal(projectMercToScreen(back, 0, 0, 0, 0, 800, 600), null, 'behind the camera');
});

// ── labels + provenance strings ─────────────────────────────────────────────

test('destination label text per source: FILED / PREDICTED / NONE draws no label', () => {
  assert.equal(planDestLabel(plan(filedWire)), 'SFO → LAX · FILED');
  assert.equal(planDestLabel(plan(predictedWire)), 'SFO → SIN · PREDICTED');
  assert.equal(planDestLabel(plan({ ...predictedWire, source: 'HISTORY_PREDICTED' })), 'SFO → SIN · PREDICTED');
  assert.equal(planDestLabel(plan(noneWire)), null);
  assert.equal(planDestLabel(plan({ ...filedWire, origin: null })), '? → LAX · FILED');
  // an unknown source is NEVER promoted to FILED
  assert.equal(plan({ ...filedWire, source: 'MYSTERY' }).source, 'NONE');
});

test('deviation + age strings go through the unit formatter', () => {
  const p = plan({ ...filedWire, deviation: { state: 'OFF_PLAN', crossTrackNm: 10, since: 1 } });
  assert.equal(deviationText(p.deviation, (km, d) => fmtKm(km, d, 'metric')), 'OFF PLAN by 18.5 km');
  assert.equal(deviationText(p.deviation, (km, d) => fmtKm(km, d, 'imperial')), 'OFF PLAN by 11.5 mi');
  assert.equal(deviationText(plan(filedWire).deviation, fmtKm), 'ON PLAN');
  assert.equal(fmtAgeShort(12), '12 s');
  assert.equal(fmtAgeShort(600), '10 min');
  assert.equal(fmtAgeShort(null), 'age unknown');
});

test('query string: unit-explicit altitude (ft + m), omitted unknowns', () => {
  assert.equal(planQueryString({}), '');
  const q = new URLSearchParams(planQueryString({ callsign: ' ual1 ', lat: 37.5, lon: -122.25, altM: 10000, trkDeg: -90 }).slice(1));
  assert.equal(q.get('callsign'), 'UAL1');
  assert.equal(q.get('alt'), '32808', 'feet');
  assert.equal(q.get('altM'), '10000');
  assert.equal(q.get('trk'), '270');
  assert.equal(q.get('lat'), '37.50000');
});

test('deviation re-fetch gate: >10 nm, not in flight, rate-limited', () => {
  assert.equal(shouldDeviationRefetch(PLAN_DEVIATION_REFETCH_M - 1, 0, 1e9, false), false);
  assert.equal(shouldDeviationRefetch(PLAN_DEVIATION_REFETCH_M + 1, 0, 1e9, false), true);
  assert.equal(shouldDeviationRefetch(PLAN_DEVIATION_REFETCH_M + 1, 0, 1e9, true), false, 'one request at a time');
  assert.equal(shouldDeviationRefetch(PLAN_DEVIATION_REFETCH_M + 1, 1000, 5000, false), false, 'rate-limited');
});

test('datum: DEM read only near the plane; airport elevation near airports; terrain-off = height above the flat plane', () => {
  const d = densifyPlan(plan(filedWire).points);
  let demCalls = 0;
  const out = computePlanDatum(d, {
    terrainOn: false, exag: 1, meshGround: () => 0,
    demGround: () => { demCalls++; return 500; },
  }, { lon: SFO.lon, lat: SFO.lat }, [SFO, LAX]);
  assert.ok(demCalls > 0 && demCalls < d.n, `DEM bounded to ${PLAN_DEM_RADIUS_M / 1000} km of the plane (${demCalls}/${d.n})`);
  const last = d.n - 1;
  near(out.altDisp[last], Math.max(0, 38 - 38), 1e-3, 'LAX vertex: field elevation stands in for unknown ground');
  assert.equal(out.drapeBelowM, 0);
  const on = computePlanDatum(d, { terrainOn: true, exag: 2, meshGround: () => 300, demGround: () => null }, null);
  assert.equal(on.groundZ[0], 300, 'terrain on: base rides the rendered mesh');
  assert.equal(on.drapeBelowM, CURTAIN_BELOW_TERRAIN_M * 2, 'drape overlap scaled by exaggeration');
});

test('datum: unknown ground is interpolated along the route — no curtain step at the DEM-radius edge', () => {
  const v = Float64Array.from([NaN, 100, NaN, NaN, 400, NaN]);
  const a = Float64Array.from([0, 10, 20, 30, 40, 50]);
  assert.deepEqual(Array.from(fillAlongRoute(v, a)), [100, 100, 200, 300, 400, 400], 'hold ends, lerp between');
  assert.deepEqual(Array.from(fillAlongRoute(Float64Array.from([NaN, NaN]), a.subarray(0, 2))), [0, 0], 'all unknown → the flat datum');
  // high ground (1500 m) under the plane, sea-level airports far away:
  // the terrain-off curtain top must stay continuous across the radius edge
  const d = densifyPlan([
    { lon: -105, lat: 39.8, altM: 11000 }, { lon: -95, lat: 39.8, altM: 11000 },
  ], 600, 1500);
  const out = computePlanDatum(d, { terrainOn: false, exag: 1, meshGround: () => 0, demGround: () => 1500 },
    { lon: -104.9, lat: 39.8 }, [{ lat: 39.8, lon: -95, elevM: 0 }]);
  let maxStep = 0;
  for (let i = 1; i < d.n; i++) maxStep = Math.max(maxStep, Math.abs(out.altDisp[i] - out.altDisp[i - 1]));
  assert.ok(maxStep < 50, `no step in the curtain top (max ${maxStep.toFixed(1)} m between vertices)`);
  near(out.altDisp[0], 11000 - 1500, 1e-3, 'near the plane: exactly the live curtain datum (MSL − DEM)');
});

// ── the controller lifecycle ────────────────────────────────────────────────

function fakeMap() {
  const layers = new Map<string, unknown>();
  return {
    layers,
    getLayer: (id: string) => layers.get(id),
    addLayer: (l: unknown) => { layers.set((l as { id: string }).id, l); },
    removeLayer: (id: string) => { layers.delete(id); },
    getTerrain: () => null,
    queryTerrainElevation: () => 0,
    triggerRepaint: () => {},
  };
}

function flush() { return new Promise((r) => setTimeout(r, 0)); }

test('controller: fetch on select, draw, abort + full teardown on deselect', async () => {
  const map = fakeMap();
  const store = createPlanRouteStore();
  const signals: AbortSignal[] = [];
  const urls: string[] = [];
  let intervals = 0, cleared = 0;
  let gate: (() => void) | null = null;
  const fetchImpl = (async (url: string, init?: RequestInit) => {
    urls.push(url);
    signals.push(init!.signal!);
    if (urls.length === 1) await new Promise<void>((r) => { gate = r; }); // hold the first response
    return { ok: true, status: 200, json: async () => filedWire } as Response;
  }) as unknown as typeof fetch;
  const loop = manualLoop();
  const h = startPlanRoute({
    map, hex: 'a1b2c3', store, fetchImpl, loop,
    getLive: () => ({ lon: -121.5, lat: 37.0, altM: 9000, trkDeg: 140, callsign: 'UAL1' }),
    getSeam: () => null,
    setInterval: (() => { intervals++; return 7; }) as any,
    clearInterval: (() => { cleared++; }) as any,
  });
  assert.equal(urls.length, 0, 'first fetch deferred one task (the live fix is seeded by then)');
  await flush();
  assert.equal(store.get().status, 'loading', 'designed loading state while the request is out');
  assert.match(urls[0], /^\/api\/data\/aircraft\/plan\/a1b2c3\?callsign=UAL1&lat=/);
  gate!();
  await flush();
  assert.equal(store.get().status, 'ok');
  assert.ok(map.layers.has(PLAN_LAYER_ID), 'layer added on a drawable plan');
  assert.equal(h.layer.hasPlan(), true);
  // an in-flight refresh is aborted by teardown
  h.refetch('test');
  const pending = signals[signals.length - 1];
  h.stop();
  assert.equal(pending.aborted, true, 'in-flight request aborted on deselect');
  assert.equal(map.layers.has(PLAN_LAYER_ID), false, 'layer removed');
  assert.equal(h.layer.getVertexCount(), 0, 'layer disposed');
  assert.equal(loop.registrationCount, 0, 'no frame work left behind');
  assert.equal(intervals, 1);
  assert.equal(cleared, 1, 'refresh interval cleared');
  assert.equal(store.get().status, 'idle', 'row state reset');
  assert.equal(PLAN_REFRESH_MS, 60_000);
});

test('controller: NONE source draws NOTHING (no layer, no geometry) and the row says so', async () => {
  const map = fakeMap();
  const store = createPlanRouteStore();
  const fetchImpl = (async () => ({ ok: true, status: 200, json: async () => noneWire } as Response)) as unknown as typeof fetch;
  const h = startPlanRoute({
    map, hex: 'c3d4e5', store, fetchImpl, loop: manualLoop(),
    getLive: () => null, getSeam: () => null,
    setInterval: (() => 1) as any, clearInterval: (() => {}) as any,
  });
  await flush();
  assert.equal(store.get().status, 'none');
  assert.equal(map.layers.has(PLAN_LAYER_ID), false, 'nothing added to the map');
  assert.equal(h.layer.hasPlan(), false);
  assert.equal(h.layer.getVertexCount(), 0);
  h.stop();
});

test('controller: unchanged 60 s refresh does not rebuild; HTTP failure keeps the last plan (Law V)', async () => {
  const map = fakeMap();
  const store = createPlanRouteStore();
  let mode: 'ok' | 'fail' = 'ok';
  const fetchImpl = (async () => (mode === 'ok'
    ? { ok: true, status: 200, json: async () => filedWire }
    : { ok: false, status: 503, json: async () => ({}) }) as Response) as unknown as typeof fetch;
  let tick: (() => void) | null = null;
  const h = startPlanRoute({
    map, hex: 'a1b2c3', store, fetchImpl, loop: manualLoop(),
    getLive: () => null, getSeam: () => null,
    setInterval: ((fn: () => void) => { tick = fn; return 1; }) as any, clearInterval: (() => {}) as any,
  });
  await flush();
  const slot = (h.layer as any).cur;
  tick!();
  await flush();
  assert.equal((h.layer as any).cur, slot, 'identical plan → same geometry slot (no rebuild, no crossfade)');
  mode = 'fail';
  tick!();
  await flush();
  assert.equal(store.get().status, 'ok', 'last plan still shown');
  assert.equal(store.get().refreshFailed, true, 'and the row says the refresh failed');
  assert.equal(h.layer.hasPlan(), true);
  h.stop();
});

test('controller: >10 nm off the drawn plan triggers an immediate re-fetch', async () => {
  const map = fakeMap();
  const store = createPlanRouteStore();
  let fetches = 0;
  const fetchImpl = (async () => { fetches++; return { ok: true, status: 200, json: async () => filedWire } as Response; }) as unknown as typeof fetch;
  const loop = manualLoop();
  let clock = 0;
  // the plane is ~0.5° (≈ 45 km) east of the SFO→LAX plan
  const m = lonLatToMercator(-120.0, 36.8);
  const h = startPlanRoute({
    map, hex: 'a1b2c3', store, fetchImpl, loop, now: () => clock,
    getLive: () => null,
    getSeam: () => ({ mercX: m.x, mercY: m.y, altM: 11000, groundZ: 0 }),
    setInterval: (() => 1) as any, clearInterval: (() => {}) as any,
  });
  await flush();
  assert.equal(fetches, 1);
  clock = 20_000; // past the rate limit
  loop.tick(16);
  assert.ok((h.layer.getLocation()?.crossTrackM ?? 0) > PLAN_DEVIATION_REFETCH_M);
  assert.equal(fetches, 2, 'deviation re-fetch fired from the frame loop');
  h.stop();
});

test('controller: late DEM tiles refine the SAME plan in place; unchanged numbers never touch the GPU', async () => {
  const map = fakeMap();
  const store = createPlanRouteStore();
  const timers: (() => void)[] = [];
  let dem: number | null = null;
  const fetchImpl = (async () => ({ ok: true, status: 200, json: async () => filedWire } as Response)) as unknown as typeof fetch;
  const h = startPlanRoute({
    map, hex: 'a1b2c3', store, fetchImpl, loop: manualLoop(),
    getLive: () => ({ lon: -121.5, lat: 37.0, altM: 9000, trkDeg: 140, callsign: 'UAL1' }),
    getSeam: () => null,
    demGround: () => dem,
    setInterval: (() => 1) as any, clearInterval: (() => {}) as any,
    setTimeout: ((fn: () => void) => { timers.push(fn); return timers.length; }) as any,
    clearTimeout: (() => {}) as any,
  });
  timers.shift()!(); // the deferred first fetch
  await flush();
  const first = (h.layer as any).cur;
  assert.ok(first, 'plan installed');
  assert.equal(timers.length, 1, 'DEM near the plane still pending → one bounded retry scheduled');
  timers.shift()!(); // retry: DEM still pending → identical numbers
  assert.equal((h.layer as any).cur, first, 'unchanged datum → no rebuild');
  dem = 500;
  timers.shift()!(); // retry: DEM landed → in-place refinement
  const second = (h.layer as any).cur;
  assert.notEqual(second, first, 'refined geometry installed');
  assert.equal((h.layer as any).prev, null, 'no crossfade for a datum refinement');
  assert.equal(timers.length, 0, 'retries are bounded (PLAN_DEM_RETRIES) and stop once the DEM answered');
  h.stop();
});

test('plan geometry key: identical plans match, any moved point differs', () => {
  assert.equal(planGeometryKey(plan(filedWire)), planGeometryKey(plan({ ...filedWire, ageSec: 99 })));
  assert.notEqual(planGeometryKey(plan(filedWire)),
    planGeometryKey(plan({ ...filedWire, points: [...filedWire.points.slice(0, 3), { ...filedWire.points[3], lat: 34 }] })));
  assert.equal(planGeometryKey(plan(noneWire)), 'none');
});

test('shouldCallsignRefetch: NONE/error + a callsign the last request lacked (null -> value counts), rate-limited', () => {
  const base = { status: 'none', queried: null, current: 'N843S', lastFetchStartMs: 0, nowMs: 6_000, inFlight: false };
  assert.equal(PLAN_CALLSIGN_REFETCH_MIN_GAP_MS, 5_000);
  assert.equal(shouldCallsignRefetch(base), true, 'the N843S case: NONE sent without callsign, callsign now known');
  assert.equal(shouldCallsignRefetch({ ...base, status: 'error' }), true);
  assert.equal(shouldCallsignRefetch({ ...base, queried: 'n843s ' }), false, 'same callsign (normalized) -> no refetch');
  assert.equal(shouldCallsignRefetch({ ...base, queried: 'DAL1' }), true, 'a different callsign counts');
  assert.equal(shouldCallsignRefetch({ ...base, status: 'ok' }), false, 'a drawn plan is never churned');
  assert.equal(shouldCallsignRefetch({ ...base, status: 'loading' }), false);
  assert.equal(shouldCallsignRefetch({ ...base, current: null }), false);
  assert.equal(shouldCallsignRefetch({ ...base, current: '  ' }), false);
  assert.equal(shouldCallsignRefetch({ ...base, inFlight: true }), false);
  assert.equal(shouldCallsignRefetch({ ...base, queried: undefined }), false, 'nothing sent yet: the select fetch carries it');
  assert.equal(shouldCallsignRefetch({ ...base, nowMs: 4_999 }), false, 'rate limit');
});

test('controller: NONE without a callsign re-asks immediately once the callsign appears (then stops)', async () => {
  const map = fakeMap();
  const store = createPlanRouteStore();
  const urls: string[] = [];
  const fetchImpl = (async (url: string) => {
    urls.push(url);
    const body = url.includes('callsign=N843S') ? { ...filedWire, hex: 'ab8c8e', callsign: 'N843S' } : { ...noneWire, hex: 'ab8c8e' };
    return { ok: true, status: 200, json: async () => body } as Response;
  }) as unknown as typeof fetch;
  const loop = manualLoop();
  let clock = 0;
  let live: { lon: number; lat: number; altM: number | null; trkDeg: number | null; callsign: string | null } | null = null;
  let known: string | null = null;
  const h = startPlanRoute({
    map, hex: 'ab8c8e', store, fetchImpl, loop, now: () => clock,
    getLive: () => live, getCallsign: () => known, getSeam: () => null,
    setInterval: (() => 1) as any, clearInterval: (() => {}) as any,
  });
  await flush();
  assert.equal(urls.length, 1);
  assert.doesNotMatch(urls[0], /callsign=/, 'the card opened before any callsign existed');
  assert.equal(store.get().status, 'none');
  loop.tick(16);
  assert.equal(urls.length, 1, 'no callsign yet -> no churn');
  // the live row arrives (callsign from the feed); still inside the 5 s gap
  live = { lon: -83.9, lat: 36.1, altM: 13000, trkDeg: 190, callsign: 'N843S' };
  clock = 3_000;
  loop.tick(16);
  assert.equal(urls.length, 1, 'rate-limited');
  clock = 5_000;
  loop.tick(16);
  assert.equal(urls.length, 2, 'refetched from the frame loop once the gap passed');
  assert.match(urls[1], /callsign=N843S/);
  await flush();
  assert.equal(store.get().status, 'ok', 'the FILED plan replaces NONE');
  clock = 60_000;
  loop.tick(16);
  assert.equal(urls.length, 2, 'a drawn plan never re-asks on this rule');
  h.stop();
});

test('controller: no live fix but a known callsign (off-viewport watched plane) -> the query still carries it', async () => {
  const map = fakeMap();
  const store = createPlanRouteStore();
  const urls: string[] = [];
  const fetchImpl = (async (url: string) => { urls.push(url); return { ok: true, status: 200, json: async () => noneWire } as Response; }) as unknown as typeof fetch;
  const h = startPlanRoute({
    map, hex: 'ab8c8e', store, fetchImpl, loop: manualLoop(),
    getLive: () => null, getCallsign: () => 'n843s', getSeam: () => null,
    setInterval: (() => 1) as any, clearInterval: (() => {}) as any,
  });
  await flush();
  assert.equal(urls[0], '/api/data/aircraft/plan/ab8c8e?callsign=N843S');
  h.stop();
});
