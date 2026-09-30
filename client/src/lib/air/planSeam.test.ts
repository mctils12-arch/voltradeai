// FORWARD-ONLY SEAM, TERMINAL VECTORING DISPLAY and WAYPOINT LABELS of the
// gray planned-route curtain (2026-09-30 — live AAL892R → KAUS: an aircraft
// being vectored ~5 nm beside its FILED arrival, 25 nm out, drew a connector
// SIDEWAYS to the perpendicular foot on the route, then a 90° corner).
// Pins: the connector joins AHEAD of the plane in its direction of travel
// (never perpendicular / backwards), inside the destination's terminal area
// it is drawn as faint dashed "ATC vectors" with no curtain, an on-route
// plane is unchanged, server-flagged vectoring segments render the same way,
// and the named-fix labels exclude synthesized names and respect the cap.
// Run: npx tsx --test client/src/lib/air/planSeam.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { CustomRenderMethodInput, Map as MapLibreMap } from 'maplibre-gl';
import {
  PlanCurtainLayer,
  densifyPlan,
  buildPlanGeometry,
  buildSeamVertices,
  locateOnPlan,
  locateSeam,
  fixesAhead,
  declutterLabels,
  PC_VERT_STRIDE,
  PG_TRACE,
  PG_CURTAIN,
  PLAN_VECTOR_EDGE_WIDTH_PX,
  PLAN_TRACE_WIDTH_PX,
  PLAN_FIX_LABEL_MAX,
  PLAN_FIX_LABEL_MIN_PX,
  type PlanGeometry,
  type PlanSeam,
  type PlanFixSlot,
  type DensePlan,
} from './planCurtainLayer.js';
import { FT_VERTS_PER_SEG } from './flightTrackLayer.js';
import {
  normalizePlan,
  planGeometryKey,
  planVectoringText,
  planVectoringAllowed,
  labelableFixIndices,
  type FlightPlan,
} from './flightPlan.js';
import { planFixesOnDense } from './planRouteController.js';
import { lonLatToMercator } from '../orbital/satBuffer.js';
import { FrameLoop, type FrameHost } from '../../render/frameCore.js';
import {
  angleDiffDeg,
  initialBearingDeg,
  haversineNm,
  seamLeadNm,
  PRESENT_POSITION_ON_PLAN_NAME,
} from '../../../../shared/flightPlanGeometry.js';

// ── fixtures ────────────────────────────────────────────────────────────────

const KDFW = { icao: 'KDFW', iata: 'DFW', name: 'Dallas Fort Worth Intl', lat: 32.8968, lon: -97.038, elevM: 185 };
const KAUS = { icao: 'KAUS', iata: 'AUS', name: 'Austin Bergstrom Intl', lat: 30.1975, lon: -97.662, elevM: 165 };

/** FILED arrival straight down the 97.662°W meridian into KAUS. */
const ausWire = (extraPoints?: unknown[]) => ({
  hex: 'a1b2c3', callsign: 'AAL892R',
  source: 'FILED_FAA', label: 'FILED — FAA SWIM flight plan KDFW→KAUS',
  origin: KDFW, destination: KAUS,
  cruiseAltFt: 34000, cruiseAltEstimated: false,
  points: extraPoints ?? [
    { lon: KDFW.lon, lat: KDFW.lat, altM: 185, altEstimated: false, name: 'DFW' },
    { lon: -97.4, lat: 32.0, altM: 10363, altEstimated: false, name: 'FIXAA' },
    { lon: -97.662, lat: 31.2, altM: 6000, altEstimated: true, name: 'FIXBB' },
    { lon: -97.662, lat: 30.8, altM: 3500, altEstimated: true, name: 'FIXCC' },
    { lon: -97.662, lat: 30.55, altM: 2400, altEstimated: true, name: 'FIXDD' },
    { lon: KAUS.lon, lat: KAUS.lat, altM: 165, altEstimated: false, name: 'AUS' },
  ],
  originalPoints: null,
  deviation: { state: 'UNKNOWN', crossTrackNm: null, since: null },
  events: [], fetchedAt: 1759000000000, ageSec: 12,
  honesty: 'Route FILED with the FAA (SWIM SFDPS).',
  pathEstimated: false,
  terminalVectoring: true,
});

const plan = (w: unknown): FlightPlan => normalizePlan(w) as FlightPlan;

const flatGeom = (d: DensePlan): PlanGeometry => buildPlanGeometry({
  dense: d, altDisp: Float32Array.from(d.altM), groundZ: new Float32Array(d.n), drapeBelowM: 0,
});

/** 25 nm north of KAUS, 5 nm EAST of the arrival, tracking to the airport */
const VEC = { lat: KAUS.lat + 25 / 60, lon: KAUS.lon + 5 / (60 * Math.cos(((KAUS.lat + 25 / 60) * Math.PI) / 180)) };
const TRK = initialBearingDeg(VEC, KAUS);

const ll = (d: DensePlan, i: number) => ({ lat: d.lat[i], lon: d.lon[i] });
/** interior angle (deg) at vertex j between j→plane and j→(j+1) */
const interiorAt = (d: DensePlan, j: number, plane: { lat: number; lon: number }) =>
  180 - angleDiffDeg(initialBearingDeg(plane, ll(d, j)), initialBearingDeg(ll(d, j), ll(d, j + 1)));

const quadCount = (v: Float32Array) => v.length / (PC_VERT_STRIDE * FT_VERTS_PER_SEG);
/** a_ext.z (width, negative = dashed) of quad q's first vertex */
const quadWidth = (v: Float32Array, q: number) => v[q * FT_VERTS_PER_SEG * PC_VERT_STRIDE + 8];
/** 1 when quad q is a world-space wall (side 0), else 0 */
const isWall = (v: Float32Array, q: number) => v[q * FT_VERTS_PER_SEG * PC_VERT_STRIDE + 6] === 0 ? 1 : 0;

function manualLoop(): FrameLoop {
  const host: FrameHost = {
    now: () => 0,
    scheduler: { request: () => 1, cancel: () => {} },
    visibility: { isHidden: () => false, subscribe: () => () => {} },
  };
  return new FrameLoop(() => host);
}

const seamAt = (lon: number, lat: number, altM: number): PlanSeam => {
  const m = lonLatToMercator(lon, lat);
  return { mercX: m.x, mercY: m.y, altM, groundZ: 0 };
};

// ── 1. the regression ───────────────────────────────────────────────────────

test('regression AAL892R: 5 nm beside the filed route 25 nm out → connector joins AHEAD, angle ≥ 110°, never backwards, vectoring', () => {
  const d = densifyPlan(plan(ausWire()).points);
  const loc = locateSeam(d, VEC.lon, VEC.lat, -1, TRK, true)!;
  assert.ok(loc, 'located');
  const xtNm = loc.crossTrackM / 1852;
  assert.ok(xtNm > 4.5 && xtNm < 5.5, `fixture ~5 nm off (${xtNm})`);
  const j = loc.join!;
  assert.ok(j >= loc.ahead && d.alongM[j] > loc.projAlongM, 'join lies ahead along-track of the projection');
  assert.ok(j < d.n - 1, 'joins the route, not merely the destination');
  const conn = initialBearingDeg(VEC, ll(d, j));
  assert.ok(angleDiffDeg(conn, TRK) <= 90, `never backwards: dot(connector, track) >= 0 (${conn} vs ${TRK})`);
  assert.ok(interiorAt(d, j, VEC) >= 110, `connector meets the next leg at ${interiorAt(d, j, VEC).toFixed(1)}° (>= 110°)`);
  assert.ok(haversineNm(VEC, ll(d, j)) >= seamLeadNm(xtNm) - 0.05, 'lead >= max(3 nm, 2 × cross-track)');
  assert.equal(loc.vectoring, true, 'terminal area (25 nm) and > 2 nm off → ATC vectors');
  // contrast: the old rule joined the first vertex ahead of the perpendicular foot
  assert.ok(interiorAt(d, loc.ahead, VEC) < 110, 'the old perpendicular join formed the ~90° corner');

  // the seam piece is drawn as vectors: dashed trace + faint dashed edge, NO curtain
  const g = flatGeom(d);
  const v = buildSeamVertices(seamAt(VEC.lon, VEC.lat, 2700), g, loc);
  assert.equal(quadCount(v), 2, 'trace + edge only');
  assert.equal(isWall(v, 0) + isWall(v, 1), 0, 'no curtain wall under ATC vectors');
  assert.equal(quadWidth(v, 0), -PLAN_TRACE_WIDTH_PX, 'dashed ground trace');
  assert.equal(quadWidth(v, 1), -PLAN_VECTOR_EDGE_WIDTH_PX, 'thin dashed top edge');
  // it ends ON the join vertex
  const endX = v[(FT_VERTS_PER_SEG * PC_VERT_STRIDE) + 2 * PC_VERT_STRIDE];
  assert.equal(endX, Math.fround(g.merc[j * 2]));
});

test('vectoring display is suppressed when not allowed (FILED route-text-only great circle) — the join stays forward', () => {
  const d = densifyPlan(plan(ausWire()).points);
  const loc = locateSeam(d, VEC.lon, VEC.lat, -1, TRK, false)!;
  assert.equal(loc.vectoring, false);
  assert.ok(interiorAt(d, loc.join!, VEC) >= 110);
  const estimatedFiled = plan({ ...ausWire(), pathEstimated: true });
  assert.equal(planVectoringAllowed(estimatedFiled), false);
  assert.equal(planVectoringAllowed(plan(ausWire())), true);
  assert.equal(planVectoringAllowed(plan({ ...ausWire(), source: 'ROUTE_DB_PREDICTED', pathEstimated: true })), true);
});

test('unknown track: next vertex ahead of the projection, never behind', () => {
  const d = densifyPlan(plan(ausWire()).points);
  const loc = locateSeam(d, VEC.lon, VEC.lat, -1, null, true)!;
  assert.ok(d.alongM[loc.join!] > loc.projAlongM);
  assert.ok(d.lat[loc.join!] < VEC.lat, 'south of the aircraft — toward KAUS');
});

test('layer: draws the fixed plan FROM THE JOIN (cut corner hidden) and reads the track source per seam update', () => {
  const loop = manualLoop();
  const layer = new PlanCurtainLayer({ loop });
  const d = densifyPlan(plan(ausWire()).points);
  const g = flatGeom(d);
  layer.setSeamSource(() => seamAt(VEC.lon, VEC.lat, 2700));
  let trkReads = 0;
  layer.setTrackSource(() => { trkReads++; return TRK; });
  layer.setPlan(g);
  loop.tick(16);
  const loc = layer.getLocation()!;
  assert.ok(trkReads >= 1, 'track read on the seam update');
  assert.ok(loc.join! > loc.ahead, 'the join skips the sideways corner');
  const { gl, draws } = fakeGl();
  layer.onAdd(fakeMap() as unknown as MapLibreMap, gl);
  layer.render(gl, renderArgs(0, 0, 1));
  assert.equal(draws[0][1] / 24, g.segQuad[PG_TRACE][loc.join!], 'fixed trace drawn from the join');
  layer.dispose();
});

// ── 2. on-route: unchanged ──────────────────────────────────────────────────

test('on-route en-route plane: join = the next vertex ahead, no vectoring — the seam is byte-identical to before', () => {
  const d = densifyPlan(plan(ausWire()).points);
  const g = flatGeom(d);
  const k = 40;
  const lon = d.lon[k] + (d.lon[k + 1] - d.lon[k]) * 0.3;
  const lat = d.lat[k] + (d.lat[k + 1] - d.lat[k]) * 0.3;
  const trk = initialBearingDeg(ll(d, k), ll(d, k + 1));
  const oldLoc = locateOnPlan(d, lon, lat)!;
  const loc = locateSeam(d, lon, lat, -1, trk, true)!;
  assert.equal(loc.join, loc.ahead);
  assert.equal(loc.join, k + 1);
  assert.equal(loc.vectoring, false);
  const seam = seamAt(lon, lat, 10000);
  assert.deepEqual(Array.from(buildSeamVertices(seam, g, loc)), Array.from(buildSeamVertices(seam, g, oldLoc)));
});

// ── 3. server-flagged vectoring segments ────────────────────────────────────

test('server-flagged connector (point.vectors): normalized, keyed, densified as vectoring, drawn with no curtain', () => {
  const pts = [
    { lon: -97.566, lat: 30.614, altM: 2743, altEstimated: false, name: PRESENT_POSITION_ON_PLAN_NAME, vectors: true },
    { lon: -97.662, lat: 30.45, altM: 1900, altEstimated: true },
    { lon: KAUS.lon, lat: KAUS.lat, altM: 165, altEstimated: false, name: 'AUS' },
  ];
  const p = plan(ausWire(pts));
  assert.equal(p.points[0].vectors, true);
  assert.equal(p.points[1].vectors, undefined);
  const unflagged = plan(ausWire(pts.map(({ vectors: _v, ...rest }) => rest)));
  assert.notEqual(planGeometryKey(p), planGeometryKey(unflagged), 'the flag is part of the geometry key');
  assert.equal(planVectoringText(p), 'ATC vectors — not part of the filed route');
  assert.equal(planVectoringText(plan({ ...ausWire(pts), source: 'HISTORY_PREDICTED' })),
    'ATC vectors — not part of the predicted route');
  assert.equal(planVectoringText(plan({ ...ausWire(pts), terminalVectoring: false })), null);

  const d = densifyPlan(p.points);
  const firstLeg = d.src.indexOf(1); // dense index of input point 1
  assert.ok(firstLeg > 1, 'the connector was densified');
  for (let i = 0; i < firstLeg; i++) assert.equal(d.vec[i], 1, `connector vertex ${i} flagged`);
  for (let i = firstLeg; i < d.n; i++) assert.equal(d.vec[i], 0, `route vertex ${i} not flagged`);
  const g = flatGeom(d);
  const curtainQuads = g.segQuad[PG_CURTAIN][d.n - 1] - g.segQuad[PG_CURTAIN][0];
  assert.equal(curtainQuads, d.n - 1 - firstLeg, 'no curtain on the vectoring connector');
  // the plane flying along the flagged connector: seam drawn as vectors too
  const loc = locateSeam(d, (d.lon[1] + d.lon[2]) / 2, (d.lat[1] + d.lat[2]) / 2, -1, TRK, true)!;
  assert.equal(loc.vectoring, true);
});

test('densify src map: every input point maps to its dense vertex, inserted vertices are −1', () => {
  const p = plan(ausWire());
  const d = densifyPlan(p.points);
  for (let i = 0; i < p.points.length; i++) {
    const k = d.src.indexOf(i);
    assert.ok(k >= 0, `input ${i} present`);
    assert.equal(d.lat[k], p.points[i].lat);
    assert.equal(d.lon[k], p.points[i].lon);
  }
  assert.ok(Array.from(d.src).filter((s) => s < 0).length > 50, 'great-circle inserts carry −1');
});

// ── 4. waypoint labels ──────────────────────────────────────────────────────

test('labels: named fixes only — synthesized names and the destination excluded', () => {
  const pts = [
    { lon: -97.4, lat: 32.0, altM: 9000, altEstimated: false, name: 'FIXAA' },
    { lon: -97.5, lat: 31.5, altM: 9000, altEstimated: true, name: 'TOD (est.)' },
    { lon: -97.566, lat: 30.9, altM: 2743, altEstimated: false, name: PRESENT_POSITION_ON_PLAN_NAME },
    { lon: -97.662, lat: 30.8, altM: null, altEstimated: true },
    { lon: -97.662, lat: 30.55, altM: 2400, altEstimated: true, name: 'FIXDD' },
    { lon: -97.66, lat: 30.4, altM: 2000, altEstimated: true, name: 'present position' },
    { lon: KAUS.lon, lat: KAUS.lat, altM: 165, altEstimated: false, name: 'AUS' },
  ];
  const p = plan(ausWire(pts));
  assert.deepEqual(labelableFixIndices(p.points), [0, 4]);
  const d = densifyPlan(p.points);
  const fixes = planFixesOnDense(p.points, d);
  assert.deepEqual(fixes.map((f) => f.name), ['FIXAA', 'FIXDD']);
  for (const f of fixes) {
    const src = p.points.find((q) => q.name === f.name)!;
    assert.equal(d.lat[f.k], src.lat);
  }
});

test('labels: the nearest-ahead cap and the declutter spacing', () => {
  const fixes = Array.from({ length: 30 }, (_, i) => ({ k: i * 10, name: `F${i}` }));
  const a = fixesAhead(fixes, 55, PLAN_FIX_LABEL_MAX);
  assert.equal(a.length, PLAN_FIX_LABEL_MAX);
  assert.equal(a[0].name, 'F6', 'first fix at/after the join');
  assert.ok(a.every((f) => f.k >= 55));
  assert.equal(fixesAhead(fixes, 285, PLAN_FIX_LABEL_MAX).length, 1);
  const show = declutterLabels([{ x: 0, y: 0 }, { x: 10, y: 0 }, null, { x: 40, y: 0 }, { x: 45, y: 0 }], PLAN_FIX_LABEL_MIN_PX);
  assert.deepEqual(show, [true, false, false, true, false]);
});

test('layer: labels positioned per frame from the render matrix — ≤ cap, only fixes ahead, decluttered, hidden on dispose', () => {
  // a long straight route with 30 named fixes, plane near its start
  const pts = Array.from({ length: 32 }, (_, i) => ({
    lon: -97.662, lat: 33 - i * 0.1, altM: 3000, altEstimated: false, name: i === 31 ? 'DEST' : `FX${String(i).padStart(2, '0')}`,
  }));
  const p = plan(ausWire(pts));
  const d = densifyPlan(p.points);
  const g = flatGeom(d);
  const loop = manualLoop();
  const layer = new PlanCurtainLayer({ loop });
  const plane = { lon: -97.662, lat: 33 - 2.35 * 0.1 }; // between FX02 and FX03, on the route
  layer.setSeamSource(() => seamAt(plane.lon, plane.lat, 3000));
  layer.setTrackSource(() => 180);
  layer.setPlan(g);
  const slots = makeSlots(PLAN_FIX_LABEL_MAX);
  layer.setFixLabels({ fixes: planFixesOnDense(p.points, d), slots });
  loop.tick(16);
  const { gl } = fakeGl();
  layer.onAdd(fakeMap() as unknown as MapLibreMap, gl);
  const c = lonLatToMercator(-97.662, 32.0);
  layer.render(gl, renderArgs(c.x, c.y, 180));
  const shown = slots.filter((s) => s.el.style.display !== 'none');
  assert.ok(shown.length > 0 && shown.length <= PLAN_FIX_LABEL_MAX, `${shown.length} shown`);
  const names = shown.map((s) => s.text.textContent);
  assert.equal(names[0], 'FX03', 'nearest fix AHEAD of the plane first (FX00-02 are behind)');
  assert.ok(!names.includes('DEST'), 'destination has its own label');
  const xy = shown.filter((s) => s.text.style.display !== 'none').map((s) => {
    const m = /translate3d\(([-\d.]+)px, ([-\d.]+)px/.exec(s.el.style.transform)!;
    return { x: Number(m[1]), y: Number(m[2]) };
  });
  for (let i = 0; i < xy.length; i++) for (let k = i + 1; k < xy.length; k++) {
    assert.ok(Math.hypot(xy[i].x - xy[k].x, xy[i].y - xy[k].y) >= PLAN_FIX_LABEL_MIN_PX, 'visible texts respect the declutter spacing');
  }
  layer.dispose();
  assert.ok(slots.every((s) => s.el.style.display === 'none'), 'every label hidden on dispose');
});

// ── fakes ───────────────────────────────────────────────────────────────────

function makeSlots(n: number): PlanFixSlot[] {
  return Array.from({ length: n }, () => ({
    el: { style: { transform: '', display: 'none' } },
    text: { textContent: null, style: { display: '' } },
  }));
}

function fakeMap() {
  return {
    getCanvas: () => ({ clientWidth: 800, clientHeight: 600 }),
    triggerRepaint: () => {},
    getTerrain: () => null,
  };
}

/** column-major matrix: clip = ((mx − cx)·s, −(my − cy)·s, 0, 1) */
function renderArgs(cx: number, cy: number, s: number): CustomRenderMethodInput {
  const m = new Float32Array(16);
  m[0] = s; m[5] = -s; m[12] = -cx * s; m[13] = cy * s; m[15] = 1;
  return {
    shaderData: { variantName: 'mercator', vertexShaderPrelude: '', define: '' },
    defaultProjectionData: {
      mainMatrix: m, tileMercatorCoords: [0, 0, 1, 1],
      clippingPlane: [0, 0, 0, 0], projectionTransition: 0, fallbackMatrix: new Float32Array(16),
    },
  } as unknown as CustomRenderMethodInput;
}

function fakeGl() {
  const draws: [number, number][] = [];
  const noop = () => {};
  const gl = {
    BLEND: 1, SRC_ALPHA: 2, ONE_MINUS_SRC_ALPHA: 3, DEPTH_TEST: 4, LEQUAL: 5, CULL_FACE: 6,
    ARRAY_BUFFER: 7, ELEMENT_ARRAY_BUFFER: 8, STATIC_DRAW: 9, TRIANGLES: 10, UNSIGNED_INT: 11,
    FLOAT: 13, VERTEX_SHADER: 14, FRAGMENT_SHADER: 15, COMPILE_STATUS: 16, LINK_STATUS: 17, DYNAMIC_DRAW: 18,
    drawingBufferWidth: 800, drawingBufferHeight: 600,
    enable: noop, disable: noop, depthMask: noop, depthFunc: noop, depthRange: noop, blendFunc: noop,
    createShader: () => ({}), shaderSource: noop, compileShader: noop, getShaderParameter: () => true, deleteShader: noop,
    createProgram: () => ({}), attachShader: noop, linkProgram: noop, getProgramParameter: () => true, useProgram: noop,
    getAttribLocation: () => 0, getUniformLocation: () => ({}),
    uniformMatrix4fv: noop, uniform4f: noop, uniform1f: noop, uniform2f: noop,
    createBuffer: () => ({}), deleteBuffer: noop, deleteProgram: noop, bindBuffer: noop,
    bufferData: noop, bufferSubData: noop,
    enableVertexAttribArray: noop, vertexAttribPointer: noop, disableVertexAttribArray: noop,
    drawElements: (_m: number, count: number, _t: number, off: number) => { draws.push([count, off]); },
  };
  return { gl: gl as unknown as WebGL2RenderingContext, draws };
}
