// Terrain-following + device-aware budgets for the selected aircraft's flown
// track and gray plan (human 2026-09-30). The rendered-terrain reads
// (map.queryTerrainElevation) cannot run headless — terrain tiles are
// blocked in the harness — so these pin the PURE sampling/densify logic the
// page and the plan controller feed with those reads.
// Run: npx tsx --test client/src/lib/air/terrainFollow.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  GROUND_TOL_M, cumulativeAlongM, currentDeviceTier, lerpPoints, refineForGround, subdivisionCount, trackBudget,
} from './terrainFollow.js';
import { buildTailVertices, FT_MAX_FEATURES, FT_VERT_STRIDE, FT_VERTS_PER_SEG } from './flightTrackLayer.js';
import {
  buildPlanGeometry, buildSeamVertices, densifyPlan, locateOnPlan, PC_VERT_STRIDE, SEAM_MAX_QUADS, SEAM_MAX_SUBDIV,
} from './planCurtainLayer.js';
import { shouldMeshRefresh, computePlanDatum } from './planRouteController.js';
import { lonLatToMercator } from '../orbital/satBuffer.js';

test('device budgets: caps, sampling and poll interval scale with the tier (never above the layer caps)', () => {
  const f = trackBudget('full'), r = trackBudget('reduced'), m = trackBudget('minimal');
  assert.ok(f.trackMaxPoints > r.trackMaxPoints && r.trackMaxPoints > m.trackMaxPoints);
  assert.ok(f.groundStepM < r.groundStepM && r.groundStepM < m.groundStepM, 'finer terrain sampling on stronger devices');
  assert.ok(f.tailMaxSubdiv > m.tailMaxSubdiv && f.seamMaxSubdiv > m.seamMaxSubdiv);
  assert.equal(f.fastPollMs, 2500);
  assert.equal(m.fastPollMs, 4000, 'minimal tier polls the fast lane every 4 s');
  assert.ok(f.trackMaxPoints <= FT_MAX_FEATURES, 'the declared Law IV cap is the ceiling');
  assert.ok(f.seamMaxSubdiv <= SEAM_MAX_SUBDIV);
  assert.deepEqual(trackBudget(undefined), f, 'unknown tier -> full (the frame governor steps it down)');
  const g = globalThis as { __vtDeviceTier?: unknown };
  const prev = g.__vtDeviceTier;
  g.__vtDeviceTier = { tier: 'minimal', pixelRatioCap: 1, reasons: [] };
  assert.equal(currentDeviceTier(), 'minimal', 'read from the renderer-capability classification, not the UA');
  g.__vtDeviceTier = undefined;
  assert.equal(currentDeviceTier(), 'full');
  g.__vtDeviceTier = prev;
});

test('subdivisionCount / lerpPoints: long segments split to the sampling step, bounded', () => {
  assert.equal(subdivisionCount(100, 300, 24), 1, 'short segment stays one piece');
  assert.equal(subdivisionCount(3000, 300, 24), 10);
  assert.equal(subdivisionCount(50_000, 300, 24), 24, 'bounded by the tier');
  assert.equal(subdivisionCount(Number.NaN, 300, 24), 1);
  const pts = lerpPoints(0, 0, 10, 20, 4);
  assert.deepEqual(pts, [[2.5, 5, 0.25], [5, 10, 0.5], [7.5, 15, 0.75]], 'interior points only');
  assert.deepEqual(lerpPoints(0, 0, 1, 1, 1), []);
});

test('refineForGround: a ridge between uniformly kept vertices earns vertices; flat ground earns none', () => {
  // 1,000 samples, 100 m apart; a 600 m ridge centered at sample 505
  const n = 1000;
  const along = new Float64Array(n), ground = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    along[i] = i * 100;
    ground[i] = Math.max(0, 600 - Math.abs(i - 505) * 20);
  }
  const uniform = Array.from({ length: 11 }, (_, k) => k * 99 + (k === 10 ? 9 : 0)); // 0..999 every ~99
  const refined = refineForGround(uniform, along, ground, 40, GROUND_TOL_M);
  assert.ok(refined.length > uniform.length, 'extra vertices spent');
  assert.ok(refined.length <= 40, 'never above the cap');
  assert.ok(refined.includes(505), 'the ridge crest is kept');
  for (const u of uniform) assert.ok(refined.includes(u), 'uniform (gap-boundary) vertices never dropped');
  assert.deepEqual(refined, [...refined].sort((a, b) => a - b), 'sorted');
  // the straight bottom edge between kept vertices now stays within tolerance near the ridge
  const flat = refineForGround(uniform, along, new Float32Array(n), 40, GROUND_TOL_M);
  assert.deepEqual(flat, uniform, 'flat ground: nothing added');
  assert.deepEqual(refineForGround([0], along, ground, 40, GROUND_TOL_M), [0]);
});

test('cumulativeAlongM: monotone great-circle distances', () => {
  const a = cumulativeAlongM([{ lat: 0, lon: 0 }, { lat: 0, lon: 1 }, { lat: 0, lon: 2 }]);
  assert.equal(a[0], 0);
  assert.ok(Math.abs(a[1] - 111_195) < 200);
  assert.ok(Math.abs(a[2] - 2 * a[1]) < 1);
});

test('flown-track tail: interior points put the trace + curtain bottom on the sampled ground', () => {
  const A = lonLatToMercator(-106, 39), B = lonLatToMercator(-105.8, 39);
  const base = {
    fromMercX: A.x, fromMercY: A.y, fromAltM: 11000, fromGroundZ: 1500,
    toMercX: B.x, toMercY: B.y, toAltM: 11000, toGroundZ: 1600, altMin: 0, altMax: 12000, drapeBelowM: 0,
  };
  const straight = buildTailVertices(base, 1);
  assert.equal(straight.length / (FT_VERT_STRIDE * FT_VERTS_PER_SEG), 3, 'one piece = trace + curtain + line');
  const mid = lonLatToMercator(-105.9, 39);
  const bent = buildTailVertices({ ...base, inner: [{ mercX: mid.x, mercY: mid.y, altM: 11000, groundZ: 3900 }] }, 1);
  assert.equal(bent.length / (FT_VERT_STRIDE * FT_VERTS_PER_SEG), 6, 'two pieces');
  // the ground trace's shared vertex carries the ridge height (+ the 16 m lift)
  const zs: number[] = [];
  for (let v = 0; v < 8; v++) zs.push(bent[v * FT_VERT_STRIDE + 2]);
  assert.ok(zs.some((z) => Math.abs(z - 3916) < 1), `ridge vertex on the sampled ground (${zs})`);
});

test('plan seam/vectoring connector: subdivided onto the rendered ground with terrain on; one piece without', () => {
  const pts = [
    { lon: -106.0, lat: 39.0, altM: 11000, altEstimated: false, name: 'A' },
    { lon: -105.0, lat: 39.0, altM: 11000, altEstimated: false, name: 'B' },
  ];
  const dense = densifyPlan(pts);
  const geom = buildPlanGeometry({ dense, altDisp: Float32Array.from(dense.altM), groundZ: new Float32Array(dense.n), drapeBelowM: 0 });
  // plane 30 km south of the route: a long connector
  const sm = lonLatToMercator(-105.8, 38.73);
  const seam = { mercX: sm.x, mercY: sm.y, altM: 11000, groundZ: 0 };
  const loc = locateOnPlan(dense, -105.8, 38.73)!;
  const one = buildSeamVertices(seam, geom, loc);
  const quadsOne = one.length / (PC_VERT_STRIDE * FT_VERTS_PER_SEG);
  let calls = 0;
  const many = buildSeamVertices(seam, geom, loc, { groundAt: () => { calls++; return 2500; }, stepM: 2000, maxSubdiv: 8 });
  const quadsMany = many.length / (PC_VERT_STRIDE * FT_VERTS_PER_SEG);
  assert.ok(quadsMany > quadsOne, `subdivided (${quadsOne} -> ${quadsMany} quads)`);
  assert.ok(quadsMany <= SEAM_MAX_QUADS, 'fits the preallocated seam buffer');
  assert.ok(calls > 0 && calls <= 8, 'ground sampled at interior points, bounded by the budget');
  let onRidge = false;
  for (let v = 0; v < many.length / PC_VERT_STRIDE; v++) if (Math.abs(many[v * PC_VERT_STRIDE + 2] - 2516) < 1) onRidge = true;
  assert.ok(onRidge, 'an interior trace vertex sits on the sampled ground (+16 m)');
  const unknown = buildSeamVertices(seam, geom, loc, { groundAt: () => null, stepM: 2000, maxSubdiv: 8 });
  assert.ok(unknown.length >= one.length, 'unknown ground keeps the straight interpolation, never NaN');
  for (const x of unknown) assert.ok(Number.isFinite(x));
});

test('plan: rendered-mesh gaps are counted and re-read from the frame loop, bounded', () => {
  const pts = [
    { lon: -106.0, lat: 39.0, altM: 11000, altEstimated: false, name: 'A' },
    { lon: -104.0, lat: 39.0, altM: 11000, altEstimated: false, name: 'B' },
  ];
  const dense = densifyPlan(pts);
  let loaded = false;
  const readers = {
    terrainOn: true, exag: 1,
    meshGround: (lon: number) => (loaded || lon < -105 ? 2000 : 0),
    demGround: () => null,
  };
  const before = computePlanDatum(dense, readers, null);
  assert.ok(before.meshMissing > 0, 'vertices beyond loaded tiles are counted');
  loaded = true;
  const after = computePlanDatum(dense, readers, null);
  assert.equal(after.meshMissing, 0);
  assert.ok(after.groundZ.every((g) => g === 2000), 'once the mesh loads the whole plan sits on it');
  const off = computePlanDatum(dense, { ...readers, terrainOn: false }, null);
  assert.equal(off.meshMissing, 0, 'terrain off never asks for the mesh');
  const base = { terrainOn: true, meshMissing: 5, lastMs: 0, nowMs: 2000, done: 0, everyMs: 2000, maxTimes: 30 };
  assert.equal(shouldMeshRefresh(base), true);
  assert.equal(shouldMeshRefresh({ ...base, nowMs: 1999 }), false, 'rate-limited');
  assert.equal(shouldMeshRefresh({ ...base, done: 30 }), false, 'bounded (tiles that never load)');
  assert.equal(shouldMeshRefresh({ ...base, meshMissing: 0 }), false);
  assert.equal(shouldMeshRefresh({ ...base, terrainOn: false }), false);
});
