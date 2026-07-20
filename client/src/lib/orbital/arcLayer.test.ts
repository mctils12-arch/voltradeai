// Arc ribbon layer — EARTH TWIN round 10b (2026-07-16). Pins the extruded
// screen-space ribbon upgrade of arcLayer.ts: vertex layout (both endpoints
// + side/end selector per vertex), quad/index structure, gap discipline
// (no quad ever spans an ARC_GAP or the antimeridian), the shader's
// constant-pixel-width extrusion contract, and the unchanged public API
// (setArcs semantics, exploding-GL no-op). orbitArc.test.ts keeps the
// original split-rule and GLOBE-cull pins; this file owns the ribbon detail.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ArcLayer,
  buildArcVertices,
  buildArcIndices,
  buildWallVertices,
  defaultWallRamp,
  ARC_VERT_SRC,
  ARC_FRAG_SRC,
  ARC_VERT_STRIDE,
  ARC_VERTS_PER_SEG,
  ARC_INDICES_PER_SEG,
  ARC_DEFAULT_WIDTH_PX,
  WALL_ALPHA_TOP,
  WALL_ALPHA_BOTTOM,
  WALL_CRUISE_ALT_M,
} from './arcLayer.js';
import { ARC_GAP } from './orbitArc.js';
import { BAND_COLORS } from '../air/airLayer.js';

test('vertex layout: quad per segment carries BOTH endpoints + side/dirSign/width', () => {
  const pts = new Float32Array([0.10, 0.50, 400_000, 0.11, 0.52, 401_000]);
  const v = buildArcVertices([{ pts, color: [0.2, 0.4, 0.6, 0.8] }]);
  assert.equal(ARC_VERT_STRIDE, 13, 'stride pinned: pos(3)+other(3)+ext(3)+rgba(4)');
  assert.equal(v.length, ARC_VERTS_PER_SEG * ARC_VERT_STRIDE, 'one segment = one 4-vertex quad');
  const vert = (i: number) => Array.from(v.subarray(i * ARC_VERT_STRIDE, (i + 1) * ARC_VERT_STRIDE));
  const near = (a: number, b: number, msg: string) => assert.ok(Math.abs(a - b) < 1e-4, `${msg} (${a} vs ${b})`);
  // vertices 0,1 sit at endpoint A and carry B as "other"; 2,3 the reverse
  for (const [i, self, other, side, dirSign] of [
    [0, [0.10, 0.50, 400_000], [0.11, 0.52, 401_000], -1, +1],
    [1, [0.10, 0.50, 400_000], [0.11, 0.52, 401_000], +1, +1],
    [2, [0.11, 0.52, 401_000], [0.10, 0.50, 400_000], -1, -1],
    [3, [0.11, 0.52, 401_000], [0.10, 0.50, 400_000], +1, -1],
  ] as [number, number[], number[], number, number][]) {
    const w = vert(i);
    for (let k = 0; k < 3; k++) near(w[k], self[k], `vertex ${i} a_pos[${k}]`);
    for (let k = 0; k < 3; k++) near(w[3 + k], other[k], `vertex ${i} a_other[${k}]`);
    assert.equal(w[6], side, `vertex ${i} side selector`);
    assert.equal(w[7], dirSign, `vertex ${i} dirSign (start=+1 / end=-1)`);
    assert.equal(w[8], 0, `vertex ${i} width: no per-arc override = 0 sentinel (u_width applies)`);
    near(w[9], 0.2, `vertex ${i} color r`);
    near(w[10], 0.4, `vertex ${i} color g`);
    near(w[11], 0.6, `vertex ${i} color b`);
    near(w[12], 0.8, `vertex ${i} color a`);
  }
});

test('per-arc widthPx override is packed per vertex; absent stays backward-compatible', () => {
  const pts = new Float32Array([0.1, 0.5, 1000, 0.11, 0.5, 1000]);
  const plain = buildArcVertices([{ pts, color: [1, 1, 1, 1] }]);
  const wide = buildArcVertices([{ pts, color: [1, 1, 1, 1], widthPx: 5.5 }]);
  for (let i = 0; i < ARC_VERTS_PER_SEG; i++) {
    assert.equal(plain[i * ARC_VERT_STRIDE + 8], 0, 'no override → 0 (shader uses u_width default)');
    assert.equal(wide[i * ARC_VERT_STRIDE + 8], 5.5, 'override carried on every quad vertex');
  }
  assert.equal(ARC_DEFAULT_WIDTH_PX, 3, 'default ribbon width ~3px');
});

test('gap handling: no quad spans an ARC_GAP or the antimeridian', () => {
  const pts = new Float32Array([
    0.10, 0.5, 400_000,
    0.11, 0.5, 401_000,
    0.12, 0.5, ARC_GAP,   // gap — both adjacent segments dropped
    0.13, 0.5, 402_000,
    0.95, 0.5, 403_000,   // antimeridian jump — dropped
    0.96, 0.5, 404_000,
  ]);
  const v = buildArcVertices([{ pts, color: [1, 1, 1, 1] }]);
  const nSeg = v.length / ARC_VERT_STRIDE / ARC_VERTS_PER_SEG;
  assert.equal(nSeg, 2, 'exactly the two good segments survive (same rule as GL_LINES)');
  for (let i = 0; i < v.length / ARC_VERT_STRIDE; i++) {
    const base = i * ARC_VERT_STRIDE;
    assert.notEqual(v[base + 2], ARC_GAP, `vertex ${i}: no gap endpoint in any quad`);
    assert.notEqual(v[base + 5], ARC_GAP, `vertex ${i}: no gap "other" endpoint in any quad`);
    assert.ok(Math.abs(v[base] - v[base + 3]) <= 0.5, `vertex ${i}: quad never bridges the antimeridian`);
  }
  // indices are per-quad blocks — a triangle can never reference two quads
  const idx = buildArcIndices(nSeg);
  assert.equal(idx.length, nSeg * ARC_INDICES_PER_SEG);
  for (let s = 0; s < nSeg; s++) {
    for (let k = 0; k < ARC_INDICES_PER_SEG; k++) {
      const ix = idx[s * ARC_INDICES_PER_SEG + k];
      assert.ok(ix >= s * ARC_VERTS_PER_SEG && ix < (s + 1) * ARC_VERTS_PER_SEG,
        `index ${ix} stays inside quad ${s} — no cross-gap triangle possible`);
    }
  }
});

test('buildArcIndices: two triangles per quad, both endpoint pairs used', () => {
  assert.deepEqual(Array.from(buildArcIndices(1)), [0, 1, 2, 2, 1, 3]);
  assert.deepEqual(Array.from(buildArcIndices(2).slice(6)), [4, 5, 6, 6, 5, 7], 'second quad offset by 4 verts');
});

test('shader contract: constant-pixel extrusion by w/viewport, width uniform, cull as before', () => {
  const src = ARC_VERT_SRC('/* prelude stub */', '#define GLOBE');
  // extrusion: both endpoints projected at TRUE altitude, ribbon extruded in
  // screen space and scaled back by w so the width is pixel-constant
  assert.ok(src.includes('projectTileFor3D(a_pos.xy, a_pos.z)'), 'this endpoint at REAL altitude');
  assert.ok(src.includes('projectTileFor3D(a_other.xy, a_other.z)'), 'other endpoint projected for the screen direction');
  assert.ok(src.includes('uniform float u_width'), 'default width is a uniform');
  assert.ok(src.includes('uniform vec2 u_viewport'), 'viewport uniform for px→NDC');
  assert.ok(src.includes('(widthPx * 0.5) * 2.0 / u_viewport'), 'half-width in px converted via viewport');
  assert.ok(src.includes('offs * self.w'), 'offset scaled by w — constant PIXEL width at every zoom/tilt');
  assert.ok(src.includes('a_ext.z > 0.0 ? a_ext.z : u_width'), 'per-arc width override falls back to the uniform');
  // far-side cull: same GLOBE-guarded fragment-flag pattern as before —
  // vertices are never snapped for culling (ribbons would smear)
  const ifdef = src.indexOf('#ifdef GLOBE');
  const endif = src.indexOf('#endif');
  const outside = src.slice(0, ifdef) + src.slice(endif);
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(src.slice(ifdef, endif).includes(sym) && !outside.includes(sym), `${sym} inside the GLOBE guard`);
  }
  assert.ok(src.includes('0.998001'), 'cull radius² unchanged (OCCLUSION_RADIUS²)');
  assert.ok(src.includes('v_cull = 1.0'), 'far side flags the FRAGMENT stage — never a snapped vertex');
});

test('fragment contract: cull discard + edge rim (alpha 1 core → 0.55 edges)', () => {
  assert.ok(ARC_FRAG_SRC.includes('if (v_cull > 0.01) discard'), 'far-side discard kept');
  assert.ok(ARC_FRAG_SRC.includes('mix(1.0, 0.55, abs(v_edge))'), 'subtle rim via the across-ribbon coordinate');
});

// ---------------------------------------------------------------------------
// ALTITUDE CURTAIN WALLS (setWalls) — solid world-space quads ground→path.
// ---------------------------------------------------------------------------

test('wall vertex layout: quad = [top_i, bottom_i, top_j, bottom_j], world-space (side/dirSign/width all 0)', () => {
  const pts = new Float32Array([0.10, 0.50, 4000, 0.11, 0.52, 8000]);
  const flat = (): [number, number, number, number] => [0.1, 0.2, 0.3, 1];
  const v = buildWallVertices([{ pts }], flat);
  assert.equal(v.length, ARC_VERTS_PER_SEG * ARC_VERT_STRIDE, 'one segment = one 4-vertex quad, ribbon stride reused');
  const vert = (i: number) => Array.from(v.subarray(i * ARC_VERT_STRIDE, (i + 1) * ARC_VERT_STRIDE));
  const near = (a: number, b: number, msg: string) => assert.ok(Math.abs(a - b) < 1e-4, `${msg} (${a} vs ${b})`);
  // top_i, bottom_i, top_j, bottom_j — top at TRUE altitude, bottom at the
  // SAME mercator at 0 (never a snapped or shifted vertex)
  for (const [i, pos] of [
    [0, [0.10, 0.50, 4000]],
    [1, [0.10, 0.50, 0]],
    [2, [0.11, 0.52, 8000]],
    [3, [0.11, 0.52, 0]],
  ] as [number, number[]][]) {
    const w = vert(i);
    for (let k = 0; k < 3; k++) near(w[k], pos[k], `vertex ${i} a_pos[${k}]`);
    // a_other = self: with side 0 the extrusion is zeroed, the quad is a
    // pure world-space surface — both rows through projectTileFor3D
    for (let k = 0; k < 3; k++) near(w[3 + k], pos[k], `vertex ${i} a_other = self [${k}]`);
    assert.equal(w[6], 0, `vertex ${i} side = 0 (no screen extrusion, no rim fade)`);
    assert.equal(w[7], 0, `vertex ${i} dirSign = 0 (unused for walls)`);
    assert.equal(w[8], 0, `vertex ${i} widthPx = 0 (irrelevant at side 0)`);
  }
});

test('wall contract: top at its altitude, bottom at zero, across a multi-point strip', () => {
  const pts = new Float32Array([
    0.10, 0.50, 1000,
    0.11, 0.51, 2000,
    0.12, 0.52, 3000,
    0.13, 0.53, 4000,
  ]);
  const v = buildWallVertices([{ pts }]);
  const nQuads = v.length / ARC_VERT_STRIDE / ARC_VERTS_PER_SEG;
  assert.equal(nQuads, 3, 'n points → n-1 quads');
  for (let q = 0; q < nQuads; q++) {
    for (const [off, srcIdx, top] of [
      [0, q, true], [1, q, false], [2, q + 1, true], [3, q + 1, false],
    ] as [number, number, boolean][]) {
      const base = (q * ARC_VERTS_PER_SEG + off) * ARC_VERT_STRIDE;
      assert.equal(v[base], pts[srcIdx * 3], `quad ${q} vert ${off} mercX matches source point`);
      assert.equal(v[base + 1], pts[srcIdx * 3 + 1], `quad ${q} vert ${off} mercY matches source point`);
      assert.equal(v[base + 2], top ? pts[srcIdx * 3 + 2] : 0,
        `quad ${q} vert ${off} ${top ? 'top at TRUE altitude' : 'bottom at zero'}`);
    }
  }
});

test('wall bottoms: terrain-following base per point, clamped to its top; absent/short → 0', () => {
  const pts = new Float32Array([0.10, 0.50, 4000, 0.11, 0.52, 8000, 0.12, 0.54, 500]);
  const bottoms = new Float32Array([1200, 2500, 900]); // 900 > top 500 → clamps
  const v = buildWallVertices([{ pts, bottoms }]);
  const z = (i: number) => v[i * ARC_VERT_STRIDE + 2];
  // quad 0 = [top_i, bottom_i, top_j, bottom_j]
  assert.equal(z(0), 4000, 'top_i stays at its altitude');
  assert.equal(z(1), 1200, 'bottom_i sits on the terrain, not sea level');
  assert.equal(z(2), 8000, 'top_j unchanged');
  assert.equal(z(3), 2500, 'bottom_j from its OWN point');
  // quad 1: a ground reading above its own top clamps to the top — a bad
  // broadcast altitude below the DEM can never invert the quad
  assert.equal(z(6), 500, 'low top stays');
  assert.equal(z(7), 500, 'bottom clamped to the top');
  // short bottoms: uncovered tail points fall back to sea level
  const vShort = buildWallVertices([{ pts, bottoms: new Float32Array([1200]) }]);
  assert.equal(vShort[1 * ARC_VERT_STRIDE + 2], 1200, 'covered point uses its base');
  assert.equal(vShort[3 * ARC_VERT_STRIDE + 2], 0, 'uncovered point falls back to 0');
});

test('wall gap handling: ARC_GAP breaks the curtain, antimeridian too — no cross-gap quads', () => {
  const pts = new Float32Array([
    0.10, 0.5, 4000,
    0.11, 0.5, 5000,
    0.12, 0.5, ARC_GAP,   // gap — both adjacent segments dropped
    0.13, 0.5, 6000,
    0.95, 0.5, 7000,      // antimeridian jump — dropped
    0.96, 0.5, 8000,
  ]);
  const v = buildWallVertices([{ pts }]);
  const nQuads = v.length / ARC_VERT_STRIDE / ARC_VERTS_PER_SEG;
  assert.equal(nQuads, 2, 'exactly the two good segments survive (same rule as the ribbons)');
  for (let i = 0; i < v.length / ARC_VERT_STRIDE; i++) {
    const base = i * ARC_VERT_STRIDE;
    assert.notEqual(v[base + 2], ARC_GAP, `vertex ${i}: no gap altitude in any quad`);
  }
  // walls share the per-quad index builder — a triangle can never reference
  // two quads, so nothing can bridge the gap downstream either
  const idx = buildArcIndices(nQuads);
  for (let s = 0; s < nQuads; s++) {
    for (let k = 0; k < ARC_INDICES_PER_SEG; k++) {
      const ix = idx[s * ARC_INDICES_PER_SEG + k];
      assert.ok(ix >= s * ARC_VERTS_PER_SEG && ix < (s + 1) * ARC_VERTS_PER_SEG,
        `index ${ix} stays inside quad ${s}`);
    }
  }
});

test('wall ramp: evaluated CPU-side per POINT; bottom keeps the hue at lower alpha', () => {
  const calls: number[] = [];
  const ramp = (altM: number): [number, number, number, number] => {
    calls.push(altM);
    return [altM / 1000, altM / 2000, altM / 4000, 0.8];
  };
  const pts = new Float32Array([0.10, 0.50, 1000, 0.11, 0.51, 2000]);
  const v = buildWallVertices([{ pts }], ramp);
  assert.deepEqual(calls, [1000, 2000], 'ramp called once per point of the segment');
  const near = (a: number, b: number, msg: string) => assert.ok(Math.abs(a - b) < 1e-5, `${msg} (${a} vs ${b})`);
  const rgba = (i: number) => Array.from(v.subarray(i * ARC_VERT_STRIDE + 9, i * ARC_VERT_STRIDE + 13));
  // top_i and bottom_i carry point i's ramp color; top_j/bottom_j point j's
  for (const [i, rgb, aMul] of [
    [0, [1.0, 0.5, 0.25], WALL_ALPHA_TOP],
    [1, [1.0, 0.5, 0.25], WALL_ALPHA_BOTTOM],
    [2, [2.0, 1.0, 0.5], WALL_ALPHA_TOP],
    [3, [2.0, 1.0, 0.5], WALL_ALPHA_BOTTOM],
  ] as [number, number[], number][]) {
    const c = rgba(i);
    for (let k = 0; k < 3; k++) near(c[k], rgb[k], `vertex ${i} hue from ITS point's altitude, top and bottom alike`);
    near(c[3], 0.8 * aMul, `vertex ${i} alpha = ramp alpha × ${aMul}`);
  }
  assert.equal(WALL_ALPHA_TOP, 0.55, 'top alpha multiplier pinned');
  assert.equal(WALL_ALPHA_BOTTOM, 0.25, 'bottom alpha multiplier pinned');
});

test('default wall ramp agrees with the AIR altitude bands (map + wall never disagree)', () => {
  const near = (a: number, b: number, msg: string) => assert.ok(Math.abs(a - b) < 1e-6, `${msg} (${a} vs ${b})`);
  const low = BAND_COLORS[1];    // low amber (#fbb24c)
  const cruise = BAND_COLORS[2]; // cruise blue (#4d9fff)
  for (let k = 0; k < 3; k++) {
    near(defaultWallRamp(0)[k], low[k], `0 m = exact LOW band stop [${k}]`);
    near(defaultWallRamp(WALL_CRUISE_ALT_M)[k], cruise[k], `${WALL_CRUISE_ALT_M} m = exact CRUISE band stop [${k}]`);
    near(defaultWallRamp(12_000)[k], cruise[k], `above cruise clamps to the CRUISE stop [${k}]`);
    const mid = defaultWallRamp(WALL_CRUISE_ALT_M / 2)[k];
    assert.ok((mid >= Math.min(low[k], cruise[k]) - 1e-9) && (mid <= Math.max(low[k], cruise[k]) + 1e-9),
      `midpoint interpolates between the stops [${k}]`);
  }
  assert.equal(WALL_CRUISE_ALT_M, 3000, 'LOW→CRUISE threshold = the air layer band split');
  assert.equal(defaultWallRamp(0)[3], 1, 'ramp alpha 1 — WALL_ALPHA_TOP/BOTTOM apply on top');
});

test('wall build cost: full rebuild at 2000 points stays trivial (live-tick rebuild path)', () => {
  const n = 2000;
  const pts = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pts[i * 3] = 0.1 + i * 1e-5;
    pts[i * 3 + 1] = 0.5;
    pts[i * 3 + 2] = 500 + (i % 100) * 100;
  }
  const bottoms = new Float32Array(n);
  for (let i = 0; i < n; i++) bottoms[i] = (i % 50) * 20; // terrain-following path
  const iters = 50;
  buildWallVertices([{ pts, bottoms }]); // warm-up
  const t0 = performance.now();
  let len = 0;
  for (let it = 0; it < iters; it++) len = buildWallVertices([{ pts, bottoms }]).length;
  const perBuild = (performance.now() - t0) / iters;
  assert.equal(len, (n - 1) * ARC_VERTS_PER_SEG * ARC_VERT_STRIDE, '1999 quads built');
  // observed ~0.1–0.5 ms; 10 ms is the generous flake-proof ceiling that
  // still pins "rebuild every live tick is fine, no append fast-path needed"
  assert.ok(perBuild < 10, `2000-pt wall rebuild must stay trivial (${perBuild.toFixed(3)} ms)`);
});

test('API smoke (no GL): setWalls coexists with setArcs; empty render never touches GL', () => {
  const layer = new ArcLayer();
  const explodingGl = new Proxy({}, { get() { throw new Error('gl touched with nothing set'); } });
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false);
  // 3-point wall → 2 quads → 8 verts
  layer.setWalls([{ pts: new Float32Array([0.10, 0.5, 1000, 0.11, 0.5, 2000, 0.12, 0.5, 3000]) }]);
  assert.equal(layer.getVertexCount(), 8, 'wall quads counted');
  // arcs stack on top of walls — one combined buffer, one index pattern
  layer.setArcs([{ pts: new Float32Array([0.1, 0.5, 400_000, 0.11, 0.5, 400_000]), color: [1, 1, 1, 1] }]);
  assert.equal(layer.getVertexCount(), 12, 'arc quad + wall quads coexist');
  layer.setWalls(null);
  assert.equal(layer.getVertexCount(), 4, 'clearing walls keeps the arcs');
  layer.setArcs(null);
  assert.equal(layer.getVertexCount(), 0, 'null clears back to zero cost');
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false);
});

test('API smoke (no GL): setArcs semantics unchanged — empty render is a no-op', () => {
  const layer = new ArcLayer();
  assert.equal(layer.getVertexCount(), 0);
  const explodingGl = new Proxy({}, { get() { throw new Error('gl touched with no arcs'); } });
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false, 'no-arc render never touches GL');
  layer.setArcs([{ pts: new Float32Array([0.1, 0.5, 400_000, 0.11, 0.5, 400_000]), color: [1, 1, 1, 1] }]);
  assert.equal(layer.getVertexCount(), 4, 'one segment = one quad');
  layer.setArcs(null);
  assert.equal(layer.getVertexCount(), 0, 'null clears back to zero cost');
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false);
});
