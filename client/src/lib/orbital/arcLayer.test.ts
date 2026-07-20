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
  ARC_VERT_SRC,
  ARC_FRAG_SRC,
  ARC_VERT_STRIDE,
  ARC_VERTS_PER_SEG,
  ARC_INDICES_PER_SEG,
  ARC_DEFAULT_WIDTH_PX,
} from './arcLayer.js';
import { ARC_GAP } from './orbitArc.js';

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
  assert.ok(src.includes('projectTileFor3D(a_pos.xy, vtProjElev(a_pos.z, a_pos.y))'), 'this endpoint at REAL altitude');
  assert.ok(src.includes('projectTileFor3D(a_other.xy, vtProjElev(a_other.z, a_other.y))'), 'other endpoint projected for the screen direction');
  assert.ok(src.includes('uniform float u_width'), 'default width is a uniform');
  assert.ok(src.includes('uniform vec2 u_viewport'), 'viewport uniform for px→NDC');
  assert.ok(src.includes('(widthPx * 0.5) * 2.0 / u_viewport'), 'half-width in px converted via viewport');
  assert.ok(src.includes('offs * self.w'), 'offset scaled by w — constant PIXEL width at every zoom/tilt');
  assert.ok(src.includes('a_ext.z > 0.0 ? a_ext.z : u_width'), 'per-arc width override falls back to the uniform');
  // far-side cull: same GLOBE-guarded fragment-flag pattern as before —
  // vertices are never snapped for culling (ribbons would smear)
  // vtProjElev (2026-07-20) adds a second GLOBE guard — collect ALL guarded
  // blocks; the invariant (globe symbols never leak outside a guard) stands
  const __guards = Array.from(src.matchAll(/#ifdef GLOBE[\s\S]*?#endif/g)).map((m) => m[0]);
  const __guarded = __guards.join('\n');
  const __outside = src.replace(/#ifdef GLOBE[\s\S]*?#endif/g, '');
  const outside = __outside;
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(__guarded.includes(sym) && !outside.includes(sym), `${sym} inside the GLOBE guard`);
  }
  assert.ok(src.includes('0.998001'), 'cull radius² unchanged (OCCLUSION_RADIUS²)');
  assert.ok(src.includes('v_cull = 1.0'), 'far side flags the FRAGMENT stage — never a snapped vertex');
});

test('fragment contract: cull discard + edge rim (alpha 1 core → 0.55 edges)', () => {
  assert.ok(ARC_FRAG_SRC.includes('if (v_cull > 0.01) discard'), 'far-side discard kept');
  assert.ok(ARC_FRAG_SRC.includes('mix(1.0, 0.55, abs(v_edge))'), 'subtle rim via the across-ribbon coordinate');
});

// (The ALTITUDE CURTAIN WALL mode and its pins moved OUT 2026-07-20 with
// the flight-track replacement: the aircraft curtain contract — 40m
// below-terrain drape, 34% alpha, teal→violet ramp, depth rules — is now
// pinned by lib/air/flightTrackLayer.test.ts. This file owns ribbons only.)

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

test('SELF-HEALING failures (2026-07-20, flightTrackLayer pattern): one throw retries, 5 disable, fresh arcs re-arm', () => {
  const layer = new ArcLayer();
  const explodingGl = new Proxy({}, { get() { throw new Error('broken gl'); } });
  layer.setArcs([{ pts: new Float32Array([0.1, 0.5, 400_000, 0.11, 0.5, 400_000]), color: [1, 1, 1, 1] }]);
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false, 'ONE failure stays retryable (transient context loss)');
  for (let i = 0; i < 4; i++) layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), true, 'a persistent streak (5) self-disables');
  layer.render(explodingGl as any, {} as any); // disabled → silent no-op, MUST NOT throw
  layer.setArcs([{ pts: new Float32Array([0.2, 0.5, 500_000, 0.21, 0.5, 500_000]), color: [1, 1, 1, 1] }]);
  assert.equal(layer.getRenderFailed(), false, 'fresh arcs re-arm the retry');
});
