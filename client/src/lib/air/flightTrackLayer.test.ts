// Flight track layer — pins THE CRITICAL FIX contract from the approved
// handoff (design_handoff_flight_track_3d, 2026-07-20): curtain bottoms
// 40m BELOW the terrain (exaggeration-aware, never clamped to the top),
// 34% uniform curtain alpha with the ×[.25,.28,.45] darkened bottom,
// honest gap/antimeridian discipline, the double-sided/no-depth-write/
// depth-test render rules, and the tag's CPU screen projection.
// Run: npx tsx --test client/src/lib/air/flightTrackLayer.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  FlightTrackLayer,
  buildTrackVertices,
  buildTailVertices,
  buildQuadIndices,
  FT_VERT_SRC,
  FT_FRAG_SRC,
  FT_MARKER_VERT_SRC,
  FT_VERT_STRIDE,
  FT_VERTS_PER_SEG,
  FT_INDICES_PER_SEG,
  MARKER_LINE_RGBA,
  MARKER_GLYPH_RGBA,
  type TrackGeomInput,
} from './flightTrackLayer.js';
import {
  CURTAIN_ALPHA,
  CURTAIN_BOTTOM_MUL,
  CURTAIN_BELOW_TERRAIN_M,
  TRACE_ABOVE_TERRAIN_M,
  TRACE_RGBA,
  TRACE_WIDTH_PX,
  ALT_LINE_WIDTH_PX,
  altRampColor,
} from './trackModel.js';

const near = (a: number, b: number, tol: number, msg: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} (${a} vs ${b})`);

const input2 = (over: Partial<TrackGeomInput> = {}): TrackGeomInput => ({
  merc: new Float32Array([0.20, 0.40, 0.201, 0.401]),
  altM: new Float32Array([1000, 2000]),
  groundZ: new Float32Array([260, 390]), // display-datum ground (e.g. 200/300 real × 1.3)
  altMin: 1000,
  altMax: 2000,
  ...over,
});

const vertAt = (v: Float32Array, i: number) =>
  Array.from(v.subarray(i * FT_VERT_STRIDE, (i + 1) * FT_VERT_STRIDE));

test('one segment with altitude → trace + curtain + line quads, in that draw order', () => {
  const v = buildTrackVertices(input2(), 1.3);
  assert.equal(v.length / FT_VERT_STRIDE / FT_VERTS_PER_SEG, 3, 'three quads: trace, curtain, line');
  // quad 0 = trace ribbon at ground + 16m (scaled by altScale)
  const t0 = vertAt(v, 0);
  near(t0[2], 260 + TRACE_ABOVE_TERRAIN_M * 1.3, 1e-3, 'trace z = groundZ + 16·exag');
  near(t0[9], TRACE_RGBA[0], 1e-6, 'trace color #1e5fd6 (float32-packed)');
  assert.equal(t0[8], TRACE_WIDTH_PX, 'trace width 3px');
  // quad 2 = altitude line at alt·exag with 4px width
  const l0 = vertAt(v, 8);
  near(l0[2], 1000 * 1.3, 1e-3, 'line z = altM × exaggeration');
  assert.equal(l0[8], ALT_LINE_WIDTH_PX, 'altitude line width 4px');
});

test('THE CURTAIN: top at altitude, bottom 40m BELOW terrain (exaggeration-aware), 34% alpha, darkened bottom', () => {
  const exag = 1.3;
  const v = buildTrackVertices(input2(), exag);
  // quad 1 = curtain: [top_i, bottom_i, top_j, bottom_j]
  const topI = vertAt(v, 4);
  const botI = vertAt(v, 5);
  near(topI[2], 1000 * exag, 1e-3, 'curtain top = altM × exag');
  near(botI[2], 260 - CURTAIN_BELOW_TERRAIN_M * exag, 1e-3,
    'curtain bottom = groundZ − 40·exag (BELOW terrain — the drape fix, never clamped to top)');
  // wall vertices: side/dir/width all 0 (world-space quad, no rim fade)
  assert.deepEqual([topI[6], topI[7], topI[8]], [0, 0, 0]);
  // colors: top = ramp(alt) at 34% alpha; bottom = top × [.25,.28,.45], same alpha
  const ramp = altRampColor(1000, 1000, 2000);
  near(topI[9], ramp[0], 1e-6, 'top r = ramp');
  near(topI[12], CURTAIN_ALPHA, 1e-6, 'top alpha 0.34');
  near(botI[9], ramp[0] * CURTAIN_BOTTOM_MUL[0], 1e-6, 'bottom r darkened ×0.25');
  near(botI[10], ramp[1] * CURTAIN_BOTTOM_MUL[1], 1e-6, 'bottom g darkened ×0.28');
  near(botI[11], ramp[2] * CURTAIN_BOTTOM_MUL[2], 1e-6, 'bottom b darkened ×0.45');
  near(botI[12], CURTAIN_ALPHA, 1e-6, 'bottom alpha stays 0.34 (darkening is in the COLOR)');
});

test('drapeBelowM 0 (terrain off): curtain bottom sits AT the flat base', () => {
  const v = buildTrackVertices(input2({ groundZ: new Float32Array([0, 0]), drapeBelowM: 0 }), 1);
  const botI = vertAt(v, 5);
  assert.equal(botI[2], 0, 'no drape overlap without terrain');
});

test('altitude gaps: trace continues (real position history), curtain + line break honestly', () => {
  const v = buildTrackVertices(input2({ altM: new Float32Array([1000, NaN]) }), 1);
  assert.equal(v.length / FT_VERT_STRIDE / FT_VERTS_PER_SEG, 1, 'only the trace quad survives a gap');
  const t0 = vertAt(v, 0);
  assert.equal(t0[8], TRACE_WIDTH_PX, 'the survivor is the ground trace');
});

test('antimeridian jumps split every geometry honestly', () => {
  const v = buildTrackVertices(input2({ merc: new Float32Array([0.01, 0.4, 0.99, 0.4]) }), 1);
  assert.equal(v.length, 0, 'no quad bridges the antimeridian');
});

test('quad indices never cross a quad boundary', () => {
  const idx = buildQuadIndices(2);
  assert.equal(idx.length, 2 * FT_INDICES_PER_SEG);
  assert.deepEqual(Array.from(idx.subarray(0, 6)), [0, 1, 2, 2, 1, 3]);
  assert.deepEqual(Array.from(idx.subarray(6, 12)), [4, 5, 6, 6, 5, 7]);
});

test('tail builder emits the same layout from one from→to pair', () => {
  const v = buildTailVertices({
    fromMercX: 0.2, fromMercY: 0.4, fromAltM: 1500, fromGroundZ: 300,
    toMercX: 0.2001, toMercY: 0.4001, toAltM: 1520, toGroundZ: 310,
    altMin: 1000, altMax: 2000,
  }, 1);
  assert.equal(v.length / FT_VERT_STRIDE / FT_VERTS_PER_SEG, 3, 'trace + curtain + line tail quads');
});

test('shader contract pins: projectTileFor3D, GLOBE far-side fragment discard, wall rim exemption', () => {
  const vs = FT_VERT_SRC('/*prelude*/', '#define GLOBE');
  assert.ok(vs.includes('projectTileFor3D(a_pos.xy, vtProjElev(a_pos.z, a_pos.y))'), 'true-altitude projection');
  assert.ok(vs.includes('0.998001'), 'far-side cull constant (OCCLUSION_RADIUS²)');
  assert.ok(vs.includes('v_cull'), 'cull flagged to the fragment stage, vertices never snapped');
  assert.ok(FT_FRAG_SRC.includes('discard'), 'fragment discard');
  assert.ok(FT_FRAG_SRC.includes('abs(v_edge)'), 'rim uses |side| — side-0 wall vertices keep full 34% alpha');
  const mvs = FT_MARKER_VERT_SRC('/*prelude*/', '');
  assert.ok(mvs.includes('mix(u_groundZ, u_altZ, a_altFrac)'), 'marker drop line spans ground→altitude');
});

test('render sets THE CRITICAL FIX GL state: depth-test on, depth-write OFF, cull OFF, full depth range', () => {
  const layer = new FlightTrackLayer();
  layer.setTrack(input2(), 1);
  const calls: string[] = [];
  const flags = { depthMask: null as boolean | null, cullDisabled: false, depthTestEnabled: false, depthRange: null as [number, number] | null };
  const GL = {
    BLEND: 1, SRC_ALPHA: 2, ONE_MINUS_SRC_ALPHA: 3, DEPTH_TEST: 4, LEQUAL: 5, CULL_FACE: 6,
    ARRAY_BUFFER: 7, ELEMENT_ARRAY_BUFFER: 8, STATIC_DRAW: 9, TRIANGLES: 10, UNSIGNED_INT: 11,
    LINES: 12, FLOAT: 13, VERTEX_SHADER: 14, FRAGMENT_SHADER: 15, COMPILE_STATUS: 16, LINK_STATUS: 17,
  };
  const gl: any = {
    ...GL,
    drawingBufferWidth: 800, drawingBufferHeight: 600,
    enable: (c: number) => { if (c === GL.DEPTH_TEST) flags.depthTestEnabled = true; calls.push(`enable${c}`); },
    disable: (c: number) => { if (c === GL.CULL_FACE) flags.cullDisabled = true; },
    depthMask: (v: boolean) => { flags.depthMask = v; },
    depthFunc: () => {},
    depthRange: (a: number, b: number) => { flags.depthRange = [a, b]; },
    blendFunc: () => {},
    createShader: () => ({}), shaderSource: () => {}, compileShader: () => {},
    getShaderParameter: () => true, deleteShader: () => {},
    createProgram: () => ({}), attachShader: () => {}, linkProgram: () => {},
    getProgramParameter: () => true, useProgram: () => {},
    getAttribLocation: () => 0, getUniformLocation: () => ({}),
    uniformMatrix4fv: () => {}, uniform4f: () => {}, uniform1f: () => {}, uniform2f: () => {}, uniform4fv: () => {},
    createBuffer: () => ({}), bindBuffer: () => {}, bufferData: () => {},
    enableVertexAttribArray: () => {}, vertexAttribPointer: () => {}, disableVertexAttribArray: () => {},
    vertexAttrib2f: () => {}, vertexAttrib1f: () => {},
    drawElements: () => { calls.push('drawElements'); }, drawArrays: () => {},
    deleteProgram: () => {}, deleteBuffer: () => {},
  };
  // node has no WebGL2RenderingContext — the layer's render() gate is typeof-safe
  (globalThis as any).WebGL2RenderingContext = (globalThis as any).WebGL2RenderingContext || function () {};
  const args: any = {
    shaderData: { variantName: 'mercator', vertexShaderPrelude: 'uniform mat4 u_projection_matrix;', define: '' },
    defaultProjectionData: {
      mainMatrix: new Float32Array(16), tileMercatorCoords: [0, 0, 1, 1],
      clippingPlane: [0, 0, 0, 0], projectionTransition: 0, fallbackMatrix: new Float32Array(16),
    },
  };
  (layer as any).renderInner(gl, args, true, false);
  assert.equal(flags.depthTestEnabled, true, 'depth TEST enabled (terrain occludes the curtain correctly)');
  assert.equal(flags.depthMask, false, 'depth WRITE off (translucent geometry never z-fights)');
  assert.equal(flags.cullDisabled, true, 'CULL_FACE disabled (double-sided — survives any tilt)');
  assert.deepEqual(flags.depthRange, [0, 1], 'full depth range (the 2d-sublayer pinning undone)');
  assert.ok(calls.includes('drawElements'), 'geometry drawn');
});

test('render is a no-op with nothing set; GL failures SELF-HEAL (streak + re-arm), never a session-long latch', () => {
  const layer = new FlightTrackLayer();
  const exploding: any = new Proxy({}, { get: () => { throw new Error('boom'); } });
  layer.render(exploding, {} as any); // nothing set → returns before touching gl
  assert.equal(layer.getRenderFailed(), false, 'no-op path never touches gl');
  layer.setTrack(input2(), 1);
  // one transient failure (context loss during a terrain hiccup) must NOT
  // kill the layer for the session — the 2026-07-20 "trail does not show" bug
  layer.render(exploding, {} as any);
  assert.equal(layer.getRenderFailed(), false, 'a single failure is retried, not latched');
  // a persistent streak (5 consecutive) disables it…
  for (let i = 0; i < 4; i++) layer.render(exploding, {} as any);
  assert.equal(layer.getRenderFailed(), true, 'persistent failures disable the layer');
  layer.render(exploding, {} as any); // disabled → early return, no throw
  // …and fresh data (every poll calls setTrack) re-arms a clean rebuild
  layer.setTrack(input2(), 1);
  assert.equal(layer.getRenderFailed(), false, 'new data re-arms rendering after recovery');
});

test('projectToScreen uses the cached frame matrix (mercator: matrix × [merc, altMeters])', () => {
  const layer = new FlightTrackLayer();
  // identity-ish matrix: clip = [x, y, z, 1] → screen center mapping checkable
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  (layer as any).lastMainMatrix = m;
  (layer as any).lastTransition = 0;
  (layer as any).altScale = 2;
  const p = layer.projectToScreen(0, 0, 100, 800, 600)!;
  assert.deepEqual([p.x, p.y], [400, 300], 'origin projects to screen center');
  // behind-camera (w <= 0) → null
  const m2 = new Float32Array(16); m2[15] = -1;
  (layer as any).lastMainMatrix = m2;
  assert.equal(layer.projectToScreen(0, 0, 100, 800, 600), null);
  // no frame yet → null
  (layer as any).lastMainMatrix = null;
  assert.equal(layer.projectToScreen(0, 0, 0, 800, 600), null);
});

test('marker colors are the handoff values', () => {
  near(MARKER_GLYPH_RGBA[0], 0x8f / 255, 1e-6, 'glyph #8fd0ff');
  near(MARKER_LINE_RGBA[3], 0.65, 1e-6, 'drop line at 65%');
});
