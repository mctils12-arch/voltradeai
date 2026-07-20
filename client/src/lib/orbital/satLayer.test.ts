// Pins the satellite vertex shader's far-side cull to the CPU mirror in
// ./occlusion. GLSL can't import TS, so satLayer.ts INLINES earthOccludes /
// cameraFromClippingPlane — these tests are the sync mechanism: if someone
// edits the shader math or the occlusion constants, one side fails here.
//
// The shader itself can't execute under node (no WebGL); occlusion.test.ts
// exercises the identical math numerically. Here we assert the structural
// contract of the generated source: the cull exists, uses the same formulas
// and constants, is guarded to the globe variant, and never runs mid
// projection transition.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  VERT_SRC,
  shouldSkipTickRepaint,
  MAX_GROUND_TRACK_SPEED_MPS,
  MAX_GLIDE_SEC,
  GLIDE_FRAME_SEC,
  glideDtSec,
  shouldRequestGlideFrame,
  glideRepaintDelaySec,
  tickAnchorFromEpoch,
  MAX_ANCHOR_LAG_MS,
} from './satLayer.js';
import { metersPerPixel } from '../lod.js';
import { OCCLUSION_RADIUS } from './occlusion.js';

const PRELUDE = '/* prelude stub */';
const src = VERT_SRC(PRELUDE, '#define GLOBE');

test('far-side cull block exists and is guarded to the GLOBE shader variant', () => {
  // vtProjElev (2026-07-20) adds a second GLOBE guard — collect ALL guarded
  // blocks; the invariant (globe symbols never leak outside a guard) stands
  const __guards = Array.from(src.matchAll(/#ifdef GLOBE[\s\S]*?#endif/g)).map((m) => m[0]);
  const __guarded = __guards.join('\n');
  const __outside = src.replace(/#ifdef GLOBE[\s\S]*?#endif/g, '');
  assert.ok(__guards.length > 0, 'has #ifdef GLOBE guards');
  // globe-only symbols must ONLY appear inside a guard, or the mercator
  // variant (whose prelude lacks them) would fail to compile.
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(__guarded.includes(sym), `cull block uses ${sym}`);
    const outside = __outside;
    assert.ok(!outside.includes(sym), `${sym} must not leak outside #ifdef GLOBE`);
  }
});

test('cull mirrors occlusion.ts: camera reconstruction, segment test, radius²', () => {
  // camera = plane.xyz · (-1/plane.w)  (cameraFromClippingPlane)
  assert.ok(
    src.includes('u_projection_clipping_plane.xyz * (-1.0 / u_projection_clipping_plane.w)'),
    'reconstructs the camera exactly as cameraFromClippingPlane does',
  );
  // segment–sphere closest-approach test (earthOccludes)
  assert.ok(src.includes('float t = -dot(cam, v) / dot(v, v);'), 'closest-approach parameter');
  assert.ok(src.includes('t > 0.0 && t < 1.0'), 'strictly-between-endpoints check');
  // the inlined constant must be OCCLUSION_RADIUS² (limb anti-flicker bias)
  const r2 = (OCCLUSION_RADIUS * OCCLUSION_RADIUS).toFixed(6);
  assert.ok(src.includes(r2), `inlines OCCLUSION_RADIUS² (${r2})`);
});

test('cull is disabled mid globe↔mercator transition and on a degenerate plane', () => {
  assert.ok(
    src.includes('u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0'),
    'gated to full globe mode with a valid clipping plane',
  );
});

test('altitude enters the sphere position (GEO must cull differently from LEO)', () => {
  // Since the glide (2026-07-17) the cull runs on the GLIDED position
  // (g_xy/g_alt) — it must hide the point where it is DRAWN.
  assert.ok(
    src.includes('projectToSphere(g_xy) * (1.0 + g_alt / GLOBE_RADIUS)'),
    'satellite sphere position includes (glided) altitude, matching mercatorToSphere',
  );
});

test('sentinel slots are still culled before the far-side test', () => {
  const sentinel = src.indexOf('cls < 0.0');
  // the far-side cull block is the one that projects the glided position
  // (the vtProjElev guard higher up is a unit conversion, not the cull)
  const cull = src.indexOf('projectToSphere(g_xy)');
  assert.ok(sentinel >= 0 && sentinel < cull, 'sentinel check precedes the far-side cull');
});

// ── LOD envelope fade (EARTH TWIN A1 / E0-2) ──

test('LOD: u_opacity is declared OUTSIDE the GLOBE guard and scales the final alpha', () => {
  // vtProjElev (2026-07-20) adds a second GLOBE guard — collect ALL guarded
  // blocks; the invariant (globe symbols never leak outside a guard) stands
  const __guards = Array.from(src.matchAll(/#ifdef GLOBE[\s\S]*?#endif/g)).map((m) => m[0]);
  const __guarded = __guards.join('\n');
  const __outside = src.replace(/#ifdef GLOBE[\s\S]*?#endif/g, '');
  const outside = __outside;
  assert.ok(outside.includes('uniform float u_opacity;'),
    'u_opacity must compile in BOTH projection variants (declared outside #ifdef GLOBE)');
  const alphaMul = src.indexOf('v_color.a *= u_opacity;');
  const classColor = src.indexOf('cls < 0.5 ? u_colorLEO');
  assert.ok(alphaMul > classColor && classColor >= 0,
    'the fade multiplies alpha AFTER class-color selection so it composes with class alpha');
});

test('LOD: setGlobalOpacity clamps to 0..1 and getGlobalOpacity reports it (no-GL smoke)', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const layer = new SatLayer();
  assert.equal(layer.getGlobalOpacity(), 1, 'fully visible by default');
  layer.setGlobalOpacity(0.4);
  assert.equal(layer.getGlobalOpacity(), 0.4);
  layer.setGlobalOpacity(-5);
  assert.equal(layer.getGlobalOpacity(), 0, 'clamped below');
  layer.setGlobalOpacity(7);
  assert.equal(layer.getGlobalOpacity(), 1, 'clamped above');
});

// ── SYMBOLS NOT DOTS (human-directed 2026-07-15) ──

test('SHAPES: a_shape declared outside the GLOBE guard; glyph branches exist; dot stays the unidentified fallback', () => {
  // vtProjElev (2026-07-20) adds a second GLOBE guard — collect ALL guarded
  // blocks; the invariant (globe symbols never leak outside a guard) stands
  const __guards = Array.from(src.matchAll(/#ifdef GLOBE[\s\S]*?#endif/g)).map((m) => m[0]);
  const __guarded = __guards.join('\n');
  const __outside = src.replace(/#ifdef GLOBE[\s\S]*?#endif/g, '');
  const outside = __outside;
  assert.ok(outside.includes('in float a_shape;'), 'a_shape compiles in BOTH projection variants');
  assert.ok(src.includes('v_shape = a_shape;'), 'shape code reaches the fragment stage');
  assert.ok(/a_shape > 0\.5 \? u_size \* 2\.6 : u_size/.test(src),
    'identified glyphs get the larger point size; unidentified dots keep the original');
});

test('SHAPES: setShapeCodes stores/clears; misaligned buffers are ignorable by contract (dots, never mislabels)', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const layer = new SatLayer();
  assert.equal(layer.getShapeCodes(), null, 'no catalog joined yet = all dots');
  const codes = new Float32Array([1, 2, 3, 0]);
  layer.setShapeCodes(codes);
  assert.equal(layer.getShapeCodes(), codes);
  layer.setShapeCodes(null);
  assert.equal(layer.getShapeCodes(), null, 'clearing returns to all-dots');
});

test('LOD: at opacity 0 render() is a no-op even with data loaded (zero GPU cost when hidden)', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const { SAT_STRIDE } = await import('./satBuffer.js');
  const layer = new SatLayer();
  layer.updatePositions(new Float32Array(SAT_STRIDE * 2), { shown: 2, deepSpaceSkipped: 0, invalidSkipped: 0 });
  layer.setGlobalOpacity(0);
  // a gl stub whose every property THROWS on access: render must return
  // before touching the context at all
  const explodingGl = new Proxy({}, { get() { throw new Error('gl touched while fully faded'); } });
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false,
    'render() must early-out before any GL work when the LOD fade is 0 (and never trip the failure latch)');
});

// ── PER-FRAME VELOCITY GLIDE (human directive 2026-07-17) ──

test('GLIDE shader contract: a_vel + u_dtSec exist OUTSIDE the GLOBE guard; position and cull use the glided coords', () => {
  // vtProjElev (2026-07-20) adds a second GLOBE guard — collect ALL guarded
  // blocks; the invariant (globe symbols never leak outside a guard) stands
  const __guards = Array.from(src.matchAll(/#ifdef GLOBE[\s\S]*?#endif/g)).map((m) => m[0]);
  const __guarded = __guards.join('\n');
  const __outside = src.replace(/#ifdef GLOBE[\s\S]*?#endif/g, '');
  const outside = __outside;
  assert.ok(outside.includes('in vec3 a_vel;'), 'a_vel compiles in BOTH projection variants');
  assert.ok(outside.includes('uniform float u_dtSec;'), 'u_dtSec compiles in BOTH projection variants');
  // glided position: base + velocity × dt, antimeridian-wrapped X, pole-clamped Y
  assert.ok(src.includes('fract(a_data.x + a_vel.x * u_dtSec)'), 'X glides and wraps via fract');
  assert.ok(src.includes('clamp(a_data.y + a_vel.y * u_dtSec, 0.0, 1.0)'), 'Y glides and clamps at the poles');
  assert.ok(src.includes('float g_alt = a_data.z + a_vel.z * u_dtSec;'), 'altitude glides too');
  assert.ok(src.includes('projectTileFor3D(g_xy, vtProjElev(g_alt, g_xy.y))'), 'the DRAWN position is the glided one');
  assert.ok(!src.includes('projectTileFor3D(a_data.xy'), 'no path draws the un-glided position anymore');
});

test('GLIDE: sentinel slots exit before any glide math (no gliding of fabricated zeros)', () => {
  const sentinel = src.indexOf('cls < 0.0');
  const glide = src.indexOf('a_vel.x * u_dtSec');
  assert.ok(sentinel >= 0 && glide > sentinel, 'sentinel early-return precedes the glide computation');
});

test('glideDtSec: elapsed seconds, clamped to the honesty cap; broken/absent clock glides zero', () => {
  assert.equal(MAX_GLIDE_SEC, 2.5, 'cap constant pinned — beyond this the first-order glide would be fiction');
  assert.equal(glideDtSec(1000, 1000), 0, 'same instant = no glide');
  assert.equal(glideDtSec(1400, 1000), 0.4, 'sub-cap elapsed passes through');
  assert.equal(glideDtSec(1000 + MAX_GLIDE_SEC * 1000, 1000), MAX_GLIDE_SEC, 'exactly at the cap');
  assert.equal(glideDtSec(1000 + 60_000, 1000), MAX_GLIDE_SEC, 'a stale worker HOLDS at the cap, never runs away');
  assert.equal(glideDtSec(500, 1000), 0, 'clock behind the anchor = no glide (never negative)');
  assert.equal(glideDtSec(NaN, 1000), 0, 'broken clock = raw tick positions (pre-glide behavior)');
});

test('shouldRequestGlideFrame: frame-visible motion gates the continuous repaint (perf lever intact)', () => {
  // zoomed way out: one frame of worst-case motion is sub-pixel — no
  // continuous repaint; the 1Hz tick path still repaints on accumulation.
  assert.equal(shouldRequestGlideFrame(0, 0), false);
  // zoomed in: per-frame motion is visible — glide frames must flow.
  assert.equal(shouldRequestGlideFrame(0, 15), true);
  // consistency with the underlying displacement math
  assert.equal(shouldRequestGlideFrame(0, 8), !shouldSkipTickRepaint(0, 8, GLIDE_FRAME_SEC));
});

// ── PULSE FIX (live report 2026-07-18: rhythmic 1Hz snapping) ──

test('glideRepaintDelaySec: closes the dead band — paced repaints where per-tick motion is visible but per-frame is not', () => {
  // continuous zone (close zoom): per-frame motion visible -> repaint every frame
  assert.equal(glideRepaintDelaySec(0, 15, 1), 0);
  // DEAD-BAND REGRESSION PIN (the measured pulse, drive_pulse.mjs): at
  // z7/equator the old binary gate requested NO glide frames…
  assert.equal(shouldRequestGlideFrame(0, 7), false, 'per-frame gate is off at z7');
  // …while one 1s tick of worst-case motion was clearly visible (~13px) —
  // so the only repaint was the tick buffer swap: the pulse WAS the tick.
  assert.equal(shouldSkipTickRepaint(0, 7, 1), false, 'a full tick of motion IS visible at z7');
  const d = glideRepaintDelaySec(0, 7, 1);
  assert.ok(d != null && d > 0 && d < 1, 'the band now gets a finite paced delay');
  const expected = metersPerPixel(0, 7) / MAX_GROUND_TRACK_SPEED_MPS;
  assert.ok(Math.abs(d! - expected) < 1e-12, 'delay = time for ~one pixel of worst-case motion');
  // idle zone (far out): even a full tick is sub-pixel -> no glide repaint
  // owed; tick accumulation covers it and the map may idle (perf preserved).
  assert.equal(glideRepaintDelaySec(0, 0, 1), null);
  // no declared tick stream: the horizon falls back to the glide cap
  assert.equal(glideRepaintDelaySec(0, 0, null), null, 'z0 is sub-pixel over even the full glide cap');
  const d2 = glideRepaintDelaySec(0, 5, null);
  assert.ok(d2 != null && d2 > 0 && d2 < MAX_GLIDE_SEC, 'one-off updates still pace inside the cap');
  // fail-open: broken camera numbers -> repaint, never a silent freeze
  assert.equal(glideRepaintDelaySec(NaN, 7, 1), 0);
  assert.equal(glideRepaintDelaySec(0, NaN, 1), 0);
});

test('tickAnchorFromEpoch: maps the worker propagation epoch into the layer clock; bad lags fall back to arrival', () => {
  // positions propagated at Date 5000, arriving at Date 5100 / perf 1000:
  // the anchor must sit 100ms BEFORE arrival in the perf clock.
  assert.equal(tickAnchorFromEpoch(5000, 5100, 1000), 900);
  assert.equal(tickAnchorFromEpoch(5100, 5100, 1000), 1000, 'zero lag = arrival');
  assert.equal(tickAnchorFromEpoch(6000, 5100, 1000), 1000, 'epoch in the future (skew) = arrival fallback');
  assert.equal(tickAnchorFromEpoch(5100 - MAX_ANCHOR_LAG_MS - 1, 5100, 1000), 1000, 'absurd lag (clock jump) = arrival fallback');
  assert.equal(tickAnchorFromEpoch(NaN, 5100, 1000), 1000, 'broken epoch = arrival fallback');
});

// A gl stub complete enough to run renderInner (compile+draw) under node.
function makeGlStub() {
  const gl: any = new Proxy({}, {
    get(_t, prop: string | symbol) {
      const p = String(prop);
      if (p === 'getShaderParameter' || p === 'getProgramParameter') return () => true;
      if (p === 'getAttribLocation') return (_prog: unknown, name: string) => (name === 'a_data' ? 0 : name === 'a_vel' ? 1 : 2);
      if (p === 'getUniformLocation') return () => ({});
      if (p === 'createShader' || p === 'createProgram' || p === 'createBuffer') return () => ({});
      return () => undefined; // every other method is a no-op; constants used only as args
    },
  });
  return gl;
}
const RENDER_ARGS = () => ({
  shaderData: { variantName: 'mercator', vertexShaderPrelude: '', define: '' },
  defaultProjectionData: {
    mainMatrix: new Float32Array(16),
    tileMercatorCoords: [0, 0, 1, 1],
    clippingPlane: [0, 0, 0, 0],
    projectionTransition: 0,
    fallbackMatrix: new Float32Array(16),
  },
}) as any;

test('render in the dead band schedules ONE paced repaint timer instead of a per-frame loop (pulse regression)', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const { SAT_STRIDE } = await import('./satBuffer.js');
  let repaints = 0;
  const timers: Array<{ cb: () => void; ms: number }> = [];
  const fakeMap = { getZoom: () => 7, getCenter: () => ({ lat: 0 }), triggerRepaint: () => { repaints++; } };
  const layer = new SatLayer({
    now: () => 1100,
    setTimeoutFn: (cb, ms) => { timers.push({ cb: cb as () => void, ms }); return timers.length; },
    clearTimeoutFn: () => {},
  });
  const gl = makeGlStub();
  layer.onAdd(fakeMap as any, gl);
  layer.updatePositions(new Float32Array(SAT_STRIDE), { shown: 1, deepSpaceSkipped: 0, invalidSkipped: 0 }, 1);
  layer.setTickTime(1000); // anchored 100ms ago in the fake clock
  const before = repaints;
  layer.render(gl, RENDER_ARGS());
  assert.equal(layer.getRenderFailed(), false, 'stubbed render path must not trip the failure latch');
  assert.equal(repaints, before, 'dead band: no synchronous per-frame repaint request');
  assert.equal(timers.length, 1, 'exactly one paced repaint timer scheduled');
  const expectedMs = (metersPerPixel(0, 7) / MAX_GROUND_TRACK_SPEED_MPS) * 1000;
  assert.ok(Math.abs(timers[0].ms - expectedMs) < 1e-6, 'timer waits until ~one pixel of worst-case motion');
  // a second render while the timer is pending must not stack another
  layer.render(gl, RENDER_ARGS());
  assert.equal(timers.length, 1, 'single pending timer, never a pile-up');
  // firing the timer requests the frame that draws the next glide step
  timers[0].cb();
  assert.equal(repaints, before + 1, 'paced timer triggers the repaint');
});

test('render at close zoom keeps the continuous glide loop (no timer, direct repaint request)', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const { SAT_STRIDE } = await import('./satBuffer.js');
  let repaints = 0;
  const timers: Array<{ cb: () => void; ms: number }> = [];
  const fakeMap = { getZoom: () => 15, getCenter: () => ({ lat: 0 }), triggerRepaint: () => { repaints++; } };
  const layer = new SatLayer({
    now: () => 1100,
    setTimeoutFn: (cb, ms) => { timers.push({ cb: cb as () => void, ms }); return timers.length; },
    clearTimeoutFn: () => {},
  });
  const gl = makeGlStub();
  layer.onAdd(fakeMap as any, gl);
  layer.updatePositions(new Float32Array(SAT_STRIDE), { shown: 1, deepSpaceSkipped: 0, invalidSkipped: 0 }, 1);
  layer.setTickTime(1000);
  const before = repaints;
  layer.render(gl, RENDER_ARGS());
  assert.equal(layer.getRenderFailed(), false);
  assert.equal(repaints, before + 1, 'close zoom: per-frame repaint request continues');
  assert.equal(timers.length, 0, 'no paced timer needed when the per-frame loop runs');
});

test('setTickTime: stores the anchor (defaulting to the injected clock) and requests a frame', async () => {
  const { SatLayer } = await import('./satLayer.js');
  let repaints = 0;
  const fakeMap = { getZoom: () => 0, getCenter: () => ({ lat: 0 }), triggerRepaint: () => { repaints++; } };
  const layer = new SatLayer({ now: () => 5000 });
  assert.equal(layer.getTickTime(), null, 'glide unwired by default — layer renders raw tick positions');
  layer.onAdd(fakeMap as any, { createBuffer: () => ({}) } as any);
  layer.setTickTime();
  assert.equal(layer.getTickTime(), 5000, 'no-arg anchors at the injected clock');
  assert.equal(repaints, 1, 'anchoring (re)starts the glide frame loop');
  layer.setTickTime(6250);
  assert.equal(layer.getTickTime(), 6250, 'explicit anchor honored');
});

// ── PERF: 1Hz orbital repaint (scale_program.md queue item (c)) ──

test('shouldSkipTickRepaint: matches metersPerPixel directly (skip iff worst-case displacement is sub-pixel)', () => {
  // zoomed way out at the equator: 8000 m/s worst-case ground track moves
  // far less than one screen pixel per second (78271m/px at z0).
  assert.equal(shouldSkipTickRepaint(0, 0, 1), true);
  assert.equal(metersPerPixel(0, 0) > MAX_GROUND_TRACK_SPEED_MPS, true, 'sanity: z0 really is sub-pixel for 1s');
  // zoomed in close: meters/px is tiny, worst-case displacement dwarfs it.
  assert.equal(shouldSkipTickRepaint(0, 15, 1), false);
  assert.equal(metersPerPixel(0, 15) < MAX_GROUND_TRACK_SPEED_MPS, true, 'sanity: z15 is well past one pixel for 1s');
  // longer elapsed time crosses the threshold even at a zoom that was
  // sub-pixel for a single tick — accumulation must matter.
  assert.equal(shouldSkipTickRepaint(0, 0, 1), true);
  assert.equal(shouldSkipTickRepaint(0, 0, 100), false, 'accumulated 100s of drift is no longer sub-pixel at z0');
});

test('shouldSkipTickRepaint: fails open (never skips) on non-finite or non-positive input', () => {
  assert.equal(shouldSkipTickRepaint(NaN, 0, 1), false);
  assert.equal(shouldSkipTickRepaint(0, NaN, 1), false);
  assert.equal(shouldSkipTickRepaint(0, 0, NaN), false);
  assert.equal(shouldSkipTickRepaint(0, 0, 0), false);
  assert.equal(shouldSkipTickRepaint(0, 0, -1), false);
});

test('updatePositions: sub-pixel ticks at low zoom accumulate and skip triggerRepaint until the drift would be visible', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const { SAT_STRIDE } = await import('./satBuffer.js');
  let repaints = 0;
  const fakeMap = {
    getZoom: () => 0,
    getCenter: () => ({ lat: 0 }),
    triggerRepaint: () => { repaints++; },
  };
  const layer = new SatLayer();
  layer.onAdd(fakeMap as any, { createBuffer: () => ({}) } as any);
  const buf = new Float32Array(SAT_STRIDE * 2);

  layer.updatePositions(buf, { shown: 2, deepSpaceSkipped: 0, invalidSkipped: 0 }, 1);
  assert.equal(repaints, 0, 'first sub-pixel tick at z0 must not force a repaint');
  assert.equal(layer.getPositions(), buf, 'the buffer itself is always updated regardless of the repaint decision');

  for (let i = 0; i < 40; i++) {
    layer.updatePositions(buf, { shown: 2, deepSpaceSkipped: 0, invalidSkipped: 0 }, 1);
  }
  assert.ok(repaints >= 1, 'accumulated drift over ~40s must eventually force a repaint (never freezes forever)');
});

test('updatePositions: always repaints immediately at a close-in zoom (no accumulation needed)', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const { SAT_STRIDE } = await import('./satBuffer.js');
  let repaints = 0;
  const fakeMap = {
    getZoom: () => 15,
    getCenter: () => ({ lat: 0 }),
    triggerRepaint: () => { repaints++; },
  };
  const layer = new SatLayer();
  layer.onAdd(fakeMap as any, { createBuffer: () => ({}) } as any);
  const buf = new Float32Array(SAT_STRIDE * 2);
  layer.updatePositions(buf, { shown: 2, deepSpaceSkipped: 0, invalidSkipped: 0 }, 1);
  assert.equal(repaints, 1, 'a single tick at z15 already exceeds one pixel of worst-case drift');
});

test('updatePositions: omitting tickIntervalSec always repaints (one-off updates, e.g. non-worker callers)', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const { SAT_STRIDE } = await import('./satBuffer.js');
  let repaints = 0;
  const fakeMap = {
    getZoom: () => 0,
    getCenter: () => ({ lat: 0 }),
    triggerRepaint: () => { repaints++; },
  };
  const layer = new SatLayer();
  layer.onAdd(fakeMap as any, { createBuffer: () => ({}) } as any);
  const buf = new Float32Array(SAT_STRIDE * 2);
  layer.updatePositions(buf, { shown: 2, deepSpaceSkipped: 0, invalidSkipped: 0 });
  layer.updatePositions(buf, { shown: 2, deepSpaceSkipped: 0, invalidSkipped: 0 });
  assert.equal(repaints, 2, 'no tickIntervalSec means every call repaints, matching pre-existing behavior');
});

// ── B3 SIM CLOCK (celestial v2, 2026-07-18): time-warped glide + epoch
// mapping. THE REGRESSION CONTRACT: at rate 1 / simulated ≡ real, every new
// function is bit-identical to its pre-B3 counterpart. ──

test('glideDtSecWarp: rate 1 is EXACTLY glideDtSec over the whole input space', async () => {
  const { glideDtSecWarp } = await import('./satLayer.js');
  for (const [now, anchor] of [
    [1000, 0], [1000, 999], [1000, 1000], [1000, 1001], [5000, 0],
    [1000, NaN], [NaN, 0], [0, -1e12],
  ] as Array<[number, number]>) {
    assert.equal(glideDtSecWarp(now, anchor, 1), glideDtSec(now, anchor), `identity at (${now}, ${anchor})`);
  }
});

test('glideDtSecWarp: scales elapsed real time by the sim rate, same honesty cap, 0 when paused', async () => {
  const { glideDtSecWarp } = await import('./satLayer.js');
  // 100 real ms at 10× = 1 simulated second of glide
  assert.equal(glideDtSecWarp(1100, 1000, 10), 1);
  // the MAX_GLIDE_SEC cap applies to SIMULATED seconds — never loosened
  assert.equal(glideDtSecWarp(2000, 1000, 60), MAX_GLIDE_SEC, 'warp hits the cap');
  assert.equal(glideDtSecWarp(9_999_999, 0, 86400), MAX_GLIDE_SEC);
  // paused (rate 0) glides zero — frozen, not extrapolated
  assert.equal(glideDtSecWarp(99_999, 0, 0), 0);
  // broken rate freezes rather than fabricates
  assert.equal(glideDtSecWarp(2000, 1000, NaN), 0);
});

test('tickAnchorFromSimEpoch: rate 1 is EXACTLY tickAnchorFromEpoch, including every fallback', async () => {
  const { tickAnchorFromSimEpoch } = await import('./satLayer.js');
  for (const [epoch, nowD] of [
    [5000, 5100], [5100, 5100], [6000, 5100],
    [5100 - MAX_ANCHOR_LAG_MS - 1, 5100], [NaN, 5100],
  ] as Array<[number, number]>) {
    assert.equal(
      tickAnchorFromSimEpoch(epoch, nowD, 1000, 1),
      tickAnchorFromEpoch(epoch, nowD, 1000),
      `identity at epoch=${epoch}`,
    );
  }
});

test('tickAnchorFromSimEpoch: maps simulated lag into real layer-clock ms via the rate', async () => {
  const { tickAnchorFromSimEpoch } = await import('./satLayer.js');
  // 6000 simulated ms of lag at 60× = 100 real ms before arrival
  assert.equal(tickAnchorFromSimEpoch(0, 6000, 1000, 60), 900);
  // the wall-clock skew guard applies to the REAL-mapped lag: a huge sim
  // lag at high rate is a tiny real lag — legitimate, not skew
  assert.equal(tickAnchorFromSimEpoch(0, 3_600_000, 1000, 3600), 0, '1h sim lag at 3600× = 1s real');
  // absurd real-mapped lag falls back to arrival (pre-fix behavior, never worse)
  assert.equal(tickAnchorFromSimEpoch(0, (MAX_ANCHOR_LAG_MS + 1) * 60, 1000, 60), 1000);
  // epoch ahead of sim-now (skew) falls back to arrival
  assert.equal(tickAnchorFromSimEpoch(7000, 6000, 1000, 60), 1000);
  // paused / broken rates anchor at arrival (glide is zero there anyway)
  assert.equal(tickAnchorFromSimEpoch(0, 6000, 1000, 0), 1000);
  assert.equal(tickAnchorFromSimEpoch(0, 6000, 1000, NaN), 1000);
});

test('setTimeScale: defaults to 1 (pre-B3 render math), clamps garbage to 0, round-trips', async () => {
  const { SatLayer } = await import('./satLayer.js');
  const layer = new SatLayer({ id: 't-warp' });
  assert.equal(layer.getTimeScale(), 1, 'default is the realtime identity');
  layer.setTimeScale(60);
  assert.equal(layer.getTimeScale(), 60);
  layer.setTimeScale(0);
  assert.equal(layer.getTimeScale(), 0, 'paused');
  layer.setTimeScale(-5);
  assert.equal(layer.getTimeScale(), 0, 'negative clamps to paused');
  layer.setTimeScale(NaN);
  assert.equal(layer.getTimeScale(), 0, 'NaN clamps to paused');
  layer.setTimeScale(1);
  assert.equal(layer.getTimeScale(), 1);
});
